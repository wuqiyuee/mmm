"""联合估计每日提示效应，吸收公司×年度季度和日期固定效应。

所有日期共同估计27项控制的系数，每个可识别交易日保留独立提示系数。
固定效应和每日提示列都嵌套在季度内，因此逐季吸收、累计舒尔补方程，
避免构造全样本行数×全部日期数的稠密矩阵。公司聚类得分跨季度汇总。
本模块导入时不读取数据、不估计、不写文件。
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components


PROJECT_ROOT = Path(__file__).resolve().parents[4]
COMMON_DIR = PROJECT_ROOT / "code/07_实证分析/90_公共模块"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from 主回归口径 import CONTROL_COLUMNS, CUE_COLUMN, SPEC_ID
from 图2图3扩展_日期固定效应回归公共模块 import _residualize_quarter_block
from 提示系数回归公共模块 import print_progress


def _group_sums(values, codes, group_count):
    """按连续整数编号求列和，不创建观测数×分组数的稠密矩阵。"""
    membership = sparse.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))),
        shape=(group_count, len(codes)),
    )
    return np.asarray(membership @ values)


def _quarter_residuals(reg, positions, outcome, controls, candidate_codes):
    """仅在本季度构造日提示列，并共同消除两组固定效应。"""
    part = reg.iloc[positions]
    firm_codes, global_firms = pd.factorize(part["_cluster_firm"], sort=False)
    date_codes, global_dates = pd.factorize(part["_fe_date"], sort=True)
    selected = np.isin(np.asarray(global_dates), candidate_codes)
    daily_codes = np.asarray(global_dates)[selected]
    column_for_date = np.full(len(global_dates), -1, dtype="int64")
    column_for_date[selected] = np.arange(len(daily_codes))
    row_columns = column_for_date[date_codes]
    ncontrols = len(controls)
    values = np.zeros((len(part), 1 + ncontrols + len(daily_codes)))
    values[:, 0] = part[outcome].to_numpy(dtype="float64")
    values[:, 1:1 + ncontrols] = part[controls].to_numpy(dtype="float64")
    selected_rows = np.flatnonzero(row_columns >= 0)
    values[selected_rows, 1 + ncontrols + row_columns[selected_rows]] = (
        part[CUE_COLUMN].to_numpy(dtype="float64")[selected_rows]
    )
    residualized = _residualize_quarter_block(values, firm_codes, date_codes)
    return (
        residualized[:, 0],
        residualized[:, 1:1 + ncontrols],
        residualized[:, 1 + ncontrols:],
        firm_codes, np.asarray(global_firms), date_codes,
        np.asarray(global_dates), daily_codes,
    )


def _fixed_effect_rank(firm_codes, date_codes):
    """本季度公司、日期二部图的组数减去连通分量数。"""
    firm_count = int(firm_codes.max()) + 1
    date_count = int(date_codes.max()) + 1
    counts = sparse.csr_matrix(
        (np.ones(len(firm_codes)), (firm_codes, date_codes)),
        shape=(firm_count, date_count),
    )
    components = connected_components(
        (counts.T @ counts).tocsr(), directed=False, return_labels=False,
    )
    return firm_count + date_count - int(components)


def _influence_square_sum(local_scores, global_scores, global_second_moment, loading):
    """计算全部聚类影响得分平方和，包含本季度没有观测的聚类。"""
    local_square = np.einsum("ij,ij->j", local_scores, local_scores)
    cross = local_scores.T @ global_scores
    common_square = np.einsum(
        "ij,jk,ik->i", loading, global_second_moment, loading,
    )
    return local_square - 2 * np.einsum("ij,ij->i", cross, loading) + common_square


def estimate_joint_daily(data, dependent_variable, significance_level, scale=100):
    """返回每日样本统计、联合回归全样本及逐日提示系数和双聚类推断。"""
    if not 0 < significance_level < 1:
        raise ValueError("显著性水平必须介于0和1之间。")
    needed = ["StkcdRecall", "Trddt", "yearQuarter", "Filling"]
    numeric = [dependent_variable, CUE_COLUMN] + CONTROL_COLUMNS
    missing = [column for column in needed + numeric if column not in data]
    if missing:
        raise ValueError(f"联合日度回归缺少字段：{missing}")
    if data[["StkcdRecall", "Trddt"]].isna().any().any():
        raise ValueError("联合日度回归存在缺失公司日期键。")
    if data.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("联合日度回归的公司日期键不唯一。")
    if not data[CUE_COLUMN].dropna().isin([0, 1]).all():
        raise ValueError("提示变量存在0、1以外的取值。")

    # 一、先确定完整观测，再递归删除两组固定效应单例。
    valid = data[needed].notna().all(axis=1)
    for column in numeric:
        valid &= np.isfinite(data[column].to_numpy())
    count_data = pd.DataFrame({
        "Trddt": data["Trddt"], "_valid": valid,
        "_valid_cue": valid & data[CUE_COLUMN].eq(1),
        "_valid_noncue": valid & data[CUE_COLUMN].eq(0),
    })
    counts = count_data.groupby("Trddt", sort=True).agg(
        面板观测数=("_valid", "size"), 有效观测数=("_valid", "sum"),
        有效提示观测数=("_valid_cue", "sum"), 有效未提示观测数=("_valid_noncue", "sum"),
    )
    del count_data
    counts["两组均有有效观测"] = (
        counts["有效提示观测数"].gt(0) & counts["有效未提示观测数"].gt(0)
    )
    reg = data.loc[valid, needed + numeric].copy()
    if reg.empty:
        raise ValueError("联合日度回归没有完整有效观测。")
    reg["_fe_firm_quarter"] = reg.groupby(
        ["StkcdRecall", "yearQuarter"], observed=True, sort=False,
    ).ngroup()
    reg["_fe_date"] = pd.factorize(reg["Trddt"], sort=True)[0]
    singleton_removed = 0
    while len(reg):
        singleton = np.zeros(len(reg), dtype=bool)
        for column in ["_fe_firm_quarter", "_fe_date"]:
            codes = reg[column].to_numpy(dtype="int64")
            group_sizes = np.bincount(codes)
            singleton |= group_sizes[codes] == 1
        if not singleton.any():
            break
        singleton_removed += int(singleton.sum())
        reg = reg.loc[~singleton].copy()
    if reg.empty:
        raise ValueError("递归删除双固定效应单例后没有剩余观测。")
    reg = reg.reset_index(drop=True)
    # 面板观测数保留进入量；有效样本和两组标志统一按删单例后的实际样本报告。
    retained_counts = reg.groupby("Trddt", sort=True).agg(
        有效观测数=("StkcdRecall", "size"), 有效提示观测数=(CUE_COLUMN, "sum"),
    ).reindex(counts.index, fill_value=0)
    counts["有效观测数"] = retained_counts["有效观测数"].astype("int64")
    counts["有效提示观测数"] = retained_counts["有效提示观测数"].astype("int64")
    counts["有效未提示观测数"] = counts["有效观测数"] - counts["有效提示观测数"]
    counts["两组均有有效观测"] = (
        counts["有效提示观测数"].gt(0) & counts["有效未提示观测数"].gt(0)
    )
    reg["_cluster_firm"], firm_labels = pd.factorize(reg["StkcdRecall"], sort=False)
    reg["_fe_date"], date_labels = pd.factorize(reg["Trddt"], sort=True)
    firm_count, date_count = len(firm_labels), len(date_labels)
    if min(firm_count, date_count) < 2:
        raise ValueError("公司和日期聚类均至少需要2组。")
    daily_nobs = np.bincount(reg["_fe_date"], minlength=date_count)
    daily_cues = np.bincount(
        reg["_fe_date"], weights=reg[CUE_COLUMN], minlength=date_count,
    ).astype("int64")
    candidate_codes = np.flatnonzero((daily_cues > 0) & (daily_cues < daily_nobs))
    if not len(candidate_codes):
        raise ValueError("删除单例后，没有同时存在提示和未提示观测的日期。")
    # 无提示日和全为提示日仍参与固定效应及共用控制系数估计。
    quarters = reg.groupby("yearQuarter", observed=True, sort=True).indices
    ncontrols = len(CONTROL_COLUMNS)
    schur = np.zeros((ncontrols, ncontrols))
    rhs = np.zeros(ncontrols)
    quarter_info = []
    fixed_effect_rank = 0

    # 二、逐季剖除每日提示列，累计全期共用控制的正规方程。
    quarter_total = len(quarters)
    print_progress("步骤1/3 整理季度方程（本阶段季度数进度）", 0, quarter_total)
    for completed, (quarter, positions) in enumerate(quarters.items(), start=1):
        y, x, z, firms, _, dates, _, daily_codes = _quarter_residuals(
            reg, positions, dependent_variable, CONTROL_COLUMNS, candidate_codes,
        )
        fixed_effect_rank += _fixed_effect_rank(firms, dates)
        a = z.T @ z
        b = z.T @ x
        zty = z.T @ y
        if len(daily_codes):
            # 单列近零时条件数仍为1，须另比较吸收前后的列范数。
            raw_norms = np.maximum(daily_cues[daily_codes], 1)
            if (np.diag(a) <= raw_norms * 1e-12).any():
                raise ValueError(
                    f"{quarter}季度存在被双固定效应吸收的每日提示列；"
                    "联合估计已停止，不把浮点残差当作可识别变化。"
                )
            if not np.isfinite(a).all() or np.linalg.cond(a) >= 1e12:
                raise ValueError(
                    f"{quarter}季度的每日提示列在双固定效应后不能独立识别；"
                    "联合估计已停止，不采用任意广义逆或归一化。"
                )
            try:
                np.linalg.cholesky(a)
                a_inverse = np.linalg.inv(a)
            except np.linalg.LinAlgError as error:
                raise ValueError(f"{quarter}季度的每日提示列秩不足。") from error
        else:
            a_inverse = np.empty((0, 0))
        loading = a_inverse @ b
        profiled_x = x - z @ loading
        schur += profiled_x.T @ profiled_x
        rhs += profiled_x.T @ y
        quarter_info.append({
            "季度": quarter, "位置": positions, "日期编号": daily_codes,
            "日提示逆矩阵": a_inverse, "控制载荷": loading, "日提示因变量积": zty,
            "日提示控制积": b,
        })
        del y, x, z, profiled_x
        print_progress(
            "步骤1/3 整理季度方程（本阶段季度数进度）", completed, quarter_total,
            f"当前季度：{quarter}",
        )

    # 先按全矩阵统一尺度排除数值零，再在标准化矩阵中判断独立列。
    # 精确共线判据接近机器精度；尚未秩亏但病态的情形明确报错。
    diagonal = np.diag(schur)
    zero_tolerance = max(float(diagonal.max()), 1.0) * 1e-12
    eligible = np.flatnonzero(diagonal > zero_tolerance)
    standardized = np.zeros_like(schur)
    if len(eligible):
        norms = np.sqrt(diagonal[eligible])
        standardized[np.ix_(eligible, eligible)] = (
            schur[np.ix_(eligible, eligible)] / np.outer(norms, norms)
        )
    rank_tolerance = 100 * np.finfo("float64").eps
    active = []
    for column in eligible:
        trial = active + [int(column)]
        trial_gram = standardized[np.ix_(trial, trial)]
        if np.linalg.matrix_rank(trial_gram, tol=rank_tolerance) <= len(active):
            continue
        if np.linalg.cond(trial_gram) >= 1e12:
            raise ValueError("联合估计的标准化控制矩阵病态，不能静默省略近共线项。")
        active.append(int(column))
    active_controls = [CONTROL_COLUMNS[index] for index in active]
    omitted_controls = [column for column in CONTROL_COLUMNS if column not in active_controls]
    reduced_schur = schur[np.ix_(active, active)]
    if active and np.linalg.cond(reduced_schur) >= 1e12:
        raise ValueError("联合估计的共用控制矩阵条件数过大，无法稳定识别。")
    inverse_schur = np.linalg.inv(reduced_schur) if active else np.empty((0, 0))
    gamma = inverse_schur @ rhs[active]
    slope_count = len(active) + len(candidate_codes)
    nobs = len(reg)
    degrees = nobs - fixed_effect_rank - slope_count
    # 与正式主回归一致：协方差分母包含显式常数，但不再扣吸收固定效应秩。
    nvars = 1 + slope_count
    if degrees <= 0 or nobs <= nvars:
        raise ValueError("联合日度估计的残差自由度不足。")
    for info in quarter_info:
        info["控制载荷"] = info["控制载荷"][:, active]
        info["日系数"] = info["日提示逆矩阵"] @ (
            info["日提示因变量积"] - info["日提示控制积"][:, active] @ gamma
        )
        info.pop("日提示控制积")
        info.pop("日提示因变量积")

    # 三、再次逐季计算残差，先汇总跨季度的公司聚类共用控制得分。
    firm_scores = np.zeros((firm_count, len(active)))
    date_scores = np.zeros((date_count, len(active)))
    individual_second = np.zeros((len(active), len(active)))
    print_progress("步骤2/3 汇总聚类得分（本阶段季度数进度）", 0, quarter_total)
    for completed, info in enumerate(quarter_info, start=1):
        y, x, z, firms, global_firms, dates, global_dates, daily_codes = _quarter_residuals(
            reg, info["位置"], dependent_variable, active_controls, candidate_codes,
        )
        if not np.array_equal(daily_codes, info["日期编号"]):
            raise ValueError("重复读取季度数据时，每日提示列顺序发生变化。")
        residuals = y - x @ gamma - z @ info["日系数"]
        profiled_x = x - z @ info["控制载荷"]
        scores = profiled_x * residuals[:, None]
        firm_scores[global_firms] += _group_sums(scores, firms, len(global_firms))
        date_scores[global_dates] = _group_sums(scores, dates, len(global_dates))
        individual_second += scores.T @ scores
        local_scores = (z @ info["日提示逆矩阵"]) * residuals[:, None]
        info["单观测局部平方和"] = np.einsum("ij,ij->j", local_scores, local_scores)
        info["单观测交叉积"] = local_scores.T @ scores
        del y, x, z, residuals, profiled_x, scores, local_scores
        print_progress(
            "步骤2/3 汇总聚类得分（本阶段季度数进度）", completed, quarter_total,
            f"当前季度：{info['季度']}",
        )
    firm_second = firm_scores.T @ firm_scores
    date_second = date_scores.T @ date_scores
    common_scale = (nobs - 1) / (nobs - nvars)
    firm_scale = firm_count / (firm_count - 1) * common_scale
    date_scale = date_count / (date_count - 1) * common_scale
    individual_scale = nobs / (nobs - 1) * common_scale
    critical = float(stats.t.ppf(1 - significance_level / 2, degrees))
    rows = []
    for date_code, date in enumerate(date_labels):
        rows.append({
            "日期": date, "观测数": int(daily_nobs[date_code]),
            "提示观测数": int(daily_cues[date_code]),
            "未提示观测数": int(daily_nobs[date_code] - daily_cues[date_code]),
            "保留控制变量数": len(active), "省略控制变量": "、".join(omitted_controls),
            "系数_百分点": np.nan, "标准误_百分点": np.nan,
            "置信下限_百分点": np.nan, "置信上限_百分点": np.nan,
            "显著性概率": np.nan,
            "状态": "提示变量被日期固定效应吸收",
            "口径标识": SPEC_ID, "提示变量": CUE_COLUMN,
            "联合回归观测数": nobs, "联合每日提示系数数": len(candidate_codes),
            "公司聚类数": firm_count, "日期聚类数": date_count,
            "删除单例观测数": singleton_removed,
            "残差自由度": degrees, "固定效应实际秩": fixed_effect_rank,
        })

    # 四、最后逐季计算日系数影响得分，使用全期公司和日期得分修正。
    print_progress("步骤3/3 每日系数推断（本阶段季度数进度）", 0, quarter_total)
    for completed, info in enumerate(quarter_info, start=1):
        if not len(info["日期编号"]):
            print_progress(
                "步骤3/3 每日系数推断（本阶段季度数进度）", completed, quarter_total,
                f"当前季度：{info['季度']}（无可识别日提示列）",
            )
            continue
        y, x, z, firms, global_firms, dates, global_dates, _ = _quarter_residuals(
            reg, info["位置"], dependent_variable, active_controls, candidate_codes,
        )
        residuals = y - x @ gamma - z @ info["日系数"]
        local_scores = (z @ info["日提示逆矩阵"]) * residuals[:, None]
        firm_local = _group_sums(local_scores, firms, len(global_firms))
        date_local = _group_sums(local_scores, dates, len(global_dates))
        loading = info["控制载荷"] @ inverse_schur
        firm_diagonal = _influence_square_sum(
            firm_local, firm_scores[global_firms], firm_second, loading,
        )
        date_diagonal = _influence_square_sum(
            date_local, date_scores[global_dates], date_second, loading,
        )
        individual_diagonal = (
            info["单观测局部平方和"]
            - 2 * np.einsum("ij,ij->i", info["单观测交叉积"], loading)
            + np.einsum("ij,jk,ik->i", loading, individual_second, loading)
        )
        variances = (
            firm_scale * firm_diagonal + date_scale * date_diagonal
            - individual_scale * individual_diagonal
        )
        for local_column, date_code in enumerate(info["日期编号"]):
            row = rows[int(date_code)]
            beta = float(info["日系数"][local_column])
            row["系数_百分点"] = beta * scale
            variance = float(variances[local_column])
            if not np.isfinite(beta):
                row["状态"] = "联合回归系数非有限"
            elif not np.isfinite(variance) or variance <= 0:
                row["状态"] = "双向聚类方差非正或无效，保留点估计"
            else:
                standard_error = float(np.sqrt(variance))
                row.update({
                    "标准误_百分点": standard_error * scale,
                    "置信下限_百分点": (beta - critical * standard_error) * scale,
                    "置信上限_百分点": (beta + critical * standard_error) * scale,
                    "显著性概率": float(2 * stats.t.sf(abs(beta / standard_error), degrees)),
                    "状态": "成功",
                })
        del y, x, z, residuals, local_scores, firm_local, date_local
        print_progress(
            "步骤3/3 每日系数推断（本阶段季度数进度）", completed, quarter_total,
            f"当前季度：{info['季度']}",
        )

    results = pd.DataFrame(rows).sort_values("日期").reset_index(drop=True)
    abnormal_counts = results.loc[results["状态"].ne("成功"), "状态"].value_counts()
    abnormal_summary = "、".join(
        f"{status}：{count}日" for status, count in abnormal_counts.items()
    ) or "无"
    print(
        f"联合日度估计完成：样本量{nobs:,}，可识别每日提示系数{len(candidate_codes):,}个，"
        f"删除固定效应单例{singleton_removed:,}条。",
        flush=True,
    )
    print(
        f"省略控制变量：{'、'.join(omitted_controls) or '无'}；非正常状态：{abnormal_summary}。",
        flush=True,
    )
    regression_data = reg.drop(columns=["_fe_firm_quarter", "_fe_date", "_cluster_firm"])
    return counts, regression_data, results
