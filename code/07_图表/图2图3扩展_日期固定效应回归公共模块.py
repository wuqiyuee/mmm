"""
收益动态回归的公共估计核心。

本文件只接收已经按某个事件期对齐好的横截面日度数据，不负责构造
累计收益或单日收益。固定效应、聚类协方差和百分比单位与当前
“图2_收益动态和长期反转_加入日期固定效应.py”保持一致：

1. 吸收被提示公司 × 年度季度固定效应；
2. 吸收事件起点交易日期固定效应；
3. 按被提示公司和事件起点交易日期双向聚类；
4. 使用有限样本修正；
5. 原始收益率以小数进入回归，结果以百分点输出。
"""

from __future__ import annotations

import gc
from collections.abc import Sequence

import numpy as np
import pandas as pd
from pyhdfe import create as create_hdfe
from scipy import sparse, stats
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.csgraph import connected_components


# 条件较差时，等价的矩阵乘法重排可能放大与原路径的舍入差异。
# 此阈值只决定方差的计算路径，不改变下方系数求解器的 1e12 阈值。
_SCALAR_COVARIANCE_MAX_CONDITION = 1e6
# 正常季度约有60个交易日；异常的大日期块继续使用原迭代算法。
_QUARTER_DIRECT_MAX_DATES = 128


def significance_stars(p_value: float) -> str:
    """根据双侧检验的 p 值添加显著性星号。"""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    data_name: str,
) -> None:
    """检查输入表是否包含指定字段。"""
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{data_name} 缺少必要变量：{missing}")


def drop_fe_singletons_iteratively(
    reg: pd.DataFrame,
    fe_code_cols: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """递归删除任一固定效应维度中的单例组。"""
    total_removed = 0

    while True:
        if reg.empty:
            raise ValueError("删除固定效应单例后没有剩余观测。")

        singleton = np.zeros(len(reg), dtype=bool)

        for column in fe_code_cols:
            codes = reg[column].to_numpy(dtype=np.int64, copy=False)
            if codes.min() < 0:
                raise ValueError(f"{column} 中存在无效的负数分组编号。")
            counts = np.bincount(codes)
            singleton |= counts[codes] == 1

        removed_now = int(singleton.sum())
        if removed_now == 0:
            break

        total_removed += removed_now
        reg = reg.loc[~singleton].copy()

    return reg, total_removed


def _cluster_score_meat(
    x: np.ndarray,
    residuals: np.ndarray,
    cluster_codes: np.ndarray,
    cluster_count: int,
) -> np.ndarray:
    """计算一个聚类维度的未缩放得分外积和。"""
    group_scores = np.empty(
        (cluster_count, x.shape[1]),
        dtype=np.float64,
    )

    for column in range(x.shape[1]):
        group_scores[:, column] = np.bincount(
            cluster_codes,
            weights=x[:, column] * residuals,
            minlength=cluster_count,
        )

    return group_scores.T @ group_scores


def _observation_score_meat(
    x: np.ndarray,
    residuals: np.ndarray,
    chunk_size: int = 250_000,
) -> np.ndarray:
    """按块计算公司×日期唯一交集组的得分外积和。"""
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)

    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        scores = x[start:stop] * residuals[start:stop, None]
        meat += scores.T @ scores

    return meat


def two_way_cluster_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    firm_codes: np.ndarray,
    date_codes: np.ndarray,
    xpx_inverse: np.ndarray,
) -> np.ndarray:
    """计算公司和事件起点日期双向聚类协方差。"""
    nobs, nvars = x.shape
    firm_count = int(firm_codes.max()) + 1
    date_count = int(date_codes.max()) + 1

    if firm_count < 2:
        raise ValueError("被提示公司聚类数少于2，无法计算聚类标准误。")
    if date_count < 2:
        raise ValueError("日期聚类数少于2，无法计算聚类标准误。")
    if nobs <= nvars:
        raise ValueError("回归观测数不大于解释变量数。")

    firm_meat = _cluster_score_meat(
        x,
        residuals,
        firm_codes,
        firm_count,
    )
    date_meat = _cluster_score_meat(
        x,
        residuals,
        date_codes,
        date_count,
    )
    intersection_meat = _observation_score_meat(x, residuals)

    common_scale = (nobs - 1) / (nobs - nvars)
    firm_scale = firm_count / (firm_count - 1) * common_scale
    date_scale = date_count / (date_count - 1) * common_scale
    intersection_scale = nobs / (nobs - 1) * common_scale

    meat = (
        firm_scale * firm_meat
        + date_scale * date_meat
        - intersection_scale * intersection_meat
    )
    covariance = xpx_inverse @ meat @ xpx_inverse
    return (covariance + covariance.T) / 2


def two_way_cluster_coefficient_variance(
    x: np.ndarray,
    residuals: np.ndarray,
    firm_codes: np.ndarray,
    date_codes: np.ndarray,
    xpx_inverse: np.ndarray,
    coefficient_position: int,
    *,
    xpx_condition: float | None = None,
) -> float:
    """只计算一个系数的双向聚类方差，沿用完整协方差的修正口径。

    调用方须保证公司×日期为唯一键，与完整协方差函数的前提一致。
    先将每行得分投影至所需系数，再聚类求和，避免为全部解释变量
    构造得分外积矩阵。分别使用逆矩阵的行和列，保留原矩阵乘法
    的双线性形式，不额外假定数值计算得到的逆矩阵严格对称。
    条件数过大或非有限时回退完整协方差，保留近奇异样本的原有
    数值计算结果。调用方可传入已有的交叉积矩阵条件数以免重复计算。
    """
    nobs, nvars = x.shape
    firm_count = int(firm_codes.max()) + 1
    date_count = int(date_codes.max()) + 1

    if firm_count < 2:
        raise ValueError("被提示公司聚类数少于2，无法计算聚类标准误。")
    if date_count < 2:
        raise ValueError("日期聚类数少于2，无法计算聚类标准误。")
    if nobs <= nvars:
        raise ValueError("回归观测数不大于解释变量数。")

    if xpx_condition is None:
        xpx_condition = float(np.linalg.cond(xpx_inverse))
    if (
        not np.isfinite(xpx_condition)
        or xpx_condition >= _SCALAR_COVARIANCE_MAX_CONDITION
    ):
        return float(
            two_way_cluster_covariance(
                x, residuals, firm_codes, date_codes, xpx_inverse
            )[coefficient_position, coefficient_position]
        )

    left_scores = x @ xpx_inverse[coefficient_position, :]
    right_scores = x @ xpx_inverse[:, coefficient_position]
    left_scores *= residuals
    right_scores *= residuals

    firm_left = np.bincount(
        firm_codes, weights=left_scores, minlength=firm_count
    )
    firm_right = np.bincount(
        firm_codes, weights=right_scores, minlength=firm_count
    )
    date_left = np.bincount(
        date_codes, weights=left_scores, minlength=date_count
    )
    date_right = np.bincount(
        date_codes, weights=right_scores, minlength=date_count
    )

    common_scale = (nobs - 1) / (nobs - nvars)
    firm_scale = firm_count / (firm_count - 1) * common_scale
    date_scale = date_count / (date_count - 1) * common_scale
    intersection_scale = nobs / (nobs - 1) * common_scale

    return float(
        firm_scale * (firm_left @ firm_right)
        + date_scale * (date_left @ date_right)
        - intersection_scale * (left_scores @ right_scores)
    )


def _residualize_regression_data_map(
    reg: pd.DataFrame,
    outcome: str,
    rhs_cols: Sequence[str],
    fe_ids: np.ndarray,
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """分批吸收两组固定效应，降低宽矩阵的峰值内存。"""
    if batch_size <= 0:
        raise ValueError("固定效应残差化批次大小必须为正数。")

    rhs_cols = list(rhs_cols)
    all_columns = [outcome, *rhs_cols]
    y_residualized = np.empty(len(reg), dtype=np.float64)
    x_residualized = np.empty(
        (len(reg), len(rhs_cols)),
        dtype=np.float64,
    )

    algorithm = create_hdfe(
        fe_ids,
        drop_singletons=False,
        compute_degrees=False,
        residualize_method="map",
        options={
            "tol": 1e-8,
            "transform": "symmetric",
            "acceleration": "cg",
        },
    )

    for start in range(0, len(all_columns), batch_size):
        stop = min(start + batch_size, len(all_columns))
        columns = all_columns[start:stop]
        values = reg[columns].to_numpy(dtype=np.float64, copy=True)
        residualized = algorithm.residualize(values)

        for local_position, global_position in enumerate(
            range(start, stop)
        ):
            if global_position == 0:
                y_residualized[:] = residualized[:, local_position]
            else:
                x_residualized[:, global_position - 1] = residualized[
                    :, local_position
                ]

        del values, residualized
        gc.collect()

    del algorithm
    gc.collect()
    return y_residualized, x_residualized


def _residualize_quarter_block(
    values: np.ndarray,
    firm_codes: np.ndarray,
    date_codes: np.ndarray,
) -> np.ndarray:
    """直接消除一个季度内的公司和日期固定效应。

    先公司去均值，再求解日期固定效应的舒尔补方程。每个日期
    连通分量设一个基准日期，保证不平衡或季内断连样本也适用；
    不加入岭项，不对投影矩阵进行任意秩截断。
    """
    nobs = len(values)
    firm_count = int(firm_codes.max()) + 1
    date_count = int(date_codes.max()) + 1
    if date_count > _QUARTER_DIRECT_MAX_DATES:
        raise ValueError("季度内日期过多，改用原固定效应迭代算法。")

    firm_nobs = np.bincount(firm_codes).astype(np.float64)
    date_nobs = np.bincount(date_codes).astype(np.float64)
    row_positions = np.arange(nobs, dtype=np.int64)
    ones = np.ones(nobs, dtype=np.float64)
    firm_sum = sparse.csr_matrix(
        (ones, (firm_codes, row_positions)),
        shape=(firm_count, nobs),
    )
    date_sum = sparse.csr_matrix(
        (ones, (date_codes, row_positions)),
        shape=(date_count, nobs),
    )
    counts = sparse.csr_matrix(
        (ones, (firm_codes, date_codes)),
        shape=(firm_count, date_count),
    )
    _, component_labels = connected_components(
        (counts.T @ counts).tocsr(), directed=False,
    )
    reference_dates = np.unique(component_labels, return_index=True)[1]
    free_dates = np.setdiff1d(
        np.arange(date_count), reference_dates, assume_unique=True,
    )
    schur = np.diag(date_nobs) - (
        counts.T @ counts.multiply(1.0 / firm_nobs[:, None])
    ).toarray()
    schur = (schur + schur.T) / 2

    firm_means = (firm_sum @ values) / firm_nobs[:, None]
    within_firm = values - firm_means[firm_codes]
    date_rhs = date_sum @ within_firm
    date_effects = np.zeros((date_count, values.shape[1]), dtype=np.float64)
    if free_dates.size:
        factor = cho_factor(
            schur[np.ix_(free_dates, free_dates)],
            lower=True, check_finite=False,
        )
        date_effects[free_dates] = cho_solve(
            factor, date_rhs[free_dates], check_finite=False,
        )
    firm_date_means = (counts @ date_effects) / firm_nobs[:, None]
    residualized = (
        within_firm
        - date_effects[date_codes]
        + firm_date_means[firm_codes]
    )
    if not np.isfinite(residualized).all():
        raise FloatingPointError("季度直接残差化得到非有限结果。")
    return residualized


def _residualize_regression_data_by_quarter(
    reg: pd.DataFrame,
    outcome: str,
    rhs_cols: Sequence[str],
    fe_ids: np.ndarray,
    yearquarter_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """逐季处理全部回归列，只分配一份全样本因变量和解释变量。"""
    if yearquarter_col not in reg or reg[yearquarter_col].isna().any():
        raise ValueError("季度直接残差化需要完整的年度季度字段。")
    if fe_ids.shape != (len(reg), 2) or (fe_ids < 0).any():
        raise ValueError("固定效应编号的维数或取值无效。")
    # 正常入口已连续编号。此上限避免异常编号导致巨大检查数组。
    if int(fe_ids.max()) >= len(reg):
        raise ValueError("固定效应编号不是预期的连续编号。")

    all_columns = [outcome, *rhs_cols]
    y_residualized = np.empty(len(reg), dtype=np.float64)
    x_residualized = np.empty((len(reg), len(rhs_cols)), dtype=np.float64)
    seen_firm_quarters = np.zeros(int(fe_ids[:, 0].max()) + 1, dtype=bool)
    seen_dates = np.zeros(int(fe_ids[:, 1].max()) + 1, dtype=bool)
    # groupby.indices 返回原始索引标签；回归表在 dropna 后索引可能不连续，
    # 这里必须根据季度编码生成位置数组，才能与 fe_ids 的行位置一致。
    quarter_codes = pd.factorize(reg[yearquarter_col], sort=False)[0]
    quarter_positions = {
        int(quarter): np.flatnonzero(quarter_codes == quarter)
        for quarter in np.unique(quarter_codes)
    }

    for positions in quarter_positions.values():
        block_ids = fe_ids[positions]
        firm_codes, global_firm_ids = pd.factorize(block_ids[:, 0], sort=False)
        date_codes, global_date_ids = pd.factorize(block_ids[:, 1], sort=False)
        # 两组固定效应均须嵌套在季度内，否则季度分块不等价。
        if (
            seen_firm_quarters[global_firm_ids].any()
            or seen_dates[global_date_ids].any()
        ):
            raise ValueError("固定效应组跨越多个季度，改用原迭代算法。")
        seen_firm_quarters[global_firm_ids] = True
        seen_dates[global_date_ids] = True
        values = reg.iloc[positions][all_columns].to_numpy(
            dtype=np.float64, copy=True,
        )
        residualized = _residualize_quarter_block(values, firm_codes, date_codes)
        y_residualized[positions] = residualized[:, 0]
        x_residualized[positions, :] = residualized[:, 1:]
        del values, residualized, block_ids

    return y_residualized, x_residualized


def residualize_regression_data(
    reg: pd.DataFrame,
    outcome: str,
    rhs_cols: Sequence[str],
    fe_ids: np.ndarray,
    batch_size: int = 4,
    *,
    yearquarter_col: str = "yearQuarter",
    residualization_method: str = "quarter_direct",
) -> tuple[np.ndarray, np.ndarray]:
    """默认逐季直接吸收固定效应，保留原迭代算法供复现和回退。

    quarter_direct（季度直接解）一次处理一个季度的全部回归列；
    map（交替投影）沿用原有收敛容差和列批次。季度无法分块、
    日期块过大或直接分解失败时，自动回退原有交替投影算法。
    batch_size（列批次大小）继续控制原算法及回退路径。
    """
    if batch_size <= 0:
        raise ValueError("固定效应残差化批次大小必须为正数。")
    if residualization_method not in {"quarter_direct", "map"}:
        raise ValueError("固定效应残差化方法必须为季度直接解或交替投影。")
    rhs_cols = list(rhs_cols)
    if residualization_method == "quarter_direct":
        try:
            direct_result = _residualize_regression_data_by_quarter(
                reg, outcome, rhs_cols, fe_ids, yearquarter_col,
            )
            # 直接解与MAP（交替投影）的浮点误差在病态解释变量矩阵中
            # 可能被系数放大；此时改走原路径，保留原结果的数值行为。
            xpx_condition = float(
                np.linalg.cond(direct_result[1].T @ direct_result[1])
            )
            if (
                not np.isfinite(xpx_condition)
                or xpx_condition >= _SCALAR_COVARIANCE_MAX_CONDITION
            ):
                del direct_result
                return _residualize_regression_data_map(
                    reg, outcome, rhs_cols, fe_ids, batch_size=batch_size,
                )
            return direct_result
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            # 离开异常处理后再分配原路径矩阵，释放失败分块的局部数组。
            pass
    return _residualize_regression_data_map(
        reg, outcome, rhs_cols, fe_ids, batch_size=batch_size,
    )


def run_date_fe_regression(
    data: pd.DataFrame,
    *,
    horizon: int,
    outcome: str,
    outcome_label: str,
    rhs_cols: Sequence[str],
    cue_col: str,
    firm_col: str = "StkcdRecall",
    date_col: str = "Trddt",
    yearquarter_col: str = "yearQuarter",
    residualization_batch_size: int = 4,
    residualization_method: str = "quarter_direct",
    full_covariance_diagnostics: bool = False,
) -> dict[str, object]:
    """估计一个相对交易日的双固定效应、双向聚类回归。

    默认只计算最终输出所需的核心提示系数方差。将
    full_covariance_diagnostics（完整协方差诊断）设为 True（真）
    可使用原完整协方差路径，并检查全部解释变量的方差是否为负；
    默认快速路径只执行核心提示系数的方差及标准误有效性检查。
    residualization_method（残差化方法）默认按季度直接求解；设为
    map（交替投影）可复现原固定效应吸收路径。
    """
    rhs_cols = list(rhs_cols)
    if cue_col not in rhs_cols:
        raise ValueError("核心提示变量必须包含在解释变量列表中。")

    needed = list(
        dict.fromkeys(
            [
                outcome,
                firm_col,
                date_col,
                yearquarter_col,
                *rhs_cols,
            ]
        )
    )
    require_columns(data, needed, "收益动态回归输入")

    reg = data[needed].dropna().copy()
    if reg.empty:
        raise ValueError(f"相对交易日 t+{horizon} 没有完整回归观测。")
    if reg.duplicated([firm_col, date_col]).any():
        raise ValueError("回归样本的被提示公司和事件起点日期不是唯一键。")

    reg["_fe_firm_yearquarter"] = (
        reg.groupby(
            [firm_col, yearquarter_col],
            observed=True,
            sort=False,
        )
        .ngroup()
        .astype("int64")
    )
    reg["_fe_date"] = pd.factorize(reg[date_col], sort=False)[0].astype(
        "int64"
    )

    reg, singleton_removed = drop_fe_singletons_iteratively(
        reg,
        ["_fe_firm_yearquarter", "_fe_date"],
    )

    # 删除单例后重新编号，保证固定效应和聚类编号连续。
    reg["_fe_firm_yearquarter"] = (
        reg.groupby(
            [firm_col, yearquarter_col],
            observed=True,
            sort=False,
        )
        .ngroup()
        .astype("int64")
    )
    reg["_fe_date"] = pd.factorize(reg[date_col], sort=False)[0].astype(
        "int64"
    )

    fe_ids = np.column_stack(
        [
            reg["_fe_firm_yearquarter"].to_numpy(
                dtype=np.int64,
                copy=False,
            ),
            reg["_fe_date"].to_numpy(dtype=np.int64, copy=False),
        ]
    )
    firm_codes = pd.factorize(reg[firm_col], sort=False)[0].astype(
        np.int64,
        copy=False,
    )
    date_codes = pd.factorize(reg[date_col], sort=False)[0].astype(
        np.int64,
        copy=False,
    )

    y_residualized, x_residualized = residualize_regression_data(
        reg,
        outcome,
        rhs_cols,
        fe_ids,
        batch_size=residualization_batch_size,
        yearquarter_col=yearquarter_col,
        residualization_method=residualization_method,
    )

    xpx = x_residualized.T @ x_residualized
    xpy = x_residualized.T @ y_residualized
    xpx_condition = float(np.linalg.cond(xpx))

    if np.isfinite(xpx_condition) and xpx_condition < 1e12:
        parameters = np.linalg.solve(xpx, xpy).reshape(-1)
        xpx_inverse = np.linalg.inv(xpx)
        coefficient_solver = "正规方程（条件数已检查）"
    else:
        parameters = np.linalg.lstsq(
            x_residualized,
            y_residualized,
            rcond=None,
        )[0].reshape(-1)
        xpx_inverse = np.linalg.pinv(xpx)
        coefficient_solver = "最小二乘分解"

    residuals = y_residualized - x_residualized @ parameters
    cue_position = rhs_cols.index(cue_col)
    if full_covariance_diagnostics:
        covariance = two_way_cluster_covariance(
            x_residualized,
            residuals,
            firm_codes,
            date_codes,
            xpx_inverse,
        )
        variances = np.diag(covariance)
        if (variances < -1e-12).any():
            raise ValueError(
                f"相对交易日 t+{horizon} 出现明显为负的协方差对角元素。"
            )
        cue_variance = float(variances[cue_position])
        del covariance, variances
    else:
        cue_variance = two_way_cluster_coefficient_variance(
            x_residualized,
            residuals,
            firm_codes,
            date_codes,
            xpx_inverse,
            cue_position,
            xpx_condition=xpx_condition,
        )

    if cue_variance < -1e-12:
        raise ValueError(
            f"相对交易日 t+{horizon} 出现明显为负的协方差对角元素。"
        )

    coefficient_raw = float(parameters[cue_position])
    standard_error_raw = float(np.sqrt(np.maximum(cue_variance, 0.0)))
    if not np.isfinite(standard_error_raw) or standard_error_raw <= 0:
        raise ValueError(f"相对交易日 t+{horizon} 的核心变量标准误无效。")

    t_value = coefficient_raw / standard_error_raw
    nobs = len(reg)
    absorbed_parameter_count = (
        reg["_fe_firm_yearquarter"].nunique()
        + reg["_fe_date"].nunique()
        - 1
    )
    residual_degrees = nobs - int(
        absorbed_parameter_count + x_residualized.shape[1]
    )
    if residual_degrees <= 0:
        raise ValueError(f"相对交易日 t+{horizon} 的残差自由度不为正。")

    p_value = float(
        2 * stats.t.sf(abs(t_value), residual_degrees)
    )
    original_y = reg[outcome].to_numpy(dtype=np.float64, copy=True)
    centered_y = original_y - original_y.mean()
    residual_sum_squares = float(residuals @ residuals)
    total_sum_squares = float(centered_y @ centered_y)
    if total_sum_squares <= 0:
        raise ValueError(f"相对交易日 t+{horizon} 的因变量没有有效变异。")
    r_squared = max(1 - residual_sum_squares / total_sum_squares, 0.0)

    coefficient = coefficient_raw * 100
    standard_error = standard_error_raw * 100
    row: dict[str, object] = {
        "窗口序号": horizon,
        "相对交易日": f"t+{horizon}",
        "因变量": outcome_label,
        "结果单位": "百分点",
        "线索系数": coefficient,
        "聚类标准误": standard_error,
        "置信区间下界": coefficient - 1.96 * standard_error,
        "置信区间上界": coefficient + 1.96 * standard_error,
        "置信区间口径": "逐点95%置信区间",
        "t值": t_value,
        "p值": p_value,
        "显著性": significance_stars(p_value),
        "控制变量": "是",
        "日期固定效应": "是",
        "日期固定效应基准": "事件起点交易日期",
        "被提示公司×年度季度固定效应": "是",
        "聚类方式": "被提示公司和事件起点交易日期双向聚类",
        "有限样本修正": "是",
        "回归样本量": nobs,
        "被提示公司聚类数": int(reg[firm_col].nunique()),
        "日期聚类数": int(reg[date_col].nunique()),
        "删除单例组观测数": singleton_removed,
        "解释变量矩阵条件数": xpx_condition,
        "系数求解方法": coefficient_solver,
        "R方": r_squared,
    }

    del (
        fe_ids,
        firm_codes,
        date_codes,
        y_residualized,
        x_residualized,
        xpx,
        xpy,
        xpx_inverse,
        parameters,
        residuals,
        original_y,
        centered_y,
        reg,
    )
    gc.collect()
    return row
