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
from scipy import stats


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


def residualize_regression_data(
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
) -> dict[str, object]:
    """估计一个相对交易日的双固定效应、双向聚类回归。"""
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

    standard_errors = np.sqrt(np.maximum(variances, 0.0))
    cue_position = rhs_cols.index(cue_col)
    coefficient_raw = float(parameters[cue_position])
    standard_error_raw = float(standard_errors[cue_position])
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
        covariance,
        variances,
        standard_errors,
        original_y,
        centered_y,
        reg,
    )
    gc.collect()
    return row
