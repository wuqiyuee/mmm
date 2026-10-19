"""收益动态的公司季度固定效应估计核心。

本模块接收已经按相对交易日对齐的数据，仅吸收被提示公司×年度季度
固定效应。事件起点交易日期用于双向聚类，不作为固定效应吸收。
聚类协方差复用日期固定效应版本的有限样本修正；原始小数收益率
进入回归，核心系数、标准误和置信区间统一以百分点返回。
"""

from __future__ import annotations

import gc
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from 图2图3扩展_日期固定效应回归公共模块 import (
    drop_fe_singletons_iteratively,
    require_columns,
    significance_stars,
    two_way_cluster_coefficient_variance,
)


DATE_FE_BASE_DEFINITION = "不适用（未加入日期固定效应）"


def _demean_column(
    values: np.ndarray,
    group_codes: np.ndarray,
    group_counts: np.ndarray,
) -> np.ndarray:
    """对一列数值做公司季度组内去均值，不生成完整的组均值宽表。"""
    if not np.isfinite(values).all():
        raise ValueError("回归变量包含非有限数值。")
    group_sums = np.bincount(
        group_codes,
        weights=values,
        minlength=len(group_counts),
    )
    values -= (group_sums / group_counts)[group_codes]
    return values


def run_firm_quarter_fe_regression(
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
) -> dict[str, object]:
    """估计一个相对交易日的单固定效应、双向聚类回归。

    公司季度单例组在估计前删除；只有一条观测的日期仍可用于日期
    聚类，不因日期组大小而删除观测。完全吸收或共线的解释变量会
    明确报错，不自动改变调用方给定的控制变量集合。
    """
    rhs_cols = list(rhs_cols)
    if cue_col not in rhs_cols:
        raise ValueError("核心提示变量必须包含在解释变量列表中。")
    if len(rhs_cols) != len(set(rhs_cols)):
        raise ValueError("解释变量列表包含重复字段。")

    needed = list(
        dict.fromkeys(
            [outcome, firm_col, date_col, yearquarter_col, *rhs_cols]
        )
    )
    require_columns(data, needed, "收益动态回归输入")
    reg = data[needed].dropna().copy()
    if reg.empty:
        raise ValueError(f"相对交易日 t+{horizon} 没有完整回归观测。")
    if reg.duplicated([firm_col, date_col]).any():
        raise ValueError("回归样本的被提示公司和事件起点日期不是唯一键。")

    reg["_fe_firm_yearquarter"] = (
        reg.groupby([firm_col, yearquarter_col], observed=True, sort=False)
        .ngroup()
        .astype("int64")
    )
    reg, singleton_removed = drop_fe_singletons_iteratively(
        reg,
        ["_fe_firm_yearquarter"],
    )
    # 删除单例后重新编号，保证分组编号连续且组计数均为正数。
    reg["_fe_firm_yearquarter"] = (
        reg.groupby([firm_col, yearquarter_col], observed=True, sort=False)
        .ngroup()
        .astype("int64")
    )
    group_codes = reg["_fe_firm_yearquarter"].to_numpy(
        dtype=np.int64,
        copy=False,
    )
    group_counts = np.bincount(group_codes)
    nobs = len(reg)
    nvars = len(rhs_cols)
    residual_degrees = nobs - len(group_counts) - nvars
    if residual_degrees <= 0:
        raise ValueError(f"相对交易日 t+{horizon} 的残差自由度不为正。")

    firm_codes = pd.factorize(reg[firm_col], sort=False)[0].astype(
        np.int64,
        copy=False,
    )
    date_codes = pd.factorize(reg[date_col], sort=False)[0].astype(
        np.int64,
        copy=False,
    )
    original_y = reg[outcome].to_numpy(dtype=np.float64, copy=True)
    y_residualized = _demean_column(
        original_y.copy(),
        group_codes,
        group_counts,
    )
    x_residualized = np.empty((nobs, nvars), dtype=np.float64)
    absorbed_columns: list[str] = []
    rounding_scale = max(100, int(group_counts.max())) * np.finfo(float).eps
    for position, column in enumerate(rhs_cols):
        values = reg[column].to_numpy(dtype=np.float64, copy=True)
        original_norm = float(np.linalg.norm(values))
        values = _demean_column(values, group_codes, group_counts)
        if np.linalg.norm(values) <= rounding_scale * max(original_norm, 1.0):
            absorbed_columns.append(column)
        x_residualized[:, position] = values

    if cue_col in absorbed_columns:
        raise ValueError(
            f"相对交易日 t+{horizon} 的线索变量被公司季度固定效应完全吸收。"
        )
    if absorbed_columns:
        raise ValueError(
            f"相对交易日 t+{horizon} 的控制变量被公司季度固定效应完全吸收："
            f"{absorbed_columns}。请检查控制变量设定。"
        )

    xpx = x_residualized.T @ x_residualized
    xpy = x_residualized.T @ y_residualized
    # 先标准化各列再检查秩，避免量纲差异被误判为完全共线。
    column_norms = np.sqrt(np.diag(xpx))
    standardized_xpx = xpx / column_norms[:, None] / column_norms[None, :]
    if np.linalg.matrix_rank(standardized_xpx) < nvars:
        raise ValueError(
            f"相对交易日 t+{horizon} 的解释变量组内矩阵秩不足，"
            "无法分别识别全部系数。请检查共线控制变量。"
        )

    xpx_condition = float(np.linalg.cond(xpx))
    if np.isfinite(xpx_condition) and xpx_condition < 1e12:
        parameters = np.linalg.solve(xpx, xpy).reshape(-1)
        xpx_inverse = np.linalg.inv(xpx)
        coefficient_solver = "正规方程（条件数已检查）"
    else:
        parameters, _, estimated_rank, _ = np.linalg.lstsq(
            x_residualized,
            y_residualized,
            rcond=None,
        )
        if estimated_rank < nvars:
            raise ValueError(
                f"相对交易日 t+{horizon} 的解释变量组内矩阵数值秩不足。"
            )
        parameters = parameters.reshape(-1)
        xpx_inverse = np.linalg.pinv(xpx)
        coefficient_solver = "最小二乘分解"

    residuals = y_residualized - x_residualized @ parameters
    cue_position = rhs_cols.index(cue_col)
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
    if not np.isfinite(coefficient_raw):
        raise ValueError(f"相对交易日 t+{horizon} 的核心变量系数无效。")

    t_value = coefficient_raw / standard_error_raw
    p_value = float(2 * stats.t.sf(abs(t_value), residual_degrees))
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
        "日期固定效应": "否",
        "日期固定效应基准": DATE_FE_BASE_DEFINITION,
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
        group_codes,
        group_counts,
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
        standardized_xpx,
        reg,
    )
    gc.collect()
    return row
