"""表5心理机制：规格映射与输入表头校验。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import time
from typing import Callable, Iterator, Sequence
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from linearmodels.iv import AbsorbingLS
from linearmodels.panel.utility import (
    AbsorbingEffectError,
    AbsorbingEffectWarning,
)
from scipy import sparse, stats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"
DEFAULT_RESULT_DIR = ROOT / "中间结果/08_回归结果"
DEFAULT_IMAGE_DIR = ROOT / "图表结果/论文复现/表格"
DEFAULT_CACHE_PATH = ROOT / "work/表5低内存缓存/表5所需字段.parquet"
DEFAULT_CHUNK_SIZE = 250_000
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
A4_LANDSCAPE_FIGSIZE = (297 / 25.4, 210 / 25.4)


@dataclass(frozen=True)
class OutcomeSpec:
    panel: str
    column: str
    label: str


@dataclass(frozen=True)
class MechanismSpec:
    order: int
    key: str
    label: str
    first_90: str
    first_5090: str
    second_90: str | None
    second_5090: str | None
    comparison_mode: str
    source_note: str


@dataclass(frozen=True)
class ComparisonPair:
    size_key: str
    size_label: str
    first_col: str
    second_col: str
    first_label: str
    second_label: str


class NonEstimableModelError(ValueError):
    """模型因缺少有效变异或识别组而无法估计。"""


OUTCOME_SPECS = (
    OutcomeSpec("A", "chAdjreturn_t_tp1_Recall_w", "[t, t+1]"),
    OutcomeSpec("B", "chAdjreturn_t_tp10_Recall_w", "[t, t+10]"),
)

OWN_EA_COLS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)
DISCLOSURE_COLS = [
    "ownMajorDisclosure",
    "ownPerformanceForecast",
]
MEDIA_CONTROL_COLS = [
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]

_SUE_NOTES = {
    1: "历史未调整每股收益",
    2: "历史调整每股收益",
    3: "分析师预测中位数",
}
_SHARED_MECHANISM_ORDERS = (1, 2, 3, 4, 9)
_SUE_MECHANISM_ORDERS = (5, 6, 7, 8)

_BASELINE_90 = "quarterLag1size90plus"
_BASELINE_5090 = "quarterLag1size50_90"
_INTERFERENCE_FIELDS = (
    "lowInterferenceEWLag1S90p",
    "lowInterferenceEWLag1S5090",
    "highInterferenceEWLag1S90p",
    "highInterferenceEWLag1S5090",
)
_SIZE_GROUPS = (
    ("90plus", "市值前10%", _BASELINE_90),
    ("50_90", "市值50%至90%", _BASELINE_5090),
)


def build_mechanism_specs(sue_id: int) -> tuple[MechanismSpec, ...]:
    """返回指定 SUE（盈余惊喜）口径的九类表5心理机制。"""
    if type(sue_id) is not int or sue_id not in (1, 2, 3):
        raise ValueError("SUE口径只能是1、2或3。")

    shared_specs = (
        MechanismSpec(
            1,
            "high_total_volume",
            "异常总成交量",
            "lag1hiavolsize90plus",
            "lag1hiavolsize50_90",
            None,
            None,
            "strong_vs_rest",
            "共同高异常总成交量强关联提示，与同规模其他关联提示比较。",
        ),
        MechanismSpec(
            2,
            "high_small_trade_volume",
            "异常散户成交量（A股小单替代）",
            "lag1hiavolSsize90plus",
            "lag1hiavolSsize50_90",
            None,
            None,
            "strong_vs_rest",
            "A股小单异常成交量为原文零售交易量口径的中国替代；"
            "强关联提示与同规模其他关联提示比较。",
        ),
        MechanismSpec(
            3,
            "high_media",
            "媒体覆盖",
            "lag1hiAEVsize90plus",
            "lag1hiAEVsize50_90",
            None,
            None,
            "strong_vs_rest",
            "共同高媒体关注强关联提示，与同规模其他关联提示比较。",
        ),
        MechanismSpec(
            4,
            "top_mover",
            "价格异动",
            "lag1topMoversize90plus",
            "lag1topMoversize50_90",
            None,
            None,
            "strong_vs_rest",
            "共同TopMover强关联提示，与同规模其他关联提示比较。",
        ),
    )
    sue_specs = (
        MechanismSpec(
            5,
            "positive_sue",
            "正向盈余惊喜",
            f"lag1sue{sue_id}possize90plus",
            f"lag1sue{sue_id}possize50_90",
            None,
            None,
            "strong_vs_rest",
            f"共同正向SUE{sue_id}强关联提示，与同规模其他关联提示比较。",
        ),
        MechanismSpec(
            6,
            "negative_sue",
            "负向盈余惊喜",
            f"lag1sue{sue_id}negsize90plus",
            f"lag1sue{sue_id}negsize50_90",
            None,
            None,
            "strong_vs_rest",
            f"共同负向SUE{sue_id}强关联提示，与同规模其他关联提示比较。",
        ),
        MechanismSpec(
            7,
            "extreme_positive_sue",
            "极端正向盈余惊喜",
            f"lag1hi_sue{sue_id}_possize90plus",
            f"lag1hi_sue{sue_id}_possize50_90",
            None,
            None,
            "strong_vs_rest",
            f"共同极端正向SUE{sue_id}强关联提示，与同规模其他关联提示比较。",
        ),
        MechanismSpec(
            8,
            "extreme_negative_sue",
            "极端负向盈余惊喜",
            f"lag1hi_sue{sue_id}_negsize90plus",
            f"lag1hi_sue{sue_id}_negsize50_90",
            None,
            None,
            "strong_vs_rest",
            f"共同极端负向SUE{sue_id}强关联提示，与同规模其他关联提示比较。",
        ),
    )
    interference_spec = MechanismSpec(
        9,
        "interference",
        "干扰程度",
        _INTERFERENCE_FIELDS[0],
        _INTERFERENCE_FIELDS[1],
        _INTERFERENCE_FIELDS[2],
        _INTERFERENCE_FIELDS[3],
        "low_vs_high",
        "按年份、等权共同公告公司数量二分；"
        "低干扰界定为强关联提示，高干扰界定为其他关联提示。",
    )
    return shared_specs + sue_specs + (interference_spec,)


def required_input_columns() -> list[str]:
    """返回表5全部模型所需列，供表头检查和后续紧凑读取共同使用。"""
    columns = [
        "StkcdRecall",
        "Trddt",
        "year",
        "yearQuarter",
        *(outcome.column for outcome in OUTCOME_SPECS),
        *OWN_EA_COLS,
        *DISCLOSURE_COLS,
        *MEDIA_CONTROL_COLS,
        _BASELINE_90,
        _BASELINE_5090,
    ]
    for sue_id in (1, 2, 3):
        for spec in build_mechanism_specs(sue_id):
            columns.extend((spec.first_90, spec.first_5090))
            if spec.second_90 is not None:
                columns.append(spec.second_90)
            if spec.second_5090 is not None:
                columns.append(spec.second_5090)
    return list(dict.fromkeys(columns))


def _validate_binary_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    """校验机制比较中每个核心提示字段只能为二元值或缺失。"""
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError("机制比较缺少以下核心字段：" + "、".join(missing))

    for column in columns:
        values = data[column]
        invalid = values.notna() & ~values.isin((0, 1))
        if invalid.any():
            raise ValueError(f"核心字段{column}只能取0、1或缺失。")


def _comparison_column_name(spec: MechanismSpec, size_key: str, role: str) -> str:
    """返回不会在不同机制或规模组之间覆盖的比较变量名。"""
    return f"table5_{spec.key}_{size_key}_{role}"


def _comparison_pairs_for_spec(
    spec: MechanismSpec,
) -> tuple[ComparisonPair, ComparisonPair]:
    """返回机制对应的两组比较定义，不依赖实际数据。"""
    if spec.comparison_mode == "strong_vs_rest":
        role_names = ("strong", "rest")
        role_labels = ("强关联提示", "其他关联提示")
    elif spec.comparison_mode == "low_vs_high":
        role_names = ("low", "high")
        role_labels = ("强关联提示", "其他关联提示")
    else:
        raise ValueError(f"不支持的机制比较方式：{spec.comparison_mode}。")

    pairs = [
        ComparisonPair(
            size_key,
            size_label,
            _comparison_column_name(spec, size_key, role_names[0]),
            _comparison_column_name(spec, size_key, role_names[1]),
            role_labels[0],
            role_labels[1],
        )
        for size_key, size_label, _ in _SIZE_GROUPS
    ]
    return pairs[0], pairs[1]


def _nullable_binary(values: pd.Series) -> pd.Series:
    """将已校验的提示字段规范为可保留缺失值的二元序列。"""
    return values.astype("Int64")


def materialize_comparisons(
    data: pd.DataFrame,
    spec: MechanismSpec,
) -> tuple[pd.DataFrame, tuple[ComparisonPair, ComparisonPair]]:
    """构造一类机制的两个规模比较变量，并在回归前校验二元和子集口径。"""
    output = data.copy()
    pairs: list[ComparisonPair] = []

    if spec.comparison_mode == "strong_vs_rest":
        for size_key, size_label, baseline_col in _SIZE_GROUPS:
            strong_col = spec.first_90 if size_key == "90plus" else spec.first_5090
            _validate_binary_columns(output, (baseline_col, strong_col))

            baseline = output[baseline_col]
            strong = output[strong_col]
            invalid_subset = strong.eq(1) & baseline.eq(0)
            if invalid_subset.any():
                raise ValueError(
                    f"强关联提示字段{strong_col}为1时，对应基准提示字段"
                    f"{baseline_col}不能为0。"
                )

            first_col = _comparison_column_name(spec, size_key, "strong")
            second_col = _comparison_column_name(spec, size_key, "rest")
            output[first_col] = _nullable_binary(strong)

            rest = pd.Series(pd.NA, index=output.index, dtype="Int64")
            known = baseline.notna() & strong.notna()
            rest.loc[known] = (
                (baseline.loc[known] == 1) & (strong.loc[known] == 0)
            ).astype("Int64")
            output[second_col] = rest
            pairs.append(
                ComparisonPair(
                    size_key,
                    size_label,
                    first_col,
                    second_col,
                    "强关联提示",
                    "其他关联提示",
                )
            )
    elif spec.comparison_mode == "low_vs_high":
        for size_key, size_label, _ in _SIZE_GROUPS:
            low_col = spec.first_90 if size_key == "90plus" else spec.first_5090
            high_col = spec.second_90 if size_key == "90plus" else spec.second_5090
            if high_col is None:
                raise ValueError("干扰机制缺少高干扰提示字段。")
            _validate_binary_columns(output, (low_col, high_col))

            first_col = _comparison_column_name(spec, size_key, "low")
            second_col = _comparison_column_name(spec, size_key, "high")
            output[first_col] = _nullable_binary(output[low_col])
            output[second_col] = _nullable_binary(output[high_col])
            pairs.append(
                ComparisonPair(
                    size_key,
                    size_label,
                    first_col,
                    second_col,
                    "强关联提示",
                    "其他关联提示",
                )
            )
    else:
        raise ValueError(f"不支持的机制比较方式：{spec.comparison_mode}。")

    return output, (pairs[0], pairs[1])


def drop_fe_singletons_iteratively(
    reg: pd.DataFrame,
    fe_code_cols: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """递归删除固定效应单例组，直至各固定效应维度都不再产生单例。"""
    if not fe_code_cols:
        return reg, 0

    total_removed = 0
    while True:
        singleton = np.zeros(len(reg), dtype=bool)
        for column in fe_code_cols:
            codes = reg[column].to_numpy(dtype=np.int64, copy=False)
            counts = np.bincount(codes)
            singleton |= counts[codes] == 1

        removed_now = int(singleton.sum())
        if removed_now == 0:
            return reg, total_removed

        total_removed += removed_now
        reg = reg.loc[~singleton].copy()


def _mechanism_source_columns(spec: MechanismSpec) -> list[str]:
    """返回当前机制构造四个比较变量实际需要读取的源字段。"""
    if spec.comparison_mode == "strong_vs_rest":
        return [
            _BASELINE_90,
            _BASELINE_5090,
            spec.first_90,
            spec.first_5090,
        ]
    if spec.comparison_mode == "low_vs_high":
        columns = [
            spec.first_90,
            spec.first_5090,
            spec.second_90,
            spec.second_5090,
        ]
        if any(column is None for column in columns):
            raise ValueError(f"{spec.label}机制缺少低高组比较字段。")
        return [str(column) for column in columns]
    raise ValueError(f"不支持的机制比较方式：{spec.comparison_mode}。")


def _regression_context(outcome: OutcomeSpec, spec: MechanismSpec) -> str:
    """返回回归校验错误中统一使用的中文模型定位信息。"""
    return (
        f"{spec.label}机制、收益窗口{outcome.label}、"
        f"因变量{outcome.column}"
    )


def _check_core_variation(
    reg: pd.DataFrame,
    core_cols: Sequence[str],
    outcome: OutcomeSpec,
    spec: MechanismSpec,
    fe_col: str | None = None,
) -> None:
    """校验四个核心变量有总体变异，并且不会被固定效应完全吸收。"""
    for column in core_cols:
        if reg[column].nunique(dropna=True) < 2:
            raise NonEstimableModelError(
                f"{_regression_context(outcome, spec)}："
                f"核心变量{column}没有有效变异，无法估计。"
            )
        if fe_col is not None:
            has_within_group_variation = (
                reg.groupby(fe_col, observed=True, sort=False)[column]
                .nunique(dropna=True)
                .gt(1)
                .any()
            )
            if not has_within_group_variation:
                raise NonEstimableModelError(
                    f"{_regression_context(outcome, spec)}："
                    f"核心变量{column}被公司×年度季度固定效应"
                    "完全吸收，无法估计。"
                )


def significance_stars(p_value: float) -> str:
    """按双侧检验的 p 值返回常用显著性星号。"""
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def format_coefficient(
    coefficient: float,
    standard_error: float,
    p_value: float,
    digits: int = 3,
) -> str:
    """把百分点系数、显著性和括号内聚类标准误排成一个表格单元。"""
    if pd.isna(coefficient) or pd.isna(standard_error):
        return ""
    return (
        f"{coefficient:.{digits}f}{significance_stars(p_value)}\n"
        f"({standard_error:.{digits}f})"
    )


def format_p_value(p_value: float) -> str:
    """紧凑显示系数相等检验的 p 值。"""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def run_mechanism_regression(
    data: pd.DataFrame,
    outcome: OutcomeSpec,
    spec: MechanismSpec,
    control_cols: Sequence[str],
    *,
    date_fixed_effects: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """在同一模型中估计四个机制变量，并返回轻量主结果与系数明细。"""
    context = _regression_context(outcome, spec)
    controls = list(dict.fromkeys(control_cols))
    if "const" in controls:
        raise ValueError(
            f"{context}：控制变量不能命名为const，以免覆盖显式常数项。"
        )

    try:
        source_cols = _mechanism_source_columns(spec)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context}：{exc}") from exc
    needed = list(
        dict.fromkeys(
            [
                outcome.column,
                "StkcdRecall",
                "Trddt",
                "yearQuarter",
                *controls,
                *source_cols,
            ]
        )
    )
    missing = [column for column in needed if column not in data.columns]
    if missing:
        raise ValueError(
            f"{_regression_context(outcome, spec)}缺少以下回归字段："
            + "、".join(missing)
        )

    # 性能约束：仅复制当前机制的四个源字段来构造比较变量，
    # 避免在全样本上同时保留两份控制变量宽表。
    compact = data.loc[:, source_cols]
    try:
        materialized, pairs = materialize_comparisons(compact, spec)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context}：{exc}") from exc
    core_cols = [
        column
        for pair in pairs
        for column in (pair.first_col, pair.second_col)
    ]
    model_cols = [
        outcome.column,
        "StkcdRecall",
        "Trddt",
        "yearQuarter",
        *controls,
    ]
    reg = pd.concat(
        [
            data.loc[:, model_cols],
            materialized.loc[:, core_cols],
        ],
        axis=1,
        copy=False,
    ).dropna().copy()
    del compact, materialized
    if reg.empty:
        raise NonEstimableModelError(
            f"{_regression_context(outcome, spec)}：统一删缺失后没有可用观测。"
        )
    _check_core_variation(reg, core_cols, outcome, spec)

    reg["_fe_firm_yearquarter"] = (
        reg.groupby(
            ["StkcdRecall", "yearQuarter"],
            observed=True,
            sort=False,
        )
        .ngroup()
        .astype("int64")
    )
    fe_code_cols = ["_fe_firm_yearquarter"]
    if date_fixed_effects:
        reg["_fe_date"] = pd.factorize(
            reg["Trddt"],
            sort=False,
        )[0].astype("int64")
        fe_code_cols.append("_fe_date")
    reg, singleton_removed = drop_fe_singletons_iteratively(
        reg,
        fe_code_cols,
    )
    if reg.empty:
        raise NonEstimableModelError(
            f"{_regression_context(outcome, spec)}：递归删除固定效应单例后"
            "没有可用观测。"
        )

    _check_core_variation(
        reg,
        core_cols,
        outcome,
        spec,
        fe_col="_fe_firm_yearquarter",
    )
    fe_count = int(reg["_fe_firm_yearquarter"].nunique())
    firm_count = int(reg["StkcdRecall"].nunique())
    date_count = int(reg["Trddt"].nunique())
    if fe_count < 2:
        raise NonEstimableModelError(
            f"{_regression_context(outcome, spec)}："
            "被提示公司×年度季度固定效应至少需要两个组。"
        )
    if firm_count < 2:
        raise NonEstimableModelError(
            f"{_regression_context(outcome, spec)}：被提示公司聚类至少需要两个组。"
        )
    if date_count < 2:
        raise NonEstimableModelError(
            f"{_regression_context(outcome, spec)}：交易日期聚类至少需要两个组。"
        )

    absorb_columns = {
        "firm_yearquarter": pd.Categorical(
            reg["_fe_firm_yearquarter"]
        )
    }
    if date_fixed_effects:
        absorb_columns["date"] = pd.Categorical(reg["_fe_date"])
    absorb = pd.DataFrame(absorb_columns, index=reg.index)
    clusters = pd.DataFrame(
        {
            "firm": pd.factorize(reg["StkcdRecall"], sort=False)[0],
            "date": pd.factorize(reg["Trddt"], sort=False)[0],
        },
        index=reg.index,
    )
    x_cols = [*core_cols, *controls]
    try:
        exog = reg[x_cols].astype("float64")
        dependent = reg[outcome.column].astype("float64")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{context}：数值转换失败：{exc}") from exc
    exog.insert(0, "const", 1.0)

    try:
        model = AbsorbingLS(
            dependent=dependent,
            exog=exog,
            absorb=absorb,
            drop_absorbed=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AbsorbingEffectWarning)
            result = model.fit(
                cov_type="clustered",
                clusters=clusters,
                debiased=True,
            )
    except (
        AbsorbingEffectError,
        ValueError,
        TypeError,
        OverflowError,
        np.linalg.LinAlgError,
    ) as exc:
        raise ValueError(f"{context}：模型拟合失败：{exc}") from exc

    params = result.params
    unestimated_core = [
        column for column in core_cols if column not in params.index
    ]
    if unestimated_core:
        raise NonEstimableModelError(
            f"{_regression_context(outcome, spec)}：核心变量"
            + "、".join(unestimated_core)
            + "与其他解释变量存在完全共线性，因而未估计。"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        standard_errors = result.std_errors
        t_statistics = result.tstats
        p_values = result.pvalues
    inference_fields = (
        ("系数", params),
        ("聚类标准误", standard_errors),
        ("t值", t_statistics),
        ("p值", p_values),
    )
    for field_label, values in inference_fields:
        invalid_columns = [
            column
            for column in core_cols
            if not np.isfinite(float(values[column]))
        ]
        if invalid_columns:
            raise NonEstimableModelError(
                f"{_regression_context(outcome, spec)}：核心变量"
                + "、".join(invalid_columns)
                + f"的{field_label}为非有限值，无法进行可靠推断。"
            )

    diagnostics: dict[str, object] = {
        "面板": outcome.panel,
        "收益窗口": outcome.label,
        "因变量": outcome.column,
        "机制序号": spec.order,
        "机制键": spec.key,
        "机制": spec.label,
        "控制变量": "是" if controls else "否",
        "控制变量数": len(controls),
        "显式常数项": "是",
        "被提示公司×年度季度固定效应": "是",
        "日期固定效应": "是" if date_fixed_effects else "否",
        "聚类方式": "被提示公司和交易日期双向聚类",
        "有限样本自由度修正": "是",
        "回归样本量": int(result.nobs),
        "被提示公司聚类数": firm_count,
        "日期聚类数": date_count,
        "固定效应组数": fe_count,
        "日期固定效应组数": date_count if date_fixed_effects else 0,
        "删除单例组观测数": int(singleton_removed),
        "R方": float(result.rsquared),
    }

    variable_metadata = {
        pair.first_col: (pair, pair.first_label, 1)
        for pair in pairs
    }
    variable_metadata.update(
        {
            pair.second_col: (pair, pair.second_label, 2)
            for pair in pairs
        }
    )
    coefficient_rows: list[dict[str, object]] = []
    for variable_order, variable in enumerate(core_cols, start=1):
        pair, variable_label, within_pair_order = variable_metadata[variable]
        beta = float(params[variable])
        standard_error = float(standard_errors[variable])
        t_value = float(t_statistics[variable])
        p_value = float(p_values[variable])
        coefficient_rows.append(
            {
                **diagnostics,
                "规模组": pair.size_key,
                "规模组标签": pair.size_label,
                "变量": variable,
                "变量顺序": variable_order,
                "组内变量顺序": within_pair_order,
                "变量标签": variable_label,
                "变量类型": "机制核心变量",
                "估计状态": "已估计",
                "系数": beta,
                "聚类标准误": standard_error,
                "系数（百分点）": beta * 100,
                "聚类标准误（百分点）": standard_error * 100,
                "t值": t_value,
                "p值": p_value,
                "显著性": significance_stars(p_value),
            }
        )

    main_rows: list[dict[str, object]] = []
    for pair in pairs:
        wald = result.wald_test(
            formula=f"{pair.first_col} = {pair.second_col}"
        )
        first_beta = float(params[pair.first_col])
        second_beta = float(params[pair.second_col])
        first_standard_error = float(standard_errors[pair.first_col])
        second_standard_error = float(standard_errors[pair.second_col])
        first_p_value = float(p_values[pair.first_col])
        second_p_value = float(p_values[pair.second_col])
        coefficient_difference = first_beta - second_beta
        main_rows.append(
            {
                **diagnostics,
                "规模组": pair.size_key,
                "规模组标签": pair.size_label,
                "第一组变量": pair.first_col,
                "第一组标签": pair.first_label,
                "第二组变量": pair.second_col,
                "第二组标签": pair.second_label,
                "第一组系数": first_beta,
                "第二组系数": second_beta,
                "第一组聚类标准误": first_standard_error,
                "第二组聚类标准误": second_standard_error,
                "第一组聚类标准误（百分点）": first_standard_error * 100,
                "第二组聚类标准误（百分点）": second_standard_error * 100,
                "第一组p值": first_p_value,
                "第二组p值": second_p_value,
                "系数差": coefficient_difference,
                "系数差（百分点）": coefficient_difference * 100,
                "Wald检验式": f"{pair.first_col} = {pair.second_col}",
                "Wald统计量": float(np.asarray(wald.stat).squeeze()),
                "Wald自由度": int(wald.df),
                "Wald p值": float(wald.pval),
            }
        )

    return main_rows, coefficient_rows


def _iter_streaming_model_batches(
    cache_path: str | Path,
    outcome: OutcomeSpec,
    spec: MechanismSpec,
    control_cols: Sequence[str],
    *,
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    """逐批读取并构造一个模型的完整案例样本，不把整表载入内存。"""
    context = _regression_context(outcome, spec)
    source_cols = _mechanism_source_columns(spec)
    needed = model_required_columns(
        outcome,
        spec,
        control_cols,
    )
    cache = Path(cache_path)
    try:
        parquet = pq.ParquetFile(cache)
    except (OSError, pa.ArrowInvalid) as exc:
        raise ValueError(f"{context}：无法读取低内存缓存：{exc}") from exc
    available = set(parquet.schema_arrow.names)
    missing = [column for column in needed if column not in available]
    if missing:
        raise ValueError(
            f"{context}：低内存缓存缺少以下回归字段："
            + "、".join(missing)
        )

    model_cols = [
        outcome.column,
        "StkcdRecall",
        "Trddt",
        "yearQuarter",
        *control_cols,
    ]
    try:
        batches = parquet.iter_batches(
            batch_size=batch_size,
            columns=needed,
        )
        for arrow_batch in batches:
            data = arrow_batch.to_pandas()
            data["Trddt"] = pd.to_datetime(data["Trddt"], errors="raise")
            data["StkcdRecall"] = data["StkcdRecall"].astype("string")
            data["yearQuarter"] = data["yearQuarter"].astype("string")
            try:
                materialized, pairs = materialize_comparisons(
                    data.loc[:, source_cols],
                    spec,
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{context}：{exc}") from exc
            core_cols = [
                column
                for pair in pairs
                for column in (pair.first_col, pair.second_col)
            ]
            reg = pd.concat(
                [
                    data.loc[:, model_cols],
                    materialized.loc[:, core_cols],
                ],
                axis=1,
                copy=False,
            ).dropna()
            if not reg.empty:
                yield reg
    except (
        OSError,
        ValueError,
        TypeError,
        OverflowError,
        pa.ArrowInvalid,
    ):
        raise


def _streaming_numeric_frame(
    reg: pd.DataFrame,
    numeric_cols: Sequence[str],
    context: str,
) -> pd.DataFrame:
    """把一个已删缺失分块转换为双精度数值矩阵。"""
    try:
        return reg.loc[:, numeric_cols].astype("float64")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{context}：数值转换失败：{exc}") from exc


def _streaming_fixed_effect_positions(
    reg: pd.DataFrame,
    group_index: pd.MultiIndex,
    date_index: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """把分块中的两组固定效应键映射到全样本统计位置。"""
    group_keys = pd.MultiIndex.from_arrays(
        [
            reg["StkcdRecall"].astype("string").to_numpy(dtype=object),
            reg["yearQuarter"].astype("string").to_numpy(dtype=object),
        ],
        names=["StkcdRecall", "yearQuarter"],
    )
    group_positions = group_index.get_indexer(group_keys)
    date_positions = date_index.get_indexer(pd.DatetimeIndex(reg["Trddt"]))
    if np.any(group_positions < 0) or np.any(date_positions < 0):
        raise RuntimeError("分块固定效应键与首遍全样本统计不一致。")
    return (
        group_positions.astype(np.int32, copy=False),
        date_positions.astype(np.int32, copy=False),
    )


def _transform_streaming_batch(
    reg: pd.DataFrame,
    numeric_cols: Sequence[str],
    eligible_index: pd.MultiIndex,
    group_means: np.ndarray,
    grand_means: np.ndarray,
    context: str,
    retain: np.ndarray | None = None,
    *,
    date_index: pd.DatetimeIndex | None = None,
    date_effects: np.ndarray | None = None,
    group_date_effect_means: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """删除固定效应单例并生成与AbsorbingLS相同的加回均值变换。"""
    keys = pd.MultiIndex.from_arrays(
        [
            reg["StkcdRecall"].astype("string").to_numpy(dtype=object),
            reg["yearQuarter"].astype("string").to_numpy(dtype=object),
        ],
        names=["StkcdRecall", "yearQuarter"],
    )
    positions = eligible_index.get_indexer(keys)
    keep = positions >= 0
    if not np.any(keep):
        empty_x = np.empty((0, len(numeric_cols)))
        return (
            empty_x,
            np.empty(0),
            np.empty(0, dtype=object),
            np.empty(0, dtype=np.int64),
            np.empty(0),
        )

    kept = reg.loc[keep]
    raw = _streaming_numeric_frame(
        kept,
        numeric_cols,
        context,
    ).to_numpy()
    positions = positions[keep]
    transformed = raw - group_means[positions]
    if date_effects is not None or group_date_effect_means is not None:
        if (
            date_index is None
            or date_effects is None
            or group_date_effect_means is None
        ):
            raise ValueError("日期固定效应变换参数必须同时提供。")
        date_positions = date_index.get_indexer(
            pd.DatetimeIndex(kept["Trddt"])
        )
        if np.any(date_positions < 0):
            raise RuntimeError("分块中出现未纳入日期固定效应统计的交易日期。")
        transformed = (
            transformed
            - date_effects[date_positions]
            + group_date_effect_means[positions]
        )
    transformed = transformed + grand_means
    y_tilde = transformed[:, 0]
    x_tilde = np.empty(
        (len(transformed), len(numeric_cols)),
        dtype=np.float64,
    )
    x_tilde[:, 0] = 1.0
    x_tilde[:, 1:] = transformed[:, 1:]
    if retain is not None:
        x_tilde = x_tilde[:, retain]
    firms = (
        kept["StkcdRecall"]
        .astype("string")
        .to_numpy(dtype=object)
    )
    dates = (
        kept["Trddt"]
        .to_numpy(dtype="datetime64[ns]", copy=False)
        .astype(np.int64, copy=False)
    )
    return x_tilde, y_tilde, firms, dates, raw[:, 0]


def _update_tsqr(
    current_r: np.ndarray | None,
    block: np.ndarray,
) -> np.ndarray:
    """用分层QR累计一个分块，避免保存完整解释变量矩阵。"""
    if block.size == 0:
        if current_r is None:
            raise ValueError("空分块不能初始化分层QR。")
        return current_r
    block_r = np.linalg.qr(block, mode="reduced")[1]
    if current_r is None:
        return block_r
    return np.linalg.qr(
        np.vstack([current_r, block_r]),
        mode="reduced",
    )[1]


def _retained_columns_from_tsqr(
    qr_r: np.ndarray,
    x_columns: Sequence[str],
) -> np.ndarray:
    """镜像linearmodels对被吸收或完全共线列的保留规则。"""
    column_count = len(x_columns)
    x_r = qr_r[:column_count, :column_count]
    if np.linalg.matrix_rank(x_r) == column_count:
        return np.arange(column_count, dtype=np.int64)

    xpx = x_r.T @ x_r
    eigenvalues = np.linalg.eigvalsh(xpx)
    if float(eigenvalues.max()) == 0.0:
        return np.empty(0, dtype=np.int64)
    tolerance = (
        float(eigenvalues.max())
        * column_count
        * np.finfo(np.float64).eps
    )
    absorbed_count = int(np.count_nonzero(eigenvalues < tolerance))
    diagonal = np.abs(np.diag(x_r))
    threshold = np.sort(diagonal)[absorbed_count]
    drop = set(np.flatnonzero(diagonal < threshold).tolist())
    return np.asarray(
        [
            index
            for index in range(column_count)
            if index not in drop
        ],
        dtype=np.int64,
    )


def _merge_cluster_scores(
    store: dict[object, np.ndarray],
    keys: np.ndarray,
    scores: np.ndarray,
) -> None:
    """按聚类键合并一个分块的得分和。"""
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    grouped = np.empty(
        (len(unique_keys), scores.shape[1]),
        dtype=np.float64,
    )
    for column in range(scores.shape[1]):
        grouped[:, column] = np.bincount(
            inverse,
            weights=scores[:, column],
            minlength=len(unique_keys),
        )
    for key, values in zip(unique_keys.tolist(), grouped, strict=True):
        previous = store.get(key)
        if previous is None:
            store[key] = values.copy()
        else:
            previous += values


def _streaming_result_rows(
    *,
    outcome: OutcomeSpec,
    spec: MechanismSpec,
    controls: Sequence[str],
    pairs: Sequence[ComparisonPair],
    core_cols: Sequence[str],
    params: pd.Series,
    covariance: pd.DataFrame,
    standard_errors: pd.Series,
    t_statistics: pd.Series,
    p_values: pd.Series,
    nobs: int,
    firm_count: int,
    date_count: int,
    fe_count: int,
    date_fe_count: int,
    date_fixed_effects: bool,
    singleton_removed: int,
    rsquared: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """把分块推断结果整理成与黄金估计器完全相同的两类行。"""
    inference_fields = (
        ("系数", params),
        ("聚类标准误", standard_errors),
        ("t值", t_statistics),
        ("p值", p_values),
    )
    for field_label, values in inference_fields:
        invalid_columns = [
            column
            for column in core_cols
            if not np.isfinite(float(values[column]))
        ]
        if invalid_columns:
            raise NonEstimableModelError(
                f"{_regression_context(outcome, spec)}：核心变量"
                + "、".join(invalid_columns)
                + f"的{field_label}为非有限值，无法进行可靠推断。"
            )

    diagnostics: dict[str, object] = {
        "面板": outcome.panel,
        "收益窗口": outcome.label,
        "因变量": outcome.column,
        "机制序号": spec.order,
        "机制键": spec.key,
        "机制": spec.label,
        "控制变量": "是" if controls else "否",
        "控制变量数": len(controls),
        "显式常数项": "是",
        "被提示公司×年度季度固定效应": "是",
        "日期固定效应": "是" if date_fixed_effects else "否",
        "聚类方式": "被提示公司和交易日期双向聚类",
        "有限样本自由度修正": "是",
        "回归样本量": int(nobs),
        "被提示公司聚类数": int(firm_count),
        "日期聚类数": int(date_count),
        "固定效应组数": int(fe_count),
        "日期固定效应组数": int(date_fe_count),
        "删除单例组观测数": int(singleton_removed),
        "R方": float(rsquared),
    }
    variable_metadata = {
        pair.first_col: (pair, pair.first_label, 1)
        for pair in pairs
    }
    variable_metadata.update(
        {
            pair.second_col: (pair, pair.second_label, 2)
            for pair in pairs
        }
    )

    coefficient_rows: list[dict[str, object]] = []
    for variable_order, variable in enumerate(core_cols, start=1):
        pair, variable_label, within_pair_order = variable_metadata[variable]
        beta = float(params[variable])
        standard_error = float(standard_errors[variable])
        t_value = float(t_statistics[variable])
        p_value = float(p_values[variable])
        coefficient_rows.append(
            {
                **diagnostics,
                "规模组": pair.size_key,
                "规模组标签": pair.size_label,
                "变量": variable,
                "变量顺序": variable_order,
                "组内变量顺序": within_pair_order,
                "变量标签": variable_label,
                "变量类型": "机制核心变量",
                "估计状态": "已估计",
                "系数": beta,
                "聚类标准误": standard_error,
                "系数（百分点）": beta * 100,
                "聚类标准误（百分点）": standard_error * 100,
                "t值": t_value,
                "p值": p_value,
                "显著性": significance_stars(p_value),
            }
        )

    main_rows: list[dict[str, object]] = []
    for pair in pairs:
        first_beta = float(params[pair.first_col])
        second_beta = float(params[pair.second_col])
        first_standard_error = float(standard_errors[pair.first_col])
        second_standard_error = float(standard_errors[pair.second_col])
        first_p_value = float(p_values[pair.first_col])
        second_p_value = float(p_values[pair.second_col])
        coefficient_difference = first_beta - second_beta
        difference_variance = float(
            covariance.loc[pair.first_col, pair.first_col]
            + covariance.loc[pair.second_col, pair.second_col]
            - 2 * covariance.loc[pair.first_col, pair.second_col]
        )
        wald_statistic = coefficient_difference**2 / difference_variance
        wald_p_value = 1.0 - stats.chi2.cdf(wald_statistic, 1)
        main_rows.append(
            {
                **diagnostics,
                "规模组": pair.size_key,
                "规模组标签": pair.size_label,
                "第一组变量": pair.first_col,
                "第一组标签": pair.first_label,
                "第二组变量": pair.second_col,
                "第二组标签": pair.second_label,
                "第一组系数": first_beta,
                "第二组系数": second_beta,
                "第一组聚类标准误": first_standard_error,
                "第二组聚类标准误": second_standard_error,
                "第一组聚类标准误（百分点）": first_standard_error * 100,
                "第二组聚类标准误（百分点）": second_standard_error * 100,
                "第一组p值": first_p_value,
                "第二组p值": second_p_value,
                "系数差": coefficient_difference,
                "系数差（百分点）": coefficient_difference * 100,
                "Wald检验式": f"{pair.first_col} = {pair.second_col}",
                "Wald统计量": float(wald_statistic),
                "Wald自由度": 1,
                "Wald p值": float(wald_p_value),
            }
        )
    return main_rows, coefficient_rows


def run_mechanism_regression_streaming(
    cache_path: str | Path,
    outcome: OutcomeSpec,
    spec: MechanismSpec,
    control_cols: Sequence[str],
    *,
    batch_size: int = DEFAULT_CHUNK_SIZE,
    date_fixed_effects: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """用有界分块复现固定效应回归和双向聚类推断。"""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size必须是正整数。")
    controls = list(dict.fromkeys(control_cols))
    context = _regression_context(outcome, spec)
    if "const" in controls:
        raise ValueError(
            f"{context}：控制变量不能命名为const，以免覆盖显式常数项。"
        )

    pairs = _comparison_pairs_for_spec(spec)
    core_cols = [
        column
        for pair in pairs
        for column in (pair.first_col, pair.second_col)
    ]
    numeric_cols = [outcome.column, *core_cols, *controls]
    x_columns = ["const", *core_cols, *controls]
    stats_blocks: list[pd.DataFrame] = []
    date_stats_blocks: list[pd.DataFrame] = []
    core_minimum = np.full(len(core_cols), np.inf)
    core_maximum = np.full(len(core_cols), -np.inf)
    complete_observations = 0

    for reg in _iter_streaming_model_batches(
        cache_path,
        outcome,
        spec,
        controls,
        batch_size=batch_size,
    ):
        numeric = _streaming_numeric_frame(reg, numeric_cols, context)
        core_values = numeric.loc[:, core_cols].to_numpy()
        core_minimum = np.minimum(
            core_minimum,
            np.min(core_values, axis=0),
        )
        core_maximum = np.maximum(
            core_maximum,
            np.max(core_values, axis=0),
        )
        block = numeric.reset_index(drop=True)
        block.insert(
            0,
            "yearQuarter",
            reg["yearQuarter"].astype("string").to_numpy(dtype=object),
        )
        block.insert(
            0,
            "StkcdRecall",
            reg["StkcdRecall"].astype("string").to_numpy(dtype=object),
        )
        grouped = block.groupby(
            ["StkcdRecall", "yearQuarter"],
            observed=True,
            sort=False,
        )
        sums = grouped[numeric_cols].sum()
        sums["_count"] = grouped.size().astype("float64")
        stats_blocks.append(sums)
        if date_fixed_effects:
            date_block = numeric.reset_index(drop=True)
            date_block.insert(
                0,
                "Trddt",
                reg["Trddt"].to_numpy(dtype="datetime64[ns]", copy=False),
            )
            date_grouped = date_block.groupby(
                "Trddt",
                observed=True,
                sort=False,
            )
            date_sums = date_grouped[numeric_cols].sum()
            date_sums["_count"] = date_grouped.size().astype("float64")
            date_stats_blocks.append(date_sums)
        complete_observations += len(reg)

    if complete_observations == 0 or not stats_blocks:
        raise NonEstimableModelError(
            f"{context}：统一删缺失后没有可用观测。"
        )
    for column, minimum, maximum in zip(
        core_cols,
        core_minimum,
        core_maximum,
        strict=True,
    ):
        if minimum == maximum:
            raise NonEstimableModelError(
                f"{context}：核心变量{column}没有有效变异，无法估计。"
            )

    group_stats = (
        pd.concat(stats_blocks, axis=0)
        .groupby(level=[0, 1], sort=False)
        .sum()
    )
    del stats_blocks

    date_index: pd.DatetimeIndex | None = None
    date_effects: np.ndarray | None = None
    group_date_effect_means: np.ndarray | None = None
    date_fe_count = 0

    if date_fixed_effects:
        date_stats = (
            pd.concat(date_stats_blocks, axis=0)
            .groupby(level=0, sort=False)
            .sum()
        )
        del date_stats_blocks
        full_group_index = group_stats.index
        full_date_index = pd.DatetimeIndex(date_stats.index)
        group_position_blocks: list[np.ndarray] = []
        date_position_blocks: list[np.ndarray] = []
        for reg in _iter_streaming_model_batches(
            cache_path,
            outcome,
            spec,
            controls,
            batch_size=batch_size,
        ):
            group_positions, date_positions = (
                _streaming_fixed_effect_positions(
                    reg,
                    full_group_index,
                    full_date_index,
                )
            )
            group_position_blocks.append(group_positions)
            date_position_blocks.append(date_positions)

        all_group_positions = np.concatenate(group_position_blocks)
        all_date_positions = np.concatenate(date_position_blocks)
        del group_position_blocks, date_position_blocks
        if len(all_group_positions) != complete_observations:
            raise RuntimeError("固定效应关联矩阵的样本量与首遍统计不一致。")
        incidence = sparse.csr_matrix(
            (
                np.ones(complete_observations, dtype=np.float64),
                (all_group_positions, all_date_positions),
            ),
            shape=(len(full_group_index), len(full_date_index)),
        )

        alive_groups = np.ones(len(full_group_index), dtype=bool)
        alive_dates = np.ones(len(full_date_index), dtype=bool)
        while True:
            active_incidence = incidence[alive_groups][:, alive_dates]
            group_degrees = np.asarray(
                active_incidence.sum(axis=1)
            ).ravel()
            date_degrees = np.asarray(
                active_incidence.sum(axis=0)
            ).ravel()
            group_ids = np.flatnonzero(alive_groups)
            date_ids = np.flatnonzero(alive_dates)
            next_alive_groups = alive_groups.copy()
            next_alive_dates = alive_dates.copy()
            next_alive_groups[group_ids] = group_degrees > 1
            next_alive_dates[date_ids] = date_degrees > 1
            if (
                np.array_equal(next_alive_groups, alive_groups)
                and np.array_equal(next_alive_dates, alive_dates)
            ):
                break
            alive_groups = next_alive_groups
            alive_dates = next_alive_dates

        incidence = incidence[alive_groups][:, alive_dates].tocsr()
        nobs = int(round(float(incidence.sum())))
        singleton_removed = complete_observations - nobs
        if nobs == 0:
            raise NonEstimableModelError(
                f"{context}：递归删除固定效应单例后没有可用观测。"
            )

        if singleton_removed:
            group_sums_full = np.zeros(
                (len(full_group_index), len(numeric_cols)),
                dtype=np.float64,
            )
            date_sums_full = np.zeros(
                (len(full_date_index), len(numeric_cols)),
                dtype=np.float64,
            )
            group_counts_full = np.zeros(
                len(full_group_index),
                dtype=np.int64,
            )
            date_counts_full = np.zeros(
                len(full_date_index),
                dtype=np.int64,
            )
            for reg in _iter_streaming_model_batches(
                cache_path,
                outcome,
                spec,
                controls,
                batch_size=batch_size,
            ):
                group_positions, date_positions = (
                    _streaming_fixed_effect_positions(
                        reg,
                        full_group_index,
                        full_date_index,
                    )
                )
                keep = (
                    alive_groups[group_positions]
                    & alive_dates[date_positions]
                )
                if not np.any(keep):
                    continue
                kept_groups = group_positions[keep]
                kept_dates = date_positions[keep]
                values = _streaming_numeric_frame(
                    reg.loc[keep],
                    numeric_cols,
                    context,
                ).to_numpy()
                group_counts_full += np.bincount(
                    kept_groups,
                    minlength=len(full_group_index),
                )
                date_counts_full += np.bincount(
                    kept_dates,
                    minlength=len(full_date_index),
                )
                for column in range(len(numeric_cols)):
                    group_sums_full[:, column] += np.bincount(
                        kept_groups,
                        weights=values[:, column],
                        minlength=len(full_group_index),
                    )
                    date_sums_full[:, column] += np.bincount(
                        kept_dates,
                        weights=values[:, column],
                        minlength=len(full_date_index),
                    )
            group_sums = group_sums_full[alive_groups]
            date_sums = date_sums_full[alive_dates]
            group_counts = group_counts_full[alive_groups]
            date_counts = date_counts_full[alive_dates]
            del (
                group_sums_full,
                date_sums_full,
                group_counts_full,
                date_counts_full,
            )
        else:
            group_sums = group_stats.loc[:, numeric_cols].to_numpy()
            date_sums = date_stats.loc[:, numeric_cols].to_numpy()
            group_counts = (
                group_stats["_count"].to_numpy(dtype=np.int64)
            )
            date_counts = date_stats["_count"].to_numpy(dtype=np.int64)

        eligible_index = full_group_index[alive_groups]
        date_index = full_date_index[alive_dates]
        group_stats = pd.DataFrame(
            group_sums,
            index=eligible_index,
            columns=numeric_cols,
        )
        counts = pd.Series(group_counts, index=eligible_index)
        observed_group_counts = np.asarray(
            incidence.sum(axis=1)
        ).ravel()
        observed_date_counts = np.asarray(
            incidence.sum(axis=0)
        ).ravel()
        if (
            not np.array_equal(observed_group_counts, group_counts)
            or not np.array_equal(observed_date_counts, date_counts)
        ):
            raise RuntimeError("两组固定效应计数与关联矩阵不一致。")

        group_quarters = (
            eligible_index.get_level_values("yearQuarter")
            .astype(str)
            .to_numpy()
        )
        quarter_levels, group_quarter_codes = np.unique(
            group_quarters,
            return_inverse=True,
        )
        date_quarter_code_mean = (
            incidence.T @ group_quarter_codes.astype(np.float64)
        ) / date_counts
        date_quarter_code_square_mean = (
            incidence.T
            @ np.square(group_quarter_codes.astype(np.float64))
        ) / date_counts
        date_quarter_code_variance = (
            date_quarter_code_square_mean
            - np.square(date_quarter_code_mean)
        )
        if np.any(np.abs(date_quarter_code_variance) > 1e-12):
            raise ValueError(
                f"{context}：同一交易日期映射到多个yearQuarter，"
                "无法同时吸收两组固定效应。"
            )
        date_quarter_codes = np.rint(
            date_quarter_code_mean
        ).astype(np.int64)
        date_quarters = quarter_levels[date_quarter_codes]

        group_means = group_sums / group_counts[:, None]
        grand_means = group_sums.sum(axis=0) / nobs
        weighted_incidence = incidence.multiply(
            (1.0 / group_counts)[:, None]
        )
        date_laplacian = (
            sparse.diags(date_counts.astype(np.float64))
            - incidence.T @ weighted_incidence
        ).tocsr()
        date_rhs = date_sums - incidence.T @ group_means
        date_effects = np.zeros_like(date_rhs)
        for quarter in pd.unique(date_quarters):
            date_ids = np.flatnonzero(date_quarters == quarter)
            block = date_laplacian[date_ids][:, date_ids].toarray()
            date_effects[date_ids] = np.linalg.lstsq(
                block,
                date_rhs[date_ids],
                rcond=None,
            )[0]
        group_date_effect_means = (
            incidence @ date_effects
        ) / group_counts[:, None]
        fe_count = len(eligible_index)
        date_fe_count = len(date_index)
        del (
            all_group_positions,
            all_date_positions,
            incidence,
            weighted_incidence,
            date_laplacian,
            date_rhs,
            date_stats,
            group_sums,
            date_sums,
            group_counts,
            date_counts,
        )
    else:
        counts = group_stats["_count"].astype("int64")
        eligible = counts.gt(1)
        singleton_removed = int(counts.loc[~eligible].sum())
        group_stats = group_stats.loc[eligible]
        counts = counts.loc[eligible]
        nobs = int(counts.sum())
        eligible_index = group_stats.index
        group_means = (
            group_stats.loc[:, numeric_cols].to_numpy()
            / counts.to_numpy()[:, None]
        )
        grand_means = (
            group_stats.loc[:, numeric_cols].sum(axis=0).to_numpy()
            / nobs
        )
        fe_count = len(group_stats)

    if group_stats.empty:
        raise NonEstimableModelError(
            f"{context}：递归删除固定效应单例后没有可用观测。"
        )
    for column in core_cols:
        sums = group_stats[column].to_numpy()
        group_counts = counts.to_numpy()
        if not np.any((sums > 0) & (sums < group_counts)):
            raise NonEstimableModelError(
                f"{context}：核心变量{column}被公司×年度季度固定效应"
                "完全吸收，无法估计。"
            )

    def collect_qr(
        retain: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float, int]:
        qr_r: np.ndarray | None = None
        y_sum = 0.0
        y_square_sum = 0.0
        seen = 0
        for reg in _iter_streaming_model_batches(
            cache_path,
            outcome,
            spec,
            controls,
            batch_size=batch_size,
        ):
            x_tilde, y_tilde, _, _, raw_y = _transform_streaming_batch(
                reg,
                numeric_cols,
                eligible_index,
                group_means,
                grand_means,
                context,
                retain,
                date_index=date_index,
                date_effects=date_effects,
                group_date_effect_means=group_date_effect_means,
            )
            if len(y_tilde) == 0:
                continue
            qr_r = _update_tsqr(
                qr_r,
                np.column_stack([x_tilde, y_tilde]),
            )
            y_sum += float(raw_y.sum())
            y_square_sum += float(raw_y @ raw_y)
            seen += len(y_tilde)
        if qr_r is None or seen != nobs:
            raise RuntimeError("分块QR累计的样本量与固定效应统计不一致。")
        return qr_r, y_sum, y_square_sum, seen

    try:
        qr_r, y_sum, y_square_sum, _ = collect_qr()
        retain = _retained_columns_from_tsqr(qr_r, x_columns)
        if len(retain) == 0:
            raise ValueError("所有解释变量均被固定效应吸收。")
        unestimated_core = [
            column
            for column in core_cols
            if x_columns.index(column) not in set(retain.tolist())
        ]
        if unestimated_core:
            raise NonEstimableModelError(
                f"{context}：核心变量"
                + "、".join(unestimated_core)
                + "与其他解释变量存在完全共线性，因而未估计。"
            )
        if len(retain) != len(x_columns):
            qr_r, y_sum, y_square_sum, _ = collect_qr(retain)
        retained_columns = [x_columns[index] for index in retain]
        parameter_count = len(retained_columns)
        cross_products = qr_r.T @ qr_r
        xtx = cross_products[:parameter_count, :parameter_count]
        xty = cross_products[:parameter_count, parameter_count]
        beta = np.linalg.solve(
            qr_r[:parameter_count, :parameter_count],
            qr_r[:parameter_count, parameter_count],
        )
    except NonEstimableModelError:
        raise
    except (
        ValueError,
        TypeError,
        OverflowError,
        np.linalg.LinAlgError,
    ) as exc:
        raise ValueError(f"{context}：模型拟合失败：{exc}") from exc

    firm_scores: dict[object, np.ndarray] = {}
    date_scores: dict[object, np.ndarray] = {}
    intersection_meat = np.zeros(
        (parameter_count, parameter_count),
        dtype=np.float64,
    )
    residual_ss = 0.0
    covariance_observations = 0
    for reg in _iter_streaming_model_batches(
        cache_path,
        outcome,
        spec,
        controls,
        batch_size=batch_size,
    ):
        x_tilde, y_tilde, firms, dates, _ = _transform_streaming_batch(
            reg,
            numeric_cols,
            eligible_index,
            group_means,
            grand_means,
            context,
            retain,
            date_index=date_index,
            date_effects=date_effects,
            group_date_effect_means=group_date_effect_means,
        )
        if len(y_tilde) == 0:
            continue
        residuals = y_tilde - x_tilde @ beta
        scores = x_tilde * residuals[:, None]
        _merge_cluster_scores(firm_scores, firms, scores)
        _merge_cluster_scores(date_scores, dates, scores)
        intersection_meat += scores.T @ scores
        residual_ss += float(residuals @ residuals)
        covariance_observations += len(y_tilde)
    if covariance_observations != nobs:
        raise RuntimeError("分块聚类累计的样本量与固定效应统计不一致。")

    firm_count = len(firm_scores)
    date_count = len(date_scores)
    if firm_count < 2:
        raise NonEstimableModelError(
            f"{context}：被提示公司聚类至少需要两个组。"
        )
    if date_count < 2:
        raise NonEstimableModelError(
            f"{context}：交易日期聚类至少需要两个组。"
        )
    if fe_count < 2:
        raise NonEstimableModelError(
            f"{context}：被提示公司×年度季度固定效应至少需要两个组。"
        )
    if nobs <= parameter_count:
        raise NonEstimableModelError(
            f"{context}：有效样本量不足以进行有限样本修正。"
        )

    firm_matrix = np.vstack(list(firm_scores.values()))
    date_matrix = np.vstack(list(date_scores.values()))
    firm_meat = firm_matrix.T @ firm_matrix
    date_meat = date_matrix.T @ date_matrix

    def correction(cluster_count: int) -> float:
        return (
            cluster_count
            / (cluster_count - 1)
            * (nobs - 1)
            / (nobs - parameter_count)
        )

    combined_meat = (
        correction(firm_count) * firm_meat
        + correction(date_count) * date_meat
        - correction(nobs) * intersection_meat
    )
    try:
        xtx_inverse = np.linalg.inv(xtx)
        covariance_array = (
            xtx_inverse
            @ combined_meat
            @ xtx_inverse
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{context}：模型拟合失败：{exc}") from exc
    covariance_array = (covariance_array + covariance_array.T) / 2
    params = pd.Series(beta, index=retained_columns, name="parameter")
    covariance = pd.DataFrame(
        covariance_array,
        index=retained_columns,
        columns=retained_columns,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        standard_errors = pd.Series(
            np.sqrt(np.diag(covariance_array)),
            index=retained_columns,
            name="stderr",
        )
        t_statistics = pd.Series(
            beta / standard_errors.to_numpy(),
            index=retained_columns,
            name="tstat",
        )
    df_model = (
        fe_count
        + date_fe_count
        - 2
        + parameter_count
        if date_fixed_effects
        else fe_count - 1 + parameter_count
    )
    df_resid = nobs - df_model
    p_values = pd.Series(
        2.0
        - 2.0
        * stats.t.cdf(
            np.abs(t_statistics.to_numpy()),
            df_resid,
        ),
        index=retained_columns,
        name="pvalue",
    )
    total_ss = y_square_sum - y_sum**2 / nobs
    rsquared = max(1.0 - residual_ss / total_ss, 0.0)

    return _streaming_result_rows(
        outcome=outcome,
        spec=spec,
        controls=controls,
        pairs=pairs,
        core_cols=core_cols,
        params=params,
        covariance=covariance,
        standard_errors=standard_errors,
        t_statistics=t_statistics,
        p_values=p_values,
        nobs=nobs,
        firm_count=firm_count,
        date_count=date_count,
        fe_count=fe_count,
        date_fe_count=date_fe_count,
        date_fixed_effects=date_fixed_effects,
        singleton_removed=singleton_removed,
        rsquared=rsquared,
    )


def _non_estimable_rows(
    outcome: OutcomeSpec,
    spec: MechanismSpec,
    control_cols: Sequence[str],
    reason: str,
    *,
    date_fixed_effects: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """为无法识别的模型生成结构完整、数值留空的结果行。"""
    pairs = _comparison_pairs_for_spec(spec)
    missing_value = float("nan")
    diagnostics: dict[str, object] = {
        "面板": outcome.panel,
        "收益窗口": outcome.label,
        "因变量": outcome.column,
        "机制序号": spec.order,
        "机制键": spec.key,
        "机制": spec.label,
        "控制变量": "是" if control_cols else "否",
        "控制变量数": len(control_cols),
        "显式常数项": "是",
        "被提示公司×年度季度固定效应": "是",
        "日期固定效应": "是" if date_fixed_effects else "否",
        "聚类方式": "被提示公司和交易日期双向聚类",
        "有限样本自由度修正": "是",
        "估计状态": "不可估计",
        "不可估计原因": reason,
        "回归样本量": missing_value,
        "被提示公司聚类数": missing_value,
        "日期聚类数": missing_value,
        "固定效应组数": missing_value,
        "日期固定效应组数": missing_value,
        "删除单例组观测数": missing_value,
        "R方": missing_value,
    }

    main_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    variable_order = 0
    for pair in pairs:
        main_rows.append(
            {
                **diagnostics,
                "规模组": pair.size_key,
                "规模组标签": pair.size_label,
                "第一组变量": pair.first_col,
                "第一组标签": pair.first_label,
                "第二组变量": pair.second_col,
                "第二组标签": pair.second_label,
                "第一组系数": missing_value,
                "第二组系数": missing_value,
                "第一组聚类标准误": missing_value,
                "第二组聚类标准误": missing_value,
                "第一组聚类标准误（百分点）": missing_value,
                "第二组聚类标准误（百分点）": missing_value,
                "第一组p值": missing_value,
                "第二组p值": missing_value,
                "系数差": missing_value,
                "系数差（百分点）": missing_value,
                "Wald检验式": f"{pair.first_col} = {pair.second_col}",
                "Wald统计量": missing_value,
                "Wald自由度": missing_value,
                "Wald p值": missing_value,
            }
        )
        for within_pair_order, (variable, label) in enumerate(
            (
                (pair.first_col, pair.first_label),
                (pair.second_col, pair.second_label),
            ),
            start=1,
        ):
            variable_order += 1
            coefficient_rows.append(
                {
                    **diagnostics,
                    "规模组": pair.size_key,
                    "规模组标签": pair.size_label,
                    "变量": variable,
                    "变量顺序": variable_order,
                    "组内变量顺序": within_pair_order,
                    "变量标签": label,
                    "变量类型": "机制核心变量",
                    "系数": missing_value,
                    "聚类标准误": missing_value,
                    "系数（百分点）": missing_value,
                    "聚类标准误（百分点）": missing_value,
                    "t值": missing_value,
                    "p值": missing_value,
                    "显著性": "",
                }
            )
    return main_rows, coefficient_rows


def _add_result_metadata(
    rows: Sequence[dict[str, object]],
    sue_id: int,
    spec: MechanismSpec,
    control_cols: Sequence[str],
    *,
    main_result: bool,
    date_fixed_effects: bool = False,
) -> list[dict[str, object]]:
    """复制轻量结果行，并补齐跨三种SUE一致的输出元数据。"""
    output: list[dict[str, object]] = []
    for row in rows:
        normalized = dict(row)
        normalized.setdefault("估计状态", "已估计")
        normalized.setdefault("不可估计原因", "")
        normalized.update(
            {
                "SUE口径": f"SUE{sue_id}",
                "SUE口径说明": _SUE_NOTES[sue_id],
                "机制编号": spec.order,
                "机制名称": normalized.get("机制", spec.label),
                "控制变量": "、".join(control_cols),
                "固定效应": (
                    "被提示公司×年度季度固定效应、交易日期固定效应"
                    if date_fixed_effects
                    else "被提示公司×年度季度固定效应"
                ),
                "口径说明": spec.source_note,
            }
        )
        if main_result:
            normalized.update(
                {
                    "强关联提示变量": normalized["第一组变量"],
                    "其他关联提示变量": normalized["第二组变量"],
                    "强关联提示系数": normalized["第一组系数"],
                    "其他关联提示系数": normalized["第二组系数"],
                    "强关联提示聚类标准误": normalized["第一组聚类标准误"],
                    "其他关联提示聚类标准误": normalized["第二组聚类标准误"],
                    "强关联提示p值": normalized["第一组p值"],
                    "其他关联提示p值": normalized["第二组p值"],
                    "沃尔德统计量": normalized["Wald统计量"],
                    "沃尔德自由度": normalized["Wald自由度"],
                    "沃尔德p值": normalized["Wald p值"],
                }
            )
        output.append(normalized)
    return output


def _sort_result_table(
    rows: Sequence[dict[str, object]],
    *,
    coefficient_detail: bool,
) -> pd.DataFrame:
    """按面板、机制、规模组和组内变量顺序稳定整理结果。"""
    table = pd.DataFrame(rows)
    panel_order = {"A": 0, "B": 1}
    size_order = {"90plus": 0, "50_90": 1}
    table["_面板排序"] = table["面板"].map(panel_order)
    table["_规模组排序"] = table["规模组"].map(size_order)
    sort_columns = ["_面板排序", "机制序号", "_规模组排序"]
    if coefficient_detail:
        sort_columns.append("组内变量顺序")
    return (
        table.sort_values(sort_columns, kind="stable")
        .drop(columns=["_面板排序", "_规模组排序"])
        .reset_index(drop=True)
    )


def _orchestrate_all_models(
    execute_regression: Callable[
        [OutcomeSpec, MechanismSpec, Sequence[str]],
        tuple[list[dict[str, object]], list[dict[str, object]]],
    ],
    *,
    date_fixed_effects: bool = False,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """按固定顺序编排34个模型，并返回三种SUE的两类结果。"""
    control_cols = [
        *OWN_EA_COLS,
        *DISCLOSURE_COLS,
        *MEDIA_CONTROL_COLS,
    ]
    shared_specs = tuple(
        spec
        for spec in build_mechanism_specs(1)
        if spec.order in _SHARED_MECHANISM_ORDERS
    )
    shared_cache: dict[
        tuple[str, str],
        tuple[list[dict[str, object]], list[dict[str, object]]],
    ] = {}
    model_number = 0

    def execute_model(
        outcome: OutcomeSpec,
        spec: MechanismSpec,
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        str,
    ]:
        try:
            main_rows, coefficient_rows = execute_regression(
                outcome,
                spec,
                control_cols,
            )
            return main_rows, coefficient_rows, ""
        except NonEstimableModelError as exc:
            reason = str(exc)
            main_rows, coefficient_rows = _non_estimable_rows(
                outcome,
                spec,
                control_cols,
                reason,
                date_fixed_effects=date_fixed_effects,
            )
            return main_rows, coefficient_rows, reason

    for outcome in OUTCOME_SPECS:
        for spec in shared_specs:
            main_rows, coefficient_rows, unavailable_reason = execute_model(
                outcome,
                spec,
            )
            shared_cache[(outcome.panel, spec.key)] = (
                [dict(row) for row in main_rows],
                [dict(row) for row in coefficient_rows],
            )
            model_number += 1
            print(
                f"{model_number}/34：收益窗口{outcome.label}，"
                f"机制{spec.order}（{spec.label}）"
                + (
                    f"；不可估计：{unavailable_reason}"
                    if unavailable_reason
                    else ""
                )
            )
            del main_rows, coefficient_rows

    main_results: dict[int, pd.DataFrame] = {}
    coefficient_details: dict[int, pd.DataFrame] = {}
    for sue_id in (1, 2, 3):
        specs = build_mechanism_specs(sue_id)
        specs_by_order = {spec.order: spec for spec in specs}
        sue_main_rows: list[dict[str, object]] = []
        sue_coefficient_rows: list[dict[str, object]] = []

        for outcome in OUTCOME_SPECS:
            for spec in specs:
                if spec.order not in _SUE_MECHANISM_ORDERS:
                    continue
                main_rows, coefficient_rows, unavailable_reason = execute_model(
                    outcome,
                    spec,
                )
                sue_main_rows.extend(
                    _add_result_metadata(
                        main_rows,
                        sue_id,
                        spec,
                        control_cols,
                        main_result=True,
                        date_fixed_effects=date_fixed_effects,
                    )
                )
                sue_coefficient_rows.extend(
                    _add_result_metadata(
                        coefficient_rows,
                        sue_id,
                        spec,
                        control_cols,
                        main_result=False,
                        date_fixed_effects=date_fixed_effects,
                    )
                )
                model_number += 1
                print(
                    f"{model_number}/34：收益窗口{outcome.label}，"
                    f"机制{spec.order}（{spec.label}）"
                    + (
                        f"；不可估计：{unavailable_reason}"
                        if unavailable_reason
                        else ""
                    )
                )
                del main_rows, coefficient_rows

        for outcome in OUTCOME_SPECS:
            for mechanism_order in _SHARED_MECHANISM_ORDERS:
                spec = specs_by_order[mechanism_order]
                cached_main, cached_coefficients = shared_cache[
                    (outcome.panel, spec.key)
                ]
                sue_main_rows.extend(
                    _add_result_metadata(
                        cached_main,
                        sue_id,
                        spec,
                        control_cols,
                        main_result=True,
                        date_fixed_effects=date_fixed_effects,
                    )
                )
                sue_coefficient_rows.extend(
                    _add_result_metadata(
                        cached_coefficients,
                        sue_id,
                        spec,
                        control_cols,
                        main_result=False,
                        date_fixed_effects=date_fixed_effects,
                    )
                )

        main_results[sue_id] = _sort_result_table(
            sue_main_rows,
            coefficient_detail=False,
        )
        coefficient_details[sue_id] = _sort_result_table(
            sue_coefficient_rows,
            coefficient_detail=True,
        )
        del sue_main_rows, sue_coefficient_rows

    return main_results, coefficient_details


def run_all_models(
    data: pd.DataFrame,
    regression_runner: Callable[
        [
            pd.DataFrame,
            OutcomeSpec,
            MechanismSpec,
            Sequence[str],
        ],
        tuple[list[dict[str, object]], list[dict[str, object]]],
    ] = run_mechanism_regression,
    *,
    date_fixed_effects: bool = False,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """在同一内存面板上串行估计34个模型。"""
    def execute(
        outcome: OutcomeSpec,
        spec: MechanismSpec,
        controls: Sequence[str],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if date_fixed_effects:
            return regression_runner(
                data,
                outcome,
                spec,
                controls,
                date_fixed_effects=True,
            )
        return regression_runner(data, outcome, spec, controls)

    return _orchestrate_all_models(
        execute,
        date_fixed_effects=date_fixed_effects,
    )


def model_required_columns(
    outcome: OutcomeSpec,
    spec: MechanismSpec,
    control_cols: Sequence[str],
) -> list[str]:
    """返回一个模型从列式缓存实际需要读取的字段。"""
    return list(
        dict.fromkeys(
            [
                outcome.column,
                "StkcdRecall",
                "Trddt",
                "yearQuarter",
                *control_cols,
                *_mechanism_source_columns(spec),
            ]
        )
    )


def load_cached_model_data(
    cache_path: str | Path,
    outcome: OutcomeSpec,
    spec: MechanismSpec,
    control_cols: Sequence[str],
) -> pd.DataFrame:
    """从列式缓存仅读取一个模型需要的字段并规范键类型。"""
    context = _regression_context(outcome, spec)
    cache = Path(cache_path)
    columns = model_required_columns(outcome, spec, control_cols)
    try:
        available = set(pq.ParquetFile(cache).schema_arrow.names)
    except (OSError, pa.ArrowInvalid) as exc:
        raise ValueError(f"{context}：无法读取低内存缓存：{exc}") from exc
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(
            f"{context}：低内存缓存缺少以下回归字段："
            + "、".join(missing)
        )

    try:
        data = pd.read_parquet(cache, columns=columns)
    except (OSError, ValueError, TypeError, pa.ArrowInvalid) as exc:
        raise ValueError(f"{context}：读取低内存缓存失败：{exc}") from exc
    data["Trddt"] = pd.to_datetime(data["Trddt"], errors="raise")
    data["StkcdRecall"] = data["StkcdRecall"].astype("string").astype("category")
    data["yearQuarter"] = data["yearQuarter"].astype("string").astype("category")
    return data


def run_all_models_low_memory(
    cache_path: str | Path,
    regression_runner: Callable[
        [
            pd.DataFrame,
            OutcomeSpec,
            MechanismSpec,
            Sequence[str],
        ],
        tuple[list[dict[str, object]], list[dict[str, object]]],
    ] = run_mechanism_regression,
    data_loader: Callable[
        [str | Path, OutcomeSpec, MechanismSpec, Sequence[str]],
        pd.DataFrame,
    ] = load_cached_model_data,
    *,
    date_fixed_effects: bool = False,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """逐模型读取和释放列式缓存数据，保持34个模型编排不变。"""
    cache = Path(cache_path)

    def execute(
        outcome: OutcomeSpec,
        spec: MechanismSpec,
        controls: Sequence[str],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        model_data = data_loader(cache, outcome, spec, controls)
        try:
            if date_fixed_effects:
                return regression_runner(
                    model_data,
                    outcome,
                    spec,
                    controls,
                    date_fixed_effects=True,
                )
            return regression_runner(model_data, outcome, spec, controls)
        finally:
            del model_data
            gc.collect()

    return _orchestrate_all_models(
        execute,
        date_fixed_effects=date_fixed_effects,
    )


def run_all_models_streaming(
    cache_path: str | Path,
    *,
    batch_size: int = DEFAULT_CHUNK_SIZE,
    regression_runner: Callable[
        [
            str | Path,
            OutcomeSpec,
            MechanismSpec,
            Sequence[str],
        ],
        tuple[list[dict[str, object]], list[dict[str, object]]],
    ] = run_mechanism_regression_streaming,
    date_fixed_effects: bool = False,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """用有界分块估计器串行编排34个模型。"""
    cache = Path(cache_path)

    def execute(
        outcome: OutcomeSpec,
        spec: MechanismSpec,
        controls: Sequence[str],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if date_fixed_effects:
            return regression_runner(
                cache,
                outcome,
                spec,
                controls,
                batch_size=batch_size,
                date_fixed_effects=True,
            )
        return regression_runner(
            cache,
            outcome,
            spec,
            controls,
            batch_size=batch_size,
        )

    return _orchestrate_all_models(
        execute,
        date_fixed_effects=date_fixed_effects,
    )


_TABLE5_DRAW_REQUIRED_COLUMNS = {
    "SUE口径",
    "SUE口径说明",
    "面板",
    "收益窗口",
    "机制编号",
    "机制名称",
    "规模组",
    "强关联提示系数",
    "强关联提示聚类标准误",
    "强关联提示p值",
    "其他关联提示系数",
    "其他关联提示聚类标准误",
    "其他关联提示p值",
    "沃尔德p值",
    "回归样本量",
    "R方",
    "估计状态",
}


def _validate_draw_summary(summary: pd.DataFrame, sue_id: int) -> None:
    """校验两张单窗口图片共同需要的36条主结果。"""
    if type(sue_id) is not int or sue_id not in (1, 2, 3):
        raise ValueError("绘图SUE口径只能是1、2或3。")
    missing = sorted(_TABLE5_DRAW_REQUIRED_COLUMNS - set(summary.columns))
    if missing:
        raise ValueError(
            f"SUE{sue_id}主结果缺少以下绘图字段：" + "、".join(missing)
        )
    expected_sue = f"SUE{sue_id}"
    if set(summary["SUE口径"].dropna().astype(str)) != {expected_sue}:
        raise ValueError(f"绘图数据并非单一的{expected_sue}口径。")
    if len(summary) != 36:
        raise ValueError(
            f"{expected_sue}主结果应有36行，实际为{len(summary)}行。"
        )
    key_cols = ["面板", "机制编号", "规模组"]
    if summary.duplicated(key_cols).any():
        raise ValueError(
            f"{expected_sue}主结果的面板、机制编号和规模组不是唯一键。"
        )
    expected_keys = {
        (panel, mechanism, size)
        for panel in ("A", "B")
        for mechanism in range(1, 10)
        for size in ("90plus", "50_90")
    }
    actual_keys = {
        (str(panel), int(mechanism), str(size))
        for panel, mechanism, size in summary[key_cols].itertuples(
            index=False,
            name=None,
        )
    }
    if actual_keys != expected_keys:
        raise ValueError(f"{expected_sue}主结果的面板、机制或规模组不完整。")


def _mechanism_header(spec: MechanismSpec) -> str:
    """把机制名称换行为适合九列宽表的列标题。"""
    labels = {
        "异常总成交量": "异常总\n成交量",
        "异常散户成交量（A股小单替代）": "异常散户成交量\n（A股小单替代）",
        "媒体覆盖": "媒体\n覆盖",
        "价格异动": "价格\n异动",
        "正向盈余惊喜": "正向\n盈余惊喜",
        "负向盈余惊喜": "负向\n盈余惊喜",
        "极端正向盈余惊喜": "极端正向\n盈余惊喜",
        "极端负向盈余惊喜": "极端负向\n盈余惊喜",
        "干扰程度": "干扰\n程度",
    }
    return f"({spec.order})\n{labels.get(spec.label, spec.label)}"


def draw_table5(
    summary: pd.DataFrame,
    sue_id: int,
    output_path: str | Path,
    *,
    panel: str = "A",
    window_role: str = "主口径",
    dpi: int = 300,
    date_fixed_effects: bool = False,
) -> Path:
    """绘制一张仅包含一个收益窗口和九类机制的论文式三线表图片。"""
    if dpi < 300:
        raise ValueError("正式表5图片分辨率不能低于300 DPI。")
    if panel not in {"A", "B"}:
        raise ValueError("绘图收益窗口只能对应面板A或面板B。")
    if not str(window_role).strip():
        raise ValueError("绘图收益窗口角色不能为空。")
    _validate_draw_summary(summary, sue_id)
    if date_fixed_effects:
        if "日期固定效应" not in summary.columns:
            raise ValueError("日期固定效应版图片缺少日期固定效应结果字段。")
        date_fe_values = set(
            summary["日期固定效应"].dropna().astype(str)
        )
        if date_fe_values != {"是"}:
            raise ValueError(
                "日期固定效应版图片要求日期固定效应全部为“是”；"
                f"实际取值为：{sorted(date_fe_values)}。"
            )

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = (
        FontProperties(fname=str(FONT_PATH))
        if FONT_PATH.exists()
        else FontProperties(family="sans-serif")
    )
    font_bold = (
        FontProperties(fname=str(FONT_PATH), weight="bold")
        if FONT_PATH.exists()
        else FontProperties(family="sans-serif", weight="bold")
    )

    specs = build_mechanism_specs(sue_id)
    lookup = {
        (str(row["面板"]), int(row["机制编号"]), str(row["规模组"])): row
        for _, row in summary.iterrows()
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=A4_LANDSCAPE_FIGSIZE)
    fig.subplots_adjust(left=0.012, right=0.995, top=0.995, bottom=0.012)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#20252B"
    muted = "#56616B"
    accent = "#284B63"
    light_rule = "#B7C0C8"
    label_right = 0.165
    data_left = label_right + 0.004
    data_right = 0.992
    column_width = (data_right - data_left) / 9
    column_centers = [
        data_left + (index + 0.5) * column_width
        for index in range(9)
    ]

    ax.text(
        0.01,
        0.978,
        (
            f"表 5  心理机制检验（{window_role}"
            + ("；加入日期固定效应" if date_fixed_effects else "")
            + "）"
        ),
        ha="left",
        va="top",
        fontsize=18,
        color=ink,
        fontproperties=font_bold,
    )
    ax.text(
        0.01,
        0.948,
        (
            f"SUE{sue_id}：{_SUE_NOTES[sue_id]}；"
            "因变量为被提示公司的特征调整累计收益率，系数与标准误单位为百分点"
        ),
        ha="left",
        va="top",
        fontsize=10.5,
        color=muted,
        fontproperties=font,
    )

    row_definitions = [
        ("市值前10%：强关联提示", "first_90"),
        ("市值前10%：其他关联提示", "second_90"),
        ("市值前10%：系数相等检验 p 值", "wald_90"),
        ("市值50%至90%：强关联提示", "first_5090"),
        ("市值50%至90%：其他关联提示", "second_5090"),
        ("市值50%至90%：系数相等检验 p 值", "wald_5090"),
        ("控制变量", "controls"),
        ("被提示公司×年度季度固定效应", "fixed_effects"),
    ]
    if date_fixed_effects:
        row_definitions.append(("交易日期固定效应", "date_fixed_effects"))
    row_definitions.extend(
        [
            ("回归样本量", "observations"),
            ("R²", "r_squared"),
        ]
    )

    def panel_values(panel: str, value_key: str) -> list[str]:
        values: list[str] = []
        for spec in specs:
            size = "50_90" if value_key.endswith("5090") else "90plus"
            if value_key in {
                "controls",
                "fixed_effects",
                "date_fixed_effects",
                "observations",
                "r_squared",
            }:
                size = "90plus"
            row = lookup[(panel, spec.order, size)]
            unavailable = str(row["估计状态"]) == "不可估计"
            if unavailable and (
                value_key.startswith("first_")
                or value_key.startswith("second_")
                or value_key.startswith("wald_")
            ):
                values.append("不可估计")
            elif value_key.startswith("first_"):
                values.append(
                    format_coefficient(
                        float(row["强关联提示系数"]) * 100,
                        float(row["强关联提示聚类标准误"]) * 100,
                        float(row["强关联提示p值"]),
                    )
                )
            elif value_key.startswith("second_"):
                values.append(
                    format_coefficient(
                        float(row["其他关联提示系数"]) * 100,
                        float(row["其他关联提示聚类标准误"]) * 100,
                        float(row["其他关联提示p值"]),
                    )
                )
            elif value_key.startswith("wald_"):
                values.append(format_p_value(float(row["沃尔德p值"])))
            elif value_key == "controls":
                values.append("是")
            elif value_key == "fixed_effects":
                values.append("是")
            elif value_key == "date_fixed_effects":
                values.append(str(row["日期固定效应"]))
            elif value_key == "observations":
                observations = pd.to_numeric(
                    pd.Series([row["回归样本量"]]),
                    errors="coerce",
                ).iloc[0]
                values.append(
                    "—" if pd.isna(observations) else f"{int(observations):,}"
                )
            else:
                r_squared = pd.to_numeric(
                    pd.Series([row["R方"]]),
                    errors="coerce",
                ).iloc[0]
                values.append(
                    "—" if pd.isna(r_squared) else f"{float(r_squared):.4f}"
                )
        return values

    def draw_window_table(panel: str, top: float, bottom: float) -> None:
        outcome = next(item for item in OUTCOME_SPECS if item.panel == panel)
        ax.text(
            0.01,
            top + 0.017,
            f"收益窗口 {outcome.label}",
            ha="left",
            va="bottom",
            fontsize=11.5,
            color=ink,
            fontproperties=font_bold,
        )
        ax.plot(
            [0.01, data_right],
            [top, top],
            color=accent,
            linewidth=1.25,
            solid_capstyle="butt",
        )
        header_height = 0.066
        header_bottom = top - header_height
        ax.text(
            0.01,
            top - header_height / 2,
            "机制与模型信息",
            ha="left",
            va="center",
            fontsize=8.6,
            color=ink,
            fontproperties=font_bold,
        )
        for center, spec in zip(column_centers, specs, strict=True):
            ax.text(
                center,
                top - header_height / 2,
                _mechanism_header(spec),
                ha="center",
                va="center",
                fontsize=7.5,
                color=ink,
                linespacing=1.15,
                fontproperties=font_bold,
            )
        ax.plot(
            [0.01, data_right],
            [header_bottom, header_bottom],
            color=ink,
            linewidth=0.75,
            solid_capstyle="butt",
        )

        row_height = (header_bottom - bottom) / len(row_definitions)
        for row_index, (row_label, value_key) in enumerate(row_definitions):
            center_y = header_bottom - (row_index + 0.5) * row_height
            ax.text(
                0.01,
                center_y,
                row_label,
                ha="left",
                va="center",
                fontsize=7.4,
                color=ink if row_index < 6 else muted,
                fontproperties=font,
            )
            values = panel_values(panel, value_key)
            for center, value in zip(column_centers, values, strict=True):
                ax.text(
                    center,
                    center_y,
                    value,
                    ha="center",
                    va="center",
                    fontsize=7.6 if row_index < 6 else 7.8,
                    color=ink,
                    linespacing=1.25,
                    fontproperties=font,
                )
            if row_index in (2, 5):
                y = header_bottom - (row_index + 1) * row_height
                ax.plot(
                    [0.01, data_right],
                    [y, y],
                    color=light_rule,
                    linewidth=0.6,
                    solid_capstyle="butt",
                )
        ax.plot(
            [0.01, data_right],
            [bottom, bottom],
            color=ink,
            linewidth=0.9,
            solid_capstyle="butt",
        )

    draw_window_table(panel, 0.875, 0.185)

    inference_note = (
        "模型同时吸收被提示公司×年度季度和交易日期固定效应；"
        if date_fixed_effects
        else ""
    )
    notes = (
        "注：强关联提示指按各列机制判定的强底层关联，其他关联提示指同规模组内未满足该列强关联条件的提示；"
        "干扰程度列将低干扰界定为强关联、高干扰界定为其他关联。",
        inference_note
        + "括号内为按被提示公司和交易日期双向聚类的标准误；*、**、*** 分别表示在10%、5%和1%水平上显著。",
        "异常散户成交量使用A股小单异常成交量，作为原文两种美国散户交易识别方法的中国替代口径。",
        "因当前数据没有可用的具体公告时间戳，原文“交易时段公告”和“盘后公告”机制不估计，也不将缺失时点填为零。",
    )
    for index, note in enumerate(notes):
        ax.text(
            0.01,
            0.152 - index * 0.024,
            note,
            ha="left",
            va="top",
            fontsize=7.3,
            color=muted,
            fontproperties=font,
        )

    try:
        fig.savefig(
            output,
            dpi=dpi,
            facecolor="white",
            metadata={
                "Title": (
                    f"表5 心理机制检验 SUE{sue_id} "
                    f"{next(item.label for item in OUTCOME_SPECS if item.panel == panel)} "
                    f"{window_role}"
                    + (" 加入日期固定效应" if date_fixed_effects else "")
                ),
                "Creator": "表5_心理机制.py",
            },
        )
    finally:
        plt.close(fig)
    return output


def write_outputs(
    main_results: dict[int, pd.DataFrame],
    coefficient_details: dict[int, pd.DataFrame],
    result_dir: str | Path = DEFAULT_RESULT_DIR,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    *,
    date_fixed_effects: bool = False,
) -> list[Path]:
    """保存三份主结果、三份系数明细和六张单窗口图片。"""
    if set(main_results) != {1, 2, 3}:
        raise ValueError("主结果必须同时包含SUE1、SUE2和SUE3。")
    if set(coefficient_details) != {1, 2, 3}:
        raise ValueError("系数明细必须同时包含SUE1、SUE2和SUE3。")

    result_root = Path(result_dir)
    image_root = Path(image_dir)
    result_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    output_suffix = "_加入日期固定效应" if date_fixed_effects else ""

    for sue_id in (1, 2, 3):
        if date_fixed_effects:
            date_fe_values = set(
                main_results[sue_id]["日期固定效应"]
                .dropna()
                .astype(str)
            )
            if date_fe_values != {"是"}:
                raise ValueError(
                    f"SUE{sue_id}日期固定效应结果要求"
                    "日期固定效应全部为“是”。"
                )
        summary_path = (
            result_root
            / f"表5_SUE{sue_id}_心理机制{output_suffix}结果.csv"
        )
        detail_path = (
            result_root
            / f"表5_SUE{sue_id}_心理机制{output_suffix}系数明细.csv"
        )
        main_image_path = (
            image_root
            / f"table5_SUE{sue_id}_心理机制{output_suffix}.png"
        )
        extension_image_path = (
            image_root
            / (
                f"table5_SUE{sue_id}_心理机制_t_tp10扩展分析"
                f"{output_suffix}.png"
            )
        )
        main_results[sue_id].to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        coefficient_details[sue_id].to_csv(
            detail_path,
            index=False,
            encoding="utf-8-sig",
        )
        draw_table5(
            main_results[sue_id],
            sue_id,
            main_image_path,
            panel="A",
            window_role="主口径",
            dpi=300,
            date_fixed_effects=date_fixed_effects,
        )
        draw_table5(
            main_results[sue_id],
            sue_id,
            extension_image_path,
            panel="B",
            window_role="十日扩展分析",
            dpi=300,
            date_fixed_effects=date_fixed_effects,
        )
        written.extend(
            (
                summary_path,
                detail_path,
                main_image_path,
                extension_image_path,
            )
        )

    return written


def read_and_validate_header(input_path: str | Path) -> tuple[str, ...]:
    """仅读取 CSV（逗号分隔值文件）首行，并一次报告全部缺失必需字段。"""
    path = Path(input_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])

    available = set(header)
    missing = [column for column in required_input_columns() if column not in available]
    if missing:
        raise ValueError("输入文件表头缺少以下必需字段：" + "、".join(missing))
    return tuple(header)


def cache_manifest_path(cache_path: str | Path) -> Path:
    """返回低内存列式缓存对应的清单路径。"""
    path = Path(cache_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def _input_column_dtypes() -> dict[str, str]:
    """返回内存模式和分块模式共同使用的输入字段类型。"""
    columns = required_input_columns()
    text_columns = {"StkcdRecall", "yearQuarter"}
    numeric_columns = [
        column
        for column in columns
        if column not in text_columns | {"Trddt"}
    ]
    return {
        "StkcdRecall": "string",
        "yearQuarter": "string",
        **{column: "float32" for column in numeric_columns},
    }


def _normalize_input_chunk(
    chunk: pd.DataFrame,
    *,
    start_year: int | None,
    end_year: int | None,
) -> pd.DataFrame:
    """规范一个输入分块的日期、公司代码和年份筛选。"""
    chunk["Trddt"] = pd.to_datetime(chunk["Trddt"], errors="raise")
    chunk["StkcdRecall"] = (
        chunk["StkcdRecall"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    if start_year is not None or end_year is not None:
        keep = pd.Series(True, index=chunk.index)
        if start_year is not None:
            keep &= chunk["year"].ge(start_year)
        if end_year is not None:
            keep &= chunk["year"].le(end_year)
        chunk = chunk.loc[keep].copy()
    return chunk


def _encode_company_date_keys(chunk: pd.DataFrame) -> np.ndarray:
    """把公司代码和交易日期无损编码为用于磁盘排序的64位键。"""
    key_columns = ["StkcdRecall", "Trddt"]
    if chunk[key_columns].isna().any().any():
        raise ValueError("StkcdRecall和Trddt唯一键字段不能缺失。")

    try:
        firm_numbers = pd.to_numeric(
            chunk["StkcdRecall"],
            errors="raise",
        ).to_numpy(dtype=np.uint64, copy=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("StkcdRecall必须是可转换为整数的六位公司代码。") from exc
    if np.any(firm_numbers > np.uint64(999_999)):
        raise ValueError("StkcdRecall必须是不超过六位的公司代码。")

    date_days = (
        chunk["Trddt"]
        .to_numpy(dtype="datetime64[D]", copy=False)
        .astype(np.int64, copy=False)
    )
    int32 = np.iinfo(np.int32)
    if np.any(date_days < int32.min) or np.any(date_days > int32.max):
        raise ValueError("Trddt超出低内存唯一键编码支持的日期范围。")
    unsigned_days = (date_days - int32.min).astype(np.uint64, copy=False)
    return (firm_numbers << np.uint64(32)) | unsigned_days


def _cache_fingerprint(
    input_path: str | Path,
    *,
    start_year: int | None,
    end_year: int | None,
    nrows: int | None,
) -> dict[str, object]:
    """返回决定列式缓存能否安全复用的源数据指纹。"""
    source = Path(input_path).resolve()
    stat = source.stat()
    return {
        "source_path": str(source),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "columns": required_input_columns(),
        "start_year": start_year,
        "end_year": end_year,
        "nrows": nrows,
    }


def build_low_memory_cache(
    input_path: str | Path,
    cache_path: str | Path,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    nrows: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, object]:
    """分块校验原始面板并原子生成可按列读取的Parquet缓存。"""
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError("开始年份不能晚于结束年份。")
    if nrows is not None and (type(nrows) is not int or nrows <= 0):
        raise ValueError("nrows必须是正整数。")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size必须是正整数。")

    source = Path(input_path)
    read_and_validate_header(source)
    fingerprint = _cache_fingerprint(
        source,
        start_year=start_year,
        end_year=end_year,
        nrows=nrows,
    )
    cache = Path(cache_path)
    manifest_path = cache_manifest_path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    building_cache = cache.with_suffix(cache.suffix + ".building")
    building_manifest = manifest_path.with_suffix(
        manifest_path.suffix + ".building"
    )
    key_path = cache.with_suffix(cache.suffix + ".keys.building")
    for temporary in (building_cache, building_manifest, key_path):
        temporary.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    row_count = 0
    companies: set[str] = set()
    minimum_date: pd.Timestamp | None = None
    maximum_date: pd.Timestamp | None = None
    arrow_schema: pa.Schema | None = None

    try:
        reader = pd.read_csv(
            source,
            usecols=required_input_columns(),
            dtype=_input_column_dtypes(),
            parse_dates=["Trddt"],
            nrows=nrows,
            chunksize=chunk_size,
            low_memory=False,
        )
        for chunk in reader:
            chunk = _normalize_input_chunk(
                chunk,
                start_year=start_year,
                end_year=end_year,
            )
            if chunk.empty:
                continue
            if chunk["yearQuarter"].isna().any():
                raise ValueError("yearQuarter固定效应字段不能缺失。")

            encoded_keys = _encode_company_date_keys(chunk)
            with key_path.open("ab") as key_handle:
                encoded_keys.tofile(key_handle)

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                arrow_schema = table.schema
                writer = pq.ParquetWriter(
                    building_cache,
                    arrow_schema,
                    compression="zstd",
                )
            elif table.schema != arrow_schema:
                table = table.cast(arrow_schema)
            writer.write_table(table, row_group_size=len(chunk))

            row_count += len(chunk)
            companies.update(chunk["StkcdRecall"].astype(str).unique())
            chunk_minimum = pd.Timestamp(chunk["Trddt"].min())
            chunk_maximum = pd.Timestamp(chunk["Trddt"].max())
            minimum_date = (
                chunk_minimum
                if minimum_date is None
                else min(minimum_date, chunk_minimum)
            )
            maximum_date = (
                chunk_maximum
                if maximum_date is None
                else max(maximum_date, chunk_maximum)
            )

        if writer is None or row_count == 0:
            raise ValueError("年份筛选和删缺失后没有可写入缓存的观测。")
        writer.close()
        writer = None

        key_count = key_path.stat().st_size // np.dtype(np.uint64).itemsize
        if key_count != row_count:
            raise RuntimeError("低内存唯一键文件的行数与缓存行数不一致。")
        keys = np.memmap(
            key_path,
            dtype=np.uint64,
            mode="r+",
            shape=(key_count,),
        )
        keys.sort()
        has_duplicate = bool(
            key_count > 1 and np.any(keys[1:] == keys[:-1])
        )
        del keys
        if has_duplicate:
            raise ValueError(
                "StkcdRecall与Trddt不是唯一键，请先处理重复的被提示公司交易日。"
            )

        manifest: dict[str, object] = {
            **fingerprint,
            "row_count": int(row_count),
            "company_count": int(len(companies)),
            "minimum_date": minimum_date.strftime("%Y-%m-%d"),
            "maximum_date": maximum_date.strftime("%Y-%m-%d"),
            "chunk_size": int(chunk_size),
            "complete": True,
        }
        building_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(building_cache, cache)
        os.replace(building_manifest, manifest_path)
        return manifest
    finally:
        if writer is not None:
            writer.close()
        for temporary in (building_cache, building_manifest, key_path):
            temporary.unlink(missing_ok=True)


def cache_is_reusable(
    input_path: str | Path,
    cache_path: str | Path,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    nrows: int | None = None,
) -> bool:
    """判断现有列式缓存是否完整并与本次源数据口径完全一致。"""
    cache = Path(cache_path)
    manifest_path = cache_manifest_path(cache)
    if not cache.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = _cache_fingerprint(
            input_path,
            start_year=start_year,
            end_year=end_year,
            nrows=nrows,
        )
        if not isinstance(manifest, dict) or manifest.get("complete") is not True:
            return False
        if any(manifest.get(key) != value for key, value in expected.items()):
            return False
        metadata = pq.ParquetFile(cache).metadata
        return (
            metadata is not None
            and int(metadata.num_rows) == int(manifest["row_count"])
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        pa.ArrowInvalid,
    ):
        return False


def load_input_data(
    input_path: str | Path,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """按需读取表5字段，规范唯一键并使用紧凑数据类型。"""
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError("开始年份不能晚于结束年份。")
    if nrows is not None and (type(nrows) is not int or nrows <= 0):
        raise ValueError("nrows必须是正整数。")

    columns = required_input_columns()
    data = pd.read_csv(
        Path(input_path),
        usecols=columns,
        dtype=_input_column_dtypes(),
        parse_dates=["Trddt"],
        nrows=nrows,
        low_memory=False,
    )
    data["Trddt"] = pd.to_datetime(data["Trddt"], errors="raise")
    data["StkcdRecall"] = (
        data["StkcdRecall"]
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )

    if start_year is not None or end_year is not None:
        keep = pd.Series(True, index=data.index)
        if start_year is not None:
            keep &= data["year"].ge(start_year)
        if end_year is not None:
            keep &= data["year"].le(end_year)
        data = data.loc[keep].copy()

    key_columns = ["StkcdRecall", "Trddt"]
    if data[key_columns].isna().any().any():
        raise ValueError("StkcdRecall和Trddt唯一键字段不能缺失。")
    if data["yearQuarter"].isna().any():
        raise ValueError("yearQuarter固定效应字段不能缺失。")
    if data.duplicated(key_columns).any():
        raise ValueError(
            "StkcdRecall与Trddt不是唯一键，请先处理重复的被提示公司交易日。"
        )

    data["StkcdRecall"] = data["StkcdRecall"].astype("category")
    data["yearQuarter"] = data["yearQuarter"].astype("category")
    return data


def _positive_int(value: str) -> int:
    """供命令行使用的正整数转换器。"""
    converted = int(value)
    if converted <= 0:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return converted


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析独立表5回归、输出和开发试跑参数。"""
    parser = argparse.ArgumentParser(description="复现表5可识别的心理机制检验")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--execution-mode",
        choices=("auto", "in-memory", "low-memory"),
        default="auto",
        help="执行模式；自动模式对正式全样本使用低内存缓存",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="低内存Parquet列式缓存路径",
    )
    parser.add_argument(
        "--chunk-size",
        type=_positive_int,
        default=DEFAULT_CHUNK_SIZE,
        help="低内存模式每次读取的原始CSV行数",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="忽略已有缓存并重新构建",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="正式输出成功后仍保留低内存缓存",
    )
    parser.add_argument(
        "--date-fixed-effects",
        action="store_true",
        help="另行估计并输出加入交易日期固定效应的版本",
    )
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument(
        "--nrows",
        type=_positive_int,
        help="仅用于开发试跑的CSV前N行；正式运行请省略",
    )
    parser.add_argument("--check-only", action="store_true", help="仅校验输入CSV表头")
    return parser.parse_args(argv)


def resolve_execution_mode(args: argparse.Namespace) -> str:
    """把自动模式解析为本次实际采用的内存策略。"""
    if args.execution_mode != "auto":
        return str(args.execution_mode)
    return "in-memory" if args.nrows is not None else "low-memory"


def main(argv: Sequence[str] | None = None) -> None:
    """执行表头校验、数据读取、34个模型和十二个文件输出。"""
    started = time.perf_counter()
    args = parse_args(argv)
    if (
        args.start_year is not None
        and args.end_year is not None
        and args.start_year > args.end_year
    ):
        raise ValueError("开始年份不能晚于结束年份。")

    read_and_validate_header(args.input)
    if args.check_only:
        print(f"表头校验通过：共需{len(required_input_columns())}个字段。")
        return

    execution_mode = resolve_execution_mode(args)
    fixed_effect_label = (
        "被提示公司×年度季度和交易日期固定效应"
        if args.date_fixed_effects
        else "被提示公司×年度季度固定效应"
    )
    print(f"固定效应口径：{fixed_effect_label}。")
    if execution_mode == "in-memory":
        print("执行模式：内存模式。")
        print("读取表5所需的被提示公司×交易日日度面板……")
        data = load_input_data(
            args.input,
            start_year=args.start_year,
            end_year=args.end_year,
            nrows=args.nrows,
        )
        print(
            f"数据读取完成：{len(data):,}行，"
            f"{data['StkcdRecall'].nunique():,}家被提示公司。"
        )
        if args.date_fixed_effects:
            main_results, coefficient_details = run_all_models(
                data,
                date_fixed_effects=True,
            )
        else:
            main_results, coefficient_details = run_all_models(data)
        del data
        gc.collect()
    else:
        print("执行模式：低内存分块缓存模式。")
        reusable = (
            not args.rebuild_cache
            and cache_is_reusable(
                args.input,
                args.cache_path,
                start_year=args.start_year,
                end_year=args.end_year,
                nrows=args.nrows,
            )
        )
        if reusable:
            print(f"复用低内存缓存：{args.cache_path.resolve()}")
        else:
            print("分块构建表5低内存列式缓存……")
            manifest = build_low_memory_cache(
                args.input,
                args.cache_path,
                start_year=args.start_year,
                end_year=args.end_year,
                nrows=args.nrows,
                chunk_size=args.chunk_size,
            )
            print(
                f"缓存构建完成：{int(manifest['row_count']):,}行，"
                f"{int(manifest['company_count']):,}家被提示公司。"
            )
        if args.date_fixed_effects:
            main_results, coefficient_details = run_all_models_streaming(
                args.cache_path,
                batch_size=args.chunk_size,
                date_fixed_effects=True,
            )
        else:
            main_results, coefficient_details = run_all_models_streaming(
                args.cache_path,
                batch_size=args.chunk_size,
            )

    if args.date_fixed_effects:
        written = write_outputs(
            main_results,
            coefficient_details,
            args.result_dir,
            args.image_dir,
            date_fixed_effects=True,
        )
    else:
        written = write_outputs(
            main_results,
            coefficient_details,
            args.result_dir,
            args.image_dir,
        )
    if execution_mode == "low-memory" and not args.keep_cache:
        args.cache_path.unlink(missing_ok=True)
        cache_manifest_path(args.cache_path).unlink(missing_ok=True)
        print("正式输出成功，已清理低内存缓存。")
    elapsed = time.perf_counter() - started
    print(
        f"表5复现完成：34个模型、{len(written)}个输出文件，"
        f"总耗时{elapsed / 60:.1f}分钟。"
    )
    for path in written:
        print(f"已保存：{path.resolve()}")


if __name__ == "__main__":
    main()
