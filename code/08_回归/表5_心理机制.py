"""表5心理机制：规格映射与输入表头校验。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"


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
    "positiveNews",
    "negativeNews",
    "logIncreaseAEV",
    "logDecreaseAEV",
]

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
            "共同高异常总成交量强联系提示，与同规模其余提示比较。",
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
            "A股小单异常成交量为原文零售交易量口径的中国替代。",
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
            "共同高媒体关注强联系提示，与同规模其余提示比较。",
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
            "共同TopMover强联系提示，与同规模其余提示比较。",
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
            f"共同正向SUE{sue_id}强联系提示，与同规模其余提示比较。",
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
            f"共同负向SUE{sue_id}强联系提示，与同规模其余提示比较。",
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
            f"共同极端正向SUE{sue_id}强联系提示，与同规模其余提示比较。",
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
            f"共同极端负向SUE{sue_id}强联系提示，与同规模其余提示比较。",
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
        "按年份、等权共同公告公司数量二分的低干扰与高干扰提示比较。",
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
                    f"强联系提示字段{strong_col}为1时，对应基准提示字段"
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
                    "强联系提示",
                    "其余提示",
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
                    "低干扰提示",
                    "高干扰提示",
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
            raise ValueError(
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
                raise ValueError(
                    f"{_regression_context(outcome, spec)}："
                    f"核心变量{column}被公司×年度季度固定效应"
                    "完全吸收，无法估计。"
                )


def _significance_stars(p_value: float) -> str:
    """按双侧检验的 p 值返回常用显著性星号。"""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def run_mechanism_regression(
    data: pd.DataFrame,
    outcome: OutcomeSpec,
    spec: MechanismSpec,
    control_cols: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """在同一模型中估计四个机制变量，并返回轻量主结果与系数明细。"""
    controls = list(dict.fromkeys(control_cols))
    if "const" in controls:
        raise ValueError("控制变量不能命名为const，以免覆盖显式常数项。")

    source_cols = _mechanism_source_columns(spec)
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

    # 性能约束：先从宽表切出当前模型所需字段，再构造比较变量。
    compact = data.loc[:, needed].copy()
    materialized, pairs = materialize_comparisons(compact, spec)
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
        *core_cols,
        *controls,
    ]
    reg = materialized.loc[:, model_cols].dropna().copy()
    if reg.empty:
        raise ValueError(
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
    reg, singleton_removed = drop_fe_singletons_iteratively(
        reg,
        ["_fe_firm_yearquarter"],
    )
    if reg.empty:
        raise ValueError(
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
        raise ValueError(
            f"{_regression_context(outcome, spec)}："
            "被提示公司×年度季度固定效应至少需要两个组。"
        )
    if firm_count < 2:
        raise ValueError(
            f"{_regression_context(outcome, spec)}：被提示公司聚类至少需要两个组。"
        )
    if date_count < 2:
        raise ValueError(
            f"{_regression_context(outcome, spec)}：交易日期聚类至少需要两个组。"
        )

    absorb = pd.DataFrame(
        {
            "firm_yearquarter": pd.Categorical(
                reg["_fe_firm_yearquarter"]
            )
        },
        index=reg.index,
    )
    clusters = pd.DataFrame(
        {
            "firm": pd.factorize(reg["StkcdRecall"], sort=False)[0],
            "date": pd.factorize(reg["Trddt"], sort=False)[0],
        },
        index=reg.index,
    )
    x_cols = [*core_cols, *controls]
    exog = reg[x_cols].astype("float64")
    exog.insert(0, "const", 1.0)

    model = AbsorbingLS(
        dependent=reg[outcome.column].astype("float64"),
        exog=exog,
        absorb=absorb,
        drop_absorbed=True,
    )
    result = model.fit(
        cov_type="clustered",
        clusters=clusters,
        debiased=True,
    )

    unestimated_core = [
        column for column in core_cols if column not in result.params.index
    ]
    if unestimated_core:
        raise ValueError(
            f"{_regression_context(outcome, spec)}：核心变量"
            + "、".join(unestimated_core)
            + "被固定效应吸收或未估计。"
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
        "日期固定效应": "否",
        "聚类方式": "被提示公司和交易日期双向聚类",
        "有限样本自由度修正": "是",
        "回归样本量": int(result.nobs),
        "被提示公司聚类数": firm_count,
        "日期聚类数": date_count,
        "固定效应组数": fe_count,
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
        beta = float(result.params[variable])
        standard_error = float(result.std_errors[variable])
        t_value = float(result.tstats[variable])
        p_value = float(result.pvalues[variable])
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
                "显著性": _significance_stars(p_value),
            }
        )

    main_rows: list[dict[str, object]] = []
    for pair in pairs:
        wald = result.wald_test(
            formula=f"{pair.first_col} = {pair.second_col}"
        )
        first_beta = float(result.params[pair.first_col])
        second_beta = float(result.params[pair.second_col])
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
                "系数差": coefficient_difference,
                "系数差（百分点）": coefficient_difference * 100,
                "Wald检验式": f"{pair.first_col} = {pair.second_col}",
                "Wald统计量": float(np.asarray(wald.stat).squeeze()),
                "Wald自由度": int(wald.df),
                "Wald p值": float(wald.pval),
            }
        )

    return main_rows, coefficient_rows


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析本阶段可用的表头检查命令行参数。"""
    parser = argparse.ArgumentParser(description="表5心理机制规格与输入表头校验")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--check-only", action="store_true", help="仅校验输入CSV表头")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """运行只读表头校验。"""
    args = parse_args(argv)
    if args.check_only:
        read_and_validate_header(args.input)
        print(f"表头校验通过：共需{len(required_input_columns())}个字段。")
        return
    raise SystemExit("当前阶段仅支持 --check-only。")


if __name__ == "__main__":
    main()
