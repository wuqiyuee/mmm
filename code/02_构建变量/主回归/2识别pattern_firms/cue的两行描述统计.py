#!/usr/bin/env python3
"""从正式回归面板生成提示变量的两行描述统计。"""

from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

import pandas as pd


def find_project_root() -> Path:
    """从当前脚本位置向上查找项目根目录。"""
    here = Path(__file__).resolve()

    for parent in [here.parent, *here.parents]:
        if (parent / "code").exists() and (parent / "中间结果").exists():
            return parent

    return here.parents[3]


PROJECT_ROOT = find_project_root()

# 正式回归面板已经包含基准提示变量及提示公司数量。
REG_SAMPLE_PATH = (
    PROJECT_ROOT
    / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"
)

OUT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "中间结果/04_合并数据/cue_summary_two_rows.csv"
)

SOURCE_CUE_COL = "quarterLag1size90plus"
SOURCE_COUNT_COL = "nrCuesLag1S90p"

INPUT_COLUMNS = [
    "StkcdRecall",
    "Trddt",
    SOURCE_CUE_COL,
    SOURCE_COUNT_COL,
]


def require_columns(columns: pd.Index, required: list[str]) -> None:
    """检查输入文件是否包含必要变量。"""
    missing = sorted(set(required) - set(columns))

    if missing:
        raise ValueError(f"正式回归面板缺少必要变量：{missing}")

# 分位数采用 Pandas 默认线性插值方法
def summarize_series(series: pd.Series) -> pd.Series:
    """计算与原文表1对应的描述统计量。"""
    series = pd.to_numeric(series, errors="coerce").dropna()

    return pd.Series(
        {
            "Mean（均值）": series.mean(),
            "p25（第25百分位）": series.quantile(0.25),
            "p50（中位数）": series.quantile(0.50),
            "p75（第75百分位）": series.quantile(0.75),
            "Std. Dev.（标准差）": series.std(ddof=1),
            "Min（最小值）": series.min(),
            "Max（最大值）": series.max(),
            "N（样本量）": int(series.shape[0]),
        }
    )


def load_regression_sample(path: Path) -> pd.DataFrame:
    """读取并检查正式回归面板中的提示变量。"""
    if not path.exists():
        raise FileNotFoundError(f"正式回归面板不存在：{path}")

    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    require_columns(header.columns, INPUT_COLUMNS)

    reg = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=INPUT_COLUMNS,
        dtype={"StkcdRecall": "string"},
        low_memory=False,
    )

    reg["Trddt"] = pd.to_datetime(reg["Trddt"], errors="coerce")

    invalid_key = reg["StkcdRecall"].isna() | reg["Trddt"].isna()
    if invalid_key.any():
        raise ValueError(
            "正式回归面板存在无法识别的被提示公司代码或交易日期："
            f"{int(invalid_key.sum()):,} 行"
        )

    duplicate_count = int(
        reg.duplicated(["StkcdRecall", "Trddt"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            "正式回归面板的被提示公司—交易日期键不唯一："
            f"{duplicate_count:,} 条重复记录"
        )

    cue_source = pd.to_numeric(
        reg[SOURCE_CUE_COL],
        errors="coerce",
    )
    count_source = pd.to_numeric(
        reg[SOURCE_COUNT_COL],
        errors="coerce",
    )

    invalid_cue = cue_source.isna() | ~cue_source.isin([0, 1])
    if invalid_cue.any():
        raise ValueError(
            f"{SOURCE_CUE_COL} 存在缺失值或非0/1取值："
            f"{int(invalid_cue.sum()):,} 行"
        )

    if count_source.lt(0).fillna(False).any():
        raise ValueError(f"{SOURCE_COUNT_COL} 存在负数")

    cue_equals_one = cue_source.eq(1)

    invalid_positive_cue = (
        cue_equals_one
        & (count_source.isna() | count_source.lt(1))
    )
    if invalid_positive_cue.any():
        raise ValueError(
            "存在提示标记等于1、但提示公司数量缺失或小于1的记录："
            f"{int(invalid_positive_cue.sum()):,} 行"
        )

    invalid_zero_cue = (
        cue_source.eq(0)
        & count_source.fillna(0).gt(0)
    )
    if invalid_zero_cue.any():
        raise ValueError(
            "存在提示标记等于0、但提示公司数量大于0的记录："
            f"{int(invalid_zero_cue.sum()):,} 行"
        )

    reg["Cue"] = cue_source.astype("int8")

    # 提示公司数量只在发生提示的公司—交易日中参与统计。
    reg["Number_of_cues"] = count_source.where(
        reg["Cue"].eq(1)
    )

    return reg


def main() -> None:
    """生成提示虚拟变量和提示公司数量的两行描述统计。"""
    print("正在读取正式回归面板……")
    reg = load_regression_sample(REG_SAMPLE_PATH)

    summary = pd.DataFrame(
        {
            "Cue（提示虚拟变量）": summarize_series(reg["Cue"]),
            "Number of cues（提示公司数量）": summarize_series(
                reg["Number_of_cues"]
            ),
        }
    ).T

    summary.index.name = "变量"

    display_summary = summary.copy()
    decimal_columns = [
        "Mean（均值）",
        "p25（第25百分位）",
        "p50（中位数）",
        "p75（第75百分位）",
        "Std. Dev.（标准差）",
        "Min（最小值）",
        "Max（最大值）",
    ]

    for column in decimal_columns:
        display_summary[column] = display_summary[column].map(
            lambda value: (
                f"{value:.4f}" if pd.notna(value) else ""
            )
        )

    display_summary["N（样本量）"] = display_summary[
        "N（样本量）"
    ].map(lambda value: f"{int(value):,}")

    print("\n提示变量描述统计：")
    print(display_summary)

    print("\n关键检查量：")
    print(f"正式回归样本行数：{len(reg):,}")
    print(f"提示虚拟变量均值：{reg['Cue'].mean():.6f}")
    print(f"发生提示的公司—交易日数量：{int(reg['Cue'].sum()):,}")
    print(
        "发生提示时的平均提示公司数量："
        f"{reg.loc[reg['Cue'].eq(1), 'Number_of_cues'].mean():.6f}"
    )

    OUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        OUT_SUMMARY_PATH,
        encoding="utf-8-sig",
    )

    print(f"\n描述统计已保存至：{OUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()