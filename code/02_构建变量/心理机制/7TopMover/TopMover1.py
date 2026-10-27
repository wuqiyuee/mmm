import pandas as pd
from pathlib import Path
import numpy as np

# =============================================================================
## 路径设置
# =============================================================================

ROOT = Path(__file__).resolve().parents[4]
TRD_Dalyr_DGTW_path = ROOT / '中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv'
OUT_DIR = ROOT / '中间结果/02_变量构造结果/7TopMover'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出模式：both=同时输出两套，top100=只输出前100名，top10pct=只输出前10%
OUTPUT_MODE = "both"
TOP100_N = 100
TOP10_RATE = 0.10
TOP100_OUT = OUT_DIR / "TopMover1_top100.csv"
TOP10_OUT = OUT_DIR / "TopMover1_top10pct.csv"

VALID_OUTPUT_MODES = {"both", "top100", "top10pct"}
if OUTPUT_MODE not in VALID_OUTPUT_MODES:
    raise ValueError(
        f"OUTPUT_MODE 必须是 {VALID_OUTPUT_MODES}，当前是 {OUTPUT_MODE}"
    )

# 输出字段
OUTPUT_COLS = [
    "Stkcd",
    "Trddt",
    "ret",
    "size",
    "abs_ret",
    "top_ret_rank",
]

#===============================================================================
## 工具函数
#===============================================================================

def normalize_stock_code(x: object) -> str:
    """标准化股票代码，填充为6位数，并转换为字符串"""
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s == '':
        return ''
    return f"{int(float(s)):06d}"


def load_daily_panel(input_path):
    """读取日度股票面板，并删除收益率或规模缺失的观测。"""
    df = pd.read_csv(
        input_path,
        encoding="utf-8",
        usecols=["Stkcd", "td", "Trddt", "ret", "size"],
        parse_dates=["Trddt"]
    )

    df["Stkcd"] = df["Stkcd"].apply(normalize_stock_code)

    ret_missing = int(df["ret"].isna().sum())
    size_missing = int(df["size"].isna().sum())

    df = df.dropna(
        subset=["ret", "size"]
    ).copy()

    return df, ret_missing, size_missing


def prepare_ranked_panel(df, top10_rate):
    """剔除每日最低30%规模股票，并完成两套口径共用的排名。"""
    df = df.copy()

    # 每个交易日计算规模30%分位点
    df["size_p30"] = (
        df
        .groupby("Trddt")["size"]
        .transform(lambda s: s.quantile(0.30))
    )

    # 剔除每日规模最小的30%
    df = df.loc[
        df["size"] > df["size_p30"]
    ].copy()

    # 计算剔除最低30%后，每天剩余的股票数量
    df["eligible_count"] = (
        df
        .groupby("Trddt")["Stkcd"]
        .transform("size")
    )

    # 计算每天前10%对应的股票数量
    df["top10_n"] = np.ceil(
        df["eligible_count"] * top10_rate
    ).astype("int64")

    # 计算绝对收益率
    df["abs_ret"] = df["ret"].abs()

    # 每天按绝对收益率、规模和股票代码排序
    df = df.sort_values(
        ["Trddt", "abs_ret", "size", "Stkcd"],
        ascending=[True, False, False, True]
    )

    # 计算每天的排名
    df["top_ret_rank"] = (
        df
        .groupby("Trddt")
        .cumcount()
        + 1
    )

    return df


def select_variant(df, variant, top100_n):
    """按照指定口径选择 TopMover 股票。"""
    if variant == "top100":
        mask = df["top_ret_rank"] <= top100_n

    elif variant == "top10pct":
        mask = df["top_ret_rank"] <= df["top10_n"]

    else:
        raise ValueError(
            "variant 必须是 'top100' 或 'top10pct'"
        )

    selected = df.loc[
        mask,
        OUTPUT_COLS
    ].copy()

    selected["topMover"] = 1

    return selected

def save_variant(selected, output_path, label):
    """保存已经筛选好的 TopMover 结果。"""
    selected.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    daily_counts = selected.groupby("Trddt").size()

    print(f"\n{label}输出完成")
    print("输出文件：", output_path)
    print("TopMover 行数：", len(selected))
    print("TopMover 日期数：", selected["Trddt"].nunique())

    if not daily_counts.empty:
        print("每天入选数量最小值：", daily_counts.min())
        print("每天入选数量平均值：", daily_counts.mean())
        print("每天入选数量最大值：", daily_counts.max())

def main():
    """执行 TopMover 两套口径的完整流程。"""

    # 读取并清洗日度股票面板
    TRD_Dalyr, na1, na2 = load_daily_panel(
        TRD_Dalyr_DGTW_path
    )

    # 剔除最低30%，计算前10%数量并完成每日排名
    TRD_Dalyr = prepare_ranked_panel(
        TRD_Dalyr,
        TOP10_RATE
    )

    print("原始 ret 缺失数：", na1)
    print("原始 size 缺失数：", na2)
    print(
        "剔除缺失和最低30%规模后样本数：",
        len(TRD_Dalyr)
    )

    # 输出前100名口径
    if OUTPUT_MODE in {"both", "top100"}:
        save_variant(
            select_variant(
                TRD_Dalyr,
                "top100",
                TOP100_N
            ),
            TOP100_OUT,
            "前100名口径"
        )

    # 输出前10%口径
    if OUTPUT_MODE in {"both", "top10pct"}:
        save_variant(
            select_variant(
                TRD_Dalyr,
                "top10pct",
                TOP100_N
            ),
            TOP10_OUT,
            "前10%口径"
        )


if __name__ == "__main__":
    main()
