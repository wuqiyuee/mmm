
"""使用现成官方标签，构造供下游合并的基础特征调整收益率。

标签直接读取国泰安股票分配表，只在内存中整理字段和编码，不另存标签表。
官方三个分组原值为1高、5低，保留这一含义，不与自建标签的组合编号混用。

计算顺序：读取官方标签 → 准备基础公司日 → 逐窗计算组合基准
→ 计算个股收益减组合基准 → 保存两张基础结果表。

收益计算规则对应同目录《00_DGTW全流程.py》（特征基准主线全流程）：
1. 收益起点为2000—2025年，沿用基础清洗资格索引和日度构建面板。
2. 每年7月至次年6月使用同一形成年度标签，不补齐官方缺失年度标签。
3. 基准成员要求标签有效、滞后三日流通市值有效且大于零、该窗收益非缺失。
4. 每个窗口独立选择成员，直接对整窗收益加权，不要求54类收益共同完整。
5. 输出基础资格内且标签有效的公司日；目标公司自身不要求权重有效。
6. 共有过去3个单日、当日、50个未来累计窗口，共54类未缩尾调整收益。
   未来窗口包含起点和终点；第1窗为两日，第50窗为51日。

本文件独立运行，不调用其他项目脚本。不依赖最终回归面板，不做缩尾、
回归筛样、提示口径筛选或绘图。填充记录和窗口缺失规则沿用上游及主线。
运行时覆盖本文件指定的两个官方收益输出，不覆盖自建标签结果。
"""

from pathlib import Path

import numpy as np
import pandas as pd


# 一、输入、输出和计算参数。
ROOT = Path(__file__).resolve().parents[4]
BASE_SAMPLE_INDEX_PATH = ROOT / "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv"
DAILY_VARIABLE_PANEL_PATH = ROOT / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv"
OFFICIAL_LABEL_PATH = (
    ROOT / "input/CSMAR/DGTW（2000至2025）/DGTW股票分配表/STK_MKT_DGTWASSINGMENTS.csv"
)
OUTPUT_DIR = ROOT / "中间结果/02_变量构造结果/1DGTW特征调整收益率"
BENCHMARK_RETURN_PATH = OUTPUT_DIR / "官方标签_组合基准收益.csv"
ADJUSTED_RETURN_PATH = OUTPUT_DIR / "官方标签_特征调整收益率.csv"

START_DATE = "2000-01-01"
END_DATE = "2025-12-31"
CHUNK_SIZE = 250_000

# 后缀tm3/tm2/tm1为过去第3/2/1个单日，t为当日，t_t1至t_t50为未来累计窗。
RETURN_SUFFIXES = ["tm3", "tm2", "tm1", "t"] + [f"t_t{i}" for i in range(1, 51)]
RAW_RETURN_COLUMNS = [f"raw_ret_{suffix}" for suffix in RETURN_SUFFIXES]
BENCHMARK_COLUMNS = [f"bench_ret_{suffix}" for suffix in RETURN_SUFFIXES]
ADJUSTED_RETURN_COLUMNS = [f"chAdj_ret_{suffix}" for suffix in RETURN_SUFFIXES]

# 五个识别字段依次为公司代码、交易日期、交易日序号、形成年度和组合编号。
ADJUSTED_OUTPUT_COLUMNS = [
    "Stkcd", "Trddt", "td", "formation_year", "portfolioNumber",
    *ADJUSTED_RETURN_COLUMNS,
]


def normalize_stock_codes(series):
    """与主线一致，将股票代码整理成六位字符串。"""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def assert_unique_key(frame, columns, name):
    """检查合并键，避免重复观测改变组合权重。"""
    duplicate_count = frame.duplicated(columns).sum()
    if duplicate_count:
        raise ValueError(f"{name}的{columns}存在重复：{duplicate_count:,}条。")


def normalize_bool_flag(series):
    """沿用主线的权重有效标志解释方式。"""
    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n", ""}
    bad_values = set()

    def convert_one(value):
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in true_values:
                return True
            if lowered in false_values:
                return False
        bad_values.add(str(value))
        return False

    result = series.map(convert_one).astype(bool)
    if bad_values:
        raise ValueError(f"流通市值有效标志存在无法识别的取值：{sorted(bad_values)[:5]}。")
    return result


# 二、直接读取官方年度标签，只在内存中建立合并所需字段。
def read_official_labels():
    """保留非北交所版本，不重新分组、不反转分组、不导出标签。"""
    labels = pd.read_csv(
        OFFICIAL_LABEL_PATH,
        encoding="utf-8-sig",
        usecols=["Symbol", "TradingYear", "IsNotBSE", "MarketValue", "BooktoMarket", "Momentum"],
        dtype={"Symbol": "string"},
    )
    labels = labels.loc[labels["IsNotBSE"].eq(1)].copy()
    labels = labels.rename(columns={"Symbol": "Stkcd", "TradingYear": "formation_year"})
    labels["Stkcd"] = normalize_stock_codes(labels["Stkcd"])
    if labels.empty or labels[["Stkcd", "formation_year"]].isna().any().any():
        raise ValueError("官方非北交所标签为空，或公司、形成年度缺失。")
    year = pd.to_numeric(labels["formation_year"], errors="raise")
    if not year.mod(1).eq(0).all():
        raise ValueError("官方标签的形成年度必须为整数。")
    labels["formation_year"] = year.astype("int16")
    assert_unique_key(labels, ["Stkcd", "formation_year"], "官方年度标签")

    # 三列原始含义依次为规模分组、账面市值比分组、动量分组，均为1至5。
    for column in ["MarketValue", "BooktoMarket", "Momentum"]:
        if not labels[column].isin([1, 2, 3, 4, 5]).all():
            raise ValueError(f"官方分组不是1至5的整数：{column}。")
    labels["portfolioNumber"] = (
        (labels["MarketValue"] - 1) * 25
        + (labels["BooktoMarket"] - 1) * 5
        + labels["Momentum"]
    ).astype("int16")
    labels["dateBeg"] = pd.to_datetime(labels["formation_year"].astype(str) + "-07-01")
    labels["dateEnd"] = pd.to_datetime((labels["formation_year"] + 1).astype(str) + "-06-30")
    return labels[["Stkcd", "formation_year", "dateBeg", "dateEnd", "portfolioNumber"]]


# 三、以基础资格索引为左表，匹配原始收益、权重和年度组合。
def prepare_daily_panel(labels):
    """输出资格池内的公司日；此处不按权重或窗口收益删除目标行。"""
    sample = pd.read_csv(
        BASE_SAMPLE_INDEX_PATH,
        encoding="utf-8-sig",
        usecols=["Stkcd", "Trddt", "td"],
        dtype={"Stkcd": "string"},
        parse_dates=["Trddt"],
    )
    sample["Stkcd"] = normalize_stock_codes(sample["Stkcd"])
    sample["td"] = pd.to_numeric(sample["td"], errors="raise").astype("int64")
    sample = sample.loc[sample["Trddt"].between(START_DATE, END_DATE)].reset_index(drop=True)
    assert_unique_key(sample, ["Stkcd", "td"], "基础清洗资格索引")
    if sample.empty:
        raise ValueError("指定起点日期范围内没有基础清洗资格记录。")

    variable_columns = [
        "Stkcd", "td", *RAW_RETURN_COLUMNS,
        "float_mktcap_tm3", "float_mktcap_tm3_valid",
    ]
    variables = pd.read_csv(
        DAILY_VARIABLE_PANEL_PATH,
        encoding="utf-8",
        usecols=variable_columns,
        dtype={"Stkcd": "string"},
    )
    variables["Stkcd"] = normalize_stock_codes(variables["Stkcd"])
    variables["td"] = pd.to_numeric(variables["td"], errors="raise").astype("int64")
    assert_unique_key(variables, ["Stkcd", "td"], "日度构建面板")

    # 左合并与主线相同：源表未覆盖的资格键仍保留，其收益和权重为空。
    daily = sample.merge(variables, on=["Stkcd", "td"], how="left", validate="1:1")
    daily["float_mktcap_tm3_valid"] = normalize_bool_flag(daily["float_mktcap_tm3_valid"])
    daily["weight_tm3"] = daily["float_mktcap_tm3"]
    daily["formation_year"] = (
        daily["Trddt"].dt.year - daily["Trddt"].dt.month.le(6).astype(int)
    ).astype("int16")
    daily = daily.merge(labels, on=["Stkcd", "formation_year"], how="left", validate="m:1")
    valid_date = (
        daily["dateBeg"].notna()
        & daily["dateEnd"].notna()
        & daily["Trddt"].between(daily["dateBeg"], daily["dateEnd"])
    )
    daily["portfolioNumber"] = daily["portfolioNumber"].where(valid_date)
    return daily.drop(columns=["dateBeg", "dateEnd"])


# 四、逐窗口独立选择基准成员，对整窗收益直接作市值加权平均。
def calculate_benchmark_returns(daily):
    """每窗同时保存基准、股票数及权重总和，字段与主线一致。"""
    frames = []
    for suffix, raw_column, benchmark_column in zip(
        RETURN_SUFFIXES, RAW_RETURN_COLUMNS, BENCHMARK_COLUMNS,
    ):
        sum_weight_column = f"sum_weight_{suffix}"
        n_stocks_column = f"n_stocks_{suffix}"
        members = daily.loc[
            daily["float_mktcap_tm3_valid"]
            & daily["portfolioNumber"].notna()
            & daily["weight_tm3"].gt(0)
            & daily[raw_column].notna(),
            ["Trddt", "portfolioNumber", "Stkcd", "weight_tm3", raw_column],
        ].copy()
        members["weighted_ret"] = members["weight_tm3"] * members[raw_column]
        frame = members.groupby(["Trddt", "portfolioNumber"], as_index=False).agg(
            **{
                sum_weight_column: ("weight_tm3", "sum"),
                "sum_weighted_ret": ("weighted_ret", "sum"),
                n_stocks_column: ("Stkcd", "nunique"),
            }
        )
        frame[benchmark_column] = frame["sum_weighted_ret"] / frame[sum_weight_column]
        frames.append(frame[[
            "Trddt", "portfolioNumber", benchmark_column, n_stocks_column, sum_weight_column,
        ]])
        print(f"组合基准已完成：{benchmark_column}（窗口基准收益）。", flush=True)

    benchmark = frames[0]
    for frame in frames[1:]:
        benchmark = benchmark.merge(
            frame, on=["Trddt", "portfolioNumber"], how="outer", validate="1:1",
        )
    benchmark = benchmark.sort_values(["Trddt", "portfolioNumber"]).reset_index(drop=True)
    assert_unique_key(benchmark, ["Trddt", "portfolioNumber"], "官方组合基准")
    return benchmark


# 五、仅要求目标公司标签有效；逐块计算收益差并保存，不缩尾、不额外筛样。
def write_adjusted_returns(daily, benchmark):
    """一行代表公司×交易日；缺失的个股收益或基准自然产生空调整收益。"""
    targets = daily.loc[daily["portfolioNumber"].notna()].copy()
    targets["portfolioNumber"] = targets["portfolioNumber"].astype("int16")
    first_chunk = True
    rows_written = 0
    for start in range(0, len(targets), CHUNK_SIZE):
        chunk = targets.iloc[start:start + CHUNK_SIZE].merge(
            benchmark[["Trddt", "portfolioNumber", *BENCHMARK_COLUMNS]],
            on=["Trddt", "portfolioNumber"], how="left", validate="m:1",
        )
        for raw_column, benchmark_column, adjusted_column in zip(
            RAW_RETURN_COLUMNS, BENCHMARK_COLUMNS, ADJUSTED_RETURN_COLUMNS,
        ):
            chunk[adjusted_column] = chunk[raw_column] - chunk[benchmark_column]
        chunk[ADJUSTED_OUTPUT_COLUMNS].to_csv(
            ADJUSTED_RETURN_PATH,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8-sig" if first_chunk else "utf-8",
        )
        first_chunk = False
        rows_written += len(chunk)
        print(f"未缩尾调整收益已写出{rows_written:,}行。", flush=True)
    if first_chunk:
        targets.reindex(columns=ADJUSTED_OUTPUT_COLUMNS).to_csv(
            ADJUSTED_RETURN_PATH, index=False, encoding="utf-8-sig",
        )
    return rows_written


def main():
    """依次整理标签、构造基准和保存基础收益；导入本文件不会运行。"""
    for path in [BASE_SAMPLE_INDEX_PATH, DAILY_VARIABLE_PANEL_PATH, OFFICIAL_LABEL_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入文件：{path}")
    if CHUNK_SIZE <= 0 or pd.Timestamp(START_DATE) > pd.Timestamp(END_DATE):
        raise ValueError("分块行数必须大于零，起始日期不能晚于结束日期。")

    print("读取现成官方年度标签，仅在内存中整理。", flush=True)
    labels = read_official_labels()
    print("读取基础资格、日度原始收益及滞后三日流通市值。", flush=True)
    daily = prepare_daily_panel(labels)
    print(f"基础资格公司日{len(daily):,}行，有标签{daily['portfolioNumber'].notna().sum():,}行。", flush=True)
    benchmark = calculate_benchmark_returns(daily)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(BENCHMARK_RETURN_PATH, index=False, encoding="utf-8-sig")
    print(f"官方组合基准已保存：{BENCHMARK_RETURN_PATH}", flush=True)
    rows_written = write_adjusted_returns(daily, benchmark)
    print(f"官方未缩尾调整收益已保存：{ADJUSTED_RETURN_PATH}；{rows_written:,}行。", flush=True)


if __name__ == "__main__":
    main()
