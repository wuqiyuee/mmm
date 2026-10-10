"""
DGTW 特征调整收益率全流程脚本。

这个脚本把三步 DGTW 流程放在同一个入口里：
1. 生成股票-形成年度层面的 DGTW 组合标签；
2. 生成交易日-DGTW 组合层面的市值加权基准收益；
3. 生成股票-交易日层面的 DGTW 特征调整收益率。

脚本默认覆盖已有输出。如需跳过已有输出，显式添加 --no-overwrite。

此脚本与 窗口有效性口径 文件下的脚本区别在于：
未对 被解释变量累计收益窗口期内的填充观测（停牌或暂停上市）作出一定比例或者严格的要求。
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


ROOT = Path("/Users/qiyuewu/Desktop/mmm复现数据包")

BASE_SAMPLE_INDEX_PATH = ROOT / "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv"
DAILY_VARIABLE_PANEL_PATH = ROOT / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv"
BALANCE_SHEET_PATH = ROOT / "input/CSMAR/资产负债表1990-12-31至2026-06-30/FS_Combas.csv"

DGTW_OUTPUT_DIR = ROOT / "中间结果/02_变量构造结果/1DGTW特征调整收益率"
PORTFOLIO_ASSIGN_PATH = DGTW_OUTPUT_DIR / "DGTW形成期组合标签.csv"
BENCHMARK_RETURN_PATH = DGTW_OUTPUT_DIR / "DGTW组合基准收益.csv"
ADJUSTED_RETURN_PATH = DGTW_OUTPUT_DIR / "DGTW特征调整收益率.csv"

ALL_STEPS = ("labels", "benchmark", "adjusted")
LAG_RETURN_OFFSETS = (3, 2, 1)
FORWARD_RETURN_HORIZONS = tuple(range(1, 51))
BENCHMARK_RETURN_SPECS = (
    tuple((f"raw_ret_tm{i}", f"bench_ret_tm{i}") for i in LAG_RETURN_OFFSETS)
    + (("raw_ret_t", "bench_ret_t"),)
    + tuple(
        (f"raw_ret_t_t{i}", f"bench_ret_t_t{i}")
        for i in FORWARD_RETURN_HORIZONS
    )
)
ADJUSTED_RETURN_SPECS = tuple(
    (raw_col, bench_col, bench_col.replace("bench_ret_", "chAdj_ret_", 1))
    for raw_col, bench_col in BENCHMARK_RETURN_SPECS
)
RAW_RETURN_COLUMNS = [raw_col for raw_col, _ in BENCHMARK_RETURN_SPECS]
BENCHMARK_COLUMNS = [bench_col for _, bench_col in BENCHMARK_RETURN_SPECS]
ADJUSTED_RETURN_COLUMNS = [adjusted_col for _, _, adjusted_col in ADJUSTED_RETURN_SPECS]
BENCHMARK_VARIABLE_COLUMNS = [
    "Stkcd",
    "td",
    *RAW_RETURN_COLUMNS,
    "float_mktcap_tm3",
    "float_mktcap_tm3_valid",
]
ADJUSTED_OUTPUT_COLUMNS = [
    "Stkcd",
    "Trddt",
    "td",
    "formation_year",
    "portfolioNumber",
    *ADJUSTED_RETURN_COLUMNS,
]


FINAL_OUTPUT_START_DATE = pd.Timestamp("2000-01-01")
FINAL_OUTPUT_END_DATE = pd.Timestamp("2025-12-31")


def normalize_stock_code(value):
    """标准化单个股票代码为六位字符串。"""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text == "":
        return pd.NA
    match = re.search(r"\d{1,6}", text)
    if match is None:
        return pd.NA
    return match.group(0).zfill(6)


def normalize_stock_code_series(series):
    """标准化一列股票代码为六位字符串。"""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def require_columns(df, columns, name):
    """检查 DataFrame 是否包含必要字段。"""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要变量：{missing}")


def assert_unique_key(df, key_cols, name):
    """检查指定字段组合是否唯一。"""
    duplicate_count = df.duplicated(key_cols).sum()
    if duplicate_count:
        raise ValueError(f"{name} 的 {key_cols} 存在重复：{duplicate_count:,} 条")


def resolve_steps(steps):
    """将命令行步骤解析为 labels -> benchmark -> adjusted 的依赖顺序。"""
    if "all" in steps:
        return list(ALL_STEPS)
    requested = set(steps)
    return [step for step in ALL_STEPS if step in requested]


def should_run_step(output_path, overwrite):
    """判断某一步是否应该运行：允许覆盖，或输出尚不存在时运行。"""
    return overwrite or not Path(output_path).exists()


def normalize_bool_flag(series, name):
    """将 True/False、1/0、yes/no 等常见标记统一为 bool。"""
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
            bad_values.add(value)
            return False
        bad_values.add(value)
        return False

    out = series.map(convert_one).astype(bool)
    if bad_values:
        preview = sorted(map(str, bad_values))[:5]
        raise ValueError(f"{name} 存在无法识别的布尔取值：{preview}")
    return out


def add_holding_formation_year(df):
    """
    生成交易日对应的 DGTW 形成年度。

    DGTW 组合每年 7 月开始生效，到下一年 6 月结束；
    因此 1-6 月交易日使用上一年 6 月形成的组合标签。
    """
    out = df.copy()
    trddt = pd.to_datetime(out["Trddt"], errors="coerce")
    out["formation_year"] = np.where(
        trddt.dt.month <= 6,
        trddt.dt.year - 1,
        trddt.dt.year,
    ).astype("int16")
    return out


def assign_5groups(x, p20, p40, p60, p80):
    """根据 20/40/60/80 分位点把变量划分为五组。"""
    return np.select(
        [
            x <= p20,
            (x > p20) & (x <= p40),
            (x > p40) & (x <= p60),
            (x > p60) & (x <= p80),
            x > p80,
        ],
        [1, 2, 3, 4, 5],
        default=np.nan,
    )


def read_daily_panel_for_labels():
    """读取构造 DGTW 标签所需的日度字段。"""
    usecols = [
        "Stkcd",
        "td",
        "Trddt",
        "Markettype",
        "ret",
        "sizew",
        "size",
        "trade_days_1y",
    ]
    daily = pd.read_csv(
        DAILY_VARIABLE_PANEL_PATH,
        encoding="utf-8",
        usecols=usecols,
        dtype={"Stkcd": "string"},
        parse_dates=["Trddt"],
    )
    require_columns(daily, usecols, "DGTW 构建用日度面板")
    daily["Stkcd"] = normalize_stock_code_series(daily["Stkcd"])
    daily = daily.sort_values(["Stkcd", "td"]).reset_index(drop=True)
    daily["month"] = daily["Trddt"].dt.month
    daily["year"] = daily["Trddt"].dt.year
    return daily


def add_momentum_sorting_variable(daily):
    """
    计算形成期动量变量 return_tm252_tm20。

    口径为过去 252 个交易日至过去 20 个交易日之间的复合收益。
    """
    out = daily.copy()
    out["log_1p_ret"] = np.log1p(out["ret"])
    out["logret_rolling_252"] = (
        out.groupby("Stkcd")["log_1p_ret"]
        .rolling(window=252, min_periods=252)
        .sum()
        .reset_index(level=0, drop=True)
    )
    out["logret_rolling_20"] = (
        out.groupby("Stkcd")["log_1p_ret"]
        .rolling(window=20, min_periods=20)
        .sum()
        .reset_index(level=0, drop=True)
    )
    out["return_tm252_tm20"] = (
        np.expm1(out["logret_rolling_252"] - out["logret_rolling_20"]) * 100
    )
    return out.drop(columns=["log_1p_ret", "logret_rolling_252", "logret_rolling_20"])


def extract_market_month_end(daily, month):
    """提取每年指定月份的全市场最后一个交易日观测。"""
    month_panel = daily.loc[daily["month"].eq(month)].copy()
    month_panel["last_month_trading_day"] = month_panel.groupby("year")["Trddt"].transform("max")
    out = month_panel.loc[
        month_panel["Trddt"].eq(month_panel["last_month_trading_day"])
    ].copy()
    return out.drop(columns=["last_month_trading_day"])


def load_book_equity():
    """读取资产负债表并计算普通股账面价值 be。"""
    comp = pd.read_csv(
        BALANCE_SHEET_PATH,
        encoding="utf-8",
        usecols=[
            "Stkcd",
            "Accper",
            "Typrep",
            "A001222000",
            "A002208000",
            "A003112000",
            "A003112101",
            "A003112201",
            "A003100000",
            "A003000000",
        ],
        parse_dates=["Accper"],
    )
    comp = comp.loc[comp["Typrep"].eq("A")].copy()
    comp = comp.rename(
        columns={
            "A001222000": "deferred_tax_assets",
            "A002208000": "deferred_tax_liabilities",
            "A003112000": "other_equity_instruments",
            "A003112101": "preferred_stock",
            "A003112201": "perpetual_bond",
            "A003100000": "equity_parent",
            "A003000000": "equity_total",
        }
    )
    comp["Stkcd"] = comp["Stkcd"].apply(normalize_stock_code)
    comp = comp.loc[comp["Accper"].dt.month.eq(12)].copy()
    comp["year"] = comp["Accper"].dt.year

    value_cols = [
        "deferred_tax_assets",
        "deferred_tax_liabilities",
        "other_equity_instruments",
        "preferred_stock",
        "perpetual_bond",
        "equity_parent",
        "equity_total",
    ]
    for col in value_cols:
        comp[col] = pd.to_numeric(comp[col], errors="coerce")

    comp["eq"] = np.where(
        comp["equity_parent"].isna(),
        comp["equity_total"],
        comp["equity_parent"],
    )
    comp["preferred_stock"] = comp["preferred_stock"].fillna(0)
    comp["perpetual_bond"] = comp["perpetual_bond"].fillna(0)
    comp["preferred_plus_perpetual"] = comp["preferred_stock"] + comp["perpetual_bond"]
    comp["other_equity_instruments"] = comp["other_equity_instruments"].fillna(
        comp["preferred_plus_perpetual"]
    )
    comp["deferred_tax_liabilities"] = comp["deferred_tax_liabilities"].fillna(0)
    comp["deferred_tax_assets"] = comp["deferred_tax_assets"].fillna(0)
    comp["be"] = (
        comp["eq"]
        - comp["other_equity_instruments"]
        + comp["deferred_tax_liabilities"]
        - comp["deferred_tax_assets"]
    )
    comp = comp.loc[comp["be"].ge(0), ["Stkcd", "Accper", "year", "be"]].copy()
    return comp


def build_dgtw_portfolio_labels():
    """生成股票-形成年度层面的 DGTW 组合标签。"""
    daily = add_momentum_sorting_variable(read_daily_panel_for_labels())

    june = extract_market_month_end(daily, month=6)
    june["year_merge"] = june["year"] - 1

    december = extract_market_month_end(daily, month=12)
    december = december.rename(
        columns={"size": "size_prev", "sizew": "sizew_prev", "year": "year_merge"}
    )

    june = june.merge(
        december[["Stkcd", "year_merge", "size_prev", "sizew_prev"]],
        how="left",
        on=["Stkcd", "year_merge"],
        validate="m:1",
    )

    comp = load_book_equity().rename(columns={"year": "year_merge"})
    sortvars = comp.merge(june, how="inner", on=["Stkcd", "year_merge"])
    sortvars = sortvars.loc[sortvars["size_prev"].gt(0)].copy()
    sortvars["bm"] = sortvars["be"] / sortvars["size_prev"]
    sortvars = sortvars.loc[sortvars["trade_days_1y"].ge(120)].copy()
    sortvars = sortvars.dropna(subset=["size", "bm", "return_tm252_tm20"]).copy()

    break_univ = sortvars.loc[sortvars["Markettype"].isin([1, 4])].copy()
    quantiles = [0.2, 0.4, 0.6, 0.8]
    breakpoints = (
        break_univ.groupby("year")[["size", "bm", "return_tm252_tm20"]]
        .quantile(quantiles)
        .unstack()
    )
    breakpoints.columns = [f"{var}_p{int(q * 100)}" for var, q in breakpoints.columns]
    sortvars = sortvars.merge(
        breakpoints.reset_index(),
        on="year",
        how="left",
        validate="m:1",
    )

    sortvars["sizeGroup"] = assign_5groups(
        sortvars["size"],
        sortvars["size_p20"],
        sortvars["size_p40"],
        sortvars["size_p60"],
        sortvars["size_p80"],
    )
    sortvars["bookToMarketGroup"] = assign_5groups(
        sortvars["bm"],
        sortvars["bm_p20"],
        sortvars["bm_p40"],
        sortvars["bm_p60"],
        sortvars["bm_p80"],
    )
    sortvars["momentumGroup"] = assign_5groups(
        sortvars["return_tm252_tm20"],
        sortvars["return_tm252_tm20_p20"],
        sortvars["return_tm252_tm20_p40"],
        sortvars["return_tm252_tm20_p60"],
        sortvars["return_tm252_tm20_p80"],
    )
    sortvars = sortvars.dropna(
        subset=["sizeGroup", "bookToMarketGroup", "momentumGroup"]
    ).copy()
    sortvars[["sizeGroup", "bookToMarketGroup", "momentumGroup"]] = (
        sortvars[["sizeGroup", "bookToMarketGroup", "momentumGroup"]].astype("int8")
    )
    sortvars["portfolioNumber"] = (
        (sortvars["sizeGroup"] - 1) * 25
        + (sortvars["bookToMarketGroup"] - 1) * 5
        + sortvars["momentumGroup"]
    ).astype("int16")
    sortvars["dateBeg"] = pd.to_datetime(sortvars["year"].astype(int).astype(str) + "-07-01")
    sortvars["dateEnd"] = pd.to_datetime(
        (sortvars["year"].astype(int) + 1).astype(str) + "-06-30"
    )

    portfolio_assign = sortvars[
        [
            "Stkcd",
            "year",
            "Trddt",
            "dateBeg",
            "dateEnd",
            "Markettype",
            "size",
            "bm",
            "return_tm252_tm20",
            "sizeGroup",
            "bookToMarketGroup",
            "momentumGroup",
            "portfolioNumber",
        ]
    ].drop_duplicates(["Stkcd", "year"])
    portfolio_assign = portfolio_assign.rename(
        columns={"year": "formation_year", "Trddt": "formation_date"}
    ).copy()
    assert_unique_key(portfolio_assign, ["Stkcd", "formation_year"], "DGTW 组合标签")
    return portfolio_assign


def load_portfolio_assign():
    """读取已生成的 DGTW 组合标签。"""
    portfolio = pd.read_csv(
        PORTFOLIO_ASSIGN_PATH,
        encoding="utf-8-sig",
        parse_dates=["formation_date", "dateBeg", "dateEnd"],
        dtype={"Stkcd": "string"},
    )
    portfolio["Stkcd"] = normalize_stock_code_series(portfolio["Stkcd"])
    portfolio["formation_year"] = pd.to_numeric(
        portfolio["formation_year"], errors="coerce"
    ).astype("int16")
    portfolio["portfolioNumber"] = pd.to_numeric(
        portfolio["portfolioNumber"], errors="coerce"
    ).astype("int16")
    assert_unique_key(portfolio, ["Stkcd", "formation_year"], "DGTW 组合标签")
    return portfolio


def read_base_sample_index():
    """读取基础清洗后股票-交易日索引。"""
    sample = pd.read_csv(
        BASE_SAMPLE_INDEX_PATH,
        encoding="utf-8-sig",
        usecols=["Stkcd", "Trddt", "td"],
        dtype={"Stkcd": "string"},
        parse_dates=["Trddt"],
    )
    sample["Stkcd"] = normalize_stock_code_series(sample["Stkcd"])
    sample["td"] = pd.to_numeric(sample["td"], errors="coerce").astype("int64")
    
     # 只限制收益窗口的起始日 t，不影响前后缓冲数据的使用
    output_date = sample["Trddt"].between(
        FINAL_OUTPUT_START_DATE,
        FINAL_OUTPUT_END_DATE,
    )
    sample = sample.loc[output_date].copy()
    sample = sample.reset_index(drop=True)

    assert_unique_key(
        sample,
        ["Stkcd", "td"],
        "基础清洗后个股日度样本索引",
    )
    return sample


def attach_portfolio(daily, portfolio_assign):
    """把股票-形成年度组合标签合并到股票-交易日面板。"""
    assign_cols = ["Stkcd", "formation_year", "dateBeg", "dateEnd", "portfolioNumber"]
    require_columns(portfolio_assign, assign_cols, "DGTW 组合标签")
    out = add_holding_formation_year(daily)
    out = out.merge(
        portfolio_assign[assign_cols],
        how="left",
        on=["Stkcd", "formation_year"],
        validate="m:1",
    )
    valid_date = (
        out["dateBeg"].notna()
        & out["dateEnd"].notna()
        & out["Trddt"].between(out["dateBeg"], out["dateEnd"])
    )
    out["portfolioNumber"] = out["portfolioNumber"].where(valid_date)
    return out.drop(columns=["dateBeg", "dateEnd"], errors="ignore")


def prepare_benchmark_panel(portfolio_assign):
    """准备计算 benchmark 和后续特征调整收益率可复用的日度面板。"""
    sample_index = read_base_sample_index()
    variable_panel = pd.read_csv(
        DAILY_VARIABLE_PANEL_PATH,
        encoding="utf-8",
        usecols=BENCHMARK_VARIABLE_COLUMNS,
        dtype={"Stkcd": "string"},
    )
    require_columns(variable_panel, BENCHMARK_VARIABLE_COLUMNS, "DGTW 构建用日度面板")
    variable_panel["Stkcd"] = normalize_stock_code_series(variable_panel["Stkcd"])
    variable_panel["td"] = pd.to_numeric(variable_panel["td"], errors="coerce").astype("int64")
    assert_unique_key(variable_panel, ["Stkcd", "td"], "DGTW 构建用日度面板")

    daily = sample_index.merge(
        variable_panel,
        how="left",
        on=["Stkcd", "td"],
        validate="1:1",
    )
    daily["float_mktcap_tm3_valid"] = normalize_bool_flag(
        daily["float_mktcap_tm3_valid"],
        "float_mktcap_tm3_valid",
    )
    daily["weight_tm3"] = daily["float_mktcap_tm3"]
    daily = attach_portfolio(daily, portfolio_assign)
    return daily


def benchmark_suffix(out_col):
    """提取 benchmark 字段的窗口后缀。"""
    return out_col.replace("bench_ret_", "", 1) if out_col.startswith("bench_ret_") else out_col


def make_value_weighted_benchmark(df, ret_col, out_col):
    """计算交易日-DGTW 组合层面的市值加权基准收益。"""
    suffix = benchmark_suffix(out_col)
    sum_weight_col = f"sum_weight_{suffix}"
    n_stocks_col = f"n_stocks_{suffix}"
    tmp = df.loc[
        df["float_mktcap_tm3_valid"]
        & df["portfolioNumber"].notna()
        & df["weight_tm3"].gt(0)
        & df[ret_col].notna(),
        ["Trddt", "portfolioNumber", "Stkcd", "weight_tm3", ret_col],
    ].copy()
    if tmp.empty:
        return pd.DataFrame(
            columns=["Trddt", "portfolioNumber", out_col, n_stocks_col, sum_weight_col]
        )
    tmp["weighted_ret"] = tmp["weight_tm3"] * tmp[ret_col]
    out = (
        tmp.groupby(["Trddt", "portfolioNumber"], as_index=False)
        .agg(
            **{
                sum_weight_col: ("weight_tm3", "sum"),
                "sum_weighted_ret": ("weighted_ret", "sum"),
                n_stocks_col: ("Stkcd", "nunique"),
            }
        )
    )
    out[out_col] = out["sum_weighted_ret"] / out[sum_weight_col]
    return out.drop(columns=["sum_weighted_ret"])[
        ["Trddt", "portfolioNumber", out_col, n_stocks_col, sum_weight_col]
    ]


def build_dgtw_benchmark_returns(portfolio_assign):
    """生成 DGTW 组合基准收益，并返回可复用的日度面板。"""
    daily = prepare_benchmark_panel(portfolio_assign)
    benchmark_frames = [
        make_value_weighted_benchmark(daily, ret_col=ret_col, out_col=out_col)
        for ret_col, out_col in BENCHMARK_RETURN_SPECS
    ]
    benchmark_returns = benchmark_frames[0]
    for frame in benchmark_frames[1:]:
        benchmark_returns = benchmark_returns.merge(
            frame,
            on=["Trddt", "portfolioNumber"],
            how="outer",
        )
    benchmark_returns = (
        benchmark_returns.sort_values(["Trddt", "portfolioNumber"])
        .reset_index(drop=True)
    )
    return benchmark_returns, daily


def load_benchmark_returns():
    """读取已生成的 DGTW 组合基准收益。"""
    benchmark = pd.read_csv(
        BENCHMARK_RETURN_PATH,
        encoding="utf-8-sig",
        usecols=["Trddt", "portfolioNumber", *BENCHMARK_COLUMNS],
        parse_dates=["Trddt"],
    )
    benchmark["portfolioNumber"] = pd.to_numeric(
        benchmark["portfolioNumber"], errors="coerce"
    ).astype("int16")
    assert_unique_key(benchmark, ["Trddt", "portfolioNumber"], "DGTW 组合基准收益")
    return benchmark


def calculate_adjusted_returns(df, specs=ADJUSTED_RETURN_SPECS):
    """按窗口逐列计算 raw stock return - DGTW benchmark return。"""
    out = df.copy()
    require_columns(
        out,
        [column for raw_col, bench_col, _ in specs for column in (raw_col, bench_col)],
        "特征调整收益率输入",
    )
    for raw_col, bench_col, adjusted_col in specs:
        out[adjusted_col] = out[raw_col] - out[bench_col]
    return out


def sample_keys_from_portfolio_assign(portfolio_assign):
    """为流式 adjusted 步骤构造可匹配 DGTW 组合的股票-交易日键。"""
    sample = attach_portfolio(read_base_sample_index(), portfolio_assign)
    sample = sample.loc[
        sample["portfolioNumber"].notna(),
        ["Stkcd", "td", "Trddt", "formation_year", "portfolioNumber"],
    ].copy()
    sample["portfolioNumber"] = sample["portfolioNumber"].astype("int16")
    assert_unique_key(sample, ["Stkcd", "td"], "可匹配 DGTW 组合的样本键")
    return sample


def write_adjusted_returns_from_daily_panel(daily, benchmark_returns, chunk_size):
    """使用 benchmark 阶段留下的日度面板生成特征调整收益率，避免重读大 CSV。"""
    first_chunk = True
    total_output_rows = 0
    valid_daily = daily.loc[daily["portfolioNumber"].notna()].copy()
    valid_daily["portfolioNumber"] = valid_daily["portfolioNumber"].astype("int16")

    for start in range(0, len(valid_daily), chunk_size):
        chunk = valid_daily.iloc[start : start + chunk_size].copy()
        merged = chunk.merge(
            benchmark_returns,
            on=["Trddt", "portfolioNumber"],
            how="left",
            validate="m:1",
        )
        out = calculate_adjusted_returns(merged)[ADJUSTED_OUTPUT_COLUMNS].copy()
        out.to_csv(
            ADJUSTED_RETURN_PATH,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8-sig" if first_chunk else "utf-8",
        )
        first_chunk = False
        total_output_rows += len(out)
        print(f"adjusted chunk rows={len(out):,}, total_out={total_output_rows:,}", flush=True)
    return total_output_rows


def write_adjusted_returns_from_source_chunks(portfolio_assign, benchmark_returns, chunk_size):
    """从原始日度面板按块读取并生成特征调整收益率。"""
    sample_keys = sample_keys_from_portfolio_assign(portfolio_assign)
    first_chunk = True
    total_output_rows = 0
    reader = pd.read_csv(
        DAILY_VARIABLE_PANEL_PATH,
        encoding="utf-8",
        usecols=["Stkcd", "td", *RAW_RETURN_COLUMNS],
        dtype={"Stkcd": "string"},
        chunksize=chunk_size,
    )
    for chunk_number, raw_chunk in enumerate(reader, start=1):
        raw_chunk["Stkcd"] = normalize_stock_code_series(raw_chunk["Stkcd"])
        raw_chunk["td"] = pd.to_numeric(raw_chunk["td"], errors="coerce").astype("int64")
        merged = raw_chunk.merge(
            sample_keys,
            on=["Stkcd", "td"],
            how="inner",
            validate="1:1",
        )
        if merged.empty:
            print(f"adjusted chunk {chunk_number}: matched=0", flush=True)
            continue
        merged = merged.merge(
            benchmark_returns,
            on=["Trddt", "portfolioNumber"],
            how="left",
            validate="m:1",
        )
        out = calculate_adjusted_returns(merged)[ADJUSTED_OUTPUT_COLUMNS].copy()
        out.to_csv(
            ADJUSTED_RETURN_PATH,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8-sig" if first_chunk else "utf-8",
        )
        first_chunk = False
        total_output_rows += len(out)
        print(
            f"adjusted chunk {chunk_number}: matched={len(out):,}, total_out={total_output_rows:,}",
            flush=True,
        )
    return total_output_rows


def save_dataframe(df, path):
    """保存 DataFrame 到 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run the DGTW full workflow.")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", *ALL_STEPS],
        help="Steps to run. Default: all.",
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    overwrite_group.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip steps whose outputs already exist.",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Chunk size for adjusted return writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned steps and output paths without running calculations.",
    )
    return parser.parse_args()


def print_plan(steps, overwrite):
    """打印本次计划运行的步骤和输出路径。"""
    print(f"steps: {steps}")
    print(f"overwrite: {overwrite}")
    print(f"labels output: {PORTFOLIO_ASSIGN_PATH}")
    print(f"benchmark output: {BENCHMARK_RETURN_PATH}")
    print(f"adjusted output: {ADJUSTED_RETURN_PATH}")


def main():
    """按依赖顺序运行 DGTW 全流程。"""
    args = parse_args()
    steps = resolve_steps(args.steps)
    print_plan(steps, args.overwrite)
    if args.dry_run:
        return

    portfolio_assign = None
    benchmark_returns = None
    benchmark_daily_panel = None

    if "labels" in steps:
        if should_run_step(PORTFOLIO_ASSIGN_PATH, args.overwrite):
            print("开始生成 DGTW 组合标签...", flush=True)
            portfolio_assign = build_dgtw_portfolio_labels()
            save_dataframe(portfolio_assign, PORTFOLIO_ASSIGN_PATH)
            print(f"DGTW 组合标签已保存：{PORTFOLIO_ASSIGN_PATH}", flush=True)
        else:
            print(f"跳过 labels：输出已存在 {PORTFOLIO_ASSIGN_PATH}", flush=True)

    if any(step in steps for step in ["benchmark", "adjusted"]):
        portfolio_assign = portfolio_assign if portfolio_assign is not None else load_portfolio_assign()

    if "benchmark" in steps:
        if should_run_step(BENCHMARK_RETURN_PATH, args.overwrite):
            print("开始生成 DGTW 组合基准收益...", flush=True)
            benchmark_returns, benchmark_daily_panel = build_dgtw_benchmark_returns(portfolio_assign)
            save_dataframe(benchmark_returns, BENCHMARK_RETURN_PATH)
            print(f"DGTW 组合基准收益已保存：{BENCHMARK_RETURN_PATH}", flush=True)
        else:
            print(f"跳过 benchmark：输出已存在 {BENCHMARK_RETURN_PATH}", flush=True)

    if "adjusted" in steps:
        if should_run_step(ADJUSTED_RETURN_PATH, args.overwrite):
            benchmark_returns = (
                benchmark_returns if benchmark_returns is not None else load_benchmark_returns()
            )
            print("开始生成 DGTW 特征调整收益率...", flush=True)
            if benchmark_daily_panel is not None:
                rows = write_adjusted_returns_from_daily_panel(
                    benchmark_daily_panel,
                    benchmark_returns,
                    args.chunk_size,
                )
            else:
                rows = write_adjusted_returns_from_source_chunks(
                    portfolio_assign,
                    benchmark_returns,
                    args.chunk_size,
                )
            print(f"DGTW 特征调整收益率已保存：{ADJUSTED_RETURN_PATH}", flush=True)
            print(f"写出行数：{rows:,}", flush=True)
        else:
            print(f"跳过 adjusted：输出已存在 {ADJUSTED_RETURN_PATH}", flush=True)


if __name__ == "__main__":
    main()
