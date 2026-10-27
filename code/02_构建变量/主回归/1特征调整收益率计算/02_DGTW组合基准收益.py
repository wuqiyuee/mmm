import re #用来做正则匹配，主要用于从股票代码中提取数字。
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 基础设置
# =============================================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 1000)

ROOT = Path(__file__).resolve().parents[4]

base_sample_index_path = ROOT / "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv"
daily_variable_panel_path = ROOT / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv"
portfolio_assign_path = ROOT / "中间结果/02_变量构造结果/DGTW/DGTW形成期组合标签.csv"
benchmark_out_path = ROOT / "中间结果/02_变量构造结果/DGTW/DGTW组合基准收益.csv"

LAG_RETURN_OFFSETS = (3, 2, 1)
FORWARD_RETURN_HORIZONS = tuple(range(1, 51))
LAG_RETURN_SPECS = [
    (f"raw_ret_tm{i}", f"bench_ret_tm{i}")
    for i in LAG_RETURN_OFFSETS
]
RETURN_WINDOW_SPECS = (
    LAG_RETURN_SPECS
    + [("raw_ret_t", "bench_ret_t")]
    + [
        (f"raw_ret_t_t{i}", f"bench_ret_t_t{i}")
        for i in FORWARD_RETURN_HORIZONS
    ]
)
BENCHMARK_VARIABLE_COLUMNS = [
    "Stkcd",
    "td",
    *[ret_col for ret_col, _ in RETURN_WINDOW_SPECS],
    "float_mktcap_tm3",
    "float_mktcap_tm3_valid",
]


# =============================================================================
# 工具函数
# =============================================================================

def normalize_stock_code(x):
    """
    标准化单个股票代码。
    输入：数字、字符串或缺失值；输出：6 位股票代码字符串，无法识别时返回 pd.NA。
    """
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()
    if s == "":
        return pd.NA

    m = re.search(r"\d{1,6}", s)
    if m is None:
        return pd.NA

    return m.group(0).zfill(6)


def require_columns(df, cols, name):
    """
    检查数据表是否包含必要字段。
    输入：待检查 DataFrame、必要字段列表、表名；输出：无，缺字段时抛出 ValueError。
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要变量：{missing}")


def add_holding_formation_year(df):
    """
    生成持有期交易日对应的 DGTW 形成年份。
    输入：含 Trddt 的日度面板；输出：新增 formation_year 的副本。
    规则：1-6 月取上一年，7-12 月取当年。
    """
    out = df.copy()
    out["formation_year"] = np.where(
        out["Trddt"].dt.month <= 6,
        out["Trddt"].dt.year - 1,
        out["Trddt"].dt.year,
    ).astype("int16")
    return out


def assert_unique_key(df, key_cols, name):
    """
    检查指定字段组合是否唯一。
    输入：DataFrame、唯一键字段、表名；输出：无，发现重复时抛出 ValueError。
    """
    dup_n = df.duplicated(key_cols).sum()
    if dup_n:
        raise ValueError(f"{name} 的 {key_cols} 存在重复：{dup_n:,} 条")


def normalize_bool_flag(series, name):
    """
    将常见布尔标记统一为 bool。
    输入：Series 和字段名；输出：等长 bool Series，缺失值按 False 处理。
    识别 True/False、1/0、yes/no、y/n 等；无法识别时抛出 ValueError。
    """
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


def prepare_benchmark_panel(sample_index, variable_panel):
    """
    准备 benchmark 计算用的日度股票面板。
    输入：基础样本索引 sample_index、个股日度变量 variable_panel。
    输出：按 Stkcd × td 合并后的面板，包含收益窗口、t-3 流通市值和 weight_tm3。
    """
    index_cols = ["Stkcd", "td", "Trddt"]
    require_columns(sample_index, index_cols, "基础清洗后个股日度样本索引")
    require_columns(variable_panel, BENCHMARK_VARIABLE_COLUMNS, "DGTW构建用个股日度面板")

    index = sample_index[index_cols].copy()
    index["Stkcd"] = index["Stkcd"].apply(normalize_stock_code)
    index["Trddt"] = pd.to_datetime(index["Trddt"])
    assert_unique_key(index, ["Stkcd", "td"], "基础清洗后个股日度样本索引")

    variables = variable_panel[BENCHMARK_VARIABLE_COLUMNS].copy()
    variables["Stkcd"] = variables["Stkcd"].apply(normalize_stock_code)
    assert_unique_key(variables, ["Stkcd", "td"], "DGTW构建用个股日度面板")

    out = index.merge(
        variables,
        how="left",
        on=["Stkcd", "td"],
        validate="1:1",
    )
    out["float_mktcap_tm3_valid"] = normalize_bool_flag(
        out["float_mktcap_tm3_valid"],
        "float_mktcap_tm3_valid",
    )
    out["weight_tm3"] = out["float_mktcap_tm3"]
    return out


def attach_portfolio(daily, portfolio_assign):
    """
    合并形成期 DGTW 组合标签到日度面板。
    输入：日度面板 daily、形成期标签 portfolio_assign。
    输出：新增 portfolioNumber 等字段的面板；持有期区间外的组合标签置空。
    """
    assign_cols = [
        "Stkcd",
        "formation_year",
        "dateBeg",
        "dateEnd",
        "portfolioNumber",
    ]
    require_columns(portfolio_assign, assign_cols, "portfolio_assign")

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
    return out


def benchmark_suffix(out_col):
    """
    提取 benchmark 收益字段的窗口后缀。
    输入：如 bench_ret_t_t10 的字段名；输出：如 t_t10 的后缀。
    """
    if out_col.startswith("bench_ret_"):
        return out_col.replace("bench_ret_", "", 1)
    return out_col


def make_value_weighted_benchmark(df, ret_col, ok_col, out_col):
    """
    计算 Trddt × portfolioNumber 的市值加权 benchmark 收益。
    输入：日度面板、收益列 ret_col、额外有效性标记 ok_col、输出列名 out_col。
    输出：组合收益、有效股票数、权重合计。
    口径：仅使用 ok_col 为 True、组合编号非缺失、权重大于 0、收益非缺失的记录。
    """
    suffix = benchmark_suffix(out_col)
    sum_weight_col = f"sum_weight_{suffix}"
    n_stocks_col = f"n_stocks_{suffix}"

    keep_cols = ["Trddt", "portfolioNumber", "Stkcd", "weight_tm3", ret_col]
    tmp = df.loc[
        df[ok_col]
        & df["portfolioNumber"].notna()
        & df["weight_tm3"].gt(0)
        & df[ret_col].notna(),
        keep_cols,
    ].copy()

    if tmp.empty:
        return pd.DataFrame(
            columns=["Trddt", "portfolioNumber", out_col, n_stocks_col, sum_weight_col]
        )

    tmp["weighted_ret"] = tmp["weight_tm3"] * tmp[ret_col]

    out = (
        tmp
        .groupby(["Trddt", "portfolioNumber"], as_index=False)
        .agg(
            **{
                sum_weight_col: ("weight_tm3", "sum"),
                "sum_weighted_ret": ("weighted_ret", "sum"),
                n_stocks_col: ("Stkcd", "nunique"),
            }
        )
    )

    out[out_col] = out["sum_weighted_ret"] / out[sum_weight_col]
    out = out.drop(columns=["sum_weighted_ret"])

    return out[["Trddt", "portfolioNumber", out_col, n_stocks_col, sum_weight_col]]


# =============================================================================
# 主程序
# =============================================================================

def main():
    """
    主程序：读取样本索引、日度变量和 DGTW 组合标签。
    输出：各收益窗口的组合基准收益，保存到 benchmark_out_path。
    """
    print("读取基础样本...", flush=True)
    sample_index = pd.read_csv(
        base_sample_index_path,
        encoding="utf-8-sig",
        parse_dates=["Trddt"],
    )
    print("读取 16G 日度收益面板，这一步会比较久...", flush=True)
    variable_panel = pd.read_csv(
        daily_variable_panel_path,
        encoding="utf-8",
        usecols=BENCHMARK_VARIABLE_COLUMNS
    )

    print("读取组合标签...", flush=True)
    portfolio_assign = pd.read_csv(
        portfolio_assign_path,
        encoding="utf-8-sig",
        parse_dates=["dateBeg", "dateEnd"],
    )
    portfolio_assign["Stkcd"] = portfolio_assign["Stkcd"].apply(normalize_stock_code)
    print("日度收益面板读取完成，开始合并...", flush=True)

    daily = prepare_benchmark_panel(sample_index, variable_panel)
    daily = attach_portfolio(daily, portfolio_assign)

    benchmark_frames = []
    for ret_col, out_col in RETURN_WINDOW_SPECS:
        ok_col = "_benchmark_mktcap_valid"
        daily[ok_col] = daily["float_mktcap_tm3_valid"]
        benchmark_frames.append(
            make_value_weighted_benchmark(
                daily,
                ret_col=ret_col,
                ok_col=ok_col,
                out_col=out_col,
            )
        )
        daily.drop(columns=[ok_col], inplace=True)

    benchmark_returns = benchmark_frames[0]
    for frame in benchmark_frames[1:]:
        benchmark_returns = benchmark_returns.merge(
            frame,
            on=["Trddt", "portfolioNumber"],
            how="outer",
        )

    benchmark_returns = (
        benchmark_returns
        .sort_values(["Trddt", "portfolioNumber"])
        .reset_index(drop=True)
    )

    benchmark_out_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_returns.to_csv(
        benchmark_out_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("DGTW benchmark 收益计算完成")
    for (ret_col, out_col), frame in zip(RETURN_WINDOW_SPECS, benchmark_frames):
        if out_col in {
            "bench_ret_tm3",
            "bench_ret_tm2",
            "bench_ret_tm1",
            "bench_ret_t",
            "bench_ret_t_t1",
            "bench_ret_t_t10",
            "bench_ret_t_t50",
        }:
            print(f"{out_col} 行数：{len(frame):,}")
    print(f"benchmark_returns 行数：{len(benchmark_returns):,}")
    print(f"已保存：{benchmark_out_path}")


if __name__ == "__main__":
    main()
