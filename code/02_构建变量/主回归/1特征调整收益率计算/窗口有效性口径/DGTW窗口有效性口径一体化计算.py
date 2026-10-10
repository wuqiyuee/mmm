import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/qiyuewu/Desktop/mmm复现数据包")

base_sample_index_path = ROOT / "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv"
daily_variable_panel_path = ROOT / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv"
portfolio_assign_path = ROOT / "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW形成期组合标签.csv"
output_root = ROOT / "中间结果/02_变量构造结果/1DGTW特征调整收益率/窗口有效性口径"

LAG_RETURN_OFFSETS = (3, 2, 1)
FORWARD_RETURN_HORIZONS = tuple(range(1, 51))
DIAGNOSTIC_HORIZONS = {1, 10, 50}
POLICY_THRESHOLDS = {
    "strict": 1.0,
    "pct70": 0.70,
    "pct80": 0.80,
}

LAG_RETURN_SPECS = tuple(
    (f"raw_ret_tm{i}", f"bench_ret_tm{i}", f"chAdj_ret_tm{i}", None)
    for i in LAG_RETURN_OFFSETS
)
CURRENT_RETURN_SPEC = (("raw_ret_t", "bench_ret_t", "chAdj_ret_t", None),)
FORWARD_RETURN_SPECS = tuple(
    (f"raw_ret_t_t{i}", f"bench_ret_t_t{i}", f"chAdj_ret_t_t{i}", i)
    for i in FORWARD_RETURN_HORIZONS
)
RETURN_WINDOW_SPECS = LAG_RETURN_SPECS + CURRENT_RETURN_SPEC + FORWARD_RETURN_SPECS
RAW_RETURN_COLUMNS = [raw_col for raw_col, _, _, _ in RETURN_WINDOW_SPECS]
ADJUSTED_RETURN_COLUMNS = [out_col for _, _, out_col, _ in RETURN_WINDOW_SPECS]


def normalize_stock_code_series(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def require_columns(df, columns, name):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要变量：{missing}")


def assert_unique_key(df, key_cols, name):
    dup_count = df.duplicated(key_cols).sum()
    if dup_count:
        raise ValueError(f"{name} 的 {key_cols} 存在重复：{dup_count:,} 条")


def normalize_bool_flag(series):
    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n", ""}

    def convert_one(value):
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        lowered = str(value).strip().lower()
        if lowered in true_values:
            return True
        if lowered in false_values:
            return False
        raise ValueError(f"无法识别的布尔取值：{value!r}")

    return series.map(convert_one).astype(bool)


def add_holding_formation_year(df):
    out = df.copy()
    trddt = pd.to_datetime(out["Trddt"], errors="coerce")
    out["formation_year"] = np.where(
        trddt.dt.month <= 6,
        trddt.dt.year - 1,
        trddt.dt.year,
    ).astype("int16")
    return out


def clean_count_threshold(window_len, policy):
    ratio = POLICY_THRESHOLDS[policy]
    return window_len if ratio == 1.0 else math.ceil(window_len * ratio)


def add_forward_window_validity(df, horizons=FORWARD_RETURN_HORIZONS, policies=("strict", "pct70")):
    """
    在完整 Stkcd-td 面板上计算 forward window 的 clean-day 有效性。

    valid_clean_day 表示该 Stkcd-td 出现在基础清洗样本中，已经排除了
    Filling!=0、金融行业、上市未满一年、ST/PT 等异常状态。
    对 [t,t+k]，先要求同一股票存在完整 td...td+k，再按口径判断 clean day 数量。
    """
    out = (
        df
        .sort_values(["Stkcd", "td"])
        .reset_index(drop=True)
        .copy()
    )
    out["valid_clean_day"] = normalize_bool_flag(out["valid_clean_day"])

    grouped = out.groupby("Stkcd", sort=False)
    clean_nobs = pd.Series(0, index=out.index, dtype="Int16")
    existing_nobs = pd.Series(0, index=out.index, dtype="Int16")

    for k in range(0, max(horizons) + 1):
        shifted_td = grouped["td"].shift(-k)
        shifted_clean = grouped["valid_clean_day"].shift(-k)
        shifted_clean = shifted_clean.where(shifted_clean.notna(), False).astype(bool)
        exists_k = shifted_td.eq(out["td"] + k)
        clean_k = exists_k & shifted_clean

        existing_nobs = (existing_nobs + exists_k.astype("int8")).astype("Int16")
        clean_nobs = (clean_nobs + clean_k.astype("int8")).astype("Int16")

        if k not in horizons:
            continue

        window_len = k + 1
        exists_full = existing_nobs.eq(window_len)
        if k in DIAGNOSTIC_HORIZONS:
            out[f"window_t_t{k}_clean_nobs"] = clean_nobs
            out[f"window_t_t{k}_exists_full"] = exists_full

        for policy in policies:
            threshold = clean_count_threshold(window_len, policy)
            out[f"window_t_t{k}_valid_{policy}"] = exists_full & clean_nobs.ge(threshold)

    return out


def load_base_clean_sample():
    sample = pd.read_csv(
        base_sample_index_path,
        encoding="utf-8-sig",
        usecols=["Stkcd", "Trddt", "td"],
        dtype={"Stkcd": "string"},
        parse_dates=["Trddt"],
    )
    sample["Stkcd"] = normalize_stock_code_series(sample["Stkcd"])
    sample["td"] = pd.to_numeric(sample["td"], errors="coerce").astype("int64")
    sample = sample.dropna(subset=["Stkcd", "Trddt", "td"])
    assert_unique_key(sample, ["Stkcd", "td"], "基础清洗样本")
    return sample


def load_daily_panel():
    usecols = [
        "Stkcd",
        "td",
        "Trddt",
        *RAW_RETURN_COLUMNS,
        "float_mktcap_tm3",
        "float_mktcap_tm3_valid",
    ]
    daily = pd.read_csv(
        daily_variable_panel_path,
        encoding="utf-8",
        usecols=usecols,
        dtype={"Stkcd": "string"},
        parse_dates=["Trddt"],
    )
    require_columns(daily, usecols, "DGTW构建用个股日度面板")
    daily["Stkcd"] = normalize_stock_code_series(daily["Stkcd"])
    daily["td"] = pd.to_numeric(daily["td"], errors="coerce").astype("int64")
    daily["float_mktcap_tm3_valid"] = normalize_bool_flag(daily["float_mktcap_tm3_valid"])
    assert_unique_key(daily, ["Stkcd", "td"], "DGTW构建用个股日度面板")
    return daily


def attach_clean_day_flag(daily, base_sample):
    valid_keys = base_sample[["Stkcd", "td"]].copy()
    valid_keys["valid_clean_day"] = True

    out = daily.merge(
        valid_keys,
        on=["Stkcd", "td"],
        how="left",
        validate="1:1",
    )
    out["valid_clean_day"] = out["valid_clean_day"].fillna(False)
    return out


def attach_portfolio(daily):
    portfolio = pd.read_csv(
        portfolio_assign_path,
        encoding="utf-8-sig",
        usecols=["Stkcd", "formation_year", "dateBeg", "dateEnd", "portfolioNumber"],
        dtype={"Stkcd": "string"},
        parse_dates=["dateBeg", "dateEnd"],
    )
    portfolio["Stkcd"] = normalize_stock_code_series(portfolio["Stkcd"])
    portfolio["formation_year"] = pd.to_numeric(
        portfolio["formation_year"],
        errors="coerce",
    ).astype("int16")
    portfolio["portfolioNumber"] = pd.to_numeric(
        portfolio["portfolioNumber"],
        errors="coerce",
    ).astype("Int16")
    assert_unique_key(portfolio, ["Stkcd", "formation_year"], "DGTW形成期组合标签")

    out = add_holding_formation_year(daily)
    out = out.merge(
        portfolio,
        on=["Stkcd", "formation_year"],
        how="left",
        validate="m:1",
    )
    valid_date = (
        out["portfolioNumber"].notna()
        & out["dateBeg"].notna()
        & out["dateEnd"].notna()
        & out["Trddt"].between(out["dateBeg"], out["dateEnd"])
    )
    out = out.loc[valid_date].copy()
    out["portfolioNumber"] = out["portfolioNumber"].astype("int16")
    out = out.drop(columns=["dateBeg", "dateEnd"])
    return out


def make_value_weighted_benchmark(df, raw_col, bench_col, horizon, policy):
    ok = (
        df["float_mktcap_tm3_valid"]
        & df["float_mktcap_tm3"].gt(0)
        & df["portfolioNumber"].notna()
        & df[raw_col].notna()
    )
    if horizon is not None:
        ok = ok & df[f"window_t_t{horizon}_valid_{policy}"]

    tmp = df.loc[
        ok,
        ["Trddt", "portfolioNumber", "Stkcd", "float_mktcap_tm3", raw_col],
    ].copy()

    suffix = bench_col.replace("bench_ret_", "", 1)
    n_stocks_col = f"n_stocks_{suffix}"
    sum_weight_col = f"sum_weight_{suffix}"

    if tmp.empty:
        return pd.DataFrame(
            columns=["Trddt", "portfolioNumber", bench_col, n_stocks_col, sum_weight_col]
        )

    tmp["weighted_ret"] = tmp["float_mktcap_tm3"] * tmp[raw_col]
    out = (
        tmp
        .groupby(["Trddt", "portfolioNumber"], as_index=False)
        .agg(
            **{
                sum_weight_col: ("float_mktcap_tm3", "sum"),
                "sum_weighted_ret": ("weighted_ret", "sum"),
                n_stocks_col: ("Stkcd", "nunique"),
            }
        )
    )
    out[bench_col] = out["sum_weighted_ret"] / out[sum_weight_col]
    return out[["Trddt", "portfolioNumber", bench_col, n_stocks_col, sum_weight_col]]


def build_benchmark(df, policy):
    benchmark_frames = []
    for raw_col, bench_col, _, horizon in RETURN_WINDOW_SPECS:
        benchmark_frames.append(
            make_value_weighted_benchmark(
                df=df,
                raw_col=raw_col,
                bench_col=bench_col,
                horizon=horizon,
                policy=policy,
            )
        )

    benchmark = benchmark_frames[0]
    for frame in benchmark_frames[1:]:
        benchmark = benchmark.merge(
            frame,
            on=["Trddt", "portfolioNumber"],
            how="outer",
        )

    return (
        benchmark
        .sort_values(["Trddt", "portfolioNumber"])
        .reset_index(drop=True)
    )


def calculate_adjusted_returns_for_policy(df, policy, horizons=FORWARD_RETURN_HORIZONS):
    out = df.copy()
    for raw_col, bench_col, adjusted_col, horizon in RETURN_WINDOW_SPECS:
        if raw_col not in out.columns or bench_col not in out.columns:
            continue
        if horizon is not None and horizon not in horizons:
            continue

        out[adjusted_col] = out[raw_col] - out[bench_col]
        if horizon is not None:
            out.loc[~out[f"window_t_t{horizon}_valid_{policy}"], adjusted_col] = np.nan
    return out


def build_adjusted_returns(sample_panel, benchmark, policy):
    merged = sample_panel.merge(
        benchmark,
        on=["Trddt", "portfolioNumber"],
        how="left",
        validate="m:1",
    )
    merged = calculate_adjusted_returns_for_policy(merged, policy)
    keep_cols = [
        "Stkcd",
        "Trddt",
        "td",
        "formation_year",
        "portfolioNumber",
        *ADJUSTED_RETURN_COLUMNS,
    ]
    return merged[keep_cols].copy()


def write_policy_outputs(sample_panel, policy):
    policy_dir = output_root / policy
    policy_dir.mkdir(parents=True, exist_ok=True)

    flag_cols = [
        f"window_t_t{k}_valid_{policy}"
        for k in FORWARD_RETURN_HORIZONS
    ]
    diagnostic_cols = []
    for k in sorted(DIAGNOSTIC_HORIZONS):
        diagnostic_cols.extend([
            f"window_t_t{k}_clean_nobs",
            f"window_t_t{k}_exists_full",
        ])

    label_cols = [
        "Stkcd",
        "Trddt",
        "td",
        "formation_year",
        "portfolioNumber",
        "valid_clean_day",
        *diagnostic_cols,
        *flag_cols,
    ]
    label_path = policy_dir / f"DGTW窗口有效性标签_{policy}.csv"
    benchmark_path = policy_dir / f"DGTW组合基准收益_{policy}.csv"
    adjusted_path = policy_dir / f"DGTW特征调整收益率_{policy}.csv"

    sample_panel[label_cols].to_csv(label_path, index=False, encoding="utf-8-sig")
    print(f"[{policy}] 标签已保存：{label_path}", flush=True)

    benchmark = build_benchmark(sample_panel, policy)
    benchmark.to_csv(benchmark_path, index=False, encoding="utf-8-sig")
    print(f"[{policy}] benchmark 已保存：{benchmark_path}", flush=True)

    adjusted = build_adjusted_returns(sample_panel, benchmark, policy)
    adjusted.to_csv(adjusted_path, index=False, encoding="utf-8-sig")
    print(f"[{policy}] 特征调整收益率已保存：{adjusted_path}", flush=True)

    return {
        "policy": policy,
        "label_rows": len(sample_panel),
        "benchmark_rows": len(benchmark),
        "adjusted_rows": len(adjusted),
        "chAdj_ret_t_t1_missing": int(adjusted["chAdj_ret_t_t1"].isna().sum()),
        "chAdj_ret_t_t10_missing": int(adjusted["chAdj_ret_t_t10"].isna().sum()),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成 DGTW 窗口有效性口径下的标签、benchmark 和特征调整收益率。",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["strict", "pct70"],
        choices=sorted(POLICY_THRESHOLDS),
        help="要输出的窗口有效性口径。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root.mkdir(parents=True, exist_ok=True)

    print("读取基础清洗样本...", flush=True)
    base_sample = load_base_clean_sample()
    print(f"基础清洗样本行数：{len(base_sample):,}", flush=True)

    print("读取 DGTW 构建用日度面板...", flush=True)
    daily = load_daily_panel()
    print(f"DGTW 日度面板行数：{len(daily):,}", flush=True)

    print("标记 valid_clean_day...", flush=True)
    daily = attach_clean_day_flag(daily, base_sample)

    print("计算 forward window clean 有效性...", flush=True)
    daily = add_forward_window_validity(daily, policies=tuple(args.policies))

    print("合并 DGTW 形成期组合标签...", flush=True)
    daily = attach_portfolio(daily)
    sample_panel = daily.loc[daily["valid_clean_day"]].copy()
    assert_unique_key(sample_panel, ["Stkcd", "td"], "窗口有效性样本面板")
    print(f"带 DGTW 组合的基础清洗样本行数：{len(sample_panel):,}", flush=True)

    summaries = []
    for policy in args.policies:
        print(f"开始输出口径：{policy}", flush=True)
        summaries.append(write_policy_outputs(sample_panel, policy))

    print("全部口径输出完成。", flush=True)
    for summary in summaries:
        policy = summary["policy"]
        print(
            f"[{policy}] labels={summary['label_rows']:,}, "
            f"benchmark={summary['benchmark_rows']:,}, "
            f"adjusted={summary['adjusted_rows']:,}, "
            f"chAdj_t_t1_missing={summary['chAdj_ret_t_t1_missing']:,}, "
            f"chAdj_t_t10_missing={summary['chAdj_ret_t_t10_missing']:,}",
            flush=True,
        )


if __name__ == "__main__":
    main()
