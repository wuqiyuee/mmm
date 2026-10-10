from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from 日度因子工具 import make_ff_2x3_factors, weighted_mean_by_group  # noqa: E402


DAILY_DEFAULT = ROOT / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv"
ANNUAL_LABEL_DEFAULT = ROOT / "中间结果/02_变量构造结果/DGTW/DGTW形成期组合标签.csv"
CALENDAR_DEFAULT = ROOT / "中间结果/01_清洗后样本/市场交易日历_td.csv"
OUT_DEFAULT = ROOT / "中间结果/06_交易策略/自算日度因子.csv"


def normalize_stock_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.extract(r"(\d{1,6})", expand=False).str.zfill(6)


def parse_market_types(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def assign_three_groups(
    df: pd.DataFrame,
    signal_col: str,
    low_bp_col: str,
    high_bp_col: str,
    out_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.select(
        [
            out[signal_col] <= out[low_bp_col],
            (out[signal_col] > out[low_bp_col]) & (out[signal_col] <= out[high_bp_col]),
            out[signal_col] > out[high_bp_col],
        ],
        ["L", "M", "H"],
        default=pd.NA,
    )
    return out


def load_daily_panel(path: Path, weight_col: str, start_year: int | None, end_year: int | None, nrows: int | None) -> pd.DataFrame:
    usecols = ["Stkcd", "td", "Trddt", "Markettype", "ret", weight_col]
    print(f"读取日度清洗面板：{path}")
    daily = pd.read_csv(
        path,
        usecols=usecols,
        dtype={"Stkcd": "string", "td": "Int32", "Markettype": "float32"},
        parse_dates=["Trddt"],
        nrows=nrows,
        low_memory=False,
    )
    daily["Stkcd"] = normalize_stock_code(daily["Stkcd"])
    daily["ret"] = pd.to_numeric(daily["ret"], errors="coerce").astype("float64")
    daily[weight_col] = pd.to_numeric(daily[weight_col], errors="coerce").astype("float64")
    daily["year"] = daily["Trddt"].dt.year
    if start_year is not None:
        daily = daily.loc[daily["year"] >= start_year].copy()
    if end_year is not None:
        daily = daily.loc[daily["year"] <= end_year].copy()
    daily = daily.sort_values(["Stkcd", "td"]).reset_index(drop=True)
    print(f"日度面板行数：{len(daily):,}")
    return daily


def load_calendar_month_ends(path: Path, start_year: int | None, end_year: int | None) -> pd.Series:
    cal = pd.read_csv(path, usecols=["Trddt"], parse_dates=["Trddt"])
    if start_year is not None:
        cal = cal.loc[cal["Trddt"].dt.year >= start_year].copy()
    if end_year is not None:
        cal = cal.loc[cal["Trddt"].dt.year <= end_year].copy()
    cal["month"] = cal["Trddt"].dt.to_period("M").astype(str)
    return cal.groupby("month")["Trddt"].max().sort_values()


def build_market_factor(daily: pd.DataFrame, weight_col: str, rf_file: Path | None, rf_date_col: str, rf_col: str, rf_in_percent: bool) -> pd.DataFrame:
    valid = daily.loc[daily["ret"].notna() & daily[weight_col].gt(0)].copy()
    mkt = weighted_mean_by_group(
        valid,
        group_cols=["Trddt"],
        return_col="ret",
        weight_col=weight_col,
        value_name="mkt",
    )
    mkt = mkt.rename(columns={"n_stocks": "n_mkt", "sum_weight": "sum_weight_mkt"})

    if rf_file is None:
        mkt["rf"] = 0.0
        mkt["rf_assumed_zero"] = 1
    else:
        rf = pd.read_csv(rf_file, usecols=[rf_date_col, rf_col], parse_dates=[rf_date_col])
        rf = rf.rename(columns={rf_date_col: "Trddt", rf_col: "rf"})
        rf["rf"] = pd.to_numeric(rf["rf"], errors="coerce")
        if rf_in_percent:
            rf["rf"] = rf["rf"] / 100
        mkt = mkt.merge(rf[["Trddt", "rf"]], on="Trddt", how="left")
        mkt["rf_assumed_zero"] = mkt["rf"].isna().astype("int8")
        mkt["rf"] = mkt["rf"].fillna(0)

    mkt["mktrf"] = mkt["mkt"] - mkt["rf"]
    return mkt


def build_annual_size_value_factors(
    daily: pd.DataFrame,
    labels_path: Path,
    breakpoint_markettypes: tuple[int, ...],
    weight_col: str,
) -> pd.DataFrame:
    print(f"读取年度 DGTW 标签：{labels_path}")
    labels = pd.read_csv(
        labels_path,
        usecols=["Stkcd", "formation_year", "Markettype", "size", "bm"],
        dtype={"Stkcd": "string"},
    )
    labels["Stkcd"] = normalize_stock_code(labels["Stkcd"])
    labels["formation_year"] = pd.to_numeric(labels["formation_year"], errors="coerce").astype("Int64")
    labels["Markettype"] = pd.to_numeric(labels["Markettype"], errors="coerce")
    labels["size"] = pd.to_numeric(labels["size"], errors="coerce")
    labels["bm"] = pd.to_numeric(labels["bm"], errors="coerce")
    labels = labels.dropna(subset=["Stkcd", "formation_year", "size", "bm"]).copy()

    break_univ = labels.loc[labels["Markettype"].isin(breakpoint_markettypes)].copy()
    bp = (
        break_univ.groupby("formation_year")
        .agg(size_median=("size", "median"), bm_p30=("bm", lambda x: x.quantile(0.3)), bm_p70=("bm", lambda x: x.quantile(0.7)))
        .reset_index()
    )
    labels = labels.merge(bp, on="formation_year", how="left", validate="m:1")
    labels["size_port"] = np.where(labels["size"] <= labels["size_median"], "S", "B")
    labels = assign_three_groups(labels, "bm", "bm_p30", "bm_p70", "bm_port")
    labels = labels.dropna(subset=["size_port", "bm_port"]).copy()
    labels["formation_weight"] = labels["size"]

    returns = daily.loc[:, ["Stkcd", "Trddt", "year", "ret"]].copy()
    returns["month"] = returns["Trddt"].dt.month
    returns["formation_year"] = np.where(returns["month"] >= 7, returns["year"], returns["year"] - 1).astype("int64")
    returns = returns.drop(columns=["month", "year"])

    merged = returns.merge(
        labels[["Stkcd", "formation_year", "size_port", "bm_port", "formation_weight"]],
        on=["Stkcd", "formation_year"],
        how="inner",
    )
    legs = weighted_mean_by_group(
        merged,
        group_cols=["Trddt", "size_port", "bm_port"],
        return_col="ret",
        weight_col="formation_weight",
        value_name="leg_ret",
    )
    factors = make_ff_2x3_factors(
        legs,
        date_col="Trddt",
        size_col="size_port",
        char_col="bm_port",
        return_col="leg_ret",
        smb_col="smb",
        hml_col="hml",
    )
    return factors


def build_monthly_signal_assignments(
    daily: pd.DataFrame,
    weight_col: str,
    month_ends: pd.Series,
    breakpoint_markettypes: tuple[int, ...],
) -> pd.DataFrame:
    print("计算月末动量和短期反转排序变量")
    signal = daily.loc[:, ["Stkcd", "td", "Trddt", "Markettype", "ret", weight_col]].copy()
    signal = signal.sort_values(["Stkcd", "td"]).reset_index(drop=True)
    signal["log_1p_ret"] = np.log1p(signal["ret"])
    grouped = signal.groupby("Stkcd", sort=False)["log_1p_ret"]
    roll20 = grouped.rolling(window=20, min_periods=20).sum().reset_index(level=0, drop=True)
    roll252 = grouped.rolling(window=252, min_periods=252).sum().reset_index(level=0, drop=True)
    signal["mom_signal"] = np.expm1(roll252 - roll20)
    signal["st_rev_signal"] = np.expm1(roll20)

    month_end_dates = set(month_ends.dt.strftime("%Y-%m-%d"))
    signal["date_key"] = signal["Trddt"].dt.strftime("%Y-%m-%d")
    formation = signal.loc[signal["date_key"].isin(month_end_dates)].copy()
    formation = formation.drop(columns=["date_key", "log_1p_ret"])
    formation["formation_month"] = formation["Trddt"].dt.to_period("M").astype(str)
    formation["hold_month"] = (formation["Trddt"] + MonthBegin(1)).dt.to_period("M").astype(str)
    formation = formation.rename(columns={weight_col: "formation_weight"})

    break_univ = formation.loc[formation["Markettype"].isin(breakpoint_markettypes)].copy()
    bp = (
        break_univ.groupby("formation_month")
        .agg(
            mom_p30=("mom_signal", lambda x: x.quantile(0.3)),
            mom_p70=("mom_signal", lambda x: x.quantile(0.7)),
            st_rev_p30=("st_rev_signal", lambda x: x.quantile(0.3)),
            st_rev_p70=("st_rev_signal", lambda x: x.quantile(0.7)),
        )
        .reset_index()
    )
    formation = formation.merge(bp, on="formation_month", how="left", validate="m:1")
    formation = assign_three_groups(formation, "mom_signal", "mom_p30", "mom_p70", "mom_port")
    formation = assign_three_groups(formation, "st_rev_signal", "st_rev_p30", "st_rev_p70", "st_rev_port")
    formation = formation.loc[formation["formation_weight"].notna() & formation["formation_weight"].gt(0)].copy()
    return formation[["Stkcd", "formation_month", "hold_month", "formation_weight", "mom_port", "st_rev_port"]]


def build_monthly_signal_factors(daily: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    returns = daily.loc[:, ["Stkcd", "Trddt", "ret"]].copy()
    returns["hold_month"] = returns["Trddt"].dt.to_period("M").astype(str)
    merged = returns.merge(assignments, on=["Stkcd", "hold_month"], how="inner")

    mom_legs = weighted_mean_by_group(
        merged.dropna(subset=["mom_port"]),
        group_cols=["Trddt", "mom_port"],
        return_col="ret",
        weight_col="formation_weight",
        value_name="mom_leg_ret",
    )
    mom_wide = mom_legs.pivot_table(index="Trddt", columns="mom_port", values="mom_leg_ret", aggfunc="first").reset_index()
    for col in ["H", "L"]:
        if col not in mom_wide.columns:
            mom_wide[col] = pd.NA
    mom_wide["mom"] = mom_wide["H"] - mom_wide["L"]

    st_legs = weighted_mean_by_group(
        merged.dropna(subset=["st_rev_port"]),
        group_cols=["Trddt", "st_rev_port"],
        return_col="ret",
        weight_col="formation_weight",
        value_name="st_rev_leg_ret",
    )
    st_wide = st_legs.pivot_table(index="Trddt", columns="st_rev_port", values="st_rev_leg_ret", aggfunc="first").reset_index()
    for col in ["H", "L"]:
        if col not in st_wide.columns:
            st_wide[col] = pd.NA
    st_wide["st_rev"] = st_wide["L"] - st_wide["H"]

    return mom_wide[["Trddt", "mom"]].merge(st_wide[["Trddt", "st_rev"]], on="Trddt", how="outer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于本文清洗后 A 股样本自算日度五因子。")
    parser.add_argument("--daily-path", type=Path, default=DAILY_DEFAULT)
    parser.add_argument("--annual-label-path", type=Path, default=ANNUAL_LABEL_DEFAULT)
    parser.add_argument("--calendar-path", type=Path, default=CALENDAR_DEFAULT)
    parser.add_argument("--out-path", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--weight-col", choices=["size", "sizew"], default="size")
    parser.add_argument("--breakpoint-markettypes", default="1,4")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--nrows", type=int, default=None, help="只读取前 n 行，用于快速试跑。")
    parser.add_argument("--rf-file", type=Path, default=None)
    parser.add_argument("--rf-date-col", default="Trddt")
    parser.add_argument("--rf-col", default="rf")
    parser.add_argument("--rf-in-percent", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    breakpoint_markettypes = parse_market_types(args.breakpoint_markettypes)

    daily = load_daily_panel(args.daily_path, args.weight_col, args.start_year, args.end_year, args.nrows)
    month_ends = load_calendar_month_ends(args.calendar_path, args.start_year, args.end_year)

    mkt = build_market_factor(daily, args.weight_col, args.rf_file, args.rf_date_col, args.rf_col, args.rf_in_percent)
    size_value = build_annual_size_value_factors(daily, args.annual_label_path, breakpoint_markettypes, args.weight_col)
    assignments = build_monthly_signal_assignments(daily, args.weight_col, month_ends, breakpoint_markettypes)
    momentum_reversal = build_monthly_signal_factors(daily, assignments)

    factors = (
        mkt.merge(size_value, on="Trddt", how="outer")
        .merge(momentum_reversal, on="Trddt", how="outer")
        .sort_values("Trddt")
        .reset_index(drop=True)
    )
    factors["year"] = factors["Trddt"].dt.year
    keep = [
        "Trddt",
        "year",
        "mktrf",
        "mkt",
        "rf",
        "rf_assumed_zero",
        "smb",
        "hml",
        "mom",
        "st_rev",
        "n_mkt",
        "sum_weight_mkt",
    ]
    factors = factors.loc[:, [c for c in keep if c in factors.columns]]

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(args.out_path, index=False, encoding="utf-8-sig")

    print(f"输出文件：{args.out_path}")
    print(f"输出行数：{len(factors):,}")
    if len(factors) > 0:
        print(f"日期范围：{factors['Trddt'].min().date()} 至 {factors['Trddt'].max().date()}")
    print("核心变量缺失数：")
    print(factors[["mktrf", "smb", "hml", "mom", "st_rev"]].isna().sum())


if __name__ == "__main__":
    main()
