# 筛选 A 股市场股票：清洗金融行业、上市未满一年、ST 状态观测值
# 用于计算 DGTW 分组标签。
#
# 清洗方式：
# 1. 将个股填充至完整交易日序列，计算过去 1 年交易天数和过去 1 个月交易天数。
# 2. 删除金融行业、上市未满一年、ST 状态观测值。
# 3. 再用 td 匹配回完整面板。
#
# 不合格的日度观测值除了 Stkcd、Trddt、td 以外，其他变量会变成 NaN。
# 后续使用 shift + rolling 计算动量指标时，只要窗口期内有一条缺失值，mom 指标就会缺失。
# 后续还会在 6 月份年度数据上筛选掉过去 1 年交易天数不足 120 天的观测。

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 当前脚本假定放在：mmm复现数据包/code/01_清洗数据/xxx.py
# 因此 parents[2] 指向项目根目录：mmm复现数据包
ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from mytools import normalize_stock_code  # noqa: E402


# =============================================================================
# 参数设置
# =============================================================================

@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    daily_path: Path = ROOT / "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_2000_2025.csv"
    industry_path: Path = ROOT / "input/CSMAR/上市公司行业分类表/STK_INDUSTRYCLASS.csv"
    company_path: Path = ROOT / "input/CSMAR/公司文件/TRD_Co.csv"
    calendar_path: Path = ROOT / "中间结果/01_清洗后样本/TRD_Cale_td.csv"
    full_panel_path: Path = ROOT / "中间结果/01_清洗后样本/full_panel.csv"
    out_file: Path = ROOT / "中间结果/01_清洗后样本/TRD_Dalyr_DGTW.csv"


A_SHARE_MARKET_TYPES = (1, 4, 16, 32)
FILLING_VALUES = (0, 1)
NORMAL_TRADING_STATUS = (1, 4, 7, 10, 13)

DAILY_USECOLS = [
    "Stkcd", "Trddt", "Markettype", "Filling", "Trdsta",
    "Dnshrtrd", "Dretwd", "Dretnd", "Dsmvosd", "Dsmvtll",
]
NUMERIC_COLS = ["Dnshrtrd", "Dretwd", "Dretnd", "Dsmvosd", "Dsmvtll"]
RENAME_DAILY_COLS = {
    "Dretwd": "ret",
    "Dretnd": "retn",
}

INDUSTRY_USECOLS = ["ImplementDate", "IndustryClassificationID", "Symbol", "IndustryCode"]
COMPANY_USECOLS = ["Stkcd", "Listdt"]
OUTPUT_DROP_COLS = ["Trdsta", "Filling", "has_volume", "Listdt"]

INDUSTRY_STANDARD_DATES = (
    pd.Timestamp("2012-10-26"),
    pd.Timestamp("2023-05-01"),
)


# =============================================================================
# 通用设置与检查函数
# =============================================================================

def setup_runtime() -> None:
    warnings.filterwarnings("ignore")
    pd.options.mode.chained_assignment = None
    logging.getLogger("matplotlib.font_manager").disabled = True

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)


def ensure_output_dirs(cfg: Config) -> None:
    cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.full_panel_path.parent.mkdir(parents=True, exist_ok=True)


def print_section(title: str) -> None:
    print("" + "=" * 80)
    print(title)
    print("=" * 80)


def check_unique_stock_td(df: pd.DataFrame, name: str) -> None:
    dup_n = df.duplicated(["Stkcd", "td"]).sum()
    print(f"{name} 的 Stkcd-td 重复观测数：{dup_n}")

    if dup_n > 0:
        print(
            df.loc[
                df.duplicated(["Stkcd", "td"], keep=False),
                ["Stkcd", "Trddt", "td"],
            ].head(30)
        )


def drop_duplicate_stock_td(df: pd.DataFrame, name: str) -> pd.DataFrame:
    dup_n = df.duplicated(["Stkcd", "td"]).sum()
    print(f"{name} 的 Stkcd-td 重复观测数：{dup_n}")

    if dup_n == 0:
        return df

    print(
        df.loc[
            df.duplicated(["Stkcd", "td"], keep=False),
            ["Stkcd", "Trddt", "td"],
        ].head(30)
    )

    return (
        df
        .sort_values(["Stkcd", "td", "Trddt"])
        .drop_duplicates(["Stkcd", "td"], keep="last")
        .reset_index(drop=True)
    )


def report_td_continuity(df: pd.DataFrame, name: str) -> None:
    tmp = df.sort_values(["Stkcd", "td"]).copy()
    tmp["td_gap"] = tmp.groupby("Stkcd")["td"].diff()

    print(f"{name} 内部 td_gap 分布：")
    print(tmp["td_gap"].value_counts(dropna=False).head(20))

    gap = tmp.loc[tmp["td_gap"].notna() & (tmp["td_gap"] != 1)].copy()
    print(f"{name} 中 td 不连续的观测数：{len(gap)}")
    print(f"{name} 中 td 不连续的股票数：{gap['Stkcd'].nunique()}")

    if len(gap) > 0:
        print(gap[["Stkcd", "Trddt", "td", "td_gap"]].head(30))


# =============================================================================
# 读取与基础筛选
# =============================================================================

def load_daily_data(cfg: Config) -> pd.DataFrame:
    print_section("读取日度数据、保留 A 股市场数据")

    df = pd.read_csv(
        cfg.daily_path,
        encoding="utf-8",
        usecols=DAILY_USECOLS,
        parse_dates=["Trddt"],
    )

    df["Stkcd"] = df["Stkcd"].apply(normalize_stock_code)
    df["year"] = df["Trddt"].dt.year

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns=RENAME_DAILY_COLS)

    df["sizew"] = df["Dsmvosd"] * 1000
    df["size"] = df["Dsmvtll"] * 1000
    df = df.drop(columns=["Dsmvosd", "Dsmvtll"])

    before = len(df)
    df = df.loc[
        df["Markettype"].isin(A_SHARE_MARKET_TYPES)
        & df["Filling"].isin(FILLING_VALUES)
    ].copy()

    print("原始日度观测数：", before)
    print("筛选 A 股和 Filling 后观测数：", len(df))
    print("删除观测数：", before - len(df))

    return df


def load_trading_calendar(cfg: Config) -> pd.DataFrame:
    calendar = pd.read_csv(cfg.calendar_path, encoding="utf-8")
    calendar["Trddt"] = pd.to_datetime(calendar["Trddt"])
    return calendar


def attach_trading_day(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    print_section("匹配统一交易日 td")

    df_td = pd.merge(
        df,
        calendar[["Trddt", "td"]],
        on=["Trddt"],
        how="left",
    )

    print("没有匹配到 td 的观测数字：", df_td["td"].isna().sum())

    df_td = df_td.dropna(subset=["td"]).copy()
    df_td["td"] = df_td["td"].astype(int)

    df_td = (
        df_td
        .sort_values(["Stkcd", "td"])
        .reset_index(drop=True)
    )

    df_td = drop_duplicate_stock_td(df_td, "TRD_Dalyr_td")
    return df_td


# =============================================================================
# 构造完整交易日面板
# =============================================================================

def build_full_panel(df_td: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    print_section("构造完整 Stkcd-td 面板")

    td_range = (
        df_td
        .groupby("Stkcd", as_index=False)["td"]
        .agg(td_min="min", td_max="max")
    )

    lengths = (td_range["td_max"] - td_range["td_min"] + 1).astype(int).to_numpy()

    full_panel = pd.DataFrame({
        "Stkcd": np.repeat(td_range["Stkcd"].to_numpy(), lengths),
        "td": np.concatenate([
            np.arange(lo, hi + 1, dtype=np.int64)
            for lo, hi in zip(td_range["td_min"], td_range["td_max"])
        ]),
    })

    full_panel = pd.merge(
        full_panel,
        calendar[["td", "Trddt"]],
        on="td",
        how="left",
    )

    print("完整面板股票数：", full_panel["Stkcd"].nunique())
    print("完整面板观测数：", len(full_panel))

    return full_panel


def complete_daily_panel(
    df_td: pd.DataFrame,
    full_panel: pd.DataFrame,
    original_columns: list[str],
) -> pd.DataFrame:
    print_section("将个股填充至完整交易日序列")

    completed = pd.merge(
        full_panel,
        df_td.drop(columns=["Trddt"]),
        on=["Stkcd", "td"],
        how="left",
    )

    completed = completed[original_columns]

    print("补齐前样本数：", len(df_td))
    print("补齐后样本数：", len(completed))
    print("新增补齐观测数：", len(completed) - len(df_td))

    report_td_continuity(completed, "TRD_Dalyr1")
    return completed


# =============================================================================
# 计算交易天数指标
# =============================================================================

def rolling_trade_days(
    df: pd.DataFrame,
    window: str,
    min_periods: int,
) -> pd.Series:
    rolled = (
        df
        .groupby("Stkcd")
        .rolling(
            window=window,
            on="Trddt",
            min_periods=min_periods,
            closed="right",
        )["has_volume"]
        .sum()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )

    return pd.Series(rolled, index=df.index)


def add_trade_day_counts(df: pd.DataFrame) -> pd.DataFrame:
    print_section("在完整面板上计算交易天数指标")

    out = (
        df
        .sort_values(["Stkcd", "td"])
        .reset_index(drop=True)
    )

    out["has_volume"] = out["Dnshrtrd"].fillna(0).gt(0).astype("int8")
    out = out.drop(columns=["Dnshrtrd"])

    out = (
        out
        .sort_values(["Stkcd", "Trddt"])
        .reset_index(drop=True)
        .copy()
    )

    out["trade_days_1m"] = rolling_trade_days(out, window="30D", min_periods=15)
    out["trade_days_1y"] = rolling_trade_days(out, window="365D", min_periods=120)

    out["trade_days_1m"] = out["trade_days_1m"].astype("Int16")
    out["trade_days_1y"] = out["trade_days_1y"].astype("Int16")

    print("trade_days_1m 非缺失观测数：", out["trade_days_1m"].notna().sum())
    print("trade_days_1y 非缺失观测数：", out["trade_days_1y"].notna().sum())

    return out


# =============================================================================
# 行业匹配与金融行业剔除
# =============================================================================

def load_industry_data(cfg: Config) -> pd.DataFrame:
    industry = pd.read_csv(
        cfg.industry_path,
        encoding="utf-8",
        usecols=INDUSTRY_USECOLS,
        parse_dates=["ImplementDate"],
    )

    industry = industry.dropna(subset=["ImplementDate", "Symbol", "IndustryCode"]).copy()

    industry["IndustryCode"] = (
        industry["IndustryCode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    industry = industry.loc[
        industry["IndustryClassificationID"].isin(["P0201", "P0207", "P0221"])
    ].copy()

    industry["is_fin"] = (
        ((industry["IndustryClassificationID"] == "P0201") & industry["IndustryCode"].str.startswith("I"))
        | ((industry["IndustryClassificationID"] == "P0207") & industry["IndustryCode"].str.startswith("J"))
        | ((industry["IndustryClassificationID"] == "P0221") & industry["IndustryCode"].str.startswith("J"))
    ).astype(int)

    industry = industry.rename(columns={
        "Symbol": "Stkcd",
        "IndustryClassificationID": "std_id",
    })

    industry["Stkcd"] = industry["Stkcd"].apply(normalize_stock_code)

    industry = (
        industry
        .sort_values(["Stkcd", "std_id", "ImplementDate"])
        .drop_duplicates(["Stkcd", "std_id", "ImplementDate"], keep="last")
    )

    return industry


def assign_industry_standard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_2012, date_2023 = INDUSTRY_STANDARD_DATES

    out["std_id"] = np.select(
        [
            out["Trddt"] < date_2012,
            (out["Trddt"] >= date_2012) & (out["Trddt"] < date_2023),
            out["Trddt"] >= date_2023,
        ],
        ["P0201", "P0207", "P0221"],
        default="P0201",
    )

    return out


def merge_industry(df: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    print_section("按当期行业匹配金融行业标识")

    left = (
        assign_industry_standard(df)
        .sort_values(["Trddt", "Stkcd", "std_id"])
        .reset_index(drop=True)
        .copy()
    )

    right = (
        industry[["ImplementDate", "Stkcd", "std_id", "IndustryCode", "is_fin"]]
        .dropna(subset=["ImplementDate"])
        .sort_values(["ImplementDate", "Stkcd", "std_id"])
        .reset_index(drop=True)
        .copy()
    )

    merged = pd.merge_asof(
        left,
        right,
        left_on="Trddt",
        right_on="ImplementDate",
        by=["Stkcd", "std_id"],
        direction="backward",
        allow_exact_matches=True,
    )

    match_success = merged[merged["is_fin"].notna()].copy()
    match_fail = merged[merged["is_fin"].isna()].copy()

    print("=" * 80)
    print("1. 匹配总体情况")
    print(f"   总观测数：{len(merged)}")
    print(f"   匹配成功：{len(match_success)}（占比 {len(match_success) / len(merged):.2%}）")
    print(f"   匹配失败：{len(match_fail)}（占比 {len(match_fail) / len(merged):.2%}）")
    print("=" * 80)

    if len(match_fail) > 0:
        print("--- 涉及股票 ---")
        print(f"   匹配失败的股票共 {match_fail['Stkcd'].nunique()} 只")
        print("   前 20 只失败次数最多的股票：")
        print(match_fail["Stkcd"].value_counts().head(20).to_string())

    return merged


def filter_non_financial(df: pd.DataFrame) -> pd.DataFrame:
    print_section("剔除金融行业观测")

    original_len = len(df)

    fin_obs = df["is_fin"].fillna(0).eq(1).sum()
    fin_stocks = df.loc[
        df["is_fin"].fillna(0).eq(1),
        "Stkcd",
    ].nunique()

    out = df.loc[df["is_fin"].fillna(0) != 1].copy()

    print(f"按当期行业口径识别出曾在某些时点属于金融行业的股票共 {fin_stocks} 只")
    print(f"删除的金融行业 股票-日期 观测共 {fin_obs} 条")
    print(f"已从日度收益文件中按当期行业剔除金融股，样本从 {original_len} 下降为 {len(out)}")

    out = out.drop(columns=["std_id", "ImplementDate", "is_fin"]).copy()
    return out


# =============================================================================
# 上市日期与 ST 状态筛选
# =============================================================================

def load_company_data(cfg: Config) -> pd.DataFrame:
    company = pd.read_csv(
        cfg.company_path,
        encoding="utf-8",
        usecols=COMPANY_USECOLS,
        parse_dates=["Listdt"],
    )

    company["Stkcd"] = company["Stkcd"].apply(normalize_stock_code)
    return company


def report_company_duplicates(company: pd.DataFrame) -> None:
    company_check = company[["Stkcd", "Listdt"]].copy()

    dup_stk_mask = company_check.duplicated(subset=["Stkcd"], keep=False)
    dup_stk = company_check.loc[dup_stk_mask].sort_values(["Stkcd", "Listdt"])

    print("TRD_Co 总行数：", len(company_check))
    print("TRD_Co 股票数：", company_check["Stkcd"].nunique())
    print("Stkcd 重复行数：", len(dup_stk))
    print("Stkcd 重复股票数：", dup_stk["Stkcd"].nunique())

    if len(dup_stk) > 0:
        print("重复 Stkcd 示例：")
        print(dup_stk.head(50))

    print("TRD_Co 类型：")
    print(company.dtypes)


def filter_listing_age(df: pd.DataFrame, company: pd.DataFrame) -> pd.DataFrame:
    print_section("删除上市不满 1 年的日度观测值")

    report_company_duplicates(company)

    merged = pd.merge(
        df,
        company,
        how="left",
        on="Stkcd",
    )

    merged["Listdt"] = pd.to_datetime(merged["Listdt"])
    merged["Listdt_1y"] = merged["Listdt"] + pd.DateOffset(years=1)

    out = merged.loc[
        merged["Trddt"] >= merged["Listdt_1y"]
    ].copy()

    out = out.drop(columns=["Listdt_1y"])

    print(f"未匹配上市日期观测数：{merged['Listdt'].isna().sum()}")
    print(f"未匹配上市日期股票数：{merged.loc[merged['Listdt'].isna(), 'Stkcd'].nunique()}")
    print(f"删除上市不满一年观测值前的样本数：{len(merged)}")
    print(f"删除上市不满一年观测值后的样本数：{len(out)}")
    print(f"删除上市不满一年观测值共：{len(merged) - len(out)} 条")

    return out


def filter_normal_trading_status(df: pd.DataFrame) -> pd.DataFrame:
    print_section("删除 ST 等异常交易状态的观测值")

    out = df.loc[df["Trdsta"].isin(NORMAL_TRADING_STATUS)].copy()

    print(f"删除 ST 等异常交易状态观测值前的样本数：{len(df)}")
    print(f"删除 ST 等异常交易状态观测值后的样本数：{len(out)}")
    print(f"删除 ST 等异常交易状态观测值共：{len(df) - len(out)} 条")

    return out


# =============================================================================
# 最终 merge 回完整面板
# =============================================================================

def merge_back_to_full_panel(
    full_panel: pd.DataFrame,
    cleaned_daily: pd.DataFrame,
) -> pd.DataFrame:
    print_section("将清洗后样本 merge 回完整面板")

    valid_stocks = cleaned_daily["Stkcd"].drop_duplicates()

    full_panel_valid = full_panel.loc[
        full_panel["Stkcd"].isin(valid_stocks)
    ].copy()

    final = pd.merge(
        full_panel_valid,
        cleaned_daily.drop(columns=["Trddt"]),
        how="left",
        on=["Stkcd", "td"],
    )

    final = final.drop(columns=OUTPUT_DROP_COLS)

    nan_ret_count = final["ret"].isna().sum()
    print("最终 ret 缺失观测数：", nan_ret_count)
    print("最终股票数：", final["Stkcd"].nunique())
    print("最终观测数：", len(final))

    return final


# =============================================================================
# 主程序
# =============================================================================

def main() -> None:
    setup_runtime()
    cfg = Config()
    ensure_output_dirs(cfg)

    daily = load_daily_data(cfg)
    calendar = load_trading_calendar(cfg)

    daily_td = attach_trading_day(daily, calendar)
    original_columns = daily_td.columns.tolist()

    full_panel = build_full_panel(daily_td, calendar)
    full_panel.to_csv(
        cfg.full_panel_path,
        encoding="utf-8",
        index=False,
    )

    completed = complete_daily_panel(daily_td, full_panel, original_columns)
    completed = add_trade_day_counts(completed)

    industry = load_industry_data(cfg)
    daily_with_industry = merge_industry(completed, industry)
    non_financial = filter_non_financial(daily_with_industry)

    company = load_company_data(cfg)
    listed_one_year = filter_listing_age(non_financial, company)
    cleaned = filter_normal_trading_status(listed_one_year)

    final = merge_back_to_full_panel(full_panel, cleaned)
    final.to_csv(cfg.out_file, index=False, encoding="utf-8")

    print_section("完成")
    print(f"已保存完整面板：{cfg.full_panel_path}")
    print(f"已保存清洗后 DGTW 日度面板：{cfg.out_file}")


if __name__ == "__main__":
    main()
