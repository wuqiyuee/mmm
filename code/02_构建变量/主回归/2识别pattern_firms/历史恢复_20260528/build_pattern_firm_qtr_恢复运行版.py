#!/usr/bin/env python3
"""按 2026 年 5 月 28 日保存的旧算法重建模式公司季度标记。

恢复来源：版本记录 4811223；该脚本与 6 月 8 日、6 月 11 日保存的版本一致。
同目录的 build_pattern_firm_qtr_原样.py 保留历史文件的原始内容。
本恢复版仅适配现有输入路径、日历字段名称和独立输出目录。
当前原始数据已更新，因此本次重算不等于找回 7 月的原始结果文件。

历史算法完整保留：同类财报连续 3 期；公告日和财报截止日均限于
2000—2025 年；按证券代码和交易所筛选沪深 A 股；默认保留公告日等于
财报截止日的记录；保留旧版休市公告的交易日规律标签计算方式。
这些历史行为用于追溯结果变化，不代表对当前主线识别规则的修改。

输出列：
- Stkcd
- Accper
- PatternFirmQtr

这是一个独立脚本，只输出精简的公司-财报季度标识表。
默认使用 CSMAR 的“年、中、季报基本情况文件” IAR_Rept.csv。
其中 Annodt 是公告日，Reptyp 是报告类型，Annowk 是公告星期。

默认口径是同一公司、同一报告类型连续 3 期遵循同一披露规律。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


# 这些是 Noh et al. 附录 A 里使用的 10 类公告日期规律。
# 后续如果同一条公司-季度观测同时命中多个规律，会按这个顺序作为优先级之一。
PATTERN_PRIORITY: List[str] = [
    "same_kth_weekday_same_month",  # 同月第 k 个星期几，例如 4 月第 2 个周二
    "same_kth_weekday_since_accper",  # 财报截止日后第 k 个相同星期几
    "same_weekday_same_nth_week_since_accper",  # 财报截止日后第 n 周的同一星期几
    "same_weekday_same_nth_week_same_month",  # 同月第 n 周的同一星期几
    "same_kth_trading_day_same_month",  # 同月第 k 个交易日
    "same_kth_nonweekend_day_same_month",  # 同月第 k 个非周末日
    "same_kth_nonweekend_day_since_accper",  # 财报截止日后第 k 个非周末日
    "same_kth_trading_day_since_accper",  # 财报截止日后第 k 个交易日
    "same_calendar_month_and_date",  # 固定公历月日，例如每年 4 月 30 日
    "same_kth_day_since_accper",  # 财报截止日后第 k 个自然日
]

# IAR_Rept.csv 里的 Reptyp 报告类型：
# 1=第一季度季报，2=中报，3=第三季度季报，4=年报。
REPORT_TYPE_MAP = {
    1: "Q1",
    2: "H1",
    3: "Q3",
    4: "FY",
}


def find_project_root() -> Path:
    """自动定位项目根目录。

    输入：当前脚本路径 __file__。
    处理：从脚本所在目录逐级向上找，优先选择同时包含 input 和 code 的目录。
    输出：项目根目录路径。

    这样不依赖绝对路径；只要整个项目文件夹结构不变，
    换电脑或换用户名后仍然可以自动找到默认输入和输出位置。
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "input").exists() and (parent / "code").exists():
            return parent

    # 如果还没放 input 文件夹，就按当前文件夹层级回退到项目根目录。
    # 当前脚本位于: 项目根目录/code/02_构建变量/识别pattern_firms/
    return here.parents[3]


PROJECT_ROOT = find_project_root()
DEFAULT_ANN_FILE = PROJECT_ROOT / "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv"
DEFAULT_CAL_FILE = PROJECT_ROOT / "input/CSMAR/日历文件(1990-12-19至2026-08-20)/CER_Cale.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "中间结果/02_变量构造结果/6PatternFirmQtr_旧口径恢复_20260528"
DEFAULT_OUT_FILE = "PatternFirmQtr.csv"


def normalize_stock_code(x: object) -> str:
    """把证券代码统一为六位字符串。

    输入变量：
    - x：原始 Stkcd，可能是数字、字符串或缺失值。

    处理方式：
    - 缺失值返回空字符串。
    - 数字型代码先转成整数，再用 0 补足六位。
    - 字符串代码去掉前后空格后，用 0 补足六位。

    输出变量：
    - Stkcd6：六位证券代码字符串，例如 1 -> "000001"。
    """
    if pd.isna(x):
        return ""
    try:
        return f"{int(float(x)):06d}"
    except Exception:
        return str(x).strip().zfill(6)


def classify_share_class(code: str, sctcd: int) -> str:
    """根据证券代码和交易所代码识别股票板块。

    输入变量：
    - code：六位证券代码 Stkcd6。
    - sctcd：CSMAR 里的交易所代码，1=上海，2=深圳，3=北京。

    处理方式：
    - 上海：区分主板 A 股、科创板、B 股。
    - 深圳：区分主板/中小板 A 股、创业板、B 股。
    - 北京：识别北交所股票，兼容旧代码和 920 新代码。
      本项目的 A 股样本口径不包含北交所；这里识别 bj_a 只是为了后续明确排除。

    输出变量：
    - share_class：例如 sh_a_main、sh_star、sz_a_main、sz_chinext、bj_a。
    """
    if not code:
        return "other"
    if sctcd == 1:
        if code.startswith("900"):
            return "sh_b"
        if code.startswith(("688", "689")):
            return "sh_star"
        if code.startswith(("600", "601", "603", "605")):
            return "sh_a_main"
        return "other"
    if sctcd == 2:
        if code.startswith("200"):
            return "sz_b"
        if code.startswith(("300", "301")):
            return "sz_chinext"
        if code.startswith(("000", "001", "002", "003")):
            return "sz_a_main"
        return "other"
    if sctcd == 3:
        if code.startswith(("920", "8", "9", "4")):
            return "bj_a"
        return "other"
    return "other"


def keep_a_share(share_class: str, scope: str) -> bool:
    """判断一条记录是否保留在 A 股样本里。

    输入变量：
    - share_class：classify_share_class 得到的股票类别。
    - scope：用户传入的样本范围。
      all_a 在本项目中表示沪深 A 股，main_a_only 表示只保留沪深主板。

    处理方式：
    - all_a：保留沪深主板、科创板、创业板，不包含北交所。
    - main_a_only：只保留沪深主板 A 股。

    输出变量：
    - 布尔值：真表示保留，假表示删除。
    """
    if scope == "all_a":
        return share_class in {"sh_a_main", "sh_star", "sz_a_main", "sz_chinext"}
    return share_class in {"sh_a_main", "sz_a_main"}


def infer_report_type(reptyp: pd.Series) -> pd.Series:
    """根据 Reptyp 直接生成报告类型。

    输入变量：
    - reptyp：IAR_Rept.csv 中的 Reptyp，1=一季报，2=中报，3=三季报，4=年报。

    处理方式：
    - 先把 Reptyp 转成数值，再映射到 Q1、H1、Q3、FY。

    输出变量：
    - report_type：Q1/H1/Q3/FY；无法识别的 Reptyp 会得到缺失值。
    """
    return pd.to_numeric(reptyp, errors="coerce").map(REPORT_TYPE_MAP)


def build_a_share_calendar(cal: pd.DataFrame) -> pd.DataFrame:
    """从原始交易日历构造统一的沪深 A 股交易日变量。

    输入变量：
    - cal：交易日历表，至少包含 Markettype、Clddt、State。

    处理方式：
    - 使用 Markettype=1 的上证 A 股日历作为统一市场日历。
    - 本地检查发现上证 A 股、深证 A 股、创业板和科创板开休市状态完全一致。
    - 构造 is_open、open_cum、trading_day_of_month。

    输出变量：
    - a_share_cal：沪深 A 股逐日交易日历。
    """
    cal = cal.copy()
    cal["Clddt"] = pd.to_datetime(cal["Clddt"], errors="coerce")
    cal = cal.dropna(subset=["Clddt"]).sort_values(["Markettype", "Clddt"])

    base = cal.loc[cal["Markettype"] == 1, ["Clddt", "State"]].copy()
    if base.empty:
        raise ValueError("交易日历中没有 Markettype=1 的上证 A 股日历，无法构造统一 A 股日历")
    base = base.sort_values("Clddt").drop_duplicates("Clddt", keep="first")
    base["ym"] = base["Clddt"].dt.to_period("M").astype(str)
    base["is_open"] = (base["State"] == "O").astype(int)
    base["open_cum"] = base["is_open"].cumsum()
    base["trading_day_of_month"] = base.groupby("ym")["is_open"].cumsum()
    base.loc[base["is_open"] == 0, "trading_day_of_month"] = np.nan
    return base[["Clddt", "is_open", "open_cum", "trading_day_of_month"]].copy()


def weekday_sun0(d: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """把日期转换为星期数，且设定周日=0。

    输入变量：日期序列。
    处理方式：pandas 默认周一=0，这里转换成周日=0、周一=1、...、周六=6。
    输出变量：ann_weekday_sun0 等星期变量。
    """
    if isinstance(d, pd.DatetimeIndex):
        return pd.Series((d.weekday + 1) % 7, index=d)
    return (d.dt.weekday + 1) % 7


def week_start_sunday(d: pd.Series) -> pd.Series:
    """计算每个日期所在周的周日日期。

    输入变量：日期序列。
    处理方式：日期减去 weekday_sun0 对应天数。
    输出变量：该日期所在周的周日，用于计算“距期末第几周”。
    """
    return d - pd.to_timedelta(weekday_sun0(d), unit="D")


def add_calendar_features(df: pd.DataFrame, a_share_cal: pd.DataFrame) -> pd.DataFrame:
    """为每条公告记录构造 10 类公告日期规律候选变量。

    输入变量：
    - df：清洗后的公告表，核心列是 Stkcd6、Accper、Actudt。
    - a_share_cal：build_a_share_calendar 生成的统一沪深 A 股交易日历。

    处理方式：
    - 从 Actudt 构造月份、日期、星期、月内第几个星期几等变量。
    - 从 Accper 到 Actudt 的距离构造“距财报截止日第几天/周/交易日”等变量。
    - 涉及交易日顺序的变量统一使用上证 A 股日历，因为沪深 A 股各板块开休市状态一致。
    - 把每类规律拼成稳定的字符串标签，例如 SM_4_2_3、TA_18。

    输出变量：
    - 原 df 加上 10 个公告日期规律字符串列，列名对应 PATTERN_PRIORITY。
    """
    out = df.copy()
    out["ann_month"] = out["Actudt"].dt.month
    out["ann_day"] = out["Actudt"].dt.day
    out["ann_weekday_mon0"] = out["Actudt"].dt.weekday
    out["ann_weekday_sun0"] = weekday_sun0(out["Actudt"])
    out["day_diff_since_accper"] = (out["Actudt"] - out["Accper"]).dt.days

    out["kth_weekday_same_month_k"] = ((out["ann_day"] - 1) // 7) + 1

    month_start = pd.to_datetime(out["Actudt"].values.astype("datetime64[M]"))
    month_start_sun0 = pd.Series((month_start.weekday + 1) % 7, index=out.index)
    out["nth_week_same_month_n"] = ((out["ann_day"] + month_start_sun0 - 1) // 7) + 1

    start_after_acc = out["Accper"] + pd.Timedelta(days=1)
    first_same_wd_offset = (out["ann_weekday_mon0"] - start_after_acc.dt.weekday) % 7
    first_same_wd = start_after_acc + pd.to_timedelta(first_same_wd_offset, unit="D")
    out["kth_weekday_since_accper_k"] = 1 + ((out["Actudt"] - first_same_wd).dt.days // 7)

    out["nth_week_since_accper_n"] = (
        (week_start_sunday(out["Actudt"]) - week_start_sunday(out["Accper"])).dt.days // 7
    )

    out["calendar_month"] = out["ann_month"]
    out["calendar_day"] = out["ann_day"]
    out["kth_day_since_accper_k"] = out["day_diff_since_accper"]

    weekday_mask_ann = out["ann_weekday_mon0"] <= 4
    out["kth_nonweekend_day_same_month_k"] = np.where(
        weekday_mask_ann,
        np.busday_count(
            out["Actudt"].values.astype("datetime64[M]"),
            (out["Actudt"] + pd.Timedelta(days=1)).values.astype("datetime64[D]"),
            weekmask="1111100",
        ),
        np.nan,
    )
    out["kth_nonweekend_day_since_accper_k"] = np.where(
        weekday_mask_ann,
        np.busday_count(
            (out["Accper"] + pd.Timedelta(days=1)).values.astype("datetime64[D]"),
            (out["Actudt"] + pd.Timedelta(days=1)).values.astype("datetime64[D]"),
            weekmask="1111100",
        ),
        np.nan,
    )

    temp = out[["Actudt", "Accper"]].copy()
    ann_map = a_share_cal[["Clddt", "trading_day_of_month", "open_cum"]].rename(
        columns={
            "Clddt": "Actudt",
            "trading_day_of_month": "ann_trading_day_of_month",
            "open_cum": "ann_open_cum",
        }
    )
    acc_map = a_share_cal[["Clddt", "open_cum"]].rename(
        columns={"Clddt": "Accper", "open_cum": "acc_open_cum"}
    )

    temp = temp.merge(ann_map, on="Actudt", how="left")
    temp = temp.merge(acc_map, on="Accper", how="left")
    out["kth_trading_day_same_month_k"] = temp["ann_trading_day_of_month"].values
    out["kth_trading_day_since_accper_k"] = (
        temp["ann_open_cum"] - temp["acc_open_cum"]
    ).values

    def make_str(mask: pd.Series, prefix: str, *parts: Iterable[pd.Series]) -> pd.Series:
        """把若干数字变量拼成一个公告日期规律标签。"""
        s = pd.Series(np.nan, index=out.index, dtype=object)
        if mask.any():
            joined = pd.DataFrame({f"p{i}": p[mask] for i, p in enumerate(parts)})
            joined = joined.astype("Int64").astype(str)
            s.loc[mask] = prefix + joined.apply(lambda r: "_".join(r.values.tolist()), axis=1)
        return s

    out["same_kth_weekday_same_month"] = make_str(
        out["Actudt"].notna(),
        "SM_",
        out["ann_month"],
        out["ann_weekday_sun0"],
        out["kth_weekday_same_month_k"],
    )
    out["same_kth_weekday_since_accper"] = make_str(
        out["Actudt"].notna(),
        "SA_",
        out["ann_weekday_sun0"],
        out["kth_weekday_since_accper_k"],
    )
    out["same_weekday_same_nth_week_since_accper"] = make_str(
        out["Actudt"].notna(),
        "SWA_",
        out["ann_weekday_sun0"],
        out["nth_week_since_accper_n"],
    )
    out["same_weekday_same_nth_week_same_month"] = make_str(
        out["Actudt"].notna(),
        "SWM_",
        out["ann_month"],
        out["ann_weekday_sun0"],
        out["nth_week_same_month_n"],
    )
    out["same_kth_trading_day_same_month"] = make_str(
        out["kth_trading_day_same_month_k"].notna(),
        "TM_",
        out["ann_month"],
        out["kth_trading_day_same_month_k"],
    )
    out["same_kth_nonweekend_day_same_month"] = make_str(
        out["kth_nonweekend_day_same_month_k"].notna(),
        "NM_",
        out["ann_month"],
        out["kth_nonweekend_day_same_month_k"],
    )
    out["same_kth_nonweekend_day_since_accper"] = make_str(
        out["kth_nonweekend_day_since_accper_k"].notna(),
        "NA_",
        out["kth_nonweekend_day_since_accper_k"],
    )
    out["same_kth_trading_day_since_accper"] = make_str(
        out["kth_trading_day_since_accper_k"].notna(),
        "TA_",
        out["kth_trading_day_since_accper_k"],
    )
    out["same_calendar_month_and_date"] = make_str(
        out["Actudt"].notna(),
        "CD_",
        out["calendar_month"],
        out["calendar_day"],
    )
    out["same_kth_day_since_accper"] = make_str(
        out["kth_day_since_accper_k"].notna(),
        "DA_",
        out["kth_day_since_accper_k"],
    )
    return out


def compute_consecutive_flags(
    df: pd.DataFrame, pattern_cols: List[str], min_consecutive: int
) -> pd.DataFrame:
    """判断每类公告日期规律是否连续满足指定期数。

    输入变量：
    - df：已经包含 10 类公告日期规律候选变量的公告表。
    - pattern_cols：需要检查的公告日期规律列名。
    - min_consecutive：连续多少期才进入规律样本；本脚本默认是 3。

    处理方式：
    - 在 Stkcd6 × report_type 内按 Accper 排序。
    - 只有相邻 Accper 正好相差一年且月日相同，才算同一报告期的连续观测。
    - 对每个公告日期规律列计算连续段长度和当前所在位置。

    输出变量：
    - 每个公告日期规律对应三列：
      {pattern}__run_size、{pattern}__pos_in_run、{pattern}__qualified。
      其中 pattern 表示具体的公告日期规律列名。
    """
    out = df.copy()
    out = out.sort_values(["Stkcd6", "report_type", "Accper"]).reset_index(drop=True)

    prev_accper = out.groupby(["Stkcd6", "report_type"])["Accper"].shift(1)
    same_month_day = out["Accper"].dt.strftime("%m-%d").eq(prev_accper.dt.strftime("%m-%d"))
    one_year_apart = out["Accper"].dt.year.eq(prev_accper.dt.year + 1)
    is_consecutive_prev = (same_month_day & one_year_apart).fillna(False)

    for col in pattern_cols:
        prev_val = out.groupby(["Stkcd6", "report_type"])[col].shift(1)
        boundary = (~is_consecutive_prev) | out[col].isna() | prev_val.isna() | out[col].ne(prev_val)
        run_id = boundary.groupby([out["Stkcd6"], out["report_type"]]).cumsum()
        run_size = out.groupby(["Stkcd6", "report_type", run_id])[col].transform("size")
        pos_in_run = out.groupby(["Stkcd6", "report_type", run_id]).cumcount() + 1

        run_size = run_size.where(out[col].notna(), 0)
        pos_in_run = pos_in_run.where(out[col].notna(), 0)
        out[f"{col}__run_size"] = run_size
        out[f"{col}__pos_in_run"] = pos_in_run
        out[f"{col}__qualified"] = (
            out[col].notna() & (run_size >= min_consecutive) & (pos_in_run >= min_consecutive)
        )
    return out


def select_primary_pattern(df: pd.DataFrame, pattern_cols: List[str]) -> pd.DataFrame:
    """把多类公告日期规律的结果汇总成季度层面的规律标识。

    输入变量：
    - df：已经包含 {pattern}__qualified 等连续性判断结果的公告表。
    - pattern_cols：10 类公告日期规律列名。

    处理方式：
    - 只要任一公告日期规律合格，该 Stkcd6-Accper 就是规律季度观测。
    - 如果同时命中多个规律，先选连续段长度更长的；相同则按 PATTERN_PRIORITY 选。

    输出变量：
    - num_qualifying_patterns：该观测命中的公告日期规律数量。
    - is_pattern_observation：季度层面的规律公司标识，后续会改名为 PatternFirmQtr。
    - selected_pattern_type/value：保留主规律信息，便于检查，但最终三列表不输出。
    """
    out = df.copy()
    qual_cols = [f"{c}__qualified" for c in pattern_cols]
    out["num_qualifying_patterns"] = out[qual_cols].sum(axis=1)
    out["is_pattern_observation"] = out["num_qualifying_patterns"] > 0

    out["selected_pattern_type"] = pd.Series(np.nan, index=out.index, dtype=object)
    out["selected_pattern_value"] = pd.Series(np.nan, index=out.index, dtype=object)
    out["selected_pattern_run_size"] = np.nan
    out["selected_pattern_pos_in_run"] = np.nan

    best_run = pd.Series(-1, index=out.index, dtype=float)
    for col in pattern_cols:
        qual = out[f"{col}__qualified"]
        run_size = out[f"{col}__run_size"]
        better = qual & (run_size > best_run)
        out.loc[better, "selected_pattern_type"] = col
        out.loc[better, "selected_pattern_value"] = out.loc[better, col]
        out.loc[better, "selected_pattern_run_size"] = out.loc[better, f"{col}__run_size"]
        out.loc[better, "selected_pattern_pos_in_run"] = out.loc[better, f"{col}__pos_in_run"]
        best_run.loc[better] = run_size.loc[better]

    for col in pattern_cols:
        qual = out[f"{col}__qualified"]
        tie = qual & (out[f"{col}__run_size"] == out["selected_pattern_run_size"])
        replace = tie & out["selected_pattern_type"].ne(col)
        priority_col = PATTERN_PRIORITY.index(col)
        selected_priority = out["selected_pattern_type"].map(
            lambda x: PATTERN_PRIORITY.index(x) if isinstance(x, str) else 999
        )
        replace = replace & (priority_col < selected_priority)
        out.loc[replace, "selected_pattern_type"] = col
        out.loc[replace, "selected_pattern_value"] = out.loc[replace, col]
        out.loc[replace, "selected_pattern_run_size"] = out.loc[replace, f"{col}__run_size"]
        out.loc[replace, "selected_pattern_pos_in_run"] = out.loc[replace, f"{col}__pos_in_run"]
    return out


def parse_args() -> argparse.Namespace:
    """读取命令行参数。

    输入：用户在终端传入的路径、日期区间、连续期数阈值等参数。
    处理：如果用户没有传参数，就使用上方定义的默认路径和默认口径。
    输出：argparse.Namespace，后续用 args.ann_file、args.min_consecutive 等字段取值。
    """
    parser = argparse.ArgumentParser(
        description="生成 Stkcd-Accper-PatternFirmQtr 三列表。"
    )
    parser.add_argument("--ann-file", default=str(DEFAULT_ANN_FILE), help="年、中、季报基本情况文件路径")
    parser.add_argument("--cal-file", default=str(DEFAULT_CAL_FILE), help="交易日历文件路径")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出文件夹")
    parser.add_argument("--out-file", default=DEFAULT_OUT_FILE, help="输出CSV文件名")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--min-consecutive", type=int, default=3)
    parser.add_argument(
        "--a-share-scope",
        choices=["all_a", "main_a_only"],
        default="all_a",
        help="all_a保留沪深A股板块（沪深主板、科创板、创业板），不含北交所；main_a_only仅保留沪深主板A股。",
    )
    parser.add_argument(
        "--drop-actudt-equals-accper",
        action="store_true",
        help="剔除公告日等于财报截止日的记录。",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: set[str], file_label: str) -> None:
    """检查输入表是否包含必要列。

    输入变量：
    - df：读取进来的公告表或交易日历表。
    - required：脚本运行必须存在的列名集合。
    - file_label：报错时显示的文件类型名称。

    处理方式：用 required 减去 df.columns，找出缺失列。
    输出结果：没有缺失列时不返回任何值；有缺失列时直接报错，避免后续变量构造出错。
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_label} 缺少列: {sorted(missing)}")


def main() -> None:
    """生成 Stkcd-Accper-PatternFirmQtr 三列表。

    主要输入：
    - ann：年、中、季报基本情况表，核心原始变量是 Stkcd、Stknme、Reptyp、Sctcd、Accper、Annodt、Annowk。
    - cal：交易日历表，核心原始变量是 Markettype、Clddt、State。

    主要处理：
    1. 清洗年、中、季报基本情况表，构造 Stkcd6、Actudt、share_class、report_type。
    2. 根据公告日期和交易日历构造各种公告日期规律候选变量。
    3. 判断每个公司-财报类型是否连续 min_consecutive 期满足同一公告日期规律。
    4. 把 is_pattern_observation 整理成最终的 PatternFirmQtr。

    最终输出：
    - Stkcd：六位证券代码。
    - Accper：财报截止日。
    - PatternFirmQtr：该公司该财报期是否属于规律公司季度观测，1 是，0 否。
    """
    args = parse_args()

    # 1. 把命令行参数转成路径对象，明确输入文件和输出文件位置。
    # ann_file：年、中、季报基本情况数据；cal_file：交易日历；out_path：最终三列表 CSV。
    ann_file = Path(args.ann_file)
    cal_file = Path(args.cal_file)
    out_dir = Path(args.out_dir)
    out_path = out_dir / args.out_file

    # 2. 先检查输入文件是否存在。这里不读取数据，只做路径层面的保护。
    if not ann_file.exists():
        raise FileNotFoundError(f"公告文件不存在: {ann_file}")
    if not cal_file.exists():
        raise FileNotFoundError(f"交易日历文件不存在: {cal_file}")

    # 3. 读取原始数据。
    # ann 保留年、中、季报基本情况记录；cal 保留交易日历，用于计算“第几个交易日”类公告日期规律。
    ann = pd.read_csv(ann_file, low_memory=False)
    cal = pd.read_csv(cal_file, low_memory=False)

    # 现有日历将同一开休市状态字段命名为 Statse，旧算法内部使用 State。
    # 这里只兼容字段名，保留原始 O/C 开休市编码和旧版日历计算方法。
    if "State" not in cal.columns and "Statse" in cal.columns:
        cal = cal.rename(columns={"Statse": "State"})

    # 4. 检查后续会用到的核心列。
    # 年、中、季报基本情况表需要证券代码、证券名称、报告类型、交易所代码、财报截止日、报告公布日。
    # 交易日历需要市场类型、日历日期、开休市状态。
    require_columns(ann, {"Stkcd", "Stknme", "Reptyp", "Sctcd", "Accper", "Annodt"}, "公告文件")
    require_columns(cal, {"Markettype", "Clddt", "State"}, "交易日历文件")

    # 5. 清洗公告表的基础变量。
    # Stkcd6：把原始 Stkcd 统一成六位证券代码。
    # Accper：财报截止日，转成 pandas 日期。
    # Actudt：脚本内部统一使用的公告日，由 IAR_Rept.csv 的 Annodt 转换得到。
    # Annowk：报告公布星期，仅用于一致性检查；pattern 计算仍以 Actudt 日期本身为准。
    # dropna 后保留能识别公司、交易所、财报期和披露日的记录。
    ann = ann.copy()
    ann["Stkcd6"] = ann["Stkcd"].map(normalize_stock_code)
    ann["Accper"] = pd.to_datetime(ann["Accper"], errors="coerce")
    ann["Actudt"] = pd.to_datetime(ann["Annodt"], errors="coerce")
    if "Annowk" in ann.columns:
        ann["Annowk_calc"] = weekday_sun0(ann["Actudt"])
        ann["Annowk"] = pd.to_numeric(ann["Annowk"], errors="coerce")
        ann["Annowk_mismatch"] = (
            ann["Annowk"].notna()
            & ann["Actudt"].notna()
            & ann["Annowk"].ne(ann["Annowk_calc"])
        )
        # Annowk 校验记录：
        # 2026-05-26 检查 IAR_Rept.csv 时发现：
        # - 原始表共有 285,703 行。
        # - Annodt 和 Annowk 都有效的记录有 285,699 行。
        # - 原始表中 Annodt 推算星期与 Annowk 不一致的记录有 56 行。
        # - 按本脚本默认沪深 A 股筛选口径处理后，样本有 270,351 行。
        # - 默认筛选样本中不一致记录有 53 行。
        # 例子：2002-05-09 实际是星期四，按 CSMAR 编码应为 4，但原表 Annowk=5。
        # 因此本脚本以 Annodt 日期重新计算星期，Annowk 只作为数据质量校验字段。
    ann = ann.dropna(subset=["Accper", "Actudt", "Stkcd6", "Sctcd"])

    # 6. 构造样本筛选和分组变量。
    # share_class：根据证券代码和 Sctcd 判断是沪主板、科创板、深主板、创业板、北交所等。
    # 默认 all_a 样本只保留沪深 A 股，北交所 bj_a 会在 keep_a_share 中被排除。
    # report_type：根据 IAR_Rept.csv 的 Reptyp 直接得到 Q1、H1、Q3、FY。
    ann["share_class"] = [
        classify_share_class(code, int(sctcd))
        for code, sctcd in zip(ann["Stkcd6"], ann["Sctcd"])
    ]
    ann = ann[ann["share_class"].map(lambda x: keep_a_share(x, args.a_share_scope))].copy()
    ann["report_type"] = infer_report_type(ann["Reptyp"])
    ann = ann.dropna(subset=["report_type"])

    # 7. 按用户设定的日期区间筛选样本。
    # 这里同时约束实际披露日 Actudt 和财报截止日 Accper。
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    ann = ann[(ann["Actudt"] >= start_date) & (ann["Actudt"] <= end_date)].copy()
    ann = ann[(ann["Accper"] >= start_date) & (ann["Accper"] <= end_date)].copy()

    # 8. 可选保守处理：剔除实际披露日等于财报截止日的记录。
    # 默认不剔除；只有运行时传入 --drop-actudt-equals-accper 才会执行。
    if args.drop_actudt_equals_accper:
        ann = ann[ann["Actudt"] != ann["Accper"]].copy()

    # 9. 对同一公司同一财报截止日去重。
    # 排序后保留最早 Actudt 的一条记录，保证结果不依赖原始 CSV 的行顺序。
    ann = (
        ann.sort_values(["Stkcd6", "Accper", "Actudt"])
        .drop_duplicates(["Stkcd6", "Accper"], keep="first")
        .copy()
    )

    # 10. 识别规律公司季度观测。
    # a_share_cal：由上证 A 股日历构造的统一沪深 A 股日历，包含是否开市、累计交易日、月内第几个交易日。
    # add_calendar_features：给 ann 增加 10 类公告日期规律候选变量。
    # compute_consecutive_flags：判断每类公告日期规律是否连续满足 min_consecutive 期。
    # select_primary_pattern：得到 is_pattern_observation，即最终的季度层面规律标识。
    a_share_cal = build_a_share_calendar(cal)
    ann = add_calendar_features(ann, a_share_cal)
    ann = compute_consecutive_flags(ann, PATTERN_PRIORITY, args.min_consecutive)
    ann = select_primary_pattern(ann, PATTERN_PRIORITY)

    # 11. 整理最终三列表。
    # 原始识别结果中的 is_pattern_observation 是布尔值：
    # 真表示该 Stkcd6-Accper 是规律公司季度观测，假表示不是。
    # 这里重命名为 PatternFirmQtr，并转成 1/0，方便后续回归合并。
    out = ann[["Stkcd6", "Accper", "is_pattern_observation"]].copy()
    out = out.rename(
        columns={
            "Stkcd6": "Stkcd",
            "is_pattern_observation": "PatternFirmQtr",
        }
    )
    out["Accper"] = out["Accper"].dt.strftime("%Y-%m-%d")
    out["PatternFirmQtr"] = out["PatternFirmQtr"].astype(int)
    out = out.sort_values(["Stkcd", "Accper"]).reset_index(drop=True)

    # 12. 写出 CSV。
    # utf-8-sig 方便 Excel 直接打开中文路径和内容时不乱码。
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 13. 打印几个检查量，方便确认输出行数和 PatternFirmQtr 的分布。
    print(f"[OK] wrote {len(out):,} rows to {out_path}")
    print(f"PatternFirmQtr mean = {out['PatternFirmQtr'].mean():.6f}")
    print(f"PatternFirmQtr count = {int(out['PatternFirmQtr'].sum()):,}")
    if "Annowk_mismatch" in ann.columns:
        print(f"Annowk mismatch count = {int(ann['Annowk_mismatch'].sum()):,}")


if __name__ == "__main__":
    main()
