# -*- coding: utf-8 -*-
"""
按公告日距离构造 Pattern Firms（模式公司报告）记忆矩阵。

本脚本与原来的财报期滞后版本分开保存，暂时不读取或构造盈余惊喜变量。
核心单位是“公司—公告日”：同一家公司同一天发布年报和一季报时，
两份报告保留在报告层，但公司公告事件只保留一条。

处理链条：
1. 读取全部财报明细，并合并 PatternFirmQtr；
2. 汇总为唯一的公司—公告日事件；
3. 合并公告日新闻关注度、事件交易日异常成交量和价格异动；
4. 构造历史同日公告关系；
5. 将当前公告事件与过去 364 天的历史关系连接；
6. 在每个当前公司对内，按历史共同公告距离从近到远排序，生成最近的
   lag1 至 lag4；同时保留历史公告距离变量。

本脚本不读取 sue.csv，也不生成任何盈余惊喜变量。
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.simplefilter(action="ignore", category=PerformanceWarning)

try:
    from tqdm.auto import tqdm
except ImportError:  # 没有安装 tqdm 时仍允许脚本运行
    def tqdm(iterable, **kwargs):
        return iterable


# 公告日为交易日时保留当天；非交易日顺延到下一个交易日。
EVENT_DATE_MODE = "next_if_nontrading"

# 历史公告回看窗口，以及最多保留的最近历史公告次数。
LOOKBACK_DAYS = 364
MEMORY_RANK_LABELS = (1, 2, 3, 4)


# ============================================================================
# 工具函数
# ============================================================================

def normalize_stock_code(value: object) -> str:
    """将股票代码标准化为六位字符串。"""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    return f"{int(float(text)):06d}"


def xtile_by_group(series: pd.Series, n: int) -> pd.Series:
    """复现 Stata xtile：并列值不按出现顺序拆开。"""
    if n < 2:
        raise ValueError("分组数量 n 必须至少为 2。")

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="Int64",
    )

    non_missing = series.dropna()
    count = len(non_missing)
    if count == 0:
        return result

    values = non_missing.to_numpy(dtype="float64")
    sorted_values = np.sort(values)
    cutpoints = np.empty(n - 1, dtype="float64")

    for k in range(1, n):
        lower_index, remainder = divmod(count * k, n)

        if remainder == 0:
            cutpoints[k - 1] = (
                sorted_values[lower_index - 1]
                + sorted_values[lower_index]
            ) / 2
        else:
            cutpoints[k - 1] = sorted_values[lower_index]

    groups = (
        np.searchsorted(cutpoints, values, side="left")
        + 1
    )
    result.loc[non_missing.index] = groups
    return result


def check_unique(
    data: pd.DataFrame,
    keys: list[str],
    name: str,
) -> None:
    """检查指定主键是否唯一。"""
    duplicate_count = int(data.duplicated(keys).sum())
    if duplicate_count:
        raise ValueError(
            f"{name}在主键 {keys} 上存在重复记录：{duplicate_count}"
        )


# ============================================================================
# 路径与输入检查
# ============================================================================

def resolve_root(root: Optional[str | Path] = None) -> Path:
    """获取项目根目录。"""
    if root is not None:
        return Path(root).expanduser().resolve()

    env_root = os.getenv("MMM_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (
            (candidate / "input").exists()
            and (candidate / "中间结果").exists()
        ):
            return candidate

    return current


REL_PATHS = {
    "IAR_Rept": Path(
        "input/CSMAR/"
        "年、中、季报基本情况文件1990-12-31至2026-06-30/"
        "IAR_Rept.csv"
    ),
    "pattern_firm_qtr": Path(
        "中间结果/02_变量构造结果/"
        "6PatternFirmQtr/PatternFirmQtr.csv"
    ),
    "AES_AEV": Path(
        "中间结果/02_变量构造结果/"
        "3报刊新闻_AES_AEV/报刊新闻_AES_AEV.csv"
    ),
    "avol": Path(
        "中间结果/02_变量构造结果/"
        "2交易行为OIB、Avol/Avol_CSMAR_prep.csv"
    ),
    "TopMover": Path(
        "中间结果/02_变量构造结果/"
        "7TopMover/TopMover1_top100.csv"
    ),
    "calendar": Path(
        "中间结果/01_清洗后样本/市场交易日历_td.csv"
    ),
    "output_csv": Path(
        "中间结果/03_记忆矩阵/"
        "MemoryMatrixPattern_公告日距离版.csv"
    ),
}


def build_paths(root: Path) -> dict[str, Path]:
    """把项目根目录和相对路径组合起来。"""
    return {
        key: root / relative_path
        for key, relative_path in REL_PATHS.items()
    }


def check_input_paths(paths: dict[str, Path]) -> None:
    """检查必要输入文件是否存在。"""
    missing = [
        str(path)
        for key, path in paths.items()
        if key != "output_csv" and not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "以下输入文件不存在，请检查项目根目录或数据路径：\n"
            + "\n".join(f"- {path}" for path in missing)
        )


# ============================================================================
# 第一步：读取报告明细，并构造唯一公司—公告日事件
# ============================================================================

def read_report_details(paths: dict[str, Path]) -> pd.DataFrame:
    """
    读取全部报告明细，并合并 PatternFirmQtr。

    这里不按 Stkcd + Annodt 去重，因为同日年报和一季报需要同时保留。
    PatternFirmQtr 缺失时保留缺失，不视为非模式报告
    """
    reports = pd.read_csv(
        paths["IAR_Rept"],
        encoding="utf-8-sig",
        usecols=["Stkcd", "Accper", "Annodt", "Reptyp"],
        parse_dates=["Accper", "Annodt"],
    )
    reports["Stkcd"] = reports["Stkcd"].apply(normalize_stock_code)
    reports["Reptyp"] = pd.to_numeric(
        reports["Reptyp"],
        errors="coerce",
    )

    # 国泰安中 Annodt == Accper 表示有财务数据但未披露相关公告。
    # 这类记录不是实际公告，不参与公告日事件汇总。
    n_undisclosed = int(
        reports["Annodt"].eq(reports["Accper"]).sum()
    )
    print("未披露占位记录数：", n_undisclosed)

    reports = reports.loc[
        reports["Annodt"].ne(reports["Accper"])
    ].copy()

    check_unique(
        reports,
        ["Stkcd", "Accper"],
        "IAR_Rept",
    )

    pattern = pd.read_csv(
        paths["pattern_firm_qtr"],
        encoding="utf-8-sig",
        usecols=["Stkcd", "Accper", "PatternFirmQtr"],
        parse_dates=["Accper"],
    )
    pattern["Stkcd"] = pattern["Stkcd"].apply(
        normalize_stock_code
    )
    pattern["PatternFirmQtr"] = pd.to_numeric(
        pattern["PatternFirmQtr"],
        errors="coerce",
    )

    check_unique(
        pattern,
        ["Stkcd", "Accper"],
        "PatternFirmQtr",
    )

    reports = reports.merge(
        pattern,
        how="left",
        on=["Stkcd", "Accper"],
        validate="1:1",
    )

    missing_annodt = int(reports["Annodt"].isna().sum())
    if missing_annodt:
        print(
            "公告日缺失记录数，无法构造公告事件：",
            missing_annodt,
        )
        reports = reports.dropna(subset=["Annodt"])

    print("报告明细总行数：", len(reports))
    print(
        "模式报告行数：",
        int(reports["PatternFirmQtr"].eq(1).sum()),
    )
    print(
        "历史足够但未识别模式的报告行数：",
        int(reports["PatternFirmQtr"].eq(0).sum()),
    )
    print(
        "标记缺失或未匹配的报告行数：",
        int(reports["PatternFirmQtr"].isna().sum()),
    )

    return reports


def build_company_day_events(
    reports: pd.DataFrame,
) -> pd.DataFrame:
    """
    将报告明细汇总为公司—公告日事件。

    事件是否保留由当天是否至少有一份模式报告决定；报告数量和报告类型
    统计基于当天全部报告，避免把非模式的同日报告误记为不存在。
    """
    data = reports.copy()
    data["is_pattern"] = data["PatternFirmQtr"].eq(1).astype("int8")
    data["is_nonpattern"] = data["PatternFirmQtr"].eq(0).astype("int8")
    data["is_annual"] = data["Reptyp"].eq(4).astype("int8")
    data["is_q1"] = data["Reptyp"].eq(1).astype("int8")

    data["pattern_annual"] = (
        data["is_pattern"] * data["is_annual"]
    ).astype("int8")
    data["pattern_q1"] = (
        data["is_pattern"] * data["is_q1"]
    ).astype("int8")

    events = (
        data.groupby(
            ["Stkcd", "Annodt"],
            as_index=False,
        )
        .agg(
            nReports=("Accper", "nunique"),
            nPatternReports=("is_pattern", "sum"),
            hasAnnual=("is_annual", "max"),
            hasQ1=("is_q1", "max"),
            patternAnnual=("pattern_annual", "max"),
            patternQ1=("pattern_q1", "max"),
            hasNonPatternReport=("is_nonpattern", "max"),
        )
    )

    events["eventPattern"] = (
        events["nPatternReports"] > 0
    ).astype("int8")
    events["multipleReports"] = (
        events["nReports"] > 1
    ).astype("int8")


    # PatternFirmQtr 上游当前只覆盖沪深A股；这里仅保留当天
    # 至少有一份模式报告的公司，故当前配对两端均为沪深A股。
    # 本脚本未单独检查市场类型；更换上游标记表后需重新核查。

    events = events.loc[
        events["eventPattern"].eq(1)
    ].copy()

    check_unique(
        events,
        ["Stkcd", "Annodt"],
        "公司公告事件表",
    )

    print("模式公司公告事件数量：", len(events))
    print(
        "同日多报告事件数量：",
        int(events["multipleReports"].sum()),
    )
    print(
        "同日两份报告且两份均为模式报告数量：",
        int(
            events["nPatternReports"].ge(2).sum()
        ),
    )

    return events


# ============================================================================
# 第二步：公告日映射到事件交易日，并合并公司日关注度
# ============================================================================

def add_event_trading_date(
    events: pd.DataFrame,
    calendar_path: Path,
) -> pd.DataFrame:
    """给公告事件添加公告后首个交易日和交易日序号。"""
    calendar = pd.read_csv(
        calendar_path,
        encoding="utf-8-sig",
        usecols=["Trddt", "td"],
        parse_dates=["Trddt"],
    ).sort_values("Trddt")

    if calendar[["Trddt", "td"]].isna().any().any():
        raise ValueError("交易日历的 Trddt 或 td 存在缺失值。")
    if calendar["Trddt"].duplicated().any():
        raise ValueError("交易日历中存在重复的 Trddt。")
    if calendar["td"].duplicated().any():
        raise ValueError("交易日历中存在重复的 td。")

    date_key = (
        events[["Annodt"]]
        .drop_duplicates()
        .sort_values("Annodt")
    )

    date_map = pd.merge_asof(
        date_key,
        calendar.rename(
            columns={
                "Trddt": "event_Trddt",
                "td": "event_td",
            }
        ),
        left_on="Annodt",
        right_on="event_Trddt",
        direction="forward",
        allow_exact_matches=(
            EVENT_DATE_MODE == "next_if_nontrading"
        ),
    )

    result = events.merge(
        date_map,
        how="left",
        on="Annodt",
        validate="m:1",
    )

    if result[["event_Trddt", "event_td"]].isna().any().any():
        raise ValueError(
            "部分公告日期无法映射到事件交易日，请检查交易日历。"
        )

    return result


def merge_attention_features(
    events: pd.DataFrame,
    paths: dict[str, Path],
) -> pd.DataFrame:
    """在唯一公司—公告日层面合并新闻、成交量和价格异动变量。"""
    result = events.copy()

    # 新闻关注度按公告自然日合并。
    aev = pd.read_csv(
        paths["AES_AEV"],
        encoding="utf-8-sig",
        usecols=["Stkcd", "date", "AEV"],
        parse_dates=["date"],
    )
    aev["Stkcd"] = aev["Stkcd"].apply(normalize_stock_code)
    aev["AEV"] = pd.to_numeric(aev["AEV"], errors="coerce")

    result = result.merge(
        aev,
        how="left",
        left_on=["Stkcd", "Annodt"],
        right_on=["Stkcd", "date"],
        validate="m:1",
    ).drop(columns=["date"])
    result["AEV_nonmiss"] = result["AEV"].fillna(0)

    # 异常成交量按公告后首个交易日合并。
    avol = pd.read_csv(
        paths["avol"],
        encoding="utf-8-sig",
        usecols=["Stkcd", "Trddt", "td", "avol", "avolS"],
        parse_dates=["Trddt"],
    )
    avol["Stkcd"] = avol["Stkcd"].apply(normalize_stock_code)
    for col in ["td", "avol", "avolS"]:
        avol[col] = pd.to_numeric(avol[col], errors="coerce")

    result = result.merge(
        avol,
        how="left",
        left_on=["Stkcd", "event_Trddt"],
        right_on=["Stkcd", "Trddt"],
        suffixes=("", "_avol"),
        validate="m:1",
    ).drop(columns=["Trddt"])

    td_mismatch = (
        result["event_td"].notna()
        & result["td"].notna()
        & result["event_td"].ne(result["td"])
    )
    if td_mismatch.any():
        raise ValueError(
            "事件交易日序号与异常成交量表的 td 不一致。"
        )

    # 价格异动按公告后首个交易日合并。
    top_mover = pd.read_csv(
        paths["TopMover"],
        encoding="utf-8-sig",
        usecols=["Stkcd", "Trddt", "topMover"],
        parse_dates=["Trddt"],
    )
    top_mover["Stkcd"] = top_mover["Stkcd"].apply(
        normalize_stock_code
    )
    top_mover["topMover"] = pd.to_numeric(
        top_mover["topMover"],
        errors="coerce",
    )

    result = result.merge(
        top_mover,
        how="left",
        left_on=["Stkcd", "event_Trddt"],
        right_on=["Stkcd", "Trddt"],
        suffixes=("", "_topMover"),
        validate="m:1",
    ).drop(columns=["Trddt"])
    result["topMover"] = result["topMover"].fillna(0).astype("int8")

    return result


def build_attention_dummies(events: pd.DataFrame) -> pd.DataFrame:
    """在公司公告日层面构造高新闻关注和高异常成交量标记。"""
    result = events.copy()
    result["year"] = result["Annodt"].dt.year

    result["AEV_nonmiss_medGr"] = (
        result.groupby("year")["AEV_nonmiss"]
        .transform(lambda s: xtile_by_group(s, 2))
    )

    for col in ["avol", "avolS"]:
        result[f"{col}_medGr"] = (
            result.groupby("year")[col]
            .transform(lambda s: xtile_by_group(s, 2))
        )

    result["hi_AEV"] = (
        result["AEV_nonmiss_medGr"] == 2
    ).fillna(False).astype("int8")
    result["hi_avol"] = (
        result["avol_medGr"] == 2
    ).fillna(False).astype("int8")
    result["hi_avolS"] = (
        result["avolS_medGr"] == 2
    ).fillna(False).astype("int8")

    return result


# ============================================================================
# 第三步：构造历史共同公告关系和当前提示事件
# ============================================================================

EVENT_FEATURES = [
    "hi_AEV",
    "hi_avol",
    "hi_avolS",
    "topMover",
    "nReports",
    "nPatternReports",
    "multipleReports",
    "hasNonPatternReport",
]


def build_encoding_pairs(events: pd.DataFrame) -> pd.DataFrame:
    """构造历史同日公告的有方向公司对。"""
    check_unique(events, ["Stkcd", "Annodt"], "公司公告事件表")

    left = events.rename(
        columns={"Annodt": "EncodingAnnodt"}
    )[["EncodingAnnodt", "Stkcd", *EVENT_FEATURES]]

    right_rename = {
        "Stkcd": "StkcdRecall",
        "Annodt": "EncodingAnnodt",
        **{
            col: f"{col}Recall"
            for col in EVENT_FEATURES
        },
    }
    right = events.rename(
        columns=right_rename
    )[[
        "EncodingAnnodt",
        "StkcdRecall",
        *[f"{col}Recall" for col in EVENT_FEATURES],
    ]]

    pairs = left.merge(
        right,
        how="inner",
        on="EncodingAnnodt",
        validate="many_to_many",
    )
    pairs = pairs.loc[
        pairs["Stkcd"] != pairs["StkcdRecall"]
    ].copy()

    # 统计该线索公司在这次共同公告中遇到的其他公司数量。
    pairs["nrFirmsAnnouncing"] = (
        pairs.groupby(
            ["EncodingAnnodt", "Stkcd"]
        )["StkcdRecall"]
        .transform("nunique")
    )

    for col in ["hi_AEV", "hi_avol", "hi_avolS"]:
        pairs[f"both_{col}"] = (
            pairs[col].eq(1)
            & pairs[f"{col}Recall"].eq(1)
        ).astype("int8")

    pairs["both_topMover"] = (
        pairs["topMover"].eq(1)
        & pairs["topMoverRecall"].eq(1)
    ).astype("int8")
    pairs["any_topMover"] = (
        pairs["topMover"].eq(1)
        | pairs["topMoverRecall"].eq(1)
    ).astype("int8")
    pairs["encodingMultiReport"] = (
        pairs["multipleReports"].eq(1)
        | pairs["multipleReportsRecall"].eq(1)
    ).astype("int8")

    check_unique(
        pairs,
        ["EncodingAnnodt", "Stkcd", "StkcdRecall"],
        "历史共同公告公司对",
    )
    return pairs


def build_current_cues(events: pd.DataFrame) -> pd.DataFrame:
    """每个合格模式公司公告日都可以成为当前提示事件。"""
    current_cols = [
        "Annodt",
        "event_Trddt",
        "event_td",
        "Stkcd",
        "hi_AEV",
        "hi_avol",
        "hi_avolS",
        "topMover",
        "nReports",
        "nPatternReports",
        "hasAnnual",
        "hasQ1",
        "patternAnnual",
        "patternQ1",
        "multipleReports",
        "hasNonPatternReport",
    ]

    cues = events[current_cols].rename(
        columns={
            "Annodt": "CueAnnodt",
        }
    )
    check_unique(cues, ["CueAnnodt", "Stkcd"], "当前提示事件表")
    return cues


# ============================================================================
# 第四步：按公告日距离排序构造记忆矩阵
# ============================================================================

def aggregate_memory_rows(temp: pd.DataFrame) -> pd.DataFrame:
    """将同一当前公司对的历史公告按距离次序整理为一行。"""
    keys = ["CueAnnodt", "Stkcd", "StkcdRecall"]
    check_unique(
        temp,
        [*keys, "memoryRankLag"],
        "公司对历史公告次序",
    )
    current_cols = [
        "event_Trddt",
        "event_td",
        "current_hi_AEV",
        "current_hi_avol",
        "current_hi_avolS",
        "current_topMover",
        "current_nReports",
        "current_nPatternReports",
        "current_hasAnnual",
        "current_hasQ1",
        "current_patternAnnual",
        "current_patternQ1",
        "current_multipleReports",
        "current_hasNonPatternReport",
    ]

    base_agg = {
        col: (col, "first")
        for col in current_cols
        if col in temp.columns
    }
    base_agg["closestMemoryAgeDays"] = (
        "memoryAgeDays",
        "min",
    )
    base_agg["nHistoricalOverlapDates"] = (
        "EncodingAnnodt",
        "nunique",
    )

    result = temp.groupby(keys, as_index=False).agg(**base_agg)

    lag_feature_map = {
        "hi_AEV": "both_hi_AEV",
        "hiavol": "both_hi_avol",
        "hiavolS": "both_hi_avolS",
        "topMover": "both_topMover",
        "anyTopMover": "any_topMover",
        "encodingMultiReport": "encodingMultiReport",
    }

    for lag in MEMORY_RANK_LABELS:
        sub = temp.loc[
            temp["memoryRankLag"].eq(lag)
        ].copy()

        if sub.empty:
            result[f"hasOverlapLag{lag}"] = 0
            result[f"daysSinceOverlapAnnLag{lag}"] = np.nan
            result[f"nrFirmsAnnLag{lag}"] = np.nan
            result[f"lag{lag}encodingMultiReport"] = 0
            for output_suffix in lag_feature_map:
                if output_suffix != "encodingMultiReport":
                    result[f"lag{lag}{output_suffix}"] = 0
            continue

        # 每个 rank 只对应一次历史共同公告，直接读取该事件的特征。
        grouped = sub.groupby(keys, as_index=False)
        lag_values = grouped.agg(
            **{
                f"hasOverlapLag{lag}": (
                    "memoryRankLag",
                    lambda values: 1,
                ),
                f"daysSinceOverlapAnnLag{lag}": (
                    "memoryAgeDays",
                    "first",
                ),
                **{
                    f"lag{lag}{output_suffix}": (
                        source_col,
                        "first",
                    )
                    for output_suffix, source_col
                    in lag_feature_map.items()
                },
                f"nrFirmsAnnLag{lag}": (
                    "nrFirmsAnnouncing",
                    "first",
                ),
            }
        )
        result = result.merge(
            lag_values,
            how="left",
            on=keys,
            validate="1:1",
        )

    result = result.rename(columns={"CueAnnodt": "Annodt"})

    integer_cols = [
        col
        for col in result.columns
        if col.startswith("hasOverlapLag")
        or col.startswith("lag")
        or col.startswith("current_")
    ]
    for col in integer_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype("int8")

    return result


def build_memory_matrix(
    cues: pd.DataFrame,
    encoding_pairs: pd.DataFrame,
    test_days: Optional[int] = None,
) -> pd.DataFrame:
    """连接当前提示事件与过去364天并按距离排序的历史共同公告。"""
    if test_days is not None and test_days < 0:
        raise ValueError("test_days 不能为负数。")

    cue_dates = (
        cues["CueAnnodt"]
        .drop_duplicates()
        .sort_values()
    )
    if test_days is not None:
        cue_dates = cue_dates.iloc[:test_days]

    current_rename = {
        col: f"current_{col}"
        for col in [
            "hi_AEV",
            "hi_avol",
            "hi_avolS",
            "topMover",
            "nReports",
            "nPatternReports",
            "hasAnnual",
            "hasQ1",
            "patternAnnual",
            "patternQ1",
            "multipleReports",
            "hasNonPatternReport",
        ]
    }

    current = cues.rename(columns=current_rename)
    results: list[pd.DataFrame] = []

    for current_date in tqdm(
        cue_dates,
        desc="正在构造公告日距离记忆矩阵",
        unit="day",
        ncols=80,
        mininterval=1.0,
        leave=True,
        ascii=True,
    ):
        current_cues = current.loc[
            current["CueAnnodt"].eq(current_date)
        ].copy()
        if current_cues.empty:
            continue

        window_start = current_date - pd.Timedelta(days=LOOKBACK_DAYS)
        historical = encoding_pairs.loc[
            encoding_pairs["EncodingAnnodt"].ge(window_start)
            & encoding_pairs["EncodingAnnodt"].lt(current_date)
        ].copy()
        if historical.empty:
            continue

        # 线索公司今天的公告，连接该公司过去形成的所有历史关系。
        temp = current_cues.merge(
            historical,
            how="inner",
            on="Stkcd",
            validate="many_to_many",
        )
        if temp.empty:
            continue

        temp["memoryAgeDays"] = (
            temp["CueAnnodt"]
            - temp["EncodingAnnodt"]
        ).dt.days
        rank_keys = [
            "CueAnnodt",
            "Stkcd",
            "StkcdRecall",
        ]
        # 以公司对为单位排序：距离越小，历史公告次序越靠前。
        temp = temp.sort_values(
            [*rank_keys, "memoryAgeDays", "EncodingAnnodt"],
            kind="stable",
        )
        temp["memoryRankLag"] = (
            temp.groupby(rank_keys, sort=False)
            .cumcount()
            .add(1)
            .astype("Int64")
        )

        results.append(aggregate_memory_rows(temp))

    if not results:
        return pd.DataFrame()

    memory = pd.concat(results, ignore_index=True)
    memory = memory.sort_values(
        ["Annodt", "Stkcd", "StkcdRecall"]
    ).reset_index(drop=True)
    return memory


# ============================================================================
# 第五步：字段检查、保存和命令行入口
# ============================================================================

def validate_memory_matrix(memory: pd.DataFrame) -> None:
    """检查最终记忆矩阵的主键、自己配对和公告距离变量。"""
    if memory.empty:
        print("记忆矩阵为空，没有可检查的记录。")
        return

    required = {
        "Annodt",
        "event_Trddt",
        "event_td",
        "Stkcd",
        "StkcdRecall",
        "closestMemoryAgeDays",
    }
    missing = sorted(required - set(memory.columns))
    if missing:
        raise ValueError(
            "记忆矩阵缺少必要字段：" + ", ".join(missing)
        )

    self_pairs = int(
        memory["Stkcd"].eq(memory["StkcdRecall"]).sum()
    )
    if self_pairs:
        raise ValueError(
            f"记忆矩阵中存在自己配自己记录：{self_pairs}"
        )

    duplicate_count = int(
        memory.duplicated(
            ["Annodt", "Stkcd", "StkcdRecall"]
        ).sum()
    )
    if duplicate_count:
        raise ValueError(
            "记忆矩阵的公司对主键存在重复："
            f"{duplicate_count}"
        )

    invalid_age = (
        memory["closestMemoryAgeDays"].le(0)
        | memory["closestMemoryAgeDays"].gt(LOOKBACK_DAYS)
    ).sum()
    if invalid_age:
        raise ValueError(
            f"记忆矩阵存在不在1至{LOOKBACK_DAYS}天范围内的历史距离："
            f"{int(invalid_age)}"
        )


def run_pipeline(
    root: Optional[str | Path] = None,
    output_csv: Optional[str | Path] = None,
    test_days: Optional[int] = None,
) -> pd.DataFrame:
    """读取数据、构造公告日距离记忆矩阵并保存结果。"""
    project_root = resolve_root(root)
    paths = build_paths(project_root)

    if output_csv is not None:
        output_path = Path(output_csv).expanduser()
        if not output_path.is_absolute():
            output_path = project_root / output_path
        paths["output_csv"] = output_path.resolve()

    check_input_paths(paths)

    # 1. 报告明细层：同一公司同一天的多份报告全部保留。
    reports = read_report_details(paths)

    # 2. 公司公告事件层：一天一家公司一行。
    events = build_company_day_events(reports)
    events = add_event_trading_date(events, paths["calendar"])

    # 3. 关注度和价格异动只在公司公告日层面合并一次。
    events = merge_attention_features(events, paths)
    events = build_attention_dummies(events)

    # 4. 历史共同公告关系和当前提示事件分开构造。
    encoding_pairs = build_encoding_pairs(events)
    cues = build_current_cues(events)

    # 5. 按公告日距离排序构造最终记忆矩阵。
    memory = build_memory_matrix(
        cues,
        encoding_pairs,
        test_days=test_days,
    )
    validate_memory_matrix(memory)

    output_path = paths["output_csv"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memory.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("保存成功：", output_path)
    print("最终表形状：", memory.shape)
    if not memory.empty:
        print(
            "最终唯一键重复数：",
            int(
                memory.duplicated(
                    ["Annodt", "Stkcd", "StkcdRecall"]
                ).sum()
            ),
        )
        print("最终列数：", len(memory.columns))

    return memory


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="按公告日距离构造 Pattern Firms 记忆矩阵"
    )
    parser.add_argument(
        "--root",
        default=None,
        help="项目根目录；默认自动寻找。",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="自定义输出 CSV 路径。",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=None,
        help="只计算最早的 N 个当前公告日，用于测试。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    run_pipeline(
        root=args.root,
        output_csv=args.output_csv,
        test_days=args.test_days,
    )


if __name__ == "__main__":
    main()
