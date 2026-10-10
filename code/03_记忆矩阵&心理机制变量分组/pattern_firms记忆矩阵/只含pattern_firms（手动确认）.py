# 将公告自然日映射至事件交易日：
# 交易日公告保留当天，非交易日公告顺延到下一个交易日。

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from tqdm.auto import tqdm

warnings.simplefilter(action="ignore", category=PerformanceWarning)


EVENT_DATE_MODE = "next_if_nontrading"

VALID_EVENT_DATE_MODES = {
    "next_if_nontrading",
    "strict_next_trading",
}

if EVENT_DATE_MODE not in VALID_EVENT_DATE_MODES:
    raise ValueError(
        f"EVENT_DATE_MODE 只能是 {VALID_EVENT_DATE_MODES}，"
        f"当前是 {EVENT_DATE_MODE}"
    )
#===============================================================================
## 工具函数
#===============================================================================
def normalize_stock_code(x:object) -> str:
    """标准化股票代码，填充为6位数，并转换为字符串"""
    if pd.isna(x):
        return ''
    s=str(x).strip()
    if s=='':
        return ''
    return f"{int(float(s)):06d}"

def xtile_by_group(
    s: pd.Series,
    n: int,
) -> pd.Series:
    """复现 Stata xtile 的默认分位数组；并列值不拆分。"""
    if n < 2:
        raise ValueError(
            "分组数量 n 必须至少为 2。"
        )

    out = pd.Series(
        pd.NA,
        index=s.index,
        dtype="Int64",
    )

    non_missing = s.dropna()
    m = len(non_missing)

    if m == 0:
        return out

    values = non_missing.to_numpy(
        dtype="float64"
    )
    sorted_values = np.sort(values)

    cutpoints = np.empty(
        n - 1,
        dtype="float64",
    )

    for k in range(1, n):
        lower_index, remainder = divmod(
            m * k,
            n,
        )

        if remainder == 0:
            cutpoints[k - 1] = (
                sorted_values[lower_index - 1]
                + sorted_values[lower_index]
            ) / 2
        else:
            cutpoints[k - 1] = (
                sorted_values[lower_index]
            )

    groups = (
        np.searchsorted(
            cutpoints,
            values,
            side="left",
        )
        + 1
    )

    out.loc[non_missing.index] = groups

    return out

def dummy_int(s):
    """
    把变量转成 0/1 整数。
    缺失值处理为 0。
    """
    return s.fillna(0).astype(int)
#===============================================================================
## 路径设置
#===============================================================================

def resolve_root(root: Optional[str | Path] = None) -> Path:
    """
    获取项目根目录。

    优先顺序：
    1. 使用函数传入的目录；
    2. 使用环境变量 MMM_ROOT；
    3. 从当前目录向上寻找同时包含 input 和 中间结果 的目录；
    4. 如果仍未找到，就使用当前目录。
    """
    if root is not None:
        return Path(root).expanduser().resolve()

    env_root = os.getenv("MMM_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "input").exists() and (
            candidate / "中间结果"
        ).exists():
            return candidate

    return cwd

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
        "3报刊新闻_AES_AEV/"
        "报刊新闻_AES_AEV.csv"
    ),
    "avol": Path(
        "中间结果/02_变量构造结果/"
        "2交易行为OIB、Avol/Avol_CSMAR_prep.csv"
    ),
    "sue": Path(
        "中间结果/02_变量构造结果/"
        "4sue/sue.csv"
    ),
    "TopMover": Path(
        "中间结果/02_变量构造结果/"
        "7TopMover/TopMover1_top100.csv"
    ),
    "calendar": Path(
        "中间结果/01_清洗后样本/"
        "市场交易日历_td.csv"
    ),
    "output_csv": Path(
        "中间结果/03_记忆矩阵/"
        "MemoryMatrixPattern(交易变量用公告后首个交易日).csv"
    ),
}


def build_paths(root: Path) -> Dict[str, Path]:
    """把项目根目录与各个相对路径组合起来。"""
    return {
        key: root / relative_path
        for key, relative_path in REL_PATHS.items()
    }

def check_input_paths(paths: Dict[str, Path]) -> None:
    """检查所有输入文件是否存在。"""
    missing = [
        str(path)
        for key, path in paths.items()
        if key != "output_csv" and not path.exists()
    ]

    if missing:
        missing_text = "\n".join(
            f"- {path}" for path in missing
        )
        raise FileNotFoundError(
            "以下输入文件不存在，请检查项目根目录或数据路径：\n"
            f"{missing_text}"
        )


#===============================================================================
## 读取并筛选基础公告样本
#===============================================================================
def add_event_trading_date(
    events: pd.DataFrame,
    calendar_path: Path,
) -> pd.DataFrame:
    """为公告事件添加用于合并交易变量的事件交易日。"""
    calendar = pd.read_csv(
        calendar_path,
        encoding="utf-8",
        usecols=["Trddt", "td"],
        parse_dates=["Trddt"],
    ).sort_values("Trddt")

    if calendar[["Trddt", "td"]].isna().any().any():
        raise ValueError(
            "交易日历的 Trddt 或 td 存在缺失值。"
        )

    if calendar["Trddt"].duplicated().any():
        raise ValueError(
            "交易日历中存在重复的 Trddt。"
        )

    if calendar["td"].duplicated().any():
        raise ValueError(
            "交易日历中存在重复的 td。"
        )

    annodt_key = (
        events[["Annodt"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Annodt")
    )

    date_map = pd.merge_asof(
        annodt_key,
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

    if result["event_Trddt"].isna().any():
        raise ValueError(
            "部分公告日期无法映射到交易日，"
            "请检查交易日历的日期范围。"
        )

    result["Trddt"] = result["event_Trddt"]

    return result



def read_base_data(paths: Dict[str, Path]) -> pd.DataFrame:
    """读取公告数据，并筛选 PatternFirmQtr 等于 1 的样本。"""
    IAR_Rept = pd.read_csv(
        paths["IAR_Rept"],
        encoding="utf-8",
        usecols=["Stkcd", "Accper", "Annodt"],
        parse_dates=["Accper", "Annodt"],
    )

    IAR_Rept["Stkcd"] = (
        IAR_Rept["Stkcd"].apply(normalize_stock_code)
    )

    # 公告日等于财报截止日是未披露占位值，不参与公告配对。
    n_undisclosed = int(IAR_Rept["Annodt"].eq(IAR_Rept["Accper"]).sum())
    print("未披露占位记录数：", n_undisclosed)
    IAR_Rept = IAR_Rept.loc[
        IAR_Rept["Annodt"].ne(IAR_Rept["Accper"])
    ].copy()

    # 同一公司同一公告日存在多个财报期时，保留较晚的财报期。
    rows_before_dedup = len(IAR_Rept)

    duplicate_company_day = IAR_Rept.duplicated(
        ["Stkcd", "Annodt"],
        keep=False,
    )

    IAR_Rept = (
        IAR_Rept
        .sort_values(
            ["Stkcd", "Annodt", "Accper"],
            kind="stable",
        )
        .drop_duplicates(
            ["Stkcd", "Annodt"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    print(
        "同公司同公告日重复记录数：",
        int(duplicate_company_day.sum()),
    )
    print(
        "按原文口径删除记录数：",
        rows_before_dedup - len(IAR_Rept),
    )


    pattern_firm_qtr = pd.read_csv(
        paths["pattern_firm_qtr"],
        encoding="utf-8",
        parse_dates=["Accper"],
    )

    pattern_firm_qtr["Stkcd"] = (
        pattern_firm_qtr["Stkcd"].apply(normalize_stock_code)
    )

    IAR_Rept1 = pd.merge(
        IAR_Rept,
        pattern_firm_qtr,
        how="inner",
        on=["Stkcd", "Accper"],
        validate="1:1",
    )

    if IAR_Rept1.empty:
        raise ValueError(
            "公告表与 PatternFirmQtr 按 Stkcd 和 Accper "
            "合并后没有匹配记录。"
        )

    # 此处只筛选模式公司季度，没有显式限定证券市场类型。
    # 核查当前数据后，入选记录均为沪深A股，不含B股或北交所A股。
    # 这是当前数据的结果，不代表 PatternFirmQtr == 1 本身会排除B股；
    # 上游数据更新后需重新核查市场类型。
   
    IAR_Rept2 = IAR_Rept1.loc[
        IAR_Rept1["PatternFirmQtr"] == 1
    ].copy()

    print("全样本数量：", len(IAR_Rept1))
    print("目标公司样本数量：", len(IAR_Rept2))
    print(
        "目标公司样本占比：",
        len(IAR_Rept2) / len(IAR_Rept1),
    )

    # 新闻变量仍然按照真实公告自然日合并。
    IAR_Rept2["date"] = IAR_Rept2["Annodt"]

    IAR_Rept2 = add_event_trading_date(
        IAR_Rept2,
        paths["calendar"],
    )

    return IAR_Rept2

#===============================================================================
## 合并公告日特征变量
#===============================================================================

def merge_announcement_features(
    base_events: pd.DataFrame,
    paths: Dict[str, Path],
) -> pd.DataFrame:
    """
    合并新闻关注度、异常交易量、盈余惊喜和价格异动股标记。
    
    其中：异常交易量和价格异动通过市场的第一个反应日的反应来间接衡量收到的关注度。
    """

    # 1. 新闻关注度：按照真实公告自然日合并。
    aev = pd.read_csv(
        paths["AES_AEV"],
        encoding="utf-8-sig",
        usecols=["Stkcd", "date", "AEV"],
        parse_dates=["date"],
    )
    aev["Stkcd"] = aev["Stkcd"].apply(
        normalize_stock_code
    )
    aev["AEV"] = pd.to_numeric(aev["AEV"])

    events = pd.merge(
        base_events,
        aev,
        how="left",
        on=["Stkcd", "date"],
        validate="m:1",
    )

    # 没有可用新闻记录时，按当前研究口径记为 0。
    events["AEV_nonmiss"] = events["AEV"].fillna(0)

    # 2. 异常交易量：按照事件交易日合并。
    # 前面已经将非交易日公告的反应日调整到下一交易日。
    avol = pd.read_csv(
        paths["avol"],
        encoding="utf-8-sig",
        usecols=[
            "Stkcd",
            "Trddt",
            "td",
            "avol",
            "avolS",
        ],
        parse_dates=["Trddt"],
    )
    avol["Stkcd"] = avol["Stkcd"].apply(
        normalize_stock_code
    )

    for col in ["td", "avol", "avolS"]:
        avol[col] = pd.to_numeric(avol[col])

    events = pd.merge(
        events,
        avol,
        how="left",
        on=["Stkcd", "Trddt"],
        validate="m:1",
    )

    # 两套交易日序号应当一致。
    td_mismatch = (
        events["event_td"].notna()
        & events["td"].notna()
        & events["event_td"].ne(events["td"])
    )

    if td_mismatch.any():
        raise ValueError(
            "事件交易日序号与异常交易量表的 td 不一致。"
        )

    # 3. 盈余惊喜：按照公司代码和财报截止日合并。
    sue = pd.read_csv(
        paths["sue"],
        encoding="utf-8-sig",
        usecols=[
            "Stkcd",
            "Accper",
            "sue1",
            "sue2",
            "sue3",
        ],
        parse_dates=["Accper"],
    )
    sue["Stkcd"] = sue["Stkcd"].apply(
        normalize_stock_code
    )

    for col in ["sue1", "sue2", "sue3"]:
        sue[col] = pd.to_numeric(sue[col])

    events = pd.merge(
        events,
        sue,
        how="left",
        on=["Stkcd", "Accper"],
        validate="m:1",
    )

    # 这里只关心盈余惊喜的大小，不区分正负方向。
    for col in ["sue1", "sue2", "sue3"]:
        events[f"{col}_abs"] = events[col].abs()

    # 4. 高绝对收益价格异动股：按照事件交易日合并。
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
        top_mover["topMover"]
    )

    events = pd.merge(
        events,
        top_mover,
        how="left",
        on=["Stkcd", "Trddt"],
        validate="m:1",
    )

    # 未进入每日绝对收益率前100名的公司记为 0。
    events["topMover"] = (
        events["topMover"]
        .fillna(0)
        .astype("int8")
    )

    return events

#===============================================================================
## 构造公告显著性指标，用于后续的心理机制检验
#===============================================================================

def build_attention_dummies(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """按公告年份构造高关注、高交易量和高盈余惊喜指标。"""
    events = events.copy()
    events["year"] = events["Annodt"].dt.year

    # 新闻关注度和异常交易量：按年划分为高低两组。
    events["AEV_nonmiss_medGr"] = (
        events.groupby("year")["AEV_nonmiss"]
        .transform(lambda s: xtile_by_group(s, 2))
    )

    for col in ["avol", "avolS"]:
        events[f"{col}_medGr"] = (
            events.groupby("year")[col]
            .transform(lambda s: xtile_by_group(s, 2))
        )

    # 盈余惊喜绝对值：按年划分为十组。
    for col in [
        "sue1_abs",
        "sue2_abs",
        "sue3_abs",
    ]:
        events[f"{col}_dec"] = (
            events.groupby("year")[col]
            .transform(lambda s: xtile_by_group(s, 10))
        )

    # 高新闻关注和高异常交易量：位于年度较高一组。
    events["hi_AEV"] = (
        events["AEV_nonmiss_medGr"] == 2
    ).fillna(False).astype(int)

    events["hi_avol"] = (
        events["avol_medGr"] == 2
    ).fillna(False).astype(int)

    events["hi_avolS"] = (
        events["avolS_medGr"] == 2
    ).fillna(False).astype(int)

    # 高盈余惊喜：位于年度最高十分位组。
    # 同时区分正向和负向盈余惊喜。
    for k in range(1, 4):
        events[f"hi_sue{k}_abs"] = (
            events[f"sue{k}_abs_dec"] == 10
        ).fillna(False).astype(int)

        events[f"sue{k}_pos"] = (
            (events[f"sue{k}"] > 0)
            & events[f"sue{k}"].notna()
        ).astype(int)

        events[f"sue{k}_neg"] = (
            (events[f"sue{k}"] < 0)
            & events[f"sue{k}"].notna()
        ).astype(int)

    keep_cols = [
        "Stkcd",
        "Annodt",
        "Accper",
        "hi_AEV",
        "hi_avol",
        "hi_avolS",
        "hi_sue1_abs",
        "hi_sue2_abs",
        "hi_sue3_abs",
        "sue1_pos",
        "sue2_pos",
        "sue3_pos",
        "sue1_neg",
        "sue2_neg",
        "sue3_neg",
        "topMover",
    ]

    return events[keep_cols].copy()


#===============================================================================
## 生成同日公告公司对
#===============================================================================

def make_same_day_pairs(
    event_flags: pd.DataFrame,
) -> pd.DataFrame:
    """生成同一公告日内的有方向公司对。"""

    duplicate_company_day = event_flags.duplicated(
        ["Stkcd", "Annodt"],
        keep=False,
    )

    if duplicate_company_day.any():
        duplicate_count = int(
            duplicate_company_day.sum()
        )

        raise ValueError(
            "同一家公司在同一公告日存在多条记录，"
            "不能直接构造公司对。"
            f"重复记录数：{duplicate_count}"
        )
    rename_cols = [
        "Stkcd",
        "Accper",
        "hi_AEV",
        "hi_avol",
        "hi_avolS",
        "hi_sue1_abs",
        "hi_sue2_abs",
        "hi_sue3_abs",
        "sue1_pos",
        "sue2_pos",
        "sue3_pos",
        "sue1_neg",
        "sue2_neg",
        "sue3_neg",
        "topMover",
    ]

    rename_dict = {
        col: f"{col}Recall"
        for col in rename_cols
    }

    recall_events = event_flags.rename(
        columns=rename_dict
    )

    pairs = recall_events.merge(
        event_flags,
        on="Annodt",
        how="inner",
        validate="m:m",
    )

    # 删除同一家公司与自己的配对。
    pairs = pairs.loc[
        pairs["Stkcd"] != pairs["StkcdRecall"]
    ].copy()

    # 把公司对的三个关键字段放到最前面。
    front_cols = [
        "Annodt",
        "Stkcd",
        "StkcdRecall",
    ]

    other_cols = [
        col for col in pairs.columns
        if col not in front_cols
    ]

    pairs = pairs[front_cols + other_cols]

    return pairs.sort_values(
        ["Annodt", "Stkcd", "StkcdRecall"]
    ).reset_index(drop=True)

#===============================================================================
## 滚动构造 记忆矩阵 。这里基于原文，但是原文的逻辑有问题
#===============================================================================

# 如果 A 当前季度公告日没有出现在“同日公告公司对”表中，也就是当天没有其他 Pattern firm
# 和 A 同日公告，那么即使 A 公司上季度和其他公司一起公告过，这次公告也不会被识别为 cueing event。

# IAR_Rept4_merge 里的 Stkcd*Accper 不全：
# 原始 IAR_Rept.csv 有 285,703 个唯一 Stkcd*Accper，
# 脚本先筛成 PatternFirmQtr==1 后剩 22,387 个，
# 而进入 IAR_Rept4_merge 的只有 22,037 个，少了 350 个。
# 原因是 IAR_Rept4_merge 按 Annodt 自连接后又剔除了同公司配对，
# 所以只有“同一天还有另一个 pattern firm 公告”的事件才会保留。

def get_memory_vars() -> Dict[str, Tuple[str, str]]:
    """定义最终变量名与两家公司原始变量之间的对应关系。"""
    return {
        "hiAEV": ("hi_AEV", "hi_AEVRecall"),
        "hiavol": ("hi_avol", "hi_avolRecall"),
        "hiavolS": ("hi_avolS", "hi_avolSRecall"),

        "hi_sue1_abs": ("hi_sue1_abs", "hi_sue1_absRecall"),
        "hi_sue2_abs": ("hi_sue2_abs", "hi_sue2_absRecall"),
        "hi_sue3_abs": ("hi_sue3_abs", "hi_sue3_absRecall"),

        "sue1pos": ("sue1_pos", "sue1_posRecall"),
        "sue2pos": ("sue2_pos", "sue2_posRecall"),
        "sue3pos": ("sue3_pos", "sue3_posRecall"),

        "sue1neg": ("sue1_neg", "sue1_negRecall"),
        "sue2neg": ("sue2_neg", "sue2_negRecall"),
        "sue3neg": ("sue3_neg", "sue3_negRecall"),

        "topMover": ("topMover", "topMoverRecall"),
    }


def get_memory_core_columns() -> list[str]:
    """返回补入事件交易日前，记忆矩阵应有的 98 个字段。"""
    memory_suffixes = list(get_memory_vars().keys())
    memory_suffixes.append("anyTopMover")

    key_cols = [
        "Annodt",
        "Stkcd",
        "StkcdRecall",
    ]

    quarter_lag_cols = [
        f"quarterLag{lag}Alt"
        for lag in range(0, 5)
    ]

    regular_lag_cols = [
        f"lag{lag}{suffix}"
        for lag in range(1, 5)
        for suffix in memory_suffixes
    ]

    extreme_sue_cols = [
        f"lag{lag}hi_sue{k}_{direction}"
        for lag in range(1, 5)
        for k in range(1, 4)
        for direction in ["pos", "neg"]
    ]

    nr_firms_cols = [
        f"nrFirmsAnnLag{lag}Alt"
        for lag in range(0, 5)
    ]

    days_since_cols = [
        f"daysSinceOverlapLag{lag}Alt"
        for lag in range(0, 5)
    ]

    return (
        key_cols
        + quarter_lag_cols
        + regular_lag_cols
        + extreme_sue_cols
        + nr_firms_cols
        + days_since_cols
    )


def add_quarter_lag_columns(
    temp: pd.DataFrame,
    pair_cols: list[str],
) -> pd.DataFrame:
    """生成季度距离变量和 quarterLag0Alt 至 quarterLag4Alt。"""
    temp = temp.copy()

    # 以窗口内最大的 Accper（财报截止日）划分季度滞后。
    # 如果公司后来补发较旧财报，当天的共同公告可能被归入历史滞后期。
    # 当前参与配对的模式公司样本中，此类提示事件核查为 0 次；上游数据更新后需重新核查。
    temp["maxAccper_byStkcd"] = (
        temp.groupby("Stkcd")["Accper"].transform("max")
    )

    temp["days_from_current_accper"] = (
        temp["maxAccper_byStkcd"] - temp["Accper"]
    ).dt.days

    diff_days = temp["days_from_current_accper"]

    # 先生成逐行的季度位置变量。
    dummy_data = {
        "dummyQuarterLag0Alt": (
            temp["Accper"] == temp["maxAccper_byStkcd"]
        ).astype(int)
    }

    for lag in range(1, 5):
        lower = (lag - 1) * 100
        upper = lag * 100

        dummy_data[f"dummyQuarterLag{lag}Alt"] = (
            (diff_days > lower) &
            (diff_days <= upper)
        ).astype(int)

    temp = pd.concat(
        [temp, pd.DataFrame(dummy_data, index=temp.index)],
        axis=1,
    )

    # 再汇总到线索公司—被提示公司的公司对层面。
    dummy_cols = [
        f"dummyQuarterLag{lag}Alt"
        for lag in range(0, 5)
    ]

    quarter_lags = (
        temp.groupby(pair_cols)[dummy_cols]
        .transform("max")
        .fillna(0)
        .astype(int)
    )

    quarter_lags.columns = [
        f"quarterLag{lag}Alt"
        for lag in range(0, 5)
    ]

    return pd.concat(
        [temp, quarter_lags],
        axis=1,
    )


def add_regular_memory_lags(
    temp: pd.DataFrame,
    pair_cols: list[str],
    memory_vars: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """生成普通强关联的 dummyLag 和 lag 变量。"""
    dummy_data = {}

    # 新增口径：历史共同公告时，至少一家公司是价格异动公司。
    any_top_mover = (
        dummy_int(temp["topMover"]).eq(1)
        | dummy_int(temp["topMoverRecall"]).eq(1)
    ).astype(int)

    # 先判断每条历史记录是否满足双方同时高关注。
    for lag in range(1, 5):
        lag_col = f"dummyQuarterLag{lag}Alt"
        lag_indicator = dummy_int(temp[lag_col])

        for suffix, (cue_var, recall_var) in memory_vars.items():
            dummy_col = f"dummyLag{lag}{suffix}"

            dummy_data[dummy_col] = (
                lag_indicator
                * dummy_int(temp[cue_var])
                * dummy_int(temp[recall_var])
            )

        dummy_data[f"dummyLag{lag}anyTopMover"] = (
            lag_indicator * any_top_mover
        )

    temp = pd.concat(
        [temp, pd.DataFrame(dummy_data, index=temp.index)],
        axis=1,
    )

    # 再汇总到线索公司—被提示公司的公司对层面。
    dummy_cols = list(dummy_data.keys())

    lag_data = (
        temp.groupby(pair_cols)[dummy_cols]
        .transform("max")
        .fillna(0)
        .astype(int)
    )

    lag_data.columns = [
        col.replace("dummyLag", "lag", 1)
        for col in dummy_cols
    ]

    return pd.concat(
        [temp, lag_data],
        axis=1,
    )


def add_extreme_sue_lags(
    temp: pd.DataFrame,
    pair_cols: list[str],
) -> pd.DataFrame:
    """生成双方同为极端正向或极端负向盈余惊喜的变量。"""
    dummy_data = {}

    for lag in range(1, 5):
        lag_col = f"dummyQuarterLag{lag}Alt"
        lag_indicator = dummy_int(temp[lag_col])

        for k in range(1, 4):
            dummy_pos_col = f"dummyLag{lag}hi_sue{k}_pos"
            dummy_neg_col = f"dummyLag{lag}hi_sue{k}_neg"

            # 双方的盈余惊喜都很高，而且方向都为正。
            dummy_data[dummy_pos_col] = (
                lag_indicator
                * dummy_int(temp[f"sue{k}_pos"])
                * dummy_int(temp[f"sue{k}_posRecall"])
                * dummy_int(temp[f"hi_sue{k}_abs"])
                * dummy_int(temp[f"hi_sue{k}_absRecall"])
            )

            # 双方的盈余惊喜都很高，而且方向都为负。
            dummy_data[dummy_neg_col] = (
                lag_indicator
                * dummy_int(temp[f"sue{k}_neg"])
                * dummy_int(temp[f"sue{k}_negRecall"])
                * dummy_int(temp[f"hi_sue{k}_abs"])
                * dummy_int(temp[f"hi_sue{k}_absRecall"])
            )

    temp = pd.concat(
        [temp, pd.DataFrame(dummy_data, index=temp.index)],
        axis=1,
    )

    dummy_cols = list(dummy_data.keys())

    lag_data = (
        temp.groupby(pair_cols)[dummy_cols]
        .transform("max")
        .fillna(0)
        .astype(int)
    )

    lag_data.columns = [
        col.replace("dummyLag", "lag", 1)
        for col in dummy_cols
    ]

    return pd.concat(
        [temp, lag_data],
        axis=1,
    )


def add_nr_firms_announcing(
    temp: pd.DataFrame,
    pair_cols: list[str],
) -> pd.DataFrame:
    """生成不同季度位置对应的同日公告公司数量。"""
    temp = temp.copy()

    # 统计每家线索公司在每个公告日对应多少家其他公告公司。
    temp["nrFirmsAnnouncing"] = (
        temp.groupby(
            ["Stkcd", "Annodt"]
        )["StkcdRecall"].transform("size")
    )

    int_data = {}

    for lag in range(0, 5):
        lag_col = f"dummyQuarterLag{lag}Alt"
        int_col = f"nrFirmsAnnLag{lag}Alt_int"

        int_data[int_col] = np.where(
            temp[lag_col] == 1,
            temp["nrFirmsAnnouncing"],
            np.nan,
        )

    temp = pd.concat(
        [temp, pd.DataFrame(int_data, index=temp.index)],
        axis=1,
    )

    int_cols = list(int_data.keys())

    final_data = (
        temp.groupby(pair_cols)[int_cols]
        .transform("max")
    )

    final_data.columns = [
        col.replace("_int", "")
        for col in int_cols
    ]

    return pd.concat(
        [temp, final_data],
        axis=1,
    )

def add_days_since_overlap(
    temp: pd.DataFrame,
    pair_cols: list[str],
) -> pd.DataFrame:
    """按照财报截止日差生成 daysSinceOverlapLag0Alt 至 Lag4Alt。"""
    int_data = {}

    for lag in range(0, 5):
        lag_col = f"dummyQuarterLag{lag}Alt"
        int_col = f"daysSinceOverlapLag{lag}Alt_int"

        if lag == 0:
            int_data[int_col] = np.where(
                temp[lag_col] == 1,
                0,
                np.nan,
            )
        else:
            int_data[int_col] = np.where(
                temp[lag_col] == 1,
                temp["days_from_current_accper"],
                np.nan,
            )

    temp = pd.concat(
        [temp, pd.DataFrame(int_data, index=temp.index)],
        axis=1,
    )

    int_cols = list(int_data.keys())

    # 同一公司对在同一 lag 内存在多条记录时，沿用原文取最大值。
    final_data = (
        temp.groupby(pair_cols)[int_cols]
        .transform("max")
    )

    final_data.columns = [
        col.replace("_int", "")
        for col in int_cols
    ]

    return pd.concat(
        [temp, final_data],
        axis=1,
    )


def keep_final_memory_columns(
    memory_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """检查并保留记忆矩阵应输出的 98 个核心字段。"""
    expected_cols = get_memory_core_columns()

    missing_cols = [
        col
        for col in expected_cols
        if col not in memory_matrix.columns
    ]

    if missing_cols:
        raise ValueError(
            "记忆矩阵缺少必要字段："
            + ", ".join(missing_cols)
        )

    return memory_matrix[expected_cols].copy()


def build_memory_matrix(
    same_day_pairs: pd.DataFrame,
    test_days: Optional[int] = None,
) -> pd.DataFrame:
    """按照原文口径滚动构造记忆矩阵。"""

    if test_days is not None and test_days < 0:
        raise ValueError(
            "test_days 不能为负数。"
        )

    pair_cols = ["Stkcd", "StkcdRecall"]
    memory_vars = get_memory_vars()
    memory_list: list[pd.DataFrame] = []

    # 继续使用同日公告公司对中的日期，保持原文样本口径。
    annodt_list = (
        same_day_pairs["Annodt"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    # 测试时可以只计算前几个公告日；None 表示计算全部日期。
    if test_days is not None:
        annodt_list = annodt_list.iloc[:test_days]

    for current_date in tqdm(
        annodt_list,
        desc="正在构造 MemoryMatrix",
        unit="day",
        ncols=80,
        mininterval=1.0,
        leave=True,
        ascii=True,
    ):
        min_annodt = current_date - pd.Timedelta(days=364)

        # 1. 取当前公告日及此前364天的滚动窗口。
        temp = same_day_pairs[
            (
                same_day_pairs["Annodt"]
                >= min_annodt
            )
            & (
                same_day_pairs["Annodt"]
                <= current_date
            )
        ].copy()

        if temp.empty:
            continue

        # 2. 构造各类记忆变量。
        temp = add_quarter_lag_columns(
            temp,
            pair_cols,
        )

        temp = add_regular_memory_lags(
            temp,
            pair_cols,
            memory_vars,
        )

        temp = add_extreme_sue_lags(
            temp,
            pair_cols,
        )

        temp = add_nr_firms_announcing(
            temp,
            pair_cols,
        )

        temp = add_days_since_overlap(
            temp,
            pair_cols,
        )

        # 整理前面多次增加列产生的内存碎片。
        temp = temp.copy()

        # 3. 每个有方向公司对只保留最后一条历史共同公告。
        temp = (
            temp.sort_values(
                ["Stkcd", "StkcdRecall", "Annodt"]
            )
            .groupby(
                pair_cols,
                as_index=False,
            )
            .tail(1)
            .copy()
        )

        # 4. 把历史公告日改成线索公司的当前公告日。
        temp["maxAnnodt_byStkcd"] = (
            temp.groupby("Stkcd")["Annodt"]
            .transform("max")
        )

        temp["Annodt"] = temp["maxAnnodt_byStkcd"]

        # 5. 只保留本次循环对应的当前公告事件。
        temp = temp[
            temp["Annodt"] == current_date
        ].copy()

        if temp.empty:
            continue

        memory_list.append(temp)

    if not memory_list:
        empty_matrix = pd.DataFrame(
            {
                col: pd.Series(dtype="float64")
                for col in get_memory_core_columns()
            }
        )

        empty_matrix["Annodt"] = pd.Series(
            dtype="datetime64[ns]"
        )
        empty_matrix["Stkcd"] = pd.Series(
            dtype="string"
        )
        empty_matrix["StkcdRecall"] = pd.Series(
            dtype="string"
        )

        return empty_matrix

    memory_matrix = pd.concat(
        memory_list,
        ignore_index=True,
    )

    return keep_final_memory_columns(
        memory_matrix
    )


def add_event_date_columns(
    memory_matrix: pd.DataFrame,
    base_events: pd.DataFrame,
) -> pd.DataFrame:
    """为最终记忆矩阵补回当前公告对应的事件交易日。"""
    event_key = (
        base_events[
            ["Annodt", "event_Trddt", "event_td"]
        ]
        .drop_duplicates()
    )

    if event_key.duplicated("Annodt").any():
        raise ValueError(
            "同一个 Annodt 对应了多个事件交易日。"
        )

    result = memory_matrix.merge(
        event_key,
        on="Annodt",
        how="left",
        validate="m:1",
    )

    if result[
        ["event_Trddt", "event_td"]
    ].isna().any().any():
        raise ValueError(
            "部分记忆矩阵记录没有匹配到事件交易日。"
        )

    front_cols = [
        "Annodt",
        "event_Trddt",
        "event_td",
        "Stkcd",
        "StkcdRecall",
    ]

    other_cols = [
        col
        for col in result.columns
        if col not in front_cols
    ]

    return result[front_cols + other_cols]


def validate_final_memory_matrix(
    memory_matrix: pd.DataFrame,
) -> None:
    """检查最终记忆矩阵的字段和主键。"""
    core_cols = [
        col
        for col in get_memory_core_columns()
        if col not in {
            "Annodt",
            "Stkcd",
            "StkcdRecall",
        }
    ]

    expected_cols = [
        "Annodt",
        "event_Trddt",
        "event_td",
        "Stkcd",
        "StkcdRecall",
        *core_cols,
    ]

    missing_cols = [
        col
        for col in expected_cols
        if col not in memory_matrix.columns
    ]

    unexpected_cols = [
        col
        for col in memory_matrix.columns
        if col not in expected_cols
    ]

    if missing_cols or unexpected_cols:
        raise ValueError(
            "最终记忆矩阵字段不符合预期；"
            f"缺少字段：{missing_cols}；"
            f"多余字段：{unexpected_cols}"
        )

    self_pair_count = int(
        (
            memory_matrix["Stkcd"]
            == memory_matrix["StkcdRecall"]
        ).sum()
    )

    if self_pair_count > 0:
        raise ValueError(
            "最终记忆矩阵中存在自己配自己的记录；"
            f"记录数：{self_pair_count}"
        )

    duplicate_count = int(
        memory_matrix.duplicated(
            ["Stkcd", "Annodt", "StkcdRecall"]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "最终记忆矩阵的公司对主键存在重复；"
            f"重复记录数：{duplicate_count}"
        )


def run_pipeline(
    root: Optional[str | Path] = None,
    output_csv: Optional[str | Path] = None,
    test_days: Optional[int] = None,
) -> pd.DataFrame:
    """读取数据、构造记忆矩阵并保存结果。"""
    project_root = resolve_root(root)
    paths = build_paths(project_root)

    # 如果手动指定输出位置，就覆盖默认输出路径。
    if output_csv is not None:
        paths["output_csv"] = (
            Path(output_csv)
            .expanduser()
            .resolve()
        )

    check_input_paths(paths)

    # 1. 读取并筛选基础公告样本。
    base_events = read_base_data(paths)

    # 2. 合并公告日特征。
    events_with_features = merge_announcement_features(
        base_events,
        paths,
    )

    # 3. 构造高关注度等虚拟变量。
    event_flags = build_attention_dummies(
        events_with_features
    )

    # 4. 构造有方向的同日公告公司对。
    same_day_pairs = make_same_day_pairs(
        event_flags
    )

    # 5. 滚动构造记忆矩阵。
    memory_matrix = build_memory_matrix(
        same_day_pairs,
        test_days=test_days,
    )

    # 6. 为当前公告日补回正确的事件交易日。
    memory_matrix = add_event_date_columns(
        memory_matrix,
        base_events,
    )

    # 7. 保存前检查最终字段和主键。
    validate_final_memory_matrix(
        memory_matrix
    )

    # 8. 保存结果。
    output_path = paths["output_csv"]
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("正在保存记忆矩阵...")

    memory_matrix.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("保存成功！")
    print(output_path)
    print("最终表形状：", memory_matrix.shape)



    print("列名：")
    for idx, col in enumerate(
        memory_matrix.columns,
        start=1,
    ):
        print(idx, col)

    return memory_matrix



def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""
    parser = argparse.ArgumentParser(
        description="构造 Pattern Firms 记忆矩阵"
    )

    parser.add_argument(
        "--root",
        default=None,
        help=(
            "项目根目录。默认读取环境变量 MMM_ROOT；"
            "如果没有，则从当前目录向上寻找。"
        ),
    )

    parser.add_argument(
        "--output-csv",
        default=None,
        help="自定义输出文件路径。",
    )

    parser.add_argument(
        "--test-days",
        type=int,
        default=None,
        help=(
            "只计算前 N 个公告日，用于测试；"
            "正式运行时不设置。"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """脚本主入口。"""
    args = parse_args()

    run_pipeline(
        root=args.root,
        output_csv=args.output_csv,
        test_days=args.test_days,
    )


if __name__ == "__main__":
    main()

# 最终预计 100 个变量：
# 5 个日期及公司字段
# + 5 个 quarterLag 变量
# + 56 个普通 lag 强关联变量
#   其中包括原文的 52 个变量和新增的 4 个 anyTopMover 变量
# + 24 个极端正向/负向 SUE 变量
# + 5 个 nrFirmsAnnLag 变量
# + 5 个 daysSinceOverlapLag 变量
# = 100 个变量
