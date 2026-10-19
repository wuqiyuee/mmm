# 本文件由旧零散回归目录迁移；仅调整项目定位、公共工具引用和任务输出位置。
"""在现有模式公司最终面板上，检验公告日距离口径的记忆远近。

本脚本为独立扩展；直接运行仅在终端打印结果，不覆盖面板或既有回归结果。
仅显式指定 --save-dir（保存目录）时，才在该目录下新建带时间戳的结果子目录。

一、关系与年龄
    来源是“MemoryMatrixPattern_公告日距离版.csv”（公告日距离版记忆矩阵）。
    上游的第 1—4 次表示历史共同公告由近到远的次序，并非固定天数区间。
    本脚本读取最近一次共同公告距当前公告的自然日数，默认划分为
    1—90、91—180、181—270、271—364 天；可用 --age-cuts（分组上限）调整。
    对同一被提示公司、同一事件交易日，先筛选合格的大市值提示公司，
    再取全部合格关系中的最小年龄；各组因此互斥。
    系数描述“当日最近期合格提示关系”的年龄异质性，并非旧记忆的独立效应。

二、样本与筛选
    默认 2005—2025 年，因变量为两日、十一日缩尾后的特征调整收益。
    使用已生成的最终面板，它已对两种收益和控制变量筛样本、缩尾；不重新缩尾。
    大市值使用事件前第三个交易日总市值与事件上月末 90 分位断点比较。
    延续原合并脚本的行业、一级供应链和个人分析师共同覆盖排除规则。
    分析师覆盖在新关系全集上重新识别，不能直接套用旧矩阵的正记录文件。
    公告日已经考虑收盘后披露；仅非交易日映射至下一交易日。

三、回归
    各年龄组提示虚拟变量同时入模；参照组为未匹配到合格提示关系的公司日。
    使用原主回归的 27 个控制变量，吸收日期及被提示公司×年度季度固定效应，
    递归删除固定效应单例组，并按被提示公司、日期双向聚类，启用小样本调整。
    输出组间系数相等的联合沃尔德检验及两两比较；后者另给出霍尔姆校正。

四、解释边界
    距离版同日任一报告满足模式即可保留，与原财报期滞后版事件口径有差异。
    因此本检验是“公告日距离口径扩展”，不是只给原提示变量增加年龄分组。
    零值沿用关系表未匹配填零的定义，不能据此断言原始历史完整且确实无关系。
    已存在关系的年龄缺失或非法时直接报错，绝不把它当作无提示填零。
    两窗口联合检验与单个系数的显著性不作跨窗口多重检验校正；
    两两比较的霍尔姆校正仅在各收益窗口内部进行。
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from scipy.stats import chi2


def find_project_root() -> Path:
    """沿脚本祖先寻找项目标识，迁移目录后不依赖固定父目录层数。"""
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "AGENTS.md").is_file()
            and (candidate / "code").is_dir()
            and (candidate / "中间结果").is_dir()
        ):
            return candidate
    raise FileNotFoundError("未找到同时包含项目说明、代码和中间结果的项目根目录。")


ROOT = find_project_root()
TASK_OUTPUT_DIR = ROOT / "图表结果/02_扩展分析/其他扩展检验/记忆远近检验"
MAX_AGE = 364
KEY = ["Annodt", "Stkcd", "StkcdRecall"]
DAY_KEY = ["StkcdRecall", "Trddt"]
OUTCOMES = {
    "chAdjreturn_t_tp1_Recall_w": "两日收益",
    "chAdjreturn_t_tp10_Recall_w": "十一日收益",
}
CONTROLS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
    + ["ownMajorDisclosure", "ownPerformanceForecast"]
    + ["aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV"]
)
RELATIVE_PATHS = {
    "matrix": "中间结果/03_记忆矩阵/MemoryMatrixPattern_公告日距离版.csv",
    "panel": "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
    "calendar": "中间结果/01_清洗后样本/市场交易日历_td.csv",
    "size": "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv",
    "breakpoints": "中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv",
    "industry": "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv",
    "supply": "input/CSMAR/供应链网络关系指标表2001-12-31至2025-12-31/SC_NetworkRelationsIndex.csv",
    "forecast": "中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv",
    "analyst_helper": (
        "code/03_记忆矩阵&心理机制变量分组/识别分析师共同覆盖/"
        "识别分析师共同覆盖（日度Annodt合并检验）.py"
    ),
}


def stock_codes(values: pd.Series) -> pd.Series:
    """保留缺失，把其余股票代码统一为六位；不把缺失变成虚构代码。"""
    result = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    result = result.mask(result.eq(""))
    invalid = result.notna() & ~result.str.fullmatch(r"[0-9]{1,6}", na=False)
    if invalid.any():
        raise ValueError(f"存在非法股票代码：{result.loc[invalid].head().tolist()}")
    return result.str.zfill(6)


def check_keys(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{label}存在缺失键或重复键：{keys}")


def record(diagnostics: list[dict], item: str, value: int) -> None:
    diagnostics.append({"项目": item, "数量": int(value)})
    print(f"{item}：{int(value):,}", flush=True)


def read_matrix(paths: dict, start: int, end: int, diagnostics: list[dict]) -> pd.DataFrame:
    columns = KEY + [
        "event_Trddt", "event_td", "closestMemoryAgeDays",
        "hasOverlapLag1", "daysSinceOverlapAnnLag1", "nHistoricalOverlapDates",
    ]
    matrix = pd.read_csv(
        paths["matrix"], usecols=columns, encoding="utf-8-sig",
        dtype={"Stkcd": "string", "StkcdRecall": "string"},
    )
    for column in ["Annodt", "event_Trddt"]:
        matrix[column] = pd.to_datetime(matrix[column], errors="raise").dt.normalize()
        if matrix[column].isna().any():
            raise ValueError(f"关系矩阵的 {column}（公告或事件日期）存在缺失。")
    matrix = matrix.loc[matrix["event_Trddt"].dt.year.between(start, end)].copy()
    if matrix.empty:
        raise ValueError("指定年份没有公告日距离版关系。")
    for column in ["Stkcd", "StkcdRecall"]:
        matrix[column] = stock_codes(matrix[column])
    check_keys(matrix, KEY, "公告日距离版关系矩阵")
    if matrix["Stkcd"].eq(matrix["StkcdRecall"]).any():
        raise ValueError("关系矩阵包含公司自身与自身的配对。")
    numeric = columns[4:]
    for column in numeric:
        matrix[column] = pd.to_numeric(matrix[column], errors="raise")
        values = matrix[column].to_numpy(dtype="float64")
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"关系矩阵的 {column}（交易日序号、年龄或次数字段）不是完整整数。")
    age = matrix["closestMemoryAgeDays"]
    if (
        not age.between(1, MAX_AGE).all()
        or not age.eq(matrix["daysSinceOverlapAnnLag1"]).all()
        or not matrix["hasOverlapLag1"].eq(1).all()
        or not matrix["nHistoricalOverlapDates"].ge(1).all()
    ):
        raise ValueError("最近记忆年龄、第一次共同公告和历史次数不一致。")

    calendar = pd.read_csv(paths["calendar"], usecols=["Trddt", "td"])
    calendar["Trddt"] = pd.to_datetime(calendar["Trddt"], errors="raise")
    calendar["td"] = pd.to_numeric(calendar["td"], errors="raise")
    check_keys(calendar, ["Trddt"], "市场交易日历")
    check_keys(calendar, ["td"], "市场交易日历序号")
    calendar = calendar.sort_values("Trddt").reset_index(drop=True)
    positions = calendar["Trddt"].searchsorted(matrix["Annodt"], side="left")
    if (positions == len(calendar)).any():
        raise ValueError("市场交易日历不能覆盖所有公告日。")
    expected = calendar.iloc[positions]
    if (
        not np.array_equal(expected["Trddt"].to_numpy(), matrix["event_Trddt"].to_numpy())
        or not np.array_equal(expected["td"].to_numpy(), matrix["event_td"].to_numpy())
    ):
        raise ValueError("矩阵事件日不符合交易日保留、非交易日向后映射的口径。")
    matrix["event_td"] = matrix["event_td"].astype("int64")
    matrix["closestMemoryAgeDays"] = age.astype("int16")
    record(diagnostics, "年份范围内原始关系数", len(matrix))
    return matrix[KEY + ["event_Trddt", "event_td", "closestMemoryAgeDays"]]


def read_needed_daily(
    path: Path, needed: pd.DataFrame, value_columns: list[str], chunksize: int,
    *, industry: bool = False,
) -> pd.DataFrame:
    """分块读取所需公司和交易日；行业重复取首条，与主合并脚本一致。"""
    keys = ["Stkcd", "td"]
    needed_index = pd.MultiIndex.from_frame(needed[keys].drop_duplicates())
    parts = []
    dtype = {"Stkcd": "string"}
    if industry:
        dtype["IndustryCode"] = "string"
    with pd.read_csv(path, usecols=keys + value_columns, dtype=dtype, chunksize=chunksize) as reader:
        for block in reader:
            block["Stkcd"] = stock_codes(block["Stkcd"])
            block["td"] = pd.to_numeric(block["td"], errors="raise")
            block = block.dropna(subset=keys + (value_columns if industry else []))
            keep = pd.MultiIndex.from_frame(block[keys]).isin(needed_index)
            if keep.any():
                parts.append(block.loc[keep].copy())
    if not parts:
        if industry:
            # 没有匹配行业时按原合并口径保留关系，并在下一步统计行业缺失。
            return pd.DataFrame({
                "Stkcd": pd.Series(dtype="string"), "td": pd.Series(dtype="int64"),
                **{column: pd.Series(dtype="string") for column in value_columns},
            })
        raise ValueError(f"日度输入未匹配到任何所需公司日期：{path}")
    result = pd.concat(parts, ignore_index=True)
    if industry:
        result = result.drop_duplicates(keys, keep="first")
    check_keys(result, keys, "日度匹配数据")
    return result


def filter_large_cues(matrix: pd.DataFrame, paths: dict, chunksize: int, diagnostics: list[dict]) -> pd.DataFrame:
    pairs = matrix.copy()
    pairs["td_tm3"] = pairs["event_td"] - 3
    needed = pairs[["Stkcd", "td_tm3"]].rename(columns={"td_tm3": "td"})
    size = read_needed_daily(paths["size"], needed, ["size"], chunksize)
    size = size.rename(columns={"td": "td_tm3", "size": "mktcap_tm3"})
    pairs = pairs.merge(size, on=["Stkcd", "td_tm3"], how="left", validate="many_to_one")
    bp_columns = [f"marketCap_{n}th_percentile" for n in [90, 50, 10]]
    breakpoints = pd.read_csv(paths["breakpoints"], usecols=["Trdmnt"] + bp_columns)
    breakpoints["lookup_month"] = pd.to_datetime(breakpoints["Trdmnt"], errors="raise").dt.to_period("M")
    check_keys(breakpoints, ["lookup_month"], "月度市值断点")
    pairs["lookup_month"] = pairs["event_Trddt"].dt.to_period("M") - 1
    pairs = pairs.merge(
        breakpoints[["lookup_month"] + bp_columns], on="lookup_month",
        how="left", validate="many_to_one",
    )
    required = ["mktcap_tm3"] + bp_columns
    for column in required:
        pairs[column] = pd.to_numeric(pairs[column], errors="raise")
    complete = np.isfinite(pairs[required].to_numpy(dtype="float64")).all(axis=1)
    record(diagnostics, "市值或必要断点缺失或非有限值的关系数", (~complete).sum())
    pairs = pairs.loc[complete].copy()
    large = pairs["mktcap_tm3"].gt(pairs["marketCap_90th_percentile"])
    record(diagnostics, "剔除非大市值提示关系数", (~large).sum())
    pairs = pairs.loc[large, matrix.columns].copy()
    if pairs.empty:
        raise ValueError("没有满足大市值条件的提示关系。")
    return pairs


def filter_industry_supply(pairs: pd.DataFrame, paths: dict, chunksize: int, diagnostics: list[dict]) -> pd.DataFrame:
    needed = pd.concat([
        pairs[["Stkcd", "event_td"]].rename(columns={"event_td": "td"}),
        pairs[["StkcdRecall", "event_td"]].rename(columns={"StkcdRecall": "Stkcd", "event_td": "td"}),
    ], ignore_index=True).drop_duplicates()
    industry = read_needed_daily(paths["industry"], needed, ["IndustryCode"], chunksize, industry=True)
    original_columns = pairs.columns.tolist()
    pairs = pairs.merge(
        industry.rename(columns={"td": "event_td", "IndustryCode": "cue_industry"}),
        on=["Stkcd", "event_td"], how="left", validate="many_to_one",
    ).merge(
        industry.rename(columns={"Stkcd": "StkcdRecall", "td": "event_td", "IndustryCode": "recall_industry"}),
        on=["StkcdRecall", "event_td"], how="left", validate="many_to_one",
    )
    known = pairs[["cue_industry", "recall_industry"]].notna().all(axis=1)
    same = known & pairs["cue_industry"].eq(pairs["recall_industry"]).fillna(False)
    record(diagnostics, "行业缺失但按原口径保留的关系数", (~known).sum())
    record(diagnostics, "剔除同行业关系数", same.sum())
    pairs = pairs.loc[~same, original_columns].copy()

    supply_columns = ["Symbol", "EndDate", "StateTypeCode", "PSCBusinessRelation", "PSCSymbol"]
    supply = pd.read_csv(paths["supply"], usecols=supply_columns, dtype="string")
    keep = (
        supply["StateTypeCode"].eq("1")
        & supply["PSCBusinessRelation"].isin(["1", "2"])
        & supply["PSCSymbol"].str.fullmatch(r"[0368][0-9]{5}", na=False)
    )
    supply = supply.loc[keep].copy()
    supply["yearCue"] = pd.to_datetime(supply["EndDate"], errors="raise").dt.year
    supply["Stkcd"] = stock_codes(supply["Symbol"])
    supply["StkcdRecall"] = stock_codes(supply["PSCSymbol"])
    supply = supply.dropna(subset=["yearCue", "Stkcd", "StkcdRecall"])
    supply = supply.loc[supply["Stkcd"].ne(supply["StkcdRecall"]), ["yearCue", "Stkcd", "StkcdRecall"]]
    reverse = supply.rename(columns={"Stkcd": "StkcdRecall", "StkcdRecall": "Stkcd"})
    symmetric = pd.concat([supply, reverse], ignore_index=True).drop_duplicates().assign(_vertical=1)
    pairs["yearCue"] = pairs["Annodt"].dt.year
    pairs = pairs.merge(symmetric, on=["yearCue", "Stkcd", "StkcdRecall"], how="left", validate="many_to_one")
    vertical = pairs["_vertical"].eq(1)
    record(diagnostics, "剔除一级供应链关系数", vertical.sum())
    return pairs.loc[~vertical, original_columns].copy()


def filter_analysts(pairs: pd.DataFrame, paths: dict, diagnostics: list[dict]) -> pd.DataFrame:
    """在本次全部候选关系上重算覆盖；按公告月分批，限制缓存内存。"""
    if pairs.empty:
        raise ValueError("行业和供应链排除后没有候选提示关系。")
    spec = importlib.util.spec_from_file_location("memory_age_analyst_helper", paths["analyst_helper"])
    if spec is None or spec.loader is None:
        raise ImportError("无法加载分析师共同覆盖辅助函数。")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    print("按新关系重新识别公告日前 365 天的个人分析师共同覆盖……", flush=True)
    forecast = helper.read_analyst_forecast(paths["forecast"])
    stocks = pd.Index(pd.concat([pairs["Stkcd"], pairs["StkcdRecall"]]).unique())
    start = pairs["Annodt"].min() - pd.Timedelta(days=365)
    end = pairs["Annodt"].max()
    forecast = forecast.loc[
        forecast["Stkcd"].isin(stocks) & forecast["anndats"].ge(start) & forecast["anndats"].lt(end)
    ].copy()
    coverage = helper.build_split_coverage(forecast)
    coverage_index = helper.build_fast_coverage_index(coverage)
    record(diagnostics, "候选关系中未在覆盖索引出现的公司数", (~stocks.isin(list(coverage_index))).sum())
    del forecast, coverage
    gc.collect()
    positive = []
    for month, block in pairs.groupby(pairs["Annodt"].dt.to_period("M"), sort=True):
        print(f"识别分析师共同覆盖：{month}，{len(block):,} 条关系", flush=True)
        overlap = helper.identify_overlap_by_annodt_daily_sets(
            block[KEY], coverage_index=coverage_index, lookback_days=365,
            id_mode="split", show_progress=False,
        )
        if not overlap.empty:
            positive.append(overlap[KEY])
    if positive:
        excluded = pd.concat(positive, ignore_index=True).drop_duplicates().assign(_overlap=1)
        joined = pairs.merge(excluded, on=KEY, how="left", validate="one_to_one")
        keep = joined["_overlap"].isna()
        result = joined.loc[keep, pairs.columns].copy()
    else:
        result = pairs.copy()
    record(diagnostics, "剔除个人分析师共同覆盖关系数", len(pairs) - len(result))
    record(diagnostics, "最终合格提示关系数", len(result))
    if result.empty:
        raise ValueError("没有剩余的合格提示关系。")
    return result


def make_daily_cues(pairs: pd.DataFrame, cuts: list[int], diagnostics: list[dict]) -> pd.DataFrame:
    """先按关系年龄分组，再取公司日的最近年龄；另保留跨组提示的诊断。"""
    pairs = pairs.rename(columns={"event_Trddt": "Trddt"}).copy()
    pairs["_pair_age_group"] = pd.cut(
        pairs["closestMemoryAgeDays"], bins=[0] + cuts, labels=False, right=True,
    ) + 1
    daily = pairs.groupby(DAY_KEY, as_index=False, sort=False).agg(
        nearest_age=("closestMemoryAgeDays", "min"),
        n_cue_firms=("Stkcd", "nunique"),
        n_age_groups=("_pair_age_group", "nunique"),
    )
    daily["age_group"] = (
        pd.cut(daily["nearest_age"], bins=[0] + cuts, labels=False, right=True) + 1
    ).astype("int16")
    check_keys(daily, DAY_KEY, "公司日提示表")
    record(diagnostics, "存在合格提示的公司日数（合并最终面板前）", len(daily))
    record(diagnostics, "同日合格提示涉及多个年龄组的公司日数（合并前）", daily["n_age_groups"].gt(1).sum())
    return daily


def read_panel(path: Path, start: int, end: int, chunksize: int) -> pd.DataFrame:
    print("分块读取现有最终模式面板的收益和控制变量……", flush=True)
    numeric = list(OUTCOMES) + CONTROLS + ["quarterLag1size90plus"]
    dtype = {column: "float32" for column in numeric}
    dtype.update({"StkcdRecall": "string", "yearQuarter": "string"})
    columns = DAY_KEY + ["yearQuarter"] + numeric
    parts = []
    with pd.read_csv(path, usecols=columns, dtype=dtype, chunksize=chunksize) as reader:
        for number, block in enumerate(reader, 1):
            block["Trddt"] = pd.to_datetime(block["Trddt"], errors="raise")
            if block["Trddt"].isna().any():
                raise ValueError("最终面板存在缺失交易日期。")
            keep = block["Trddt"].dt.year.between(start, end)
            if keep.any():
                parts.append(block.loc[keep].copy())
            if number % 10 == 0:
                print(f"已读取 {number} 个数据块。", flush=True)
    if not parts:
        raise ValueError("最终面板在指定年份范围内没有观测。")
    panel = pd.concat(parts, ignore_index=True)
    del parts
    panel["StkcdRecall"] = stock_codes(panel["StkcdRecall"])
    check_keys(panel, DAY_KEY, "最终模式面板")
    if panel["yearQuarter"].isna().any():
        raise ValueError("最终模式面板缺少年度季度固定效应字段。")
    for column in ["StkcdRecall", "yearQuarter"]:
        panel[column] = panel[column].astype("category")
    return panel


def attach_groups(panel: pd.DataFrame, daily: pd.DataFrame, diagnostics: list[dict]) -> pd.DataFrame:
    """只把未匹配到关系的年龄组编码填零，年龄本身仍保持缺失。"""
    # 按索引对齐，避免合并时复制整张宽面板。
    target = pd.MultiIndex.from_frame(panel[DAY_KEY])
    aligned = daily.set_index(DAY_KEY).reindex(target)
    panel["age_group"] = aligned["age_group"].fillna(0).to_numpy(dtype="int16")
    record(diagnostics, "年份范围内最终面板公司日数", len(panel))
    record(diagnostics, "面板中存在新口径提示的公司日数", panel["age_group"].gt(0).sum())
    record(diagnostics, "合格提示公司日未进入现有最终面板的数量", len(daily) - panel["age_group"].gt(0).sum())
    record(diagnostics, "面板中同日涉及多个年龄组的提示公司日数", aligned["n_age_groups"].gt(1).sum())
    old = panel["quarterLag1size90plus"]
    if not old.isin([0, 1]).all():
        raise ValueError("最终面板的旧提示标记不是完整的零一变量。")
    new = panel["age_group"].gt(0)
    for old_value, old_label in [(0, "无"), (1, "有")]:
        for new_value, new_label in [(False, "无"), (True, "有")]:
            record(
                diagnostics, f"旧口径{old_label}提示、新口径{new_label}提示的公司日数",
                (old.eq(old_value) & new.eq(new_value)).sum(),
            )
    return panel.drop(columns=["quarterLag1size90plus"])


def drop_singletons(reg: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(reg)
    while len(reg):
        singleton = np.zeros(len(reg), dtype=bool)
        for column in ["_firm_quarter", "_date"]:
            codes = reg[column].to_numpy(dtype="int64")
            singleton |= np.bincount(codes)[codes] == 1
        if not singleton.any():
            break
        reg = reg.loc[~singleton].copy()
    return reg, before - len(reg)


def wald_test(beta: np.ndarray, covariance: np.ndarray, restriction: np.ndarray) -> dict:
    """采用完整协方差计算沃尔德卡方检验；协方差异常时不伪造显著性。"""
    difference = restriction @ beta
    variance = restriction @ covariance @ restriction.T
    variance = (variance + variance.T) / 2
    dimension = len(difference)
    empty = {"沃尔德统计量": np.nan, "自由度": dimension, "p值": np.nan}
    if not np.isfinite(variance).all() or not np.isfinite(difference).all():
        return {**empty, "检验状态": "系数或受限协方差含非有限值，未检验"}
    eigenvalues = np.linalg.eigvalsh(variance)
    tolerance = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny) * 1e-10
    if float(eigenvalues.min()) <= tolerance:
        return {**empty, "检验状态": "受限协方差不正定或秩不足，未检验"}
    statistic = float(difference @ np.linalg.solve(variance, difference))
    return {
        "沃尔德统计量": statistic, "自由度": dimension,
        "p值": float(chi2.sf(statistic, dimension)), "检验状态": "已检验",
    }


def holm_adjust(pvalues: list[float]) -> np.ndarray:
    """校正一个收益窗口内的所有两两比较；存在无法检验项则不校正。"""
    values = np.asarray(pvalues, dtype=float)
    if not np.isfinite(values).all():
        return np.full(len(values), np.nan)
    order = np.argsort(values)
    corrected = np.minimum(1.0, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty_like(values)
    result[order] = corrected
    return result


def fit_outcome(panel: pd.DataFrame, outcome: str, labels: list[str]) -> dict[str, list[dict]]:
    title = OUTCOMES[outcome]
    print(f"开始估计{title}：各年龄组同时入模。", flush=True)
    needed = DAY_KEY + ["yearQuarter", "age_group", outcome] + CONTROLS
    reg = panel[needed].dropna().copy()
    numeric = [outcome] + CONTROLS
    # 列级处理无穷值，避免额外构建整张双精度矩阵。
    finite = np.ones(len(reg), dtype=bool)
    for column in numeric:
        finite &= np.isfinite(reg[column].to_numpy())
    reg = reg.loc[finite].copy()
    complete_count = len(reg)
    reg["_firm_quarter"] = reg.groupby(
        ["StkcdRecall", "yearQuarter"], observed=True, sort=False,
    ).ngroup().astype("int64")
    reg["_date"] = pd.factorize(reg["Trddt"], sort=False)[0].astype("int64")
    reg, singleton_count = drop_singletons(reg)
    if reg.empty:
        raise ValueError(f"{title}的有效回归样本为空。")
    counts = reg["age_group"].value_counts()
    print("回归有效样本的各组公司日数：", flush=True)
    for group in range(len(labels) + 1):
        name = "无合格提示参照组" if group == 0 else labels[group - 1]
        print(f"  {name}：{int(counts.get(group, 0)):,}", flush=True)
    for group in range(len(labels) + 1):
        if int(counts.get(group, 0)) == 0:
            name = "无合格提示参照组" if group == 0 else labels[group - 1]
            raise ValueError(f"{title}中{name}没有有效观测；请依据研究设计调整分组上限。")

    cue_columns = [f"memory_age_{i}" for i in range(1, len(labels) + 1)]
    for number, column in enumerate(cue_columns, 1):
        reg[column] = reg["age_group"].eq(number).astype("int8")
    x_columns = cue_columns + CONTROLS
    regressors = reg[x_columns].astype("float64")
    regressors.insert(0, "const", 1.0)
    absorb = pd.DataFrame({
        "firm_quarter": pd.Categorical(reg["_firm_quarter"]),
        "date": pd.Categorical(reg["_date"]),
    }, index=reg.index)
    clusters = pd.DataFrame({
        "firm": pd.factorize(reg["StkcdRecall"], sort=False)[0],
        "date": pd.factorize(reg["Trddt"], sort=False)[0],
    }, index=reg.index)
    model = AbsorbingLS(
        dependent=reg[outcome].astype("float64"), exog=regressors,
        absorb=absorb, drop_absorbed=True,
    )
    result = model.fit(cov_type="clustered", clusters=clusters, debiased=True, use_cache=False)
    missing = [column for column in cue_columns if column not in result.params.index]
    if missing:
        raise ValueError(f"{title}的部分年龄组被固定效应吸收，不能比较：{missing}")
    beta = result.params.loc[cue_columns].to_numpy(dtype=float)
    covariance = result.cov.loc[cue_columns, cue_columns].to_numpy(dtype=float)
    coefficient_rows = []
    for column in x_columns:
        estimated = column in result.params.index
        group_label = labels[cue_columns.index(column)] if column in cue_columns else "控制变量"
        coefficient_rows.append({
            "收益窗口": title, "变量": column, "年龄组或变量类型": group_label,
            "系数（百分点）": float(result.params[column]) * 100 if estimated else np.nan,
            "聚类标准误（百分点）": float(result.std_errors[column]) * 100 if estimated else np.nan,
            "t值": float(result.tstats[column]) if estimated else np.nan,
            "p值": float(result.pvalues[column]) if estimated else np.nan,
            "估计状态": "已估计" if estimated else "被固定效应吸收",
        })
    # K 组全部相等只需要 K−1 个独立约束，不能把全部两两比较当成独立约束。
    equality = np.zeros((len(labels) - 1, len(labels)))
    equality[:, 0] = 1.0
    for row in range(len(labels) - 1):
        equality[row, row + 1] = -1.0
    joint_rows = [
        {"收益窗口": title, "原假设": "各年龄组系数全部相等", **wald_test(beta, covariance, equality)},
        {"收益窗口": title, "原假设": "各年龄组系数全部为零", **wald_test(beta, covariance, np.eye(len(labels)))},
    ]
    pair_rows = []
    for left, right in combinations(range(len(labels)), 2):
        restriction = np.zeros((1, len(labels)))
        restriction[0, left], restriction[0, right] = 1.0, -1.0
        variance = float((restriction @ covariance @ restriction.T)[0, 0])
        pair_rows.append({
            "收益窗口": title, "比较": f"{labels[left]}减{labels[right]}",
            "系数差（百分点）": float(beta[left] - beta[right]) * 100,
            "系数差标准误（百分点）": np.sqrt(variance) * 100 if variance > 0 else np.nan,
            **wald_test(beta, covariance, restriction),
        })
    adjusted = holm_adjust([row["p值"] for row in pair_rows])
    for row, value in zip(pair_rows, adjusted):
        row["霍尔姆校正p值"] = float(value)
    summary = {
        "收益窗口": title, "回归样本量": int(result.nobs),
        "缺失或非有限值剔除数": len(panel) - complete_count,
        "递归单例组剔除数": singleton_count,
        "被提示公司数": int(reg["StkcdRecall"].nunique()),
        "交易日期数": int(reg["Trddt"].nunique()), "R方": float(result.rsquared),
        "固定效应": "日期及被提示公司×年度季度", "聚类": "被提示公司及日期，含小样本调整",
    }
    group_rows = [
        {
            "收益窗口": title, "年龄组": "无合格提示参照组" if number == 0 else labels[number - 1],
            "公司日数": int(counts.get(number, 0)), "占回归样本比例": int(counts.get(number, 0)) / len(reg),
        }
        for number in range(len(labels) + 1)
    ]
    tables = {
        "模型汇总": [summary], "系数明细": coefficient_rows,
        "组间联合检验": joint_rows, "两两比较": pair_rows, "分组样本量": group_rows,
    }
    print(pd.DataFrame(coefficient_rows[:len(labels)]).to_string(index=False), flush=True)
    print(pd.DataFrame(joint_rows).to_string(index=False), flush=True)
    del result, model, regressors, absorb, clusters, reg
    gc.collect()
    return tables


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在现有模式面板上检验公告日距离口径的记忆远近。")
    parser.add_argument("--root", type=Path, default=ROOT, help="项目根目录")
    parser.add_argument("--start-year", type=int, default=2005, help="回归起始年份，包含该年")
    parser.add_argument("--end-year", type=int, default=2025, help="回归结束年份，包含该年；目前不超过2025")
    parser.add_argument("--age-cuts", type=int, nargs="+", default=[90, 180, 270, 364], help="各组自然日上限，至少两组，末项必须为364")
    parser.add_argument("--chunksize", type=int, default=250_000, help="大文件每次读取的行数")
    parser.add_argument("--save-dir", type=Path, nargs="?", const=TASK_OUTPUT_DIR, default=None, help="显式保存开关；不带目录时保存到本任务目录/时间戳，也可指定保存目录")
    args = parser.parse_args()
    if args.start_year > args.end_year or args.end_year > 2025:
        parser.error("起始年不能晚于结束年；现有分析师及供应链输入口径只支持截至2025年。")
    cuts = args.age_cuts
    if len(cuts) < 2 or cuts[0] < 1 or cuts[-1] != MAX_AGE or any(a >= b for a, b in zip(cuts, cuts[1:])):
        parser.error("分组上限应为严格递增的正整数，至少两项，最后一项为364。")
    if args.chunksize < 1:
        parser.error("分块行数必须为正数。")
    return args


def main() -> None:
    args = parse_arguments()
    paths = {key: args.root.expanduser().resolve() / value for key, value in RELATIVE_PATHS.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"缺少必要输入：{path}")
    provenance = {
        key: {"路径": str(path), "字节数": path.stat().st_size, "修改时间纳秒": path.stat().st_mtime_ns}
        for key, path in paths.items()
    }
    labels = [f"{lower + 1}—{upper}天" for lower, upper in zip([0] + args.age_cuts[:-1], args.age_cuts)]
    print(f"样本年份：{args.start_year}—{args.end_year}；年龄分组：{'、'.join(labels)}", flush=True)
    print("年龄取当日所有合格关系的最小值；使用现有最终模式面板的基准样本和缩尾收益。", flush=True)
    print("未匹配关系填零只表示表内未识别到合格提示，不能据此证明历史原始数据完整。", flush=True)
    diagnostics: list[dict] = []
    matrix = read_matrix(paths, args.start_year, args.end_year, diagnostics)
    pairs = filter_large_cues(matrix, paths, args.chunksize, diagnostics)
    del matrix
    pairs = filter_industry_supply(pairs, paths, args.chunksize, diagnostics)
    pairs = filter_analysts(pairs, paths, diagnostics)
    daily = make_daily_cues(pairs, args.age_cuts, diagnostics)
    del pairs
    gc.collect()
    panel = read_panel(paths["panel"], args.start_year, args.end_year, args.chunksize)
    panel = attach_groups(panel, daily, diagnostics)
    del daily
    tables: dict[str, list[dict]] = {"筛选诊断": diagnostics}
    for outcome in OUTCOMES:
        for name, rows in fit_outcome(panel, outcome, labels).items():
            tables.setdefault(name, []).extend(rows)
    del panel
    for name in ["模型汇总", "分组样本量", "两两比较"]:
        print(f"\n{name}\n{pd.DataFrame(tables[name]).to_string(index=False)}", flush=True)
    print("\n判断远近是否不同，应看组间相等检验，不能仅比较一组显著、另一组不显著。", flush=True)
    print("两两比较给出各窗口内部的霍尔姆校正；单组系数和两个窗口的联合检验未经跨窗口校正。", flush=True)
    if args.save_dir is not None:
        # 显式选择保存时才创建文件；每次单独建目录，禁止覆盖既有结果。
        for key, path in paths.items():
            current = path.stat()
            if current.st_size != provenance[key]["字节数"] or current.st_mtime_ns != provenance[key]["修改时间纳秒"]:
                raise RuntimeError(f"运行期间输入文件发生变化，未保存结果：{path}")
        destination = args.save_dir.expanduser().resolve() / f"记忆远近_{datetime.now():%Y%m%d_%H%M%S_%f}"
        destination.mkdir(parents=True, exist_ok=False)
        for name, rows in tables.items():
            pd.DataFrame(rows).to_csv(destination / f"{name}.csv", index=False, encoding="utf-8-sig")
        metadata = {
            "年份": [args.start_year, args.end_year], "年龄组": labels,
            "归组规则": "同一被提示公司事件日全部合格关系的最近共同公告自然日年龄",
            "样本边界": "沿用现有最终模式面板；属于公告日距离口径扩展",
            "零值解释": "未匹配到合格关系；不保证历史原始数据充分",
            "输入": provenance,
        }
        (destination / "运行口径.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"独立结果已保存至：{destination}", flush=True)
    else:
        print("\n本次仅打印结果，未保存或覆盖任何结果文件。", flush=True)


if __name__ == "__main__":
    main()
