import pandas as pd
import numpy as np
import gc
from dateutil.relativedelta import * 
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    # 进度条是可选依赖；缺少时直接迭代，不影响分块计算。
    def tqdm(iterable, **_kwargs):
        return iterable
import datetime 

#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)


#===============================================================================
## 工具函数
#===============================================================================

def normalize_stock_code(x: object) -> str:
    """标准化股票代码，填充为6位数，并转换为字符串"""
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s == '':
        return ''
    return f"{int(float(s)):06d}"


# =============================================================================
## 路径设置
# =============================================================================

def build_full_paths(root):
    """根据项目根目录，构造全样本脚本的输入、输出路径。"""
    root = Path(root)

    relative_paths = {
        "MemoryMatrixFull_path": (
            "中间结果/03_记忆矩阵/"
            "MemoryMatrixFull(封装优化版).csv"
        ),
        "TRD_Cale_td_path": (
            "中间结果/01_清洗后样本/市场交易日历_td.csv"
        ),
        "TRD_Dalyr_DGTW_path": (
            "中间结果/01_清洗后样本/"
            "DGTW构建用个股日度面板.csv"
        ),
        "marketcap_group_path": (
            "中间结果/02_变量构造结果/"
            "8月度市值分组/月度市值分组.csv"
        ),
        "industry_path": (
            "中间结果/02_变量构造结果/同行业识别/"
            "个股行业代码完整面板_RESSET补缺.csv"
        ),
        "SC_NetWorkRelationsIndex_path": (
            "input/CSMAR/"
            "供应链网络关系指标表2001-12-31至2025-12-31/"
            "SC_NetworkRelationsIndex.csv"
        ),
        "IAR_Rept_path": (
            "input/CSMAR/"
            "年、中、季报基本情况文件1990-12-31至2026-06-30/"
            "IAR_Rept.csv"
        ),
        "AES_AEV_controls_path": (
            "中间结果/02_变量构造结果/3报刊新闻_AES_AEV/"
            "报刊新闻_AES_AEV_真实交易日控制变量.csv"
        ),
        "adjusted_return_path": (
            "中间结果/02_变量构造结果/1DGTW特征调整收益率/"
            "DGTW特征调整收益率.csv"
        ),
        "performance_forecast_controls_path": (
            "中间结果/02_变量构造结果/5公司公告事件/"
            "业绩预告快报控制变量_公告日期口径.csv"
        ),
        "other_event_controls_path": (
            "中间结果/02_变量构造结果/5公司公告事件/"
            "专题研究事件控制变量_公告日期口径.csv"
        ),
        "sample_intermediate_out_path": (
            "中间结果/04_合并数据/Sample_intermediate_full.csv"
        ),
        "cue_events_out_path": (
            "中间结果/04_合并数据/Full_cue_events_after_collapse.csv"
        ),
        "output_path": (
            "中间结果/04_合并数据/Full样本_最终回归面板.csv"
        ),
    }

    return {
        name: root / relative_path
        for name, relative_path in relative_paths.items()
    }


# =============================================================================
# 参数设置
# =============================================================================

def build_full_config(
    *,
    drop_analyst_overlap,
    analyst_id_mode,
    analyst_lookback_days,
    marketcap_breakpoint_mode,
    event_date_mode,
    mm_chunksize=1_000_000,
    reuse_existing_cue_events=False,
):
    """集中构造并校验全样本脚本的运行参数。"""

    if not isinstance(drop_analyst_overlap, bool):
        raise TypeError("共同分析师覆盖排除开关必须为布尔值。")

    if not isinstance(reuse_existing_cue_events, bool):
        raise TypeError("提示事件表复用开关必须为布尔值。")

    if analyst_id_mode not in {"split", "team"}:
        raise ValueError(
            "分析师身份口径只能是 "
            "'split'（个人）或 'team'（完整团队）。"
        )

    if marketcap_breakpoint_mode not in {
        "prev_month",
        "same_month",
    }:
        raise ValueError(
            "市值断点月份只能是 "
            "'prev_month'（上月）或 'same_month'（当月）。"
        )

    if event_date_mode not in {
        "next_if_nontrading",
        "strict_next_trading",
    }:
        raise ValueError(
            "事件日口径只能是 "
            "'next_if_nontrading'（仅非交易日顺延）或 "
            "'strict_next_trading'（严格使用下一交易日）。"
        )

    positive_integer_params = {
        "分析师回看窗口天数": analyst_lookback_days,
        "记忆矩阵分块行数": mm_chunksize,
    }

    for name, value in positive_integer_params.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name}必须为正整数。")

    return {
        "DROP_ANALYST_OVERLAP": drop_analyst_overlap,
        "ANALYST_ID_MODE": analyst_id_mode,
        "ANALYST_LOOKBACK_DAYS": analyst_lookback_days,
        "MARKETCAP_BREAKPOINT_MODE": marketcap_breakpoint_mode,
        "EVENT_DATE_MODE": event_date_mode,
        "MM_CHUNKSIZE": mm_chunksize,
        "REUSE_EXISTING_CUE_EVENTS": reuse_existing_cue_events,
    }


#===============================================================================
# 读取记忆矩阵：分块筛选并 collapse 到 被提示公司 × 日期
#===============================================================================
def combine_cue_parts(parts):
    if not parts:
        return pd.DataFrame(
            columns=["StkcdRecall", "event_Trddt", "quarterLag1size90plus"]
        )

    return (
        pd.concat(parts, ignore_index=True)
        .groupby(["StkcdRecall", "event_Trddt"], as_index=False)["quarterLag1size90plus"]
        .max()
    )


def load_full_calendar(calendar_path):
    """读取市场交易日历，按日期排序并整理交易日序号。"""

    calendar = pd.read_csv(
        calendar_path,
        encoding="utf-8",
        usecols=["Trddt", "td"],
        parse_dates=["Trddt"],
    ).sort_values("Trddt")

    calendar["td"] = pd.to_numeric(
        calendar["td"],
        errors="coerce",
    )

    return calendar


def load_full_marketcap_index(daily_panel_path):
    """读取日度总市值，建立后续查询事件前三交易日市值的索引。"""

    mktcap = pd.read_csv(
        daily_panel_path,
        encoding="utf-8",
        usecols=["Stkcd", "td", "size"],
        dtype={"Stkcd": "string"},
    )

    mktcap["Stkcd"] = (
        mktcap["Stkcd"].apply(normalize_stock_code)
    )
    mktcap["td"] = pd.to_numeric(
        mktcap["td"],
        errors="coerce",
    )

    mktcap = mktcap.rename(
        columns={
            "td": "td_tm3",
            "size": "mktcap_tm3",
        }
    )

    mktcap_index = mktcap.set_index(
        ["Stkcd", "td_tm3"]
    )["mktcap_tm3"]

    if not mktcap_index.index.is_unique:
        raise ValueError("日度市值表的股票代码—交易日键不唯一。")

    del mktcap
    gc.collect()

    return mktcap_index

def load_full_marketcap_breakpoints(breakpoint_path):
    """读取月度市值断点，整理用于匹配的月份字段。"""

    marketcap_group = pd.read_csv(
        breakpoint_path,
        encoding="utf-8",
    )

    marketcap_group["Trdmnt"] = pd.to_datetime(
        marketcap_group["Trdmnt"]
    )

    return marketcap_group.rename(
        columns={"Trdmnt": "Trdmnt_breakpoint"}
    )


def load_full_industry_index(industry_path):
    """读取日度行业数据，建立公司—交易日行业查询索引。"""

    daily_industry = pd.read_csv(
        industry_path,
        encoding="utf-8",
        usecols=["Stkcd", "td", "IndustryCode"],
        dtype={
            "Stkcd": "string",
            "IndustryCode": "string",
        },
    )

    daily_industry["Stkcd"] = (
        daily_industry["Stkcd"].apply(normalize_stock_code)
    )
    daily_industry["td"] = pd.to_numeric(
        daily_industry["td"],
        errors="coerce",
    )

    daily_industry = (
        daily_industry
        .dropna(subset=["Stkcd", "td", "IndustryCode"])
        .drop_duplicates(["Stkcd", "td"])
    )

    industry_index = daily_industry.set_index(
        ["Stkcd", "td"]
    )["IndustryCode"]

    del daily_industry
    gc.collect()

    return industry_index


def load_full_supply_chain_index(supply_chain_path):
    """读取一级供应链关系，构造用于排除公司对的年度双向索引。"""

    supply_chain_raw = pd.read_csv(
        supply_chain_path,
        encoding="utf-8",
        usecols=[
            "Symbol",
            "EndDate",
            "StateTypeCode",
            "PSCBusinessRelation",
            "PSCSymbol",
        ],
        dtype={
            "Symbol": "string",
            "StateTypeCode": "string",
            "PSCBusinessRelation": "string",
            "PSCSymbol": "string",
        },
        parse_dates=["EndDate"],
    )

    supply_chain_raw["yearCue"] = (
        supply_chain_raw["EndDate"].dt.year
    )
    supply_chain_raw["Symbol"] = (
        supply_chain_raw["Symbol"].apply(normalize_stock_code)
    )

    # 关联方代码先按原始格式筛选，再进行标准化。
    is_a_share_pair = (
        supply_chain_raw["StateTypeCode"].eq("1")
        & supply_chain_raw["PSCSymbol"].str.fullmatch(
            r"[0368][0-9]{5}",
            na=False,
        )
        & supply_chain_raw["PSCBusinessRelation"].isin(["1", "2"])
    )

    supply_chain = supply_chain_raw.loc[is_a_share_pair].copy()
    supply_chain["PSCSymbol"] = (
        supply_chain["PSCSymbol"].apply(normalize_stock_code)
    )

    # 沿用原脚本的上下游方向转换规则。
    supply_chain["upstream_stkcd"] = np.where(
        supply_chain["PSCBusinessRelation"].eq("1"),
        supply_chain["Symbol"],
        supply_chain["PSCSymbol"],
    )

    supply_chain["downstream_stkcd"] = np.where(
        supply_chain["PSCBusinessRelation"].eq("1"),
        supply_chain["PSCSymbol"],
        supply_chain["Symbol"],
    )

    # 删除自身配对，并对年度有向关系去重。
    supply_chain = supply_chain.loc[
        supply_chain["upstream_stkcd"]
        != supply_chain["downstream_stkcd"],
        ["yearCue", "upstream_stkcd", "downstream_stkcd"],
    ].drop_duplicates()

    # 任一方向存在关系，都纳入公司对排除索引。
    vertical_pairs_forward = supply_chain.rename(
        columns={
            "upstream_stkcd": "Stkcd",
            "downstream_stkcd": "StkcdRecall",
        }
    )[["yearCue", "Stkcd", "StkcdRecall"]]

    vertical_pairs_reverse = supply_chain.rename(
        columns={
            "downstream_stkcd": "Stkcd",
            "upstream_stkcd": "StkcdRecall",
        }
    )[["yearCue", "Stkcd", "StkcdRecall"]]

    vertical_pairs = (
        pd.concat(
            [vertical_pairs_forward, vertical_pairs_reverse],
            ignore_index=True,
        )
        .drop_duplicates()
        .assign(vertical_related=1)
    )

    return vertical_pairs.set_index(
        ["yearCue", "Stkcd", "StkcdRecall"]
    ).index

def load_full_analyst_exclusion_index(
    overlap_root,
    id_mode,
    lookback_days,
):
    """读取已有共同分析师覆盖结果，构造公司对排除索引。"""

    id_mode_folder = {
        "split": "拆开AnanmID",
        "team": "不拆开AnanmID",
    }

    if id_mode not in id_mode_folder:
        raise ValueError(
            "分析师身份口径只能是 "
            "'split'（个人）或 'team'（完整团队）。"
        )

    analyst_overlap_path = (
        Path(overlap_root)
        / id_mode_folder[id_mode]
        / f"Full样本_分析师共同覆盖_滚动{lookback_days}天.csv"
    )

    analyst_overlap = pd.read_csv(
        analyst_overlap_path,
        encoding="utf-8-sig",
        usecols=[
            "Annodt",
            "Stkcd",
            "StkcdRecall",
            "same_analyst_overlap",
            "n_common_analysts",
        ],
        dtype={
            "Stkcd": "string",
            "StkcdRecall": "string",
            "same_analyst_overlap": "Int64",
            "n_common_analysts": "Int64",
        },
        parse_dates=["Annodt"],
    )

    analyst_overlap["Stkcd"] = (
        analyst_overlap["Stkcd"].apply(normalize_stock_code)
    )
    analyst_overlap["StkcdRecall"] = (
        analyst_overlap["StkcdRecall"].apply(normalize_stock_code)
    )

    # 沿用原规则：重复公司对公告日保留首条。
    analyst_overlap = analyst_overlap.drop_duplicates(
        ["Annodt", "Stkcd", "StkcdRecall"]
    )

    # 来源文件只记录已识别共同覆盖的公司对。
    # 沿用标记非缺失即纳入排除索引的判定。
    analyst_exclusion_index = analyst_overlap.loc[
        analyst_overlap["same_analyst_overlap"].notna(),
        ["Annodt", "Stkcd", "StkcdRecall"],
    ].set_index(
        ["Annodt", "Stkcd", "StkcdRecall"]
    ).index

    del analyst_overlap
    gc.collect()

    return analyst_exclusion_index


def load_full_auxiliary_tables(
    paths,
    config,
    analyst_overlap_root,
):
    """统一读取构造全样本提示事件所需的辅助表和索引。"""

    print("正在读取辅助表......")

    calendar = load_full_calendar(
        calendar_path=paths["TRD_Cale_td_path"],
    )

    mktcap_index = load_full_marketcap_index(
        daily_panel_path=paths["TRD_Dalyr_DGTW_path"],
    )

    marketcap_group_merge = load_full_marketcap_breakpoints(
        breakpoint_path=paths["marketcap_group_path"],
    )

    industry_index = load_full_industry_index(
        industry_path=paths["industry_path"],
    )

    vertical_pair_index = load_full_supply_chain_index(
        supply_chain_path=paths["SC_NetWorkRelationsIndex_path"],
    )

    analyst_exclusion_index = load_full_analyst_exclusion_index(
        overlap_root=analyst_overlap_root,
        id_mode=config["ANALYST_ID_MODE"],
        lookback_days=config["ANALYST_LOOKBACK_DAYS"],
    )

    print("辅助表读取完成。")

    return {
        "calendar": calendar,
        "mktcap_index": mktcap_index,
        "marketcap_group_merge": marketcap_group_merge,
        "industry_index": industry_index,
        "vertical_pair_index": vertical_pair_index,
        "analyst_exclusion_index": analyst_exclusion_index,
    }


def load_cached_full_cue_events(cue_events_path):
    """读取已有全样本提示事件表，并检查公司—日期键是否重复。"""

    cue_events_for_panel = pd.read_csv(
        cue_events_path,
        encoding="utf-8-sig",
        parse_dates=["date"],
        dtype={"StkcdRecall": "string"},
    )

    cue_events_for_panel["StkcdRecall"] = (
        cue_events_for_panel["StkcdRecall"]
        .apply(normalize_stock_code)
    )

    duplicate_count = cue_events_for_panel.duplicated(
        ["StkcdRecall", "date"]
    ).sum()

    if duplicate_count != 0:
        raise ValueError(
            "缓存的全样本提示事件表存在被提示公司—日期重复键。"
        )

    print(f"复用已生成的全样本提示事件表：{cue_events_path}")
    print("缓存行数：", len(cue_events_for_panel))

    return cue_events_for_panel


def process_mm_chunk(
    mm,
    *,
    calendar,
    mktcap_index,
    marketcap_group_merge,
    industry_index,
    vertical_pair_index,
    analyst_exclusion_index,
    config,
):
    """处理一个记忆矩阵数据块，生成被提示公司事件日提示标记。"""

    # 从传入的配置中取得本函数需要的三项设置。
    EVENT_DATE_MODE = config["EVENT_DATE_MODE"]
    MARKETCAP_BREAKPOINT_MODE = config["MARKETCAP_BREAKPOINT_MODE"]
    DROP_ANALYST_OVERLAP = config["DROP_ANALYST_OVERLAP"]

    mm = mm.copy()

    mm["Stkcd"] = mm["Stkcd"].apply(normalize_stock_code)
    mm["StkcdRecall"] = mm["StkcdRecall"].apply(normalize_stock_code)
    mm = mm.loc[(mm["Stkcd"] != "") & (mm["StkcdRecall"] != "")].copy()

    if mm.empty:
        return combine_cue_parts([])

    annodt_key = (
        mm[["Annodt"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Annodt")
    )

    annodt_key = pd.merge_asof(
        annodt_key,
        calendar.rename(columns={"Trddt": "event_Trddt", "td": "event_td"}),
        left_on="Annodt",
        right_on="event_Trddt",
        direction="forward",
        allow_exact_matches=(EVENT_DATE_MODE == "next_if_nontrading"),
    )

    mm = mm.merge(
        annodt_key,
        on="Annodt",
        how="left",
        validate="m:1",
    )

    mm["event_Trddt"] = pd.to_datetime(mm["event_Trddt"], errors="coerce")
    mm["event_td"] = pd.to_numeric(mm["event_td"], errors="coerce")
    mm["td_tm3"] = mm["event_td"] - 3

    mktcap_keys = pd.MultiIndex.from_frame(mm[["Stkcd", "td_tm3"]])
    mm["mktcap_tm3"] = mktcap_index.reindex(mktcap_keys).to_numpy()

    if MARKETCAP_BREAKPOINT_MODE == "prev_month":
        mm["Trdmnt_for_breakpoint"] = (
            mm["event_Trddt"].dt.to_period("M").sub(1).dt.to_timestamp()
        )
    elif MARKETCAP_BREAKPOINT_MODE == "same_month":
        mm["Trdmnt_for_breakpoint"] = (
            mm["event_Trddt"].dt.to_period("M").dt.to_timestamp()
        )
    else:
        raise ValueError("MARKETCAP_BREAKPOINT_MODE 只能是 'prev_month' 或 'same_month'")

    mm = mm.merge(
        marketcap_group_merge,
        left_on="Trdmnt_for_breakpoint",
        right_on="Trdmnt_breakpoint",
        how="left",
        validate="m:1",
    )

    mm = mm.dropna(
        subset=[
            "mktcap_tm3",
            "marketCap_90th_percentile",
            "marketCap_50th_percentile",
            "marketCap_10th_percentile",
        ]
    ).copy()

    if mm.empty:
        return combine_cue_parts([])

    mm_key = mm[["Annodt", "event_td", "Stkcd", "StkcdRecall"]].copy()

    cue_industry_keys = pd.MultiIndex.from_arrays(
        [mm_key["Stkcd"], mm_key["event_td"]]
    )
    recall_industry_keys = pd.MultiIndex.from_arrays(
        [mm_key["StkcdRecall"], mm_key["event_td"]]
    )
    mm_key["cue_industry_code"] = industry_index.reindex(
        cue_industry_keys
    ).to_numpy()
    mm_key["recall_industry_code"] = industry_index.reindex(
        recall_industry_keys
    ).to_numpy()

    mm_key["same_industry"] = (
        mm_key["cue_industry_code"].notna()
        & mm_key["recall_industry_code"].notna()
        & (mm_key["cue_industry_code"] == mm_key["recall_industry_code"])
    )

    mm_key = mm_key.loc[~mm_key["same_industry"]].copy()

    mm_key["yearCue"] = mm_key["Annodt"].dt.year

    vertical_keys = pd.MultiIndex.from_frame(
        mm_key[["yearCue", "Stkcd", "StkcdRecall"]]
    )
    mm_key = mm_key.loc[~vertical_keys.isin(vertical_pair_index)].copy()

    if DROP_ANALYST_OVERLAP:
        analyst_keys = pd.MultiIndex.from_frame(
            mm_key[["Annodt", "Stkcd", "StkcdRecall"]]
        )
        mm_key = mm_key.loc[
            ~analyst_keys.isin(analyst_exclusion_index)
        ].copy()

    key_cols = ["Annodt", "Stkcd", "StkcdRecall"]
    mm_key_keep = mm_key[key_cols].drop_duplicates()

    mm = mm.merge(
        mm_key_keep,
        on=key_cols,
        how="inner",
        validate="m:1",
    )

    if mm.empty:
        return combine_cue_parts([])

    mm["quarterLag1Alt"] = (
        pd.to_numeric(mm["quarterLag1Alt"], errors="coerce")
        .fillna(0)
        .astype("int8")
    )

    mm["size90plus"] = (
        mm["mktcap_tm3"] > mm["marketCap_90th_percentile"]
    ).astype("int8")

    mm["quarterLag1size90plus"] = (
        mm["quarterLag1Alt"] * mm["size90plus"]
    ).astype("int8")

    part = (
        mm.groupby(["StkcdRecall", "event_Trddt"], as_index=False)["quarterLag1size90plus"]
        .max()
    )

    part = part.loc[part["event_Trddt"].notna()].copy()

    return part


def rebuild_full_cue_events(
    matrix_path,
    *,
    calendar,
    mktcap_index,
    marketcap_group_merge,
    industry_index,
    vertical_pair_index,
    analyst_exclusion_index,
    config,
):
    """分块读取记忆矩阵，聚合全样本的大市值提示事件。"""

    print("正在分块重算全样本提示事件......")
    collapsed_parts = []

    reader = pd.read_csv(
        matrix_path,
        encoding="utf-8",
        usecols=[
            "Annodt",
            "Stkcd",
            "StkcdRecall",
            "quarterLag1Alt",
        ],
        dtype={
            "Stkcd": "string",
            "StkcdRecall": "string",
        },
        chunksize=config["MM_CHUNKSIZE"],
    )

    for chunk_no, chunk in enumerate(reader, start=1):
        lag1 = pd.to_numeric(
            chunk["quarterLag1Alt"],
            errors="coerce",
        )

        if not lag1.dropna().isin([0, 1]).all():
            raise ValueError("上一季度关系标记不是二值变量。")

        chunk = chunk.loc[lag1.eq(1)].copy()
        if chunk.empty:
            continue

        chunk["Annodt"] = pd.to_datetime(
            chunk["Annodt"],
            errors="coerce",
        )

        part = process_mm_chunk(
            chunk,
            calendar=calendar,
            mktcap_index=mktcap_index,
            marketcap_group_merge=marketcap_group_merge,
            industry_index=industry_index,
            vertical_pair_index=vertical_pair_index,
            analyst_exclusion_index=analyst_exclusion_index,
            config=config,
        )

        part = part.loc[
            part["quarterLag1size90plus"].eq(1)
        ]

        if not part.empty:
            collapsed_parts.append(part)

        # 累积25份非空结果时，先归并一次。
        if len(collapsed_parts) >= 25:
            collapsed_parts = [
                combine_cue_parts(collapsed_parts)
            ]

        if chunk_no % 10 == 0:
            print(f"已处理记忆矩阵 {chunk_no:,} 个分块")

        del chunk, part
        gc.collect()

    # 最后再次归并，合并跨块出现的同一公司日。
    return combine_cue_parts(collapsed_parts)


def save_full_cue_events(collapsed, output_path):
    """整理并保存提示事件表，返回供日度主面板合并使用的表。"""

    output_path = Path(output_path)

    # 保留原来的日期整理方式和输出列顺序。
    collapsed["date"] = collapsed["event_Trddt"]

    cue_events_for_panel = collapsed.drop(
        columns=["event_Trddt"],
        errors="ignore",
    ).copy()

    cue_events_for_panel = (
        cue_events_for_panel
        .sort_values(["StkcdRecall", "date"])
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cue_events_for_panel.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print(
        "\n已保存用于后续主面板合并的提示事件表：",
        output_path,
    )
    print("保存行数：", len(cue_events_for_panel))
    print("保存列数：", cue_events_for_panel.shape[1])
    print(
        "上一季度大市值提示标记合计：",
        cue_events_for_panel["quarterLag1size90plus"].sum(),
    )

    return cue_events_for_panel


def build_full_cue_events(
    paths,
    config,
    analyst_overlap_root,
):
    """统一组织辅助数据读取、缓存复用、分块重算及事件表保存。"""

    auxiliary_tables = load_full_auxiliary_tables(
        paths=paths,
        config=config,
        analyst_overlap_root=analyst_overlap_root,
    )

    cue_events_path = Path(paths["cue_events_out_path"])

    if (
        config["REUSE_EXISTING_CUE_EVENTS"]
        and cue_events_path.exists()
    ):
        cue_events_for_panel = load_cached_full_cue_events(
            cue_events_path=cue_events_path,
        )
    else:
        collapsed = rebuild_full_cue_events(
            matrix_path=paths["MemoryMatrixFull_path"],
            calendar=auxiliary_tables["calendar"],
            mktcap_index=auxiliary_tables["mktcap_index"],
            marketcap_group_merge=auxiliary_tables[
                "marketcap_group_merge"
            ],
            industry_index=auxiliary_tables["industry_index"],
            vertical_pair_index=auxiliary_tables[
                "vertical_pair_index"
            ],
            analyst_exclusion_index=auxiliary_tables[
                "analyst_exclusion_index"
            ],
            config=config,
        )

        cue_events_for_panel = save_full_cue_events(
            collapsed=collapsed,
            output_path=cue_events_path,
        )

        del collapsed

    # 两个分支结束后，统一释放辅助数据。
    del auxiliary_tables
    gc.collect()

    return cue_events_for_panel

def load_full_daily_panel(daily_panel_path):
    """读取个股日度面板，整理为被提示公司日度主面板。"""

    base_return_cols = [
        "raw_ret_tm3",
        "raw_ret_tm2",
        "raw_ret_tm1",
        "raw_ret_t",
    ]
    forward_return_cols = [
        f"raw_ret_t_t{i}"
        for i in range(1, 51)
    ]

    daily_panel = pd.read_csv(
        daily_panel_path,
        encoding="utf-8",
        usecols=(
            ["Stkcd", "td", "Trddt", "year", "mktcap_tm3"]
            + base_return_cols
            + forward_return_cols
        ),
        dtype=(
            {
                "Stkcd": "string",
                "td": "int32",
                "year": "float32",
                "mktcap_tm3": "float32",
            }
            | {
                col: "float32"
                for col in base_return_cols + forward_return_cols
            }
        ),
        parse_dates=["Trddt"],
    )

    daily_panel["Stkcd"] = (
        daily_panel["Stkcd"].apply(normalize_stock_code)
    )

    daily_panel = (
        daily_panel
        .rename(columns={"Stkcd": "StkcdRecall"})
        .drop(columns=["IndustryCode"], errors="ignore")
    )

    raw_return_rename = {
        "raw_ret_tm3": "return_tm3_Recall",
        "raw_ret_tm2": "return_tm2_Recall",
        "raw_ret_tm1": "return_tm1_Recall",
        "raw_ret_t": "return_t_Recall",
    }
    raw_return_rename.update({
        f"raw_ret_t_t{i}": f"return_t_tp{i}_Recall"
        for i in range(1, 51)
    })

    return daily_panel.rename(columns=raw_return_rename)


def add_full_own_earnings_announcement_flag(
    daily_panel,
    report_path,
    event_date_mode,
):
    """映射自身财报公告日期，并合并自身公告日标记。"""

    report = pd.read_csv(
        report_path,
        usecols=["Stkcd", "Accper", "Annodt"],
        parse_dates=["Accper", "Annodt"],
    )

    report["Stkcd"] = (
        report["Stkcd"].apply(normalize_stock_code)
    )
    report = report.rename(
        columns={"Stkcd": "StkcdRecall"}
    )

    calendar = (
        daily_panel[["Trddt", "td"]]
        .drop_duplicates()
        .sort_values("Trddt")
    )

    ann_key = (
        report[["Annodt"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Annodt")
    )

    ann_key = pd.merge_asof(
        ann_key,
        calendar,
        left_on="Annodt",
        right_on="Trddt",
        direction="forward",
        allow_exact_matches=(
            event_date_mode == "next_if_nontrading"
        ),
    )

    own_ea = (
        report
        .merge(
            ann_key,
            on="Annodt",
            how="left",
            validate="m:1",
        )
        .dropna(subset=["Trddt"])
        [["StkcdRecall", "Trddt"]]
        .drop_duplicates()
    )
    own_ea["ownEA"] = 1

    daily_panel = daily_panel.merge(
        own_ea,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="1:1",
    )
    daily_panel["ownEA"] = (
        daily_panel["ownEA"].fillna(0).astype("int8")
    )

    own_ea_count = daily_panel["ownEA"].eq(1).sum()
    print("自身公告标记为1的数量：", own_ea_count)
    print("自身公告标记为1的比例：", own_ea_count / len(daily_panel))

    unmatched_own_ea = own_ea.merge(
        daily_panel[["StkcdRecall", "Trddt"]],
        on=["StkcdRecall", "Trddt"],
        how="left",
        indicator=True,
    )
    unmatched_own_ea = unmatched_own_ea.loc[
        unmatched_own_ea["_merge"].eq("left_only")
    ].copy()

    print("未匹配自身公告事件数：", len(unmatched_own_ea))
    print(
        unmatched_own_ea["Trddt"]
        .dt.year.value_counts().sort_index()
    )

    return daily_panel


# 生成被提示公司自身盈余公告在事件日前后 10 个交易日内的控制变量
# 这些变量用于控制自身公告窗口对收益的污染
#===============================================================================
# 构造 own EA 前后 10 个交易日 dummy
# 当前 TRD_Dalyr 层级：StkcdRecall × Trddt
#===============================================================================

def add_full_own_ea_windows(daily_panel, window=10):
    """生成自身财报公告前后各10个交易日的逐日控制标记。"""

    dup_n = daily_panel.duplicated(
        ["StkcdRecall", "td"]
    ).sum()
    print("日度面板的公司—交易日重复数：", dup_n)

    if dup_n != 0:
        raise ValueError("日度面板的公司—交易日键不唯一。")

    own_ea_td = (
        daily_panel.loc[
            daily_panel["ownEA"].eq(1),
            ["StkcdRecall", "td"],
        ]
        .drop_duplicates()
        .copy()
    )
    print("自身财报公告事件数：", len(own_ea_td))

    for i in range(1, window + 1):
        # 将公告事件向前映射：当前日之后第i日有公告。
        future_ea = own_ea_td.copy()
        future_ea["td"] = future_ea["td"] - i
        future_ea = (
            future_ea
            .drop_duplicates(["StkcdRecall", "td"])
            .assign(**{f"ownEA_tp{i}": 1})
        )

        daily_panel = daily_panel.merge(
            future_ea,
            on=["StkcdRecall", "td"],
            how="left",
            validate="1:1",
        )
        daily_panel[f"ownEA_tp{i}"] = (
            daily_panel[f"ownEA_tp{i}"]
            .fillna(0)
            .astype("int8")
        )

        # 将公告事件向后映射：当前日之前第i日有公告。
        past_ea = own_ea_td.copy()
        past_ea["td"] = past_ea["td"] + i
        past_ea = (
            past_ea
            .drop_duplicates(["StkcdRecall", "td"])
            .assign(**{f"ownEA_tm{i}": 1})
        )

        daily_panel = daily_panel.merge(
            past_ea,
            on=["StkcdRecall", "td"],
            how="left",
            validate="1:1",
        )
        daily_panel[f"ownEA_tm{i}"] = (
            daily_panel[f"ownEA_tm{i}"]
            .fillna(0)
            .astype("int8")
        )

    print("自身公告前后窗口变量生成完成。")

    own_ea_window_cols = (
        [f"ownEA_tp{i}" for i in range(1, window + 1)]
        + ["ownEA"]
        + [f"ownEA_tm{i}" for i in range(1, window + 1)]
    )
    print(daily_panel[own_ea_window_cols].sum())

    return daily_panel


def merge_full_own_event_controls(
    daily_panel,
    performance_forecast_path,
    other_event_path,
):
    """合并自身业绩预告或快报、十类专题事件控制变量。"""

    performance_forecast_controls = pd.read_csv(
        performance_forecast_path,
        encoding="utf-8-sig",
        usecols=[
            "StkcdRecall",
            "Trddt",
            "ownPerformanceForecastBrief",
        ],
        dtype={"StkcdRecall": "string"},
        parse_dates=["Trddt"],
    )

    other_event_controls = pd.read_csv(
        other_event_path,
        encoding="utf-8-sig",
        usecols=["StkcdRecall", "Trddt", "ownOtherEventER"],
        dtype={"StkcdRecall": "string"},
        parse_dates=["Trddt"],
    )

    for controls_name, controls_df in [
        ("业绩预告快报", performance_forecast_controls),
        ("专题事件", other_event_controls),
    ]:
        controls_df["StkcdRecall"] = (
            controls_df["StkcdRecall"].apply(normalize_stock_code)
        )

        missing_key = (
            controls_df["StkcdRecall"].eq("")
            | controls_df["Trddt"].isna()
        ).sum()
        duplicate_key = controls_df.duplicated(
            ["StkcdRecall", "Trddt"]
        ).sum()

        print(
            f"{controls_name}控制表缺失键数：{missing_key}，"
            f"重复键数：{duplicate_key}"
        )

        if missing_key != 0 or duplicate_key != 0:
            raise ValueError(
                f"{controls_name}控制表的公司—交易日键存在缺失或重复。"
            )

    performance_forecast_controls["ownPerformanceForecastBrief"] = (
        pd.to_numeric(
            performance_forecast_controls["ownPerformanceForecastBrief"],
            errors="raise",
        )
        .fillna(0)
        .astype("int8")
    )
    other_event_controls["ownOtherEventER"] = (
        pd.to_numeric(
            other_event_controls["ownOtherEventER"],
            errors="raise",
        )
        .fillna(0)
        .astype("int8")
    )

    # 上游已整理交易日期，此处直接合并。
    daily_panel = daily_panel.merge(
        performance_forecast_controls,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="1:1",
    )
    daily_panel = daily_panel.merge(
        other_event_controls,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="1:1",
    )

    own_event_cols = [
        "ownPerformanceForecastBrief",
        "ownOtherEventER",
    ]
    daily_panel[own_event_cols] = (
        daily_panel[own_event_cols].fillna(0).astype("int8")
    )

    print(
        "专题研究事件标记为1的数量：",
        daily_panel["ownOtherEventER"].sum(),
    )
    print(
        "业绩预告或快报标记为1的数量：",
        daily_panel["ownPerformanceForecastBrief"].sum(),
    )
    print(
        "两个公告控制标记同时为1的数量：",
        (
            daily_panel["ownOtherEventER"].eq(1)
            & daily_panel["ownPerformanceForecastBrief"].eq(1)
        ).sum(),
    )

    return daily_panel


#===============================================================================
# 合并上游已构造的真实交易日新闻控制变量
#===============================================================================
def merge_full_news_controls(daily_panel, news_path):
    """读取真实交易日新闻控制变量，并合并到日度主面板。"""

    news_controls = pd.read_csv(
        news_path,
        encoding="utf-8-sig",
        usecols=[
            "Stkcd",
            "date",
            "AES",
            "AEV",
            "aesIncrease",
            "aesDecrease",
            "logIncreaseAEV",
            "logDecreaseAEV",
        ],
        dtype={"Stkcd": "string"},
        parse_dates=["date"],
    )

    news_controls["Stkcd"] = (
        news_controls["Stkcd"].apply(normalize_stock_code)
    )
    news_controls = news_controls.rename(
        columns={
            "Stkcd": "StkcdRecall",
            "date": "Trddt",
        }
    )

    daily_panel["Trddt"] = pd.to_datetime(daily_panel["Trddt"])
    news_controls["Trddt"] = pd.to_datetime(news_controls["Trddt"])

    duplicate_news = news_controls.duplicated(
        ["StkcdRecall", "Trddt"]
    ).sum()
    print("真实交易日新闻控制表中公司—交易日重复数：", duplicate_news)

    if duplicate_news != 0:
        raise ValueError(
            "真实交易日新闻控制表在公司—交易日层面不唯一，"
            "请先检查上游构造。"
        )

    news_numeric_cols = [
        "AES",
        "AEV",
        "aesIncrease",
        "aesDecrease",
        "logIncreaseAEV",
        "logDecreaseAEV",
    ]
    news_controls[news_numeric_cols] = (
        news_controls[news_numeric_cols].apply(
            pd.to_numeric,
            errors="raise",
        )
    )

    if (news_controls["AEV"].dropna() < 0).any():
        raise ValueError("真实交易日新闻控制表中的新闻关注度不应出现负数。")

    print("真实交易日新闻控制表行数：", len(news_controls))

    daily_panel = daily_panel.merge(
        news_controls,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="1:1",
    )

    del news_controls
    gc.collect()

    print("合并真实交易日新闻控制变量后主面板表结构：", daily_panel.shape)
    for news_col in news_numeric_cols:
        print(f"{news_col} 非缺失数量：", daily_panel[news_col].notna().sum())

    return daily_panel


def merge_full_cue_events(daily_panel, cue_events):
    """合并提示事件，并将未匹配的提示标记填为零。"""

    cue_panel = cue_events.copy()

    daily_panel["Trddt"] = pd.to_datetime(daily_panel["Trddt"])
    cue_panel["date"] = pd.to_datetime(cue_panel["date"])

    duplicate_main = daily_panel.duplicated(
        ["StkcdRecall", "Trddt"]
    ).sum()
    print("日度主面板中公司—交易日重复数：", duplicate_main)

    if duplicate_main != 0:
        raise ValueError(
            "日度主面板在公司—交易日层面不唯一，请先检查主面板。"
        )

    duplicate_cue = cue_panel.duplicated(
        ["StkcdRecall", "date"]
    ).sum()
    print("提示事件表中公司—事件日重复数：", duplicate_cue)

    if duplicate_cue != 0:
        raise ValueError(
            "提示事件表在公司—事件日层面不唯一，请检查事件聚合步骤。"
        )

    print("提示事件总数：", len(cue_panel))

    daily_panel = daily_panel.merge(
        cue_panel,
        left_on=["StkcdRecall", "Trddt"],
        right_on=["StkcdRecall", "date"],
        how="left",
        validate="1:1",
    )

    matched_cue_count = daily_panel["date"].notna().sum()
    print("提示事件能匹配到日度主面板数量：", matched_cue_count)
    print(
        "提示事件匹配不到日度主面板数量：",
        len(cue_panel) - matched_cue_count,
    )

    del cue_panel
    daily_panel = daily_panel.drop(
        columns=["date"],
        errors="ignore",
    )

    print("合并提示事件后的主面板表结构：", daily_panel.shape)
    print(
        "合并后公司—交易日重复数：",
        daily_panel.duplicated(["StkcdRecall", "Trddt"]).sum(),
    )

    cue_dummy_cols = [
        "quarterLag1size90plus",
    ]
    cue_dummy_cols = [
        col
        for col in cue_dummy_cols
        if col in daily_panel.columns
    ]

    daily_panel[cue_dummy_cols] = (
        daily_panel[cue_dummy_cols]
        .fillna(0)
        .astype("int8")
    )

    print("提示标记缺失值已设为零")
    print(
        daily_panel[cue_dummy_cols]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )

    return daily_panel


#===============================================================================
# 合并特征调整收益率
#===============================================================================
def merge_full_adjusted_returns(daily_panel, adjusted_return_path):
    """读取被提示公司的特征调整收益率，并合并到日度主面板。"""

    adjusted_returns = pd.read_csv(
        adjusted_return_path,
        encoding="utf-8-sig",
        dtype=(
            {
                "Stkcd": "string",
                "portfolioNumber": "float32",
            }
            | {
                column: "float32"
                for column in (
                    [
                        "chAdj_ret_tm3",
                        "chAdj_ret_tm2",
                        "chAdj_ret_tm1",
                        "chAdj_ret_t",
                    ]
                    + [
                        f"chAdj_ret_t_t{i}"
                        for i in range(1, 51)
                    ]
                )
            }
        ),
        parse_dates=["Trddt"],
    )

    adjusted_returns["Stkcd"] = (
        adjusted_returns["Stkcd"].apply(normalize_stock_code)
    )

    adjusted_returns = adjusted_returns.rename(
        columns={
            "Stkcd": "StkcdRecall",
            "chAdj_ret_tm3": "chAdjreturn_tm3_Recall",
            "chAdj_ret_tm2": "chAdjreturn_tm2_Recall",
            "chAdj_ret_tm1": "chAdjreturn_tm1_Recall",
            "chAdj_ret_t": "chAdjreturn_t_Recall",
        }
    )

    rename_forward = {
        f"chAdj_ret_t_t{i}": f"chAdjreturn_t_tp{i}_Recall"
        for i in range(1, 51)
    }
    adjusted_returns = adjusted_returns.rename(
        columns=rename_forward
    )

    keep_adj_cols = (
        ["StkcdRecall", "Trddt", "portfolioNumber"]
        + [
            col
            for col in adjusted_returns.columns
            if col.startswith("chAdjreturn_")
        ]
    )
    adjusted_returns = adjusted_returns[keep_adj_cols].copy()

    duplicate_adj = adjusted_returns.duplicated(
        ["StkcdRecall", "Trddt"]
    ).sum()
    print("特征调整收益率表中公司—交易日重复数：", duplicate_adj)

    if duplicate_adj != 0:
        raise ValueError(
            "特征调整收益率表在公司—交易日层面不唯一。"
        )

    daily_panel = daily_panel.merge(
        adjusted_returns,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="1:1",
    )

    del adjusted_returns
    gc.collect()

    print("合并特征调整收益率后表结构：", daily_panel.shape)
    for col in [
        "chAdjreturn_t_Recall",
        "chAdjreturn_t_tp1_Recall",
        "chAdjreturn_t_tp10_Recall",
    ]:
        print(f"{col} 缺失比例：", daily_panel[col].isna().mean())

    return daily_panel


#===============================================================================
# 最终整理样本并进行缩尾
#===============================================================================
# 缩尾方式是：全样本 1% winsorize（缩尾），不是按年份缩尾。
# 也就是把变量低于第 1 百分位的值替换为第 1 百分位，把高于第 99 百分位的值替换为第 99 百分位，
# 原变量保留，新生成 _w 变量。

def winsorize_limits(s, p=0.01):
    """
    只计算缩尾阈值，不立刻生成缩尾后的整列。
    """
    lower = s.quantile(p)
    upper = s.quantile(1 - p)
    return lower, upper


def add_winsor_col(df, col, winsor_limits):
    """使用传入的全样本缩尾阈值，生成缩尾后的变量。"""

    lower, upper = winsor_limits[col]
    df[f"{col}_w"] = df[col].clip(
        lower=lower,
        upper=upper,
    )

def prepare_full_panel_for_baseline(daily_panel):
    """计算全样本缩尾阈值，并准备基准样本筛选所需的面板。"""

    print("开始缩尾......")

    # 本脚本缩尾变量：4项单日及10项累计窗口调整收益率。
    dep_vars = [
        "chAdjreturn_tm3_Recall",
        "chAdjreturn_tm2_Recall",
        "chAdjreturn_tm1_Recall",
        "chAdjreturn_t_Recall",
    ] + [
        f"chAdjreturn_t_tp{i}_Recall"
        for i in range(1, 11)
    ]

    winsor_source_cols = [
        col
        for col in dep_vars
        if col in daily_panel.columns
    ]

    # 阈值在基准样本筛选前的全日度面板上计算。
    winsor_limits = {}
    for col in winsor_source_cols:
        winsor_limits[col] = winsorize_limits(
            daily_panel[col],
            p=0.01,
        )

    # 先生成两个基准回归需要的因变量缩尾列。
    for col in [
        "chAdjreturn_t_tp1_Recall",
        "chAdjreturn_t_tp10_Recall",
    ]:
        if col in winsor_limits:
            add_winsor_col(
                daily_panel,
                col,
                winsor_limits,
            )

    daily_panel["quarter"] = daily_panel["Trddt"].dt.quarter
    daily_panel["yearQuarter"] = (
        daily_panel["Trddt"].dt.to_period("Q")
    )

    front_cols = ["StkcdRecall", "Trddt"]
    other_cols = [
        col
        for col in daily_panel.columns
        if col not in front_cols
    ]

    daily_panel = (
        daily_panel[front_cols + other_cols]
        .sort_values(["StkcdRecall", "Trddt"])
        .reset_index(drop=True)
    )

    return daily_panel, winsor_source_cols, winsor_limits

#===============================================================================
# 复刻原文两个基准回归的 e(sample) 筛选
#===============================================================================

def baseline_regression_sample_mask(df, dep_col, rhs_cols, fe_cols, cluster_cols=None):
    """
    近似复刻 Stata reghdfe 后的 e(sample)：
    1. 因变量、解释变量、固定效应变量、聚类变量不能缺失；
    2. 吸收 StkcdRecall × yearQuarter 固定效应后，删除 singleton 固定效应组。
    """
    cluster_cols = cluster_cols or []
    needed_cols = [dep_col] + rhs_cols + fe_cols + cluster_cols

    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少基准回归变量，不能复刻 e(sample)：{missing_cols}")

    mask = df[needed_cols].notna().all(axis=1)

    fe_key = (
        df.loc[mask, fe_cols]
        .astype(str)
        .agg("×".join, axis=1)
    )

    fe_count = fe_key.value_counts()
    keep_index = fe_key.loc[fe_key.map(fe_count) > 1].index

    out = pd.Series(False, index=df.index)
    out.loc[keep_index] = True
    return out

def filter_full_baseline_sample(daily_panel):
    """按照两个基准回归的口径，顺序筛选日度面板。"""

    own_ea_cols = (
        [f"ownEA_tm{i}" for i in range(10, 0, -1)]
        + ["ownEA"]
        + [f"ownEA_tp{i}" for i in range(1, 11)]
    )

    own_disclosure_cols = [
        "ownOtherEventER",
        "ownPerformanceForecastBrief",
    ]

    baseline_rhs = (
        ["quarterLag1size90plus"]
        + own_ea_cols
        + own_disclosure_cols
        + [
            "aesIncrease",
            "aesDecrease",
            "logIncreaseAEV",
            "logDecreaseAEV",
        ]
    )

    print("开始筛选 [t, t+1] 基准回归样本......")
    mask_tp1 = baseline_regression_sample_mask(
        daily_panel,
        dep_col="chAdjreturn_t_tp1_Recall_w",
        rhs_cols=baseline_rhs,
        fe_cols=["StkcdRecall", "yearQuarter"],
        cluster_cols=["StkcdRecall", "Trddt"],
    )

    daily_panel = daily_panel.loc[mask_tp1].copy()
    print("[t, t+1] 筛选后样本量：", len(daily_panel))

    print("开始筛选 [t, t+10] 基准回归样本......")
    mask_tp10 = baseline_regression_sample_mask(
        daily_panel,
        dep_col="chAdjreturn_t_tp10_Recall_w",
        rhs_cols=baseline_rhs,
        fe_cols=["StkcdRecall", "yearQuarter"],
        cluster_cols=["StkcdRecall", "Trddt"],
    )

    daily_panel = daily_panel.loc[mask_tp10].copy()
    print("[t, t+10] 筛选后样本量：", len(daily_panel))

    return daily_panel


#===============================================================================
# 基准样本筛选后，补生成其余缩尾变量
#===============================================================================

def complete_full_winsor_columns(
    daily_panel,
    winsor_source_cols,
    winsor_limits,
):
    """沿用筛选前的阈值，补生成尚未生成的缩尾列。"""

    print("基准样本筛选后补生成其余缩尾变量......")

    for col in winsor_source_cols:
        w_col = f"{col}_w"

        if col in daily_panel.columns and w_col not in daily_panel.columns:
            add_winsor_col(
                daily_panel,
                col,
                winsor_limits,
            )

    print("最终同时进入两个基准回归的样本量：", len(daily_panel))
    print(
        "最终提示标记合计：",
        daily_panel["quarterLag1size90plus"].sum(),
    )

    return daily_panel


#===============================================================================
# 最终输出前整理变量
#===============================================================================
def finalize_full_panel_columns(daily_panel):
    """整理全样本最终输出字段及排列顺序。"""

    # 统一被提示公司事件前第三交易日总市值的输出名称。
    daily_panel["marketCap_Minus3_Recall"] = daily_panel["mktcap_tm3"]

    expected_return_cols = [
        "return_tm3_Recall",
        "return_tm2_Recall",
        "return_tm1_Recall",
        "return_t_Recall",
    ] + [
        f"return_t_tp{i}_Recall"
        for i in range(1, 51)
    ]

    missing_return_cols = [
        col
        for col in expected_return_cols
        if col not in daily_panel.columns
    ]

    if missing_return_cols:
        raise ValueError(
            "先把 raw_ret_* 重命名为 return_*_Recall，"
            "再删除 raw_ret_* 辅助列。"
            f" 当前缺少：{missing_return_cols[:10]}"
        )

    drop_cols_final = [
        "AES",
        "AEV",
        "ret",
        "retn",
        "Markettype",
        "size",
        "sizew",
        "trade_days_1m",
        "trade_days_1y",
        "mktcap_tm3",
        "float_mktcap_tm3",
        "mktcap_tm3_valid",
        "float_mktcap_tm3_valid",
        "portfolioNumber",
    ]

    drop_cols_final += [
        col
        for col in daily_panel.columns
        if col.startswith("raw_ret_")
    ]

    drop_cols_final = list(dict.fromkeys(drop_cols_final))
    drop_cols_existing = [
        col
        for col in drop_cols_final
        if col in daily_panel.columns
    ]

    print("最终输出前删除中间变量数量：", len(drop_cols_existing))
    print(drop_cols_existing)

    daily_panel = daily_panel.drop(columns=drop_cols_existing)

    front_cols = [
        "StkcdRecall",
        "Trddt",
        "td",
        "year",
        "quarter",
        "yearQuarter",
    ]
    front_cols = [
        col
        for col in front_cols
        if col in daily_panel.columns
    ]
    other_cols = [
        col
        for col in daily_panel.columns
        if col not in front_cols
    ]

    daily_panel = (
        daily_panel[front_cols + other_cols]
        .sort_values(["StkcdRecall", "Trddt"])
        .reset_index(drop=True)
    )

    return daily_panel


#===============================================================================
# 输出最终文件
#===============================================================================

def save_full_regression_panel(daily_panel, output_path):
    """保存全样本最终回归面板。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("正在输出全样本最终回归面板......")

    daily_panel.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print("全样本最终回归面板输出成功！")
    print("输出路径：", output_path)
    print("最终表结构：", daily_panel.shape)


def run_full_pipeline(paths, config, analyst_overlap_root):
    """按既定顺序执行全样本合并流程，返回最终回归面板。"""

    cue_events = build_full_cue_events(
        paths=paths,
        config=config,
        analyst_overlap_root=analyst_overlap_root,
    )

    daily_panel = load_full_daily_panel(
        daily_panel_path=paths["TRD_Dalyr_DGTW_path"],
    )

    daily_panel = add_full_own_earnings_announcement_flag(
        daily_panel=daily_panel,
        report_path=paths["IAR_Rept_path"],
        event_date_mode=config["EVENT_DATE_MODE"],
    )

    daily_panel = add_full_own_ea_windows(
        daily_panel=daily_panel,
        window=10,
    )

    daily_panel = merge_full_own_event_controls(
        daily_panel=daily_panel,
        performance_forecast_path=paths[
            "performance_forecast_controls_path"
        ],
        other_event_path=paths["other_event_controls_path"],
    )

    daily_panel = merge_full_news_controls(
        daily_panel=daily_panel,
        news_path=paths["AES_AEV_controls_path"],
    )

    daily_panel = merge_full_cue_events(
        daily_panel=daily_panel,
        cue_events=cue_events,
    )

    daily_panel = merge_full_adjusted_returns(
        daily_panel=daily_panel,
        adjusted_return_path=paths["adjusted_return_path"],
    )

    (
        daily_panel,
        winsor_source_cols,
        winsor_limits,
    ) = prepare_full_panel_for_baseline(
        daily_panel=daily_panel,
    )

    daily_panel = filter_full_baseline_sample(
        daily_panel=daily_panel,
    )

    daily_panel = complete_full_winsor_columns(
        daily_panel=daily_panel,
        winsor_source_cols=winsor_source_cols,
        winsor_limits=winsor_limits,
    )

    daily_panel = finalize_full_panel_columns(
        daily_panel=daily_panel,
    )

    save_full_regression_panel(
        daily_panel=daily_panel,
        output_path=paths["output_path"],
    )

    return daily_panel


def main(
    root="/Users/qiyuewu/Desktop/mmm复现数据包",
    *,
    drop_analyst_overlap=True,
    analyst_id_mode="split",
    analyst_lookback_days=365,
    marketcap_breakpoint_mode="prev_month",
    event_date_mode="next_if_nontrading",
    mm_chunksize=1_000_000,
    reuse_existing_cue_events=False,
):
    """集中设置路径和参数，启动全样本合并流程。"""

    root = Path(root)
    paths = build_full_paths(root)

    config = build_full_config(
        drop_analyst_overlap=drop_analyst_overlap,
        analyst_id_mode=analyst_id_mode,
        analyst_lookback_days=analyst_lookback_days,
        marketcap_breakpoint_mode=marketcap_breakpoint_mode,
        event_date_mode=event_date_mode,
        mm_chunksize=mm_chunksize,
        reuse_existing_cue_events=reuse_existing_cue_events,
    )

    analyst_overlap_root = (
        root / "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后"
    )

    return run_full_pipeline(
        paths=paths,
        config=config,
        analyst_overlap_root=analyst_overlap_root,
    )


if __name__ == "__main__":
    main()