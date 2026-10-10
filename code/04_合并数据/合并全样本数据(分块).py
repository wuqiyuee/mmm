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

ROOT = Path('/Users/qiyuewu/Desktop/mmm复现数据包')

MemoryMatrixFull_path = ROOT / '中间结果/03_记忆矩阵/MemoryMatrixFull(封装优化版).csv'
TRD_Cale_td_path = ROOT / '中间结果/01_清洗后样本/市场交易日历_td.csv'
TRD_Dalyr_DGTW_path = ROOT / '中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv'
marketcap_group_path = ROOT / '中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv'
industry_path = ROOT / '中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv'
SC_NetWorkRelationsIndex_path = ROOT/'input/CSMAR/供应链网络关系指标表2001-12-31至2025-12-31/SC_NetworkRelationsIndex.csv'
IAR_Rept_path = ROOT/'input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv'
AES_AEV_controls_path = ROOT/'中间结果/02_变量构造结果/3报刊新闻_AES_AEV/报刊新闻_AES_AEV_真实交易日控制变量.csv'
sample_intermediate_out_path = ROOT / "中间结果/04_合并数据/Sample_intermediate_full.csv"
adjusted_return_path = ROOT / "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW特征调整收益率.csv"
output_path = ROOT/'中间结果/04_合并数据/Full样本_最终回归面板.csv'
performance_forecast_controls_path = (
    ROOT / "中间结果/02_变量构造结果/5公司公告事件/业绩预告快报控制变量_公告日期口径.csv"
)
other_event_controls_path = (
    ROOT / "中间结果/02_变量构造结果/5公司公告事件/专题研究事件控制变量_公告日期口径.csv"
)

cue_events_out_path = ROOT / '中间结果/04_合并数据/Full_cue_events_after_collapse.csv'
cue_events_out_path.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 参数设置
# =============================================================================

# 1. 分析师共同覆盖过滤参数
DROP_ANALYST_OVERLAP = True
# True: 删除有共同分析师覆盖的公司对
# False: 只合并标记，不删除

ANALYST_OVERLAP_ROOT = ROOT / '中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后'

ANALYST_ID_MODE = 'split'
# 'split': 拆开 AnanmID，按个人分析师识别，推荐主口径
# 'team': 不拆开 AnanmID，按原始 AnanmID 字符串识别

ANALYST_LOOKBACK_DAYS = 365
# 可选：365 或 45

# 2.市值排名数据参数
MARKETCAP_BREAKPOINT_MODE = "prev_month"

# 事件反应日口径
EVENT_DATE_MODE = "next_if_nontrading"
# "next_if_nontrading": 主口径，只调整非交易日
# "strict_next_trading": 稳健性口径，交易日也顺延

VALID_EVENT_DATE_MODES = {"next_if_nontrading", "strict_next_trading"}

if EVENT_DATE_MODE not in VALID_EVENT_DATE_MODES:
    raise ValueError(
        f"EVENT_DATE_MODE 只能是 {VALID_EVENT_DATE_MODES}，当前是 {EVENT_DATE_MODE}"
    )
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


print("正在读取辅助表......")

calendar = pd.read_csv(
    TRD_Cale_td_path,
    encoding="utf-8",
    usecols=["Trddt", "td"],
    parse_dates=["Trddt"],
).sort_values("Trddt")

calendar["td"] = pd.to_numeric(calendar["td"], errors="coerce")

mktcap = pd.read_csv(
    TRD_Dalyr_DGTW_path,
    encoding="utf-8",
    usecols=["Stkcd", "td", "size"],
    dtype={"Stkcd": "string"},
)

mktcap["Stkcd"] = mktcap["Stkcd"].apply(normalize_stock_code)
mktcap["td"] = pd.to_numeric(mktcap["td"], errors="coerce")

mktcap = mktcap.rename(
    columns={
        "td": "td_tm3",
        "size": "mktcap_tm3",
    }
)
mktcap_index = mktcap.set_index(["Stkcd", "td_tm3"])["mktcap_tm3"]
if not mktcap_index.index.is_unique:
    raise ValueError("日度市值表的 Stkcd-td 键不唯一。")
del mktcap
gc.collect()

marketcap_group = pd.read_csv(
    marketcap_group_path,
    encoding="utf-8",
)

marketcap_group["Trdmnt"] = pd.to_datetime(marketcap_group["Trdmnt"])
marketcap_group_merge = marketcap_group.rename(
    columns={"Trdmnt": "Trdmnt_breakpoint"}
)

daily_industry = pd.read_csv(
    industry_path,
    encoding="utf-8",
    usecols=["Stkcd", "td", "IndustryCode"],
    dtype={"Stkcd": "string", "IndustryCode": "string"},
)

daily_industry["Stkcd"] = daily_industry["Stkcd"].apply(normalize_stock_code)
daily_industry["td"] = pd.to_numeric(daily_industry["td"], errors="coerce")

daily_industry = (
    daily_industry
    .dropna(subset=["Stkcd", "td", "IndustryCode"])
    .drop_duplicates(["Stkcd", "td"])
)

industry_index = daily_industry.set_index(["Stkcd", "td"])["IndustryCode"]
del daily_industry
gc.collect()

SC_NetWorkRelationsIndex = pd.read_csv(
    SC_NetWorkRelationsIndex_path,
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

SC_NetWorkRelationsIndex["yearCue"] = SC_NetWorkRelationsIndex["EndDate"].dt.year
SC_NetWorkRelationsIndex["Symbol"] = SC_NetWorkRelationsIndex["Symbol"].apply(normalize_stock_code)

is_a_share_pair = (
    SC_NetWorkRelationsIndex["StateTypeCode"].eq("1")
    & SC_NetWorkRelationsIndex["PSCSymbol"].str.fullmatch(r"[0368][0-9]{5}", na=False)
    & SC_NetWorkRelationsIndex["PSCBusinessRelation"].isin(["1", "2"])
)

supply_chain = SC_NetWorkRelationsIndex.loc[is_a_share_pair].copy()
supply_chain["PSCSymbol"] = supply_chain["PSCSymbol"].apply(normalize_stock_code)

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

supply_chain = supply_chain.loc[
    supply_chain["upstream_stkcd"] != supply_chain["downstream_stkcd"],
    ["yearCue", "upstream_stkcd", "downstream_stkcd"],
].drop_duplicates()

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
    pd.concat([vertical_pairs_forward, vertical_pairs_reverse], ignore_index=True)
    .drop_duplicates()
    .assign(vertical_related=1)
)
vertical_pair_index = vertical_pairs.set_index(
    ["yearCue", "Stkcd", "StkcdRecall"]
).index
del vertical_pairs

id_mode_folder = {
    "split": "拆开AnanmID",
    "team": "不拆开AnanmID",
}

if ANALYST_ID_MODE not in id_mode_folder:
    raise ValueError("ANALYST_ID_MODE 只能是 'split' 或 'team'")

analyst_overlap_path = (
    ANALYST_OVERLAP_ROOT
    / id_mode_folder[ANALYST_ID_MODE]
    / f"Full样本_分析师共同覆盖_滚动{ANALYST_LOOKBACK_DAYS}天.csv"
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

analyst_overlap["Stkcd"] = analyst_overlap["Stkcd"].apply(normalize_stock_code)
analyst_overlap["StkcdRecall"] = analyst_overlap["StkcdRecall"].apply(normalize_stock_code)

analyst_overlap = analyst_overlap.drop_duplicates(
    ["Annodt", "Stkcd", "StkcdRecall"]
)
analyst_exclusion_index = analyst_overlap.loc[
    analyst_overlap["same_analyst_overlap"].notna(),
    ["Annodt", "Stkcd", "StkcdRecall"],
].set_index(["Annodt", "Stkcd", "StkcdRecall"]).index
del analyst_overlap
gc.collect()

print("辅助表读取完成。")


def process_mm_chunk(mm):
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


REUSE_EXISTING_CUE_EVENTS = False
# 复用缓存须手动开启；上游记忆矩阵更新后，应从当前输入重算。
collapsed_parts = []
collapsed = pd.DataFrame()

if REUSE_EXISTING_CUE_EVENTS and cue_events_out_path.exists():
    cue_events_for_panel = pd.read_csv(
        cue_events_out_path,
        encoding="utf-8-sig",
        parse_dates=["date"],
        dtype={"StkcdRecall": "string"},
    )
    cue_events_for_panel["StkcdRecall"] = cue_events_for_panel[
        "StkcdRecall"
    ].apply(normalize_stock_code)
    duplicate_count = cue_events_for_panel.duplicated(
        ["StkcdRecall", "date"]
    ).sum()
    if duplicate_count != 0:
        raise ValueError("缓存的 Full cue 事件表存在 StkcdRecall-date 重复键。")
    print(f"复用已生成的 Full cue 事件表：{cue_events_out_path}")
    print("缓存行数：", len(cue_events_for_panel))
else:
    print("正在分块重算 Full cue 事件......")
    reader = pd.read_csv(
        MemoryMatrixFull_path,
        encoding="utf-8",
        usecols=["Annodt", "Stkcd", "StkcdRecall", "quarterLag1Alt"],
        dtype={
            "Stkcd": "string",
            "StkcdRecall": "string",
        },
        chunksize=1_000_000,
    )

    for chunk_no, chunk in enumerate(reader, start=1):
        lag1 = pd.to_numeric(chunk["quarterLag1Alt"], errors="coerce")
        if not lag1.dropna().isin([0, 1]).all():
            raise ValueError("Full 记忆矩阵的 quarterLag1Alt 不是二值变量。")
        chunk = chunk.loc[lag1.eq(1)].copy()
        if chunk.empty:
            continue
        chunk["Annodt"] = pd.to_datetime(chunk["Annodt"], errors="coerce")
        part = process_mm_chunk(chunk)
        part = part.loc[part["quarterLag1size90plus"].eq(1)]
        if not part.empty:
            collapsed_parts.append(part)
        if len(collapsed_parts) >= 25:
            collapsed_parts = [combine_cue_parts(collapsed_parts)]
        if chunk_no % 10 == 0:
            print(f"已处理记忆矩阵 {chunk_no:,} 百万行")
        del chunk, part
        gc.collect()

    collapsed = combine_cue_parts(collapsed_parts)
    collapsed["date"] = collapsed["event_Trddt"]

    cue_events_for_panel = collapsed.drop(
        columns=["event_Trddt"],
        errors="ignore",
    ).copy()
    cue_events_for_panel = cue_events_for_panel.sort_values(
        ["StkcdRecall", "date"]
    ).reset_index(drop=True)
    cue_events_for_panel.to_csv(
        cue_events_out_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print(f"\n已保存用于后续主面板合并的 cue 事件表：{cue_events_out_path}")
    print("保存行数：", len(cue_events_for_panel))
    print("保存列数：", cue_events_for_panel.shape[1])
    print("quarterLag1size90plus 合计：", cue_events_for_panel["quarterLag1size90plus"].sum())

del calendar
del mktcap_index
del marketcap_group
del marketcap_group_merge
del industry_index
del SC_NetWorkRelationsIndex
del supply_chain
del vertical_pairs_forward
del vertical_pairs_reverse
del vertical_pair_index
del analyst_exclusion_index
del collapsed_parts
del collapsed

gc.collect()

#===============================================================================
# 从股票日度数据开始构建完整主样本，再合并 cue 事件
#===============================================================================
TRD_Dalyr = pd.read_csv(
    TRD_Dalyr_DGTW_path,
    encoding='utf-8',
    usecols=(
        ["Stkcd", "td", "Trddt", "year", "mktcap_tm3",
         "raw_ret_tm3", "raw_ret_tm2", "raw_ret_tm1", "raw_ret_t"]
        + [f"raw_ret_t_t{i}" for i in range(1, 51)]
    ),
    dtype=(
        {"Stkcd": "string", "td": "int32", "year": "float32", "mktcap_tm3": "float32"}
        | {
            column: "float32"
            for column in (
                ["raw_ret_tm3", "raw_ret_tm2", "raw_ret_tm1", "raw_ret_t"]
                + [f"raw_ret_t_t{i}" for i in range(1, 51)]
            )
        }
    ),
    parse_dates=['Trddt']
)

TRD_Dalyr['Stkcd'] = TRD_Dalyr['Stkcd'].apply(normalize_stock_code)

TRD_Dalyr = TRD_Dalyr.rename(
    columns={'Stkcd':'StkcdRecall'}
).drop(columns=['IndustryCode'], errors='ignore')

# 将原始收益率变量改成和原文一致的命名
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

TRD_Dalyr = TRD_Dalyr.rename(columns=raw_return_rename)


# 标记被提示公司自身财报公告日，用于后续回归控制
IAR_Rept = pd.read_csv(
    IAR_Rept_path,
    usecols=['Stkcd','Accper','Annodt'],
    parse_dates=['Accper','Annodt']
)

IAR_Rept['Stkcd'] = IAR_Rept['Stkcd'].apply(normalize_stock_code)
IAR_Rept = IAR_Rept.rename(columns={
    'Stkcd':'StkcdRecall'
})

# 把自身公告自然日映射到用于收益反应的事件交易日
calendar = (
    TRD_Dalyr[['Trddt', 'td']]
    .drop_duplicates()
    .sort_values('Trddt')
)

ann_key = (
    IAR_Rept[['Annodt']]
    .dropna()
    .drop_duplicates()
    .sort_values('Annodt')
)

ann_key = pd.merge_asof(
    ann_key,
    calendar,
    left_on='Annodt',
    right_on='Trddt',
    direction='forward',
    allow_exact_matches=(EVENT_DATE_MODE == "next_if_nontrading")
)

# 生成 股票 × 事件交易日 × ownEA
own_ea = (
    IAR_Rept
    .merge(ann_key, on='Annodt', how='left', validate='m:1')
    .dropna(subset=['Trddt'])
    [['StkcdRecall', 'Trddt']]
    .drop_duplicates()
)

own_ea['ownEA'] = 1

# 合并回日度面板
TRD_Dalyr = TRD_Dalyr.merge(
    own_ea,
    on=['StkcdRecall', 'Trddt'],
    how='left',
    validate='1:1'
)

TRD_Dalyr['ownEA'] = TRD_Dalyr['ownEA'].fillna(0).astype('int8')

print("ownEA为1的数量：",(TRD_Dalyr['ownEA']==1).sum())
print("ownEA为1的比例：",((TRD_Dalyr['ownEA']==1).sum())/len(TRD_Dalyr))
unmatched_own_ea = own_ea.merge(
    TRD_Dalyr[['StkcdRecall', 'Trddt']],
    on=['StkcdRecall', 'Trddt'],
    how='left',
    indicator=True
)

unmatched_own_ea = unmatched_own_ea.loc[
    unmatched_own_ea['_merge'].eq('left_only')
].copy()

print('未匹配自身公告事件数：', len(unmatched_own_ea))
print(unmatched_own_ea['Trddt'].dt.year.value_counts().sort_index())


# 生成被提示公司自身盈余公告在事件日前后 10 个交易日内的控制变量
# 这些变量用于控制自身公告窗口对收益的污染
#===============================================================================
# 构造 own EA 前后 10 个交易日 dummy
# 当前 TRD_Dalyr 层级：StkcdRecall × Trddt
#===============================================================================

# 先检查主面板唯一性
dup_n = TRD_Dalyr.duplicated(['StkcdRecall', 'td']).sum()
print("TRD_Dalyr 中 StkcdRecall-td 重复数：", dup_n)

if dup_n != 0:
    raise ValueError("TRD_Dalyr 中 StkcdRecall-td 不唯一，请先检查是否存在重复交易日记录。")

# 提取自身公告所在的 股票-交易日序号
own_ea_td = (
    TRD_Dalyr.loc[TRD_Dalyr['ownEA'].eq(1), ['StkcdRecall', 'td']]
    .drop_duplicates()
    .copy()
)

print("ownEA 事件数：", len(own_ea_td))

for i in range(1, 11):
    # ------------------------------------------------------------
    # ownEA_tp{i}: 当前日后 i 个交易日会发生 ownEA
    # 如果公告日在 td=100，那么 td=100-i 的行 ownEA_tp{i}=1
    # ------------------------------------------------------------
    future_ea = own_ea_td.copy()
    future_ea['td'] = future_ea['td'] - i
    future_ea = (
        future_ea
        .drop_duplicates(['StkcdRecall', 'td'])
        .assign(**{f'ownEA_tp{i}': 1})
    )

    TRD_Dalyr = TRD_Dalyr.merge(
        future_ea,
        on=['StkcdRecall', 'td'],
        how='left',
        validate='1:1'
    )

    TRD_Dalyr[f'ownEA_tp{i}'] = (
        TRD_Dalyr[f'ownEA_tp{i}']
        .fillna(0)
        .astype('int8')
    )

    # ------------------------------------------------------------
    # ownEA_tm{i}: 当前日前 i 个交易日发生过 ownEA
    # 如果公告日在 td=100，那么 td=100+i 的行 ownEA_tm{i}=1
    # ------------------------------------------------------------
    past_ea = own_ea_td.copy()
    past_ea['td'] = past_ea['td'] + i
    past_ea = (
        past_ea
        .drop_duplicates(['StkcdRecall', 'td'])
        .assign(**{f'ownEA_tm{i}': 1})
    )

    TRD_Dalyr = TRD_Dalyr.merge(
        past_ea,
        on=['StkcdRecall', 'td'],
        how='left',
        validate='1:1'
    )

    TRD_Dalyr[f'ownEA_tm{i}'] = (
        TRD_Dalyr[f'ownEA_tm{i}']
        .fillna(0)
        .astype('int8')
    )

print("ownEA 前后窗口变量生成完成。")

own_ea_window_cols = (
    [f'ownEA_tp{i}' for i in range(1, 11)]
    + ['ownEA']
    + [f'ownEA_tm{i}' for i in range(1, 11)]
)

print(TRD_Dalyr[own_ea_window_cols].sum())


#===============================================================================
# 合并被提示公司业绩预告/快报与专题事件控制变量
#===============================================================================

performance_forecast_controls = pd.read_csv(
    performance_forecast_controls_path,
    encoding="utf-8-sig",
    usecols=["StkcdRecall", "Trddt", "ownPerformanceForecastBrief"],
    dtype={"StkcdRecall": "string"},
    parse_dates=["Trddt"]
)

other_event_controls = pd.read_csv(
    other_event_controls_path,
    encoding="utf-8-sig",
    usecols=["StkcdRecall", "Trddt", "ownOtherEventER"],
    dtype={"StkcdRecall": "string"},
    parse_dates=["Trddt"]
)

for controls_name, controls_df in [
    ("业绩预告快报", performance_forecast_controls),
    ("专题事件", other_event_controls),
]:
    controls_df["StkcdRecall"] = controls_df["StkcdRecall"].apply(
        normalize_stock_code
    )

    missing_key = (
        controls_df["StkcdRecall"].eq("") | controls_df["Trddt"].isna()
    ).sum()
    duplicate_key = controls_df.duplicated(
        ["StkcdRecall", "Trddt"]
    ).sum()
    print(
        f"{controls_name}控制表缺失键数：{missing_key}，重复键数：{duplicate_key}"
    )

    if missing_key != 0 or duplicate_key != 0:
        raise ValueError(
            f"{controls_name}控制表在 StkcdRecall-Trddt 层面存在缺失或重复，请先检查。"
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
    pd.to_numeric(other_event_controls["ownOtherEventER"], errors="raise")
    .fillna(0)
    .astype("int8")
)

# 两张表均已将非交易日公告顺延至下一交易日；直接按被提示公司和交易日合并。
TRD_Dalyr = TRD_Dalyr.merge(
    performance_forecast_controls,
    on=["StkcdRecall", "Trddt"],
    how="left",
    validate="1:1"
)
TRD_Dalyr = TRD_Dalyr.merge(
    other_event_controls,
    on=["StkcdRecall", "Trddt"],
    how="left",
    validate="1:1"
)

TRD_Dalyr[["ownPerformanceForecastBrief", "ownOtherEventER"]] = (
    TRD_Dalyr[["ownPerformanceForecastBrief", "ownOtherEventER"]]
    .fillna(0)
    .astype("int8")
)

# 保持现有回归变量接口。注意 ownMajorDisclosure 现由十类专题事件并集构造，
# 与旧文字识别表的“重大公告窄口径”定义不同。
TRD_Dalyr["ownMajorDisclosure"] = TRD_Dalyr["ownOtherEventER"].astype("int8")

TRD_Dalyr["ownPerformanceForecast"] = (
    TRD_Dalyr["ownPerformanceForecastBrief"]
    .astype("int8")
)

print("ownMajorDisclosure=1 数量：", TRD_Dalyr["ownMajorDisclosure"].sum())
print("ownPerformanceForecast=1 数量：", TRD_Dalyr["ownPerformanceForecast"].sum())
print(
    "ownMajorDisclosure 与 ownPerformanceForecast 同时为1数量：",
    (
        TRD_Dalyr["ownMajorDisclosure"].eq(1)
        & TRD_Dalyr["ownPerformanceForecast"].eq(1)
    ).sum()
)


#===============================================================================
# 合并上游已构造的真实交易日新闻控制变量
#===============================================================================

AES_AEV_controls = pd.read_csv(
    AES_AEV_controls_path,
    encoding='utf-8-sig',
    usecols=[
        'Stkcd',
        'date',
        'AES',
        'AEV',
        'aesIncrease',
        'aesDecrease',
        'logIncreaseAEV',
        'logDecreaseAEV',
    ],
    dtype={'Stkcd': 'string'},
    parse_dates=['date']
)

AES_AEV_controls['Stkcd'] = AES_AEV_controls['Stkcd'].apply(
    normalize_stock_code
)

AES_AEV_controls = AES_AEV_controls.rename(columns={
    'Stkcd': 'StkcdRecall',
    'date': 'Trddt'
})

TRD_Dalyr['Trddt'] = pd.to_datetime(TRD_Dalyr['Trddt'])
AES_AEV_controls['Trddt'] = pd.to_datetime(AES_AEV_controls['Trddt'])

# 检查上游控制表是否在 股票-真实交易日 层面唯一
dup_news = AES_AEV_controls.duplicated(['StkcdRecall', 'Trddt']).sum()
print('真实交易日新闻控制表中 StkcdRecall-Trddt 重复数：', dup_news)

if dup_news != 0:
    raise ValueError(
        '真实交易日新闻控制表在 StkcdRecall-Trddt 层面不唯一，请先检查上游构造。'
    )

news_numeric_cols = [
    'AES',
    'AEV',
    'aesIncrease',
    'aesDecrease',
    'logIncreaseAEV',
    'logDecreaseAEV',
]
AES_AEV_controls[news_numeric_cols] = AES_AEV_controls[news_numeric_cols].apply(
    pd.to_numeric,
    errors='raise',
)

if (AES_AEV_controls['AEV'].dropna() < 0).any():
    raise ValueError('真实交易日新闻控制表中的 AEV 不应出现负数。')

print('真实交易日新闻控制表行数：', len(AES_AEV_controls))

# 正式合并
TRD_Dalyr = TRD_Dalyr.merge(
    AES_AEV_controls,
    on=['StkcdRecall', 'Trddt'],
    how='left',
    validate='1:1'
)
del AES_AEV_controls
gc.collect()

print('合并真实交易日 AES/AEV 控制变量后 TRD_Dalyr 表结构：', TRD_Dalyr.shape)
for news_col in news_numeric_cols:
    print(f'{news_col} 非缺失数量：', TRD_Dalyr[news_col].notna().sum())

try:
    cue_panel = cue_events_for_panel.copy()
except NameError:
    cue_panel = pd.read_csv(
        cue_events_out_path,
        encoding="utf-8-sig",
        parse_dates=["date"],
        dtype={"StkcdRecall": "string"}
    )
    cue_panel["StkcdRecall"] = cue_panel["StkcdRecall"].apply(normalize_stock_code)

TRD_Dalyr["Trddt"] = pd.to_datetime(TRD_Dalyr["Trddt"])
cue_panel["date"] = pd.to_datetime(cue_panel["date"])

# 检查主面板唯一性
dup_main = TRD_Dalyr.duplicated(["StkcdRecall", "Trddt"]).sum()
print("TRD_Dalyr 中 StkcdRecall-Trddt 重复数：", dup_main)

if dup_main != 0:
    raise ValueError("TRD_Dalyr 中 StkcdRecall-Trddt 不唯一，请先检查主面板。")

# 检查 cue 表唯一性
dup_cue = cue_panel.duplicated(["StkcdRecall", "date"]).sum()
print("cue_panel 中 StkcdRecall-date 重复数：", dup_cue)

if dup_cue != 0:
    raise ValueError("cue_panel 中 StkcdRecall-date 不唯一，请检查 collapse 步骤。")

print("cue 事件总数：", len(cue_panel))

# 正式合并
TRD_Dalyr = TRD_Dalyr.merge(
    cue_panel,
    left_on=["StkcdRecall", "Trddt"],
    right_on=["StkcdRecall", "date"],
    how="left",
    validate="1:1"
)
matched_cue_count = TRD_Dalyr["date"].notna().sum()
print("cue 能匹配到日度主面板数量：", matched_cue_count)
print("cue 匹配不到日度主面板数量：", len(cue_panel) - matched_cue_count)
del cue_panel
TRD_Dalyr = TRD_Dalyr.drop(columns=["date"], errors="ignore")

print("合并 cue 事件后的 TRD_Dalyr 表结构：", TRD_Dalyr.shape)
print("合并后 StkcdRecall-Trddt 重复数：",
      TRD_Dalyr.duplicated(["StkcdRecall", "Trddt"]).sum())

#===============================================================================
# 没有 cue 的日期，将 cue dummy 缺失值设为 0
#===============================================================================

cue_dummy_cols = [
    "quarterLag1size90plus",
]

cue_dummy_cols = [c for c in cue_dummy_cols if c in TRD_Dalyr.columns]

TRD_Dalyr[cue_dummy_cols] = (
    TRD_Dalyr[cue_dummy_cols]
    .fillna(0)
    .astype("int8")
)

print("cue dummy 缺失值已设为 0")
print(TRD_Dalyr[cue_dummy_cols].sum().sort_values(ascending=False).head(20))

#===============================================================================
# 合并特征调整收益率
#===============================================================================
adj = pd.read_csv(
    adjusted_return_path,
    encoding="utf-8-sig",
    dtype=(
        {"Stkcd": "string", "portfolioNumber": "float32"}
        | {
            column: "float32"
            for column in (
                ["chAdj_ret_tm3", "chAdj_ret_tm2", "chAdj_ret_tm1", "chAdj_ret_t"]
                + [f"chAdj_ret_t_t{i}" for i in range(1, 51)]
            )
        }
    ),
    parse_dates=["Trddt"]
)

adj["Stkcd"] = adj["Stkcd"].apply(normalize_stock_code)

adj = adj.rename(columns={
    "Stkcd": "StkcdRecall",
    "chAdj_ret_tm3": "chAdjreturn_tm3_Recall",
    "chAdj_ret_tm2": "chAdjreturn_tm2_Recall",
    "chAdj_ret_tm1": "chAdjreturn_tm1_Recall",
    "chAdj_ret_t": "chAdjreturn_t_Recall",
})

rename_forward = {
    f"chAdj_ret_t_t{i}": f"chAdjreturn_t_tp{i}_Recall"
    for i in range(1, 51)
}
adj = adj.rename(columns=rename_forward)

keep_adj_cols = (
    ["StkcdRecall", "Trddt", "portfolioNumber"]
    + [c for c in adj.columns if c.startswith("chAdjreturn_")]
)

adj = adj[keep_adj_cols].copy()

dup_adj = adj.duplicated(["StkcdRecall", "Trddt"]).sum()
print("DGTW特征调整收益率 StkcdRecall-Trddt 重复数：", dup_adj)
if dup_adj != 0:
    raise ValueError("DGTW特征调整收益率在 StkcdRecall-Trddt 层面不唯一。")

TRD_Dalyr = TRD_Dalyr.merge(
    adj,
    on=["StkcdRecall", "Trddt"],
    how="left",
    validate="1:1",
)
del adj
gc.collect()

print("合并特征调整收益率后表结构：", TRD_Dalyr.shape)
print("chAdjreturn_t_Recall 缺失比例：", TRD_Dalyr["chAdjreturn_t_Recall"].isna().mean())
print("chAdjreturn_t_tp1_Recall 缺失比例：", TRD_Dalyr["chAdjreturn_t_tp1_Recall"].isna().mean())
print("chAdjreturn_t_tp10_Recall 缺失比例：", TRD_Dalyr["chAdjreturn_t_tp10_Recall"].isna().mean())

#===============================================================================
# 最终整理样本并进行缩尾
#===============================================================================
# 缩尾方式是：全样本 1% winsorize（缩尾），不是按年份缩尾。
# 也就是把变量低于第 1 百分位的值替换为第 1 百分位，把高于第 99 百分位的值替换为第 99 百分位，
# 原变量保留，新生成 _w 变量。
print("开始缩尾......")
def winsorize_limits(s, p=0.01):
    """
    只计算缩尾阈值，不立刻生成缩尾后的整列。
    """
    lower = s.quantile(p)
    upper = s.quantile(1 - p)
    return lower, upper


def add_winsor_col(df, col):
    """
    使用预先在全样本上算好的缩尾阈值生成 _w 变量。
    """
    lower, upper = winsor_limits[col]
    df[f"{col}_w"] = df[col].clip(lower=lower, upper=upper)


# 对主要因变量进行 1% winsorize
dep_vars = [
    "chAdjreturn_tm3_Recall",
    "chAdjreturn_tm2_Recall",
    "chAdjreturn_tm1_Recall",
    "chAdjreturn_t_Recall",
]

dep_vars += [
    f"chAdjreturn_t_tp{i}_Recall"
    for i in range(1, 11)
]
# 注意：阈值仍然在筛基准样本前的全样本上计算

# 原文缩尾变量清单：收益率窗口 + cue 盈余惊喜
winsor_source_cols = [
    col
    for col in dep_vars 
    if col in TRD_Dalyr.columns
]

winsor_limits = {}
for col in winsor_source_cols:
    winsor_limits[col] = winsorize_limits(TRD_Dalyr[col], p=0.01)

# 筛基准样本前，只先生成后面马上要用的两个因变量缩尾列
for col in ["chAdjreturn_t_tp1_Recall", "chAdjreturn_t_tp10_Recall"]:
    if col in winsor_limits:
        add_winsor_col(TRD_Dalyr, col)

# 生成日历季度和年度-季度固定效应变量
TRD_Dalyr["quarter"] = TRD_Dalyr["Trddt"].dt.quarter
TRD_Dalyr["yearQuarter"] = TRD_Dalyr["Trddt"].dt.to_period("Q")

# 按被提示公司和日期排序，并把核心标识变量放在前面
front_cols = ["StkcdRecall", "Trddt"]
other_cols = [c for c in TRD_Dalyr.columns if c not in front_cols]

TRD_Dalyr = (
    TRD_Dalyr[front_cols + other_cols]
    .sort_values(["StkcdRecall", "Trddt"])
    .reset_index(drop=True)
)
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


own_ea_cols = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)

own_disclosure_cols = [
    "ownMajorDisclosure",
    "ownPerformanceForecast",
]

baseline_rhs = (
    ["quarterLag1size90plus"]
    + own_ea_cols
    + own_disclosure_cols
    + ["aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV"]
)

print("开始筛选 [t, t+1] 基准回归样本......")
mask_tp1 = baseline_regression_sample_mask(
    TRD_Dalyr,
    dep_col="chAdjreturn_t_tp1_Recall_w",
    rhs_cols=baseline_rhs,
    fe_cols=["StkcdRecall", "yearQuarter"],
    cluster_cols=["StkcdRecall", "Trddt"],
)

TRD_Dalyr = TRD_Dalyr.loc[mask_tp1].copy()
print("[t, t+1] 筛选后样本量：", len(TRD_Dalyr))

print("开始筛选 [t, t+10] 基准回归样本......")
mask_tp10 = baseline_regression_sample_mask(
    TRD_Dalyr,
    dep_col="chAdjreturn_t_tp10_Recall_w",
    rhs_cols=baseline_rhs,
    fe_cols=["StkcdRecall", "yearQuarter"],
    cluster_cols=["StkcdRecall", "Trddt"],
)

TRD_Dalyr = TRD_Dalyr.loc[mask_tp10].copy()
print("[t, t+10] 筛选后样本量：", len(TRD_Dalyr))

#===============================================================================
# 基准样本筛选后，补生成其余缩尾变量
#===============================================================================

print("基准样本筛选后补生成其余缩尾变量......")

for col in winsor_source_cols:
    w_col = f"{col}_w"
    if col in TRD_Dalyr.columns and w_col not in TRD_Dalyr.columns:
        add_winsor_col(TRD_Dalyr, col)

print("最终同时进入两个基准回归的样本量：", len(TRD_Dalyr))
print("最终 cue 数量：", TRD_Dalyr["quarterLag1size90plus"].sum())

#===============================================================================
# 最终输出前整理变量
#===============================================================================

TRD_Dalyr["marketCap_Minus3_Recall"] = TRD_Dalyr["mktcap_tm3"]

expected_return_cols = (
    ["return_tm3_Recall", "return_tm2_Recall", "return_tm1_Recall", "return_t_Recall"]
    + [f"return_t_tp{i}_Recall" for i in range(1, 51)]
)

missing_return_cols = [
    c for c in expected_return_cols
    if c not in TRD_Dalyr.columns
]

if missing_return_cols:
    raise ValueError(
        "先把 raw_ret_* 重命名为 return_*_Recall，再删除 raw_ret_* 辅助列。"
        f" 当前缺少：{missing_return_cols[:10]}"
    )

drop_cols_final = [
    "AES",
    "AEV",
    "ownPerformanceForecastBrief",
    "ownOtherEventER",
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
    c for c in TRD_Dalyr.columns
    if c.startswith("raw_ret_")
]

drop_cols_final = list(dict.fromkeys(drop_cols_final))
drop_cols_existing = [c for c in drop_cols_final if c in TRD_Dalyr.columns]

print("最终输出前删除中间变量数量：", len(drop_cols_existing))
print(drop_cols_existing)

TRD_Dalyr = TRD_Dalyr.drop(columns=drop_cols_existing)

# 把核心标识变量放在前面
front_cols = ["StkcdRecall", "Trddt", "td", "year", "quarter", "yearQuarter"]
front_cols = [c for c in front_cols if c in TRD_Dalyr.columns]
other_cols = [c for c in TRD_Dalyr.columns if c not in front_cols]

TRD_Dalyr = (
    TRD_Dalyr[front_cols + other_cols]
    .sort_values(["StkcdRecall", "Trddt"])
    .reset_index(drop=True)
)

#===============================================================================
# 输出最终文件
#===============================================================================

print("正在输出 Full 样本最终回归面板......")

TRD_Dalyr.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d"
)

print("Full 样本最终回归面板输出成功！")
print("输出路径：", output_path)
print("最终表结构：", TRD_Dalyr.shape)
