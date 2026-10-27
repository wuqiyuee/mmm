import pandas as pd
import numpy as np
from dateutil.relativedelta import * 
from pathlib import Path
from tqdm import tqdm
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

ROOT = Path(__file__).resolve().parents[2]

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

# 3.事件反应日口径
EVENT_DATE_MODE = "next_if_nontrading"
# "next_if_nontrading": 主口径，只调整非交易日
# "strict_next_trading": 稳健性口径，交易日也顺延

VALID_EVENT_DATE_MODES = {"next_if_nontrading", "strict_next_trading"}

if EVENT_DATE_MODE not in VALID_EVENT_DATE_MODES:
    raise ValueError(
        f"EVENT_DATE_MODE 只能是 {VALID_EVENT_DATE_MODES}，当前是 {EVENT_DATE_MODE}"
    )

#===============================================================================
# 读取记忆矩阵
#===============================================================================

print("正在统计全样本记忆矩阵行数，用于显示读取进度......")
chunk_size = 500_000

with open(MemoryMatrixFull_path, "r", encoding="utf-8") as f:
    total_rows = max(sum(1 for _ in f) - 1, 0)

print("正在分块读取全样本记忆矩阵......")
mm_chunks = []

reader = pd.read_csv(
    MemoryMatrixFull_path,
    encoding="utf-8",
    parse_dates=["Annodt", "Accper"],
    chunksize=chunk_size,
)

with tqdm(total=total_rows, desc="读取全样本记忆矩阵", unit="行") as pbar:
    for chunk in reader:
        mm_chunks.append(chunk)
        pbar.update(len(chunk))

mm = pd.concat(mm_chunks, ignore_index=True) if mm_chunks else pd.DataFrame()
# 旧方案最终仍把全部分块拼回内存；全样本可能耗尽内存，正式复跑请用分块版。

print(f"全样本记忆矩阵读取成功！共 {len(mm):,} 行")

mm['Stkcd'] = mm['Stkcd'].apply(normalize_stock_code)
mm['StkcdRecall'] = mm['StkcdRecall'].apply(normalize_stock_code)

# 1.把公告自然日 Annodt 映射到事件交易日，用于匹配市值
calendar = pd.read_csv(
    TRD_Cale_td_path,
    encoding="utf-8",
    usecols=["Trddt", "td"],
    parse_dates=["Trddt"],
).sort_values("Trddt")

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


# 2.读取日度市值

print("正在读取市值数据......")
mktcap = pd.read_csv(
    TRD_Dalyr_DGTW_path,
    encoding='utf-8',
    usecols=['Stkcd', 'td', 'size', 'sizew'],
    dtype={'Stkcd': 'string'}
)
print("市值数据读取成功！")

mktcap['Stkcd'] = mktcap['Stkcd'].apply(normalize_stock_code)
mktcap['td'] = pd.to_numeric(mktcap['td'], errors='coerce')

mktcap = mktcap.rename(
    columns={
        'td': 'td_tm3',
        'size': 'mktcap_tm3',
        'sizew': 'float_mktcap_tm3'
    }
)

# 3. 为了省内存，先对 MemoryMatrixFull 中需要合并的 Stkcd-td_tm3 去重
mktcap_key = (
    mm[['Stkcd', 'td_tm3']]
    .drop_duplicates()
    .merge(
        mktcap[['Stkcd', 'td_tm3', 'mktcap_tm3', 'float_mktcap_tm3']],
        how='left',
        on=['Stkcd', 'td_tm3'],
        validate='m:1'
    )
)

# 4. 合并回完整 MemoryMatrixPattern
mm = mm.merge(
    mktcap_key,
    how='left',
    on=['Stkcd', 'td_tm3'],
    validate='m:1'
)

print('合并 t-3 市值后行数：', len(mm))
print('t-3 总市值缺失数：', mm['mktcap_tm3'].isna().sum())
print('t-3 总市值缺失比例：', mm['mktcap_tm3'].isna().mean())

# 合并市值排名数据

# 本文在复现过程中发现，原始代码在识别“大市值 cue 公司”时可能存在前视信息使用问题。
# 具体而言，代码先基于 CRSP 月度数据按 yearMonth 计算 NYSE 股票市值分位数断点，
# 随后将公告日 ANNDATS_ACT 所在月份的市值断点合并至日度样本，并以公告日前第 3 个交易日的
# 公司市值 marketCap_Minus3 是否超过当月 marketCap_90th_percentile 来定义大市值 cue 公司。
# 由于月度市值断点基于月末市值计算，对于发生在月初或月中的公告而言，当月月末断点在公告日时尚不可观测，
# 因此该处理可能引入前视偏差。为避免使用未来信息，
# 本文在复现中改用公告日前最近一个已知月末，即上月末的市值分位数断点，作为大市值公司划分标准。
marketcap_group = pd.read_csv(
    marketcap_group_path,
    encoding='utf-8',
) 

mm["event_Trddt"] = pd.to_datetime(mm["event_Trddt"])
marketcap_group["Trdmnt"] = pd.to_datetime(marketcap_group["Trdmnt"])

# 选择本月和上月两种模式
if MARKETCAP_BREAKPOINT_MODE == "prev_month":
    mm["Trdmnt_for_breakpoint"] = (
        mm["event_Trddt"]
        .dt.to_period("M")
        .sub(1)
        .dt.to_timestamp()
    )
elif MARKETCAP_BREAKPOINT_MODE == "same_month":
    mm["Trdmnt_for_breakpoint"] = (
        mm["event_Trddt"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )
else:
    raise ValueError("MARKETCAP_BREAKPOINT_MODE 只能是 'prev_month' 或 'same_month'")

# 为避免右表 key 名混淆，先改名
marketcap_group_merge = marketcap_group.rename(
    columns={"Trdmnt": "Trdmnt_breakpoint"}
)

# 合并 marketcap_group 
mm = mm.merge(
    marketcap_group_merge,
    left_on="Trdmnt_for_breakpoint",
    right_on="Trdmnt_breakpoint",
    how="left",
    validate="m:1"
)

print(mm["Trdmnt_breakpoint"].isna().sum())
print(mm.loc[mm["Trdmnt_breakpoint"].isna(), ["event_Trddt", "Trdmnt_breakpoint"]].head())
print("合并断点之后的行数",len(mm))

# 删除市值/市值断点存在缺失值的（这里暗含了对样本的清洗）
mm = mm.dropna(subset=[
    "mktcap_tm3",
    "marketCap_90th_percentile",
    "marketCap_50th_percentile",
    "marketCap_10th_percentile",
]).copy()
print("删除t-3市值或者市值断点存在缺失值的观测值后剩余的样本数",len(mm))

#===============================================================================
# 筛选符合条件的 cue-recall 公司对
#===============================================================================

mm_key = mm[['Annodt', 'event_td', 'Stkcd', 'StkcdRecall']].copy()
mm_key['Stkcd'] = mm_key['Stkcd'].apply(normalize_stock_code)
mm_key['StkcdRecall'] = mm_key['StkcdRecall'].apply(normalize_stock_code)

# 排除同行业的公司对(需要先匹配行业信息)
daily_industry = pd.read_csv(
    industry_path,
    encoding='utf-8',
    usecols=['Stkcd','td','IndustryCode'],
     dtype={'Stkcd': 'string', 'IndustryCode': 'string'}
)

daily_industry['Stkcd'] = daily_industry['Stkcd'].apply(normalize_stock_code)
daily_industry['td'] = pd.to_numeric(daily_industry['td'], errors='coerce')

daily_industry = (
    daily_industry
    .dropna(subset=['Stkcd', 'td', 'IndustryCode'])
    .drop_duplicates(['Stkcd', 'td'])
)

cue_industry = daily_industry.rename(columns={
    'td': 'event_td',
    'IndustryCode': 'cue_industry_code'
})

recall_industry = daily_industry.rename(columns={
    'Stkcd': 'StkcdRecall',
    'td': 'event_td',
    'IndustryCode': 'recall_industry_code'
})

mm_key = mm_key.merge(
    cue_industry[['Stkcd', 'event_td', 'cue_industry_code']],
    on=['Stkcd', 'event_td'],
    how='left',
    validate='m:1'
)# 合并提示公司的行业

mm_key = mm_key.merge(
    recall_industry[['StkcdRecall', 'event_td', 'recall_industry_code']],
    on=['StkcdRecall', 'event_td'],
    how='left',
    validate='m:1'
)# 合并被提示公司的行业

mm_key['same_industry'] = (
    mm_key['cue_industry_code'].notna()
    & mm_key['recall_industry_code'].notna()
    & (mm_key['cue_industry_code'] == mm_key['recall_industry_code'])
)

print('同行业公司对数量：', mm_key['same_industry'].sum())
print('同行业公司对比例：', mm_key['same_industry'].mean())

mm_key = mm_key.loc[~mm_key['same_industry']].copy()

# 删除上下游垂直相关公司对
#（这里仅排除一级供应链关系，后续如果需要的话还可以排除二级供应链关系作为稳健性检验）

mm_key['yearCue'] = mm_key['Annodt'].dt.year

SC_NetWorkRelationsIndex = pd.read_csv(
    SC_NetWorkRelationsIndex_path,
    encoding='utf-8',
    usecols=[
        'Symbol',
        'EndDate',
        'StateTypeCode',
        'PSCBusinessRelation',
        'PSCSymbol'
    ],
    dtype={
        'Symbol': 'string',
        'StateTypeCode': 'string',
        'PSCBusinessRelation': 'string',
        'PSCSymbol': 'string',
    },    
    parse_dates=['EndDate']
)

# 一级供应链字段：每一行都有
# 二级供应链字段：只有部分行有

SC_NetWorkRelationsIndex['yearCue'] = SC_NetWorkRelationsIndex['EndDate'].dt.year

SC_NetWorkRelationsIndex['Symbol']=SC_NetWorkRelationsIndex['Symbol'].apply(normalize_stock_code)
# 其实这里已经都是6位数字代码 

is_a_share_pair = (
    SC_NetWorkRelationsIndex['StateTypeCode'].eq('1') # 筛选出合并报表
    & SC_NetWorkRelationsIndex['PSCSymbol'].str.fullmatch(r'[0368][0-9]{5}', na=False)
    # 排除美股、港股市场的（这里不能进行6位数字填充标准化，因为5位数的是港股）
    & SC_NetWorkRelationsIndex['PSCBusinessRelation'].isin(['1', '2']) 
    # 实际也只有这两个取值
) # 筛选出A股中合并报表的数据

supply_chain = SC_NetWorkRelationsIndex.loc[is_a_share_pair].copy()

# 转化为有向关系
# PSCBusinessRelation == 1: Symbol 的客户是 PSCSymbol，所以 Symbol -> PSCSymbol
# PSCBusinessRelation == 2: Symbol 的供应商是 PSCSymbol，所以 PSCSymbol -> Symbol

supply_chain['upstream_stkcd'] = np.where(
    supply_chain['PSCBusinessRelation'].eq('1'),
    supply_chain['Symbol'],
    supply_chain['PSCSymbol']
)

supply_chain['downstream_stkcd'] = np.where(
    supply_chain['PSCBusinessRelation'].eq('1'),
    supply_chain['PSCSymbol'],
    supply_chain['Symbol']
) 

supply_chain = supply_chain.loc[
    supply_chain['upstream_stkcd'] != supply_chain['downstream_stkcd'],
    ['yearCue', 'upstream_stkcd', 'downstream_stkcd']
].drop_duplicates() # 删除自己和自己是上下游的配对，其实合并报表中并没有这种情况（母公司报表中有一条）


# 只要 A 和 B 任一方向存在供应链关系，就同时生成 A-B 和 B-A 两条可匹配记录。
vertical_pairs_forward = supply_chain.rename(columns={
    'upstream_stkcd': 'Stkcd',
    'downstream_stkcd': 'StkcdRecall'
})[['yearCue', 'Stkcd', 'StkcdRecall']]

vertical_pairs_reverse = supply_chain.rename(columns={
    'downstream_stkcd': 'Stkcd',
    'upstream_stkcd': 'StkcdRecall'
})[['yearCue', 'Stkcd', 'StkcdRecall']]

vertical_pairs = (
    pd.concat([vertical_pairs_forward, vertical_pairs_reverse], ignore_index=True)
    .drop_duplicates()
    .assign(vertical_related=1)
)

mm_key = mm_key.merge(
    vertical_pairs,
    on=['yearCue', 'Stkcd', 'StkcdRecall'],
    how='left',
    validate='m:1'# 同一对公司可能在同一年有很多公告日
) 

print('上下游垂直相关公司对数量：', mm_key['vertical_related'].eq(1).sum())
print('上下游垂直相关公司对比例：', mm_key['vertical_related'].eq(1).mean())
print('供应链关系公司对-年度数：', len(vertical_pairs))
print(vertical_pairs.head())
print(mm_key.loc[
    mm_key['vertical_related'].eq(1),
    ['Annodt', 'yearCue', 'Stkcd', 'StkcdRecall']
].head(20))

mm_key = mm_key.loc[mm_key['vertical_related'].isna()].copy()

mm_key = mm_key.drop(
    columns=['yearCue', 'vertical_related'],
    errors='ignore'
)


# 删除有共同分析师覆盖的公司对
id_mode_folder = {
    'split': '拆开AnanmID',
    'team': '不拆开AnanmID',
}

if ANALYST_ID_MODE not in id_mode_folder:
    raise ValueError("ANALYST_ID_MODE 只能是 'split' 或 'team'")

analyst_overlap_path = (
    ANALYST_OVERLAP_ROOT
    / id_mode_folder[ANALYST_ID_MODE]
    / f'Full样本_分析师共同覆盖_滚动{ANALYST_LOOKBACK_DAYS}天.csv'
)

analyst_overlap = pd.read_csv(
    analyst_overlap_path,
    encoding='utf-8-sig',
    usecols=[
        'Annodt',
        'Stkcd',
        'StkcdRecall',
        'same_analyst_overlap',
        'n_common_analysts',
    ],
    dtype={
        'Stkcd': 'string',
        'StkcdRecall': 'string',
        'same_analyst_overlap': 'Int64',
        'n_common_analysts': 'Int64',
    },
    parse_dates=['Annodt']
)

analyst_overlap['Stkcd'] = analyst_overlap['Stkcd'].apply(normalize_stock_code)
analyst_overlap['StkcdRecall'] = analyst_overlap['StkcdRecall'].apply(normalize_stock_code)

analyst_overlap = (
    analyst_overlap
    .drop_duplicates(['Annodt', 'Stkcd', 'StkcdRecall'])
)

mm_key = mm_key.merge(
    analyst_overlap,
    on=['Annodt', 'Stkcd', 'StkcdRecall'],
    how='left',
    validate='m:1'
)

analyst_overlap_count = mm_key['same_analyst_overlap'].notna().sum()
analyst_overlap_ratio = mm_key['same_analyst_overlap'].notna().mean()

print('分析师共同覆盖公司对数量：', analyst_overlap_count)
print('分析师共同覆盖公司对比例：', analyst_overlap_ratio)

if DROP_ANALYST_OVERLAP:
    mm_key = mm_key.loc[mm_key['same_analyst_overlap'].isna()].copy()
    mm_key = mm_key.drop(
        columns=['same_analyst_overlap', 'n_common_analysts'],
        errors='ignore'
    )
else:
    mm_key['same_analyst_overlap'] = (
        mm_key['same_analyst_overlap']
        .fillna(0)
        .astype('int8')
    )
    mm_key['n_common_analysts'] = (
        mm_key['n_common_analysts']
        .fillna(0)
        .astype('int16')
    )


# 用剩余的键去删选原表
key_cols = ['Annodt', 'Stkcd', 'StkcdRecall']

mm_key_keep = mm_key[key_cols].drop_duplicates()

duplicate_count = mm_key_keep.duplicated(key_cols).sum()
print('筛选键重复数：', duplicate_count)

n_before = len(mm)

mm = mm.merge(
    mm_key_keep,
    on=key_cols,
    how='inner',
    validate='1:1'
)

print('筛选前样本数：', n_before)
print('筛选后样本数：', len(mm))
print('删除的不合格公司对数量：', n_before - len(mm))

#===============================================================================
# 构造 cue dummy
#===============================================================================
# 基准 cue 只关注提示公司是否属于市值前 10%
mm["size90plus"] = (
    mm["mktcap_tm3"] > mm["marketCap_90th_percentile"]
).astype(int)



# 生成大市值cue
MAX_LAG = 1   # 只生成 lag1；如果想生成 lag1-lag4，改成 4

for i in range(1, MAX_LAG + 1):

    lag_col = f"quarterLag{i}Alt"

    # 如果当前表里没有这个 lag 变量，就跳过
    if lag_col not in mm.columns:
        print(f"跳过：{lag_col} 不在当前表中")
        continue

    # 主 cue：第 i 期共同公告记忆 × 大市值提示公司
    mm[f"quarterLag{i}size90plus"] = (
        mm[lag_col] * mm["size90plus"]
    ).astype("int8")

#===============================================================================
# 把公司对层面数据汇总到被提示公司-日期层面
#===============================================================================
df = mm.copy()

group_keys = ["StkcdRecall", "event_Trddt"]

# 1. 排序
df = df.sort_values(
    ["StkcdRecall", "event_Trddt", "mktcap_tm3", "Stkcd"]
).reset_index(drop=True)

# 2. 聚合规则
agg_dict = {
    # 可选：当天有多少个大市值提示公司
    # "nrCuesLag1S90p": ("quarterLag1size90plus", "sum"),

    # 主回归 cue dummy：是否至少有一个大市值提示事件
    "quarterLag1size90plus": ("quarterLag1size90plus", "max"),
}

# 3. 检查聚合所需变量是否存在
needed_cols = set(group_keys)
for _, value in agg_dict.items():
    needed_cols.add(value[0])

missing_cols = [c for c in needed_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"以下变量缺失，无法 collapse：{missing_cols}")

# 4. collapse 到 被提示公司 × 事件交易日
collapsed = (
    df.groupby(group_keys, dropna=False)
      .agg(**agg_dict)
      .reset_index()
)

collapsed["date"] = collapsed["event_Trddt"]

print("collapse 前表结构：", df.shape)
print("collapse 后表结构：", collapsed.shape)
print("collapse 后重复键数量：", collapsed.duplicated(group_keys).sum())
print(collapsed[["quarterLag1size90plus"]].sum())



print("\n核心变量合计：")
check_cols = [
    "quarterLag1size90plus",
]
check_cols = [c for c in check_cols if c in collapsed.columns]
print(collapsed[check_cols].sum().sort_values(ascending=False))


collapsed = collapsed.sort_values(["StkcdRecall", "date"]).reset_index(drop=True)

# 主面板只保留最终可用于回归的 cue 变量，并保存这个精简版本。
# event_Trddt 与 date 完全一致；
panel_drop_cols = (
    ["event_Trddt"]
    + [
        col for col in collapsed.columns
        if col.startswith(("sumSue", "sumMarketCap", "sumNrFirmsAnn"))
    ]
)

cue_events_for_panel = collapsed.drop(columns=panel_drop_cols, errors="ignore").copy()

cue_events_for_panel.to_csv(
    cue_events_out_path,
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d",
)

print(f"\n已保存用于后续主面板合并的 cue 事件表：{cue_events_out_path}")
print("保存行数：", len(cue_events_for_panel))
print("保存列数：", cue_events_for_panel.shape[1])
print("已删除的中间变量：")
print(panel_drop_cols)


#===============================================================================
# 从股票日度数据开始构建完整主样本，再合并 cue 事件
#===============================================================================
TRD_Dalyr = pd.read_csv(
    TRD_Dalyr_DGTW_path,
    encoding='utf-8',
    parse_dates=['Trddt']
)

TRD_Dalyr['Stkcd'] = TRD_Dalyr['Stkcd'].apply(normalize_stock_code)

TRD_Dalyr = TRD_Dalyr.rename(
    columns={'Stkcd':'StkcdRecall'}
).drop(columns=['IndustryCode'])

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

# 合并前检查匹配情况
news_match_check = AES_AEV_controls[['StkcdRecall', 'Trddt']].merge(
    TRD_Dalyr[['StkcdRecall', 'Trddt']],
    on=['StkcdRecall', 'Trddt'],
    how='left',
    indicator=True
)

print('新闻数据总行数：', len(news_match_check))
print('能匹配到交易日面板的新闻行数：', (news_match_check['_merge'] == 'both').sum())
print('不能匹配到交易日面板的新闻行数：', (news_match_check['_merge'] == 'left_only').sum())
print('不能匹配比例：', (news_match_check['_merge'] == 'left_only').mean())

# 正式合并
TRD_Dalyr = TRD_Dalyr.merge(
    AES_AEV_controls,
    on=['StkcdRecall', 'Trddt'],
    how='left',
    validate='1:1'
)

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

# 检查 cue 事件能否匹配到主面板
merge_check = cue_panel[["StkcdRecall", "date"]].merge(
    TRD_Dalyr[["StkcdRecall", "Trddt"]],
    left_on=["StkcdRecall", "date"],
    right_on=["StkcdRecall", "Trddt"],
    how="left",
    indicator=True
)

print("cue 事件总数：", len(merge_check))
print("cue 能匹配到日度主面板数量：", (merge_check["_merge"] == "both").sum())
print("cue 匹配不到日度主面板数量：", (merge_check["_merge"] == "left_only").sum())
print("cue 匹配不到比例：", (merge_check["_merge"] == "left_only").mean())

# 正式合并
TRD_Dalyr = TRD_Dalyr.merge(
    cue_panel,
    left_on=["StkcdRecall", "Trddt"],
    right_on=["StkcdRecall", "date"],
    how="left",
    validate="1:1"
)

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
    dtype={"Stkcd": "string"},
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
