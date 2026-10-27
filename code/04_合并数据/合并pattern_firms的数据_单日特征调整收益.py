import pandas as pd
import numpy as np
import gc
from dateutil.relativedelta import * 
from pathlib import Path
import datetime 


# 本脚本只生成单日收益扩展所需的窄基础面板；导出后立即结束，
# 不读取累计收益文件，也不覆盖原有累计收益回归面板。
SINGLE_DAY_BASE_ONLY = True

# 单日收益扩展和旧累计收益基准回归共同使用的右端变量。
# 提前声明后，单日基础面板可以在读取大体量 DGTW 收益文件之前导出。
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


def save_single_day_return_regression_base(
    panel: pd.DataFrame,
    output_path: Path,
    rhs_cols: list[str],
) -> None:
    """保存不含任何收益因变量的单日收益回归窄基础面板。"""
    base_cols = list(
        dict.fromkeys(
            [
                "StkcdRecall",
                "Trddt",
                "td",
                "yearQuarter",
                *rhs_cols,
            ]
        )
    )
    missing = [column for column in base_cols if column not in panel.columns]
    if missing:
        raise ValueError(
            "无法生成单日收益回归基础面板，缺少变量："
            f"{missing}"
        )

    base = panel[base_cols].copy()
    duplicate_count = base.duplicated(["StkcdRecall", "td"]).sum()
    if duplicate_count != 0:
        raise ValueError(
            "单日收益回归基础面板的 StkcdRecall-td 不唯一："
            f"{duplicate_count:,} 条重复观测。"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    print(f"单日收益回归基础面板已保存：{output_path}")
    print(f"单日收益回归基础面板结构：{base.shape}")


def add_sue_denominator_inputs(data):
    """为三种 SUE 构造各自的有效 cue 数量和有效市值分母。"""
    for sue_id in [1, 2, 3]:
        valid_cue = (
            data["quarterLag1size90plus"].eq(1)
            & data[f"sue{sue_id}"].notna()
        )
        data[f"validSue{sue_id}CueLag1S90p"] = valid_cue.astype("int8")
        data[f"validSue{sue_id}CueMarketCapLag1S90p"] = np.where(
            valid_cue,
            data["mktcap_tm3"],
            0.0,
        )
    return data


def sue_denominator_agg_specs():
    """返回三种 SUE 的非缺失 cue 数量和市值加总规则。"""
    specs = {}
    for sue_id in [1, 2, 3]:
        specs[f"sumSue{sue_id}ValidCueCountLag1S90p"] = (
            f"validSue{sue_id}CueLag1S90p",
            "sum",
        )
        specs[f"sumSue{sue_id}ValidCueMarketCapLag1S90p"] = (
            f"validSue{sue_id}CueMarketCapLag1S90p",
            "sum",
        )
    return specs


def calculate_sue_cue_averages(collapsed, denominator_mode):
    """按指定分母口径计算三种 SUE 的等权和市值加权平均。"""
    valid_modes = {"nonmissing_sue", "all_cues"}
    if denominator_mode not in valid_modes:
        raise ValueError(
            "SUE_DENOMINATOR_MODE 只能是 "
            f"{valid_modes}，当前是 {denominator_mode}"
        )

    for sue_id in [1, 2, 3]:
        if denominator_mode == "nonmissing_sue":
            ew_denominator = collapsed[
                f"sumSue{sue_id}ValidCueCountLag1S90p"
            ]
            vw_denominator = collapsed[
                f"sumSue{sue_id}ValidCueMarketCapLag1S90p"
            ]
        else:
            ew_denominator = collapsed["nrCuesLag1S90p"]
            vw_denominator = collapsed["sumMarketCap_Minus3Lag1S90p"]

        collapsed[f"sue{sue_id}CueLag1S90pEW"] = (
            collapsed[f"sumSue{sue_id}Lag1S90p"]
            .div(ew_denominator)
            .where(ew_denominator.ne(0), np.nan)
        )
        collapsed[f"sue{sue_id}CueLag1S90pVW"] = (
            collapsed[f"sumSue{sue_id}Lag1XmarketCapS90p"]
            .div(vw_denominator)
            .where(vw_denominator.ne(0), np.nan)
        )

    return collapsed


# =============================================================================
## 路径设置
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

MemoryMatrixPattern_path = ROOT / '中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv'
TRD_Cale_td_path = ROOT / '中间结果/01_清洗后样本/市场交易日历_td.csv'
TRD_Dalyr_DGTW_path = ROOT / '中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv'
marketcap_group_path = ROOT / '中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv'
industry_path = ROOT / '中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv'
sue_path = ROOT/'中间结果/02_变量构造结果/4sue/sue.csv'
SC_NetWorkRelationsIndex_path = ROOT/'input/CSMAR/供应链网络关系指标表2001-12-31至2025-12-31/SC_NetworkRelationsIndex.csv'
IAR_Rept_path = ROOT/'input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv'
# 数行者分段新闻：2010年前报刊，2010年起网络。
AES_AEV_controls_path = ROOT/'中间结果/02_变量构造结果/3分段新闻_AES_AEV/DATAGO_分段新闻_真实交易日控制变量.csv'
sample_intermediate_out_path = ROOT / "中间结果/04_合并数据/Sample_intermediate_pattern.csv"
adjusted_return_path = ROOT / "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW特征调整收益率.csv"
output_path = ROOT/'中间结果/04_合并数据/Pattern样本_最终回归面板.csv'
daily_return_regression_base_path = (
    ROOT
    / "中间结果/04_合并数据"
    / "Pattern样本_单日收益回归基础面板.csv"
)
performance_forecast_controls_path = (
    ROOT / "中间结果/02_变量构造结果/5公司公告事件/业绩预告快报控制变量_公告日期口径.csv"
)
other_event_controls_path = (
    ROOT / "中间结果/02_变量构造结果/5公司公告事件/专题研究事件控制变量_公告日期口径.csv"
)

cue_events_out_path = ROOT / '中间结果/04_合并数据/Pattern_cue_events_after_collapse.csv'
cue_events_out_path.parent.mkdir(parents=True, exist_ok=True)

# 盈余惊喜聚合分母口径
SUE_DENOMINATOR_MODE = "nonmissing_sue"
# "nonmissing_sue": 每种 SUE 只使用该变量非缺失的 cue 数量/市值作分母
# "all_cues": 三种 SUE 都使用全部 cue 数量/市值作分母，对齐原文代码

VALID_SUE_DENOMINATOR_MODES = {"nonmissing_sue", "all_cues"}
if SUE_DENOMINATOR_MODE not in VALID_SUE_DENOMINATOR_MODES:
    raise ValueError(
        "SUE_DENOMINATOR_MODE 只能是 "
        f"{VALID_SUE_DENOMINATOR_MODES}，当前是 {SUE_DENOMINATOR_MODE}"
    )

# =============================================================================
## 分析师共同覆盖过滤参数
# =============================================================================

DROP_ANALYST_OVERLAP = True
# True: 删除有共同分析师覆盖的公司对
# False: 只合并标记，不删除

ANALYST_OVERLAP_ROOT = ROOT / '中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后'

ANALYST_ID_MODE = 'split'
# 'split': 拆开 AnanmID，按个人分析师识别，推荐主口径
# 'team': 不拆开 AnanmID，按原始 AnanmID 字符串识别

ANALYST_LOOKBACK_DAYS = 365
# 可选：365 或 45


# 市值断点依据
MARKETCAP_BREAKPOINT_MODE = "prev_month"  
# "prev_month": 使用上月市值断点，避免前视
# "same_month": 使用公告所在月市值断点，对齐原文


# 事件反应日口径
EVENT_DATE_MODE = "next_if_nontrading"

VALID_EVENT_DATE_MODES = {"next_if_nontrading", "strict_next_trading"}

if EVENT_DATE_MODE not in VALID_EVENT_DATE_MODES:
    raise ValueError(
        f"EVENT_DATE_MODE 只能是 {VALID_EVENT_DATE_MODES}，当前是 {EVENT_DATE_MODE}"
    )
# "next_if_nontrading": 主口径，只调整非交易日
# "strict_next_trading": 稳健性口径，交易日也顺延


# 读取记忆矩阵
mm = pd.read_csv(
    MemoryMatrixPattern_path,
    encoding='utf-8',
    parse_dates=['Annodt','event_Trddt']
)

mm['Stkcd'] = mm['Stkcd'].apply(normalize_stock_code)
mm['StkcdRecall'] = mm['StkcdRecall'].apply(normalize_stock_code)

# 合并市值数据
# 所以需要先将anndot调整到交易日，再匹配t-3的市值。

# 1. 读取交易日历，把公告日映射到事件交易日(之前记忆矩阵生成了)
required_event_cols = {"event_Trddt", "event_td"}
missing_event_cols = required_event_cols - set(mm.columns)
if missing_event_cols:
    raise ValueError(f"MemoryMatrixPattern 缺少事件交易日字段: {missing_event_cols}")

mm["event_Trddt"] = pd.to_datetime(mm["event_Trddt"], errors="coerce")
mm["event_td"] = pd.to_numeric(mm["event_td"], errors="coerce")
mm["td_tm3"] = mm["event_td"] - 3
# 得到td-3序号用于下一步匹配市值

# 2.读取日度市值
mktcap = pd.read_csv(
    TRD_Dalyr_DGTW_path,
    encoding='utf-8',
    usecols=['Stkcd', 'td', 'size', 'sizew'],
    dtype={'Stkcd': 'string'}
)

mktcap['Stkcd'] = mktcap['Stkcd'].apply(normalize_stock_code)
mktcap['td'] = pd.to_numeric(mktcap['td'], errors='coerce')

mktcap = mktcap.rename(
    columns={
        'td': 'td_tm3',
        'size': 'mktcap_tm3',
        'sizew': 'float_mktcap_tm3'
    }
)

# 3. 为了省内存，先对 MemoryMatrixPattern 中需要合并的 Stkcd-td_tm3 去重
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



# 合并盈余惊喜数据
sue = pd.read_csv(
    sue_path,
    encoding='utf-8',
    usecols=['Stkcd','Accper','sue1','sue2','sue3'],
    parse_dates=['Accper']
)

sue['Stkcd'] = sue['Stkcd'].apply(normalize_stock_code)

# 先找当前记忆矩阵中公告日的财报截止日，再合并
current_accper = pd.read_csv(
    IAR_Rept_path,
    usecols=["Stkcd", "Annodt", "Accper"],
    parse_dates=["Annodt", "Accper"],
    dtype={"Stkcd": "string"}
)

current_accper["Stkcd"] = current_accper["Stkcd"].apply(normalize_stock_code)

current_accper = (
    current_accper
    .sort_values(["Stkcd", "Annodt", "Accper"])
    .groupby(["Stkcd", "Annodt"], as_index=False)
    .head(1)
    .rename(columns={"Accper": "currentAccper"})
)# 去重

mm = mm.merge(
    current_accper,
    on=["Stkcd", "Annodt"],
    how="left",
    validate="m:1"
)

mm = mm.merge(
    sue,
    left_on=["Stkcd", "currentAccper"],
    right_on=["Stkcd", "Accper"],
    how="left",
    validate="m:1"
)

mm = mm.drop(columns=["Accper"], errors="ignore")

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
    / f'Pattern样本_分析师共同覆盖_滚动{ANALYST_LOOKBACK_DAYS}天.csv'
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
# 构造 cue dummy: 大公司cue、强记忆cue
#===============================================================================

# 按照提示公司 Stkcd 的公告日前市值分位点生成规模分组虚拟变量
mm["size90plus"] = (
    mm["mktcap_tm3"] > mm["marketCap_90th_percentile"]
).astype(int)

mm["size50_90"] = (
    (mm["mktcap_tm3"] > mm["marketCap_50th_percentile"])
    & (mm["mktcap_tm3"] <= mm["marketCap_90th_percentile"])
).astype(int)

mm["size10_50"] = (
    (mm["mktcap_tm3"] >= mm["marketCap_10th_percentile"])
    & (mm["mktcap_tm3"] <= mm["marketCap_50th_percentile"])
).astype(int)

mm["size10minus"] = (
    mm["mktcap_tm3"] < mm["marketCap_10th_percentile"]
).astype(int)

# 已经检查都没有缺失值，可以相乘构造虚拟变量

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

    # 其他规模组 cue
    mm[f"quarterLag{i}size50_90"] = (
        mm[lag_col] * mm["size50_90"]
    ).astype("int8")

    mm[f"quarterLag{i}size10_50"] = (
        mm[lag_col] * mm["size10_50"]
    ).astype("int8")

    mm[f"quarterLag{i}size10minus"] = (
        mm[lag_col] * mm["size10minus"]
    ).astype("int8")

# ============================================================
# 生成 attention / volume × size 的强记忆 cue
# ============================================================

MAX_LAG = 1

size_groups = [
    "size90plus",
    "size50_90",
]
attention_volume_suffixes = [
    "hiAEV",
    "hiavol",
    "hiavolS",
]

for i in range(1, MAX_LAG + 1):
    for size in size_groups:
        for suffix in attention_volume_suffixes:
            src = f"lag{i}{suffix}"
            dst = f"lag{i}{suffix}{size}"

            if src in mm.columns and size in mm.columns:
                mm[dst] = (mm[src] * mm[size]).astype("int8")

# ============================================================
# 生成 SUE / 盈余惊喜 × size 的强记忆 cue
# ============================================================

sue_suffixes = [
    "hi_sue1_abs",
    "hi_sue2_abs",
    "hi_sue3_abs",
    "sue1pos",
    "sue2pos",
    "sue3pos",
    "sue1neg",
    "sue2neg",
    "sue3neg",
# 双方同为极端正向 / 极端负向 SUE（盈余惊喜）
    "hi_sue1_pos",
    "hi_sue2_pos",
    "hi_sue3_pos",
    "hi_sue1_neg",
    "hi_sue2_neg",
    "hi_sue3_neg",
]

for i in range(1, MAX_LAG + 1):
    for size in size_groups:
        for suffix in sue_suffixes:
            src = f"lag{i}{suffix}"
            dst = f"lag{i}{suffix}{size}"

            if src in mm.columns and size in mm.columns:
                mm[dst] = (mm[src] * mm[size]).astype("int8")

# ============================================================
# 生成 topMover × size 的强记忆 cue
# ============================================================

for i in range(1, MAX_LAG + 1):
    for size in size_groups:
        src = f"lag{i}topMover"
        dst = f"lag{i}topMover{size}"

        if src in mm.columns and size in mm.columns:
            mm[dst] = (mm[src] * mm[size]).astype("int8")

# ============================================================
# 计算大市值 cue 的盈余惊喜及市值加权盈余惊喜
# ============================================================

MAX_LAG = 1

for sue_col in ["sue1", "sue2", "sue3"]:
    for i in range(1, MAX_LAG + 1):
        lag_col = f"quarterLag{i}Alt"

        mm[f"{sue_col}Lag{i}S90p"] = (
            mm[sue_col] * mm[lag_col] * mm["size90plus"]
        )

        mm[f"{sue_col}Lag{i}XmarketCapS90p"] = (
            mm[sue_col] * mm[lag_col] * mm["mktcap_tm3"] * mm["size90plus"]
        )

# 为每一种 SUE 单独记录非缺失 cue 数量和相应市值。
# 这些分母只在 SUE_DENOMINATOR_MODE == "nonmissing_sue" 时使用。
mm = add_sue_denominator_inputs(mm)

# ============================================================
# 保存不同规模 cue 的市值，用于后续市值加权
# 如果这一行是“大市值 lag1 cue”，就保留 cue 公司的市值；否则记为 0。
# ============================================================

for i in range(1, MAX_LAG + 1):
    mm[f"marketCap_Minus3Lag{i}S90p"] = (
        mm["mktcap_tm3"] * mm[f"quarterLag{i}size90plus"]
    )

    mm[f"marketCap_Minus3Lag{i}S5090"] = (
        mm["mktcap_tm3"] * mm[f"quarterLag{i}size50_90"]
    )
# ============================================================
# 计算不同规模 cue 对应的共同公告公司数量
# ============================================================

for i in range(1, MAX_LAG + 1):
    nr_col = f"nrFirmsAnnLag{i}Alt" # 上i季度那次共同公告记忆形成时，当天一共有多少家公司同时公告。

    mm[f"nrFirmsAnnLag{i}S90p"] = (
        mm[nr_col] * mm["size90plus"]
    )
# 如果当前 cue 公司是大市值（前10%）公司，就保留它对应的历史共同公告数量；
# 如果当前 cue 公司不是大市值公司，就记为 0。


    mm[f"nrFirmsAnnLag{i}S5090"] = (
        mm[nr_col] * mm["size50_90"]
    )

# ============================================================
# 4. 计算市值加权的共同公告公司数量
# ============================================================

for i in range(1, MAX_LAG + 1):
    nr_col = f"nrFirmsAnnLag{i}Alt"

    mm[f"nrFirmsAnnLag{i}XmarketCapS90p"] = (
        mm[nr_col] * mm["mktcap_tm3"] * mm["size90plus"]
    ) # 大公司前10%市值加总

    mm[f"nrFirmsAnnLag{i}XmarketCapS5090"] = (
        mm[nr_col] * mm["mktcap_tm3"] * mm["size50_90"]
    )
# ============================================================
# 5. 保存 collapse 前样本
# ============================================================

sort_cols = [
    "Stkcd",
    "Annodt",
    "StkcdRecall",
]

sort_cols_existing = [c for c in sort_cols if c in mm.columns]

mm = mm.sort_values(sort_cols_existing).reset_index(drop=True)

# ============================================================
# Collapse data
# 当前 mm 层级：Stkcd × event_Trddt × StkcdRecall
# 目标层级：StkcdRecall × event_Trddt
# ============================================================

df = mm.copy()

group_keys = ["StkcdRecall", "event_Trddt"]

# 1. 排序
df = df.sort_values(
    ["StkcdRecall", "event_Trddt", "mktcap_tm3", "Stkcd"]
).reset_index(drop=True)

# 2. sum 聚合规则
agg_dict = {
    # cue 数量
    "nrCuesLag1S90p": ("quarterLag1size90plus", "sum"),
    "nrCuesLag1S5090": ("quarterLag1size50_90", "sum"),
    "nrCuesLag1S1050": ("quarterLag1size10_50", "sum"),
    "nrCuesLag1S10m": ("quarterLag1size10minus", "sum"),

    # SUE 加总：等权分子
    "sumSue1Lag1S90p": ("sue1Lag1S90p", "sum"),
    "sumSue2Lag1S90p": ("sue2Lag1S90p", "sum"),
    "sumSue3Lag1S90p": ("sue3Lag1S90p", "sum"),

    # SUE × 市值加总：市值加权分子
    "sumSue1Lag1XmarketCapS90p": ("sue1Lag1XmarketCapS90p", "sum"),
    "sumSue2Lag1XmarketCapS90p": ("sue2Lag1XmarketCapS90p", "sum"),
    "sumSue3Lag1XmarketCapS90p": ("sue3Lag1XmarketCapS90p", "sum"),

    # 市值分母
    "sumMarketCap_Minus3Lag1S90p": ("marketCap_Minus3Lag1S90p", "sum"),
    "sumMarketCap_Minus3Lag1S5090": ("marketCap_Minus3Lag1S5090", "sum"),

    # interference 等权分子
    "sumNrFirmsAnnLag1S90p": ("nrFirmsAnnLag1S90p", "sum"),
    "sumNrFirmsAnnLag1S5090": ("nrFirmsAnnLag1S5090", "sum"),

    # interference 市值加权分子
    "sumNrFirmsAnnLag1XmarketCapS90p": ("nrFirmsAnnLag1XmarketCapS90p", "sum"),
    "sumNrFirmsAnnLag1XmarketCapS5090": ("nrFirmsAnnLag1XmarketCapS5090", "sum"),
}

# 加入三种 SUE 各自的非缺失 cue 数量和有效市值分母。
agg_dict.update(sue_denominator_agg_specs())

# 3. max 聚合变量
max_cols = [
    # baseline cue by size
    "quarterLag1size90plus",
    "quarterLag1size50_90",
    "quarterLag1size10_50",
    "quarterLag1size10minus",

    # attention / volume × size
    "lag1hiAEVsize90plus",
    "lag1hiavolsize90plus",
    "lag1hiavolSsize90plus",
    "lag1hiAEVsize50_90",
    "lag1hiavolsize50_90",
    "lag1hiavolSsize50_90",

    # SUE absolute × size
    "lag1hi_sue1_abssize90plus",
    "lag1hi_sue2_abssize90plus",
    "lag1hi_sue3_abssize90plus",
    "lag1hi_sue1_abssize50_90",
    "lag1hi_sue2_abssize50_90",
    "lag1hi_sue3_abssize50_90",

    # SUE positive × size
    "lag1sue1possize90plus",
    "lag1sue2possize90plus",
    "lag1sue3possize90plus",
    "lag1sue1possize50_90",
    "lag1sue2possize50_90",
    "lag1sue3possize50_90",

    # SUE negative × size
    "lag1sue1negsize90plus",
    "lag1sue2negsize90plus",
    "lag1sue3negsize90plus",
    "lag1sue1negsize50_90",
    "lag1sue2negsize50_90",
    "lag1sue3negsize50_90",

    # SUE high positive × size
    "lag1hi_sue1_possize90plus",
    "lag1hi_sue2_possize90plus",
    "lag1hi_sue3_possize90plus",
    "lag1hi_sue1_possize50_90",
    "lag1hi_sue2_possize50_90",
    "lag1hi_sue3_possize50_90",

    # SUE high negative × size
    "lag1hi_sue1_negsize90plus",
    "lag1hi_sue2_negsize90plus",
    "lag1hi_sue3_negsize90plus",
    "lag1hi_sue1_negsize50_90",
    "lag1hi_sue2_negsize50_90",
    "lag1hi_sue3_negsize50_90",

    # top mover × size
    "lag1topMoversize90plus",
    "lag1topMoversize50_90",
]

for col in max_cols:
    if col in df.columns:
        agg_dict[col] = (col, "max")

# 4. 检查聚合所需变量是否都存在
needed_cols = set(group_keys)
for _, value in agg_dict.items():
    needed_cols.add(value[0])

missing_cols = [c for c in needed_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"以下变量缺失，无法 collapse：{missing_cols}")

# 5. collapse
collapsed = (
    df.groupby(group_keys, dropna=False)
      .agg(**agg_dict)
      .reset_index()
)

# 6. 按开关选择分母口径，计算 SUE 的等权和市值加权 cue 强度
collapsed = calculate_sue_cue_averages(
    collapsed,
    denominator_mode=SUE_DENOMINATOR_MODE,
)
print(f"SUE 聚合分母口径：{SUE_DENOMINATOR_MODE}")

# 7. 计算 interference
collapsed["nrFirmsAnnLag1VWS90p"] = np.where(
    collapsed["sumMarketCap_Minus3Lag1S90p"] != 0,
    collapsed["sumNrFirmsAnnLag1XmarketCapS90p"]
    / collapsed["sumMarketCap_Minus3Lag1S90p"],
    np.nan,
)

collapsed["nrFirmsAnnLag1VWS5090"] = np.where(
    collapsed["sumMarketCap_Minus3Lag1S5090"] != 0,
    collapsed["sumNrFirmsAnnLag1XmarketCapS5090"]
    / collapsed["sumMarketCap_Minus3Lag1S5090"],
    np.nan,
)

collapsed["nrFirmsAnnLag1EWS90p"] = np.where(
    collapsed["nrCuesLag1S90p"] != 0,
    collapsed["sumNrFirmsAnnLag1S90p"]
    / collapsed["nrCuesLag1S90p"],
    np.nan,
)

collapsed["nrFirmsAnnLag1EWS5090"] = np.where(
    collapsed["nrCuesLag1S5090"] != 0,
    collapsed["sumNrFirmsAnnLag1S5090"]
    / collapsed["nrCuesLag1S5090"],
    np.nan,
)

# 8. 生成 date，用于后续合并日度收益
collapsed["date"] = collapsed["event_Trddt"]

# 9. 输出检查
print("collapse 前表结构：", df.shape)
print("collapse 后表结构：", collapsed.shape)
print("collapse 后重复键数量：", collapsed.duplicated(group_keys).sum())

print("\n核心变量合计：")
check_cols = [
    "nrCuesLag1S90p",
    "nrCuesLag1S5090",
    "nrCuesLag1S1050",
    "nrCuesLag1S10m",
    "quarterLag1size90plus",
    "lag1hiAEVsize90plus",
    "lag1hiavolsize90plus",
    "lag1hi_sue1_possize90plus",
    "lag1hi_sue1_negsize90plus",
    "lag1hi_sue2_possize90plus",
    "lag1hi_sue2_negsize90plus",
    "lag1topMoversize90plus",
]
check_cols = [c for c in check_cols if c in collapsed.columns]
print(collapsed[check_cols].sum().sort_values(ascending=False))


collapsed = collapsed.sort_values(["StkcdRecall", "date"]).reset_index(drop=True)

# 主面板只保留最终可用于回归的 cue 变量，并保存这个精简版本。
# event_Trddt 与 date 完全一致；sum* 变量只是计算 EW/VW 指标的中间分子或分母。
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

if SINGLE_DAY_BASE_ONLY:
    # 单日基础面板只需要聚合后的 cue 表，及时释放构造阶段的大表。
    del mm, mm_key, mktcap, daily_industry, analyst_overlap, collapsed
    gc.collect()


#===============================================================================
# 从股票日度数据开始构建主样本，再合并 cue 事件数据
#===============================================================================

TRD_Dalyr = pd.read_csv(
    TRD_Dalyr_DGTW_path,
    encoding='utf-8',
    usecols=(
        ["Stkcd", "td", "Trddt"] if SINGLE_DAY_BASE_ONLY else None
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

# 保持现有回归变量接口。ownMajorDisclosure 现由十类专题事件并集构造，
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
    usecols=(
        ['Stkcd', 'date', 'aesIncrease', 'aesDecrease',
         'logIncreaseAEV', 'logDecreaseAEV']
        if SINGLE_DAY_BASE_ONLY else
        ['Stkcd', 'date', 'AES', 'AEV', 'aesIncrease', 'aesDecrease',
         'logIncreaseAEV', 'logDecreaseAEV']
    ),
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
    col for col in [
        'AES', 'AEV', 'aesIncrease', 'aesDecrease',
        'logIncreaseAEV', 'logDecreaseAEV',
    ]
    if col in AES_AEV_controls.columns
]
AES_AEV_controls[news_numeric_cols] = AES_AEV_controls[news_numeric_cols].apply(
    pd.to_numeric,
    errors='raise',
)

if 'AEV' in AES_AEV_controls and (AES_AEV_controls['AEV'].dropna() < 0).any():
    raise ValueError('真实交易日新闻控制表中的 AEV 不应出现负数。')

# 单日基础面板规模较大，匹配率在最终输出流式检查中核对。
if not SINGLE_DAY_BASE_ONLY:
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
    del news_match_check

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

print("媒体控制变量汇总：")
print(TRD_Dalyr[
    [
        "aesIncrease",
        "aesDecrease",
        "logIncreaseAEV",
        "logDecreaseAEV"
    ]
].describe())

#===============================================================================
# 合并 collapse 后的 cue 事件表
# 当前主面板：TRD_Dalyr，层级 StkcdRecall × Trddt
# cue 表：cue_events_for_panel，层级 StkcdRecall × date
#===============================================================================

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
    "quarterLag1size50_90",
    "quarterLag1size10_50",
    "quarterLag1size10minus",

    "lag1hiAEVsize90plus",
    "lag1hiavolsize90plus",
    "lag1hiavolSsize90plus",
    "lag1hiAEVsize50_90",
    "lag1hiavolsize50_90",
    "lag1hiavolSsize50_90",

    "lag1hi_sue1_abssize90plus",
    "lag1hi_sue2_abssize90plus",
    "lag1hi_sue3_abssize90plus",
    "lag1hi_sue1_abssize50_90",
    "lag1hi_sue2_abssize50_90",
    "lag1hi_sue3_abssize50_90",

    "lag1sue1possize90plus",
    "lag1sue2possize90plus",
    "lag1sue3possize90plus",
    "lag1sue1possize50_90",
    "lag1sue2possize50_90",
    "lag1sue3possize50_90",

    "lag1sue1negsize90plus",
    "lag1sue2negsize90plus",
    "lag1sue3negsize90plus",
    "lag1sue1negsize50_90",
    "lag1sue2negsize50_90",
    "lag1sue3negsize50_90",

    "lag1hi_sue1_possize90plus",
    "lag1hi_sue2_possize90plus",
    "lag1hi_sue3_possize90plus",
    "lag1hi_sue1_possize50_90",
    "lag1hi_sue2_possize50_90",
    "lag1hi_sue3_possize50_90",

    "lag1hi_sue1_negsize90plus",
    "lag1hi_sue2_negsize90plus",
    "lag1hi_sue3_negsize90plus",
    "lag1hi_sue1_negsize50_90",
    "lag1hi_sue2_negsize50_90",
    "lag1hi_sue3_negsize50_90",

    "lag1topMoversize90plus",
    "lag1topMoversize50_90",
]

cue_dummy_cols = [c for c in cue_dummy_cols if c in TRD_Dalyr.columns]

TRD_Dalyr[cue_dummy_cols] = (
    TRD_Dalyr[cue_dummy_cols]
    .fillna(0)
    .astype("int8")
)

print("cue dummy 缺失值已设为 0")
print(TRD_Dalyr[cue_dummy_cols].sum().sort_values(ascending=False).head(20))

cue_count_cols = [
    "nrCuesLag1S90p",
    "nrCuesLag1S5090",
    "nrCuesLag1S1050",
    "nrCuesLag1S10m",
]

cue_count_cols = [c for c in cue_count_cols if c in TRD_Dalyr.columns]

TRD_Dalyr[cue_count_cols] = (
    TRD_Dalyr[cue_count_cols]
    .fillna(0)
)

# 生成日历季度和年度-季度固定效应字段。单日收益扩展的中性基础面板
# 不需要读取或合并任何累计收益因变量，因此放在 DGTW 大文件合并之前。
TRD_Dalyr["quarter"] = TRD_Dalyr["Trddt"].dt.quarter
TRD_Dalyr["yearQuarter"] = TRD_Dalyr["Trddt"].dt.to_period("Q")

save_single_day_return_regression_base(
    panel=TRD_Dalyr,
    output_path=daily_return_regression_base_path,
    rhs_cols=baseline_rhs,
)

if SINGLE_DAY_BASE_ONLY:
    print(
        "已按 --single-day-base-only 仅生成单日收益回归基础面板；"
        "现有累计收益最终回归面板未被覆盖。"
    )
    raise SystemExit(0)

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

# 构造 interference 高低组虚拟变量
# interference 衡量同一个 cue 场景中同时出现的其他公告数量

# 只是兜底的写法，防止没有year这一列
if "year" not in TRD_Dalyr.columns:
    TRD_Dalyr["year"] = TRD_Dalyr["Trddt"].dt.year

def by_year_xtile_2(df, value_col, year_col="year"):
    """
    按年份把变量分成低/高两组：低组=1，高组=2。
    """
    def _split_one_year(s):
        nonmissing = s.dropna()

        # 如果该年份没有有效值，返回缺失
        if nonmissing.empty:
            return pd.Series(np.nan, index=s.index)

        # 如果该年份有效取值不足以分成两组，则全部设为缺失
        # 避免 pandas qcut 因重复断点报错
        if nonmissing.nunique() < 2:
            return pd.Series(np.nan, index=s.index)

        out = pd.Series(np.nan, index=s.index)

        try:
            out.loc[nonmissing.index] = pd.qcut(
                nonmissing,
                q=2,
                labels=[1, 2],
                duplicates="drop"
            ).astype(float)
        except ValueError:
            out.loc[nonmissing.index] = np.nan

        return out

    return df.groupby(year_col, group_keys=False)[value_col].apply(_split_one_year)

# 当前只对 lag1 构造，但循环写法可以扩展到其他 lag
for i in range(1, 2):
    # (1) 等权版本：对所有 cue 等权平均
    for suffix in ["S90p", "S5090"]:
        value_col = f"nrFirmsAnnLag{i}EW{suffix}"
        group_col = f"interferenceGroupEWLag{i}{suffix}"
        low_col = f"lowInterferenceEWLag{i}{suffix}"
        high_col = f"highInterferenceEWLag{i}{suffix}"

        TRD_Dalyr[group_col] = by_year_xtile_2(TRD_Dalyr, value_col)
        TRD_Dalyr[low_col] = (TRD_Dalyr[group_col] == 1).astype("int8")
        TRD_Dalyr[high_col] = (TRD_Dalyr[group_col] == 2).astype("int8")

    # (2) 市值加权版本：按照提示公司市值给 cue 加权
    for suffix in ["S90p", "S5090"]:
        value_col = f"nrFirmsAnnLag{i}VW{suffix}"
        group_col = f"interferenceGroupVWLag{i}{suffix}"
        low_col = f"lowInterferenceVWLag{i}{suffix}"
        high_col = f"highInterferenceVWLag{i}{suffix}"

        TRD_Dalyr[group_col] = by_year_xtile_2(TRD_Dalyr, value_col)
        TRD_Dalyr[low_col] = (TRD_Dalyr[group_col] == 1).astype("int8")
        TRD_Dalyr[high_col] = (TRD_Dalyr[group_col] == 2).astype("int8")


# 把 cue 的盈余惊喜变量分成五分位
def xtile_nq(series, q):
    """
    按分位点阈值分组，重复值很多时，允许某些组为空，不把整列设为缺失。
    """
    out = pd.Series(np.nan, index=series.index)
    nonmissing = series.dropna()

    if nonmissing.empty:
        return out

    probs = [k / q for k in range(1, q)]
    cutoffs = nonmissing.quantile(probs).to_numpy()

    values = nonmissing.to_numpy()
    groups = np.searchsorted(cutoffs, values, side="right") + 1

    out.loc[nonmissing.index] = groups
    return out


# 当前只处理 lag1，但循环可以扩展到其他 lag
for i in range(1, 2):
    for sue_id in [1, 2, 3]:
        TRD_Dalyr[f"sue{sue_id}CueLag{i}VWS90p_quintile"] = xtile_nq(
            TRD_Dalyr[f"sue{sue_id}CueLag{i}S90pVW"],
            q=5
        )# 大市值 cue 的市值加权盈余惊喜五分位

        TRD_Dalyr[f"sue{sue_id}CueLag{i}EWS90p_quintile"] = xtile_nq(
            TRD_Dalyr[f"sue{sue_id}CueLag{i}S90pEW"],
            q=5
        )# 大市值 cue 的等权盈余惊喜五分位

TRD_Dalyr["marketCap_Minus3_Recall"] = TRD_Dalyr["mktcap_tm3"]

TRD_Dalyr["marketCap_Minus3_Recall_medGr"] = by_year_xtile_2(
    TRD_Dalyr,
    "marketCap_Minus3_Recall",
    year_col="year"
) # 按年份把被提示公司市值分成高低两组

TRD_Dalyr["smallFirm"] = (
    TRD_Dalyr["marketCap_Minus3_Recall_medGr"] == 1
).astype("int8") # 如果被提示公司市值低于当年中位数，则 smallFirm = 1


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
    for i in range(1, 51)
]

# 对 cue 盈余惊喜变量进行 1% winsorize
sue_winsor_cols = []
for i in range(1, 2):
    for sue_id in [1, 2, 3]:
        for weight in ["VW", "EW"]:
            sue_winsor_cols.append(f"sue{sue_id}CueLag{i}S90p{weight}")

# 原文缩尾变量清单：收益率窗口 + cue 盈余惊喜
winsor_source_cols = [
    col
    for col in dep_vars + sue_winsor_cols
    if col in TRD_Dalyr.columns
]

# 注意：阈值仍然在筛基准样本前的全样本上计算
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

def baseline_regression_sample_mask(df, dep_col, rhs_cols, fe_cols, cluster_cols=None):
    """
    1. 回归变量、固定效应变量、聚类变量不能缺失；
    2. 吸收固定效应后，删除 singleton（单例组），即固定效应组内只有 1 条观测的组。
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


# 准备控制变量
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

# 在任何累计收益因变量参与样本筛选之前，单独保存一个窄基础面板。
# 该文件只包含公司日键、固定效应字段和右端变量；不包含任何累计
# 或单日收益因变量。单日收益扩展分析会从 DGTW 股票日度表按
# “StkcdRecall + (td+h)”另行匹配目标日收益，从而避免新结果继续
# 受 [t,t+1] 和 [t,t+10] 累计收益完整性筛选约束。
daily_return_regression_base_cols = list(
    dict.fromkeys(
        [
            "StkcdRecall",
            "Trddt",
            "td",
            "yearQuarter",
            *baseline_rhs,
        ]
    )
)

missing_daily_return_base_cols = [
    col
    for col in daily_return_regression_base_cols
    if col not in TRD_Dalyr.columns
]
if missing_daily_return_base_cols:
    raise ValueError(
        "无法生成单日收益回归基础面板，缺少变量："
        f"{missing_daily_return_base_cols}"
    )

daily_return_regression_base = TRD_Dalyr[
    daily_return_regression_base_cols
].copy()

daily_return_base_duplicate_count = (
    daily_return_regression_base
    .duplicated(["StkcdRecall", "td"])
    .sum()
)
if daily_return_base_duplicate_count != 0:
    raise ValueError(
        "单日收益回归基础面板的 StkcdRecall-td 不唯一："
        f"{daily_return_base_duplicate_count:,} 条重复观测。"
    )

daily_return_regression_base_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)
daily_return_regression_base.to_csv(
    daily_return_regression_base_path,
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d",
)
print(
    "单日收益回归基础面板已保存："
    f"{daily_return_regression_base_path}"
)
print(
    "单日收益回归基础面板结构："
    f"{daily_return_regression_base.shape}"
)
del daily_return_regression_base

if SINGLE_DAY_BASE_ONLY:
    print(
        "已按 --single-day-base-only 仅生成单日收益回归基础面板；"
        "现有累计收益最终回归面板未被覆盖。"
    )
    raise SystemExit(0)

# [t, t+1] 基准回归样本
mask_tp1 = baseline_regression_sample_mask(
    TRD_Dalyr,
    dep_col="chAdjreturn_t_tp1_Recall_w",
    rhs_cols=baseline_rhs,
    fe_cols=["StkcdRecall", "yearQuarter"],
    cluster_cols=["StkcdRecall", "Trddt"],
)

TRD_Dalyr = TRD_Dalyr.loc[mask_tp1].copy()


# [t, t+10] 基准回归样本
mask_tp10 = baseline_regression_sample_mask(
    TRD_Dalyr,
    dep_col="chAdjreturn_t_tp10_Recall_w",
    rhs_cols=baseline_rhs,
    fe_cols=["StkcdRecall", "yearQuarter"],
    cluster_cols=["StkcdRecall", "Trddt"],
)

TRD_Dalyr = TRD_Dalyr.loc[mask_tp10].copy()

#===============================================================================
# 统计：一次提示公司的公告会提示多少家被提示公司
#===============================================================================

# 一、提取最终回归样本中的“被提示公司 × 交易日”
firm_days_in_sample = (
    TRD_Dalyr[
        ["StkcdRecall", "Trddt"]
    ]
    .drop_duplicates()
    .copy()
)

# 二、从折叠前配对明细中，只保留基准大市值提示事件
cue_pairs = (
    mm.loc[
        mm["quarterLag1size90plus"].eq(1),
        ["Stkcd", "event_Trddt", "StkcdRecall"]
    ]
    .rename(columns={"event_Trddt": "Trddt"})
    .dropna(subset=["Stkcd", "StkcdRecall", "Trddt"])
    .drop_duplicates(
        ["Stkcd", "Trddt", "StkcdRecall"]
    )
    .copy()
)

# 三、只保留确实存在于最终回归样本中的被提示公司日
cue_pairs_final_sample = cue_pairs.merge(
    firm_days_in_sample,
    on=["StkcdRecall", "Trddt"],
    how="inner",
    validate="many_to_one",
)

# 四、检查“提示公司 × 日期 × 被提示公司”是否唯一
duplicate_count = cue_pairs_final_sample.duplicated(
    ["Stkcd", "Trddt", "StkcdRecall"]
).sum()

if duplicate_count != 0:
    raise ValueError(
        "提示公司、日期和被提示公司的组合存在重复，请先检查配对明细。"
    )

# 五、按“提示公司 × 公告交易日”统计被提示公司数量
nr_firms_cued = (
    cue_pairs_final_sample
    .groupby(
        ["Stkcd", "Trddt"],
        observed=True,
    )["StkcdRecall"]
    .nunique()
    .rename("nrFirmsCued")
)

# 六、生成与原文 summarize, detail 类似的描述性统计
nr_firms_cued_summary = pd.Series(
    {
        "提示公司公告次数": nr_firms_cued.shape[0],
        "均值": nr_firms_cued.mean(),
        "第25百分位": nr_firms_cued.quantile(0.25),
        "中位数": nr_firms_cued.median(),
        "第75百分位": nr_firms_cued.quantile(0.75),
        "标准差": nr_firms_cued.std(ddof=1),
        "最小值": nr_firms_cued.min(),
        "最大值": nr_firms_cued.max(),
    },
    name="每次公告提示的公司数量",
)

print("\n一次提示公司公告会提示多少家被提示公司：")
print(nr_firms_cued_summary)




print("基准样本筛选后补生成其余缩尾变量......")

for col in winsor_source_cols:
    w_col = f"{col}_w"
    if col in TRD_Dalyr.columns and w_col not in TRD_Dalyr.columns:
        add_winsor_col(TRD_Dalyr, col)

print("最终同时进入两个基准回归的样本量：", len(TRD_Dalyr))
print("最终 cue 数量：", TRD_Dalyr["quarterLag1size90plus"].sum())

#===============================================================================
# 构造 future cue 虚拟变量：cue_tp1 到 cue_tp10
# cue_tp{i}=1 表示当前日后第 i 个交易日会发生基准 cue
#===============================================================================

# 当前主面板唯一层级应为 StkcdRecall × td
dup_n = TRD_Dalyr.duplicated(["StkcdRecall", "td"]).sum()
print("构造 future cue 前 StkcdRecall-td 重复数：", dup_n)

if dup_n != 0:
    raise ValueError("TRD_Dalyr 中 StkcdRecall-td 不唯一，不能构造 future cue。")

# 提取真实 cue 发生日
cue_days = (
    TRD_Dalyr.loc[
        TRD_Dalyr["quarterLag1size90plus"].eq(1),
        ["StkcdRecall", "td"]
    ]
    .drop_duplicates()
    .copy()
)

print("基准 cue 事件数：", len(cue_days))

for i in range(1, 11):
    # 如果 cue 发生在 td = T，
    # 那么 td = T - i 的那一行应标记 cue_tp{i}=1
    future_cue = cue_days.copy()
    future_cue["td"] = future_cue["td"] - i

    future_cue = (
        future_cue
        .drop_duplicates(["StkcdRecall", "td"])
        .assign(**{f"cue_tp{i}": 1})
    )

    TRD_Dalyr = TRD_Dalyr.merge(
        future_cue,
        on=["StkcdRecall", "td"],
        how="left",
        validate="1:1"
    )

    TRD_Dalyr[f"cue_tp{i}"] = (
        TRD_Dalyr[f"cue_tp{i}"]
        .fillna(0)
        .astype("int8")
    )

print("future cue 虚拟变量生成完成。")
print(TRD_Dalyr[[f"cue_tp{i}" for i in range(1, 11)]].sum())

#===============================================================================
# 删除最终输出不需要的中间变量
#===============================================================================

# 确认原始收益率已经按原文命名，避免误删 raw_ret_* 里的核心收益变量
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
    # 上游新闻表的 AES/AEV 原值；最终只保留四个新闻变化控制变量
    "AES",
    "AEV",

    # smallFirm 和 interference 高低组的中间分组变量
    "marketCap_Minus3_Recall_medGr",
    "interferenceGroupEWLag1S90p",
    "interferenceGroupEWLag1S5090",
    "interferenceGroupVWLag1S90p",
    "interferenceGroupVWLag1S5090",

    # 两张自身公告控制表的来源字段，最终只保留回归接口列
    "ownPerformanceForecastBrief",
    "ownOtherEventER",

    # 日度面板和 DGTW 构造中的辅助字段
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

# raw_ret_* 中剩下的应只是不再需要的 nobs / valid_full 等辅助字段
drop_cols_final += [
    c for c in TRD_Dalyr.columns
    if c.startswith("raw_ret_")
]

drop_cols_final = list(dict.fromkeys(drop_cols_final))
drop_cols_existing = [c for c in drop_cols_final if c in TRD_Dalyr.columns]

print("最终输出前删除中间变量数量：", len(drop_cols_existing))
print(drop_cols_existing)

TRD_Dalyr = TRD_Dalyr.drop(columns=drop_cols_existing)

#===============================================================================
# 变量中文标签
#===============================================================================

variable_labels = {
    "StkcdRecall": "被提示公司股票代码",
    "Trddt": "交易日期",
    "td": "交易日序号",
    "marketCap_Minus3_Recall": "被提示公司 t-3 日总市值",

    "return_t_Recall": "被提示公司 t 日原始收益率",
    "ownEA": "自身盈余公告日虚拟变量，等于1表示被提示公司在 t 日发布盈余公告",

    "ownMajorDisclosure": "自身专题事件控制变量，等于1表示被提示公司 t 日发生十类专题事件之一",
    "ownPerformanceForecast": "自身业绩预告或业绩快报控制变量，等于1表示被提示公司 t 日发生相应公告",
    "aesIncrease": "新闻情绪上升虚拟变量，等于1表示被提示公司 t 日新闻情绪指标较上一真实交易日上升；由上游真实交易日新闻表构造",
    "aesDecrease": "新闻情绪下降虚拟变量，等于1表示被提示公司 t 日新闻情绪指标较上一真实交易日下降；由上游真实交易日新闻表构造",
    "logIncreaseAEV": "新闻覆盖量上升幅度的对数值，log(1+上升量)",
    "logDecreaseAEV": "新闻覆盖量下降幅度的对数值，log(1+下降量)",

    "nrCuesLag1S90p": "lag1 大市值提示公司数量，提示公司市值高于90分位",
    "nrCuesLag1S5090": "lag1 中高市值提示公司数量，提示公司市值位于50-90分位",
    "nrCuesLag1S1050": "lag1 中低市值提示公司数量，提示公司市值位于10-50分位",
    "nrCuesLag1S10m": "lag1 小市值提示公司数量，提示公司市值低于10分位",

    "quarterLag1size90plus": "基准 cue 虚拟变量，等于1表示存在 lag1 大市值提示事件",
    "quarterLag1size50_90": "中高市值 cue 虚拟变量，等于1表示存在50-90分位市值提示事件",
    "quarterLag1size10_50": "中低市值 cue 虚拟变量，等于1表示存在10-50分位市值提示事件",
    "quarterLag1size10minus": "小市值 cue 虚拟变量，等于1表示存在10分位以下市值提示事件",

    "lowInterferenceEWLag1S90p": "低干扰大市值 cue 虚拟变量，等权 interference 分组为低组",
    "highInterferenceEWLag1S90p": "高干扰大市值 cue 虚拟变量，等权 interference 分组为高组",
    "lowInterferenceVWLag1S90p": "低干扰大市值 cue 虚拟变量，市值加权 interference 分组为低组",
    "highInterferenceVWLag1S90p": "高干扰大市值 cue 虚拟变量，市值加权 interference 分组为高组",

    "nrFirmsAnnLag1EWS90p": "lag1 大市值 cue 的共同公告公司数量，按 cue 等权平均",
    "nrFirmsAnnLag1EWS5090": "lag1 中高市值 cue 的共同公告公司数量，按 cue 等权平均",
    "nrFirmsAnnLag1VWS90p": "lag1 大市值 cue 的共同公告公司数量，按提示公司市值加权平均",
    "nrFirmsAnnLag1VWS5090": "lag1 中高市值 cue 的共同公告公司数量，按提示公司市值加权平均",

    "smallFirm": "小市值被提示公司虚拟变量，等于1表示被提示公司 t-3 市值低于当年中位数",
    "quarter": "交易日期所属日历季度",
    "yearQuarter": "交易日期所属年度-季度",
    "year": "交易日期所属年份",
}

for i in range(1, 4):
    variable_labels[f"return_tm{i}_Recall"] = f"被提示公司 t-{i} 日原始收益率"
    variable_labels[f"chAdjreturn_tm{i}_Recall"] = f"被提示公司 t-{i} 日特征调整收益率"

for i in range(1, 51):
    variable_labels[f"return_t_tp{i}_Recall"] = f"被提示公司 [t, t+{i}] 窗口原始累计收益率"
    variable_labels[f"chAdjreturn_t_tp{i}_Recall"] = (
        f"被提示公司 [t, t+{i}] 窗口特征调整累计收益率"
    )

for i in range(1, 11):
    variable_labels[f"ownEA_tp{i}"] = f"未来自身公告控制变量，等于1表示被提示公司在 t+{i} 日发布盈余公告"
    variable_labels[f"ownEA_tm{i}"] = f"过去自身公告控制变量，等于1表示被提示公司在 t-{i} 日发布盈余公告"
    variable_labels[f"cue_tp{i}"] = f"未来 cue 控制变量，等于1表示被提示公司在 t+{i} 日发生基准 cue"

for sue_id in [1, 2, 3]:
    variable_labels[f"sue{sue_id}CueLag1S90pVW"] = (
        f"lag1 大市值提示公司的 SUE{sue_id} 盈余惊喜，按提示公司市值加权平均"
    )
    variable_labels[f"sue{sue_id}CueLag1S90pEW"] = (
        f"lag1 大市值提示公司的 SUE{sue_id} 盈余惊喜，按 cue 等权平均"
    )
    variable_labels[f"sue{sue_id}CueLag1VWS90p_quintile"] = (
        f"lag1 大市值提示公司市值加权 SUE{sue_id} 盈余惊喜五分位"
    )
    variable_labels[f"sue{sue_id}CueLag1EWS90p_quintile"] = (
        f"lag1 大市值提示公司等权 SUE{sue_id} 盈余惊喜五分位"
    )

print("正在输出最终文件......")

TRD_Dalyr.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d"
)

print("最终文件输出成功!")
