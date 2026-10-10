# 除金融行业股票、新上市一年内的股票从一开始剔除外，其他筛选分四阶段：
#
# 1. 断点分位数样本：严格
#    ① 要求 sizew、BM、momentum 可得；
#    ② [t-252,t-20] 以及 [t-20,t] 窗口内 ST/*ST/PT 天数不能过多、流动性不能太差；
#    ③ 对 momentum 两个端点施加陈旧度约束，防止动量信号不可靠；
#    ④ 仅使用主板/原口径市场股票计算 size、BM、momentum 分位点。
#
# 2. DGTW 标签分配样本：相对宽松
#    ① 不要求参与断点；
#    ② 只要 sizew、BM、momentum 可得，且 momentum 端点不过度陈旧，
#       即可使用第1步得到的断点分配 DGTW 组合标签；
#    ③ 因此，部分不能参与断点的股票仍可拥有 DGTW 标签。
#
# 3. benchmark 成分池：严格
#       benchmark 按 DGTW 组合标签分组计算，因此成分股必须拥有可靠标签；
#    ① benchmark 基础池要求形成期 momentum 可靠：
#       [t-252,t-20] 以及 [t-20,t] 窗口内 ST/*ST/PT、停牌填充、无交易不能过多；
#       并施加 momentum 端点陈旧度约束；
#    ② benchmark 池不要求必须是主板市场股票                                  ；
#    ③ benchmark 在持有期内按收益窗口动态筛选：
#       - t 日 benchmark：当日 ST/*ST/PT、停牌/暂停上市填充、成交量为0或收益缺失的股票不参与；
#       - [t,t+1] benchmark：窗口内出现 ST/*ST/PT 或成交量为0的股票不参与；
#       - [t,t+10] benchmark：窗口内出现 ST/*ST/PT 的股票不参与；
#         t 和 t+10 需正常交易，窗口内无交易天数不超过3天。
#
# 4. 主回归样本：严格
#    保留同时能够计算 [t,t+1] 和 [t,t+10] 特征调整收益率的 firm-day：
#    ① [t,t+1] 内出现 ST/*ST/PT 或成交量为0则剔除；
#    ② [t,t+10] 内出现 ST/*ST/PT 则剔除；
#       t 和 t+10 需正常交易，窗口内无交易天数不超过3天；
#    ③ 要求对应 DGTW benchmark 非缺失，因此 Return_t_t1 和 Return_t_t10 均非缺失。


import  pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
from dateutil.relativedelta import * 
from pandas.tseries.offsets import * 
import warnings
warnings.filterwarnings('ignore') 
pd.options.mode.chained_assignment = None 
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)#不自动转行
pd.set_option('display.width', 1000)#设置很长的虚拟宽度

USE_MOM_STALE_FILTER=True
USE_MOM_LIQ_FILTER=True
USE_MOM_ST_FILTER=True
SORT_MODE='direct'
EXPORT_SAMPLE_KEY=True

#读取日度股价的数据，用于提取市值并排名、计算动量(填充了停牌和暂停上市的数据)
TRD_Dalyr = pd.read_csv(
'/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMAR/个股日度收益率（升级测试）/CSMAR_日个股回报率(升级测试)_2000_2025.csv',
    encoding='utf-8')

TRD_Dalyr = TRD_Dalyr.loc[
    (TRD_Dalyr['Markettype'].isin([1,4,16,32]))
    &(TRD_Dalyr['Filling'].isin([0,1]))].copy()

#筛选A股市场的股票，包括主板以及科创板、中小板。保留填充了非节假日的数据。
TRD_Dalyr['Trddt'] = pd.to_datetime(TRD_Dalyr['Trddt'])
TRD_Dalyr['jdate'] = TRD_Dalyr['Trddt'] + MonthEnd(0)
TRD_Dalyr['size'] = TRD_Dalyr['Dsmvtll'] * 1000  
TRD_Dalyr['sizew'] = TRD_Dalyr['Dsmvosd'] * 1000  
TRD_Dalyr=TRD_Dalyr.rename(columns={'Dretwd':'ret'})

def normalize_stock_code(x:object) -> str:#定义一个函数，输入任意类型的，输出为字符串
    """标准化股票代码"""
    if pd.isna(x):#如果输入的是缺失值
        return ""#返回空字符串
    s=str(x).strip()#将输入的x先转换为字符串格式，并清除前后的空白字符后赋值给s
    if s=="":#如果s是空（字符串）
        return ""#那么返回空（字符串）
    return f"{int(float(s)):06d}"
#否则先将s转换为浮点数，然后转换为整数，最后左边补0为6位整数。f"..."最终返回为字符串

TRD_Dalyr['Stkcd']=TRD_Dalyr['Stkcd'].apply(normalize_stock_code)
TRD_Dalyr=TRD_Dalyr[['Stkcd','Trddt','jdate','sizew','size','ret','Trdsta','Markettype','Filling','Dnshrtrd']]

# =========================================================
# 按“当期行业”剔除金融行业股票：使用 STK_INDUSTRYCLASS
# =========================================================

# 证监会行业分类2012年版
# 对应《上市公司行业分类指引（2012年修订）》。证监会公告明确写的是：“自公布之日起施行”，公告日期为 2012年10月26日。

#《中国上市公司协会上市公司行业统计分类指引》，正式施行时间是 2023-05-01。
# 中上协官网的《指引》附则明确写的是：“于2023年5月1日施行。”

industry = pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMAR/上市公司行业分类表/STK_INDUSTRYCLASS.csv',
    encoding='utf-8'
)

industry = industry[['ImplementDate', 'IndustryClassificationID', 'Symbol', 'IndustryCode']].copy()
industry['ImplementDate'] = pd.to_datetime(industry['ImplementDate'])
industry['IndustryCode'] = industry['IndustryCode'].astype(str).str.strip().str.upper()

industry = industry.loc[
    industry['IndustryClassificationID'].isin(['P0201', 'P0207', 'P0221'])
].copy()

industry = industry.dropna(subset=['ImplementDate', 'Symbol', 'IndustryCode'])
#只要这三个字段里缺了任何一个，这行数据就直接扔掉。

industry['is_fin'] = (
    ((industry['IndustryClassificationID'] == 'P0201') & industry['IndustryCode'].str.startswith('I')) |
    ((industry['IndustryClassificationID'] == 'P0207') & industry['IndustryCode'].str.startswith('J')) |
    ((industry['IndustryClassificationID'] == 'P0221') & industry['IndustryCode'].str.startswith('J'))
).astype(int)

industry = industry.rename(columns={
    'Symbol': 'Stkcd',
    'IndustryClassificationID': 'std_id'
})

industry['Stkcd']=industry['Stkcd'].apply(normalize_stock_code)

industry = (
    industry
    .sort_values(['Stkcd', 'std_id','ImplementDate' ])
    .drop_duplicates(['Stkcd', 'std_id', 'ImplementDate'], keep='last')
)
TRD_Dalyr['std_id'] = np.select(
    [
        TRD_Dalyr['Trddt'] < pd.Timestamp('2012-10-26'),
        (TRD_Dalyr['Trddt'] >= pd.Timestamp('2012-10-26')) & (TRD_Dalyr['Trddt'] < pd.Timestamp('2023-05-01')),
        TRD_Dalyr['Trddt'] >= pd.Timestamp('2023-05-01')
    ],
    ['P0201', 'P0207', 'P0221'],
    default='P0201'
)
# pd.Timestamp ()：Pandas 库中专门生成「时间戳 / 日期对象」的函数
# 把文本格式的日期（如'2023-05-01'）转换成 Pandas 能识别、能直接比较的日期类型

left = TRD_Dalyr.sort_values([ 'Trddt','Stkcd', 'std_id']).reset_index(drop=True).copy()
#.reset_index(drop=True)：生成新的索引（并且扔掉旧的索引）
right = industry[['ImplementDate', 'Stkcd', 'std_id', 'IndustryCode', 'is_fin']] \
    .dropna(subset=['ImplementDate']) \
    .sort_values(['ImplementDate', 'Stkcd', 'std_id']) \
    .reset_index(drop=True) \
    .copy()

TRD_Dalyr_fin = pd.merge_asof(
    left,
    right,
    left_on='Trddt',
    right_on='ImplementDate',
    by=['Stkcd', 'std_id'],
    direction='backward',
    allow_exact_matches=True#如果交易日期 = 行业实施日期，允许直接精确匹配；如果没有，再找更早的行业
)
#pd.merge_asof ： 时间序列「最近匹配」，专门解决「非每日数据」匹配；


# =========================================================
# 检查一下行业表的匹配情况
# =========================================================
# 1. 先把匹配结果拆成「成功」和「失败」两部分
match_success = TRD_Dalyr_fin[TRD_Dalyr_fin['is_fin'].notna()].copy()
match_fail = TRD_Dalyr_fin[TRD_Dalyr_fin['is_fin'].isna()].copy()
print("="*80)
print(f"1. 匹配总体情况")
print(f"   总观测数：{len(TRD_Dalyr_fin)}")
print(f"   匹配成功：{len(match_success)}（占比 {len(match_success)/len(TRD_Dalyr_fin):.2%}）")
print(f"   匹配失败：{len(match_fail)}（占比 {len(match_fail)/len(TRD_Dalyr_fin):.2%}）")
print("="*80)
if len(match_fail) > 0:
    #看一下失败样本的【股票列表】（是个别股票还是普遍现象？）
    print(f"\n   --- 涉及股票 ---")
    print(f"   匹配失败的股票共 {match_fail['Stkcd'].nunique()} 只")
    print(f"   前20只失败次数最多的股票：")
    print(match_fail['Stkcd'].value_counts().head(20).to_string())

original_len = len(TRD_Dalyr_fin)
fin_obs = TRD_Dalyr_fin['is_fin'].fillna(0).eq(1).sum()
fin_stocks = TRD_Dalyr_fin.loc[TRD_Dalyr_fin['is_fin'].fillna(0).eq(1), 'Stkcd'].nunique()
TRD_Dalyr1 = TRD_Dalyr_fin.loc[
    TRD_Dalyr_fin['is_fin'].fillna(0) != 1
].copy()

print(f"按当期行业口径识别出曾在某些时点属于金融行业的股票共 {fin_stocks} 只")
print(f"删除的金融行业 股票-日期 观测共 {fin_obs} 条")
print(f"已从日度收益文件中按当期行业剔除金融股，样本从{original_len}下降为{len(TRD_Dalyr1)}")

TRD_Dalyr1 = TRD_Dalyr1.drop(columns=['std_id', 'ImplementDate', 'is_fin']).copy()

TRD_Co=pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMAR/公司文件/TRD_Co.csv',
    encoding='utf-8'
)

TRD_Co['Stkcd']=TRD_Co['Stkcd'].apply(normalize_stock_code)
TRD_Co['Listdt']=pd.to_datetime(TRD_Co['Listdt'],errors='coerce')
TRD_Dalyr1=pd.merge(
    TRD_Dalyr1,
    TRD_Co[['Stkcd','Listdt']],
    how='left',
    on='Stkcd'
)
TRD_Dalyr1['Listdt_1y']=TRD_Dalyr1['Listdt']+pd.DateOffset(years=1)
TRD_Dalyr2=TRD_Dalyr1.loc[TRD_Dalyr1['Trddt']>TRD_Dalyr1['Listdt_1y']]
TRD_Dalyr2=TRD_Dalyr2.drop(columns=['Listdt_1y'])
print(f'删除上市不满一年观测值前的样本数：{len(TRD_Dalyr1)}')
print(f'删除上市不满一年观测值后的样本数：{len(TRD_Dalyr2)}')
print(f'删除上市不满一年观测值共：{len(TRD_Dalyr1)-len(TRD_Dalyr2)}条')


################################################################################
################ 1. 日度底表：算基础变量、生成动量可靠或者可得的两个信号################
################################################################################
TRD_Cale=pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMAR/日历文件2000-01-01至2025-12-31/TRD_Cale.csv',
    encoding='utf-8'
)

TRD_Cale['Clddt']=pd.to_datetime(TRD_Cale['Clddt'])
TRD_Cale=TRD_Cale.loc[(TRD_Cale['Markettype']==1)&
                      (TRD_Cale['State']=='O')]
TRD_Cale=TRD_Cale.rename(columns=
                {'Clddt':'Trddt'})
TRD_Cale=TRD_Cale.sort_values(['Trddt']).reset_index(drop=True)
TRD_Cale['td']=np.arange(1,len(TRD_Cale)+1)


TRD_Dalyr2=pd.merge(
    TRD_Dalyr2,
    TRD_Cale[['Trddt','td']],
    on=['Trddt'],
    how='left'
)
print(f'没有匹配到交易序号的观测数：{TRD_Dalyr2['td'].isna().sum()}')

TRD_Dalyr2 = TRD_Dalyr2.sort_values(['Stkcd', 'Trddt']).reset_index(drop=True)

# 价格指数
TRD_Dalyr2['prcIndex'] = (
    TRD_Dalyr2.groupby('Stkcd')['ret']
    .transform(lambda x: (1 + x).cumprod())
)

#准备端点表
endpoint=(
    TRD_Dalyr2[['Stkcd','td','prcIndex','sizew']]
    .drop_duplicates(['Stkcd','td'])
    .copy()
)

def add_endpoint_value(df,endpoint,offset,input_col,out_col):
    """
    给df增加某个市场交易日偏移端点的变量。
    offset = -20  表示 t-20
    offset = -252 表示 t-252
    offset = 1    表示 t+1
    offset = 10   表示 t+10
    """
    tmp=endpoint[['Stkcd','td',input_col]].copy()
    tmp=tmp.rename(columns={
        'td':'target_td',
        input_col:out_col
    })
    out=df.copy()
    out['target_td']=out['td']+ offset
    
    out=out.merge(
        tmp,
        on=['Stkcd','target_td'],
        how='left',
    )

    out=out.drop(columns=['target_td'])
    return out

TRD_Dalyr2=add_endpoint_value(
    TRD_Dalyr2,endpoint,
    offset=-20,
    input_col='prcIndex',
    out_col='prcIndex_t20'
)


TRD_Dalyr2=add_endpoint_value(
    TRD_Dalyr2,endpoint,
    offset=-252,
    input_col='prcIndex',
    out_col='prcIndex_t252'
)

TRD_Dalyr2['return_tm252_tm20']=(
    TRD_Dalyr2['prcIndex_t20']/TRD_Dalyr2['prcIndex_t252']-1
)*100

TRD_Dalyr2=TRD_Dalyr2.drop(columns=['prcIndex_t20','prcIndex_t252'])



TRD_Dalyr2=add_endpoint_value(
    TRD_Dalyr2,endpoint,
    offset=-1,
    input_col='prcIndex',
    out_col='prcIndex_tm1'
)


TRD_Dalyr2=add_endpoint_value(
    TRD_Dalyr2,endpoint,
    offset=1,
    input_col='prcIndex',
    out_col='prcIndex_tp1'
)

TRD_Dalyr2=add_endpoint_value(
    TRD_Dalyr2,endpoint,
    offset=10,
    input_col='prcIndex',
    out_col='prcIndex_tp10'
)

TRD_Dalyr2['raw_ret_t']=(
    TRD_Dalyr2['prcIndex']/TRD_Dalyr2['prcIndex_tm1']-1
)*100

TRD_Dalyr2['raw_ret_t_t1']=(
    TRD_Dalyr2['prcIndex_tp1']/TRD_Dalyr2['prcIndex_tm1']-1
)*100

TRD_Dalyr2['raw_ret_t_t10']=(
    TRD_Dalyr2['prcIndex_tp10']/TRD_Dalyr2['prcIndex_tm1']-1
)*100

TRD_Dalyr2=TRD_Dalyr2.drop(columns=['prcIndex_tp1','prcIndex_tm1','prcIndex_tp10'])

TRD_Dalyr2=add_endpoint_value(
    TRD_Dalyr2,endpoint,
    offset=-3,
    input_col='sizew',
    out_col='weight_t3'
)

#### 判断动量有效性
## ①端点陈旧度约束
# 最近一次真实交易日（Filling=0 表示真实交易日）
TRD_Dalyr2['last_trade_td'] = TRD_Dalyr2['td'].where(TRD_Dalyr2['Filling'].eq(0))
TRD_Dalyr2['last_trade_td'] = TRD_Dalyr2.groupby('Stkcd')['last_trade_td'].ffill()
# 市场交易日序号 -> 陈旧度stale：当前交易日序号 - 上一次真实成交的交易日序号
TRD_Dalyr2['stale'] = TRD_Dalyr2['td'] - TRD_Dalyr2['last_trade_td']
TRD_Dalyr2=TRD_Dalyr2.drop(columns=['last_trade_td'])
print(f'端点陈旧度的均值：{TRD_Dalyr2['stale'].mean()}')
# 动量端点陈旧度
endpoint=(
    TRD_Dalyr2[['Stkcd','td','stale']]
    .drop_duplicates(['Stkcd','td'])
    .copy()
)
TRD_Dalyr2=add_endpoint_value(
    TRD_Dalyr2,endpoint,
    offset=-20,
    input_col='stale',
    out_col='stale_t20'
)
TRD_Dalyr2=add_endpoint_value(
    TRD_Dalyr2,endpoint,
    offset=-252,
    input_col='stale',
    out_col='stale_t252'
)

TRD_Dalyr2=TRD_Dalyr2.drop(columns=['stale'])

## ②流动性约束：近期窗口与动量形成窗口分别计算
# has_trading_volume 表示成交量大于0。
TRD_Dalyr2['has_trading_volume'] = TRD_Dalyr2['Dnshrtrd'].fillna(0).gt(0)
TRD_Dalyr2=TRD_Dalyr2.sort_values(['Stkcd','td']).reset_index(drop=True)
# 近期窗口：默认 [t-20, t]
TRD_Dalyr2['trade_days_recent'] = (
    TRD_Dalyr2.groupby('Stkcd')['has_trading_volume']
    .rolling(window=20,min_periods=20)
    .sum()
    .reset_index(level=0, drop=True)
)
# 动量形成窗口：[t-252, t-20]，长度 233 个交易日。
TRD_Dalyr2['trade_days_mom'] = (
    TRD_Dalyr2.groupby('Stkcd')['has_trading_volume']
    .transform(lambda x: x.shift(20).rolling(window=233).sum())
)

## ③ST、*ST、PT天数约束
# 正常状态标记（非ST、*ST、PT股票）
TRD_Dalyr2['non_ST'] = TRD_Dalyr2['Trdsta'].isin([1, 4, 7, 10, 13])
# ST / *ST / PT 标记
TRD_Dalyr2['ST_flag'] = (~TRD_Dalyr2['non_ST']).astype(int)
# 近期窗口内 ST / *ST / PT 天数
TRD_Dalyr2['st_days_recent'] = (
    TRD_Dalyr2.groupby('Stkcd')['ST_flag']
    .rolling(window=20,min_periods=20)
    .sum()
    .reset_index(level=0, drop=True)
)
# 动量形成窗口 [t-252, t-20] 的 ST / *ST / PT 天数
TRD_Dalyr2['st_days_mom'] = (
    TRD_Dalyr2.groupby('Stkcd')['ST_flag']
    .transform(lambda x: x.shift(20).rolling(window=233).sum())
)

#可得的动量需要满足动量指标不为空、端点不陈旧
avaliable_mom = (
    (TRD_Dalyr2['return_tm252_tm20'].notna())&
    (TRD_Dalyr2['stale_t20'] <= 5) &
        (TRD_Dalyr2['stale_t252'] <= 20)
    )

#可靠的动量额外要求流动性足够、不是长期ST股票
reliable_mom = avaliable_mom
if USE_MOM_LIQ_FILTER:#流动性约束
    reliable_mom = (
        reliable_mom &
        (TRD_Dalyr2['trade_days_recent'] >= 10) &
        (TRD_Dalyr2['trade_days_mom'] >= 180)
    )

if USE_MOM_ST_FILTER:#ST约束
    reliable_mom = (
        reliable_mom &
        (TRD_Dalyr2['st_days_recent'] <= 10) &
        (TRD_Dalyr2['st_days_mom'] <= 120)
    )

# 标签分配阶段使用的最低限度 momentum 可靠性：momentum 可得，且两个端点不过度陈旧。
# 与 valid_mom_signal 不同，它不要求整个动量窗口/近期窗口的ST、停牌、流动性都满足严格条件。
TRD_Dalyr2['avaliable_mom'] = avaliable_mom 
TRD_Dalyr2['reliable_mom'] = reliable_mom

TRD_Dalyr2 = TRD_Dalyr2.drop(
    columns=[
        'stale_t20','stale_t252',
        'trade_days_recent','trade_days_mom',
        'st_days_recent','st_days_mom',
    ],
    errors='ignore'
)

################################################################################
################################ 2. 计算bm，合并凑齐 size/bm/mom ##################
################################################################################

# 月末面板：每只股票每月最后一个交易日观测
TRD_Mnth = (
    TRD_Dalyr2
    .sort_values(['Stkcd', 'Trddt'])
    .groupby(['Stkcd', 'jdate'])
    .tail(1)
    .copy()
)
# -------------------------------------------------
# 额外准备：上一年12月末市值（用于 BM 分母）
# 注意：这里用  size，而不是 sizew
# -------------------------------------------------
dec_me = TRD_Mnth.loc[TRD_Mnth['jdate'].dt.month == 12, ['Stkcd', 'jdate', 'size']].copy()
dec_me['Connect_year'] = dec_me['jdate'].dt.year
dec_me = dec_me.rename(columns={'size': 'me_dec'})
dec_me = dec_me[['Stkcd', 'Connect_year', 'me_dec']].copy()

# 只取 6 月作为形成期
TRD_Mnth_6 = TRD_Mnth.loc[TRD_Mnth['jdate'].dt.month == 6].copy()
TRD_Mnth_6['Connect_year'] = TRD_Mnth_6['jdate'].dt.year - 1

# 读取资产负债表，计算上年12月末账面价值
FS_Combas = pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMAR/资产负债表2000-01-01至2025-12-31/FS_Combas.csv',
    encoding='utf-8'
)

comp = FS_Combas.copy()
comp = comp.loc[comp['Typrep'] == 'A']

comp = comp.rename(columns={
    'Accper': 'date',
    'A001222000': '递延所得税资产',
    'A002208000': '递延所得税负债',
    'A003112000': 'oth_eq_inst',
    'A003100000': 'eqToParent',
    'A003000000': 'teq',
    'A003112101': 'pref'
})

comp['date'] = pd.to_datetime(comp['date'])
comp = comp.loc[comp['date'].dt.month == 12].copy()
comp['year'] = comp['date'].dt.year
comp['Stkcd']=comp['Stkcd'].apply(normalize_stock_code)

comp = comp[['Stkcd', 'date', 'oth_eq_inst', 'eqToParent', 'teq',
             'year', '递延所得税负债', '递延所得税资产', 'pref']].copy()

# 账面权益
comp['eq'] = np.where(comp['eqToParent'].isnull(), comp['teq'], comp['eqToParent'])
comp['eq'] = np.where(comp['eq'].isnull(), 0, comp['eq'])

comp['oth_eq_inst'] = np.where(comp['oth_eq_inst'].isnull(), comp['pref'], comp['oth_eq_inst'])
comp['oth_eq_inst'] = np.where(comp['oth_eq_inst'].isnull(), 0, comp['oth_eq_inst'])

comp['递延所得税负债'] = comp['递延所得税负债'].fillna(0)
comp['递延所得税资产'] = comp['递延所得税资产'].fillna(0)

comp['be'] = (
    comp['eq']
    - comp['oth_eq_inst']
    + comp['递延所得税负债']
    - comp['递延所得税资产']
)

comp = comp.loc[comp['be'] >= 0, ['Stkcd', 'year', 'be']].copy()
comp = comp.rename(columns={'year': 'Connect_year'})


# 合并上年末账面价值，计算 BM
TRD_Mnth_6 = pd.merge(
    TRD_Mnth_6,
    comp,
    how='left',
    on=['Stkcd', 'Connect_year']
)
# 合并上一年12月末市值（BM分母）
TRD_Mnth_6 = pd.merge(
    TRD_Mnth_6,
    dec_me,
    how='left',
    on=['Stkcd', 'Connect_year']
)
# BM = 上年12月账面价值 / 上年12月末市值
TRD_Mnth_6['bm'] = np.where(
    TRD_Mnth_6['me_dec'] > 0,
    TRD_Mnth_6['be'] / TRD_Mnth_6['me_dec'],
    np.nan
)

TRD_Mnth_6 = TRD_Mnth_6.drop(
    columns=['be', 'me_dec', 'Connect_year','raw_ret_t_t1','raw_ret_t_t10','raw_ret_t'])

################################################################################
################################ 3. 分别判定能参与端点和能获得标签的 #################
################################################################################

break_point_eligble = (
    TRD_Mnth_6['sizew'].notna() &
    TRD_Mnth_6['bm'].notna() &
    TRD_Mnth_6['return_tm252_tm20'].notna()&
    TRD_Mnth_6['reliable_mom']&
    TRD_Mnth_6['Markettype'].isin([1,4])
)

assign_eligble=(
    TRD_Mnth_6['sizew'].notna() &
    TRD_Mnth_6['bm'].notna() &
    TRD_Mnth_6['return_tm252_tm20'].notna()&
    TRD_Mnth_6['avaliable_mom']
)

TRD_Mnth_6['break_point_eligble'] = break_point_eligble
TRD_Mnth_6['assign_eligble'] = assign_eligble

break_universe = TRD_Mnth_6.loc[TRD_Mnth_6['break_point_eligble']].copy()
assign_universe = TRD_Mnth_6.loc[TRD_Mnth_6['assign_eligble']].copy()

print("3. 参与断点和能获得标签的资格筛选完成")
print(f"TRD_Mnth_6 行数: {len(TRD_Mnth_6):,}")
print(f"break_point_eligble 行数: {TRD_Mnth_6['break_point_eligble'].sum():,}")
print(f"assign_eligble 行数: {TRD_Mnth_6['assign_eligble'].sum():,}")
print("=" * 80)


################################################################################
########################### 4. 用断点样本计算分位数，并给assign样本赋标签 ############
################################################################################

# =========================================================
# 排序辅助函数
# =========================================================

def make_quantile_breaks(df, by_cols, value_col, prefix):
    """
    计算指定变量的 20/40/60/80 分位点。
    
    by_cols:
        直接排序时：['jdate']
        嵌套排序时：['jdate', 'sizeGroup'] 或 ['jdate', 'sizeGroup', 'bookToMarketGroup']
    """
    bk = (
        df
        .groupby(by_cols)[value_col]
        .quantile([0.2, 0.4, 0.6, 0.8])
        .unstack()
        .reset_index()
        .rename(columns={
            0.2: f'{prefix}_p20',
            0.4: f'{prefix}_p40',
            0.6: f'{prefix}_p60',
            0.8: f'{prefix}_p80'
        })
    )
    return bk


def assign_quintile(df, value_col, prefix, out_col):
    """
    根据 20/40/60/80 分位点分成 1-5 组。
    约定：1 = 最低/最小组，5 = 最高/最大组。
    """
    x = df[value_col]
    p20 = df[f'{prefix}_p20']
    p40 = df[f'{prefix}_p40']
    p60 = df[f'{prefix}_p60']
    p80 = df[f'{prefix}_p80']

    valid = (
        x.notna() &
        p20.notna() &
        p40.notna() &
        p60.notna() &
        p80.notna()
    )

    df[out_col] = np.nan

    df.loc[valid & (x <= p20), out_col] = 1
    df.loc[valid & (x > p20) & (x <= p40), out_col] = 2
    df.loc[valid & (x > p40) & (x <= p60), out_col] = 3
    df.loc[valid & (x > p60) & (x <= p80), out_col] = 4
    df.loc[valid & (x > p80), out_col] = 5

    return df


def drop_break_columns(df):
    """删除所有分位点断点列，避免后续表太宽。"""
    break_cols = [
        c for c in df.columns
        if c.endswith('_p20') or c.endswith('_p40') or c.endswith('_p60') or c.endswith('_p80')
    ]
    return df.drop(columns=break_cols, errors='ignore')


# =========================================================
# 4A. 直接排序：Memory Moves Markets 作者 code 口径
# =========================================================
if SORT_MODE == "direct":

    print("4. 当前排序模式：direct，size / BM / momentum 独立三重排序")

    # 三个变量都在同一个形成期横截面内分别计算断点。
    size_bk = make_quantile_breaks(
        break_universe,
        by_cols=['jdate'],
        value_col='sizew',
        prefix='size'
    )

    bm_bk = make_quantile_breaks(
        break_universe,
        by_cols=['jdate'],
        value_col='bm',
        prefix='bm'
    )

    mom_bk = make_quantile_breaks(
        break_universe,
        by_cols=['jdate'],
        value_col='return_tm252_tm20',
        prefix='mom'
    )

    # 标签分配使用较宽松 assign 样本；这些股票不参与断点，但可以拿到DGTW标签。
    port = (
        assign_universe
        .merge(size_bk, on='jdate', how='left')
        .merge(bm_bk, on='jdate', how='left')
        .merge(mom_bk, on='jdate', how='left')
    )

    port = assign_quintile(
        port,
        value_col='sizew',
        prefix='size',
        out_col='sizeGroup'
    )

    port = assign_quintile(
        port,
        value_col='bm',
        prefix='bm',
        out_col='bookToMarketGroup'
    )

    port = assign_quintile(
        port,
        value_col='return_tm252_tm20',
        prefix='mom',
        out_col='momentumGroup'
    )

    port = drop_break_columns(port)


# =========================================================
# 4B. 嵌套排序：经典 DGTW 口径
# =========================================================
elif SORT_MODE == "nested":

    print("4. 当前排序模式：nested，size -> size内BM -> size×BM内momentum 嵌套排序")

    # -------------------------
    # 4.1 Size 断点与分组
    # -------------------------

    size_bk = make_quantile_breaks(
        break_universe,
        by_cols=['jdate'],
        value_col='sizew',
        prefix='size'
    )

    # 先给assign样本分size组。
    port = assign_universe.merge(size_bk, how='left', on='jdate')
    port = assign_quintile(
        port,
        value_col='sizew',
        prefix='size',
        out_col='sizeGroup'
    )
    port = drop_break_columns(port)

    # 同时给break样本分size组，用于后续嵌套断点。
    break_port = break_universe.merge(size_bk, how='left', on='jdate')
    break_port = assign_quintile(
        break_port,
        value_col='sizew',
        prefix='size',
        out_col='sizeGroup'
    )
    break_port = drop_break_columns(break_port)

    # -------------------------
    # 4.2 BM 断点与分组：在 size 组内
    # -------------------------
    bm_break_sample = break_port.loc[
        break_port['sizeGroup'].notna()
    ].copy()

    bm_bk = make_quantile_breaks(
        bm_break_sample,
        by_cols=['jdate', 'sizeGroup'],
        value_col='bm',
        prefix='bm'
    )

    port = port.merge(
        bm_bk,
        how='left',
        on=['jdate', 'sizeGroup']
    )
    port = assign_quintile(
        port,
        value_col='bm',
        prefix='bm',
        out_col='bookToMarketGroup'
    )
    port = drop_break_columns(port)

    break_port = break_port.merge(
        bm_bk,
        how='left',
        on=['jdate', 'sizeGroup']
    )
    break_port = assign_quintile(
        break_port,
        value_col='bm',
        prefix='bm',
        out_col='bookToMarketGroup'
    )
    break_port = drop_break_columns(break_port)

    # -------------------------
    # 4.3 MOM 断点与分组：在 size × BM 组内
    # -------------------------
    mom_break_sample = break_port.loc[
        break_port['sizeGroup'].notna() &
        break_port['bookToMarketGroup'].notna()
    ].copy()

    mom_bk = make_quantile_breaks(
        mom_break_sample,
        by_cols=['jdate', 'sizeGroup', 'bookToMarketGroup'],
        value_col='return_tm252_tm20',
        prefix='mom'
    )

    port = port.merge(
        mom_bk,
        how='left',
        on=['jdate', 'sizeGroup', 'bookToMarketGroup']
    )
    port = assign_quintile(
        port,
        value_col='return_tm252_tm20',
        prefix='mom',
        out_col='momentumGroup'
    )
    port = drop_break_columns(port)


else:
    raise ValueError("SORT_MODE 只能是 'direct' 或 'nested'")


# =========================================================
# 4C. 形成最终 DGTW portfolio 标签
# =========================================================

port1 = port.dropna(
    subset=['sizeGroup', 'bookToMarketGroup', 'momentumGroup']
).copy()

port1[['sizeGroup', 'bookToMarketGroup', 'momentumGroup']] = (
    port1[['sizeGroup', 'bookToMarketGroup', 'momentumGroup']]
    .astype(int)
    .astype(str)
)

port1['dgtw_port'] = (
    port1['sizeGroup'] +
    port1['bookToMarketGroup'] +
    port1['momentumGroup']
)

port2 = port1[['Stkcd', 'jdate', 'dgtw_port']].copy()
port2['jyear'] = port1['jdate'].dt.year
port2 = port2.rename(columns={'jdate': 'formdate'})
port2 = port2[['Stkcd', 'formdate', 'jyear', 'dgtw_port']].copy()

print("4. DGTW 组合标签构建完成")
print(f"排序模式 SORT_MODE: {SORT_MODE}")
print(port2.head())

#port2 所有 assign_universe 中成功获得 DGTW 标签的股票-年份表。

################################################################################
########################### 4D. 给 port2 增加 benchmark 基础池标记 ################
################################################################################
# benchmark基础池：形成期 momentum 可靠，且 sizew、BM、momentum 可得
# 注意：benchmark池不限制 Markettype 为主板

benchmark_base_flag = TRD_Mnth_6.loc[
    TRD_Mnth_6['sizew'].notna() &
    TRD_Mnth_6['bm'].notna() &
    TRD_Mnth_6['return_tm252_tm20'].notna() &
    TRD_Mnth_6['reliable_mom'],
    ['Stkcd', 'jdate']
].copy()

benchmark_base_flag['jyear'] = benchmark_base_flag['jdate'].dt.year
benchmark_base_flag['benchmark_base'] = True

benchmark_base_flag = (
    benchmark_base_flag[['Stkcd', 'jyear', 'benchmark_base']]
    .drop_duplicates()
)

port2 = port2.merge(
    benchmark_base_flag,
    on=['Stkcd', 'jyear'],
    how='left'
)

port2['benchmark_base'] = port2['benchmark_base'].fillna(False)

print("4D. benchmark基础池标记完成")
print(port2[['Stkcd', 'formdate', 'jyear', 'dgtw_port', 'benchmark_base']].head())
print("=" * 80)


################################################################################
################################## 5. benchmark：计算 ###########################
################################################################################

# 日度持有表：直接从完整日度底表出发
TRD_Dalyr3 = TRD_Dalyr2[
    [
        'Stkcd', 'Trddt', 'Filling', 'weight_t3', 'Trdsta',
        'raw_ret_t', 'raw_ret_t_t1', 'raw_ret_t_t10',
        'non_ST', 'has_trading_volume'
    ]
].copy()

# 形成期标签与持有期日期的映射：
# 7月到次年6月使用上一年6月形成的 DGTW 组合
TRD_Dalyr3['month'] = TRD_Dalyr3['Trddt'].dt.month
TRD_Dalyr3['jyear'] = np.where(
    TRD_Dalyr3['month'] <= 6,
    TRD_Dalyr3['Trddt'].dt.year - 1,
    TRD_Dalyr3['Trddt'].dt.year
)

# 合并 DGTW 标签和 benchmark 基础池标记
TRD_Dalyr3 = pd.merge(
    TRD_Dalyr3,
    port2[['Stkcd', 'jyear', 'dgtw_port', 'benchmark_base']],
    on=['Stkcd', 'jyear'],
    how='left'
)

TRD_Dalyr3['benchmark_base'] = TRD_Dalyr3['benchmark_base'].fillna(False)

# 不要提前过滤，先在完整日度面板上计算窗口条件
bench = TRD_Dalyr3.copy()
bench = bench.sort_values(['Stkcd', 'Trddt']).reset_index(drop=True)

# 正常交易日：非填充日 + 成交量大于0
bench['normal_trade_day'] = (
    bench['Filling'].eq(0) &
    bench['has_trading_volume'].fillna(False)
)

# benchmark 基础资格：
# 有 DGTW 标签 + 形成期属于干净 benchmark 基础池 + t-3 权重大于0
bench['benchmark_base_ok'] = (
    bench['dgtw_port'].notna() &
    bench['benchmark_base'].eq(True) &
    bench['weight_t3'].gt(0)
)

# ---------------------------------------------------------
# 1. t日 benchmark 资格
# ---------------------------------------------------------
bench['benchmark_t_eligible'] = (
    bench['benchmark_base_ok'] &
    bench['non_ST'] &
    bench['normal_trade_day'] &
    bench['raw_ret_t'].notna()
)

# ---------------------------------------------------------
# 2. [t,t+1] benchmark 资格
# t 和 t+1 都非ST、都正常交易，且 [t,t+1] 收益非缺失
# ---------------------------------------------------------
bench['bench_t_t1_eligible'] = (
    bench['benchmark_base_ok'] &
    bench['non_ST'] &
    bench.groupby('Stkcd')['non_ST'].shift(-1).fillna(False) &
    bench['normal_trade_day'] &
    bench.groupby('Stkcd')['normal_trade_day'].shift(-1).fillna(False) &
    bench['raw_ret_t_t1'].notna()
)

# ---------------------------------------------------------
# 3. [t,t+10] benchmark 资格
# 11天都非ST；t和t+10正常交易；11天内正常交易天数至少8天；
# [t,t+10]收益非缺失
# ---------------------------------------------------------

# [t,t+10] 11天全部非ST
bench['bench_non_ST_11'] = (
    bench.groupby('Stkcd')['non_ST']
    .transform(lambda x: x.astype(int)
               .rolling(window=11, min_periods=11)
               .sum()
               .shift(-10))
    .eq(11)
)

# [t,t+10] 窗口内正常交易天数至少8天
bench['bench_trade_days_11'] = (
    bench.groupby('Stkcd')['normal_trade_day']
    .transform(lambda x: x.astype(int)
               .rolling(window=11, min_periods=11)
               .sum()
               .shift(-10))
    .ge(8)
)

# t 和 t+10 必须正常交易
bench['bench_endpoint_trade_11'] = (
    bench['normal_trade_day'] &
    bench.groupby('Stkcd')['normal_trade_day'].shift(-10).fillna(False)
)

bench['bench_t_t10_eligible'] = (
    bench['benchmark_base_ok'] &
    bench['bench_non_ST_11'] &
    bench['bench_trade_days_11'] &
    bench['bench_endpoint_trade_11'] &
    bench['raw_ret_t_t10'].notna()
)

# 统一转为 bool
bench['benchmark_t_eligible'] = bench['benchmark_t_eligible'].fillna(False).astype(bool)
bench['bench_t_t1_eligible'] = bench['bench_t_t1_eligible'].fillna(False).astype(bool)
bench['bench_t_t10_eligible'] = bench['bench_t_t10_eligible'].fillna(False).astype(bool)

print("5. benchmark资格筛选完成")
print(f"benchmark_t_eligible: {bench['benchmark_t_eligible'].sum():,}")
print(f"bench_t_t1_eligible:  {bench['bench_t_t1_eligible'].sum():,}")
print(f"bench_t_t10_eligible: {bench['bench_t_t10_eligible'].sum():,}")
print("=" * 80)


################################################################################
############################ 6. 计算 benchmark 收益率 #############################
################################################################################

def make_vw_benchmark(df, ret_col, ok_col, out_col):
    """
    对每个 Trddt × dgtw_port，计算市值加权 benchmark 收益率。
    """
    tmp = df.loc[
        df[ok_col] &
        df['weight_t3'].gt(0) &
        df[ret_col].notna()
    ].copy()

    tmp['wret'] = tmp['weight_t3'] * tmp[ret_col]

    out = (
        tmp
        .groupby(['Trddt', 'dgtw_port'], as_index=False)
        .agg(
            sum_weight_t3=('weight_t3', 'sum'),
            sum_wret=('wret', 'sum'),
            n_stocks=('Stkcd', 'nunique')
        )
    )

    out[out_col] = out['sum_wret'] / out['sum_weight_t3']

    return out[['Trddt', 'dgtw_port', out_col, 'n_stocks']].copy()


# 1. t日 benchmark
bench_t = make_vw_benchmark(
    bench,
    ret_col='raw_ret_t',
    ok_col='benchmark_t_eligible',
    out_col='bench_ret_t'
)

# 2. [t,t+1] benchmark
bench_t_t1 = make_vw_benchmark(
    bench,
    ret_col='raw_ret_t_t1',
    ok_col='bench_t_t1_eligible',
    out_col='bench_ret_t_t1'
)

# 3. [t,t+10] benchmark
bench_t_t10 = make_vw_benchmark(
    bench,
    ret_col='raw_ret_t_t10',
    ok_col='bench_t_t10_eligible',
    out_col='bench_ret_t_t10'
)

# 合并三个 horizon 的 benchmark
benchmark_ret = (
    bench_t
    .rename(columns={'n_stocks': 'n_stocks_t'})
    .merge(
        bench_t_t1.rename(columns={'n_stocks': 'n_stocks_t_t1'}),
        on=['Trddt', 'dgtw_port'],
        how='outer'
    )
    .merge(
        bench_t_t10.rename(columns={'n_stocks': 'n_stocks_t_t10'}),
        on=['Trddt', 'dgtw_port'],
        how='outer'
    )
)

print("6. benchmark收益率计算完成")
print(f"bench_t 行数:       {len(bench_t):,}")
print(f"bench_t_t1 行数:    {len(bench_t_t1):,}")
print(f"bench_t_t10 行数:   {len(bench_t_t10):,}")
print(f"benchmark_ret 行数: {len(benchmark_ret):,}")
print(benchmark_ret.head())
print("=" * 80)


################################################################################
########################### 7. 计算全部个股特征调整收益 ############################
################################################################################

TRD_Dalyr3 = pd.merge(
    TRD_Dalyr3,
    benchmark_ret[
        ['Trddt', 'dgtw_port', 'bench_ret_t', 'bench_ret_t_t1', 'bench_ret_t_t10']
    ],
    on=['Trddt', 'dgtw_port'],
    how='left'
)

TRD_Dalyr3['charadj_ret_t'] = (
    TRD_Dalyr3['raw_ret_t'] - TRD_Dalyr3['bench_ret_t']
)

TRD_Dalyr3['Return_t_t1'] = (
    TRD_Dalyr3['raw_ret_t_t1'] - TRD_Dalyr3['bench_ret_t_t1']
)

TRD_Dalyr3['Return_t_t10'] = (
    TRD_Dalyr3['raw_ret_t_t10'] - TRD_Dalyr3['bench_ret_t_t10']
)

TRD_Dalyr3 = TRD_Dalyr3.sort_values(['Stkcd', 'Trddt']).reset_index(drop=True)

print("7. 全部个股特征调整收益计算完成")
print(TRD_Dalyr3[['Stkcd', 'Trddt', 'dgtw_port', 'Return_t_t1', 'Return_t_t10']].head())
print("=" * 80)


################################################################################
########################### 8. 主回归资格筛选 #####################################
################################################################################

TRD_Dalyr4 = TRD_Dalyr3.copy()
TRD_Dalyr4 = TRD_Dalyr4.sort_values(['Stkcd', 'Trddt']).reset_index(drop=True)

# 正常交易日：非填充日 + 成交量大于0
TRD_Dalyr4['normal_trade_day'] = (
    TRD_Dalyr4['Filling'].eq(0) &
    TRD_Dalyr4['has_trading_volume'].fillna(False)
)

# ---------------------------------------------------------
# [t,t+1] 回归资格
# t 和 t+1 都非ST、都正常交易，且 Return_t_t1 非缺失
# ---------------------------------------------------------
TRD_Dalyr4['reg_t_t1_eligible'] = (
    TRD_Dalyr4['dgtw_port'].notna() &
    TRD_Dalyr4['non_ST'] &
    TRD_Dalyr4.groupby('Stkcd')['non_ST'].shift(-1).fillna(False) &
    TRD_Dalyr4['normal_trade_day'] &
    TRD_Dalyr4.groupby('Stkcd')['normal_trade_day'].shift(-1).fillna(False) &
    TRD_Dalyr4['Return_t_t1'].notna()
)

# ---------------------------------------------------------
# [t,t+10] 回归资格
# 11天都非ST；t和t+10正常交易；11天内正常交易天数至少8天；
# Return_t_t10 非缺失
# ---------------------------------------------------------

TRD_Dalyr4['reg_non_ST_11'] = (
    TRD_Dalyr4.groupby('Stkcd')['non_ST']
    .transform(lambda x: x.astype(int)
               .rolling(window=11, min_periods=11)
               .sum()
               .shift(-10))
    .eq(11)
)

TRD_Dalyr4['reg_trade_days_11'] = (
    TRD_Dalyr4.groupby('Stkcd')['normal_trade_day']
    .transform(lambda x: x.astype(int)
               .rolling(window=11, min_periods=11)
               .sum()
               .shift(-10))
    .ge(8)
)

TRD_Dalyr4['reg_endpoint_trade_11'] = (
    TRD_Dalyr4['normal_trade_day'] &
    TRD_Dalyr4.groupby('Stkcd')['normal_trade_day'].shift(-10).fillna(False)
)

TRD_Dalyr4['reg_t_t10_eligible'] = (
    TRD_Dalyr4['dgtw_port'].notna() &
    TRD_Dalyr4['reg_non_ST_11'] &
    TRD_Dalyr4['reg_trade_days_11'] &
    TRD_Dalyr4['reg_endpoint_trade_11'] &
    TRD_Dalyr4['Return_t_t10'].notna()
)

# 可选：t日特征调整收益可用标记
TRD_Dalyr4['reg_return_t_eligible'] = (
    TRD_Dalyr4['dgtw_port'].notna() &
    TRD_Dalyr4['non_ST'] &
    TRD_Dalyr4['normal_trade_day'] &
    TRD_Dalyr4['charadj_ret_t'].notna()
)

print("8. 主回归资格标记完成")
print(TRD_Dalyr4[
    ['Stkcd', 'Trddt', 'reg_t_t1_eligible', 'reg_t_t10_eligible']
].head())
print("=" * 80)


################################################################################
########################### 9. 最终回归样本 #######################################
################################################################################

# 共同样本：同时能够计算 [t,t+1] 和 [t,t+10] 特征调整收益率
final_reg = TRD_Dalyr4.loc[
    TRD_Dalyr4['reg_t_t1_eligible'] &
    TRD_Dalyr4['reg_t_t10_eligible']
].copy()

print("9. 最终回归样本筛选完成")
print(f"final_daily 行数: {len(TRD_Dalyr4):,}")
print(f"final_reg 行数: {len(final_reg):,}")
print("=" * 80)
################################################################################
########################### 9. 导出最终样本键 #####################################
################################################################################

from pathlib import Path

if EXPORT_SAMPLE_KEY:
    sample_key = final_reg[['Stkcd', 'Trddt']].drop_duplicates().copy()
    sample_key['Stkcd'] = sample_key['Stkcd'].astype(str).str.zfill(6)

    sample_out = Path(r"/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/DGTW_final_sample.csv")
    sample_out.parent.mkdir(parents=True, exist_ok=True)
    sample_key.to_csv(sample_out, index=False, encoding="utf-8-sig")

    print(f"DGTW最终样本键已保存：{sample_out}")
else:
    print("EXPORT_SAMPLE_KEY=False，未导出DGTW最终样本键。")
print("=" * 80)

################################################################################
########################### 10. 对最终回归样本做缩尾 ###############################
################################################################################

print("开始对最终回归样本做缩尾处理......")

def winsorize_series(s, p=0.01):
    lb = s.quantile(p)
    ub = s.quantile(1 - p)
    return s.clip(lower=lb, upper=ub), lb, ub

final_reg['charadj_ret_t'], lb0, ub0 = winsorize_series(final_reg['charadj_ret_t'])
final_reg['Return_t_t1'], lb1, ub1 = winsorize_series(final_reg['Return_t_t1'])
final_reg['Return_t_t10'], lb10, ub10 = winsorize_series(final_reg['Return_t_t10'])

print(f"charadj_ret_t winsor区间: [{lb0:.4f}, {ub0:.4f}]")
print(f"Return_t_t1 winsor区间:  [{lb1:.4f}, {ub1:.4f}]")
print(f"Return_t_t10 winsor区间: [{lb10:.4f}, {ub10:.4f}]")
print("=" * 80)

################################################################################
########################### 11. 后面的描述统计和可视化 #############################
################################################################################

stats_cols = ['Return_t_t1', 'Return_t_t10']
stats_df = final_reg[stats_cols].describe().T
stats_df = stats_df[['mean','25%','50%','75%','std','min','max','count']]
stats_df.columns = ['Mean','p25','p50','p75','SD','Min','Max','N']
stats_df.index = ['Return in % over [t,t+1]', 'Return in % over [t,t+10]']

stats_formatted = stats_df.copy()
stats_formatted['N'] = stats_formatted['N'].astype(int).apply(lambda x: f"{x:,}")
for col in ['Mean','p25','p50','p75','SD','Min','Max']:
    stats_formatted[col] = stats_formatted[col].apply(lambda x: f"{x:.4f}")

print(stats_formatted)
print("=" * 80)

################################################################################
############################ 12. 绘制描述统计表 ##################################
################################################################################

plt.rcParams['font.family'] = ['Times New Roman', 'Arial Unicode MS', 'PingFang SC', 'STSong', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 3))
ax.axis('off')

table_data = stats_formatted.reset_index()
col_labels = [''] + stats_formatted.columns.tolist()
cell_text = table_data.values

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)

col_width_list = [0.2] + [0.1] * 8
for (row, col), cell in table.get_celld().items():
    cell.set_width(col_width_list[col])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

for (row, col), cell in table.get_celld().items():
    cell.set_linewidth(0)

    if row == 0:
        cell.visible_edges = 'TB'
        cell.set_linewidth(1.5)
        cell.set_edgecolor('black')
        cell.set_text_props(weight='bold')
    elif row == len(stats_formatted):
        cell.visible_edges = 'B'
        cell.set_linewidth(1.5)
        cell.set_edgecolor('black')

    if col == 0:
        cell.set_text_props(ha='left')

ax.set_title(
    '特征调整收益率的描述性统计',
    fontsize=14,
    weight='bold',
    pad=15,
    loc='left'
)

plt.tight_layout()
plt.show()
# plt.savefig('descriptive_stats.pdf', dpi=300, bbox_inches='tight')