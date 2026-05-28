# Rptdt[报告日期] 盈利预测指标公布的时间，即报告的撰写日期。(没有缺失值)
# DeclareDate[发布日期] 是研究报告平台发布日期。（有少量的缺失值）
# 目前决定主要使用发布日期，如果发布日期是缺失值的话就使用报告日期

# 参照中文文献的做法使用股本进行调整

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import * #处理时间金融序列的日期计算
import datetime 
from pathlib import Path
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def normalize_stock_code(x:object) -> str:
    """标准化股票代码，填充为6位数，并转换为字符串"""
    if pd.isna(x):
        return ''
    s=str(x).strip()
    if s=='':
        return ''
    return f"{int(float(s)):06d}"

begdate = '01/01/2000'
enddate = '12/31/2025'

#===============================================================================
# Step 1. 读取预测数据 
#===============================================================================

AF_Forecast = pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv',
    encoding='utf-8',
    usecols=['Stkcd','InstitutionID','AnanmID','Ananm','Feps',
             'Fenddt','Rptdt','DeclareDate']
)
# Rptdt[报告日期] 盈利预测指标公布的时间，即报告的撰写日期。
# DeclareDate[发布日期] 是研究报告平台发布日期。
# 目前决定主要使用发布日期，如果发布日期是缺失值的话就使用报告日期


AF_Forecast['Stkcd'] = AF_Forecast['Stkcd'].apply(normalize_stock_code)
AF_Forecast["InstitutionID"] = (
    AF_Forecast["InstitutionID"]
    .astype("string")
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

AF_Forecast['Fenddt'] = pd.to_datetime(AF_Forecast['Fenddt'])
AF_Forecast = AF_Forecast.rename(
    columns={'Fenddt':'Accper'}
)
AF_Forecast = AF_Forecast.loc[AF_Forecast.Accper<= pd.Timestamp('2025-12-31')]

AF_Forecast['Rptdt'] = pd.to_datetime(AF_Forecast['Rptdt'])
AF_Forecast['DeclareDate'] = pd.to_datetime(AF_Forecast['DeclareDate'])
AF_Forecast['anndats'] = AF_Forecast['DeclareDate'].fillna(AF_Forecast['Rptdt'])
AF_Forecast = AF_Forecast.drop(columns=['Rptdt','DeclareDate'])

# 先按照公司、财报截止日、券商ID、分析师ID、宣布时间排序
# 按照公司、财报截止日、券商ID、分析师ID分组，然后取最后一条记录
AF_Forecast = AF_Forecast.sort_values(
    by=['Stkcd','Accper','InstitutionID','AnanmID','anndats']
)

AF_Forecast_1 = AF_Forecast.groupby(['Stkcd','Accper','InstitutionID','AnanmID'])\
.apply(lambda x: x.index[-1]).to_frame().reset_index()
# 取最后一条记录的索引，把取到的这列索引series转换成表格中的一列（pandas默认命名为0），把groupby的四个字段变成普通列

AF_Forecast_1 = AF_Forecast_1.set_index(0)
# 把名称为”0“的列设置为索引

AF_Forecast = pd.merge(AF_Forecast,AF_Forecast_1['AnanmID'],left_index=True,right_index=True)
# 原表AF_Forecast的索引和AF_Forecast_1的索引进行内连接（how的默认值是‘inner’）
# 只有两边索引都存在的行，才会被保留）
AF_Forecast = AF_Forecast.drop(['AnanmID_y'],axis=1).rename(columns={'AnanmID_x':'AnanmID'})


#===============================================================================
# Step 2. 连接预测数据与真实数据 
#===============================================================================

# 只保留在实际公告日前90天的预测数据

# 读取实际公告的基本情况
IAR_Rept = pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/input/CSMAR/年、中、季报基本情况文件2000-01-01至2025-12-31/IAR_Rept.csv',
    encoding='utf-8',
    usecols=['Stkcd','Reptyp','Accper','Sctcd','Annodt','Erana','Eranb']
)


IAR_Rept['Stkcd'] = IAR_Rept['Stkcd'].apply(normalize_stock_code)
IAR_Rept['Accper'] = pd.to_datetime(IAR_Rept['Accper'])
IAR_Rept = IAR_Rept.loc[IAR_Rept['Sctcd'].isin([1,2])]
IAR_Rept['Annodt'] = pd.to_datetime(IAR_Rept['Annodt'])


IAR_Rept = IAR_Rept.drop(columns=['Sctcd'])
IAR_Rept = IAR_Rept .loc[
    (IAR_Rept ["Annodt"] <= pd.Timestamp("2025-12-31"))
].copy()

AF_Forecast1 = pd.merge(
    IAR_Rept,
    AF_Forecast,
    how='left',
    on=['Stkcd','Accper']
)

AF_Forecast1['dgap'] = AF_Forecast1.Annodt - AF_Forecast1.anndats

AF_Forecast1['flag']= np.where(
    (AF_Forecast1.dgap>=datetime.timedelta(days=0))
    &(AF_Forecast1.dgap<=datetime.timedelta(days=90))
    &(AF_Forecast1.anndats.notna())
    &(AF_Forecast1.Annodt.notna()),1,0)

AF_Forecast1 = AF_Forecast1.loc[AF_Forecast1.flag==1].drop(['flag','dgap'],axis=1)

# 提取所有的事件日期
AF_Forecast1_dt1 = AF_Forecast1[['Stkcd','anndats']].drop_duplicates()#提取预测公告日
AF_Forecast1_dt2 = AF_Forecast1[['Stkcd','Annodt']].drop_duplicates().rename(columns={'Annodt':'anndats'})#提取实际公告日期
# 拼接在一起
AF_Forecast_anndats = pd.concat([AF_Forecast1_dt1,AF_Forecast1_dt2]).drop_duplicates()

# 把不再需要的表格设置为空从而释放内存
AF_Forecast1_dt1 = None
AF_Forecast1_dt2 = None

# 把所有的预测发布日期和实际公告日期调整到最近的交易日，为了提取调整因子（这里暂时参照中文文献的做法使用股本进行调整）
unique_anndats = AF_Forecast_anndats[['anndats']].drop_duplicates()

# 读取市场交易日历数据
TRD_Cale = pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/input/CSMAR/日历文件2000-01-01至2025-12-31/TRD_Cale.csv',
    encoding='utf-8'
)

TRD_Cale['Clddt'] = pd.to_datetime(TRD_Cale['Clddt'])

TRD_Cale = TRD_Cale.loc[
    (TRD_Cale['Markettype']==1)
    &(TRD_Cale['State']=='O')
]

unique_anndats = (
    unique_anndats[['anndats']]
    .dropna()
    .drop_duplicates()
    .sort_values('anndats')
    .reset_index(drop=True)
)

cal = (
    TRD_Cale[['Clddt']]
    .dropna()
    .sort_values('Clddt')
    .reset_index(drop=True)
)

event_trade_date = pd.merge_asof(
    unique_anndats,
    cal,
    left_on='anndats',
    right_on='Clddt',
    direction='backward'
)
# 向前寻找和左表键（预测公告日）最接近的右表键（市场交易日）

event_trade_date['days_shift_back']=(
    event_trade_date['anndats']-event_trade_date['Clddt']
).dt.days

print("\n事件日向前匹配最近交易日结果：")
print("原始唯一 anndats 数量：", unique_anndats["anndats"].nunique())
print("成功匹配数量：", event_trade_date["Clddt"].notna().sum())
print("未匹配数量：", event_trade_date["Clddt"].isna().sum())

print("\n向前调整天数分布：")
print(event_trade_date["days_shift_back"].value_counts(dropna=False).sort_index())

unmatched = event_trade_date.loc[event_trade_date["Clddt"].isna()]
if len(unmatched) > 0:
    print("\n仍未匹配的 anndats：")
    print(unmatched)

unique_anndats = None


# 把预期eps的股本调整到盈余公告日

# 使用cfacshr 调整方法：
# 预测EPS_adj = 预测EPS × cfacshr_公告日 / cfacshr_预测日

# 使用总股本调整方法：
# 预测EPS_adj = 预测EPS × 预测日总股本 / 公告日总股本

# ① 按照 CSMAR 字段 eps=每股收益/普通股总股数，应使用普通股总股数进行调整

# ② 查看 CSMAR 的股本变动文件，样本范围内只有2只股票的优先股取值不为0
# 股票：000509 日期：2008/12/31	股本变动原因：00008（其他）	总股数：250009885 优先股：106641
# （在此之前优先股为0，在此之后不知道哪天优先股变为0且再也没增加）
# 股票：600671 日期：2000/10/9	股本变动原因：30000	总股数：121778628 优先股：18900000
# （从上市首日起就一直有优先股，再次之后不知道哪天优先股变为0且再也没增加）
# 鉴于此，优先股的影响很小，决定使用总股本来进行调整

# ③ 对比了 CSMAR 数据库中的 股票历史日行情信息表(后复权)&股票历史日行情信息表(前复权)，这两张表上的总股数字段是一致的，故决定任意采用一个

TotalShare = pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/00_合并原始数据/CSMAR_股票历史日行情信息表_前复权_2000_2025_合并结果.csv',
    encoding='utf-8',
    usecols=['TradingDate','Symbol','TotalShare']
)

TotalShare['TradingDate'] = pd.to_datetime(TotalShare['TradingDate'])
TotalShare = TotalShare.rename(
    columns={
        'Symbol':'Stkcd',
        'TradingDate':'Clddt'
    }
)

TotalShare['Stkcd'] = TotalShare['Stkcd'].apply(normalize_stock_code)

# 检查一下有没有重复值
duplicate = TotalShare.duplicated(subset=['Stkcd','Clddt']).sum()
# 没有重复值

AF_Forecast_anndats = pd.merge(
    AF_Forecast_anndats,
    event_trade_date,
    how='left',
    on=['anndats']
)

# 按照证券标识和日期匹配得到总股本
AF_Forecast_anndats = pd.merge(
    AF_Forecast_anndats, 
    TotalShare, 
    how='left', 
    on=['Stkcd','Clddt']
)

TRD_Cale = None


#===============================================================================
# Step 3.使用总股本调整估计值
#===============================================================================
AF_Forecast1 = pd.merge(
    AF_Forecast1,
    AF_Forecast_anndats,
    how='inner',
    on=['Stkcd','anndats']
)

AF_Forecast1 = AF_Forecast1.drop(['anndats','Clddt','days_shift_back'],axis=1)\
    .rename(columns={'TotalShare':'TotalShare_ann'})

AF_Forecast1 = pd.merge(
    AF_Forecast1,
    AF_Forecast_anndats,
    how='inner',
    left_on=['Stkcd','Annodt'],
    right_on=['Stkcd','anndats']
)

AF_Forecast1 = AF_Forecast1.drop(['anndats','Clddt','days_shift_back'],axis=1)\
    .rename(columns={'TotalShare':'TotalShare_rep'})

AF_Forecast1['new_Feps'] = (AF_Forecast1['TotalShare_ann']/AF_Forecast1['TotalShare_rep']) * AF_Forecast1['Feps']
# 把预期值的总股数从预测报告期调整为实际公告日

# 检查一个公司、一个财报截止日、一家券商、一个分析师只应该有一个预测数据（检查结果确实是0）
duplicate = AF_Forecast1.duplicated(subset=['Stkcd','Accper','InstitutionID','AnanmID']).sum()

AF_Forecast = None
AF_Forecast_anndats = None

# 计算盈余公告前90天内的分析师预测中位数
medest = AF_Forecast1.groupby(['Stkcd','Accper','Annodt','Erana'])\
    ['new_Feps'].agg(['median','count']).reset_index()


medest = medest.rename(
    columns={
        'median':'medest',
        'count':'numest'
    }
)

AF_Forecast1 = None

#===============================================================================
# Step 4. 合并公司财务数据（左连接，没有预期数据也能计算sue2和sue3）
#===============================================================================
# 经过一番检查发现，A股上市公司基本都满足总资产>=0的要求。

IAR_Rept1 = pd.merge(
    IAR_Rept,
    medest[['Stkcd','Accper','medest','numest']],
    how='left',
    on=['Stkcd','Accper']
)

duplicate = IAR_Rept1.duplicated(subset=['Stkcd','Accper']).sum()
# 检查结果为0

IAR_Rept1 = IAR_Rept1.sort_values(
    by=['Stkcd','Reptyp','Accper']
).drop_duplicates()

medest = None

#===============================================================================
# Step 5. 计算标准化的盈余惊喜
#===============================================================================
sue = IAR_Rept1.sort_values(by=['Accper','Stkcd'])
sue['fyearq'] = sue['Accper'].dt.year
sue = sue.rename(
    columns={
        'Reptyp':'fqtr',
    }
)

# 匹配财报截止日的总股数
TotalShare = TotalShare.sort_values(by=['Clddt','Stkcd'])

sue = pd.merge_asof(
    sue,
    TotalShare,
    by= 'Stkcd', 
    left_on='Accper',
    right_on='Clddt',
    direction='backward'
)

sue['days_shift_back'] = (
    sue['Accper']-sue['Clddt']
).dt.days


print("\n使用财报截止日匹配总股数的情况：")
print("未成功匹配数量：", sue["TotalShare"].isna().sum())
# 有 1753 条未匹配记录，都是因为财报截止日 Accper 早于这只股票在 TotalShare 日行情表里的第一条可用记录。
# 新上市公司：公司上市后，数据库里收录了其上市前或上市初期相关报告期的财报公告，
# 但日行情表里的 TotalShare 只能从该股票开始有行情记录之后才有。

print("\n向前调整天数分布：")
print(sue["days_shift_back"].value_counts(dropna=False).sort_index())
# days_shift_back过大是股票层面的长期停牌、暂停上市、数据断档问题。
# 现在的数据中 days_shift_back∈[5，10]没有任何样本，
# 所以筛选 days_shift_back>5、>7、>10的结果其实一样，都会发现 12 条异常数据。
# 为了不显得过度武断，使用 >10，将days_shift_back>10的TotalShare设置为空。

sue.loc[sue['days_shift_back']>10,'TotalShare'] = np.nan
print("现在缺失总股数的观测值数量：", sue["TotalShare"].isna().sum())

sue = sue.drop(
    columns=['Annodt','Clddt','days_shift_back']
).sort_values(by=['Stkcd','fqtr','fyearq'])


sue['dif_fyearq'] = sue.groupby(['Stkcd','fqtr'])['fyearq'].diff()
sue['lagStkcd'] = sue['Stkcd'].shift(1)
cond_year = sue['dif_fyearq']==1
# 条件：当前纪录与上一条记录是同一个公司，同一个财季的连续两年数据


sue['lagErana']   = np.where(cond_year, sue['Erana'].shift(1), None)
sue['lagEranb']   = np.where(cond_year, sue['Eranb'].shift(1), None)
sue['lagTotalShare']   = np.where(cond_year, sue['TotalShare'].shift(1), None)

# 处理公司的第一条数据
cond_Stkcd = sue['Stkcd'] != sue['lagStkcd'] #第一条数据
sue['lagErana'] = np.where(cond_Stkcd,  None, sue['Erana'].shift(1))
sue['lagEranb'] = np.where(cond_Stkcd, None, sue['Eranb'].shift(1))
sue['lagTotalShare'] = np.where(cond_Stkcd, None, sue['TotalShare'].shift(1))

sue['actual1'] = sue['Erana']# 当期的未调整每股收益
sue['actual2'] = sue['Eranb']# 当期的调整每股收益

sue['expected1'] = sue['lagErana']*sue['lagTotalShare']/sue['TotalShare']
sue['expected2'] = sue['lagEranb']*sue['lagTotalShare']/sue['TotalShare']


# 匹配一下股价
Price = pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_2000_2025.csv',
    encoding='utf-8',
    usecols=['Stkcd','Trddt','Clsprc']
)

Price['Stkcd'] = Price['Stkcd'].apply(normalize_stock_code)
Price['Trddt'] = pd.to_datetime(Price['Trddt'])

duplicate = Price.duplicated(subset=['Stkcd','Trddt']).sum()

sue = sue.sort_values(by=['Accper','Stkcd'])
Price = Price.sort_values(by=['Trddt','Stkcd'])

# 鉴于Price是交易日的数据,使用
sue = pd.merge_asof(
    sue,
    Price,
    by='Stkcd',
    left_on='Accper',
    right_on='Trddt'
)

sue['days_shift_back'] = (
    sue['Accper']-sue['Trddt']
).dt.days

print("未成功匹配财报截止日股价的样本数",(sue['Clsprc'].isna()).sum())
print("匹配季末股价的前移天数",sue['days_shift_back'].value_counts(dropna=False).sort_index())

sue.loc[sue['days_shift_back']>10,'Clsprc'] = np.nan
print("现在缺失股价的观测值数量：", sue["Clsprc"].isna().sum())

sue = sue.drop(
    columns=['days_shift_back','dif_fyearq']
).sort_values(by=['Stkcd','Accper'])

# 鉴于分子已经调整到财报截止日，所以分母的股价取财报截止日的也不需要进行额外的调整
# 计算sue
sue['sue1'] = (sue['actual1']-sue['expected1'])/sue['Clsprc']
sue['sue2'] = (sue['actual2']-sue['expected2'])/sue['Clsprc']
sue['sue3'] = (sue['actual1']-sue['medest'])/sue['Clsprc']

# 差异：
# ①预测期窗口不同。MMM保留公告日前2到45天的预测，这里保留公告日前0到90天。
# ②股价分母不同。MMM使用的是实际盈余公告日前3个交易日的收盘价，这里使用的财报截止日前最近一条收盘价。
# ③MMM原文没有进行股本调整，这里使用总股本进行调整口径到财报截止日。

output_path = Path('/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/02_变量构造结果/sue.csv')
output_path.parent.mkdir(exist_ok=True)
sue.to_csv(
    output_path,
    index = False,
    encoding='utf-8-sig'
)

print(f'已成功保存计算结果到：{output_path}')