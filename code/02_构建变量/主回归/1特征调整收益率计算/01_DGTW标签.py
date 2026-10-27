import  pandas as pd
import numpy as np  
import re
import matplotlib.pyplot as plt 
from dateutil.relativedelta import * 
from pandas.tseries.offsets import * 
from pathlib import Path
import warnings
warnings.filterwarnings('ignore') 
pd.options.mode.chained_assignment = None 
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

# =============================================================================
# 基础设置(路径设置，方便查看读取了哪些文件)
# =============================================================================

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)#不自动转行
pd.set_option('display.width', 1000)#设置很长的虚拟宽度

ROOT = Path(__file__).resolve().parents[4]

td_path = ROOT/ '中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv'  # 对样本进行了清洗并标号
TRD_Dalyr_DGTW_path = ROOT/ '中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv' # 清洗了的用于构建标签的样本
FS_Combas_path = ROOT /'input/CSMAR/资产负债表2000-01-01至2025-12-31/FS_Combas.csv'


portfolio_assign_path = ROOT / '中间结果/02_变量构造结果/DGTW/DGTW形成期组合标签.csv'

portfolio_assign_path.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 工具函数
# =============================================================================

# 标准化股票代码
def normalize_stock_code(x):
    """标准化股票代码为 6 位字符串。"""
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()
    if s == '':
        return pd.NA

    m = re.search(r'\d{1,6}', s)
    if m is None:
        return pd.NA

    return m.group(0).zfill(6)

# =============================================================================
# 处理日度股价的数据，用于提取市值并排名、计算动量
# =============================================================================

# 读取清洗了用于计算DGTW标签的日度股价的数据
TRD_Dalyr = pd.read_csv(
    TRD_Dalyr_DGTW_path,
    encoding='utf-8',
    parse_dates=['Trddt']
)

TRD_Dalyr['Stkcd'] = TRD_Dalyr['Stkcd'].apply(normalize_stock_code)
TRD_Dalyr['Trddt'] = pd.to_datetime(TRD_Dalyr['Trddt'])

# 完整面板过去1个月的交易天数、过去1年的交易天数指标。
# 但是金融行业观测值、上市不满1年的观测值、ST观测值的相关变量是缺失值
TRD_Dalyr = TRD_Dalyr.sort_values(["Stkcd", "td"]).reset_index(drop=True)

TRD_Dalyr['jdate'] = TRD_Dalyr['Trddt'] + MonthEnd(0)
TRD_Dalyr['month'] = TRD_Dalyr['Trddt'].dt.month
TRD_Dalyr['year'] = TRD_Dalyr['Trddt'].dt.year

nan_count = (TRD_Dalyr['ret'].isna()).sum()
# 1907728（封装前的）
# 1907728 (封装后的)

TRD_Dalyr['log_1p_ret'] = np.log1p(TRD_Dalyr['ret'])

TRD_Dalyr['logret_rolling_252']=(
    TRD_Dalyr
    .groupby('Stkcd')['log_1p_ret']
    .rolling(window=252,min_periods=252)
    .sum()
    .reset_index(level=0,drop=True)
)

TRD_Dalyr['logret_rolling_20']=(
    TRD_Dalyr
    .groupby('Stkcd')['log_1p_ret']
    .rolling(window=20,min_periods=20)
    .sum()
    .reset_index(level=0,drop=True)
) #要求过去20天收益不是缺失值（不是金融、新股、ST）

TRD_Dalyr["return_tm252_tm20"] = (
    np.expm1(
        TRD_Dalyr["logret_rolling_252"] -
        TRD_Dalyr["logret_rolling_20"]
    ) * 100
)#要求过去252天收益不是缺失值（不是金融、新股、ST）

TRD_Dalyr["return_tm252_tm20"] = (
    np.expm1(
        TRD_Dalyr["logret_rolling_252"] -
        TRD_Dalyr["logret_rolling_20"]
    ) * 100
)

TRD_Dalyr = TRD_Dalyr.drop(columns=[
    "log_1p_ret",
    "logret_rolling_252",
    "logret_rolling_20",
])

# 即使 TRD_Dalyr_DGTW 是按照市场日历填充的完整面板，它也是按照“每只股票自己的起始 td 到结束 td”填充的。
# 也就是说，如果某只股票在某一年 6 月中旬之后就没有进入面板了，
# 比如退市、样本结束、原始日度数据到 6 月某天为止，那么这只股票那一年 6 月的最后一条观测就可能是
#  6 月 14 日、6 月 20 日，而不是全市场 6 月最后交易日。

june = TRD_Dalyr.loc[TRD_Dalyr['month']==6].copy()
june["last_june_trading_day"] = (
    june.groupby("year")["Trddt"].transform("max")
)# 取整个市场最后一天的交易日
TRD_Dalyr_6 = june.loc[
    june["Trddt"] == june["last_june_trading_day"]
].copy() # 提取6月末的数据（按照市场日期）

TRD_Dalyr_6 = TRD_Dalyr_6.drop(columns=['last_june_trading_day'])
print(TRD_Dalyr_6['Trddt'].value_counts(dropna=False).reset_index())

dup = TRD_Dalyr_6.duplicated(subset=['Stkcd','year']).sum() # 检查每只股票每年6月份1条数据
TRD_Dalyr_6['year_merge'] = TRD_Dalyr_6['year'] - 1 # 计算一个用于合并的年份变量


December = TRD_Dalyr.loc[TRD_Dalyr['month']==12].copy()
December["last_december_trading_day"] = (
    December.groupby("year")["Trddt"].transform("max")
)# 取整个市场最后一天的交易日
TRD_Dalyr_12 = December.loc[
    December["Trddt"] == December["last_december_trading_day"]
].copy() # 提取6月末的数据（按照市场日期）

TRD_Dalyr_12 = TRD_Dalyr_12.drop(columns=['last_december_trading_day'])
print(TRD_Dalyr_12['Trddt'].value_counts(dropna=False).reset_index())

dup = TRD_Dalyr_12.duplicated(subset=['Stkcd','year']).sum() # 检查每只股票每年12月份1条数据

TRD_Dalyr_12 = TRD_Dalyr_12.rename(
    columns={'size':'size_prev',
             'sizew':'sizew_prev',
             'year':'year_merge'}
) # 为了合并进行的重命名

TRD_Dalyr_6 = pd.merge(
    TRD_Dalyr_6,
    TRD_Dalyr_12[['Stkcd','year_merge','size_prev','sizew_prev']],
    how='left',
    on=['Stkcd','year_merge'],
) # 合并6月末数据与上年12月末的市值

dup = TRD_Dalyr_6.duplicated(subset=['Stkcd','year']).sum() # 检验每只股票每年6月份1条数据

#===============================================================================
# 读取资产负债表的数据（季度数据），计算账面价值，合并市值数据后计算bm
#===============================================================================
# 读取资产负债表
FS_Combas = pd.read_csv(
    FS_Combas_path,
    encoding='utf-8',
    usecols=['Stkcd','Accper','Typrep',
             'A001222000','A002208000',
             'A003112000','A003112101','A003112201',
             'A003100000','A003000000'],
    parse_dates=['Accper']
)
comp=FS_Combas.copy()
comp=comp.loc[comp['Typrep']=='A']
comp = comp.drop(columns=['Typrep'])
comp=comp.rename(
    columns={
        'A001222000':'递延所得税资产',
        'A002208000':'递延所得税负债',

        'A003112000':'其他权益工具', 
        'A003112101':'优先股',
        'A003112201':'永续债',
        
        'A003100000':'归属于母公司所有者权益合计',
        'A003000000':'所有者权益合计',
})
comp['Stkcd'] = comp['Stkcd'].apply(normalize_stock_code)
comp['Accper']=pd.to_datetime(comp['Accper'])
comp=comp.loc[comp['Accper'].dt.month==12] # 原表为季度数据，只保留12月的数据
comp['year']=comp['Accper'].dt.year

for col in  ['递延所得税资产','递延所得税负债',
             '其他权益工具','优先股','永续债',
             '归属于母公司所有者权益合计','所有者权益合计']:
    comp[col] = pd.to_numeric(comp[col],errors='coerce')

# 计算 账面价值 = 归属于母公司的所有者权益合计 - 其他权益工具 + 递延所得税负债 - 递延所得税资产


# ①归属于母公司所有者权益合计，如果有的话优先使用，没有的话取所有者权益合计。
na1 = comp['归属于母公司所有者权益合计'].isna().sum()
comp['eq']=np.where(
    comp['归属于母公司所有者权益合计'].isnull(),
    comp['所有者权益合计'],
    comp['归属于母公司所有者权益合计'])
panduan = (comp['eq']!=comp['归属于母公司所有者权益合计']).sum() # 实质上归属于母公司所有者权益合计没有缺失值
na2 = comp['eq'].isna().sum() # 检查一下所有者权益缺失值的情况，没有缺失值


# ②其他权益工具=优先股+永续债，
# 如果其他权益工具科目本身有值就取其他权益工具科目
# 否则取 优先股+永续债
# 如果优先股和永续债也都是缺失值，其他权益工具就会是0
comp['优先股'] = comp['优先股'].fillna(0)
comp['永续债'] = comp['永续债'].fillna(0)
comp['优先股+永续债'] = comp['优先股']+comp['永续债']
comp['其他权益工具'] = comp['其他权益工具'].fillna(comp['优先股+永续债'])

# ③递延所得税资产和递延所得税负债就直接填充缺失值为0
comp['递延所得税负债'] = comp['递延所得税负债'].fillna(0)
comp['递延所得税资产'] = comp['递延所得税资产'].fillna(0)

# 计算普通股的账面价值
comp['be']=comp['eq']-comp['其他权益工具']+comp['递延所得税负债']-comp['递延所得税资产']#计算所有者账面价值
comp = comp.loc[comp['be']>=0]#筛选出账面价值大于0的
jiancha_be_xiaoyu0 = (comp['be']<0).sum()# 没有账面价值小于0的
comp = comp[['Stkcd','Accper','year','be']].copy()


#合并市值和账面市值,计算账面市值比
comp = comp.rename(columns={'year':'year_merge'}
                   )
comp1=pd.merge(
    comp,
    TRD_Dalyr_6,
    how='inner',
    on=['Stkcd','year_merge']
)

comp1=comp1.loc[comp1['size_prev']>0]
comp1['bm']=comp1['be']/comp1['size_prev']

# 此处得到的comp1是已经部分清洗了的A股市场的股票在6月末的市值、动量指标，以及上年末的账面市值比指标
# 还需要进一步清洗,排除过去1年交易天数过少的

comp1 = comp1.loc[
    comp1["trade_days_1y"] >= 120
].copy()#删除过去一年交易天数过少的

comp1 = comp1.dropna(
    subset=["size", "bm", "return_tm252_tm20"]
).copy()

#完成最终清洗，可以用于标签

#===============================================================================
# 1. 使用 comp1 构造 DGTW / characteristic portfolio 标签
# 当前 comp1 层级：Stkcd × formation year
# 每行代表某只股票在某年6月末的排序变量
#===============================================================================

sortvars = comp1.copy()

# 确认关键变量存在
required_cols = ["Stkcd", "year", "size", "bm", "return_tm252_tm20"]
missing = [c for c in required_cols if c not in sortvars.columns]
if missing:
    raise ValueError(f"comp1 缺少必要变量：{missing}")

# 为了稳妥，再删一次缺失
sortvars = sortvars.dropna(
    subset=["Stkcd", "year", "size", "bm", "return_tm252_tm20"]
).copy()

# 只用主板股票断点，可以改成类似：
break_univ = sortvars.loc[sortvars["Markettype"].isin([1,4])].copy()

#===============================================================================
# 2. 每年计算 size / bm / momentum 的 20/40/60/80 分位点
#===============================================================================

quantiles = [0.2, 0.4, 0.6, 0.8]

bp = (
    break_univ
    .groupby("year")[["size", "bm", "return_tm252_tm20"]]
    .quantile(quantiles)
    .unstack()
)

# 展平列名
bp.columns = [
    f"{var}_p{int(q * 100)}"
    for var, q in bp.columns
]

bp = bp.reset_index()

# 合并断点回 sortvars
sortvars = sortvars.merge(
    bp,
    on="year",
    how="left",
    validate="m:1"
)


#===============================================================================
# 3. 根据断点分组
#===============================================================================

def assign_5groups(x, p20, p40, p60, p80):
    return np.select(
        [
            x <= p20,
            (x > p20) & (x <= p40),
            (x > p40) & (x <= p60),
            (x > p60) & (x <= p80),
            x > p80,
        ],
        [1, 2, 3, 4, 5],
        default=np.nan,
    )

sortvars["sizeGroup"] = assign_5groups(
    sortvars["size"],
    sortvars["size_p20"],
    sortvars["size_p40"],
    sortvars["size_p60"],
    sortvars["size_p80"],
)

sortvars["bookToMarketGroup"] = assign_5groups(
    sortvars["bm"],
    sortvars["bm_p20"],
    sortvars["bm_p40"],
    sortvars["bm_p60"],
    sortvars["bm_p80"],
)

sortvars["momentumGroup"] = assign_5groups(
    sortvars["return_tm252_tm20"],
    sortvars["return_tm252_tm20_p20"],
    sortvars["return_tm252_tm20_p40"],
    sortvars["return_tm252_tm20_p60"],
    sortvars["return_tm252_tm20_p80"],
)

# 删除无法分组的
sortvars = sortvars.dropna(
    subset=["sizeGroup", "bookToMarketGroup", "momentumGroup"]
).copy()

sortvars[["sizeGroup", "bookToMarketGroup", "momentumGroup"]] = (
    sortvars[["sizeGroup", "bookToMarketGroup", "momentumGroup"]]
    .astype("int8")
)
#===============================================================================
# 4. 生成稳定的 1-125 portfolioNumber
#===============================================================================

sortvars["portfolioNumber"] = (
    (sortvars["sizeGroup"] - 1) * 25
    + (sortvars["bookToMarketGroup"] - 1) * 5
    + sortvars["momentumGroup"]
).astype("int16")

#===============================================================================
# 5. 生成组合有效期：formation year 的 7月1日 到 次年6月30日
#===============================================================================

sortvars["dateBeg"] = pd.to_datetime(
    sortvars["year"].astype(int).astype(str) + "-07-01"
)

sortvars["dateEnd"] = pd.to_datetime(
    (sortvars["year"].astype(int) + 1).astype(str) + "-06-30"
)

portfolio_assign = sortvars[
    [
        "Stkcd",
        "year",
        "Trddt",
        "dateBeg",
        "dateEnd",
        "Markettype",
        "size",
        "bm",
        "return_tm252_tm20",
        "sizeGroup",
        "bookToMarketGroup",
        "momentumGroup",
        "portfolioNumber",
    ]
].drop_duplicates(["Stkcd", "year"]).copy()

portfolio_assign = portfolio_assign.rename(
    columns={
        "year": "formation_year",
        "Trddt": "formation_date",
    }
)

print("portfolio assignment 行数：", len(portfolio_assign))
print("portfolioNumber 范围：")
print(portfolio_assign["portfolioNumber"].describe())

portfolio_assign.to_csv(
    portfolio_assign_path,
    index=False,
    encoding="utf-8-sig",
)

print(f"DGTW形成期标签已保存：{portfolio_assign_path}")
