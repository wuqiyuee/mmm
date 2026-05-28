import  pandas as pd
import numpy as np 
import datetime as dt 
import matplotlib.pyplot as plt 
from dateutil.relativedelta import * 
from pandas.tseries.offsets import * 
import warnings
warnings.filterwarnings('ignore') 
pd.options.mode.chained_assignment = None 
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_columns', 12)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 1000)


# 读取月度收益文件用于提取市值和计算动量
TRD_Mnth=pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMAR/月个股回报率文件2000-01至2025-12/TRD_Mnth.csv',
    encoding='utf-8'
)
TRD_Mnth=TRD_Mnth.loc[TRD_Mnth['Markettype'].isin([1,4,16,32])]
TRD_Mnth['Trdmnt']=pd.to_datetime(TRD_Mnth['Trdmnt'])
TRD_Mnth=TRD_Mnth.rename(columns={
    'Msmvosd':'sizew',
    'Msmvttl':'size',
    'Mretwd':'ret',
})
TRD_Mnth['size']=TRD_Mnth['size']*1000
TRD_Mnth['sizew']=TRD_Mnth['sizew']*1000
TRD_Mnth=TRD_Mnth[['Stkcd','Trdmnt','Clsdt','size','sizew','Ndaytrd','ret','Markettype']]

#计算动量
TRD_Mnth=TRD_Mnth.sort_values(['Stkcd','Trdmnt'])
TRD_Mnth['is_valid'] = TRD_Mnth['Clsdt']!='DD'
TRD_Mnth['ret_filled'] = TRD_Mnth['ret'].fillna(0)
TRD_Mnth['form_year']=np.where(
    TRD_Mnth['Trdmnt'].dt.month>=7,
    TRD_Mnth['Trdmnt'].dt.year+1,
    TRD_Mnth['Trdmnt'].dt.year
)

def calculate_mom(x):
    valid_months=x['is_valid'].sum()
    cum_ret=np.prod(1+x['ret_filled'])-1
    return pd.Series({'mom':cum_ret,'valid_months':valid_months})

print("正在计算月度复合动量...")
mom_yearly=TRD_Mnth.groupby(['Stkcd','form_year']).apply(calculate_mom).reset_index()
mom_yearly=mom_yearly[(mom_yearly['valid_months']>=6)
                      &(mom_yearly['form_year']!=2000)
                      &(mom_yearly['form_year']!=2026)].copy()




#读取资产负债表，计算账面市值比
comp=pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMR/资产负债表2000-01-01至2025-12-31/FS_Combas.csv',
    encoding='utf-8'
)
comp['Accper']=pd.to_datetime(comp['Accper'])
comp['month']=comp['Accper'].dt.month
comp['year']=comp['Accper'].dt.year
comp=comp.loc[(comp['Typrep']=='A')&
              (comp['month']==12)]
comp=comp.rename(
    columns={
        'A001222000':'递延所得税资产',
        'A002208000':'递延所得税负债',
        'A003112101':'pref',
        'A003000000':'teq',
    }
)

comp['递延所得税资产']=comp['递延所得税资产'].fillna(0)
comp['递延所得税负债']=comp['递延所得税负债'].fillna(0)
comp['pref']=comp['pref'].fillna(0)
comp['teq']=comp['teq'].fillna(0)

comp['be']=comp['teq']-comp['pref']+comp['递延所得税负债']-comp['递延所得税资产']
comp=comp[['Stkcd','Accper','year','be','Typrep']]

TRD_Mnth1=TRD_Mnth[['Stkcd','Trdmnt','size','sizew','Markettype']].copy()
TRD_Mnth1['month']=TRD_Mnth1['Trdmnt'].dt.month
TRD_Mnth1['year']=TRD_Mnth1['Trdmnt'].dt.year

TRD_Mnth2=TRD_Mnth1.loc[TRD_Mnth1['month']==12]
TRD_Mnth2=TRD_Mnth2[['Stkcd','Trdmnt','size','sizew','year','Markettype']]

comp1=comp.merge(
    TRD_Mnth2,
    left_on=['Stkcd','year'],
    right_on=['Stkcd','year'],
    how='inner'
)

comp1['BtM']=comp1['be']/comp1['sizew']
comp1=comp1.loc[comp1['BtM']>0]

##进行行业调整
STK_IndustryClassAnl=pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMR/上市公司行业分类年度表2001-12-31至2025-12-31/STK_IndustryClassAnl.csv',
    encoding='utf-8'
)

STK_IndustryClassAnl=STK_IndustryClassAnl.loc[
    STK_IndustryClassAnl['IndustryClassificationID'].isin(['P0201','P0207','P0221'])]


STK_IndustryClassAnl['EndDate']=pd.to_datetime(STK_IndustryClassAnl['EndDate'])
STK_IndustryClassAnl['year']=STK_IndustryClassAnl['EndDate'].dt.year
STK_IndustryClassAnl=STK_IndustryClassAnl.rename(
    columns={'Symbol':'Stkcd'
             }
)

STK_IndustryClassAnl=STK_IndustryClassAnl[['EndDate','year','Stkcd','IndustryCode2']]

#公司数据匹配行业分类
comp2=comp1.merge(
    STK_IndustryClassAnl,
    left_on=['Stkcd','year'],
    right_on=['Stkcd','year'],
    how='inner'
)

industry_stock_count=(
    comp2.groupby(['year','IndustryCode2'])['Stkcd']
    .nunique()
    .reset_index(name='StockCount')
)


comp3=comp2[['Stkcd','year','BtM','IndustryCode2','be','sizew','size']].merge(
    industry_stock_count,
    on=['year','IndustryCode2'],
    how='left'
)

comp3=comp3.loc[comp3['StockCount']>1].sort_values(['year','IndustryCode2'])

#计算行业均值
adj_BE=comp3.groupby(['year','IndustryCode2'])['be'].sum()
adj_ME=comp3.groupby(['year','IndustryCode2'])['sizew'].sum()

adj=adj_BE/adj_ME
adj=adj.rename('adj').reset_index()

#均值匹配回公司数据
comp4=pd.merge(comp3,adj,on=['year','IndustryCode2'],how='left')

comp4['ln_BtM']=np.log(comp4['BtM'])
comp4['ln_adj']=np.log(comp4['adj'])
comp4['ln_diff']=comp4['ln_BtM']-comp4['ln_adj']

std=comp4.groupby(['year','IndustryCode2'])['ln_diff'].std(ddof=0).reset_index(name='std')
comp4=comp4.merge(
    std,
    on=['year','IndustryCode2'],
    how='left'
)

comp4['BTMadj']=comp4['ln_diff']/comp4['std']

comp4=comp4[['Stkcd','year','BtM','sizew','size','BTMadj']]

#合并三个指标
#comp4包含上年末的账面市值比
#mom_yearly包含动量指标
TRD_Mnth3=TRD_Mnth1.loc[TRD_Mnth1['month']==6]
#TRD_Mnth3包含6月底的市值指标

size_mom=pd.merge(
    TRD_Mnth3,
    mom_yearly[['Stkcd','form_year','mom']],
    left_on=['Stkcd','year'],
    right_on=['Stkcd','form_year'],
    how='inner'
)

comp4['form_year']=comp4['year']+1
size_mom_bm=pd.merge(
    size_mom[['Stkcd','Trdmnt','size','sizew','form_year','mom','Markettype']],
    comp4[['Stkcd','year','form_year','BtM','BTMadj']],
    on=['Stkcd','form_year'],
    how='inner'
)

#开始三分位
#筛选出主板市场进行断点

port=size_mom_bm.copy()

port1=TRD_Mnth3.loc[TRD_Mnth3['Markettype'].isin([1,4,16,32])]#port1中仅包含A股主板市场
port1=port1[['Stkcd','Trdmnt','size','sizew']]

port1=port1.sort_values(['Trdmnt','Stkcd']).drop_duplicates()
size_bk=port1.groupby(['Trdmnt'])['sizew']\
    .describe(percentiles=[.2,.4,.6,.8]).reset_index()
size_bk=size_bk[['Trdmnt','20%','40%','60%','80%']].rename(
    columns={
        '20%':'dec20',
        '40%':'dec40',
        '60%':'dec60',
        '80%':'dec80',
    }
)

port2=pd.merge(port,size_bk,how='left',on='Trdmnt')

def size_group(row):
    if 0<=row['sizew']<row['dec20']:
        value=5
    elif row['sizew']<row['dec40']:
        value=4
    elif row['sizew']<row['dec60']:
        value=3
    elif row['sizew']<row['dec80']:
        value=2
    elif row['sizew']>=row['dec80']:
        value=1
    else:
        value=np.nan
    return value

port2['group']=port2.apply(size_group,axis=1)
port2.drop(columns=['dec20','dec40','dec60','dec80'],inplace=True)#port2全市场




port3=port2.loc[port2['Markettype'].isin([1,4,16,32])]#port3仅包含主板市场
port3=port3.sort_values(['form_year','group','Stkcd']).drop_duplicates()
bm_bk=port3.groupby(['form_year','group'])['BTMadj']\
    .describe(percentiles=[.2,.4,.6,.8]).reset_index()
bm_bk=bm_bk[['form_year','group','20%','40%','60%','80%']].rename(
    columns={
        '20%':'dec20',
        '40%':'dec40',
        '60%':'dec60',
        '80%':'dec80',
    }
)

port4=pd.merge(port2,bm_bk,how='left',on=['form_year','group'])

def bm_group(row):
    if row['BTMadj']<row['dec20']:
        value=5
    elif row['BTMadj']<row['dec40']:
        value=4
    elif row['BTMadj']<row['dec60']:
        value=3
    elif row['BTMadj']<row['dec80']:
        value=2
    elif row['BTMadj']>=row['dec80']:
        value=1
    else:
        value=np.nan
    return value

port4['bmr']=port4.apply(bm_group,axis=1)
port4.drop(columns=['dec20','dec40','dec60','dec80'],inplace=True)


port5=port4.loc[port4['Markettype'].isin([1,4,16,32])].copy()
port5=port5.sort_values(['form_year','group','bmr','Stkcd']).drop_duplicates()
mom_bk=port5.groupby(['form_year','group','bmr'])['mom']\
    .describe(percentiles=[.2,.4,.6,.8]).reset_index()
mom_bk=mom_bk[['form_year','group','bmr','20%','40%','60%','80%']].rename(
    columns={
        '20%':'dec20',
        '40%':'dec40',
        '60%':'dec60',
        '80%':'dec80',
    }
)

port6=pd.merge(port4,mom_bk,how='left',on=['form_year','group','bmr'])

def mom_group(row):
    if row['mom']<row['dec20']:
        value=5
    elif row['mom']<row['dec40']:
        value=4
    elif row['mom']<row['dec60']:
        value=3
    elif row['mom']<row['dec80']:
        value=2
    elif row['mom']>=row['dec80']:
        value=1
    else:
        value=np.nan
    return value

port6['momr']=port6.apply(mom_group,axis=1)
port6.drop(columns=['dec20','dec40','dec60','dec80'],inplace=True)
port6 = port6.dropna(subset=['group', 'bmr', 'momr'])
port6[['group','bmr','momr']]=port6[['group','bmr','momr']].astype(int).astype(str)
port6['dgtw_port']=port6['group']+port6['bmr']+port6['momr']


#将组合匹配回原来的月度收益文件
TRD_Mnth5=TRD_Mnth.drop(columns=['is_valid','ret_filled','form_year'])
TRD_Mnth5['Connect_year'] = np.where(TRD_Mnth5['Trdmnt'].dt.month <= 6,
                                     TRD_Mnth5['Trdmnt'].dt.year - 1,
                                     TRD_Mnth5['Trdmnt'].dt.year)

TRD_Mnth6=pd.merge(
    TRD_Mnth5,
    port6[['Stkcd','form_year','dgtw_port']],
    left_on=['Stkcd','Connect_year'],
    right_on=['Stkcd','form_year'],
    how='left'
)

TRD_Mnth6=TRD_Mnth6.dropna()
TRD_Mnth6=TRD_Mnth6.sort_values(['Trdmnt','dgtw_port'])

def wavg(group,avg_name,weight_name):
    d=group[avg_name]
    w=group[weight_name]
    return (d*w).sum()/w.sum()

print("正在计算每月125个组合的加权平均收益...")
benchmark=TRD_Mnth6.groupby(['Trdmnt','dgtw_port']).apply(
    wavg,'ret','sizew'
).reset_index().rename(columns={0:'benchmark_ret'})


TRD_Mnth7=pd.merge(
    benchmark,
    TRD_Mnth6,
    on=['Trdmnt','dgtw_port'],
    how='left'
)

TRD_Mnth7['xret']=TRD_Mnth7['ret']-TRD_Mnth7['benchmark_ret']
TRD_Mnth7=TRD_Mnth7.drop(columns=['Connect_year','form_year'])

####检验模块
csmar_assign=pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMR/DGTW/DGTW股票分配表/STK_MKT_DGTWASSINGMENTS.csv',
    encoding='utf-8'
)
csmar_bench=pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMR/DGTW/DGTW特征基准指标/STK_MKT_DGTWBENCH.csv',
    encoding='utf-8'
)



#检验组合
my_assign=port6[['Stkcd','form_year','group','bmr','momr','dgtw_port']].copy()
my_assign=my_assign.rename(columns={
    'Stkcd':'Symbol',
    'form_year':'TradingYear',
    'group':'my_MarketValue',
    'bmr':'my_BooktoMarket',
    'momr':'my_Momentum'
})

my_assign[['my_MarketValue','my_BooktoMarket','my_Momentum']]=my_assign[['my_MarketValue','my_BooktoMarket','my_Momentum']].astype(float)
csmar_assign['Symbol']=csmar_assign['Symbol'].astype(int)

csmar_assign=csmar_assign[csmar_assign['IsNotBSE']==1].copy()
merge_assign=pd.merge(
    csmar_assign,
    my_assign,
    on=['Symbol','TradingYear'],
    how='inner'
)

match_mv=(merge_assign['MarketValue']==merge_assign['my_MarketValue']).mean()
match_bm=(merge_assign['BooktoMarket']==merge_assign['my_BooktoMarket']).mean()
match_mom=(merge_assign['Momentum']==merge_assign['my_Momentum']).mean()
match_all=((merge_assign['MarketValue']==merge_assign['my_MarketValue'])&
           (merge_assign['BooktoMarket']==merge_assign['my_BooktoMarket'])&
           (merge_assign['Momentum']==merge_assign['my_Momentum'])).mean()


print("\n【股票分配匹配率（1-5分类）】")
print(f"参与对比的样本量：{len(merge_assign)}条")
print(f"市值组（Size）匹配率：{match_mv:.2%}")
print(f"账面市值比（BtM）匹配率：{match_bm:.2%}")
print(f"动量组（Mom）匹配率：{match_mom:.2%}")
print(f"完全一致（125组合）匹配率：{match_all:.2%}")


#检验基准
my_bench = benchmark.copy()
# 将 Trdmnt 转换为 YYYY-MM 格式与 CSMAR 对齐
my_bench['Trdmnt_str'] = my_bench['Trdmnt'].dt.strftime('%Y-%m')

# 拆解你的 dgtw_port 回 3 个维度 (假设格式为 '111', '125' 等)
my_bench['my_MarketValue'] = my_bench['dgtw_port'].str[0].astype(int)
my_bench['my_BooktoMarket'] = my_bench['dgtw_port'].str[1].astype(int)
my_bench['my_Momentum'] = my_bench['dgtw_port'].str[2].astype(int)

csmar_bench=csmar_bench[csmar_bench['IsNotBSE']==1].copy()

# 合并两份收益率表
merge_bench = pd.merge(
    csmar_bench,
    my_bench,
    left_on=['TradingMonth', 'MarketValue', 'BooktoMarket', 'Momentum'],
    right_on=['Trdmnt_str', 'my_MarketValue', 'my_BooktoMarket', 'my_Momentum'],
    how='inner'
)

# 计算差异与相关性
merge_bench['Ret_Diff'] = merge_bench['BenchmarkReturns'] - merge_bench['benchmark_ret']
mae = merge_bench['Ret_Diff'].abs().mean()
corr = merge_bench['BenchmarkReturns'].corr(merge_bench['benchmark_ret'])

print("\n【基准收益率匹配度】")
print(f"参与比对月度组合数: {len(merge_bench)} 条")
print(f"平均绝对误差 (MAE): {mae:.6f}")
print(f"收益率皮尔逊相关系数: {corr:.4f}")
