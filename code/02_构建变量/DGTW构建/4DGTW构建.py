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

ROOT = Path('/Users/qiyuewu/Desktop/mmm复现数据包')

TRD_Dalyr_path = ROOT/ '中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_2000_2025.csv'
td_path = ROOT/ '中间结果/01_清洗后样本/TRD_Dalyr_td.csv'  # 对样本进行了清洗并标号
TRD_Dalyr_DGTW_path = ROOT/ '中间结果/01_清洗后样本/TRD_Dalyr_DGTW.csv' # 只筛选了A股市场的开市日期并标号
FS_Combas_path = ROOT /'input/CSMAR/资产负债表2000-01-01至2025-12-31/FS_Combas.csv'


out_path = ROOT / '中间结果/02_变量构造结果/DGTW_prep.csv'
fig_path = ROOT / '描述性统计图/return_table1.png'

out_path.parent.mkdir(parents=True, exist_ok=True)
fig_path.parent.mkdir(parents=True, exist_ok=True)

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

# 读取日度股价的数据
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

# 此处得到的comp1是全部A股市场的股票在6月末的市值、动量指标，以及上年末的账面市值比指标
# 还需要进一步清洗,排除过去1年交易天数过少的

comp1 = comp1.loc[
    comp1["trade_days_1y"] >= 120
].copy()

comp1 = comp1.dropna(
    subset=["size", "bm", "return_tm252_tm20"]
).copy()

#完成最终清洗，可以用于标签

################################################################################
########################进行三重排序，构建DGTW组合##################################
###################ƒ#############################################################

# 开始进行三重排序：市值、账面市值比、动量
port1=size_mom_bm.sort_values(['jdate','group','Stkcd']).drop_duplicates()
port2=port1.loc[port1['Markettype'].isin([1,4])]
bm_bk=port2.groupby(['jdate','group'])['bm'].describe(percentiles=[.2,.4,.6,.8])\
.reset_index()

bm_bk=bm_bk[['jdate','group','20%','40%','60%','80%']].rename(
    columns={
    '20%':'dec20',
    '40%':'dec40',
    '60%':'dec60',
    '80%':'dec80'})
port3=pd.merge(port1,bm_bk,how='left',on=['jdate','group'])

def bm_group(row):
    if row['bm']<row['dec20']:
        value=5
    elif row['bm']<row['dec40']:
        value=4
    elif row['bm']<row['dec60']:
        value=3
    elif row['bm']<row['dec80']:
        value=2
    elif row['bm']>=row['dec80']:
        value=1
    else:
        value=np.nan
    return value

port3['bmr']=port3.apply(bm_group,axis=1)
port3.drop(columns=['dec20', 'dec40', 'dec60', 'dec80'], inplace=True)


port3=port3.sort_values(['jdate','group','bmr'])
port4=port3.loc[port3['Markettype'].isin([1,4])].copy()

mom_bk=port4.groupby(['jdate','group','bmr'])['return_tm252_tm20'].\
    describe(percentiles=[.2,.4,.6,.8]).reset_index()

mom_bk=mom_bk[['jdate','group','bmr','20%','40%','60%','80%']].rename(
    columns={
    '20%':'dec20',
    '40%':'dec40',
    '60%':'dec60',
    '80%':'dec80'})
port5=pd.merge(port3,mom_bk,how='left',on=['jdate','group','bmr'])

def mom_group(row):
    if row['return_tm252_tm20']<row['dec20']:
        value=5
    elif row['return_tm252_tm20']<row['dec40']:
        value=4
    elif row['return_tm252_tm20']<row['dec60']:
        value=3
    elif row['return_tm252_tm20']<row['dec80']:
        value=2
    elif row['return_tm252_tm20']>=row['dec80']:
        value=1
    else:
        value=np.nan
    return value
port5['momr']=port5.apply(mom_group,axis=1)
port5.drop(columns=['dec20', 'dec40', 'dec60', 'dec80'], inplace=True)




#1表示最大，5表示最小
port6=port5.copy()
port6 = port6.dropna(subset=['group', 'bmr', 'momr'])
port6[['group','bmr','momr']]=port6[['group','bmr','momr']].astype(int).astype(str)
port6['dgtw_port']=port6['group']+port6['bmr']+port6['momr']
port7=port6[['Stkcd','jdate','sizew','return_tm252_tm20','dgtw_port']]
port7['jyear']=port7['jdate'].dt.year
port7=port7.sort_values(['Stkcd','jdate'])
port7=port7.rename(columns={'jdate':'formdate',})
port7=port7[['Stkcd','formdate','jyear','sizew','dgtw_port']]
print(f"构建的DGTW组合：\n{port7.head()}")

################################################################################
##################################计算基准收益率##################################
################################################################################
mom1=mom.sort_values(['Stkcd','Trddt']).copy()

mom1['month']=mom1['Trddt'].dt.month
mom1['jyear']=np.where(
    mom1['month']<=6,
    mom1['Trddt'].dt.year-1,
    mom1['Trddt'].dt.year
)

mom_daily_clean = mom1[['Stkcd', 'Trddt', 'weight_t3', 'jyear',
                        'ret', 'prcIndex', 'Trdsta',
                        'raw_ret_t', 'raw_ret_t_t1', 'raw_ret_t_t10']].copy()

dgtw_daily = pd.merge(
    mom_daily_clean, 
    port7[['Stkcd', 'jyear', 'dgtw_port']], 
    on=['Stkcd', 'jyear'], 
    how='left'
)

#剔除无法匹配标签或权重缺失的行
dgtw_daily=dgtw_daily.sort_values(['Stkcd','Trddt']).reset_index(drop=True)

dgtw_daily = dgtw_daily.dropna(
    subset=['dgtw_port', 'weight_t3', 'prcIndex',
            'raw_ret_t', 'raw_ret_t_t1', 'raw_ret_t_t10']
).copy()

#再计算组合在同一窗口下的价值加权
for col in ['raw_ret_t','raw_ret_t_t1','raw_ret_t_t10']:
    dgtw_daily[f'w_{col}']=dgtw_daily[col]*dgtw_daily['weight_t3']

bench=(
    dgtw_daily
    .groupby(['Trddt','dgtw_port'],as_index=False)
    .agg(
        sum_weight_t3=('weight_t3','sum'),
        sum_w_raw_ret_t=('w_raw_ret_t','sum'),
        sum_w_raw_ret_t_t1=('w_raw_ret_t_t1','sum'),
        sum_w_raw_ret_t_t10=('w_raw_ret_t_t10','sum'),
    )
    
)

bench['bench_ret_t']=bench['sum_w_raw_ret_t']/bench['sum_weight_t3']
bench['bench_ret_t_t1']=bench['sum_w_raw_ret_t_t1']/bench['sum_weight_t3']
bench['bench_ret_t_t10']=bench['sum_w_raw_ret_t_t10']/bench['sum_weight_t3']
bench=bench[['Trddt','dgtw_port','bench_ret_t','bench_ret_t_t1','bench_ret_t_t10']]


#合并回主表并计算日度超额收益 (Excess Return)
final_daily = pd.merge(
    dgtw_daily, 
    bench, 
    on=['Trddt', 'dgtw_port'], 
    how='left'
)
final_daily['charadj_ret_t'] = final_daily['raw_ret_t'] - final_daily['bench_ret_t']
final_daily['Return_t_t1'] = final_daily['raw_ret_t_t1'] - final_daily['bench_ret_t_t1']
final_daily['Return_t_t10'] = final_daily['raw_ret_t_t10'] - final_daily['bench_ret_t_t10']

final_daily=final_daily.loc[final_daily['Trdsta'].isin([1,4,7,10,13])].copy()

print("日度 DGTW 超额收益计算完成！")
print(final_daily[['Stkcd', 'Trddt', 'dgtw_port', 
                   'Return_t_t1', 'Return_t_t10']].head())


from pathlib import Path

sample_key = final_daily[['Stkcd', 'Trddt']].drop_duplicates().copy()
sample_key['Stkcd'] = sample_key['Stkcd'].astype(str).str.zfill(6)

sample_out = Path(r"/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/DGTW_final_sample1.csv")
sample_out.parent.mkdir(parents=True, exist_ok=True)
sample_key.to_csv(sample_out, index=False, encoding="utf-8-sig")

print(f"DGTW最终样本键已保存：{sample_out}")


#stats_return = final_daily['dgtw_daily_xret'].describe()
#print("\ndgtw_daily_xret的统计：")
#print(stats_return)



#===============================================================================
#========================对特征调整收益率进行缩尾处理================================
#===============================================================================
print("\n"+"="*50)
print("开始进行特证调整收益率的缩尾处理......")
print("="*50)

def winsorize_series(s,p=0.01):
    lb=s.quantile(p)
    ub=s.quantile(1-p)
    return s.clip(lower=lb,upper=ub),lb,ub

final_daily['charadj_ret_t'],lb0,ub0=winsorize_series(final_daily['charadj_ret_t'])
final_daily['Return_t_t1'],lb1,ub1=winsorize_series(final_daily['Return_t_t1'])
final_daily['Return_t_t10'],lb10,ub10=winsorize_series(final_daily['Return_t_t10'])

print(f"charadj_ret_t winsor区间: [{lb0:.4f}, {ub0:.4f}]")
print(f"Return_t_t1 winsor区间:  [{lb1:.4f}, {ub1:.4f}]")
print(f"Return_t_t10 winsor区间: [{lb10:.4f}, {ub10:.4f}]")
print("✅未来[t,t+1]和[t,t+10]累计收益率计算完成！")

###生成描述性统计
stats_cols=['Return_t_t1','Return_t_t10']
stats_df=final_daily[stats_cols].describe().T
#对需要分析的列进行描述行统计并转置
stats_df=stats_df[['mean','25%','50%','75%','std','min','max','count']]
stats_df.columns=['Mean','p25','p50','p75','SD','Min','Max','N']
stats_df.index=['Return in % over [t,t+1]','Return in % over [t,t+10]']

####修改数据的显示格式
stats_formatted=stats_df.copy()
###N显示为带千分位逗号的整数，f"{x:,}"表示给x加千分位符
stats_formatted['N']=stats_formatted['N'].astype(int).apply(lambda x: f"{x:,}")
###其余的变量保留四位有效数字
for col in ['Mean','p25','p50','p75','SD','Min','Max']:
    stats_formatted[col]=stats_formatted[col].apply(lambda x: f"{x:.4f}")



################################################################################
#######################激动人心的绘图时刻到了！#####################################
################################################################################

# 全局字体配置：确保中英文正常显示，英文用Times New Roman，中文兼容多系统
plt.rcParams['font.family'] = ['Times New Roman', 'Arial Unicode MS', 'PingFang SC', 'STSong', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False 

# 创建画布：加宽画布避免列拥挤，微调高度适配行高
fig, ax = plt.subplots(figsize=(16, 3))
ax.axis('off')  # 隐藏坐标轴

# -------------------------- 核心修改：整合行标签到表格数据 --------------------------
# 把行索引（Return那两行）变成表格的第一列，彻底规避col=-1的特殊单元格问题
table_data = stats_formatted.reset_index()
# 列名最前面加一个空字符串，对应行标签列的表头
col_labels = [''] + stats_formatted.columns.tolist()
# 单元格内容为整合后的完整数据
cell_text = table_data.values

# 创建表格（不再需要rowLabels参数，避免特殊列的边界问题）
table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)
col_width_list = [0.2] + [0.1]*8
for (row, col), cell in table.get_celld().items():
    cell.set_width(col_width_list[col])
# 基础样式设置
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)  # 行高翻倍，避免文字拥挤

# -------------------------- 三线表绘制（修复后三条线完全等长） --------------------------
for (row, col), cell in table.get_celld().items():
    # 先清空所有单元格的默认边框，避免样式干扰
    cell.set_linewidth(0)
    
    # 1. 处理表头行：绘制顶线+表头分隔线，同时表头文字加粗
    if row == 0:
        cell.visible_edges = 'TB'  # 显示单元格上下边框
        cell.set_linewidth(1.5)
        cell.set_edgecolor('black')
        cell.set_text_props(weight='bold')  # 表头文字加粗，符合学术规范
    
    # 2. 处理表格最后一行：绘制完整底线
    elif row == len(stats_formatted):  # 数据总行数=stats_formatted的行数
        cell.visible_edges = 'B'  # 显示单元格底部边框
        cell.set_linewidth(1.5)
        cell.set_edgecolor('black')
    
    # 3. 第一列（行标签列）左对齐，符合学术表格惯例
    if col == 0:
        cell.set_text_props(ha='left')

# 添加标题：用面向对象的ax.set_title，比plt.title更稳定，避免布局错乱
ax.set_title(
    '特征调整收益率的描述性统计',
    fontsize=14,
    weight='bold',
    pad=15,  # 标题与表格的间距，避免贴太近
    loc='left'
)

# 自动调整布局，彻底避免标题、表格边缘被截断
plt.tight_layout()

# 显示图片，取消注释下方代码可直接保存高清PDF（论文常用格式）
plt.show()
# plt.savefig('descriptive_stats.pdf', dpi=300, bbox_inches='tight')