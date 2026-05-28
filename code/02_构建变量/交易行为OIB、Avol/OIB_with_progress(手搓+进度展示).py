# ①CSMAR里面的主动与否是通过算法判断的
# 然后再将主动的分为大中小单，也就是说后面的小单的买卖性质也是通过前面的算法判断的

# ②CSMAR数据中小单的成交量、成交金额都存在负值，不过量比较小，作为异常值设置为NaN
# （怀疑是因为小单是由总量-大单-中单倒挤得到所导致的）

# ③RESSET和CSMAR的数据还是存在差异的


# =============================================================================
# 说明
# =============================================================================
# 使用小单（金额小于5万元）来近似代替散户
# OI = (buy - sell) / (buy + sell)，即净买入占总交易的比例
# 使用 volume 作为主要代理指标，amount 和 order 作为稳健性检验
#
# 这版的问题：
# ① 主动买入或者卖出是基于算法（详见CSMAR数据库）
# ② 数据并不是基于交易日历，剔除了停牌和暂停上市的数据（86.62%）。
#
# 采用股票自身市场交易日历并结合停牌起止时点进行检验后发现，
# 个股买卖不平衡指标表 缺失交易日中约 86.25% 可由全天停牌、暂停上市或退市解释；
# 若进一步将盘中部分停牌重叠纳入，相关比例达到 86.62%。
# 这表明，个股买卖不平衡指标表并 非完整股票—交易日平衡面板，
# 但其缺口主要来源于停牌，而非数据库随机漏记。

import pandas as pd
import numpy as np
from pathlib import Path
pd.set_option('display.max_columns', None)   # 显示所有列
pd.set_option('display.width', None)         # 自动适应控制台宽度
pd.set_option('display.max_colwidth', None)  # 每列内容不截断


def print_progress(message):
    print(f"[进度] {message}", flush=True)


def print_loop_progress(label, current, total):
    interval = max(1, total // 10)
    if current == 1 or current == total or current % interval == 0:
        print_progress(f"{label}: {current}/{total} ({current / total:.0%})")


ROOT = Path('/Users/qiyuewu/Desktop/mmm复现数据包')
HF_BSImbalance_file = ROOT / '中间结果/00_合并原始数据/CSMAR_个股买卖不平衡指标表_日_合并结果.csv'
td_file = ROOT / '中间结果/01_清洗后样本/TRD_Dalyr_td.csv'

out_file = ROOT / '中间结果/02_变量构造结果/交易行为OIB、Avol/OIB_CSMAR_prep.csv'
fig_file = ROOT / '描述性统计图/OIB_table1.png'

def normalize_stock_code(x:object) -> str:
    """标准化股票代码，填充为6位数，并转换为字符串"""
    if pd.isna(x):
        return ''
    s=str(x).strip()
    if s=='':
        return ''
    return f"{int(float(s)):06d}"

#===============================================================================
#读取数据
#===============================================================================
print_progress("开始读取个股买卖不平衡指标表")
HF_BSImbalance=pd.read_csv(
    HF_BSImbalance_file,
    encoding='utf-8',
    usecols=['Stkcd','Trddt',
             'B_Num','S_Num',#次
             'B_Volume','S_Volume',#股
             'B_Num_S','S_Num_S',#次
             'B_Volume_S','S_Volume_S',#股
             #'B_Amount_S','S_Amount_S',#元(暂时没用上这个变量)
             #'B_Order_S','S_Order_S',
             ]#笔，数据缺失严重（缺失比例30%）
)

HF_BSImbalance['Stkcd']=HF_BSImbalance['Stkcd'].apply(normalize_stock_code)
print_progress(f"个股买卖不平衡指标表读取完成，行数：{len(HF_BSImbalance):,}")

#===============================================================================
# 处理成交量异常值：负成交量设为缺失值
#===============================================================================

print_progress("开始处理成交量异常值")
print("=" * 80)
print("负成交量处理前：")
print("=" * 80)
print("B_Volume_S < 0 数量：", (HF_BSImbalance['B_Volume_S'] < 0).sum())
print("S_Volume_S < 0 数量：", (HF_BSImbalance['S_Volume_S'] < 0).sum())
print("B_Volume_S 缺失数量：", HF_BSImbalance['B_Volume_S'].isna().sum())
print("S_Volume_S 缺失数量：", HF_BSImbalance['S_Volume_S'].isna().sum())


# 将负成交量设为缺失值
HF_BSImbalance.loc[
    HF_BSImbalance['B_Volume_S'] < 0,
    'B_Volume_S'
] = np.nan

HF_BSImbalance.loc[
    HF_BSImbalance['S_Volume_S'] < 0,
    'S_Volume_S'
] = np.nan

print("\n" + "=" * 80)
print("负成交量处理后：")
print("=" * 80)
print("B_Volume_S < 0 数量：", (HF_BSImbalance['B_Volume_S'] < 0).sum())
print("S_Volume_S < 0 数量：", (HF_BSImbalance['S_Volume_S'] < 0).sum())
print("B_Volume_S 缺失数量：", HF_BSImbalance['B_Volume_S'].isna().sum())
print("S_Volume_S 缺失数量：", HF_BSImbalance['S_Volume_S'].isna().sum())
print_progress("成交量异常值处理完成")

#===============================================================================
# 计算订单不平衡
#===============================================================================
print_progress("开始计算订单不平衡指标")
HF_BSImbalance['OIBvol'] = (
    (HF_BSImbalance['B_Volume_S'] - HF_BSImbalance['S_Volume_S'])
    / (HF_BSImbalance['B_Volume_S'] + HF_BSImbalance['S_Volume_S'])
    * 100
)

HF_BSImbalance['OIBnum'] = (
    (HF_BSImbalance['B_Num_S'] - HF_BSImbalance['S_Num_S'])
    / (HF_BSImbalance['B_Num_S'] + HF_BSImbalance['S_Num_S'])
    * 100
)

#===============================================================================
# 简单检验异常值
#===============================================================================

print("=" * 80)
print("OIB 异常值检查")
print("=" * 80)
print("|OIBvol| > 100 数量：", (HF_BSImbalance['OIBvol'].abs() > 100).sum())
print("|OIBnum| > 100 数量：", (HF_BSImbalance['OIBnum'].abs() > 100).sum())
print("OIBvol 为 inf 数量：", np.isinf(HF_BSImbalance['OIBvol']).sum())
print("OIBnum 为 inf 数量：", np.isinf(HF_BSImbalance['OIBnum']).sum())
print("OIBvol 缺失数量：", HF_BSImbalance['OIBvol'].isna().sum())
print("OIBnum 缺失数量：", HF_BSImbalance['OIBnum'].isna().sum())
print_progress("订单不平衡指标计算完成")

#填充至完整的个股交易日序列
print_progress("开始读取个股交易日序列")
TRD_Dalyr_td = pd.read_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/下载数据/CSMAR/个股日度收益率（升级测试）/TRD_Dalyr_td.csv',
    encoding='utf-8'
)

TRD_Dalyr_td['Stkcd'] = TRD_Dalyr_td['Stkcd'].apply(normalize_stock_code)
TRD_Dalyr_td['Trddt'] = pd.to_datetime(TRD_Dalyr_td['Trddt'])
HF_BSImbalance['Trddt'] = pd.to_datetime(HF_BSImbalance['Trddt'])
print_progress(f"个股交易日序列读取完成，行数：{len(TRD_Dalyr_td):,}")

print("整行重复的数量：",TRD_Dalyr_td.duplicated().sum())
print("(Stkcd,Trddt)重复的数量：",TRD_Dalyr_td.duplicated(subset=['Stkcd','Trddt']).sum())
#没有重复值

print_progress("开始合并交易日序列和买卖不平衡数据")
HF_BSImbalance_td=pd.merge(
    TRD_Dalyr_td,
    HF_BSImbalance,
    on=['Stkcd','Trddt'],
    how='left',
    indicator=True
)

print(HF_BSImbalance_td['_merge'].value_counts())
print(HF_BSImbalance_td['_merge'].value_counts(normalize=True))
print_progress(f"合并完成，行数：{len(HF_BSImbalance_td):,}")

#计算未来第i天的OIB指标
print_progress("开始计算未来 OIB 指标")
HF_BSImbalance_td = HF_BSImbalance_td.sort_values(['Stkcd','td']).reset_index(drop=True
                                                                              )
lookup = HF_BSImbalance_td.set_index(['Stkcd','td'])
for i in range(1,51):#左闭右开
    future_index = pd.MultiIndex.from_arrays(
        [
            HF_BSImbalance_td['Stkcd'],
            HF_BSImbalance_td['td']+i
        ],
        names = ['Stkcd','td']
    )
    HF_BSImbalance_td[f'OIBvol_tp{i}']=(
        lookup['OIBvol'] #取值（带双层索引）
        .reindex(future_index) #用刚刚生成的索引去查找数据
        .to_numpy() #.to_numpy() 脱掉所有索引 / 标签，只保留纯数字数组，然后按行精准赋值给原数据表。
    )
    print_loop_progress("未来 OIBvol 指标", i, 50)

for i in range(1,11):
    future_index=pd.MultiIndex.from_arrays(
        [
            HF_BSImbalance_td['Stkcd'],
            HF_BSImbalance_td['td']+i
        ],
        names=['Stkcd','td']
    )
    HF_BSImbalance_td[f'OIBnum_tp{i}']=(
        lookup['OIBnum']
        .reindex(future_index)
        .to_numpy()
    )
    print_loop_progress("未来 OIBnum 指标", i, 10)

print_progress("未来 OIB 指标计算完成")
#计算未来累计的OIBvol
HF_BSImbalance_td['totOIBvol_t_tp1'] = (
    HF_BSImbalance_td['OIBvol']+HF_BSImbalance_td['OIBvol_tp1']
)

print_progress("开始计算累计 OIBvol")
for i in range(2,51):
    prev = i-1

    HF_BSImbalance_td[f'totOIBvol_t_tp{i}']=(
        HF_BSImbalance_td[f'totOIBvol_t_tp{prev}']
        + HF_BSImbalance_td[f'OIBvol_tp{i}']
    )
    print_loop_progress("累计 OIBvol", i - 1, 49)
#------------------------------------------------------------------------------
# 1. 基于成交量的平均 OIB
# 窗口含义：
# OIBvol_avg_t_tp1  = mean(OIBvol_t, OIBvol_t+1)
# OIBvol_avg_t_tp2  = mean(OIBvol_t, OIBvol_t+1, OIBvol_t+2)
# ...
# OIBvol_avg_t_tp50 = mean(OIBvol_t, ..., OIBvol_t+50)
#------------------------------------------------------------------------------
print_progress("开始计算平均 OIBvol")
for i in range(1,51):
    div = i + 1
    HF_BSImbalance_td[f'OIBvol_avg_t_tp{i}']=(
        HF_BSImbalance_td[f'totOIBvol_t_tp{i}']/div
    )
    #删除中间变量
    HF_BSImbalance_td = HF_BSImbalance_td.drop(
        columns=[f'totOIBvol_t_tp{i}',f'OIBvol_tp{i}']
    )
    print_loop_progress("平均 OIBvol", i, 50)
    
#计算未来累计的OIBnum
#------------------------------------------------------------------------------
# 2. 基于成交笔数的平均 OIB
# 窗口含义：
# OIBnum_avg_t_tp1  = mean(OIBnum_t, OIBnum_t+1)
# OIBnum_avg_t_tp2  = mean(OIBnum_t, OIBnum_t+1, OIBnum_t+2)
# ...
# OIBnum_avg_t_tp10 = mean(OIBnum_t, ..., OIBnum_t+10)
#------------------------------------------------------------------------------
HF_BSImbalance_td['totOIBnum_t_tp1'] = (
    HF_BSImbalance_td['OIBnum']+HF_BSImbalance_td['OIBnum_tp1']
)

print_progress("开始计算累计 OIBnum")
for i in range(2,11):
    prev = i-1

    HF_BSImbalance_td[f'totOIBnum_t_tp{i}']=(
        HF_BSImbalance_td[f'totOIBnum_t_tp{prev}']
        + HF_BSImbalance_td[f'OIBnum_tp{i}']
    )
    print_loop_progress("累计 OIBnum", i - 1, 9)

print_progress("开始计算平均 OIBnum")
for i in range(1,11):
    div = i + 1
    HF_BSImbalance_td[f'OIBnum_avg_t_tp{i}']=(
        HF_BSImbalance_td[f'totOIBnum_t_tp{i}']/div
    )
    #删除中间变量
    HF_BSImbalance_td = HF_BSImbalance_td.drop(
        columns=[f'totOIBnum_t_tp{i}',f'OIBnum_tp{i}']
    )
    print_loop_progress("平均 OIBnum", i, 10)

#===============================================================================
# 计算异常散户成交量 abnormal retail volume
# barbervol = barberbvol + barbersvol
# rollingSumbarberVolume = barbervol[_n-1] + ... + barbervol[_n-252]
# barberavol = barbervol / (rollingSumbarberVolume / 252)
#===============================================================================
print_progress("开始计算异常散户成交量")
HF_BSImbalance_td = HF_BSImbalance_td.sort_values(['Stkcd','td']).reset_index(drop=True)

#------------------------------------------------------------------------------
# 1. retail volume = buys + sells
#------------------------------------------------------------------------------
HF_BSImbalance_td['volS'] = (
    HF_BSImbalance_td['B_Volume_S']+HF_BSImbalance_td['S_Volume_S']
)

lookup_vol = HF_BSImbalance_td.set_index(['Stkcd','td'])['volS']
#------------------------------------------------------------------------------
# 2. 计算过去 252 个交易日的散户成交量之和，不包含当天 t
# rollingSumVolume = vol[_td-1] + vol[_td-2] + ... + vol[_td-252]
# 注意：
# min_periods=252 表示必须有完整的过去 252 个有效值；
# 如果过去 252 天中有任何一天缺失，rollingSumVolume 也是 NaN。
#------------------------------------------------------------------------------
past_index = pd.MultiIndex.from_arrays(
    [
        HF_BSImbalance_td['Stkcd'],
        HF_BSImbalance_td['td'] - 1
    ],
    names = ['Stkcd','td']
)

rollingSumVolume = lookup_vol.reindex(past_index).to_numpy()

print_progress("开始计算过去 252 个交易日散户成交量之和")
for i in range(2,253):
    past_index = pd.MultiIndex.from_arrays(
        [
            HF_BSImbalance_td['Stkcd'],
            HF_BSImbalance_td['td'] - i
        ],
        names=['Stkcd','td']
    )

    rollingSumVolume = (
        rollingSumVolume
        +lookup_vol.reindex(past_index).to_numpy()
    )
    print_loop_progress("过去 252 日散户成交量滚动求和", i - 1, 251)

#------------------------------------------------------------------------------
# 3. abnormal retail volume
# avol = 当前散户成交量 / 过去252个交易日平均散户成交量
#------------------------------------------------------------------------------
HF_BSImbalance_td['avolS']=(
    HF_BSImbalance_td['volS'] / (rollingSumVolume / 252)
)
print_progress("异常散户成交量计算完成")
#===============================================================================
# 计算未来的平均异常散户成交量
# avol_tp1 到 avol_tp50
#===============================================================================

lookup_avol = HF_BSImbalance_td.set_index(['Stkcd','td'])['avolS']

print_progress("开始计算未来 avolS 指标")
for i in range(1,51):
    future_index = pd.MultiIndex.from_arrays(
        [
            HF_BSImbalance_td['Stkcd'],
            HF_BSImbalance_td['td'] + i
        ],
        names = ['Stkcd','td']
    )

    HF_BSImbalance_td[f'avolS_tp{i}']=(
        lookup_avol
        .reindex(future_index)
        .to_numpy()
    )
    print_loop_progress("未来 avolS 指标", i, 50)

#------------------------------------------------------------------------------
# 计算窗口 [t, t+i] 的平均 abnormal volume
# avol_avg_t_tp1  = mean(avol_t, avol_t+1)
# avol_avg_t_tp50 = mean(avol_t, ..., avol_t+50)
#
# 注意：
# 逐项相加会让 NaN 继续向后传播；
#------------------------------------------------------------------------------
HF_BSImbalance_td['totavolS_t_tp1']=(
    HF_BSImbalance_td['avolS'] + HF_BSImbalance_td['avolS_tp1']
)

print_progress("开始计算累计 avolS")
for i in range(2,51):
    prev = i-1

    HF_BSImbalance_td[f'totavolS_t_tp{i}']=(
        HF_BSImbalance_td[f'totavolS_t_tp{prev}']
        + HF_BSImbalance_td[f'avolS_tp{i}']
    )
    print_loop_progress("累计 avolS", i - 1, 49)

print_progress("开始计算平均 avolS")
for i in range(1,51):
    div = i+1

    HF_BSImbalance_td[f'avolS_avg_t_tp{i}']=(
        HF_BSImbalance_td[f'totavolS_t_tp{i}'] / div
    )

    HF_BSImbalance_td = HF_BSImbalance_td.drop(
        columns=[f'totavolS_t_tp{i}',f'avolS_tp{i}']
    )
    print_loop_progress("平均 avolS", i, 50)

print("volS < 0 数量：", (HF_BSImbalance_td['volS'] < 0).sum())
print("avolS < 0 数量：", (HF_BSImbalance_td['avolS'] < 0).sum())
print("OIBvol_avg_t_tp1 inf 数量：", np.isinf(HF_BSImbalance_td['OIBvol_avg_t_tp1']).sum())
print("avolS_avg_t_tp1 inf 数量：", np.isinf(HF_BSImbalance_td['avolS_avg_t_tp1']).sum())


#===============================================================================
# 保存前最终整理
#===============================================================================

print_progress("开始保存前最终整理")
# 删除 Stata 中也会删除的 merge 标记
if '_merge' in HF_BSImbalance_td.columns:
    HF_BSImbalance_td = HF_BSImbalance_td.drop(columns=['_merge'])

# 整理内存碎片
HF_BSImbalance_td = HF_BSImbalance_td.copy()

print("最终数据形状：", HF_BSImbalance_td.shape)
print("最终列数：", len(HF_BSImbalance_td.columns))

print_progress("开始输出 CSV 文件")
print("正在输出....")

HF_BSImbalance_td.to_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/OIB_CSMAR_prep.csv',
    index=False,
   encoding='utf-8-sig'
)

print("结果已经保存到中间结果。")
print_progress("CSV 文件输出完成")



#===============================================================================
# 生成类似 Table 1 的 OIB 描述性统计表格图片
#===============================================================================
print_progress("开始生成描述性统计表格图片")
import matplotlib.pyplot as plt

def summary_row(s):
    s = pd.to_numeric(s, errors='coerce')
    s = s.dropna()
    return {
        'Mean': s.mean(),
        'p25': s.quantile(0.25),
        'p50': s.quantile(0.50),
        'p75': s.quantile(0.75),
        'Std.Dev.': s.std(),
        'Min': s.min(),
        'Max': s.max(),
        'N': int(s.shape[0])
    }

# 这里选 Table 1 最核心的两个窗口变量
vars_for_table = {
    'Retail OI in % over [t,t+1]': HF_BSImbalance_td['OIBvol_avg_t_tp1'],
    'Retail OI in % over [t,t+10]': HF_BSImbalance_td['OIBvol_avg_t_tp10'],
}

rows = {}
for name, s in vars_for_table.items():
    rows[name] = summary_row(s)

table1 = pd.DataFrame(rows).T

# 格式化显示
table1_print = table1.copy()
for col in ['Mean', 'p25', 'p50', 'p75', 'Std.Dev.', 'Min', 'Max']:
    table1_print[col] = table1_print[col].map(lambda x: f'{x:,.4f}')
table1_print['N'] = table1_print['N'].map(lambda x: f'{int(x):,}')

print("\nTable 1 风格描述性统计：")
print(table1_print)

#===============================================================================
# 画成表格图片
#===============================================================================

show_df = table1_print.reset_index().rename(columns={'index': 'Variable'})

fig, ax = plt.subplots(figsize=(15, 2.8))
ax.axis('off')

ax.set_title('Descriptive Statistics of Retail Order Imbalance', loc='left', pad=6)

col_widths = [0.28, 0.09, 0.09, 0.09, 0.09, 0.10, 0.09, 0.09, 0.09]

tbl = ax.table(
    cellText=show_df.values,
    colLabels=show_df.columns,
    loc='upper center',
    cellLoc='center',
    colLoc='center',
    colWidths=col_widths
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

# 第一列左对齐
for i in range(len(show_df) + 1):
    cell = tbl[(i, 0)]
    cell.set_text_props(ha='left')

plt.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.08)

table_png_file = '/Users/qiyuewu/Desktop/mmm复现数据包/描述性统计图/OIB_table1.png'
plt.savefig(table_png_file, dpi=300, bbox_inches='tight')
plt.close()

print(f'表格图片已保存：{table_png_file}')
print_progress("描述性统计表格图片生成完成")


for v in ['OIBvol_avg_t_tp1', 'OIBvol_avg_t_tp10']:
    s = pd.to_numeric(HF_BSImbalance_td[v], errors='coerce')

    print("\n变量：", v)
    print("总行数：", len(s))
    print("非缺失数：", s.notna().sum())
    print("缺失数：", s.isna().sum())
    print("等于 100 的数量：", (s == 100).sum())
    print("等于 -100 的数量：", (s == -100).sum())
    print("大于 99 的数量：", (s > 99).sum())
    print("小于 -99 的数量：", (s < -99).sum())

print_progress("全部运行完成")
