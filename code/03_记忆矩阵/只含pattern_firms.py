import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import * 
import datetime 
from pathlib import Path
from tqdm.auto import tqdm
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)

#===============================================================================
## 工具函数
#===============================================================================
def normalize_stock_code(x:object) -> str:
    """标准化股票代码，填充为6位数，并转换为字符串"""
    if pd.isna(x):
        return ''
    s=str(x).strip()
    if s=='':
        return ''
    return f"{int(float(s)):06d}"

def xtile_by_group(s,n):
    """
    s：待分组的序列series
    n：分几组
    """
    out = pd.Series(pd.NA,index=s.index,dtype="Int64")
    # 输出与原序列长度对齐的空序列，初始化为缺失值

    non_missing = s.dropna()
    m = len(non_missing)
    # 只对非缺失值分组，缺失值不参与计算。
    # 原生pd.qcut遇到缺失值要么直接报错，要么会把NA强行分到某一组，
    # 完全不符合实证研究的规范：缺失的因子值应该保持缺失，不参与分组，也不影响有效数据的分位数计算。

    if m < n :
        return out
    # 有效样本不足的时候不分组，直接返回全NA的结果，避免后续rank、qcut处理空序列抛出异常。
    
    # 用 rank(method="first") 是为了避免大量重复值导致 qcut 报错
    ranks = non_missing.rank(method="first")
    # 遇到重复值时，按元素出现的顺序分配唯一、连续的整数排名，强制所有排名无重复
    # rank 默认从小到大排名

    out.loc[non_missing.index] = pd.qcut( 
# 只修改out中，索引和non_missing匹配的位置；缺失值的位置完全不动，保持 NA。
        ranks,# 用non_missing.rank(method="first")生成的完全无重复的连续整数
        q=n,# 把序列按分位数均匀分成 n 组
        labels=range(1, n + 1) # 不使用原生区间而是标签
    ).astype("Int64")

    return out
#===============================================================================
## 路径设置
#===============================================================================

ROOT = Path('/Users/qiyuewu/Desktop/mmm复现数据包')

IAR_Rept_path = ROOT/'input/CSMAR/年、中、季报基本情况文件2000-01-01至2025-12-31/IAR_Rept.csv'
pattern_firm_qtr_path = ROOT/'中间结果/02_变量构造结果/PatternFirmQtr/PatternFirmQtr.csv'
AES_AEV_path = ROOT/'中间结果/02_变量构造结果/报刊新闻_AES_AEV/报刊新闻_AES_AEV.csv'
avol_path = ROOT/'中间结果/02_变量构造结果/交易行为OIB、Avol/OIB_CSMAR_prep.csv'
sue_path = ROOT/'中间结果/02_变量构造结果/sue/sue.csv'
TopMover_path = ROOT/'中间结果/02_变量构造结果/TopMover/TopMover.csv'

#===============================================================================
## 读取数据并构造矩阵
#===============================================================================
IAR_Rept = pd.read_csv(
    IAR_Rept_path,
    encoding='utf-8',
    usecols=['Stkcd','Accper','Annodt'],
    parse_dates=['Accper','Annodt']
)

IAR_Rept['Stkcd'] = IAR_Rept['Stkcd'].apply(normalize_stock_code)

pattern_firm_qtr = pd.read_csv(
    pattern_firm_qtr_path,
    encoding='utf-8',
    parse_dates=['Accper']
)

pattern_firm_qtr['Stkcd'] = pattern_firm_qtr['Stkcd'].apply(normalize_stock_code)

IAR_Rept1 = pd.merge(
    IAR_Rept,
    pattern_firm_qtr,
    how='inner',
    on=['Stkcd','Accper']
)
## 筛选出pattern_firms
IAR_Rept2 = IAR_Rept1.loc[IAR_Rept1['PatternFirmQtr']==1]

print("全样本数量：",len(IAR_Rept1))
print("pattern_firm样本数量：",len(IAR_Rept2))
print("pattern_firm样本占比：",len(IAR_Rept2)/len(IAR_Rept1))


# 为了方便后续合并
IAR_Rept2['date'] = IAR_Rept2['Annodt']
IAR_Rept2['Trddt'] = IAR_Rept2['Annodt']

#===============================================================================
## 合并财报发布日当天的特征变量
#===============================================================================

### 合并媒体相关数据
AES_AEV = pd.read_csv(
    AES_AEV_path,
    encoding='utf-8',
    usecols=['Stkcd','date','AEV'],
    parse_dates=['date']
)

AES_AEV['Stkcd'] = AES_AEV['Stkcd'].apply(normalize_stock_code)

AES_AEV['AEV'] = pd.to_numeric(AES_AEV['AEV'])

IAR_Rept3 = pd.merge(
    IAR_Rept2,
    AES_AEV,
    how='left',
    on=['Stkcd','date']
)

nan_count1 = (IAR_Rept3['AEV'].isna()).sum() #3843
nan_ratio1 = (IAR_Rept3['AEV'].isna()).sum()/len(IAR_Rept3) #0.17

IAR_Rept3['AEV_nonmiss'] = IAR_Rept3['AEV'].fillna(0)

# 如果不把缺失改成 0，会出现一个问题：那些没有 RavenPack 新闻的公告日会被 xtile() 
# 排除在分位数计算之外。结果就会变成只在“有新闻记录的公司-公告日”里面划分高低媒体覆盖，
# 而不是在所有公告事件中划分。这会把“没有新闻”的公司错误地从媒体覆盖分布中拿掉。



# 合并异常交易量指标数据
avol = pd.read_csv(
    avol_path,
    encoding='utf-8',
    usecols=['Stkcd','Trddt','td','avol','avolS'],
    parse_dates=['Trddt']
)

avol['Stkcd'] = avol['Stkcd'].apply(normalize_stock_code)
for col in ['td','avol','avolS']:
    avol[col] = pd.to_numeric(avol[col])

nan_count2 = (avol['avol'].isna()).sum() # 8885851 1416105
nan_ratio2 = (avol['avol'].isna()).sum()/len(avol) # 0.629 0.08454
nan_count3 = (avol['avolS'].isna()).sum() # 8893617 7537325
nan_ratio3 = (avol['avolS'].isna()).sum()/len(avol) # 0.62957 0.449972


IAR_Rept3 = pd.merge(
    IAR_Rept3,
    avol,
    how='left',
    on=['Stkcd','Trddt']
)
# 如果盈余公告日这天不是工作日，那么avol很有可能就是缺失值
# 另外avol非缺失的要求很高，要求过去252天全部非缺失（虽然原文也是这样要求的，但是这样的要求似乎过于严格）
# 所以调整了一下，基于完整面板+填充部分数据计算的。


### 合并盈余惊喜
sue = pd.read_csv(
    sue_path,
    encoding='utf-8',
    usecols=['Stkcd','Accper','sue1','sue2','sue3'],
    parse_dates=['Accper']
)

sue['Stkcd'] = sue['Stkcd'].apply(normalize_stock_code)

for col in ['sue1','sue2','sue3']:
    sue[col] = pd.to_numeric(sue[col])

IAR_Rept3 = pd.merge(
    IAR_Rept3,
    sue,
    how='left',
    on=['Stkcd','Accper']
)

for col in ['sue1','sue2','sue3']:
    IAR_Rept3[f'{col}_abs'] = IAR_Rept3[col].abs()
# 只衡量“盈余惊喜有多大”，并不关心它是好消息还是坏消息。

### 合并龙虎榜
TopMover = pd.read_csv(
    TopMover_path,
    encoding='utf-8',
    parse_dates=['Trddt']
)

TopMover['Stkcd'] = TopMover['Stkcd'].apply(normalize_stock_code)

IAR_Rept3 = pd.merge(
    IAR_Rept3,
    TopMover,
    how='left',
    on=['Stkcd','Trddt']
)

nan_count4 = (IAR_Rept3['topMover'].isna()).sum() #22370
nan_ratio4 = (IAR_Rept3['topMover'].isna()).sum()/len(IAR_Rept3) #0.99

IAR_Rept3['topMover'] = IAR_Rept3['topMover'].fillna(0)


#===============================================================================
## 合并财报发布日当天的特征变量，生成一系列用于判断联系当日是否受到强关注的虚拟变量
#===============================================================================
IAR_Rept3['year'] = IAR_Rept3['Annodt'].dt.year

IAR_Rept3["AEV_nonmiss_medGr"] = (
    IAR_Rept3.groupby("year")["AEV_nonmiss"]
      .transform(lambda s: xtile_by_group(s, 2))
)

IAR_Rept3["sue1_abs_dec"] = (
    IAR_Rept3.groupby("year")["sue1_abs"]
      .transform(lambda s: xtile_by_group(s, 10))
)

IAR_Rept3["sue2_abs_dec"] = (
    IAR_Rept3.groupby("year")["sue2_abs"]
      .transform(lambda s: xtile_by_group(s, 10))
)

IAR_Rept3["sue3_abs_dec"] = (
    IAR_Rept3.groupby("year")["sue3_abs"]
      .transform(lambda s: xtile_by_group(s, 10))
)

IAR_Rept3["avol_medGr"] = (
    IAR_Rept3.groupby("year")["avol"]
      .transform(lambda s: xtile_by_group(s, 2))
)

IAR_Rept3["avolS_medGr"] = (
    IAR_Rept3.groupby("year")["avolS"]
      .transform(lambda s: xtile_by_group(s, 2))
)

IAR_Rept3["hi_AEV"] = (IAR_Rept3["AEV_nonmiss_medGr"] == 2).fillna(False).astype(int)
IAR_Rept3["hi_avol"] = (IAR_Rept3["avol_medGr"] == 2).fillna(False).astype(int)
IAR_Rept3["hi_avolS"] = (IAR_Rept3["avolS_medGr"] == 2).fillna(False).astype(int)


IAR_Rept3["hi_sue1_abs"] = (IAR_Rept3["sue1_abs_dec"] == 10).fillna(False).astype(int)
IAR_Rept3["hi_sue2_abs"] = (IAR_Rept3["sue2_abs_dec"] == 10).fillna(False).astype(int)
IAR_Rept3["hi_sue3_abs"] = (IAR_Rept3["sue3_abs_dec"] == 10).fillna(False).astype(int)

IAR_Rept3["sue1_pos"] = ((IAR_Rept3["sue1"] > 0) & IAR_Rept3["sue1"].notna()).astype(int)
IAR_Rept3["sue2_pos"] = ((IAR_Rept3["sue2"] > 0) & IAR_Rept3["sue2"].notna()).astype(int)
IAR_Rept3["sue3_pos"] = ((IAR_Rept3["sue3"] > 0) & IAR_Rept3["sue3"].notna()).astype(int)

IAR_Rept3["sue1_neg"] = ((IAR_Rept3["sue1"] < 0) & IAR_Rept3["sue1"].notna()).astype(int)
IAR_Rept3["sue2_neg"] = ((IAR_Rept3["sue2"] < 0) & IAR_Rept3["sue2"].notna()).astype(int)
IAR_Rept3["sue3_neg"] = ((IAR_Rept3["sue3"] < 0) & IAR_Rept3["sue3"].notna()).astype(int)

IAR_Rept4 = IAR_Rept3[
    [
        "Stkcd",
        "Annodt",
        "Accper",
        "hi_AEV",
        "hi_avol",
        "hi_avolS",
        "hi_sue1_abs","hi_sue2_abs","hi_sue3_abs",
        "sue1_pos","sue2_pos","sue3_pos",
        "sue1_neg","sue2_neg","sue3_neg",
        "topMover",
    ]
].copy()

#===============================================================================
## 生成 同一盈余公告日 的公司两两配对
#===============================================================================

IAR_Rept4_Recall = IAR_Rept4.copy()
rename_cols=[
        "Stkcd",
        "Accper",
        "hi_AEV",
        "hi_avol",
        "hi_avolS",
        "hi_sue1_abs","hi_sue2_abs","hi_sue3_abs",
        "sue1_pos","sue2_pos","sue3_pos",
        "sue1_neg","sue2_neg","sue3_neg",
        "topMover",
    ]

rename_dict = {}
for col in rename_cols:
    rename_dict[col] = col + "Recall"

IAR_Rept4_Recall = IAR_Rept4_Recall .rename(columns=rename_dict)

IAR_Rept4_merge = IAR_Rept4_Recall.merge(
    IAR_Rept4,
    on='Annodt',
    how='inner'
)

IAR_Rept4_merge = IAR_Rept4_merge[
    IAR_Rept4_merge['Stkcd'] != IAR_Rept4_merge['StkcdRecall']
    ].copy()

front_cols = ["Annodt", "Stkcd", "StkcdRecall"]
other_cols = [
    col for col in IAR_Rept4_merge.columns
    if col not in front_cols
]

IAR_Rept4_merge = IAR_Rept4_merge[front_cols + other_cols]

IAR_Rept4_merge = IAR_Rept4_merge.sort_values(by=['Annodt','Stkcd','StkcdRecall'])
# 得到同日公告公司对。StkcdRecall是 可能 被Stkcd Recall的公式


#===============================================================================
## 滚动构造 记忆矩阵 。这里基于原文，但是原文的逻辑有问题
#===============================================================================

# 如果 A 当前季度公告日没有出现在“同日公告公司对”表中，也就是当天没有其他 Pattern firm
# 和 A 同日公告，那么即使 A 公司上季度和其他公司一起公告过，这次公告也不会被识别为 cueing event。

# IAR_Rept4_merge 里的 Stkcd*Accper 不全：
# 原始 IAR_Rept.csv 有 285,703 个唯一 Stkcd*Accper，
# 脚本先筛成 PatternFirmQtr==1 后剩 22,387 个，
# 而进入 IAR_Rept4_merge 的只有 22,037 个，少了 350 个。
# 原因是 IAR_Rept4_merge 按 Annodt 自连接后又剔除了同公司配对，
# 所以只有“同一天还有另一个 pattern firm 公告”的事件才会保留。
# 这 350 个缺失项里，344 个在原始全样本中其实有其他公司同日公告，只是那些公司不是 pattern firm；
# 真正按原始全样本看也完全没有其他公司同日公告的只有 6 个。

memory_list = []

for i in tqdm(
    pd.date_range(
        IAR_Rept4_merge['Annodt'].min(),
        IAR_Rept4_merge['Annodt'].max(),
        freq='D'
    ),
    desc="正在构造 MemoryMatrix",
    unit="day"
):
    temp = IAR_Rept4_merge.copy()
    minAnnodt = i - pd.Timedelta(days=364)
    maxAnnodt = i

    temp = temp[
        (temp['Annodt']>=minAnnodt)&
        (temp['Annodt']<=maxAnnodt)
    ].copy()

    if temp.empty:
        continue

    temp["maxAccper_byStkcd"] = (
        temp.groupby("Stkcd")["Accper"]
            .transform("max")
    ) # 当前滚动窗口内，Stkcd 作为线索公司时，最大的 Accper。

    diff_days = (
        temp["maxAccper_byStkcd"] - temp["Accper"]
    ).dt.days

    temp["dummyQuarterLag0Alt"] = (
        temp["Accper"] == temp["maxAccper_byStkcd"]
    ).astype(int) # 当前这条历史同日公告记录，是不是发生在线索公司当前窗口内最新季度。

    temp["dummyQuarterLag1Alt"] = (
        (diff_days > 0) &
        (diff_days <= 100)
    ).astype(int) # 当前这条历史同日公告记录，是不是发生在线索公司上一季度。

    temp["dummyQuarterLag2Alt"] = (
        (diff_days > 100) &
        (diff_days <= 200)
    ).astype(int)

    temp["dummyQuarterLag3Alt"] = (
        (diff_days > 200) &
        (diff_days <= 300)
    ).astype(int)

    temp["dummyQuarterLag4Alt"] = (
        (diff_days > 300) &
        (diff_days <= 400)
    ).astype(int)

# 对同一个 Stkcd–StkcdRecall 公司对，只要窗口内有任意一次属于 lag 1，那么 quarterLag1Alt = 1。
    temp["quarterLag0Alt"] = (
        temp.groupby(["Stkcd", "StkcdRecall"])["dummyQuarterLag0Alt"]
            .transform("max")
    )

    temp["quarterLag1Alt"] = (
        temp.groupby(["Stkcd", "StkcdRecall"])["dummyQuarterLag1Alt"]
            .transform("max")
    )

    temp["quarterLag2Alt"] = (
        temp.groupby(["Stkcd", "StkcdRecall"])["dummyQuarterLag2Alt"]
            .transform("max")
    )

    temp["quarterLag3Alt"] = (
        temp.groupby(["Stkcd", "StkcdRecall"])["dummyQuarterLag3Alt"]
            .transform("max")
    )

    temp["quarterLag4Alt"] = (
        temp.groupby(["Stkcd", "StkcdRecall"])["dummyQuarterLag4Alt"]
            .transform("max")
    )
# 每个 Stkcd–StkcdRecall 公司对，在当前窗口内只保留最后一条同日公告记录。
    temp = (
        temp.sort_values(["Stkcd", "StkcdRecall", "Annodt"])
            .groupby(["Stkcd", "StkcdRecall"], as_index=False)
            .tail(1)
            .copy()
    )

# 把历史同日公告日替换成线索公司的当前公告日
    temp["maxAnnodt_byStkcd"] = (
        temp.groupby("Stkcd")["Annodt"]
            .transform("max")
    )

    temp["Annodt"] = temp["maxAnnodt_byStkcd"]
# 只保留当前循环日期这一天的结果
    temp = temp[temp["Annodt"] == maxAnnodt].copy()
# 如果这一天没有结果，就跳过：   
    if temp.empty:
        continue
    
    memory_list.append(temp)

MemoryMatrixPattern = pd.concat(memory_list, ignore_index=True)

# 删除中间变量
drop_cols = [
    "maxAccper_byStkcd",
    "maxAnnodt_byStkcd",
    "dummyQuarterLag0Alt",
    "dummyQuarterLag1Alt",
    "dummyQuarterLag2Alt",
    "dummyQuarterLag3Alt",
    "dummyQuarterLag4Alt",
]

MemoryMatrixPattern = MemoryMatrixPattern.drop(
    columns=[col for col in drop_cols if col in MemoryMatrixPattern.columns]
)

# 调整列顺序
front_cols = ["Annodt", "Stkcd", "StkcdRecall","Accper"]

MemoryMatrixPattern = MemoryMatrixPattern[
    front_cols +
    [col for col in MemoryMatrixPattern.columns if col not in front_cols]
]

print()