# 没有映射到下一个交易日

import pandas as pd
import numpy as np
from dateutil.relativedelta import * 
import warnings
from pathlib import Path
from tqdm import tqdm
from pandas.errors import PerformanceWarning
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)
warnings.simplefilter(action="ignore", category=PerformanceWarning)
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


def dummy_int(s):
    """
    把变量转成 0/1 整数。
    缺失值处理为 0。
    """
    return s.fillna(0).astype(int)
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

output_path = ROOT / "中间结果/03_记忆矩阵/MemoryMatrixPattern1.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
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
IAR_Rept2 = IAR_Rept1.loc[IAR_Rept1['PatternFirmQtr']==1].copy()
# 只要右边是筛选/切片/tail()/head()/条件过滤出来的新表，并且后面还要改它，就加 .copy()。
# 即使左边还是同一个变量名，也建议加。

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

pair_cols = ["Stkcd", "StkcdRecall"]

memory_vars = {
    "hiAEV": ("hi_AEV", "hi_AEVRecall"),
    "hiavol": ("hi_avol", "hi_avolRecall"),
    "hiavolS": ("hi_avolS", "hi_avolSRecall"),

    "hi_sue1_abs": ("hi_sue1_abs", "hi_sue1_absRecall"),
    "hi_sue2_abs": ("hi_sue2_abs", "hi_sue2_absRecall"),
    "hi_sue3_abs": ("hi_sue3_abs", "hi_sue3_absRecall"),

    "sue1pos": ("sue1_pos", "sue1_posRecall"),
    "sue2pos": ("sue2_pos", "sue2_posRecall"),
    "sue3pos": ("sue3_pos", "sue3_posRecall"),

    "sue1neg": ("sue1_neg", "sue1_negRecall"),
    "sue2neg": ("sue2_neg", "sue2_negRecall"),
    "sue3neg": ("sue3_neg", "sue3_negRecall"),

    "topMover": ("topMover", "topMoverRecall"),
}
# 此字典的结构为，"最终变量名后缀": ("线索公司的变量", "被提示公司的变量")

memory_list = []
# 每循环一个公告日 i，都会生成一张当天的临时 memory matrix。
# memory_list 就是把每天的临时表先放进去，最后一次性 pd.concat()。
# 它不是最终表，只是一个容器。

annodt_list = (
    IAR_Rept4_merge["Annodt"]
    .dropna()
    .drop_duplicates()
    .sort_values()
) # 只遍历真实公告日期

for i in tqdm(
    annodt_list,
    desc="正在构造 MemoryMatrix",
    unit="day",
    ncols=80,
    mininterval=1.0,
    leave=True,
    ascii=True
):
    minAnnodt = i - pd.Timedelta(days=364)
    maxAnnodt = i

    # 1. 当前日期 i 的过去一年滚动窗口
    temp = IAR_Rept4_merge[
        (IAR_Rept4_merge["Annodt"] >= minAnnodt) &
        (IAR_Rept4_merge["Annodt"] <= maxAnnodt)
    ].copy()

    if temp.empty:
        continue

    # 2. 每个线索公司在窗口内最新的财报季度
    temp["maxAccper_byStkcd"] = (
        temp.groupby("Stkcd")["Accper"]
            .transform("max")
    )

    temp["days_from_current_accper"] = (
        temp["maxAccper_byStkcd"] - temp["Accper"]
    ).dt.days # 这里用的是财报季度结束日差值，不是公告日差值。

    diff_days = temp["days_from_current_accper"]

    # 3. 逐行判断这条历史共同公告属于 lag0-lag4 哪一类
    temp["dummyQuarterLag0Alt"] = (
        temp["Accper"] == temp["maxAccper_byStkcd"]
    ).astype(int)

    for lag in range(1, 5):
        lower = (lag - 1) * 100
        upper = lag * 100

        temp[f"dummyQuarterLag{lag}Alt"] = (
            (diff_days > lower) & (diff_days <= upper)
        ).astype(int)

    # 4. 公司对层面的 quarterLag（汇总）
    for l in range(0, 5):
        dummy_col = f"dummyQuarterLag{l}Alt"
        final_col = f"quarterLag{l}Alt"

        temp[final_col] = (
            temp.groupby(pair_cols)[dummy_col]
                .transform("max")
                .fillna(0)
                .astype(int)
        )
    # dummyQuarterLag1Alt，逐行，标识这一条记录是不是 lag1
    # quarterLag1Alt公司对，标识这个公司对在窗口内是否出现过 lag1

    # 5. 生成普通强关联 dummyLag 变量
    for l in range(1, 5):
        lag_col = f"dummyQuarterLag{l}Alt"

        for suffix, (cue_var, recall_var) in memory_vars.items():
            dummy_col = f"dummyLag{l}{suffix}"

            temp[dummy_col] = (
                dummy_int(temp[lag_col])
                * dummy_int(temp[cue_var])
                * dummy_int(temp[recall_var])
            )

    # 6. 把普通 dummyLag 汇总成公司对层面的 lag 变量
    for l in range(1, 5):
        for suffix in memory_vars.keys():
            dummy_col = f"dummyLag{l}{suffix}"
            final_col = f"lag{l}{suffix}"

            temp[final_col] = (
                temp.groupby(pair_cols)[dummy_col]
                    .transform("max")
                    .fillna(0)
                    .astype(int)
            )

    # 7. 极端正向 / 极端负向 SUE
    for l in range(1, 5):
        lag_col = f"dummyQuarterLag{l}Alt"

        for k in range(1, 4):
            # 极端正向 SUE
            dummy_pos_col = f"dummyLag{l}hi_sue{k}_pos"
            final_pos_col = f"lag{l}hi_sue{k}_pos"

            temp[dummy_pos_col] = (
                dummy_int(temp[lag_col])
                * dummy_int(temp[f"sue{k}_pos"])
                * dummy_int(temp[f"sue{k}_posRecall"])
                * dummy_int(temp[f"hi_sue{k}_abs"])
                * dummy_int(temp[f"hi_sue{k}_absRecall"])
            )

            temp[final_pos_col] = (
                temp.groupby(pair_cols)[dummy_pos_col]
                    .transform("max")
                    .fillna(0)
                    .astype(int)
            )

            # 极端负向 SUE
            dummy_neg_col = f"dummyLag{l}hi_sue{k}_neg"
            final_neg_col = f"lag{l}hi_sue{k}_neg"

            temp[dummy_neg_col] = (
                dummy_int(temp[lag_col])
                * dummy_int(temp[f"sue{k}_neg"])
                * dummy_int(temp[f"sue{k}_negRecall"])
                * dummy_int(temp[f"hi_sue{k}_abs"])
                * dummy_int(temp[f"hi_sue{k}_absRecall"])
            )

            temp[final_neg_col] = (
                temp.groupby(pair_cols)[dummy_neg_col]
                    .transform("max")
                    .fillna(0)
                    .astype(int)
            )

    # 8. 计算同一天公告的其他公司数量
    temp["nrFirmsAnnouncing"] = (
        temp.groupby(["Stkcd", "Annodt"])["StkcdRecall"]
            .transform("size")
    )

    for l in range(0, 5):
        lag_col = f"dummyQuarterLag{l}Alt"
        int_col = f"nrFirmsAnnLag{l}Alt_int"

        temp[int_col] = np.where(
            temp[lag_col] == 1,
            temp["nrFirmsAnnouncing"],
            np.nan
        )

    for l in range(0, 5):
        int_col = f"nrFirmsAnnLag{l}Alt_int"
        final_col = f"nrFirmsAnnLag{l}Alt"

        temp[final_col] = (
            temp.groupby(pair_cols)[int_col]
                .transform("max")
        )

    # 9. 计算距离历史 overlap 的天数，用于衡量记忆的远近程度。
    for l in range(0, 5):
        lag_col = f"dummyQuarterLag{l}Alt"
        int_col = f"daysSinceOverlapLag{l}Alt_int"

        if l == 0:
            temp[int_col] = np.where(
                temp[lag_col] == 1,
                0,
                np.nan
            )
        else:
            temp[int_col] = np.where(
                temp[lag_col] == 1,
                temp["days_from_current_accper"],
                np.nan
            )

    for l in range(0, 5):
        int_col = f"daysSinceOverlapLag{l}Alt_int"
        final_col = f"daysSinceOverlapLag{l}Alt"

        temp[final_col] = (
            temp.groupby(pair_cols)[int_col]
                .transform("max")
        )

    # 10. 到这里所有窗口内信息已经汇总完成
    # 每个 Stkcd-StkcdRecall 公司对只保留最后一条历史共同公告记录
    temp = (
        temp.sort_values(["Stkcd", "StkcdRecall", "Annodt"])
            .groupby(pair_cols, as_index=False)
            .tail(1)
            .copy()
    )

    # 11. 把历史 Annodt 替换成线索公司当前最新公告日
    temp["maxAnnodt_byStkcd"] = (
        temp.groupby("Stkcd")["Annodt"]
            .transform("max")
    )

    temp["Annodt"] = temp["maxAnnodt_byStkcd"]

    # 12. 只保留当前循环日期 i 的结果
    temp = temp[temp["Annodt"] == maxAnnodt].copy()

    if temp.empty:
        continue

    memory_list.append(temp)


MemoryMatrixPattern = pd.concat(memory_list, ignore_index=True)

# 只保留后续需要的变量
keep_cols = (
    ["Annodt", "Stkcd", "StkcdRecall"]
    + [col for col in MemoryMatrixPattern.columns if col.startswith("quarterLag")]
    + [col for col in MemoryMatrixPattern.columns if col.startswith("lag")]
    + [col for col in MemoryMatrixPattern.columns if col.startswith("nrFirmsAnnLag") and col.endswith("Alt")]
    + [col for col in MemoryMatrixPattern.columns if col.startswith("daysSinceOverlapLag") and col.endswith("Alt")]
)

# 去重，保持顺序
keep_cols = list(dict.fromkeys(keep_cols))

MemoryMatrixPattern = MemoryMatrixPattern[keep_cols].copy()

# 查看列名
for idx, col in enumerate(MemoryMatrixPattern.columns, start=1):
    print(idx, col)
# 只有当前公告日也形成同日公告配对的公司，才被允许成为线索公司。
# 因为代码是从“同日公告公司对表”出发，所以只有当前日期 i 也出现在同日公告公司对表里的公司，
# 才能成为线索公司。如果某家公司当前是单独公告，没有和其他公司同日公告，那么识别不到它作为 cueing firm 的事件。

# 在日期 Annodt = i，线索公司 Stkcd 当前有公告事件；
# StkcdRecall 是它可能提示到的公司；
# quarterLag1Alt、quarterLag2Alt 等变量说明这段记忆关系是在几个季度前由同日公告形成的。

# 最终表里的 Annodt 不是历史同日公告日，而是线索公司当前公告日；
# lag 变量才告诉我们这段历史关系发生在多久以前。



print("正在保存记忆矩阵...")
MemoryMatrixPattern.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("保存成功！")

# 3 个主键变量
# + 5 个 quarterLag 变量
# + 52 个普通 lag 强关联变量
# + 24 个极端正向/负向 SUE 变量
# + 5 个 nrFirmsAnnLag 变量
# + 5 个 daysSinceOverlapLag 变量
# = 94 个变量
