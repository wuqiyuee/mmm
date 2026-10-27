# 核心目标是生成：MemoryMatrix
# 某公司今天公告时，市场可能会“联想到”哪些过去曾经和它同日公告的公司。

import pandas as pd
import numpy as np
from dateutil.relativedelta import * 
from pathlib import Path
from tqdm import tqdm
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

def dummy_int(s):
    """
    把变量转成 0/1 整数。
    缺失值处理为 0。
    """
    return s.fillna(0).astype(int)
#===============================================================================
## 路径设置
#===============================================================================

ROOT = Path(__file__).resolve().parents[3]
IAR_Rept_path = ROOT/'input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv'

output_path = ROOT / "中间结果/03_记忆矩阵/MemoryMatrixFull1.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

if output_path.exists():
    output_path.unlink()
#===============================================================================
## 读取数据并构造矩阵(公司对)
#===============================================================================
IAR_Rept = pd.read_csv(
    IAR_Rept_path,
    encoding='utf-8',
    usecols=['Stkcd','Accper','Annodt'],
    parse_dates=['Accper','Annodt']
)

IAR_Rept['Stkcd'] = IAR_Rept['Stkcd'].apply(normalize_stock_code)


IAR_Rept_Recall = IAR_Rept.copy()

IAR_Rept_Recall = IAR_Rept_Recall.rename(
    columns={
        'Stkcd':'StkcdRecall',
        'Accper':'AccperRecall'
    }
)

# 配对
IAR_Rept_merge = IAR_Rept_Recall .merge(
    IAR_Rept,
    on="Annodt",
    how="inner"
)

# 删除自己和自己的配对
IAR_Rept_merge = IAR_Rept_merge[
    IAR_Rept_merge["Stkcd"] != IAR_Rept_merge["StkcdRecall"]
].copy()


IAR_Rept_merge = IAR_Rept_merge[
    ["Annodt", "Stkcd", "StkcdRecall", "Accper", "AccperRecall"]
].sort_values(
    ["Annodt", "Stkcd", "StkcdRecall"]
)

#===============================================================================
## 滚动构造 记忆矩阵 。这里基于原文，但是原文的逻辑有问题
#===============================================================================
pair_cols = ["Stkcd", "StkcdRecall"]
# 每循环一个公告日 i，都会生成一张当天的临时 memory matrix。
# 不再把结果放进 memory_list，而是每轮直接追加写入 CSV，避免最后 concat 爆内存。

keep_cols = [
        "Stkcd",
        "StkcdRecall",
        "Annodt",
        "quarterLag0Alt",
        "quarterLag1Alt",
        "quarterLag2Alt",
        "quarterLag3Alt",
        "quarterLag4Alt",
        "nrFirmsAnnLag0Alt",
        "nrFirmsAnnLag1Alt",
        "nrFirmsAnnLag2Alt",
        "nrFirmsAnnLag3Alt",
        "nrFirmsAnnLag4Alt",
    ]

wrote_header = False

annodt_list = (
    IAR_Rept_merge["Annodt"]
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
    temp = IAR_Rept_merge[
        (IAR_Rept_merge["Annodt"] >= minAnnodt) &
        (IAR_Rept_merge["Annodt"] <= maxAnnodt)
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

    temp = temp[keep_cols].copy()

    temp = temp.sort_values(
        ["Stkcd", "Annodt", "StkcdRecall"]
    ).reset_index(drop=True)

    temp.to_csv(
        output_path,
        mode="a",
        header=not wrote_header,
        index=False,
        encoding="utf-8-sig" if not wrote_header else "utf-8"
    )

    wrote_header = True

print("保存成功！")