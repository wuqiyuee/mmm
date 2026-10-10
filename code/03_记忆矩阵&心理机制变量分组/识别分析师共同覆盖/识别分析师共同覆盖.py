# 对于每个 cue firm - recall firm 公司对，
# 如果在 cue firm 公告日前的窗口内，
# 存在至少一个分析师同时覆盖 cue firm 和 recall firm，
# 那么把这个公司对标记为 analyst-following overlap。

import pandas as pd
import numpy as np
from dateutil.relativedelta import * # 自然月、自然年
from pathlib import Path
from tqdm import tqdm
import datetime # 支持天、小时、分钟等固定长度时间

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


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


def get_analyst_set(coverage_by_stock, stkcd, annodt, lookback_days):
    """
    coverage_by_stock: 按 Stkcd 分组后的分析师覆盖字典
    stkcd: 公司代码
    annodt: cue firm 公告日
    lookback_days: 向前回看多少天
    """
    coverage_df = coverage_by_stock.get(stkcd) #dict.get(key)
    # 找到匹配的 key：直接返回对应的value—— 也就是该股票的完整分析师覆盖子 DataFrame；

    if coverage_df is None:
        return set()
    # 如果某只股票分析师从来没有预测过，就会返回空集合

    start_date = annodt - datetime.timedelta(days=lookback_days)

    tmp = coverage_df.loc[
        (coverage_df['anndats'] >= start_date)
        & (coverage_df['anndats'] < annodt)
    ]

    return set(tmp['analyst_key']) # 返回分析师集合

#===============================================================================
## 路径设置
#===============================================================================

ROOT = Path('/Users/qiyuewu/Desktop/mmm复现数据包')
AF_Forecast_path = ROOT / '中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv'

output_root = ROOT / "中间结果/03_记忆矩阵/分析师共同覆盖"
output_split = output_root / "拆开AnanmID"
output_team = output_root / "不拆开AnanmID"
output_compare = output_root / "对比汇总"

output_split.mkdir(parents=True, exist_ok=True)
output_team.mkdir(parents=True, exist_ok=True)
output_compare.mkdir(parents=True, exist_ok=True)


#===============================================================================
## 读取分析师预测文件
#===============================================================================

AF_Forecast = pd.read_csv(
    AF_Forecast_path,
    encoding='utf-8',
    usecols=['Stkcd', 'InstitutionID', 'AnanmID', 'Ananm',
             'Fenddt', 'Rptdt', 'DeclareDate']
)

# Rptdt[报告日期] 盈利预测指标公布的时间，即报告的撰写日期。
# DeclareDate[发布日期] 是研究报告平台发布日期。
# 目前决定主要使用发布日期，如果发布日期是缺失值的话就使用报告日期。

AF_Forecast['Stkcd'] = AF_Forecast['Stkcd'].apply(normalize_stock_code)

AF_Forecast["InstitutionID"] = (
    AF_Forecast["InstitutionID"]
    .astype("string")
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)  # 删除末尾的 .0
)

AF_Forecast['Fenddt'] = pd.to_datetime(AF_Forecast['Fenddt'])
AF_Forecast = AF_Forecast.rename(
    columns={'Fenddt': 'Accper'}
)

AF_Forecast = AF_Forecast.loc[
    AF_Forecast.Accper <= pd.Timestamp('2025-12-31')
]

AF_Forecast['Rptdt'] = pd.to_datetime(AF_Forecast['Rptdt'])
AF_Forecast['DeclareDate'] = pd.to_datetime(AF_Forecast['DeclareDate'])
AF_Forecast['anndats'] = AF_Forecast['DeclareDate'].fillna(AF_Forecast['Rptdt'])

AF_Forecast = AF_Forecast.drop(columns=['Rptdt', 'DeclareDate'])

#===============================================================================
# 构造分析师覆盖表：不拆开 AnanmID
#===============================================================================

AF_Team = AF_Forecast[
    ['Stkcd', 'InstitutionID', 'AnanmID', 'Ananm', 'anndats']
].copy()

AF_Team['analyst_key'] = (
    AF_Team['AnanmID']
    .astype('string')
    .str.strip()
    .str.replace(r'\s+', '', regex=True)
)

AF_Team = AF_Team.loc[
    (AF_Team['Stkcd'] != '')
    & (AF_Team['analyst_key'].notna())
    & (AF_Team['analyst_key'] != '')
    & (AF_Team['anndats'].notna())
].copy()

AF_Team = AF_Team.drop_duplicates(
    ['Stkcd', 'analyst_key', 'anndats']
)

AF_Team['id_mode'] = 'team'

#===============================================================================
# 构造分析师覆盖表：拆开 AnanmID
#===============================================================================

AF_Split = AF_Forecast[
    ['Stkcd', 'InstitutionID', 'AnanmID', 'Ananm', 'anndats']
].copy()

AF_Split['analyst_key'] = (
    AF_Split['AnanmID']
    .astype('string')
    .str.strip()
    .str.replace(r'\s+', '', regex=True)
    .str.split(',')
)

AF_Split = AF_Split.explode('analyst_key')

AF_Split['analyst_key'] = (
    AF_Split['analyst_key']
    .astype('string')
    .str.strip()
)

AF_Split = AF_Split.loc[
    (AF_Split['Stkcd'] != '')
    & (AF_Split['analyst_key'].notna())
    & (AF_Split['analyst_key'] != '')
    & (AF_Split['anndats'].notna())
].copy()

AF_Split = AF_Split.drop_duplicates(
    ['Stkcd', 'analyst_key', 'anndats']
)

AF_Split['id_mode'] = 'split'


#===============================================================================
# 最小加速：按股票代码提前分组
#===============================================================================

AF_Team_by_stock = dict(tuple(AF_Team.groupby('Stkcd'))) 
# 解决groupby迭代器消耗的问题
# 返回的是{'Stkcd':DataFrame}
#{
#    "具体的股票代码1": 该股票对应的完整分析师覆盖子DataFrame,
#    "具体的股票代码2": 该股票对应的完整分析师覆盖子DataFrame,
#    "具体的股票代码3": 该股票对应的完整分析师覆盖子DataFrame,
#    ...
#}
AF_Split_by_stock = dict(tuple(AF_Split.groupby('Stkcd')))


#===============================================================================
# 读取 MemoryMatrix
#===============================================================================

MemoryMatrix = pd.read_csv(
    ROOT / "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv",
    encoding='utf-8',
    usecols=['Annodt', 'Stkcd', 'StkcdRecall'],
    parse_dates=['Annodt']
)

MemoryMatrix['Stkcd'] = MemoryMatrix['Stkcd'].apply(normalize_stock_code)
MemoryMatrix['StkcdRecall'] = MemoryMatrix['StkcdRecall'].apply(normalize_stock_code)
MemoryMatrix['Annodt'] = pd.to_datetime(MemoryMatrix['Annodt'])

MemoryMatrix = MemoryMatrix.drop_duplicates(
    ['Annodt', 'Stkcd', 'StkcdRecall']
)

#===============================================================================
# 识别共同分析师覆盖
#===============================================================================

def identify_common_analyst_overlap(memory_df, coverage_by_stock, lookback_days, id_mode):
    """
    对 MemoryMatrix 中每个 cue-recall 公司对，
    判断公告日前 lookback_days 天内是否存在共同分析师覆盖。
    """
    result_list = []

    for _, row in tqdm(memory_df.iterrows(), total=len(memory_df)):
    # memory_df.iterrows()：遍历 DataFrame 的每一行，返回(行索引, 行内容)的二元组；
    # _：用下划线接收行索引，代表我们不需要行索引，直接丢弃，是 Python 的通用简洁写法；
    # total=len(memory_df)：手动指定总样本量，让 tqdm 的进度条和剩余时间计算 100% 准确，避免进度条显示不全。
        annodt = row['Annodt']
        stkcd = row['Stkcd']
        stkcd_recall = row['StkcdRecall']

    # 分别获取两家公司的窗口期分析师集合
        cue_analysts = get_analyst_set(
            coverage_by_stock,
            stkcd,
            annodt,
            lookback_days
        )

        recall_analysts = get_analyst_set(
            coverage_by_stock,
            stkcd_recall,
            annodt,
            lookback_days
        )

        # 计算共同分析师
        common_analysts = cue_analysts & recall_analysts

        # 只保存「存在共同分析师」的有效配对（追加到结果列表中）
        if len(common_analysts) > 0:
            result_list.append({
                'Annodt': annodt, # cue公司的公告日期
                'Stkcd': stkcd, # cue公司
                'StkcdRecall': stkcd_recall, # 被cue的公司
                'same_analyst_overlap': 1, # 分析师共同覆盖标记
                'n_common_analysts': len(common_analysts), # 共同分析师的数量
                'lookback_days': lookback_days, # 记录向前看的窗口期
                'id_mode': id_mode # 记录是否拆分了分析师id
            })
    return pd.DataFrame(result_list) # 将结果列表拼接成表格


#===============================================================================
# 分别运行拆开和不拆开的共同覆盖识别
#===============================================================================

lookback_list = [45, 365]

summary_list = []

for lookback_days in lookback_list:

    print(f"正在识别拆开 AnanmID 口径，窗口 = {lookback_days} 天")
    Overlap_Split = identify_common_analyst_overlap(
        MemoryMatrix,
        AF_Split_by_stock,
        lookback_days,
        id_mode='split'
    )

    print(f"正在识别不拆开 AnanmID 口径，窗口 = {lookback_days} 天")
    Overlap_Team = identify_common_analyst_overlap(
        MemoryMatrix,
        AF_Team_by_stock,
        lookback_days,
        id_mode='team'
    )

    Overlap_Split.to_csv(
        output_split / f"Pattern样本_分析师共同覆盖_滚动{lookback_days}天.csv",
        index=False,
        encoding='utf-8-sig'
    )

    Overlap_Team.to_csv(
        output_team / f"Pattern样本_分析师共同覆盖_滚动{lookback_days}天.csv",
        index=False,
        encoding='utf-8-sig'
    )

    split_keys = set(
        zip(
            Overlap_Split['Annodt'],
            Overlap_Split['Stkcd'],
            Overlap_Split['StkcdRecall']
        )
    )

    team_keys = set(
        zip(
            Overlap_Team['Annodt'],
            Overlap_Team['Stkcd'],
            Overlap_Team['StkcdRecall']
        )
    )

    both_keys = split_keys & team_keys
    split_only_keys = split_keys - team_keys
    team_only_keys = team_keys - split_keys

    summary_list.append({
        'lookback_days': lookback_days,
        'matrix_rows_checked': len(MemoryMatrix),
        'split_overlap_rows': len(split_keys),
        'team_overlap_rows': len(team_keys),
        'both_overlap_rows': len(both_keys),
        'split_only_rows': len(split_only_keys),
        'team_only_rows': len(team_only_keys),
        'split_only_share_among_split': (
            len(split_only_keys) / len(split_keys)
            if len(split_keys) > 0 else np.nan
        ),
        'team_only_share_among_team': (
            len(team_only_keys) / len(team_keys)
            if len(team_keys) > 0 else np.nan
        )
    })


#===============================================================================
# 保存拆开与不拆开的对比汇总
#===============================================================================

Summary = pd.DataFrame(summary_list)

Summary.to_csv(
    output_compare / "Pattern样本_分析师共同覆盖_拆开与不拆开AnanmID对比汇总.csv",
    index=False,
    encoding='utf-8-sig'
)

print(Summary)

#===============================================================================
# 分块读取 MemoryMatrixFull，并识别共同分析师覆盖
#===============================================================================

MemoryMatrixFull_path = ROOT / "中间结果/03_记忆矩阵/MemoryMatrixFull(封装优化版).csv"

lookback_list = [45, 365]
chunksize = 500000

summary_list_full = []

for lookback_days in lookback_list:

    split_file = output_split / f"Full样本_分析师共同覆盖_滚动{lookback_days}天.csv"
    team_file = output_team / f"Full样本_分析师共同覆盖_滚动{lookback_days}天.csv"

    split_keys = set()
    team_keys = set()
    matrix_rows_checked = 0

    for chunk_no, MemoryMatrixFull in enumerate(
        pd.read_csv(
            MemoryMatrixFull_path,
            encoding='utf-8',
            usecols=['Annodt', 'Stkcd', 'StkcdRecall'],
            chunksize=chunksize
        ),
        start=1
    ):
        print(f"Full样本，窗口 = {lookback_days} 天，正在处理第 {chunk_no} 个分块")

        MemoryMatrixFull['Stkcd'] = MemoryMatrixFull['Stkcd'].apply(normalize_stock_code)
        MemoryMatrixFull['StkcdRecall'] = MemoryMatrixFull['StkcdRecall'].apply(normalize_stock_code)
        MemoryMatrixFull['Annodt'] = pd.to_datetime(MemoryMatrixFull['Annodt'])

        MemoryMatrixFull = MemoryMatrixFull.drop_duplicates(
            ['Annodt', 'Stkcd', 'StkcdRecall']
        )

        matrix_rows_checked += len(MemoryMatrixFull)

        Overlap_Split_Full = identify_common_analyst_overlap(
            MemoryMatrixFull,
            AF_Split_by_stock,
            lookback_days,
            id_mode='split'
        )

        Overlap_Team_Full = identify_common_analyst_overlap(
            MemoryMatrixFull,
            AF_Team_by_stock,
            lookback_days,
            id_mode='team'
        )

        if not Overlap_Split_Full.empty:
            split_keys.update(
                zip(
                    Overlap_Split_Full['Annodt'],
                    Overlap_Split_Full['Stkcd'],
                    Overlap_Split_Full['StkcdRecall']
                )
            )

        if not Overlap_Team_Full.empty:
            team_keys.update(
                zip(
                    Overlap_Team_Full['Annodt'],
                    Overlap_Team_Full['Stkcd'],
                    Overlap_Team_Full['StkcdRecall']
                )
            )

        Overlap_Split_Full.to_csv(
            split_file,
            mode='w' if chunk_no == 1 else 'a',
            header=(chunk_no == 1),
            index=False,
            encoding='utf-8-sig'
        )

        Overlap_Team_Full.to_csv(
            team_file,
            mode='w' if chunk_no == 1 else 'a',
            header=(chunk_no == 1),
            index=False,
            encoding='utf-8-sig'
        )

    both_keys = split_keys & team_keys
    split_only_keys = split_keys - team_keys
    team_only_keys = team_keys - split_keys

    summary_list_full.append({
        'lookback_days': lookback_days,
        'matrix_rows_checked': matrix_rows_checked,
        'split_overlap_rows': len(split_keys),
        'team_overlap_rows': len(team_keys),
        'both_overlap_rows': len(both_keys),
        'split_only_rows': len(split_only_keys),
        'team_only_rows': len(team_only_keys),
        'split_only_share_among_split': (
            len(split_only_keys) / len(split_keys)
            if len(split_keys) > 0 else np.nan
        ),
        'team_only_share_among_team': (
            len(team_only_keys) / len(team_keys)
            if len(team_keys) > 0 else np.nan
        )
    })

Summary_Full = pd.DataFrame(summary_list_full)

Summary_Full.to_csv(
    output_compare / "Full样本_分析师共同覆盖_拆开与不拆开AnanmID对比汇总.csv",
    index=False,
    encoding='utf-8-sig'
)

print(Summary_Full)