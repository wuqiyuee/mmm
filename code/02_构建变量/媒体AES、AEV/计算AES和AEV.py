"""
构造 CNRDS 报刊新闻口径的 AES/AEV。

这个脚本只在自然日新闻面板上构造基础媒体变量 AES 和 AEV。
positiveNews、negativeNews、logIncreaseAEV、logDecreaseAEV 应在合并到最终
交易日样本后再用 add_news_controls_on_panel() 计算，才能让“上一期”对应
原 Stata 代码里的上一条交易日观测。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_INPUT=Path('/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/00_合并原始数据/报刊财经新闻量化统计_1994_2024.csv')
DEFAULT_OUTPUT=Path('/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/02_变量构造结果/报刊新闻_AES_AEV/报刊新闻_AES_AEV.csv')
DEFAULT_SAMPLE_START="2000-01-01"
DEFAULT_SAMPLE_END="2024-12-31"
DEFAULT_ROLLING_WINDOW=91

def try_read_csv(path:Path,usecols:list[str])-> pd.DataFrame:
    """
    读取 CNRDS 报刊财经新闻量化统计 CSV。

    输入：
    - path：CSV 文件路径。
    - usecols：只读取后续构造 AES/AEV 所需的字段。

    处理：
    - 依次尝试 utf-8-sig、gb18030、gbk 和 pandas 默认编码，避免中文 CSV 编码不一致导致读取失败。

    输出：
    - 只包含 usecols 字段的 DataFrame。
    """
    encodings:Iterable[str|None]=['utf-8-sig','gb18030','gbk',None]
    last_err:Exception|None=None
    for enc in encodings:
        try:
            return pd.read_csv(path,usecols=usecols,encoding=enc)
        except Exception as e:
            last_err=e
    raise RuntimeError(f"Failed to read {path}. Last error:{last_err}")

def build_AES_AEV(
    df:pd.DataFrame,
    use_title:bool=False,
    sample_start:str|pd.Timestamp=DEFAULT_SAMPLE_START,
    sample_end:str|pd.Timestamp=DEFAULT_SAMPLE_END,
    rolling_window:int=DEFAULT_ROLLING_WINDOW,
)->pd.DataFrame:
    """
    根据公司-日期层面的 CNRDS 新闻量和情绪分类字段构造 AES、AEV。

    输入字段：
    - Scode、Coname、Date：股票代码、公司简称、统计日期。
    - Newsnum_Title、Newsnum_Cont：标题/正文出现该公司的日度新闻数。
    - Posnews_All、Neunews_All、Negnews_All：日度正面/中性/负面新闻数。

    主要处理：
    - 标准化股票代码和日期。
    - 将新闻数量字段转成数值，缺失按 0 处理。
    - 对同一 Stkcd-date 的重复记录取最大值去重。
    - 使用 sample_start 之前的 rolling_window-1 天作为历史回看期。
    - 对每只股票补齐滚动计算所需的自然日，缺失新闻数填 0。
    - 不在股票首次新闻日前人为补 0；若可观察历史不足 rolling_window 天，AES/AEV 保持缺失。
    - 用过去 rolling_window 个自然日的正面、负面新闻数滚动和计算 AES 和 AEV。
    - 输出只保留 sample_start 到 sample_end 之间的自然日媒体变量。

    输出变量：
    - AEV：过去 rolling_window 个自然日正面新闻数 + 负面新闻数。
    - AES：过去 rolling_window 个自然日正面新闻数 / 非中性新闻数 * 100。

    注意：
    - positiveNews、negativeNews、logIncreaseAEV、logDecreaseAEV 不在这里最终生成；
      应在最终交易日样本上调用 add_news_controls_on_panel() 计算。
    """
    sample_start=pd.to_datetime(sample_start)
    sample_end=pd.to_datetime(sample_end)
    if sample_end < sample_start:
        raise ValueError("sample_end must be greater than or equal to sample_start")
    if rolling_window < 1:
        raise ValueError("rolling_window must be a positive integer")

    rename_map={
        "Scode":"Stkcd",
        "Coname":"company_name",
        "Date":"date",
        "Newsnum_Title":"news_title_count",
        "Newsnum_Cont":"news_content_count",
        "Posnews_All":"pos_count",
        "Neunews_All":"neu_count",
        "Negnews_All":"neg_count",
    }

    df=df.rename(columns=rename_map).copy()

    # 股票代码可能被读成数字或带交易所/其他字符；这里只提取数字并补齐为 6 位 A 股代码。
    df['Stkcd']=df['Stkcd'].astype(str).str.extract(r"(\d+)",expand=False).str.zfill(6)
    #② 正则表达式 r"(\d+)" \d = 匹配 数字（0-9）
    # + = 匹配 至少 1 个，连续的数字
    # ( ) = 捕获组，告诉程序：我只要括号里的这部分内容

    # Date 是 CNRDS 的统计日期，按自然日 0:00-24:00 统计；无法解析的日期会被置为缺失并删除。
    df['date']=pd.to_datetime(df['date'],errors="coerce")
    df=df.dropna(subset=['Stkcd','date'])

    numeric_cols=[
        "news_title_count",
        "news_content_count",
        "pos_count",
        "neu_count",
        "neg_count",
    ]

    for c in numeric_cols:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)

    # 只保留样本期和历史回看期需要的新闻记录，减少后续自然日补齐的规模。
    lookback_start=sample_start-pd.Timedelta(days=rolling_window-1)
    df=df[(df['date']>=lookback_start)&(df['date']<=sample_end)].copy()

    # 每个股票-日期保留一条记录；如有重复或冲突，按各新闻数量字段取最大值去重。
    agg=(
        df.groupby(['Stkcd','date'],as_index=False)[numeric_cols]
        .max()
        .sort_values(['Stkcd','date'])
    )#取较大值去重
#--- 重复值占比统计 ---
#原始数据总行数：1896655
#去重后总行数：1896207
#重复的记录行数：896
#涉及重复的组合数：448
#---
#重复行数占比：0.05%
#重复组合占比：0.02%

#在 448 组冲突数据中：
#因公司简称（Coname）不一致造成的冲突组数：9
#单纯因新闻数值不一致造成的冲突组数：439

    # Keep a daily news count proxy as a reference column.
    daily_count_col="news_title_count" if use_title else "news_content_count"
    agg['news_count_daily']=agg[daily_count_col]

    # 逐只股票构造自然日面板。没有新闻的日期补 0，保证 91 天滚动窗口按连续自然日计算。
    out_parts:list[pd.DataFrame]=[]
    for stkcd,g in agg.groupby('Stkcd',sort=False):
        g=g.sort_values('date').copy()

        stock_min=g['date'].min()
        stock_max=g['date'].max()

        # 只使用真实可观察到的新闻历史。若股票首次新闻日晚于历史回看起点，
        # 不再向首次新闻日前补 0；rolling(..., min_periods=rolling_window)
        # 会让历史不足 rolling_window 天的 AES/AEV 保持缺失。
        # 末尾仍延长 rolling_window-1 天，保留最后新闻对后续自然日窗口的影响。
        panel_start=max(lookback_start,stock_min)
        panel_end=min(sample_end,stock_max+pd.Timedelta(days=rolling_window-1))

        if panel_end < sample_start:
            continue

        stock_dates=pd.date_range(panel_start,panel_end,freq='D')

        g=g.set_index('date').reindex(stock_dates)
        g.index.name='date'
        g['Stkcd']=stkcd

        for c in numeric_cols+['news_count_daily']:
            g[c]=g[c].fillna(0)

        # 过去 rolling_window 个自然日分别累计正面和负面新闻；中性新闻不进入 RavenPack AES 的非中性分母。
        pos_roll=g['pos_count'].rolling(window=rolling_window,min_periods=rolling_window).sum()
        neg_roll=g['neg_count'].rolling(window=rolling_window,min_periods=rolling_window).sum()
        non_neutral=pos_roll+neg_roll

        # AEV = 过去 rolling_window 个自然日的非中性新闻量，即正面新闻数 + 负面新闻数。
        g['AEV']=non_neutral

        # AES = 正面/非中性
        g['AES']=np.where(non_neutral>0,100.0*pos_roll/non_neutral,np.nan)

        out_part=g.reset_index()
        output_start=max(sample_start,stock_min)
        out_part=out_part[(out_part['date']>=output_start)&(out_part['date']<=sample_end)]
        if not out_part.empty:
            out_parts.append(out_part)

    out=pd.concat(out_parts,ignore_index=True)

    # 最终只保留后续合并和回归需要的股票、日期、原始新闻数、AES/AEV。
    out=out[
        [
            "Stkcd",
            "date",
            "news_title_count",
            "news_content_count",
            "pos_count",
            "neg_count",
            "news_count_daily",
            "AEV",
            "AES",
        ]
    ]
    return out

def add_news_controls_on_panel(
    panel:pd.DataFrame,
    group_col:str="Stkcd",
    date_col:str="date",
    aes_col:str="AES",
    aev_col:str="AEV",
)->pd.DataFrame:
    """
    在最终交易日样本上生成新闻控制变量，贴近原 Stata 代码的处理方式。

    输入：
    - panel：已经合并了 AES 和 AEV 的最终样本，通常是一行一个股票-交易日。
    - group_col/date_col：股票代码列和日期列。
    - aes_col/aev_col：AES 和 AEV 列。

    输出：
    - AES_filled：按股票向前填补后的 AES。
    - positiveNews/negativeNews：AES_filled 相比上一条样本观测改善/恶化。
    - AEV_filled：按股票向前填补 AEV，并将仍缺失的值置 0。
    - logIncreaseAEV/logDecreaseAEV：AEV_filled 相比上一条样本观测的上升/下降幅度。
    """
    out=panel.sort_values([group_col,date_col]).copy()
    by_stock=out.groupby(group_col,sort=False)

    out['AES_filled']=by_stock[aes_col].ffill()
    aes_lag=out.groupby(group_col,sort=False)['AES_filled'].shift(1)
    out['positiveNews']=(
        ((out['AES_filled']>aes_lag)&aes_lag.notna())|
        ((out['AES_filled']>50)&aes_lag.isna())
    )&out['AES_filled'].notna()
    out['negativeNews']=(
        ((out['AES_filled']<aes_lag)&aes_lag.notna())|
        ((out['AES_filled']<50)&aes_lag.isna())
    )&out['AES_filled'].notna()

    out['AEV_filled']=by_stock[aev_col].ffill().fillna(0)
    aev_lag=out.groupby(group_col,sort=False)['AEV_filled'].shift(1).fillna(0)
    delta_aev=out['AEV_filled']-aev_lag
    out['logIncreaseAEV']=np.log1p(delta_aev.clip(lower=0))
    out['logDecreaseAEV']=np.log1p((-delta_aev).clip(lower=0))
    return out

def main() -> None:
    """
    命令行入口。

    处理流程：
    - 解析输入路径、输出路径和 news_count_daily 参考列设置。
    - 从包含历史回看期的 CNRDS 合并表读取必要字段。
    - 调用 build_AES_AEV 构造自然日 AES、AEV。
    - 将结果保存为 UTF-8-SIG CSV，方便 Excel 和后续 Python/Stata 读取。
    """
    parser =argparse.ArgumentParser(description="构建AES和AEV指标")
    parser.add_argument("--input",type=Path,default=DEFAULT_INPUT,help="报刊新闻数据储存路径")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT,help="输出路径")
    parser.add_argument("--sample-start",default=DEFAULT_SAMPLE_START,help="输出样本开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--sample-end",default=DEFAULT_SAMPLE_END,help="输出样本结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--rolling-window",type=int,default=DEFAULT_ROLLING_WINDOW,help="AES/AEV 滚动自然日窗口长度")
    parser.add_argument(
        "--use-title-count",
        action='store_true',
        help="使用标题出现该公司的新闻总数而不是内容出现该公司的新闻总数作为news_count_daily参考列；AEV仍按rolling-window正面+负面新闻滚动和计算"
    )
    args=parser.parse_args()

    usecols=[
        "Scode",
        "Coname",
        "Date",
        "Newsnum_Title",
        "Newsnum_Cont",
        "Posnews_All",
        "Neunews_All",
        "Negnews_All",
    ]
    df=try_read_csv(args.input,usecols=usecols)
    out=build_AES_AEV(
        df,
        use_title=args.use_title_count,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        rolling_window=args.rolling_window,
    )

    args.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output, index=False,encoding='utf-8-sig')

    print(f"[OK] wrote {args.output}")
    print(out.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
