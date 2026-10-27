"""
构造 CNRDS 报刊新闻口径的 AES、AEV 及四个新闻控制变量。

处理分为两个阶段：

1. 在连续自然日新闻面板上，使用过去 91 个自然日计算 AES 和 AEV；
2. 读取 CSMAR 个股日度数据，只保留 A 股 Filling=0 的真实交易观测，
   将自然日 AES、AEV 合并到真实交易日面板，并计算：
   aesIncrease、aesDecrease、logIncreaseAEV、logDecreaseAEV。

自然日结果和真实交易日结果分别保存，避免影响记忆矩阵对自然日 AEV 的使用。
"""

from __future__ import annotations # 让类型注解更灵活，减少因为定义顺序造成的问题

import argparse # argparse 是 Python 标准库，用来处理命令行参数。
from pathlib import Path
from typing import Iterable # Iterable 通常只是类型提示，不负责真正处理数据

import numpy as np
import pandas as pd

# 根据脚本位置定位项目根目录，整个项目移动后无需修改路径。
ROOT = Path(__file__).resolve().parents[4]


DEFAULT_INPUT = ROOT / '中间结果/00_合并原始数据/报刊财经新闻量化统计_1994_2025.csv'



AES_AEV_DIR = ROOT / '中间结果/02_变量构造结果/3报刊新闻_AES_AEV'
# 自然日 AES、AEV 输出，继续供记忆矩阵使用
DEFAULT_OUTPUT = AES_AEV_DIR / "报刊新闻_AES_AEV.csv"

# 新增的真实交易日新闻控制变量输出
DEFAULT_CONTROLS_OUTPUT = (
    AES_AEV_DIR / "报刊新闻_AES_AEV_真实交易日控制变量.csv"
)

# 用来识别个股真实交易日
DEFAULT_STOCK_DAILY_INPUT = ROOT / '中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv'



DEFAULT_SAMPLE_START = "1994-01-01"
DEFAULT_SAMPLE_END = "2025-12-31"
DEFAULT_ROLLING_WINDOW = 91
DEFAULT_READ_CHUNKSIZE = 500_000

# 当前项目使用的 A 股市场类型
A_SHARE_MARKET_TYPES = (1, 4, 16, 32)

def try_read_csv(path:Path,usecols:list[str])-> pd.DataFrame: #括号里面是两个参数，冒号 : 后面不是默认值，而是类型注解
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
    # ：冒号后面接类型注解
    # Iterable 表示encodings是“可以被循环遍历的东西”
    # str|None表示encodings里面的元素可以是字符串str，也可以是None空
    # 变量名: 类型 = 值
    last_err:Exception|None=None
    # last_err 这个变量以后可以存一个“异常对象”（Exception），也可以什么都不存（None）
    for enc in encodings:
        try:
            return pd.read_csv(path,usecols=usecols,encoding=enc)
        except Exception as e: # 如果刚才发生了一个错误，就把这个错误临时命名为 e，不直接中断程序
            last_err=e
    raise RuntimeError(f"无法读取 {path}。最后一次错误是:{last_err}")
    # RuntimeError(...) 是创建一个运行时错误，主动把错误抛出来


def normalize_stock_codes(series: pd.Series) -> pd.Series:
    """
    将股票代码标准化为六位字符串。

    可以处理：
    - 1
    - 1.0
    - 000001
    - 带有其他字符的股票代码
    """
    return (
        series.astype("string")
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(6)
    )

    # .str表示要对字符串进行字符串操作
    # .str.extract(...)从字符串里“提取”符合某种规则的内容
    # 正则表达式 r"(\d+)" ，r“（）” = 不要让 Python 乱解释字符串里的 \
    # \d = 一个数字（0-9）
    # + = 前面的东西出现一次或多次
    # \d+就是：连续的一个或多个数字
    # ( ) = 捕获组，告诉程序：我只要括号里的这部分内容
    # 默认会提取第一段连续数字


def build_AES_AEV(# 构建AES和AEV变量
    df:pd.DataFrame, # 输入数据表
    use_title:bool=False, # 使用标题（布尔值），默认不使用
    sample_start:str|pd.Timestamp=DEFAULT_SAMPLE_START,# 样本开始日期
    sample_end:str|pd.Timestamp=DEFAULT_SAMPLE_END,# 样本结束日期
    rolling_window:int=DEFAULT_ROLLING_WINDOW,# 滚动窗口
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
    - 每只股票从 sample_start 与首次新闻日的较晚者开始构造自然日面板。
    - 面板起点后的前 rolling_window-1 个自然日用于积累历史数值；
        第 rolling_window 个自然日以“此前 rolling_window-1 日 + 当日”
        组成首个完整窗口并首次计算 AES 和 AEV。
    - 对每只股票补齐滚动计算所需的自然日，缺失新闻数填 0。
    - 不在股票首次新闻日前人为补 0；若可观察历史不足 rolling_window 天，AES/AEV 保持缺失。
    - 用过去 rolling_window 个自然日的正面、负面新闻数滚动和计算 AES 和 AEV。
    - 输出只保留 sample_start 到 sample_end 之间的自然日媒体变量。

    输出变量：
    - AEV：过去 rolling_window 个自然日正面新闻数 + 负面新闻数。
    - AES：过去 rolling_window 个自然日正面新闻数 / 非中性新闻数 * 100。

    """
    sample_start=pd.to_datetime(sample_start)
    sample_end=pd.to_datetime(sample_end)
    if sample_end < sample_start:
        raise ValueError("样本结束时间必须要晚于样本开始时间")
    if rolling_window < 1:
        raise ValueError("滚动窗口必须是正整数")

    rename_map={
        "Scode":"Stkcd",
        "Coname":"company_name",
        "Date":"date",
        "Newsnum_Title":"news_title_count",# 标题出现该公司的新闻总数
        "Newsnum_Cont":"news_content_count",# 内容出现该公司的新闻总数
        "Posnews_All":"pos_count",# 正面新闻数（全部）
        "Neunews_All":"neu_count",
        "Negnews_All":"neg_count",
    }

    df=df.rename(columns=rename_map).copy()

    # 股票代码可能被读成数字或带交易所/其他字符；这里只提取数字并补齐为 6 位 A 股代码。
    df["Stkcd"] = normalize_stock_codes(df["Stkcd"])

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

    # --- 重复值占比统计 ---
    # 原始数据总行数：1896655
    # 去重后总行数：1896207
    # 重复的记录行数：896
    # 涉及重复的组合数：448
    # ---
    # 重复行数占比：0.05%
    # 重复组合占比：0.02%

    # 在 448 组冲突数据中：
    # 因公司简称（Coname）不一致造成的冲突组数：9
    # 单纯因新闻数值不一致造成的冲突组数：439

    # 保留每日新闻数量代理数据
    daily_count_col="news_title_count" if use_title else "news_content_count"
    agg['news_count_daily']=agg[daily_count_col]

    # 逐只股票构造自然日面板。没有新闻的日期补 0，保证 91 天滚动窗口按连续自然日计算。
    out_parts:list[pd.DataFrame]=[]# out_parts是一个列表（默认值为空列表），里面的元素是表格
    for stkcd,g in agg.groupby('Stkcd',sort=False):
        # agg.groupby('Stkcd',sort=False)相当于按股票代码把agg这个大表分为一个个小表
        g=g.sort_values('date').copy()

        stock_min=g['date'].min()# 某只股票最早出现的时间

        # 只使用真实可观察到的新闻历史。若股票首次新闻日晚于历史回看起点，
        # 不再向首次新闻日前补 0；rolling(..., min_periods=rolling_window)
        # 会让历史不足 rolling_window 天的 AES/AEV 保持缺失。
        # 个股自然日面板只构造到 sample_end，不向样本结束日之后延长。
        panel_start = max(
            sample_start,
            stock_min,
        )# 个股面板开始的时间
        # 首次出现新闻以后一直补到样本期末。
        # 这样最后一篇新闻退出 91 个自然日窗口后，AEV 会正确下降到 0，
        # 不会在交易日合并时错误地长期沿用最后一次非零 AEV。
        panel_end = sample_end
        # 个股面板结束的时间

        if panel_end < sample_start:# 如果某只股票的结束日期比整个研究样本的开始日期还早
            continue # 就直接跳过

        stock_dates=pd.date_range(panel_start,panel_end,freq='D')# 生成个度日度序列

        g=g.set_index('date').reindex(stock_dates)
        # 先把 date 设成索引，然后按照 stock_dates 这一整套日期重新排列，并把缺失日期补出来。
        # 这对后面的rolling很重要
        g.index.name='date' # 给索引起名为“date”
        g['Stkcd']=stkcd # 给每个字表都新增一列“Stkcd”,并使用stkcd赋值

        for c in numeric_cols+['news_count_daily']:
            g[c]=g[c].fillna(0) # 将缺失值填充为0

        # 过去 rolling_window 个自然日分别累计正面和负面新闻；中性新闻不进入 RavenPack AES 的非中性分母。
        pos_roll=g['pos_count'].rolling(window=rolling_window,min_periods=rolling_window).sum()
        neg_roll=g['neg_count'].rolling(window=rolling_window,min_periods=rolling_window).sum()
        non_neutral=pos_roll+neg_roll

        # AEV = 过去 rolling_window 个自然日的非中性新闻量，即正面新闻数 + 负面新闻数。
        g['AEV']=non_neutral

        # AES = 正面/非中性
        g['AES']=np.where(non_neutral>0,100.0*pos_roll/non_neutral,np.nan)

        out_part=g.reset_index()# 把 date 从索引变回普通列
        output_start=max(sample_start,stock_min)
        out_part=out_part[(out_part['date']>=output_start)&(out_part['date']<=sample_end)]# 确认日期范围
        if not out_part.empty:
            out_parts.append(out_part)# 追加结果

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


def load_actual_trading_days(
    path: Path,
    sample_start: str | pd.Timestamp,
    sample_end: str | pd.Timestamp,
    chunksize: int = DEFAULT_READ_CHUNKSIZE,
) -> pd.DataFrame:
    """
    从 CSMAR 个股日度数据中读取真实个股交易日。

    只保留：
    1. A 股市场；
    2. Filling=0 的真实交易观测；
    3. 指定样本日期范围。

    此处不删除金融业、ST/PT 或上市未满一年观测，
    因为新闻变化变量的“上一期”应当是上一真实交易日，
    而不是上一个符合最终回归资格的日期。
    """
    sample_start = pd.to_datetime(sample_start)
    sample_end = pd.to_datetime(sample_end)

    if sample_end < sample_start:
        raise ValueError("样本结束日期不能早于样本开始日期")

    if chunksize < 1:
        raise ValueError("chunksize 必须是正整数")

    usecols = [
        "Stkcd",
        "Trddt",
        "Markettype",
        "Filling",
    ]

    parts: list[pd.DataFrame] = []

    reader = pd.read_csv(
        path,
        usecols=usecols,
        encoding="utf-8-sig",
        dtype={"Stkcd": "string"},
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk in reader:
        stock_codes = normalize_stock_codes(chunk["Stkcd"])
        trading_dates = pd.to_datetime(
            chunk["Trddt"],
            errors="coerce",
        )
        market_type = pd.to_numeric(
            chunk["Markettype"],
            errors="coerce",
        )
        filling = pd.to_numeric(
            chunk["Filling"],
            errors="coerce",
        )

        keep = (
            market_type.isin(A_SHARE_MARKET_TYPES)
            & filling.eq(0)
            & trading_dates.between(sample_start, sample_end)
            & stock_codes.notna()
            & trading_dates.notna()
        )

        if not keep.any():
            continue

        part = pd.DataFrame(
            {
                "Stkcd": stock_codes.loc[keep],
                "date": trading_dates.loc[keep],
            }
        )

        parts.append(part)

    if not parts:
        raise ValueError("筛选后没有任何 A 股真实交易日观测")

    actual_trading_days = pd.concat(
        parts,
        ignore_index=True,
    )

    duplicate_mask = actual_trading_days.duplicated(
        ["Stkcd", "date"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_examples = (
            actual_trading_days.loc[
                duplicate_mask,
                ["Stkcd", "date"],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "A 股真实交易日索引存在重复的 Stkcd-date："
            f"{duplicate_examples}"
        )

    actual_trading_days = (
        actual_trading_days
        .sort_values(
            ["Stkcd", "date"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return actual_trading_days




def add_news_controls_on_panel(
    panel: pd.DataFrame,
    group_col: str = "Stkcd",
    date_col: str = "date",
    aes_col: str = "AES",
    aev_col: str = "AEV",
) -> pd.DataFrame:
    """
    在全部 A 股真实交易日面板上生成新闻控制变量。

    这里的上一期是同一股票的上一真实交易日，而不是：
    - 上一自然日；
    - 上一市场开市日；
    - 上一个符合最终回归样本要求的日期。

    输出：
    - AES_filled：按股票向前填补后的 AES；
    - aesIncrease：AES 相比上一真实交易日上升；
    - aesDecrease：AES 相比上一真实交易日下降；
    - AEV_filled：按股票向前填补 AEV，初始缺失设为 0；
    - logIncreaseAEV：AEV 上升幅度的对数；
    - logDecreaseAEV：AEV 下降幅度的对数。
    """
    required_columns = {
        group_col,
        date_col,
        aes_col,
        aev_col,
    }

    missing_columns = required_columns.difference(panel.columns)

    if missing_columns:
        raise ValueError(
            f"计算新闻控制变量缺少字段：{sorted(missing_columns)}"
        )

    out = panel.copy()

    out[date_col] = pd.to_datetime(
        out[date_col],
        errors="coerce",
    )
    out[aes_col] = pd.to_numeric(
        out[aes_col],
        errors="coerce",
    )
    out[aev_col] = pd.to_numeric(
        out[aev_col],
        errors="coerce",
    )

    if out[group_col].isna().any() or out[date_col].isna().any():
        raise ValueError("股票代码或交易日期存在无法识别的缺失值")

    duplicate_mask = out.duplicated(
        [group_col, date_col],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "交易日新闻面板在股票代码—日期层面不唯一"
        )

    if (out[aev_col].dropna() < 0).any():
        raise ValueError("AEV 不应出现负数")

    out = (
        out.sort_values(
            [group_col, date_col],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 1. 新闻情绪变化
    # ------------------------------------------------------------------

    out["AES_filled"] = (
        out.groupby(group_col, sort=False)[aes_col]
        .ffill()
    )

    aes_lag = (
        out.groupby(group_col, sort=False)["AES_filled"]
        .shift(1)
    )

    out["aesIncrease"] = (
        (
            (
                (out["AES_filled"] > aes_lag)
                & aes_lag.notna()
            )
            |
            (
                (out["AES_filled"] > 50)
                & aes_lag.isna()
            )
        )
        & out["AES_filled"].notna()
    ).astype("int8")

    out["aesDecrease"] = (
        (
            (
                (out["AES_filled"] < aes_lag)
                & aes_lag.notna()
            )
            |
            (
                (out["AES_filled"] < 50)
                & aes_lag.isna()
            )
        )
        & out["AES_filled"].notna()
    ).astype("int8")

    # ------------------------------------------------------------------
    # 2. 新闻关注度变化
    # ------------------------------------------------------------------

    out["AEV_filled"] = (
        out.groupby(group_col, sort=False)[aev_col]
        .ffill()
    )

    aev_lag = (
        out.groupby(group_col, sort=False)["AEV_filled"]
        .shift(1)
    )

    # 经过历史完整性筛选后，每只股票第一行就是
    # 第一个具有完整 91 个自然日窗口的真实交易日。
    first_valid_observation = (
        out.groupby(group_col, sort=False)
        .cumcount()
        .eq(0)
    )

    delta_aev = out["AEV_filled"] - aev_lag

    # 只把第一条有效观测的变化设为 0，
    # 不把其他意外缺失值无条件转换成 0。
    delta_aev = delta_aev.mask(
        first_valid_observation,
        0.0,
    )

    out["logIncreaseAEV"] = np.log1p(
        delta_aev.clip(lower=0)
    )

    out["logDecreaseAEV"] = np.log1p(
        (-delta_aev).clip(lower=0)
    )

    return out

def build_trading_day_news_controls(
    natural_news_panel: pd.DataFrame,
    actual_trading_days: pd.DataFrame,
) -> pd.DataFrame:
    """
    将自然日 AES、AEV 合并到真实交易日索引，
    然后计算四个新闻控制变量。
    """
    news_columns = [
        "Stkcd",
        "date",
        "AES",
        "AEV",
    ]

    missing_columns = set(news_columns).difference(
        natural_news_panel.columns
    )

    if missing_columns:
        raise ValueError(
            f"自然日新闻面板缺少字段：{sorted(missing_columns)}"
        )

    news_levels = natural_news_panel[news_columns].copy()

    duplicate_mask = news_levels.duplicated(
        ["Stkcd", "date"],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "自然日 AES/AEV 在 Stkcd-date 层面不唯一"
        )

    # 必须以真实交易日索引作为左表。
    # 这样没有新闻记录的真实交易日仍然会保留。
    trading_panel = actual_trading_days.merge(
        news_levels,
        how="left",
        on=["Stkcd", "date"],
        validate="one_to_one",
        sort=False,
    )

    if len(trading_panel) != len(actual_trading_days):
        raise AssertionError(
            "合并 AES/AEV 后真实交易日面板行数发生变化"
        )

    # AEV 缺失表示 91 个自然日历史窗口尚未完整，
    # 因而不进入新闻控制变量的计算面板。
    # 注意：AEV=0 是有效观测，不能删除。
    trading_panel = (
        trading_panel.loc[trading_panel["AEV"].notna()]
        .sort_values(
            ["Stkcd", "date"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    trading_panel = add_news_controls_on_panel(
        trading_panel,
        group_col="Stkcd",
        date_col="date",
        aes_col="AES",
        aev_col="AEV",
    )

    output_columns = [
        "Stkcd",
        "date",
        "AES",
        "AEV",
        "aesIncrease",
        "aesDecrease",
        "logIncreaseAEV",
        "logDecreaseAEV",
    ]

    return trading_panel[output_columns]

def main() -> None:
    """
    先输出自然日 AES、AEV，
    再输出真实个股交易日新闻控制变量。
    """
    parser = argparse.ArgumentParser(
        description="构建自然日 AES/AEV 和真实交易日新闻控制变量"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="报刊新闻数据路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="自然日 AES/AEV 输出路径",
    )
    parser.add_argument(
        "--stock-daily-input",
        type=Path,
        default=DEFAULT_STOCK_DAILY_INPUT,
        help="CSMAR 个股日度数据路径",
    )
    parser.add_argument(
        "--controls-output",
        type=Path,
        default=DEFAULT_CONTROLS_OUTPUT,
        help="真实交易日新闻控制变量输出路径",
    )
    parser.add_argument(
        "--sample-start",
        default=DEFAULT_SAMPLE_START,
        help="输出样本开始日期，格式为 YYYY-MM-DD",
    )
    parser.add_argument(
        "--sample-end",
        default=DEFAULT_SAMPLE_END,
        help="输出样本结束日期，格式为 YYYY-MM-DD",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=DEFAULT_ROLLING_WINDOW,
        help="AES/AEV 滚动自然日窗口长度",
    )
    parser.add_argument(
        "--read-chunksize",
        type=int,
        default=DEFAULT_READ_CHUNKSIZE,
        help="分块读取个股日度数据时每块的行数",
    )
    parser.add_argument(
        "--use-title-count",
        action="store_true",
        help=(
            "使用标题新闻数作为 news_count_daily 参考列；"
            "AEV 仍然按照正面新闻数加负面新闻数计算"
        ),
    )

    args = parser.parse_args()

    if args.output.resolve() == args.controls_output.resolve():
        raise ValueError(
            "自然日输出和真实交易日控制变量输出不能使用同一路径"
        )

    usecols = [
        "Scode",
        "Coname",
        "Date",
        "Newsnum_Title",
        "Newsnum_Cont",
        "Posnews_All",
        "Neunews_All",
        "Negnews_All",
    ]

    # ------------------------------------------------------------------
    # 第一阶段：构造并保存自然日 AES、AEV
    # ------------------------------------------------------------------

    news_raw = try_read_csv(
        args.input,
        usecols=usecols,
    )

    natural_news_panel = build_AES_AEV(
        news_raw,
        use_title=args.use_title_count,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        rolling_window=args.rolling_window,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    natural_news_panel.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print(f"[完成] 自然日 AES/AEV 已输出：{args.output}")
    print(f"自然日输出行数：{len(natural_news_panel):,}")
    print(
        natural_news_panel.head(10).to_string(index=False)
    )

    # 自然日文件已经保存，后续只需要四个字段参与合并，
    # 删除其他列可以降低内存占用。
    natural_news_panel = natural_news_panel[
        ["Stkcd", "date", "AES", "AEV"]
    ].copy()

    # ------------------------------------------------------------------
    # 第二阶段：读取 Filling=0 的真实交易日
    # ------------------------------------------------------------------

    actual_trading_days = load_actual_trading_days(
        path=args.stock_daily_input,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        chunksize=args.read_chunksize,
    )

    print(
        "A 股 Filling=0 的真实交易日观测数："
        f"{len(actual_trading_days):,}"
    )
    print(
        "真实交易日股票数："
        f"{actual_trading_days['Stkcd'].nunique():,}"
    )

    # ------------------------------------------------------------------
    # 第三阶段：构造四个新闻控制变量
    # ------------------------------------------------------------------

    controls = build_trading_day_news_controls(
        natural_news_panel=natural_news_panel,
        actual_trading_days=actual_trading_days,
    )

    args.controls_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    controls.to_csv(
        args.controls_output,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print(
        "[完成] 真实交易日新闻控制变量已输出："
        f"{args.controls_output}"
    )
    print(f"真实交易日输出行数：{len(controls):,}")
    print(
        "aesIncrease=1 数量："
        f"{controls['aesIncrease'].sum():,}"
    )
    print(
        "aesDecrease=1 数量："
        f"{controls['aesDecrease'].sum():,}"
    )
    print(
        "logIncreaseAEV 非零数量："
        f"{controls['logIncreaseAEV'].ne(0).sum():,}"
    )
    print(
        "logDecreaseAEV 非零数量："
        f"{controls['logDecreaseAEV'].ne(0).sum():,}"
    )
    print(controls.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
