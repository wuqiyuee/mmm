"""用 DATAGO（数行者数据库）报刊日度计数构造两项新闻指标。

输入是已聚合的公司自然日计数表，不再读取文章明细或重新分类情感。
默认处理1998—2025年；输出起点之前的窗口历史参与计算，最后才裁输出日期。
每家公司首次新闻日从完整输入确定，不向该日期之前补零；这只是保守计算
约定，并非数据库覆盖起点。首次新闻日以后，区间内缺少记录的自然日补零，
一直延续到输出终点，包括最后一篇新闻之后的日期。

AES（综合事件情绪）= 窗口正面新闻篇数 / 窗口非中性新闻篇数 × 100。
AEV（综合事件数量）= 窗口正面新闻篇数 + 窗口负面新闻篇数。
默认窗口含当日及此前90个自然日。历史不足时两项指标均缺失；完整窗口内
无非中性新闻时，数量为零、情绪缺失。输入应保留完整历史，不能预先只裁
研究期，否则无法识别早期已有新闻、近期没有新闻的公司。

自然日结果和真实交易日结果分别保存，使用独立来源文件名。交易日识别和
四个变化控制变量复用同目录现有指标脚本；控制变量只对完整窗口计算，再
合并回全部真实交易日。因此历史不足、从未出现新闻的公司日仍保留缺失。
首个有效情绪与50比较，首个有效新闻量变化设零，沿用现有项目口径。

运行时只在项目 work（临时工作目录）下准备文件；两份结果均完成后才发布。
逐文件替换失败时尝试恢复旧结果；恢复失败则保留临时备份并报告位置。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from 计算AES和AEV import add_news_controls_on_panel, load_actual_trading_days


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = (
    ROOT / "中间结果/00_合并原始数据/DATAGO_报刊新闻量化统计_1998_2025.csv"
)
OUTPUT_DIR = ROOT / "中间结果/02_变量构造结果/3报刊新闻_AES_AEV"
DEFAULT_OUTPUT = OUTPUT_DIR / "DATAGO_报刊新闻_AES_AEV_自然日.csv"
DEFAULT_CONTROLS_OUTPUT = OUTPUT_DIR / "DATAGO_报刊新闻_AES_AEV_真实交易日控制变量.csv"
DEFAULT_STOCK_DAILY_INPUT = (
    ROOT / "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv"
)
SOURCE_START = pd.Timestamp("1998-01-01")
SOURCE_END = pd.Timestamp("2025-12-31")
DEFAULT_WINDOW = 91
DEFAULT_READ_CHUNKSIZE = 500_000

RENAME_COLUMNS = {
    "Scode": "Stkcd",
    "Date": "date",
    "Newsnum_Title": "news_title_count",
    "Newsnum_Cont": "news_content_count",
    "Posnews_All": "pos_count",
    "Neunews_All": "neu_count",
    "Negnews_All": "neg_count",
    "Unclassifiednews_All": "unclassified_count",
    "Allnews_All": "all_count",
}
DAILY_COUNT_COLUMNS = [
    "news_title_count", "news_content_count", "pos_count", "neu_count",
    "neg_count", "unclassified_count", "all_count",
]
NATURAL_COLUMNS = [
    "Stkcd", "date", *DAILY_COUNT_COLUMNS, "news_count_daily",
    "pos_count_window", "neg_count_window", "window_days", "window_complete",
    "AES", "AEV",
]
CONTROL_COLUMNS = [
    "aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV",
]
TRADING_COLUMNS = [
    "Stkcd", "date", "has_news_history", "window_days", "window_complete",
    "AES", "AEV", *CONTROL_COLUMNS,
]


class OutputPublishError(RuntimeError):
    """发布失败时记录是否必须保留临时备份供恢复。"""

    def __init__(self, message: str, preserve_temp: bool = False) -> None:
        super().__init__(message)
        self.preserve_temp = preserve_temp


def parse_sample_range(
    sample_start: str | pd.Timestamp,
    sample_end: str | pd.Timestamp,
    rolling_window: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """输出期不能超出当前来源已确认的覆盖期。"""

    start, end = pd.Timestamp(sample_start), pd.Timestamp(sample_end)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("输出起止日期不能缺失")
    start, end = start.normalize(), end.normalize()
    if end < start:
        raise ValueError("输出结束日期不能早于开始日期")
    if start < SOURCE_START or end > SOURCE_END:
        raise ValueError("当前报刊来源的输出日期应位于1998-01-01至2025-12-31")
    if not isinstance(rolling_window, int) or rolling_window < 1:
        raise ValueError("滚动窗口必须是正整数")
    return start, end


def read_daily_counts(path: Path) -> pd.DataFrame:
    """严格读取完整计数表；原始缺失值、重复键或计数冲突不能补零掩盖。"""

    table = pd.read_csv(
        path,
        usecols=list(RENAME_COLUMNS),
        dtype={"Scode": "string", "Date": "string"},
        encoding="utf-8-sig",
        low_memory=False,
    ).rename(columns=RENAME_COLUMNS)
    if table.empty:
        raise ValueError("输入新闻计数表为空")

    codes = table["Stkcd"].str.strip()
    invalid_codes = ~codes.str.fullmatch(r"\d{6}", na=False)
    if invalid_codes.any():
        raise ValueError(
            "输入必须使用六位股票代码，示例: "
            f"{table.loc[invalid_codes, 'Stkcd'].head(3).tolist()}"
        )
    table["Stkcd"] = codes
    dates = pd.to_datetime(table["date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError(
            "输入存在缺失或无法解析的自然日，示例: "
            f"{table.loc[dates.isna(), 'date'].head(3).tolist()}"
        )
    table["date"] = dates.dt.normalize()
    if not table["date"].between(SOURCE_START, SOURCE_END).all():
        raise ValueError("输入新闻日期超出当前报刊来源1998—2025年的覆盖范围")

    for column in DAILY_COUNT_COLUMNS:
        numeric = pd.to_numeric(table[column], errors="coerce")
        valid = numeric.notna() & np.isfinite(numeric) & numeric.ge(0)
        valid &= numeric.eq(np.floor(numeric))
        if not valid.all():
            raise ValueError(
                f"新闻计数 {column} 必须为非负整数，异常示例: "
                f"{table.loc[~valid, ['Stkcd', 'date', column]].head(3).to_dict('records')}"
            )
        table[column] = numeric.astype("int64")

    duplicates = table.duplicated(["Stkcd", "date"], keep=False)
    if duplicates.any():
        raise ValueError(
            "公司自然日计数表存在重复键，示例: "
            f"{table.loc[duplicates, ['Stkcd', 'date']].head(3).to_dict('records')}"
        )
    classified = table[
        ["pos_count", "neu_count", "neg_count", "unclassified_count"]
    ].sum(axis=1)
    if not classified.eq(table["all_count"]).all():
        raise ValueError("正面、中性、负面及未分类篇数之和不等于全部篇数")
    if (
        table[["news_title_count", "news_content_count"]]
        .gt(table["all_count"], axis=0)
        .any(axis=None)
    ):
        raise ValueError("标题篇数或正文篇数不能超过全部篇数")
    return table


def build_company_natural_panel(
    stock_code: str,
    daily: pd.DataFrame,
    first_news_date: pd.Timestamp,
    sample_start: pd.Timestamp,
    sample_end: pd.Timestamp,
    rolling_window: int,
    use_title_count: bool = False,
) -> pd.DataFrame:
    """先保留预热历史并滚动，最后输出研究期内的单家公司自然日表。"""

    if pd.isna(first_news_date) or first_news_date > sample_end:
        return pd.DataFrame(columns=NATURAL_COLUMNS)
    lookback_start = sample_start - pd.Timedelta(days=rolling_window - 1)
    panel_start = max(first_news_date, lookback_start)
    dates = pd.date_range(panel_start, sample_end, freq="D")
    # 输入已经通过非缺失检查，填零只作用于重新对齐后新增的日期。
    panel = (
        daily.set_index("date")[DAILY_COUNT_COLUMNS]
        .sort_index()
        .reindex(dates, fill_value=0)
        .rename_axis("date")
    )
    panel["Stkcd"] = stock_code
    count_column = "news_title_count" if use_title_count else "news_content_count"
    panel["news_count_daily"] = panel[count_column]
    panel["pos_count_window"] = panel["pos_count"].rolling(
        rolling_window, min_periods=rolling_window
    ).sum()
    panel["neg_count_window"] = panel["neg_count"].rolling(
        rolling_window, min_periods=rolling_window
    ).sum()
    panel["window_days"] = np.minimum(
        np.arange(1, len(panel) + 1), rolling_window
    )
    panel["window_complete"] = panel["window_days"].eq(rolling_window)
    panel["AEV"] = panel["pos_count_window"] + panel["neg_count_window"]
    panel["AES"] = (
        100.0 * panel["pos_count_window"] / panel["AEV"].where(panel["AEV"].gt(0))
    )
    if not panel["AEV"].notna().eq(panel["window_complete"]).all():
        raise AssertionError("窗口完整标记与数量指标的缺失状态不一致")
    if not panel["AES"].dropna().between(0, 100).all():
        raise AssertionError("情绪指标超出0—100范围")
    if panel["AEV"].dropna().lt(0).any():
        raise AssertionError("数量指标不能为负数")

    return panel.loc[panel.index >= sample_start].reset_index()[NATURAL_COLUMNS]


def build_company_trading_panel(
    actual_days: pd.DataFrame,
    natural_panel: pd.DataFrame,
    first_news_date: pd.Timestamp,
) -> pd.DataFrame:
    """先匹配全部真实交易日，只对完整窗口计算变化，再合并回原行。"""

    trading = actual_days.sort_values("date").copy()
    if natural_panel.empty:
        trading["AES"] = np.nan
        trading["AEV"] = np.nan
        trading["window_days"] = 0
    else:
        levels = natural_panel[["Stkcd", "date", "AES", "AEV", "window_days"]]
        trading = trading.merge(
            levels, on=["Stkcd", "date"], how="left", validate="one_to_one"
        )
        trading["window_days"] = trading["window_days"].fillna(0).astype("int64")
    trading["has_news_history"] = (
        trading["date"].ge(first_news_date) if pd.notna(first_news_date) else False
    )
    trading["window_complete"] = trading["AEV"].notna()

    valid = trading.loc[
        trading["window_complete"], ["Stkcd", "date", "AES", "AEV"]
    ]
    if valid.empty:
        for column in CONTROL_COLUMNS:
            trading[column] = np.nan
    else:
        # 原函数的首期处理依赖完整窗口筛选后的第一行；零新闻量必须保留。
        controls = add_news_controls_on_panel(valid)
        trading = trading.merge(
            controls[["Stkcd", "date", *CONTROL_COLUMNS]],
            on=["Stkcd", "date"], how="left", validate="one_to_one",
        )
    if len(trading) != len(actual_days):
        raise AssertionError("匹配指标后真实交易日行数发生变化")
    return trading[TRADING_COLUMNS]


def append_table(table: pd.DataFrame, path: Path, first: bool) -> None:
    """只向待发布文件追加；首批写表头，其余批次不重复写表头。"""

    table.to_csv(
        path, mode="w" if first else "a", header=first, index=False,
        encoding="utf-8-sig" if first else "utf-8", date_format="%Y-%m-%d",
    )


def publish_outputs(
    staged: list[tuple[Path, Path]], temp_dir: Path,
) -> None:
    """逐文件原子替换；第二份失败时回滚已发布文件，不宣称双文件事务。"""

    backups: dict[Path, Path] = {}
    published: list[Path] = []
    for index, (_, destination) in enumerate(staged):
        if destination.exists():
            backup = temp_dir / f"original_{index}.csv"
            shutil.copy2(destination, backup)
            backups[destination] = backup
    try:
        for pending, destination in staged:
            pending.replace(destination)
            published.append(destination)
    except BaseException as error:
        recovery_errors = []
        for destination in reversed(published):
            try:
                if destination in backups:
                    backups[destination].replace(destination)
                else:
                    destination.unlink()
            except OSError as recovery_error:
                recovery_errors.append(f"{destination}: {recovery_error}")
        if recovery_errors:
            raise OutputPublishError(
                "发布失败且部分旧结果未能恢复，备份保留在 "
                f"{temp_dir}；恢复异常: {'; '.join(recovery_errors)}",
                preserve_temp=True,
            ) from error
        raise OutputPublishError("发布失败，已恢复此前正式结果") from error


def run_pipeline(args: argparse.Namespace) -> dict[str, int]:
    """逐公司生成两张结果表，保留全量输入的首次新闻日期及公司范围。"""

    start, end = parse_sample_range(
        args.sample_start, args.sample_end, args.rolling_window
    )
    if args.read_chunksize < 1:
        raise ValueError("分块读取行数必须为正整数")
    destinations = [args.output.resolve(), args.controls_output.resolve()]
    sources = {args.input.resolve(), args.stock_daily_input.resolve()}
    if destinations[0] == destinations[1] or any(p in sources for p in destinations):
        raise ValueError("两份结果路径必须不同，且不能覆盖输入文件")
    work_dir = ROOT / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.stat().st_dev != work_dir.stat().st_dev:
            raise ValueError("正式结果与项目临时工作目录必须位于同一文件系统")
        if destination.exists() and not destination.is_file():
            raise ValueError(f"输出路径不是普通文件: {destination}")

    print("[1/4] 读取并校验完整报刊日度计数表")
    news = read_daily_counts(args.input)
    # 先获取完整输入的首次新闻日，再筛回看期；近期无新闻的老公司仍保留。
    first_news = news.groupby("Stkcd", sort=True)["date"].min()
    print(f"输入行数={len(news):,}；输入公司数={len(first_news):,}")
    print(f"未分类公司新闻篇次={int(news['unclassified_count'].sum()):,}")
    lookback_start = start - pd.Timedelta(days=args.rolling_window - 1)
    news = news.loc[news["date"].between(lookback_start, end)].reset_index(drop=True)
    news_groups = news.groupby("Stkcd", sort=False).indices

    print("[2/4] 读取 A 股实际交易日，保留真实交易观测")
    actual_days = load_actual_trading_days(
        path=args.stock_daily_input, sample_start=start, sample_end=end,
        chunksize=args.read_chunksize,
    )
    trading_groups = actual_days.groupby("Stkcd", sort=False).indices
    stocks = sorted(set(first_news.index) | set(trading_groups))
    summary = {
        "input_firms": int(len(first_news)), "natural_firms": 0,
        "natural_rows": 0, "incomplete_natural_rows": 0,
        "zero_volume_natural_rows": 0, "actual_trading_rows": int(len(actual_days)),
        "trading_rows": 0, "valid_trading_rows": 0, "no_history_trading_rows": 0,
    }
    temp_dir = Path(tempfile.mkdtemp(prefix="datago_indicators_", dir=work_dir))
    keep_temp = False
    pending_natural = temp_dir / "natural.csv"
    pending_trading = temp_dir / "trading.csv"
    try:
        print("[3/4] 按公司补齐自然日、滚动计算并匹配实际交易日")
        for number, stock in enumerate(stocks, 1):
            first_date = first_news.get(stock, pd.NaT)
            positions = news_groups.get(stock)
            daily = news.iloc[positions] if positions is not None else news.iloc[:0]
            natural = build_company_natural_panel(
                stock, daily, first_date, start, end, args.rolling_window,
                use_title_count=args.use_title_count,
            )
            if not natural.empty:
                append_table(natural, pending_natural, summary["natural_rows"] == 0)
                summary["natural_firms"] += 1
                summary["natural_rows"] += len(natural)
                summary["incomplete_natural_rows"] += int(
                    (~natural["window_complete"]).sum()
                )
                summary["zero_volume_natural_rows"] += int(natural["AEV"].eq(0).sum())

            trading_positions = trading_groups.get(stock)
            if trading_positions is not None:
                trading = build_company_trading_panel(
                    actual_days.iloc[trading_positions], natural, first_date
                )
                append_table(trading, pending_trading, summary["trading_rows"] == 0)
                summary["trading_rows"] += len(trading)
                summary["valid_trading_rows"] += int(trading["window_complete"].sum())
                summary["no_history_trading_rows"] += int(
                    (~trading["has_news_history"]).sum()
                )
            if number % 100 == 0 or number == len(stocks):
                print(f"已处理公司 {number:,}/{len(stocks):,}", flush=True)

        if summary["natural_rows"] == 0:
            raise ValueError("所选日期内没有可输出的自然日新闻指标")
        if summary["trading_rows"] != len(actual_days):
            raise AssertionError("最终真实交易日表未完整保留基础交易日观测")
        print("[4/4] 两份结果均已准备完成，发布正式文件")
        publish_outputs(
            [(pending_natural, destinations[0]), (pending_trading, destinations[1])],
            temp_dir,
        )
    except OutputPublishError as error:
        keep_temp = error.preserve_temp
        raise
    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从数行者报刊日度计数计算自然日指标及真实交易日控制变量"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="日度计数表")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="自然日结果")
    parser.add_argument(
        "--controls-output", type=Path, default=DEFAULT_CONTROLS_OUTPUT,
        help="真实交易日结果，保留未达到完整窗口的缺失观测",
    )
    parser.add_argument(
        "--stock-daily-input", type=Path, default=DEFAULT_STOCK_DAILY_INPUT,
        help="用于识别实际交易日的个股日度基础表",
    )
    parser.add_argument("--sample-start", default="1998-01-01", help="输出开始自然日")
    parser.add_argument("--sample-end", default="2025-12-31", help="输出结束自然日")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_WINDOW, help="滚动自然日数")
    parser.add_argument(
        "--read-chunksize", type=int, default=DEFAULT_READ_CHUNKSIZE,
        help="分块读取个股交易日基础表时每块的行数",
    )
    parser.add_argument(
        "--use-title-count", action="store_true",
        help="日度参考新闻量使用标题篇数；两项指标仍由正负新闻篇数计算",
    )
    args = parser.parse_args()
    print(f"滚动窗口={args.rolling_window}个自然日，包含当日")
    print("首次新闻日前不补零；完整零新闻量保留，历史不足与情绪缺失分别处理")
    summary = run_pipeline(args)
    print(f"已写出自然日结果: {args.output.resolve()}")
    print(f"已写出真实交易日结果: {args.controls_output.resolve()}")
    print(f"处理摘要: {summary}")


if __name__ == "__main__":
    main()
