"""独立试跑：2005—2009年报刊、2010—2025年网络新闻的表2收益回归。

文件名沿用原试跑入口，实际回归研究期已调整为2005—2025年。
直接运行本文件，默认先重建模式公司和完整样本两个面板，再估计24个模型。
已有本口径的两个面板、仅需重估回归时，使用 --stage 回归。
更新面板时使用 --stage 模式面板、--stage 完整面板 分步运行；
使用 --stage 全部 可依次重建两个面板并估计回归。

复用现有合并脚本的分步函数和表2估计函数，不修改来源脚本。
直接读取原合并流程保存的共享提示事件表，不重建或另存提示事件。
共享事件表须保留完整历史和既有字段；记忆矩阵或提示识别规则变化后，
应先通过原合并流程更新共享事件表，再重建本次分段新闻面板。
从筛样前日度基础表出发，先按日期合并新闻控制，再按原规则筛样。
2010年1月1日前使用原主线报刊新闻控制，此后使用数行者网络新闻控制。
两类来源各自先计算91自然日窗口及交易日变化，再选择所属日期段；
网络新闻保留2009年历史参与窗口计算，切换时不跨来源计算变化或填补缺失。
历史提示、公告窗口、公司对排除、缩尾参照范围及基准筛样顺序沿用现有口径；
研究期在上述步骤完成后限制为2005—2025年，不提前截断历史数据。

仅保存表2所需列，不生成其他表格使用的机制分组输出和未来提示标记。
面板另存至“中间结果/04_合并数据/分段新闻_2005_2025/”；
结果另存至“图表结果/01_主线图表/表2_收益回归/分段新闻_2005_2025/”。
只估计模型汇总和系数明细，不自动绘图。
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MERGE_DIR = PROJECT_ROOT / "code/04_合并数据"
NEWSPAPER_NEWS_PATH = PROJECT_ROOT / (
    "中间结果/02_变量构造结果/3报刊新闻_AES_AEV/"
    "报刊新闻_AES_AEV_真实交易日控制变量.csv"
)
NETWORK_NEWS_PATH = PROJECT_ROOT / (
    "中间结果/02_变量构造结果/3网络新闻_AES_AEV/"
    "DATAGO_网络新闻_AES_AEV_真实交易日控制变量.csv"
)
PANEL_DIR = PROJECT_ROOT / "中间结果/04_合并数据/分段新闻_2005_2025"
RESULT_DIR = PROJECT_ROOT / "图表结果/01_主线图表/表2_收益回归/分段新闻_2005_2025"
PATTERN_PATH = PANEL_DIR / "Pattern样本_分段新闻_表2回归面板.csv"
FULL_PATH = PANEL_DIR / "Full样本_分段新闻_表2回归面板.csv"
PATTERN_CUE_PATH = PROJECT_ROOT / "中间结果/04_合并数据/Pattern_cue_events_after_collapse.csv"
FULL_CUE_PATH = PROJECT_ROOT / "中间结果/04_合并数据/Full_cue_events_after_collapse.csv"
START_YEAR = 2005
END_YEAR = 2025
NEWS_SWITCH_DATE = pd.Timestamp("2010-01-01")
NEWS_SOURCE_LABEL = "2005—2009年报刊新闻；2010—2025年数行者网络新闻"
# 自身公告已考虑收盘后发布，仅在非交易日映射至下一交易日。
EVENT_DATE_MODE = "next_if_nontrading"


def load_module(name, path):
    """加载现有脚本中的函数；不会触发其直接运行入口。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def merge_news_controls_by_period(panel):
    """在基准筛样前，按交易日期合并两个来源已经算好的新闻控制变量。"""
    news_cols = [
        "Stkcd", "date", "AES", "AEV", "aesIncrease", "aesDecrease",
        "logIncreaseAEV", "logDecreaseAEV",
    ]
    news_parts = []
    for source_name, news_path in (
        ("报刊", NEWSPAPER_NEWS_PATH),
        ("网络", NETWORK_NEWS_PATH),
    ):
        part = pd.read_csv(
            news_path, usecols=news_cols, dtype={"Stkcd": "string"},
            encoding="utf-8-sig", parse_dates=["date"],
        )
        part["Stkcd"] = part["Stkcd"].str.strip().str.zfill(6)
        part["date"] = pd.to_datetime(part["date"], errors="raise")
        if (
            not part["Stkcd"].str.fullmatch(r"\d{6}", na=False).all()
            or part["date"].isna().any()
        ):
            raise ValueError(f"{source_name}新闻控制表存在无效股票代码或缺失日期。")

        # 此处保留回归期之前的历史，研究期在原基准筛样完成后统一截取。
        if source_name == "报刊":
            part = part.loc[part["date"] < NEWS_SWITCH_DATE].copy()
        else:
            part = part.loc[part["date"] >= NEWS_SWITCH_DATE].copy()
        if part.empty:
            raise ValueError(f"{source_name}新闻控制表在指定日期段没有观测。")
        print(f"选用{source_name}新闻控制：{len(part):,}行", flush=True)
        news_parts.append(part)

    news_controls = pd.concat(news_parts, ignore_index=True).rename(
        columns={"Stkcd": "StkcdRecall", "date": "Trddt"},
    )
    del news_parts, part
    keys = ["StkcdRecall", "Trddt"]
    if news_controls.duplicated(keys).any():
        raise ValueError("分段新闻控制表的被提示公司与交易日不唯一。")

    numeric_cols = news_cols[2:]
    news_controls[numeric_cols] = news_controls[numeric_cols].apply(
        pd.to_numeric, errors="raise",
    )
    if news_controls["AEV"].dropna().lt(0).any():
        raise ValueError("新闻关注度不能为负数。")
    panel["Trddt"] = pd.to_datetime(panel["Trddt"], errors="raise")
    if panel[keys].isna().any().any() or panel["StkcdRecall"].eq("").any():
        raise ValueError("日度基础面板存在缺失的被提示公司或交易日。")

    # 左合并保留全部基础观测；各来源的控制变量缺失交给原基准筛样处理。
    panel = panel.merge(news_controls, on=keys, how="left", validate="one_to_one")
    del news_controls
    gc.collect()
    print(f"合并分段新闻控制后面板行列数：{panel.shape}", flush=True)
    return panel


def save_table2_panel(panel, output_path, table2):
    """完成原基准筛样后，截取研究期并保存表2必要字段。"""
    columns = list(dict.fromkeys(
        ["StkcdRecall", "Trddt", "year", "yearQuarter", "quarterLag1size90plus"]
        + table2.return_cols
        + table2.table2_full_controls
    ))
    panel = panel.loc[
        panel["year"].between(START_YEAR, END_YEAR), columns
    ].copy()
    if panel.empty:
        raise ValueError(f"{START_YEAR}—{END_YEAR}年没有可用的分段新闻回归样本。")
    if panel[["StkcdRecall", "Trddt"]].isna().any().any():
        raise ValueError("分段新闻回归面板存在缺失的公司或交易日。")
    if panel.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("分段新闻回归面板的公司与交易日不唯一。")
    if panel[table2.media_cols].isna().any().any():
        raise ValueError("基准筛样后仍存在缺失的新闻控制变量。")

    panel = panel.sort_values(["StkcdRecall", "Trddt"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    print(f"已保存分段新闻回归面板：{output_path}", flush=True)
    print(f"行列数：{panel.shape}；日期：{panel['Trddt'].min()} 至 {panel['Trddt'].max()}")


def build_pattern_panel(table2):
    """复用共享模式提示事件，在原基准筛样之前接入分段新闻控制。"""
    if not PATTERN_CUE_PATH.is_file():
        raise FileNotFoundError(
            f"找不到共享模式提示事件表：{PATTERN_CUE_PATH}。"
            "请先通过原模式合并流程生成；本试跑不自动重建提示事件。"
        )

    source = load_module("_分段新闻试跑_模式合并", MERGE_DIR / "合并pattern_firms的数据.py")
    paths = source.build_pattern_paths(PROJECT_ROOT)
    print(f"复用共享模式提示事件：{PATTERN_CUE_PATH}", flush=True)

    # 一、从筛样前日度基础表合并自身事件、分段新闻、共享提示事件和收益。
    panel = source.load_pattern_daily_panel(daily_panel_path=paths["TRD_Dalyr_DGTW_path"])
    panel = source.add_own_earnings_announcement_flag(
        daily_panel=panel, report_path=paths["IAR_Rept_path"],
        event_date_mode=EVENT_DATE_MODE,
    )
    panel = source.add_own_ea_windows(daily_panel=panel, window=10)
    panel = source.merge_own_event_controls(
        daily_panel=panel,
        performance_forecast_path=paths["performance_forecast_controls_path"],
        other_event_path=paths["other_event_controls_path"],
    )
    panel = merge_news_controls_by_period(panel)
    # 读取共享事件的全部字段和完整历史，供后续提示标记及分组使用。
    panel = source.merge_pattern_cue_events(
        daily_panel=panel, cue_events_path=PATTERN_CUE_PATH,
    )
    panel = source.fill_pattern_cue_indicators(daily_panel=panel)
    panel = source.merge_adjusted_returns(daily_panel=panel, adjusted_return_path=paths["adjusted_return_path"])

    # 二、沿用原缩尾参照样本和两轮基准筛样，然后保存表2专用面板。
    panel = source.add_interference_groups(daily_panel=panel)
    panel = source.add_sue_cue_quantiles(daily_panel=panel)
    panel = source.add_recalled_firm_size_group(daily_panel=panel)
    winsor_limits, _ = source.compute_pattern_winsor_limits(daily_panel=panel, p=0.01)
    panel = source.prepare_pattern_baseline_inputs(daily_panel=panel, winsor_limits=winsor_limits)
    panel = source.filter_pattern_baseline_sample(daily_panel=panel)
    save_table2_panel(panel, PATTERN_PATH, table2)
    del panel
    gc.collect()


def build_full_panel(table2):
    """分块复用分段新闻版公共字段，重新匹配完整样本提示。"""
    import tempfile

    cue_path = FULL_CUE_PATH
    keys = ["StkcdRecall", "Trddt"]
    cue_col = "quarterLag1size90plus"

    if not PATTERN_PATH.is_file():
        raise FileNotFoundError(
            f"找不到分段新闻模式面板：{PATTERN_PATH}。"
            "请先使用 --stage 模式面板 生成。"
        )
    if not cue_path.is_file():
        raise FileNotFoundError(
            f"找不到共享完整提示事件表：{cue_path}。"
            "请先通过原全样本合并流程生成；本试跑不自动重建提示事件。"
        )
    print(f"复用共享完整提示事件：{cue_path}", flush=True)

    # 一、读取完整样本提示表，不再读取大型辅助表。
    cues = pd.read_csv(
        cue_path,
        usecols=["StkcdRecall", "date", cue_col],
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    ).rename(columns={"date": "Trddt"})

    cues["Trddt"] = (
        pd.to_datetime(cues["Trddt"], errors="raise")
        .dt.strftime("%Y-%m-%d")
    )

    if cues[keys].eq("").any().any():
        raise ValueError("完整样本提示表存在空键。")
    if cues.duplicated(keys).any():
        raise ValueError("完整样本提示表的公司与日期不唯一。")

    cues[cue_col] = pd.to_numeric(cues[cue_col], errors="raise")
    if not cues[cue_col].isin([0, 1]).all():
        raise ValueError("完整样本提示标记必须为0或1。")
    cues[cue_col] = cues[cue_col].astype("int8")

    # 二、逐批保留公共字段，替换提示标记。
    # 其余字段按文本读取，保留原文件中的数值和季度表示。
    reader = pd.read_csv(
        PATTERN_PATH,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
        chunksize=100_000,
    )

    work_dir = PROJECT_ROOT / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_cues = 0
    last_key = None

    with tempfile.TemporaryDirectory(
        prefix="分段新闻完整面板_", dir=work_dir
    ) as temp_dir:
        temp_path = Path(temp_dir) / "完整面板.csv"

        with temp_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as output:
            for chunk in reader:
                columns = chunk.columns.tolist()
                key_index = pd.MultiIndex.from_frame(chunk[keys])

                if (
                    chunk[keys].eq("").any().any()
                    or not key_index.is_unique
                    or not key_index.is_monotonic_increasing
                ):
                    raise ValueError("输入面板的公司日期键异常或未排序。")

                if last_key is not None and key_index[0] <= last_key:
                    raise ValueError("输入面板跨读取批次存在重复或乱序。")
                last_key = key_index[-1]

                before_rows = len(chunk)

                # 删除模式提示标记，换成完整样本提示标记。
                chunk = chunk.drop(columns=cue_col).merge(
                    cues,
                    on=keys,
                    how="left",
                    validate="one_to_one",
                    sort=False,
                )
                chunk[cue_col] = (
                    chunk[cue_col].fillna(0).astype("int8")
                )

                if len(chunk) != before_rows:
                    raise ValueError("合并前后行数不一致。")

                chunk[columns].to_csv(
                    output,
                    index=False,
                    header=(total_rows == 0),
                )

                total_rows += len(chunk)
                total_cues += int(chunk[cue_col].sum())
                print(f"已写出：{total_rows:,}行", flush=True)

        if total_rows == 0:
            raise ValueError("没有生成任何完整样本观测。")

        # 全部完成后才发布正式文件。
        temp_path.replace(FULL_PATH)

    print("完整样本面板已保存：", FULL_PATH)
    print("公司日行数：", total_rows)
    print("完整样本提示标记合计：", total_cues)

def estimate_table2(table2):
    """复用表2的两个收益窗口、六种规格及双向聚类，单独保存结果。"""
    for path in (PATTERN_PATH, FULL_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到分段新闻回归面板：{path}。"
                "请依次使用 --stage 模式面板、--stage 完整面板 生成面板，"
                "再使用 --stage 回归 估计；首次也可使用 --stage 全部。"
            )

    summary_tables = []
    coefficient_tables = []
    for path, sample_name, sample_label in (
        (PATTERN_PATH, "pattern", "A. 模式公司主样本"),
        (FULL_PATH, "full", "B. 完整样本"),
    ):
        panel = table2.load_table2_panel(path, sample_label)
        summary, coefficients = table2.run_table2_sample(
            data=panel, sample_name=sample_name, sample_label=sample_label,
        )
        summary_tables.append(summary)
        coefficient_tables.append(coefficients)
        del panel
        gc.collect()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for tables, filename in (
        (summary_tables, "表2_模型汇总.csv"),
        (coefficient_tables, "表2_系数明细.csv"),
    ):
        result = pd.concat(tables, ignore_index=True)
        result["新闻控制来源"] = NEWS_SOURCE_LABEL
        result["新闻切换日期"] = NEWS_SWITCH_DATE.strftime("%Y-%m-%d")
        result["回归起始年份"] = START_YEAR
        result["回归结束年份"] = END_YEAR
        result["新闻控制文件"] = (
            f"报刊：{NEWSPAPER_NEWS_PATH.relative_to(PROJECT_ROOT)}；"
            f"网络：{NETWORK_NEWS_PATH.relative_to(PROJECT_ROOT)}"
        )
        output_path = RESULT_DIR / filename
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"已保存分段新闻回归结果：{output_path}", flush=True)
        if filename == "表2_模型汇总.csv":
            print(result.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="2005—2025年表2独立试跑：2005—2009年报刊、2010—2025年网络新闻",
    )
    parser.add_argument(
        "--stage", choices=["全部", "模式面板", "完整面板", "回归"],
        default="全部", help="默认重建两类面板后估计回归；已有面板时可选择回归。",
    )
    args = parser.parse_args()
    if args.stage in ("全部", "模式面板"):
        for news_path in (NEWSPAPER_NEWS_PATH, NETWORK_NEWS_PATH):
            if not news_path.is_file():
                raise FileNotFoundError(f"找不到已生成的新闻控制变量表：{news_path}")

    table2 = load_module("_分段新闻试跑_表2估计", Path(__file__).with_name("估计回归.py"))
    # 仅设置本次加载实例的参数，不修改来源文件。
    table2.START_YEAR = START_YEAR
    table2.END_YEAR = END_YEAR
    table2.PATTERN_PATH = PATTERN_PATH
    table2.FULL_PATH = FULL_PATH

    print(f"本次运行阶段：{args.stage}", flush=True)
    print(f"新闻控制来源：{NEWS_SOURCE_LABEL}；回归期间：{START_YEAR}—{END_YEAR}年", flush=True)
    print(f"报刊新闻控制文件：{NEWSPAPER_NEWS_PATH}", flush=True)
    print(f"网络新闻控制文件：{NETWORK_NEWS_PATH}", flush=True)
    if args.stage in ("全部", "模式面板"):
        build_pattern_panel(table2)
    if args.stage in ("全部", "完整面板"):
        build_full_panel(table2)
    if args.stage in ("全部", "回归"):
        estimate_table2(table2)


if __name__ == "__main__":
    main()
