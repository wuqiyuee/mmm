"""生成当日至第50个交易日的单日散户订单不平衡动态图。

仅吸收被提示公司×年度季度固定效应，不加入日期固定效应。
沿用相邻日期版本的基础面板、27项控制、全源表1%/99%缩尾、
精确目标市场交易日匹配及公司和事件起点日期双向聚类口径。
目标日缺失不填零；每个相对日分别删除公司季度单例组。

默认使用2005—2025年中性基础面板和逐日可用样本。可使用
--sample-mode common50 选择当日至第50日均有效的共同样本，
--base-source legacy 选择旧累计筛选样本，--resume 继续同口径结果。
分块估计只降低内存使用，不改变固定效应和聚类方差公式。
直接运行才读取研究数据并输出独立结果，不覆盖日期固定效应版本。
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import 图3扩展_单日散户订单不平衡_加入日期固定效应 as data_source
from 单日订单不平衡_公司季度分块估计 import QuarterFEEstimator

import matplotlib.pyplot as plt
from matplotlib import font_manager


HORIZONS = data_source.HORIZONS
ANALYSIS_VERSION = "单日散户订单不平衡_公司季度固定效应_v1"


def result_metadata(*, sample_label, base_label, media_cols) -> dict:
    """保留数据口径，显式区分本次固定效应规格。"""
    metadata = data_source.result_metadata(
        sample_label=sample_label, base_label=base_label, media_cols=media_cols,
    )
    metadata.update({
        "分析版本": ANALYSIS_VERSION,
        "日期固定效应": "否",
        "日期固定效应基准": "不适用（未加入日期固定效应）",
    })
    return metadata


def resolve_output_paths(sample_mode, base_source) -> tuple[Path, Path]:
    """文件名中保留固定效应、样本和基础面板口径。"""
    sample_label = data_source.SAMPLE_MODE_LABELS[sample_mode]
    base_short = {"neutral": "中性基础面板", "legacy": "旧累计筛选样本"}[base_source]
    stem = (
        "图3扩展_单日散户订单不平衡_公司季度固定效应_"
        f"{sample_label}_{base_short}"
    )
    return (
        data_source.RESULT_DIR / f"{stem}结果.csv",
        data_source.FIGURE_DIR / f"{stem}.png",
    )


def load_completed_rows(*, resume, result_path, metadata, lower, upper) -> list[dict]:
    """续算只复用字段完整、规格及缩尾阈值相同的结果。"""
    if not resume or not result_path.exists():
        return []
    existing = pd.read_csv(result_path, encoding="utf-8-sig")
    if existing.empty:
        return []
    if "窗口序号" not in existing:
        raise ValueError("已有结果缺少窗口序号。")
    windows = pd.to_numeric(existing["窗口序号"], errors="coerce")
    if not windows.isin(HORIZONS).all():
        raise ValueError("已有结果包含无效相对交易日。")
    existing["窗口序号"] = windows.astype(int)
    required = data_source.RESULT_COMPLETION_FIELDS
    for column in [*required, *metadata]:
        if column not in existing:
            existing[column] = np.nan
    complete = existing[required].notna().all(axis=1)
    for column, expected in metadata.items():
        complete &= existing[column].astype(str).eq(str(expected))
    for column, expected in (
        ("缩尾下界（百分点）", lower), ("缩尾上界（百分点）", upper),
    ):
        complete &= np.isclose(
            pd.to_numeric(existing[column], errors="coerce"), expected,
            rtol=1e-10, atol=1e-12, equal_nan=False,
        )
    if (~complete).any():
        print("将重估字段或口径不完整的相对日：", existing.loc[~complete, "窗口序号"].tolist())
    return (
        existing.loc[complete].drop_duplicates("窗口序号", keep="last")
        .sort_values("窗口序号").to_dict("records")
    )


def validate_results(results: pd.DataFrame, metadata: dict) -> None:
    """核验全部51个相对日、真实估计规格及主要统计量。"""
    if results["窗口序号"].tolist() != HORIZONS:
        raise ValueError("结果没有完整覆盖当日至第50个交易日。")
    for column, expected in metadata.items():
        if column not in results or not results[column].astype(str).eq(str(expected)).all():
            raise ValueError(f"结果的“{column}”口径不一致。")
    if results[data_source.RESULT_COMPLETION_FIELDS].isna().any().any():
        raise ValueError("结果存在缺失的必需统计量。")
    numeric_cols = [
        "线索系数", "聚类标准误", "置信区间下界", "置信区间上界",
        "t值", "p值", "回归样本量", "被提示公司聚类数", "日期聚类数",
        "删除单例组观测数", "解释变量矩阵条件数", "R方",
        "目标日有效OIB观测数", "缩尾下界（百分点）", "缩尾上界（百分点）",
    ]
    if not np.isfinite(results[numeric_cols].to_numpy(dtype=np.float64)).all():
        raise ValueError("结果包含非有限统计量。")
    if not results["聚类标准误"].gt(0).all() or not results["p值"].between(0, 1).all():
        raise ValueError("标准误或检验概率无效。")
    if not results[["被提示公司聚类数", "日期聚类数"]].ge(2).all().all():
        raise ValueError("聚类组数不足。")
    if not results["回归样本量"].eq(
        results["目标日有效OIB观测数"] - results["删除单例组观测数"]
    ).all():
        raise ValueError("回归样本量与单例删除记录不一致。")
    for column, expected in (
        ("相对交易日", [data_source.format_relative_day(h) for h in HORIZONS]),
        ("因变量", [data_source.OUTCOME_LABELS[h] for h in HORIZONS]),
    ):
        if results[column].tolist() != expected:
            raise ValueError(f"结果的{column}标签不正确。")
    if not np.allclose(results["置信区间下界"], results["线索系数"] - 1.96 * results["聚类标准误"]):
        raise ValueError("95%置信区间下界不正确。")
    if not np.allclose(results["置信区间上界"], results["线索系数"] + 1.96 * results["聚类标准误"]):
        raise ValueError("95%置信区间上界不正确。")


def draw_single_day_figure(results, output_path, sample_label) -> None:
    """绘制逐点置信区间，并在图上列明固定效应和聚类方式。"""
    plot_data = results.sort_values("窗口序号")
    font_path = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=str(font_path)).get_name()
        ]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(10, 6))
    for spec in data_source.ERROR_BAR_SPECS:
        radius = spec["critical_value"] * plot_data["聚类标准误"]
        axis.vlines(
            plot_data["窗口序号"], plot_data["线索系数"] - radius,
            plot_data["线索系数"] + radius, colors=spec["color"],
            linewidth=spec["line_width"], label=spec["label"], zorder=spec["zorder"],
        )
    significant = plot_data["线索系数"].abs() > 1.6448536269514722 * plot_data["聚类标准误"]
    axis.scatter(
        plot_data["窗口序号"], plot_data["线索系数"],
        color=np.where(significant, "#d62728", "#1a1a1a"), s=16, zorder=4,
    )
    axis.scatter([], [], color="#d62728", s=16, label="红点：10%水平显著")
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_title(
        "单日散户订单不平衡动态\n"
        "固定效应：被提示公司×年度季度（不含日期固定效应）\n"
        f"样本：2005—2025年，{sample_label}", fontsize=11, pad=12,
    )
    axis.set_xlabel("相对交易日")
    axis.set_ylabel("提示变量系数（百分点）")
    ticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    axis.set_xticks(ticks)
    axis.set_xticklabels([data_source.format_relative_day(h) for h in ticks])
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
        frameon=False, fontsize=9, handlelength=1.4,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")
    figure.text(
        0.5, 0.015, "控制变量：27项；标准误：被提示公司与事件起点交易日期双向聚类。",
        ha="center", fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """复用共同数据准备，执行独立的公司季度规格。"""
    args = data_source.parse_args()
    sample_label = data_source.SAMPLE_MODE_LABELS[args.sample_mode]
    base_label = data_source.BASE_SOURCE_LABELS[args.base_source]
    base_path = data_source.resolve_base_path(args.base_source)
    result_path, output_path = resolve_output_paths(args.sample_mode, args.base_source)
    print("本次规格：仅被提示公司×年度季度固定效应，不含日期固定效应。")
    print(f"读取基础面板：{base_path}")
    base, media_cols, rhs_cols = data_source.read_base_panel(base_path)
    if len(rhs_cols) != 28:
        raise ValueError("预期为提示变量与27项控制变量。")
    print(f"基础面板行数：{len(base):,}；媒体控制变量：{media_cols}")
    print(f"读取单日订单不平衡源表：{data_source.DAILY_OIB_PATH}")
    daily_lookup, lower, upper = data_source.read_daily_oib_lookup()
    data_source.validate_shared_td_calendar(base, daily_lookup)
    print(f"共同缩尾阈值（百分点）：下界={lower:.10f}，上界={upper:.10f}")
    if args.sample_mode == "common50":
        common = data_source.build_common_sample_mask(base, daily_lookup, horizons=HORIZONS)
        print(f"共同样本筛选删除观测数：{int((~common).sum()):,}")
        base = base.loc[common].reset_index(drop=True)
        if base.empty:
            raise ValueError("当日至第50个交易日共同样本为空。")
    metadata = result_metadata(sample_label=sample_label, base_label=base_label, media_cols=media_cols)
    rows = load_completed_rows(
        resume=args.resume, result_path=result_path, metadata=metadata, lower=lower, upper=upper,
    )
    completed = {int(row["窗口序号"]) for row in rows}
    pending = [h for h in HORIZONS if h not in completed]
    if pending:
        lookup = data_source.DailyOutcomeLookup(daily_lookup)
        estimator = QuarterFEEstimator(base, rhs_cols, data_source.CUE_COL)
    for position, horizon in enumerate(pending, start=1):
        label = data_source.format_relative_day(horizon)
        print(f"正在估计{label}（待估计{position}/{len(pending)}）")
        aligned = data_source.attach_daily_outcome(base, lookup, horizon)
        valid = aligned[data_source.OUTCOME_VALUE_COL].notna()
        valid_count = int(valid.sum())
        if not valid_count:
            raise ValueError(f"{label}没有可用单日订单不平衡。")
        dates = aligned.loc[valid, data_source.OUTCOME_DATE_COL]
        row = estimator.fit(
            aligned, horizon=horizon, outcome=data_source.OUTCOME_VALUE_COL,
            outcome_label=data_source.OUTCOME_LABELS[horizon],
        )
        for column in (
            "日期固定效应", "日期固定效应基准", "被提示公司×年度季度固定效应",
            "聚类方式", "控制变量", "结果单位", "有限样本修正",
        ):
            if row.get(column) != metadata[column]:
                raise ValueError(f"估计器返回的{column}与规格不符。")
        row.update({
            **metadata, "相对交易日": label,
            "缩尾下界（百分点）": lower, "缩尾上界（百分点）": upper,
            "目标日有效OIB观测数": valid_count,
            "目标日日期起点": dates.min().strftime("%Y-%m-%d"),
            "目标日日期终点": dates.max().strftime("%Y-%m-%d"),
        })
        rows = [saved for saved in rows if int(saved["窗口序号"]) != horizon]
        rows.append(row)
        results = data_source.save_results(rows, result_path)
        print(
            f"{label}完成：系数={row['线索系数']:.6f}，标准误={row['聚类标准误']:.6f}，"
            f"样本量={row['回归样本量']:,}"
        )
        del aligned, valid, dates
        gc.collect()
    if not pending:
        results = data_source.save_results(rows, result_path)
    validate_results(results, metadata)
    draw_single_day_figure(results, output_path, sample_label)
    print(f"公司季度规格结果：{result_path}")
    print(f"公司季度规格图片：{output_path}")


if __name__ == "__main__":
    main()
