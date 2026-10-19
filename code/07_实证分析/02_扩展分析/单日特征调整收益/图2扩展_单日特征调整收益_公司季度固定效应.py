"""估计并绘制单日特征调整收益的公司季度固定效应版本，共51个点。

本文件是“图2扩展_单日特征调整收益_加入日期固定效应.py”的配套规格。
复用该脚本的数据读取、目标交易日匹配、共同缩尾和样本筛选函数，
但不调用其日期固定效应估计器或主函数。

因变量为被提示公司在目标交易日 t+h 的单日特征调整收益，h为0至50；
仅吸收被提示公司×年度季度固定效应，不吸收日期固定效应。
标准误仍按被提示公司和事件起点交易日期双向聚类，并采用有限样本修正。
仅按公司季度固定效应删除单例组；样本量可能与双固定效应版本不同。

默认使用中性基础面板和逐日可用样本；保留51日共同样本、旧累计筛选
样本和续算选项。结果与图片保存到同一单日收益结果目录，文件名包含
“公司季度固定效应”，不会覆盖加入日期固定效应版本。
仅在直接运行或调用主函数时读取研究数据、估计和生成结果文件。
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

# 相邻脚本只提供共同的数据准备函数；规格、估计、续算校验和绘图在本文件指定。
import 图2扩展_单日特征调整收益_加入日期固定效应 as data_source

from 图2图3扩展_公司季度固定效应回归公共模块 import (
    run_firm_quarter_fe_regression,
)

import matplotlib.pyplot as plt
from matplotlib import font_manager


HORIZONS = data_source.HORIZONS
ANALYSIS_VERSION = "单日特征调整收益_公司季度固定效应_v1"
DATE_FE_BASE_DEFINITION = "不适用（未加入日期固定效应）"
RESULT_COMPLETION_FIELDS = data_source.RESULT_COMPLETION_FIELDS


def result_metadata(sample_label: str, base_label: str) -> dict[str, str]:
    """统一生成保存、续算和最终校验使用的规格标记。"""
    return {
        "分析版本": ANALYSIS_VERSION,
        "被解释变量口径": data_source.OUTCOME_DEFINITION,
        "被解释变量来源": data_source.OUTCOME_SOURCE_DEFINITION,
        "目标日组合口径": data_source.TARGET_PORTFOLIO_DEFINITION,
        "目标日对齐口径": data_source.TARGET_ALIGNMENT_DEFINITION,
        "目标日缺失处理": data_source.TARGET_MISSING_DEFINITION,
        "样本口径": sample_label,
        "基础面板口径": base_label,
        "缩尾口径": data_source.WINSOR_DEFINITION,
        "日期固定效应": "否",
        "日期固定效应基准": DATE_FE_BASE_DEFINITION,
        "被提示公司×年度季度固定效应": "是",
        "聚类方式": data_source.CLUSTER_DEFINITION,
        "控制变量": "是",
        "结果单位": "百分点",
        "有限样本修正": "是",
    }


def resolve_output_paths(
    sample_mode: str,
    base_source: str,
) -> tuple[Path, Path]:
    """按固定效应和样本口径区分结果，避免覆盖日期固定效应版本。"""
    sample_label = data_source.SAMPLE_MODE_LABELS[sample_mode]
    base_short_label = {
        "neutral": "中性基础面板",
        "legacy": "旧累计筛选样本",
    }[base_source]
    stem = (
        "图2扩展_单日特征调整收益_公司季度固定效应_"
        f"{sample_label}_{base_short_label}"
    )
    return (
        data_source.RESULT_DIR / f"{stem}结果.csv",
        data_source.FIGURE_DIR / f"{stem}.png",
    )


def load_completed_rows(
    *,
    resume: bool,
    result_path: Path,
    sample_label: str,
    base_label: str,
    winsor_lower: float,
    winsor_upper: float,
) -> list[dict]:
    """只恢复本规格、同一样本和相同缩尾阈值的完整相对日结果。"""
    if not resume or not result_path.exists():
        return []

    existing = pd.read_csv(result_path, encoding="utf-8-sig")
    if existing.empty:
        return []
    if "窗口序号" not in existing.columns:
        raise ValueError("已有单日收益结果缺少“窗口序号”，无法继续估计。")

    window_numbers = pd.to_numeric(existing["窗口序号"], errors="coerce")
    valid_windows = window_numbers.isin(HORIZONS)
    if not valid_windows.all():
        raise ValueError("已有结果包含无效相对日，不能继续估计。")
    existing["窗口序号"] = window_numbers.astype(int)

    metadata = result_metadata(sample_label, base_label)
    for column in [*RESULT_COMPLETION_FIELDS, *metadata]:
        if column not in existing.columns:
            existing[column] = np.nan

    complete = existing[RESULT_COMPLETION_FIELDS].notna().all(axis=1)
    for column, expected in metadata.items():
        complete &= existing[column].astype(str).eq(expected)

    for column, expected in (
        ("缩尾下界", winsor_lower),
        ("缩尾上界", winsor_upper),
    ):
        complete &= np.isclose(
            pd.to_numeric(existing[column], errors="coerce"),
            expected,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=False,
        )

    incomplete = existing.loc[~complete, "窗口序号"].tolist()
    if incomplete:
        print(f"以下已有相对日的字段或口径不完整，将重新估计：{incomplete}")

    return (
        existing.loc[complete]
        .drop_duplicates("窗口序号", keep="last")
        .sort_values("窗口序号")
        .to_dict("records")
    )


def save_results(rows: Iterable[dict], result_path: Path) -> pd.DataFrame:
    """沿用逐日排序、临时文件替换和中文兼容编码的保存方式。"""
    return data_source.save_results(rows, result_path)


def validate_results(
    results: pd.DataFrame,
    *,
    sample_label: str,
    base_label: str,
) -> None:
    """绘图前核验51个单日结果及公司季度规格，防止混用日期版本。"""
    if len(results) != len(HORIZONS):
        raise ValueError("公司季度版本未包含全部51个相对交易日。")
    if results["窗口序号"].tolist() != HORIZONS:
        raise ValueError("公司季度版本的相对日不完整或顺序错误。")

    for column, expected in result_metadata(sample_label, base_label).items():
        if column not in results.columns:
            raise ValueError(f"单日收益结果缺少口径字段：{column}")
        if not results[column].astype(str).eq(expected).all():
            raise ValueError(f"单日收益结果的“{column}”标记不正确。")

    numeric_columns = [
        "线索系数", "聚类标准误", "置信区间下界", "置信区间上界",
        "t值", "p值", "回归样本量", "被提示公司聚类数", "日期聚类数",
        "删除单例组观测数", "解释变量矩阵条件数", "R方",
        "目标日有效收益观测数", "缩尾下界", "缩尾上界",
    ]
    numeric = results[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("单日收益结果包含缺失或非有限统计量。")
    if not results["聚类标准误"].gt(0).all():
        raise ValueError("单日收益结果包含非正标准误。")
    if not results["p值"].between(0, 1).all():
        raise ValueError("单日收益结果包含无效检验概率。")
    if not results["回归样本量"].gt(0).all():
        raise ValueError("单日收益结果包含无效样本量。")
    if not results[["被提示公司聚类数", "日期聚类数"]].ge(2).all().all():
        raise ValueError("单日收益结果的聚类数少于2。")

    interval_valid = (
        results["置信区间下界"].le(results["线索系数"])
        & results["线索系数"].le(results["置信区间上界"])
    )
    if not interval_valid.all():
        raise ValueError("单日收益结果包含无效置信区间。")
    if results["相对交易日"].tolist() != [f"t+{h}" for h in HORIZONS]:
        raise ValueError("结果表的相对交易日标签不正确。")
    if results["因变量"].tolist() != [
        data_source.OUTCOME_LABELS[h] for h in HORIZONS
    ]:
        raise ValueError("结果表的单日被解释变量名称不正确。")


def draw_single_day_figure(
    results: pd.DataFrame,
    output_path: Path,
    sample_label: str,
) -> None:
    """绘制公司季度规格的51个系数及逐点99%、95%、90%置信区间。"""
    plot_data = results.sort_values("窗口序号").reset_index(drop=True)
    font_path = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for spec in data_source.ERROR_BAR_SPECS:
        radius = spec["critical_value"] * plot_data["聚类标准误"]
        axis.vlines(
            plot_data["窗口序号"],
            plot_data["线索系数"] - radius,
            plot_data["线索系数"] + radius,
            colors=spec["color"],
            linewidth=spec["line_width"],
            label=spec["label"],
            zorder=spec["zorder"],
        )

    significant = (
        plot_data["线索系数"].abs()
        > 1.6448536269514722 * plot_data["聚类标准误"]
    )
    axis.scatter(
        plot_data["窗口序号"],
        plot_data["线索系数"],
        color=np.where(significant, "#d62728", "#1a1a1a"),
        s=16,
        zorder=4,
    )
    axis.scatter([], [], color="#d62728", s=16, label="红点：10%水平显著")
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_title(
        "事件当日及之后的单日特征调整收益动态\n"
        "固定效应：被提示公司×年度季度；不含交易日期固定效应\n"
        f"样本：{sample_label}",
        fontsize=11,
        pad=12,
    )
    axis.set_xlabel("相对交易日")
    axis.set_ylabel("提示变量系数（百分点）")
    ticks = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    axis.set_xticks(ticks)
    axis.set_xticklabels(["t" if tick == 0 else f"t+{tick}" for tick in ticks])
    axis.legend(loc="upper right", frameon=False, fontsize=9, handlelength=1.4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """保留与日期版本一致的样本选项和显式续算参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-mode",
        choices=sorted(data_source.SAMPLE_MODE_LABELS),
        default="available",
        help="available（逐日可用样本）；common51（51日完整共同样本）。",
    )
    parser.add_argument(
        "--base-source",
        choices=sorted(data_source.BASE_SOURCE_LABELS),
        default="neutral",
        help="neutral（中性基础面板）；legacy（旧累计收益筛选样本）。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="仅在输入和设置未变时，从同口径的已有完整结果继续估计。",
    )
    return parser.parse_args()


def main() -> None:
    """准备单日收益，逐日估计公司季度规格，并保存独立结果与图片。"""
    args = parse_args()
    sample_label = data_source.SAMPLE_MODE_LABELS[args.sample_mode]
    base_label = data_source.BASE_SOURCE_LABELS[args.base_source]
    base_path = data_source.resolve_base_path(args.base_source)
    result_path, output_path = resolve_output_paths(args.sample_mode, args.base_source)

    print("本次规格：仅被提示公司×年度季度固定效应；公司和事件起点日期双向聚类。")
    print(f"读取回归基础面板：{base_path}")
    base = data_source.read_base_panel(base_path)
    print(f"读取股票日度特征调整收益表：{data_source.DAILY_ADJUSTED_RETURN_PATH}")
    daily_lookup, winsor_lower, winsor_upper = data_source.read_daily_return_lookup()
    data_source.validate_shared_td_calendar(base, daily_lookup)
    print(
        "单日特征调整收益共同缩尾阈值："
        f"下界={winsor_lower:.10f}，上界={winsor_upper:.10f}"
    )

    if args.sample_mode == "common51":
        print("正在识别51个相对交易日均有效的共同样本……")
        common_mask = data_source.build_common_sample_mask(
            base, daily_lookup, horizons=HORIZONS,
        )
        removed = int((~common_mask).sum())
        base = base.loc[common_mask].reset_index(drop=True)
        print(f"共同样本筛选删除观测数：{removed:,}")
        if base.empty:
            raise ValueError("51日完整共同样本为空。")

    existing_rows = load_completed_rows(
        resume=args.resume,
        result_path=result_path,
        sample_label=sample_label,
        base_label=base_label,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
    )
    completed = {int(row["窗口序号"]) for row in existing_rows}
    pending = [h for h in HORIZONS if h not in completed]
    rows = existing_rows.copy()
    outcome_lookup = data_source.DailyOutcomeLookup(daily_lookup) if pending else None

    for position, horizon in enumerate(pending, start=1):
        print(
            "正在运行公司季度固定效应版单日收益回归："
            f"t+{horizon}（待估计 {position}/{len(pending)}）"
        )
        aligned = data_source.attach_daily_outcome(base, outcome_lookup, horizon)
        valid_outcome = aligned[data_source.OUTCOME_VALUE_COL].notna()
        valid_count = int(valid_outcome.sum())
        if valid_count == 0:
            raise ValueError(f"t+{horizon} 没有可用的单日特征调整收益率。")
        valid_dates = aligned.loc[valid_outcome, data_source.OUTCOME_DATE_COL]

        row = run_firm_quarter_fe_regression(
            aligned,
            horizon=horizon,
            outcome=data_source.OUTCOME_VALUE_COL,
            outcome_label=data_source.OUTCOME_LABELS[horizon],
            rhs_cols=data_source.RHS_COLS,
            cue_col=data_source.CUE_COL,
        )
        metadata = result_metadata(sample_label, base_label)
        # 先核对估计器真实返回的规格，再补充样本与收益来源说明。
        for column in (
            "日期固定效应", "日期固定效应基准",
            "被提示公司×年度季度固定效应", "聚类方式",
            "控制变量", "结果单位", "有限样本修正",
        ):
            if row.get(column) != metadata[column]:
                raise ValueError(f"估计器返回的“{column}”与当前规格不一致。")
        row.update(metadata)
        row.update({
            "缩尾下界": winsor_lower,
            "缩尾上界": winsor_upper,
            "目标日有效收益观测数": valid_count,
            "目标日日期起点": valid_dates.min().strftime("%Y-%m-%d"),
            "目标日日期终点": valid_dates.max().strftime("%Y-%m-%d"),
        })

        rows = [saved for saved in rows if int(saved["窗口序号"]) != horizon]
        rows.append(row)
        results = save_results(rows, result_path)
        print(
            f"t+{horizon} 完成：系数={row['线索系数']:.6f}，"
            f"标准误={row['聚类标准误']:.6f}，样本量={row['回归样本量']:,}"
        )
        del aligned, valid_outcome, valid_dates
        gc.collect()

    if not pending:
        print("已有结果包含全部51个相对交易日，无需重新估计。")
        results = save_results(existing_rows, result_path)

    validate_results(results, sample_label=sample_label, base_label=base_label)
    draw_single_day_figure(results, output_path, sample_label)
    print(f"公司季度固定效应版单日收益结果已保存：{result_path}")
    print(f"公司季度固定效应版单日收益图已保存：{output_path}")


if __name__ == "__main__":
    main()
