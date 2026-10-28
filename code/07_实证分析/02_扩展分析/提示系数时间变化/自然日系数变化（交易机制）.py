"""联合估计每日订单失衡提示系数，使用当前收益主回归的完整规格。

所有日期一起估计，每天保留独立的提示系数，27项控制的斜率全期共用。
吸收被提示公司×年度季度和日期固定效应，按公司、日期双向聚类；
提示、样本及分段新闻控制复用当前主回归入口，保留合格填充日起点。
订单失衡先与范围内主口径公司日全样本左合并并统一双侧1%缩尾，
再筛选共同资格及本次因变量有效值；不填补缺失，不按日或年度重新缩尾。
源订单失衡已是百分点，展示时不再乘100。
分季度计算联合回归，不构造全部观测×全部日期的稠密矩阵。
默认依次估计[t,t+1]至[t,t+10]，每个窗口分别保存一张系数图。
仅直接运行时读取数据、估计和保存图片；导入不执行这些步骤。
"""

from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
from 提示系数回归公共模块 import (
    CONTROL_COLUMNS, CUE_COLUMN, SPEC_DESCRIPTION, load_main_sample,
)
from 提示系数联合日度估计 import estimate_joint_daily


# 一、参数：窗口终点表示包含当天及随后相应数量市场交易日。
root = Path(__file__).resolve().parents[4]
image_dir = root / "图表结果/02_扩展分析/提示系数时间变化"
horizons = range(1, 11)
cue = CUE_COLUMN
control_cols = CONTROL_COLUMNS
start_date = "2005-01-01"
end_date = "2025-12-31"
significance_level = 0.10
confidence_level = 1 - significance_level
show_details = False  # 改为True时打印完整日度结果表。


def plot_daily_results(daily_results, window_label):
    """每天一个提示系数；置信区间来自同一个联合回归。"""
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_data = daily_results.loc[
        np.isfinite(daily_results["系数_百分点"])
    ].sort_values("日期")
    if plot_data.empty:
        raise ValueError("没有可估计的日期提示系数，无法绘图。")
    valid_ci = np.isfinite(plot_data[
        ["置信下限_百分点", "置信上限_百分点"]
    ]).all(axis=1)
    interval_data = plot_data.loc[valid_ci]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.vlines(
        interval_data["日期"], interval_data["置信下限_百分点"],
        interval_data["置信上限_百分点"], color="#AAB2BD", linewidth=0.8,
        alpha=0.7, label=f"{confidence_level:.0%}双向聚类置信区间",
    )
    ax.scatter(
        plot_data["日期"], plot_data["系数_百分点"],
        color="#2166AC", s=22, zorder=3, label="联合回归的每日提示系数",
    )
    if (~valid_ci).any():
        ax.scatter(
            plot_data.loc[~valid_ci, "日期"],
            plot_data.loc[~valid_ci, "系数_百分点"], color="#D97706",
            marker="^", s=42, zorder=4, label="聚类方差无效，仅展示系数",
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", which="major", labelrotation=45, length=6)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.set_xlabel("自然日期")
    ax.set_ylabel("提示系数（百分点）")
    ax.set_title(f"{window_label}平均成交量订单失衡：联合估计的每日提示系数")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def run_one_window(horizon):
    """估计一个订单失衡窗口，并保存该窗口的每日提示系数图。"""
    raw_dependent_variable = f"OIBvol_avg_t_tp{horizon}"
    dependent_variable = f"{raw_dependent_variable}_w"
    window_label = f"[t,t+{horizon}]"
    # 二、复用当前主样本；无提示日仍参与固定效应和共用控制系数估计。
    if not 0 < significance_level < 1:
        raise ValueError("显著性水平必须介于0和1之间。")
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("起始日期不得晚于结束日期。")
    data = load_main_sample(horizon, start_date, end_date, outcome_type="交易机制")

    # 三、联合估计、递归删单例、计算主回归口径的双向聚类推断。
    daily_counts, regression_data, daily_results = estimate_joint_daily(
        data, dependent_variable, significance_level, scale=1,
    )
    del data, regression_data
    if show_details:
        print("\n每日样本统计（删除固定效应单例后）：")
        print(daily_counts.to_string())
        print("\n每日联合回归结果：")
        print(daily_results.to_string(index=False))

    # 四、文件名仅保留分析类型和窗口；只有运行本脚本时才保存图片。
    if np.isfinite(daily_results["系数_百分点"]).any():
        fig, ax = plot_daily_results(daily_results, window_label)
        image_dir.mkdir(parents=True, exist_ok=True)
        figure_path = image_dir / (
            f"自然日_交易机制_窗口{window_label}.png"
        )
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("图片已保存：", figure_path)
    else:
        print("没有可估计的日期提示系数，跳过图片保存。")
    return daily_results


def main():
    """依次处理10个窗口，只保留各窗口的日度结果，避免累积完整回归样本。"""
    print("当前主回归口径：", SPEC_DESCRIPTION)
    print(f"提示变量：{cue}；控制变量{len(control_cols)}项；置信水平{confidence_level:.0%}")
    print("所有日期联合估计，每日提示系数不同，控制变量系数全期共用。")
    daily_results_by_horizon = {}
    total_windows = len(horizons)
    for number, horizon in enumerate(horizons, start=1):
        print(
            f"\n窗口{number}/{total_windows}：[t,t+{horizon}]；"
            f"已完成窗口{number - 1}/{total_windows}"
            f"（{(number - 1) / total_windows:.0%}）", flush=True,
        )
        daily_results_by_horizon[horizon] = run_one_window(horizon)
        print(
            f"已完成窗口{number}/{total_windows}（{number / total_windows:.0%}）",
            flush=True,
        )
    return daily_results_by_horizon


if __name__ == "__main__":
    daily_results_by_horizon = main()
