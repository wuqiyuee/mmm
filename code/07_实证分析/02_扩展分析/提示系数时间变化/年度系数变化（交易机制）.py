"""按当前收益主回归口径，逐年估计散户订单失衡的提示系数。

历史沪深300提示、季度主样本（含合格填充日起点）和27项控制均复用
当前主回归入口；新闻控制使用分段数行者口径，不读取旧市值90分位面板。
因变量为从当天至未来第h个市场交易日、共h+1日的平均成交量订单失衡。
先在日期范围内的当前主口径全体季度公司日观测上左合并并统一双侧1%缩尾，
再筛选主回归共同资格和订单失衡有效观测，随后分年并递归删除单例。
每年吸收被提示公司×年度季度、交易日期固定效应，采用公司和日期双向聚类；
聚类协方差有限样本修正、实际固定效应秩和残差自由度与当前收益主回归一致。
源订单失衡已经乘100，展示系数、标准误和置信区间时不再乘100。
每次直接运行依次处理[t,t+1]至[t,t+10]共10个窗口，读取数据、估计并分别保存年度系数图。
图片文件名仅包含分析类型和窗口，不附加置信水平。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel.utility import AbsorbingEffectError


# 1. 导入同目录的统一样本与估计入口，兼容从其他目录调用本脚本。
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
from 提示系数回归公共模块 import (
    CONTROL_COLUMNS,
    CUE_COLUMN,
    SPEC_DESCRIPTION,
    SPEC_ID,
    estimate_main_specification,
    load_main_sample,
    print_progress,
)


# 2. 集中设置路径、窗口、年份和显著性水平。
root = Path(__file__).resolve().parents[4]
image_dir = root / "图表结果/02_扩展分析/提示系数时间变化"

horizons = range(1, 11)
start_year = 2005
end_year = 2025
significance_level = 0.10
# 设为 True（真）时显示完整年度结果表；默认仅报告进度与汇总。
show_details = False

confidence_level = 1 - significance_level
cue = CUE_COLUMN
control_cols = CONTROL_COLUMNS


def run_one_window(horizon):
    if not isinstance(horizon, int) or not 1 <= horizon <= 50:
        raise ValueError("窗口终点必须为1—50的整数；所选窗口还必须在订单失衡源表中存在。")
    if start_year > end_year:
        raise ValueError("起始年份不能晚于结束年份。")
    if not 0 < significance_level < 1:
        raise ValueError("显著性水平必须介于0和1之间。")

    raw_dependent_variable = f"OIBvol_avg_t_tp{horizon}"
    dependent_variable = f"{raw_dependent_variable}_w"
    window_label = f"[t,t+{horizon}]"
    years = list(range(start_year, end_year + 1))
    # 3. 统一读取当前主样本、合并并缩尾；不按年度分别计算缩尾界限。
    data = load_main_sample(
        horizon,
        f"{start_year}-01-01",
        f"{end_year}-12-31",
        outcome_type="交易机制",
    )
    year_parts = {
        int(year): part.copy()
        for year, part in data.groupby(data["Trddt"].dt.year, sort=False)
    }
    del data

    # 4. 各年采用当前收益主回归的双固定效应、双向聚类及精确自由度。
    annual_rows = []
    print_progress("逐年估计", 0, len(years))
    for year_index, year in enumerate(years, 1):
        row = {
            "年份": year,
            "观测数": 0,
            "公司数": 0,
            "交易日期数": 0,
            "提示观测数": 0,
            "提示日期数": 0,
            "删除单例观测数": 0,
            "保留控制变量数": 0,
            "系数_百分点": np.nan,
            "标准误_百分点": np.nan,
            "置信下限_百分点": np.nan,
            "置信上限_百分点": np.nan,
            "显著性概率": np.nan,
            "达到显著性阈值": np.nan,
            "状态": "没有有效样本",
        }
        annual_rows.append(row)
        reg = year_parts.pop(year, None)
        if reg is None or reg.empty:
            print_progress("逐年估计", year_index, len(years), detail=f"{year}年")
            continue

        try:
            # 订单失衡源值已为百分点，换算倍数为1。
            row.update(estimate_main_specification(
                reg,
                dependent_variable,
                window_label,
                significance_level,
                scale=1,
            ))
        except (ValueError, np.linalg.LinAlgError, AbsorbingEffectError) as error:
            row["状态"] = f"未成功：{error}"
            print(f"{year}年未成功：{error}")
        print_progress("逐年估计", year_index, len(years), detail=f"{year}年")

    # 5. 汇总年度结果；没有成功年份时保留失败信息并跳过绘图。
    annual_results = pd.DataFrame(annual_rows).sort_values("年份").reset_index(drop=True)
    if show_details:
        print(f"\n{start_year}—{end_year}年年度回归结果：")
        print(annual_results.to_string(index=False))
    success = annual_results["状态"].eq("成功")
    success_count = int(success.sum())
    print(f"年度估计汇总：成功{success_count}年，未成功{len(years) - success_count}年。")
    plot_data = annual_results.loc[success].copy()
    if plot_data.empty:
        print("没有成功估计的年度结果，无法绘制系数图。")
        return annual_results

    # 6. 保存当前窗口的年度系数图，文件名仅包含分析类型和窗口。
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    error_lower = plot_data["系数_百分点"] - plot_data["置信下限_百分点"]
    error_upper = plot_data["置信上限_百分点"] - plot_data["系数_百分点"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(
        plot_data["年份"],
        plot_data["系数_百分点"],
        yerr=np.vstack([error_lower, error_upper]),
        fmt="o",
        color="#2166AC",
        ecolor="#8FAFCB",
        markersize=5,
        elinewidth=1.2,
        capsize=3,
        label=f"年度系数及{confidence_level:.0%}双向聚类置信区间",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(years)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlabel("年份")
    ax.set_ylabel("提示系数（百分点）")
    ax.set_title(f"{window_label}平均成交量订单失衡：年度提示系数变化")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    image_dir.mkdir(parents=True, exist_ok=True)
    figure_path = image_dir / (
        "年度_交易机制_"
        f"窗口{window_label}.png"
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("图片已保存：", figure_path)
    return annual_results


def main():
    """依次处理10个窗口，只保留各窗口的年度结果，避免累积完整回归样本。"""
    print(f"主回归口径：{SPEC_DESCRIPTION}")
    print(
        f"回归年份：{start_year}—{end_year}年；置信水平：{confidence_level:.0%}；"
        f"控制变量：{len(control_cols)}项；平均成交量订单失衡窗口：共{len(horizons)}个。"
    )
    annual_results_by_horizon = {}
    for window_index, horizon in enumerate(horizons, 1):
        print(
            f"\n窗口{window_index}/{len(horizons)}：[t,t+{horizon}]；"
            f"已完成窗口：{window_index - 1}/{len(horizons)}"
            f"（{(window_index - 1) / len(horizons):.0%}）",
            flush=True,
        )
        annual_results_by_horizon[horizon] = run_one_window(horizon)
        print(
            f"已完成窗口：{window_index}/{len(horizons)}"
            f"（{window_index / len(horizons):.0%}）",
            flush=True,
        )
    return annual_results_by_horizon


if __name__ == "__main__":
    annual_results_by_horizon = main()
