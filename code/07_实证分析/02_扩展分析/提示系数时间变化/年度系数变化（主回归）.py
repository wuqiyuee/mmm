"""逐年估计累计特征调整收益的提示系数，并保存年度系数图。

读取与当前表2相同的历史沪深300提示主样本、27项控制及既有缩尾收益，
保留合格填充日起点，使用分段数行者新闻控制，不回退到旧最终面板。
每年分别吸收被提示公司×年度季度、交易日期固定效应，递归删除单例；
双向聚类修正、固定效应实际秩和残差自由度均与表2第（6）列一致。
源收益以小数存储，展示系数、标准误和置信区间时乘100，单位为百分点。
每次运行依次处理[t,t+1]至[t,t+10]共10个窗口，报告完成进度并保存年度系数图。
图片文件名仅包含分析类型和窗口，不附加置信水平。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel.utility import AbsorbingEffectError


# 将本目录显式加入模块搜索路径，兼容由其他入口加载本脚本。
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
from 提示系数回归公共模块 import (
    CONTROL_COLUMNS, CUE_COLUMN, SPEC_DESCRIPTION, SPEC_ID,
    estimate_main_specification, load_main_sample, print_progress,
)


# 1. 集中设置路径、窗口、年份和显著性水平
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

# 2. 控制变量的组成、顺序和数据来源均与当前主回归共用。
control_cols = CONTROL_COLUMNS


def run_one_window(horizon):
    if not isinstance(horizon, int) or not 1 <= horizon <= 50:
        raise ValueError("窗口终点必须为1—50的整数。")
    if start_year > end_year:
        raise ValueError("起始年份不能晚于结束年份。")
    if not 0 < significance_level < 1:
        raise ValueError("显著性水平必须介于0和1之间。")

    dependent_variable = f"chAdjreturn_t_tp{horizon}_Recall_w"
    window_label = f"[t,t+{horizon}]"
    years = list(range(start_year, end_year + 1))

    # 3. 从当前主回归共同入口读取样本，再按年份划分。
    data = load_main_sample(
        horizon=horizon,
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
        outcome_type="收益",
    )
    year_parts = {
        int(year): part.copy()
        for year, part in data.groupby(data["Trddt"].dt.year, sort=False)
    }
    del data

    # 4. 逐年整理固定效应、递归删除单例并回归
    annual_rows = []
    print_progress("逐年估计", 0, len(years))
    for year_index, year in enumerate(years, 1):
        # 每个年份先保留完整结果字段，失败年份也有明确的缺失结果。
        row = {
            "年份": year,
            "口径标识": SPEC_ID,
            "提示变量": cue,
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
            "省略控制变量": [],
            "残差自由度": np.nan,
            "固定效应实际秩": np.nan,
            "状态": "没有有效样本",
        }
        annual_rows.append(row)
        reg = year_parts.pop(year, None)
        if reg is None or reg.empty:
            print_progress("逐年估计", year_index, len(years), detail=f"{year}年")
            continue

        try:
            # 5. 复用当前主回归完整规格和推断，收益展示时统一乘100。
            row.update(estimate_main_specification(
                data=reg,
                dependent_variable=dependent_variable,
                window_label=window_label,
                significance_level=significance_level,
                scale=100,
            ))
        except (ValueError, np.linalg.LinAlgError, AbsorbingEffectError) as error:
            row["状态"] = f"未成功：{error}"
            print(f"{year}年未成功：{error}")
        finally:
            del reg
        print_progress("逐年估计", year_index, len(years), detail=f"{year}年")

    # 6. 汇总年度结果；所有年份均失败时给出提示
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

    # 7. 保存当前窗口的年度系数图，文件名仅包含分析类型和窗口。
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
    ax.set_title(f"{window_label}累计特征调整收益：年度提示系数变化")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    image_dir.mkdir(parents=True, exist_ok=True)
    figure_path = image_dir / (
        "年度_主回归_"
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
        f"控制变量：{len(control_cols)}项；累计特征调整收益窗口：共{len(horizons)}个。"
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
