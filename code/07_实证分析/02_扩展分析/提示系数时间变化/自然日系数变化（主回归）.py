"""逐日估计主回归的提示系数，打印结果并保存图片。

一行样本为“被提示公司×交易日期”；每个日期单独估计横截面回归。
因变量使用最终面板已有的缩尾累计特征调整收益，展示时转为百分点。
控制变量共27个；无固定效应，使用异方差稳健标准误和双侧检验。
某组仅1条观测时，图中用橙色三角展示系数，不展示其置信区间。
默认显著性水平10%，对应90%置信区间。直接运行时才读取数据、估计和保存图片。
图片文件名包含自然日、主回归、窗口和置信水平。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 一、参数区：以后主要修改这里
# ============================================================

root = Path("/Users/qiyuewu/Desktop/mmm复现数据包")
input_path = root / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"
image_dir = root / "图表结果/02_扩展分析/提示系数时间变化"

# 窗口终点5表示[t,t+5]，包含当天和随后5个交易日。
horizon = 4
dependent_variable = f"chAdjreturn_t_tp{horizon}_Recall_w"
window_label = f"[t,t+{horizon}]"
cue = "quarterLag1size90plus"

start_date = "2005-01-01"
end_date = "2025-12-31"
chunk_size = 300_000

# 双侧显著性水平；0.10对应90%置信区间，0.05对应95%置信区间。
significance_level = 0.10
confidence_level = 1 - significance_level

# 被提示公司自身盈余公告：共21个控制变量。
own_ea_cols = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)

# 其他公告与媒体：共6个控制变量。
other_control_cols = [
    "ownOtherEventER",
    "ownPerformanceForecastBrief",
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]
control_cols = own_ea_cols + other_control_cols
needed_cols = ["StkcdRecall", "Trddt", dependent_variable, cue] + control_cols
numeric_cols = [dependent_variable, cue] + control_cols


# ============================================================
# 二、检查字段并分块读取
# ============================================================

def check_input():
    """检查参数与字段，仅显示预览，不写出文件。"""
    if not isinstance(horizon, int) or not 1 <= horizon <= 50:
        raise ValueError("窗口终点必须为1—50的整数。")
    if not 0 < significance_level < 1:
        raise ValueError("显著性水平必须在0和1之间。")
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("起始日期不得晚于结束日期。")

    header_cols = pd.read_csv(
        input_path, nrows=0, encoding="utf-8-sig"
    ).columns
    missing_cols = [col for col in needed_cols if col not in header_cols]
    if missing_cols:
        raise ValueError(f"最终面板缺少以下字段：{missing_cols}")

    preview = pd.read_csv(
        input_path,
        usecols=["StkcdRecall", "Trddt", dependent_variable, cue],
        nrows=5,
        dtype={"StkcdRecall": "string"},
        encoding="utf-8-sig",
    )
    print("前5行数据：")
    print(preview)
    print("因变量：", dependent_variable)
    print("控制变量数量：", len(control_cols))
    print(f"双侧显著性水平：{significance_level:.0%}")
    print(f"置信水平：{confidence_level:.0%}")


def read_chunks(stage):
    """按同一规则读取两次，避免统计与提取的样本口径不同。"""
    read_types = {col: "float64" for col in numeric_cols}
    read_types.update({"StkcdRecall": "string", "Trddt": "string"})
    with pd.read_csv(
        input_path,
        usecols=needed_cols,
        dtype=read_types,
        chunksize=chunk_size,
        encoding="utf-8-sig",
    ) as reader:
        for number, chunk in enumerate(reader, start=1):
            print(f"{stage}：第 {number} 个数据块……", flush=True)
            chunk["Trddt"] = pd.to_datetime(chunk["Trddt"], errors="raise")
            chunk["StkcdRecall"] = (
                chunk["StkcdRecall"].str.strip().replace("", pd.NA).str.zfill(6)
            )
            yield chunk


def get_valid_mask(data):
    """公司代码非缺失，因变量、提示变量和全部控制变量均为有限值。"""
    return (
        data["StkcdRecall"].notna()
        & np.isfinite(data[numeric_cols].to_numpy()).all(axis=1)
    )


# ============================================================
# 三、统计每日样本与年度分布
# ============================================================

def count_daily_samples():
    daily_parts = []
    for chunk in read_chunks("统计每日样本"):
        chunk = chunk.loc[chunk["Trddt"].between(start_date, end_date)].copy()
        if chunk.empty:
            continue
        if not chunk[cue].dropna().isin([0, 1]).all():
            raise ValueError("提示变量中存在0、1以外的取值。")

        valid = get_valid_mask(chunk)
        chunk["有效观测数"] = valid
        chunk["有效提示观测数"] = valid & chunk[cue].eq(1)
        chunk["有效未提示观测数"] = valid & chunk[cue].eq(0)
        daily_parts.append(chunk.groupby("Trddt").agg(
            面板观测数=("StkcdRecall", "size"),
            有效观测数=("有效观测数", "sum"),
            有效提示观测数=("有效提示观测数", "sum"),
            有效未提示观测数=("有效未提示观测数", "sum"),
        ))

    if not daily_parts:
        raise ValueError("指定日期范围内没有观测。")
    # 同一天可能分散在多个数据块中，必须再次汇总。
    daily_counts = pd.concat(daily_parts).groupby(level=0).sum().sort_index()
    daily_counts["两组均有有效观测"] = (
        daily_counts["有效提示观测数"].gt(0)
        & daily_counts["有效未提示观测数"].gt(0)
    )
    return daily_counts


def show_sample_summary(daily_counts):
    print("\n日期数：", len(daily_counts))
    print("有效观测总数：", int(daily_counts["有效观测数"].sum()))
    print("两组均有有效观测的日期数：", int(daily_counts["两组均有有效观测"].sum()))
    print("有效样本中没有提示观测的日期数：", int(daily_counts["有效提示观测数"].eq(0).sum()))
    print("\n前10个日期的统计：")
    print(daily_counts.head(10).to_string())

    yearly_counts = daily_counts.groupby(daily_counts.index.year).agg(
        交易日期数=("面板观测数", "size"),
        有效观测数=("有效观测数", "sum"),
        有效提示观测数=("有效提示观测数", "sum"),
        两组均有有效观测的日期数=("两组均有有效观测", "sum"),
    )
    yearly_counts.index.name = "年份"
    print("\n各年份的提示观测分布：")
    print(yearly_counts.to_string())

    active_dates = daily_counts.index[daily_counts["有效提示观测数"].gt(0)]
    if len(active_dates):
        print("\n最早有效提示日期：", active_dates.min().date())
        print("最晚有效提示日期：", active_dates.max().date())


# ============================================================
# 四、提取样本，核对每日行数与公司日期唯一键
# ============================================================

def extract_regression_sample(daily_counts):
    target_dates = daily_counts.index[daily_counts["两组均有有效观测"]]
    if len(target_dates) == 0:
        raise ValueError("没有同时存在两组有效观测的日期。")
    print("\n准备提取的日期数：", len(target_dates))

    regression_parts = []
    for chunk in read_chunks("提取回归样本"):
        chunk = chunk.loc[chunk["Trddt"].isin(target_dates)].copy()
        if chunk.empty:
            continue
        valid = get_valid_mask(chunk)
        if valid.any():
            regression_parts.append(chunk.loc[valid, needed_cols].copy())

    if not regression_parts:
        raise ValueError("没有提取到有效回归样本。")
    regression_data = pd.concat(regression_parts, ignore_index=True)
    actual_counts = (
        regression_data.groupby("Trddt").size().reindex(target_dates, fill_value=0)
    )
    expected_counts = daily_counts.loc[target_dates, "有效观测数"]
    if not actual_counts.eq(expected_counts).all():
        raise ValueError("提取后的日度观测数与此前统计不一致。")
    if regression_data.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("回归样本存在重复的被提示公司日期记录。")

    print("\n每日回归样本提取完成。")
    print("观测行数：", len(regression_data))
    print("日期数：", regression_data["Trddt"].nunique())
    print("各日期观测数量与公司日期唯一键核对通过。")
    return regression_data


# ============================================================
# 五、逐日回归：先保留独立控制变量，再检查提示变量能否识别
# ============================================================

def estimate_one_day(day_data):
    if day_data.empty:
        raise ValueError("传入的单日样本为空。")
    row = {
        "日期": day_data["Trddt"].iloc[0],
        "观测数": len(day_data),
        "提示观测数": int(day_data[cue].eq(1).sum()),
        "未提示观测数": int(day_data[cue].eq(0).sum()),
        "保留控制变量数": 0,
        "系数_百分点": np.nan,
        "标准误_百分点": np.nan,
        "置信下限_百分点": np.nan,
        "置信上限_百分点": np.nan,
        "显著性概率": np.nan,
        "状态": "待估计",
    }

    kept_controls = []
    omitted_controls = []
    control_matrix = np.ones((len(day_data), 1))
    for col in control_cols:
        candidate_matrix = np.column_stack([control_matrix, day_data[col].to_numpy()])
        if np.linalg.matrix_rank(candidate_matrix) > control_matrix.shape[1]:
            kept_controls.append(col)
            control_matrix = candidate_matrix
        else:
            omitted_controls.append(col)
    row["保留控制变量数"] = len(kept_controls)

    design = sm.add_constant(day_data[kept_controls + [cue]], has_constant="add")
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        row["状态"] = "提示变量与控制变量完全共线"
        return row, omitted_controls
    if len(day_data) <= design.shape[1]:
        row["状态"] = "剩余自由度不足"
        return row, omitted_controls

    fit = sm.OLS(day_data[dependent_variable], design, missing="raise").fit(
        cov_type="HC1",  # 异方差稳健标准误。
        use_t=True,      # 检验与置信区间使用t分布。
    )
    # 收益率源变量使用小数，展示系数、标准误和区间时乘100转为百分点。
    coefficient = float(fit.params[cue] * 100)
    standard_error = float(fit.bse[cue] * 100)
    ci_low, ci_high = (
        fit.conf_int(alpha=significance_level).loc[cue] * 100
    ).tolist()
    pvalue = float(fit.pvalues[cue])

    if np.isfinite([coefficient, standard_error, ci_low, ci_high, pvalue]).all():
        row.update({
            "系数_百分点": coefficient,
            "标准误_百分点": standard_error,
            "置信下限_百分点": ci_low,
            "置信上限_百分点": ci_high,
            "显著性概率": pvalue,
            "状态": "成功",
        })
    else:
        row["状态"] = "估计结果含缺失或无穷值"
    return row, omitted_controls


def estimate_all_days(regression_data):
    daily_rows = []
    for number, (_, day_data) in enumerate(
        regression_data.groupby("Trddt", sort=True), start=1
    ):
        if number % 20 == 0:
            print(f"已开始估计第 {number} 个日期……", flush=True)
        row, _ = estimate_one_day(day_data)
        daily_rows.append(row)

    daily_results = pd.DataFrame(daily_rows).sort_values("日期").reset_index(drop=True)
    success = daily_results["状态"].eq("成功")
    print("\n候选日期数：", len(daily_results))
    print("成功估计的日期数：", int(success.sum()))
    print("未成功估计的日期数：", int((~success).sum()))
    print(f"本次区间采用{confidence_level:.0%}置信水平。")
    print("\n前10个日期的回归结果：")
    print(daily_results.head(10).to_string(index=False))
    if (~success).any():
        print("\n未成功估计的日期及原因：")
        print(daily_results.loc[
            ~success, ["日期", "提示观测数", "未提示观测数", "状态"]
        ].to_string(index=False))
    return daily_results


# ============================================================
# 六、构造日度系数图，在主流程中保存
# ============================================================

def plot_daily_results(daily_results):
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_data = daily_results.loc[daily_results["状态"].eq("成功")].sort_values("日期")
    if plot_data.empty:
        raise ValueError("没有成功估计的日期，无法绘图。")

    single_group = plot_data["提示观测数"].eq(1) | plot_data["未提示观测数"].eq(1)
    regular_data = plot_data.loc[~single_group]
    single_data = plot_data.loc[single_group]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.vlines(
        regular_data["日期"], regular_data["置信下限_百分点"], regular_data["置信上限_百分点"],
        color="#AAB2BD", linewidth=0.8, alpha=0.7,
        label=f"逐日{confidence_level:.0%}稳健置信区间",
    )
    ax.scatter(
        regular_data["日期"], regular_data["系数_百分点"],
        color="#2166AC", s=22, zorder=3, label="每日回归系数",
    )
    if not single_data.empty:
        ax.scatter(
            single_data["日期"], single_data["系数_百分点"],
            color="#D97706", marker="^", s=42, zorder=4, label="某组仅1条观测",
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    ax.set_xticks(pd.to_datetime([
        f"{year}-01-01" for year in range(start_year, end_year + 1, 2)
    ]))
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("自然日期")
    ax.set_ylabel("提示系数（百分点）")
    ax.set_title(f"{window_label}累计特征调整收益：每日横截面回归系数")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


# ============================================================
# 七、主流程：只有直接运行时才执行
# ============================================================

def main():
    check_input()
    daily_counts = count_daily_samples()
    show_sample_summary(daily_counts)
    regression_data = extract_regression_sample(daily_counts)

    target_dates = daily_counts.index[daily_counts["两组均有有效观测"]]
    example_date = daily_counts.loc[target_dates, "有效提示观测数"].idxmax()
    example_data = regression_data.loc[regression_data["Trddt"].eq(example_date)]
    example_result, omitted_controls = estimate_one_day(example_data)
    print("\n单日示范结果：")
    print(pd.Series(example_result).to_string())
    print("\n该日为常量或完全共线的控制变量：")
    print(omitted_controls)

    daily_results = estimate_all_days(regression_data)
    if daily_results["状态"].eq("成功").any():
        fig, ax = plot_daily_results(daily_results)
        image_dir.mkdir(parents=True, exist_ok=True)
        figure_path = image_dir / (
            "自然日_主回归_"
            f"窗口{window_label}_{confidence_level:.0%}置信区间.png"
        )
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("图片已保存：", figure_path)
    else:
        print("没有成功估计的日期，保留结果表并跳过图片保存。")
    return daily_counts, regression_data, daily_results


if __name__ == "__main__":
    daily_counts, regression_data, daily_results = main()
