"""图2：收益动态和长期反转——公司季度固定效应版本。

直接读取选定筛样规格的沪深300正式合并面板。
回归吸收被提示公司×年度季度固定效应，
标准误按被提示公司和交易日期双向聚类。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from tqdm.auto import tqdm


def find_project_root() -> Path:
    """从当前脚本向上查找项目根目录。"""
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "AGENTS.md").is_file()
            and (candidate / "code").is_dir()
            and (candidate / "中间结果").is_dir()
        ):
            return candidate

    raise FileNotFoundError("无法找到项目根目录。")


ROOT = find_project_root()

# 输入面板的合并阶段回归样本筛选规格：5或6。
# 比较两种固定效应时，建议两个图2脚本选择相同的输入面板。
INPUT_SAMPLE_SPEC = 5

if INPUT_SAMPLE_SPEC not in (5, 6):
    raise ValueError("输入面板的筛样规格只能选择5或6。")

INPUT_PANEL_PATH = ROOT / (
    "中间结果/04_合并数据/"
    f"Pattern样本_最终回归面板_沪深300_筛样规格{INPUT_SAMPLE_SPEC}.csv"
)

# 核心提示变量。
CUE_COLUMN = "quarterLag1HS300"

# 21项自身财报公告控制变量。
OWN_EA_COLUMNS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)

# 2项其他公告控制变量。
EVENT_COLUMNS = [
    "ownOtherEventER",
    "ownPerformanceForecastBrief",
]

# 4项新闻控制变量。
NEWS_COLUMNS = [
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]

CONTROL_COLUMNS = OWN_EA_COLUMNS + EVENT_COLUMNS + NEWS_COLUMNS

# 沿用该版本已有的估计年份范围。
START_YEAR = 2005
END_YEAR = 2025
HORIZONS = list(range(1, 51))

# 仅重画时读取已有回归结果。
PLOT_ONLY = False

# 结果、进度和图片均标明输入面板的筛样规格。
OUTPUT_DIR = ROOT / "图表结果" / "01_主线图表" / "图2_收益动态"
OUTPUT_TAG = f"沪深300_筛样规格{INPUT_SAMPLE_SPEC}"

RESULT_PATH = OUTPUT_DIR / (
    f"图2_公司季度固定效应_{OUTPUT_TAG}_回归结果.csv"
)
PROGRESS_PATH = OUTPUT_DIR / (
    f"图2_公司季度固定效应_{OUTPUT_TAG}"
    "_数行者分段新闻_2005至2025年_逐窗口进度.csv"
)
OUTPUT_PATH = OUTPUT_DIR / (
    f"图2_收益动态和长期反转_公司季度固定效应_{OUTPUT_TAG}.png"
)

RESULT_FIELDS = [
    "window", "coef", "se", "ci_low", "ci_high", "nobs"
]



def load_progress(path, source_paths, curve_name):
    """核对计算版本，并恢复已经保存的窗口。"""
    input_info = []

    for source in source_paths:
        source = Path(source)
        file_stat = source.stat()
        input_info.append({
            "path": str(source.resolve()),
            "size": file_stat.st_size,
            "mtime_ns": file_stat.st_mtime_ns,
        })

    # 根据脚本内容、输入文件元信息和依赖版本生成校验标记
    config = {
        "curve": curve_name,
        "script_hash": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "inputs": input_info,
        "versions": {
            name: version(name)
            for name in [
                "numpy", "pandas", "scipy", "linearmodels", "pyhdfe"
            ]
        },
    }

    signature = hashlib.sha256(
        json.dumps(
            config, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()

    if not path.exists():
        return [], signature

    saved = pd.read_csv(
        path,
        dtype={"signature": "string"},
        float_precision="round_trip",
    )

    if saved.empty or list(saved.columns) != RESULT_FIELDS + ["signature"]:
        raise ValueError(f"进度表为空或格式不正确，请检查：{path}")

    if not saved["signature"].eq(signature).fillna(False).all():
        raise ValueError(
            f"进度表与当前输入文件、脚本或依赖版本不一致：{path}\n"
            "请更换进度表文件名，开始新一轮计算。"
        )

    values = saved[RESULT_FIELDS].apply(
        pd.to_numeric, errors="raise"
    )
    windows = values["window"]

    valid = (
        np.isfinite(values.to_numpy(dtype=float)).all()
        and windows.between(1, 50).all()
        and windows.mod(1).eq(0).all()
        and not windows.duplicated().any()
        and values["se"].ge(0).all()
        and values["nobs"].gt(0).all()
        and values["nobs"].mod(1).eq(0).all()
        and values["ci_low"].le(values["coef"]).all()
        and values["ci_high"].ge(values["coef"]).all()
    )

    if not valid:
        raise ValueError(f"进度表存在异常结果，请检查：{path}")

    values = values.astype({"window": "int64", "nobs": "int64"})
    values = values.sort_values("window")

    return values.to_dict("records"), signature


def save_progress(rows, path, signature):
    """先写临时文件，写完后再替换正式进度表。"""
    table = pd.DataFrame(rows, columns=RESULT_FIELDS)
    table = table.sort_values("window")
    table["signature"] = signature

    temporary_dir = ROOT / "work" / "图2结果暂存"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_dir / (path.name + ".tmp")
    table.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
    )
    temporary_path.replace(path)

def draw_dynamic_figure(table, title, output_path):
    """根据已经保存的回归结果绘图。"""
    # 延迟初始化绘图与字体，导入公共计算模块时不创建绘图文件。
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    font_manager.fontManager.addfont(font_path)
    zh_font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.sans-serif"] = [zh_font]
    plt.rcParams["axes.unicode_minus"] = False

    table = table.sort_values("window")

    windows = table["window"].to_numpy(dtype=float)
    coefficients = table["coef"].to_numpy(dtype=float)
    standard_errors = table["se"].to_numpy(dtype=float)

    interval_specs = [
        # 从最长到最短依次叠画，95%区间保留原回归结果
        (
            coefficients - 2.5758293035489004 * standard_errors,
            coefficients + 2.5758293035489004 * standard_errors,
            "#bdbdbd", 0.8, "99%置信区间", 1,
        ),
        (
            table["ci_low"].to_numpy(dtype=float),
            table["ci_high"].to_numpy(dtype=float),
            "#78909c", 1.6, "95%置信区间", 2,
        ),
        (
            coefficients - 1.6448536269514722 * standard_errors,
            coefficients + 1.6448536269514722 * standard_errors,
            "#1f4e79", 2.8, "90%置信区间", 3,
        ),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for lower, upper, color, width, label, order in interval_specs:
        ax.vlines(
            windows,
            lower,
            upper,
            colors=color,
            linewidth=width,
            label=label,
            zorder=order,
        )

    # 按图中90%置信区间判断10%水平显著性
    significant = (
        np.abs(coefficients)
        > 1.6448536269514722 * standard_errors
    )

    # 显著用红色，其余用黑色
    point_colors = np.where(significant, "#d62728", "#1a1a1a")

    ax.scatter(
        windows,
        coefficients,
        color=point_colors,
        s=16,
        zorder=4,
    )

    ax.axhline(0, color="#333333", linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("事件窗口")
    ax.set_ylabel("提示变量系数（百分点）")

    ticks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"[t,t+{tick}]" for tick in ticks])

    # 添加红点的图例说明
    ax.scatter(
        [],
        [],
        color="#d62728",
        s=16,
        label="红点：10%水平显著",
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=9,
        handlelength=1.4,
    )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"图已保存到：{output_path}")


def read_figure2_panel(horizons: list[int]) -> pd.DataFrame:
    """读取正式面板，保留各窗口自身的收益缺失情况。"""
    if not horizons or not set(horizons).issubset(HORIZONS):
        raise ValueError("待估计窗口必须属于1至50。")

    if not INPUT_PANEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到正式合并面板：{INPUT_PANEL_PATH}"
        )

    # 两个基础窗口在合并阶段已经完成回归样本筛选。
    baseline_return_columns = [
        "chAdjreturn_t_tp1_Recall_w",
        "chAdjreturn_t_tp10_Recall_w",
    ]

    # 读取待计算窗口，同时读取两个基础窗口进行完整性检查。
    return_columns = list(dict.fromkeys(
        baseline_return_columns
        + [
            f"chAdjreturn_t_tp{horizon}_Recall_w"
            for horizon in horizons
        ]
    ))

    numeric_columns = [
        CUE_COLUMN,
        *CONTROL_COLUMNS,
        *return_columns,
    ]

    read_columns = [
        "StkcdRecall",
        "Trddt",
        "Filling",
        "cueDefinition",
        *numeric_columns,
    ]

    column_types = {
        column: "float32"
        for column in numeric_columns
    }
    column_types.update({
        "StkcdRecall": "int32",
        "Filling": "float32",
        "cueDefinition": "string",
    })

    print(
        f"正在读取筛样规格（{INPUT_SAMPLE_SPEC}）面板："
        f"{INPUT_PANEL_PATH}",
        flush=True,
    )

    data = pd.read_csv(
        INPUT_PANEL_PATH,
        usecols=read_columns,
        dtype=column_types,
        encoding="utf-8-sig",
    )

    if data.empty:
        raise ValueError("正式合并面板没有观测。")

    data["Trddt"] = pd.to_datetime(
        data["Trddt"],
        errors="raise",
    )

    if data["Trddt"].isna().any():
        raise ValueError("正式面板中存在缺失日期。")

    if data.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("被提示公司与日期的组合不唯一。")

    if not data["cueDefinition"].eq("hs300").fillna(False).all():
        raise ValueError("输入面板不是统一的沪深300提示口径。")

    # 提示变量、27项控制和两个基础收益窗口应当完整。
    required_columns = [
        CUE_COLUMN,
        *CONTROL_COLUMNS,
        *baseline_return_columns,
    ]

    for column in required_columns:
        if not np.isfinite(data[column].to_numpy()).all():
            raise ValueError(
                f"正式面板的 {column} 存在缺失或无穷值，"
                "请核对合并阶段的回归样本筛选。"
            )

    for column in [CUE_COLUMN, "Filling"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} 必须全部为0或1。")

    # 其他累计窗口允许缺失，各次回归分别处理。
    for column in return_columns:
        if np.isinf(data[column].to_numpy()).any():
            raise ValueError(
                f"{column} 存在无穷值，请核对收益数据。"
            )

    # 沿用该版本原有的估计年份范围。
    data = data.loc[
        data["Trddt"].dt.year.between(START_YEAR, END_YEAR)
    ].copy()

    if data.empty:
        raise ValueError("指定年份范围内没有可用观测。")

    data.drop(columns=["cueDefinition"], inplace=True)
    data["Filling"] = data["Filling"].astype("uint8")
    data["StkcdRecall"] = data["StkcdRecall"].astype("category")

    # 年度季度直接根据交易日期生成。
    data["yearQuarter"] = (
        data["Trddt"].dt.to_period("Q").astype("category")
    )

    print(
        f"公司季度版本面板读取完成：{len(data):,}条观测。",
        flush=True,
    )

    return data

def run_figure2() -> list[Path]:
    """逐窗口估计，保存进度，最后整理结果并绘图。"""
    results, signature = load_progress(
        PROGRESS_PATH,
        [INPUT_PANEL_PATH],
        "图2：公司季度固定效应，原始小数收益率",
    )

    completed = {int(row["window"]) for row in results}
    remaining = [
        horizon
        for horizon in HORIZONS
        if horizon not in completed
    ]

    print(
        f"已恢复{len(completed)}个窗口，"
        f"待计算{len(remaining)}个窗口。",
        flush=True,
    )

    # 已全部完成时，直接整理已有进度。
    if remaining:
        data = read_figure2_panel(remaining)

    progress_bar = tqdm(
        remaining,
        total=len(remaining),
        desc="正在回归窗口",
        unit="个窗口",
        dynamic_ncols=True,
    )

    for horizon in progress_bar:
        progress_bar.set_postfix_str(
            f"当前窗口 {horizon}/50"
        )

        outcome = f"chAdjreturn_t_tp{horizon}_Recall_w"
        rhs_columns = [CUE_COLUMN] + CONTROL_COLUMNS
        regression_columns = [
            outcome,
            "StkcdRecall",
            "Trddt",
            "yearQuarter",
            *rhs_columns,
        ]

        # 每个收益窗口都从原面板独立选取样本。
        # 其他必要变量已在读取时核验，这里删除本窗口收益缺失。
        reg = data.loc[
            data[outcome].notna(),
            regression_columns,
        ].copy()

        if reg.empty:
            raise ValueError(
                f"窗口[t,t+{horizon}]没有可用回归观测。"
            )

        # 在当前窗口的有效样本内，检查公司×年度季度单例。
        group_sizes = reg.groupby(
            ["StkcdRecall", "yearQuarter"],
            observed=True,
            sort=False,
        )[outcome].transform("size")

        reg = reg.loc[group_sizes.gt(1)].copy()

        if reg.empty:
            raise ValueError(
                f"窗口[t,t+{horizon}]删除公司季度单例后没有剩余观测。"
            )


        # 公司与年度季度的交互固定效应。
        reg["firm_yearquarter_fe"] = pd.Categorical(
            reg["StkcdRecall"].astype(str)
            + "_"
            + reg["yearQuarter"].astype(str)
        )

        y = reg[outcome]
        X = reg[rhs_columns]
        absorb = reg[["firm_yearquarter_fe"]]

        # 日期只用于标准误聚类，不作为本版本的固定效应。
        clusters = pd.DataFrame(
            {
                "firm_cluster": pd.factorize(
                    reg["StkcdRecall"]
                )[0],
                "date_cluster": pd.factorize(
                    reg["Trddt"]
                )[0],
            },
            index=reg.index,
        )

        model = AbsorbingLS(
            dependent=y,
            exog=X,
            absorb=absorb,
        )

        fit = model.fit(
            cov_type="clustered",
            clusters=clusters,
            debiased=True,
        )

        coefficient = float(fit.params[CUE_COLUMN])
        standard_error = float(fit.std_errors[CUE_COLUMN])
        interval = fit.conf_int(level=0.95).loc[CUE_COLUMN]

        lower = float(interval.iloc[0])
        upper = float(interval.iloc[1])

        if (
            not np.isfinite([
                coefficient,
                standard_error,
                lower,
                upper,
            ]).all()
            or standard_error < 0
        ):
            raise ValueError(
                f"窗口[t,t+{horizon}]的回归结果存在异常。"
            )

        # 进度表保留原始小数收益率单位。
        results.append({
            "window": horizon,
            "coef": coefficient,
            "se": standard_error,
            "ci_low": lower,
            "ci_high": upper,
            "nobs": int(fit.nobs),
        })

        save_progress(results, PROGRESS_PATH, signature)

        # 释放本窗口对象，以及已完成的收益列。
        del reg, y, X, absorb, clusters, model, fit, interval
        del data[outcome]
        gc.collect()

    if remaining:
        del data
        gc.collect()

    result_table = (
        pd.DataFrame(results, columns=RESULT_FIELDS)
        .sort_values("window")
        .reset_index(drop=True)
    )

    # 正式结果及图片统一使用百分点。
    percentage_columns = [
        "coef", "se", "ci_low", "ci_high"
    ]
    result_table[percentage_columns] = (
        result_table[percentage_columns] * 100
    )

    result_table.to_csv(
        RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"回归结果已保存到：{RESULT_PATH}")

    draw_dynamic_figure(
        result_table,
        "收益动态和长期反转（公司×年度季度固定效应）",
        OUTPUT_PATH,
    )

    return [RESULT_PATH, PROGRESS_PATH, OUTPUT_PATH]


def main(argv=None) -> list[Path]:
    """运行回归，或读取已有结果重新绘图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=None,
        help="读取已有回归结果重新绘图。",
    )
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_only = (
        PLOT_ONLY
        if args.plot_only is None
        else args.plot_only
    )

    if plot_only:
        saved_results = pd.read_csv(RESULT_PATH)

        if (
            len(saved_results) != len(HORIZONS)
            or set(saved_results["window"]) != set(HORIZONS)
        ):
            raise ValueError("已有结果没有完整覆盖1至50窗口。")

        # 已保存结果为百分点，重画时直接使用。
        draw_dynamic_figure(
            saved_results,
            "收益动态和长期反转（公司×年度季度固定效应）",
            OUTPUT_PATH,
        )
        return [OUTPUT_PATH]

    return run_figure2()


if __name__ == "__main__":
    main()