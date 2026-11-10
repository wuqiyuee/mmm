"""
按当前表2主规格估计并绘制图2“收益动态和长期反转”。

与表2共用主回归口径：历史沪深300提示、个人365天共同覆盖排除、
数行者分段新闻、合格填充日起点及全部年份。收益沿用季度缓存中的
既有缩尾值，各窗口只检查自身收益，不要求50个窗口同时完整。

直接复用表2规格（6）的估计器：被提示公司×年度季度及交易日期
固定效应、递归删除单例、双向聚类有限样本修正和精确残差自由度。
共估计50个累计窗口，结果单位为百分点；使用同一自由度的t分布
绘制99%、95%、90%置信区间，红点表示10%水平显著。

每个窗口保存前校验口径及已有的第1、第10窗口表2锚点。
仅当数据和设置未变时使用 --resume（续算参数）；旧口径结果不能续算。
导入本模块不会读取回归数据或写文件。
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components


def find_project_root() -> Path:
    """按项目说明、代码和中间结果三个标记，从当前文件的祖先目录寻根。"""
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "AGENTS.md").is_file()
            and (candidate / "code").is_dir()
            and (candidate / "中间结果").is_dir()
        ):
            return candidate
    raise FileNotFoundError("无法找到包含项目说明、代码和中间结果的项目根目录。")


ROOT = find_project_root()
MODULE_DIR = ROOT / "code" / "07_实证分析" / "90_公共模块"
TABLE2_DIR = ROOT / "code" / "07_实证分析" / "01_主线图表" / "表2_收益回归"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from 图2图3扩展_日期固定效应回归公共模块 import two_way_cluster_covariance

# 正式输入：与表2读取同一份沪深300最终面板。
# 选择输入面板的回归样本筛选规格：5或6。
INPUT_SAMPLE_SPEC = 5

if INPUT_SAMPLE_SPEC not in (5, 6):
    raise ValueError("输入面板的筛样规格只能选择5或6。")

INPUT_PANEL_PATH = ROOT / (
    "中间结果/04_合并数据/"
    f"Pattern样本_最终回归面板_沪深300_筛样规格{INPUT_SAMPLE_SPEC}.csv"
)

# 提示变量、新闻来源及研究口径说明。
CUE_COLUMN = "quarterLag1HS300"
NEWS_SOURCE = "datago_segmented"

SPEC_ID = (
    "沪深300_个人365天_数行者分段_允许填充_全期"
    f"_筛样规格{INPUT_SAMPLE_SPEC}"
)

SPEC_DESCRIPTION = (
    "历史沪深300提示；个人365天共同覆盖排除；一级供应链及同行业排除；"
    "提示方保留金融和房地产；被提示方沿用非金融等基础资格、保留房地产，"
    "不追加市值和60个月交易史筛选；2005—2009年数行者报刊、"
    "2010—2025年数行者网络；允许合格填充日起点；保留全部年份"
)

# 21项自身财报公告控制。
OWN_EA_COLUMNS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)

# 2项其他公告控制。
EVENT_COLUMNS = [
    "ownOtherEventER",
    "ownPerformanceForecastBrief",
]

# 4项新闻控制。
NEWS_COLUMNS = [
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]

# 完整控制变量，共27项。
CONTROL_COLUMNS = OWN_EA_COLUMNS + EVENT_COLUMNS + NEWS_COLUMNS

# 表2脚本位于相邻任务目录，按实际文件路径加载，避免依赖跨目录模块名解析。
# 加载同一份正式估计器，导入过程不会执行其主函数或读取回归数据。
TABLE2_PATH = TABLE2_DIR / "估计回归.py"
if not TABLE2_PATH.is_file():
    raise FileNotFoundError(f"找不到表2估计器：{TABLE2_PATH}")

table2_spec = importlib.util.spec_from_file_location("_figure2_table2_estimator", TABLE2_PATH)
if table2_spec is None or table2_spec.loader is None:
    raise ImportError(f"无法创建表2估计器的加载入口：{TABLE2_PATH}")
table2_module = importlib.util.module_from_spec(table2_spec)
table2_spec.loader.exec_module(table2_module)
run_table2_regression = table2_module.run_return_regression
table2_specifications = table2_module.table2_specifications

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


OUTPUT_DIR = ROOT / "图表结果" / "01_主线图表" / "图2_收益动态"
OUTPUT_TAG = f"沪深300_筛样规格{INPUT_SAMPLE_SPEC}"

RESULT_PATH = OUTPUT_DIR / (
    f"图2_收益动态和长期反转_加入日期固定效应_{OUTPUT_TAG}_结果.csv"
)

OUTPUT_PATH = OUTPUT_DIR / (
    f"figure2_收益动态和长期反转_加入日期固定效应_{OUTPUT_TAG}.png"
)

TABLE2_SPECIFICATION = next(spec for spec in table2_specifications if spec["规格"] == "(6)")
if (
    TABLE2_SPECIFICATION["控制变量"] != CONTROL_COLUMNS
    or set(TABLE2_SPECIFICATION["固定效应"]) != {"date", "firm_yearquarter"}
):
    raise ValueError("表2规格（6）已变化，请核对图2的控制变量和固定效应设置。")

HORIZONS = list(range(1, 51))
OUTCOME_COLS = {
    horizon: f"chAdjreturn_t_tp{horizon}_Recall_w"
    for horizon in HORIZONS
}

RESULT_SPECIFICATION = {
    "口径标识": SPEC_ID,
    "提示变量": CUE_COLUMN,
    "新闻来源": NEWS_SOURCE,
    "表2规格": TABLE2_SPECIFICATION["规格"],
    "日期固定效应": "是",
    "被提示公司×年度季度固定效应": "是",
    "控制变量": "是",
    "结果单位": "百分点",
    "有限样本修正": "是",
}
NUMERIC_RESULT_FIELDS = [
    "线索系数", "聚类标准误", "置信区间下界", "置信区间上界",
    "t值", "p值", "回归样本量", "解释变量矩阵条件数", "R方",
    "残差自由度", "固定效应实际秩",
]
RESULT_COMPLETION_FIELDS = NUMERIC_RESULT_FIELDS + ["系数求解方法"]

# 从最长到最短绘制三档区间，保留原配色和线宽。
ERROR_BAR_SPECS = (
    {"label": "99%置信区间（1%水平）", "confidence_level": 0.99,
     "color": "#bdbdbd", "line_width": 0.8, "zorder": 1},
    {"label": "95%置信区间（5%水平）", "confidence_level": 0.95,
     "color": "#78909c", "line_width": 1.6, "zorder": 2},
    {"label": "90%置信区间（10%水平）", "confidence_level": 0.90,
     "color": "#1f4e79", "line_width": 2.8, "zorder": 3},
)


def read_figure2_panel(horizons: list[int]) -> pd.DataFrame:
    """直接读取选定的正式面板，各累计窗口分别处理自身收益缺失。"""

    if not horizons or not set(horizons).issubset(HORIZONS):
        raise ValueError("待估计窗口必须属于1至50。")

    if not INPUT_PANEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到正式合并面板：{INPUT_PANEL_PATH}"
        )

    # 两个基础窗口用于检查面板是否已完成合并阶段筛样。
    baseline_return_columns = [
        OUTCOME_COLS[1],
        OUTCOME_COLS[10],
    ]

    # 读取待估计窗口，同时保留两个基础窗口用于检查。
    return_columns = list(dict.fromkeys(
        baseline_return_columns
        + [OUTCOME_COLS[horizon] for horizon in horizons]
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

    # 检查日期、唯一键和提示公司口径。
    if data["Trddt"].isna().any():
        raise ValueError("正式面板中存在缺失日期。")

    if data.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("被提示公司与日期的组合不唯一。")

    if not data["cueDefinition"].eq("hs300").fillna(False).all():
        raise ValueError("输入面板不是统一的沪深300提示口径。")

    # 提示、27项控制和两个基础收益窗口应当完整。
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

    # 其余收益窗口允许缺失，由各次回归分别删除。
    # 无穷值属于异常，直接提示核对。
    for column in return_columns:
        if np.isinf(data[column].to_numpy()).any():
            raise ValueError(f"{column} 存在无穷值，请核对收益数据。")

    data.drop(columns=["cueDefinition"], inplace=True)
    data["Filling"] = data["Filling"].astype("uint8")
    data["StkcdRecall"] = data["StkcdRecall"].astype("category")
    data["yearQuarter"] = (
        data["Trddt"].dt.to_period("Q").astype("category")
    )

    print(
        f"图2正式面板读取完成：{len(data):,}条观测。",
        flush=True,
    )

    return data

def run_date_fe_regression(data: pd.DataFrame, horizon: int) -> dict:
    """直接复用表2规格（6），仅将结果转换为图2的字段和百分点单位。"""
    outcome = OUTCOME_COLS[horizon]
    model_row, _ = run_table2_regression(
        data=data,
        outcome=outcome,
        window_label=f"[t, t+{horizon}]",
        specification=TABLE2_SPECIFICATION,
        include_diagnostics=True,
    )
    coefficient = model_row["线索系数（百分点）"]
    standard_error = model_row["聚类标准误（百分点）"]
    critical_value = float(stats.t.ppf(0.975, model_row["残差自由度"]))
    return {
        **RESULT_SPECIFICATION,
        "口径说明": SPEC_DESCRIPTION,
        "窗口序号": horizon,
        "事件窗口": f"[t, t+{horizon}]",
        "因变量": outcome,
        "线索系数": coefficient,
        "聚类标准误": standard_error,
        "置信区间下界": coefficient - critical_value * standard_error,
        "置信区间上界": coefficient + critical_value * standard_error,
        **{
            field: model_row[field]
            for field in (
                "t值", "p值", "显著性", "回归样本量", "被提示公司聚类数",
                "日期聚类数", "删除单例组观测数", "正提示公司日数",
                "填充起始日数", "残差自由度", "固定效应实际秩",
                "解释变量矩阵条件数", "系数求解方法", "R方",
            )
        },
    }


def validate_result_specification(results: pd.DataFrame) -> None:
    """在筛除不完整窗口之前检查每一行，拒绝旧口径或混合口径续算。"""
    for column, expected in RESULT_SPECIFICATION.items():
        if column not in results or not results[column].astype(str).eq(expected).all():
            raise ValueError(
                f"已有图2结果的“{column}”与当前表2口径不一致，不能续算。"
                "请不带 --resume（续算参数）从头运行；旧文件不会在检查时被删除。"
            )


def validate_result_rows(results: pd.DataFrame) -> None:
    """检查一批完整窗口，可用于保存中间进度及最终绘图。"""
    validate_result_specification(results)
    horizons = pd.to_numeric(results["窗口序号"], errors="raise")
    if not horizons.isin(HORIZONS).all() or horizons.duplicated().any():
        raise ValueError("图2包含无效或重复窗口。")
    values = results[NUMERIC_RESULT_FIELDS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("图2结果包含缺失或非有限数值。")
    if not (results["聚类标准误"] > 0).all() or not (results["残差自由度"] > 0).all():
        raise ValueError("图2结果的标准误或残差自由度必须为正数。")
    if not results["p值"].between(0, 1).all():
        raise ValueError("图2结果包含无效p值。")
    if not (
        (results["置信区间下界"] <= results["线索系数"])
        & (results["线索系数"] <= results["置信区间上界"])
    ).all():
        raise ValueError("图2结果包含无效置信区间。")
    if results["系数求解方法"].isna().any():
        raise ValueError("图2结果缺少系数求解方法。")


def save_results(rows: list[dict]) -> pd.DataFrame:
    """检查口径和数值完整性，再保存逐窗口结果。"""
    results = (
        pd.DataFrame(rows)
        .sort_values("窗口序号")
        .reset_index(drop=True)
    )

    validate_result_rows(results)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULT_PATH.with_suffix(
        RESULT_PATH.suffix + ".tmp"
    )

    results.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
    )
    temporary_path.replace(RESULT_PATH)

    return results

def load_completed_rows(resume: bool) -> list[dict]:
    """仅接受当前口径；同口径的缺失诊断窗口重新估计。"""
    if not resume or not RESULT_PATH.exists():
        return []
    existing = pd.read_csv(RESULT_PATH)
    if existing.empty:
        return []
    validate_result_specification(existing)
    if "窗口序号" not in existing:
        raise ValueError("已有图2结果缺少窗口序号，无法续算。")
    horizons = pd.to_numeric(existing["窗口序号"], errors="raise")
    if not horizons.isin(HORIZONS).all():
        raise ValueError("已有图2结果的窗口序号必须为1至50的整数。")
    existing["窗口序号"] = horizons.astype(int)
    for column in RESULT_COMPLETION_FIELDS:
        if column not in existing:
            existing[column] = np.nan
    numeric = existing[NUMERIC_RESULT_FIELDS].apply(pd.to_numeric, errors="coerce")
    existing[NUMERIC_RESULT_FIELDS] = numeric
    complete = (
        np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
        & existing["系数求解方法"].notna()
    )
    incomplete_horizons = existing.loc[~complete, "窗口序号"].tolist()
    if incomplete_horizons:
        print(f"以下已有窗口的统计或诊断字段不完整，将重新估计：{incomplete_horizons}")
    completed = (
        existing.loc[complete]
        .drop_duplicates("窗口序号", keep="last")
        .sort_values("窗口序号")
    )
    if not completed.empty:
        validate_result_rows(completed)
    return completed.to_dict("records")


def validate_results(results: pd.DataFrame) -> None:
    """最终绘图必须有顺序完整的50个窗口。"""
    validate_result_rows(results)
    if results["窗口序号"].tolist() != HORIZONS:
        raise ValueError("日期固定效应版图2未包含顺序完整的50个窗口。")


def draw_figure2_date_fe(
    results: pd.DataFrame,
) -> None:
    """绘制加入交易日期固定效应的收益动态图。"""
    plot_data = (
        results
        .sort_values(
            "窗口序号"
        )
        .reset_index(
            drop=True
        )
    )

    font_path = Path(
        "/System/Library/Fonts/"
        "Hiragino Sans GB.ttc"
    )

    if font_path.exists():
        font_manager.fontManager.addfont(
            str(font_path)
        )

        font_name = (
            font_manager
            .FontProperties(
                fname=str(
                    font_path
                )
            )
            .get_name()
        )

        plt.rcParams[
            "font.sans-serif"
        ] = [
            font_name
        ]

    plt.rcParams[
        "axes.unicode_minus"
    ] = False

    figure, axis = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    x_values = plot_data[
        "窗口序号"
    ].to_numpy(
        dtype=np.float64,
    )

    coefficients = plot_data[
        "线索系数"
    ].to_numpy(
        dtype=np.float64,
    )

    standard_errors = plot_data[
        "聚类标准误"
    ].to_numpy(
        dtype=np.float64,
    )

    degrees = plot_data["残差自由度"].to_numpy(dtype=np.float64)

    for spec in ERROR_BAR_SPECS:
        critical_values = stats.t.ppf((1 + spec["confidence_level"]) / 2, degrees)
        interval_radius = critical_values * standard_errors
        axis.vlines(
            x_values,
            coefficients
            - interval_radius,
            coefficients
            + interval_radius,
            colors=spec[
                "color"
            ],
            linewidth=spec[
                "line_width"
            ],
            label=spec[
                "label"
            ],
            zorder=spec[
                "zorder"
            ],
        )

    # 与图中90%置信区间一致，判断10%水平显著性
    significant = plot_data["p值"].to_numpy(dtype=np.float64) < 0.10

    # 显著用红色，其余用黑色
    point_colors = np.where(significant, "#d62728", "#1a1a1a")

    axis.scatter(
        x_values,
        coefficients,
        color=point_colors,
        s=16,
        zorder=4,
    )

    # 只添加红点图例，不增加实际数据点
    axis.scatter(
        [],
        [],
        color="#d62728",
        s=16,
        label="红点：10%水平显著",
    )

    axis.axhline(
        0,
        color="#333333",
        linewidth=0.8,
    )

    axis.set_title(
        "收益动态和长期反转：历史沪深300提示\n"
        "个人365天排除；数行者分段新闻；合格填充日起点；全部年份\n"
        "固定效应：被提示公司×年度季度；交易日期（事件起点）",
        fontsize=12,
    )

    axis.set_xlabel(
        "事件窗口"
    )

    axis.set_ylabel(
        "提示变量系数（百分点）"
    )

    ticks = [
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ]

    axis.set_xticks(
        ticks
    )

    axis.set_xticklabels(
        [
            f"[t,t+{tick}]"
            for tick in ticks
        ]
    )

    axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=9,
        handlelength=1.4,
    )

    axis.spines[
        "top"
    ].set_visible(False)

    axis.spines[
        "right"
    ].set_visible(False)

    axis.tick_params(
        direction="out"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""
    parser = argparse.ArgumentParser(description="按当前表2主规格估计并绘制图2。")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="仅从口径标识和诊断完整的当前结果续算；更改口径后请从头运行。",
    )
    return parser.parse_args()


def main() -> None:
    """读取正式面板，逐窗口估计、保存并绘图。"""
    args = parse_args()

    existing_rows = load_completed_rows(resume=args.resume)
    completed = {
        int(row["窗口序号"])
        for row in existing_rows
    }
    pending = [
        horizon
        for horizon in HORIZONS
        if horizon not in completed
    ]
    rows = existing_rows.copy()

    if pending:
        data = read_figure2_panel(pending)

        for position, horizon in enumerate(pending, start=1):
            print(
                f"正在运行加入日期固定效应的图2：[t, t+{horizon}]"
                f"（待估计窗口 {position}/{len(pending)}）"
            )

            row = run_date_fe_regression(data, horizon)

            # 先检查并保存，再更新内存中的已完成记录。
            results = save_results(rows + [row])
            rows.append(row)

            print(
                f"窗口 {horizon} 完成："
                f"系数={row['线索系数']:.6f}，"
                f"标准误={row['聚类标准误']:.6f}"
            )

            data.drop(
                columns=[OUTCOME_COLS[horizon]],
                inplace=True,
            )
            gc.collect()

        del data
        gc.collect()

    else:
        print("已有结果包含全部50个完整窗口，无需重新估计。")
        results = (
            pd.DataFrame(rows)
            .sort_values("窗口序号")
            .reset_index(drop=True)
        )

    validate_results(results)
    draw_figure2_date_fe(results)

    print(f"日期固定效应版图2结果已保存：{RESULT_PATH}")
    print(f"日期固定效应版图2已保存：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
