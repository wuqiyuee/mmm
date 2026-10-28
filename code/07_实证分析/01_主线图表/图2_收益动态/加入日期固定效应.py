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
from scipy import stats


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

from 主回归口径 import (
    CONTROL_COLUMNS, CUE_COLUMN, NEWS_SOURCE, SPEC_DESCRIPTION, SPEC_ID,
    iter_main_blocks,
)
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
RESULT_PATH = OUTPUT_DIR / "图2_收益动态和长期反转_加入日期固定效应结果.csv"
OUTPUT_PATH = OUTPUT_DIR / "figure2_收益动态和长期反转_加入日期固定效应.png"
TABLE2_RESULT_PATH = ROOT / "图表结果" / "01_主线图表" / "表2_收益回归" / "表2_模型汇总.csv"

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
ANCHORS = {1: "[t, t+1]", 10: "[t, t+10]"}
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
    """按与表2完全相同的季度入口和列类型，组装待估计窗口。"""
    if not horizons or not set(horizons).issubset(HORIZONS):
        raise ValueError("待估计窗口必须属于1至50。")
    print(f"读取图2与表2共用的主样本：{SPEC_DESCRIPTION}")
    pieces = []
    for block, returns, controls, cue, common in iter_main_blocks():
        positions = np.flatnonzero(common)
        frame = pd.DataFrame(controls[positions], columns=CONTROL_COLUMNS)
        frame["StkcdRecall"] = block["firm"][positions]
        frame["Trddt"] = pd.to_datetime(block["date"][positions], unit="D", origin="unix")
        frame["Filling"] = block["filled"][positions]
        frame[CUE_COLUMN] = cue[positions]
        # 每个窗口沿用自己的有效收益；不对全部50窗共同删除缺失。
        for horizon in horizons:
            frame[OUTCOME_COLS[horizon]] = returns[positions, horizon - 1]
        pieces.append(frame)
    if not pieces:
        raise ValueError("当前沪深300主口径没有可用观测。")
    data = pd.concat(pieces, ignore_index=True)
    del pieces
    data["StkcdRecall"] = data["StkcdRecall"].astype("category")
    data["yearQuarter"] = data["Trddt"].dt.to_period("Q").astype("category")
    print(f"沪深300主样本基础观测：{len(data):,}；包含合格填充日起点。")
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


def load_table2_reference() -> pd.DataFrame:
    """估计前确认当前口径的两个表2锚点均存在。"""
    if not TABLE2_RESULT_PATH.is_file():
        raise FileNotFoundError(
            f"找不到表2模型汇总：{TABLE2_RESULT_PATH}。请先运行当前口径的表2。"
        )
    table2 = pd.read_csv(TABLE2_RESULT_PATH)
    required = {
        "样本", "收益窗口", "规格", "口径标识", "提示变量",
        "线索系数（百分点）", "聚类标准误（百分点）", "回归样本量",
    }
    missing = sorted(required - set(table2.columns))
    if missing:
        raise ValueError(f"表2模型汇总缺少当前口径校验字段：{missing}。请重新运行表2。")
    for window_label in ANCHORS.values():
        expected = table2.loc[
            table2["样本"].eq("pattern")
            & table2["收益窗口"].eq(window_label)
            & table2["规格"].eq(TABLE2_SPECIFICATION["规格"])
        ]
        if len(expected) != 1:
            raise ValueError(f"表2中无法找到唯一的{window_label}规格（6）锚点。")
        if (
            not expected["口径标识"].eq(SPEC_ID).all()
            or not expected["提示变量"].eq(CUE_COLUMN).all()
        ):
            raise ValueError("表2模型汇总仍是其他口径，请先重新运行当前表2。")
        numeric = expected[
            ["线索系数（百分点）", "聚类标准误（百分点）", "回归样本量"]
        ].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(numeric).all()
            or numeric[0, 1] <= 0
            or numeric[0, 2] <= 0
            or numeric[0, 2] != int(numeric[0, 2])
        ):
            raise ValueError(f"表2的{window_label}锚点含有无效系数、标准误或样本量。")
    return table2


def validate_against_table2(
    results: pd.DataFrame,
    horizons_to_check: set[int] | None = None,
    *,
    table2: pd.DataFrame,
) -> None:
    """按原容差校验第1、第10窗口的系数、标准误及精确样本量。"""
    validate_result_specification(results)
    if horizons_to_check is None:
        horizons_to_check = set(ANCHORS)
    if not set(horizons_to_check).issubset(ANCHORS):
        raise ValueError("表2锚点校验只接受第1和第10窗口。")
    checked = []
    for horizon, window_label in ANCHORS.items():
        if horizon not in horizons_to_check:
            continue
        expected = table2.loc[
            table2["样本"].eq("pattern")
            & table2["收益窗口"].eq(window_label)
            & table2["规格"].eq(TABLE2_SPECIFICATION["规格"])
        ]
        actual = results.loc[results["窗口序号"].eq(horizon)]
        if len(expected) != 1 or len(actual) != 1:
            raise ValueError(f"{window_label}无法完成表2规格（6）交叉校验。")
        if (
            not expected["口径标识"].eq(SPEC_ID).all()
            or not expected["提示变量"].eq(CUE_COLUMN).all()
        ):
            raise ValueError("表2锚点与当前主回归口径不一致。")
        for actual_column, expected_column in (
            ("线索系数", "线索系数（百分点）"),
            ("聚类标准误", "聚类标准误（百分点）"),
        ):
            actual_value = float(actual.iloc[0][actual_column])
            expected_value = float(expected.iloc[0][expected_column])
            if not np.isclose(actual_value, expected_value, rtol=1e-6, atol=1e-8):
                raise ValueError(
                    f"{window_label}的{actual_column}与表2规格（6）不一致："
                    f"图2={actual_value:.12g}，表2={expected_value:.12g}。"
                )
        if float(actual.iloc[0]["回归样本量"]) != float(expected.iloc[0]["回归样本量"]):
            raise ValueError(f"{window_label}的回归样本量与表2规格（6）不一致。")
        checked.append(window_label)
    if checked:
        print(f"{'、'.join(checked)}已通过表2模式公司主样本规格（6）交叉校验。")


def save_results(rows: list[dict], *, table2: pd.DataFrame) -> pd.DataFrame:
    """先校验候选结果，再原子替换进度文件，避免保存失败的锚点。"""
    results = (
        pd.DataFrame(rows)
        .sort_values("窗口序号")
        .reset_index(drop=True)
    )
    validate_result_rows(results)
    anchors_present = set(results["窗口序号"]) & set(ANCHORS)
    validate_against_table2(results, anchors_present, table2=table2)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULT_PATH.with_suffix(RESULT_PATH.suffix + ".tmp")
    results.to_csv(temporary_path, index=False, encoding="utf-8-sig")
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
    """运行50个窗口；通过表2锚点校验后保存进度和绘图。"""
    args = parse_args()
    table2 = load_table2_reference()
    existing_rows = load_completed_rows(resume=args.resume)
    completed = {int(row["窗口序号"]) for row in existing_rows}
    completed_anchors = completed & set(ANCHORS)
    if completed_anchors:
        validate_against_table2(
            pd.DataFrame(existing_rows), completed_anchors, table2=table2,
        )
    pending = [horizon for horizon in HORIZONS if horizon not in completed]
    rows = existing_rows.copy()
    if pending:
        data = read_figure2_panel(pending)
        for position, horizon in enumerate(pending, start=1):
            print(
                f"正在运行加入日期固定效应的图2：[t, t+{horizon}]"
                f"（待估计窗口 {position}/{len(pending)}）"
            )
            row = run_date_fe_regression(data, horizon)
            # 保存函数会先检查锚点；不一致时保留磁盘上先前的进度。
            results = save_results(rows + [row], table2=table2)
            rows.append(row)
            print(
                f"窗口 {horizon} 完成：系数={row['线索系数']:.6f}，"
                f"标准误={row['聚类标准误']:.6f}"
            )
            data.drop(columns=[OUTCOME_COLS[horizon]], inplace=True)
            gc.collect()
        del data
        gc.collect()
    else:
        print("已有结果包含全部50个完整窗口，无需重新估计。")
        results = pd.DataFrame(rows).sort_values("窗口序号").reset_index(drop=True)
    validate_results(results)
    validate_against_table2(results, table2=table2)
    draw_figure2_date_fe(results)
    print(f"日期固定效应版图2结果已保存：{RESULT_PATH}")
    print(f"日期固定效应版图2已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
