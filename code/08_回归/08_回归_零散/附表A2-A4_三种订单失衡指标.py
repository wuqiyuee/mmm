"""
估计加入交易日期固定效应的附表 A.2-A.4 散户订单失衡稳健性回归。

三张附表分别采用不同的订单失衡衡量口径：
1. 附表 A.2，OIBnum：买卖次数；
2. 附表 A.3，OIBamount：成交金额；
3. 附表 A.4，OIBorder：成交笔数。

固定效应：
1. 被提示公司 × 年度季度；
2. 交易日期。

标准误仍按被提示公司和交易日期双向聚类。

脚本运行结束后会为每种口径分别保存模型汇总、全部系数明细和
图片式完整系数表。
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[3]

PATTERN_PATH = (
    ROOT
    / "中间结果"
    / "04_合并数据"
    / "Pattern样本_最终回归面板.csv"
)

OIB_PATH = (
    ROOT
    / "中间结果"
    / "02_变量构造结果"
    / "2交易行为OIB、Avol"
    / "OIB_CSMAR_prep.csv"
)

RESULT_DIR = ROOT / "中间结果" / "08_回归结果"
IMAGE_DIR = ROOT / "图表结果" / "论文复现" / "表格"


START_YEAR = None
END_YEAR = None

CUE_COL = "quarterLag1size90plus"

OWN_EA_COLS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)

DISCLOSURE_COLS = [
    "ownMajorDisclosure",
    "ownPerformanceForecast",
]

MEDIA_COLS = [
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]

RHS_COLS = (
    [CUE_COL]
    + OWN_EA_COLS
    + DISCLOSURE_COLS
    + MEDIA_COLS
)

WINDOW_SPECS = [
    ("tp1", "[t, t+1]"),
    ("tp10", "[t, t+10]"),
]

OIB_METRIC_SPECS = [
    {
        "table_number": "A.2",
        "prefix": "OIBnum",
        "measure_label": "买卖次数",
        "results_path": (
            RESULT_DIR
            / "附表A2_OIBnum_买卖次数订单失衡_加入日期固定效应结果.csv"
        ),
        "coefficients_path": (
            RESULT_DIR
            / "附表A2_OIBnum_买卖次数订单失衡_加入日期固定效应系数明细.csv"
        ),
        "image_path": (
            IMAGE_DIR
            / "appendix_A2_OIBnum_买卖次数订单失衡_完整系数版_加入日期固定效应.png"
        ),
    },
    {
        "table_number": "A.3",
        "prefix": "OIBamount",
        "measure_label": "成交金额",
        "results_path": (
            RESULT_DIR
            / "附表A3_OIBamount_成交金额订单失衡_加入日期固定效应结果.csv"
        ),
        "coefficients_path": (
            RESULT_DIR
            / "附表A3_OIBamount_成交金额订单失衡_加入日期固定效应系数明细.csv"
        ),
        "image_path": (
            IMAGE_DIR
            / "appendix_A3_OIBamount_成交金额订单失衡_完整系数版_加入日期固定效应.png"
        ),
    },
    {
        "table_number": "A.4",
        "prefix": "OIBorder",
        "measure_label": "成交笔数",
        "results_path": (
            RESULT_DIR
            / "附表A4_OIBorder_成交笔数订单失衡_加入日期固定效应结果.csv"
        ),
        "coefficients_path": (
            RESULT_DIR
            / "附表A4_OIBorder_成交笔数订单失衡_加入日期固定效应系数明细.csv"
        ),
        "image_path": (
            IMAGE_DIR
            / "appendix_A4_OIBorder_成交笔数订单失衡_完整系数版_加入日期固定效应.png"
        ),
    },
]

OIB_RAW_COLS = [
    f"{metric_spec['prefix']}_avg_t_{window_suffix}"
    for metric_spec in OIB_METRIC_SPECS
    for window_suffix, _ in WINDOW_SPECS
]

WINDOW_ORDER = [
    window_label
    for _, window_label in WINDOW_SPECS
]

FONT_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
if FONT_PATH.exists():
    FONT = FontProperties(fname=str(FONT_PATH))
    FONT_BOLD = FontProperties(
        fname=str(FONT_PATH),
        weight="bold",
    )
else:
    FONT = FontProperties(family="sans-serif")
    FONT_BOLD = FontProperties(
        family="sans-serif",
        weight="bold",
    )

plt.rcParams["axes.unicode_minus"] = False


def significance_stars(p_value: float) -> str:
    """根据双侧检验的 p 值添加显著性星号。"""
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def winsorize_1pct(series: pd.Series) -> pd.Series:
    """把变量的上下各 1% 观测缩尾到相应分位点。"""
    valid = pd.to_numeric(series, errors="coerce")
    lower = valid.quantile(0.01)
    upper = valid.quantile(0.99)
    return valid.clip(lower=lower, upper=upper)


def drop_fe_singletons_iteratively(
    reg: pd.DataFrame,
    fe_code_cols: list[str],
) -> tuple[pd.DataFrame, int]:
    """递归删除任一固定效应维度中的单例组。"""
    total_removed = 0

    while True:
        singleton = np.zeros(len(reg), dtype=bool)

        for column in fe_code_cols:
            codes = reg[column].to_numpy(
                dtype=np.int64,
                copy=False,
            )
            counts = np.bincount(codes)
            singleton |= counts[codes] == 1

        removed_now = int(singleton.sum())
        if removed_now == 0:
            break

        total_removed += removed_now
        reg = reg.loc[~singleton].copy()

    return reg, total_removed


def read_appendix_robustness_date_fe_data() -> pd.DataFrame:
    """读取并整理加入日期固定效应的三张稳健性附表所需数据。"""
    panel_cols = list(dict.fromkeys(
        [
            "StkcdRecall",
            "Trddt",
            "year",
            "yearQuarter",
        ]
        + RHS_COLS
    ))

    numeric_cols = [
        column
        for column in panel_cols
        if column not in {
            "StkcdRecall",
            "Trddt",
            "yearQuarter",
        }
    ]

    panel_dtype = {
        "StkcdRecall": "string",
        "yearQuarter": "string",
    }
    panel_dtype.update({
        column: "float32"
        for column in numeric_cols
    })

    print("读取加入日期固定效应的附表 A.2-A.4 主样本……")
    data = pd.read_csv(
        PATTERN_PATH,
        usecols=panel_cols,
        dtype=panel_dtype,
        parse_dates=["Trddt"],
    )

    data["StkcdRecall"] = (
        data["StkcdRecall"]
        .str.strip()
        .str.zfill(6)
    )

    if START_YEAR is not None:
        data = data.loc[
            data["year"] >= START_YEAR
        ].copy()

    if END_YEAR is not None:
        data = data.loc[
            data["year"] <= END_YEAR
        ].copy()

    if data.duplicated(
        ["StkcdRecall", "Trddt"]
    ).any():
        raise ValueError(
            "附表 A.2-A.4 主样本的公司和日期不是唯一键。"
        )

    oib_dtype = {
        "Stkcd": "string",
        **{
            column: "float32"
            for column in OIB_RAW_COLS
        },
    }

    print("读取加入日期固定效应的三种口径订单失衡数据……")
    oib = pd.read_csv(
        OIB_PATH,
        usecols=[
            "Stkcd",
            "Trddt",
        ] + OIB_RAW_COLS,
        dtype=oib_dtype,
        parse_dates=["Trddt"],
    )

    oib["Stkcd"] = (
        oib["Stkcd"]
        .str.strip()
        .str.zfill(6)
    )
    oib = oib.rename(
        columns={
            "Stkcd": "StkcdRecall",
        }
    )

    if oib.duplicated(
        ["StkcdRecall", "Trddt"]
    ).any():
        raise ValueError(
            "订单失衡表的公司和日期不是唯一键。"
        )

    data = data.merge(
        oib,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="one_to_one",
    )

    del oib
    gc.collect()

    for column in OIB_RAW_COLS:
        data[f"{column}_w"] = (
            winsorize_1pct(data[column])
            .astype("float32")
        )

    data = data.drop(columns=OIB_RAW_COLS)

    data["StkcdRecall"] = (
        data["StkcdRecall"].astype("category")
    )
    data["yearQuarter"] = (
        data["yearQuarter"].astype("category")
    )

    return data


def run_appendix_robustness_date_fe_regression(
    data: pd.DataFrame,
    outcome: str,
) -> tuple[dict, list[dict]]:
    """估计同时吸收公司年度季度和交易日期固定效应的一个模型。"""
    needed = list(dict.fromkeys(
        [
            outcome,
            "StkcdRecall",
            "Trddt",
            "yearQuarter",
        ]
        + RHS_COLS
    ))

    reg = data[needed].dropna().copy()

    reg["_fe_firm_yearquarter"] = (
        reg.groupby(
            [
                "StkcdRecall",
                "yearQuarter",
            ],
            observed=True,
            sort=False,
        )
        .ngroup()
        .astype("int64")
    )

    reg["_fe_date"] = pd.factorize(
        reg["Trddt"],
        sort=False,
    )[0].astype("int64")

    reg, singleton_removed = (
        drop_fe_singletons_iteratively(
            reg,
            [
                "_fe_firm_yearquarter",
                "_fe_date",
            ],
        )
    )

    absorb = pd.DataFrame(
        {
            "firm_yearquarter": pd.Categorical(
                reg["_fe_firm_yearquarter"]
            ),
            "date": pd.Categorical(
                reg["_fe_date"]
            ),
        },
        index=reg.index,
    )

    clusters = pd.DataFrame(
        {
            "firm": pd.factorize(
                reg["StkcdRecall"],
                sort=False,
            )[0],
            "date": pd.factorize(
                reg["Trddt"],
                sort=False,
            )[0],
        },
        index=reg.index,
    )

    model = AbsorbingLS(
        dependent=reg[outcome].astype("float64"),
        exog=reg[RHS_COLS].astype("float64"),
        absorb=absorb,
        drop_absorbed=True,
    )

    result = model.fit(
        cov_type="clustered",
        clusters=clusters,
        debiased=True,
    )

    beta = float(result.params[CUE_COL])
    standard_error = float(
        result.std_errors[CUE_COL]
    )
    t_value = float(result.tstats[CUE_COL])
    p_value = float(result.pvalues[CUE_COL])

    model_info = {
        "因变量": outcome,
        "线索系数": beta,
        "聚类标准误": standard_error,
        "单位": "百分点",
        "t值": t_value,
        "p值": p_value,
        "显著性": significance_stars(p_value),
        "控制变量": "是",
        "日期固定效应": "是",
        "被提示公司×年度季度固定效应": "是",
        "回归样本量": int(result.nobs),
        "被提示公司聚类数": int(
            reg["StkcdRecall"].nunique()
        ),
        "日期聚类数": int(
            reg["Trddt"].nunique()
        ),
        "删除单例组观测数": singleton_removed,
        "R方": float(result.rsquared),
    }

    coefficient_rows = []

    for variable_order, variable in enumerate(
        RHS_COLS,
        start=1,
    ):
        estimated = variable in result.params.index

        if estimated:
            variable_beta = float(
                result.params[variable]
            )
            variable_standard_error = float(
                result.std_errors[variable]
            )
            variable_t_value = float(
                result.tstats[variable]
            )
            variable_p_value = float(
                result.pvalues[variable]
            )
            stars = significance_stars(
                variable_p_value
            )
            status = "已估计"
        else:
            variable_beta = np.nan
            variable_standard_error = np.nan
            variable_t_value = np.nan
            variable_p_value = np.nan
            stars = ""
            status = "被固定效应吸收"

        coefficient_rows.append({
            "因变量": outcome,
            "变量": variable,
            "变量顺序": variable_order,
            "变量类型": (
                "线索事件"
                if variable == CUE_COL
                else "控制变量"
            ),
            "估计状态": status,
            "系数": variable_beta,
            "聚类标准误": variable_standard_error,
            "单位": "百分点",
            "t值": variable_t_value,
            "p值": variable_p_value,
            "显著性": stars,
            "控制变量": "是",
            "日期固定效应": "是",
            "被提示公司×年度季度固定效应": "是",
            "回归样本量": int(result.nobs),
            "R方": float(result.rsquared),
        })

    del result, model, absorb, clusters, reg
    gc.collect()

    return model_info, coefficient_rows


def validate_plot_columns(
    frame: pd.DataFrame,
    required: list[str],
    frame_name: str,
) -> None:
    """检查图片式回归表所需字段是否齐全。"""
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{frame_name}缺少绘图字段：{'、'.join(missing)}"
        )


def format_estimate(
    coefficient: float,
    standard_error: float,
    p_value: float,
    digits: int = 4,
) -> tuple[str, str]:
    """把系数和括号内标准误格式化为论文回归表形式。"""
    if pd.isna(coefficient) or pd.isna(standard_error):
        return "", ""

    coefficient_text = (
        f"{coefficient:.{digits}f}"
        f"{significance_stars(p_value)}"
    )
    standard_error_text = (
        f"({standard_error:.{digits}f})"
    )
    return coefficient_text, standard_error_text


def variable_label(variable: str) -> str:
    """把回归变量名转换为图片表格中的中文名称。"""
    fixed_labels = {
        CUE_COL: "线索事件",
        "ownEA": "自身盈余公告 [t]",
        "ownMajorDisclosure": "其他专题事件 [t]",
        "ownPerformanceForecast": "业绩预告或快报 [t]",
        "aesIncrease": "被提示公司新闻情绪上升 [t]",
        "aesDecrease": "被提示公司新闻情绪下降 [t]",
        "logIncreaseAEV": "log(1 + 新闻量增加)",
        "logDecreaseAEV": "log(1 + 新闻量减少)",
    }
    if variable in fixed_labels:
        return fixed_labels[variable]

    lag_match = re.fullmatch(r"ownEA_tm(\d+)", variable)
    if lag_match:
        return (
            f"自身盈余公告 [t-{lag_match.group(1)}]"
        )

    lead_match = re.fullmatch(r"ownEA_tp(\d+)", variable)
    if lead_match:
        return (
            f"自身盈余公告 [t+{lead_match.group(1)}]"
        )

    return variable


def format_observations(value: float) -> str:
    """格式化回归样本量。"""
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def format_r_squared(value: float) -> str:
    """将 R 方保留四位小数。"""
    if pd.isna(value):
        return ""
    return f"{value:.4f}"


def order_appendix_robustness_models(
    summary: pd.DataFrame,
) -> list[pd.Series]:
    """按照两个订单失衡窗口排列回归模型。"""
    actual_windows = set(
        summary["订单失衡窗口"]
        .dropna()
        .astype(str)
        .tolist()
    )
    if actual_windows != set(WINDOW_ORDER):
        raise ValueError(
            "模型汇总中的订单失衡窗口不完整："
            f"预期为 {WINDOW_ORDER}，"
            f"实际为 {sorted(actual_windows)}。"
        )

    ordered = []

    for window in WINDOW_ORDER:
        matched = summary.loc[
            summary["订单失衡窗口"].eq(window)
        ]
        if len(matched) != 1:
            raise ValueError(
                f"订单失衡窗口 {window} 应恰好有一行，"
                f"实际为 {len(matched)} 行。"
            )
        ordered.append(matched.iloc[0])

    return ordered


def draw_rule(
    axes,
    y: float,
    linewidth: float = 0.8,
) -> None:
    """绘制论文式表格中的水平线。"""
    axes.plot(
        [0.02, 0.985],
        [y, y],
        color="#222222",
        linewidth=linewidth,
        solid_capstyle="butt",
        clip_on=False,
    )


def draw_appendix_robustness_date_fe_table(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制加入日期固定效应的一张稳健性附表完整系数图片。"""
    validate_plot_columns(
        summary,
        [
            "附表编号",
            "衡量口径",
            "订单失衡窗口",
            "散户识别方法",
            "控制变量",
            "日期固定效应",
            "被提示公司×年度季度固定效应",
            "回归样本量",
            "R方",
        ],
        "模型汇总结果",
    )
    validate_plot_columns(
        details,
        [
            "附表编号",
            "衡量口径",
            "订单失衡窗口",
            "变量",
            "变量顺序",
            "系数",
            "聚类标准误",
            "p值",
        ],
        "系数明细结果",
    )

    table_numbers = (
        summary["附表编号"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if len(table_numbers) != 1:
        raise ValueError(
            "每张稳健性附表图片应只包含一个附表编号，"
            f"实际为：{table_numbers}。"
        )
    table_number = table_numbers[0]

    detail_table_numbers = (
        details["附表编号"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if detail_table_numbers != table_numbers:
        raise ValueError(
            "模型汇总与系数明细的附表编号不一致："
            f"汇总为 {table_numbers}，"
            f"明细为 {detail_table_numbers}。"
        )

    measure_labels = (
        summary["衡量口径"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if len(measure_labels) != 1:
        raise ValueError(
            "每张稳健性附表图片应只包含一种衡量口径，"
            f"实际为：{measure_labels}。"
        )
    measure_label = measure_labels[0]

    detail_measure_labels = (
        details["衡量口径"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if detail_measure_labels != measure_labels:
        raise ValueError(
            "模型汇总与系数明细的衡量口径不一致："
            f"汇总为 {measure_labels}，"
            f"明细为 {detail_measure_labels}。"
        )

    detail_windows = set(
        details["订单失衡窗口"]
        .dropna()
        .astype(str)
        .tolist()
    )
    if detail_windows != set(WINDOW_ORDER):
        raise ValueError(
            "系数明细中的订单失衡窗口不完整："
            f"预期为 {WINDOW_ORDER}，"
            f"实际为 {sorted(detail_windows)}。"
        )

    model_rows = order_appendix_robustness_models(summary)
    lookup = {
        (row["订单失衡窗口"], row["变量"]): row
        for _, row in details.iterrows()
    }
    variable_order = (
        details[["变量", "变量顺序"]]
        .drop_duplicates()
        .sort_values("变量顺序")["变量"]
        .tolist()
    )

    table_rows = []

    for variable_index, variable in enumerate(
        variable_order
    ):
        coefficient_values = []
        standard_error_values = []

        for model_row in model_rows:
            detail_row = lookup.get(
                (
                    model_row["订单失衡窗口"],
                    variable,
                )
            )

            if detail_row is None:
                coefficient, standard_error = "", ""
            else:
                coefficient, standard_error = (
                    format_estimate(
                        detail_row["系数"],
                        detail_row["聚类标准误"],
                        detail_row["p值"],
                    )
                )

            coefficient_values.append(coefficient)
            standard_error_values.append(
                standard_error
            )

        table_rows.extend([
            {
                "kind": "coefficient",
                "label": variable_label(variable),
                "values": coefficient_values,
                "bold": variable == CUE_COL,
                "line_before": variable_index == 1,
            },
            {
                "kind": "standard_error",
                "label": "",
                "values": standard_error_values,
                "color": "#444444",
            },
        ])

    table_rows.extend([
        {
            "kind": "stat",
            "label": "控制变量",
            "values": [
                str(row["控制变量"])
                for row in model_rows
            ],
            "line_before": True,
        },
        {
            "kind": "stat",
            "label": "日期固定效应",
            "values": [
                str(row["日期固定效应"])
                for row in model_rows
            ],
        },
        {
            "kind": "stat",
            "label": "被提示公司×年度季度固定效应",
            "values": [
                str(
                    row[
                        "被提示公司×年度季度固定效应"
                    ]
                )
                for row in model_rows
            ],
        },
        {
            "kind": "stat",
            "label": "样本量",
            "values": [
                format_observations(row["回归样本量"])
                for row in model_rows
            ],
        },
        {
            "kind": "stat",
            "label": "R²",
            "values": [
                format_r_squared(row["R方"])
                for row in model_rows
            ],
        },
    ])

    notes = [
        (
            "注：所有模型均同时吸收被提示公司×年度季度固定效应和"
            "交易日期固定效应；括号内为按被提示公司和交易日期"
            "双向聚类的标准误。"
        ),
        "*、**、*** 分别表示在 10%、5% 和 1% 水平上显著。",
    ]

    row_heights = {
        "coefficient": 0.88,
        "standard_error": 0.76,
        "stat": 0.88,
    }
    title_height = 5.0
    header_height = 2.15
    notes_height = 0.85 * len(notes) + 0.75
    body_height = sum(
        row_heights[row["kind"]]
        for row in table_rows
    )
    total_height = (
        title_height
        + header_height
        + body_height
        + notes_height
    )
    figure_height = max(7.5, total_height * 0.29)

    figure, axes = plt.subplots(
        figsize=(12.0, figure_height)
    )
    figure.patch.set_facecolor("white")
    axes.set_facecolor("white")
    axes.set_xlim(0, 1)
    axes.set_ylim(total_height, 0)
    axes.axis("off")

    label_right = 0.34
    data_left = label_right + 0.012
    data_right = 0.975
    column_width = (
        data_right - data_left
    ) / len(WINDOW_ORDER)
    column_centers = [
        data_left + (index + 0.5) * column_width
        for index in range(len(WINDOW_ORDER))
    ]

    axes.text(
        0.02,
        0.38,
        f"附表 {table_number}（加入日期固定效应）",
        ha="left",
        va="top",
        fontproperties=FONT_BOLD,
        fontsize=21.0,
    )
    axes.text(
        0.02,
        2.02,
        f"线索事件与散户订单失衡：{measure_label}口径",
        ha="left",
        va="top",
        fontproperties=FONT_BOLD,
        fontsize=19.0,
    )
    axes.text(
        0.02,
        3.52,
        (
            f"因变量：按{measure_label}衡量的小单订单失衡；"
            "当前复现采用A股小单口径；"
            "系数与标准误的单位均为百分点"
        ),
        ha="left",
        va="top",
        color="#444444",
        fontproperties=FONT,
        fontsize=10.5,
    )

    header_top = title_height
    draw_rule(axes, header_top, linewidth=1.0)
    axes.text(
        0.046,
        header_top + 1.24,
        "变量",
        ha="left",
        va="center",
        fontproperties=FONT_BOLD,
        fontsize=9.6,
    )
    axes.text(
        sum(column_centers) / len(column_centers),
        header_top + 0.57,
        f"小单订单失衡（{measure_label}口径）",
        ha="center",
        va="center",
        fontproperties=FONT_BOLD,
        fontsize=9.2,
    )

    for center, label in zip(
        column_centers,
        WINDOW_ORDER,
    ):
        axes.text(
            center,
            header_top + 1.42,
            label,
            ha="center",
            va="center",
            fontproperties=FONT_BOLD,
            fontsize=9.2,
        )

    body_top = header_top + header_height
    draw_rule(axes, body_top, linewidth=0.9)
    y = body_top

    for row in table_rows:
        kind = row["kind"]
        height = row_heights[kind]

        if row.get("line_before"):
            draw_rule(axes, y, linewidth=0.55)

        center_y = y + height / 2
        font = (
            FONT_BOLD
            if row.get("bold", False)
            else FONT
        )
        color = row.get("color", "#111111")
        label_x = (
            0.058
            if kind == "standard_error"
            else 0.046
        )

        axes.text(
            label_x,
            center_y,
            row.get("label", ""),
            ha="left",
            va="center",
            color=color,
            fontproperties=font,
            fontsize=9.2,
        )

        for center, value in zip(
            column_centers,
            row.get("values", []),
        ):
            axes.text(
                center,
                center_y,
                value,
                ha="center",
                va="center",
                color=color,
                fontproperties=font,
                fontsize=9.2,
            )

        y += height

    draw_rule(axes, y, linewidth=1.0)

    note_y = y + 0.72
    for note in notes:
        axes.text(
            0.02,
            note_y,
            note,
            ha="left",
            va="top",
            color="#444444",
            fontproperties=FONT,
            fontsize=8.7,
        )
        note_y += 0.82

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.12,
    )
    plt.close(figure)


def main() -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = read_appendix_robustness_date_fe_data()
    for metric_spec in OIB_METRIC_SPECS:
        table_number = metric_spec["table_number"]
        measure_label = metric_spec["measure_label"]
        model_rows = []
        coefficient_rows = []

        for window_suffix, window_label in WINDOW_SPECS:
            outcome = (
                f"{metric_spec['prefix']}_avg_t_"
                f"{window_suffix}_w"
            )
            print(
                "正在运行加入日期固定效应的"
                f"附表 {table_number}："
                f"{measure_label}口径，{window_label}"
            )

            model_info, current_coefficients = (
                run_appendix_robustness_date_fe_regression(
                    data,
                    outcome,
                )
            )

            model_metadata = {
                "结果表": (
                    f"附表{table_number}"
                    "（加入日期固定效应）"
                ),
                "附表编号": table_number,
                "衡量口径": measure_label,
                "订单失衡窗口": window_label,
                "散户识别方法": "A股小单口径",
            }

            model_rows.append({
                **model_metadata,
                **model_info,
            })

            coefficient_rows.extend(
                {
                    **model_metadata,
                    **row,
                }
                for row in current_coefficients
            )

        results = pd.DataFrame(model_rows)
        coefficients = pd.DataFrame(coefficient_rows)

        results.to_csv(
            metric_spec["results_path"],
            index=False,
            encoding="utf-8-sig",
        )

        coefficients.to_csv(
            metric_spec["coefficients_path"],
            index=False,
            encoding="utf-8-sig",
        )

        draw_appendix_robustness_date_fe_table(
            summary=results,
            details=coefficients,
            output_path=metric_spec["image_path"],
        )

        print(
            f"\n附表 {table_number}：加入日期固定效应的"
            f"{measure_label}订单失衡结果"
        )
        print(results.to_string(index=False))
        print(
            f"\n附表 {table_number} 模型汇总已保存："
            f"{metric_spec['results_path']}"
        )
        print(
            f"附表 {table_number} 全部系数明细已保存："
            f"{metric_spec['coefficients_path']}"
        )
        print(
            f"附表 {table_number} 图片式表格已保存："
            f"{metric_spec['image_path']}"
        )

        del results, coefficients
        gc.collect()

    del data
    gc.collect()


if __name__ == "__main__":
    main()
