# 本文件由旧零散回归目录迁移；仅调整项目定位、公共工具引用和任务输出位置。
"""单独运行表2收益回归主结果。

本脚本与“回归.py”中的表2保持相同的样本、变量、回归规格、
固定效应、聚类方式和结果文件名，但不依赖表1或其他回归部分。
"""

from pathlib import Path
import gc
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from matplotlib.font_manager import FontProperties


# =============================================================================
# 路径设置
# =============================================================================

def find_project_root() -> Path:
    """沿脚本祖先寻找项目标识，迁移目录后不依赖固定父目录层数。"""
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "AGENTS.md").is_file()
            and (candidate / "code").is_dir()
            and (candidate / "中间结果").is_dir()
        ):
            return candidate
    raise FileNotFoundError("未找到同时包含项目说明、代码和中间结果的项目根目录。")


ROOT = find_project_root()

PATTERN_PATH = (
    ROOT
    / "中间结果"
    / "04_合并数据"
    / "Pattern样本_最终回归面板.csv"
)

FULL_PATH = (
    ROOT
    / "中间结果"
    / "04_合并数据"
    / "Full样本_最终回归面板.csv"
)

RESULT_DIR = ROOT / "图表结果/99_历史与待核对/独立脚本待核对/表2_收益回归"
TABLE2_RESULTS_PATH = RESULT_DIR / "表2_模型汇总.csv"
TABLE2_COEFFICIENTS_PATH = RESULT_DIR / "表2_系数明细.csv"
# 独立脚本图片先归入待核对目录，数值核对后再与正式主表归并。
IMAGE_DIR = RESULT_DIR
TABLE2_IMAGE_PATHS = {
    "pattern": (
        IMAGE_DIR / "table2_收益回归_完整系数版_模式公司主样本_独立脚本.png",
        IMAGE_DIR / "table2_收益回归_完整系数版_模式公司主样本_独立脚本_续页.png",
    ),
    "full": (
        IMAGE_DIR / "table2_收益回归_完整系数版_完整样本_独立脚本.png",
        IMAGE_DIR / "table2_收益回归_完整系数版_完整样本_独立脚本_续页.png",
    ),
}

FONT_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
if FONT_PATH.exists():
    FONT = FontProperties(fname=str(FONT_PATH))
    FONT_BOLD = FontProperties(fname=str(FONT_PATH), weight="bold")
else:
    FONT = FontProperties(family="sans-serif")
    FONT_BOLD = FontProperties(family="sans-serif", weight="bold")

plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# 样本期和变量设置
# =============================================================================

# 默认研究期：2005—2025年（含首尾年）。
START_YEAR = 2005
END_YEAR = 2025

OWN_EA_COLS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)

DISCLOSURE_COLS = [
    "ownOtherEventER",
    "ownPerformanceForecastBrief",
]

MEDIA_COLS = [
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]

RETURN_COLS = [
    "chAdjreturn_t_tp1_Recall_w",
    "chAdjreturn_t_tp10_Recall_w",
]

TABLE2_FULL_CONTROLS = (
    OWN_EA_COLS
    + DISCLOSURE_COLS
    + MEDIA_COLS
)

TABLE2_SPECIFICATIONS = [
    {
        "规格": "(1)",
        "规格说明": "无控制变量、无固定效应",
        "控制变量": [],
        "固定效应": [],
    },
    {
        "规格": "(2)",
        "规格说明": "自身盈余公告控制、无固定效应",
        "控制变量": OWN_EA_COLS,
        "固定效应": [],
    },
    {
        "规格": "(3)",
        "规格说明": "完整控制变量、无固定效应",
        "控制变量": TABLE2_FULL_CONTROLS,
        "固定效应": [],
    },
    {
        "规格": "(4)",
        "规格说明": "完整控制变量、日期固定效应",
        "控制变量": TABLE2_FULL_CONTROLS,
        "固定效应": ["date"],
    },
    {
        "规格": "(5)",
        "规格说明": "完整控制变量、公司×年度季度固定效应",
        "控制变量": TABLE2_FULL_CONTROLS,
        "固定效应": ["firm_yearquarter"],
    },
    {
        "规格": "(6)",
        "规格说明": "完整控制变量、日期和公司×年度季度固定效应",
        "控制变量": TABLE2_FULL_CONTROLS,
        "固定效应": ["date", "firm_yearquarter"],
    },
]

TABLE2_OUTCOMES = [
    (
        "chAdjreturn_t_tp1_Recall_w",
        "[t, t+1]",
    ),
    (
        "chAdjreturn_t_tp10_Recall_w",
        "[t, t+10]",
    ),
]


def significance_stars(p_value):
    """根据双侧检验的 p 值生成显著性星号。"""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def drop_fe_singletons_iteratively(reg, fe_code_cols):
    """递归删除固定效应单例组。"""
    if not fe_code_cols:
        return reg, 0

    total_removed = 0

    while True:
        singleton = np.zeros(len(reg), dtype=bool)

        for col in fe_code_cols:
            codes = reg[col].to_numpy(
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


def build_table2_coefficient_rows(
    result,
    x_cols,
    cue_col,
    outcome,
    window_label,
    specification,
    fe_names,
):
    """整理表2单个模型中线索变量和全部控制变量的估计结果。"""
    rows = []

    for variable_order, variable in enumerate(x_cols, start=1):
        estimated = variable in result.params.index

        if estimated:
            beta = float(result.params[variable])
            standard_error = float(result.std_errors[variable])
            t_value = float(result.tstats[variable])
            p_value = float(result.pvalues[variable])
            stars = significance_stars(p_value)
            status = "已估计"
        else:
            beta = np.nan
            standard_error = np.nan
            t_value = np.nan
            p_value = np.nan
            stars = ""
            status = "被固定效应吸收"

        rows.append({
            "收益窗口": window_label,
            "因变量": outcome,
            "规格": specification["规格"],
            "规格说明": specification["规格说明"],
            "变量": variable,
            "变量顺序": variable_order,
            "变量类型": (
                "线索事件"
                if variable == cue_col
                else "控制变量"
            ),
            "估计状态": status,
            "系数": beta,
            "聚类标准误": standard_error,
            "系数（百分点）": beta * 100,
            "聚类标准误（百分点）": standard_error * 100,
            "t值": t_value,
            "p值": p_value,
            "显著性": stars,
            "日期固定效应": (
                "是" if "date" in fe_names else "否"
            ),
            "被提示公司×年度季度固定效应": (
                "是"
                if "firm_yearquarter" in fe_names
                else "否"
            ),
            "回归样本量": int(result.nobs),
            "R方": float(result.rsquared),
        })

    return rows


def run_return_regression(
    data,
    outcome,
    window_label,
    specification,
):
    """运行表2中的一个收益回归。"""
    cue_col = "quarterLag1size90plus"
    control_cols = specification["控制变量"]
    fe_names = specification["固定效应"]

    x_cols = [cue_col] + control_cols

    needed = [outcome, "StkcdRecall", "Trddt"] + x_cols

    if "firm_yearquarter" in fe_names:
        needed.append("yearQuarter")

    needed = list(dict.fromkeys(needed))
    reg = data[needed].dropna().copy()

    fe_code_cols = []

    if "firm_yearquarter" in fe_names:
        reg["_fe_firm_yearquarter"] = (
            reg.groupby(
                ["StkcdRecall", "yearQuarter"],
                observed=True,
                sort=False,
            )
            .ngroup()
            .astype("int64")
        )
        fe_code_cols.append("_fe_firm_yearquarter")

    if "date" in fe_names:
        reg["_fe_date"] = pd.factorize(
            reg["Trddt"],
            sort=False,
        )[0].astype("int64")
        fe_code_cols.append("_fe_date")

    reg, singleton_removed = drop_fe_singletons_iteratively(
        reg,
        fe_code_cols,
    )

    absorb = None

    if fe_code_cols:
        absorb = pd.DataFrame(
            {
                col: pd.Categorical(reg[col])
                for col in fe_code_cols
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

    X = reg[x_cols].astype("float64")
    X.insert(0, "const", 1.0)

    model = AbsorbingLS(
        dependent=reg[outcome].astype("float64"),
        exog=X,
        absorb=absorb,
        drop_absorbed=True,
    )

    result = model.fit(
        cov_type="clustered",
        clusters=clusters,
        debiased=True,
    )

    if cue_col not in result.params.index:
        raise ValueError(
            f"{outcome} 的 {specification['规格']} 中，"
            f"{cue_col} 被固定效应完全吸收。"
        )

    beta = float(result.params[cue_col])
    standard_error = float(result.std_errors[cue_col])
    t_value = float(result.tstats[cue_col])
    p_value = float(result.pvalues[cue_col])

    regression_row = {
        "收益窗口": window_label,
        "因变量": outcome,
        "规格": specification["规格"],
        "规格说明": specification["规格说明"],
        "线索系数": beta,
        "聚类标准误": standard_error,
        "线索系数（百分点）": beta * 100,
        "聚类标准误（百分点）": standard_error * 100,
        "t值": t_value,
        "p值": p_value,
        "显著性": significance_stars(p_value),
        "回归样本量": int(result.nobs),
        "被提示公司聚类数": int(reg["StkcdRecall"].nunique()),
        "日期聚类数": int(reg["Trddt"].nunique()),
        "删除单例组观测数": singleton_removed,
        "R方": float(result.rsquared),
    }

    coefficient_rows = build_table2_coefficient_rows(
        result=result,
        x_cols=x_cols,
        cue_col=cue_col,
        outcome=outcome,
        window_label=window_label,
        specification=specification,
        fe_names=fe_names,
    )

    del result, model, absorb, clusters, X, reg
    gc.collect()

    return regression_row, coefficient_rows


def run_table2_sample(data, sample_name, sample_label):
    """在一个样本上运行两个收益窗口和六种回归规格。"""
    model_rows = []
    coefficient_rows = []

    for outcome, window_label in TABLE2_OUTCOMES:
        for specification in TABLE2_SPECIFICATIONS:
            print(
                f"正在运行表2：{sample_label}，"
                f"{window_label}，"
                f"规格 {specification['规格']}"
            )

            model_row, model_coefficient_rows = run_return_regression(
                data=data,
                outcome=outcome,
                window_label=window_label,
                specification=specification,
            )

            model_rows.append({
                "样本": sample_name,
                "样本说明": sample_label,
                **model_row,
            })

            coefficient_rows.extend(
                {
                    "样本": sample_name,
                    "样本说明": sample_label,
                    **row,
                }
                for row in model_coefficient_rows
            )

    return pd.DataFrame(model_rows), pd.DataFrame(coefficient_rows)


def read_table2_data(input_path):
    """读取一个样本的表2回归字段并完成统一清洗。"""
    input_cols = list(dict.fromkeys(
        [
            "StkcdRecall",
            "Trddt",
            "year",
            "yearQuarter",
            "quarterLag1size90plus",
        ]
        + RETURN_COLS
        + TABLE2_FULL_CONTROLS
    ))

    numeric_cols = [
        col
        for col in input_cols
        if col not in {"StkcdRecall", "Trddt", "yearQuarter"}
    ]

    dtype = {
        "StkcdRecall": "string",
        "yearQuarter": "string",
    }
    dtype.update({col: "float32" for col in numeric_cols})

    print(f"读取表2回归面板：{input_path.name}……")
    data = pd.read_csv(
        input_path,
        usecols=input_cols,
        dtype=dtype,
        parse_dates=["Trddt"],
    )

    data["StkcdRecall"] = (
        data["StkcdRecall"]
        .str.strip()
        .str.zfill(6)
    )

    if START_YEAR is not None:
        data = data.loc[data["year"] >= START_YEAR].copy()
    if END_YEAR is not None:
        data = data.loc[data["year"] <= END_YEAR].copy()

    if data.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError(
            f"{input_path.name} 的 StkcdRecall + Trddt 不是唯一键。"
        )

    data["StkcdRecall"] = data["StkcdRecall"].astype("category")
    data["yearQuarter"] = data["yearQuarter"].astype("category")

    print(f"表2样本行数：{len(data):,}")
    return data


def _numeric_value(value):
    """将结果表中的单个值安全转换为数值。"""
    return pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]


def _format_table2_estimate(row):
    """将表2系数和双向聚类标准误转成图片表格文字。"""
    coefficient = _numeric_value(row.get("系数（百分点）", np.nan))
    standard_error = _numeric_value(
        row.get("聚类标准误（百分点）", np.nan)
    )
    if pd.isna(coefficient):
        coefficient = _numeric_value(row.get("系数", np.nan)) * 100
    if pd.isna(standard_error):
        standard_error = _numeric_value(row.get("聚类标准误", np.nan)) * 100
    p_value = _numeric_value(row.get("p值", np.nan))

    if pd.isna(coefficient) or pd.isna(standard_error):
        return "", ""

    stars = ""
    if pd.notna(p_value):
        if p_value < 0.01:
            stars = "***"
        elif p_value < 0.05:
            stars = "**"
        elif p_value < 0.10:
            stars = "*"
    return f"{coefficient:.4f}{stars}", f"({standard_error:.4f})"


def _table2_variable_label(variable):
    """将内部变量名转换为表2完整系数表的显示名称。"""
    labels = {
        "quarterLag1size90plus": "线索事件",
        "ownEA": "自身盈余公告 [t]",
        "ownOtherEventER": "其他专题事件 [t]",
        "ownPerformanceForecastBrief": "业绩预告或快报 [t]",
        "aesIncrease": "被提示公司新闻情绪上升 [t]",
        "aesDecrease": "被提示公司新闻情绪下降 [t]",
        "logIncreaseAEV": "log(1 + 新闻量增加)",
        "logDecreaseAEV": "log(1 + 新闻量减少)",
    }
    if variable in labels:
        return labels[variable]

    lag = re.fullmatch(r"ownEA_tm(\d+)", str(variable))
    if lag:
        return f"自身盈余公告 [t-{lag.group(1)}]"
    lead = re.fullmatch(r"ownEA_tp(\d+)", str(variable))
    if lead:
        return f"自身盈余公告 [t+{lead.group(1)}]"
    return str(variable)


def _ordered_table2_models(summary, sample):
    """按收益窗口和规格顺序取得一个样本的十二个模型。"""
    selected = summary.loc[summary["样本"].eq(sample)]
    model_rows = []
    for window in ("[t, t+1]", "[t, t+10]"):
        for specification in ("(1)", "(2)", "(3)", "(4)", "(5)", "(6)"):
            matched = selected.loc[
                selected["收益窗口"].eq(window)
                & selected["规格"].eq(specification)
            ]
            if len(matched) != 1:
                raise ValueError(
                    "表2完整系数表需要每个样本、收益窗口和规格各一条结果："
                    f"样本={sample}，窗口={window}，规格={specification}。"
                )
            model_rows.append(matched.iloc[0])
    return model_rows


def _table2_specification_setting(model_rows, setting):
    """根据规格编号整理完整系数表底部的模型设定行。"""
    values = []
    for row in model_rows:
        number = int(str(row["规格"]).strip("()"))
        if setting == "own":
            enabled = number >= 2
        elif setting == "other":
            enabled = number >= 3
        elif setting == "date":
            enabled = number in {4, 6}
        elif setting == "firm_quarter":
            enabled = number in {5, 6}
        else:
            raise ValueError(f"未知的表2模型设定：{setting}")
        values.append("是" if enabled else "否")
    return values


def _save_table2_page(
    model_rows,
    details_lookup,
    variables,
    output_path,
    *,
    sample_label,
    continued,
    include_model_stats,
):
    """按参考文档的横向页面（297×210毫米）绘制表2三线表。"""
    column_labels = [row["规格"] for row in model_rows]
    group_headers = [
        (0, 5, "收益窗口 [t, t+1]"),
        (6, 11, "收益窗口 [t, t+10]"),
    ]
    table_rows = []

    for variable in variables:
        coefficients = []
        standard_errors = []
        for row in model_rows:
            detail = details_lookup.get(
                (row["收益窗口"], row["规格"], variable)
            )
            if detail is None:
                coefficient, standard_error = "", ""
            else:
                coefficient, standard_error = _format_table2_estimate(detail)
            coefficients.append(coefficient)
            standard_errors.append(standard_error)

        table_rows.extend([
            {
                "kind": "coefficient",
                "label": _table2_variable_label(variable),
                "values": coefficients,
                "bold": variable == "quarterLag1size90plus",
                "line_before": variable == "ownEA_tm10",
            },
            {
                "kind": "standard_error",
                "label": "",
                "values": standard_errors,
                "color": "#444444",
            },
        ])

    if include_model_stats:
        table_rows.extend([
            {
                "kind": "stat",
                "label": "自身盈余公告控制",
                "values": _table2_specification_setting(model_rows, "own"),
                "line_before": True,
            },
            {
                "kind": "stat",
                "label": "其他控制变量",
                "values": _table2_specification_setting(model_rows, "other"),
            },
            {
                "kind": "stat",
                "label": "日期固定效应",
                "values": _table2_specification_setting(model_rows, "date"),
            },
            {
                "kind": "stat",
                "label": "被提示公司×年度季度固定效应",
                "values": _table2_specification_setting(model_rows, "firm_quarter"),
            },
            {
                "kind": "stat",
                "label": "样本量",
                "values": [
                    f"{int(_numeric_value(row['回归样本量'])):,}"
                    for row in model_rows
                ],
            },
            {
                "kind": "stat",
                "label": r"$R^2$",
                "values": [
                    f"{_numeric_value(row['R方']):.4f}"
                    for row in model_rows
                ],
            },
        ])

    notes = (
        [
            "注：空白单元格表示相应规格未纳入该控制变量；括号内为按被提示公司和交易日期双向聚类的标准误。",
            "*、**、*** 分别表示在 10%、5% 和 1% 水平上显著。",
        ]
        if include_model_stats
        else ["续见下一页。"]
    )
    row_heights = {
        "coefficient": 0.88,
        "standard_error": 0.76,
        "stat": 0.88,
    }
    body_fontsize = 8.5
    title_height = 3.55
    header_height = 2.78
    notes_height = 0.85 * len(notes) + 0.75
    body_height = sum(row_heights[row["kind"]] for row in table_rows)
    total_height = title_height + header_height + body_height + notes_height

    figure, axes = plt.subplots(
        figsize=(297.0 / 25.4, 210.0 / 25.4),
        dpi=254,
    )
    figure.subplots_adjust(left=0.02, right=0.985, top=0.97, bottom=0.03)
    figure.patch.set_facecolor("white")
    axes.set_facecolor("white")
    axes.set_xlim(0, 1)
    axes.set_ylim(total_height, 0)
    axes.axis("off")

    # 两个收益窗口各占六列，组间留白与参考表一致。
    data_left = 0.205 + 0.012
    data_right = 0.975
    group_boundaries = {end for _, end, _ in group_headers[:-1]}
    group_gap = 0.018
    available_width = data_right - data_left - group_gap * len(group_boundaries)
    column_width = available_width / len(column_labels)
    column_lefts = []
    column_rights = []
    column_centers = []
    cursor = data_left
    for index in range(len(column_labels)):
        left = cursor
        right = left + column_width
        column_lefts.append(left)
        column_rights.append(right)
        column_centers.append((left + right) / 2)
        cursor = right + (group_gap if index in group_boundaries else 0.0)

    def draw_rule(y, linewidth=0.8):
        axes.plot(
            [0.02, 0.985],
            [y, y],
            color="#222222",
            linewidth=linewidth,
            solid_capstyle="butt",
            clip_on=False,
        )

    title = "表 2（续）" if continued else "表 2"
    axes.text(
        0.02,
        0.18,
        title,
        ha="left",
        va="top",
        fontsize=11.5,
        fontproperties=FONT_BOLD,
    )
    axes.text(
        0.02,
        0.92,
        "线索事件与被提示公司收益",
        ha="left",
        va="top",
        fontsize=14.5,
        fontproperties=FONT_BOLD,
    )
    axes.text(
        0.02,
        2.12,
        sample_label,
        ha="left",
        va="top",
        fontsize=9.1,
        color="#444444",
        fontproperties=FONT,
    )

    header_top = title_height
    draw_rule(header_top, linewidth=1.0)
    axes.text(
        0.046,
        header_top + 1.55,
        "变量",
        ha="left",
        va="center",
        fontproperties=FONT_BOLD,
        fontsize=body_fontsize + 0.4,
    )
    axes.text(
        (column_centers[0] + column_centers[-1]) / 2,
        header_top + 0.42,
        "因变量：被提示公司的特征调整收益率（%）",
        ha="center",
        va="center",
        color="#333333",
        fontproperties=FONT,
        fontsize=max(8.0, body_fontsize - 0.1),
    )
    for start, end, label in group_headers:
        axes.text(
            (column_centers[start] + column_centers[end]) / 2,
            header_top + 1.18,
            label,
            ha="center",
            va="center",
            fontproperties=FONT_BOLD,
            fontsize=body_fontsize + 0.15,
        )
        axes.plot(
            [
                column_lefts[start] + 0.08 * column_width,
                column_rights[end] - 0.08 * column_width,
            ],
            [header_top + 1.62, header_top + 1.62],
            color="#555555",
            linewidth=0.5,
            solid_capstyle="butt",
            clip_on=False,
        )
    for center, label in zip(column_centers, column_labels):
        axes.text(
            center,
            header_top + 2.16,
            label,
            ha="center",
            va="center",
            fontproperties=FONT_BOLD,
            fontsize=max(7.5, body_fontsize - 0.45),
        )

    body_top = header_top + header_height
    draw_rule(body_top, linewidth=0.9)
    y = body_top
    for row in table_rows:
        kind = row["kind"]
        height = row_heights[kind]
        if row.get("line_before"):
            draw_rule(y, linewidth=0.55)

        center_y = y + height / 2
        font = FONT_BOLD if row.get("bold", False) else FONT
        color = row.get("color", "#111111")
        label_x = 0.046 if kind != "standard_error" else 0.058
        axes.text(
            label_x,
            center_y,
            row["label"],
            ha="left",
            va="center",
            color=color,
            fontproperties=font,
            fontsize=body_fontsize,
        )
        for center, value in zip(column_centers, row["values"]):
            axes.text(
                center,
                center_y,
                value,
                ha="center",
                va="center",
                color=color,
                fontproperties=font,
                fontsize=body_fontsize,
            )
        y += height

    draw_rule(y, linewidth=1.0)
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
            fontsize=max(8.0, body_fontsize - 0.5),
        )
        note_y += 0.82

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=300,
        facecolor="white",
    )
    plt.close(figure)


def draw_table2_full_two_pages(summary, details, sample, output_paths):
    """生成指定样本的表2完整系数两页图片，不依赖外部绘图脚本。"""
    model_rows = _ordered_table2_models(summary, sample)
    selected = details.loc[details["样本"].eq(sample)]
    variables = (
        selected[["变量", "变量顺序"]]
        .drop_duplicates()
        .sort_values("变量顺序")["变量"]
        .tolist()
    )
    split_after = "ownEA_tp3"
    if split_after not in variables:
        raise ValueError("表2完整系数表找不到预定的分页变量 ownEA_tp3。")
    split_index = variables.index(split_after) + 1
    first_variables = variables[:split_index]
    second_variables = variables[split_index:]
    details_lookup = {
        (row["收益窗口"], row["规格"], row["变量"]): row
        for _, row in selected.iterrows()
    }
    sample_label = str(model_rows[0]["样本说明"])
    _save_table2_page(
        model_rows,
        details_lookup,
        first_variables,
        output_paths[0],
        sample_label=sample_label,
        continued=False,
        include_model_stats=False,
    )
    _save_table2_page(
        model_rows,
        details_lookup,
        second_variables,
        output_paths[1],
        sample_label=sample_label,
        continued=True,
        include_model_stats=True,
    )


def main():
    """单独运行表2并保存模型汇总和系数明细。"""
    pattern = read_table2_data(PATTERN_PATH)
    table2_pattern_results, table2_pattern_coefficients = run_table2_sample(
        data=pattern,
        sample_name="pattern",
        sample_label="A. 模式公司主样本",
    )
    del pattern
    gc.collect()

    full = read_table2_data(FULL_PATH)
    table2_full_results, table2_full_coefficients = run_table2_sample(
        data=full,
        sample_name="full",
        sample_label="B. 完整样本",
    )
    del full
    gc.collect()

    table2_results = pd.concat(
        [table2_pattern_results, table2_full_results],
        ignore_index=True,
    )
    table2_coefficients = pd.concat(
        [table2_pattern_coefficients, table2_full_coefficients],
        ignore_index=True,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table2_results.to_csv(
        TABLE2_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    table2_coefficients.to_csv(
        TABLE2_COEFFICIENTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    draw_table2_full_two_pages(
        table2_results,
        table2_coefficients,
        "pattern",
        TABLE2_IMAGE_PATHS["pattern"],
    )
    draw_table2_full_two_pages(
        table2_results,
        table2_coefficients,
        "full",
        TABLE2_IMAGE_PATHS["full"],
    )

    print("\n表2：收益回归结果")
    print(table2_results.to_string(index=False))
    print(f"\n表2结构化结果已保存：{TABLE2_RESULTS_PATH}")
    print(f"表2全部系数明细已保存：{TABLE2_COEFFICIENTS_PATH}")
    print(f"表2完整系数图片已保存：{TABLE2_IMAGE_PATHS['pattern'][0]}")
    print(f"表2完整系数图片已保存：{TABLE2_IMAGE_PATHS['full'][0]}")


if __name__ == "__main__":
    main()
