"""
公共回归表格绘制模块。

从 code/07_实证分析/99_历史与待核对/原综合脚本/绘制回归表格.py 提取原有绘图函数、校验函数与样式常量。
本模块只读取传入的结构化结果并绘图，不估计回归、不重算样本，
也不会在导入时读取各项任务的结果。原综合绘图脚本继续原样保留。
输入和输出路径由各任务的“绘制表格.py”明确指定。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


WINDOW_ORDER = ["[t, t+1]", "[t, t+10]"]
SPEC_ORDER = ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)"]
A4_LANDSCAPE_WIDTH_IN = 297.0 / 25.4
A4_LANDSCAPE_HEIGHT_IN = 210.0 / 25.4
TABLE4_SPEC_ORDER = ["(1)", "(2)", "(3)"]
TABLE4_CUE_ORDER = [
    "quarterLag1size90plus",
    "quarterLag1size50_90",
    "quarterLag1size10_50",
    "quarterLag1size10minus",
]

FONT_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
if FONT_PATH.exists():
    FONT = FontProperties(fname=str(FONT_PATH))
    FONT_BOLD = FontProperties(fname=str(FONT_PATH), weight="bold")
else:
    FONT = FontProperties(family="sans-serif")
    FONT_BOLD = FontProperties(family="sans-serif", weight="bold")


def validate_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    source_name: str,
) -> None:
    """检查结果文件是否包含绘图所需字段。"""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        missing_text = "、".join(missing)
        raise ValueError(f"{source_name}缺少必要字段：{missing_text}")


def significance_stars(p_value: float) -> str:
    """根据双侧检验的 p 值返回显著性星号。"""
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


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
        f"{coefficient:.{digits}f}{significance_stars(p_value)}"
    )
    standard_error_text = f"({standard_error:.{digits}f})"
    return coefficient_text, standard_error_text


def variable_label(variable: str) -> str:
    """把结果文件中的变量名转换为表格中的中文名称。"""
    fixed_labels = {
        "quarterLag1size90plus": "线索事件",
        "ownEA": "自身盈余公告 [t]",
        "ownOtherEventER": "其他专题事件 [t]",
        "ownPerformanceForecastBrief": "业绩预告或快报 [t]",
        "aesIncrease": "被提示公司新闻情绪上升 [t]",
        "aesDecrease": "被提示公司新闻情绪下降 [t]",
        "logIncreaseAEV": "log(1 + 新闻量增加)",
        "logDecreaseAEV": "log(1 + 新闻量减少)",
    }
    if variable in fixed_labels:
        return fixed_labels[variable]

    lag_match = re.fullmatch(r"ownEA_tm(\d+)", variable)
    if lag_match:
        return f"自身盈余公告 [t-{lag_match.group(1)}]"

    lead_match = re.fullmatch(r"ownEA_tp(\d+)", variable)
    if lead_match:
        return f"自身盈余公告 [t+{lead_match.group(1)}]"

    return variable


def read_result(path: Path, required: Sequence[str]) -> pd.DataFrame:
    """读取一个结果文件，并给出清晰的缺失文件或字段提示。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到回归结果文件：{path}\n请先运行生成该结果的回归脚本。"
        )
    frame = pd.read_csv(path)
    validate_columns(frame, required, path.name)
    return frame


def format_observations(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def format_r_squared(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.4f}"


def _draw_rule(ax, y: float, linewidth: float = 0.8) -> None:
    ax.plot(
        [0.02, 0.985],
        [y, y],
        color="#222222",
        linewidth=linewidth,
        solid_capstyle="butt",
        clip_on=False,
    )


def render_publication_table(
    *,
    table_number: str,
    title: str,
    subtitle: str,
    group_headers: Sequence[tuple[int, int, str]],
    column_labels: Sequence[str],
    rows: Sequence[dict],
    notes: Sequence[str],
    output_path: Path,
    width: float,
    label_right: float,
    body_fontsize: float,
    height: float | None = None,
    pdf_pages: PdfPages | None = None,
    compact_header: bool = False,
    dependent_variable_header: str | None = None,
) -> None:
    """使用文字和横线绘制不带网格的论文式三线表。"""
    row_heights = {
        "panel": 1.15,
        "coefficient": 0.88,
        "standard_error": 0.76,
        "stat": 0.88,
        "spacer": 0.35,
    }
    if compact_header:
        title_height = 3.55
        table_number_y = 0.18
        title_y = 0.92
        subtitle_y = 2.12
        table_number_fontsize = 11.5
        title_fontsize = 14.5
        subtitle_fontsize = 9.1
    else:
        title_height = 5.0
        table_number_y = 0.38
        title_y = 2.02
        subtitle_y = 3.52
        table_number_fontsize = 21.0
        title_fontsize = 19.0
        subtitle_fontsize = 10.5
    # 多行副标题为固定效应说明预留空间；单行表格的布局保持不变。
    title_height += 0.85 * subtitle.count("\n")
    header_height = 2.78 if dependent_variable_header else 2.15
    notes_height = 0.85 * len(notes) + 0.75
    body_height = sum(row_heights[row["kind"]] for row in rows)
    total_height = title_height + header_height + body_height + notes_height
    figure_height = height if height is not None else max(7.5, total_height * 0.29)

    figure_dpi = 254 if compact_header else None
    if figure_dpi is None:
        fig, ax = plt.subplots(figsize=(width, figure_height))
    else:
        fig, ax = plt.subplots(
            figsize=(width, figure_height),
            dpi=figure_dpi,
        )
    if compact_header:
        fig.subplots_adjust(left=0.02, right=0.985, top=0.97, bottom=0.03)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(total_height, 0)
    ax.axis("off")

    number_of_columns = len(column_labels)
    data_left = label_right + 0.012
    data_right = 0.975
    group_boundaries = {
        end for _, end, _ in group_headers[:-1]
    } if compact_header else set()
    group_gap = 0.018 if group_boundaries else 0.0
    available_width = (
        data_right - data_left - group_gap * len(group_boundaries)
    )
    column_width = available_width / number_of_columns
    column_lefts = []
    column_rights = []
    column_centers = []
    cursor = data_left
    for index in range(number_of_columns):
        left = cursor
        right = left + column_width
        column_lefts.append(left)
        column_rights.append(right)
        column_centers.append((left + right) / 2)
        cursor = right + (group_gap if index in group_boundaries else 0.0)

    ax.text(
        0.02,
        table_number_y,
        table_number,
        ha="left",
        va="top",
        fontproperties=FONT_BOLD,
        fontsize=table_number_fontsize,
    )
    ax.text(
        0.02,
        title_y,
        title,
        ha="left",
        va="top",
        fontproperties=FONT_BOLD,
        fontsize=title_fontsize,
    )
    ax.text(
        0.02,
        subtitle_y,
        subtitle,
        ha="left",
        va="top",
        color="#444444",
        fontproperties=FONT,
        fontsize=subtitle_fontsize,
    )

    header_top = title_height
    _draw_rule(ax, header_top, linewidth=1.0)
    variable_header_y = (
        header_top + 1.55
        if dependent_variable_header
        else header_top + 1.24
    )
    ax.text(
        0.046,
        variable_header_y,
        "变量",
        ha="left",
        va="center",
        fontproperties=FONT_BOLD,
        fontsize=body_fontsize + 0.4,
    )

    if dependent_variable_header:
        ax.text(
            (column_centers[0] + column_centers[-1]) / 2,
            header_top + 0.42,
            dependent_variable_header,
            ha="center",
            va="center",
            color="#333333",
            fontproperties=FONT,
            fontsize=max(8.0, body_fontsize - 0.1),
        )

    group_header_y = (
        header_top + 1.18
        if dependent_variable_header
        else header_top + 0.57
    )
    group_rule_y = (
        header_top + 1.62
        if dependent_variable_header
        else header_top + 0.98
    )
    column_label_y = (
        header_top + 2.16
        if dependent_variable_header
        else header_top + 1.42
    )

    for start, end, label in group_headers:
        center = (column_centers[start] + column_centers[end]) / 2
        ax.text(
            center,
            group_header_y,
            label,
            ha="center",
            va="center",
            fontproperties=FONT_BOLD,
            fontsize=body_fontsize + (0.15 if compact_header else 0.0),
        )
        if compact_header:
            ax.plot(
                [
                    column_lefts[start] + 0.08 * column_width,
                    column_rights[end] - 0.08 * column_width,
                ],
                [group_rule_y, group_rule_y],
                color="#555555",
                linewidth=0.5,
                solid_capstyle="butt",
                clip_on=False,
            )

    for center, label in zip(column_centers, column_labels):
        ax.text(
            center,
            column_label_y,
            label,
            ha="center",
            va="center",
            fontproperties=FONT_BOLD,
            fontsize=(
                max(7.5, body_fontsize - 0.45)
                if compact_header
                else body_fontsize
            ),
        )

    body_top = header_top + header_height
    _draw_rule(ax, body_top, linewidth=0.9)
    y = body_top

    for row in rows:
        kind = row["kind"]
        height = row_heights[kind]

        if row.get("line_before"):
            _draw_rule(ax, y, linewidth=row.get("line_width", 0.55))

        if kind == "spacer":
            y += height
            continue

        center_y = y + height / 2
        bold = row.get("bold", False)
        font = FONT_BOLD if bold else FONT
        color = row.get("color", "#111111")
        label_x = 0.046 if kind != "standard_error" else 0.058

        ax.text(
            label_x,
            center_y,
            row.get("label", ""),
            ha="left",
            va="center",
            color=color,
            fontproperties=font,
            fontsize=body_fontsize,
        )

        for center, value in zip(column_centers, row.get("values", [])):
            ax.text(
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

    _draw_rule(ax, y, linewidth=1.0)

    note_y = y + 0.72
    for note in notes:
        ax.text(
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
    if compact_header:
        fig.savefig(
            output_path,
            dpi=300,
            facecolor="white",
        )
    else:
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.12,
        )
    if pdf_pages is not None:
        if compact_header:
            pdf_pages.savefig(fig, facecolor="white")
        else:
            pdf_pages.savefig(
                fig,
                bbox_inches="tight",
                facecolor="white",
                pad_inches=0.12,
            )
    plt.close(fig)


def table2_model_rows(
    summary: pd.DataFrame,
    sample: str,
) -> list[pd.Series]:
    """按两类收益窗口和六种规格排列表2模型。"""
    sample_rows = summary.loc[summary["样本"].eq(sample)]
    ordered = []
    for window in WINDOW_ORDER:
        for specification in SPEC_ORDER:
            matched = sample_rows.loc[
                sample_rows["收益窗口"].eq(window)
                & sample_rows["规格"].eq(specification)
            ]
            if len(matched) != 1:
                raise ValueError(
                    f"表2中样本 {sample}、窗口 {window}、规格 "
                    f"{specification} 应恰好有一行，实际为 {len(matched)} 行。"
                )
            ordered.append(matched.iloc[0])
    return ordered


def table2_setting_values(model_rows: Sequence[pd.Series], setting: str) -> list[str]:
    values = []
    for row in model_rows:
        specification_number = int(str(row["规格"]).strip("()"))
        if setting == "own_controls":
            enabled = specification_number >= 2
        elif setting == "other_controls":
            enabled = specification_number >= 3
        elif setting == "date_fe":
            enabled = specification_number in {4, 6}
        elif setting == "firm_quarter_fe":
            enabled = specification_number in {5, 6}
        else:
            raise ValueError(f"未知的模型设置：{setting}")
        values.append("是" if enabled else "否")
    return values


def draw_table2_compact(summary: pd.DataFrame, output_path: Path) -> None:
    """绘制表2紧凑版：只展示线索系数和模型设定。"""
    rows = []
    for sample in ["pattern", "full"]:
        model_rows = table2_model_rows(summary, sample)
        sample_label = str(model_rows[0]["样本说明"])
        coefficients = []
        standard_errors = []
        for model_row in model_rows:
            coefficient, standard_error = format_estimate(
                model_row["线索系数（百分点）"],
                model_row["聚类标准误（百分点）"],
                model_row["p值"],
            )
            coefficients.append(coefficient)
            standard_errors.append(standard_error)

        rows.extend(
            [
                {
                    "kind": "panel",
                    "label": sample_label,
                    "values": [""] * 12,
                    "bold": True,
                    "line_before": bool(rows),
                },
                {
                    "kind": "coefficient",
                    "label": "线索事件",
                    "values": coefficients,
                    "bold": True,
                },
                {
                    "kind": "standard_error",
                    "label": "（聚类标准误）",
                    "values": standard_errors,
                    "color": "#444444",
                },
                {
                    "kind": "stat",
                    "label": "自身盈余公告控制",
                    "values": table2_setting_values(model_rows, "own_controls"),
                    "line_before": True,
                },
                {
                    "kind": "stat",
                    "label": "其他控制变量",
                    "values": table2_setting_values(model_rows, "other_controls"),
                },
                {
                    "kind": "stat",
                    "label": "日期固定效应",
                    "values": table2_setting_values(model_rows, "date_fe"),
                },
                {
                    "kind": "stat",
                    "label": "被提示公司×年度季度固定效应",
                    "values": table2_setting_values(model_rows, "firm_quarter_fe"),
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
                    "values": [format_r_squared(row["R方"]) for row in model_rows],
                },
            ]
        )

    render_publication_table(
        table_number="表 2",
        title="线索事件与被提示公司收益",
        subtitle="因变量：被提示公司的特征调整收益率；系数与标准误的单位均为百分点",
        group_headers=[
            (0, 5, "收益窗口 [t, t+1]"),
            (6, 11, "收益窗口 [t, t+10]"),
        ],
        column_labels=SPEC_ORDER + SPEC_ORDER,
        rows=rows,
        notes=[
            "注：括号内为按被提示公司和交易日期双向聚类的标准误。",
            "*、**、*** 分别表示在 10%、5% 和 1% 水平上显著。",
        ],
        output_path=output_path,
        width=23.0,
        label_right=0.285,
        body_fontsize=9.0,
    )


def _table2_detail_lookup(
    details: pd.DataFrame,
    sample: str,
) -> dict[tuple[str, str, str], pd.Series]:
    selected = details.loc[details["样本"].eq(sample)]
    return {
        (row["收益窗口"], row["规格"], row["变量"]): row
        for _, row in selected.iterrows()
    }


def table2_variable_order(details: pd.DataFrame, sample: str) -> list[str]:
    """返回指定样本在完整系数表中的变量顺序。"""
    return (
        details.loc[details["样本"].eq(sample), ["变量", "变量顺序"]]
        .drop_duplicates()
        .sort_values("变量顺序")["变量"]
        .tolist()
    )


def split_table2_variable_order(
    variable_order: Sequence[str],
    *,
    split_after_variable: str,
) -> tuple[list[str], list[str]]:
    """在指定变量之后将表2的变量顺序分成两页。"""
    variables = list(variable_order)
    if split_after_variable not in variables:
        raise ValueError(
            f"表2分页断点变量不存在：{split_after_variable}"
        )

    split_index = variables.index(split_after_variable) + 1
    first_page = variables[:split_index]
    second_page = variables[split_index:]
    if not first_page or not second_page:
        raise ValueError("表2分页后两页都必须包含变量。")
    return first_page, second_page


def _table2_full_coefficient_rows(
    model_rows: Sequence[pd.Series],
    lookup: dict[tuple[str, str, str], pd.Series],
    variable_order: Sequence[str],
) -> list[dict]:
    """整理表2完整系数版中的系数和标准误行。"""
    rows = []
    for variable_index, variable in enumerate(variable_order):
        coefficient_values = []
        standard_error_values = []
        for model_row in model_rows:
            detail_row = lookup.get(
                (model_row["收益窗口"], model_row["规格"], variable)
            )
            if detail_row is None:
                coefficient, standard_error = "", ""
            else:
                coefficient, standard_error = format_estimate(
                    detail_row["系数（百分点）"],
                    detail_row["聚类标准误（百分点）"],
                    detail_row["p值"],
                )
            coefficient_values.append(coefficient)
            standard_error_values.append(standard_error)

        rows.extend(
            [
                {
                    "kind": "coefficient",
                    "label": variable_label(variable),
                    "values": coefficient_values,
                    "bold": variable == "quarterLag1size90plus",
                    "line_before": variable == "ownEA_tm10",
                },
                {
                    "kind": "standard_error",
                    "label": "",
                    "values": standard_error_values,
                    "color": "#444444",
                },
            ]
        )
    return rows


def _table2_full_stat_rows(model_rows: Sequence[pd.Series]) -> list[dict]:
    """整理表2完整系数版底部的模型设定和统计量。"""
    return [
        {
            "kind": "stat",
            "label": "自身盈余公告控制",
            "values": table2_setting_values(model_rows, "own_controls"),
            "line_before": True,
        },
        {
            "kind": "stat",
            "label": "其他控制变量",
            "values": table2_setting_values(model_rows, "other_controls"),
        },
        {
            "kind": "stat",
            "label": "日期固定效应",
            "values": table2_setting_values(model_rows, "date_fe"),
        },
        {
            "kind": "stat",
            "label": "被提示公司×年度季度固定效应",
            "values": table2_setting_values(model_rows, "firm_quarter_fe"),
        },
        {
            "kind": "stat",
            "label": "样本量",
            "values": [
                format_observations(row["回归样本量"]) for row in model_rows
            ],
        },
        {
            "kind": "stat",
            "label": r"$R^2$",
            "values": [format_r_squared(row["R方"]) for row in model_rows],
        },
    ]


def _draw_table2_full_page(
    *,
    model_rows: Sequence[pd.Series],
    lookup: dict[tuple[str, str, str], pd.Series],
    variable_order: Sequence[str],
    output_path: Path,
    table_number: str,
    include_model_stats: bool,
    notes: Sequence[str],
    width: float,
    height: float | None,
    label_right: float,
    body_fontsize: float,
    pdf_pages: PdfPages | None = None,
    compact_header: bool = False,
    continued: bool = False,
) -> None:
    """绘制表2完整系数版的单个页面。"""
    rows = _table2_full_coefficient_rows(model_rows, lookup, variable_order)
    if include_model_stats:
        rows.extend(_table2_full_stat_rows(model_rows))

    sample_label = str(model_rows[0]["样本说明"])
    if compact_header:
        subtitle = sample_label
    else:
        subtitle = (
            f"{sample_label}；因变量为被提示公司的特征调整收益率；"
            "系数与标准误的单位均为百分点"
        )
    render_publication_table(
        table_number=table_number,
        title="线索事件与被提示公司收益",
        subtitle=subtitle,
        group_headers=[
            (0, 5, "收益窗口 [t, t+1]"),
            (6, 11, "收益窗口 [t, t+10]"),
        ],
        column_labels=SPEC_ORDER + SPEC_ORDER,
        rows=rows,
        notes=notes,
        output_path=output_path,
        width=width,
        height=height,
        label_right=label_right,
        body_fontsize=body_fontsize,
        pdf_pages=pdf_pages,
        compact_header=compact_header,
        dependent_variable_header=(
            "因变量：被提示公司的特征调整收益率（%）"
            if compact_header
            else None
        ),
    )


def draw_table2_full(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    sample: str,
    output_path: Path,
) -> None:
    """按样本绘制包含全部控制变量系数的表2。"""
    model_rows = table2_model_rows(summary, sample)
    lookup = _table2_detail_lookup(details, sample)
    variable_order = table2_variable_order(details, sample)
    _draw_table2_full_page(
        model_rows=model_rows,
        lookup=lookup,
        variable_order=variable_order,
        output_path=output_path,
        table_number="表 2（完整系数版）",
        include_model_stats=True,
        notes=[
            "注：空白单元格表示相应规格未纳入该控制变量；括号内为按被提示公司和交易日期双向聚类的标准误。",
            "*、**、*** 分别表示在 10%、5% 和 1% 水平上显著。",
        ],
        width=24.0,
        height=None,
        label_right=0.19,
        body_fontsize=7.6,
    )


def draw_table2_full_two_pages(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    *,
    sample: str,
    first_page_output: Path,
    second_page_output: Path,
    pdf_output: Path | None,
) -> None:
    """将表2完整系数版按原文断点导出为两张图片及可选两页 PDF。"""
    model_rows = table2_model_rows(summary, sample)
    lookup = _table2_detail_lookup(details, sample)
    variable_order = table2_variable_order(details, sample)
    first_variables, second_variables = split_table2_variable_order(
        variable_order,
        split_after_variable="ownEA_tp3",
    )

    def render_pages(pdf_pages: PdfPages | None = None) -> None:
        _draw_table2_full_page(
            model_rows=model_rows,
            lookup=lookup,
            variable_order=first_variables,
            output_path=first_page_output,
            table_number="表 2",
            include_model_stats=False,
            notes=["续见下一页。"],
            width=A4_LANDSCAPE_WIDTH_IN,
            height=A4_LANDSCAPE_HEIGHT_IN,
            label_right=0.205,
            body_fontsize=8.5,
            pdf_pages=pdf_pages,
            compact_header=True,
        )
        _draw_table2_full_page(
            model_rows=model_rows,
            lookup=lookup,
            variable_order=second_variables,
            output_path=second_page_output,
            table_number="表 2（续）",
            include_model_stats=True,
            notes=[
                "注：空白单元格表示相应规格未纳入该控制变量；括号内为按被提示公司和交易日期双向聚类的标准误。",
                "*、**、*** 分别表示在 10%、5% 和 1% 水平上显著。",
            ],
            width=A4_LANDSCAPE_WIDTH_IN,
            height=A4_LANDSCAPE_HEIGHT_IN,
            label_right=0.205,
            body_fontsize=8.5,
            pdf_pages=pdf_pages,
            compact_header=True,
            continued=True,
        )

    if pdf_output is None:
        render_pages()
    else:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_output) as pdf_pages:
            render_pages(pdf_pages)


def order_imbalance_models(summary: pd.DataFrame) -> list[pd.Series]:
    ordered = []
    for window in WINDOW_ORDER:
        matched = summary.loc[summary["订单失衡窗口"].eq(window)]
        if len(matched) != 1:
            raise ValueError(
                f"订单失衡窗口 {window} 应恰好有一行，实际为 {len(matched)} 行。"
            )
        ordered.append(matched.iloc[0])
    return ordered


def read_appendix_order_imbalance_results(
    summary_path: Path,
    detail_path: Path,
    *,
    prefix: str,
    measure_label: str,
    date_fixed_effects: str,
    table_number: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取附表结果，防止混用不同指标、窗口或固定效应的历史文件。"""
    metadata_columns = [
        "衡量口径",
        "订单失衡窗口",
        "散户识别方法",
        "因变量",
        "单位",
        "日期固定效应",
        "被提示公司×年度季度固定效应",
    ]
    if table_number is not None:
        metadata_columns.append("附表编号")
    summary = read_result(
        summary_path,
        metadata_columns
        + [
            "线索系数",
            "聚类标准误",
            "p值",
            "控制变量",
            "回归样本量",
            "R方",
        ],
    )
    details = read_result(
        detail_path,
        metadata_columns
        + ["变量", "变量顺序", "系数", "聚类标准误", "p值"],
    )
    expected_metadata = {
        "衡量口径": measure_label,
        "散户识别方法": "A股小单口径",
        "单位": "百分点",
        "日期固定效应": date_fixed_effects,
        "被提示公司×年度季度固定效应": "是",
    }
    if table_number is not None:
        expected_metadata["附表编号"] = table_number
    expected_outcomes = {
        "[t, t+1]": f"{prefix}_avg_t_tp1_w",
        "[t, t+10]": f"{prefix}_avg_t_tp10_w",
    }
    for frame, source_path in [(summary, summary_path), (details, detail_path)]:
        if frame.empty:
            raise ValueError(f"{source_path.name}没有可绘制的附表结果。")
        for column, expected in expected_metadata.items():
            values = frame[column].astype("string").str.strip()
            if values.isna().any() or not values.eq(expected).all():
                raise ValueError(
                    f"{source_path.name}的“{column}”必须全部为“{expected}”。"
                )
        if set(frame["订单失衡窗口"].dropna()) != set(WINDOW_ORDER):
            raise ValueError(
                f"{source_path.name}应且仅应包含两个窗口：{WINDOW_ORDER}。"
            )
        expected = frame["订单失衡窗口"].map(expected_outcomes)
        if expected.isna().any() or not frame["因变量"].eq(expected).all():
            raise ValueError(
                f"{source_path.name}的因变量与{measure_label}指标及窗口不一致。"
            )
    order_imbalance_models(summary)
    if len(summary) != len(WINDOW_ORDER):
        raise ValueError(f"{summary_path.name}的模型汇总应恰好有两行。")
    if details[["变量", "变量顺序"]].isna().any().any():
        raise ValueError(f"{detail_path.name}的变量名称或顺序存在缺失。")
    if details.duplicated(["订单失衡窗口", "变量"]).any():
        raise ValueError(f"{detail_path.name}存在重复的“订单失衡窗口×变量”。")
    variable_orders = details[["变量", "变量顺序"]].drop_duplicates()
    if variable_orders.duplicated("变量").any() or variable_orders.duplicated(
        "变量顺序"
    ).any():
        raise ValueError(f"{detail_path.name}的变量与变量顺序不是一一对应。")
    expected_variables = set(variable_orders["变量"])
    for window in WINDOW_ORDER:
        actual_variables = set(details.loc[details["订单失衡窗口"].eq(window), "变量"])
        if actual_variables != expected_variables:
            raise ValueError(f"{detail_path.name}的{window}窗口缺少系数明细行。")
    return summary, details


def draw_order_imbalance_full(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    *,
    table_number: str,
    title: str,
    output_path: Path,
    label_right: float = 0.54,
    additional_notes: Sequence[str] = (),
) -> None:
    """绘制表3或附表A.2-A.4的两列完整系数表。"""
    model_rows = order_imbalance_models(summary)
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

    rows = []
    for variable_index, variable in enumerate(variable_order):
        coefficient_values = []
        standard_error_values = []
        for model_row in model_rows:
            detail_row = lookup.get((model_row["订单失衡窗口"], variable))
            if detail_row is None:
                coefficient, standard_error = "", ""
            else:
                coefficient, standard_error = format_estimate(
                    detail_row["系数"],
                    detail_row["聚类标准误"],
                    detail_row["p值"],
                )
            coefficient_values.append(coefficient)
            standard_error_values.append(standard_error)

        rows.extend(
            [
                {
                    "kind": "coefficient",
                    "label": variable_label(variable),
                    "values": coefficient_values,
                    "bold": variable == "quarterLag1size90plus",
                    "line_before": variable_index == 1,
                },
                {
                    "kind": "standard_error",
                    "label": "",
                    "values": standard_error_values,
                    "color": "#444444",
                },
            ]
        )

    rows.extend(
        [
            {
                "kind": "stat",
                "label": "控制变量",
                "values": [str(row["控制变量"]) for row in model_rows],
                "line_before": True,
            },
            {
                "kind": "stat",
                "label": "日期固定效应",
                "values": [str(row["日期固定效应"]) for row in model_rows],
            },
            {
                "kind": "stat",
                "label": "被提示公司×年度季度固定效应",
                "values": [
                    str(row["被提示公司×年度季度固定效应"])
                    for row in model_rows
                ],
            },
            {
                "kind": "stat",
                "label": "样本量",
                "values": [
                    format_observations(row["回归样本量"]) for row in model_rows
                ],
            },
            {
                "kind": "stat",
                "label": "R²",
                "values": [format_r_squared(row["R方"]) for row in model_rows],
            },
        ]
    )

    measure = str(summary.iloc[0]["衡量口径"])
    method = str(summary.iloc[0]["散户识别方法"])
    render_publication_table(
        table_number=table_number,
        title=title,
        subtitle=(
            f"因变量：按{measure}衡量的小单订单失衡；当前复现采用{method}；"
            "系数与标准误的单位均为百分点"
        ),
        group_headers=[(0, 1, f"小单订单失衡（{measure}口径）")],
        column_labels=WINDOW_ORDER,
        rows=rows,
        notes=[
            *additional_notes,
            "注：括号内为按被提示公司和交易日期双向聚类的标准误。",
            "*、**、*** 分别表示在 10%、5% 和 1% 水平上显著。",
        ],
        output_path=output_path,
        width=12.0,
        label_right=label_right,
        body_fontsize=9.2,
    )


def table4_variable_label(variable: str) -> str:
    """把表4的线索公司规模组变量转换为中文标签。"""
    labels = {
        "quarterLag1size90plus": "线索：提示公司规模 > 第90百分位",
        "quarterLag1size50_90": "线索：提示公司规模为第51至90百分位",
        "quarterLag1size10_50": "线索：提示公司规模为第10至50百分位",
        "quarterLag1size10minus": "线索：提示公司规模 < 第10百分位",
    }
    return labels.get(variable, variable)


def build_table4_rows(
    results: pd.DataFrame,
    *,
    show_date_fixed_effects: bool = False,
) -> list[dict]:
    """把表4长表结果整理为三列论文式表格所需的行。"""
    model_rows = []
    coefficient_lookup = {}

    for specification in TABLE4_SPEC_ORDER:
        specification_rows = results.loc[
            results["规格"].eq(specification)
        ]
        if specification_rows.empty:
            raise ValueError(f"表4中缺少规格 {specification}。")

        for variable in TABLE4_CUE_ORDER:
            matched = specification_rows.loc[
                specification_rows["变量"].eq(variable)
            ]
            if len(matched) != 1:
                raise ValueError(
                    f"表4中规格 {specification}、变量 {variable} "
                    f"应恰好有一行，实际为 {len(matched)} 行。"
                )
            coefficient_lookup[(specification, variable)] = matched.iloc[0]

        # 同一规格的每行都保存相同的模型级统计量，
        # 因此取第一行用于底部的样本量和 R²。
        model_rows.append(specification_rows.iloc[0])

    rows = []
    for variable in TABLE4_CUE_ORDER:
        coefficients = []
        standard_errors = []
        for specification in TABLE4_SPEC_ORDER:
            result_row = coefficient_lookup[(specification, variable)]
            coefficient, standard_error = format_estimate(
                result_row["系数（百分点）"],
                result_row["聚类标准误（百分点）"],
                result_row["p值"],
            )
            coefficients.append(coefficient)
            standard_errors.append(standard_error)

        rows.extend(
            [
                {
                    "kind": "coefficient",
                    "label": table4_variable_label(variable),
                    "values": coefficients,
                },
                {
                    "kind": "standard_error",
                    "label": "",
                    "values": standard_errors,
                    "color": "#444444",
                },
            ]
        )

    statistic_rows = [
        {
            "kind": "stat",
            "label": "控制变量",
            "values": [str(row["控制变量"]) for row in model_rows],
            "line_before": True,
        },
    ]

    if show_date_fixed_effects:
        statistic_rows.append({
            "kind": "stat",
            "label": "日期固定效应",
            "values": [
                str(row["日期固定效应"])
                for row in model_rows
            ],
        })

    statistic_rows.extend(
        [
            {
                "kind": "stat",
                "label": "被提示公司×年度季度固定效应",
                "values": [
                    str(row["被提示公司×年度季度固定效应"])
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
                "label": r"$R^2$",
                "values": [format_r_squared(row["R方"]) for row in model_rows],
            },
        ]
    )
    rows.extend(statistic_rows)
    return rows


def draw_table4_extension(
    results: pd.DataFrame,
    output_path: Path,
    *,
    show_date_fixed_effects: bool = False,
) -> None:
    """绘制当前 [t, t+1] 口径的表4式规模异质性扩展分析。"""
    outcomes = set(results["因变量"].dropna().astype(str))
    expected_outcomes = {"chAdjreturn_t_tp1_Recall_w"}
    if outcomes != expected_outcomes:
        raise ValueError(
            "当前表4扩展分析仅适用于 [t, t+1] 收益结果；"
            f"实际因变量为：{sorted(outcomes)}。"
        )

    if show_date_fixed_effects:
        date_fe_values = set(
            results[
                "日期固定效应"
            ]
            .dropna()
            .astype(str)
        )
        if date_fe_values != {"是"}:
            raise ValueError(
                "加入日期固定效应的表4要求"
                "“日期固定效应”全部为“是”；"
                f"实际取值为：{sorted(date_fe_values)}。"
            )

    title = "按公司规模的横截面分析"
    notes = [
        "注：本表使用当前 [t, t+1] 回归结果，属于扩展分析，不是原文 [t, t+10] 表4的严格复现。",
        "括号内为按被提示公司和交易日期双向聚类的标准误。",
        "*、**、*** 分别表示在 10%、5% 和 1% 水平上显著。",
    ]

    if show_date_fixed_effects:
        title += "：加入日期固定效应"
        notes.insert(
            1,
            "本表在吸收被提示公司×年度季度固定效应的基础上，进一步吸收交易日期固定效应。",
        )

    render_publication_table(
        table_number="表 4（扩展分析）",
        title=title,
        subtitle=(
            "因变量：被提示公司的特征调整收益率 [t, t+1]；"
            "系数与标准误的单位均为百分点\n"
            "固定效应：被提示公司×年度季度；"
            + ("交易日期" if show_date_fixed_effects else "不含交易日期固定效应")
        ),
        group_headers=[(0, 2, "被提示公司规模")],
        column_labels=["全部\n(1)", "小型\n(2)", "大型\n(3)"],
        rows=build_table4_rows(
            results,
            show_date_fixed_effects=show_date_fixed_effects,
        ),
        notes=notes,
        output_path=output_path,
        width=12.5,
        label_right=0.39,
        body_fontsize=9.2,
    )

