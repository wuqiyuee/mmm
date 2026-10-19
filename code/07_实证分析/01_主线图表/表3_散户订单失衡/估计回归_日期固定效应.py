"""
在不覆盖现有表3结果的前提下，额外估计加入交易日期固定效应的散户订单失衡回归。

固定效应：
1. 被提示公司 × 年度季度；
2. 交易日期。

标准误仍按被提示公司和交易日期双向聚类。
"""

from __future__ import annotations

import gc
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from matplotlib.font_manager import FontProperties


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
OUTPUT_DIR = ROOT / "图表结果/01_主线图表/表3_散户订单失衡"

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

TABLE3_DATE_FE_RESULTS_PATH = OUTPUT_DIR / "表3_散户订单失衡_加入日期固定效应结果.csv"

TABLE3_DATE_FE_COEFFICIENTS_PATH = OUTPUT_DIR / "表3_散户订单失衡_加入日期固定效应系数明细.csv"

TABLE3_DATE_FE_IMAGE_PATH = OUTPUT_DIR / "table3_散户订单失衡_完整系数版_加入日期固定效应_独立脚本.png"

FONT_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
if FONT_PATH.exists():
    FONT = FontProperties(fname=str(FONT_PATH))
    FONT_BOLD = FontProperties(fname=str(FONT_PATH), weight="bold")
else:
    FONT = FontProperties(family="sans-serif")
    FONT_BOLD = FontProperties(family="sans-serif", weight="bold")

plt.rcParams["axes.unicode_minus"] = False


START_YEAR = 2005
END_YEAR = 2025

CUE_COL = "quarterLag1size90plus"

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

RHS_COLS = (
    [CUE_COL]
    + OWN_EA_COLS
    + DISCLOSURE_COLS
    + MEDIA_COLS
)

TABLE3_VOLUME_OUTCOMES = [
    (
        "OIBvol_avg_t_tp1_w",
        "[t, t+1]",
    ),
    (
        "OIBvol_avg_t_tp10_w",
        "[t, t+10]",
    ),
]

OIB_RAW_COLS = [
    "OIBvol_avg_t_tp1",
    "OIBvol_avg_t_tp10",
]


def significance_stars(p_value: float) -> str:
    """根据双侧检验的 p 值添加显著性星号。"""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def winsorize_1pct(series: pd.Series) -> pd.Series:
    """把变量的上下各1%观测缩尾到相应分位点。"""
    valid = pd.to_numeric(
        series,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    if valid.notna().sum() == 0:
        raise ValueError("订单失衡变量没有可用于计算缩尾阈值的观测。")

    lower = valid.quantile(0.01)
    upper = valid.quantile(0.99)
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("订单失衡变量无法计算有效的1%/99%缩尾阈值。")

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


def read_table3_date_fe_data() -> pd.DataFrame:
    """读取并整理新增日期固定效应版表3所需的数据。"""
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

    print("读取加入日期固定效应的表3主样本……")
    data = pd.read_csv(
        PATTERN_PATH,
        usecols=panel_cols,
        dtype=panel_dtype,
        parse_dates=["Trddt"],
    )

    if data[["StkcdRecall", "Trddt", "yearQuarter"]].isna().any().any():
        raise ValueError(
            "表3主样本的公司代码、交易日期或年度季度存在缺失。"
        )

    for column in numeric_cols:
        data[column] = data[column].replace([np.inf, -np.inf], np.nan)

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
            "表3主样本的公司和日期不是唯一键。"
        )

    oib_dtype = {
        "Stkcd": "string",
        **{
            column: "float32"
            for column in OIB_RAW_COLS
        },
    }

    print("读取加入日期固定效应的表3订单失衡数据……")
    oib = pd.read_csv(
        OIB_PATH,
        usecols=[
            "Stkcd",
            "Trddt",
        ] + OIB_RAW_COLS,
        dtype=oib_dtype,
        parse_dates=["Trddt"],
    )

    if oib[["Stkcd", "Trddt"]].isna().any().any():
        raise ValueError(
            "订单失衡表的股票代码或交易日期存在缺失。"
        )

    for column in OIB_RAW_COLS:
        oib[column] = oib[column].replace([np.inf, -np.inf], np.nan)

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


def run_table3_date_fe_regression(
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
    if reg.empty:
        raise ValueError(
            f"因变量 {outcome} 与解释变量合并删除缺失值后没有可用观测。"
        )

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

    if reg.empty:
        raise ValueError(
            f"因变量 {outcome} 删除固定效应单例组后没有可用观测。"
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

    if CUE_COL not in result.params.index:
        raise ValueError(
            f"因变量 {outcome} 中的线索变量被固定效应完全吸收，"
            "无法报告线索系数。"
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
        "删除单例组观测数":
            singleton_removed,
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
            "聚类标准误":
                variable_standard_error,
            "单位": "百分点",
            "t值": variable_t_value,
            "p值": variable_p_value,
            "显著性": stars,
            "控制变量": "是",
            "日期固定效应": "是",
            "被提示公司×年度季度固定效应":
                "是",
            "回归样本量": int(result.nobs),
            "R方": float(result.rsquared),
        })

    del (
        result,
        model,
        absorb,
        clusters,
        reg,
    )
    gc.collect()

    return model_info, coefficient_rows


def _format_table3_estimate(row: pd.Series) -> tuple[str, str]:
    """将表3系数和双向聚类标准误转成论文式表格文字。"""
    coefficient = pd.to_numeric(
        pd.Series([row.get("系数", np.nan)]),
        errors="coerce",
    ).iloc[0]
    standard_error = pd.to_numeric(
        pd.Series([row.get("聚类标准误", np.nan)]),
        errors="coerce",
    ).iloc[0]
    p_value = pd.to_numeric(
        pd.Series([row.get("p值", np.nan)]),
        errors="coerce",
    ).iloc[0]
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


def _table3_variable_label(variable: str) -> str:
    """将散户订单失衡回归的变量名转换为中文标签。"""
    labels = {
        CUE_COL: "线索事件",
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
    lag = re.fullmatch(r"ownEA_tm(\d+)", variable)
    if lag:
        return f"自身盈余公告 [t-{lag.group(1)}]"
    lead = re.fullmatch(r"ownEA_tp(\d+)", variable)
    if lead:
        return f"自身盈余公告 [t+{lead.group(1)}]"
    return variable


def draw_table3_date_fe_full(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制与集中绘表脚本一致的论文式表3三线表。"""
    window_order = ["[t, t+1]", "[t, t+10]"]
    model_rows = []
    for window in window_order:
        matched = summary.loc[summary["订单失衡窗口"].eq(window)]
        if len(matched) != 1:
            raise ValueError(f"表3完整系数表缺少或重复窗口：{window}。")
        model_rows.append(matched.iloc[0])
    variables = (
        details[["变量", "变量顺序"]]
        .drop_duplicates()
        .sort_values("变量顺序")["变量"]
        .tolist()
    )
    lookup = {
        (row["订单失衡窗口"], row["变量"]): row
        for _, row in details.iterrows()
    }
    rows: list[dict[str, object]] = []
    for variable in variables:
        coefficient_values, standard_error_values = [], []
        for model in model_rows:
            detail = lookup.get((model["订单失衡窗口"], variable))
            coefficient, standard_error = (
                _format_table3_estimate(detail)
                if detail is not None
                else ("", "")
            )
            coefficient_values.append(coefficient)
            standard_error_values.append(standard_error)

        rows.extend([
            {
                "kind": "coefficient",
                "label": _table3_variable_label(variable),
                "values": coefficient_values,
                "bold": variable == CUE_COL,
                "line_before": variable == OWN_EA_COLS[0],
            },
            {
                "kind": "standard_error",
                "label": "",
                "values": standard_error_values,
                "color": "#444444",
            },
        ])

    rows.extend([
        {
            "kind": "stat",
            "label": "控制变量",
            "values": [str(row["控制变量"]) for row in model_rows],
            "line_before": True,
        },
        {
            "kind": "stat",
            "label": "交易日期固定效应",
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
                f"{int(row['回归样本量']):,}" for row in model_rows
            ],
        },
        {
            "kind": "stat",
            "label": "R²",
            "values": [f"{float(row['R方']):.4f}" for row in model_rows],
        },
    ])

    row_heights = {
        "coefficient": 0.88,
        "standard_error": 0.76,
        "stat": 0.88,
    }
    notes = [
        "注：所有模型同时吸收被提示公司×年度季度和交易日期固定效应；",
        "括号内为按被提示公司和交易日期双向聚类的标准误；"
        "*、**、*** 分别表示在10%、5%和1%水平上显著。",
    ]
    title_height = 5.0
    header_height = 2.78
    notes_height = 0.85 * len(notes) + 0.75
    body_height = sum(row_heights[row["kind"]] for row in rows)
    total_height = (
        title_height
        + header_height
        + body_height
        + notes_height
    )
    figure_height = max(7.5, total_height * 0.29)

    figure, axes = plt.subplots(figsize=(12.0, figure_height))
    figure.subplots_adjust(
        left=0.02,
        right=0.985,
        top=0.97,
        bottom=0.03,
    )
    figure.patch.set_facecolor("white")
    axes.set_facecolor("white")
    axes.set_xlim(0, 1)
    axes.set_ylim(total_height, 0)
    axes.axis("off")

    label_right = 0.34
    data_left = label_right + 0.012
    data_right = 0.975
    column_width = (data_right - data_left) / len(window_order)
    column_lefts = [
        data_left + column_width * index
        for index in range(len(window_order))
    ]
    column_rights = [left + column_width for left in column_lefts]
    column_centers = [
        (left + right) / 2
        for left, right in zip(column_lefts, column_rights)
    ]

    def draw_rule(y: float, linewidth: float = 0.8) -> None:
        axes.plot(
            [0.02, 0.985],
            [y, y],
            color="#222222",
            linewidth=linewidth,
            solid_capstyle="butt",
            clip_on=False,
        )

    axes.text(
        0.02,
        0.38,
        "表 3（加入日期固定效应）",
        ha="left",
        va="top",
        fontproperties=FONT_BOLD,
        fontsize=21.0,
    )
    axes.text(
        0.02,
        2.02,
        "线索事件与散户订单失衡",
        ha="left",
        va="top",
        fontproperties=FONT_BOLD,
        fontsize=19.0,
    )
    axes.text(
        0.02,
        3.52,
        "因变量为按成交量衡量的小单订单失衡；"
        "系数与括号内双向聚类标准误的单位均为百分点",
        ha="left",
        va="top",
        color="#444444",
        fontproperties=FONT,
        fontsize=10.5,
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
        fontsize=9.6,
    )
    axes.text(
        (column_centers[0] + column_centers[-1]) / 2,
        header_top + 0.42,
        "因变量：按成交量衡量的小单订单失衡（百分点）",
        ha="center",
        va="center",
        color="#333333",
        fontproperties=FONT,
        fontsize=9.1,
    )
    axes.text(
        (column_centers[0] + column_centers[-1]) / 2,
        header_top + 1.18,
        "小单订单失衡（成交量口径）",
        ha="center",
        va="center",
        fontproperties=FONT_BOLD,
        fontsize=9.2,
    )
    for center, label in zip(column_centers, window_order):
        axes.text(
            center,
            header_top + 2.16,
            label,
            ha="center",
            va="center",
            fontproperties=FONT_BOLD,
            fontsize=9.2,
        )

    body_top = header_top + header_height
    draw_rule(body_top, linewidth=0.9)
    y = body_top
    for row in rows:
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
            fontsize=9.2,
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
                fontsize=9.2,
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
            fontsize=8.7,
        )
        note_y += 0.82

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(figure)


def main() -> list[Path]:
    TABLE3_DATE_FE_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = read_table3_date_fe_data()
    model_rows = []
    coefficient_rows = []

    for outcome, window_label in TABLE3_VOLUME_OUTCOMES:
        print(
            "正在运行加入日期固定效应的表3："
            f"{window_label}"
        )

        model_info, current_coefficients = (
            run_table3_date_fe_regression(
                data,
                outcome,
            )
        )

        model_metadata = {
            "结果表": "表3（加入日期固定效应）",
            "衡量口径": "成交量",
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
    coefficients = pd.DataFrame(
        coefficient_rows
    )

    results.to_csv(
        TABLE3_DATE_FE_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    coefficients.to_csv(
        TABLE3_DATE_FE_COEFFICIENTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    draw_table3_date_fe_full(
        results,
        coefficients,
        TABLE3_DATE_FE_IMAGE_PATH,
    )

    print("\n表3：加入日期固定效应的散户订单失衡结果")
    print(results.to_string(index=False))
    print(
        "\n表3新增结果已保存："
        f"{TABLE3_DATE_FE_RESULTS_PATH}"
    )
    print(
        "表3新增全部系数明细已保存："
        f"{TABLE3_DATE_FE_COEFFICIENTS_PATH}"
    )
    print(f"表3完整系数图片已保存：{TABLE3_DATE_FE_IMAGE_PATH}")

    del data, results, coefficients
    gc.collect()
    return [
        TABLE3_DATE_FE_RESULTS_PATH,
        TABLE3_DATE_FE_COEFFICIENTS_PATH,
        TABLE3_DATE_FE_IMAGE_PATH,
    ]


if __name__ == "__main__":
    main()
