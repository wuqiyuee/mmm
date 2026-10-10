"""
在不覆盖现有表3结果的前提下，额外估计加入交易日期固定效应的
散户订单失衡回归。

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

TABLE3_DATE_FE_RESULTS_PATH = (
    ROOT
    / "中间结果"
    / "08_回归结果"
    / "表3_散户订单失衡_加入日期固定效应结果.csv"
)

TABLE3_DATE_FE_COEFFICIENTS_PATH = (
    ROOT
    / "中间结果"
    / "08_回归结果"
    / "表3_散户订单失衡_加入日期固定效应系数明细.csv"
)

TABLE3_DATE_FE_IMAGE_PATH = (
    ROOT
    / "图表结果"
    / "论文复现"
    / "表格"
    / "table3_散户订单失衡_完整系数版_加入日期固定效应_独立脚本.png"
)

FONT_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
if FONT_PATH.exists():
    FONT = FontProperties(fname=str(FONT_PATH))
    FONT_BOLD = FontProperties(fname=str(FONT_PATH), weight="bold")
else:
    FONT = FontProperties(family="sans-serif")
    FONT_BOLD = FontProperties(family="sans-serif", weight="bold")

plt.rcParams["axes.unicode_minus"] = False


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
        "ownMajorDisclosure": "其他专题事件 [t]",
        "ownPerformanceForecast": "业绩预告或快报 [t]",
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
    """在本脚本内直接绘制表3日期固定效应版的完整系数表。"""
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
    rows: list[list[str]] = []
    bold_rows: set[int] = set()
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
        bold_rows.add(len(rows))
        rows.append([_table3_variable_label(variable), *coefficient_values])
        rows.append(["", *standard_error_values])

    rows.extend([
        ["控制变量", *[str(row["控制变量"]) for row in model_rows]],
        ["交易日期固定效应", *[str(row["日期固定效应"]) for row in model_rows]],
        [
            "被提示公司×年度季度固定效应",
            *[str(row["被提示公司×年度季度固定效应"]) for row in model_rows],
        ],
        ["样本量", *[f"{int(row['回归样本量']):,}" for row in model_rows]],
        ["R²", *[f"{float(row['R方']):.4f}" for row in model_rows]],
    ])

    figure_height = max(12.0, 5.4 + 0.23 * len(rows))
    figure, axes = plt.subplots(figsize=(12.0, figure_height))
    figure.patch.set_facecolor("white")
    axes.axis("off")
    axes.text(
        0.01,
        0.988,
        "表 3：线索事件与散户订单失衡（加入交易日期固定效应）",
        ha="left",
        va="top",
        fontsize=16,
        fontproperties=FONT_BOLD,
    )
    axes.text(
        0.01,
        0.957,
        "因变量为按成交量衡量的小单订单失衡；系数与括号内双向聚类标准误的单位均为百分点",
        ha="left",
        va="top",
        fontsize=8.6,
        color="#444444",
        fontproperties=FONT,
    )
    table = axes.table(
        cellText=rows,
        colLabels=["变量", *window_order],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.50, 0.25, 0.25],
        bbox=[0.01, 0.08, 0.98, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#9AA3AA")
        cell.set_linewidth(0.28)
        cell.set_facecolor("#F4F6F7" if row_index == 0 else "white")
        cell.get_text().set_fontproperties(
            FONT_BOLD
            if row_index == 0 or (row_index - 1) in bold_rows
            else FONT
        )
        if column_index == 0:
            cell.get_text().set_ha("left")
    axes.text(
        0.01,
        0.045,
        "注：所有模型同时吸收被提示公司×年度季度和交易日期固定效应；"
        "*、**、*** 分别表示在 10%、5% 和 1% 水平上显著。",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#444444",
        fontproperties=FONT,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.08,
    )
    plt.close(figure)


def main() -> None:
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


if __name__ == "__main__":
    main()
