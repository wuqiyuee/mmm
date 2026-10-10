"""
在不覆盖现有表4结果的前提下，额外估计加入交易日期固定效应的规模异质性回归。

固定效应：
1. 被提示公司 × 年度季度；
2. 交易日期。

标准误仍按被提示公司和交易日期双向聚类。
"""

from __future__ import annotations

import gc
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

TABLE4_DATE_FE_RESULTS_PATH = (
    ROOT
    / "中间结果"
    / "08_回归结果"
    / "表4_规模异质性_加入日期固定效应结果.csv"
)

TABLE4_DATE_FE_COEFFICIENTS_PATH = (
    ROOT
    / "中间结果"
    / "08_回归结果"
    / "表4_规模异质性_加入日期固定效应系数明细.csv"
)

TABLE4_DATE_FE_IMAGE_PATH = (
    ROOT
    / "图表结果"
    / "论文复现"
    / "表格"
    / "table4_规模异质性_加入日期固定效应_独立脚本.png"
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

TABLE4_OUTCOME = "chAdjreturn_t_tp1_Recall_w"

TABLE4_CUE_COLS = [
    "quarterLag1size90plus",
    "quarterLag1size50_90",
    "quarterLag1size10_50",
    "quarterLag1size10minus",
]

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

TABLE4_CONTROL_COLS = (
    OWN_EA_COLS
    + DISCLOSURE_COLS
    + MEDIA_COLS
)

TABLE4_SPECIFICATIONS = [
    {
        "规格": "(1)",
        "样本": "全部规律公告公司样本",
        "smallFirm取值": None,
    },
    {
        "规格": "(2)",
        "样本": "小型被提示公司",
        "smallFirm取值": 1,
    },
    {
        "规格": "(3)",
        "样本": "大型被提示公司",
        "smallFirm取值": 0,
    },
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


def drop_fe_singletons_iteratively(
    reg: pd.DataFrame,
    fe_code_cols: list[str],
) -> tuple[pd.DataFrame, int]:
    """递归删除任一固定效应维度中的单例组。"""
    if not fe_code_cols:
        return reg, 0

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


def read_table4_data(
    input_path: Path = PATTERN_PATH,
) -> pd.DataFrame:
    """只读取新增表4回归所需的字段。"""
    input_cols = list(dict.fromkeys(
        [
            "StkcdRecall",
            "Trddt",
            "year",
            "yearQuarter",
            "smallFirm",
            TABLE4_OUTCOME,
        ]
        + TABLE4_CUE_COLS
        + TABLE4_CONTROL_COLS
    ))

    numeric_cols = [
        column
        for column in input_cols
        if column not in {
            "StkcdRecall",
            "Trddt",
            "yearQuarter",
        }
    ]

    dtype = {
        "StkcdRecall": "string",
        "yearQuarter": "string",
    }
    dtype.update({
        column: "float32"
        for column in numeric_cols
    })

    print("读取加入日期固定效应的表4回归数据……")
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
            "表4样本的被提示公司和交易日期不是唯一键。"
        )

    data["StkcdRecall"] = (
        data["StkcdRecall"].astype("category")
    )
    data["yearQuarter"] = (
        data["yearQuarter"].astype("category")
    )

    return data


def run_table4_date_fe_regression(
    data: pd.DataFrame,
    specification: dict,
) -> tuple[dict, list[dict]]:
    """估计同时吸收公司年度季度和交易日期固定效应的一个规格。"""
    x_cols = (
        TABLE4_CUE_COLS
        + TABLE4_CONTROL_COLS
    )

    small_firm_value = specification[
        "smallFirm取值"
    ]

    if small_firm_value is None:
        sample = data
    else:
        sample = data.loc[
            data["smallFirm"].eq(
                small_firm_value
            )
        ]

    needed = list(dict.fromkeys(
        [
            TABLE4_OUTCOME,
            "StkcdRecall",
            "Trddt",
            "yearQuarter",
        ]
        + x_cols
    ))

    reg = sample[needed].dropna().copy()

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

    regressors = reg[x_cols].astype("float64")
    regressors.insert(0, "const", 1.0)

    model = AbsorbingLS(
        dependent=reg[
            TABLE4_OUTCOME
        ].astype("float64"),
        exog=regressors,
        absorb=absorb,
        drop_absorbed=True,
    )

    result = model.fit(
        cov_type="clustered",
        clusters=clusters,
        debiased=True,
    )

    model_info = {
        "规格": specification["规格"],
        "样本": specification["样本"],
        "因变量": TABLE4_OUTCOME,
        "控制变量": "是",
        "被提示公司×年度季度固定效应": "是",
        "日期固定效应": "是",
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
        x_cols,
        start=1,
    ):
        estimated = variable in result.params.index

        if estimated:
            beta = float(result.params[variable])
            standard_error = float(
                result.std_errors[variable]
            )
            t_value = float(
                result.tstats[variable]
            )
            p_value = float(
                result.pvalues[variable]
            )
            stars = significance_stars(p_value)
            status = "已估计"
        else:
            beta = np.nan
            standard_error = np.nan
            t_value = np.nan
            p_value = np.nan
            stars = ""
            status = "被固定效应吸收"

        coefficient_rows.append({
            **model_info,
            "变量": variable,
            "变量顺序": variable_order,
            "变量类型": (
                "规模线索变量"
                if variable in TABLE4_CUE_COLS
                else "控制变量"
            ),
            "估计状态": status,
            "系数": beta,
            "聚类标准误": standard_error,
            "系数（百分点）": (
                beta * 100
                if pd.notna(beta)
                else np.nan
            ),
            "聚类标准误（百分点）": (
                standard_error * 100
                if pd.notna(standard_error)
                else np.nan
            ),
            "t值": t_value,
            "p值": p_value,
            "显著性": stars,
        })

    del (
        result,
        model,
        absorb,
        clusters,
        regressors,
        reg,
    )
    gc.collect()

    return model_info, coefficient_rows


def _format_table4_estimate(row: pd.Series) -> tuple[str, str]:
    """将规模异质性回归的百分点系数和标准误格式化为表格文字。"""
    coefficient = pd.to_numeric(
        pd.Series([row.get("系数（百分点）", np.nan)]),
        errors="coerce",
    ).iloc[0]
    standard_error = pd.to_numeric(
        pd.Series([row.get("聚类标准误（百分点）", np.nan)]),
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


def draw_table4_date_fe(
    summary: pd.DataFrame,
    core_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """在本脚本内直接绘制表4日期固定效应版的核心结果表。"""
    specifications = ["(1)", "(2)", "(3)"]
    model_rows = []
    for specification in specifications:
        matched = summary.loc[summary["规格"].eq(specification)]
        if len(matched) != 1:
            raise ValueError(f"表4图片缺少或重复规格：{specification}。")
        model_rows.append(matched.iloc[0])
    lookup = {
        (row["规格"], row["变量"]): row
        for _, row in core_results.iterrows()
    }
    labels = {
        "quarterLag1size90plus": "市值前10%的线索事件",
        "quarterLag1size50_90": "市值50%至90%的线索事件",
        "quarterLag1size10_50": "市值10%至50%的线索事件",
        "quarterLag1size10minus": "市值后10%的线索事件",
    }
    rows: list[list[str]] = []
    bold_rows: set[int] = set()
    for variable in TABLE4_CUE_COLS:
        coefficient_values, standard_error_values = [], []
        for model in model_rows:
            detail = lookup.get((model["规格"], variable))
            coefficient, standard_error = (
                _format_table4_estimate(detail)
                if detail is not None
                else ("", "")
            )
            coefficient_values.append(coefficient)
            standard_error_values.append(standard_error)
        bold_rows.add(len(rows))
        rows.append([labels[variable], *coefficient_values])
        rows.append(["", *standard_error_values])
    rows.extend([
        ["控制变量", *[str(row["控制变量"]) for row in model_rows]],
        [
            "交易日期固定效应",
            *[str(row["日期固定效应"]) for row in model_rows],
        ],
        [
            "被提示公司×年度季度固定效应",
            *[str(row["被提示公司×年度季度固定效应"]) for row in model_rows],
        ],
        ["样本量", *[f"{int(row['回归样本量']):,}" for row in model_rows]],
        ["R²", *[f"{float(row['R方']):.4f}" for row in model_rows]],
    ])
    column_labels = ["变量", "(1)\n全样本", "(2)\n小型公司", "(3)\n大型公司"]
    figure, axes = plt.subplots(figsize=(12.0, 9.2))
    figure.patch.set_facecolor("white")
    axes.axis("off")
    axes.text(
        0.01,
        0.975,
        "表 4：线索事件与被提示公司收益的规模异质性",
        ha="left",
        va="top",
        fontsize=16,
        fontproperties=FONT_BOLD,
    )
    axes.text(
        0.01,
        0.937,
        "因变量为被提示公司的特征调整收益率；加入交易日期固定效应；系数与标准误单位为百分点",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
        fontproperties=FONT,
    )
    table = axes.table(
        cellText=rows,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.52, 0.16, 0.16, 0.16],
        bbox=[0.01, 0.12, 0.98, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#9AA3AA")
        cell.set_linewidth(0.35)
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
        0.075,
        "注：所有模型同时吸收被提示公司×年度季度和交易日期固定效应；"
        "括号内为按被提示公司和交易日期双向聚类的标准误。",
        ha="left",
        va="top",
        fontsize=7.4,
        color="#444444",
        fontproperties=FONT,
    )
    axes.text(
        0.01,
        0.045,
        "*、**、*** 分别表示在 10%、5% 和 1% 水平上显著。",
        ha="left",
        va="top",
        fontsize=7.4,
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
    TABLE4_DATE_FE_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = read_table4_data()
    model_rows = []
    coefficient_rows = []

    for specification in TABLE4_SPECIFICATIONS:
        print(
            "正在运行加入日期固定效应的表4："
            f"{specification['规格']}，"
            f"{specification['样本']}"
        )

        model_info, current_coefficients = (
            run_table4_date_fe_regression(
                data,
                specification,
            )
        )
        model_rows.append(model_info)
        coefficient_rows.extend(
            current_coefficients
        )

    results = pd.DataFrame(model_rows)
    coefficients = pd.DataFrame(
        coefficient_rows
    )

    core_results = (
        coefficients.loc[
            coefficients["变量"].isin(
                TABLE4_CUE_COLS
            )
        ]
        .copy()
    )

    core_results.to_csv(
        TABLE4_DATE_FE_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    coefficients.to_csv(
        TABLE4_DATE_FE_COEFFICIENTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    draw_table4_date_fe(
        results,
        core_results,
        TABLE4_DATE_FE_IMAGE_PATH,
    )

    print("\n表4：加入日期固定效应的规模异质性核心结果")
    print(core_results.to_string(index=False))
    print(
        "\n表4核心结果已保存："
        f"{TABLE4_DATE_FE_RESULTS_PATH}"
    )
    print(
        "表4全部系数明细已保存："
        f"{TABLE4_DATE_FE_COEFFICIENTS_PATH}"
    )
    print(f"表4图片已保存：{TABLE4_DATE_FE_IMAGE_PATH}")

    del (
        data,
        results,
        coefficients,
        core_results,
    )
    gc.collect()


if __name__ == "__main__":
    main()
