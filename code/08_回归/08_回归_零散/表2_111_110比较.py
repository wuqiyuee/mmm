"""
分别估计 111 与 110 两类提示事件的表2收益回归。

两组定义：
* 111：当前提示公司是模式公司，历史提示公司是模式公司，
  历史被提示公司也是模式公司；
* 110：当前提示公司是模式公司，历史提示公司是模式公司，
  历史被提示公司不是模式公司。

本脚本不把两类事件同时放进同一个回归，而是使用同一份完整日度面板，
分别合并 CueEvents_111.csv 与 CueEvents_110.csv，依次运行论文表2的
六种规格。这样两组的控制变量、固定效应、样本期和聚类方式完全一致，
可直接比较两张结果表中的提示系数。

前置脚本：
1. code/03_记忆矩阵/full_sample记忆矩阵/构造111_110记忆矩阵.py
2. code/04_合并数据/构造111_110提示事件.py

默认输入是 Full样本_最终回归面板.csv，因为 110 组的被提示公司在历史季度
上被定义为非模式公司，不能只使用原来的模式公司面板。当前项目面板中的新闻
变量名称为 aesIncrease / aesDecrease。
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from matplotlib.font_manager import FontProperties


GROUPS = ("111", "110")

BASE_PANEL_RELATIVE_PATH = Path(
    "中间结果/04_合并数据/Full样本_最终回归面板.csv"
)
EVENT_RELATIVE_PATHS = {
    group: Path(f"中间结果/04_合并数据/CueEvents_{group}.csv")
    for group in GROUPS
}
OUTPUT_RESULTS_RELATIVE_PATH = Path(
    "中间结果/08_回归结果/表2_111_110模型汇总.csv"
)
OUTPUT_COEFFICIENTS_RELATIVE_PATH = Path(
    "中间结果/08_回归结果/表2_111_110系数明细.csv"
)
IMAGE_RELATIVE_DIR = Path("图表结果/论文复现/表格/03_扩展分析/提示事件比较")

FONT_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
if FONT_PATH.exists():
    FONT = FontProperties(fname=str(FONT_PATH))
    FONT_BOLD = FontProperties(fname=str(FONT_PATH), weight="bold")
else:
    FONT = FontProperties(family="sans-serif")
    FONT_BOLD = FontProperties(family="sans-serif", weight="bold")

plt.rcParams["axes.unicode_minus"] = False

# 默认研究期：2005—2025年（含首尾年）。
START_YEAR: int | None = 2005
END_YEAR: int | None = 2025


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

TABLE2_FULL_CONTROLS = OWN_EA_COLS + DISCLOSURE_COLS + MEDIA_COLS
BASE_PANEL_COLS = list(
    dict.fromkeys(
        [
            "StkcdRecall",
            "Trddt",
            "year",
            "yearQuarter",
        ]
        + RETURN_COLS
        + TABLE2_FULL_CONTROLS
    )
)
NUMERIC_BASE_COLS = [
    col
    for col in BASE_PANEL_COLS
    if col not in {"StkcdRecall", "Trddt", "yearQuarter"}
]


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
    ("chAdjreturn_t_tp1_Recall_w", "[t, t+1]"),
    ("chAdjreturn_t_tp10_Recall_w", "[t, t+10]"),
]


def normalize_stock_code(value: object) -> str:
    """把证券代码标准化为六位字符串。"""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return f"{int(float(text)):06d}"
    except ValueError:
        return text.zfill(6)


def find_project_root(start: Path | None = None) -> Path:
    """从当前文件或传入目录向上寻找项目根目录。"""
    start_path = Path(start or __file__).resolve()
    search_start = start_path if start_path.is_dir() else start_path.parent
    for candidate in [search_start, *search_start.parents]:
        if (candidate / "input").exists() and (candidate / "code").exists():
            return candidate
    return search_start


def significance_stars(p_value: float) -> str:
    """按论文常用阈值添加显著性星号。"""
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def load_base_panel(root: Path) -> pd.DataFrame:
    """读取完整被提示公司日度面板，作为两组回归的共同底表。"""
    path = root / BASE_PANEL_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(f"找不到完整日度面板：{path}")

    dtype = {
        "StkcdRecall": "string",
        "yearQuarter": "string",
    }
    dtype.update({col: "float32" for col in NUMERIC_BASE_COLS})

    panel = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=BASE_PANEL_COLS,
        dtype=dtype,
        parse_dates=["Trddt"],
    )
    panel["StkcdRecall"] = panel["StkcdRecall"].apply(
        normalize_stock_code
    )
    panel["Trddt"] = pd.to_datetime(panel["Trddt"], errors="coerce")
    panel = panel.dropna(subset=["StkcdRecall", "Trddt"]).copy()
    panel = panel.loc[panel["StkcdRecall"].ne("")].copy()

    if START_YEAR is not None:
        panel = panel.loc[panel["year"] >= START_YEAR].copy()
    if END_YEAR is not None:
        panel = panel.loc[panel["year"] <= END_YEAR].copy()

    duplicate_count = panel.duplicated(["StkcdRecall", "Trddt"]).sum()
    if duplicate_count:
        raise ValueError(
            "完整日度面板的 StkcdRecall + Trddt 不是唯一键，"
            "不能安全合并两组提示事件。"
        )

    print(f"共同完整日度面板行数：{len(panel):,}")
    return panel


def merge_group_event(
    panel: pd.DataFrame,
    root: Path,
    group: str,
) -> pd.DataFrame:
    """将某一组正向提示事件合并到共同底表，未命中日期记为0。"""
    path = root / EVENT_RELATIVE_PATHS[group]
    if not path.exists():
        raise FileNotFoundError(
            f"找不到{group}提示事件表：{path}；请先运行前两个脚本。"
        )

    cue_col = f"_cue_{group}"
    events = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=["StkcdRecall", "date", "quarterLag1size90plus"],
        dtype={"StkcdRecall": "string"},
        parse_dates=["date"],
    )
    events["StkcdRecall"] = events["StkcdRecall"].apply(
        normalize_stock_code
    )
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events[cue_col] = pd.to_numeric(
        events["quarterLag1size90plus"], errors="coerce"
    ).fillna(0).astype("int8")
    events = events.rename(columns={"date": "Trddt"})[
        ["StkcdRecall", "Trddt", cue_col]
    ]
    events = events.dropna(subset=["StkcdRecall", "Trddt"])
    events = events.loc[events["StkcdRecall"].ne("")].copy()

    duplicate_count = events.duplicated(["StkcdRecall", "Trddt"]).sum()
    if duplicate_count:
        raise ValueError(f"{group}提示事件表存在重复的被提示公司—日期。")

    merged = panel.merge(
        events,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="one_to_one",
    )
    merged["quarterLag1size90plus"] = merged[cue_col].fillna(0).astype(
        "int8"
    )
    merged = merged.drop(columns=[cue_col])

    print(
        f"{group}组命中完整面板的公司—日期观测数："
        f"{int(merged['quarterLag1size90plus'].sum()):,}"
    )
    return merged


def drop_fe_singletons_iteratively(
    reg: pd.DataFrame,
    fe_code_cols: list[str],
) -> tuple[pd.DataFrame, int]:
    """递归删除高维固定效应中的单例观测。"""
    if not fe_code_cols or reg.empty:
        return reg, 0

    total_removed = 0
    while True:
        singleton = np.zeros(len(reg), dtype=bool)
        for col in fe_code_cols:
            codes = reg[col].to_numpy(dtype=np.int64, copy=False)
            counts = np.bincount(codes)
            singleton |= counts[codes] == 1

        removed_now = int(singleton.sum())
        if removed_now == 0:
            break

        total_removed += removed_now
        reg = reg.loc[~singleton].copy()

    return reg, total_removed


def run_return_regression(
    data: pd.DataFrame,
    outcome: str,
    window_label: str,
    specification: dict,
) -> tuple[dict, list[dict]]:
    """运行一组中的一个收益窗口和一个固定效应规格。"""
    cue_col = "quarterLag1size90plus"
    control_cols = list(specification["控制变量"])
    fe_names = list(specification["固定效应"])
    x_cols = [cue_col] + control_cols

    needed = [outcome, "StkcdRecall", "Trddt"] + x_cols
    if "firm_yearquarter" in fe_names:
        needed.append("yearQuarter")
    needed = list(dict.fromkeys(needed))

    reg = data[needed].dropna().copy()
    singleton_removed = 0
    fe_code_cols: list[str] = []

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
            reg["Trddt"], sort=False
        )[0].astype("int64")
        fe_code_cols.append("_fe_date")

    reg, singleton_removed = drop_fe_singletons_iteratively(
        reg, fe_code_cols
    )

    base_row = {
        "收益窗口": window_label,
        "因变量": outcome,
        "规格": specification["规格"],
        "规格说明": specification["规格说明"],
        "日期固定效应": "是" if "date" in fe_names else "否",
        "被提示公司×年度季度固定效应": (
            "是" if "firm_yearquarter" in fe_names else "否"
        ),
        "删除单例组观测数": singleton_removed,
    }

    if reg.empty:
        model_row = {
            **base_row,
            "估计状态": "无可用观测",
            "线索系数": np.nan,
            "聚类标准误": np.nan,
            "线索系数（百分点）": np.nan,
            "聚类标准误（百分点）": np.nan,
            "t值": np.nan,
            "p值": np.nan,
            "显著性": "",
            "回归样本量": 0,
            "被提示公司聚类数": 0,
            "日期聚类数": 0,
            "R方": np.nan,
        }
        return model_row, []

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
            "firm": pd.factorize(reg["StkcdRecall"], sort=False)[0],
            "date": pd.factorize(reg["Trddt"], sort=False)[0],
        },
        index=reg.index,
    )
    x = reg[x_cols].astype("float64")
    x.insert(0, "const", 1.0)

    model = AbsorbingLS(
        dependent=reg[outcome].astype("float64"),
        exog=x,
        absorb=absorb,
        drop_absorbed=True,
    )
    result = model.fit(
        cov_type="clustered",
        clusters=clusters,
        debiased=True,
    )

    if cue_col in result.params.index:
        beta = float(result.params[cue_col])
        standard_error = float(result.std_errors[cue_col])
        t_value = float(result.tstats[cue_col])
        p_value = float(result.pvalues[cue_col])
        status = "正常估计"
    else:
        beta = standard_error = t_value = p_value = np.nan
        status = "线索变量被固定效应吸收"

    model_row = {
        **base_row,
        "估计状态": status,
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
        "R方": float(result.rsquared),
    }

    coefficient_rows = []
    variable_order = ["const"] + x_cols
    for order, variable in enumerate(variable_order, start=1):
        if variable not in result.params.index:
            coefficient_rows.append(
                {
                    "变量顺序": order,
                    "变量": variable,
                    "变量类型": (
                        "提示事件" if variable == cue_col else "控制变量"
                    ),
                    "估计状态": "被吸收或未估计",
                    "系数": np.nan,
                    "聚类标准误": np.nan,
                    "系数（百分点）": np.nan,
                    "聚类标准误（百分点）": np.nan,
                    "t值": np.nan,
                    "p值": np.nan,
                    "显著性": "",
                }
            )
            continue

        coefficient = float(result.params[variable])
        standard_error = float(result.std_errors[variable])
        p_value = float(result.pvalues[variable])
        coefficient_rows.append(
            {
                "变量顺序": order,
                "变量": variable,
                "变量类型": (
                    "提示事件" if variable == cue_col else "控制变量"
                ),
                "估计状态": "正常估计",
                "系数": coefficient,
                "聚类标准误": standard_error,
                "系数（百分点）": coefficient * 100,
                "聚类标准误（百分点）": standard_error * 100,
                "t值": float(result.tstats[variable]),
                "p值": p_value,
                "显著性": significance_stars(p_value),
            }
        )

    for row in coefficient_rows:
        row.update(
            {
                "收益窗口": window_label,
                "因变量": outcome,
                "规格": specification["规格"],
                "规格说明": specification["规格说明"],
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
            }
        )

    del result, model, absorb, clusters, x, reg
    gc.collect()
    return model_row, coefficient_rows


def run_group_regressions(
    panel: pd.DataFrame,
    group: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对单个组运行两个收益窗口×六种规格。"""
    model_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    sample_label = (
        "111：模式—模式（当前提示、历史提示、历史被提示均为模式）"
        if group == "111"
        else "110：模式—非模式（当前提示和历史提示为模式，历史被提示为非模式）"
    )

    for outcome, window_label in TABLE2_OUTCOMES:
        for specification in TABLE2_SPECIFICATIONS:
            print(
                f"正在运行{sample_label}：{window_label}，"
                f"规格 {specification['规格']}"
            )
            model_row, coefficient_part = run_return_regression(
                data=panel,
                outcome=outcome,
                window_label=window_label,
                specification=specification,
            )
            model_rows.append(
                {
                    "样本": group,
                    "样本说明": sample_label,
                    **model_row,
                }
            )
            coefficient_rows.extend(
                {
                    "样本": group,
                    "样本说明": sample_label,
                    **row,
                }
                for row in coefficient_part
            )

    return pd.DataFrame(model_rows), pd.DataFrame(coefficient_rows)


def _result_number(value: object) -> float:
    """将回归输出中的单个值安全转换为数值。"""
    return float(
        pd.to_numeric(
            pd.Series([value]),
            errors="coerce",
        ).iloc[0]
    )


def _format_percentage_estimate(row: pd.Series) -> tuple[str, str]:
    """以百分点显示表2比较回归的系数和双向聚类标准误。"""
    coefficient = _result_number(row.get("系数（百分点）", np.nan))
    standard_error = _result_number(
        row.get("聚类标准误（百分点）", np.nan)
    )
    p_value = _result_number(row.get("p值", np.nan))
    if np.isnan(coefficient) or np.isnan(standard_error):
        return "", ""
    stars = ""
    if not np.isnan(p_value):
        if p_value < 0.01:
            stars = "***"
        elif p_value < 0.05:
            stars = "**"
        elif p_value < 0.10:
            stars = "*"
    return f"{coefficient:.4f}{stars}", f"({standard_error:.4f})"


def _display_variable(variable: str) -> str:
    """将111/110比较回归的变量名转换为图片表格标签。"""
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
    lag = re.fullmatch(r"ownEA_tm(\d+)", variable)
    if lag:
        return f"自身盈余公告 [t-{lag.group(1)}]"
    lead = re.fullmatch(r"ownEA_tp(\d+)", variable)
    if lead:
        return f"自身盈余公告 [t+{lead.group(1)}]"
    return variable


def _ordered_comparison_models(
    summary: pd.DataFrame,
    group: str,
) -> list[pd.Series]:
    """按收益窗口和规格顺序取得111或110组的十二个回归。"""
    selected = summary.loc[summary["样本"].eq(group)]
    model_rows = []
    for window in ("[t, t+1]", "[t, t+10]"):
        for specification in ("(1)", "(2)", "(3)", "(4)", "(5)", "(6)"):
            matched = selected.loc[
                selected["收益窗口"].eq(window)
                & selected["规格"].eq(specification)
            ]
            if len(matched) != 1:
                raise ValueError(
                    "111/110比较完整系数表缺少模型："
                    f"组别={group}，窗口={window}，规格={specification}。"
                )
            model_rows.append(matched.iloc[0])
    return model_rows


def _setting_values(model_rows: list[pd.Series], setting: str) -> list[str]:
    """从六种表2规格生成图片表格中的模型设定行。"""
    values = []
    for row in model_rows:
        number = int(str(row["规格"]).strip("()"))
        if setting == "own":
            enabled = number >= 2
        elif setting == "other":
            enabled = number >= 3
        elif setting == "date":
            enabled = number in {4, 6}
        else:
            enabled = number in {5, 6}
        values.append("是" if enabled else "否")
    return values


def _save_comparison_page(
    model_rows: list[pd.Series],
    lookup: dict[tuple[str, str, str], pd.Series],
    variables: list[str],
    output_path: Path,
    *,
    sample_label: str,
    continued: bool,
    include_model_stats: bool,
) -> None:
    """直接用 matplotlib（绘图库）保存111或110完整系数表的一页。"""
    column_labels = ["变量"] + [
        f"{row['收益窗口']}\n{row['规格']}"
        for row in model_rows
    ]
    values: list[list[str]] = []
    bold_rows: set[int] = set()
    for variable in variables:
        coefficients, standard_errors = [], []
        for row in model_rows:
            detail = lookup.get((row["收益窗口"], row["规格"], variable))
            coefficient, standard_error = (
                _format_percentage_estimate(detail)
                if detail is not None
                else ("", "")
            )
            coefficients.append(coefficient)
            standard_errors.append(standard_error)
        bold_rows.add(len(values))
        values.append([_display_variable(variable), *coefficients])
        values.append(["", *standard_errors])

    if include_model_stats:
        values.extend([
            ["自身盈余公告控制", *_setting_values(model_rows, "own")],
            ["其他控制变量", *_setting_values(model_rows, "other")],
            ["交易日期固定效应", *_setting_values(model_rows, "date")],
            ["被提示公司×年度季度固定效应", *_setting_values(model_rows, "firm")],
            [
                "样本量",
                *[f"{int(_result_number(row['回归样本量'])):,}" for row in model_rows],
            ],
            ["R²", *[f"{_result_number(row['R方']):.4f}" for row in model_rows]],
        ])

    figure, axes = plt.subplots(figsize=(16.5, 11.7))
    figure.patch.set_facecolor("white")
    axes.axis("off")
    axes.text(
        0.01,
        0.975,
        "表 2：111与110提示事件比较" + "（续）" if continued else "表 2：111与110提示事件比较",
        ha="left",
        va="top",
        fontsize=15,
        fontproperties=FONT_BOLD,
    )
    axes.text(
        0.01,
        0.934,
        "线索事件与被提示公司收益",
        ha="left",
        va="top",
        fontsize=12.5,
        fontproperties=FONT_BOLD,
    )
    axes.text(
        0.01,
        0.901,
        (
            f"{sample_label}；因变量为被提示公司的特征调整收益率；"
            "系数与括号内双向聚类标准误的单位均为百分点"
        ),
        ha="left",
        va="top",
        fontsize=8.1,
        color="#444444",
        fontproperties=FONT,
    )
    table = axes.table(
        cellText=values,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.24] + [0.76 / 12] * 12,
        bbox=[0.01, 0.115, 0.98, 0.745],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.8)
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
        0.07,
        "注：空白单元格表示该规格未纳入相应控制变量。"
        if include_model_stats
        else "续见下一页。",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#444444",
        fontproperties=FONT,
    )
    if include_model_stats:
        axes.text(
            0.01,
            0.042,
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


def draw_comparison_full_two_pages(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    group: str,
    first_path: Path,
    second_path: Path,
) -> None:
    """生成111或110样本的完整系数两页图片，不导入外部绘图脚本。"""
    model_rows = _ordered_comparison_models(summary, group)
    selected = details.loc[
        details["样本"].eq(group) & details["变量"].ne("const")
    ]
    variables = (
        selected[["变量", "变量顺序"]]
        .drop_duplicates()
        .sort_values("变量顺序")["变量"]
        .tolist()
    )
    if "ownEA_tp3" not in variables:
        raise ValueError("111/110比较完整系数表找不到分页变量 ownEA_tp3。")
    split_index = variables.index("ownEA_tp3") + 1
    lookup = {
        (row["收益窗口"], row["规格"], row["变量"]): row
        for _, row in selected.iterrows()
    }
    sample_label = str(model_rows[0]["样本说明"])
    _save_comparison_page(
        model_rows,
        lookup,
        variables[:split_index],
        first_path,
        sample_label=sample_label,
        continued=False,
        include_model_stats=False,
    )
    _save_comparison_page(
        model_rows,
        lookup,
        variables[split_index:],
        second_path,
        sample_label=sample_label,
        continued=True,
        include_model_stats=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分别运行111与110两类提示事件的表2回归。"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="项目根目录；省略时自动寻找。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root or find_project_root()).resolve()
    print(f"项目根目录：{root}")

    base_panel = load_base_panel(root)
    all_model_rows = []
    all_coefficient_rows = []

    for group in GROUPS:
        group_panel = merge_group_event(base_panel, root, group)
        model_rows, coefficient_rows = run_group_regressions(
            group_panel, group
        )
        all_model_rows.append(model_rows)
        all_coefficient_rows.append(coefficient_rows)
        del group_panel, model_rows, coefficient_rows
        gc.collect()

    results = pd.concat(all_model_rows, ignore_index=True)
    coefficients = pd.concat(all_coefficient_rows, ignore_index=True)

    results_path = root / OUTPUT_RESULTS_RELATIVE_PATH
    coefficients_path = root / OUTPUT_COEFFICIENTS_RELATIVE_PATH
    results_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients_path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )
    coefficients.to_csv(
        coefficients_path,
        index=False,
        encoding="utf-8-sig",
    )

    image_dir = root / IMAGE_RELATIVE_DIR
    for group in GROUPS:
        draw_comparison_full_two_pages(
            results,
            coefficients,
            group,
            image_dir / f"table2_111_110比较_{group}_完整系数版_独立脚本.png",
            image_dir / f"table2_111_110比较_{group}_完整系数版_独立脚本_续页.png",
        )

    print(f"表2模型汇总已输出：{results_path}")
    print(f"表2系数明细已输出：{coefficients_path}")
    print(f"表2比较完整系数图片已输出：{image_dir}")


if __name__ == "__main__":
    main()
