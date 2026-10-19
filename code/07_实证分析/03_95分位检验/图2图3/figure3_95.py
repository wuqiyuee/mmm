"""
在不覆盖现有图3的前提下，额外估计并绘制加入交易日期固定效应的
散户订单不平衡动态。

固定效应：
1. 被提示公司 × 年度季度；
2. 交易日期。

标准误仍按被提示公司和交易日期双向聚类。其余样本、变量、缩尾和
事件窗口口径与现有图3保持一致。
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats


def _find_project_root() -> Path:
    for ancestor in Path(__file__).resolve().parents:
        if (
            (ancestor / "AGENTS.md").is_file()
            and (ancestor / "code").is_dir()
            and (ancestor / "中间结果").is_dir()
        ):
            return ancestor
    raise FileNotFoundError("找不到包含 AGENTS.md、code 和中间结果的项目根目录。")


ROOT = _find_project_root()

# 两张95分位图复用统一的日期固定效应回归模块。
MODULE_DIR = ROOT / "code" / "07_实证分析" / "90_公共模块"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
from 图2图3扩展_日期固定效应回归公共模块 import (
    residualize_regression_data as residualize_regression_data_shared,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager



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

RESULT_PATH = ROOT / "图表结果" / "03_95分位检验" / "图3_散户订单不平衡" / "图3_95分位_散户订单不平衡动态_日期固定效应结果.csv"

TABLE3_DATE_FE_RESULT_PATH = ROOT / "__q95_no_table3_anchor__.csv"

OUTPUT_PATH = ROOT / "图表结果" / "03_95分位检验" / "图3_散户订单不平衡" / "figure3_95分位_散户订单不平衡动态_加入日期固定效应.png"


CUE_COL = "quarterLag1size95plus"

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

CONTROL_COLS = (
    OWN_EA_COLS
    + DISCLOSURE_COLS
    + MEDIA_COLS
)

RHS_COLS = [CUE_COL] + CONTROL_COLS

HORIZONS = list(range(1, 51))
START_YEAR = 2005
END_YEAR = 2025

# 双侧显著性水平与置信区间的对应关系：
# 1%、5%、10% 显著性水平分别对应 99%、95%、90% 置信区间。
# 从最长区间到最短区间依次绘制。采用浅灰、蓝灰、深蓝配色，
# 同时用递增线宽区分层级；不绘制端帽，减少50个点的视觉噪声。
ERROR_BAR_SPECS = (
    {
        "label": "99%置信区间（1%水平）",
        "critical_value": 2.5758293035489004,
        "color": "#bdbdbd",
        "line_width": 0.8,
        "zorder": 1,
    },
    {
        "label": "95%置信区间（5%水平）",
        "critical_value": 1.959963984540054,
        "color": "#78909c",
        "line_width": 1.6,
        "zorder": 2,
    },
    {
        "label": "90%置信区间（10%水平）",
        "critical_value": 1.6448536269514722,
        "color": "#1f4e79",
        "line_width": 2.8,
        "zorder": 3,
    },
)

RESIDUALIZATION_BATCH_SIZE = 4

OIB_RAW_COLS = [
    f"OIBvol_avg_t_tp{horizon}"
    for horizon in HORIZONS
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


def cluster_score_meat(
    x: np.ndarray,
    residuals: np.ndarray,
    cluster_codes: np.ndarray,
    cluster_count: int,
) -> np.ndarray:
    """计算一个聚类维度的未缩放得分外积和。"""
    group_scores = np.empty(
        (cluster_count, x.shape[1]),
        dtype=np.float64,
    )

    for column in range(x.shape[1]):
        group_scores[:, column] = np.bincount(
            cluster_codes,
            weights=(
                x[:, column]
                * residuals
            ),
            minlength=cluster_count,
        )

    return group_scores.T @ group_scores


def observation_score_meat(
    x: np.ndarray,
    residuals: np.ndarray,
    chunk_size: int = 250_000,
) -> np.ndarray:
    """
    计算公司×日期交集聚类项。

    主样本以公司和日期为唯一键，所以交集聚类中的每组只有一条
    观测。分块计算可避免构造整张得分矩阵。
    """
    meat = np.zeros(
        (x.shape[1], x.shape[1]),
        dtype=np.float64,
    )

    for start in range(
        0,
        len(x),
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            len(x),
        )
        scores = (
            x[start:stop]
            * residuals[start:stop, None]
        )
        meat += scores.T @ scores

    return meat


def two_way_cluster_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    firm_codes: np.ndarray,
    date_codes: np.ndarray,
    xpx_inverse: np.ndarray,
) -> np.ndarray:
    """
    计算与 linearmodels 聚类协方差相同的双向聚类协方差。

    直接使用残差化后的普通最小二乘得分，避免通用工具重复计算
    大型伪逆，并利用公司×日期唯一键直接计算交集聚类项。
    """
    nobs, nvars = x.shape
    firm_count = int(firm_codes.max()) + 1
    date_count = int(date_codes.max()) + 1

    firm_meat = cluster_score_meat(
        x,
        residuals,
        firm_codes,
        firm_count,
    )
    date_meat = cluster_score_meat(
        x,
        residuals,
        date_codes,
        date_count,
    )
    intersection_meat = observation_score_meat(
        x,
        residuals,
    )

    common_scale = (
        (nobs - 1)
        / (nobs - nvars)
    )
    firm_scale = (
        firm_count
        / (firm_count - 1)
        * common_scale
    )
    date_scale = (
        date_count
        / (date_count - 1)
        * common_scale
    )

    # 公司×日期是一行一个交集组，因此组数等于样本量。
    intersection_scale = (
        nobs
        / (nobs - 1)
        * common_scale
    )

    meat = (
        firm_scale * firm_meat
        + date_scale * date_meat
        - intersection_scale
        * intersection_meat
    )

    covariance = (
        xpx_inverse
        @ meat
        @ xpx_inverse
    )

    return (
        covariance + covariance.T
    ) / 2


def residualize_regression_data(
    reg: pd.DataFrame,
    outcome: str,
    fe_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """调用公共季度直接解，并保留本脚本因变量的二维列向量结构。"""
    y_residualized, x_residualized = residualize_regression_data_shared(
        reg,
        outcome,
        RHS_COLS,
        fe_ids,
        batch_size=RESIDUALIZATION_BATCH_SIZE,
    )
    return y_residualized.reshape(-1, 1), x_residualized


def read_figure3_panel() -> pd.DataFrame:
    """只读取图3各窗口共同使用的主样本字段。"""
    panel_cols = list(dict.fromkeys(
        [
            "StkcdRecall",
            "Trddt",
            "yearQuarter",
        ]
        + [column for column in RHS_COLS if column != CUE_COL]
    ))

    panel_dtype = {
        "StkcdRecall": "string",
        "yearQuarter": "string",
        **{
            column: "float32"
            for column in RHS_COLS
        },
    }

    print("读取加入日期固定效应的图3主样本……")
    data = pd.read_csv(
        PATTERN_PATH,
        usecols=panel_cols,
        dtype=panel_dtype,
        parse_dates=["Trddt"],
    )

    # 独立95%口径：从q95临时事件映射接入 cue95，不改正式面板。
    event_path = ROOT / "work/q95_tables_run/cue95_event_map.csv"
    event_map = pd.read_csv(event_path, dtype={"firm": "int64", "day": "int64"})
    event_keys = (event_map["firm"].to_numpy(dtype="int64") * 100_000
                  + event_map["day"].to_numpy(dtype="int64"))
    data_keys = (data["StkcdRecall"].to_numpy(dtype="int64") * 100_000
                 + data["Trddt"].to_numpy(dtype="datetime64[D]").astype("int64"))
    data[CUE_COL] = np.isin(data_keys, event_keys).astype("float32")
    del event_map, event_keys, data_keys
    data["StkcdRecall"] = (
        data["StkcdRecall"]
        .str.strip()
        .str.zfill(6)
    )

    if data.duplicated(
        ["StkcdRecall", "Trddt"]
    ).any():
        raise ValueError(
            "图3主样本的被提示公司和交易日期不是唯一键。"
        )

    data = data.loc[
        data["Trddt"].dt.year.between(START_YEAR, END_YEAR)
    ].copy()

    # 这两个分组在所有窗口中定义相同，预先编号可避免50次重复
    # groupby/factorize；窗口筛选只会删除行，不会改变组的归属。
    data["_fe_firm_yearquarter_all"] = (
        data.groupby(
            [
                "StkcdRecall",
                "yearQuarter",
            ],
            observed=True,
            sort=False,
        )
        .ngroup()
        .astype("int32")
    )
    data["_fe_date_all"] = pd.factorize(
        data["Trddt"],
        sort=False,
    )[0].astype("int32")

    return data


def read_figure3_batch(
    panel: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """
    分批读取订单不平衡指标并合并。

    这样可以避免一次把50个大样本因变量与回归工作表同时留在内存，
    减少内存交换，但不改变逐窗口的样本或估计口径。
    """
    raw_columns = [
        f"OIBvol_avg_t_tp{horizon}"
        for horizon in horizons
    ]

    oib_dtype = {
        "Stkcd": "string",
        **{
            column: "float32"
            for column in raw_columns
        },
    }

    print(
        "读取图3订单不平衡数据批次："
        f"{horizons}"
    )
    oib = pd.read_csv(
        OIB_PATH,
        usecols=[
            "Stkcd",
            "Trddt",
        ] + raw_columns,
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
            "图3订单不平衡数据的公司和交易日期不是唯一键。"
        )

    data = panel.merge(
        oib,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="one_to_one",
    )

    del oib
    gc.collect()

    print(
        "对当前批次订单不平衡指标进行全样本1%缩尾……"
    )

    for raw_column in raw_columns:
        valid = pd.to_numeric(
            data[raw_column],
            errors="coerce",
        )
        valid = valid.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        lower = valid.quantile(0.01)
        upper = valid.quantile(0.99)

        data[raw_column] = (
            valid.clip(
                lower=lower,
                upper=upper,
            )
            .astype("float32")
        )

    data = data.rename(
        columns={
            raw_column: f"{raw_column}_w"
            for raw_column in raw_columns
        }
    )

    data["StkcdRecall"] = (
        data["StkcdRecall"].astype("category")
    )
    data["yearQuarter"] = (
        data["yearQuarter"].astype("category")
    )

    return data


def run_date_fe_regression(
    data: pd.DataFrame,
    horizon: int,
) -> dict:
    """估计一个事件窗口的日期固定效应版图3回归。"""
    outcome = f"OIBvol_avg_t_tp{horizon}_w"

    needed = list(dict.fromkeys(
        [
            outcome,
            "StkcdRecall",
            "Trddt",
            "yearQuarter",
            "_fe_firm_yearquarter_all",
            "_fe_date_all",
        ]
        + RHS_COLS
    ))

    reg = data[needed].dropna().copy()

    reg["_fe_firm_yearquarter"] = (
        reg["_fe_firm_yearquarter_all"]
        .astype("int64")
    )

    reg["_fe_date"] = (
        reg["_fe_date_all"]
        .astype("int64")
    )

    reg, singleton_removed = (
        drop_fe_singletons_iteratively(
            reg,
            [
                "_fe_firm_yearquarter",
                "_fe_date",
            ],
        )
    )

    fe_ids = np.column_stack([
        reg["_fe_firm_yearquarter"].to_numpy(
            dtype=np.int64,
            copy=False,
        ),
        reg["_fe_date"].to_numpy(
            dtype=np.int64,
            copy=False,
        ),
    ])

    firm_codes = pd.factorize(
        reg["StkcdRecall"],
        sort=False,
    )[0].astype(
        np.int64,
        copy=False,
    )
    date_codes = pd.factorize(
        reg["Trddt"],
        sort=False,
    )[0].astype(
        np.int64,
        copy=False,
    )

    (
        y_residualized,
        x_residualized,
    ) = residualize_regression_data(
        reg,
        outcome,
        fe_ids,
    )

    xpx = (
        x_residualized.T
        @ x_residualized
    )
    xpy = (
        x_residualized.T
        @ y_residualized
    )
    xpx_condition = float(
        np.linalg.cond(xpx)
    )

    if (
        np.isfinite(xpx_condition)
        and xpx_condition < 1e12
    ):
        parameters = np.linalg.solve(
            xpx,
            xpy,
        ).reshape(-1)

        xpx_inverse = np.linalg.inv(xpx)

        coefficient_solver = (
            "正规方程（条件数已检查）"
        )
    else:
        parameters = np.linalg.lstsq(
            x_residualized,
            y_residualized,
            rcond=None,
        )[0].reshape(-1)

        xpx_inverse = np.linalg.pinv(xpx)

        coefficient_solver = (
            "最小二乘分解"
        )

    residuals = (
        y_residualized.reshape(-1)
        - x_residualized @ parameters
    )

    covariance = two_way_cluster_covariance(
        x_residualized,
        residuals,
        firm_codes,
        date_codes,
        xpx_inverse,
    )
    standard_errors = np.sqrt(
        np.diag(covariance)
    )

    column_names = RHS_COLS
    if CUE_COL not in column_names:
        raise ValueError(
            f"{outcome} 中的线索变量被固定效应完全吸收。"
        )

    cue_position = column_names.index(
        CUE_COL
    )
    coefficient = float(
        parameters[cue_position]
    )
    standard_error = float(
        standard_errors[cue_position]
    )
    t_value = (
        coefficient / standard_error
    )

    nobs = len(reg)
    absorbed_parameter_count = (
        reg["_fe_firm_yearquarter"].nunique()
        + reg["_fe_date"].nunique()
        - 1
    )
    model_parameter_count = int(
        absorbed_parameter_count
        + x_residualized.shape[1]
    )
    residual_degrees = (
        nobs - model_parameter_count
    )
    p_value = float(
        2
        * stats.t.sf(
            abs(t_value),
            residual_degrees,
        )
    )

    original_y = reg[outcome].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    centered_y = (
        original_y - original_y.mean()
    )
    residual_sum_squares = float(
        residuals @ residuals
    )
    total_sum_squares = float(
        centered_y @ centered_y
    )
    r_squared = max(
        1
        - residual_sum_squares
        / total_sum_squares,
        0.0,
    )

    row = {
        "窗口序号": horizon,
        "事件窗口": f"[t, t+{horizon}]",
        "因变量": outcome,
        "线索系数": coefficient,
        "聚类标准误": standard_error,
        "置信区间下界": (
            coefficient - 1.96 * standard_error
        ),
        "置信区间上界": (
            coefficient + 1.96 * standard_error
        ),
        "t值": t_value,
        "p值": p_value,
        "显著性": significance_stars(p_value),
        "控制变量": "是",
        "日期固定效应": "是",
        "被提示公司×年度季度固定效应": "是",
        "回归样本量": nobs,
        "被提示公司聚类数": int(
            reg["StkcdRecall"].nunique()
        ),
        "日期聚类数": int(
            reg["Trddt"].nunique()
        ),
        "删除单例组观测数": singleton_removed,
        "解释变量矩阵条件数": xpx_condition,
        "系数求解方法": coefficient_solver,
        "R方": r_squared,
    }

    del (
        fe_ids,
        firm_codes,
        date_codes,
        covariance,
        xpx,
        xpy,
        xpx_inverse,
        x_residualized,
        y_residualized,
        parameters,
        standard_errors,
        residuals,
        original_y,
        centered_y,
        reg,
    )
    gc.collect()

    return row


def save_results(rows: list[dict]) -> pd.DataFrame:
    """按窗口排序并保存结构化结果。"""
    results = (
        pd.DataFrame(rows)
        .sort_values("窗口序号")
        .reset_index(drop=True)
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return results


def validate_against_table3(
    results: pd.DataFrame,
) -> None:
    """用表3的第1和第10窗口校验动态回归口径。"""
    if not TABLE3_DATE_FE_RESULT_PATH.exists():
        print(
            "未找到日期固定效应版表3结果，"
            "跳过第1和第10窗口的交叉校验。"
        )
        return

    table3 = pd.read_csv(
        TABLE3_DATE_FE_RESULT_PATH
    )

    anchors = {
        "[t, t+1]": 1,
        "[t, t+10]": 10,
    }

    for window_label, horizon in anchors.items():
        expected = table3.loc[
            table3["订单失衡窗口"].eq(
                window_label
            )
        ]

        actual = results.loc[
            results["窗口序号"].eq(horizon)
        ]

        if len(expected) != 1 or len(actual) != 1:
            raise ValueError(
                f"{window_label} 无法完成表3交叉校验。"
            )

        for column in [
            "线索系数",
            "聚类标准误",
        ]:
            if not np.isclose(
                float(actual.iloc[0][column]),
                float(expected.iloc[0][column]),
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError(
                    f"{window_label} 的{column}与"
                    "日期固定效应版表3不一致。"
                )

    print(
        "第1和第10窗口已通过日期固定效应版表3交叉校验。"
    )


def draw_figure3_date_fe(
    results: pd.DataFrame,
) -> None:
    """绘制加入日期固定效应的散户订单不平衡动态。"""
    required_horizons = set(HORIZONS)
    actual_horizons = set(
        results["窗口序号"].astype(int)
    )

    if actual_horizons != required_horizons:
        missing = sorted(
            required_horizons - actual_horizons
        )
        raise ValueError(
            "图3结果不完整，缺少窗口："
            f"{missing}"
        )

    date_fe_values = set(
        results["日期固定效应"]
        .dropna()
        .astype(str)
    )

    if date_fe_values != {"是"}:
        raise ValueError(
            "日期固定效应版图3要求"
            "“日期固定效应”全部为“是”。"
        )

    plot_data = (
        results
        .sort_values("窗口序号")
        .reset_index(drop=True)
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
                fname=str(font_path)
            )
            .get_name()
        )
        plt.rcParams["font.sans-serif"] = [
            font_name
        ]

    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(
        figsize=(10, 5)
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

    for spec in ERROR_BAR_SPECS:
        interval_radius = (
            spec["critical_value"]
            * standard_errors
        )
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

    # 按10%显著性水平判断：90%置信区间不包含0
    significant = (
        np.abs(coefficients)
        > 1.6448536269514722 * standard_errors
    )
    point_colors = np.where(significant, "#d62728", "#1a1a1a")

    axis.scatter(
        x_values,
        coefficients,
        color=point_colors,
        s=16,
        zorder=4,
    )

    # 添加红点图例
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
        "散户订单不平衡动态（95分位提示事件）\n"
        "固定效应：被提示公司×年度季度；交易日期（事件起点）"
    )
    axis.set_xlabel("事件窗口")
    axis.set_ylabel("提示变量系数（百分点）")

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

    axis.set_xticks(ticks)
    axis.set_xticklabels([
        f"[t,t+{tick}]"
        for tick in ticks
    ])

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
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "估计并绘制加入日期固定效应的图3。"
        )
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "从已有的部分结果继续估计尚未完成的窗口。"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help=(
            "每次读入的订单不平衡窗口数量，默认5。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size 必须是正整数。"
        )

    existing_rows: list[dict] = []

    if args.resume and RESULT_PATH.exists():
        existing = pd.read_csv(RESULT_PATH)

        if not existing.empty:
            if not existing[
                "日期固定效应"
            ].astype(str).eq("是").all():
                raise ValueError(
                    "已有结果不是日期固定效应版本，"
                    "不能继续估计。"
                )

            existing_rows = (
                existing
                .drop_duplicates(
                    "窗口序号",
                    keep="last",
                )
                .to_dict("records")
            )

    completed = {
        int(row["窗口序号"])
        for row in existing_rows
    }

    pending = [
        horizon
        for horizon in HORIZONS
        if horizon not in completed
    ]

    if pending:
        panel = read_figure3_panel()
        rows = existing_rows.copy()
        completed_in_run = 0

        for batch_start in range(
            0,
            len(pending),
            args.batch_size,
        ):
            batch_horizons = pending[
                batch_start:
                batch_start + args.batch_size
            ]

            data = read_figure3_batch(
                panel,
                batch_horizons,
            )

            for horizon in batch_horizons:
                completed_in_run += 1

                print(
                    "正在运行加入日期固定效应的图3："
                    f"[t, t+{horizon}]"
                    "（待估计窗口 "
                    f"{completed_in_run}/{len(pending)}）"
                )

                rows.append(
                    run_date_fe_regression(
                        data,
                        horizon,
                    )
                )

                results = save_results(rows)
                latest = results.loc[
                    results["窗口序号"].eq(
                        horizon
                    )
                ].iloc[0]

                print(
                    f"窗口 {horizon} 完成："
                    f"系数={latest['线索系数']:.6f}，"
                    f"标准误={latest['聚类标准误']:.6f}"
                )

            del data
            gc.collect()

        del panel
        gc.collect()
    else:
        print(
            "已有结果包含全部50个窗口，"
            "无需重新估计。"
        )
        results = save_results(
            existing_rows
        )

    if len(results) != len(HORIZONS):
        raise ValueError(
            "日期固定效应版图3结果未包含全部50个窗口。"
        )

    validate_against_table3(results)
    draw_figure3_date_fe(results)

    print(
        "日期固定效应版图3结果已保存："
        f"{RESULT_PATH}"
    )
    print(
        "日期固定效应版图3已保存："
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
