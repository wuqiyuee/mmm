"""
估计并绘制加入交易日期固定效应的图2“收益动态和长期反转”。

固定效应：
1. 被提示公司 × 年度季度；
2. 交易日期。

标准误按被提示公司和交易日期双向聚类，并采用有限样本修正。
当前合并绘图脚本也已开启有限样本修正。

因变量使用最终回归面板中已经缩尾的累计特征调整收益变量，
本脚本不再次缩尾。控制变量与当前合并绘图脚本保持一致。

共估计50个累计窗口：[t,t+1]至[t,t+50]。
系数、标准误和置信区间统一转换为百分点。
图中展示99%、95%、90%正态近似置信区间；
红点表示对应窗口在10%水平上显著。

每完成一个窗口就保存一次结果。
仅在输入数据和回归设置未变化时，使用 --resume（续算参数）。
输入或回归设置变化后，应从头重新估计。
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


# 公共回归模块位于当前“图2”目录的上一级。
MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
from 图2图3扩展_日期固定效应回归公共模块 import (
    residualize_regression_data as residualize_regression_data_shared,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[3]

PATTERN_PATH = (
    ROOT
    / "中间结果"
    / "04_合并数据"
    / "Pattern样本_最终回归面板.csv"
)

RESULT_PATH = (
    ROOT
    / "中间结果"
    / "08_回归结果"
    / "图2_收益动态和长期反转_加入日期固定效应结果.csv"
)

TABLE2_RESULT_PATH = (
    ROOT
    / "中间结果"
    / "08_回归结果"
    / "表2_模型汇总.csv"
)

OUTPUT_PATH = (
    ROOT
    / "图表结果"
    / "论文复现"
    / "图形"
    / "figure2_收益动态和长期反转_加入日期固定效应.png"
)


CUE_COL = "quarterLag1size90plus"

OWN_EA_COLS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)

DISCLOSURE_COLS = [
    "ownOtherEventER",              # 被提示公司自身专题事件
    "ownPerformanceForecastBrief",  # 被提示公司自身业绩预告或快报
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

OUTCOME_COLS = {
    horizon: (
        f"chAdjreturn_t_tp{horizon}_Recall_w"
    )
    for horizon in HORIZONS
}

RESIDUALIZATION_BATCH_SIZE = 4

RESULT_COMPLETION_FIELDS = [
    "窗口序号",
    "线索系数",
    "聚类标准误",
    "置信区间下界",
    "置信区间上界",
    "t值",
    "p值",
    "回归样本量",
    "解释变量矩阵条件数",
    "系数求解方法",
    "R方",
]


def significance_stars(
    p_value: float,
) -> str:
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
        singleton = np.zeros(
            len(reg),
            dtype=bool,
        )

        for column in fe_code_cols:
            codes = reg[column].to_numpy(
                dtype=np.int64,
                copy=False,
            )

            if len(codes) == 0:
                raise ValueError(
                    "删除固定效应单例后没有剩余观测。"
                )

            if codes.min() < 0:
                raise ValueError(
                    f"{column} 中存在无效的负数分组编号。"
                )

            counts = np.bincount(codes)
            singleton |= counts[codes] == 1

        removed_now = int(
            singleton.sum()
        )

        if removed_now == 0:
            break

        total_removed += removed_now
        reg = reg.loc[
            ~singleton
        ].copy()

    return reg, total_removed


def cluster_score_meat(
    x: np.ndarray,
    residuals: np.ndarray,
    cluster_codes: np.ndarray,
    cluster_count: int,
) -> np.ndarray:
    """计算一个聚类维度的未缩放得分外积和。"""
    group_scores = np.empty(
        (
            cluster_count,
            x.shape[1],
        ),
        dtype=np.float64,
    )

    for column in range(
        x.shape[1]
    ):
        group_scores[
            :,
            column,
        ] = np.bincount(
            cluster_codes,
            weights=(
                x[:, column]
                * residuals
            ),
            minlength=cluster_count,
        )

    return (
        group_scores.T
        @ group_scores
    )


def observation_score_meat(
    x: np.ndarray,
    residuals: np.ndarray,
    chunk_size: int = 250_000,
) -> np.ndarray:
    """
    计算公司×日期交集聚类项。

    主样本以被提示公司和交易日期为唯一键，所以交集聚类中的每组
    只有一条观测。分块计算可以避免构造整张得分矩阵。
    """
    meat = np.zeros(
        (
            x.shape[1],
            x.shape[1],
        ),
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
            * residuals[
                start:stop,
                None,
            ]
        )

        meat += (
            scores.T
            @ scores
        )

    return meat


def two_way_cluster_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    firm_codes: np.ndarray,
    date_codes: np.ndarray,
    xpx_inverse: np.ndarray,
) -> np.ndarray:
    """
    计算被提示公司和交易日期双向聚类协方差。

    直接使用残差化后的普通最小二乘得分，并利用被提示公司×日期
    唯一键直接计算交集聚类项。
    """
    nobs, nvars = x.shape

    firm_count = (
        int(firm_codes.max())
        + 1
    )
    date_count = (
        int(date_codes.max())
        + 1
    )

    if firm_count < 2:
        raise ValueError(
            "被提示公司聚类数少于2，无法计算聚类标准误。"
        )

    if date_count < 2:
        raise ValueError(
            "日期聚类数少于2，无法计算聚类标准误。"
        )

    if nobs <= nvars:
        raise ValueError(
            "回归观测数不大于解释变量数。"
        )

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

    intersection_meat = (
        observation_score_meat(
            x,
            residuals,
        )
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

    # 被提示公司×日期是一行一个交集组，因此组数等于样本量。
    intersection_scale = (
        nobs
        / (nobs - 1)
        * common_scale
    )

    meat = (
        firm_scale
        * firm_meat
        + date_scale
        * date_meat
        - intersection_scale
        * intersection_meat
    )

    covariance = (
        xpx_inverse
        @ meat
        @ xpx_inverse
    )

    return (
        covariance
        + covariance.T
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


def read_figure2_panel(
    horizons: list[int],
) -> pd.DataFrame:
    """
    一次读取所有待估计窗口和共同回归变量。

    输入文件较大；一次扫描只读取待估计窗口，可以避免为每个窗口
    重复扫描同一个文件。恢复运行时，已完成窗口不会再次读入内存。
    """
    outcome_columns = [
        OUTCOME_COLS[horizon]
        for horizon in horizons
    ]

    panel_columns = list(
        dict.fromkeys(
            [
                "StkcdRecall",
                "Trddt",
                "yearQuarter",
            ]
            + RHS_COLS
            + outcome_columns
        )
    )

    available_columns = (
        pd.read_csv(
            PATTERN_PATH,
            nrows=0,
        )
        .columns
        .tolist()
    )

    missing_columns = sorted(
        set(panel_columns)
        - set(available_columns)
    )

    if missing_columns:
        raise ValueError(
            "图2输入文件缺少变量："
            f"{missing_columns}"
        )

    panel_dtype = {
        "StkcdRecall": "string",
        "yearQuarter": "string",
        **{
            column: "float32"
            for column in (
                RHS_COLS
                + outcome_columns
            )
        },
    }

    print(
        "读取加入日期固定效应的图2主样本……"
    )

    data = pd.read_csv(
        PATTERN_PATH,
        usecols=panel_columns,
        dtype=panel_dtype,
        parse_dates=["Trddt"],
    )
    data["StkcdRecall"] = (
        data["StkcdRecall"]
        .str.strip()
        .str.zfill(6)
    )

    valid_key = (
        data[
            [
                "StkcdRecall",
                "Trddt",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    if data.loc[
        valid_key
    ].duplicated(
        [
            "StkcdRecall",
            "Trddt",
        ]
    ).any():
        raise ValueError(
            "图2主样本的被提示公司和交易日期不是唯一键。"
        )

    data = data.loc[
        data["Trddt"].dt.year.between(START_YEAR, END_YEAR)
    ].copy()

    # 两个分组在所有窗口中定义相同。预先编号可以避免50次重复
    # 分组；窗口筛选只会删除行，不会改变组的归属。
    valid_firm_yearquarter = (
        data[
            [
                "StkcdRecall",
                "yearQuarter",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    data[
        "_fe_firm_yearquarter_all"
    ] = np.int32(-1)

    firm_yearquarter_codes = (
        data.loc[
            valid_firm_yearquarter
        ]
        .groupby(
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

    data.loc[
        valid_firm_yearquarter,
        "_fe_firm_yearquarter_all",
    ] = firm_yearquarter_codes.to_numpy(
        dtype=np.int32,
        copy=False,
    )

    data[
        "_fe_firm_yearquarter_all"
    ] = data[
        "_fe_firm_yearquarter_all"
    ].astype("int32")

    data["_fe_date_all"] = (
        pd.factorize(
            data["Trddt"],
            sort=False,
        )[0]
        .astype("int32")
    )

    data["StkcdRecall"] = (
        data["StkcdRecall"]
        .astype("category")
    )

    data["yearQuarter"] = (
        data["yearQuarter"]
        .astype("category")
    )

    return data


def run_date_fe_regression(
    data: pd.DataFrame,
    horizon: int,
) -> dict:
    """估计一个事件窗口的日期固定效应版图2回归。"""
    outcome = OUTCOME_COLS[
        horizon
    ]

    needed = list(
        dict.fromkeys(
            [
                outcome,
                "StkcdRecall",
                "Trddt",
                "yearQuarter",
                "_fe_firm_yearquarter_all",
                "_fe_date_all",
            ]
            + RHS_COLS
        )
    )

    reg = (
        data[
            needed
        ]
        .dropna()
        .copy()
    )

    reg[
        "_fe_firm_yearquarter"
    ] = reg[
        "_fe_firm_yearquarter_all"
    ].astype("int64")

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

    fe_ids = np.column_stack(
        [
            reg[
                "_fe_firm_yearquarter"
            ].to_numpy(
                dtype=np.int64,
                copy=False,
            ),
            reg["_fe_date"].to_numpy(
                dtype=np.int64,
                copy=False,
            ),
        ]
    )

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
        np.isfinite(
            xpx_condition
        )
        and xpx_condition < 1e12
    ):
        parameters = np.linalg.solve(
            xpx,
            xpy,
        ).reshape(-1)

        xpx_inverse = (
            np.linalg.inv(xpx)
        )

        coefficient_solver = (
            "正规方程（条件数已检查）"
        )
    else:
        parameters = np.linalg.lstsq(
            x_residualized,
            y_residualized,
            rcond=None,
        )[0].reshape(-1)

        xpx_inverse = (
            np.linalg.pinv(xpx)
        )

        coefficient_solver = (
            "最小二乘分解"
        )

    residuals = (
        y_residualized.reshape(-1)
        - x_residualized
        @ parameters
    )

    covariance = (
        two_way_cluster_covariance(
            x_residualized,
            residuals,
            firm_codes,
            date_codes,
            xpx_inverse,
        )
    )

    variances = np.diag(
        covariance
    )

    if (
        variances
        < -1e-12
    ).any():
        raise ValueError(
            f"窗口 {horizon} 出现明显为负的协方差对角元素。"
        )

    standard_errors = np.sqrt(
        np.maximum(
            variances,
            0.0,
        )
    )

    cue_position = (
        RHS_COLS.index(
            CUE_COL
        )
    )

    coefficient_raw = float(
        parameters[
            cue_position
        ]
    )

    standard_error_raw = float(
        standard_errors[
            cue_position
        ]
    )

    if (
        not np.isfinite(
            standard_error_raw
        )
        or standard_error_raw <= 0
    ):
        raise ValueError(
            f"窗口 {horizon} 的核心变量标准误无效。"
        )

    t_value = (
        coefficient_raw
        / standard_error_raw
    )

    nobs = len(reg)

    absorbed_parameter_count = (
        reg[
            "_fe_firm_yearquarter"
        ].nunique()
        + reg["_fe_date"].nunique()
        - 1
    )

    model_parameter_count = int(
        absorbed_parameter_count
        + x_residualized.shape[1]
    )

    residual_degrees = (
        nobs
        - model_parameter_count
    )

    if residual_degrees <= 0:
        raise ValueError(
            f"窗口 {horizon} 的残差自由度不为正。"
        )

    p_value = float(
        2
        * stats.t.sf(
            abs(t_value),
            residual_degrees,
        )
    )

    original_y = reg[
        outcome
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    centered_y = (
        original_y
        - original_y.mean()
    )

    residual_sum_squares = float(
        residuals
        @ residuals
    )

    total_sum_squares = float(
        centered_y
        @ centered_y
    )

    if total_sum_squares <= 0:
        raise ValueError(
            f"窗口 {horizon} 的因变量没有有效变异。"
        )

    r_squared = max(
        (
            1
            - residual_sum_squares
            / total_sum_squares
        ),
        0.0,
    )

    # 原始收益以小数存储；图2和结果表统一转换为百分点。
    coefficient = (
        coefficient_raw
        * 100
    )

    standard_error = (
        standard_error_raw
        * 100
    )

    row = {
        "窗口序号": horizon,
        "事件窗口": (
            f"[t, t+{horizon}]"
        ),
        "因变量": outcome,
        "结果单位": "百分点",
        "线索系数": coefficient,
        "聚类标准误": standard_error,
        "置信区间下界": (
            coefficient
            - 1.96
            * standard_error
        ),
        "置信区间上界": (
            coefficient
            + 1.96
            * standard_error
        ),
        "t值": t_value,
        "p值": p_value,
        "显著性": (
            significance_stars(
                p_value
            )
        ),
        "控制变量": "是",
        "日期固定效应": "是",
        "被提示公司×年度季度固定效应": "是",
        "有限样本修正": "是",
        "回归样本量": nobs,
        "被提示公司聚类数": int(
            reg[
                "StkcdRecall"
            ].nunique()
        ),
        "日期聚类数": int(
            reg["Trddt"].nunique()
        ),
        "删除单例组观测数": (
            singleton_removed
        ),
        "解释变量矩阵条件数": (
            xpx_condition
        ),
        "系数求解方法": (
            coefficient_solver
        ),
        "R方": r_squared,
    }

    del (
        fe_ids,
        firm_codes,
        date_codes,
        covariance,
        variances,
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


def save_results(
    rows: list[dict],
) -> pd.DataFrame:
    """按窗口排序并保存结构化结果。"""
    results = (
        pd.DataFrame(rows)
        .drop_duplicates(
            "窗口序号",
            keep="last",
        )
        .sort_values(
            "窗口序号"
        )
        .reset_index(
            drop=True
        )
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        RESULT_PATH.with_suffix(
            RESULT_PATH.suffix
            + ".tmp"
        )
    )

    results.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
    )

    temporary_path.replace(
        RESULT_PATH
    )

    return results


def load_completed_rows(
    resume: bool,
) -> list[dict]:
    """
    读取可安全续算的已有结果。

    只有核心结果和诊断字段都完整的窗口才视为已完成；字段不完整的
    旧窗口会从续算结果中移除，并进入待估计列表。
    """
    if (
        not resume
        or not RESULT_PATH.exists()
    ):
        return []

    existing = pd.read_csv(
        RESULT_PATH
    )

    if existing.empty:
        return []

    if "窗口序号" not in existing:
        raise ValueError(
            "已有图2结果缺少“窗口序号”，无法继续估计。"
        )

    if (
        "日期固定效应"
        in existing
        and not existing[
            "日期固定效应"
        ]
        .dropna()
        .astype(str)
        .eq("是")
        .all()
    ):
        raise ValueError(
            "已有结果包含非日期固定效应版本，不能继续估计。"
        )

    invalid_horizons = sorted(
        set(
            existing[
                "窗口序号"
            ]
            .dropna()
            .astype(int)
        )
        - set(HORIZONS)
    )

    if invalid_horizons:
        raise ValueError(
            "已有结果包含无效窗口："
            f"{invalid_horizons}"
        )

    for column in (
        RESULT_COMPLETION_FIELDS
        + [
            "日期固定效应",
            "被提示公司×年度季度固定效应",
            "控制变量",
            "结果单位",
            "有限样本修正",
        ]
    ):
        if column not in existing:
            existing[column] = np.nan

    complete_mask = (
        existing[
            RESULT_COMPLETION_FIELDS
        ]
        .notna()
        .all(axis=1)
        & existing[
            "日期固定效应"
        ]
        .astype(str)
        .eq("是")
        & existing[
            "被提示公司×年度季度固定效应"
        ]
        .astype(str)
        .eq("是")
        & existing[
            "控制变量"
        ]
        .astype(str)
        .eq("是")
        & existing[
            "结果单位"
        ]
        .astype(str)
        .eq("百分点")
        & existing[
            "有限样本修正"
        ]
        .astype(str)
        .eq("是")
    )

    incomplete_horizons = sorted(
        existing.loc[
            ~complete_mask,
            "窗口序号",
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    if incomplete_horizons:
        print(
            "以下已有窗口的字段不完整，"
            "将自动重新估计："
            f"{incomplete_horizons}"
        )

    completed = (
        existing.loc[
            complete_mask
        ]
        .drop_duplicates(
            "窗口序号",
            keep="last",
        )
        .sort_values(
            "窗口序号"
        )
    )

    return completed.to_dict(
        "records"
    )


def validate_results(
    results: pd.DataFrame,
) -> None:
    """在绘图前检查50个窗口及统计字段是否完整。"""
    if len(results) != len(
        HORIZONS
    ):
        raise ValueError(
            "日期固定效应版图2结果未包含全部50个窗口。"
        )

    actual_horizons = (
        results[
            "窗口序号"
        ]
        .astype(int)
        .tolist()
    )

    if actual_horizons != HORIZONS:
        raise ValueError(
            "日期固定效应版图2的窗口不完整或顺序错误。"
        )

    required_flags = {
        "日期固定效应": "是",
        "被提示公司×年度季度固定效应": "是",
        "控制变量": "是",
        "结果单位": "百分点",
        "有限样本修正": "是",
    }

    for column, expected in (
        required_flags.items()
    ):
        values = set(
            results[column]
            .dropna()
            .astype(str)
        )

        if values != {
            expected
        }:
            raise ValueError(
                f"{column} 的结果标记不正确。"
            )

    numeric_columns = [
        "线索系数",
        "聚类标准误",
        "置信区间下界",
        "置信区间上界",
        "t值",
        "p值",
        "回归样本量",
        "解释变量矩阵条件数",
        "R方",
    ]

    numeric_values = results[
        numeric_columns
    ].to_numpy(
        dtype=np.float64,
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            "日期固定效应版图2包含缺失或非有限数值。"
        )

    if not (
        results[
            "聚类标准误"
        ]
        > 0
    ).all():
        raise ValueError(
            "日期固定效应版图2包含非正标准误。"
        )

    if not results[
        "p值"
    ].between(
        0,
        1,
    ).all():
        raise ValueError(
            "日期固定效应版图2包含无效p值。"
        )

    interval_valid = (
        (
            results[
                "置信区间下界"
            ]
            <= results[
                "线索系数"
            ]
        )
        & (
            results[
                "线索系数"
            ]
            <= results[
                "置信区间上界"
            ]
        )
    )

    if not interval_valid.all():
        raise ValueError(
            "日期固定效应版图2包含无效置信区间。"
        )

    if results[
        "系数求解方法"
    ].isna().any():
        raise ValueError(
            "日期固定效应版图2缺少系数求解方法。"
        )


def validate_against_table2(
    results: pd.DataFrame,
    horizons_to_check: set[int]
    | None = None,
) -> None:
    """
    用表2模式公司主样本规格（6）的两个收益窗口交叉校验。

    表2中的系数和标准误同时控制交易日期固定效应与
    被提示公司×年度季度固定效应，并采用相同的双向聚类和
    有限样本修正，因此可作为图2第1和第10窗口的直接锚点。
    """
    if not TABLE2_RESULT_PATH.exists():
        print(
            "未找到表2模型汇总，"
            "跳过第1和第10窗口的交叉校验。"
        )
        return

    table2 = pd.read_csv(
        TABLE2_RESULT_PATH
    )

    required_columns = {
        "样本",
        "收益窗口",
        "规格",
        "线索系数（百分点）",
        "聚类标准误（百分点）",
        "回归样本量",
    }

    missing_columns = sorted(
        required_columns
        - set(table2.columns)
    )

    if missing_columns:
        raise ValueError(
            "表2模型汇总缺少交叉校验字段："
            f"{missing_columns}"
        )

    anchors = {
        "[t, t+1]": 1,
        "[t, t+10]": 10,
    }

    if horizons_to_check is None:
        horizons_to_check = set(
            anchors.values()
        )

    invalid_horizons = (
        set(
            horizons_to_check
        )
        - set(
            anchors.values()
        )
    )

    if invalid_horizons:
        raise ValueError(
            "表2锚点校验收到无效窗口："
            f"{sorted(invalid_horizons)}"
        )

    checked_labels = []

    for window_label, horizon in (
        anchors.items()
    ):
        if horizon not in (
            horizons_to_check
        ):
            continue

        expected = table2.loc[
            table2[
                "样本"
            ]
            .astype(str)
            .eq("pattern")
            & table2[
                "收益窗口"
            ]
            .astype(str)
            .eq(window_label)
            & table2[
                "规格"
            ]
            .astype(str)
            .eq("(6)")
        ]

        actual = results.loc[
            results[
                "窗口序号"
            ].eq(horizon)
        ]

        if (
            len(expected) != 1
            or len(actual) != 1
        ):
            raise ValueError(
                f"{window_label} 无法完成表2规格（6）交叉校验。"
            )

        comparisons = {
            "线索系数": (
                "线索系数（百分点）"
            ),
            "聚类标准误": (
                "聚类标准误（百分点）"
            ),
        }

        for actual_column, expected_column in (
            comparisons.items()
        ):
            if not np.isclose(
                float(
                    actual.iloc[
                        0
                    ][
                        actual_column
                    ]
                ),
                float(
                    expected.iloc[
                        0
                    ][
                        expected_column
                    ]
                ),
                rtol=1e-6,
                atol=1e-8,
            ):
                raise ValueError(
                    f"{window_label} 的{actual_column}"
                    "与表2模式公司主样本规格（6）不一致。"
                )

        if int(
            actual.iloc[
                0
            ][
                "回归样本量"
            ]
        ) != int(
            expected.iloc[
                0
            ][
                "回归样本量"
            ]
        ):
            raise ValueError(
                f"{window_label} 的回归样本量"
                "与表2模式公司主样本规格（6）不一致。"
            )

        checked_labels.append(
            window_label
        )

    print(
        f"{'、'.join(checked_labels)}"
        "已通过表2"
        "模式公司主样本规格（6）交叉校验。"
    )


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

    # 与图中90%置信区间一致，判断10%水平显著性
    significant = (
        np.abs(coefficients)
        > 1.6448536269514722 * standard_errors
    )

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
        "收益动态和长期反转"
        "（控制交易日期固定效应）"
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
    parser = argparse.ArgumentParser(
        description=(
            "估计并绘制加入交易日期固定效应的图2。"
        )
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "从已有的完整窗口结果继续估计。"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """运行50个收益窗口并生成独立的日期固定效应版图2。"""
    args = parse_args()

    existing_rows = (
        load_completed_rows(
            resume=args.resume,
        )
    )

    completed = {
        int(
            row[
                "窗口序号"
            ]
        )
        for row in existing_rows
    }

    completed_anchor_horizons = (
        completed
        & {
            1,
            10,
        }
    )

    if completed_anchor_horizons:
        validate_against_table2(
            pd.DataFrame(
                existing_rows
            ),
            horizons_to_check=(
                completed_anchor_horizons
            ),
        )

    pending = [
        horizon
        for horizon in HORIZONS
        if horizon
        not in completed
    ]

    if pending:
        data = read_figure2_panel(
            pending
        )

        rows = (
            existing_rows.copy()
        )

        for position, horizon in enumerate(
            pending,
            start=1,
        ):
            print(
                "正在运行加入日期固定效应的图2："
                f"[t, t+{horizon}]"
                "（待估计窗口 "
                f"{position}/{len(pending)}）"
            )

            row = (
                run_date_fe_regression(
                    data,
                    horizon,
                )
            )

            rows = [
                existing_row
                for existing_row in rows
                if int(
                    existing_row[
                        "窗口序号"
                    ]
                )
                != horizon
            ]

            rows.append(
                row
            )

            results = (
                save_results(
                    rows
                )
            )

            if horizon in {
                1,
                10,
            }:
                validate_against_table2(
                    results,
                    horizons_to_check={
                        horizon
                    },
                )

            print(
                f"窗口 {horizon} 完成："
                f"系数={row['线索系数']:.6f}，"
                f"标准误={row['聚类标准误']:.6f}"
            )

            # 当前因变量已完成后立即释放，降低后续回归的常驻内存。
            data.drop(
                columns=[
                    OUTCOME_COLS[
                        horizon
                    ]
                ],
                inplace=True,
            )

            gc.collect()

        del data
        gc.collect()
    else:
        print(
            "已有结果包含全部50个完整窗口，"
            "无需重新估计。"
        )

        results = save_results(
            existing_rows
        )

    validate_results(
        results
    )

    validate_against_table2(
        results
    )

    draw_figure2_date_fe(
        results
    )

    print(
        "日期固定效应版图2结果已保存："
        f"{RESULT_PATH}"
    )

    print(
        "日期固定效应版图2已保存："
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
