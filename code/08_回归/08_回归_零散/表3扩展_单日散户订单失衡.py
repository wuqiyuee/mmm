"""
表3的单日散户订单失衡扩展检验。

本脚本不改变、也不覆盖现有表3的累计平均 OIB（订单不平衡）结果。
新的被解释变量是被提示公司在目标交易日 t+h 的单日 OIBvol：

    OIBvol_{i,t+h}

目标日通过统一市场交易日序号严格匹配：target_td = origin_td + h。
匹配不到目标股票日或 OIBvol 缺失时保持缺失，不填充为 0，也不要求
t 至 t+h 之间的其他交易日完整。因此，本扩展检验不会产生累计窗口的
缺失传播。

默认设置：
1. 估计 t、t+1 和 t+10，可通过 --horizons 修改；
2. 使用 Pattern样本_单日收益回归基础面板.csv；
3. available（逐时点可用样本）口径；
4. 在完整日度 OIB 源表上统一使用 1%/99% 阈值缩尾；
5. 吸收被提示公司×年度季度和事件起点交易日期固定效应；
6. 按被提示公司和事件起点交易日期双向聚类，并启用有限样本修正。

示例：
    python 表3扩展_单日散户订单失衡_加入日期固定效应.py
    python 表3扩展_单日散户订单失衡_加入日期固定效应.py \
        --horizons 0 1 2 5 10 --sample-mode common
    python 表3扩展_单日散户订单失衡_加入日期固定效应.py \
        --base-source legacy --horizons 0,1,10

不同基础面板、样本口径和相对时点使用不同文件名。结果只写入
“扩展分析/单日散户订单不平衡”，不会覆盖累计口径表3结果。
"""

from __future__ import annotations

import argparse
import gc
from collections.abc import Sequence
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[3]

NEUTRAL_BASE_PATH = (
    ROOT
    / "中间结果"
    / "04_合并数据"
    / "Pattern样本_单日收益回归基础面板.csv"
)

LEGACY_BASE_PATH = (
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

RESULT_DIR = (
    ROOT
    / "中间结果"
    / "08_回归结果"
    / "扩展分析"
    / "单日散户订单不平衡"
)
IMAGE_DIR = ROOT / "图表结果" / "论文复现" / "表格"

FONT_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
if FONT_PATH.exists():
    FONT = FontProperties(fname=str(FONT_PATH))
    FONT_BOLD = FontProperties(fname=str(FONT_PATH), weight="bold")
else:
    FONT = FontProperties(family="sans-serif")
    FONT_BOLD = FontProperties(family="sans-serif", weight="bold")

plt.rcParams["axes.unicode_minus"] = False


DEFAULT_HORIZONS = (0, 1, 10)
WINSOR_TAIL = 0.01

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

AEV_COLS = [
    "logIncreaseAEV",
    "logDecreaseAEV",
]

MEDIA_COL_CANDIDATES = {
    "neutral": [
        ["aesIncrease", "aesDecrease"],
        ["positiveNews", "negativeNews"],
    ],
    "legacy": [
        ["positiveNews", "negativeNews"],
        ["aesIncrease", "aesDecrease"],
    ],
}

BASE_KEY_COLS = [
    "StkcdRecall",
    "Trddt",
    "td",
    "yearQuarter",
]

SAMPLE_MODE_LABELS = {
    "available": "逐时点可用样本",
    "common": "指定时点完整共同样本",
}

BASE_SOURCE_LABELS = {
    "neutral": "累计因变量筛选前的中性基础面板",
    "legacy": "旧累计收益最终回归面板",
}

BASE_SOURCE_SHORT_LABELS = {
    "neutral": "中性基础面板",
    "legacy": "旧累计回归面板",
}

OUTCOME_VALUE_COL = "_single_day_OIBvol_w"
OUTCOME_TD_COL = "_outcome_td"
OUTCOME_DATE_COL = "_outcome_Trddt"
TARGET_TD_COL = "_target_td"

ANALYSIS_VERSION = "表3扩展_单日散户订单失衡_v1"
OUTCOME_DEFINITION = "目标交易日t+h的单日OIBvol"
TARGET_ALIGNMENT_DEFINITION = "目标交易日序号=事件起点交易日序号+h"
TARGET_MISSING_DEFINITION = "目标股票日或OIBvol缺失时保持缺失，不填零"
WINSOR_DEFINITION = "完整日度OIBvol源表统一1%/99%阈值，先缩尾后匹配"
DATE_FE_DEFINITION = "事件起点交易日期"
CLUSTER_DEFINITION = "被提示公司和事件起点交易日期双向聚类"
FINITE_SAMPLE_DEFINITION = "是（debiased=True）"


def significance_stars(p_value: float) -> str:
    """根据双侧检验的 p 值添加显著性星号。"""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def require_columns(
    columns: Sequence[str],
    available: Sequence[str],
    data_name: str,
) -> None:
    """检查数据是否包含全部必要字段。"""
    missing = sorted(set(columns) - set(available))
    if missing:
        raise ValueError(f"{data_name} 缺少必要变量：{missing}")


def normalize_stock_code_series(series: pd.Series) -> pd.Series:
    """将股票代码统一为保留前导零的六位字符串。"""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def coerce_required_integer(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """把交易日序号转换为非缺失整数并检查取值。"""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{column_name} 存在缺失或无法解析的取值。")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{column_name} 存在非整数取值。")
    return numeric.astype("int64")


def format_relative_day(horizon: int) -> str:
    """把交易日偏移量格式化为 t 或 t+h。"""
    return "t" if horizon == 0 else f"t+{horizon}"


def outcome_label(horizon: int) -> str:
    """返回与累计平均 OIB 变量明确区分的单日因变量名称。"""
    suffix = "t" if horizon == 0 else f"tp{horizon}"
    return f"OIBvol_{suffix}_w"


def parse_horizons(tokens: Sequence[str]) -> tuple[int, ...]:
    """解析空格分隔或逗号分隔的非负交易日偏移量。"""
    values: list[int] = []

    for token in tokens:
        for part in str(token).split(","):
            cleaned = part.strip()
            if not cleaned:
                continue
            try:
                value = int(cleaned)
            except ValueError as exc:
                raise ValueError(f"无法解析相对交易日：{cleaned}") from exc
            if value < 0:
                raise ValueError("相对交易日必须为非负整数。")
            values.append(value)

    if not values:
        raise ValueError("至少需要指定一个相对交易日。")
    if len(values) != len(set(values)):
        raise ValueError("--horizons 中存在重复的相对交易日。")

    return tuple(values)


def resolve_base_path(base_source: str) -> Path:
    """根据命令行基础样本口径返回输入文件路径。"""
    if base_source == "neutral":
        return NEUTRAL_BASE_PATH
    if base_source == "legacy":
        return LEGACY_BASE_PATH
    raise ValueError(f"未知基础面板口径：{base_source}")


def detect_media_columns(
    available_columns: Sequence[str],
    base_source: str,
) -> list[str]:
    """自动识别中性面板或旧面板使用的正负新闻变量。"""
    available = set(available_columns)

    for candidate in MEDIA_COL_CANDIDATES[base_source]:
        if set(candidate).issubset(available):
            return list(candidate)

    expected = MEDIA_COL_CANDIDATES[base_source]
    raise ValueError(
        "基础面板无法识别正负新闻变量；应至少包含以下一组："
        f"{expected}"
    )


def read_base_panel(
    path: Path,
    base_source: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """读取基础面板并自动确定完整右端变量列表。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到回归基础面板：{path}")

    available = pd.read_csv(
        path,
        nrows=0,
        encoding="utf-8-sig",
    ).columns.tolist()

    media_cols = detect_media_columns(available, base_source)
    rhs_cols = (
        [CUE_COL]
        + OWN_EA_COLS
        + DISCLOSURE_COLS
        + media_cols
        + AEV_COLS
    )
    read_cols = list(dict.fromkeys(BASE_KEY_COLS + rhs_cols))
    require_columns(read_cols, available, "回归基础面板")

    numeric_cols = [column for column in rhs_cols]
    data = pd.read_csv(
        path,
        usecols=read_cols,
        dtype={
            "StkcdRecall": "string",
            "yearQuarter": "string",
            **{column: "float32" for column in numeric_cols},
        },
        parse_dates=["Trddt"],
        encoding="utf-8-sig",
    )

    data["StkcdRecall"] = normalize_stock_code_series(
        data["StkcdRecall"]
    )
    data["td"] = coerce_required_integer(data["td"], "td")
    data["Trddt"] = pd.to_datetime(data["Trddt"], errors="coerce")

    if data[["StkcdRecall", "Trddt", "yearQuarter"]].isna().any().any():
        raise ValueError("回归基础面板的公司、日期或年度季度存在缺失。")

    duplicate_count = int(
        data.duplicated(["StkcdRecall", "td"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            "回归基础面板的被提示公司和交易日序号不唯一："
            f"{duplicate_count:,} 条重复观测。"
        )

    invariant_cols = list(dict.fromkeys(BASE_KEY_COLS + rhs_cols))
    before = len(data)
    data = data.dropna(subset=invariant_cols).reset_index(drop=True)
    print(
        "基础面板中因固定效应或右端变量缺失删除的观测数："
        f"{before - len(data):,}"
    )
    if data.empty:
        raise ValueError("删除基础面板固定字段缺失后没有剩余观测。")

    return data, rhs_cols, media_cols


def read_daily_oib_lookup(
    path: Path = OIB_PATH,
) -> tuple[pd.DataFrame, float, float]:
    """仅读取四个必要字段，并在完整源表上统一缩尾 OIBvol。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到日度 OIB 数据：{path}")

    oib = pd.read_csv(
        path,
        usecols=["Stkcd", "Trddt", "td", "OIBvol"],
        dtype={
            "Stkcd": "string",
            "OIBvol": "float32",
        },
        parse_dates=["Trddt"],
        encoding="utf-8-sig",
    )

    oib["Stkcd"] = normalize_stock_code_series(oib["Stkcd"])
    oib["td"] = coerce_required_integer(oib["td"], "td")
    oib["Trddt"] = pd.to_datetime(oib["Trddt"], errors="coerce")
    oib["OIBvol"] = pd.to_numeric(oib["OIBvol"], errors="coerce")
    oib["OIBvol"] = oib["OIBvol"].replace([np.inf, -np.inf], np.nan)

    if oib[["Stkcd", "Trddt"]].isna().any().any():
        raise ValueError("日度 OIB 表存在缺失的股票代码或交易日期。")

    duplicate_count = int(oib.duplicated(["Stkcd", "td"]).sum())
    if duplicate_count:
        raise ValueError(
            "日度 OIB 表的股票和交易日序号不唯一："
            f"{duplicate_count:,} 条重复观测。"
        )

    valid = oib["OIBvol"].dropna()
    if valid.empty:
        raise ValueError("日度 OIB 表没有可用的 OIBvol。")
    if valid.abs().gt(100.0001).any():
        raise ValueError("OIBvol 超出理论取值区间 [-100, 100]。")

    lower = float(valid.quantile(WINSOR_TAIL))
    upper = float(valid.quantile(1 - WINSOR_TAIL))
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("无法得到有效的 OIBvol 统一缩尾阈值。")

    oib[OUTCOME_VALUE_COL] = (
        oib["OIBvol"]
        .clip(lower=lower, upper=upper)
        .astype("float32")
    )

    lookup = oib.rename(
        columns={
            "Stkcd": "StkcdRecall",
            "td": OUTCOME_TD_COL,
            "Trddt": OUTCOME_DATE_COL,
        }
    )[
        [
            "StkcdRecall",
            OUTCOME_TD_COL,
            OUTCOME_DATE_COL,
            OUTCOME_VALUE_COL,
        ]
    ].copy()

    del oib, valid
    gc.collect()
    return lookup, lower, upper


def extract_td_date_map(
    data: pd.DataFrame,
    td_col: str,
    date_col: str,
    data_name: str,
) -> pd.DataFrame:
    """提取并校验交易日序号与自然日期的一一对应关系。"""
    mapping = data[[td_col, date_col]].drop_duplicates().copy()
    if mapping[[td_col, date_col]].isna().any().any():
        raise ValueError(f"{data_name} 的交易日序号或日期存在缺失。")
    if mapping.duplicated(td_col, keep=False).any():
        raise ValueError(f"{data_name} 中同一交易日序号对应多个日期。")
    if mapping.duplicated(date_col, keep=False).any():
        raise ValueError(f"{data_name} 中同一交易日期对应多个序号。")
    return mapping


def validate_shared_td_calendar(
    base: pd.DataFrame,
    oib_lookup: pd.DataFrame,
) -> None:
    """确认基础面板与 OIB 表使用相同的市场交易日序号。"""
    base_map = extract_td_date_map(
        base,
        "td",
        "Trddt",
        "回归基础面板",
    ).rename(columns={"Trddt": "_base_date"})
    oib_map = extract_td_date_map(
        oib_lookup,
        OUTCOME_TD_COL,
        OUTCOME_DATE_COL,
        "日度 OIB 表",
    ).rename(columns={OUTCOME_DATE_COL: "_oib_date"})

    overlap = base_map.merge(
        oib_map,
        left_on="td",
        right_on=OUTCOME_TD_COL,
        how="inner",
        validate="1:1",
    )
    if overlap.empty:
        raise ValueError("基础面板与日度 OIB 表没有重叠的交易日序号。")
    if not overlap["_base_date"].eq(overlap["_oib_date"]).all():
        raise ValueError("基础面板与日度 OIB 表的交易日序号口径不一致。")


def attach_single_day_oib(
    base: pd.DataFrame,
    oib_lookup: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """按 StkcdRecall 和 td+h 精确匹配一个时点的单日 OIBvol。"""
    aligned = base.copy()
    aligned[TARGET_TD_COL] = aligned["td"] + int(horizon)
    aligned = aligned.merge(
        oib_lookup,
        left_on=["StkcdRecall", TARGET_TD_COL],
        right_on=["StkcdRecall", OUTCOME_TD_COL],
        how="left",
        validate="1:1",
        sort=False,
    )

    if len(aligned) != len(base):
        raise ValueError("匹配单日 OIB 后基础面板行数发生变化。")

    matched = aligned[OUTCOME_TD_COL].notna()
    if not aligned.loc[matched, OUTCOME_TD_COL].eq(
        aligned.loc[matched, TARGET_TD_COL]
    ).all():
        raise ValueError("单日 OIB 的目标交易日序号匹配错误。")

    return aligned.drop(columns=[TARGET_TD_COL])


def build_common_sample_mask(
    base: pd.DataFrame,
    oib_lookup: pd.DataFrame,
    horizons: Sequence[int],
) -> pd.Series:
    """识别所有指定相对日均有单日 OIBvol 的共同样本。"""
    valid_lookup = oib_lookup.loc[
        oib_lookup[OUTCOME_VALUE_COL].notna(),
        ["StkcdRecall", OUTCOME_TD_COL],
    ]
    valid_index = pd.MultiIndex.from_frame(valid_lookup)

    firms = base["StkcdRecall"].astype("string").to_numpy()
    origin_td = base["td"].to_numpy(dtype=np.int64, copy=False)
    common = np.ones(len(base), dtype=bool)

    for horizon in horizons:
        target_index = pd.MultiIndex.from_arrays(
            [
                firms,
                origin_td + int(horizon),
            ],
            names=["StkcdRecall", OUTCOME_TD_COL],
        )
        common &= target_index.isin(valid_index)
        if not common.any():
            break

    return pd.Series(common, index=base.index, dtype=bool)


def drop_fe_singletons_iteratively(
    reg: pd.DataFrame,
    fe_code_cols: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """递归删除任一固定效应维度中的单例组。"""
    total_removed = 0

    while True:
        if reg.empty:
            raise ValueError("删除固定效应单例后没有剩余观测。")

        singleton = np.zeros(len(reg), dtype=bool)
        for column in fe_code_cols:
            codes = reg[column].to_numpy(dtype=np.int64, copy=False)
            counts = np.bincount(codes)
            singleton |= counts[codes] == 1

        removed_now = int(singleton.sum())
        if removed_now == 0:
            break

        total_removed += removed_now
        reg = reg.loc[~singleton].copy()

    return reg, total_removed


def run_single_day_regression(
    data: pd.DataFrame,
    horizon: int,
    rhs_cols: Sequence[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """估计一个单日 OIB 模型并返回模型汇总和全部系数。"""
    rhs_cols = list(rhs_cols)
    needed = list(
        dict.fromkeys(
            [
                OUTCOME_VALUE_COL,
                "StkcdRecall",
                "Trddt",
                "yearQuarter",
                *rhs_cols,
            ]
        )
    )
    reg = data[needed].dropna().copy()
    if reg.empty:
        raise ValueError(
            f"{format_relative_day(horizon)} 没有完整的回归观测。"
        )
    if reg.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("回归样本的被提示公司和事件起点日期不是唯一键。")

    reg["_fe_firm_yearquarter"] = (
        reg.groupby(
            ["StkcdRecall", "yearQuarter"],
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

    reg, singleton_removed = drop_fe_singletons_iteratively(
        reg,
        ["_fe_firm_yearquarter", "_fe_date"],
    )

    absorb = pd.DataFrame(
        {
            "firm_yearquarter": pd.Categorical(
                reg["_fe_firm_yearquarter"]
            ),
            "event_origin_date": pd.Categorical(reg["_fe_date"]),
        },
        index=reg.index,
    )
    clusters = pd.DataFrame(
        {
            "firm": pd.factorize(
                reg["StkcdRecall"],
                sort=False,
            )[0],
            "event_origin_date": pd.factorize(
                reg["Trddt"],
                sort=False,
            )[0],
        },
        index=reg.index,
    )

    model = AbsorbingLS(
        dependent=reg[OUTCOME_VALUE_COL].astype("float64"),
        exog=reg[rhs_cols].astype("float64"),
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
            f"{format_relative_day(horizon)} 的线索变量被固定效应完全吸收。"
        )

    beta = float(result.params[CUE_COL])
    standard_error = float(result.std_errors[CUE_COL])
    t_value = float(result.tstats[CUE_COL])
    p_value = float(result.pvalues[CUE_COL])

    common_metadata: dict[str, object] = {
        "分析版本": ANALYSIS_VERSION,
        "窗口序号": int(horizon),
        "相对交易日": format_relative_day(horizon),
        "因变量": outcome_label(horizon),
        "被解释变量口径": OUTCOME_DEFINITION,
        "目标日对齐口径": TARGET_ALIGNMENT_DEFINITION,
        "目标日缺失处理": TARGET_MISSING_DEFINITION,
        "缩尾口径": WINSOR_DEFINITION,
        "结果单位": "百分点",
        "控制变量": "是",
        "日期固定效应": "是",
        "日期固定效应基准": DATE_FE_DEFINITION,
        "被提示公司×年度季度固定效应": "是",
        "聚类方式": CLUSTER_DEFINITION,
        "有限样本修正": FINITE_SAMPLE_DEFINITION,
        "回归样本量": int(result.nobs),
        "被提示公司聚类数": int(reg["StkcdRecall"].nunique()),
        "日期聚类数": int(reg["Trddt"].nunique()),
        "删除单例组观测数": singleton_removed,
        "R方": float(result.rsquared),
    }

    model_info = {
        **common_metadata,
        "线索系数": beta,
        "聚类标准误": standard_error,
        "置信区间下界": beta - 1.96 * standard_error,
        "置信区间上界": beta + 1.96 * standard_error,
        "置信区间口径": "逐点95%置信区间",
        "t值": t_value,
        "p值": p_value,
        "显著性": significance_stars(p_value),
    }

    coefficient_rows: list[dict[str, object]] = []
    for variable_order, variable in enumerate(rhs_cols, start=1):
        estimated = variable in result.params.index
        if estimated:
            variable_beta = float(result.params[variable])
            variable_se = float(result.std_errors[variable])
            variable_t = float(result.tstats[variable])
            variable_p = float(result.pvalues[variable])
            stars = significance_stars(variable_p)
            status = "已估计"
        else:
            variable_beta = np.nan
            variable_se = np.nan
            variable_t = np.nan
            variable_p = np.nan
            stars = ""
            status = "被固定效应吸收"

        coefficient_rows.append(
            {
                **common_metadata,
                "变量": variable,
                "变量顺序": variable_order,
                "变量类型": (
                    "线索事件" if variable == CUE_COL else "控制变量"
                ),
                "估计状态": status,
                "系数": variable_beta,
                "聚类标准误": variable_se,
                "t值": variable_t,
                "p值": variable_p,
                "显著性": stars,
            }
        )

    del result, model, absorb, clusters, reg
    gc.collect()
    return model_info, coefficient_rows


def _format_single_day_estimate(row: pd.Series) -> tuple[str, str]:
    """将单日订单失衡回归结果格式化为图片表格中的系数和标准误。"""
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


def _single_day_variable_label(variable: str) -> str:
    """将单日订单失衡扩展中的变量名转换为中文标签。"""
    labels = {
        CUE_COL: "线索事件",
        "ownEA": "自身盈余公告 [t]",
        "ownMajorDisclosure": "其他专题事件 [t]",
        "ownPerformanceForecast": "业绩预告或快报 [t]",
        "positiveNews": "正面新闻 [t]",
        "negativeNews": "负面新闻 [t]",
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


def draw_single_day_oib_full(
    models: pd.DataFrame,
    coefficients: pd.DataFrame,
    output_path: Path,
) -> None:
    """在本脚本内直接绘制指定相对交易日的单日 OIB 完整系数表。"""
    model_rows = [
        row
        for _, row in models.sort_values("窗口序号").iterrows()
    ]
    if not model_rows:
        raise ValueError("单日订单失衡完整系数表没有可绘制的模型结果。")
    variables = (
        coefficients[["变量", "变量顺序"]]
        .drop_duplicates()
        .sort_values("变量顺序")["变量"]
        .tolist()
    )
    lookup = {
        (int(row["窗口序号"]), row["变量"]): row
        for _, row in coefficients.iterrows()
    }
    rows: list[list[str]] = []
    bold_rows: set[int] = set()
    for variable in variables:
        coefficient_values, standard_error_values = [], []
        for model in model_rows:
            detail = lookup.get((int(model["窗口序号"]), variable))
            coefficient, standard_error = (
                _format_single_day_estimate(detail)
                if detail is not None
                else ("", "")
            )
            coefficient_values.append(coefficient)
            standard_error_values.append(standard_error)
        bold_rows.add(len(rows))
        rows.append([
            _single_day_variable_label(variable),
            *coefficient_values,
        ])
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
    column_labels = ["变量"] + [
        str(row["相对交易日"])
        for row in model_rows
    ]
    figure_width = max(12.0, 3.0 + 1.45 * len(model_rows))
    figure_height = max(12.0, 5.4 + 0.23 * len(rows))
    figure, axes = plt.subplots(figsize=(figure_width, figure_height))
    figure.patch.set_facecolor("white")
    axes.axis("off")
    first = model_rows[0]
    axes.text(
        0.01,
        0.988,
        "表 3 扩展：单日散户订单失衡（加入交易日期固定效应）",
        ha="left",
        va="top",
        fontsize=16,
        fontproperties=FONT_BOLD,
    )
    axes.text(
        0.01,
        0.957,
        (
            f"{first['基础面板口径']}；{first['样本口径']}；"
            "因变量为目标交易日的小单订单失衡"
        ),
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
        fontproperties=FONT,
    )
    column_count = len(column_labels)
    table = axes.table(
        cellText=rows,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.50] + [0.50 / (column_count - 1)] * (column_count - 1),
        bbox=[0.01, 0.08, 0.98, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2 if column_count <= 5 else 6.4)
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
        "注：所有模型同时吸收被提示公司×年度季度和事件起点交易日期固定效应；"
        "括号内为双向聚类标准误；*、**、*** 分别表示在 10%、5% 和 1% 水平上显著。",
        ha="left",
        va="top",
        fontsize=7.1,
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


def resolve_image_path(
    base_source: str,
    sample_mode: str,
    horizons: Sequence[int],
) -> Path:
    """按同一参数生成单日订单失衡完整系数表的独立图片路径。"""
    horizon_tag = "_".join(f"h{int(horizon)}" for horizon in horizons)
    stem = (
        "表3扩展_单日散户订单失衡_加入日期固定效应_"
        f"{BASE_SOURCE_SHORT_LABELS[base_source]}_"
        f"{SAMPLE_MODE_LABELS[sample_mode]}_"
        f"{horizon_tag}"
    )
    return IMAGE_DIR / f"{stem}_完整系数版_独立脚本.png"


def resolve_output_paths(
    base_source: str,
    sample_mode: str,
    horizons: Sequence[int],
) -> tuple[Path, Path]:
    """生成包含基础样本、样本模式和相对时点的独立输出路径。"""
    horizon_tag = "_".join(f"h{int(horizon)}" for horizon in horizons)
    stem = (
        "表3扩展_单日散户订单失衡_加入日期固定效应_"
        f"{BASE_SOURCE_SHORT_LABELS[base_source]}_"
        f"{SAMPLE_MODE_LABELS[sample_mode]}_"
        f"{horizon_tag}"
    )
    return (
        RESULT_DIR / f"{stem}_模型汇总.csv",
        RESULT_DIR / f"{stem}_全部系数明细.csv",
    )


def save_csv_atomically(data: pd.DataFrame, path: Path) -> None:
    """以 UTF-8 BOM 编码原子保存 CSV 结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    """读取基础面板、样本模式和相对交易日参数。"""
    parser = argparse.ArgumentParser(
        description="估计指定相对交易日的单日散户订单失衡回归。"
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=[str(value) for value in DEFAULT_HORIZONS],
        help=(
            "相对交易日，可使用空格或逗号分隔；"
            "默认：0 1 10。"
        ),
    )
    parser.add_argument(
        "--base-source",
        choices=sorted(BASE_SOURCE_LABELS),
        default="neutral",
        help=(
            "neutral=单日收益中性基础面板；"
            "legacy=旧累计收益最终回归面板。"
        ),
    )
    parser.add_argument(
        "--sample-mode",
        choices=sorted(SAMPLE_MODE_LABELS),
        default="available",
        help=(
            "available=每个时点使用该时点可用样本；"
            "common=所有指定时点均有 OIBvol 的共同样本。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """串联基础样本、单日 OIB 匹配、回归和独立输出。"""
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    base_path = resolve_base_path(args.base_source)
    model_path, coefficient_path = resolve_output_paths(
        args.base_source,
        args.sample_mode,
        horizons,
    )
    image_path = resolve_image_path(
        args.base_source,
        args.sample_mode,
        horizons,
    )

    print(f"读取回归基础面板：{base_path}")
    base, rhs_cols, media_cols = read_base_panel(
        base_path,
        args.base_source,
    )
    print(f"自动识别的正负新闻变量：{media_cols}")

    print(f"读取单日 OIB 表：{OIB_PATH}")
    oib_lookup, winsor_lower, winsor_upper = read_daily_oib_lookup()
    validate_shared_td_calendar(base, oib_lookup)
    print(
        "完整日度 OIBvol 统一缩尾阈值："
        f"下界={winsor_lower:.10f}，上界={winsor_upper:.10f}"
    )

    base_before_common = len(base)
    if args.sample_mode == "common":
        print(
            "正在识别全部指定相对交易日均有 OIBvol 的共同样本："
            f"{list(horizons)}"
        )
        common_mask = build_common_sample_mask(
            base,
            oib_lookup,
            horizons,
        )
        base = base.loc[common_mask].reset_index(drop=True)
        print(
            "共同样本筛选删除观测数："
            f"{base_before_common - len(base):,}"
        )
        if base.empty:
            raise ValueError("指定时点的完整共同样本为空。")

    model_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    for position, horizon in enumerate(horizons, start=1):
        relative_day = format_relative_day(horizon)
        print(
            "正在运行单日散户订单失衡回归："
            f"{relative_day}（{position}/{len(horizons)}）"
        )
        aligned = attach_single_day_oib(base, oib_lookup, horizon)
        valid_outcome = aligned[OUTCOME_VALUE_COL].notna()
        valid_count = int(valid_outcome.sum())
        if valid_count == 0:
            raise ValueError(f"{relative_day} 没有可用的单日 OIBvol。")

        valid_dates = aligned.loc[valid_outcome, OUTCOME_DATE_COL]
        model_info, current_coefficients = run_single_day_regression(
            aligned,
            horizon,
            rhs_cols,
        )

        run_metadata: dict[str, object] = {
            "结果表": "表3扩展（单日散户订单失衡）",
            "衡量口径": "成交量订单失衡OIBvol",
            "散户识别方法": "A股小单口径",
            "基础面板口径": BASE_SOURCE_LABELS[args.base_source],
            "样本口径": SAMPLE_MODE_LABELS[args.sample_mode],
            "指定时点": ",".join(
                format_relative_day(value) for value in horizons
            ),
            "基础面板共同样本筛选前观测数": base_before_common,
            "基础面板共同样本筛选后观测数": len(base),
            "目标日有效OIBvol观测数": valid_count,
            "目标日日期起点": valid_dates.min().strftime("%Y-%m-%d"),
            "目标日日期终点": valid_dates.max().strftime("%Y-%m-%d"),
            "缩尾下界": winsor_lower,
            "缩尾上界": winsor_upper,
            "正负新闻变量": ",".join(media_cols),
        }

        model_rows.append({**run_metadata, **model_info})
        coefficient_rows.extend(
            {**run_metadata, **row}
            for row in current_coefficients
        )

        print(
            f"{relative_day} 完成："
            f"系数={model_info['线索系数']:.6f}，"
            f"标准误={model_info['聚类标准误']:.6f}，"
            f"样本量={model_info['回归样本量']:,}"
        )

        del aligned, valid_outcome, valid_dates
        gc.collect()

    models = pd.DataFrame(model_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    save_csv_atomically(models, model_path)
    save_csv_atomically(coefficients, coefficient_path)
    draw_single_day_oib_full(models, coefficients, image_path)

    print("\n表3扩展：单日散户订单失衡模型汇总")
    print(models.to_string(index=False))
    print(f"\n模型汇总已保存：{model_path}")
    print(f"全部系数明细已保存：{coefficient_path}")
    print(f"完整系数图片已保存：{image_path}")

    del base, oib_lookup, models, coefficients
    gc.collect()


if __name__ == "__main__":
    main()
