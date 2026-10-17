"""
估计并绘制事件日 t 至 t+50 的单日散户订单不平衡动态。

新的被解释变量不是 [t,t+h] 内的平均订单不平衡，而是被提示公司在
目标市场交易日 t+h 的当日小单成交量订单不平衡 OIBvol：

    target_td = origin_td + h,  h = 0, 1, ..., 50

目标日严格按照统一市场交易日序号匹配。若目标股票日不存在，或者
目标日 OIBvol 缺失，则该相对日因变量保持缺失，不填充为 0。

主规格默认使用“旧累计因变量筛选前的中性基础面板”，每个相对日
使用该日因变量可用的观测。命令行可切换为 t 至 t+50 均有效的共同
样本，也可以使用旧最终回归面板进行可比性检验。不同样本口径使用
不同结果路径，不覆盖现有累计窗口图3、表3及其结果。

固定效应和标准误：
1. 被提示公司 × 年度季度固定效应；
2. 事件起点交易日期固定效应；
3. 被提示公司和事件起点交易日期双向聚类；
4. 使用收益动态回归公共核心中的有限样本修正。

OIBvol 原始单位为百分点。脚本先在 OIB 股票日度源表上统一使用
1%/99% 阈值缩尾，再除以 100 送入公共回归核心；公共核心最终将
系数和标准误乘以 100，因此输出结果仍以百分点计量。

每完成一个相对日便原子保存一次结果，--resume 只复用口径元数据
完全一致的已完成结果。
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# 将公共模块所在目录加入模块搜索路径
MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from 图2图3扩展_日期固定效应回归公共模块 import (
    require_columns,
    run_date_fe_regression,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


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

DAILY_OIB_PATH = (
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

FIGURE_DIR = (
    ROOT
    / "图表结果"
    / "扩展分析"
    / "单日散户订单不平衡"
)

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

NEUTRAL_MEDIA_COLS = [
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]

LEGACY_MEDIA_COLS = [
    "positiveNews",
    "negativeNews",
    "logIncreaseAEV",
    "logDecreaseAEV",
]

BASE_KEY_COLS = [
    "StkcdRecall",
    "Trddt",
    "td",
    "yearQuarter",
]

HORIZONS = list(range(0, 51))
START_YEAR = 2005
END_YEAR = 2025
WINSOR_TAIL = 0.01

# 双侧显著性水平与置信区间的对应关系：
# 1%、5%、10% 显著性水平分别对应 99%、95%、90% 置信区间。
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

OUTCOME_LABELS = {
    horizon: (
        "OIBvol_t_w"
        if horizon == 0
        else f"OIBvol_tp{horizon}_w"
    )
    for horizon in HORIZONS
}

OUTCOME_VALUE_COL = "_single_day_oibvol_scaled_w"
OUTCOME_TD_COL = "_outcome_td"
OUTCOME_DATE_COL = "_outcome_Trddt"
TARGET_TD_COL = "_target_td"

SAMPLE_MODE_LABELS = {
    "available": "逐日可用样本",
    "common50": "t至t+50完整共同样本",
}

BASE_SOURCE_LABELS = {
    "neutral": "旧累计因变量筛选前的中性基础面板",
    "legacy": "旧累计收益基准回归样本",
}

OUTCOME_DEFINITION = "目标交易日单日小单成交量订单不平衡"
OUTCOME_SOURCE_DEFINITION = "OIB_CSMAR_prep.csv中的OIBvol"
TARGET_ALIGNMENT_DEFINITION = "目标交易日序号=事件交易日序号+h"
TARGET_MISSING_DEFINITION = "目标股票日不存在或OIBvol缺失时保持缺失，不填零"
WINSOR_DEFINITION = "OIB股票日度源表共同1%/99%阈值，先缩尾后匹配目标日"
SCALE_DEFINITION = "OIBvol原值为百分点，缩尾后除以100进入回归"
DATE_FE_BASE_DEFINITION = "事件起点交易日期"
CLUSTER_DEFINITION = "被提示公司和事件起点交易日期双向聚类"
ANALYSIS_VERSION = "单日散户订单不平衡_v1"

RESULT_COMPLETION_FIELDS = [
    "窗口序号",
    "相对交易日",
    "因变量",
    "线索系数",
    "聚类标准误",
    "置信区间下界",
    "置信区间上界",
    "t值",
    "p值",
    "回归样本量",
    "被提示公司聚类数",
    "日期聚类数",
    "删除单例组观测数",
    "解释变量矩阵条件数",
    "系数求解方法",
    "R方",
    "目标日有效OIB观测数",
    "目标日日期起点",
    "目标日日期终点",
    "缩尾下界（百分点）",
    "缩尾上界（百分点）",
]


def normalize_stock_code_series(series: pd.Series) -> pd.Series:
    """将股票代码统一为保留前导零的六位字符串。"""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def format_relative_day(horizon: int) -> str:
    """把交易日偏移量格式化为 t 或 t+h。"""
    return "t" if horizon == 0 else f"t+{horizon}"


def _coerce_required_integer(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """把必要的交易日序号字段转换为非缺失整数。"""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{column_name} 存在缺失或无法解析的取值。")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{column_name} 存在非整数取值。")
    return numeric.astype("int64")


def extract_td_date_map(
    data: pd.DataFrame,
    *,
    td_col: str,
    date_col: str,
    data_name: str,
) -> pd.DataFrame:
    """提取并校验交易日序号与自然日期的一一对应关系。"""
    require_columns(data, [td_col, date_col], data_name)
    mapping = data[[td_col, date_col]].drop_duplicates().copy()
    if mapping[[td_col, date_col]].isna().any().any():
        raise ValueError(f"{data_name} 的交易日序号或交易日期存在缺失。")
    if mapping.duplicated(td_col, keep=False).any():
        raise ValueError(f"{data_name} 中同一交易日序号对应多个日期。")
    if mapping.duplicated(date_col, keep=False).any():
        raise ValueError(f"{data_name} 中同一交易日期对应多个交易日序号。")
    return mapping.sort_values(td_col).reset_index(drop=True)


def validate_shared_td_calendar(
    base: pd.DataFrame,
    daily_lookup: pd.DataFrame,
) -> None:
    """确认基础面板与OIB源表使用同一套市场交易日序号。"""
    base_map = extract_td_date_map(
        base,
        td_col="td",
        date_col="Trddt",
        data_name="回归基础面板",
    ).rename(columns={"Trddt": "_base_Trddt"})
    outcome_map = extract_td_date_map(
        daily_lookup,
        td_col=OUTCOME_TD_COL,
        date_col=OUTCOME_DATE_COL,
        data_name="单日OIB查找表",
    ).rename(columns={OUTCOME_DATE_COL: "_oib_Trddt"})

    overlap = base_map.merge(
        outcome_map,
        left_on="td",
        right_on=OUTCOME_TD_COL,
        how="inner",
        validate="1:1",
    )
    if overlap.empty:
        raise ValueError("基础面板和单日OIB表没有重叠的交易日序号。")
    if not overlap["_base_Trddt"].eq(overlap["_oib_Trddt"]).all():
        raise ValueError("基础面板和单日OIB表使用的交易日序号口径不一致。")


def prepare_daily_oib_lookup(
    daily: pd.DataFrame,
    winsor_tail: float = WINSOR_TAIL,
) -> tuple[pd.DataFrame, float, float]:
    """校验、统一缩尾并整理股票单日OIB查找表。"""
    required = ["Stkcd", "Trddt", "td", "OIBvol"]
    require_columns(daily, required, "OIB股票日度源表")

    if not 0 <= winsor_tail < 0.5:
        raise ValueError("缩尾尾部比例必须位于 [0, 0.5) 区间。")

    out = daily[required].copy()
    out["Stkcd"] = normalize_stock_code_series(out["Stkcd"])
    out["td"] = _coerce_required_integer(out["td"], "td")
    out["Trddt"] = pd.to_datetime(out["Trddt"], errors="coerce")
    out["OIBvol"] = pd.to_numeric(out["OIBvol"], errors="coerce")
    out["OIBvol"] = out["OIBvol"].replace([np.inf, -np.inf], np.nan)

    if out[["Stkcd", "Trddt"]].isna().any().any():
        raise ValueError("OIB股票日度表存在缺失的股票代码或交易日期。")

    duplicate_count = out.duplicated(["Stkcd", "td"]).sum()
    if duplicate_count:
        raise ValueError(
            "OIB股票日度表的股票和交易日序号不唯一："
            f"{duplicate_count:,} 条重复观测。"
        )

    valid_oib = out["OIBvol"].dropna()
    if valid_oib.empty:
        raise ValueError("OIB股票日度表没有可用的单日OIBvol。")
    if valid_oib.abs().gt(100.0001).any():
        raise ValueError("OIBvol 超出理论取值区间 [-100, 100]。")

    lower = float(valid_oib.quantile(winsor_tail))
    upper = float(valid_oib.quantile(1 - winsor_tail))
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("无法得到有效的单日OIB缩尾阈值。")

    # OIBvol 原值是百分点。除以100后进入公共核心，核心会在输出时
    # 再乘100，从而使最终系数与标准误仍以百分点表示。
    out[OUTCOME_VALUE_COL] = (
        out["OIBvol"]
        .clip(lower=lower, upper=upper)
        .div(100.0)
        .astype("float32")
    )

    out = out.rename(
        columns={
            "Stkcd": "StkcdRecall",
            "td": OUTCOME_TD_COL,
            "Trddt": OUTCOME_DATE_COL,
        }
    )

    lookup = out[
        [
            "StkcdRecall",
            OUTCOME_TD_COL,
            OUTCOME_DATE_COL,
            OUTCOME_VALUE_COL,
        ]
    ].copy()
    return lookup, lower, upper


def read_daily_oib_lookup(
    path: Path = DAILY_OIB_PATH,
) -> tuple[pd.DataFrame, float, float]:
    """仅读取构造单日因变量需要的四个OIB字段。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到OIB股票日度源表：{path}")

    daily = pd.read_csv(
        path,
        usecols=["Stkcd", "Trddt", "td", "OIBvol"],
        dtype={"Stkcd": "string", "td": "int32", "OIBvol": "float32"},
        parse_dates=["Trddt"],
        encoding="utf-8-sig",
    )
    return prepare_daily_oib_lookup(
        daily,
        winsor_tail=WINSOR_TAIL,
    )


def detect_media_controls(available_columns: Sequence[str]) -> list[str]:
    """根据基础面板字段自动识别中性版或旧版媒体控制变量。"""
    available = set(available_columns)
    neutral_available = set(NEUTRAL_MEDIA_COLS).issubset(available)
    legacy_available = set(LEGACY_MEDIA_COLS).issubset(available)

    if neutral_available:
        return NEUTRAL_MEDIA_COLS.copy()
    if legacy_available:
        return LEGACY_MEDIA_COLS.copy()

    raise ValueError(
        "回归基础面板既不包含中性媒体控制变量"
        f"{NEUTRAL_MEDIA_COLS}，也不包含旧版媒体控制变量"
        f"{LEGACY_MEDIA_COLS}。"
    )


def read_base_panel(
    path: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """读取基础面板，并自动确定媒体控制变量和完整右端变量。"""
    if not path.exists():
        raise FileNotFoundError(
            f"未找到回归基础面板：{path}\n"
            "中性基础面板需要先运行相应的数据合并脚本生成。"
        )

    available = pd.read_csv(
        path,
        nrows=0,
        encoding="utf-8-sig",
    ).columns.tolist()

    # 保留媒体控制变量的自动识别
    media_cols = detect_media_controls(available)
    control_cols = OWN_EA_COLS + DISCLOSURE_COLS + media_cols
    rhs_cols = [CUE_COL] + control_cols
    read_cols = list(dict.fromkeys(BASE_KEY_COLS + rhs_cols))

    # 优先读取新名称，旧表没有新名称时读取对应旧名称
    aliases = {
        "ownOtherEventER": "ownMajorDisclosure",
        "ownPerformanceForecastBrief": "ownPerformanceForecast",
    }
    source_columns = {
        column: (
            column
            if column in available
            else aliases.get(column, column)
        )
        for column in read_cols
    }

    missing = sorted(set(source_columns.values()) - set(available))
    if missing:
        raise ValueError(f"回归基础面板缺少变量：{missing}")

    data = pd.read_csv(
        path,
        usecols=list(source_columns.values()),
        dtype={
            "StkcdRecall": "string",
            "yearQuarter": "string",
            **{
                source_columns[column]: "float32"
                for column in rhs_cols
            },
        },
        parse_dates=["Trddt"],
        encoding="utf-8-sig",
    )

    # 读入后统一使用新名称
    data = data.rename(
        columns={
            source: target
            for target, source in source_columns.items()
            if source != target
        }
    )

    # 标准化股票代码和交易日序号
    data["StkcdRecall"] = normalize_stock_code_series(
        data["StkcdRecall"]
    )
    data["td"] = _coerce_required_integer(data["td"], "td")
    if data[["StkcdRecall", "Trddt"]].isna().any().any():
        raise ValueError("回归基础面板存在缺失的公司代码或事件起点日期。")

    data = data.loc[
        data["Trddt"].dt.year.between(START_YEAR, END_YEAR)
    ].copy()

    duplicate_count = data.duplicated(["StkcdRecall", "td"]).sum()
    if duplicate_count:
        raise ValueError(
            "回归基础面板的被提示公司和交易日序号不唯一："
            f"{duplicate_count:,} 条重复观测。"
        )

    invariant_columns = [
        "StkcdRecall",
        "Trddt",
        "td",
        "yearQuarter",
        *rhs_cols,
    ]
    before = len(data)
    data = data.dropna(subset=invariant_columns).reset_index(drop=True)
    print(
        "基础面板中因固定效应或右端变量缺失删除的观测数："
        f"{before - len(data):,}"
    )
    return data, media_cols, rhs_cols


class DailyOutcomeLookup:
    """一次建立公司×交易日索引，供所有相对日复用。"""

    def __init__(self, daily_lookup: pd.DataFrame) -> None:
        require_columns(
            daily_lookup,
            [
                "StkcdRecall",
                OUTCOME_TD_COL,
                OUTCOME_DATE_COL,
                OUTCOME_VALUE_COL,
            ],
            "单日OIB查找表",
        )
        key_index = pd.MultiIndex.from_frame(
            daily_lookup[["StkcdRecall", OUTCOME_TD_COL]]
        )
        if not key_index.is_unique:
            raise pd.errors.MergeError("单日OIB查找表的公司和交易日序号不唯一。")

        self._values = daily_lookup[
            [OUTCOME_TD_COL, OUTCOME_DATE_COL, OUTCOME_VALUE_COL]
        ].copy(deep=False)
        self._values.index = key_index

    def match(self, base: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """精确定位目标市场交易日；未匹配的目标日保留缺失。"""
        target_index = pd.MultiIndex.from_arrays(
            [base["StkcdRecall"], base["td"] + int(horizon)],
            names=["StkcdRecall", OUTCOME_TD_COL],
        )
        if not target_index.is_unique:
            raise pd.errors.MergeError("回归基础面板的公司和交易日序号不唯一。")
        return self._values.reindex(target_index).reset_index(drop=True)


def attach_daily_outcome(
    base: pd.DataFrame,
    daily_lookup: pd.DataFrame | DailyOutcomeLookup,
    horizon: int,
) -> pd.DataFrame:
    """按 StkcdRecall 和 td+h 精确匹配一个相对日的单日OIB。"""
    if not isinstance(horizon, (int, np.integer)):
        raise TypeError("相对交易日序号必须是整数。")
    if horizon < 0:
        raise ValueError("相对交易日序号不能为负数。")

    require_columns(base, ["StkcdRecall", "td"], "回归基础面板")
    lookup = (
        daily_lookup
        if isinstance(daily_lookup, DailyOutcomeLookup)
        else DailyOutcomeLookup(daily_lookup)
    )
    matched_values = lookup.match(base, int(horizon))

    # 右端变量和起点日期均不随窗口改变。只新增三列目标日数据，
    # 共享基础表的只读列，避免每轮复制整张控制变量宽表。
    aligned = base.copy(deep=False)
    aligned.index = pd.RangeIndex(len(base))
    for column in matched_values.columns:
        aligned[column] = matched_values[column].array
    return aligned


def build_common_sample_mask(
    base: pd.DataFrame,
    daily_lookup: pd.DataFrame,
    horizons: Sequence[int] = HORIZONS,
) -> pd.Series:
    """识别 t 至 t+50 的所有相对日均有单日OIB的共同样本。"""
    horizons = [int(horizon) for horizon in horizons]
    if not horizons:
        raise ValueError("共同样本至少需要一个相对交易日。")
    if min(horizons) < 0:
        raise ValueError("共同样本的相对交易日序号不能为负数。")

    valid_lookup = daily_lookup.loc[
        daily_lookup[OUTCOME_VALUE_COL].notna(),
        ["StkcdRecall", OUTCOME_TD_COL],
    ]
    valid_index = pd.MultiIndex.from_frame(valid_lookup)

    firm_values = base["StkcdRecall"].astype("string").to_numpy()
    origin_td = base["td"].to_numpy(dtype=np.int64, copy=False)
    common = np.ones(len(base), dtype=bool)

    for horizon in horizons:
        target_index = pd.MultiIndex.from_arrays(
            [
                firm_values,
                origin_td + horizon,
            ],
            names=["StkcdRecall", OUTCOME_TD_COL],
        )
        common &= target_index.isin(valid_index)
        if not common.any():
            break

    return pd.Series(common, index=base.index, dtype=bool)


def resolve_base_path(base_source: str) -> Path:
    """根据基础样本口径返回输入路径。"""
    if base_source == "neutral":
        return NEUTRAL_BASE_PATH
    if base_source == "legacy":
        return LEGACY_BASE_PATH
    raise ValueError(f"未知基础样本口径：{base_source}")


def resolve_output_paths(
    sample_mode: str,
    base_source: str,
) -> tuple[Path, Path]:
    """为不同样本口径生成互不覆盖的结果和图形路径。"""
    sample_label = SAMPLE_MODE_LABELS[sample_mode]
    base_short_label = {
        "neutral": "中性基础面板",
        "legacy": "旧累计筛选样本",
    }[base_source]
    stem = (
        "图3扩展_单日散户订单不平衡_加入日期固定效应_"
        f"{sample_label}_{base_short_label}"
    )
    return RESULT_DIR / f"{stem}结果.csv", FIGURE_DIR / f"{stem}.png"


def save_results(rows: Iterable[dict], result_path: Path) -> pd.DataFrame:
    """按相对日排序并原子保存结构化结果。"""
    results = (
        pd.DataFrame(list(rows))
        .drop_duplicates("窗口序号", keep="last")
        .sort_values("窗口序号")
        .reset_index(drop=True)
    )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = result_path.with_suffix(result_path.suffix + ".tmp")
    results.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
    )
    temporary_path.replace(result_path)
    return results


def result_metadata(
    *,
    sample_label: str,
    base_label: str,
    media_cols: Sequence[str],
) -> dict[str, str]:
    """返回续算和最终校验共同使用的分析口径元数据。"""
    return {
        "分析版本": ANALYSIS_VERSION,
        "被解释变量口径": OUTCOME_DEFINITION,
        "被解释变量来源": OUTCOME_SOURCE_DEFINITION,
        "目标日对齐口径": TARGET_ALIGNMENT_DEFINITION,
        "目标日缺失处理": TARGET_MISSING_DEFINITION,
        "样本口径": sample_label,
        "基础面板口径": base_label,
        "媒体控制变量": ",".join(media_cols),
        "缩尾口径": WINSOR_DEFINITION,
        "量纲转换": SCALE_DEFINITION,
        "日期固定效应": "是",
        "日期固定效应基准": DATE_FE_BASE_DEFINITION,
        "被提示公司×年度季度固定效应": "是",
        "聚类方式": CLUSTER_DEFINITION,
        "控制变量": "是",
        "结果单位": "百分点",
        "有限样本修正": "是",
    }


def load_completed_rows(
    *,
    resume: bool,
    result_path: Path,
    sample_label: str,
    base_label: str,
    media_cols: Sequence[str],
    winsor_lower: float,
    winsor_upper: float,
) -> list[dict]:
    """读取与本次数据和口径完全一致的已完成相对日。"""
    if not resume or not result_path.exists():
        return []

    existing = pd.read_csv(result_path, encoding="utf-8-sig")
    if existing.empty:
        return []
    if "窗口序号" not in existing.columns:
        raise ValueError("已有单日OIB结果缺少“窗口序号”，无法继续估计。")

    invalid_horizons = sorted(
        set(existing["窗口序号"].dropna().astype(int)) - set(HORIZONS)
    )
    if invalid_horizons:
        raise ValueError(f"已有结果包含无效相对日：{invalid_horizons}")

    metadata = result_metadata(
        sample_label=sample_label,
        base_label=base_label,
        media_cols=media_cols,
    )
    for column in [*RESULT_COMPLETION_FIELDS, *metadata]:
        if column not in existing.columns:
            existing[column] = np.nan

    complete = existing[RESULT_COMPLETION_FIELDS].notna().all(axis=1)
    for column, expected in metadata.items():
        complete &= existing[column].astype(str).eq(str(expected))

    lower_matches = np.isclose(
        pd.to_numeric(
            existing["缩尾下界（百分点）"],
            errors="coerce",
        ),
        winsor_lower,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=False,
    )
    upper_matches = np.isclose(
        pd.to_numeric(
            existing["缩尾上界（百分点）"],
            errors="coerce",
        ),
        winsor_upper,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=False,
    )
    complete &= lower_matches & upper_matches

    incomplete_horizons = sorted(
        existing.loc[~complete, "窗口序号"]
        .dropna()
        .astype(int)
        .tolist()
    )
    if incomplete_horizons:
        print(
            "以下已有相对日的字段或口径不完整，将重新估计："
            f"{incomplete_horizons}"
        )

    return (
        existing.loc[complete]
        .drop_duplicates("窗口序号", keep="last")
        .sort_values("窗口序号")
        .to_dict("records")
    )


def validate_results(
    results: pd.DataFrame,
    *,
    sample_label: str,
    base_label: str,
    media_cols: Sequence[str],
) -> None:
    """绘图前校验51个相对日、统计量和口径元数据。"""
    if len(results) != len(HORIZONS):
        raise ValueError("单日OIB结果未包含 t 至 t+50 的全部相对交易日。")
    if results["窗口序号"].astype(int).tolist() != HORIZONS:
        raise ValueError("单日OIB相对日不完整或顺序错误。")

    metadata = result_metadata(
        sample_label=sample_label,
        base_label=base_label,
        media_cols=media_cols,
    )
    for column, expected in metadata.items():
        if column not in results.columns:
            raise ValueError(f"单日OIB结果缺少口径字段：{column}")
        values = set(results[column].dropna().astype(str))
        if values != {expected}:
            raise ValueError(f"单日OIB结果的“{column}”标记不正确。")

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
        "目标日有效OIB观测数",
        "缩尾下界（百分点）",
        "缩尾上界（百分点）",
    ]
    numeric = results[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("单日OIB结果包含缺失或非有限统计量。")
    if not results["聚类标准误"].gt(0).all():
        raise ValueError("单日OIB结果包含非正标准误。")
    if not results["p值"].between(0, 1).all():
        raise ValueError("单日OIB结果包含无效p值。")

    interval_valid = (
        results["置信区间下界"].le(results["线索系数"])
        & results["线索系数"].le(results["置信区间上界"])
    )
    if not interval_valid.all():
        raise ValueError("单日OIB结果包含无效置信区间。")

    if results["相对交易日"].tolist() != [
        format_relative_day(horizon) for horizon in HORIZONS
    ]:
        raise ValueError("结果表的相对交易日标签不正确。")
    if results["因变量"].tolist() != [
        OUTCOME_LABELS[horizon] for horizon in HORIZONS
    ]:
        raise ValueError("结果表的单日OIB因变量名称不正确。")


def draw_single_day_figure(
    results: pd.DataFrame,
    output_path: Path,
    sample_label: str,
) -> None:
    """绘制51个单日系数及逐点99%、95%、90%置信区间。"""
    plot_data = results.sort_values("窗口序号").reset_index(drop=True)

    font_path = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(10, 5))
    for spec in ERROR_BAR_SPECS:
        interval_radius = spec["critical_value"] * plot_data["聚类标准误"]
        axis.vlines(
            plot_data["窗口序号"],
            plot_data["线索系数"] - interval_radius,
            plot_data["线索系数"] + interval_radius,
            colors=spec["color"],
            linewidth=spec["line_width"],
            label=spec["label"],
            zorder=spec["zorder"],
        )

    # 按10%显著性水平判断：90%置信区间不包含0
    significant = (
        plot_data["线索系数"].abs()
        > 1.6448536269514722 * plot_data["聚类标准误"]
    )
    point_colors = np.where(significant, "#d62728", "#1a1a1a")

    axis.scatter(
        plot_data["窗口序号"],
        plot_data["线索系数"],
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
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_title(
        "单日散户订单不平衡动态"
        f"（控制交易日期固定效应，{sample_label}）"
    )
    axis.set_xlabel("相对交易日")
    axis.set_ylabel("提示变量系数（百分点）")

    ticks = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    axis.set_xticks(ticks)
    axis.set_xticklabels([format_relative_day(tick) for tick in ticks])
    axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=9,
        handlelength=1.4,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """读取样本口径、基础面板和续算参数。"""
    parser = argparse.ArgumentParser(
        description="估计事件日t至t+50的单日散户订单不平衡动态。"
    )
    parser.add_argument(
        "--sample-mode",
        choices=sorted(SAMPLE_MODE_LABELS),
        default="available",
        help=(
            "available=每个相对日使用该日可用样本；"
            "common50=t至t+50均有效的共同样本。"
        ),
    )
    parser.add_argument(
        "--base-source",
        choices=sorted(BASE_SOURCE_LABELS),
        default="neutral",
        help=(
            "neutral=旧累计因变量筛选前的中性基础面板；"
            "legacy=旧累计收益最终回归样本，仅用于可比性检验。"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从数据和分析口径完全一致的已有结果继续估计。",
    )
    return parser.parse_args()


def main() -> None:
    """串联基础样本、单日OIB匹配、逐日回归和独立图形输出。"""
    args = parse_args()
    sample_label = SAMPLE_MODE_LABELS[args.sample_mode]
    base_label = BASE_SOURCE_LABELS[args.base_source]
    base_path = resolve_base_path(args.base_source)
    result_path, output_path = resolve_output_paths(
        args.sample_mode,
        args.base_source,
    )

    print(f"读取回归基础面板：{base_path}")
    base, media_cols, rhs_cols = read_base_panel(base_path)
    print(f"自动识别的媒体控制变量：{media_cols}")

    print(f"读取OIB股票日度源表：{DAILY_OIB_PATH}")
    daily_lookup, winsor_lower, winsor_upper = read_daily_oib_lookup()
    validate_shared_td_calendar(base, daily_lookup)
    print(
        "单日OIB共同缩尾阈值（百分点）："
        f"下界={winsor_lower:.10f}，上界={winsor_upper:.10f}"
    )

    if args.sample_mode == "common50":
        print("正在识别t至t+50均有单日OIB的共同样本……")
        common_mask = build_common_sample_mask(
            base,
            daily_lookup,
            horizons=HORIZONS,
        )
        removed = int((~common_mask).sum())
        base = base.loc[common_mask].reset_index(drop=True)
        print(f"共同样本筛选删除观测数：{removed:,}")
        if base.empty:
            raise ValueError("t至t+50完整共同样本为空。")

    existing_rows = load_completed_rows(
        resume=args.resume,
        result_path=result_path,
        sample_label=sample_label,
        base_label=base_label,
        media_cols=media_cols,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
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
    rows = existing_rows.copy()
    outcome_lookup = DailyOutcomeLookup(daily_lookup) if pending else None

    metadata = result_metadata(
        sample_label=sample_label,
        base_label=base_label,
        media_cols=media_cols,
    )

    for position, horizon in enumerate(pending, start=1):
        print(
            "正在运行单日散户订单不平衡回归："
            f"{format_relative_day(horizon)}"
            f"（待估计 {position}/{len(pending)}）"
        )
        aligned = attach_daily_outcome(
            base,
            outcome_lookup,
            horizon,
        )
        valid_outcome = aligned[OUTCOME_VALUE_COL].notna()
        valid_count = int(valid_outcome.sum())
        if valid_count == 0:
            raise ValueError(
                f"{format_relative_day(horizon)} 没有可用的单日OIBvol。"
            )

        valid_dates = aligned.loc[valid_outcome, OUTCOME_DATE_COL]
        outcome_label = OUTCOME_LABELS[horizon]
        row = run_date_fe_regression(
            aligned,
            horizon=horizon,
            outcome=OUTCOME_VALUE_COL,
            outcome_label=outcome_label,
            rhs_cols=rhs_cols,
            cue_col=CUE_COL,
        )
        row.update(
            {
                **metadata,
                "相对交易日": format_relative_day(horizon),
                "缩尾下界（百分点）": winsor_lower,
                "缩尾上界（百分点）": winsor_upper,
                "目标日有效OIB观测数": valid_count,
                "目标日日期起点": valid_dates.min().strftime("%Y-%m-%d"),
                "目标日日期终点": valid_dates.max().strftime("%Y-%m-%d"),
            }
        )

        rows = [
            existing
            for existing in rows
            if int(existing["窗口序号"]) != horizon
        ]
        rows.append(row)
        results = save_results(rows, result_path)
        print(
            f"{format_relative_day(horizon)} 完成："
            f"系数={row['线索系数']:.6f}，"
            f"标准误={row['聚类标准误']:.6f}，"
            f"样本量={row['回归样本量']:,}"
        )

        del aligned, valid_outcome, valid_dates
        gc.collect()

    if not pending:
        print("已有结果包含t至t+50的全部相对交易日，无需重新估计。")
        results = save_results(existing_rows, result_path)

    validate_results(
        results,
        sample_label=sample_label,
        base_label=base_label,
        media_cols=media_cols,
    )
    draw_single_day_figure(
        results,
        output_path,
        sample_label,
    )

    print(f"单日散户订单不平衡动态结果已保存：{result_path}")
    print(f"单日散户订单不平衡动态图已保存：{output_path}")


if __name__ == "__main__":
    main()
