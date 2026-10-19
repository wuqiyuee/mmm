"""
估计并绘制事件当日至第50个交易日的单日特征调整收益动态，共51个点。

新的被解释变量不是 [t,t+h] 累计收益，也不是累计收益的一阶差分，
而是被提示公司在目标交易日 t+h 的当日 DGTW 特征调整收益率：

    个股 t+h 日收益 - 个股在 t+h 日所属 DGTW 组合的当日基准收益。

目标日通过统一市场交易日序号严格匹配：target_td = origin_td + h。
缺少目标日观测时保持缺失，不将停牌、退市或样本边界缺失填成 0。

主规格默认使用“旧累计因变量筛选前的中性基础面板”，每个相对日
使用该日因变量可用的观测。命令行可切换为 51 日完整共同样本，
也可使用旧最终回归面板做可比性检验。三个口径使用不同结果文件，
不会覆盖现有累计收益图2及其结果。

固定效应和标准误：
1. 被提示公司 × 年度季度固定效应；
2. 事件起点交易日期固定效应；
3. 被提示公司和事件起点交易日期双向聚类；
4. 双向聚类协方差使用与现有日期固定效应版图2一致的有限样本修正。

收益率在 DGTW 股票日度源表上使用同一组 1%/99% 阈值缩尾，随后
再匹配到各相对交易日。这样 51 个点不会因分别估计缩尾阈值而改变
量纲。每完成一个相对日就原子保存一次结果，--resume 可继续估计。
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

# 公共回归核心统一位于实证分析的公共模块目录。
def find_project_root() -> Path:
    """按项目说明、代码和中间结果三个标记，从当前文件的祖先目录寻根。"""
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "AGENTS.md").is_file()
            and (candidate / "code").is_dir()
            and (candidate / "中间结果").is_dir()
        ):
            return candidate
    raise FileNotFoundError("无法找到包含项目说明、代码和中间结果的项目根目录。")


ROOT = find_project_root()
MODULE_DIR = ROOT / "code" / "07_实证分析" / "90_公共模块"
sys.path.insert(0, str(MODULE_DIR))

from 图2图3扩展_日期固定效应回归公共模块 import (
    require_columns,
    run_date_fe_regression,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager



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

DAILY_ADJUSTED_RETURN_PATH = (
    ROOT
    / "中间结果"
    / "02_变量构造结果"
    / "1DGTW特征调整收益率"
    / "DGTW特征调整收益率.csv"
)


RESULT_DIR = ROOT / "图表结果" / "02_扩展分析" / "单日特征调整收益"

FIGURE_DIR = RESULT_DIR

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

CONTROL_COLS = OWN_EA_COLS + DISCLOSURE_COLS + MEDIA_COLS
RHS_COLS = [CUE_COL] + CONTROL_COLS
BASE_KEY_COLS = ["StkcdRecall", "Trddt", "td", "yearQuarter"]
BASE_READ_COLS = list(dict.fromkeys(BASE_KEY_COLS + RHS_COLS))

HORIZONS = list(range(0, 51))
START_YEAR = 2005
END_YEAR = 2025
WINSOR_TAIL = 0.01

# 双侧显著性水平与置信区间的对应关系：
# 1%、5%、10% 显著性水平分别对应 99%、95%、90% 置信区间。
# 从最长区间到最短区间依次绘制。采用浅灰、蓝灰、深蓝的克制配色，
# 同时用递增线宽区分层级；不绘制端帽，减少51个点叠加时的视觉噪声。
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

# 与现有累计变量 chAdjreturn_t_tp*_Recall_w 区分：少一个“t_”表示
# 只取目标日 t+h 当日，不包含从 t 开始的累计窗口。
OUTCOME_LABELS = {
    horizon: f"chAdjreturn_tp{horizon}_Recall_w"
    for horizon in HORIZONS
}

OUTCOME_VALUE_COL = "_single_day_chadj_return_w"
OUTCOME_TD_COL = "_outcome_td"
OUTCOME_DATE_COL = "_outcome_Trddt"
TARGET_TD_COL = "_target_td"

SAMPLE_MODE_LABELS = {
    "available": "逐日可用样本",
    "common51": "51日完整共同样本（含事件当日）",
}

BASE_SOURCE_LABELS = {
    "neutral": "旧累计因变量筛选前的中性基础面板",
    "legacy": "旧累计收益基准回归样本",
}

OUTCOME_DEFINITION = "目标交易日单日DGTW特征调整收益率"
WINSOR_DEFINITION = "日度源表共同1%/99%阈值，先缩尾后匹配目标日"
OUTCOME_SOURCE_DEFINITION = "DGTW特征调整收益率.csv中的chAdj_ret_t"
TARGET_PORTFOLIO_DEFINITION = "目标交易日实际所属DGTW组合"
TARGET_ALIGNMENT_DEFINITION = "目标交易日序号=事件交易日序号+h"
TARGET_MISSING_DEFINITION = "保持缺失，不填零"
DATE_FE_BASE_DEFINITION = "事件起点交易日期"
CLUSTER_DEFINITION = "被提示公司和事件起点交易日期双向聚类"
ANALYSIS_VERSION = "单日特征调整收益_v1"

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
    "目标日有效收益观测数",
    "目标日日期起点",
    "目标日日期终点",
    "缩尾下界",
    "缩尾上界",
]


def normalize_stock_code_series(series: pd.Series) -> pd.Series:
    """将股票代码统一为保留前导零的六位字符串。"""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


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
    """确认基础面板和日度收益源表使用同一套市场交易日序号。"""
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
        data_name="单日收益查找表",
    ).rename(columns={OUTCOME_DATE_COL: "_daily_Trddt"})

    overlap = base_map.merge(
        outcome_map,
        left_on="td",
        right_on=OUTCOME_TD_COL,
        how="inner",
        validate="1:1",
    )
    if overlap.empty:
        raise ValueError("基础面板和单日收益表没有重叠的交易日序号。")
    if not overlap["_base_Trddt"].eq(overlap["_daily_Trddt"]).all():
        raise ValueError("基础面板和单日收益表使用的交易日序号口径不一致。")


def prepare_daily_return_lookup(
    daily: pd.DataFrame,
    winsor_tail: float = WINSOR_TAIL,
) -> tuple[pd.DataFrame, float, float]:
    """校验并整理股票日度特征调整收益查找表。"""
    required = ["Stkcd", "td", "Trddt", "chAdj_ret_t"]
    require_columns(daily, required, "DGTW股票日度特征调整收益表")

    if not 0 <= winsor_tail < 0.5:
        raise ValueError("缩尾尾部比例必须位于 [0, 0.5) 区间。")

    out = daily[required].copy()
    out["Stkcd"] = normalize_stock_code_series(out["Stkcd"])
    out["td"] = _coerce_required_integer(out["td"], "td")
    out["Trddt"] = pd.to_datetime(out["Trddt"], errors="coerce")
    out["chAdj_ret_t"] = pd.to_numeric(
        out["chAdj_ret_t"],
        errors="coerce",
    )

    if out[["Stkcd", "Trddt"]].isna().any().any():
        raise ValueError("DGTW股票日度表存在缺失的股票代码或交易日期。")

    duplicate_count = out.duplicated(["Stkcd", "td"]).sum()
    if duplicate_count:
        raise ValueError(
            "DGTW股票日度表的股票和交易日序号不唯一："
            f"{duplicate_count:,} 条重复观测。"
        )

    valid_returns = out["chAdj_ret_t"].dropna()
    if valid_returns.empty:
        raise ValueError("DGTW股票日度表没有可用的当日特征调整收益率。")

    lower = float(valid_returns.quantile(winsor_tail))
    upper = float(valid_returns.quantile(1 - winsor_tail))
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("无法得到有效的单日收益缩尾阈值。")

    out[OUTCOME_VALUE_COL] = (
        out["chAdj_ret_t"]
        .clip(lower=lower, upper=upper)
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


def read_daily_return_lookup(
    path: Path = DAILY_ADJUSTED_RETURN_PATH,
) -> tuple[pd.DataFrame, float, float]:
    """只读取构造单日被解释变量需要的四个日度字段。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到DGTW股票日度收益表：{path}")

    daily = pd.read_csv(
        path,
        usecols=["Stkcd", "td", "Trddt", "chAdj_ret_t"],
        dtype={"Stkcd": "string", "chAdj_ret_t": "float32"},
        parse_dates=["Trddt"],
        encoding="utf-8-sig",
    )
    return prepare_daily_return_lookup(
        daily,
        winsor_tail=WINSOR_TAIL,
    )


def read_base_panel(path: Path) -> pd.DataFrame:
    """读取不含收益因变量的回归基础字段并检查唯一键。"""
    if not path.exists():
        raise FileNotFoundError(
            f"未找到回归基础面板：{path}\n"
            "中性基础面板需要先运行合并pattern_firms的数据.py生成。"
        )

    available = pd.read_csv(
        path,
        nrows=0,
        encoding="utf-8-sig",
    ).columns.tolist()
        # 新名称对应中性基础面板保留的旧接口名称
    aliases = {
        "ownOtherEventER": "ownMajorDisclosure",
        "ownPerformanceForecastBrief": "ownPerformanceForecast",
    }

    # 优先读取新名称；源表只有旧名称时，读取对应旧列
    source_columns = {
        column: (
            column
            if column in available
            else aliases.get(column, column)
        )
        for column in BASE_READ_COLS
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
                for column in RHS_COLS
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

    # 右端变量和固定效应均与相对日无关，提前统一筛除可以减少51次合并量。
    invariant_columns = [
        "StkcdRecall",
        "Trddt",
        "td",
        "yearQuarter",
        *RHS_COLS,
    ]
    before = len(data)
    data = data.dropna(subset=invariant_columns).reset_index(drop=True)
    print(
        "基础面板中因固定效应或右端变量缺失删除的观测数："
        f"{before - len(data):,}"
    )
    return data


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
            "单日收益查找表",
        )
        key_index = pd.MultiIndex.from_frame(
            daily_lookup[["StkcdRecall", OUTCOME_TD_COL]]
        )
        if not key_index.is_unique:
            raise pd.errors.MergeError("单日收益查找表的公司和交易日序号不唯一。")

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
    """按 StkcdRecall 和 td+h 精确匹配一个相对日的单日收益。"""
    if not isinstance(horizon, (int, np.integer)):
        raise TypeError("相对交易日序号必须是整数。")
    if horizon < 0:
        raise ValueError("相对交易日序号不能为负数。")

    require_columns(
        base,
        ["StkcdRecall", "td"],
        "回归基础面板",
    )
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
    """识别所有指定相对日均有单日收益的共同样本。"""
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
        "图2扩展_单日特征调整收益_加入日期固定效应_"
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


def load_completed_rows(
    *,
    resume: bool,
    result_path: Path,
    sample_label: str,
    base_label: str,
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
        raise ValueError("已有单日收益结果缺少“窗口序号”，无法继续估计。")

    invalid_horizons = sorted(
        set(existing["窗口序号"].dropna().astype(int)) - set(HORIZONS)
    )
    if invalid_horizons:
        raise ValueError(f"已有结果包含无效相对日：{invalid_horizons}")

    metadata = {
        "分析版本": ANALYSIS_VERSION,
        "被解释变量口径": OUTCOME_DEFINITION,
        "被解释变量来源": OUTCOME_SOURCE_DEFINITION,
        "目标日组合口径": TARGET_PORTFOLIO_DEFINITION,
        "目标日对齐口径": TARGET_ALIGNMENT_DEFINITION,
        "目标日缺失处理": TARGET_MISSING_DEFINITION,
        "样本口径": sample_label,
        "基础面板口径": base_label,
        "缩尾口径": WINSOR_DEFINITION,
        "日期固定效应": "是",
        "日期固定效应基准": DATE_FE_BASE_DEFINITION,
        "被提示公司×年度季度固定效应": "是",
        "聚类方式": CLUSTER_DEFINITION,
        "控制变量": "是",
        "结果单位": "百分点",
        "有限样本修正": "是",
    }

    for column in [*RESULT_COMPLETION_FIELDS, *metadata]:
        if column not in existing.columns:
            existing[column] = np.nan

    complete = existing[RESULT_COMPLETION_FIELDS].notna().all(axis=1)
    for column, expected in metadata.items():
        complete &= existing[column].astype(str).eq(str(expected))

    lower_matches = np.isclose(
        pd.to_numeric(existing["缩尾下界"], errors="coerce"),
        winsor_lower,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=False,
    )
    upper_matches = np.isclose(
        pd.to_numeric(existing["缩尾上界"], errors="coerce"),
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
) -> None:
    """绘图前校验51个相对日、统计量和口径元数据。"""
    if len(results) != len(HORIZONS):
        raise ValueError("单日收益结果未包含全部51个相对交易日。")
    if results["窗口序号"].astype(int).tolist() != HORIZONS:
        raise ValueError("单日收益相对日不完整或顺序错误。")

    required_metadata = {
        "分析版本": ANALYSIS_VERSION,
        "被解释变量口径": OUTCOME_DEFINITION,
        "被解释变量来源": OUTCOME_SOURCE_DEFINITION,
        "目标日组合口径": TARGET_PORTFOLIO_DEFINITION,
        "目标日对齐口径": TARGET_ALIGNMENT_DEFINITION,
        "目标日缺失处理": TARGET_MISSING_DEFINITION,
        "样本口径": sample_label,
        "基础面板口径": base_label,
        "缩尾口径": WINSOR_DEFINITION,
        "结果单位": "百分点",
        "控制变量": "是",
        "日期固定效应": "是",
        "日期固定效应基准": DATE_FE_BASE_DEFINITION,
        "被提示公司×年度季度固定效应": "是",
        "聚类方式": CLUSTER_DEFINITION,
        "有限样本修正": "是",
    }
    for column, expected in required_metadata.items():
        if column not in results.columns:
            raise ValueError(f"单日收益结果缺少口径字段：{column}")
        values = set(results[column].dropna().astype(str))
        if values != {expected}:
            raise ValueError(f"单日收益结果的“{column}”标记不正确。")

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
        "目标日有效收益观测数",
        "缩尾下界",
        "缩尾上界",
    ]
    numeric = results[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("单日收益结果包含缺失或非有限统计量。")
    if not results["聚类标准误"].gt(0).all():
        raise ValueError("单日收益结果包含非正标准误。")
    if not results["p值"].between(0, 1).all():
        raise ValueError("单日收益结果包含无效p值。")

    interval_valid = (
        results["置信区间下界"].le(results["线索系数"])
        & results["线索系数"].le(results["置信区间上界"])
    )
    if not interval_valid.all():
        raise ValueError("单日收益结果包含无效置信区间。")

    if results["相对交易日"].tolist() != [
        f"t+{horizon}" for horizon in HORIZONS
    ]:
        raise ValueError("结果表的相对交易日标签不正确。")
    if results["因变量"].tolist() != [
        OUTCOME_LABELS[horizon] for horizon in HORIZONS
    ]:
        raise ValueError("结果表的单日被解释变量名称不正确。")


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

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for spec in ERROR_BAR_SPECS:
        interval_radius = (
            spec["critical_value"] * plot_data["聚类标准误"]
        )
        axis.vlines(
            plot_data["窗口序号"],
            plot_data["线索系数"] - interval_radius,
            plot_data["线索系数"] + interval_radius,
            colors=spec["color"],
            linewidth=spec["line_width"],
            label=spec["label"],
            zorder=spec["zorder"],
        )

    # 与图中90%置信区间一致，判断10%水平显著性
    significant = (
        plot_data["线索系数"].abs()
        > 1.6448536269514722 * plot_data["聚类标准误"]
    )

    # 显著用红色，其余用黑色
    point_colors = np.where(significant, "#d62728", "#1a1a1a")

    axis.scatter(
        plot_data["窗口序号"],
        plot_data["线索系数"],
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
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_title(
        "事件当日及之后的单日特征调整收益动态\n"
        "固定效应：被提示公司×年度季度；交易日期（事件起点）\n"
        f"样本：{sample_label}",
        fontsize=11,
        pad=12,
    )
    axis.set_xlabel("相对交易日")
    axis.set_ylabel("提示变量系数（百分点）")

    ticks = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    axis.set_xticks(ticks)
    axis.set_xticklabels([
        "t" if tick == 0 else f"t+{tick}"
        for tick in ticks
    ])
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
    """读取样本口径和续算参数。"""
    parser = argparse.ArgumentParser(
        description="估计事件当日至第50个交易日的单日特征调整收益动态，共51个点。"
    )
    parser.add_argument(
        "--sample-mode",
        choices=sorted(SAMPLE_MODE_LABELS),
        default="available",
        help=(
            "available=每个相对日使用该日可用样本；"
            "common51=51个相对日均有效的共同样本。"
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
        help="从相同口径的已有完整结果继续估计。",
    )
    return parser.parse_args()


def main() -> None:
    """串联基础样本、目标日收益、逐日回归和独立图形输出。"""
    args = parse_args()
    sample_label = SAMPLE_MODE_LABELS[args.sample_mode]
    base_label = BASE_SOURCE_LABELS[args.base_source]
    base_path = resolve_base_path(args.base_source)
    result_path, output_path = resolve_output_paths(
        args.sample_mode,
        args.base_source,
    )

    print(f"读取回归基础面板：{base_path}")
    base = read_base_panel(base_path)

    print(f"读取DGTW股票日度收益表：{DAILY_ADJUSTED_RETURN_PATH}")
    daily_lookup, winsor_lower, winsor_upper = read_daily_return_lookup()
    validate_shared_td_calendar(base, daily_lookup)
    print(
        "单日特征调整收益共同缩尾阈值："
        f"下界={winsor_lower:.10f}，上界={winsor_upper:.10f}"
    )

    if args.sample_mode == "common51":
        print("正在识别51个相对交易日均有效的共同样本……")
        common_mask = build_common_sample_mask(
            base,
            daily_lookup,
            horizons=HORIZONS,
        )
        removed = int((~common_mask).sum())
        base = base.loc[common_mask].reset_index(drop=True)
        print(f"共同样本筛选删除观测数：{removed:,}")
        if base.empty:
            raise ValueError("51日完整共同样本为空。")

    existing_rows = load_completed_rows(
        resume=args.resume,
        result_path=result_path,
        sample_label=sample_label,
        base_label=base_label,
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

    for position, horizon in enumerate(pending, start=1):
        print(
            "正在运行单日特征调整收益回归："
            f"t+{horizon}（待估计 {position}/{len(pending)}）"
        )
        aligned = attach_daily_outcome(
            base,
            outcome_lookup,
            horizon,
        )
        valid_outcome = aligned[OUTCOME_VALUE_COL].notna()
        valid_count = int(valid_outcome.sum())
        if valid_count == 0:
            raise ValueError(f"t+{horizon} 没有可用的单日特征调整收益率。")

        valid_dates = aligned.loc[valid_outcome, OUTCOME_DATE_COL]
        outcome_label = OUTCOME_LABELS[horizon]
        row = run_date_fe_regression(
            aligned,
            horizon=horizon,
            outcome=OUTCOME_VALUE_COL,
            outcome_label=outcome_label,
            rhs_cols=RHS_COLS,
            cue_col=CUE_COL,
        )
        row.update(
            {
                "分析版本": ANALYSIS_VERSION,
                "被解释变量口径": OUTCOME_DEFINITION,
                "被解释变量来源": OUTCOME_SOURCE_DEFINITION,
                "目标日组合口径": TARGET_PORTFOLIO_DEFINITION,
                "目标日对齐口径": TARGET_ALIGNMENT_DEFINITION,
                "目标日缺失处理": TARGET_MISSING_DEFINITION,
                "样本口径": sample_label,
                "基础面板口径": base_label,
                "缩尾口径": WINSOR_DEFINITION,
                "缩尾下界": winsor_lower,
                "缩尾上界": winsor_upper,
                "目标日有效收益观测数": valid_count,
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
            f"t+{horizon} 完成："
            f"系数={row['线索系数']:.6f}，"
            f"标准误={row['聚类标准误']:.6f}，"
            f"样本量={row['回归样本量']:,}"
        )

        del aligned, valid_outcome, valid_dates
        gc.collect()

    if not pending:
        print("已有结果包含全部51个相对交易日，无需重新估计。")
        results = save_results(existing_rows, result_path)

    validate_results(
        results,
        sample_label=sample_label,
        base_label=base_label,
    )
    draw_single_day_figure(
        results,
        output_path,
        sample_label,
    )

    print(f"单日收益动态结果已保存：{result_path}")
    print(f"单日收益动态图已保存：{output_path}")


if __name__ == "__main__":
    main()
