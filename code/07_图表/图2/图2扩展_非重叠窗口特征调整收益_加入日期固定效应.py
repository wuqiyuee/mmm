"""
估计并绘制公告后非重叠窗口的 DGTW 特征调整收益。

检验窗口为：
    [t,t+1]、[t+2,t+5]、[t+6,t+10]、[t+11,t+20]、
    [t+21,t+30]、[t+31,t+40]、[t+41,t+50]。

对任意窗口 [t+a,t+b]，脚本不会把单日特征调整收益简单相加，
而是在目标起点 t+a 对应的 DGTW 日度源表行上，直接读取覆盖
b-a+1 个交易日的完整窗口特征调整收益：

    chAdj_ret_t_t(b-a)
    = 个股在 [t+a,t+b] 的复合收益
      - 目标起点实际 DGTW 组合在 [t+a,t+b] 的基准复合收益。

因此，[t+2,t+5] 使用 t+2 行的 chAdj_ret_t_t3，[t+6,t+10]
使用 t+6 行的 chAdj_ret_t_t4；四个十日窗口均使用各自起点行的
chAdj_ret_t_t9。若目标起点不存在，或源表完整窗口变量缺失，结果保持缺失，
不额外填零。

为了与单日收益扩展脚本保持一致，每种窗口长度对应的收益字段先在完整
DGTW 股票日度源表上按 1%/99% 阈值缩尾，再匹配事件样本。这是新的
稳健性检验口径，与原累计图在事件样本内逐因变量缩尾的口径并不完全相同。

回归规格与日期固定效应版图2一致：
1. 被提示公司 × 年度季度固定效应；
2. 事件起点交易日期固定效应；
3. 被提示公司和事件起点交易日期双向聚类；
4. 使用相同的右端控制变量和有限样本修正；
5. 原始收益率以小数进入回归，结果以百分点输出。

脚本拥有独立结果目录和图形目录，不修改、不覆盖现有累计收益图或
单日收益图。默认逐窗口使用该窗口可用样本；也可切换为七个窗口
全部有效的共同样本。--plot-only 可在已有结果上只重新绘图。
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections.abc import Iterable
from dataclasses import dataclass
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

DGTW_ADJUSTED_RETURN_PATH = (
    ROOT
    / "中间结果"
    / "02_变量构造结果"
    / "1DGTW特征调整收益率"
    / "DGTW特征调整收益率.csv"
)


RESULT_DIR = (
    ROOT
    / "中间结果"
    / "08_回归结果"
    / "扩展分析"
    / "非重叠窗口特征调整收益"
)

FIGURE_DIR = (
    ROOT
    / "图表结果"
    / "扩展分析"
    / "非重叠窗口特征调整收益"
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


@dataclass(frozen=True)
class WindowSpec:
    """一个非重叠事件窗口的固定定义。"""

    window_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def source_horizon(self) -> int:
        return self.end - self.start

    @property
    def source_column(self) -> str:
        if self.source_horizon == 0:
            return "chAdj_ret_t"
        return f"chAdj_ret_t_t{self.source_horizon}"

    @property
    def label(self) -> str:
        return f"[{format_relative_day(self.start)},{format_relative_day(self.end)}]"

    @property
    def outcome_label(self) -> str:
        start = "t" if self.start == 0 else f"tp{self.start}"
        end = "t" if self.end == 0 else f"tp{self.end}"
        return f"chAdjreturn_{start}_{end}_Recall_w"


WINDOW_SPECS = (
    WindowSpec(1, 0, 1),
    WindowSpec(2, 2, 5),
    WindowSpec(3, 6, 10),
    WindowSpec(4, 11, 20),
    WindowSpec(5, 21, 30),
    WindowSpec(6, 31, 40),
    WindowSpec(7, 41, 50),
)

WINDOW_BY_ID = {spec.window_id: spec for spec in WINDOW_SPECS}
SOURCE_RETURN_COLS = tuple(
    dict.fromkeys(spec.source_column for spec in WINDOW_SPECS)
)

WINSOR_TAIL = 0.01
SOURCE_TD_COL = "_source_start_td"
SOURCE_DATE_COL = "_source_start_Trddt"
TARGET_START_TD_COL = "_target_start_td"
OUTCOME_VALUE_COL = "_interval_chadj_return_w"

SAMPLE_MODE_LABELS = {
    "available": "逐窗口可用样本",
    "common7": "七区间完整共同样本",
}

BASE_SOURCE_LABELS = {
    "neutral": "旧累计因变量筛选前的中性基础面板",
    "legacy": "旧累计收益基准回归样本",
}

ANALYSIS_VERSION = "非重叠窗口特征调整收益_v1"
OUTCOME_DEFINITION = "目标区间完整复合DGTW特征调整收益率"
OUTCOME_SOURCE_DEFINITION = (
    "DGTW特征调整收益率.csv中目标起点对应的完整窗口变量"
)
TARGET_PORTFOLIO_DEFINITION = "目标区间起点实际所属DGTW组合"
TARGET_ALIGNMENT_DEFINITION = "目标区间起点交易日序号=事件交易日序号+区间起点"
TARGET_MISSING_DEFINITION = "目标起点不存在或DGTW完整窗口变量缺失时保持缺失，不额外填零"
WINSOR_DEFINITION = "按窗口长度对应源变量分别使用全DGTW源表1%/99%阈值"
DATE_FE_BASE_DEFINITION = "事件起点交易日期"
CLUSTER_DEFINITION = "被提示公司和事件起点交易日期双向聚类"

RESULT_COMPLETION_FIELDS = [
    "窗口序号",
    "事件窗口",
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
    "目标区间有效收益观测数",
    "目标区间起点日期最小值",
    "目标区间起点日期最大值",
    "按区间长度折算系数",
    "按区间长度折算标准误",
]

# 双侧显著性水平1%、5%、10%分别对应99%、95%、90%置信区间。
ERROR_BAR_SPECS = (
    {
        "label": "99%置信区间（1%水平）",
        "critical_value": 2.5758293035489004,
        "color": "#bdbdbd",
        "line_width": 0.9,
        "zorder": 1,
    },
    {
        "label": "95%置信区间（5%水平）",
        "critical_value": 1.959963984540054,
        "color": "#78909c",
        "line_width": 1.8,
        "zorder": 2,
    },
    {
        "label": "90%置信区间（10%水平）",
        "critical_value": 1.6448536269514722,
        "color": "#1f4e79",
        "line_width": 3.0,
        "zorder": 3,
    },
)


def format_relative_day(offset: int) -> str:
    """把交易日偏移量格式化为 t 或 t+h。"""
    return "t" if offset == 0 else f"t+{offset}"


def validate_window_specs() -> None:
    """检查七个窗口编号唯一、首尾相接并完整覆盖 t 至 t+50。"""
    if len(WINDOW_BY_ID) != len(WINDOW_SPECS):
        raise ValueError("非重叠窗口存在重复的窗口编号。")

    expected_start = 0
    for expected_id, spec in enumerate(WINDOW_SPECS, start=1):
        if spec.window_id != expected_id:
            raise ValueError("非重叠窗口编号必须从1开始连续排列。")
        if spec.start != expected_start or spec.end < spec.start:
            raise ValueError("非重叠窗口必须首尾相接且区间有效。")
        expected_start = spec.end + 1

    if expected_start != 51:
        raise ValueError("非重叠窗口必须完整覆盖 t 至 t+50。")


def winsor_column(source_column: str) -> str:
    """返回源收益字段对应的内部缩尾字段名。"""
    return f"_winsor_{source_column}"


def normalize_stock_code_series(series: pd.Series) -> pd.Series:
    """把股票代码统一为六位字符串。"""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("股票代码存在无法标准化的值。")
    return numeric.astype("int64").astype("string").str.zfill(6)


def coerce_required_integer(series: pd.Series, column: str) -> pd.Series:
    """把不允许缺失的交易日序号转换为整数。"""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{column} 存在缺失或无法转换的值。")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{column} 存在非整数值。")
    return numeric.astype("int64")


def extract_td_date_map(
    data: pd.DataFrame,
    *,
    td_col: str,
    date_col: str,
    data_name: str,
) -> pd.DataFrame:
    """提取并检查交易日序号和交易日期的一一对应关系。"""
    mapping = data[[td_col, date_col]].drop_duplicates().copy()
    if mapping[td_col].duplicated().any():
        raise ValueError(f"{data_name} 中同一交易日序号对应多个交易日期。")
    if mapping[date_col].duplicated().any():
        raise ValueError(f"{data_name} 中同一交易日期对应多个交易日序号。")
    return mapping.sort_values(td_col).reset_index(drop=True)


def validate_shared_td_calendar(
    base: pd.DataFrame,
    lookup: pd.DataFrame,
) -> None:
    """检查基础面板和DGTW收益表是否使用同一套交易日序号。"""
    base_map = extract_td_date_map(
        base,
        td_col="td",
        date_col="Trddt",
        data_name="回归基础面板",
    ).rename(columns={"Trddt": "_base_Trddt"})
    source_map = extract_td_date_map(
        lookup,
        td_col=SOURCE_TD_COL,
        date_col=SOURCE_DATE_COL,
        data_name="DGTW区间收益表",
    ).rename(columns={SOURCE_DATE_COL: "_source_Trddt"})

    overlap = base_map.merge(
        source_map,
        left_on="td",
        right_on=SOURCE_TD_COL,
        how="inner",
        validate="1:1",
    )
    if overlap.empty:
        raise ValueError("基础面板和DGTW收益表没有重叠的交易日序号。")
    if not overlap["_base_Trddt"].eq(overlap["_source_Trddt"]).all():
        raise ValueError("基础面板和DGTW收益表使用的交易日序号口径不一致。")


def read_base_panel(path: Path) -> pd.DataFrame:
    """读取不含收益因变量的回归基础面板。"""
    if not path.exists():
        raise FileNotFoundError(
            f"未找到回归基础面板：{path}\n"
            "中性基础面板需要先运行新增的单日收益合并脚本生成。"
        )

    available = pd.read_csv(
        path,
        nrows=0,
        encoding="utf-8-sig",
    ).columns.tolist()

    # 优先读取新名称；旧表没有新名称时，读取对应旧名称
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

    # 标准化股票代码，并确认交易日序号为整数
    data["StkcdRecall"] = normalize_stock_code_series(data["StkcdRecall"])
    data["td"] = coerce_required_integer(data["td"], "td")

    if data[["StkcdRecall", "Trddt"]].isna().any().any():
        raise ValueError("回归基础面板存在缺失的公司代码或事件起点日期。")
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
        *RHS_COLS,
    ]
    before = len(data)
    data = data.dropna(subset=invariant_columns).reset_index(drop=True)
    print(
        "基础面板中因固定效应或右端变量缺失删除的观测数："
        f"{before - len(data):,}"
    )
    return data


def prepare_interval_return_lookup(
    source: pd.DataFrame,
    winsor_tail: float = WINSOR_TAIL,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """校验DGTW窗口收益，并按每种窗口长度分别进行全样本缩尾。"""
    required = ["Stkcd", "td", "Trddt", *SOURCE_RETURN_COLS]
    require_columns(source, required, "DGTW区间特征调整收益表")

    if not 0 <= winsor_tail < 0.5:
        raise ValueError("缩尾尾部比例必须位于 [0, 0.5) 区间。")

    out = source[required].copy()
    out["Stkcd"] = normalize_stock_code_series(out["Stkcd"])
    out["td"] = coerce_required_integer(out["td"], "td")
    out["Trddt"] = pd.to_datetime(out["Trddt"], errors="coerce")
    if out[["Stkcd", "Trddt"]].isna().any().any():
        raise ValueError("DGTW收益表存在缺失的股票代码或交易日期。")

    duplicate_count = out.duplicated(["Stkcd", "td"]).sum()
    if duplicate_count:
        raise ValueError(
            "DGTW收益表的股票和交易日序号不唯一："
            f"{duplicate_count:,} 条重复观测。"
        )

    thresholds: dict[str, tuple[float, float]] = {}
    for source_column in SOURCE_RETURN_COLS:
        out[source_column] = pd.to_numeric(
            out[source_column],
            errors="coerce",
        )
        valid = out[source_column].dropna()
        if valid.empty:
            raise ValueError(f"DGTW收益字段 {source_column} 没有有效观测。")

        lower = float(valid.quantile(winsor_tail))
        upper = float(valid.quantile(1 - winsor_tail))
        if not np.isfinite([lower, upper]).all() or lower > upper:
            raise ValueError(f"DGTW收益字段 {source_column} 的缩尾阈值无效。")

        thresholds[source_column] = (lower, upper)
        out[winsor_column(source_column)] = (
            out[source_column]
            .clip(lower=lower, upper=upper)
            .astype("float32")
        )

    out = out.rename(
        columns={
            "Stkcd": "StkcdRecall",
            "td": SOURCE_TD_COL,
            "Trddt": SOURCE_DATE_COL,
        }
    )
    lookup_columns = [
        "StkcdRecall",
        SOURCE_TD_COL,
        SOURCE_DATE_COL,
        *[winsor_column(column) for column in SOURCE_RETURN_COLS],
    ]
    return out[lookup_columns].copy(), thresholds


def read_interval_return_lookup(
    path: Path = DGTW_ADJUSTED_RETURN_PATH,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """只读取非重叠窗口实际需要的DGTW收益字段。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到DGTW特征调整收益表：{path}")

    source = pd.read_csv(
        path,
        usecols=["Stkcd", "td", "Trddt", *SOURCE_RETURN_COLS],
        dtype={
            "Stkcd": "string",
            **{column: "float32" for column in SOURCE_RETURN_COLS},
        },
        parse_dates=["Trddt"],
        encoding="utf-8-sig",
    )
    return prepare_interval_return_lookup(source)


def attach_interval_outcome(
    base: pd.DataFrame,
    lookup: pd.DataFrame,
    spec: WindowSpec,
) -> pd.DataFrame:
    """把一个非重叠窗口的收益精确匹配到事件起点面板。"""
    if spec.start < 0 or spec.end < spec.start:
        raise ValueError(f"事件窗口定义无效：{spec}")

    source_value_column = winsor_column(spec.source_column)
    require_columns(
        lookup,
        [
            "StkcdRecall",
            SOURCE_TD_COL,
            SOURCE_DATE_COL,
            source_value_column,
        ],
        "DGTW区间收益查找表",
    )

    window_lookup = lookup[
        [
            "StkcdRecall",
            SOURCE_TD_COL,
            SOURCE_DATE_COL,
            source_value_column,
        ]
    ].rename(columns={source_value_column: OUTCOME_VALUE_COL})

    aligned = base.copy()
    aligned[TARGET_START_TD_COL] = aligned["td"] + spec.start
    aligned = aligned.merge(
        window_lookup,
        left_on=["StkcdRecall", TARGET_START_TD_COL],
        right_on=["StkcdRecall", SOURCE_TD_COL],
        how="left",
        validate="1:1",
    )

    if len(aligned) != len(base):
        raise ValueError(f"匹配窗口 {spec.label} 后基础面板行数发生变化。")

    matched = aligned[SOURCE_TD_COL].notna()
    if not aligned.loc[matched, SOURCE_TD_COL].eq(
        aligned.loc[matched, TARGET_START_TD_COL]
    ).all():
        raise ValueError(f"窗口 {spec.label} 的目标起点交易日匹配错误。")

    return aligned.drop(columns=[TARGET_START_TD_COL])


def build_common_sample_mask(
    base: pd.DataFrame,
    lookup: pd.DataFrame,
) -> pd.Series:
    """识别七个非重叠窗口均具有完整收益的事件观测。"""
    firm_values = base["StkcdRecall"].astype("string").to_numpy()
    origin_td = base["td"].to_numpy(dtype=np.int64, copy=False)
    common = np.ones(len(base), dtype=bool)
    valid_index_cache: dict[str, pd.MultiIndex] = {}

    for spec in WINDOW_SPECS:
        source_value_column = winsor_column(spec.source_column)
        if source_value_column not in valid_index_cache:
            valid_rows = lookup.loc[
                lookup[source_value_column].notna(),
                ["StkcdRecall", SOURCE_TD_COL],
            ]
            valid_index_cache[source_value_column] = pd.MultiIndex.from_frame(
                valid_rows
            )

        target_index = pd.MultiIndex.from_arrays(
            [firm_values, origin_td + spec.start],
            names=["StkcdRecall", SOURCE_TD_COL],
        )
        common &= target_index.isin(valid_index_cache[source_value_column])
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
        "图2扩展_非重叠窗口特征调整收益_加入日期固定效应_"
        f"{sample_label}_{base_short_label}"
    )
    return RESULT_DIR / f"{stem}结果.csv", FIGURE_DIR / f"{stem}.png"


def expected_row_metadata(
    spec: WindowSpec,
    *,
    sample_label: str,
    base_label: str,
    thresholds: dict[str, tuple[float, float]],
) -> dict[str, object]:
    """返回某一窗口恢复运行和最终校验所需的固定口径。"""
    lower, upper = thresholds[spec.source_column]
    return {
        "窗口序号": spec.window_id,
        "事件窗口": spec.label,
        "相对交易日": spec.label,
        "区间起点": spec.start,
        "区间终点": spec.end,
        "区间交易日数": spec.length,
        "因变量": spec.outcome_label,
        "源收益变量": spec.source_column,
        "分析版本": ANALYSIS_VERSION,
        "被解释变量口径": OUTCOME_DEFINITION,
        "被解释变量来源": OUTCOME_SOURCE_DEFINITION,
        "目标日组合口径": TARGET_PORTFOLIO_DEFINITION,
        "目标日对齐口径": TARGET_ALIGNMENT_DEFINITION,
        "目标日缺失处理": TARGET_MISSING_DEFINITION,
        "样本口径": sample_label,
        "基础面板口径": base_label,
        "缩尾口径": WINSOR_DEFINITION,
        "缩尾下界": lower,
        "缩尾上界": upper,
        "结果单位": "百分点",
        "控制变量": "是",
        "日期固定效应": "是",
        "日期固定效应基准": DATE_FE_BASE_DEFINITION,
        "被提示公司×年度季度固定效应": "是",
        "聚类方式": CLUSTER_DEFINITION,
        "有限样本修正": "是",
    }


def save_results(rows: Iterable[dict], result_path: Path) -> pd.DataFrame:
    """按窗口顺序原子保存结构化结果。"""
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


def metadata_value_matches(actual: object, expected: object) -> bool:
    """比较恢复运行时的字符串、整数和浮点口径。"""
    if isinstance(expected, float):
        actual_numeric = pd.to_numeric(pd.Series([actual]), errors="coerce").iloc[0]
        return bool(
            np.isclose(
                actual_numeric,
                expected,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=False,
            )
        )
    if isinstance(expected, int):
        actual_numeric = pd.to_numeric(pd.Series([actual]), errors="coerce").iloc[0]
        return bool(
            pd.notna(actual_numeric)
            and np.isfinite(actual_numeric)
            and float(actual_numeric).is_integer()
            and int(actual_numeric) == expected
        )
    return str(actual) == str(expected)


def load_completed_rows(
    *,
    resume: bool,
    result_path: Path,
    sample_label: str,
    base_label: str,
    thresholds: dict[str, tuple[float, float]],
) -> list[dict]:
    """读取口径完全一致的已完成窗口，支持中断后继续估计。"""
    if not resume or not result_path.exists():
        return []

    existing = pd.read_csv(result_path, encoding="utf-8-sig")
    if existing.empty:
        return []
    if "窗口序号" not in existing.columns:
        raise ValueError("已有结果缺少“窗口序号”，无法继续估计。")

    numeric_ids = pd.to_numeric(existing["窗口序号"], errors="coerce")
    finite_ids = numeric_ids.map(
        lambda value: bool(pd.notna(value) and np.isfinite(value))
    )
    valid_integer_ids = finite_ids & numeric_ids.map(
        lambda value: bool(
            pd.notna(value)
            and np.isfinite(value)
            and float(value).is_integer()
        )
    )
    if not valid_integer_ids.all():
        raise ValueError("已有结果包含缺失、非有限或非整数的窗口编号。")

    integer_ids = numeric_ids.astype("int64")
    unknown_ids = sorted(set(integer_ids) - set(WINDOW_BY_ID))
    if unknown_ids:
        raise ValueError(f"已有结果包含未知窗口编号：{unknown_ids}")
    duplicated_ids = sorted(integer_ids[integer_ids.duplicated()].unique().tolist())
    if duplicated_ids:
        raise ValueError(f"已有结果包含重复窗口编号：{duplicated_ids}")

    completed: list[dict] = []
    incomplete_ids: list[int] = []
    for row in existing.to_dict("records"):
        window_id = int(row["窗口序号"])
        spec = WINDOW_BY_ID[window_id]
        expected = expected_row_metadata(
            spec,
            sample_label=sample_label,
            base_label=base_label,
            thresholds=thresholds,
        )
        fields_complete = all(pd.notna(row.get(field)) for field in RESULT_COMPLETION_FIELDS)
        metadata_complete = all(
            metadata_value_matches(row.get(column), value)
            for column, value in expected.items()
        )
        if fields_complete and metadata_complete:
            completed.append(row)
        else:
            incomplete_ids.append(spec.window_id)

    if incomplete_ids:
        print(
            "以下已有窗口的统计量或口径不完整，将重新估计："
            f"{sorted(set(incomplete_ids))}"
        )
    return sorted(completed, key=lambda row: int(row["窗口序号"]))


def validate_results(
    results: pd.DataFrame,
    *,
    sample_label: str,
    base_label: str,
    thresholds: dict[str, tuple[float, float]] | None,
) -> None:
    """绘图前校验七个窗口、统计量和口径元数据。"""
    require_columns(results, ["窗口序号"], "非重叠窗口结果")
    raw_ids = pd.to_numeric(results["窗口序号"], errors="coerce")
    valid_ids = raw_ids.map(
        lambda value: bool(
            pd.notna(value)
            and np.isfinite(value)
            and float(value).is_integer()
        )
    )
    if not valid_ids.all():
        raise ValueError("非重叠窗口结果包含无效的窗口编号。")

    expected_ids = [spec.window_id for spec in WINDOW_SPECS]
    if len(results) != len(WINDOW_SPECS):
        raise ValueError("结果未包含全部七个非重叠窗口。")
    if raw_ids.astype("int64").tolist() != expected_ids:
        raise ValueError("非重叠窗口结果不完整或顺序错误。")

    required_numeric = [
        "线索系数",
        "聚类标准误",
        "置信区间下界",
        "置信区间上界",
        "t值",
        "p值",
        "回归样本量",
        "解释变量矩阵条件数",
        "R方",
        "目标区间有效收益观测数",
        "缩尾下界",
        "缩尾上界",
    ]
    require_columns(results, required_numeric, "非重叠窗口结果")
    numeric = results[required_numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("非重叠窗口结果包含缺失或非有限统计量。")
    if not numeric["聚类标准误"].gt(0).all():
        raise ValueError("非重叠窗口结果包含非正标准误。")
    if not numeric["p值"].between(0, 1).all():
        raise ValueError("非重叠窗口结果包含无效p值。")

    interval_valid = (
        numeric["置信区间下界"].le(numeric["线索系数"])
        & numeric["线索系数"].le(numeric["置信区间上界"])
    )
    if not interval_valid.all():
        raise ValueError("非重叠窗口结果包含无效置信区间。")

    for spec, row in zip(WINDOW_SPECS, results.to_dict("records")):
        if thresholds is None:
            row_thresholds = {
                spec.source_column: (
                    float(row["缩尾下界"]),
                    float(row["缩尾上界"]),
                )
            }
        else:
            row_thresholds = thresholds

        expected = expected_row_metadata(
            spec,
            sample_label=sample_label,
            base_label=base_label,
            thresholds=row_thresholds,
        )
        for column, value in expected.items():
            if column not in row:
                raise ValueError(f"结果缺少口径字段：{column}")
            if not metadata_value_matches(row[column], value):
                raise ValueError(
                    f"窗口 {spec.label} 的“{column}”口径与当前脚本不一致。"
                )

    # 四个十日窗口共用同一个全源表变量，缩尾阈值必须完全一致。
    t9_rows = results.loc[results["源收益变量"].eq("chAdj_ret_t_t9")]
    if len(t9_rows) != 4:
        raise ValueError("十日非重叠窗口与源收益变量的映射不完整。")
    for column in ["缩尾下界", "缩尾上界"]:
        values = pd.to_numeric(t9_rows[column], errors="coerce").to_numpy()
        if not np.isclose(values, values[0], rtol=1e-10, atol=1e-12).all():
            raise ValueError(f"四个十日窗口的{column}不一致。")


def draw_interval_figure(
    results: pd.DataFrame,
    output_path: Path,
    sample_label: str,
) -> None:
    """绘制七个非重叠窗口系数及99%、95%、90%置信区间。"""
    plot_data = results.sort_values("窗口序号").reset_index(drop=True)

    font_path = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False

    x_positions = np.arange(1, len(plot_data) + 1)
    figure, axis = plt.subplots(figsize=(10.5, 5.4))

    for interval_spec in ERROR_BAR_SPECS:
        interval_radius = (
            interval_spec["critical_value"] * plot_data["聚类标准误"]
        )
        axis.vlines(
            x_positions,
            plot_data["线索系数"] - interval_radius,
            plot_data["线索系数"] + interval_radius,
            colors=interval_spec["color"],
            linewidth=interval_spec["line_width"],
            label=interval_spec["label"],
            zorder=interval_spec["zorder"],
        )

    # 按10%显著性水平判断：90%置信区间不包含0
    significant = (
        plot_data["线索系数"].abs()
        > 1.6448536269514722 * plot_data["聚类标准误"]
    )
    point_colors = np.where(significant, "#d62728", "#1a1a1a")

    axis.scatter(
        x_positions,
        plot_data["线索系数"],
        color=point_colors,
        s=24,
        zorder=4,
    )

    # 添加红点图例
    axis.scatter(
        [],
        [],
        color="#d62728",
        s=24,
        label="红点：10%水平显著",
    )
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_title(
        "公告后非重叠窗口特征调整收益"
        f"（控制事件起点交易日期固定效应，{sample_label}）"
    )
    axis.set_xlabel("非重叠事件窗口")
    axis.set_ylabel("提示变量系数（百分点）")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(plot_data["事件窗口"].tolist())
    axis.legend(
        loc="best",
        frameon=False,
        fontsize=9,
        handlelength=1.4,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")
    axis.margins(x=0.06)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """读取样本口径、续算和仅绘图参数。"""
    parser = argparse.ArgumentParser(
        description="估计公告后七个非重叠窗口的DGTW特征调整收益。"
    )
    parser.add_argument(
        "--sample-mode",
        choices=sorted(SAMPLE_MODE_LABELS),
        default="available",
        help=(
            "available=每个窗口使用该窗口可用样本；"
            "common7=七个窗口均有效的共同样本。"
        ),
    )
    parser.add_argument(
        "--base-source",
        choices=sorted(BASE_SOURCE_LABELS),
        default="neutral",
        help=(
            "neutral=旧累计因变量筛选前的中性基础面板；"
            "legacy=旧累计收益最终回归样本，仅用于基础样本可比性检验。"
        ),
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--resume",
        action="store_true",
        help="从相同口径的已有完整窗口结果继续估计。",
    )
    run_mode.add_argument(
        "--plot-only",
        action="store_true",
        help="不运行回归，只读取当前口径已有结果并重新绘图。",
    )
    return parser.parse_args()


def main() -> None:
    """串联基础样本、非重叠区间收益、回归和独立图形输出。"""
    validate_window_specs()
    args = parse_args()
    sample_label = SAMPLE_MODE_LABELS[args.sample_mode]
    base_label = BASE_SOURCE_LABELS[args.base_source]
    result_path, output_path = resolve_output_paths(
        args.sample_mode,
        args.base_source,
    )

    if args.plot_only:
        if not result_path.exists():
            raise FileNotFoundError(f"仅绘图模式未找到已有结果：{result_path}")
        results = (
            pd.read_csv(result_path, encoding="utf-8-sig")
            .sort_values("窗口序号")
            .reset_index(drop=True)
        )
        validate_results(
            results,
            sample_label=sample_label,
            base_label=base_label,
            thresholds=None,
        )
        draw_interval_figure(results, output_path, sample_label)
        print(f"非重叠窗口图已重新保存：{output_path}")
        return

    base_path = resolve_base_path(args.base_source)
    print(f"读取回归基础面板：{base_path}")
    base = read_base_panel(base_path)

    print(f"读取DGTW特征调整收益表：{DGTW_ADJUSTED_RETURN_PATH}")
    lookup, thresholds = read_interval_return_lookup()
    validate_shared_td_calendar(base, lookup)
    for source_column, (lower, upper) in thresholds.items():
        print(
            f"{source_column} 全源表缩尾阈值："
            f"下界={lower:.10f}，上界={upper:.10f}"
        )

    if args.sample_mode == "common7":
        print("正在识别七个非重叠窗口均有效的共同样本……")
        common_mask = build_common_sample_mask(base, lookup)
        removed = int((~common_mask).sum())
        base = base.loc[common_mask].reset_index(drop=True)
        print(f"七区间共同样本筛选删除观测数：{removed:,}")
        if base.empty:
            raise ValueError("七个非重叠窗口的完整共同样本为空。")

    existing_rows = load_completed_rows(
        resume=args.resume,
        result_path=result_path,
        sample_label=sample_label,
        base_label=base_label,
        thresholds=thresholds,
    )
    completed_ids = {int(row["窗口序号"]) for row in existing_rows}
    pending_specs = [
        spec for spec in WINDOW_SPECS if spec.window_id not in completed_ids
    ]
    rows = existing_rows.copy()

    for position, spec in enumerate(pending_specs, start=1):
        print(
            "正在运行非重叠窗口收益回归："
            f"{spec.label}（待估计 {position}/{len(pending_specs)}）"
        )
        aligned = attach_interval_outcome(base, lookup, spec)
        valid_outcome = aligned[OUTCOME_VALUE_COL].notna()
        valid_count = int(valid_outcome.sum())
        if valid_count == 0:
            raise ValueError(f"窗口 {spec.label} 没有可用的特征调整收益率。")

        valid_dates = aligned.loc[valid_outcome, SOURCE_DATE_COL]
        row = run_date_fe_regression(
            aligned,
            horizon=spec.end,
            outcome=OUTCOME_VALUE_COL,
            outcome_label=spec.outcome_label,
            rhs_cols=RHS_COLS,
            cue_col=CUE_COL,
        )
        row.update(
            expected_row_metadata(
                spec,
                sample_label=sample_label,
                base_label=base_label,
                thresholds=thresholds,
            )
        )
        row.update(
            {
                "目标区间有效收益观测数": valid_count,
                "目标区间起点日期最小值": valid_dates.min().strftime("%Y-%m-%d"),
                "目标区间起点日期最大值": valid_dates.max().strftime("%Y-%m-%d"),
                "按区间长度折算系数": row["线索系数"] / spec.length,
                "按区间长度折算标准误": row["聚类标准误"] / spec.length,
            }
        )

        rows = [
            existing
            for existing in rows
            if int(existing["窗口序号"]) != spec.window_id
        ]
        rows.append(row)
        results = save_results(rows, result_path)
        print(
            f"{spec.label} 完成："
            f"系数={row['线索系数']:.6f}，"
            f"标准误={row['聚类标准误']:.6f}，"
            f"p值={row['p值']:.6f}，"
            f"样本量={row['回归样本量']:,}"
        )

        del aligned, valid_outcome, valid_dates
        gc.collect()

    if not pending_specs:
        print("已有结果包含全部七个非重叠窗口，无需重新估计。")
        results = save_results(existing_rows, result_path)

    validate_results(
        results,
        sample_label=sample_label,
        base_label=base_label,
        thresholds=thresholds,
    )
    draw_interval_figure(results, output_path, sample_label)

    print("\n非重叠窗口核心结果：")
    print(
        results[
            [
                "事件窗口",
                "区间交易日数",
                "线索系数",
                "聚类标准误",
                "p值",
                "显著性",
                "回归样本量",
            ]
        ].to_string(index=False)
    )
    print(f"非重叠窗口回归结果已保存：{result_path}")
    print(f"非重叠窗口图已保存：{output_path}")


if __name__ == "__main__":
    main()
