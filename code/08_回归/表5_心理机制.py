"""表5心理机制：规格映射与输入表头校验。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"


@dataclass(frozen=True)
class OutcomeSpec:
    panel: str
    column: str
    label: str


@dataclass(frozen=True)
class MechanismSpec:
    order: int
    key: str
    label: str
    first_90: str
    first_5090: str
    second_90: str | None
    second_5090: str | None
    comparison_mode: str
    source_note: str


OUTCOME_SPECS = (
    OutcomeSpec("A", "chAdjreturn_t_tp1_Recall_w", "[t, t+1]"),
    OutcomeSpec("B", "chAdjreturn_t_tp10_Recall_w", "[t, t+10]"),
)

OWN_EA_COLS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)
DISCLOSURE_COLS = [
    "ownMajorDisclosure",
    "ownPerformanceForecast",
]
MEDIA_CONTROL_COLS = [
    "positiveNews",
    "negativeNews",
    "logIncreaseAEV",
    "logDecreaseAEV",
]

_BASELINE_90 = "quarterLag1size90plus"
_BASELINE_5090 = "quarterLag1size50_90"
_INTERFERENCE_FIELDS = (
    "lowInterferenceEWLag1S90p",
    "lowInterferenceEWLag1S5090",
    "highInterferenceEWLag1S90p",
    "highInterferenceEWLag1S5090",
)


def build_mechanism_specs(sue_id: int) -> tuple[MechanismSpec, ...]:
    """返回指定 SUE（盈余惊喜）口径的九类表5心理机制。"""
    if type(sue_id) is not int or sue_id not in (1, 2, 3):
        raise ValueError("SUE口径只能是1、2或3。")

    shared_specs = (
        MechanismSpec(
            1,
            "high_total_volume",
            "异常总成交量",
            "lag1hiavolsize90plus",
            "lag1hiavolsize50_90",
            None,
            None,
            "strong_vs_rest",
            "共同高异常总成交量强联系提示，与同规模其余提示比较。",
        ),
        MechanismSpec(
            2,
            "high_small_trade_volume",
            "异常散户成交量（A股小单替代）",
            "lag1hiavolSsize90plus",
            "lag1hiavolSsize50_90",
            None,
            None,
            "strong_vs_rest",
            "A股小单异常成交量为原文零售交易量口径的中国替代。",
        ),
        MechanismSpec(
            3,
            "high_media",
            "媒体覆盖",
            "lag1hiAEVsize90plus",
            "lag1hiAEVsize50_90",
            None,
            None,
            "strong_vs_rest",
            "共同高媒体关注强联系提示，与同规模其余提示比较。",
        ),
        MechanismSpec(
            4,
            "top_mover",
            "价格异动",
            "lag1topMoversize90plus",
            "lag1topMoversize50_90",
            None,
            None,
            "strong_vs_rest",
            "共同TopMover强联系提示，与同规模其余提示比较。",
        ),
    )
    sue_specs = (
        MechanismSpec(
            5,
            "positive_sue",
            "正向盈余惊喜",
            f"lag1sue{sue_id}possize90plus",
            f"lag1sue{sue_id}possize50_90",
            None,
            None,
            "strong_vs_rest",
            f"共同正向SUE{sue_id}强联系提示，与同规模其余提示比较。",
        ),
        MechanismSpec(
            6,
            "negative_sue",
            "负向盈余惊喜",
            f"lag1sue{sue_id}negsize90plus",
            f"lag1sue{sue_id}negsize50_90",
            None,
            None,
            "strong_vs_rest",
            f"共同负向SUE{sue_id}强联系提示，与同规模其余提示比较。",
        ),
        MechanismSpec(
            7,
            "extreme_positive_sue",
            "极端正向盈余惊喜",
            f"lag1hi_sue{sue_id}_possize90plus",
            f"lag1hi_sue{sue_id}_possize50_90",
            None,
            None,
            "strong_vs_rest",
            f"共同极端正向SUE{sue_id}强联系提示，与同规模其余提示比较。",
        ),
        MechanismSpec(
            8,
            "extreme_negative_sue",
            "极端负向盈余惊喜",
            f"lag1hi_sue{sue_id}_negsize90plus",
            f"lag1hi_sue{sue_id}_negsize50_90",
            None,
            None,
            "strong_vs_rest",
            f"共同极端负向SUE{sue_id}强联系提示，与同规模其余提示比较。",
        ),
    )
    interference_spec = MechanismSpec(
        9,
        "interference",
        "干扰程度",
        _INTERFERENCE_FIELDS[0],
        _INTERFERENCE_FIELDS[1],
        _INTERFERENCE_FIELDS[2],
        _INTERFERENCE_FIELDS[3],
        "low_vs_high",
        "按年份、等权共同公告公司数量二分的低干扰与高干扰提示比较。",
    )
    return shared_specs + sue_specs + (interference_spec,)


def required_input_columns() -> list[str]:
    """返回表5全部模型所需列，供表头检查和后续紧凑读取共同使用。"""
    columns = [
        "StkcdRecall",
        "Trddt",
        "year",
        "yearQuarter",
        *(outcome.column for outcome in OUTCOME_SPECS),
        *OWN_EA_COLS,
        *DISCLOSURE_COLS,
        *MEDIA_CONTROL_COLS,
        _BASELINE_90,
        _BASELINE_5090,
    ]
    for sue_id in (1, 2, 3):
        for spec in build_mechanism_specs(sue_id):
            columns.extend((spec.first_90, spec.first_5090))
            if spec.second_90 is not None:
                columns.append(spec.second_90)
            if spec.second_5090 is not None:
                columns.append(spec.second_5090)
    return list(dict.fromkeys(columns))


def read_and_validate_header(input_path: str | Path) -> tuple[str, ...]:
    """仅读取 CSV（逗号分隔值文件）首行，并一次报告全部缺失必需字段。"""
    path = Path(input_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])

    available = set(header)
    missing = [column for column in required_input_columns() if column not in available]
    if missing:
        raise ValueError("输入文件表头缺少以下必需字段：" + "、".join(missing))
    return tuple(header)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析本阶段可用的表头检查命令行参数。"""
    parser = argparse.ArgumentParser(description="表5心理机制规格与输入表头校验")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--check-only", action="store_true", help="仅校验输入CSV表头")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """运行只读表头校验。"""
    args = parse_args(argv)
    if args.check_only:
        read_and_validate_header(args.input)
        print(f"表头校验通过：共需{len(required_input_columns())}个字段。")
        return
    raise SystemExit("当前阶段仅支持 --check-only。")


if __name__ == "__main__":
    main()
