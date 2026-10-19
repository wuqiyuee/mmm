"""依次调用已拆分的主线估计任务，不包含计算逻辑副本。

默认仅显示任务清单；显式提供 --run（执行）后按顺序估计。
包括表1统计、表2、表3基准、表4两日窗口与附表A2基准。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "90_公共模块"))
from 任务入口 import _load_module, task_paths


TASK_NAMES = (
    "表1_描述性统计",
    "表2_收益回归",
    "表3_散户订单失衡",
    "表4_规模异质性",
    "附表A2_买卖次数订单失衡",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="实际执行各任务估计；默认只显示路径。")
    args = parser.parse_args()
    for name in TASK_NAMES:
        folder, output = task_paths(name, __file__)
        print(f"{name}：{folder / '估计回归.py'}")
        print(f"输出：{output}")
        if args.run:
            _load_module(folder / "估计回归.py").main()


if __name__ == "__main__":
    main()
