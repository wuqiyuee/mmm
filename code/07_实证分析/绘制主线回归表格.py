"""只读取逐表数值结果，依次生成主线表格与日期固定效应附表。

这是原综合绘表入口的轻量调度器，各表的绘图代码位于各自任务目录。
不读取日度面板，不重新估计。缺少某项数值结果时立即报告对应文件。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "90_公共模块"))
from 任务入口 import _load_module, task_paths


DRAW_TASKS = (
    ("表2_收益回归", {}),
    ("表3_散户订单失衡", {}),
    ("表3_散户订单失衡", {"date_fixed_effects": True}),
    ("表4_规模异质性", {}),
    ("表4_规模异质性", {"date_fixed_effects": True}),
    ("附表A2_买卖次数订单失衡", {}),
    ("附表A2-A4_日期固定效应", {}),
)


def main() -> list[Path]:
    written = []
    for name, options in DRAW_TASKS:
        folder, output = task_paths(name, __file__)
        module = _load_module(folder / "绘制表格.py")
        written.extend(module.main(result_dir=output, **options))
    return written


if __name__ == "__main__":
    main()
