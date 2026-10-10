"""独立运行表5的“公司×年度季度固定效应 + 交易日期固定效应”版本。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


FORMAL_TABLE5_PATH = (
    Path(__file__).resolve().parents[1] / "表5_心理机制.py"
)


def load_formal_table5_module():
    """明确加载上一级的正式表5脚本，避免与当前同名文件递归导入。"""
    spec = importlib.util.spec_from_file_location(
        "formal_table5_regression",
        FORMAL_TABLE5_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载正式表5脚本：{FORMAL_TABLE5_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """保留基准版文件不变，并把日期固定效应版写入独立文件。"""
    table5 = load_formal_table5_module()
    table5.main(["--date-fixed-effects", *sys.argv[1:]])


if __name__ == "__main__":
    main()
