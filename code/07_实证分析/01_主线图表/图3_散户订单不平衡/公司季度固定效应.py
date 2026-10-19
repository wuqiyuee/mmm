"""独立运行图3的公司季度固定效应口径，不触发另一张图。

样本、窗口、缩尾和估计保持原合并脚本口径；仅重画时读取本任务目录已有结果。
本文件的主函数返回生成文件路径，导入时不读取研究数据或写文件。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


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
COMMON_DIR = ROOT / "code" / "07_实证分析" / "90_公共模块"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from 图2图3_公司季度固定效应 import main as run_figure


def main(argv=None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", default=None, help="只从本任务已有完整数值结果重画。")
    args = parser.parse_args(argv)
    return run_figure(3, plot_only=args.plot_only)


if __name__ == "__main__":
    main()
