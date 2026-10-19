"""图2_收益动态的运行入口，先明确选择固定效应口径。

默认加入日期固定效应。两个口径的图片、数值与进度在同一任务目录按文件名区分。
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        choices=("公司季度固定效应", "加入日期固定效应"),
        default="加入日期固定效应",
        help="本次固定效应口径。",
    )
    args, remaining = parser.parse_known_args()
    entry = Path(__file__).resolve().with_name(args.spec + ".py")
    subprocess.run([sys.executable, str(entry), *remaining], check=True)


if __name__ == "__main__":
    main()
