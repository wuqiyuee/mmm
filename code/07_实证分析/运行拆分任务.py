"""显示全部实证任务的脚本与结果位置，或启动指定任务。

直接运行只显示路径，不读取面板、不估计回归、不生成图表。
显式提供 --task（任务名）后执行所选入口，后续参数交给该入口解析。
原综合脚本已归档保留，不作为正式任务入口。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR / "90_公共模块"))
from 任务入口 import find_project_root
from 实证任务目录 import CATALOG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(CATALOG), help="显式选择需要执行的任务。")
    args, remaining = parser.parse_known_args()
    root = find_project_root(__file__)
    if args.task is None:
        if remaining:
            parser.error("执行参数前请先提供 --task（任务名）。")
        for name, entry in CATALOG.items():
            print(f"任务：{name}")
            print(entry["说明"])
            print(f"脚本：{ANALYSIS_DIR / entry['脚本']}")
            print(f"结果：{root / '图表结果' / entry['输出']}")
            print()
        print(f"原综合脚本：{ANALYSIS_DIR / '99_历史与待核对/原综合脚本'}")
        return
    entry = CATALOG[args.task]
    script = ANALYSIS_DIR / entry["脚本"]
    if not script.is_file():
        raise FileNotFoundError(f"任务入口不存在：{script}")
    print(f"任务：{args.task}", flush=True)
    print(f"结果目录：{root / '图表结果' / entry['输出']}", flush=True)
    subprocess.run(
        [sys.executable, str(script), *entry["参数"], *remaining],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
