"""本任务的独立入口；估计、绘图和结果位置集中在本任务中。

直接运行执行本任务；使用 --stage 查看（查看阶段）仅显示路径。
使用 --stage 估计（估计阶段）或 --stage 绘图（绘图阶段）可分开执行。
表1统计与绘图同步完成；日期固定效应附表支持独立估计与绘图。
两份原综合脚本保留不变；本入口不会执行其他表格任务。
"""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    source = Path(__file__).resolve()
    for parent in source.parents:
        if (
            (parent / "AGENTS.md").is_file()
            and (parent / "code").is_dir()
            and (parent / "中间结果").is_dir()
        ):
            public_dir = parent / "code" / "07_实证分析" / "90_公共模块"
            sys.path.insert(0, str(public_dir))# 把公共模块目录加入搜索路径
            # 这样 from 任务入口 import run_task 才能找到
            break# 如果找到了符合条件的项目根目录，就停止继续向上搜索。
    else:
        raise FileNotFoundError("未找到包含项目说明、代码和中间结果的项目根目录。")
    from 任务入口 import run_task

    return run_task(source)# 它把当前入口文件路径传给 run_task()


if __name__ == "__main__":
    main()

