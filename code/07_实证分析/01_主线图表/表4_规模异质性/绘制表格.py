"""只读取表4规模异质性结果，保留当前两日收益窗口的扩展口径。

默认读取并写入“图表结果/01_主线图表/表4_规模异质性”。
本表使用事件当日至下一交易日收益，并非原文较长窗口的严格复现。
日期固定效应结果需另行生成，本模块不会自动搜索旧目录。
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_project_root() -> Path:
    for ancestor in Path(__file__).resolve().parents:
        if (
            (ancestor / "AGENTS.md").is_file()
            and (ancestor / "code").is_dir()
            and (ancestor / "中间结果").is_dir()
        ):
            return ancestor
    raise FileNotFoundError("找不到包含 AGENTS.md、code 和中间结果的项目根目录。")


PROJECT_ROOT = _find_project_root()
COMMON_DIR = PROJECT_ROOT / "code" / "07_实证分析" / "90_公共模块"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from 回归表格绘制 import draw_table4_extension, read_result


OUTPUT_DIR = PROJECT_ROOT / "图表结果" / "01_主线图表" / "表4_规模异质性"


def main(
    date_fixed_effects: bool = False,
    result_dir: Path | None = None,
) -> list[Path]:
    """只绘制本任务的选定固定效应口径，返回图片路径。"""
    input_dir = Path(result_dir) if result_dir is not None else OUTPUT_DIR
    if date_fixed_effects:
        result_name = "表4_规模异质性_加入日期固定效应结果.csv"
        image_name = "table4_规模异质性_t_tp1扩展分析_加入日期固定效应.png"
    else:
        result_name = "表4_规模异质性结果.csv"
        image_name = "table4_规模异质性_t_tp1扩展分析.png"
    columns = [
        "规格", "样本", "因变量", "控制变量", "被提示公司×年度季度固定效应",
        "回归样本量", "R方", "变量", "变量顺序", "系数（百分点）",
        "聚类标准误（百分点）", "p值",
    ]
    if date_fixed_effects:
        columns.append("日期固定效应")
    results = read_result(input_dir / result_name, columns)
    output = OUTPUT_DIR / image_name
    draw_table4_extension(
        results,
        output,
        show_date_fixed_effects=date_fixed_effects,
    )
    print(f"表4绘制完成：{output}")
    return [output]


if __name__ == "__main__":
    main()
