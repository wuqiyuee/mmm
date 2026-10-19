"""只读取本任务结果，绘制表2紧凑版、两类样本分页图片及主样本便携文档。

默认读取并写入“图表结果/01_主线图表/表2_收益回归”。
可显式传入结果读取目录；图片仍统一保存到本任务结果目录。
本模块不重新估计回归，也不依赖其他表的结果。
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

from 回归表格绘制 import (
    draw_table2_compact,
    draw_table2_full_two_pages,
    read_result,
)


OUTPUT_DIR = PROJECT_ROOT / "图表结果" / "01_主线图表" / "表2_收益回归"


def main(result_dir: Path | None = None) -> list[Path]:
    """从本任务或显式指定目录读取表2数值结果，返回全部生成文件路径。"""
    input_dir = Path(result_dir) if result_dir is not None else OUTPUT_DIR
    summary = read_result(
        input_dir / "表2_模型汇总.csv",
        [
            "样本", "样本说明", "收益窗口", "规格", "线索系数（百分点）",
            "聚类标准误（百分点）", "p值", "回归样本量", "R方",
        ],
    )
    details = read_result(
        input_dir / "表2_系数明细.csv",
        [
            "样本", "收益窗口", "规格", "变量", "变量顺序",
            "系数（百分点）", "聚类标准误（百分点）", "p值",
        ],
    )
    compact = OUTPUT_DIR / "table2_收益回归_紧凑版.png"
    pattern_first = OUTPUT_DIR / "table2_收益回归_完整系数版_模式公司主样本.png"
    pattern_second = OUTPUT_DIR / "table2_收益回归_完整系数版_模式公司主样本_续页.png"
    pattern_pdf = OUTPUT_DIR / "table2_收益回归_完整系数版_模式公司主样本.pdf"
    full_first = OUTPUT_DIR / "table2_收益回归_完整系数版_完整样本.png"
    full_second = OUTPUT_DIR / "table2_收益回归_完整系数版_完整样本_续页.png"

    draw_table2_compact(summary, compact)
    draw_table2_full_two_pages(
        summary,
        details,
        sample="pattern",
        first_page_output=pattern_first,
        second_page_output=pattern_second,
        pdf_output=pattern_pdf,
    )
    draw_table2_full_two_pages(
        summary,
        details,
        sample="full",
        first_page_output=full_first,
        second_page_output=full_second,
        pdf_output=None,
    )
    outputs = [compact, pattern_first, pattern_second, pattern_pdf, full_first, full_second]
    print("表2绘制完成：")
    for output in outputs:
        print(f"- {output}")
    return outputs


if __name__ == "__main__":
    main()
