"""只读取本任务三种指标的日期固定效应结果，绘制附表A.2至A.4。

默认读取和写入“图表结果/01_主线图表/附表A2-A4_日期固定效应”。
估计由同一任务目录内的“估计回归.py”完成。
可显式传入其他数值读取目录；本模块不会自动搜索或回退。
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
    draw_order_imbalance_full,
    read_appendix_order_imbalance_results,
)


OUTPUT_DIR = PROJECT_ROOT / "图表结果" / "01_主线图表" / "附表A2-A4_日期固定效应"
DEFAULT_RESULT_DIR = OUTPUT_DIR
METRIC_SPECS = [
    ("A.2", "A2", "OIBnum", "买卖次数"),
    ("A.3", "A3", "OIBamount", "成交金额"),
    ("A.4", "A4", "OIBorder", "成交笔数"),
]


def main(result_dir: Path | None = None) -> list[Path]:
    """只读取三种日期固定效应指标的汇总及系数文件，返回三个图片路径。"""
    input_dir = Path(result_dir) if result_dir is not None else DEFAULT_RESULT_DIR
    prepared = []
    for table_number, table_code, prefix, measure_label in METRIC_SPECS:
        stem = f"附表{table_code}_{prefix}_{measure_label}订单失衡_加入日期固定效应"
        summary, details = read_appendix_order_imbalance_results(
            input_dir / f"{stem}结果.csv",
            input_dir / f"{stem}系数明细.csv",
            prefix=prefix,
            measure_label=measure_label,
            date_fixed_effects="是",
            table_number=table_number,
        )
        output = OUTPUT_DIR / (
            f"appendix_{table_code}_{prefix}_{measure_label}订单失衡"
            "_完整系数版_加入日期固定效应.png"
        )
        prepared.append((table_number, measure_label, summary, details, output))

    outputs = []
    for table_number, measure_label, summary, details, output in prepared:
        draw_order_imbalance_full(
            summary,
            details,
            table_number=f"附表 {table_number}（本项目扩展：加入日期固定效应）",
            title=f"线索事件与小单{measure_label}订单失衡",
            output_path=output,
            label_right=0.34,
            additional_notes=[
                "本表为本项目扩展结果，同时吸收被提示公司×年度季度和交易日期固定效应。",
            ],
        )
        outputs.append(output)
    print("附表A.2至A.4日期固定效应结果绘制完成：")
    for output in outputs:
        print(f"- {output}")
    return outputs


if __name__ == "__main__":
    main()
