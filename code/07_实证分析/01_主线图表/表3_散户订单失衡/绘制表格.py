"""只读取表3结果并绘图；默认基准口径，可显式选择日期固定效应口径。

默认读取并写入“图表结果/01_主线图表/表3_散户订单失衡”。
日期固定效应结果需由对应估计脚本另行生成；本模块不会自动搜索旧目录。
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

from 回归表格绘制 import draw_order_imbalance_full, read_result


OUTPUT_DIR = PROJECT_ROOT / "图表结果" / "01_主线图表" / "表3_散户订单失衡"


def main(
    date_fixed_effects: bool = False,
    result_dir: Path | None = None,
) -> list[Path]:
    """只绘制选定口径；结果读取目录可显式指定，图片始终写入本任务目录。"""
    input_dir = Path(result_dir) if result_dir is not None else OUTPUT_DIR
    if date_fixed_effects:
        summary_name = "表3_散户订单失衡_加入日期固定效应结果.csv"
        detail_name = "表3_散户订单失衡_加入日期固定效应系数明细.csv"
        image_name = "table3_散户订单失衡_完整系数版_加入日期固定效应.png"
        table_number = "表 3（加入日期固定效应）"
    else:
        summary_name = "表3_散户订单失衡结果.csv"
        detail_name = "表3_系数明细.csv"
        image_name = "table3_散户订单失衡_完整系数版.png"
        table_number = "表 3"

    summary = read_result(
        input_dir / summary_name,
        [
            "衡量口径", "订单失衡窗口", "散户识别方法", "线索系数", "聚类标准误",
            "p值", "控制变量", "日期固定效应", "被提示公司×年度季度固定效应",
            "回归样本量", "R方",
        ],
    )
    detail_columns = ["订单失衡窗口", "变量", "变量顺序", "系数", "聚类标准误", "p值"]
    if date_fixed_effects:
        detail_columns.append("日期固定效应")
    details = read_result(input_dir / detail_name, detail_columns)
    expected_date_fe = "是" if date_fixed_effects else "否"
    for frame, source_name in [(summary, summary_name), (details, detail_name)]:
        if "日期固定效应" in frame.columns:
            values = frame["日期固定效应"].astype("string").str.strip()
            if values.isna().any() or not values.eq(expected_date_fe).all():
                raise ValueError(
                    f"{source_name}的日期固定效应应全部为“{expected_date_fe}”。"
                )

    output = OUTPUT_DIR / image_name
    draw_order_imbalance_full(
        summary,
        details,
        table_number=table_number,
        title="线索事件与散户订单失衡",
        output_path=output,
        label_right=0.34,
    )
    print(f"表3绘制完成：{output}")
    return [output]


if __name__ == "__main__":
    main()
