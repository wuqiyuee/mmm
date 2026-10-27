"""只读取本任务选定口径结果，绘制模式公司表2及主样本便携文档。

默认读取并写入“图表结果/01_主线图表/表2_收益回归”。
可显式传入结果读取目录；图片仍统一保存到本任务结果目录。
本模块不重新估计回归，也不依赖其他表的结果。
旧口径数值须先重新估计；完整样本旧图不删除，也不纳入本次输出。
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
    SPEC_ORDER,
    WINDOW_ORDER,
    draw_table2_compact,
    draw_table2_full_two_pages,
    read_result,
)
from 主回归口径 import CUE_COLUMN, SPEC_ID


OUTPUT_DIR = PROJECT_ROOT / "图表结果" / "01_主线图表" / "表2_收益回归"
TABLE2_NOTES = [
    "注：历史沪深300提示；剔除前365天个人分析师共同覆盖关系，保留提示方金融和房地产公司。",
    "新闻：数行者2005—2009年报刊、2010—2025年网络；允许合格填充日起点，保留全期年份。",
    "主规格为（6）：27项控制、公司×年度季度和日期固定效应；括号内为公司与日期双向聚类标准误。",
    "R²采用原回归库总体口径；空白表示未纳入、被吸收或聚类方差非正，详见系数明细。",
    "星号基于双侧检验原始概率值，未作多重检验调整；*、**、***分别表示10%、5%、1%显著水平。",
]


def validate_current_results(summary, details) -> None:
    """拒绝旧口径、混合样本及不能完整对应十二个模型的数值结果。"""
    rerun_hint = "请先运行本任务新版“估计回归.py”生成选定口径的数值结果，再绘制表2。"
    model_key = ["样本", "收益窗口", "规格"]
    expected_models = {
        ("pattern", window, specification)
        for window in WINDOW_ORDER
        for specification in SPEC_ORDER
    }
    for frame, name, key in (
        (summary, "模型汇总", model_key),
        (details, "系数明细", model_key + ["变量"]),
    ):
        if frame.empty:
            raise ValueError(f"表2{name}为空。{rerun_hint}")
        for column, expected in (("口径标识", SPEC_ID), ("提示变量", CUE_COLUMN)):
            if column not in frame.columns or not frame[column].eq(expected).all():
                raise ValueError(f"表2{name}的{column}不是当前选定口径。{rerun_hint}")
        if not frame["样本"].eq("pattern").all():
            raise ValueError(f"表2{name}必须仅包含模式公司主样本。{rerun_hint}")
        if frame[key].isna().any().any() or frame.duplicated(key).any():
            raise ValueError(f"表2{name}的唯一键缺失或重复。{rerun_hint}")
        actual_models = set(frame[model_key].itertuples(index=False, name=None))
        if actual_models != expected_models:
            raise ValueError(f"表2{name}必须完整对应两种窗口、六种规格的十二个模型。{rerun_hint}")

    cue_models = set(
        details.loc[details["变量"].eq(CUE_COLUMN), model_key]
        .itertuples(index=False, name=None)
    )
    if cue_models != expected_models:
        raise ValueError(f"表2系数明细必须逐模型包含当前沪深300提示变量。{rerun_hint}")


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
    validate_current_results(summary, details)
    compact = OUTPUT_DIR / "table2_收益回归_紧凑版.png"
    pattern_first = OUTPUT_DIR / "table2_收益回归_完整系数版_模式公司主样本.png"
    pattern_second = OUTPUT_DIR / "table2_收益回归_完整系数版_模式公司主样本_续页.png"
    pattern_pdf = OUTPUT_DIR / "table2_收益回归_完整系数版_模式公司主样本.pdf"
    draw_table2_compact(summary, compact, notes=TABLE2_NOTES)
    draw_table2_full_two_pages(
        summary,
        details,
        sample="pattern",
        first_page_output=pattern_first,
        second_page_output=pattern_second,
        pdf_output=pattern_pdf,
        notes=TABLE2_NOTES,
    )
    outputs = [compact, pattern_first, pattern_second, pattern_pdf]
    print("表2绘制完成：")
    for output in outputs:
        print(f"- {output}")
    return outputs


if __name__ == "__main__":
    main()
