"""只读取表3结果并绘图；默认基准口径，可显式选择日期固定效应口径。

默认读取并写入“图表结果/01_主线图表/表3_散户订单失衡”。
可用 --panel（输入面板）、--result-dir（结果目录）或 --summary（结果汇总文件）选择结果。
图片保存在选定结果所在目录；日期固定效应结果需由对应估计脚本另行生成。
"""

from __future__ import annotations

import argparse
import json
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
    date_fixed_effects: bool | None = None,
    result_dir: Path | None = None,
    *,
    panel_path: Path | None = None,
    summary_path: Path | None = None,
) -> list[Path]:
    """只读取所选数值结果，按配套口径说明绘图并保存至结果同目录。"""
    if result_dir is not None and summary_path is not None:
        raise ValueError("请只指定结果目录或结果汇总文件中的一种。")
    panel_info = None
    selected_panel = None
    panel_output_dir = None
    if panel_path is not None:
        selected_panel = Path(panel_path).expanduser().resolve()
        if not selected_panel.is_file():
            raise FileNotFoundError(f"输入面板不存在：{selected_panel}")
        with selected_panel.with_suffix(".json").open(encoding="utf-8") as file:
            panel_info = json.load(file)
        required_fields = {"cue_label", "return_label", "regression_sample_spec", "output_path"}
        missing_fields = required_fields - panel_info.keys()
        if missing_fields:
            raise ValueError(f"面板摘要缺少字段：{sorted(missing_fields)}。")
        if Path(panel_info["output_path"]).resolve() != selected_panel:
            raise ValueError("口径摘要对应的面板与本次选择不一致。")
        sample_spec = int(panel_info["regression_sample_spec"])
        if sample_spec not in (5, 6):
            raise ValueError("无法识别摘要中的合并筛样规格。")
        if selected_panel.name != f"最终回归面板_筛样规格{sample_spec}.csv":
            raise ValueError("面板文件名与摘要中的筛样规格不一致。")
        panel_output_dir = (
            PROJECT_ROOT / "图表结果/01_主线图表"
            / panel_info["cue_label"] / panel_info["return_label"]
            / f"合并筛样规格{sample_spec}" / "表3_散户订单失衡"
        )

    selected_date_fe = bool(date_fixed_effects)
    selected_summary = None
    if summary_path is not None:
        selected_summary = Path(summary_path).expanduser().resolve()
        if selected_summary.name == "表3_散户订单失衡_加入日期固定效应结果.csv":
            selected_date_fe = True
        elif selected_summary.name == "表3_散户订单失衡结果.csv":
            selected_date_fe = False
        else:
            raise ValueError("请选择表3估计脚本生成的结果汇总文件。")
        if date_fixed_effects is not None and bool(date_fixed_effects) != selected_date_fe:
            raise ValueError("指定的固定效应版本与结果汇总文件名不一致。")
        input_dir = selected_summary.parent
    elif result_dir is not None:
        input_dir = Path(result_dir).expanduser().resolve()
    elif panel_output_dir is not None:
        input_dir = panel_output_dir / (
            "加入日期固定效应" if selected_date_fe else "公司季度固定效应"
        )
    else:
        input_dir = OUTPUT_DIR

    run_info = None
    run_info_path = input_dir / "运行口径.json"
    if run_info_path.is_file():
        with run_info_path.open(encoding="utf-8") as file:
            run_info = json.load(file)
        required_fields = {"cue_label", "return_label", "regression_sample_spec", "fixed_effects"}
        missing_fields = required_fields - run_info.keys()
        if missing_fields:
            raise ValueError(f"运行口径摘要缺少字段：{sorted(missing_fields)}。")
        fixed_effects = run_info["fixed_effects"]
        if fixed_effects not in {"公司季度固定效应", "加入日期固定效应"}:
            raise ValueError("运行口径摘要中的固定效应版本无法识别。")
        recorded_date_fe = fixed_effects == "加入日期固定效应"
        if (date_fixed_effects is not None or selected_summary is not None) and recorded_date_fe != selected_date_fe:
            raise ValueError("选择的固定效应版本与运行口径摘要不一致。")
        selected_date_fe = recorded_date_fe
        if panel_info is not None:
            if "input_panel_path" not in run_info:
                raise ValueError("运行口径摘要缺少输入面板路径，无法核对所选面板。")
            for key in ("cue_label", "return_label", "regression_sample_spec"):
                if run_info[key] != panel_info[key]:
                    raise ValueError(f"结果与所选面板的“{key}”不一致。")
            if Path(run_info["input_panel_path"]).resolve() != selected_panel:
                raise ValueError("结果使用的输入面板与本次选择不一致。")
    elif panel_info is not None:
        expected_dir = panel_output_dir / (
            "加入日期固定效应" if selected_date_fe else "公司季度固定效应"
        )
        if input_dir.resolve() != expected_dir.resolve():
            raise ValueError("结果目录缺少运行口径摘要，无法核对所选面板。")
        run_info = panel_info

    if selected_date_fe:
        summary_name = "表3_散户订单失衡_加入日期固定效应结果.csv"
        detail_name = "表3_散户订单失衡_加入日期固定效应系数明细.csv"
        image_name = "table3_散户订单失衡_完整系数版_加入日期固定效应.png"
        table_number = "表 3（加入日期固定效应）"
    else:
        summary_name = "表3_散户订单失衡结果.csv"
        detail_name = "表3_系数明细.csv"
        image_name = "table3_散户订单失衡_完整系数版.png"
        table_number = "表 3"

    actual_summary_path = input_dir / summary_name
    actual_detail_path = input_dir / detail_name
    if run_info is not None:
        for key, actual_path in (("result_path", actual_summary_path), ("coefficients_path", actual_detail_path)):
            if key in run_info and Path(run_info[key]).resolve() != actual_path.resolve():
                raise ValueError("运行口径摘要记录的数值结果路径与本次选择不一致。")
    summary = read_result(
        actual_summary_path,
        [
            "衡量口径", "订单失衡窗口", "散户识别方法", "线索系数", "聚类标准误",
            "p值", "控制变量", "日期固定效应", "被提示公司×年度季度固定效应",
            "回归样本量", "R方",
        ],
    )
    detail_columns = ["订单失衡窗口", "变量", "变量顺序", "系数", "聚类标准误", "p值"]
    if selected_date_fe:
        detail_columns.append("日期固定效应")
    details = read_result(actual_detail_path, detail_columns)
    expected_date_fe = "是" if selected_date_fe else "否"
    for frame, source_name in [(summary, summary_name), (details, detail_name)]:
        if "日期固定效应" in frame.columns:
            values = frame["日期固定效应"].astype("string").str.strip()
            if values.isna().any() or not values.eq(expected_date_fe).all():
                raise ValueError(
                    f"{source_name}的日期固定效应应全部为“{expected_date_fe}”。"
                )

    title = "线索事件与散户订单失衡"
    additional_notes = []
    if run_info is not None:
        title += f"（{run_info['cue_label']}）"
        additional_notes.append(
            f"样本：{run_info['cue_label']}提示；{run_info['return_label']}对应样本；"
            f"合并筛样规格{run_info['regression_sample_spec']}。"
        )
    output = input_dir / image_name
    draw_order_imbalance_full(
        summary,
        details,
        table_number=table_number,
        title=title,
        output_path=output,
        label_right=0.34,
        additional_notes=additional_notes,
    )
    print(f"表3绘制完成：{output}")
    return [output]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取指定表3数值结果并绘图，不重新估计。")
    parser.add_argument("--panel", type=Path, default=None, help="最终面板路径，用同名摘要定位对应结果目录。")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--result-dir", type=Path, default=None, help="明确指定数值结果目录。")
    selection.add_argument("--summary", type=Path, default=None, help="明确指定结果汇总文件。")
    parser.add_argument("--date-fixed-effects", action="store_true", default=None, help="选择加入日期固定效应版本。")
    cli_args = parser.parse_args()
    main(
        date_fixed_effects=cli_args.date_fixed_effects,
        result_dir=cli_args.result_dir,
        panel_path=cli_args.panel,
        summary_path=cli_args.summary,
    )
