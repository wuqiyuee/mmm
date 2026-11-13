"""读取明确选定的回归结果，绘制模式公司表2。

直接运行时使用顶部选定的模型汇总文件，也可由任务入口传入实际结果路径。
系数明细根据汇总文件名确定；提示口径、被解释变量口径和筛样规格从结果表读取。
图片和便携文档按这三个维度分目录保存。
本脚本不重新估计回归，也不依赖其他表的结果。
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
# 选择本次需要绘制的模型汇总结果。
# 文件名应与估计回归脚本实际生成的模型汇总文件一致。
SELECTED_SUMMARY_PATH = PROJECT_ROOT / (
    "图表结果/01_主线图表/"
    "市值90分位/自算标签收益/"
    "合并筛样规格5/表2_收益回归/"
    "表2_模型汇总_市值90分位_自算标签收益_个人365天_数行者分段_允许填充_全期_筛样规格5.csv"
)


def validate_current_results(summary, details) -> dict:
    """核对两张结果表的口径元数据、模型完整性和唯一键。"""
    rerun_hint = (
        "请先运行新版“估计回归.py”生成当前口径结果，再绘制表2。"
    )

    model_key = ["样本", "收益窗口", "规格"]
    metadata_columns = [
        "提示口径名称",
        "被解释变量口径名称",
        "合并筛样规格",
        "口径标识",
        "提示变量",
    ]

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

        for column in metadata_columns:
            if column not in frame.columns:
                raise ValueError(
                    f"表2{name}缺少字段“{column}”。{rerun_hint}"
                )
            if frame[column].isna().any() or frame[column].nunique() != 1:
                raise ValueError(
                    f"表2{name}的“{column}”缺失或存在多个取值。"
                    f"{rerun_hint}"
                )

        if not frame["样本"].eq("pattern").all():
            raise ValueError(
                f"表2{name}必须仅包含模式公司主样本。{rerun_hint}"
            )

        if frame[key].isna().any().any() or frame.duplicated(key).any():
            raise ValueError(
                f"表2{name}的唯一键缺失或重复。{rerun_hint}"
            )

        actual_models = set(
            frame[model_key].itertuples(index=False, name=None)
        )
        if actual_models != expected_models:
            raise ValueError(
                f"表2{name}必须完整对应两种窗口、六种规格的十二个模型。"
                f"{rerun_hint}"
            )

    summary_values = summary.iloc[0]
    detail_values = details.iloc[0]

    for column in [
        "提示口径名称",
        "被解释变量口径名称",
        "口径标识",
        "提示变量",
    ]:
        if str(summary_values[column]) != str(detail_values[column]):
            raise ValueError(
                f"模型汇总和系数明细的“{column}”不一致。"
                f"{rerun_hint}"
            )

    summary_sample_spec = int(summary_values["合并筛样规格"])
    detail_sample_spec = int(detail_values["合并筛样规格"])

    if summary_sample_spec != detail_sample_spec:
        raise ValueError(
            f"模型汇总和系数明细的“合并筛样规格”不一致。"
            f"{rerun_hint}"
        )

    if summary_sample_spec not in (5, 6):
        raise ValueError(
            f"无法识别合并筛样规格：{summary_sample_spec}。"
        )

    cue_column = str(summary_values["提示变量"])
    cue_models = set(
        details.loc[details["变量"].eq(cue_column), model_key]
        .itertuples(index=False, name=None)
    )

    if cue_models != expected_models:
        raise ValueError(
            f"表2系数明细必须逐模型包含当前提示变量“{cue_column}”。"
            f"{rerun_hint}"
        )

    return {
        "cue_label": str(summary_values["提示口径名称"]),
        "return_label": str(summary_values["被解释变量口径名称"]),
        "sample_spec": summary_sample_spec,
        "spec_id": str(summary_values["口径标识"]),
        "cue_column": cue_column,
    }


def main(
    result_dir: Path | None = None,
    *,
    summary_path: Path | None = None,
) -> list[Path]:
    """读取明确选定的结果，按其口径绘制并保存表2。"""

    # 1. 明确本次使用的模型汇总文件。
    if summary_path is not None and result_dir is not None:
        raise ValueError("请只指定模型汇总文件或数值结果目录中的一种。")

    if summary_path is not None:
        selected_path = Path(summary_path)

    elif result_dir is not None:
        input_dir = Path(result_dir)

        if not input_dir.is_dir():
            raise FileNotFoundError(
                f"数值结果目录不存在：{input_dir}"
            )

        candidates = sorted(
            path
            for path in input_dir.glob("表2_模型汇总_*.csv")
            if path.is_file()
        )

        if len(candidates) != 1:
            raise ValueError(
                f"指定目录中发现{len(candidates)}份模型汇总文件。"
                "请明确指定一份模型汇总文件。"
            )

        selected_path = candidates[0]

    else:
        selected_path = SELECTED_SUMMARY_PATH

    selected_path = selected_path.resolve()

    # 2. 从汇总文件名取得文件标签，找到配套系数明细。
    prefix = "表2_模型汇总_"

    if (
        selected_path.suffix != ".csv"
        or not selected_path.stem.startswith(prefix)
    ):
        raise ValueError(
            "模型汇总文件应命名为“表2_模型汇总_口径标识.csv”。"
        )

    output_tag = selected_path.stem[len(prefix):]

    if not output_tag:
        raise ValueError("模型汇总文件名缺少口径标识。")

    details_path = selected_path.with_name(
        f"表2_系数明细_{output_tag}.csv"
    )

    # 3. 读取两张结果表并核对口径和十二个模型。
    summary = read_result(
        selected_path,
        [
            "样本", "样本说明", "收益窗口", "规格",
            "线索系数（百分点）", "聚类标准误（百分点）",
            "p值", "回归样本量", "R方",
        ],
    )

    details = read_result(
        details_path,
        [
            "样本", "收益窗口", "规格", "变量", "变量顺序",
            "系数（百分点）", "聚类标准误（百分点）", "p值",
        ],
    )

    result_info = validate_current_results(summary, details)

    if output_tag != result_info["spec_id"]:
        raise ValueError("结果文件名与表内口径标识不一致。")

    # 4. 根据结果表中的元数据生成图片目录。
    cue_label = result_info["cue_label"]
    return_label = result_info["return_label"]
    sample_spec = result_info["sample_spec"]

    output_dir = (
        PROJECT_ROOT
        / "图表结果"
        / "01_主线图表"
        / cue_label
        / return_label
        / f"合并筛样规格{sample_spec}"
        / "表2_收益回归"
    )

    # 5. 图注使用本次结果对应的口径。
    notes = [
        (
            f"注：{cue_label}提示；被解释变量：{return_label}；"
            f"合并筛样规格{sample_spec}。"
        ),
        (
            "剔除前365天个人分析师共同覆盖关系，"
            "保留提示方金融和房地产公司。"
        ),
        (
            "新闻：数行者2005—2009年报刊、2010—2025年网络；"
            "允许合格填充日起点，保留全期年份。"
        ),
        (
            "主规格为（6）：27项控制、公司×年度季度和日期固定效应；"
            "括号内为公司与日期双向聚类标准误。"
        ),
        (
            "R²采用原回归库总体口径；空白表示未纳入、被吸收"
            "或聚类方差非正，详见系数明细。"
        ),
        (
            "星号基于双侧检验原始概率值，未作多重检验调整；"
            "*、**、***分别表示10%、5%、1%显著水平。"
        ),
    ]

    # 6. 沿用原有四项绘图输出。
    compact = output_dir / (
        f"table2_收益回归_紧凑版_{output_tag}.png"
    )
    pattern_first = output_dir / (
        f"table2_收益回归_完整系数版_模式公司主样本_{output_tag}.png"
    )
    pattern_second = output_dir / (
        f"table2_收益回归_完整系数版_模式公司主样本_续页_{output_tag}.png"
    )
    pattern_pdf = output_dir / (
        f"table2_收益回归_完整系数版_模式公司主样本_{output_tag}.pdf"
    )

    draw_table2_compact(
        summary,
        compact,
        notes=notes,
    )

    draw_table2_full_two_pages(
        summary,
        details,
        sample="pattern",
        first_page_output=pattern_first,
        second_page_output=pattern_second,
        pdf_output=pattern_pdf,
        notes=notes,
    )

    outputs = [
        compact,
        pattern_first,
        pattern_second,
        pattern_pdf,
    ]

    print("模型汇总：", selected_path)
    print("系数明细：", details_path)
    print("提示口径：", cue_label)
    print("被解释变量口径：", return_label)
    print("合并筛样规格：", sample_spec)
    print("表2绘制完成：")
    for output in outputs:
        print(output)

    return outputs

if __name__ == "__main__":
    main()
