"""95分位表5心理机制的独立入口。

只读取共享准备模块生成的独立面板，不写入主线数据或主线结果目录。
默认估计两个收益窗口、九种机制和三种盈余惊喜定义，共34个不同模型；
共同机制在三份结果中复用，吸收被提示公司×年度季度及交易日期固定效应，
标准误按被提示公司和交易日期双向聚类。

兼容说明：旧估计器的机制构造依赖内部键90plus（原高规模组）和
50_90（原中规模组）。此处只在估计器内部保留这两个键，实际读取的源字段
全部切换为95以上和50—95；对外表格、变量名和诊断文字统一转换为新口径。
价格异动沿用上游any（历史共同公告时任意一家价格异动）的实际构造。
本入口不调用旧脚本的缓存准备、主函数或绘图函数。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence


PREPARATION_PATH = (
    Path(__file__).resolve().parents[1] / "公共模块" / "准备95分位机制面板.py"
)

SIZE_GROUPS = (
    ("95plus", "95分位以上"),
    ("50_95", "50—95分位"),
)
SUE_NOTES = {
    1: "历史未调整每股收益",
    2: "历史调整每股收益",
    3: "分析师预测中位数",
}
MECHANISM_LABELS = {
    1: "异常总成交量",
    2: "异常散户成交量",
    3: "媒体覆盖",
    4: "价格异动\n（历史任意一家）",
    5: "正向盈余惊喜",
    6: "负向盈余惊喜",
    7: "极端正向盈余惊喜",
    8: "极端负向盈余惊喜",
    9: "干扰程度",
}


def load_preparation_module() -> Any:
    """加载本组独立准备模块；加载本身不执行面板准备。"""
    name = "_q95_table5_preparation"
    spec = importlib.util.spec_from_file_location(name, PREPARATION_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载95分位共享准备模块：{PREPARATION_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def remap_source_column(column: str | None) -> str | None:
    """把每个机制的规模交互源字段切换为新规模边界。"""
    if column is None:
        return None
    return (
        column.replace("size90plus", "size95plus")
        .replace("size50_90", "size50_95")
        .replace("S90p", "S95p")
        .replace("S5090", "S5095")
    )


def configure_estimator(preparation: Any) -> Any:
    """复用旧估计器，重新绑定全部机制定义而非只替换基准提示列。"""
    source = (
        preparation.ROOT
        / "code/07_实证分析/01_主线图表/表5_心理机制/估计回归与绘图.py"
    )
    estimator = preparation.load_module(source, "_q95_table5_estimator")
    original_builder = estimator.build_mechanism_specs

    def build_mechanism_specs(sue_id: int) -> tuple[Any, ...]:
        specifications = []
        for specification in original_builder(sue_id):
            label = specification.label
            note = specification.source_note
            if specification.order == 4:
                label = "价格异动（历史任意一家）"
                note = (
                    "历史共同公告时任意一家发生价格异动的强关联提示，"
                    "与同规模其他关联提示比较；沿用上游实际构造。"
                )
            elif specification.order == 9:
                note = (
                    "在95以上和50—95规模组内分别重算历史共同公告公司数量的"
                    "等权均值，沿用筛选前日度面板的年度二分口径；"
                    "低干扰为强关联提示，高干扰为其他关联提示。"
                )
            specifications.append(
                replace(
                    specification,
                    label=label,
                    first_90=remap_source_column(specification.first_90),
                    first_5090=remap_source_column(specification.first_5090),
                    second_90=remap_source_column(specification.second_90),
                    second_5090=remap_source_column(specification.second_5090),
                    source_note=note,
                )
            )
        return tuple(specifications)

    estimator._BASELINE_90 = "quarterLag1size95plus"
    estimator._BASELINE_5090 = "quarterLag1size50_95"
    # 内部键保留旧值以兼容现有规模分支；标签与源字段均使用95分位口径。
    estimator._SIZE_GROUPS = (
        ("90plus", "市值前5%", estimator._BASELINE_90),
        ("50_90", "市值50%至95%", estimator._BASELINE_5090),
    )
    estimator.build_mechanism_specs = build_mechanism_specs
    controls = [
        *estimator.OWN_EA_COLS,
        *estimator.DISCLOSURE_COLS,
        *estimator.MEDIA_CONTROL_COLS,
    ]
    if len(controls) != 27 or controls != list(preparation.CONTROL_COLS):
        raise ValueError("共享面板与表5估计器的27项控制变量不一致。")
    return estimator


def public_text(value: Any) -> Any:
    """导出时转换内部比较变量名、规模键和不可估计原因中的旧规模名。"""
    if not isinstance(value, str):
        return value
    return (
        value.replace("S90p", "S95p")
        .replace("S5090", "S5095")
        .replace("90plus", "95plus")
        .replace("50_90", "50_95")
        .replace("市值前10%", "市值前5%")
        .replace("50%至90%", "50%至95%")
    )


def export_table(table: Any) -> Any:
    """保留估计状态及完整诊断，不填补不可估计模型的数值。"""
    output = table.copy()
    for column in output.columns:
        output[column] = output[column].map(public_text)
    output["提示公司规模口径"] = "95分位以上；50—95分位"
    output["当前市值口径"] = "事件前第3个交易日总市值，对比事件上月市值断点"
    output["价格异动实际口径"] = "历史共同公告时任意一家价格异动"
    return output


def finite_number(value: Any) -> float | None:
    """缺失和非有限值只用于显示留空，不转换为零。"""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def stars(value: Any) -> str:
    p_value = finite_number(value)
    if p_value is None:
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def estimate_cell(row: Any, role: str) -> str:
    if str(row["估计状态"]) != "已估计":
        return "不可估计"
    coefficient = finite_number(row[f"{role}系数"])
    standard_error = finite_number(row[f"{role}聚类标准误"])
    if coefficient is None or standard_error is None:
        return "缺失"
    return (
        f"{100 * coefficient:.4f}{stars(row[f'{role}p值'])}"
        f"\n({100 * standard_error:.4f})"
    )


def p_value_cell(row: Any) -> str:
    if str(row["估计状态"]) != "已估计":
        return "—"
    p_value = finite_number(row["沃尔德p值"])
    if p_value is None:
        return "缺失"
    return "<0.0001" if p_value < 0.0001 else f"{p_value:.4f}"


def validate_summary(summary: Any) -> None:
    """检查双窗口九机制两规模的结果结构，保留不可估计占位行。"""
    required = {
        "面板", "机制编号", "规模组", "估计状态", "不可估计原因",
        "强关联提示系数", "强关联提示聚类标准误", "强关联提示p值",
        "其他关联提示系数", "其他关联提示聚类标准误", "其他关联提示p值",
        "沃尔德p值", "回归样本量", "R方", "日期固定效应",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError("95分位表5结果缺少字段：" + "、".join(missing))
    expected = {
        (panel, mechanism, size)
        for panel in ("A", "B")
        for mechanism in range(1, 10)
        for size, _ in SIZE_GROUPS
    }
    actual = [
        (str(row["面板"]), int(row["机制编号"]), str(row["规模组"]))
        for _, row in summary.iterrows()
    ]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("95分位表5应有双窗口×九机制×两规模的36行唯一结果。")
    if set(summary["日期固定效应"].astype(str)) != {"是"}:
        raise ValueError("95分位表5要求全部模型加入交易日期固定效应。")
    if not set(summary["估计状态"].astype(str)).issubset({"已估计", "不可估计"}):
        raise ValueError("95分位表5存在未知估计状态。")


def model_statistic(rows: list[Any], column: str) -> str:
    """同一模型两规模组应共享样本量与拟合优度。"""
    numbers = [finite_number(row[column]) for row in rows]
    if numbers[0] != numbers[1]:
        raise ValueError(f"同一机制两个规模组的{column}不一致。")
    value = numbers[0]
    if value is None:
        return "—"
    return f"{int(value):,}" if column == "回归样本量" else f"{value:.4f}"


def draw_table(summary: Any, sue_id: int, output_path: Path, font_path: Path) -> None:
    """独立绘制双窗口中文表格，不使用主线图中硬编码的90分位标签。"""
    validate_summary(summary)
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = (
        FontProperties(fname=str(font_path))
        if font_path.exists()
        else FontProperties(family="sans-serif")
    )
    lookup = {
        (str(row["面板"]), int(row["机制编号"]), str(row["规模组"])): row
        for _, row in summary.iterrows()
    }
    headers = [
        "心理机制", "95分位以上\n强关联提示", "95分位以上\n其他关联提示",
        "系数相等\n检验概率", "50—95分位\n强关联提示", "50—95分位\n其他关联提示",
        "系数相等\n检验概率", "样本量", "拟合优度",
    ]
    widths = [0.16, 0.13, 0.13, 0.07, 0.13, 0.13, 0.07, 0.12, 0.06]
    figure, axes = plt.subplots(figsize=(18, 13))
    figure.patch.set_facecolor("white")
    axes.axis("off")
    axes.text(
        0.5, 0.975, f"表5　95分位提示事件的心理机制（盈余惊喜口径{sue_id}）",
        ha="center", va="top", fontproperties=font, fontsize=20, weight="bold",
    )
    axes.text(
        0.5, 0.935, f"盈余惊喜定义：{SUE_NOTES[sue_id]}；系数及括号内双向聚类标准误的单位均为百分点",
        ha="center", va="top", fontproperties=font, fontsize=11,
    )
    for panel, title, bottom, title_y in (
        ("A", "短窗口：事件当日至后第1个交易日", 0.535, 0.89),
        ("B", "长窗口：事件当日至后第10个交易日", 0.145, 0.50),
    ):
        rows = []
        for order in range(1, 10):
            pair_rows = [lookup[(panel, order, size)] for size, _ in SIZE_GROUPS]
            cells = [MECHANISM_LABELS[order]]
            for row in pair_rows:
                cells.extend([
                    estimate_cell(row, "强关联提示"),
                    estimate_cell(row, "其他关联提示"),
                    p_value_cell(row),
                ])
            cells.extend([
                model_statistic(pair_rows, "回归样本量"),
                model_statistic(pair_rows, "R方"),
            ])
            rows.append(cells)
        axes.text(
            0.01, title_y, title, ha="left", va="top",
            fontproperties=font, fontsize=13, weight="bold",
        )
        table = axes.table(
            cellText=rows, colLabels=headers, colWidths=widths,
            cellLoc="center", colLoc="center", bbox=[0.01, bottom, 0.98, 0.31],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row_index, column_index), cell in table.get_celld().items():
            cell.get_text().set_fontproperties(font)
            cell.get_text().set_fontsize(9)
            cell.set_edgecolor("#CCCCCC")
            cell.set_linewidth(0.45)
            cell.PAD = 0.035
            if row_index == 0:
                cell.set_facecolor("#E8EDF2")
                cell.get_text().set_weight("bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#F7F8FA")
            if column_index == 0:
                cell.get_text().set_ha("left")
    notes = [
        "注：每个机制模型同时估计两规模组的强关联提示和其他关联提示四个变量；干扰机制分别对应低干扰和高干扰。",
        "95分位以上指事件前第3个交易日总市值严格高于事件上月95分位断点；50—95组为高于50分位且不高于95分位。",
        "全部模型加入27项控制变量、被提示公司×年度季度及交易日期固定效应；按被提示公司和交易日期双向聚类。",
        "价格异动沿用历史共同公告时任意一家发生异动的实际口径；异常散户成交量使用小单替代。",
        "系数相等检验为同规模两系数的沃尔德检验；*、**、***分别对应10%、5%、1%显著性水平。",
        "不可估计模型不填补系数或显著性；完整原因、聚类数和单例删除数见配套结果表。",
    ]
    axes.text(
        0.01, 0.115, "\n".join(notes), ha="left", va="top",
        fontproperties=font, fontsize=9, linespacing=1.6,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_outputs(
    main_results: dict[int, Any],
    coefficient_details: dict[int, Any],
    output_dir: Path,
    font_path: Path,
) -> list[Path]:
    """输出三份双窗口主结果、三份核心系数明细及三张双窗口图片。"""
    if set(main_results) != {1, 2, 3} or set(coefficient_details) != {1, 2, 3}:
        raise ValueError("95分位表5必须包含三种盈余惊喜定义的完整结果。")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for sue_id in (1, 2, 3):
        summary = export_table(main_results[sue_id])
        details = export_table(coefficient_details[sue_id])
        validate_summary(summary)
        stem = f"表5_95分位_盈余惊喜口径{sue_id}_心理机制_加入日期固定效应"
        result_path = output_dir / f"{stem}结果.csv"
        detail_path = output_dir / f"{stem}系数明细.csv"
        image_path = output_dir / f"{stem}_双窗口.png"
        summary.to_csv(result_path, index=False, encoding="utf-8-sig")
        details.to_csv(detail_path, index=False, encoding="utf-8-sig")
        draw_table(summary, sue_id, image_path, font_path)
        written.extend((result_path, detail_path, image_path))
    return written


def positive_integer(value: str) -> int:
    converted = int(value)
    if converted <= 0:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return converted


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="95分位表5心理机制独立入口")
    parser.add_argument("--prepare-only", action="store_true", help="只准备独立面板，不估计或绘图")
    parser.add_argument("--rebuild-panel", action="store_true", help="重新构建独立95分位机制面板")
    parser.add_argument("--cache-path", type=Path, help="独立95分位机制面板缓存路径")
    parser.add_argument("--chunk-size", type=positive_integer, default=250_000, help="分块读取与估计的每批行数")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    sys.dont_write_bytecode = True
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "2"
    preparation = load_preparation_module()
    requested_cache = args.cache_path if args.cache_path is not None else preparation.DEFAULT_CACHE
    # 先由共享模块校验或准备缓存，再配置旧估计器，避免调用旧缓存接口。
    cache_path = preparation.prepare_panel(
        requested_cache, rebuild=args.rebuild_panel, chunk_size=args.chunk_size,
    )
    print(f"95分位心理机制面板：{cache_path}", flush=True)
    if args.prepare_only:
        return [Path(cache_path)]
    scratch = preparation.ROOT / "work/表5_95分位运行缓存"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(scratch / "绘图缓存")
    os.environ["XDG_CACHE_HOME"] = str(scratch / "缓存")
    estimator = configure_estimator(preparation)
    main_results, coefficient_details = estimator.run_all_models_streaming(
        cache_path, batch_size=args.chunk_size, date_fixed_effects=True,
    )
    output_dir = preparation.ROOT / "图表结果/03_95分位检验/表5_心理机制"
    written = write_outputs(
        main_results, coefficient_details, output_dir, estimator.FONT_PATH,
    )
    for path in written:
        print(f"已保存：{path}", flush=True)
    return written


if __name__ == "__main__":
    main()
