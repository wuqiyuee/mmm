"""使用共享95分位机制面板，独立生成表4规模异质性结果。

默认估计全部、小型和大型被提示公司三个模型，因变量均为[t,t+1]
特征调整收益率；吸收被提示公司×年度季度和交易日期固定效应，
标准误按被提示公司及交易日期双向聚类。只有显式运行本入口时才准备
缓存、估计或输出文件；仅准备缓存参数会在估计前返回。
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
SHARED_PATH = MODULE_DIR.parent / "公共模块/准备95分位机制面板.py"
CUE_COLS = [
    "quarterLag1size95plus",
    "quarterLag1size50_95",
    "quarterLag1size10_50",
    "quarterLag1size10minus",
]
OUTCOME = "chAdjreturn_t_tp1_Recall_w"
LABELS = {
    "quarterLag1size95plus": "提示公司市值高于95分位",
    "quarterLag1size50_95": "提示公司市值50至95分位",
    "quarterLag1size10_50": "提示公司市值10至50分位",
    "quarterLag1size10minus": "提示公司市值低于10分位",
}


def load_shared_module():
    """读取公共准备模块，导入本入口时不准备数据。"""
    name = "q95_table4_shared"
    spec = importlib.util.spec_from_file_location(name, SHARED_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法读取95分位公共准备模块：{SHARED_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def positive_int(value: str) -> int:
    """限制分块行数为正整数。"""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("分块行数必须为正整数。")
    return number


def read_table4_panel(cache_path: Path, control_cols: list[str]):
    """仅读取表4所需列，沿用共享缓存中的被提示公司规模划分。"""
    import numpy as np
    import pandas as pd

    columns = list(dict.fromkeys([
        "StkcdRecall", "Trddt", "year", "yearQuarter", "smallFirm",
        OUTCOME, *CUE_COLS, *control_cols,
    ]))
    data = pd.read_parquet(cache_path, columns=columns)
    data["Trddt"] = pd.to_datetime(data["Trddt"], errors="raise")
    if data[["StkcdRecall", "Trddt", "year", "yearQuarter"]].isna().any().any():
        raise ValueError("95分位表4缓存中的公司、日期、年份或季度键缺失。")
    data["StkcdRecall"] = (
        data["StkcdRecall"].astype("string").str.strip().str.zfill(6)
    )
    data["year"] = pd.to_numeric(data["year"], errors="raise")
    data = data.loc[data["year"].between(2005, 2025)].copy()
    if data.empty:
        raise ValueError("95分位表4缓存没有2005—2025年的观测。")
    if not data["year"].eq(data["Trddt"].dt.year).all():
        raise ValueError("95分位表4缓存的年份与交易日期不一致。")
    if data.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("95分位表4缓存的被提示公司×交易日期不是唯一键。")
    for column in ["smallFirm", *CUE_COLS]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not data[column].dropna().isin([0, 1]).all():
            raise ValueError(f"95分位表4字段{column}必须为0、1或缺失。")
    # 单个模型自行删除因变量和右端缺失；此处只规范非有限数值。
    for column in [OUTCOME, *control_cols]:
        values = pd.to_numeric(data[column], errors="raise")
        data[column] = values.mask(~np.isfinite(values), np.nan).astype("float32")
    data["StkcdRecall"] = data["StkcdRecall"].astype("category")
    data["yearQuarter"] = data["yearQuarter"].astype("string").astype("category")
    return data


def draw_table(summary, core_results, output_path: Path, estimator) -> None:
    """独立绘制95分位标签，避免调用主线90分位图表。"""
    import matplotlib.pyplot as plt

    model_rows = []
    for specification in ["(1)", "(2)", "(3)"]:
        selected = summary.loc[summary["规格"].eq(specification)]
        if len(selected) != 1:
            raise ValueError(f"95分位表4缺少或重复规格：{specification}")
        model_rows.append(selected.iloc[0])
    lookup = {
        (row["规格"], row["变量"]): row
        for _, row in core_results.iterrows()
    }
    rows = []
    coefficient_rows = set()
    for variable in CUE_COLS:
        estimates, errors = [], []
        for model in model_rows:
            key = (model["规格"], variable)
            if key not in lookup:
                raise ValueError(f"95分位表4系数明细缺少：{key}")
            coefficient, standard_error = estimator._format_table4_estimate(lookup[key])
            estimates.append(coefficient)
            errors.append(standard_error)
        coefficient_rows.add(len(rows))
        rows.append([LABELS[variable], *estimates])
        rows.append(["", *errors])
    rows.extend([
        ["控制变量", *[str(row["控制变量"]) for row in model_rows]],
        ["交易日期固定效应", *[str(row["日期固定效应"]) for row in model_rows]],
        [
            "被提示公司×年度季度固定效应",
            *[str(row["被提示公司×年度季度固定效应"]) for row in model_rows],
        ],
        ["样本量", *[f"{int(row['回归样本量']):,}" for row in model_rows]],
        ["决定系数", *[f"{float(row['R方']):.4f}" for row in model_rows]],
    ])
    figure, axes = plt.subplots(figsize=(12.0, 9.2))
    figure.patch.set_facecolor("white")
    axes.axis("off")
    axes.text(
        0.01, 0.975, "表4：95分位提示事件与被提示公司收益的规模异质性",
        ha="left", va="top", fontsize=16, fontproperties=estimator.FONT_BOLD,
    )
    axes.text(
        0.01, 0.937,
        "因变量为被提示公司[t,t+1]两日特征调整收益率；系数与标准误单位为百分点",
        ha="left", va="top", fontsize=9, color="#444444", fontproperties=estimator.FONT,
    )
    table = axes.table(
        cellText=rows,
        colLabels=["变量", "(1)\n全部被提示公司", "(2)\n小型被提示公司", "(3)\n大型被提示公司"],
        cellLoc="center", colLoc="center", colWidths=[0.49, 0.17, 0.17, 0.17],
        bbox=[0.01, 0.16, 0.98, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#9AA3AA")
        cell.set_linewidth(0.35)
        cell.set_facecolor("#F4F6F7" if row_index == 0 else "white")
        cell.get_text().set_fontproperties(
            estimator.FONT_BOLD
            if row_index == 0 or row_index - 1 in coefficient_rows
            else estimator.FONT
        )
        if column_index == 0:
            cell.get_text().set_ha("left")
    notes = [
        "注：提示公司规模按事件前第3个交易日总市值与事件上月断点划分；四档为95以上、50至95、10至50和10以下。",
        "被提示公司的小型、大型分组沿用正式面板的年度规模划分；全部模型吸收公司×年度季度及交易日期固定效应。",
        "括号内为按被提示公司和交易日期双向聚类的标准误；*、**、*** 分别表示在10%、5%和1%水平上显著。",
        "空白系数表示该提示变量未被估计，具体状态见系数明细；研究期为2005—2025年。",
    ]
    for index, note in enumerate(notes):
        axes.text(
            0.01, 0.12 - 0.026 * index, note,
            ha="left", va="top", fontsize=7.2, color="#444444", fontproperties=estimator.FONT,
        )
    try:
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    finally:
        plt.close(figure)


def main() -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="仅准备共享95分位面板后返回")
    parser.add_argument("--rebuild-panel", action="store_true", help="重新构造共享95分位面板")
    parser.add_argument("--cache-path", type=Path, help="共享95分位面板的列式缓存路径")
    parser.add_argument("--chunk-size", type=positive_int, default=250_000, help="准备面板时每块读取的行数")
    args = parser.parse_args()

    sys.dont_write_bytecode = True
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[name] = "2"
    shared = load_shared_module()
    cache_path = args.cache_path if args.cache_path is not None else shared.DEFAULT_CACHE
    cache_path = shared.prepare_panel(
        cache_path=cache_path, rebuild=args.rebuild_panel, chunk_size=args.chunk_size,
    )
    if args.prepare_only:
        print(f"共享95分位机制面板已准备：{cache_path}", flush=True)
        return [Path(cache_path)]

    # 运行时只配置独立目录，原主线模块的估计入口不被调用。
    scratch = shared.ROOT / "work/表4_95分位运行缓存"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(scratch / "绘图缓存")
    os.environ["XDG_CACHE_HOME"] = str(scratch / "缓存")
    import pandas as pd

    source = shared.ROOT / "code/07_实证分析/01_主线图表/表4_规模异质性/估计回归_日期固定效应.py"
    estimator = shared.load_module(source, "q95_table4_date_fe_estimator")
    estimator.TABLE4_CUE_COLS = list(CUE_COLS)
    estimator.TABLE4_CONTROL_COLS = list(shared.CONTROL_COLS)
    estimator.TABLE4_OUTCOME = OUTCOME
    output_dir = shared.ROOT / "图表结果/03_95分位检验/表4_规模异质性"
    estimator.OUTPUT_DIR = output_dir
    data = read_table4_panel(Path(cache_path), list(shared.CONTROL_COLS))
    model_rows, coefficient_rows = [], []
    for specification in estimator.TABLE4_SPECIFICATIONS:
        print(
            f"正在估计95分位表4：{specification['规格']}，{specification['样本']}",
            flush=True,
        )
        model_info, current_coefficients = estimator.run_table4_date_fe_regression(data, specification)
        model_rows.append(model_info)
        coefficient_rows.extend(current_coefficients)
        gc.collect()
    results = pd.DataFrame(model_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    core_results = coefficients.loc[coefficients["变量"].isin(CUE_COLS)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "表4_95分位_规模异质性_加入日期固定效应结果.csv"
    coefficients_path = output_dir / "表4_95分位_规模异质性_加入日期固定效应系数明细.csv"
    image_path = output_dir / "表4_95分位_规模异质性_加入日期固定效应.png"
    core_results.to_csv(results_path, index=False, encoding="utf-8-sig")
    coefficients.to_csv(coefficients_path, index=False, encoding="utf-8-sig")
    draw_table(results, core_results, image_path, estimator)
    for path in [results_path, coefficients_path, image_path]:
        print(f"95分位表4输出：{path}", flush=True)
    return [results_path, coefficients_path, image_path]


if __name__ == "__main__":
    main()
