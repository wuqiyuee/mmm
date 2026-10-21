"""估计95分位图2，仅控制被提示公司×年度季度固定效应。

对应同目录的 figure2_95.py（95分位图2日期固定效应脚本）。
沿用其95分位事件映射、27个控制变量、2005—2025年样本和50个
累计收益窗口；直接使用面板中已经缩尾的收益，不再次缩尾。

本脚本不吸收交易日期固定效应，但标准误仍按被提示公司和交易日期
双向聚类，并开启有限样本修正。估计方法沿用主线公司季度版本。

输入：最终模式公司面板，以及既有95分位准备任务生成的事件映射。
输出：95分位图2目录中的图片、回归结果和逐窗口进度。
结果中的系数、标准误和置信区间为百分点；进度表保留原始小数单位。
默认核对输入和计算版本后自动续算；修改输入或脚本后须移走旧进度表。
可用 --plot-only（仅重画参数）读取本脚本的完整结果重新绘图。
导入本文件不会读取研究面板或执行回归。
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


# 一、输入输出与回归口径。
def find_project_root() -> Path:
    """从脚本位置向上查找项目根目录。"""
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "AGENTS.md").is_file()
            and (candidate / "code").is_dir()
            and (candidate / "中间结果").is_dir()
        ):
            return candidate
    raise FileNotFoundError("找不到项目根目录。")


ROOT = find_project_root()
COMMON_DIR = ROOT / "code/07_实证分析/90_公共模块"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from 图2图3_公司季度固定效应 import (
    draw_dynamic_figure,
    load_progress,
    save_progress,
)


PATTERN_PATH = ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"
EVENT_PATH = ROOT / "work/q95_tables_run/cue95_event_map.csv"
OUTPUT_DIR = ROOT / "图表结果/03_95分位检验/图2_收益动态"
RESULT_PATH = OUTPUT_DIR / "图2_95分位_收益动态和长期反转_公司季度固定效应结果.csv"
PROGRESS_PATH = OUTPUT_DIR / "图2_95分位_公司季度固定效应_2005至2025年_逐窗口进度.csv"
OUTPUT_PATH = OUTPUT_DIR / "figure2_95分位_收益动态和长期反转_公司季度固定效应.png"

CUE_COL = "quarterLag1size95plus"
CONTROL_COLS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
    + [
        "ownOtherEventER",
        "ownPerformanceForecastBrief",
        "aesIncrease",
        "aesDecrease",
        "logIncreaseAEV",
        "logDecreaseAEV",
    ]
)
START_YEAR = 2005
END_YEAR = 2025
HORIZONS = list(range(1, 51))
TITLE = (
    "收益动态和长期反转（95分位提示事件）\n"
    "固定效应：被提示公司×年度季度；不控制交易日期固定效应"
)


# 二、读取待估计窗口，并从95分位事件映射生成提示变量。
def read_panel(horizons: list[int]) -> pd.DataFrame:
    if not EVENT_PATH.is_file():
        raise FileNotFoundError(
            f"缺少95分位事件映射：{EVENT_PATH}\n"
            "请先按现有95分位表2表3准备流程生成并保留该事件映射。"
        )

    outcome_cols = [f"chAdjreturn_t_tp{i}_Recall_w" for i in horizons]
    columns = ["StkcdRecall", "Trddt", "yearQuarter"] + CONTROL_COLS + outcome_cols
    available = pd.read_csv(PATTERN_PATH, nrows=0).columns
    missing = sorted(set(columns) - set(available))
    if missing:
        raise ValueError(f"最终模式公司面板缺少变量：{missing}")

    print("正在读取95分位图2的公司季度固定效应样本……", flush=True)
    data = pd.read_csv(
        PATTERN_PATH,
        usecols=columns,
        dtype={
            "StkcdRecall": "string",
            "yearQuarter": "string",
            **{col: "float32" for col in CONTROL_COLS + outcome_cols},
        },
        parse_dates=["Trddt"],
    )
    data["StkcdRecall"] = data["StkcdRecall"].str.strip().str.zfill(6)
    if data[["StkcdRecall", "Trddt"]].isna().any().any():
        raise ValueError("模式公司面板的公司代码或交易日期存在缺失。")
    if data.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("模式公司面板的被提示公司×交易日期不是唯一键。")

    data = data.loc[data["Trddt"].dt.year.between(START_YEAR, END_YEAR)].copy()

    # 映射中的日期是自1970年1月1日起的整数日序号，不是年月日整数。
    event_map = pd.read_csv(
        EVENT_PATH,
        usecols=["firm", "day"],
        dtype={"firm": "int64", "day": "int64"},
    )
    event_keys = (
        event_map["firm"].to_numpy(dtype="int64") * 100_000
        + event_map["day"].to_numpy(dtype="int64")
    )
    data_keys = (
        data["StkcdRecall"].to_numpy(dtype="int64") * 100_000
        + data["Trddt"].to_numpy(dtype="datetime64[D]").astype("int64")
    )
    data[CUE_COL] = np.isin(data_keys, event_keys).astype("float32")
    if not data[CUE_COL].eq(1).any():
        raise ValueError("当前样本期没有匹配到95分位提示事件，请检查事件映射。")

    data["StkcdRecall"] = data["StkcdRecall"].astype("category")
    data["yearQuarter"] = data["yearQuarter"].astype("category")
    return data


# 三、逐窗口估计：只吸收公司季度，日期仅用于标准误聚类。
def estimate_window(data: pd.DataFrame, horizon: int) -> dict:
    outcome = f"chAdjreturn_t_tp{horizon}_Recall_w"
    rhs_cols = [CUE_COL] + CONTROL_COLS
    columns = [outcome, "StkcdRecall", "Trddt", "yearQuarter"] + rhs_cols
    reg = data[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if reg.empty:
        raise ValueError(f"窗口[t,t+{horizon}]没有可用回归观测。")

    reg["firm_yearquarter_fe"] = pd.Categorical(
        reg["StkcdRecall"].astype(str) + "_" + reg["yearQuarter"].astype(str)
    )
    clusters = pd.DataFrame(
        {
            "firm_cluster": pd.factorize(reg["StkcdRecall"])[0],
            "date_cluster": pd.factorize(reg["Trddt"])[0],
        },
        index=reg.index,
    )
    model = AbsorbingLS(
        dependent=reg[outcome],
        exog=reg[rhs_cols],
        absorb=reg[["firm_yearquarter_fe"]],
    )
    fit = model.fit(cov_type="clustered", clusters=clusters, debiased=True)
    interval = fit.conf_int(level=0.95).loc[CUE_COL]

    # 与主线进度表一致，先保留小数收益单位；最终输出时统一乘100。
    return {
        "window": horizon,
        "coef": float(fit.params[CUE_COL]),
        "se": float(fit.std_errors[CUE_COL]),
        "ci_low": float(interval.iloc[0]),
        "ci_high": float(interval.iloc[1]),
        "nobs": int(fit.nobs),
    }


# 四、保存结果和图片；已有进度经过版本核对后可自动恢复。
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="仅从本脚本的完整回归结果重画，不读取研究面板。",
    )
    args = parser.parse_args()

    if args.plot_only:
        results = pd.read_csv(RESULT_PATH)
    else:
        # 签名同时核对正式面板、95分位事件映射、本脚本和公共模块。
        rows, signature = load_progress(
            PROGRESS_PATH,
            [PATTERN_PATH, EVENT_PATH, Path(__file__).resolve()],
            "图2：95分位提示；仅公司季度固定效应；原始小数收益率",
        )
        completed = {int(row["window"]) for row in rows}
        pending = [h for h in HORIZONS if h not in completed]
        print(f"已恢复{len(completed)}个窗口，待估计{len(pending)}个。", flush=True)

        if pending:
            data = read_panel(pending)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            for horizon in pending:
                print(f"正在估计95分位图2：[t,t+{horizon}]", flush=True)
                row = estimate_window(data, horizon)
                rows.append(row)
                save_progress(rows, PROGRESS_PATH, signature)
                print(
                    f"窗口{horizon}完成：系数={row['coef'] * 100:.6f}，"
                    f"标准误={row['se'] * 100:.6f}（均为百分点）",
                    flush=True,
                )
                data.drop(columns=[f"chAdjreturn_t_tp{horizon}_Recall_w"], inplace=True)
                gc.collect()
            del data
            gc.collect()

        results = pd.DataFrame(rows).sort_values("window").reset_index(drop=True)
        results[["coef", "se", "ci_low", "ci_high"]] *= 100
        results["提示变量"] = CUE_COL
        results["结果单位"] = "百分点"
        results["日期固定效应"] = "否"
        results["被提示公司×年度季度固定效应"] = "是"
        results["聚类方式"] = "被提示公司和交易日期双向聚类"
        results["有限样本修正"] = "是"

    if sorted(results["window"].tolist()) != HORIZONS:
        raise ValueError("图2结果必须包含不重复的1至50号完整窗口。")
    if not results["日期固定效应"].eq("否").all():
        raise ValueError("结果混入了其他固定效应规格，不能用于本脚本绘图。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        results.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
        print(f"回归结果已保存：{RESULT_PATH}", flush=True)
    draw_dynamic_figure(results, TITLE, OUTPUT_PATH)


if __name__ == "__main__":
    main()
