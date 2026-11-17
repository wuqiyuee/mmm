"""表3：散户订单失衡（成交量基准规格）的独立估计入口。

输入：规律公告公司最终回归面板与原订单失衡预处理表。
输出：默认保存至原表3目录；指定 --panel（输入面板）后按口径和固定效应分目录。
对应的完整系数图片由本任务绘图入口保存到同一任务目录。
口径：2005—2025年；公司日左合并；各因变量在合并样本中独立双侧1%缩尾；
逐模型仅删除本模型因变量、线索、控制和固定效应所需字段缺失，随后删除公司季度单例。
吸收被提示公司×年度季度固定效应，不吸收日期固定效应；公司与日期双向聚类，
保留有限样本修正。导入不读面板、不创建结果目录，显式运行入口函数才估计并写结果。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import sys

import pandas as pd


def find_project_root() -> Path:
    """沿当前脚本的祖先寻找项目标识，避免依赖固定目录层数。"""
    for candidate in Path(__file__).resolve().parents:
        if ((candidate / "AGENTS.md").is_file()
                and (candidate / "code").is_dir()
                and (candidate / "中间结果").is_dir()):
            return candidate
    raise FileNotFoundError("未找到含项目说明、代码及中间结果目录的项目根目录。")


PROJECT_ROOT = find_project_root()
SHARED_DIR = PROJECT_ROOT / "code/07_实证分析/90_公共模块"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from 订单失衡估计 import run_oib_regression, winsorize_1pct

PATTERN_PATH = PROJECT_ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"
DEFAULT_PATTERN_PATH = PATTERN_PATH
OIB_PATH = PROJECT_ROOT / "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv"
START_YEAR = 2005
END_YEAR = 2025

# 控制变量的构成、顺序与原回归脚本一致。
own_ea_cols = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)
disclosure_cols = ["ownOtherEventER", "ownPerformanceForecastBrief"]
media_cols = ["aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV"]
rhs_cols = ["quarterLag1size90plus"] + own_ea_cols + disclosure_cols + media_cols

TASK_NAME = '表3_散户订单失衡'
OUTPUT_DIR = PROJECT_ROOT / "图表结果/01_主线图表" / TASK_NAME
DEFAULT_OUTPUT_DIR = OUTPUT_DIR
RESULTS_PATH = OUTPUT_DIR / '表3_散户订单失衡结果.csv'
COEFFICIENTS_PATH = OUTPUT_DIR / '表3_系数明细.csv'
RUN_INFO = None
RAW_OUTCOMES = ['OIBvol_avg_t_tp1', 'OIBvol_avg_t_tp10']
OUTCOMES = [(RAW_OUTCOMES[0] + "_w", "[t, t+1]"), (RAW_OUTCOMES[1] + "_w", "[t, t+10]")]


def configure_input_output(panel_path: Path | None) -> None:
    """仅设置输入、口径说明和输出路径，不改变样本处理及估计步骤。"""
    global PATTERN_PATH, OUTPUT_DIR, RESULTS_PATH, COEFFICIENTS_PATH, RUN_INFO
    PATTERN_PATH = DEFAULT_PATTERN_PATH
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    RUN_INFO = None
    if panel_path is not None:
        PATTERN_PATH = Path(panel_path).expanduser().resolve()
        if not PATTERN_PATH.is_file():
            raise FileNotFoundError(f"输入面板不存在：{PATTERN_PATH}")
        summary_path = PATTERN_PATH.with_suffix(".json")
        with summary_path.open(encoding="utf-8") as file:
            panel_info = json.load(file)
        required_fields = {
            "cue_definition", "cue_label", "cue_column", "return_definition",
            "return_label", "regression_sample_spec", "output_path",
        }
        missing_fields = required_fields - panel_info.keys()
        if missing_fields:
            raise ValueError(f"面板摘要缺少字段：{sorted(missing_fields)}。")
        if Path(panel_info["output_path"]).resolve() != PATTERN_PATH:
            raise ValueError("口径摘要对应的面板与本次选择不一致。")
        if panel_info["cue_column"] != rhs_cols[0]:
            raise ValueError("面板摘要中的提示变量与本脚本使用的字段不一致。")
        sample_spec = int(panel_info["regression_sample_spec"])
        if sample_spec not in (5, 6):
            raise ValueError("无法识别摘要中的合并筛样规格。")
        if PATTERN_PATH.name != f"最终回归面板_筛样规格{sample_spec}.csv":
            raise ValueError("面板文件名与摘要中的筛样规格不一致。")
        OUTPUT_DIR = (
            PROJECT_ROOT / "图表结果/01_主线图表"
            / panel_info["cue_label"] / panel_info["return_label"]
            / f"合并筛样规格{sample_spec}" / TASK_NAME / "公司季度固定效应"
        )
        RUN_INFO = {
            key: panel_info[key]
            for key in required_fields - {"output_path"}
        }
        RUN_INFO.update({
            "input_panel_path": str(PATTERN_PATH),
            "panel_summary_path": str(summary_path),
            "fixed_effects": "公司季度固定效应",
            "output_dir": str(OUTPUT_DIR),
        })
    RESULTS_PATH = OUTPUT_DIR / "表3_散户订单失衡结果.csv"
    COEFFICIENTS_PATH = OUTPUT_DIR / "表3_系数明细.csv"
    if RUN_INFO is not None:
        RUN_INFO.update({
            "result_path": str(RESULTS_PATH),
            "coefficients_path": str(COEFFICIENTS_PATH),
        })


def prepare_data() -> pd.DataFrame:
    """只读本任务字段；先按公司日左合并，再在2005—2025合并样本上缩尾。"""
    panel_cols = list(dict.fromkeys(
        ["StkcdRecall", "Trddt", "year", "yearQuarter"] + rhs_cols
    ))
    numeric_cols = [
        col for col in panel_cols
        if col not in {"StkcdRecall", "Trddt", "yearQuarter"}
    ]
    panel_dtype = {"StkcdRecall": "string", "yearQuarter": "string"}
    panel_dtype.update({col: "float32" for col in numeric_cols})
    print("读取本任务所需规律公告公司样本……")
    data = pd.read_csv(
        PATTERN_PATH, usecols=panel_cols, dtype=panel_dtype, parse_dates=["Trddt"],
    )
    data["StkcdRecall"] = data["StkcdRecall"].str.strip().str.zfill(6)
    if START_YEAR is not None:
        data = data.loc[data["year"] >= START_YEAR].copy()
    if END_YEAR is not None:
        data = data.loc[data["year"] <= END_YEAR].copy()
    if data.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("主样本的被提示公司和日期不是唯一键。")

    # 两个独立任务分别只读取自己使用的两个因变量；不交叉要求另一口径完整。
    oib_dtype = {"Stkcd": "string"}
    oib_dtype.update({col: "float32" for col in RAW_OUTCOMES})
    oib = pd.read_csv(
        OIB_PATH, usecols=["Stkcd", "Trddt"] + RAW_OUTCOMES,
        dtype=oib_dtype, parse_dates=["Trddt"],
    )
    oib["Stkcd"] = oib["Stkcd"].str.strip().str.zfill(6)
    oib = oib.rename(columns={"Stkcd": "StkcdRecall"})
    if oib.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("订单失衡表的被提示公司和日期不是唯一键。")
    data = data.merge(
        oib, on=["StkcdRecall", "Trddt"], how="left", validate="one_to_one",
    )
    del oib
    gc.collect()

    # 与原顺序一致：缩尾时尚未因控制缺失或固定效应单例删除观测。
    # 订单失衡源表已经使用百分点，本任务不再次乘100。
    for col in RAW_OUTCOMES:
        data[f"{col}_w"] = winsorize_1pct(data[col]).astype("float32")
    data = data.drop(columns=RAW_OUTCOMES)
    data["StkcdRecall"] = data["StkcdRecall"].astype("category")
    data["yearQuarter"] = data["yearQuarter"].astype("category")
    return data


def main(panel_path: Path | None = None) -> list[Path]:
    """独立完成本任务的估计与数值表输出，返回两个输出文件路径。"""
    configure_input_output(panel_path)
    data = prepare_data()
    model_rows = []
    coefficient_rows_all = []
    for outcome, window_label in OUTCOMES:
        _, regression_info, coefficient_rows = run_oib_regression(data, outcome, rhs_cols)
        metadata = {"结果表": "表3", "衡量口径": "成交量", "订单失衡窗口": window_label, "散户识别方法": "A股小单口径"}
        model_rows.append({**metadata, **regression_info})
        coefficient_rows_all.extend({**metadata, **row} for row in coefficient_rows)
    results = pd.DataFrame(model_rows)
    coefficients = pd.DataFrame(coefficient_rows_all)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    coefficients.to_csv(COEFFICIENTS_PATH, index=False, encoding="utf-8-sig")
    if RUN_INFO is not None:
        with (OUTPUT_DIR / "运行口径.json").open("w", encoding="utf-8") as file:
            json.dump(RUN_INFO, file, ensure_ascii=False, indent=2)
    print('表3：散户订单失衡（成交量基准规格）')
    print(results.to_string(index=False))
    print(f"结果已保存：{RESULTS_PATH}")
    print(f"全部系数明细已保存：{COEFFICIENTS_PATH}")
    del data
    gc.collect()
    return [RESULTS_PATH, COEFFICIENTS_PATH]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取指定面板，估计表3公司季度固定效应版本。")
    parser.add_argument("--panel", type=Path, default=None, help="最终面板路径；省略时沿用顶部默认输入和输出。")
    cli_args = parser.parse_args()
    main(panel_path=cli_args.panel)
