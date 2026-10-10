from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


ROOT = Path("/Users/qiyuewu/Desktop/mmm复现数据包")
MODULE_PATH = ROOT / "code/08_回归/表5_心理机制.py"
CACHE_PATH = ROOT / "work/表5低内存缓存/全样本.parquet"


# 加载现有的表5完整估计模块
module_name = "table5_psychological_mechanism"
spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)

if spec is None or spec.loader is None:
    raise ImportError(f"无法加载回归模块：{MODULE_PATH}")

table5 = importlib.util.module_from_spec(spec)
sys.modules[module_name] = table5

try:
    spec.loader.exec_module(table5)
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少运行依赖，请先在实际使用的 Python 环境中安装："
        "pyarrow、linearmodels。"
    ) from exc


# 强制检查：t至t+1必须是主窗口，t至t+10必须是补充窗口
expected_outcomes = [
    ("chAdjreturn_t_tp1_Recall_w", "[t, t+1]"),
    ("chAdjreturn_t_tp10_Recall_w", "[t, t+10]"),
]

actual_outcomes = [
    (item.column, item.label)
    for item in table5.OUTCOME_SPECS
]

if actual_outcomes != expected_outcomes:
    raise ValueError(
        f"收益窗口顺序不符合主、补充设定：{actual_outcomes}"
    )


# 从低内存缓存逐模型读取并估计34个模型
# 包括两个收益窗口、九类机制和三种SUE口径
main_results, coefficient_details = (
    table5.run_all_models_low_memory(CACHE_PATH)
)


# 合并三种SUE口径的主结果
results = pd.concat(
    [
        main_results[sue_id].copy()
        for sue_id in (1, 2, 3)
    ],
    ignore_index=True,
)


# 明确标记主窗口和补充窗口
results["窗口类型"] = results["面板"].map(
    {
        "A": "主窗口",
        "B": "补充窗口",
    }
)

results["_窗口排序"] = results["面板"].map(
    {
        "A": 0,
        "B": 1,
    }
)

results["_SUE排序"] = results["SUE口径"].map(
    {
        "SUE1": 1,
        "SUE2": 2,
        "SUE3": 3,
    }
)

results["_规模排序"] = results["规模组"].map(
    {
        "90plus": 0,
        "50_90": 1,
    }
)


# 主窗口排在最前面
results = (
    results
    .sort_values(
        [
            "_窗口排序",
            "机制编号",
            "_SUE排序",
            "_规模排序",
        ],
        kind="stable",
    )
    .drop(
        columns=[
            "_窗口排序",
            "_SUE排序",
            "_规模排序",
        ]
    )
    .reset_index(drop=True)
)


# 仅在控制台展示，不写出文件
show_columns = [
    "窗口类型",
    "收益窗口",
    "机制编号",
    "机制名称",
    "SUE口径",
    "规模组标签",
    "强关联提示系数",
    "其他关联提示系数",
    "系数差",
    "沃尔德p值",
    "回归样本量",
    "R方",
    "估计状态",
    "不可估计原因",
]

with pd.option_context(
    "display.max_rows", None,
    "display.max_columns", None,
    "display.width", 240,
):
    print(
        results
        .reindex(columns=show_columns)
        .to_string(index=False)
    )
