"""表1描述性统计的交互式探索脚本。

本脚本供 VS Code Interactive Window（交互式窗口）逐单元执行，
复用“估计回归.py”和公共模块中的参数与计算函数。
默认只读取和整理数据，不自动生成图片；最后一个单元可按需执行绘图。
"""

# %% 00. 导入公共模块和参数
from pathlib import Path
import gc
import sys

import pandas as pd
from IPython.display import Image, display


TASK_DIR = Path(
    "/Volumes/苹果资料/mmm复现数据包/code/07_实证分析/01_主线图表/表1_描述性统计"
)
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from 估计回归 import (  # noqa: E402
    END_YEAR,
    OIB_PATH,
    PATTERN_PATH,
    START_YEAR,
    TABLE1_IMAGE_PATH,
    export_table1_image,
    main_cols,
    main_dtype,
    return_cols,
    rhs_cols,
    run_oib_regression,
    sue_cols,
    summary_statistics,
    winsorize_1pct,
)


# %% 01. 读取模式公司回归面板
panel = pd.read_csv(
    PATTERN_PATH,
    usecols=main_cols,
    dtype=main_dtype,
    parse_dates=["Trddt"],
)

panel["StkcdRecall"] = (
    panel["StkcdRecall"].str.strip().str.zfill(6)
)
panel = panel.loc[
    panel["year"].between(START_YEAR, END_YEAR)
].copy()

if panel.duplicated(["StkcdRecall", "Trddt"]).any():
    raise ValueError("模式公司面板的 StkcdRecall + Trddt 不是唯一键。")

print("模式公司面板：", panel.shape)
print("变量数量：", len(panel.columns))
display(panel.head())


# %% 02. 读取订单失衡数据
oib_raw_cols = [
    "OIBvol_avg_t_tp1",
    "OIBvol_avg_t_tp10",
]

oib = pd.read_csv(
    OIB_PATH,
    usecols=["Stkcd", "Trddt"] + oib_raw_cols,
    dtype={
        "Stkcd": "string",
        "OIBvol_avg_t_tp1": "float32",
        "OIBvol_avg_t_tp10": "float32",
    },
    parse_dates=["Trddt"],
)

oib["Stkcd"] = oib["Stkcd"].str.strip().str.zfill(6)
oib = oib.rename(columns={"Stkcd": "StkcdRecall"})

if oib.duplicated(["StkcdRecall", "Trddt"]).any():
    raise ValueError("订单失衡表的 StkcdRecall + Trddt 不是唯一键。")

print("订单失衡数据：", oib.shape)
display(oib.head())


# %% 03. 合并主面板和订单失衡数据
panel = panel.merge(
    oib,
    on=["StkcdRecall", "Trddt"],
    how="left",
    validate="one_to_one",
)

print("合并后面板：", panel.shape)
print(
    "OIB tp1 缺失率：",
    panel["OIBvol_avg_t_tp1"].isna().mean(),
)
print(
    "OIB tp10 缺失率：",
    panel["OIBvol_avg_t_tp10"].isna().mean(),
)
display(panel.head(20))

del oib
gc.collect()


# %% 04. 订单失衡缩尾和类型整理
oib_w_cols = []
for col in oib_raw_cols:
    w_col = f"{col}_w"
    panel[w_col] = winsorize_1pct(panel[col]).astype("float32")
    oib_w_cols.append(w_col)

panel = panel.drop(columns=oib_raw_cols)
panel["StkcdRecall"] = panel["StkcdRecall"].astype("category")
panel["yearQuarter"] = panel["yearQuarter"].astype("category")

print("缩尾后的订单失衡变量：", oib_w_cols)
display(panel[oib_w_cols].describe())


# %% 05. 描述性统计
description_rows = {}

for col in return_cols:
    description_rows[f"{col}_百分比"] = summary_statistics(
        panel[col] * 100
    )

description_rows["quarterLag1size90plus"] = summary_statistics(
    panel["quarterLag1size90plus"]
)

cue_sample = panel["quarterLag1size90plus"].eq(1)
description_rows["nrCuesLag1S90p"] = summary_statistics(
    panel.loc[cue_sample, "nrCuesLag1S90p"]
)

for col in sue_cols:
    description_rows[col] = summary_statistics(
        panel.loc[cue_sample, col]
    )

description_table = pd.DataFrame.from_dict(
    description_rows,
    orient="index",
)

display(description_table)


# %% 06. 订单失衡回归核对
regression_rows = []

for outcome in oib_w_cols:
    stats, regression_info, _ = run_oib_regression(
        panel,# 当前数据表
        outcome,# 当前回归的因变量，例如 OIBvol_avg_t_tp1_w
        rhs_cols,# 自变量和控制变量列表
    )
    # 函数最后返回 outcome_stats：当前因变量的描述性统计；
    #  regression_info：当前回归的总体结果；
    #  coefficient_rows：详细的变量系数结果，这里暂时不用。
    # _：第三个结果暂时不用，所以用下划线接收。
    description_rows[outcome] = stats
    regression_rows.append(regression_info)

description_table = pd.DataFrame.from_dict(
    description_rows,
    orient="index",
)# 把回归结果转换成数据表，因为是字典套字典，所以需要加上from_dict(）
regression_check = pd.DataFrame(regression_rows)
# 单纯的字典组成的列表，每个字典就是一行

display(description_table)
display(regression_check)


# %% 07. 按需生成表1图片
# 只有明确需要更新图片时才执行本单元。
export_table1_image(description_table, TABLE1_IMAGE_PATH)
print(f"表1图片已输出：{TABLE1_IMAGE_PATH}")
# 读取刚生成的图片，在交互式窗口中显示
display(
    Image(
        filename=str(TABLE1_IMAGE_PATH),
        width=1200,
    )
)
# %%
