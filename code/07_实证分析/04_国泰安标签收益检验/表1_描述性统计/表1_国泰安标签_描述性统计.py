"""表1：国泰安标签调整收益与沪深300提示的描述性统计。

输入是沪深300最终回归面板、国泰安标签调整收益和订单失衡正式中间表。
国泰安标签调整收益须先由“图2_收益动态/01_生成国泰安标签基准与调整收益.py”生成。
收益按回归筛样前全样本、各窗口自身的1%和99%分位点缩尾。
表中样本来自沪深300最终回归面板，再按表2两窗及控制变量完整规则筛选；
不额外设置总市值1亿元或股价1元门槛。该面板与过去试算样本可能不同。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# 一、输入和输出文件。
project_dir = Path(__file__).resolve().parents[4]
panel_file = project_dir / "中间结果/04_合并数据/Pattern样本_最终回归面板_沪深300.csv"
return_file = project_dir / "中间结果/02_变量构造结果/1DGTW特征调整收益率/国泰安标签_沪深300_50窗调整收益.csv"
oib_file = project_dir / "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv"
figure_file = project_dir / "图表结果/04_国泰安标签收益检验/表1_描述性统计/表1_国泰安标签_沪深300提示_未加市值股价筛选.png"

if not return_file.is_file():
    raise FileNotFoundError(f"找不到国泰安标签调整收益：{return_file}\n请先运行图2的‘01_生成国泰安标签基准与调整收益.py’。")

# 二、读取正式沪深300面板；保留两窗原有收益、27项控制和提示变量完整的观测。
announcement_cols = ([f"ownEA_tm{i}" for i in range(10, 0, -1)] + ["ownEA"]
                     + [f"ownEA_tp{i}" for i in range(1, 11)]
                     + ["ownOtherEventER", "ownPerformanceForecastBrief"])
news_cols = ["aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV"]
original_return_cols = ["chAdjreturn_t_tp1_Recall_w", "chAdjreturn_t_tp10_Recall_w"]
sue_cols = [f"sue{i}CueLag1HS300{weight}_w" for i in (1, 2, 3) for weight in ("EW", "VW")]
panel_cols = (["StkcdRecall", "Trddt", "quarterLag1HS300", "nrCuesLag1HS300"]
              + announcement_cols + news_cols + original_return_cols + sue_cols)
numeric_cols = [col for col in panel_cols if col not in ("StkcdRecall", "Trddt")]
dtypes = {col: "float32" for col in numeric_cols}
dtypes["StkcdRecall"] = "int32"
panel = pd.read_csv(panel_file, usecols=panel_cols, dtype=dtypes, parse_dates=["Trddt"])
if panel.duplicated(["StkcdRecall", "Trddt"]).any():
    raise ValueError("沪深300最终面板存在重复的被提示公司和日期。")
panel = panel.loc[panel["Trddt"].between("2005-04-08", "2025-12-31")].copy()
complete_cols = ["quarterLag1HS300"] + announcement_cols + news_cols + original_return_cols
panel = panel.loc[np.isfinite(panel[complete_cols]).all(axis=1)].copy()
if not panel["quarterLag1HS300"].isin([0, 1]).all():
    raise ValueError("沪深300提示变量必须为0或1。")

# 三、合并国泰安标签收益，并要求两窗均有效。
official_return_cols = ["chAdjOfficial_t_tp1_w", "chAdjOfficial_t_tp10_w"]
returns = pd.read_csv(return_file, usecols=["StkcdRecall", "Trddt"] + official_return_cols,
                      dtype={"StkcdRecall": "int32"}, parse_dates=["Trddt"])
if returns.duplicated(["StkcdRecall", "Trddt"]).any():
    raise ValueError("国泰安标签调整收益存在重复的被提示公司和日期。")
panel = panel.merge(returns, on=["StkcdRecall", "Trddt"], how="left", validate="one_to_one", indicator=True)
if not panel["_merge"].eq("both").all():
    raise ValueError("国泰安收益表未覆盖全部面板公司日期，可能尚未完整写出，请核对。")
panel = panel.drop(columns="_merge")
del returns
panel = panel.loc[np.isfinite(panel[official_return_cols]).all(axis=1)].copy()

# 四、与表2第（6）列的固定效应处理一致，递归删除公司季度和日期单例。
panel["year_quarter"] = panel["Trddt"].dt.to_period("Q")
while len(panel):
    firm_quarter_n = panel.groupby(["StkcdRecall", "year_quarter"])["StkcdRecall"].transform("size")
    date_n = panel.groupby("Trddt")["StkcdRecall"].transform("size")
    keep = firm_quarter_n.gt(1) & date_n.gt(1)
    if keep.all():
        break
    panel = panel.loc[keep].copy()
if panel.empty:
    raise ValueError("共同有效样本为空，无法生成表1。")
print(f"表1共同样本：{len(panel):,} 行。", flush=True)

# 五、按公司和日期匹配订单失衡，在本表样本内按各窗口有效值缩尾。
oib_cols = ["OIBvol_avg_t_tp1", "OIBvol_avg_t_tp10"]
oib = pd.read_csv(oib_file, usecols=["Stkcd", "Trddt"] + oib_cols,
                  dtype={"Stkcd": "int32"}, parse_dates=["Trddt"])
oib = oib.rename(columns={"Stkcd": "StkcdRecall"})
if oib.duplicated(["StkcdRecall", "Trddt"]).any():
    raise ValueError("订单失衡表存在重复的公司和日期。")
panel = panel.merge(oib, on=["StkcdRecall", "Trddt"], how="left", validate="one_to_one")
del oib
for col in oib_cols:
    panel[col] = pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if panel[col].notna().sum() == 0:
        raise ValueError(f"本表没有有效的订单失衡数据：{col}")
    lower, upper = panel[col].quantile([0.01, 0.99])
    panel[col] = panel[col].clip(lower, upper)

# 六、计算描述性统计。提示数量和盈余惊喜只在沪深300提示日统计。
def describe_stats(values):
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return ["—"] * 8
    stats = [values.mean(), values.quantile(0.25), values.median(), values.quantile(0.75),
             values.std(ddof=1), values.min(), values.max()]
    return [f"{value:.4f}" for value in stats] + [f"{len(values):,}"]


cue_days = panel["quarterLag1HS300"].eq(1)
table_rows = [
    ("特征调整收益率（%），[t, t+1]", describe_stats(panel[official_return_cols[0]] * 100)),
    ("特征调整收益率（%），[t, t+10]", describe_stats(panel[official_return_cols[1]] * 100)),
    ("小单成交量订单失衡（%），[t, t+1]", describe_stats(panel[oib_cols[0]])),
    ("小单成交量订单失衡（%），[t, t+10]", describe_stats(panel[oib_cols[1]])),
    ("沪深300提示事件", describe_stats(panel["quarterLag1HS300"])),
    ("沪深300提示数量", describe_stats(panel.loc[cue_days, "nrCuesLag1HS300"])),
]
for sue_num in (1, 2, 3):
    for weight, label in (("EW", "等权"), ("VW", "市值加权")):
        col = f"sue{sue_num}CueLag1HS300{weight}_w"
        table_rows.append((f"盈余惊喜{sue_num}（{label}）", describe_stats(panel.loc[cue_days, col])))

# 七、参照主线表1绘制三线表。
font_path = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
font = font_manager.FontProperties(fname=str(font_path)) if font_path.is_file() else font_manager.FontProperties(family="sans-serif")
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(16, 8.8))
ax.axis("off")
ax.text(0.015, 0.965, "表 1", fontsize=18, fontweight="bold", fontproperties=font)
ax.text(0.015, 0.915, "描述性统计：国泰安标签与沪深300提示", fontsize=18, fontweight="bold", fontproperties=font)
ax.text(0.015, 0.875, "样本：沪深300最终回归面板，2005—2025年；按表2第（6）列规则筛选。",
        fontsize=10.5, color="#444444", fontproperties=font)
table = ax.table(
    cellText=[[name] + stats for name, stats in table_rows],
    colLabels=["变量", "均值", "第25百分位", "中位数", "第75百分位", "标准差", "最小值", "最大值", "样本量"],
    cellLoc="right", colLoc="right",
    colWidths=[0.25, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.105],
    bbox=[0.015, 0.23, 0.97, 0.62],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
for (row, col), cell in table.get_celld().items():
    cell.set_facecolor("white")
    cell.get_text().set_fontproperties(font)
    cell.get_text().set_ha("left" if col == 0 else "right")
    cell.visible_edges = "BT" if row == 0 else ("B" if row == len(table_rows) else "")
    cell.set_linewidth(1.0 if row in (0, len(table_rows)) else 0)

notes = [
    "注：国泰安收益按筛样前全样本逐窗1%和99%分位缩尾；不额外设置总市值1亿元或未复权股价1元门槛。",
    "样本要求原有收益、国泰安调整收益的两窗及27项控制有效，并递归删除公司季度和日期单例。",
    "提示数量和盈余惊喜仅按沪深300提示日统计；盈余惊喜沿用最终面板已有的缩尾值。",
    "订单失衡在本表样本各窗口的有效观测中按1%和99%分位缩尾；与主线表1订单失衡回归样本不完全相同。",
]
for row, text in enumerate(notes):
    ax.text(0.015, 0.19 - row * 0.035, text, fontsize=9.5, color="#444444", fontproperties=font)
figure_file.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_file, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"表1图片已保存：{figure_file}")
