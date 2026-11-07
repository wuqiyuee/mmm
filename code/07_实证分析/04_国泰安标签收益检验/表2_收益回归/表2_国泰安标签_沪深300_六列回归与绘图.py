"""表2：国泰安标签调整收益与沪深300提示的六列回归。

输入为正式沪深300最终回归面板和国泰安标签50窗调整收益表。后者由
“图2_收益动态/01_生成国泰安标签基准与调整收益.py”生成，需先自行运行。
国泰安调整收益按回归筛样前全样本、各窗口自身的1%和99%分位点缩尾。
两个窗口要求原有收益、国泰安收益与27项控制共同有效；各列再处理自身单例。
正式面板本身已经过样本筛选，与早先加入填充日起点的历史试算不一定相同。
回归和绘图使用的字段、路径和计算步骤均写在本文件，不读取工作目录缓存。
"""

from pathlib import Path
import json
import os
import sys
import time

sys.dont_write_bytecode = True
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.csgraph import connected_components


# 一、读取正式面板及国泰安标签调整收益。
root = Path(__file__).resolve().parents[4]
panel_path = root / "中间结果/04_合并数据/Pattern样本_最终回归面板_沪深300.csv"
return_path = root / "中间结果/02_变量构造结果/1DGTW特征调整收益率/国泰安标签_沪深300_50窗调整收益.csv"
output_path = root / "图表结果/04_国泰安标签收益检验/表2_收益回归/表2_国泰安标签_沪深300_正式面板.png"

if not panel_path.is_file():
    raise FileNotFoundError(f"找不到沪深300最终回归面板：{panel_path}")
if not return_path.is_file():
    raise FileNotFoundError(
        f"找不到国泰安标签调整收益表：{return_path}\n"
        "请先运行图2目录中的“01_生成国泰安标签基准与调整收益.py”。"
    )

started = time.monotonic()
ea_cols = [f"ownEA_tm{i}" for i in range(10, 0, -1)] + ["ownEA"] + [f"ownEA_tp{i}" for i in range(1, 11)]
other_cols = ["ownOtherEventER", "ownPerformanceForecastBrief", "aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV"]
control_cols = ea_cols + other_cols
old_cols = ["chAdjreturn_t_tp1_Recall_w", "chAdjreturn_t_tp10_Recall_w"]
official_cols = ["chAdjOfficial_t_tp1_w", "chAdjOfficial_t_tp10_w"]
keys = ["StkcdRecall", "Trddt"]
panel_cols = keys + ["quarterLag1HS300"] + control_cols + old_cols
types = {"StkcdRecall": "int32", **{name: "float32" for name in panel_cols if name not in keys}}

# 只读取研究期和回归所需列，节省读取宽面板所需的内存。
parts = []
for part in pd.read_csv(panel_path, usecols=panel_cols, dtype=types, parse_dates=["Trddt"], chunksize=300_000):
    part = part.loc[part["Trddt"].between("2005-04-08", "2025-12-31")]
    if not part.empty:
        parts.append(part)
if not parts:
    raise ValueError("沪深300最终回归面板没有研究期样本。")
panel = pd.concat(parts, ignore_index=True)
del parts
if panel.duplicated(keys).any():
    raise ValueError("沪深300最终面板的被提示公司和日期不是唯一键。")

returns = pd.read_csv(return_path, usecols=keys + official_cols,
                      dtype={"StkcdRecall": "int32", **{col: "float64" for col in official_cols}},
                      parse_dates=["Trddt"])
if returns.duplicated(keys).any():
    raise ValueError("国泰安标签调整收益表的被提示公司和日期不是唯一键。")
panel = panel.merge(returns, on=keys, how="left", validate="one_to_one", indicator=True)
if not panel["_merge"].eq("both").all():
    raise ValueError("国泰安收益表未覆盖全部面板公司日期，可能尚未完整写出，请核对。")
panel = panel.drop(columns="_merge")
del returns

# 两窗和全部控制均有效，供六列共同使用；不追加市值或股价门槛。
required = ["quarterLag1HS300"] + control_cols + old_cols + official_cols
valid = np.isfinite(panel[required].to_numpy(dtype="float64")).all(axis=1)
panel = panel.loc[valid].copy()
if panel.empty:
    raise ValueError("两窗收益和全部控制共同有效的样本为空，请检查输入口径。")
if not panel["quarterLag1HS300"].isin([0, 1]).all():
    raise ValueError("沪深300提示变量必须是0或1。")
panel["quarter"] = panel["Trddt"].dt.to_period("Q")
all_firms = np.sort(panel["StkcdRecall"].unique())
all_dates = np.sort(panel["Trddt"].unique())
print(f"共同有效样本：{len(panel):,} 条；正式面板可能与历史试算样本不同。", flush=True)


# 二、逐季度处理固定效应。公司在季度内的效应对应公司×年度季度效应。
def double_fe(values, firm, date):
    """同时吸收公司季度和日期固定效应，返回处理后的矩阵和固定效应秩。"""
    nf, nd = int(firm.max() + 1), int(date.max() + 1)
    row = np.arange(len(firm))
    firm_matrix = sparse.csr_matrix((np.ones(len(firm)), (firm, row)), shape=(nf, len(firm)))
    date_matrix = sparse.csr_matrix((np.ones(len(firm)), (date, row)), shape=(nd, len(firm)))
    counts = sparse.csr_matrix((np.ones(len(firm)), (firm, date)), shape=(nf, nd))
    firm_sizes = np.bincount(firm)
    components, labels = connected_components((counts.T @ counts).tocsr(), directed=False)
    reference = np.unique(labels, return_index=True)[1]
    free_dates = np.setdiff1d(np.arange(nd), reference)
    schur = np.diag(np.bincount(date)) - (counts.T @ counts.multiply(1 / firm_sizes[:, None])).toarray()
    schur = (schur + schur.T) / 2
    residual = values.copy()
    residual -= ((firm_matrix @ residual) / firm_sizes[:, None])[firm]
    effect = np.zeros((nd, values.shape[1]))
    effect[free_dates] = cho_solve(cho_factor(schur[np.ix_(free_dates, free_dates)], lower=True, check_finite=False),
                                   (date_matrix @ residual)[free_dates], check_finite=False)
    residual -= effect[date]
    residual += ((counts @ effect) / firm_sizes[:, None])[firm]
    return residual, nf + nd - components


# 固定效应类型：0无、1日期、2公司季度、3公司季度和日期。
configs = [(0, 0), (21, 0), (27, 0), (27, 1), (27, 2), (27, 3)]
models = []
for controls, fe in configs:
    columns = 1 + controls + (fe == 0)
    models.append({
        "xx": np.zeros((columns, columns)), "xy": np.zeros((columns, 2)),
        "n": 0, "rank": 0, "firms": np.zeros(len(all_firms), dtype=bool),
        "dates": np.zeros(len(all_dates), dtype=bool),
        "firm_scores": np.zeros((len(all_firms), 2)), "date_meat": np.zeros(2),
        "intersection": np.zeros(2),
    })

prepared = []
for quarter, block in panel.groupby("quarter", sort=True):
    firms = block["StkcdRecall"].to_numpy(dtype="int32")
    dates = block["Trddt"].to_numpy(dtype="datetime64[ns]")
    values = block[official_cols + ["quarterLag1HS300"] + control_cols].to_numpy(dtype="float64")
    quarter_groups = {}
    for fe in range(4):
        pos = np.arange(len(block))
        if fe:
            while len(pos):
                _, firm_id = np.unique(firms[pos], return_inverse=True)
                _, date_id = np.unique(dates[pos], return_inverse=True)
                keep = np.ones(len(pos), dtype=bool)
                if fe in (2, 3):
                    keep &= np.bincount(firm_id)[firm_id] > 1
                if fe in (1, 3):
                    keep &= np.bincount(date_id)[date_id] > 1
                if keep.all():
                    break
                pos = pos[keep]
        if not len(pos):
            continue
        _, firm_id = np.unique(firms[pos], return_inverse=True)
        _, date_id = np.unique(dates[pos], return_inverse=True)
        v = values[pos].copy()
        if fe == 3:
            v, rank = double_fe(v, firm_id, date_id)
        elif fe:
            group = firm_id if fe == 2 else date_id
            counts = np.bincount(group)
            sums = np.zeros((len(counts), v.shape[1]))
            np.add.at(sums, group, v)
            v -= sums[group] / counts[group, None]
            rank = len(counts)
        else:
            rank = 0
        quarter_groups[fe] = {
            "v": v, "firm": np.searchsorted(all_firms, firms[pos]),
            "date": np.searchsorted(all_dates, dates[pos]), "date_id": date_id,
            "rank": rank,
        }
    prepared.append(quarter_groups)

    # 不逐季度分别回归；累加各季度矩阵后统一估计系数。
    for model, (controls, fe) in zip(models, configs):
        if fe not in quarter_groups:
            continue
        q = quarter_groups[fe]
        y = q["v"][:, :2]
        x = q["v"][:, 2:3 + controls]
        if fe == 0:
            x = np.column_stack([x, np.ones(len(x))])
        model["xx"] += x.T @ x
        model["xy"] += x.T @ y
        model["n"] += len(x)
        model["rank"] += q["rank"]
        model["firms"][q["firm"]] = True
        model["dates"][q["date"]] = True
    print(f"已处理 {quarter}，用时 {time.monotonic() - started:.1f} 秒。", flush=True)
del panel


# 三、求系数，按公司与日期双向聚类计算标准误。
for model in models:
    if model["n"] == 0 or not np.isfinite(np.linalg.cond(model["xx"])) or np.linalg.cond(model["xx"]) >= 1e12:
        raise ValueError("某列样本为空或解释变量矩阵接近奇异，无法估计。")
    model["inverse"] = np.linalg.inv(model["xx"])
    model["beta"] = np.linalg.solve(model["xx"], model["xy"])

for quarter_groups in prepared:
    for model, (controls, fe) in zip(models, configs):
        if fe not in quarter_groups:
            continue
        q = quarter_groups[fe]
        y = q["v"][:, :2]
        x = q["v"][:, 2:3 + controls]
        if fe == 0:
            x = np.column_stack([x, np.ones(len(x))])
        residual = y - x @ model["beta"]
        score = (x @ model["inverse"][:, 0])[:, None] * residual
        for h in range(2):
            model["firm_scores"][:, h] += np.bincount(q["firm"], weights=score[:, h], minlength=len(all_firms))
            date_scores = np.bincount(q["date_id"], weights=score[:, h])
            model["date_meat"][h] += date_scores @ date_scores
        model["intersection"] += np.square(score).sum(axis=0)

results = []
for model, (controls, fe) in zip(models, configs):
    n = model["n"]
    nf = int(model["firms"].sum())
    nd = int(model["dates"].sum())
    k = 1 + controls
    if nf < 2 or nd < 2 or n <= k + 1:
        raise ValueError("某列样本或聚类组不足，无法计算标准误。")
    variance = (n - 1) / (n - k - 1) * (
        nf / (nf - 1) * np.square(model["firm_scores"]).sum(axis=0)
        + nd / (nd - 1) * model["date_meat"] - n / (n - 1) * model["intersection"]
    )
    df = n - model["rank"] - k - (fe == 0)
    if not np.isfinite(variance).all() or (variance <= 0).any() or df <= 0:
        raise ValueError("某列聚类方差或残差自由度无效。")
    beta = model["beta"][0]
    se = np.sqrt(variance)
    probability = 2 * stats.t.sf(np.abs(beta / se), df)
    result = {"样本量": n, "系数百分点": (beta * 100).tolist(),
              "标准误百分点": (se * 100).tolist(), "概率": probability.tolist()}
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


# 四、绘制两窗六列图片。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
font = font_manager.FontProperties(fname=str(font_path)) if font_path.exists() else font_manager.FontProperties(family="sans-serif")
plt.rcParams["axes.unicode_minus"] = False
coefficients, standard_errors, sample_sizes = [], [], []
for window in range(2):
    for result in results:
        p = result["概率"][window]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        coefficients.append(f"{result['系数百分点'][window]:.4f}{stars}")
        standard_errors.append(f"({result['标准误百分点'][window]:.4f})")
        sample_sizes.append(f"{result['样本量']:,}")
table_rows = [
    ["沪深300提示"] + coefficients,
    ["（聚类标准误）"] + standard_errors,
    ["自身盈余公告控制"] + ["否", "是", "是", "是", "是", "是"] * 2,
    ["其他控制变量"] + ["否", "否", "是", "是", "是", "是"] * 2,
    ["日期固定效应"] + ["否", "否", "否", "是", "否", "是"] * 2,
    ["被提示公司×年度季度固定效应"] + ["否", "否", "否", "否", "是", "是"] * 2,
    ["样本量"] + sample_sizes,
]
fig, ax = plt.subplots(figsize=(23, 7.7))
ax.axis("off")
ax.text(0.015, 0.97, "表 2", fontsize=18, fontproperties=font, transform=ax.transAxes)
ax.text(0.015, 0.92, "线索事件与被提示公司收益", fontsize=18, fontproperties=font, transform=ax.transAxes)
ax.text(0.015, 0.875, "因变量：国泰安标签特征调整收益率；系数与标准误单位为百分点", fontsize=11, fontproperties=font, transform=ax.transAxes)
ax.text(0.46, 0.825, "收益窗口 [t, t+1]", ha="center", fontsize=11, fontproperties=font, transform=ax.transAxes)
ax.text(0.81, 0.825, "收益窗口 [t, t+10]", ha="center", fontsize=11, fontproperties=font, transform=ax.transAxes)
table = ax.table(cellText=table_rows, colLabels=["变量"] + [f"({i})" for i in range(1, 7)] * 2,
                 cellLoc="center", colLoc="center", colWidths=[0.28] + [0.06] * 12, bbox=[0.015, 0.35, 0.97, 0.45])
table.auto_set_font_size(False)
table.set_fontsize(10)
for (row, col), cell in table.get_celld().items():
    cell.set_facecolor("white")
    cell.get_text().set_fontproperties(font)
    cell.get_text().set_ha("left" if col == 0 else "center")
    cell.visible_edges = "BT" if row == 0 else "T" if row == 3 else "B" if row == len(table_rows) else ""
    cell.set_linewidth(0.8 if row in (0, 3, len(table_rows)) else 0)
notes = [
    "注：国泰安收益按筛样前全样本逐窗1%和99%分位缩尾；未追加总市值1亿元和股价1元门槛。",
    "样本要求旧收益及国泰安调整收益的两种窗口和全部27项控制均有效；正式面板已按其原有规则筛选。",
    "第（6）列含被提示公司×年度季度和日期固定效应；括号内为公司和日期双向聚类标准误。",
    "*、**、***分别表示双侧检验在10%、5%、1%水平显著；本表结果可能与历史试算不同。",
]
for i, note in enumerate(notes):
    ax.text(0.015, 0.30 - i * 0.035, note, fontsize=10, color="#444444", fontproperties=font, transform=ax.transAxes)
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"表2图片已保存：{output_path}；用时 {time.monotonic() - started:.1f} 秒。", flush=True)
