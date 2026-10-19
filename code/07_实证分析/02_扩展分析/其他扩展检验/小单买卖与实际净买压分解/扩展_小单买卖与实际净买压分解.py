# 本文件由旧零散回归目录迁移；仅调整项目定位、公共工具引用和任务输出位置。
"""独立检验提示后的小单订单失衡来自买入增加、卖出减少还是相对比例变化。

默认2005—2025年，使用原模式公司最终面板、27项控制、日期和公司季度
固定效应，公司及日期双向聚类。只读复用“扩展_散户交易占比异质性.py”
的季度精确吸收、线性方程求解及基础工具，不运行该脚本的主程序。
不修改任何现有脚本或输入。直接运行只打印；--save-dir（保存目录）
在指定路径新建时间戳结果目录。大型中间矩阵只在work/临时保存并自动清理。

事前基线固定为市场交易日t-20到t-1，20日必须全部有效，停牌和缺失不填0。
目标窗口为[t,t+1]或[t,t+10]，每一天必须满足现有窗口资格和原交易值有效。
流通市值严格匹配t-3；原日行情Dsmvosd单位为千元，乘1000转换为元。
分子为主动小单交易，不是账户身份下的全部散户交易或真实资金净流入。

对于量/额分别令f(B,S)=100*(B-S)/(B+S)，B0/S0为事前20日小单均值。
逐目标日对称分解：
  买贡献=[f(B,S0)-f(B0,S0)+f(B,S)-f(B0,S)]/2；
  卖贡献=[f(B0,S)-f(B0,S0)+f(B,S)-f(B,S0)]/2。
逐行、窗口平均及同样本回归系数均满足：目标失衡=基线失衡+买贡献+卖贡献。
这是会计分解，不能据此确认买卖变化的因果来源。基线本身与提示相关时
必须保留基线系数，不能用买卖贡献之和替代原目标失衡系数。

规模端用窗口日均值减事前日均值：买、卖及净买入变化均除以事前20日
同一只股票的全部主动成交量/金额均值。另报告万元及相对t-3流通市值
的累计净买入额。比率均报告百分点，市值比率另给基点，明确日均和累计。
不分别缩尾分解指标，以保留恒等式；附加预先固定的共同样本稳健性：
剔除未来日均总主动成交量或成交额相对事前均值高于各自99分位的整行。
同时对可用样本成交量失衡窗口上下各1%缩尾作对照，并比较共同样本收益。
全部检验未作跨指标多重检验调整；不存在把不显著直接解释为经济量很小。
"""

from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "2")

import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats


def find_project_root() -> Path:
    """沿脚本祖先寻找项目标识，迁移目录后不依赖固定父目录层数。"""
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "AGENTS.md").is_file()
            and (candidate / "code").is_dir()
            and (candidate / "中间结果").is_dir()
        ):
            return candidate
    raise FileNotFoundError("未找到同时包含项目说明、代码和中间结果的项目根目录。")


ROOT = find_project_root()
TASK_OUTPUT_DIR = ROOT / "图表结果/02_扩展分析/其他扩展检验/小单买卖与实际净买压分解"
HELPER_PATH = (ROOT / "code/07_实证分析/02_扩展分析/其他扩展检验/散户交易占比异质性/扩展_散户交易占比异质性.py")
spec = importlib.util.spec_from_file_location("flow_regression_tools", HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
log, RHS, CUE = helper.log, helper.RHS, helper.CUE
INPUTS = {
    "主面板": ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
    "交易原始表": ROOT / "中间结果/00_合并原始数据/CSMAR_个股买卖不平衡指标表_日_合并结果.csv",
    "目标日资格": ROOT / "中间结果/01_清洗后样本/窗口变量目标日资格索引_基础清洗含停牌填充_td.csv",
    "交易日历": ROOT / "中间结果/01_清洗后样本/市场交易日历_td.csv",
    "日行情": ROOT / "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv",
}
SOURCE_COLS = ["B_Volume_S", "S_Volume_S", "B_Volume", "S_Volume", "B_Amount_S", "S_Amount_S", "B_Amount", "S_Amount"]
METRICS = [
    ("oib_v", "目标成交量订单失衡", "百分点"),
    ("oib_v_w", "目标成交量订单失衡_可用样本缩尾对照", "百分点"),
    ("base_v", "事前成交量订单失衡基线", "百分点"),
    ("buy_v", "成交量订单失衡的买入变化贡献", "百分点"),
    ("sell_v", "成交量订单失衡的卖出变化贡献", "百分点"),
    ("oib_a", "目标成交金额订单失衡", "百分点"),
    ("base_a", "事前成交金额订单失衡基线", "百分点"),
    ("buy_a", "成交金额订单失衡的买入变化贡献", "百分点"),
    ("sell_a", "成交金额订单失衡的卖出变化贡献", "百分点"),
    ("db_v", "小单买入日均量变化占事前总主动成交量", "百分点"),
    ("ds_v", "小单卖出日均量变化占事前总主动成交量", "百分点"),
    ("dn_v", "小单净买入日均量变化占事前总主动成交量", "百分点"),
    ("db_a", "小单买入日均额变化占事前总主动成交额", "百分点"),
    ("ds_a", "小单卖出日均额变化占事前总主动成交额", "百分点"),
    ("dn_a", "小单净买入日均额变化占事前总主动成交额", "百分点"),
    ("db_yuan", "小单买入日均金额变化", "万元"),
    ("ds_yuan", "小单卖出日均金额变化", "万元"),
    ("dn_yuan", "小单净买入日均金额变化", "万元"),
    ("net_cap", "目标窗口累计小单净买入额占事前流通市值", "基点"),
    ("dnet_cap", "相对事前基线的累计小单净买入额变化占事前流通市值", "基点"),
    ("net_total", "目标小单日均净买入额占事前总主动成交额", "百分点"),
    ("ret", "特征调整收益", "百分点"),
]
NAMES = [x[0] for x in METRICS]
IDX = {name: i for i, name in enumerate(NAMES)}


def firm_td_key(firm: np.ndarray, td: np.ndarray) -> np.ndarray:
    if np.any(td < -9999) or np.any(td >= 10000):
        raise ValueError("交易日序号超出编码范围。")
    return firm.astype("int64") * 20000 + td.astype("int64") + 10000


def read_inputs(args: argparse.Namespace, audit: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = pd.read_csv(INPUTS["交易日历"], usecols=["Trddt", "td"], parse_dates=["Trddt"]).drop_duplicates()
    if calendar.Trddt.duplicated().any() or calendar.td.duplicated().any():
        raise ValueError("市场交易日历存在日期或交易日序号冲突。")
    date_to_td = pd.Series(calendar.td.to_numpy(), index=calendar.Trddt)
    cols = ["StkcdRecall", "Trddt", "td", *RHS, *helper.RETURNS.values()]
    pieces = []
    log("读取原主回归面板及控制变量")
    for number, chunk in enumerate(pd.read_csv(
        INPUTS["主面板"], usecols=cols, parse_dates=["Trddt"], chunksize=args.chunk_size,
        dtype={"StkcdRecall": "int32", "td": "int32", **{c: "float32" for c in RHS + list(helper.RETURNS.values())}},
    ), 1):
        chunk = chunk.loc[chunk.Trddt.dt.year.between(args.start_year, args.end_year)].copy()
        if chunk.empty:
            continue
        chunk["q"] = helper.quarter_codes(chunk.Trddt)
        chunk["day"] = helper.day_codes(chunk.Trddt)
        pieces.append(chunk.rename(columns={"StkcdRecall": "firm"}).drop(columns="Trddt"))
        if number % 8 == 0:
            log(f"主面板已读取第{number}块")
    panel = pd.concat(pieces, ignore_index=True)
    del pieces
    if panel.duplicated(["firm", "td"]).any() or not panel[CUE].isin([0, 1]).all():
        raise ValueError("主面板存在重复键或提示变量不是零一。")
    audit.append({"阶段": "主面板", "项目": "研究期公司日", "数值": len(panel)})
    lower_td, upper_td = int(panel.td.min()) - 23, int(panel.td.max()) + 10
    firms = panel.firm.unique()
    eligible = pd.read_csv(INPUTS["目标日资格"], usecols=["Stkcd", "td"], dtype={"Stkcd": "int32", "td": "int32"})
    eligible = eligible.loc[eligible.td.between(lower_td, upper_td) & eligible.Stkcd.isin(firms)]
    eligible_keys = firm_td_key(eligible.Stkcd.to_numpy(), eligible.td.to_numpy())
    eligible_keys.sort()
    if np.any(eligible_keys[1:] == eligible_keys[:-1]):
        raise ValueError("目标日资格索引重复。")
    del eligible
    pieces, invalid_count, subset_bad_count = [], 0, 0
    log("读取小单及全股主动成交量、成交额，统一检查有效性")
    for number, chunk in enumerate(pd.read_csv(
        INPUTS["交易原始表"], usecols=["Stkcd", "Trddt", *SOURCE_COLS],
        dtype={"Stkcd": "int32", **{c: "float64" for c in SOURCE_COLS}},
        parse_dates=["Trddt"], chunksize=args.chunk_size,
    ), 1):
        td = date_to_td.reindex(chunk.Trddt).to_numpy()
        keep = np.isfinite(td) & (td >= lower_td) & (td <= upper_td) & chunk.Stkcd.isin(firms).to_numpy()
        chunk = chunk.loc[keep].copy()
        if chunk.empty:
            continue
        td = td[keep].astype("int32")
        candidate = firm_td_key(chunk.Stkcd.to_numpy(), td)
        _, matched = helper.matched_positions(eligible_keys, candidate)
        values = chunk[SOURCE_COLS].to_numpy()
        valid = matched & np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
        total_v, total_a = values[:, 2] + values[:, 3], values[:, 6] + values[:, 7]
        small_v, small_a = values[:, 0] + values[:, 1], values[:, 4] + values[:, 5]
        valid &= (total_v > 0) & (total_a > 0) & (small_v > 0) & (small_a > 0)
        invalid_count += int((matched & ~valid).sum())
        bad = (small_v > total_v + np.maximum(.01, total_v * 1e-9)) | (small_a > total_a + np.maximum(.01, total_a * 1e-9))
        subset_bad_count += int((valid & bad).sum())
        valid &= ~bad
        pieces.append(pd.DataFrame({
            "firm": chunk.Stkcd.to_numpy()[valid], "td": td[valid],
            "bv": values[valid, 0], "sv": values[valid, 1], "tv": total_v[valid],
            "ba": values[valid, 4], "sa": values[valid, 5], "ta": total_a[valid],
        }))
        if number % 8 == 0:
            log(f"交易原始表已读取第{number}块")
    raw = pd.concat(pieces, ignore_index=True).sort_values(["firm", "td"], kind="stable").reset_index(drop=True)
    del pieces, eligible_keys
    if raw.duplicated(["firm", "td"]).any():
        raise ValueError("有效原始交易表公司日期重复。")
    audit.extend([
        {"阶段": "原始交易", "项目": "目标资格内量额缺失负值或分母非正行", "数值": invalid_count},
        {"阶段": "原始交易", "项目": "小单合计超过全部主动合计行", "数值": subset_bad_count},
        {"阶段": "原始交易", "项目": "共同有效股票日", "数值": len(raw)},
    ])
    # 市值严格对应事件日前第三个市场交易日，不以最近一条记录替代。
    desired = firm_td_key(panel.firm.to_numpy(), panel.td.to_numpy() - 3)
    order = np.argsort(desired)
    desired_sorted = desired[order]
    caps = np.full(len(panel), np.nan)
    matched_seen = np.zeros(len(panel), dtype=bool)
    log("读取日行情，严格匹配事件前第三个交易日流通市值")
    for number, chunk in enumerate(pd.read_csv(
        INPUTS["日行情"], usecols=["Stkcd", "Trddt", "Dsmvosd"], parse_dates=["Trddt"],
        dtype={"Stkcd": "int32", "Dsmvosd": "float64"}, chunksize=args.chunk_size,
    ), 1):
        td = date_to_td.reindex(chunk.Trddt).to_numpy()
        valid = np.isfinite(td) & (td >= lower_td) & (td <= upper_td)
        if not valid.any():
            continue
        key = firm_td_key(chunk.Stkcd.to_numpy()[valid], td[valid].astype("int32"))
        positions, matched = helper.matched_positions(desired_sorted, key)
        target = order[positions[matched]]
        if len(np.unique(target)) != len(target) or matched_seen[target].any():
            raise ValueError("日行情公司日期重复。")
        cap_values = chunk.Dsmvosd.to_numpy()[valid][matched] * 1000
        caps[target] = np.where(cap_values > 0, cap_values, np.nan)
        matched_seen[target] = True
        if number % 12 == 0:
            log(f"日行情已读取第{number}块")
    panel["cap_tm3"] = caps
    del desired, desired_sorted, order, matched_seen
    gc.collect()
    return panel, raw


def f(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 100 * (b - s) / (b + s)


def construct_outcomes(panel: pd.DataFrame, raw: pd.DataFrame, workspace: Path, audit: list[dict]) -> dict[int, np.memmap]:
    """按股票建立稠密市场交易日轴，历史、未来窗口严格按日序号匹配。"""
    n = len(panel)
    output = {h: np.lib.format.open_memmap(workspace / f"结局_{h}.npy", mode="w+", dtype="float64", shape=(n, len(METRICS))) for h in (1, 10)}
    ratios = {h: np.full((n, 2), np.nan, dtype="float32") for h in (1, 10)}
    for values in output.values():
        values[:] = np.nan
    raw_positions = raw.groupby("firm", sort=False).indices
    panel_positions = panel.groupby("firm", sort=False).indices
    max_row_error = 0.0
    log("构造事前20日基线、目标窗口和买卖两端的精确贡献")
    for number, (firm, positions) in enumerate(panel_positions.items(), 1):
        source_positions = raw_positions.get(firm)
        if source_positions is None:
            continue
        source = raw.iloc[source_positions]
        start = min(int(source.td.min()), int(panel.td.iloc[positions].min()) - 20)
        stop = max(int(source.td.max()), int(panel.td.iloc[positions].max()) + 10)
        dense = np.full((stop - start + 1, 6), np.nan)
        dense[source.td.to_numpy() - start] = source[["bv", "sv", "tv", "ba", "sa", "ta"]].to_numpy()
        valid_dense = np.isfinite(dense).all(axis=1)
        cs = np.vstack([np.zeros(6), np.cumsum(np.nan_to_num(dense, nan=0.0), axis=0)])
        counts = np.r_[0, np.cumsum(valid_dense)]
        target = panel.td.iloc[positions].to_numpy() - start
        base = (cs[target] - cs[target - 20]) / 20
        base_valid = (counts[target] - counts[target - 20]) == 20
        base_valid &= (base[:, [0, 1, 3, 4]] > 0).all(axis=1)
        cap = panel.cap_tm3.iloc[positions].to_numpy()
        for h in (1, 10):
            length = h + 1
            y = np.full((len(positions), len(METRICS)), np.nan)
            future = (cs[target + length] - cs[target]) / length
            keep = base_valid & ((counts[target + length] - counts[target]) == length) & np.isfinite(cap) & (cap > 0)
            if not keep.any():
                continue
            at = np.flatnonzero(keep)
            bv, sv, tv, ba, sa, ta = base[at].T
            fbv, fsv, ftv, fba, fsa, fta = future[at].T
            for offset, suffix, b0, s0 in [(0, "v", bv, sv), (3, "a", ba, sa)]:
                target_oib = np.zeros(len(at))
                buy_contribution = np.zeros(len(at))
                sell_contribution = np.zeros(len(at))
                base_oib = f(b0, s0)
                for j in range(length):
                    daily = dense[target[at] + j]
                    b1, s1 = daily[:, offset], daily[:, offset + 1]
                    actual = f(b1, s1)
                    cross_buy, cross_sell = f(b1, s0), f(b0, s1)
                    target_oib += actual / length
                    buy_contribution += (cross_buy - base_oib + actual - cross_sell) / (2 * length)
                    sell_contribution += (cross_sell - base_oib + actual - cross_buy) / (2 * length)
                y[at, IDX[f"oib_{suffix}"]] = target_oib
                y[at, IDX[f"base_{suffix}"]] = base_oib
                y[at, IDX[f"buy_{suffix}"]] = buy_contribution
                y[at, IDX[f"sell_{suffix}"]] = sell_contribution
                max_row_error = max(max_row_error, float(np.max(np.abs(target_oib - base_oib - buy_contribution - sell_contribution))))
            for suffix, change_b, change_s, denominator in [
                ("v", fbv - bv, fsv - sv, tv), ("a", fba - ba, fsa - sa, ta),
            ]:
                y[at, IDX[f"db_{suffix}"]] = 100 * change_b / denominator
                y[at, IDX[f"ds_{suffix}"]] = 100 * change_s / denominator
                y[at, IDX[f"dn_{suffix}"]] = 100 * (change_b - change_s) / denominator
            db, ds = fba - ba, fsa - sa
            y[at, IDX["db_yuan"]] = db / 10000
            y[at, IDX["ds_yuan"]] = ds / 10000
            y[at, IDX["dn_yuan"]] = (db - ds) / 10000
            y[at, IDX["net_cap"]] = length * (fba - fsa) / cap[at] * 10000
            y[at, IDX["dnet_cap"]] = length * (db - ds) / cap[at] * 10000
            y[at, IDX["net_total"]] = (fba - fsa) / ta * 100
            y[at, IDX["ret"]] = panel[helper.RETURNS[h]].iloc[positions].to_numpy()[at] * 100
            output[h][positions] = y
            ratios[h][positions[at]] = np.column_stack([ftv / tv, fta / ta])
        if number % 600 == 0:
            log(f"已构造{number}家公司的目标与历史窗口")
    if max_row_error > 1e-10:
        raise AssertionError(f"比例分解逐行恒等式失败：{max_row_error}")
    audit.append({"阶段": "恒等式", "项目": "逐行比例分解最大绝对误差", "数值": max_row_error})
    for h, y in output.items():
        raw_oib = np.asarray(y[:, IDX["oib_v"]])
        lower, upper = np.nanquantile(raw_oib, [.01, .99])
        y[:, IDX["oib_v_w"]] = np.clip(raw_oib, lower, upper)
        thresholds = np.nanquantile(ratios[h], .99, axis=0)
        panel[f"robust_{h}"] = (ratios[h][:, 0] <= thresholds[0]) & (ratios[h][:, 1] <= thresholds[1])
        audit.extend([
            {"阶段": f"窗口{h}", "项目": "成交量订单失衡共同样本缩尾下界", "数值": lower},
            {"阶段": f"窗口{h}", "项目": "成交量订单失衡共同样本缩尾上界", "数值": upper},
            {"阶段": f"窗口{h}", "项目": "总主动成交量相对事前均值99分位", "数值": thresholds[0]},
            {"阶段": f"窗口{h}", "项目": "总主动成交额相对事前均值99分位", "数值": thresholds[1]},
        ])
        y.flush()
    return output


def select_blocks(panel: pd.DataFrame, y: np.ndarray, h: int, robust: bool) -> list[tuple[int, np.ndarray]]:
    valid = np.isfinite(y).all(axis=1)
    for c in RHS:
        valid &= np.isfinite(panel[c].to_numpy())
    if robust:
        valid &= panel[f"robust_{h}"].to_numpy()
    blocks = []
    for q in sorted(panel.loc[valid, "q"].unique()):
        positions = np.flatnonzero(valid & (panel.q.to_numpy() == q))
        keep = helper.remove_singletons(panel.firm.to_numpy()[positions], panel.day.to_numpy()[positions])
        positions = positions[keep]
        if len(positions):
            blocks.append((int(q), positions))
    if not blocks:
        raise ValueError("没有完整共同样本。")
    return blocks


def estimate(panel: pd.DataFrame, y: np.ndarray, h: int, robust: bool, workspace: Path, *, quiet: bool = False) -> tuple[list[dict], dict]:
    started = time.monotonic()
    blocks = select_blocks(panel, y, h, robust)
    all_positions = np.concatenate([p for _, p in blocks])
    firms = np.unique(panel.firm.to_numpy()[all_positions])
    dates = np.unique(panel.day.to_numpy()[all_positions])
    p, m, n = len(RHS), len(METRICS), len(all_positions)
    a, b = np.zeros((p, p)), np.zeros((p, m))
    fe_count = 0
    cache = Path(tempfile.mkdtemp(prefix="净买压残差_", dir=workspace))
    paths = []
    cue_count = int(panel[CUE].to_numpy()[all_positions].sum())
    try:
        for number, (q, positions) in enumerate(blocks):
            frame = panel.iloc[positions]
            residualized = helper.absorb_block(
                np.column_stack([frame[RHS].to_numpy(dtype=float), y[positions]]),
                frame.firm.to_numpy(), frame.day.to_numpy(),
            )
            x, yr = residualized[:, :p], residualized[:, p:]
            a += x.T @ x
            b += x.T @ yr
            fe_count += frame.firm.nunique() + frame.day.nunique()
            path = cache / f"{number:03d}.npz"
            np.savez(path, x=x, y=yr, firm=frame.firm.to_numpy(), day=frame.day.to_numpy())
            paths.append(path)
            if not quiet and number % 20 == 0:
                log(f"窗口{h}，{'共同截除极端活跃度' if robust else '主检验'}：已吸收{helper.quarter_label(q)}")
        beta, cue_column, active, condition = helper.solve_moments(a, b)
        nvars = len(active)
        degrees = n - nvars - (fe_count - 1)
        fs, ds = np.zeros((len(firms), m)), np.zeros((len(dates), m))
        intersection = np.zeros((m, m))
        for path in paths:
            with np.load(path) as block:
                errors = block["y"] - block["x"] @ beta
                influence = (block["x"] @ cue_column)[:, None] * errors
                fi, di = np.searchsorted(firms, block["firm"]), np.searchsorted(dates, block["day"])
                for j in range(m):
                    fs[:, j] += np.bincount(fi, weights=influence[:, j], minlength=len(firms))
                    ds[:, j] += np.bincount(di, weights=influence[:, j], minlength=len(dates))
                intersection += influence.T @ influence
        common_scale = (n - 1) / (n - nvars)
        covariance = common_scale * (
            len(firms) / (len(firms) - 1) * (fs.T @ fs)
            + len(dates) / (len(dates) - 1) * (ds.T @ ds)
            - n / (n - 1) * intersection
        )
        covariance = (covariance + covariance.T) / 2
        if not np.isfinite(covariance).all() or (np.diag(covariance) <= 0).any():
            raise ValueError("提示系数联合协方差包含无效方差。")
        identities = {
            "量比例分解": beta[0, IDX["oib_v"]] - beta[0, IDX["base_v"]] - beta[0, IDX["buy_v"]] - beta[0, IDX["sell_v"]],
            "额比例分解": beta[0, IDX["oib_a"]] - beta[0, IDX["base_a"]] - beta[0, IDX["buy_a"]] - beta[0, IDX["sell_a"]],
            **{f"{suffix}净额分解": beta[0, IDX[f"dn_{suffix}"]] - beta[0, IDX[f"db_{suffix}"]] + beta[0, IDX[f"ds_{suffix}"]] for suffix in ["v", "a", "yuan"]},
        }
        if max(abs(v) for v in identities.values()) > 1e-7:
            raise AssertionError(f"回归系数恒等式失败：{identities}")
        rows = []
        for j, (name, label, unit) in enumerate(METRICS):
            effect, se = float(beta[0, j]), float(np.sqrt(covariance[j, j]))
            pv = float(2 * stats.t.sf(abs(effect / se), degrees))
            rows.append({
                "窗口": f"[t,t+{h}]", "样本版本": "共同截除极端活跃度" if robust else "主检验",
                "指标": name, "含义": label, "单位": unit, "提示系数": effect, "聚类标准误": se,
                "95%区间下界": effect - stats.t.ppf(.975, degrees) * se,
                "95%区间上界": effect + stats.t.ppf(.975, degrees) * se,
                "p值": pv, "显著性": "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .1 else "",
                "共同样本量": n, "提示公司日数": cue_count,
                "公司聚类数": len(firms), "日期聚类数": len(dates),
                "完整控制": "是", "固定效应": "日期及公司年度季度", "残差自由度": degrees,
                "标准化交叉积条件数": condition, "耗时秒": round(time.monotonic() - started, 2),
            })
        # 净买卖差异与两端贡献差异都来自联合协方差，而非将标准误平方相加。
        comparisons = []
        for suffix in ("v", "a"):
            contrast = np.zeros(m)
            contrast[IDX[f"buy_{suffix}"]] = 1
            contrast[IDX[f"sell_{suffix}"]] = -1
            effect = float(contrast @ beta[0])
            variance = float(contrast @ covariance @ contrast)
            if variance <= 0:
                raise ValueError("买卖贡献差异方差无效。")
            se = np.sqrt(variance)
            comparisons.append({
                "窗口": f"[t,t+{h}]", "样本版本": "共同截除极端活跃度" if robust else "主检验",
                "比较": f"{suffix}买入贡献减卖出贡献", "系数差": effect, "标准误": se,
                "p值": float(2 * stats.t.sf(abs(effect / se), degrees)),
            })
        details = {"beta": beta, "covariance": covariance, "active": active, "identities": identities, "comparisons": comparisons, "positions": all_positions}
        return rows, details
    finally:
        shutil.rmtree(cache)


def validate(panel: pd.DataFrame, y: np.ndarray, workspace: Path) -> list[dict]:
    from linearmodels.iv import AbsorbingLS
    eligible = np.isfinite(y).all(axis=1)
    cue_quarters = sorted(panel.loc[eligible & (panel[CUE] == 1), "q"].unique())[-4:]
    candidate = panel.loc[eligible & panel.q.isin(cue_quarters)]
    cue_firms = candidate.loc[candidate[CUE] == 1, "firm"].unique()
    others = np.sort(candidate.firm.unique())[::max(1, candidate.firm.nunique() // 80)]
    pos = candidate.index[candidate.firm.isin(np.union1d(cue_firms, others))].to_numpy()
    subset, sy = panel.loc[pos].reset_index(drop=True), y[pos]
    _, details = estimate(subset, sy, 1, False, workspace, quiet=True)
    chosen = details["positions"]
    reg = subset.iloc[chosen].reset_index(drop=True)
    active = details["active"]
    exog = reg[RHS].iloc[:, active].astype(float)
    absorb = pd.DataFrame({
        "firm_q": pd.Categorical(reg.firm.astype(str) + "_" + reg.q.astype(str)),
        "date": pd.Categorical(reg.day),
    })
    clusters = pd.DataFrame({"firm": reg.firm, "date": reg.day})
    checks = []
    for column in ["oib_v", "buy_v", "sell_v", "dn_a", "dnet_cap"]:
        j = IDX[column]
        result = AbsorbingLS(sy[chosen, j], exog, absorb=absorb).fit(cov_type="clustered", clusters=clusters, debiased=True)
        actual, expected = float(details["beta"][0, j]), float(result.params[CUE])
        actual_var, expected_var = float(details["covariance"][j, j]), float(result.cov.loc[CUE, CUE])
        if not np.isclose(actual, expected, atol=1e-7, rtol=1e-6) or not np.isclose(actual_var, expected_var, atol=1e-8, rtol=2e-5):
            raise AssertionError(f"实际子样本估计器核验失败：{column}")
        checks.append({"指标": column, "样本量": len(chosen), "系数误差": actual - expected, "聚类方差误差": actual_var - expected_var, "通过": True})
    log("实际数据子样本的5项系数及聚类方差核验通过")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--chunk-size", type=int, default=400000)
    parser.add_argument("--save-dir", type=Path, nargs="?", const=TASK_OUTPUT_DIR, default=None, help="显式保存开关；不带目录时保存到本任务目录/时间戳，也可指定保存目录")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("起始年不能晚于结束年。")
    for path in [*INPUTS.values(), HELPER_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    destination = args.save_dir.resolve() / stamp if args.save_dir else None
    if destination:
        destination.mkdir(parents=True, exist_ok=False)
    parameters = {
        "日期": stamp, "起始年": args.start_year, "结束年": args.end_year,
        "基线": "t-20至t-1，20个市场交易日全部有效，不填0，不跨缺失日滚动",
        "对称分解": "目标日逐日分解再平均；基线使用事前平均买卖量的比率，不是事前比率平均",
        "资金规模": "小单主动买卖成交差额，不代表账户资金净流入；万元指标为日均变化，市值比率为累计",
        "稳健性": "共同剔除未来总主动量或额相对事前均值高于99分位的整行，不分别缩尾分量",
        "推断": "公司与日期双向聚类，有限样本修正；未作跨指标多重检验调整",
        "脚本": str(Path(__file__).resolve()), "脚本SHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "输入": {name: {"路径": str(path), "大小": path.stat().st_size, "修改时间纳秒": path.stat().st_mtime_ns} for name, path in INPUTS.items()},
    }
    audit, rows, comparisons, identities, checks = [], [], [], [], []
    work_root = ROOT / "work"
    work_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="小单净买压检验_", dir=work_root) as temporary:
        workspace = Path(temporary)
        panel, raw = read_inputs(args, audit)
        outcomes = construct_outcomes(panel, raw, workspace, audit)
        del raw
        gc.collect()
        checks = validate(panel, outcomes[1], workspace)
        for h in (1, 10):
            for robust in (False, True):
                result, details = estimate(panel, outcomes[h], h, robust, workspace)
                rows.extend(result)
                comparisons.extend(details["comparisons"])
                identities.append({"窗口": h, "稳健性": robust, **details["identities"]})
                log(f"窗口{h}，{'共同截除极端活跃度' if robust else '主检验'}完成，样本{result[0]['共同样本量']:,}行")
                if destination:
                    pd.DataFrame(rows).to_csv(destination / "核心检验结果.csv", index=False, encoding="utf-8-sig")
                    pd.DataFrame(comparisons).to_csv(destination / "买卖贡献差异检验.csv", index=False, encoding="utf-8-sig")
        if destination:
            pd.DataFrame(audit).to_csv(destination / "数据与口径审计.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(identities).to_csv(destination / "回归系数恒等核验.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(checks).to_csv(destination / "估计器核验.csv", index=False, encoding="utf-8-sig")
            (destination / "参数与来源.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = pd.DataFrame(rows)
        print(summary[["窗口", "样本版本", "含义", "单位", "提示系数", "聚类标准误", "p值", "共同样本量"]].to_string(index=False, float_format=lambda v: f"{v:.7f}"), flush=True)
        for matrix in outcomes.values():
            matrix.flush()
            matrix._mmap.close()
        log(f"检验完成，结果位置：{destination if destination else '仅终端打印'}")
    log("临时矩阵已清理")


if __name__ == "__main__":
    main()
