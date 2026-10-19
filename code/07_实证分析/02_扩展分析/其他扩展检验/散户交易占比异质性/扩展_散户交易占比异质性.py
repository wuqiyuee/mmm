# 本文件由旧零散回归目录迁移；仅调整项目定位、公共工具引用和任务输出位置。
"""按被提示公司事前小单交易占比，独立检验收益和订单失衡异质性。

直接运行只打印结果；--save-dir（保存目录）会新建带时间戳的结果子目录。
不修改现有脚本、数据或既有结果。临时残差矩阵仅写入项目 work/，结束后清理。

分组口径：上一自然季度，在基础清洗样本中的有效股票日，计算
    sum(小单主动买入额+小单主动卖出额) / sum(全部主动买入额+全部主动卖出额)。
四项金额须同时非缺失、非负，总金额为正，小单额不得超过总额。
至少40个有效交易日才合格；每季度每公司只参与一次中位数计算。
高于中位数为高组，其余为低组，下一自然季度使用；缺失季度不跨期填补。
这是小单交易代理指标，不是账户身份识别，也不是小单净买入比例。

样本与模型：默认2005—2025年，继承现有模式公司最终面板和27项控制。
订单失衡先在研究期完整主面板中按原表3口径上下各1%缩尾，再筛分组。
每个窗口分别取收益、订单失衡、控制及历史占比共同有效样本；
递归删除组别日期和公司季度固定效应单例，基准与分组模型共用这一样本。
基准吸收日期、公司季度；分组模型允许提示系数和全部控制系数分组，
并吸收组别×日期、公司季度，与分别分组估计的系数等价。
两组差异用公司及原始日期双向聚类的联合协方差检验，保留跨组相关。
有限样本修正和显著性检验沿用原表2/表3的吸收最小二乘口径。

估计采用逐季度精确吸收、分块正规方程和提示系数影响函数，控制内存。
固定效应直接消除算法参考现有图2图3扩展公共模块，本文件不导入或改写它。
运行前用实际数据子样本与 linearmodels（线性模型库）的吸收估计器核验。

示例：
    /opt/miniconda3/bin/python 扩展_散户交易占比异质性.py
"""

from __future__ import annotations

import os

# 限制数值库线程，避免后台检验占满电脑；须在导入数值库之前设置。
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "2")

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.csgraph import connected_components


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
TASK_OUTPUT_DIR = ROOT / "图表结果/02_扩展分析/其他扩展检验/散户交易占比异质性"
CUE = "quarterLag1size90plus"
CONTROLS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
    + ["ownOtherEventER", "ownPerformanceForecastBrief"]
    + ["aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV"]
)
RHS = [CUE, *CONTROLS]
PATHS = {
    "交易原始表": ROOT / "中间结果/00_合并原始数据/CSMAR_个股买卖不平衡指标表_日_合并结果.csv",
    "基础样本索引": ROOT / "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv",
    "模式公司面板": ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
    "订单失衡表": ROOT / "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv",
}
AMOUNTS = ["B_Amount_S", "S_Amount_S", "B_Amount", "S_Amount"]
RETURNS = {h: f"chAdjreturn_t_tp{h}_Recall_w" for h in (1, 10)}
OIB = {h: f"OIBvol_avg_t_tp{h}" for h in (1, 10)}


def log(message: str) -> None:
    print(f"[{datetime.now(ZoneInfo('Asia/Shanghai')):%H:%M:%S}] {message}", flush=True)


def quarter_codes(dates: pd.Series) -> np.ndarray:
    return (dates.dt.year.to_numpy() * 4 + (dates.dt.month.to_numpy() - 1) // 3).astype("int32")


def quarter_label(code: int) -> str:
    return f"{int(code) // 4}年第{int(code) % 4 + 1}季度"


def day_codes(dates: pd.Series) -> np.ndarray:
    return dates.to_numpy(dtype="datetime64[D]").astype("int32")


def keys(firms: np.ndarray, days: np.ndarray) -> np.ndarray:
    return days.astype("int64") * 1_000_000 + firms.astype("int64")


def matched_positions(sorted_keys: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_keys, candidates)
    matched = positions < len(sorted_keys)
    safe = np.minimum(positions, len(sorted_keys) - 1)
    matched &= sorted_keys[safe] == candidates
    return positions, matched


def build_groups(args: argparse.Namespace, audit: list[dict]) -> pd.DataFrame:
    """只用基础清洗股票日构造历史占比，季度缺口不作前向填补。"""
    source_min = args.start_year * 4 - 1
    source_max = args.end_year * 4 + 2
    eligible_parts = []
    log("读取事前分组所需的基础样本日期索引")
    for chunk in pd.read_csv(
        PATHS["基础样本索引"], usecols=["Stkcd", "Trddt"],
        dtype={"Stkcd": "int32"}, parse_dates=["Trddt"], chunksize=args.chunk_size,
    ):
        q = quarter_codes(chunk["Trddt"])
        keep = (q >= source_min) & (q <= source_max)
        eligible_parts.append(keys(chunk["Stkcd"].to_numpy()[keep], day_codes(chunk["Trddt"])[keep]))
    eligible = np.concatenate(eligible_parts)
    del eligible_parts
    eligible.sort()
    if len(eligible) == 0 or np.any(eligible[1:] == eligible[:-1]):
        raise ValueError("基础样本索引为空或公司日期重复。")
    partials, seen_keys = [], []
    raw_count = invalid_amount = subset_violation = used_count = 0
    log("按块读取原始交易金额，构造季度分子、分母与有效天数")
    for number, chunk in enumerate(pd.read_csv(
        PATHS["交易原始表"], usecols=["Stkcd", "Trddt", *AMOUNTS],
        dtype={"Stkcd": "int32", **{c: "float64" for c in AMOUNTS}},
        parse_dates=["Trddt"], chunksize=args.chunk_size,
    ), start=1):
        raw_count += len(chunk)
        q = quarter_codes(chunk["Trddt"])
        temporal = (q >= source_min) & (q <= source_max)
        chunk = chunk.loc[temporal].copy()
        q = q[temporal]
        if chunk.empty:
            continue
        key = keys(chunk["Stkcd"].to_numpy(), day_codes(chunk["Trddt"]))
        _, in_pool = matched_positions(eligible, key)
        amounts = chunk[AMOUNTS].to_numpy()
        total = amounts[:, 2] + amounts[:, 3]
        small = amounts[:, 0] + amounts[:, 1]
        valid = np.isfinite(amounts).all(axis=1) & (amounts >= 0).all(axis=1) & (total > 0)
        invalid_amount += int((in_pool & ~valid).sum())
        subset_bad = valid & (small > total + np.maximum(0.01, total * 1e-9))
        subset_violation += int((in_pool & subset_bad).sum())
        keep = in_pool & valid & ~subset_bad
        seen_keys.append(key[in_pool])
        used_count += int(keep.sum())
        part = pd.DataFrame({
            "firm": chunk["Stkcd"].to_numpy()[keep], "source_q": q[keep],
            "small_amount": small[keep], "total_amount": total[keep], "valid_days": 1,
        }).groupby(["firm", "source_q"], as_index=False, sort=False).sum()
        partials.append(part)
        if number % 4 == 0:
            log(f"已读取原始交易表{raw_count:,}行，有效匹配{used_count:,}行")
    matched_keys = np.concatenate(seen_keys)
    matched_keys.sort()
    if np.any(matched_keys[1:] == matched_keys[:-1]):
        raise ValueError("原始交易表在基础样本公司日期内存在重复记录。")
    grouped = pd.concat(partials, ignore_index=True).groupby(
        ["firm", "source_q"], as_index=False, sort=True,
    )[["small_amount", "total_amount", "valid_days"]].sum()
    audit.extend([
        {"阶段": "历史占比", "项目": "原始交易表读取行数", "数量": raw_count},
        {"阶段": "历史占比", "项目": "基础样本内金额缺失负值或总额非正天数", "数量": invalid_amount},
        {"阶段": "历史占比", "项目": "基础样本内小单金额超过总额天数", "数量": subset_violation},
        {"阶段": "历史占比", "项目": "有效交易日记录", "数量": used_count},
        {"阶段": "历史占比", "项目": "未达到有效天数门槛的公司季度", "数量": int((grouped.valid_days < args.min_days).sum())},
    ])
    grouped = grouped.loc[grouped.valid_days >= args.min_days].copy()
    grouped["share"] = grouped.small_amount / grouped.total_amount
    if grouped.empty or not grouped.share.between(0, 1 + 1e-9).all():
        raise ValueError("没有可用季度占比，或占比超出零到一。")
    grouped["median"] = grouped.groupby("source_q")["share"].transform("median")
    grouped["high"] = (grouped.share > grouped["median"]).astype("int8")
    grouped["q"] = grouped.source_q + 1
    if grouped.duplicated(["firm", "q"]).any():
        raise ValueError("季度分组键重复。")
    log(f"构造完成：{len(grouped):,}个合格公司季度，中位数按公司等权计算")
    del eligible, matched_keys, seen_keys, partials
    gc.collect()
    return grouped


def read_panel(args: argparse.Namespace, groups: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    """只读取所需字段；金额分组覆盖不足的行暂保留，以复现订单失衡缩尾总体。"""
    selected = ["StkcdRecall", "Trddt", *RHS, *RETURNS.values()]
    group_keys = (groups.firm.to_numpy(dtype="int64") * 10000 + groups.q.to_numpy()).astype("int64")
    order = np.argsort(group_keys)
    group_keys = group_keys[order]
    high_values = groups.high.to_numpy()[order]
    pieces = []
    log("按块读取模式公司最终面板，仅保留本检验需要的字段")
    for number, chunk in enumerate(pd.read_csv(
        PATHS["模式公司面板"], usecols=selected,
        dtype={"StkcdRecall": "int32", **{c: "float32" for c in selected if c not in {"StkcdRecall", "Trddt"}}},
        parse_dates=["Trddt"], chunksize=args.chunk_size,
    ), start=1):
        year = chunk.Trddt.dt.year
        chunk = chunk.loc[year.between(args.start_year, args.end_year)].copy()
        if chunk.empty:
            continue
        chunk["q"] = quarter_codes(chunk.Trddt)
        chunk["day"] = day_codes(chunk.Trddt)
        chunk = chunk.rename(columns={"StkcdRecall": "firm"}).drop(columns="Trddt")
        candidate = chunk.firm.to_numpy(dtype="int64") * 10000 + chunk.q.to_numpy()
        positions, matched = matched_positions(group_keys, candidate)
        high = np.full(len(chunk), -1, dtype="int8")
        high[matched] = high_values[positions[matched]]
        chunk["high"] = high
        pieces.append(chunk)
        if number % 4 == 0:
            log(f"已读取主面板第{number}块")
    data = pd.concat(pieces, ignore_index=True)
    del pieces
    key = keys(data.firm.to_numpy(), data.day.to_numpy())
    order = np.argsort(key)
    sorted_keys = key[order]
    if len(data) == 0 or np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError("研究期主面板为空或公司日期重复。")
    cue = data[CUE].to_numpy()
    if not np.isin(cue, [0, 1]).all():
        raise ValueError("提示变量存在缺失或非零一取值。")
    audit.extend([
        {"阶段": "主面板", "项目": "研究期公司日", "数量": len(data)},
        {"阶段": "主面板", "项目": "未匹配合格事前分组公司日", "数量": int((data.high < 0).sum())},
        {"阶段": "主面板", "项目": "未匹配合格事前分组的提示公司日", "数量": int(((data.high < 0) & (data[CUE] == 1)).sum())},
    ])
    log(f"研究期主面板{len(data):,}行；开始匹配订单失衡")
    matched_before = np.zeros(len(data), dtype=bool)
    oib_values = {h: np.full(len(data), np.nan, dtype="float32") for h in OIB}
    for number, chunk in enumerate(pd.read_csv(
        PATHS["订单失衡表"], usecols=["Stkcd", "Trddt", *OIB.values()],
        dtype={"Stkcd": "int32", **{c: "float32" for c in OIB.values()}},
        parse_dates=["Trddt"], chunksize=args.chunk_size,
    ), start=1):
        candidate = keys(chunk.Stkcd.to_numpy(), day_codes(chunk.Trddt))
        positions, matched = matched_positions(sorted_keys, candidate)
        target = order[positions[matched]]
        if len(np.unique(target)) != len(target) or matched_before[target].any():
            raise ValueError("订单失衡表包含重复的研究期公司日期。")
        matched_before[target] = True
        for h, column in OIB.items():
            oib_values[h][target] = chunk[column].to_numpy()[matched]
        if number % 6 == 0:
            log(f"订单失衡已读取第{number}块")
    for h in OIB:
        values = oib_values[h]
        values[~np.isfinite(values)] = np.nan
        lower, upper = np.nanquantile(values, [0.01, 0.99])
        data[f"oib_{h}"] = np.clip(values, lower, upper)
        audit.extend([
            {"阶段": "缩尾", "项目": f"{h}窗口下界（百分点）", "数量": float(lower)},
            {"阶段": "缩尾", "项目": f"{h}窗口上界（百分点）", "数量": float(upper)},
        ])
    del oib_values, order, sorted_keys, key, matched_before
    gc.collect()
    return data


def remove_singletons(firms: np.ndarray, days: np.ndarray) -> np.ndarray:
    keep = np.ones(len(firms), dtype=bool)
    while True:
        positions = np.flatnonzero(keep)
        f = pd.factorize(firms[positions], sort=False)[0]
        d = pd.factorize(days[positions], sort=False)[0]
        bad = (np.bincount(f)[f] == 1) | (np.bincount(d)[d] == 1)
        if not bad.any():
            return keep
        keep[positions[bad]] = False


def common_blocks(data: pd.DataFrame, h: int, audit: list[dict]) -> list[tuple[int, int, np.ndarray]]:
    valid = (data.high.to_numpy() >= 0)
    for column in [*RHS, RETURNS[h], f"oib_{h}"]:
        valid &= np.isfinite(data[column].to_numpy())
    before = int(valid.sum())
    blocks = []
    for q in sorted(data.loc[valid, "q"].unique()):
        for group in (0, 1):
            positions = np.flatnonzero(valid & (data.q.to_numpy() == q) & (data.high.to_numpy() == group))
            if not len(positions):
                continue
            keep = remove_singletons(data.firm.to_numpy()[positions], data.day.to_numpy()[positions])
            positions = positions[keep]
            if len(positions):
                blocks.append((int(q), group, positions))
    after = sum(len(p) for _, _, p in blocks)
    audit.extend([
        {"阶段": f"{h}窗口共同样本", "项目": "完整有效公司日", "数量": before},
        {"阶段": f"{h}窗口共同样本", "项目": "组别日期与公司季度单例删除", "数量": before - after},
        {"阶段": f"{h}窗口共同样本", "项目": "最终公司日", "数量": after},
    ])
    if not after:
        raise ValueError(f"{h}窗口没有共同有效样本。")
    return blocks


def absorb_block(values: np.ndarray, firms: np.ndarray, days: np.ndarray) -> np.ndarray:
    """精确消除季内公司和日期效应，包括不平衡面板及断连分量。"""
    f, _ = pd.factorize(firms, sort=False)
    d, _ = pd.factorize(days, sort=False)
    nf, nd = int(f.max()) + 1, int(d.max()) + 1
    n = len(values)
    rows, ones = np.arange(n), np.ones(n)
    fs = sparse.csr_matrix((ones, (f, rows)), shape=(nf, n))
    ds = sparse.csr_matrix((ones, (d, rows)), shape=(nd, n))
    counts = sparse.csr_matrix((ones, (f, d)), shape=(nf, nd))
    fn, dn = np.bincount(f).astype(float), np.bincount(d).astype(float)
    _, components = connected_components((counts.T @ counts).tocsr(), directed=False)
    references = np.unique(components, return_index=True)[1]
    free = np.setdiff1d(np.arange(nd), references)
    schur = np.diag(dn) - (counts.T @ counts.multiply(1 / fn[:, None])).toarray()
    schur = (schur + schur.T) / 2
    within = values - ((fs @ values) / fn[:, None])[f]
    effects = np.zeros((nd, values.shape[1]))
    if len(free):
        factor = cho_factor(schur[np.ix_(free, free)], lower=True, check_finite=False)
        effects[free] = cho_solve(factor, (ds @ within)[free], check_finite=False)
    result = within - effects[d] + ((counts @ effects) / fn[:, None])[f]
    if not np.isfinite(result).all():
        raise FloatingPointError("固定效应残差化出现非有限值。")
    # 检查消除后，公司和日期组内均值确实为零。
    error = max(float(np.abs(fs @ result).max()), float(np.abs(ds @ result).max()))
    if error > max(1e-7, float(np.abs(values).max()) * n * 1e-10):
        raise FloatingPointError(f"固定效应投影误差过大：{error}")
    return result


def solve_moments(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    diagonal = np.diag(a)
    active = np.flatnonzero(diagonal > max(1e-18, float(diagonal.max()) * 1e-18))
    if 0 not in active:
        raise ValueError("某组提示变量被固定效应完全吸收，不能报告组间差异。")
    scale = np.sqrt(diagonal[active])
    normalized = a[np.ix_(active, active)] / scale[:, None] / scale[None, :]
    condition = float(np.linalg.cond(normalized))
    if not np.isfinite(condition) or condition >= 1e12:
        raise ValueError(f"解释变量条件识别不足，标准化交叉积条件数为{condition}")
    normalized_inverse = np.linalg.inv(normalized)
    inverse = normalized_inverse / scale[:, None] / scale[None, :]
    beta = np.zeros_like(b)
    beta[active] = inverse @ b[active]
    cue_column = np.zeros(len(a))
    cue_column[active] = inverse[:, np.flatnonzero(active == 0)[0]]
    return beta, cue_column, active, condition


def estimate_pair(
    data: pd.DataFrame, blocks: list[tuple[int, int, np.ndarray]], h: int,
    split: bool, workspace: Path, *, quiet: bool = False,
) -> tuple[list[dict], list[dict], dict]:
    """一次估计同一窗口两个结局；跨组公司和日期相关保留在联合协方差中。"""
    started = time.monotonic()
    groups = 2 if split else 1
    p = len(RHS)
    a = np.zeros((groups, p, p))
    b = np.zeros((groups, p, 2))
    group_n = np.zeros(groups, dtype="int64")
    group_cues = np.zeros(groups, dtype="int64")
    group_days, group_firms = [set() for _ in range(groups)], [set() for _ in range(groups)]
    firm_quarter_count = date_fe_count = 0
    positions_all = np.concatenate([p for _, _, p in blocks])
    firms_all = np.unique(data.firm.to_numpy()[positions_all])
    days_all = np.unique(data.day.to_numpy()[positions_all])
    if not split:
        regrouped = {}
        for q, _, positions in blocks:
            regrouped.setdefault(q, []).append(positions)
        fitting_blocks = [(q, 0, np.concatenate(parts)) for q, parts in sorted(regrouped.items())]
    else:
        fitting_blocks = blocks
    cache = Path(tempfile.mkdtemp(prefix=f"窗口{h}_{'分组' if split else '基准'}_", dir=workspace))
    paths = []
    try:
        for number, (q, group, positions) in enumerate(fitting_blocks):
            frame = data.iloc[positions]
            x = frame[RHS].to_numpy(dtype="float64")
            y = np.column_stack([
                frame[RETURNS[h]].to_numpy(dtype="float64") * 100,
                frame[f"oib_{h}"].to_numpy(dtype="float64"),
            ])
            firms, days = frame.firm.to_numpy(), frame.day.to_numpy()
            residualized = absorb_block(np.column_stack([x, y]), firms, days)
            xr, yr = residualized[:, :p], residualized[:, p:]
            a[group] += xr.T @ xr
            b[group] += xr.T @ yr
            group_n[group] += len(frame)
            group_cues[group] += int(frame[CUE].sum())
            group_firms[group].update(firms.tolist())
            group_days[group].update(days.tolist())
            firm_quarter_count += len(np.unique(firms))
            date_fe_count += len(np.unique(days))
            path = cache / f"{number:03d}.npz"
            np.savez(path, x=xr, y=yr, firm=firms, day=days, group=np.int8(group))
            paths.append(path)
            if not quiet and number % 20 == 0:
                log(f"{'高低组' if split else '共同样本基准'}，窗口{h}：已吸收{quarter_label(q)}固定效应")
        solved = [solve_moments(a[g], b[g]) for g in range(groups)]
        k = sum(len(s[2]) for s in solved)
        n = int(group_n.sum())
        # 与原吸收估计器一致的类别数量近似自由度，未另作固定效应嵌套调整。
        df_resid = n - k - (firm_quarter_count + date_fe_count - 1)
        if df_resid <= 0:
            raise ValueError("残差自由度不为正。")
        firm_scores = np.zeros((groups, 2, len(firms_all)))
        date_scores = np.zeros((groups, 2, len(days_all)))
        observation_meat = np.zeros((groups, 2))
        for path in paths:
            with np.load(path) as block:
                g = int(block["group"])
                x, y = block["x"], block["y"]
                beta, cue_column, _, _ = solved[g]
                errors = y - x @ beta
                influence = (x @ cue_column)[:, None] * errors
                fi = np.searchsorted(firms_all, block["firm"])
                di = np.searchsorted(days_all, block["day"])
                for outcome in (0, 1):
                    score = influence[:, outcome]
                    firm_scores[g, outcome] += np.bincount(fi, weights=score, minlength=len(firms_all))
                    date_scores[g, outcome] += np.bincount(di, weights=score, minlength=len(days_all))
                    observation_meat[g, outcome] += score @ score
        common_scale = (n - 1) / (n - k)
        f_scale = common_scale * len(firms_all) / (len(firms_all) - 1)
        d_scale = common_scale * len(days_all) / (len(days_all) - 1)
        i_scale = common_scale * n / (n - 1)
        rows, coefficients, covariances = [], [], []
        for outcome, label in enumerate(("特征调整收益", "小单成交量订单失衡均值")):
            fs, ds = firm_scores[:, outcome], date_scores[:, outcome]
            covariance = f_scale * fs @ fs.T + d_scale * ds @ ds.T - i_scale * np.diag(observation_meat[:, outcome])
            covariance = (covariance + covariance.T) / 2
            covariances.append(covariance)
            tests = [("共同样本基准", np.array([1.0]))] if not split else [
                ("低小单交易占比组", np.array([1.0, 0.0])),
                ("高小单交易占比组", np.array([0.0, 1.0])),
                ("高组减低组", np.array([-1.0, 1.0])),
            ]
            cue_betas = np.array([s[0][0, outcome] for s in solved])
            for role, contrast in tests:
                effect = float(contrast @ cue_betas)
                variance = float(contrast @ covariance @ contrast)
                if variance <= 0 or not np.isfinite(variance):
                    raise ValueError(f"{label}的{role}聚类方差无效。")
                se = np.sqrt(variance)
                tvalue = effect / se
                pvalue = float(2 * stats.t.sf(abs(tvalue), df_resid))
                selected_groups = np.flatnonzero(contrast)
                rows.append({
                    "结局": label, "窗口": f"[t,t+{h}]", "结果项": role,
                    "系数（百分点）": effect, "聚类标准误（百分点）": se,
                    "t值": tvalue, "p值": pvalue,
                    "显著性": "***" if pvalue < .01 else "**" if pvalue < .05 else "*" if pvalue < .1 else "",
                    "95%区间下界": effect - stats.t.ppf(.975, df_resid) * se,
                    "95%区间上界": effect + stats.t.ppf(.975, df_resid) * se,
                    "模型总样本量": n, "对应组公司日数": int(group_n[selected_groups].sum()),
                    "对应组提示公司日数": int(group_cues[selected_groups].sum()),
                    "公司聚类数": len(firms_all), "日期聚类数": len(days_all),
                    "残差自由度": df_resid, "完整控制": "是",
                    "固定效应": "组别×日期、公司×年度季度" if split else "日期、公司×年度季度",
                    "控制系数分组": "是" if split else "否",
                    "组间协方差": float(covariance[0, 1]) if split else np.nan,
                    "耗时秒": round(time.monotonic() - started, 2),
                })
            for g, (beta, _, active, condition) in enumerate(solved):
                for j, column in enumerate(RHS):
                    coefficients.append({
                        "结局": label, "窗口": f"[t,t+{h}]",
                        "模型": "高低组完全交互" if split else "共同样本基准",
                        "组别": ("低组", "高组")[g] if split else "合并样本",
                        "变量": column, "系数（百分点）": float(beta[j, outcome]) if j in active else np.nan,
                        "估计状态": "已估计" if j in active else "被固定效应吸收",
                        "标准化交叉积条件数": condition,
                    })
        details = {"beta": [s[0] for s in solved], "active": [s[2] for s in solved], "cov": covariances}
        return rows, coefficients, details
    finally:
        shutil.rmtree(cache)


def validate_estimator(data: pd.DataFrame, h: int, workspace: Path) -> list[dict]:
    """用实际非平衡子样本，对照成熟吸收估计器的系数和双向聚类协方差。"""
    from linearmodels.iv import AbsorbingLS

    available = data.loc[(data.high >= 0) & np.isfinite(data[f"oib_{h}"]), :]
    quarters = sorted(available.q.unique())
    # 选择最后四个有提示的季度；保留提示公司并加入确定性的参照公司。
    cue_quarters = sorted(available.loc[available[CUE] == 1, "q"].unique())
    chosen = cue_quarters[-4:] if len(cue_quarters) >= 4 else quarters[-4:]
    candidate = available.loc[available.q.isin(chosen)]
    cue_firms = candidate.loc[candidate[CUE] == 1, "firm"].unique()
    other_firms = np.sort(candidate.firm.unique())[::max(1, candidate.firm.nunique() // 120)]
    subset = candidate.loc[candidate.firm.isin(np.union1d(cue_firms, other_firms))].copy().reset_index(drop=True)
    del available, candidate
    validation = []
    for split in (False, True):
        blocks = common_blocks(subset, h, [])
        rows, _, details = estimate_pair(subset, blocks, h, split, workspace, quiet=True)
        positions = np.concatenate([p for _, _, p in blocks])
        reg = subset.iloc[positions].reset_index(drop=True)
        exog_parts, names = [], []
        for g in range(2 if split else 1):
            active = details["active"][g]
            mask = (reg.high.to_numpy() == g).astype(float) if split else np.ones(len(reg))
            exog_parts.append(reg[RHS].to_numpy(dtype=float)[:, active] * mask[:, None])
            names.extend([f"g{g}_{RHS[j]}" for j in active])
        exog = pd.DataFrame(np.column_stack(exog_parts), columns=names)
        absorb = pd.DataFrame({
            "firm_quarter": pd.Categorical(reg.firm.astype(str) + "_" + reg.q.astype(str)),
            "date": pd.Categorical(reg.day.astype(str) + ("_" + reg.high.astype(str) if split else "")),
        })
        clusters = pd.DataFrame({"firm": reg.firm, "date": reg.day})
        for outcome, column in enumerate([RETURNS[h], f"oib_{h}"]):
            y = reg[column].astype(float) * (100 if outcome == 0 else 1)
            result = AbsorbingLS(y, exog, absorb=absorb, drop_absorbed=False).fit(
                cov_type="clustered", clusters=clusters, debiased=True,
            )
            focus = [f"g{g}_{CUE}" for g in range(2 if split else 1)]
            expected = result.params.loc[focus].to_numpy()
            actual = np.array([b[0, outcome] for b in details["beta"]])
            expected_cov = result.cov.loc[focus, focus].to_numpy()
            actual_cov = details["cov"][outcome]
            coefficient_error = float(np.max(np.abs(actual - expected)))
            covariance_error = float(np.max(np.abs(actual_cov - expected_cov)))
            if not np.allclose(actual, expected, rtol=2e-6, atol=2e-8):
                raise AssertionError(f"实际数据系数对照失败：{actual}，{expected}")
            if not np.allclose(actual_cov, expected_cov, rtol=2e-5, atol=2e-9):
                raise AssertionError(f"实际数据聚类协方差对照失败：{actual_cov}，{expected_cov}")
            validation.append({
                "模型": "完全交互" if split else "基准", "结局": column,
                "核验样本量": len(reg), "最大系数误差": coefficient_error,
                "最大协方差误差": covariance_error, "通过": True,
            })
    log(f"实际数据估计器核验通过：{len(validation)}项系数及聚类协方差对照")
    return validation


def save_results(directory: Path, rows: list[dict], coefficients: list[dict], audit: list[dict], groups: pd.DataFrame, validation: list[dict], parameters: dict) -> None:
    pd.DataFrame(rows).to_csv(directory / "核心结果与组间差异.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coefficients).to_csv(directory / "全部系数.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(directory / "样本与口径审计.csv", index=False, encoding="utf-8-sig")
    if not (directory / "季度分组.csv").exists():
        export = groups.copy()
        export["firm"] = export.firm.astype(str).str.zfill(6)
        export["来源季度"] = export.source_q.map(quarter_label)
        export["适用季度"] = export.q.map(quarter_label)
        export = export.rename(columns={
            "firm": "被提示公司代码", "small_amount": "小单成交额合计", "total_amount": "全部主动成交额合计",
            "valid_days": "有效交易日数", "share": "事前小单占比", "median": "季度中位数", "high": "高占比组",
        })
        export.to_csv(directory / "季度分组.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(validation).to_csv(directory / "估计器核验.csv", index=False, encoding="utf-8-sig")
    (directory / "参数与来源.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-year", type=int, default=2005, help="事件起点起始年")
    parser.add_argument("--end-year", type=int, default=2025, help="事件起点结束年")
    parser.add_argument("--min-days", type=int, default=40, help="上一自然季度最低有效交易日数")
    parser.add_argument("--chunk-size", type=int, default=400_000, help="读取大表时每块行数")
    parser.add_argument("--save-dir", type=Path, nargs="?", const=TASK_OUTPUT_DIR, default=None, help="显式保存开关；不带目录时保存到本任务目录/时间戳，也可指定保存目录")
    args = parser.parse_args()
    if args.start_year > args.end_year or args.min_days < 2 or args.chunk_size < 1000:
        parser.error("年份范围、有效天数或分块大小无效。")
    for label, path in PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"{label}不存在：{path}")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    save_dir = None
    if args.save_dir:
        save_dir = args.save_dir.resolve() / stamp
        save_dir.mkdir(parents=True, exist_ok=False)
    parameters = {
        "运行时间": stamp, "起始年": args.start_year, "结束年": args.end_year,
        "上一季度最低有效日": args.min_days, "分组": "上一自然季度小单金额合计占比，每季度公司等权中位数二分",
        "阈值股票池": "来源季度基础清洗样本中至少达到最低有效天数的全部公司，不限提示公司",
        "共同样本": "每窗口收益及订单失衡、完整控制、历史分组共同有效，基准和分组模型保持一致",
        "推断": "原公司和原日期双向聚类；有限样本修正；四结局未经多重检验调整",
        "解释边界": "小单交易代理，非账户身份；分组同时可能反映规模流动性，结果不能独立确认因果机制",
        "脚本": str(Path(__file__).resolve()),
        "脚本SHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "输入": {label: {"路径": str(path), "字节数": path.stat().st_size, "修改时间纳秒": path.stat().st_mtime_ns} for label, path in PATHS.items()},
    }
    audit, rows, coefficients, validation = [], [], [], []
    work_root = ROOT / "work"
    work_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="散户占比检验_", dir=work_root) as temporary:
        workspace = Path(temporary)
        groups = build_groups(args, audit)
        data = read_panel(args, groups, audit)
        validation = validate_estimator(data, 1, workspace)
        for h in (1, 10):
            blocks = common_blocks(data, h, audit)
            log(f"窗口{h}共同样本{sum(len(p) for _, _, p in blocks):,}行，开始正式估计")
            for split in (False, True):
                result_rows, result_coefficients, _ = estimate_pair(data, blocks, h, split, workspace)
                rows.extend(result_rows)
                coefficients.extend(result_coefficients)
                if save_dir:
                    save_results(save_dir, rows, coefficients, audit, groups, validation, parameters)
                log(f"窗口{h}，{'高低组及差异检验' if split else '共同样本基准'}完成")
        summary = pd.DataFrame(rows)[[
            "结局", "窗口", "结果项", "系数（百分点）", "聚类标准误（百分点）", "p值", "对应组公司日数", "对应组提示公司日数",
        ]]
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"), flush=True)
        if save_dir:
            log(f"新结果保存于：{save_dir}")
    log("检验完成，临时残差矩阵已清理")


if __name__ == "__main__":
    main()
