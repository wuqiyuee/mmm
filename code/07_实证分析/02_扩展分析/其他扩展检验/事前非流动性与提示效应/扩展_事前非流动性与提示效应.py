# 本文件由旧零散回归目录迁移；仅调整项目定位、公共工具引用和任务输出位置。
"""在修正后原模式公司面板上，独立检验事前非流动性与提示效应的交互。

默认2005—2025年，不重新构造提示关系，不使用已知误识别的旧版提示事件。
只读复用“扩展_散户交易占比异质性.py”的精确固定效应吸收与求解工具，
不运行该脚本主程序，不修改任何现有脚本、输入或既有结果。
直接运行只打印；--save-dir（保存目录）会新建时间戳结果目录。
大型临时矩阵仅保存在项目work/，完成后自动清理。

原始事前指标：过去20个市场交易日[t-20,t-1]中，
  日均(|含红利日收益率| / 该股票日成交金额) * 1e8。
按市场交易日序号定位，不能跳过停牌向前凑足20条；至少15个有效日。
历史股票日必须匹配基础清洗样本，真实非填充、允许的交易状态、
有效收益及严格为正的成交金额；缺失不填0。成交金额单位为人民币元。

为避免原始指标长尾主导，预先固定主交互使用同日横截面百分位：
  u=(当日平均名次-0.5)/当日有效公司数，z=2*u-1。
排名先在原主面板当日历史指标有效的公司中计算，再筛订单失衡等结局。
因此z=0对应非流动性50分位，z=-0.5/+0.5对应25/75分位。
这检验相对非流动性的线性梯度，不是原始阿米胡德数值的单位斜率。

回归：提示+非流动性主效应+提示×非流动性+原27项控制，吸收日期和
公司×年度季度，按公司、实际日期双向聚类，有限样本修正。
每窗口收益与成交量订单失衡共同有效，基准与交互模型完全共用样本。
订单失衡先在研究期原主面板合并总体上下各1%缩尾，再筛流动性样本；
收益沿用原面板的缩尾值。收益×100为百分点，订单失衡本已为百分点。
报告25/50/75分位的提示效应、75减25差异、置信区间与联合检验。
两日收益交互为主要检验；十一日及交易指标为补充，不作跨指标校正。
仅某一分位显著不能证明异质性，须依据交互差异检验。
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
TASK_OUTPUT_DIR = ROOT / "图表结果/02_扩展分析/其他扩展检验/事前非流动性与提示效应"
HELPER = (ROOT / "code/07_实证分析/02_扩展分析/其他扩展检验/散户交易占比异质性/扩展_散户交易占比异质性.py")
_spec = importlib.util.spec_from_file_location("illiquidity_regression_tools", HELPER)
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)
log, CUE, CONTROLS = helper.log, helper.CUE, helper.CONTROLS
INPUTS = {
    "模式公司主面板": helper.PATHS["模式公司面板"],
    "日行情": ROOT / "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv",
    "基础清洗索引": helper.PATHS["基础样本索引"],
    "交易日历": ROOT / "中间结果/01_清洗后样本/市场交易日历_td.csv",
    "订单失衡": helper.PATHS["订单失衡表"],
}
LIQ, INTERACTION = "illiquidity_rank_centered", "cue_x_illiquidity"


def key(firm: np.ndarray, td: np.ndarray) -> np.ndarray:
    if np.any(td < -9999) or np.any(td >= 10000):
        raise ValueError("交易日序号超出键编码范围。")
    return firm.astype("int64") * 20000 + td.astype("int64") + 10000


def read_panel(args: argparse.Namespace, audit: list[dict]) -> pd.DataFrame:
    columns = ["StkcdRecall", "Trddt", "td", CUE, *CONTROLS, *helper.RETURNS.values()]
    pieces = []
    log("读取当前修正口径主面板，保留原提示、控制与收益")
    for number, chunk in enumerate(pd.read_csv(
        INPUTS["模式公司主面板"], usecols=columns, parse_dates=["Trddt"], chunksize=args.chunk_size,
        dtype={"StkcdRecall": "int32", "td": "int32", **{c: "float32" for c in [CUE, *CONTROLS, *helper.RETURNS.values()]}},
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
    if panel.empty or panel.duplicated(["firm", "td"]).any() or not panel[CUE].isin([0, 1]).all():
        raise ValueError("研究期面板为空、公司日期重复或提示变量不是零一。")
    audit.append({"阶段": "主面板", "项目": "研究期公司日", "数值": len(panel)})
    return panel


def attach_illiquidity(panel: pd.DataFrame, args: argparse.Namespace, audit: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    calendar = pd.read_csv(INPUTS["交易日历"], usecols=["Trddt", "td"], parse_dates=["Trddt"]).drop_duplicates()
    if calendar.Trddt.duplicated().any() or calendar.td.duplicated().any():
        raise ValueError("市场日历存在冲突。")
    day_map = pd.Series(calendar.td.to_numpy(), index=calendar.Trddt)
    lower, upper = int(panel.td.min()) - args.window, int(panel.td.max()) - 1
    firms = panel.firm.unique()
    eligible = pd.read_csv(INPUTS["基础清洗索引"], usecols=["Stkcd", "td"], dtype={"Stkcd": "int32", "td": "int32"})
    eligible = eligible.loc[eligible.td.between(lower, upper) & eligible.Stkcd.isin(firms)]
    eligible_keys = key(eligible.Stkcd.to_numpy(), eligible.td.to_numpy())
    eligible_keys.sort()
    if np.any(eligible_keys[1:] == eligible_keys[:-1]):
        raise ValueError("基础清洗索引重复。")
    del eligible
    pieces, invalid, valid_days = [], 0, 0
    log("从真实交易日收益及成交金额构造日度阿米胡德值")
    columns = ["Stkcd", "Trddt", "Dretwd", "Dnvaltrd", "Markettype", "Trdsta", "Filling"]
    for number, chunk in enumerate(pd.read_csv(
        INPUTS["日行情"], usecols=columns, parse_dates=["Trddt"], chunksize=args.chunk_size,
        dtype={"Stkcd": "int32", **{c: "float64" for c in columns if c not in {"Stkcd", "Trddt"}}},
    ), 1):
        td = day_map.reindex(chunk.Trddt).to_numpy()
        temporal = np.isfinite(td) & (td >= lower) & (td <= upper) & chunk.Stkcd.isin(firms).to_numpy()
        chunk = chunk.loc[temporal].copy()
        if chunk.empty:
            continue
        td = td[temporal].astype("int32")
        _, in_pool = helper.matched_positions(eligible_keys, key(chunk.Stkcd.to_numpy(), td))
        valid = in_pool & np.isfinite(chunk.Dretwd.to_numpy()) & np.isfinite(chunk.Dnvaltrd.to_numpy())
        valid &= (chunk.Dnvaltrd > 0).to_numpy() & (chunk.Filling == 0).to_numpy()
        valid &= chunk.Markettype.isin((1, 4, 16, 32)).to_numpy() & chunk.Trdsta.isin((1, 4, 7, 10, 13)).to_numpy()
        invalid += int((in_pool & ~valid).sum())
        values = np.abs(chunk.Dretwd.to_numpy()[valid]) / chunk.Dnvaltrd.to_numpy()[valid] * 1e8
        pieces.append(pd.DataFrame({"firm": chunk.Stkcd.to_numpy()[valid], "td": td[valid], "daily_illiq": values}))
        valid_days += int(valid.sum())
        if number % 12 == 0:
            log(f"日行情已读取第{number}块，有效股票日{valid_days:,}行")
    raw = pd.concat(pieces, ignore_index=True).sort_values(["firm", "td"], kind="stable").reset_index(drop=True)
    if raw.duplicated(["firm", "td"]).any():
        raise ValueError("原日行情存在重复的有效公司日期。")
    del pieces, eligible_keys
    raw_groups = raw.groupby("firm", sort=False).indices
    panel_groups = panel.groupby("firm", sort=False).indices
    illiq = np.full(len(panel), np.nan)
    nvalid = np.zeros(len(panel), dtype="int16")
    checks = []
    log(f"按市场交易日计算前{args.window}日均值，最低{args.min_days}日有效")
    for number, (firm, positions) in enumerate(panel_groups.items(), 1):
        source = raw_groups.get(firm)
        if source is None:
            continue
        rows = raw.iloc[source]
        start = min(int(rows.td.min()), int(panel.td.iloc[positions].min()) - args.window)
        stop = max(int(rows.td.max()), int(panel.td.iloc[positions].max()))
        dense = np.full(stop - start + 1, np.nan)
        dense[rows.td.to_numpy() - start] = rows.daily_illiq.to_numpy()
        target = panel.td.iloc[positions].to_numpy() - start
        counts = np.r_[0, np.cumsum(np.isfinite(dense))]
        sums = np.r_[0.0, np.cumsum(np.nan_to_num(dense, nan=0.0))]
        count = counts[target] - counts[target - args.window]
        total = sums[target] - sums[target - args.window]
        eligible = count >= args.min_days
        illiq[positions[eligible]] = total[eligible] / count[eligible]
        nvalid[positions] = count
        # 确定性抽查不同公司的实际窗口；严格切片不跨过停牌追溯更早记录。
        if number % 59 == 0 and eligible.any():
            at = np.flatnonzero(eligible)[len(np.flatnonzero(eligible)) // 2]
            expected = np.nanmean(dense[target[at] - args.window:target[at]])
            actual = illiq[positions[at]]
            if not np.isclose(actual, expected, atol=1e-9, rtol=1e-8):
                raise AssertionError("过去交易日窗口直接切片核验失败。")
            checks.append({"核验": "市场交易日窗口直接切片", "公司": int(firm), "交易日序号": int(panel.td.iloc[positions[at]]), "误差": float(actual - expected), "通过": True})
    if np.any(illiq[np.isfinite(illiq)] < -1e-9):
        raise ValueError("非流动性指标出现负数。")
    illiq = np.where(np.isfinite(illiq), np.maximum(illiq, 0), np.nan)
    panel["amihud_pre"] = illiq
    panel["amihud_valid_days"] = nvalid
    # 平均名次处理并列值；百分位先在原研究股票池计算，不受后续结局缺失影响。
    rank = panel.groupby("day")["amihud_pre"].rank(method="average")
    count = panel.groupby("day")["amihud_pre"].transform("count")
    panel[LIQ] = (((rank - .5) / count) * 2 - 1).astype("float64")
    panel[INTERACTION] = panel[CUE].to_numpy() * panel[LIQ].to_numpy()
    audit.extend([
        {"阶段": "事前流动性", "项目": "基础样本内缺失收益成交额或资格异常的股票日", "数值": invalid},
        {"阶段": "事前流动性", "项目": "有效历史股票日", "数值": valid_days},
        {"阶段": "事前流动性", "项目": "事前有效日不足公司日", "数值": int(np.isnan(illiq).sum())},
        {"阶段": "事前流动性", "项目": "事前有效日不足的提示公司日", "数值": int((np.isnan(illiq) & (panel[CUE].to_numpy() == 1)).sum())},
    ])
    summary = panel.amihud_pre.describe(percentiles=[.01, .25, .5, .75, .99])
    audit.extend({"阶段": "原始非流动性分布", "项目": str(k), "数值": float(v)} for k, v in summary.items())
    del raw, raw_groups, panel_groups
    gc.collect()
    log(f"事前非流动性已匹配{int(panel.amihud_pre.notna().sum()):,}个公司日")
    return panel, checks


def attach_oib(panel: pd.DataFrame, args: argparse.Namespace, audit: list[dict]) -> pd.DataFrame:
    panel_keys = key(panel.firm.to_numpy(), panel.td.to_numpy())
    order = np.argsort(panel_keys)
    sorted_keys = panel_keys[order]
    seen = np.zeros(len(panel), dtype=bool)
    values = {h: np.full(len(panel), np.nan, dtype="float32") for h in (1, 10)}
    log("匹配订单失衡，先按原主面板总体缩尾，再取流动性共同样本")
    for number, chunk in enumerate(pd.read_csv(
        INPUTS["订单失衡"], usecols=["Stkcd", "td", *helper.OIB.values()],
        dtype={"Stkcd": "int32", "td": "int32", **{c: "float32" for c in helper.OIB.values()}},
        chunksize=args.chunk_size,
    ), 1):
        positions, matched = helper.matched_positions(sorted_keys, key(chunk.Stkcd.to_numpy(), chunk.td.to_numpy()))
        target = order[positions[matched]]
        if len(np.unique(target)) != len(target) or seen[target].any():
            raise ValueError("订单失衡公司日期重复。")
        seen[target] = True
        for h, c in helper.OIB.items():
            values[h][target] = chunk[c].to_numpy()[matched]
        if number % 12 == 0:
            log(f"订单失衡已读取第{number}块")
    for h, value in values.items():
        value[~np.isfinite(value)] = np.nan
        lower, upper = np.nanquantile(value, [.01, .99])
        panel[f"oib_{h}"] = np.clip(value, lower, upper)
        audit.extend([
            {"阶段": f"订单失衡窗口{h}", "项目": "原主面板缩尾下界", "数值": float(lower)},
            {"阶段": f"订单失衡窗口{h}", "项目": "原主面板缩尾上界", "数值": float(upper)},
        ])
    return panel


def common_blocks(panel: pd.DataFrame, h: int) -> list[tuple[int, np.ndarray]]:
    valid = np.isfinite(panel[LIQ].to_numpy())
    for c in [CUE, *CONTROLS, helper.RETURNS[h], f"oib_{h}"]:
        valid &= np.isfinite(panel[c].to_numpy())
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


def estimate(panel: pd.DataFrame, h: int, interaction: bool, workspace: Path, *, quiet: bool = False) -> tuple[list[dict], list[dict], dict]:
    started = time.monotonic()
    rhs = [CUE, LIQ, INTERACTION, *CONTROLS] if interaction else [CUE, *CONTROLS]
    focus = [0, 2] if interaction else [0]
    blocks = common_blocks(panel, h)
    all_positions = np.concatenate([p for _, p in blocks])
    firms, dates = np.unique(panel.firm.to_numpy()[all_positions]), np.unique(panel.day.to_numpy()[all_positions])
    p, nfoc, n = len(rhs), len(focus), len(all_positions)
    a, b = np.zeros((p, p)), np.zeros((p, 2))
    fe_count = 0
    cache = Path(tempfile.mkdtemp(prefix="流动性交互残差_", dir=workspace))
    paths = []
    try:
        for number, (q, positions) in enumerate(blocks):
            frame = panel.iloc[positions]
            y = np.column_stack([frame[helper.RETURNS[h]].to_numpy(dtype=float) * 100, frame[f"oib_{h}"].to_numpy(dtype=float)])
            residualized = helper.absorb_block(np.column_stack([frame[rhs].to_numpy(dtype=float), y]), frame.firm.to_numpy(), frame.day.to_numpy())
            x, yr = residualized[:, :p], residualized[:, p:]
            a += x.T @ x
            b += x.T @ yr
            fe_count += frame.firm.nunique() + frame.day.nunique()
            path = cache / f"{number:03d}.npz"
            np.savez(path, x=x, y=yr, firm=frame.firm.to_numpy(), day=frame.day.to_numpy())
            paths.append(path)
            if not quiet and number % 20 == 0:
                log(f"窗口{h}，{'交互模型' if interaction else '共同样本基准'}：已吸收{helper.quarter_label(q)}")
        beta, _, active, condition = helper.solve_moments(a, b)
        if not set(focus).issubset(active):
            raise ValueError("提示或交互项被固定效应完全吸收。")
        inverse = np.zeros((p, p))
        # 对角标准化求逆，与系数求解使用同一有效列及归一化矩阵。
        scale = np.sqrt(np.diag(a)[active])
        normalized = a[np.ix_(active, active)] / scale[:, None] / scale[None, :]
        inverse[np.ix_(active, active)] = np.linalg.inv(normalized) / scale[:, None] / scale[None, :]
        directions = inverse[:, focus]
        fs = np.zeros((len(firms), nfoc, 2))
        ds = np.zeros((len(dates), nfoc, 2))
        intersection = np.zeros((2, nfoc, nfoc))
        for path in paths:
            with np.load(path) as block:
                residuals = block["y"] - block["x"] @ beta
                leverage = block["x"] @ directions
                fi, di = np.searchsorted(firms, block["firm"]), np.searchsorted(dates, block["day"])
                for outcome in (0, 1):
                    scores = leverage * residuals[:, outcome, None]
                    for j in range(nfoc):
                        fs[:, j, outcome] += np.bincount(fi, weights=scores[:, j], minlength=len(firms))
                        ds[:, j, outcome] += np.bincount(di, weights=scores[:, j], minlength=len(dates))
                    intersection[outcome] += scores.T @ scores
        k = len(active)
        degrees = n - k - (fe_count - 1)
        if degrees <= 0:
            raise ValueError("残差自由度非正。")
        correction = (n - 1) / (n - k)
        covariances, rows, coefficient_rows, joints = [], [], [], []
        for outcome, label in enumerate(("特征调整收益", "小单成交量订单失衡均值")):
            covariance = correction * (
                len(firms) / (len(firms) - 1) * (fs[:, :, outcome].T @ fs[:, :, outcome])
                + len(dates) / (len(dates) - 1) * (ds[:, :, outcome].T @ ds[:, :, outcome])
                - n / (n - 1) * intersection[outcome]
            )
            covariance = (covariance + covariance.T) / 2
            if not np.isfinite(covariance).all() or (np.diag(covariance) <= 0).any():
                raise ValueError("提示及交互项聚类协方差无效。")
            covariances.append(covariance)
            contrasts = [("共同样本基准提示效应", np.array([1.0]))] if not interaction else [
                ("非流动性第25分位提示效应", np.array([1.0, -.5])),
                ("非流动性第50分位提示效应", np.array([1.0, 0.0])),
                ("非流动性第75分位提示效应", np.array([1.0, .5])),
                ("第75减第25分位差异（交互项）", np.array([0.0, 1.0])),
            ]
            cue_beta = beta[focus, outcome]
            for role, contrast in contrasts:
                effect, variance = float(contrast @ cue_beta), float(contrast @ covariance @ contrast)
                if variance <= 0:
                    raise ValueError("线性组合聚类方差无效。")
                se = np.sqrt(variance)
                pv = float(2 * stats.t.sf(abs(effect / se), degrees))
                rows.append({
                    "窗口": f"[t,t+{h}]", "结局": label, "模型": "事前非流动性交互" if interaction else "共同样本基准",
                    "结果项": role, "系数（百分点）": effect, "聚类标准误（百分点）": se,
                    "95%区间下界": effect - stats.t.ppf(.975, degrees) * se,
                    "95%区间上界": effect + stats.t.ppf(.975, degrees) * se,
                    "p值": pv, "显著性": "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .1 else "",
                    "共同样本量": n, "提示公司日数": int(panel[CUE].to_numpy()[all_positions].sum()),
                    "公司聚类数": len(firms), "日期聚类数": len(dates), "残差自由度": degrees,
                    "原27项控制": "是", "非流动性主效应": "是" if interaction else "否",
                    "固定效应": "公司×年度季度、实际日期", "聚类方式": "公司、实际日期双向聚类",
                    "标准化交叉积条件数": condition, "耗时秒": round(time.monotonic() - started, 2),
                })
            for j, c in enumerate(rhs):
                coefficient_rows.append({
                    "窗口": f"[t,t+{h}]", "结局": label, "模型": "事前非流动性交互" if interaction else "共同样本基准",
                    "变量": c, "系数（百分点）": float(beta[j, outcome]) if j in active else np.nan,
                    "状态": "已估计" if j in active else "被固定效应吸收",
                })
            if interaction:
                if np.linalg.eigvalsh(covariance).min() <= 0:
                    raise ValueError("联合提示及交互项的协方差非正定，不能做联合检验。")
                statistic = float(cue_beta @ np.linalg.solve(covariance, cue_beta))
                joints.append({"窗口": f"[t,t+{h}]", "结局": label, "假设": "提示系数与交互系数同时为零", "沃尔德统计量": statistic, "自由度": 2, "p值": float(stats.chi2.sf(statistic, 2))})
        return rows, coefficient_rows, {"beta": beta, "covariances": covariances, "active": active, "rhs": rhs, "focus": focus, "positions": all_positions, "joint_tests": joints}
    finally:
        shutil.rmtree(cache)


def validate_estimator(panel: pd.DataFrame, workspace: Path) -> list[dict]:
    from linearmodels.iv import AbsorbingLS
    complete = np.isfinite(panel[LIQ].to_numpy()) & np.isfinite(panel.oib_1.to_numpy())
    quarters = sorted(panel.loc[complete & (panel[CUE] == 1), "q"].unique())[-4:]
    candidate = panel.loc[complete & panel.q.isin(quarters)]
    cue_firms = candidate.loc[candidate[CUE] == 1, "firm"].unique()
    others = np.sort(candidate.firm.unique())[::max(1, candidate.firm.nunique() // 80)]
    subset = candidate.loc[candidate.firm.isin(np.union1d(cue_firms, others))].copy().reset_index(drop=True)
    checks = []
    for interaction in (False, True):
        _, _, details = estimate(subset, 1, interaction, workspace, quiet=True)
        reg = subset.iloc[details["positions"]].reset_index(drop=True)
        columns = [details["rhs"][j] for j in details["active"]]
        focus_names = [details["rhs"][j] for j in details["focus"]]
        absorb = pd.DataFrame({"firm_q": pd.Categorical(reg.firm.astype(str) + "_" + reg.q.astype(str)), "date": pd.Categorical(reg.day)})
        clusters = pd.DataFrame({"firm": reg.firm, "date": reg.day})
        for outcome, column in enumerate((helper.RETURNS[1], "oib_1")):
            y = reg[column].astype(float) * (100 if outcome == 0 else 1)
            result = AbsorbingLS(y, reg[columns].astype(float), absorb=absorb).fit(cov_type="clustered", clusters=clusters, debiased=True)
            expected = result.params.loc[focus_names].to_numpy()
            actual = details["beta"][details["focus"], outcome]
            expected_cov = result.cov.loc[focus_names, focus_names].to_numpy()
            actual_cov = details["covariances"][outcome]
            if not np.allclose(actual, expected, atol=1e-7, rtol=2e-6) or not np.allclose(actual_cov, expected_cov, atol=1e-8, rtol=2e-5):
                raise AssertionError("实际数据估计器系数或联合协方差核验失败。")
            checks.append({"核验": "与成熟吸收估计器对照", "模型": "交互" if interaction else "基准", "结局": column, "样本量": len(reg), "最大系数误差": float(np.max(np.abs(actual - expected))), "最大协方差误差": float(np.max(np.abs(actual_cov - expected_cov))), "通过": True})
    log("实际数据子样本的基准与交互系数、联合协方差核验全部通过")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--min-days", type=int, default=15)
    parser.add_argument("--chunk-size", type=int, default=400000)
    parser.add_argument("--save-dir", type=Path, nargs="?", const=TASK_OUTPUT_DIR, default=None, help="显式保存开关；不带目录时保存到本任务目录/时间戳，也可指定保存目录")
    args = parser.parse_args()
    if args.start_year > args.end_year or not 2 <= args.min_days <= args.window:
        parser.error("年份范围或有效日门槛无效。")
    for path in [*INPUTS.values(), HELPER]:
        if not path.exists():
            raise FileNotFoundError(path)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    destination = args.save_dir.resolve() / stamp if args.save_dir else None
    if destination:
        destination.mkdir(parents=True, exist_ok=False)
    parameters = {
        "运行时间": stamp, "起始年": args.start_year, "结束年": args.end_year,
        "事前窗口": args.window, "最低有效交易日": args.min_days,
        "原始指标": "过去市场交易日中|含红利日收益|/日成交金额（人民币元）的有效日均值乘1e8",
        "主交互转换": "当日原主面板中历史指标有效公司的平均名次百分位，z=2*(rank-0.5)/N-1；排名先于结局筛样本",
        "主检验": "两日特征调整收益的第75减第25非流动性分位提示效应差异",
        "补充检验": "十一日收益与两种订单失衡窗口，未作跨指标多重检验校正",
        "回归": "原27控制、非流动性主效应、提示、交互；公司季度与日期固定效应；公司日期双向聚类",
        "提示口径": "沿用当前主面板已修正提示变量，不改用已知有年报一季报滞后误识别的旧事件",
        "边界": "百分位线性梯度；非流动性也与规模波动性相关，交互不能独立确认价格传导因果机制",
        "脚本": str(Path(__file__).resolve()), "脚本SHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "只读工具脚本SHA256": hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        "输入": {name: {"路径": str(path), "大小": path.stat().st_size, "修改时间纳秒": path.stat().st_mtime_ns} for name, path in INPUTS.items()},
    }
    audit, rows, coefficients, joint_tests = [], [], [], []
    work_root = ROOT / "work"
    work_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="事前流动性检验_", dir=work_root) as temporary:
        workspace = Path(temporary)
        panel = read_panel(args, audit)
        panel, checks = attach_illiquidity(panel, args, audit)
        panel = attach_oib(panel, args, audit)
        checks.extend(validate_estimator(panel, workspace))
        for h in (1, 10):
            for interaction in (False, True):
                result, coefficient_rows, details = estimate(panel, h, interaction, workspace)
                rows.extend(result)
                coefficients.extend(coefficient_rows)
                joint_tests.extend(details["joint_tests"])
                log(f"窗口{h}，{'非流动性交互' if interaction else '共同样本基准'}完成，共同样本{result[0]['共同样本量']:,}行")
                if destination:
                    pd.DataFrame(rows).to_csv(destination / "核心结果与流动性差异.csv", index=False, encoding="utf-8-sig")
                    pd.DataFrame(coefficients).to_csv(destination / "全部系数.csv", index=False, encoding="utf-8-sig")
        if destination:
            pd.DataFrame(joint_tests).to_csv(destination / "提示及交互联合检验.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(audit).to_csv(destination / "数据与口径审计.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(checks).to_csv(destination / "窗口及估计器核验.csv", index=False, encoding="utf-8-sig")
            (destination / "参数与来源.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
        print(pd.DataFrame(rows)[["窗口", "结局", "结果项", "系数（百分点）", "聚类标准误（百分点）", "p值", "共同样本量"]].to_string(index=False, float_format=lambda v: f"{v:.7f}"), flush=True)
        log(f"结果位置：{destination if destination else '仅终端打印'}")
    log("检验完成，临时矩阵已清理")


if __name__ == "__main__":
    main()
