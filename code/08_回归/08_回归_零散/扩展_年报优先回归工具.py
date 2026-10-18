"""年报优先独立试验的回归工具；导入时不读取数据、不运行回归。

三个提示口径分别替换同一基准模型的提示项，不把三个提示同时放入模型。
每个收益窗口单独取三个提示、原27项控制和该窗口收益均有效的共同样本，
递归删除公司季度和日期单例。收益乘100后以百分点报告。
吸收公司×年度季度和日期固定效应，按公司、实际日期双向聚类。
年报优先减最新报告优先是两个相关估计之差；用共同样本的影响函数差
计算标准误，不能用两项标准误平方和假定两次估计独立。

只读导入已验证的精确吸收工具。临时残差仅写入调用方指定的项目工作
目录并在结束时清理；本文件不保存正式结果，也不修改任何现有文件。
"""

from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "2")

import importlib.util
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


_SPEC = importlib.util.spec_from_file_location(
    "annual_priority_absorb_tools", Path(__file__).with_name("扩展_散户交易占比异质性.py")
)
helper = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(helper)

CUES = [helper.CUE, "cue_latest_safe", "cue_annual_safe"]
MODEL_LABELS = ["现有正式提示口径", "最新报告优先安全重建", "年报优先安全重建"]
CONTROLS, RETURNS = helper.CONTROLS, helper.RETURNS


def _blocks(panel: pd.DataFrame, h: int) -> tuple[list[tuple[int, np.ndarray]], int]:
    """窗口内三个模型共用收益、控制及提示均有效的公司日。"""
    valid = np.ones(len(panel), dtype=bool)
    for name in [*CUES, *CONTROLS, RETURNS[h]]:
        valid &= np.isfinite(panel[name].to_numpy())
    before = int(valid.sum())
    qvalues, firms, days = (panel[name].to_numpy() for name in ("q", "firm", "day"))
    blocks = []
    for q in sorted(np.unique(qvalues[valid])):
        positions = np.flatnonzero(valid & (qvalues == q))
        keep = helper.remove_singletons(firms[positions], days[positions])
        if keep.any():
            blocks.append((int(q), positions[keep]))
    if not blocks:
        raise ValueError(f"收益窗口{h}没有共同有效样本。")
    return blocks, before


def _row(h: int, label: str, estimate: float, variance: float, degrees: int,
         n: int, firms: int, days: int, cue_count: int | None, condition: float,
         status: str = "可估计") -> dict:
    """所有推断字段统一使用双侧检验和残差自由度。"""
    if variance < 0 or not np.isfinite(variance):
        se, p, lower, upper = (float("nan"),) * 4
        status = "双向聚类差异方差非正，不能推断"
    elif variance == 0:
        se = 0.0
        p = 1.0 if abs(estimate) <= 1e-12 else float("nan")
        lower, upper = estimate, estimate
        status = "两口径在样本内完全相同" if p == 1 else "零聚类方差，不能推断"
    else:
        se = float(np.sqrt(variance))
        p = float(2 * stats.t.sf(abs(estimate / se), degrees))
        width = float(stats.t.ppf(.975, degrees) * se)
        lower, upper = estimate - width, estimate + width
    return {
        "窗口": f"[t,t+{h}]", "结局": "特征调整收益", "结果项": label,
        "系数（百分点）": float(estimate), "聚类标准误（百分点）": se,
        "95%区间下界": lower, "95%区间上界": upper, "p值": p,
        "显著性": "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else "",
        "共同样本量": n, "提示公司日数": cue_count,
        "公司聚类数": firms, "日期聚类数": days, "残差自由度": degrees,
        "原27项控制": "是", "固定效应": "公司×年度季度、实际日期",
        "聚类方式": "公司、实际日期双向聚类",
        "标准化交叉积条件数": condition, "推断状态": status,
        "差异检验口径": "两个单独回归的相关系数差，按共同样本影响函数差推断" if "减" in label else "单独替换提示变量的回归",
    }


def _estimate_window(panel: pd.DataFrame, h: int, workspace: Path, *, quiet: bool = False
                     ) -> tuple[list[dict], list[dict], dict]:
    blocks, before = _blocks(panel, h)
    positions = np.concatenate([p for _, p in blocks])
    n = len(positions)
    firm_values = np.unique(panel.firm.to_numpy()[positions])
    day_values = np.unique(panel.day.to_numpy()[positions])
    nf, nd = len(firm_values), len(day_values)
    if nf < 2 or nd < 2:
        raise ValueError("公司或日期聚类数不足。")
    names = [*CUES, *CONTROLS]
    model_columns = [np.r_[m, np.arange(3, len(names))] for m in range(3)]
    moments = np.zeros((len(names), len(names)))
    target = np.zeros((len(names), 1))
    # 与成熟吸收估计器相同，固定效应类别数按两个效应共用截距计数。
    fe_count = 0
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.mkdtemp(prefix="年报优先回归残差_", dir=workspace))
    paths = []
    try:
        for number, (q, at) in enumerate(blocks):
            frame = panel.iloc[at]
            values = np.column_stack([
                frame[names].to_numpy(dtype=float),
                frame[RETURNS[h]].to_numpy(dtype=float) * 100,
            ])
            within = helper.absorb_block(values, frame.firm.to_numpy(), frame.day.to_numpy())
            x, y = within[:, :-1], within[:, -1:]
            moments += x.T @ x
            target += x.T @ y
            fe_count += frame.firm.nunique() + frame.day.nunique()
            path = cache / f"{number:03d}.npz"
            np.savez(path, x=x, y=y[:, 0], firm=frame.firm.to_numpy(), day=frame.day.to_numpy())
            paths.append(path)
            if not quiet and number % 20 == 0:
                helper.log(f"年报优先比较，收益窗口{h}：已吸收{helper.quarter_label(q)}")

        betas, directions, active_columns, conditions = [], [], [], []
        for columns in model_columns:
            beta, direction, active, condition = helper.solve_moments(
                moments[np.ix_(columns, columns)], target[columns]
            )
            betas.append(beta[:, 0])
            directions.append(direction)
            active_columns.append(active)
            conditions.append(condition)

        firm_scores, day_scores = np.zeros((nf, 3)), np.zeros((nd, 3))
        intersection = np.zeros((3, 3))
        for path in paths:
            with np.load(path) as block:
                scores = np.empty((len(block["y"]), 3))
                for m, columns in enumerate(model_columns):
                    x = block["x"][:, columns]
                    residual = block["y"] - x @ betas[m]
                    scores[:, m] = (x @ directions[m]) * residual
                fi = np.searchsorted(firm_values, block["firm"])
                di = np.searchsorted(day_values, block["day"])
                for m in range(3):
                    firm_scores[:, m] += np.bincount(fi, weights=scores[:, m], minlength=nf)
                    day_scores[:, m] += np.bincount(di, weights=scores[:, m], minlength=nd)
                intersection += scores.T @ scores

        k = np.array([len(active) for active in active_columns])
        degrees = n - k - (fe_count - 1)
        if (degrees <= 0).any():
            raise ValueError("残差自由度非正。")
        # 每模型的有限样本修正取平方根后作用于影响函数，因此对角线与
        # 各自回归完全相同，非对角线保留两次估计的相关性。
        corrections = np.sqrt((n - 1) / (n - k))
        covariance = (
            nf / (nf - 1) * (firm_scores.T @ firm_scores)
            + nd / (nd - 1) * (day_scores.T @ day_scores)
            - n / (n - 1) * intersection
        ) * corrections[:, None] * corrections[None, :]
        covariance = (covariance + covariance.T) / 2
        if not np.isfinite(covariance).all() or (np.diag(covariance) <= 0).any():
            raise ValueError("某提示口径的双向聚类方差无效。")
        estimates = np.array([beta[0] for beta in betas])
        counts = [int(panel[name].to_numpy()[positions].sum()) for name in CUES]
        rows = [_row(h, MODEL_LABELS[m], estimates[m], float(covariance[m, m]),
                     int(degrees[m]), n, nf, nd, counts[m], conditions[m]) for m in range(3)]
        contrast = np.array([0., -1., 1.])
        difference = float(contrast @ estimates)
        variance = float(contrast @ covariance @ contrast)
        identical = np.array_equal(
            panel[CUES[1]].to_numpy()[positions], panel[CUES[2]].to_numpy()[positions]
        )
        if identical:
            difference, variance = 0.0, 0.0
        elif variance < 0 and abs(variance) <= np.max(np.diag(covariance)) * 1e-10:
            variance = 0.0
        rows.append(_row(h, "年报优先减最新报告优先", difference, variance,
                         int(degrees.min()), n, nf, nd, None, float(max(conditions))))
        checks = [
            {"核验": "收益窗口共同样本", "窗口": h, "完整有效行数": before,
             "递归单例删除行数": before - n, "最终共同样本量": n,
             "三个模型同一样本": True, "通过": True},
            {"核验": "提示口径差异", "窗口": h,
             "最新与年报提示不同的公司日": int(np.count_nonzero(
                 panel[CUES[1]].to_numpy()[positions] != panel[CUES[2]].to_numpy()[positions])),
             "年报与最新系数协方差": float(covariance[1, 2]),
             "相同口径零差异核验": bool(not identical or (difference == 0 and variance == 0)),
             "通过": True},
        ]
        details = {"positions": positions, "beta": betas, "active": active_columns,
                   "covariance": covariance, "degrees": degrees,
                   "model_columns": model_columns, "names": names}
        return rows, checks, details
    finally:
        shutil.rmtree(cache)


def estimate_comparison(panel: pd.DataFrame, workspace: Path) -> tuple[list[dict], list[dict]]:
    """返回核心结果行、样本及差异核验行；不保存正式输出。"""
    required = ["firm", "day", "q", *CUES, *CONTROLS, *RETURNS.values()]
    missing = [name for name in required if name not in panel]
    if missing:
        raise ValueError(f"输入面板缺少列：{missing}")
    if panel.duplicated(["firm", "day"]).any():
        raise ValueError("面板存在重复公司日，双向聚类交集不能按观察值处理。")
    for name in CUES:
        values = panel[name].dropna()
        if not values.isin((0, 1)).all():
            raise ValueError(f"提示变量{name}不是零一变量。")
    results, checks = [], []
    for h in (1, 10):
        rows, audit, _ = _estimate_window(panel, h, Path(workspace))
        results.extend(rows)
        checks.extend(audit)
    return results, checks


def validate_estimator(panel: pd.DataFrame, workspace: Path, *, target_rows: int = 200000
                       ) -> list[dict]:
    """取一个实际季度约20万行，对照成熟估计器系数及聚类协方差。

    尽可能保留全部提示公司，再按确定性间隔选参照公司。只验证数值实现；
    该样本估计不作为年报优先研究结果，正式结果仍由全量共同样本给出。
    """
    from linearmodels.iv import AbsorbingLS

    complete = np.ones(len(panel), dtype=bool)
    for name in [*CUES, *CONTROLS, RETURNS[1]]:
        complete &= np.isfinite(panel[name].to_numpy())
    quarterly = panel.loc[complete].groupby("q")[CUES].sum()
    candidate_quarters = quarterly.index[(quarterly > 0).all(axis=1)].to_numpy()
    if not len(candidate_quarters):
        raise ValueError("实际样本中没有三个提示口径均有事件的验证季度。")
    q = int(candidate_quarters[-1])
    candidate = panel.loc[complete & (panel.q.to_numpy() == q)].copy()
    if len(candidate) > target_rows * 1.1:
        cue_firms = candidate.loc[candidate[CUES].any(axis=1), "firm"].unique()
        other_firms = np.setdiff1d(candidate.firm.unique(), cue_firms)
        per_firm = max(1., len(candidate) / candidate.firm.nunique())
        needed = max(0, int(target_rows / per_firm) - len(cue_firms))
        if needed < len(other_firms):
            selected = other_firms[np.linspace(0, len(other_firms) - 1, needed, dtype=int)] if needed else []
            candidate = candidate.loc[candidate.firm.isin(np.union1d(cue_firms, selected))]
    subset = candidate.reset_index(drop=True)
    _, _, details = _estimate_window(subset, 1, Path(workspace), quiet=True)
    reg = subset.iloc[details["positions"]].reset_index(drop=True)
    absorb = pd.DataFrame({
        "firm_quarter": pd.Categorical(reg.firm.astype(str) + "_" + reg.q.astype(str)),
        "date": pd.Categorical(reg.day),
    })
    clusters = pd.DataFrame({"firm": reg.firm, "date": reg.day})
    checks = []
    for m, columns in enumerate(details["model_columns"]):
        rhs = [details["names"][columns[j]] for j in details["active"][m]]
        result = AbsorbingLS(
            reg[RETURNS[1]].astype(float) * 100, reg[rhs].astype(float), absorb=absorb
        ).fit(cov_type="clustered", clusters=clusters, debiased=True)
        name = CUES[m]
        actual, expected = details["beta"][m][0], float(result.params[name])
        actual_cov, expected_cov = float(details["covariance"][m, m]), float(result.cov.loc[name, name])
        if not np.isclose(actual, expected, atol=2e-8, rtol=2e-6):
            raise AssertionError(f"{MODEL_LABELS[m]}实际数据系数核验失败：{actual}，{expected}")
        if not np.isclose(actual_cov, expected_cov, atol=2e-9, rtol=2e-5):
            raise AssertionError(f"{MODEL_LABELS[m]}实际数据聚类协方差核验失败：{actual_cov}，{expected_cov}")
        checks.append({"核验": "实际季度与成熟吸收估计器对照", "模型": MODEL_LABELS[m],
                       "年度季度编码": q, "样本量": len(reg),
                       "系数误差": float(actual - expected),
                       "聚类协方差误差": float(actual_cov - expected_cov), "通过": True})
    helper.log(f"年报优先估计器实际样本核验通过：{len(reg):,}行、三个提示口径")
    return checks
