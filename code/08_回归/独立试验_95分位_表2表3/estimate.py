"""95分位独立试验：按原表2、表3口径分块估计，并核验完整季度。

只读取准备模块提供的面板，写入调用者指定的新结果目录。季度残差缓存
放在调用者指定的工作目录，估计结束即清理；不写入既有正式结果。
"""

from __future__ import annotations

import os

for _thread_variable in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "2"

import gc
import importlib.util
import json
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
try:
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError:  # 该独立试验不依赖线程池控制包即可完成估计
    from contextlib import contextmanager

    @contextmanager
    def threadpool_limits(*args, **kwargs):
        yield


ROOT = Path(__file__).resolve().parents[3]
_HELPER_PATH = ROOT / "code/08_回归/08_回归_零散/扩展_散户交易占比异质性.py"
_helper_spec = importlib.util.spec_from_file_location("_q95_regression_helper", _HELPER_PATH)
if _helper_spec is None or _helper_spec.loader is None:
    raise ImportError(f"无法读取辅助估计模块：{_HELPER_PATH}")
_helper = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helper)

CONTROLS = list(_helper.CONTROLS)
RETURNS = {h: f"chAdjreturn_t_tp{h}_Recall_w" for h in (1, 10)}
CUE = "cue95"
SPECS = (
    (1, [], "none"),
    (2, CONTROLS[:21], "none"),
    (3, CONTROLS, "none"),
    (4, CONTROLS, "date"),
    (5, CONTROLS, "fq"),
    (6, CONTROLS, "both"),
)


def _log(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"[95分位估计] {message}", flush=True)


def _finite_mask(data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    mask = np.ones(len(data), dtype=bool)
    for column in columns:
        mask &= np.isfinite(data[column].to_numpy(copy=False))
    return mask


def _quarter_indices(data: pd.DataFrame) -> list[tuple[int, np.ndarray]]:
    return [
        (int(q), np.asarray(positions, dtype="int64"))
        for q, positions in data.groupby("q", sort=True, observed=True).indices.items()
    ]


def _blocks(
    data: pd.DataFrame,
    quarters: list[tuple[int, np.ndarray]],
    valid: np.ndarray,
    effects: str,
) -> tuple[list[tuple[int, np.ndarray]], int]:
    """只在模型自身有效样本上删单例；季度之间没有共同固定效应组。"""
    firms, days = data.firm.to_numpy(), data.day.to_numpy()
    result = []
    removed = 0
    for q, positions in quarters:
        positions = positions[valid[positions]]
        if not len(positions):
            continue
        if effects == "both":
            keep = _helper.remove_singletons(firms[positions], days[positions])
        elif effects in {"date", "fq"}:
            values = days[positions] if effects == "date" else firms[positions]
            codes, _ = pd.factorize(values, sort=False)
            keep = np.bincount(codes)[codes] > 1
        else:
            keep = np.ones(len(positions), dtype=bool)
        removed += int((~keep).sum())
        positions = positions[keep]
        if len(positions):
            result.append((q, positions))
    if not result:
        raise ValueError("模型在删除自身缺失值和固定效应单例后没有样本。")
    return result, removed


def _demean(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    codes, _ = pd.factorize(labels, sort=False)
    count = np.bincount(codes).astype("float64")
    means = np.empty((len(count), values.shape[1]), dtype="float64")
    for column in range(values.shape[1]):
        means[:, column] = np.bincount(codes, weights=values[:, column]) / count
    return values - means[codes]


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _fit_group(
    data: pd.DataFrame,
    quarters: list[tuple[int, np.ndarray]],
    valid: np.ndarray,
    *,
    table: int,
    spec: int,
    controls: list[str],
    effects: str,
    outcomes: list[tuple[int, str, float]],
    work: Path,
    quiet: bool,
) -> list[dict]:
    """同一有效样本的收益可共享投影；订单失衡各窗口保持独立样本。"""
    columns = [CUE, *controls]
    p, m = len(columns), len(outcomes)
    blocks, removed = _blocks(data, quarters, valid, effects)
    n = sum(len(positions) for _, positions in blocks)
    firms_all = np.unique(np.concatenate([
        np.unique(data.firm.to_numpy()[positions]) for _, positions in blocks
    ]))
    days_all = np.unique(np.concatenate([
        np.unique(data.day.to_numpy()[positions]) for _, positions in blocks
    ]))
    gf, gd = len(firms_all), len(days_all)
    if min(gf, gd) <= 1:
        raise ValueError("双向聚类至少需要两个公司簇和两个日期簇。")

    # 表2有显式常数。无固定效应时必须用全部有效样本均值，不能季内去均值。
    center = None
    if effects == "none":
        center = np.array([
            sum(float(data[c].to_numpy()[positions].sum(dtype="float64"))
                for _, positions in blocks) / n
            for c in [*columns, *[outcome[1] for outcome in outcomes]]
        ])
        center[p:] *= np.array([outcome[2] for outcome in outcomes])

    a, b = np.zeros((p, p)), np.zeros((p, m))
    sy, syy, within_yy = np.zeros(m), np.zeros(m), np.zeros(m)
    cue_n = nfq = 0
    cache = Path(tempfile.mkdtemp(prefix=f"q95_t{table}_s{spec}_", dir=work))
    paths = []
    started = time.monotonic()
    try:
        for index, (q, positions) in enumerate(blocks):
            x = np.column_stack([data[c].to_numpy()[positions] for c in columns]).astype("float64")
            y = np.column_stack([
                data[c].to_numpy()[positions].astype("float64") * scale
                for _, c, scale in outcomes
            ])
            firms, days = data.firm.to_numpy()[positions], data.day.to_numpy()[positions]
            cue_n += int(x[:, 0].sum())
            nfq += len(np.unique(firms))
            sy += y.sum(axis=0)
            syy += np.einsum("ij,ij->j", y, y)
            values = np.column_stack([x, y])
            if effects == "both":
                residualized = _helper.absorb_block(values, firms, days)
            elif effects == "date":
                residualized = _demean(values, days)
            elif effects == "fq":
                residualized = _demean(values, firms)
            else:
                residualized = values - center
            xr, yr = residualized[:, :p], residualized[:, p:]
            a += xr.T @ xr
            b += xr.T @ yr
            within_yy += np.einsum("ij,ij->j", yr, yr)
            path = cache / f"{index:03d}.npz"
            np.savez(path, x=xr, y=yr, firm=firms, day=days)
            paths.append(path)
            if index % 20 == 0:
                _log(f"表{table}规格{spec}：已处理{q // 4}年第{q % 4 + 1}季度", quiet)
        beta, _, active, condition = _helper.solve_moments(a, b)
        aa = a[np.ix_(active, active)]
        # 标准化后求逆，避免量纲差异放大正规方程误差。
        scale = np.sqrt(np.diag(aa))
        inverse = np.linalg.inv(aa / scale[:, None] / scale[None, :]) / scale[:, None] / scale[None, :]
        k = len(active) + int(table == 2)
        fe_rank = {"none": 0, "date": gd, "fq": nfq, "both": nfq + gd - 1}[effects]
        df_resid = n - len(active) - fe_rank - int(table == 2 and effects == "none")
        if n <= k or df_resid <= 0:
            raise ValueError("模型残差自由度不为正。")

        # 保存全部斜率的得分，而非只计算提示系数影响函数，确保全部控制也有标准误。
        fs = np.zeros((m, gf, len(active)))
        ds = np.zeros((m, gd, len(active)))
        observation_meat = np.zeros((m, len(active), len(active)))
        sse = np.zeros(m)
        for path in paths:
            with np.load(path) as block:
                xr, yr = block["x"][:, active], block["y"]
                errors = yr - xr @ beta[active]
                sse += np.einsum("ij,ij->j", errors, errors)
                fi = np.searchsorted(firms_all, block["firm"])
                di = np.searchsorted(days_all, block["day"])
                for outcome in range(m):
                    score = xr * errors[:, [outcome]]
                    observation_meat[outcome] += score.T @ score
                    for column in range(len(active)):
                        fs[outcome, :, column] += np.bincount(fi, weights=score[:, column], minlength=gf)
                        ds[outcome, :, column] += np.bincount(di, weights=score[:, column], minlength=gd)
        total_yy = syy - sy * sy / n
        common_scale = (n - 1) / (n - k)
        rows = []
        for outcome, (h, dependent, _) in enumerate(outcomes):
            # 公司×日期唯一，故交集簇等于逐观测簇；修正项与原吸收估计器一致。
            meat = (
                gf / (gf - 1) * (fs[outcome].T @ fs[outcome])
                + gd / (gd - 1) * (ds[outcome].T @ ds[outcome])
                - n / (n - 1) * observation_meat[outcome]
            ) * common_scale
            covariance = inverse @ meat @ inverse
            covariance = (covariance + covariance.T) / 2
            with np.errstate(invalid="ignore", divide="ignore"):
                se = np.sqrt(np.diag(covariance))
                tvalues = beta[active, outcome] / se
            pvalues = 2 * stats.t.sf(np.abs(tvalues), df_resid)
            lookup = {int(column): i for i, column in enumerate(active)}
            coefficients = []
            for column, variable in enumerate(columns):
                estimated = column in lookup
                position = lookup.get(column)
                coefficients.append({
                    "variable": variable,
                    "beta": _number(beta[column, outcome]) if estimated else None,
                    "se": _number(se[position]) if estimated else None,
                    "p": _number(pvalues[position]) if estimated else None,
                    "t": _number(tvalues[position]) if estimated else None,
                    "estimated": estimated,
                })
            cue = coefficients[0]
            if cue["se"] is None or cue["se"] <= 0:
                raise ValueError("提示变量的双向聚类标准误无效。")
            rows.append({
                "table": table, "spec": spec, "h": h, "dependent": dependent,
                "beta": cue["beta"], "se": cue["se"], "p": cue["p"], "t": cue["t"],
                "n": n, "cue_n": cue_n, "r2": float(1 - sse[outcome] / total_yy[outcome]),
                "within_r2": float(1 - sse[outcome] / within_yy[outcome]),
                "df": int(df_resid), "firm_clusters": gf, "date_clusters": gd,
                "firm_quarter_groups": nfq, "fe_rank": int(fe_rank), "cov_k": k,
                "controls": len(controls), "active_slopes": len(active),
                "date_fe": effects in {"date", "both"},
                "firm_quarter_fe": effects in {"fq", "both"},
                "explicit_constant": table == 2, "singleton_removed": removed,
                "condition": condition, "unit": "百分点", "coefficients": coefficients,
            })
        _log(f"表{table}规格{spec}完成：{n:,}行，耗时{time.monotonic() - started:.1f}秒", quiet)
        return rows
    finally:
        shutil.rmtree(cache)
        gc.collect()


def estimate_models(data: pd.DataFrame, work: Path, *, quiet: bool = False) -> dict:
    """执行六种表2规格和两种表3窗口；此接口亦用于完整季度数值核验。"""
    work.mkdir(parents=True, exist_ok=True)
    quarters = _quarter_indices(data)
    table2, table3 = [], []
    for spec, controls, effects in SPECS:
        x_valid = _finite_mask(data, [CUE, *controls])
        masks = {h: x_valid & np.isfinite(data[c].to_numpy()) for h, c in RETURNS.items()}
        if np.array_equal(masks[1], masks[10]):
            table2.extend(_fit_group(
                data, quarters, masks[1], table=2, spec=spec, controls=controls,
                effects=effects, outcomes=[(h, c, 100.0) for h, c in RETURNS.items()],
                work=work, quiet=quiet,
            ))
        else:
            for h, c in RETURNS.items():
                table2.extend(_fit_group(
                    data, quarters, masks[h], table=2, spec=spec, controls=controls,
                    effects=effects, outcomes=[(h, c, 100.0)], work=work, quiet=quiet,
                ))
    # 订单失衡已在完整主面板中预缩尾，单位已经是百分点。每个窗口单独删缺失。
    for h in (1, 10):
        column = f"oib_{h}"
        table3.extend(_fit_group(
            data, quarters, _finite_mask(data, [CUE, *CONTROLS, column]),
            table=3, spec=6, controls=CONTROLS, effects="both",
            outcomes=[(h, column, 1.0)], work=work, quiet=quiet,
        ))
    return {"table2": table2, "table3": table3}


def validate_estimator(data: pd.DataFrame, work: Path) -> list[dict]:
    """完整实际季度的14模型，对照原吸收估计器的系数、标准误、概率和拟合度。"""
    from linearmodels.iv import AbsorbingLS
    from linearmodels.panel.utility import AbsorbingEffectWarning

    counts = data.loc[data[CUE] == 1].groupby("q", observed=True).size()
    if counts.empty:
        raise ValueError("实际面板没有95分位提示，无法进行实际季度验证。")
    # 提示最多的季度保留该季度全部观测，不抽公司、不抽日期。
    q = int(counts.idxmax())
    sample = data.loc[data.q == q].reset_index(drop=True)
    _log(f"实际季度数值核验：{q // 4}年第{q % 4 + 1}季度，{len(sample):,}行", False)
    ours = estimate_models(sample, work, quiet=True)
    validation = []
    quarters = _quarter_indices(sample)
    for actual in [*ours["table2"], *ours["table3"]]:
        table, spec, h = actual["table"], actual["spec"], actual["h"]
        controls = list(SPECS[spec - 1][1]) if table == 2 else CONTROLS
        effects = SPECS[spec - 1][2] if table == 2 else "both"
        columns = [CUE, *controls]
        outcome = RETURNS[h] if table == 2 else f"oib_{h}"
        valid = _finite_mask(sample, [*columns, outcome])
        blocks, _ = _blocks(sample, quarters, valid, effects)
        positions = np.concatenate([positions for _, positions in blocks])
        frame = sample.iloc[positions].reset_index(drop=True)
        exog = frame[columns].astype("float64")
        if table == 2:
            exog.insert(0, "const", 1.0)
        absorb_columns = {}
        if effects in {"fq", "both"}:
            absorb_columns["firm_quarter"] = pd.Categorical(frame.firm)
        if effects in {"date", "both"}:
            absorb_columns["date"] = pd.Categorical(frame.day)
        absorb = pd.DataFrame(absorb_columns) if absorb_columns else None
        clusters = pd.DataFrame({"firm": frame.firm, "date": frame.day})
        y = frame[outcome].astype("float64") * (100.0 if table == 2 else 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AbsorbingEffectWarning)
            reference = AbsorbingLS(y, exog, absorb=absorb, drop_absorbed=True).fit(
                cov_type="clustered", clusters=clusters, debiased=True, use_cache=False,
            )
        estimated = [row for row in actual["coefficients"] if row["estimated"]]
        names = [row["variable"] for row in estimated]
        reference_names = [name for name in reference.params.index if name != "const"]
        if names != reference_names:
            raise AssertionError(f"表{table}规格{spec}窗口{h}保留列不一致：{names}，{reference_names}")
        errors = {}
        for key, series in (
            ("beta", reference.params), ("se", reference.std_errors), ("p", reference.pvalues),
        ):
            expected = series.loc[names].to_numpy()
            observed = np.array([row[key] if row[key] is not None else np.nan for row in estimated])
            errors[key] = float(np.nanmax(np.abs(observed - expected)))
            if not np.allclose(observed, expected, rtol=2e-5, atol=2e-8, equal_nan=True):
                raise AssertionError(f"实际季度表{table}规格{spec}窗口{h}的{key}不一致，最大误差{errors[key]}")
        if actual["n"] != reference.nobs or actual["df"] != reference.df_resid:
            raise AssertionError(f"实际季度表{table}规格{spec}窗口{h}样本量或自由度不一致。")
        r2_error = abs(actual["r2"] - float(reference.rsquared))
        if r2_error > 2e-9:
            raise AssertionError(f"实际季度表{table}规格{spec}窗口{h}总体R方不一致：{r2_error}")
        validation.append({
            "test": "实际完整季度与原吸收估计器对照", "q": q,
            "table": table, "spec": spec, "h": h, "n": actual["n"],
            "df": actual["df"], "r2_error": r2_error,
            "max_beta_error": errors["beta"], "max_se_error": errors["se"],
            "max_p_error": errors["p"], "active_slopes": len(names), "passed": True,
        })
    return validation


def _panel_checks(data: pd.DataFrame) -> dict:
    required = ["firm", "day", "q", CUE, *CONTROLS, *RETURNS.values(), "oib_1", "oib_10"]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"准备面板缺少变量：{missing}")
    if data.empty or data[["firm", "day", "q"]].isna().any().any():
        raise ValueError("准备面板为空或公司、日期、季度键缺失。")
    if data.duplicated(["firm", "day"]).any():
        raise ValueError("准备面板不是公司×日期唯一，双向聚类交集公式不适用。")
    if not np.isin(data[CUE].to_numpy(), [0, 1]).all():
        raise ValueError("95分位提示列包含缺失或非零一数值。")
    if not data.q.between(2005 * 4, 2025 * 4 + 3).all():
        raise ValueError("准备面板含2005—2025年以外样本。")
    return {
        "panel_rows": len(data), "cue_n": int(data[CUE].sum()),
        "table2_all_required_finite": bool(_finite_mask(data, [CUE, *CONTROLS, *RETURNS.values()]).all()),
        "expected_reference_rows": 11110094,
        "matches_expected_reference_rows": len(data) == 11110094,
    }


def estimate_all(panel_path: Path, work: Path, output: Path) -> dict:
    """读取面板、完成实际季度核验及全样本估计，输出新试验结果。"""
    panel_path, work, output = Path(panel_path), Path(work), Path(output)
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        data = pd.read_parquet(panel_path)
        checks = _panel_checks(data)
        validation = validate_estimator(data, work)
        models = estimate_models(data, work)
    result = {
        "metadata": {
            "sample": "规律公司主样本", "start_year": 2005, "end_year": 2025,
            "cue": CUE, "cue_percentile": 95, "unit": "百分点",
            "cluster": ["被提示公司", "日期"], "debiased": True,
            "panel_path": str(panel_path.resolve()), **checks,
            "r2_definition": "1−残差平方和/原始因变量全样本去均值总平方和",
            "table2_constant": True, "table3_constant": False,
            "table3_missing_rule": "每个窗口分别删除自身缺失值和递归固定效应单例",
        },
        **models, "validation": validation,
    }
    all_rows = [*result["table2"], *result["table3"]]
    summary = [{k: v for k, v in row.items() if k != "coefficients"} for row in all_rows]
    coefficients = [
        {"table": row["table"], "spec": row["spec"], "h": row["h"], **coefficient}
        for row in all_rows for coefficient in row["coefficients"]
    ]
    (output / "regression_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coefficients).to_csv(output / "coefficients.csv", index=False, encoding="utf-8-sig")
    (output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result
