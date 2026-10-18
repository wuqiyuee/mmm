"""仅为95分位表2、表3独立试验准备面板，不修改既有输入或脚本。"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"

import gc
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PAIR = ["Annodt", "Stkcd", "StkcdRecall"]


def log(message: str) -> None:
    print(f"[{datetime.now(ZoneInfo('Asia/Shanghai')):%H:%M:%S}] {message}", flush=True)


def load_base():
    path = ROOT / "code/08_回归/08_回归_零散/扩展_年报优先提示关系对照.py"
    spec = importlib.util.spec_from_file_location("q95_prepare_existing_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def day_codes(values: pd.Series) -> np.ndarray:
    return values.to_numpy(dtype="datetime64[D]").astype("int32")


def keys(firms: np.ndarray, days: np.ndarray) -> np.ndarray:
    return days.astype("int64") * 1_000_000 + firms.astype("int64")


def matching(sorted_keys: np.ndarray, candidates: np.ndarray):
    positions = np.searchsorted(sorted_keys, candidates)
    matched = positions < len(sorted_keys)
    safe = np.minimum(positions, len(sorted_keys) - 1)
    matched &= sorted_keys[safe] == candidates
    return positions, matched


def event_keys(events: pd.DataFrame) -> np.ndarray:
    return np.sort(keys(events.StkcdRecall.astype("int32").to_numpy(), day_codes(events.event_Trddt)))


def build_events(base, audit: dict) -> dict[str, pd.DataFrame]:
    paths, merge = base.PATHS, base.merge_tools
    log("读取正式规律公司矩阵的一期滞后公司对")
    pieces = []
    matrix_rows = 0
    for chunk in pd.read_csv(
        paths["MemoryMatrixPattern_path"],
        usecols=[*PAIR, "quarterLag1Alt", "event_Trddt", "event_td"],
        dtype={"Stkcd": "string", "StkcdRecall": "string", "quarterLag1Alt": "float32", "event_td": "int32"},
        parse_dates=["Annodt", "event_Trddt"], chunksize=400_000,
    ):
        matrix_rows += len(chunk)
        mask = chunk.quarterLag1Alt.eq(1) & chunk.Annodt.dt.year.between(2005, 2025)
        pieces.append(chunk.loc[mask].drop(columns="quarterLag1Alt"))
    pairs = pd.concat(pieces, ignore_index=True)
    del pieces
    pairs.Stkcd = pairs.Stkcd.str.zfill(6)
    pairs.StkcdRecall = pairs.StkcdRecall.str.zfill(6)
    duplicate_pairs = int(pairs.duplicated(PAIR).sum())
    if duplicate_pairs:
        raise AssertionError(f"原矩阵公司对键存在重复：{duplicate_pairs}")
    audit["events"] = {"matrix_total_rows": matrix_rows, "lag1_pairs": len(pairs), "duplicate_pair_keys": duplicate_pairs,
                       "unchanged_current_day_rule": "正式规律公司矩阵原有当天至少两家规律公司公告条件"}
    assert len(pairs) == 180_226, len(pairs)

    needed_keys = np.unique(keys(pairs.Stkcd.astype("int32").to_numpy(), pairs.event_td.to_numpy() - 3))
    caps, cap_rows = [], 0
    log("扫描事件前三个交易日总市值")
    for number, chunk in enumerate(pd.read_csv(
        paths["TRD_Dalyr_DGTW_path"], usecols=["Stkcd", "td", "size"],
        dtype={"Stkcd": "int32", "td": "int32", "size": "float64"}, chunksize=400_000,
    ), 1):
        cap_rows += len(chunk)
        _, matched = matching(needed_keys, keys(chunk.Stkcd.to_numpy(), chunk.td.to_numpy()))
        if matched.any():
            caps.append(chunk.loc[matched].copy())
        if number % 12 == 0:
            log(f"日市值已扫描{cap_rows:,}行")
    cap = pd.concat(caps, ignore_index=True)
    del caps
    if cap.duplicated(["Stkcd", "td"]).any():
        raise AssertionError("日市值的待匹配公司交易日键重复")
    cap["Stkcd"] = cap.Stkcd.astype(str).str.zfill(6)
    cap["event_td"] = cap.td + 3
    pairs = pairs.merge(cap.rename(columns={"size": "mktcap_tm3"})[["Stkcd", "event_td", "mktcap_tm3"]],
                        on=["Stkcd", "event_td"], how="left", validate="m:1")
    del cap, needed_keys
    missing_cap = int(pairs.mktcap_tm3.isna().sum())
    audit["events"].update({"cap_source_rows": cap_rows, "cap_missing_pairs": missing_cap})
    assert missing_cap == 18_973, missing_cap
    pairs = merge.merge_monthly_marketcap_breakpoints(pairs, paths["marketcap_group_path"], "prev_month")
    audit["events"]["pairs_with_required_breakpoints"] = len(pairs)
    if "marketCap_95th_percentile" not in pairs:
        raise AssertionError("月度市值断点缺少95分位字段")
    pairs = pairs.loc[pairs.mktcap_tm3 > pairs.marketCap_90th_percentile].copy()
    audit["events"]["pairs_above90_before_relationship_filters"] = len(pairs)
    log(f"90分位以上候选{len(pairs):,}对，依次执行行业、供应链、个人分析师共同覆盖过滤")
    eligible = merge.filter_same_industry_pairs(pairs, paths["industry_path"])
    audit["events"]["pairs_after_industry_filter"] = len(eligible)
    eligible = merge.filter_supply_chain_pairs(eligible, paths["SC_NetWorkRelationsIndex_path"])
    audit["events"]["pairs_after_supply_filter"] = len(eligible)
    eligible = merge.filter_analyst_overlap_pairs(
        eligible, ROOT / "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后", "split", 365, True,
    )
    pairs = merge.apply_pair_filters(pairs, eligible)
    del eligible
    pairs95 = pairs.loc[pairs.mktcap_tm3 > pairs.marketCap_95th_percentile]
    audit["events"].update({"pairs90": len(pairs), "pairs95": len(pairs95),
                            "threshold_rule": "事件前三交易日总市值严格大于事件上月90或95分位断点",
                            "analyst_rule": "原矩阵同源已有个人分析师365日正共同覆盖表；未命中视为无共同覆盖"})
    assert len(pairs) == 22_088, len(pairs)
    assert len(pairs95) == 13_445, len(pairs95)
    events = {"cue90": pairs[["StkcdRecall", "event_Trddt"]].drop_duplicates(),
              "cue95": pairs95[["StkcdRecall", "event_Trddt"]].drop_duplicates()}
    audit["events"].update({"collapsed90": len(events["cue90"]), "collapsed95": len(events["cue95"])})
    assert len(events["cue90"]) == 14_052
    assert len(events["cue95"]) == 8_984
    official = pd.read_csv(paths["cue_events_out_path"], usecols=["StkcdRecall", "date", base.helper.CUE],
                           dtype={"StkcdRecall": "string", base.helper.CUE: "float32"}, parse_dates=["date"])
    official = official.loc[official.date.dt.year.between(2005, 2025) & official[base.helper.CUE].eq(1)].rename(columns={"date": "event_Trddt"})
    official_key, rebuilt_key = event_keys(official), event_keys(events["cue90"])
    missing = np.setdiff1d(official_key, rebuilt_key)
    extra = np.setdiff1d(rebuilt_key, official_key)
    audit["events"].update({"official90_missing_keys": len(missing), "official90_extra_keys": len(extra),
                            "official90_exact_match": bool(np.array_equal(official_key, rebuilt_key))})
    assert np.array_equal(official_key, rebuilt_key), "重建90分位事件与正式事件不一致"
    assert not np.setdiff1d(event_keys(events["cue95"]), rebuilt_key).size
    del pairs, pairs95, official
    gc.collect()
    log("事件核对通过：90分位14,052个股票日，95分位8,984个股票日")
    return events


def read_panel(base, events: dict, audit: dict) -> pd.DataFrame:
    controls = list(base.helper.CONTROLS)
    returns = list(base.helper.RETURNS.values())
    assert len(controls) == 27
    numeric = [base.helper.CUE, *controls, *returns]
    selected = ["StkcdRecall", "Trddt", "year", "yearQuarter", *numeric]
    cue_keys = {name: event_keys(frame) for name, frame in events.items()}
    pieces, source_rows, mismatch, invalid_quarter = [], 0, 0, 0
    inf_counts = {column: 0 for column in numeric}
    finite_counts = {column: 0 for column in numeric}
    log("扫描正式主面板，保留2005—2025年全部股票日及原收益、27项控制")
    for number, chunk in enumerate(pd.read_csv(
        base.PATHS["output_path"], usecols=selected,
        dtype={"StkcdRecall": "int32", "year": "float32", "yearQuarter": "string", **{c: "float32" for c in numeric}},
        parse_dates=["Trddt"], chunksize=250_000,
    ), 1):
        source_rows += len(chunk)
        chunk = chunk.loc[chunk.year.between(2005, 2025)].copy()
        if chunk.empty:
            continue
        if not chunk.year.eq(chunk.Trddt.dt.year).all():
            raise AssertionError("主面板year与交易日期不一致")
        quarters = pd.to_datetime(chunk.yearQuarter, errors="raise")
        q = (chunk.Trddt.dt.year.to_numpy() * 4 + (chunk.Trddt.dt.month.to_numpy() - 1) // 3).astype("int32")
        q_from_field = (quarters.dt.year.to_numpy() * 4 + (quarters.dt.month.to_numpy() - 1) // 3).astype("int32")
        invalid_quarter += int((~quarters.dt.is_quarter_end).sum() + (q != q_from_field).sum())
        if invalid_quarter:
            raise AssertionError("主面板yearQuarter不是与交易日同季度的季度末日期")
        firms, days = chunk.StkcdRecall.to_numpy(), day_codes(chunk.Trddt)
        panel_keys = keys(firms, days)
        for name, sorted_key in cue_keys.items():
            _, mask = matching(sorted_key, panel_keys)
            chunk[name] = mask.astype("float32")
        mismatch += int((chunk.cue90.to_numpy() != chunk[base.helper.CUE].to_numpy()).sum())
        if mismatch:
            raise AssertionError(f"主面板90分位逐行核对失败：{mismatch}")
        for column in numeric:
            values = chunk[column].to_numpy()
            inf = np.isinf(values)
            inf_counts[column] += int(inf.sum())
            if inf.any():
                chunk.loc[inf, column] = np.nan
            finite_counts[column] += int(np.isfinite(chunk[column].to_numpy()).sum())
        chunk["day"], chunk["q"] = days, q
        chunk = chunk.rename(columns={"StkcdRecall": "firm"}).drop(columns=["Trddt", "year", "yearQuarter", base.helper.CUE])
        pieces.append(chunk[["firm", "day", "q", "cue90", "cue95", *controls, *returns]])
        if number % 8 == 0:
            log(f"正式主面板已扫描{source_rows:,}行")
    panel = pd.concat(pieces, ignore_index=True)
    del pieces
    gc.collect()
    panel_keys = keys(panel.firm.to_numpy(), panel.day.to_numpy())
    sorted_keys = np.sort(panel_keys)
    duplicates = int(np.count_nonzero(sorted_keys[1:] == sorted_keys[:-1]))
    audit["panel"] = {"source_rows_scanned": source_rows, "rows": len(panel), "firms": int(panel.firm.nunique()),
                       "dates": int(panel.day.nunique()), "quarters": int(panel.q.nunique()), "duplicate_firm_day": duplicates,
                       "cue90": int(panel.cue90.sum()), "cue95": int(panel.cue95.sum()), "cue90_rowwise_mismatch": mismatch,
                       "quarter_field_mismatch": invalid_quarter, "infinite_to_missing": inf_counts, "finite_by_column": finite_counts,
                       "rows_removed_for_returns_controls_or_oib": 0}
    assert len(panel) == 11_110_094, len(panel)
    assert audit["panel"]["firms"] == 4_989
    assert audit["panel"]["dates"] == 5_101
    assert not duplicates
    assert audit["panel"]["cue90"] == 11_890
    assert audit["panel"]["cue95"] == 7_512
    if not (panel.cue95 <= panel.cue90).all():
        raise AssertionError("95分位提示并非90分位提示的子集")
    del panel_keys, sorted_keys
    log(f"主面板核对通过：{len(panel):,}行，90分位11,890次提示，95分位7,512次提示")
    return panel


def add_oib(panel: pd.DataFrame, base, audit: dict) -> None:
    path = ROOT / "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv"
    key = keys(panel.firm.to_numpy(), panel.day.to_numpy())
    order = np.argsort(key)
    sorted_keys = key[order]
    del key
    matched_before = np.zeros(len(panel), dtype=bool)
    values_by_window = {h: np.full(len(panel), np.nan, dtype="float32") for h in (1, 10)}
    source_rows, matched_rows = 0, 0
    log("扫描订单失衡表并按唯一公司日期匹配全部主面板")
    for number, chunk in enumerate(pd.read_csv(
        path, usecols=["Stkcd", "Trddt", *base.helper.OIB.values()],
        dtype={"Stkcd": "int32", **{c: "float32" for c in base.helper.OIB.values()}},
        parse_dates=["Trddt"], chunksize=400_000,
    ), 1):
        source_rows += len(chunk)
        candidate = keys(chunk.Stkcd.to_numpy(), day_codes(chunk.Trddt))
        positions, matched = matching(sorted_keys, candidate)
        target = order[positions[matched]]
        if len(np.unique(target)) != len(target) or matched_before[target].any():
            raise AssertionError("订单失衡表存在重复的研究期主面板公司日期键")
        matched_before[target] = True
        matched_rows += len(target)
        for h, column in base.helper.OIB.items():
            values_by_window[h][target] = chunk[column].to_numpy()[matched]
        if number % 8 == 0:
            log(f"订单失衡已扫描{source_rows:,}行")
    audit["oib"] = {"source_rows": source_rows, "matched_rows": matched_rows, "unmatched_panel_rows": int((~matched_before).sum()),
                     "winsor_sample": "全研究期主面板匹配后的各窗口非缺失分布；未先删除控制缺失或固定效应单例",
                     "units": "百分点；未额外乘100", "windows": {}}
    assert source_rows == 14_126_486, source_rows
    assert matched_rows == len(panel) and matched_before.all()
    for h, values in values_by_window.items():
        infinite = int(np.isinf(values).sum())
        values[~np.isfinite(values)] = np.nan
        lower, upper = np.nanquantile(values, [0.01, 0.99])
        count = int(np.isfinite(values).sum())
        audit["oib"]["windows"][str(h)] = {"finite": count, "missing": int(np.isnan(values).sum()),
                                                    "infinite_to_missing": infinite, "p01": float(lower), "p99": float(upper),
                                                    "below_p01": int((values < lower).sum()), "above_p99": int((values > upper).sum())}
        assert count == {1: 10_994_150, 10: 10_440_385}[h], (h, count)
        expected = {1: (-29.3166892242, 26.8421113014), 10: (-17.294304657, 14.51862381)}[h]
        assert np.allclose([lower, upper], expected, rtol=0, atol=1e-5), (h, lower, upper)
        panel[f"oib_{h}"] = np.clip(values, lower, upper).astype("float32")
        log(f"订单失衡{h}窗口：有效{count:,}行，1%/99%缩尾界为{lower:.8f}/{upper:.8f}")
    del values_by_window, order, sorted_keys, matched_before
    gc.collect()


def prepare(work: Path) -> dict:
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    audit = {"started": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "sample": "仅规律公司正式主样本；2005—2025年",
             "root": str(ROOT), "prepared_file": str(work / "panel.parquet")}
    base = load_base()
    events = build_events(base, audit)
    panel = read_panel(base, events, audit)
    add_oib(panel, base, audit)
    audit["columns"] = list(panel.columns)
    audit["dtypes"] = {c: str(t) for c, t in panel.dtypes.items()}
    log("保存独立试验面板")
    panel.to_parquet(work / "panel.parquet", index=False, engine="pyarrow", compression="zstd", row_group_size=250_000)
    audit["finished"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    audit["parquet_bytes"] = (work / "panel.parquet").stat().st_size
    (work / "preparation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"面板就绪：{len(panel):,}行、{len(panel.columns)}列")
    return audit


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
