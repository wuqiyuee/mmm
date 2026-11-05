"""用国泰安官方年度标签生成图2所需的组合基准和目标日原始收益缓存。

本脚本恢复自历史计算的准备阶段。输入数据后来曾更新，本次未重新运行。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/苹果资料/mmm复现数据包")
GRID = ROOT / "work/主回归口径组合试算_20261026"
ENRICHED = GRID / "enriched"
OUT = ROOT / "work/国泰安标签收益检验/图2"
OUT.mkdir(parents=True, exist_ok=True)
RAW_DIR = OUT / "raw_target"
RAW_DIR.mkdir(exist_ok=True)

START = np.datetime64("2005-04-08").astype("int32")
END = np.datetime64("2025-12-31").astype("int32")
H = 50
PORTS = 126
PANEL = ROOT / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv"
OFFICIAL = ROOT / "input/CSMAR/DGTW（2000至2025）/DGTW股票分配表/STK_MKT_DGTWASSINGMENTS.csv"
FILL = ROOT / "work/图2_允许填充起始日_临时/新增填充日_收益与市值.npz"

RAW_COLS = ["raw_ret_t_t" + str(i) for i in range(1, 51)]
PANEL_COLS = ["Stkcd", "Trddt", *RAW_COLS, "float_mktcap_tm3", "float_mktcap_tm3_valid"]
RAW_DTYPE = np.dtype([("firm", "i4"), ("date", "i4"), ("port", "i2"), ("raw", "f4", (H,))])


def official_labels():
    a = pd.read_csv(OFFICIAL, usecols=["Symbol", "TradingYear", "IsNotBSE", "MarketValue", "BooktoMarket", "Momentum"],
                    dtype={"Symbol": "int64", "TradingYear": "int16", "MarketValue": "int8",
                           "BooktoMarket": "int8", "Momentum": "int8"})
    a = a.loc[a["IsNotBSE"].eq(1)].copy()
    a["port"] = (a["MarketValue"].astype("int16") - 1) * 25 + (a["BooktoMarket"].astype("int16") - 1) * 5 + a["Momentum"].astype("int16")
    keys = a["Symbol"].to_numpy(dtype="int64") * 1000 + a["TradingYear"].to_numpy(dtype="int64")
    order = np.argsort(keys)
    keys = keys[order]
    ports = a["port"].to_numpy(dtype="int16")[order]
    if np.any(np.diff(keys) == 0):
        raise ValueError("官方标签的公司—形成年度键重复。")
    return keys, ports


def target_keys_and_quarters():
    keys = []
    for path in sorted(ENRICHED.glob("q*.npy")):
        a = np.load(path, mmap_mode="r")
        m = a["filled"] == 0
        keys.append(a["firm"][m].astype("int64") * 100000 + a["date"][m].astype("int64"))
    fill = np.load(FILL, mmap_mode="r")
    keys.append(fill["firm"].astype("int64") * 100000 + (fill["date"].astype("int64") + 10957))
    out = np.unique(np.concatenate(keys))
    return out


def benchmark_keys():
    """窗口变量资格索引的公司—日期键；保留填充日起点。"""
    path = ROOT / "中间结果/01_清洗后样本/窗口变量目标日资格索引_基础清洗含停牌填充_td.csv"
    pieces = []
    for part in pd.read_csv(path, usecols=["Stkcd", "Trddt"], dtype={"Stkcd": "int64"},
                            parse_dates=["Trddt"], encoding="utf-8-sig", chunksize=1_000_000):
        d = part["Trddt"].to_numpy(dtype="datetime64[D]").astype("int32")
        m = (d >= START) & (d <= END)
        if m.any():
            pieces.append(part.loc[m, "Stkcd"].to_numpy(dtype="int64") * 100000 + d[m].astype("int64"))
    return np.unique(np.concatenate(pieces))


def stage_benchmark_and_raw():
    stamp = OUT / "benchmark_complete.json"
    if stamp.exists():
        return np.load(OUT / "official_benchmark.npz", mmap_mode="r")
    started = time.monotonic()
    label_keys, label_ports = official_labels()
    target_keys = target_keys_and_quarters()
    bench_keys = benchmark_keys()
    print(f"目标起点键 {len(target_keys):,} 条；benchmark资格键 {len(bench_keys):,} 条。", flush=True)
    date_count = int(END - START + 1)
    sumw = np.zeros((date_count * PORTS, H), dtype="float64")
    sumwr = np.zeros((date_count * PORTS, H), dtype="float64")
    handles = {}
    for path in sorted(ENRICHED.glob("q*.npy")):
        q = int(path.stem[1:])
        handles[q] = (RAW_DIR / f"q{q:02d}.bin").open("wb")
    nread = 0
    nmatched = 0
    for number, part in enumerate(pd.read_csv(PANEL, usecols=PANEL_COLS, chunksize=300_000,
                                               dtype={"Stkcd": "int64", **{c: "float32" for c in RAW_COLS},
                                                      "float_mktcap_tm3": "float64"},
                                               parse_dates=["Trddt"], encoding="utf-8"), start=1):
        dates = part["Trddt"].to_numpy(dtype="datetime64[D]").astype("int32")
        in_range = (dates >= START) & (dates <= END)
        if not in_range.any():
            continue
        part = part.loc[in_range].copy()
        dates = dates[in_range]
        firms = part["Stkcd"].to_numpy(dtype="int64")
        panel_key = firms * 100000 + dates.astype("int64")
        bpos = np.searchsorted(bench_keys, panel_key)
        bkeep = (bpos < len(bench_keys))
        bkeep[bkeep] &= bench_keys[bpos[bkeep]] == panel_key[bkeep]
        if not bkeep.any():
            continue
        # benchmark与目标原始收益均只保留窗口资格池中的公司—日期观测。
        part = part.loc[bkeep].copy()
        dates = dates[bkeep]
        firms = firms[bkeep]
        years = pd.DatetimeIndex(part["Trddt"]).year.to_numpy(dtype="int64")
        months = pd.DatetimeIndex(part["Trddt"]).month.to_numpy(dtype="int64")
        form = np.where(months <= 6, years - 1, years)
        lkey = firms * 1000 + form
        pos = np.searchsorted(label_keys, lkey)
        has = (pos < len(label_keys))
        has[has] &= label_keys[pos[has]] == lkey[has]
        ports = np.zeros(len(part), dtype="int16")
        ports[has] = label_ports[pos[has]]
        weight = part["float_mktcap_tm3"].to_numpy(dtype="float64")
        valid_weight = np.isfinite(weight) & (weight > 0) & has
        raw = part[RAW_COLS].to_numpy(dtype="float64", copy=False)
        if valid_weight.any():
            row = np.flatnonzero(valid_weight)
            did = dates[row] - START
            gid = did.astype("int64") * PORTS + ports[row].astype("int64")
            order = np.argsort(gid, kind="mergesort")
            gs = gid[order]
            ws = weight[row][order]
            rs = raw[row][order]
            ug, starts = np.unique(gs, return_index=True)
            finite = np.isfinite(rs)
            sumwr_part = np.add.reduceat(np.nan_to_num(rs, nan=0.0) * ws[:, None], starts, axis=0)
            sumw_part = np.add.reduceat(finite * ws[:, None], starts, axis=0)
            flat_wr = sumwr.reshape(-1, H)
            flat_w = sumw.reshape(-1, H)
            np.add.at(flat_wr, ug, sumwr_part)
            np.add.at(flat_w, ug, sumw_part)
        # 目标日原始累计收益写入季度二进制缓存；填充日起点不在基础日度面板中，另行处理。
        key = firms * 100000 + dates.astype("int64")
        mp = np.searchsorted(target_keys, key)
        hit = (mp < len(target_keys))
        hit[hit] &= target_keys[mp[hit]] == key[hit]
        hit &= has
        if hit.any():
            hitrow = np.flatnonzero(hit)
            rec = np.zeros(len(hitrow), dtype=RAW_DTYPE)
            rec["firm"] = firms[hitrow]
            rec["date"] = dates[hitrow]
            rec["port"] = ports[hitrow]
            rec["raw"] = raw[hitrow].astype("float32")
            qnum = ((pd.DatetimeIndex(part.iloc[hitrow]["Trddt"]).year.to_numpy() - 2005) * 4
                    + pd.DatetimeIndex(part.iloc[hitrow]["Trddt"]).quarter.to_numpy() - 1)
            for q in np.unique(qnum):
                # enriched/q01 对应 2005 年第二季度，直接使用 qnum 作为文件编号。
                handles[int(q)].write(rec[qnum == q].tobytes())
            nmatched += len(rec)
        nread += len(part)
        if number % 20 == 0:
            print(f"日度面板已处理 {nread:,} 行，实际目标键 {nmatched:,} 条，用时 {time.monotonic()-started:.1f}s", flush=True)
    for f in handles.values():
        f.close()
    np.savez(OUT / "official_benchmark.npz", sumw=sumw, sumwr=sumwr, start=np.array(START), ports=np.arange(PORTS))
    stamp.write_text(json.dumps({"rows_read": nread, "target_actual": nmatched,
                                 "seconds": time.monotonic()-started}, ensure_ascii=False, indent=2))
    return np.load(OUT / "official_benchmark.npz", mmap_mode="r")

if __name__ == "__main__":
    stage_benchmark_and_raw()
