# 本文件由旧零散回归目录迁移；仅调整项目定位、公共工具引用和任务输出位置。
"""独立比较同日年报与一季报的报告取舍，不改动原脚本和正式数据。

只在年报和随后一季报同公司同日并存时改取年报；其它重复日仍取最新期。
两组先选报告再筛模式季度，历史共同公告严格早于当前公告，报告期差
为1至100天、公告日期差不超过364天。保持原矩阵的当前日须至少两家
模式公司公告条件。以当前选定财报期为滞后锚，防止当日被误归历史。
先核对最新报告组和正式记忆矩阵的一阶滞后公司对；再重查行业、供应链、
个人分析师365日共同覆盖以及事件前三日总市值/上月断点，最终只替换
现有回归面板的提示变量。2005—2025年、原收益、原控制与固定效应不变。
使用 -B 运行以避免生成或更新既有缓存。保存时仅新建时间戳目录。
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
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

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
TASK_OUTPUT_DIR = ROOT / "图表结果/02_扩展分析/其他扩展检验/年报优先提示关系对照"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_module("annual_trial_base", (ROOT / "code/07_实证分析/02_扩展分析/其他扩展检验/散户交易占比异质性/扩展_散户交易占比异质性.py"))
merge_tools = load_module("annual_trial_merge", ROOT / "code/04_合并数据/合并pattern_firms的数据.py")
analyst_tools = load_module("annual_trial_analyst", ROOT / "code/03_记忆矩阵&心理机制变量分组/识别分析师共同覆盖/识别分析师共同覆盖（日度Annodt合并检验）.py")
PATHS = merge_tools.build_pattern_paths(ROOT)
PATHS.update({
    "pattern": ROOT / "中间结果/02_变量构造结果/6PatternFirmQtr/PatternFirmQtr.csv",
    "forecast": ROOT / "中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv",
})
PAIR = ["Annodt", "Stkcd", "StkcdRecall"]
MODES = {"cue_latest_safe": "一季报优先重建", "cue_annual_safe": "年报优先"}
log = helper.log


def add_audit(audit, stage, item, value):
    audit.append({"阶段": stage, "项目": item, "数值": value})


def pair_index(frame):
    return pd.MultiIndex.from_frame(frame[PAIR])


def select_reports(audit):
    raw = pd.read_csv(PATHS["IAR_Rept_path"], usecols=["Stkcd", "Reptyp", "Accper", "Annodt"],
                      dtype={"Stkcd": "string"}, parse_dates=["Accper", "Annodt"])
    raw["Stkcd"] = raw.Stkcd.str.zfill(6)
    raw = raw.loc[raw.Annodt.ne(raw.Accper)].dropna(subset=["Stkcd", "Accper", "Annodt"]).copy()
    raw = raw.sort_values(["Stkcd", "Annodt", "Accper"], kind="stable")
    latest = raw.drop_duplicates(["Stkcd", "Annodt"], keep="last").copy()
    annual = raw.loc[raw.Reptyp.eq(4) & raw.Accper.dt.month.eq(12) & raw.Accper.dt.day.eq(31)]
    q1 = raw.loc[raw.Reptyp.eq(1) & raw.Accper.dt.month.eq(3) & raw.Accper.dt.day.eq(31)]
    # 只切换相邻年报与一季报，排除补发跨年度报告的其它情形。
    both = annual.merge(q1, on=["Stkcd", "Annodt"], suffixes=("_annual", "_q1"))
    both = both.loc[(both.Accper_q1 - both.Accper_annual).dt.days.between(89, 91)]
    replacements = both[["Stkcd", "Annodt", "Accper_annual"]].drop_duplicates(["Stkcd", "Annodt"])
    alternative = latest.merge(replacements, on=["Stkcd", "Annodt"], how="left", validate="1:1")
    changed = alternative.Accper_annual.notna() & alternative.Accper.ne(alternative.Accper_annual)
    alternative.loc[changed, "Accper"] = alternative.loc[changed, "Accper_annual"]
    alternative.loc[changed, "Reptyp"] = 4
    alternative = alternative.drop(columns="Accper_annual")
    pattern = pd.read_csv(PATHS["pattern"], dtype={"Stkcd": "string"}, parse_dates=["Accper"])
    pattern["Stkcd"] = pattern.Stkcd.str.zfill(6)
    outputs = {}
    for name, selected in zip(MODES, (latest, alternative)):
        outputs[name] = selected.merge(pattern, on=["Stkcd", "Accper"], how="left", validate="1:1")
    changes = outputs["cue_latest_safe"].merge(outputs["cue_annual_safe"], on=["Stkcd", "Annodt"], suffixes=("_latest", "_annual"), validate="1:1")
    changes = changes.loc[changes.Accper_latest.ne(changes.Accper_annual)].copy()
    period = changes.Annodt.dt.year.between(2005, 2025)
    for label, part in (("全期", changes), ("研究期", changes.loc[period])):
        add_audit(audit, "报告取舍", label + "切换公司公告日", len(part))
        for old, new, item in ((True, False, "模式资格由有变无"), (False, True, "模式资格由无变有"), (True, True, "两者均为模式季度")):
            count = (part.PatternFirmQtr_latest.eq(1).eq(old) & part.PatternFirmQtr_annual.eq(1).eq(new)).sum()
            add_audit(audit, "报告取舍", label + item, int(count))
    for name in outputs:
        outputs[name] = outputs[name].loc[outputs[name].PatternFirmQtr.eq(1), ["Stkcd", "Annodt", "Accper"]].copy()
        count = outputs[name].groupby("Annodt").Stkcd.transform("size")
        outputs[name]["current_eligible"] = count.ge(2)
        add_audit(audit, MODES[name], "模式公司公告日", len(outputs[name]))
    return outputs, changes


def build_lag1(events, name, audit):
    current = events.loc[events.current_eligible & events.Annodt.dt.year.between(2005, 2025), ["Stkcd", "Annodt", "Accper"]]
    histories = events[["Stkcd", "Annodt", "Accper"]].rename(columns={"Annodt": "history_date", "Accper": "history_accper"})
    links = current.merge(histories, on="Stkcd", how="inner")
    lag = (links.Accper - links.history_accper).dt.days
    distance = (links.Annodt - links.history_date).dt.days
    links = links.loc[lag.between(1, 100) & distance.between(1, 364)].copy()
    partners = events[["Stkcd", "Annodt"]].rename(columns={"Stkcd": "StkcdRecall", "Annodt": "history_date"})
    joined = links.merge(partners, on="history_date", how="inner")
    joined = joined.loc[joined.Stkcd.ne(joined.StkcdRecall)]
    if not ((joined.history_date < joined.Annodt) & (joined.history_accper < joined.Accper)).all():
        raise AssertionError("历史公告或财报期发生同日误归。")
    pairs = joined[PAIR].drop_duplicates().reset_index(drop=True)
    add_audit(audit, MODES[name], "严格历史的一阶滞后公司对", len(pairs))
    log(f"{MODES[name]}：重建{len(pairs):,}条一阶滞后公司对")
    return pairs


def compare_original_matrix(rebuilt, audit):
    original = []
    for chunk in pd.read_csv(PATHS["MemoryMatrixPattern_path"], usecols=[*PAIR, "quarterLag1Alt"], dtype={"Stkcd": "string", "StkcdRecall": "string"}, parse_dates=["Annodt"], chunksize=300000):
        original.append(chunk.loc[chunk.Annodt.dt.year.between(2005, 2025) & chunk.quarterLag1Alt.eq(1), PAIR])
    original = pd.concat(original, ignore_index=True).drop_duplicates()
    old, new = pair_index(original), pair_index(rebuilt)
    missing, extra = old.difference(new), new.difference(old)
    add_audit(audit, "基线核对", "重建较正式矩阵缺少公司对", len(missing))
    add_audit(audit, "基线核对", "重建较正式矩阵增加公司对", len(extra))
    log(f"正式矩阵核对：缺少{len(missing):,}，增加{len(extra):,}")
    return pd.concat([missing.to_frame(index=False).assign(类型="重建缺少"), extra.to_frame(index=False).assign(类型="重建增加")], ignore_index=True)


def map_dates(pairs):
    calendar = pd.read_csv(PATHS["TRD_Cale_td_path"], usecols=["Trddt", "td"], parse_dates=["Trddt"]).drop_duplicates()
    if calendar.Trddt.duplicated().any() or calendar.td.duplicated().any():
        raise ValueError("交易日历存在冲突。")
    calendar = calendar.sort_values("Trddt")
    date_values = calendar.Trddt.to_numpy()
    positions = np.searchsorted(date_values, pairs.Annodt.to_numpy())
    if (positions >= len(calendar)).any():
        raise ValueError("事件超出交易日历。")
    pairs = pairs.copy()
    pairs["event_Trddt"] = date_values[positions]
    pairs["event_td"] = calendar.td.to_numpy()[positions]
    return pairs


def filter_pairs(candidates, args, audit):
    candidates = map_dates(candidates)
    needed = candidates[["Stkcd", "event_td"]].drop_duplicates().rename(columns={"event_td": "td"})
    needed["td"] -= 3
    needed["firm"] = needed.Stkcd.astype("int32")
    lookup = pd.MultiIndex.from_frame(needed[["firm", "td"]])
    caps = []
    log("读取事件前三个交易日总市值，保持上月市值断点")
    for i, chunk in enumerate(pd.read_csv(PATHS["TRD_Dalyr_DGTW_path"], usecols=["Stkcd", "td", "size"], dtype={"Stkcd": "int32", "td": "int32", "size": "float64"}, chunksize=args.chunk_size), 1):
        idx = pd.MultiIndex.from_arrays([chunk.Stkcd, chunk.td], names=["firm", "td"])
        mask = idx.isin(lookup)
        if mask.any():
            caps.append(chunk.loc[mask].copy())
        if i % 12 == 0:
            log(f"日市值已读取第{i}块")
    caps = pd.concat(caps, ignore_index=True)
    caps.Stkcd = caps.Stkcd.astype(str).str.zfill(6)
    caps["event_td"] = caps.td + 3
    candidates = candidates.merge(caps.rename(columns={"size": "mktcap_tm3"})[["Stkcd", "event_td", "mktcap_tm3"]], on=["Stkcd", "event_td"], how="left", validate="m:1")
    candidates = merge_tools.merge_monthly_marketcap_breakpoints(candidates, PATHS["marketcap_group_path"], "prev_month")
    candidates = candidates.loc[candidates.mktcap_tm3 > candidates.marketCap_90th_percentile].copy()
    add_audit(audit, "两组公司对并集", "市值前十分位且市值断点有效", len(candidates))
    filtered = merge_tools.filter_same_industry_pairs(candidates, PATHS["industry_path"])
    filtered = merge_tools.filter_supply_chain_pairs(filtered, PATHS["SC_NetWorkRelationsIndex_path"])
    log("为两套口径候选公司对重查个人分析师共同覆盖")
    forecast = analyst_tools.read_analyst_forecast(PATHS["forecast"])
    coverage = analyst_tools.build_split_coverage(forecast)
    del forecast
    coverage_index = analyst_tools.build_fast_coverage_index(coverage)
    del coverage
    overlap = analyst_tools.identify_overlap_by_annodt_daily_sets(filtered[PAIR], coverage_index, 365, "split", show_progress=False)
    overlap_keys = pair_index(overlap) if len(overlap) else pd.MultiIndex.from_frame(filtered[PAIR].iloc[:0])
    filtered = filtered.loc[~pair_index(filtered).isin(overlap_keys), PAIR].copy()
    add_audit(audit, "两组公司对并集", "共同分析师排除公司对", len(overlap))
    add_audit(audit, "两组公司对并集", "三轮关联过滤后", len(filtered))
    candidates = candidates.merge(filtered, on=PAIR, how="inner", validate="1:1")
    del coverage_index
    gc.collect()
    return candidates


def collapse_and_compare(pairs, filtered, audit):
    events, surviving = {}, {}
    for mode, table in pairs.items():
        surviving[mode] = table.merge(filtered[PAIR + ["event_Trddt", "event_td"]], on=PAIR, how="inner", validate="1:1")
        event = surviving[mode][["StkcdRecall", "event_Trddt"]].drop_duplicates()
        events[mode] = event.assign(**{mode: 1})
        add_audit(audit, MODES[mode], "过滤后大市值提示公司对", len(surviving[mode]))
        add_audit(audit, MODES[mode], "过滤后被提示公司交易日", len(event))
    for stage, tables, cols in (("过滤前公司对差异", pairs, PAIR), ("过滤后公司对差异", surviving, PAIR), ("最终提示股票日差异", events, ["StkcdRecall", "event_Trddt"])):
        a = pd.MultiIndex.from_frame(tables["cue_latest_safe"][cols])
        b = pd.MultiIndex.from_frame(tables["cue_annual_safe"][cols])
        for item, value in (("共有", len(a.intersection(b))), ("仅一季报优先", len(a.difference(b))), ("仅年报优先", len(b.difference(a)))):
            add_audit(audit, stage, item, value)
    official = pd.read_csv(PATHS["cue_events_out_path"], usecols=["StkcdRecall", "date", helper.CUE], dtype={"StkcdRecall": "string"}, parse_dates=["date"])
    official = official.loc[official.date.dt.year.between(2005, 2025) & official[helper.CUE].eq(1)].rename(columns={"date": "event_Trddt"})
    a, b = pd.MultiIndex.from_frame(official[["StkcdRecall", "event_Trddt"]]), pd.MultiIndex.from_frame(events["cue_latest_safe"][["StkcdRecall", "event_Trddt"]])
    add_audit(audit, "基线核对", "重建较正式事件表缺少股票日", len(a.difference(b)))
    add_audit(audit, "基线核对", "重建较正式事件表增加股票日", len(b.difference(a)))
    return events


def read_panel(events, args, audit):
    columns = ["StkcdRecall", "Trddt", helper.CUE, *helper.CONTROLS, *helper.RETURNS.values()]
    pieces = []
    log("读取原回归面板，只替换两套提示变量")
    for number, chunk in enumerate(pd.read_csv(PATHS["output_path"], usecols=columns, parse_dates=["Trddt"], dtype={"StkcdRecall": "int32", **{c: "float32" for c in columns if c not in {"StkcdRecall", "Trddt"}}}, chunksize=args.chunk_size), 1):
        chunk = chunk.loc[chunk.Trddt.dt.year.between(2005, 2025)].copy()
        if chunk.empty:
            continue
        chunk["q"] = helper.quarter_codes(chunk.Trddt)
        chunk["day"] = helper.day_codes(chunk.Trddt)
        pieces.append(chunk.rename(columns={"StkcdRecall": "firm"}).drop(columns="Trddt"))
        if number % 10 == 0:
            log(f"回归面板已读取第{number}块")
    panel = pd.concat(pieces, ignore_index=True)
    for mode, event in events.items():
        event_keys = np.sort(helper.keys(event.StkcdRecall.astype("int32").to_numpy(), helper.day_codes(event.event_Trddt)))
        _, mask = helper.matched_positions(event_keys, helper.keys(panel.firm.to_numpy(), panel.day.to_numpy()))
        panel[mode] = mask.astype("int8")
        add_audit(audit, "回归面板", MODES[mode] + "提示股票日", int(mask.sum()))
    latest, annual, official = panel.cue_latest_safe.to_numpy().astype(bool), panel.cue_annual_safe.to_numpy().astype(bool), panel[helper.CUE].to_numpy().astype(bool)
    for item, mask in (("重建与现有不同股票日", latest != official), ("年报优先新增提示股票日", annual & ~latest), ("年报优先删除提示股票日", latest & ~annual), ("两口径共有提示股票日", latest & annual)):
        add_audit(audit, "回归面板", item, int(mask.sum()))
    add_audit(audit, "回归面板", "公司日期重复", int(panel.duplicated(["firm", "day"]).sum()))
    if panel.duplicated(["firm", "day"]).any():
        raise ValueError("原主面板公司日期重复。")
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, nargs="?", const=TASK_OUTPUT_DIR, default=None, help="显式保存开关；不带目录时保存到本任务目录/时间戳，也可指定保存目录")
    parser.add_argument("--chunk-size", type=int, default=300000)
    args = parser.parse_args()
    inputs = {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in PATHS.values() if p.exists() and p.is_file()}
    old_scripts = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "code").rglob("*.py")}
    audit = []
    selections, changes = select_reports(audit)
    pairs = {mode: build_lag1(events, mode, audit) for mode, events in selections.items()}
    mismatch = compare_original_matrix(pairs["cue_latest_safe"], audit)
    candidates = pd.concat(list(pairs.values()), ignore_index=True).drop_duplicates(PAIR)
    filtered = filter_pairs(candidates, args, audit)
    events = collapse_and_compare(pairs, filtered, audit)
    panel = read_panel(events, args, audit)
    regression = load_module("annual_trial_regression", (ROOT / "code/07_实证分析/90_公共模块/扩展_年报优先回归工具.py"))
    with tempfile.TemporaryDirectory(prefix="annual_priority_", dir=ROOT / "work") as tmp:
        result, checks = regression.estimate_comparison(panel, Path(tmp))
        validation = regression.validate_estimator(panel, Path(tmp))
    for path, meta in inputs.items():
        p = Path(path)
        if p.stat().st_size != meta["bytes"] or p.stat().st_mtime_ns != meta["mtime_ns"]:
            raise AssertionError(f"输入文件发生变化：{p}")
    for path, sha in old_scripts.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise AssertionError(f"已有脚本发生变化：{path}")
    add_audit(audit, "文件保护", "核对未改动既有脚本数量", len(old_scripts))
    result = pd.DataFrame(result)
    audit_df = pd.DataFrame(audit)
    print(result.to_string(index=False))
    print(audit_df.to_string(index=False))
    if args.save_dir:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        output = args.save_dir / stamp
        output.mkdir(parents=True, exist_ok=False)
        result.to_csv(output / "主回归口径比较.csv", index=False, encoding="utf-8-sig")
        audit_df.to_csv(output / "报告取舍与提示关系审计.csv", index=False, encoding="utf-8-sig")
        changes.to_csv(output / "同日年报一季报切换明细.csv", index=False, encoding="utf-8-sig")
        mismatch.to_csv(output / "最新期重建与正式矩阵差异.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(checks + validation).to_csv(output / "回归计算核验.csv", index=False, encoding="utf-8-sig")
        (output / "输入与脚本保护.json").write_text(json.dumps({"inputs": inputs, "script_sha256": old_scripts}, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"独立结果目录：{output}")


if __name__ == "__main__":
    main()
