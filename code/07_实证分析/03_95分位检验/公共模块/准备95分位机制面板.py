"""为95分位表4、表5准备独立面板；导入本文件不会读取数据或写文件。

运行入口在相邻的“表4_规模异质性”和“表5_心理机制”目录。
准备时从公司对记忆矩阵重建95以上、50—95、10—50、10以下提示；
历史强关联先与提示公司规模交互，再聚合为被提示公司×事件交易日。
收益、27项控制、被提示公司大小和最终样本行均沿用正式最终面板。
干扰程度在最终收益/控制筛选之前的日度样本上按年度重新二分。
临时缓存和审计只在项目 work（临时工作目录）下生成，不改正式输入。
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _find_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "AGENTS.md").is_file() and (parent / "中间结果").is_dir():
            return parent
    raise FileNotFoundError("找不到项目根目录。")


ROOT = _find_root()
DEFAULT_CACHE = ROOT / "work/95分位表4表5/机制面板.parquet"
MERGE_SCRIPT = ROOT / "code/04_合并数据/合并pattern_firms的数据.py"
START_YEAR, END_YEAR = 2005, 2025
PAIR_KEYS = ["Annodt", "Stkcd", "StkcdRecall"]
DAILY_KEYS = ["StkcdRecall", "Trddt"]
CONTROL_COLS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
    + ["ownOtherEventER", "ownPerformanceForecastBrief"]
    + ["aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV"]
)
OUTCOME_COLS = ["chAdjreturn_t_tp1_Recall_w", "chAdjreturn_t_tp10_Recall_w"]
CUE_COLS = [
    "quarterLag1size95plus", "quarterLag1size50_95",
    "quarterLag1size10_50", "quarterLag1size10minus",
]
HISTORY_COLS = ["lag1hiAEV", "lag1hiavol", "lag1hiavolS"] + [
    f"lag1{suffix}"
    for sue_id in (1, 2, 3)
    for suffix in (
        f"sue{sue_id}pos", f"sue{sue_id}neg",
        f"hi_sue{sue_id}_pos", f"hi_sue{sue_id}_neg",
    )
] + ["lag1anyTopMover"]
MECHANISM_COLS = [
    f"{source.replace('anyTopMover', 'topMover')}size{size}"
    for size in ("95plus", "50_95") for source in HISTORY_COLS
]
INTERFERENCE_COLS = [
    f"{level}InterferenceEWLag1{suffix}"
    for suffix in ("S95p", "S5095") for level in ("low", "high")
]
BINARY_COLS = CUE_COLS + MECHANISM_COLS + INTERFERENCE_COLS
PANEL_COLS = (
    DAILY_KEYS + ["year", "yearQuarter", "smallFirm"]
    + OUTCOME_COLS + CONTROL_COLS + BINARY_COLS
)
CHECK_COLUMNS = {
    "quarterLag1size90plus": "_check90",
    "quarterLag1size50_90": "_check5090",
}


def load_module(path: Path, name: str):
    """按绝对路径加载模块，登记名称以支持数据类定义。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _log(message: str) -> None:
    print(f"[95分位面板] {message}", flush=True)


def _codes(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if result.isna().any() or not result.str.fullmatch(r"\d{6}").all():
        raise ValueError("股票代码包含缺失或非六位数字。")
    return result


def _packed(firms, days) -> np.ndarray:
    return np.asarray(days, dtype="int64") * 1_000_000 + np.asarray(firms, dtype="int64")


def _date_keys(firms, dates) -> np.ndarray:
    days = np.asarray(dates, dtype="datetime64[D]").astype("int64")
    return _packed(firms, days)


def _matches(sorted_keys: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if len(sorted_keys) == 0:
        return np.zeros(len(candidates), dtype=bool)
    positions = np.searchsorted(sorted_keys, candidates)
    return (positions < len(sorted_keys)) & (sorted_keys[np.minimum(positions, len(sorted_keys) - 1)] == candidates)


def _read_pairs(paths: dict, chunk_size: int, audit: dict) -> pd.DataFrame:
    columns = PAIR_KEYS + ["event_Trddt", "event_td", "quarterLag1Alt", "nrFirmsAnnLag1Alt",
                           "currentAccpersLag1", "lag1AnnualAnchor", "lag1Q1Anchor"] + HISTORY_COLS
    pieces, scanned = [], 0
    for chunk in pd.read_csv(
        paths["MemoryMatrixPattern_path"], usecols=columns,
        dtype={"Stkcd": "string", "StkcdRecall": "string", "currentAccpersLag1": "string",
               "lag1AnnualAnchor": "int8", "lag1Q1Anchor": "int8"},
        parse_dates=["Annodt", "event_Trddt"], chunksize=chunk_size,
    ):
        scanned += len(chunk)
        keep = chunk["quarterLag1Alt"].eq(1) & chunk["event_Trddt"].dt.year.between(START_YEAR, END_YEAR)
        pieces.append(chunk.loc[keep].copy())
    if not pieces:
        raise ValueError("记忆矩阵没有数据。")
    pairs = pd.concat(pieces, ignore_index=True)
    if pairs.empty:
        raise ValueError("研究期内没有一期历史关系。")
    for column in ("Stkcd", "StkcdRecall"):
        pairs[column] = _codes(pairs[column])
    if pairs[PAIR_KEYS + ["event_Trddt", "event_td"]].isna().any().any():
        raise ValueError("公司对或事件交易日键含缺失。")
    if pairs.duplicated(PAIR_KEYS).any():
        raise ValueError("公司对公告日键重复。")
    td = pd.to_numeric(pairs["event_td"], errors="raise")
    if not np.isfinite(td).all() or not np.equal(td, np.floor(td)).all():
        raise ValueError("事件交易日序号必须是有限整数。")
    pairs["event_td"] = td.astype("int32")
    for column in HISTORY_COLS:
        if not pairs[column].isin([0, 1]).all():
            raise ValueError(f"历史机制源字段必须完整且为零一值：{column}")
        pairs[column] = pairs[column].astype("int8")
    counts = pd.to_numeric(pairs["nrFirmsAnnLag1Alt"], errors="raise")
    if np.isinf(counts).any() or counts.dropna().lt(0).any():
        raise ValueError("历史共同公告公司数量包含无穷或负值。")
    pairs["nrFirmsAnnLag1Alt"] = counts
    audit["矩阵扫描行数"] = scanned
    audit["一期关系公司对数"] = len(pairs)
    return pairs


def _add_marketcap_and_daily_membership(pairs: pd.DataFrame, path: Path, chunk_size: int, audit: dict):
    """一次扫描同时取得提示公司前三交易日市值和筛选前事件日的日度样本资格。"""
    cap_keys = np.unique(_packed(pairs["Stkcd"].astype("int64"), pairs["event_td"] - 3))
    recall_keys = np.unique(_date_keys(pairs["StkcdRecall"].astype("int64"), pairs["event_Trddt"]))
    cap_pieces, present_pieces = [], []
    for chunk in pd.read_csv(
        path, usecols=["Stkcd", "td", "Trddt", "size"],
        dtype={"Stkcd": "int32", "td": "int32", "size": "float64"},
        parse_dates=["Trddt"], chunksize=chunk_size,
    ):
        hit = _matches(cap_keys, _packed(chunk["Stkcd"], chunk["td"]))
        if hit.any():
            cap_pieces.append(chunk.loc[hit, ["Stkcd", "td", "size"]].copy())
        daily_keys = _date_keys(chunk["Stkcd"], chunk["Trddt"])
        present = _matches(recall_keys, daily_keys)
        if present.any():
            present_pieces.append(daily_keys[present])
    if not cap_pieces or not present_pieces:
        raise ValueError("日度源表未匹配到提示市值或被提示公司事件日。")
    caps = pd.concat(cap_pieces, ignore_index=True)
    if caps.duplicated(["Stkcd", "td"]).any():
        raise ValueError("待匹配市值键重复。")
    caps["Stkcd"] = _codes(caps["Stkcd"])
    caps["event_td"] = caps["td"] + 3
    caps = caps.rename(columns={"size": "mktcap_tm3"})
    pairs = pairs.merge(caps[["Stkcd", "event_td", "mktcap_tm3"]], on=["Stkcd", "event_td"], how="left", validate="many_to_one")
    audit["提示公司前三交易日市值缺失数"] = int(pairs["mktcap_tm3"].isna().sum())
    return pairs, np.unique(np.concatenate(present_pieces))


def _industry_filter(pairs: pd.DataFrame, path: Path, chunk_size: int) -> pd.DataFrame:
    """只读取待匹配行业键；缺失不排除、重复保留首条，与正式规则一致。"""
    needed = np.unique(np.concatenate([
        _packed(pairs["Stkcd"].astype("int64"), pairs["event_td"]),
        _packed(pairs["StkcdRecall"].astype("int64"), pairs["event_td"]),
    ]))
    pieces = []
    for chunk in pd.read_csv(
        path, usecols=["Stkcd", "td", "IndustryCode"],
        dtype={"Stkcd": "string", "IndustryCode": "string"}, chunksize=chunk_size,
    ):
        chunk["td"] = pd.to_numeric(chunk["td"], errors="coerce")
        chunk = chunk.dropna(subset=["Stkcd", "td", "IndustryCode"]).copy()
        chunk["Stkcd"] = _codes(chunk["Stkcd"])
        hit = _matches(needed, _packed(chunk["Stkcd"].astype("int64"), chunk["td"]))
        if hit.any():
            pieces.append(chunk.loc[hit].copy())
    if not pieces:
        _log("行业表没有匹配记录，依正式规则保留双方行业未知的公司对。")
        return pairs
    industry = pd.concat(pieces, ignore_index=True).drop_duplicates(["Stkcd", "td"])
    industry = industry.rename(columns={"td": "event_td"})
    data = pairs.merge(industry.rename(columns={"IndustryCode": "_cue_industry"}), on=["Stkcd", "event_td"], how="left", validate="many_to_one")
    data = data.merge(industry.rename(columns={"Stkcd": "StkcdRecall", "IndustryCode": "_recall_industry"}), on=["StkcdRecall", "event_td"], how="left", validate="many_to_one")
    same = data["_cue_industry"].notna() & data["_recall_industry"].notna() & data["_cue_industry"].eq(data["_recall_industry"])
    return data.loc[~same].drop(columns=["_cue_industry", "_recall_industry"])


def _build_events(merge, paths: dict, chunk_size: int, audit: dict) -> pd.DataFrame:
    pairs = _read_pairs(paths, chunk_size, audit)
    pairs, prefilter_daily_keys = _add_marketcap_and_daily_membership(pairs, paths["TRD_Dalyr_DGTW_path"], chunk_size, audit)
    pairs = merge.merge_monthly_marketcap_breakpoints(pairs, paths["marketcap_group_path"], "prev_month")
    required = ["mktcap_tm3", *[f"marketCap_{p}th_percentile" for p in (10, 50, 90, 95)]]
    missing = set(required) - set(pairs.columns)
    if missing:
        raise ValueError(f"月度断点缺少字段：{sorted(missing)}")
    pairs = pairs.replace({column: {np.inf: np.nan, -np.inf: np.nan} for column in required})
    pairs = pairs.dropna(subset=required).copy()
    if not (
        pairs["marketCap_10th_percentile"].le(pairs["marketCap_50th_percentile"])
        & pairs["marketCap_50th_percentile"].le(pairs["marketCap_90th_percentile"])
        & pairs["marketCap_90th_percentile"].le(pairs["marketCap_95th_percentile"])
    ).all():
        raise ValueError("市值断点顺序不合法。")
    pairs = _industry_filter(pairs, paths["industry_path"], chunk_size)
    audit["行业过滤后公司对数"] = len(pairs)
    eligible = merge.filter_supply_chain_pairs(pairs[PAIR_KEYS].copy(), paths["SC_NetWorkRelationsIndex_path"])
    audit["供应链过滤后公司对数"] = len(eligible)
    eligible = merge.filter_analyst_overlap_pairs(
        eligible, ROOT / "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后", "split", 365, True,
        matrix_path=paths["MemoryMatrixPattern_path"],
        forecast_path=paths["analyst_forecast_path"],
    )
    pairs = merge.apply_pair_filters(pairs, eligible)
    audit["三轮过滤后公司对数"] = len(pairs)
    if pairs.empty:
        raise ValueError("关联排除后没有有效公司对。")
    pairs = merge.deduplicate_pattern_event_pairs(pairs)
    audit["交易日去重后公司对数"] = len(pairs)

    cap = pairs["mktcap_tm3"]
    p10, p50, p90, p95 = [pairs[f"marketCap_{p}th_percentile"] for p in (10, 50, 90, 95)]
    groups = {
        "95plus": cap.gt(p95), "50_95": cap.gt(p50) & cap.le(p95),
        "10_50": cap.ge(p10) & cap.le(p50), "10minus": cap.lt(p10),
    }
    if not np.equal(np.column_stack(list(groups.values())).sum(axis=1), 1).all():
        raise ValueError("公司对规模组未构成完整且互斥的四档。")
    for size, indicator in groups.items():
        pairs[f"quarterLag1size{size}"] = indicator.astype("int8")
    pairs["_check90"] = cap.gt(p90).astype("int8")
    pairs["_check5090"] = (cap.gt(p50) & cap.le(p90)).astype("int8")
    aggregations = {column: "max" for column in CUE_COLS + list(CHECK_COLUMNS.values())}
    for size, suffix in (("95plus", "S95p"), ("50_95", "S5095")):
        size_col = f"quarterLag1size{size}"
        for source in HISTORY_COLS:
            target = f"{source.replace('anyTopMover', 'topMover')}size{size}"
            pairs[target] = (pairs[source] * pairs[size_col]).astype("int8")
            aggregations[target] = "max"
        numerator = f"_interference_sum_{suffix}"
        count = f"_interference_n_{suffix}"
        # 原聚合 sum（求和）忽略缺失，而分母包含该档全部提示公司。
        pairs[numerator] = pairs["nrFirmsAnnLag1Alt"].fillna(0) * pairs[size_col]
        pairs[count] = pairs[size_col]
        aggregations[numerator] = "sum"
        aggregations[count] = "sum"
    events = pairs.groupby(["StkcdRecall", "event_Trddt"], sort=True).agg(aggregations).reset_index()
    events = events.rename(columns={"event_Trddt": "Trddt"})
    audit["聚合前95分位公司对数"] = int(pairs[CUE_COLS[0]].sum())
    del pairs, eligible
    gc.collect()

    # 正式主线在收益/控制完整案例筛选之前划干扰组。非事件日连续指标缺失，
    # 不进入分位数；事件表与最初日度源表求交即得到同一非缺失分组样本。
    present = _matches(prefilter_daily_keys, _date_keys(events["StkcdRecall"].astype("int64"), events["Trddt"]))
    events = events.loc[present].copy()
    events["year"] = events["Trddt"].dt.year
    audit["干扰分组年份统计"] = {}
    for suffix in ("S95p", "S5095"):
        numerator, count = f"_interference_sum_{suffix}", f"_interference_n_{suffix}"
        value = f"nrFirmsAnnLag1EW{suffix}"
        events[value] = events[numerator].div(events[count].replace(0, np.nan))
        grouped = merge.by_year_xtile_2(events, value, year_col="year")
        events[f"lowInterferenceEWLag1{suffix}"] = grouped.eq(1).astype("int8")
        events[f"highInterferenceEWLag1{suffix}"] = grouped.eq(2).astype("int8")
        audit["干扰分组年份统计"][suffix] = {
            str(year): {"有效事件日数": int(group[value].notna().sum()), "低干扰数": int(group[f"lowInterferenceEWLag1{suffix}"].sum()), "高干扰数": int(group[f"highInterferenceEWLag1{suffix}"].sum())}
            for year, group in events.groupby("year")
        }
    return events[DAILY_KEYS + BINARY_COLS + list(CHECK_COLUMNS.values())]


def _fingerprint(paths: dict) -> dict:
    sources = [
        paths[key] for key in (
            "MemoryMatrixPattern_path", "TRD_Dalyr_DGTW_path", "marketcap_group_path",
            "industry_path", "SC_NetWorkRelationsIndex_path", "analyst_forecast_path", "output_path",
        )
    ] + [
        ROOT / "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后/拆开AnanmID/Pattern样本_分析师共同覆盖_滚动365天.csv",
        MERGE_SCRIPT,
    ]
    return {
        "版本": 1, "研究期": [START_YEAR, END_YEAR],
        "准备代码摘要": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "输入": {str(path.relative_to(ROOT)): {"大小": path.stat().st_size, "修改时间": path.stat().st_mtime_ns} for path in sources},
        "缓存列": PANEL_COLS,
    }


def _write_panel(source: Path, events: pd.DataFrame, target: Path, chunk_size: int, audit: dict) -> None:
    selected = DAILY_KEYS + ["year", "yearQuarter", "smallFirm"] + OUTCOME_COLS + CONTROL_COLS + list(CHECK_COLUMNS)
    dtypes = {"StkcdRecall": "string", "yearQuarter": "string"}
    dtypes.update({column: "float32" for column in selected if column not in DAILY_KEYS + ["yearQuarter"]})
    writer, key_blocks, row_count, cue_count, mismatch = None, [], 0, 0, 0
    try:
        for chunk in pd.read_csv(source, usecols=selected, dtype=dtypes, parse_dates=["Trddt"], chunksize=chunk_size):
            chunk = chunk.loc[chunk["year"].between(START_YEAR, END_YEAR)].copy()
            if chunk.empty:
                continue
            chunk["StkcdRecall"] = _codes(chunk["StkcdRecall"])
            if chunk[DAILY_KEYS + ["yearQuarter"]].isna().any().any():
                raise ValueError("正式最终面板的公司、日期或年度季度键含缺失。")
            expected_quarter = chunk["Trddt"].dt.to_period("Q")
            actual_quarter = pd.to_datetime(chunk["yearQuarter"], errors="raise").dt.to_period("Q")
            if not expected_quarter.eq(actual_quarter).all() or not chunk["year"].eq(chunk["Trddt"].dt.year).all():
                raise ValueError("正式面板年份/季度与交易日期不一致。")
            valid_small = chunk["smallFirm"].dropna().isin([0, 1]).all()
            if not valid_small:
                raise ValueError("被提示公司大小标记包含非零一值。")
            chunk = chunk.merge(events, on=DAILY_KEYS, how="left", validate="one_to_one")
            for old, check in CHECK_COLUMNS.items():
                errors = ~chunk[old].eq(chunk[check].fillna(0))
                mismatch += int(errors.sum())
                if errors.any():
                    raise ValueError(f"重建90分位对照与正式面板不一致：{old}，本块{int(errors.sum())}行。")
            chunk[BINARY_COLS] = chunk[BINARY_COLS].fillna(0).astype("int8")
            numeric = OUTCOME_COLS + CONTROL_COLS + ["smallFirm"]
            chunk[numeric] = chunk[numeric].replace([np.inf, -np.inf], np.nan)
            chunk["year"] = chunk["year"].astype("int16")
            chunk = chunk.loc[:, PANEL_COLS]
            key_blocks.append(_date_keys(chunk["StkcdRecall"].astype("int64"), chunk["Trddt"]))
            row_count += len(chunk)
            cue_count += int(chunk[CUE_COLS[0]].sum())
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression="zstd")
            writer.write_table(table)
            _log(f"已准备{row_count:,}行。")
        if row_count == 0:
            raise ValueError("研究期内最终面板为空。")
        all_keys = np.concatenate(key_blocks)
        all_keys.sort()
        if np.any(all_keys[1:] == all_keys[:-1]):
            raise ValueError("最终面板存在跨分块公司日期重复键。")
    finally:
        if writer is not None:
            writer.close()
    if cue_count == 0:
        raise ValueError("最终面板没有95分位提示事件。")
    audit.update({"面板行数": row_count, "95分位提示股票日数": cue_count, "90分位对照不一致行数": mismatch})


def prepare_panel(cache_path: Path = DEFAULT_CACHE, *, rebuild: bool = False, chunk_size: int = 250_000) -> Path:
    """构建或校验后复用缓存；写文件仅在用户运行入口后发生。"""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("每块行数必须为正整数。")
    cache = Path(cache_path).expanduser().resolve()
    work = (ROOT / "work").resolve()
    if not cache.is_relative_to(work) or cache == work or cache.suffix != ".parquet":
        raise ValueError("缓存必须是项目临时工作目录下的独立列式面板文件。")
    merge = load_module(MERGE_SCRIPT, "_q95_mechanism_merge")
    paths = merge.build_pattern_paths(ROOT)
    fingerprint = _fingerprint(paths)
    manifest_path = cache.with_suffix(".parquet.manifest.json")
    if not rebuild and cache.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = manifest.get("指纹") == fingerprint and pq.read_schema(cache).names == PANEL_COLS
            valid = valid and pq.read_metadata(cache).num_rows == manifest["审计"]["面板行数"]
        except (OSError, ValueError, KeyError, pa.ArrowInvalid):
            valid = False
        if valid:
            _log(f"复用已核对输入来源和字段的缓存：{cache}")
            return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    lock = cache.with_suffix(".parquet.lock")
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"已有任务准备同一缓存；请等待该任务结束。如上次中断，请先确认没有任务运行，再删除锁文件：{lock}") from exc
    temporary = cache.with_suffix(".parquet.building")
    manifest_temporary = manifest_path.with_suffix(".json.building")
    try:
        with handle:
            handle.write(str(os.getpid()))
        audit = {
            "开始时间": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "样本": "正式规律公司最终样本，2005—2025年；沿用原收益和27项控制。",
            "提示门槛": "事件前三交易日总市值与事件上月市值断点比较。",
            "机制构造": "公司对层级与95以上、50—95规模组交互，股票日层级取最大值。",
            "干扰分组": "新规模组内历史共同公告公司数量等权平均；在最终基准样本筛选之前的日度样本中按年度二分，并列值不拆分。",
            "价格异动": "历史共同公告时任意一家公司进入异动组，沿用主线实际口径。",
        }
        events = _build_events(merge, paths, chunk_size, audit)
        _write_panel(paths["output_path"], events, temporary, chunk_size, audit)
        if _fingerprint(paths) != fingerprint:
            raise RuntimeError("准备期间输入或准备代码发生变化，未发布缓存。")
        audit["结束时间"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        manifest_temporary.write_text(json.dumps({"指纹": fingerprint, "审计": audit}, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(cache)
        manifest_temporary.replace(manifest_path)
        _log(f"准备完成：{audit['面板行数']:,}行，95分位提示{audit['95分位提示股票日数']:,}个股票日。")
        return cache
    finally:
        temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
