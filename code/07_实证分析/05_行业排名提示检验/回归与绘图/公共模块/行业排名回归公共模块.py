"""行业前4、前5回归的共同数据入口和估计方法。

一行代表“被提示公司×交易日”。只读取已经完成的行业面板，不重建提示、
收益或缩尾。复用项目工作目录中的季度列缓存，回归残差留在内存，
不再为每个模型写出再读回大临时文件。读取缓存时逐窗筛选，
只有最终有效样本完全相同的窗口才共享投影，不要求50窗共同完整。

无固定效应模型含常数；固定效应模型吸收公司×年度季度、日期或两者。
双向聚类得分跨季度累计，防止将同一公司在不同季度误当作不同公司簇。
有限样本修正和精确吸收秩沿用当前主表2的定义；表和图共用此函数。
导入本模块不读面板、不写文件、不执行回归。
依赖：numpy（数值数组）、pandas（数据表）、scipy（科学计算）。
"""

from __future__ import annotations

import gc
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager

# 限制数值库线程，避免同时占满本机处理器；导入数值库前设置才生效。
for thread_option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                      "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_option] = "2"

import numpy as np
import pandas as pd
from scipy import linalg, sparse, stats
from scipy.sparse.csgraph import connected_components


# 一、回归口径。变量顺序明确列出，便于对照和自行修改。
ENGINE_VERSION = "industry_regression_v1"
# 数学口径没有变化；计算实现版本单独记录，优化代码不会使已完成模型作废。
COMPUTATION_VERSION = "memory_grouped_v2"
OWN_EA_COLUMNS = (
    [f"ownEA_tm{j}" for j in range(10, 0, -1)]
    + ["ownEA"] + [f"ownEA_tp{j}" for j in range(1, 11)]
)
CONTROLS = OWN_EA_COLUMNS + [
    "ownOtherEventER", "ownPerformanceForecastBrief",
    "aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV",
]
RETURN_COLUMNS = [f"chAdjreturn_t_tp{h}_Recall_w" for h in range(1, 51)]
SPECS = {
    1: ([], "none", "无控制变量、无固定效应"),
    2: (OWN_EA_COLUMNS, "none", "自身盈余公告控制、无固定效应"),
    3: (CONTROLS, "none", "完整控制变量、无固定效应"),
    4: (CONTROLS, "date", "完整控制变量、交易日期固定效应"),
    5: (CONTROLS, "fq", "完整控制变量、公司×年度季度固定效应"),
    6: (CONTROLS, "both", "完整控制变量、公司×年度季度和交易日期固定效应"),
}


def find_root() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "AGENTS.md").is_file() and (folder / "中间结果").is_dir():
            return folder
    raise FileNotFoundError("找不到包含项目说明和中间结果的项目根目录。")


def output_root(root: Path) -> Path:
    return Path(root) / "图表结果/05_行业排名提示检验"


def signature_hash(signature: dict) -> str:
    raw = json.dumps(signature, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def runtime_guard(root: Path, task_name: str):
    """两个入口共用一把进程锁；第二个入口等待，避免并发准备和争抢内存。"""
    folder = Path(root) / "work/行业排名回归"
    folder.mkdir(parents=True, exist_ok=True)
    # 不删除锁文件；进程退出或中断会自动释放系统锁，没有遗留锁的判断歧义。
    with (folder / "运行锁.lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{task_name}：另一行业回归任务正在运行，等待其完成后继续。", flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(f"{task_name}；进程号{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _same_input_definition(left: dict, right: dict) -> bool:
    # 模块内容摘要是实现来源；其余输入、列序、统计版本及推断口径均须一致。
    return ({key: value for key, value in left.items() if key != "module_sha256"}
            == {key: value for key, value in right.items() if key != "module_sha256"})


def _find_compatible_cache(root: Path, signature: dict) -> dict | None:
    """仅实现变化时复用完工缓存，并优先保留已有结果引用的输入签名。"""
    top_n = signature["top_n"]
    folder = Path(root) / "work/行业排名回归/季度缓存"
    preferred = set()
    result_paths = [
        output_root(root) / f"回归结果/行业前{top_n}名/回归核对摘要.json",
        output_root(root) / f"累计系数图/行业前{top_n}名_回归核对摘要.json",
    ]
    for path in result_paths:
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            identity = result.get("输入签名", result.get("设置", {}).get("输入签名"))
            if identity:
                preferred.add(identity)
    candidates = []
    pattern = f"行业前{top_n}名_{signature['start_year']}_{signature['end_year']}_*/完成清单.json"
    for path in folder.glob(pattern):
        cached = json.loads(path.read_text(encoding="utf-8"))
        old_signature = cached.get("signature", {})
        if not _same_input_definition(old_signature, signature):
            continue
        if cached.get("input_id") != signature_hash(old_signature):
            raise ValueError(f"季度缓存的输入签名校验失败：{path}")
        if Path(cached["path"]).resolve() != path.parent.resolve():
            raise ValueError(f"季度缓存清单记录的目录与实际位置不符：{path}")
        candidates.append((cached["input_id"] not in preferred, path.stat().st_mtime_ns, cached))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:2])
    cached = candidates[0][2]
    _validate_cache(cached)
    return cached


def _file_state(path: Path) -> dict:
    info = path.stat()
    return {"path": str(path.resolve()), "size": info.st_size,
            "mtime_ns": info.st_mtime_ns}


def _atomic_write(path: Path, writer) -> None:
    """临时写出只放项目工作目录，写完后原子替换目标文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    work = find_root() / "work/行业排名回归/写出临时文件"
    work.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=work, suffix=path.suffix)
    os.close(handle)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path: Path, value) -> None:
    _atomic_write(path, lambda temporary: temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    ))


def save_csv(path: Path, rows) -> None:
    _atomic_write(path, lambda temporary: pd.DataFrame(rows).to_csv(
        temporary, index=False, encoding="utf-8-sig",
    ))


def _digest_values(digest, values: np.ndarray) -> None:
    # 逐小块计算摘要，避免额外复制整个季度的宽数组。
    for start in range(0, len(values), 10_000):
        digest.update(np.ascontiguousarray(values[start:start + 10_000],
                                         dtype="<f8").tobytes())


# 二、只扫描一次大面板，建立供表和图共同使用的季度列缓存。
def prepare_cache(root: Path, top_n: int, start_year: int, end_year: int,
                  chunk_size: int = 250_000) -> dict:
    if top_n not in (4, 5):
        raise ValueError("本任务只接受行业前4或前5。")
    if not 1900 <= start_year <= end_year <= 2200 or chunk_size < 1:
        raise ValueError("年份范围或分块行数无效。")
    root = Path(root).resolve()
    folder = root / f"中间结果/04_合并数据/行业排名提示检验/行业前{top_n}名"
    source = folder / f"行业前{top_n}名_最终回归面板.csv"
    audit_path = folder / "构造口径与核对摘要.json"
    if not source.is_file() or not audit_path.is_file():
        raise FileNotFoundError(f"行业前{top_n}名的面板或构造摘要缺失：{folder}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("cue_definition") != "industry_top" or audit.get("industry_top_n") != top_n:
        raise ValueError("面板构造摘要与行业提示定义不一致。")
    if start_year < audit["start_year"] or end_year > audit["end_year"]:
        raise ValueError("要求的年份超出已完成面板范围，请修改入口参数。")
    cue_column = f"quarterLag1IndTop{top_n}"
    x_columns = [cue_column, *CONTROLS]
    signature = {
        "version": ENGINE_VERSION,
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "panel": _file_state(source),
        "panel_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "top_n": top_n, "start_year": start_year, "end_year": end_year,
        "x_columns": x_columns, "returns": RETURN_COLUMNS,
        "numeric_dtype": "float64", "outcome_scale": 100,
        "inference": "主表2含常数的有限样本修正；精确固定效应秩；残差自由度双侧t检验",
        "comparison_note": audit.get("comparison_note", ""),
    }
    input_id = signature_hash(signature)
    cache_root = root / "work/行业排名回归/季度缓存"
    cache_path = cache_root / f"行业前{top_n}名_{start_year}_{end_year}_{input_id[:20]}"
    manifest_path = cache_path / "完成清单.json"
    compatible = _find_compatible_cache(root, signature)
    if compatible is not None:
        print(f"复用行业前{top_n}名已完成季度缓存：{compatible['path']}", flush=True)
        return compatible
    if manifest_path.is_file():
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        if result.get("signature") != signature or result.get("input_id") != input_id:
            raise ValueError(f"季度缓存签名不一致，请核对：{manifest_path}")
        _validate_cache(result)
        print(f"复用行业前{top_n}名季度缓存：{cache_path}", flush=True)
        return result
    if cache_path.exists():
        raise ValueError(f"季度缓存未完成，请移走此任务的不完整缓存后重试：{cache_path}")

    required = ["StkcdRecall", "Trddt", "yearQuarter", "cueDefinition",
                "industryTopN", *x_columns, *RETURN_COLUMNS]
    header = pd.read_csv(source, nrows=0, encoding="utf-8-sig").columns
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"行业面板缺少必要字段：{missing}")
    cache_root.mkdir(parents=True, exist_ok=True)
    # 构造失败时自动清理本次临时碎片；只在全部季度核对通过后保留正式缓存。
    with tempfile.TemporaryDirectory(prefix="正在准备_", dir=cache_root) as temp_name:
        staging = Path(temp_name)
        parts = staging / "读取碎片"
        parts.mkdir()
        quarter_parts: dict[int, list[Path]] = {}
        source_rows = selected_rows = cue_days = 0
        all_firms = set()
        dtype = {c: "float64" for c in [*x_columns, *RETURN_COLUMNS, "industryTopN"]}
        dtype.update({c: "string" for c in ["StkcdRecall", "Trddt", "yearQuarter", "cueDefinition"]})
        reader = pd.read_csv(source, usecols=required, dtype=dtype,
                             chunksize=chunk_size, encoding="utf-8-sig")
        for part_number, chunk in enumerate(reader):
            source_rows += len(chunk)
            dates = pd.to_datetime(chunk["Trddt"], errors="raise")
            if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
                raise ValueError("交易日期缺失或包含非零时分秒。")
            selected = dates.dt.year.between(start_year, end_year)
            if not selected.any():
                continue
            chunk = chunk.loc[selected].copy()
            dates = dates.loc[selected]
            firms = chunk["StkcdRecall"].str.strip()
            if not firms.str.fullmatch(r"\d{6}").fillna(False).all():
                raise ValueError("被提示公司代码不是六位数字或存在缺失。")
            quarters = dates.dt.to_period("Q")
            recorded_quarters = pd.to_datetime(chunk["yearQuarter"], errors="raise")
            if recorded_quarters.isna().any() or not recorded_quarters.dt.to_period("Q").eq(quarters).all():
                raise ValueError("年度季度字段与交易日期不一致。")
            if not chunk["cueDefinition"].eq("industry_top").fillna(False).all() or not chunk["industryTopN"].eq(top_n).all():
                raise ValueError("面板行内的行业提示定义或前几名参数冲突。")
            if not chunk[cue_column].isin([0, 1]).all():
                raise ValueError("主提示变量含缺失或非零一取值，不能直接补零。")
            firm = firms.to_numpy(dtype="int32")
            day = dates.to_numpy(dtype="datetime64[D]").astype("int32")
            q = quarters.astype("int64").to_numpy()
            x = chunk[x_columns].to_numpy(dtype="float64")
            y = chunk[RETURN_COLUMNS].to_numpy(dtype="float64")
            selected_rows += len(chunk)
            cue_days += int(x[:, 0].sum())
            all_firms.update(int(value) for value in np.unique(firm))
            for quarter in np.unique(q):
                positions = np.flatnonzero(q == quarter)
                path = parts / f"q{quarter}_{part_number}.npz"
                np.savez(path, firm=firm[positions], day=day[positions],
                         x=x[positions], y=y[positions])
                quarter_parts.setdefault(int(quarter), []).append(path)
            print(f"行业前{top_n}名：已扫描{source_rows:,}行", flush=True)
            del chunk, x, y
        if source_rows != int(audit["panel_rows"]):
            raise ValueError("面板实际行数与构造摘要不一致，未保留季度缓存。")
        if not selected_rows:
            raise ValueError("指定年份没有可用的公司日观测。")
        if start_year == audit["start_year"] and end_year == audit["end_year"]:
            if cue_days != int(audit["cue_firm_days"]) or len(all_firms) != int(audit["panel_firms"]):
                raise ValueError("提示日数或公司数与面板构造摘要不一致。")
        quarters_info = []
        for quarter, paths in sorted(quarter_parts.items()):
            arrays = {name: [] for name in ("firm", "day", "x", "y")}
            for path in paths:
                with np.load(path, allow_pickle=False) as piece:
                    for name in arrays:
                        arrays[name].append(piece[name])
            merged = {name: np.concatenate(values) for name, values in arrays.items()}
            del arrays
            key = merged["day"].astype("int64") * 1_000_000 + merged["firm"]
            order = np.argsort(key)
            sorted_key = key[order]
            if np.any(np.diff(sorted_key) == 0):
                raise ValueError(f"季度{quarter}存在重复的被提示公司×交易日。")
            quarter_dir = staging / f"q{quarter}"
            quarter_dir.mkdir()
            for name in merged:
                values = merged[name][order]
                # 列优先保存使每次只读取某一收益窗口时不必加载其余49列。
                np.save(quarter_dir / f"{name}.npy", np.asfortranarray(values))
                del values
            base_digest = hashlib.sha256()
            # 从已排序的列缓存分块取值，不再复制整季排序后的宽数组。
            sorted_x = np.load(quarter_dir / "x.npy", mmap_mode="r")
            sorted_y = np.load(quarter_dir / "y.npy", mmap_mode="r")
            _digest_values(base_digest, sorted_x[:, 1:])
            _digest_values(base_digest, sorted_y)
            del sorted_x, sorted_y
            files = {f"q{quarter}/{name}.npy": _file_state(quarter_dir / f"{name}.npy")["size"]
                     for name in merged}
            quarters_info.append({
                "quarter": quarter, "rows": len(key), "files": files,
                "key_digest": hashlib.sha256(sorted_key.astype("<i8").tobytes()).hexdigest(),
                "base_data_digest": base_digest.hexdigest(),
            })
            for path in paths:
                path.unlink()
            del merged, key, order, sorted_key
            gc.collect()
        parts.rmdir()
        if _file_state(source) != signature["panel"]:
            raise ValueError("读取过程中源面板发生变化，未保留缓存，请待面板完成后重新运行。")
        result = {
            "signature": signature, "input_id": input_id, "path": str(cache_path),
            "root": str(root), "top_n": top_n, "rows": selected_rows,
            "cue_days": cue_days, "firms": len(all_firms), "quarters": quarters_info,
        }
        (staging / "完成清单.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        # 同卷整体改名，完成清单和全部季度文件同时发布；避免中断留下半套缓存。
        # 临时上下文随后清理已经不存在的旧路径，不触及正式缓存。
        staging.rename(cache_path)
    _validate_cache(result)
    print(f"行业前{top_n}名季度缓存完成：{selected_rows:,}行、{cue_days:,}个提示公司日", flush=True)
    return result


def _validate_cache(cache: dict) -> None:
    total = 0
    for info in cache["quarters"]:
        total += info["rows"]
        for relative, expected_size in info["files"].items():
            path = Path(cache["path"]) / relative
            if not path.is_file() or path.stat().st_size != expected_size:
                raise ValueError(f"季度缓存缺失或大小不符，请重新准备此缓存：{path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            field = path.stem
            if field == "x":
                shape = (info["rows"], len(cache["signature"]["x_columns"]))
            elif field == "y":
                shape = (info["rows"], len(RETURN_COLUMNS))
            else:
                shape = (info["rows"],)
            expected_dtype = np.dtype("float64" if field in ("x", "y") else "int32")
            if array.shape != shape or array.dtype != expected_dtype:
                raise ValueError(f"季度缓存的形状或数据类型不符：{path}")
            del array
    if not cache["quarters"] or total != cache["rows"]:
        raise ValueError("季度缓存清单为空或总行数不一致。")


def verify_base_keys(caches: list[dict]) -> None:
    """同时运行前4、前5时验证基础键、控制与收益一致，以及提示的包含关系。"""
    if len(caches) < 2:
        return
    reference = caches[0]
    for current in caches[1:]:
        fields = ("quarter", "rows", "key_digest", "base_data_digest")
        left = [{key: info[key] for key in fields} for info in reference["quarters"]]
        right = [{key: info[key] for key in fields} for info in current["quarters"]]
        if left != right:
            raise ValueError("前4与前5基础公司日键、控制或收益不一致，不能作为同基础样本比较。")
        for info in reference["quarters"]:
            name = f"q{info['quarter']}/x.npy"
            a = np.load(Path(reference["path"]) / name, mmap_mode="r")[:, 0]
            b = np.load(Path(current["path"]) / name, mmap_mode="r")[:, 0]
            smaller, larger = (a, b) if reference["top_n"] < current["top_n"] else (b, a)
            if np.any(smaller > larger):
                raise ValueError("行业前4提示并非行业前5提示的子集。")
    print("已核对前4、前5基础键、控制和收益一致，提示包含关系正确。", flush=True)


# 三、逐窗口确定有效样本；最终样本逐行相同才可以共享固定效应处理。
def _load_quarter_columns(cache: dict, quarter: int, x_indices: list[int], horizons: list[int]):
    folder = Path(cache["path"]) / f"q{quarter}"
    x = np.load(folder / "x.npy", mmap_mode="r")[:, x_indices]
    y = np.load(folder / "y.npy", mmap_mode="r")[:, [h - 1 for h in horizons]]
    firm = np.load(folder / "firm.npy", mmap_mode="r")
    day = np.load(folder / "day.npy", mmap_mode="r")
    return x, y, firm, day


def _remove_singletons(valid: np.ndarray, axes: list[np.ndarray]) -> np.ndarray:
    keep = valid.copy()
    while axes and keep.any():
        positions = np.flatnonzero(keep)
        drop = np.zeros(len(positions), dtype=bool)
        for codes in axes:
            counts = np.bincount(codes[positions], minlength=int(codes.max()) + 1)
            drop |= counts[codes[positions]] == 1
        if not drop.any():
            break
        keep[positions[drop]] = False
    return keep


def _window_samples(cache: dict, horizons: list[int], x_indices: list[int], effects: str):
    masks = np.empty((len(horizons), cache["rows"]), dtype=bool)
    missing = np.zeros(len(horizons), dtype="int64")
    singletons = np.zeros(len(horizons), dtype="int64")
    start = 0
    for info in cache["quarters"]:
        x, y, firm, day = _load_quarter_columns(cache, info["quarter"], x_indices, horizons)
        valid_x = np.isfinite(x).all(axis=1)
        axes = []
        if effects in ("fq", "both"):
            axes.append(pd.factorize(firm, sort=False)[0])
        if effects in ("date", "both"):
            axes.append(pd.factorize(day, sort=False)[0])
        end = start + info["rows"]
        for j in range(len(horizons)):
            valid = valid_x & np.isfinite(y[:, j])
            keep = _remove_singletons(valid, axes)
            missing[j] += int((~valid).sum())
            singletons[j] += int(valid.sum() - keep.sum())
            masks[j, start:end] = keep
        start = end
        del x, y, firm, day, axes
    if start != cache["rows"]:
        raise ValueError("窗口有效样本长度与季度缓存不一致。")
    groups = []
    for j, horizon in enumerate(horizons):
        digest = hashlib.sha256(np.packbits(masks[j]).tobytes()).hexdigest()
        matched = None
        for group in groups:
            if group["digest"] == digest and np.array_equal(group["keep"], masks[j]):
                matched = group
                break
        if matched is None:
            matched = {"digest": digest, "keep": masks[j], "outcomes": []}
            groups.append(matched)
        # 即使最终样本相同，原始缺失数和删除单例数也分别按窗口保留。
        matched["outcomes"].append((horizon, int(missing[j]), int(singletons[j])))
    return groups


def _demean(values: np.ndarray, codes: np.ndarray, groups: int) -> np.ndarray:
    counts = np.bincount(codes, minlength=groups)
    means = np.column_stack([
        np.bincount(codes, weights=values[:, j], minlength=groups) / counts
        for j in range(values.shape[1])
    ])
    return values - means[codes]


def _absorb_both(values: np.ndarray, firms: np.ndarray, days: np.ndarray):
    """季度直接解，采用既有日期固定效应公共模块的舒尔补方法。

    只复制其短数值核心，避免导入历史入口和额外的固定效应库。
    连通分量既决定基准日期，也用于精确计算本季度吸收秩。
    """
    n, nf, nd = len(values), int(firms.max()) + 1, int(days.max()) + 1
    counts_f = np.bincount(firms).astype("float64")
    counts_d = np.bincount(days).astype("float64")
    row = np.arange(n)
    ones = np.ones(n)
    sum_f = sparse.csr_matrix((ones, (firms, row)), shape=(nf, n))
    sum_d = sparse.csr_matrix((ones, (days, row)), shape=(nd, n))
    counts = sparse.csr_matrix((ones, (firms, days)), shape=(nf, nd))
    components, labels = connected_components((counts.T @ counts).tocsr(), directed=False)
    reference_days = np.unique(labels, return_index=True)[1]
    free_days = np.setdiff1d(np.arange(nd), reference_days)
    schur = np.diag(counts_d) - (counts.T @ counts.multiply(1 / counts_f[:, None])).toarray()
    schur = (schur + schur.T) / 2
    within = values - ((sum_f @ values) / counts_f[:, None])[firms]
    date_effects = np.zeros((nd, values.shape[1]))
    if free_days.size:
        factor = linalg.cho_factor(schur[np.ix_(free_days, free_days)], lower=True)
        date_effects[free_days] = linalg.cho_solve(factor, (sum_d @ within)[free_days])
    result = within - date_effects[days] + ((counts @ date_effects) / counts_f[:, None])[firms]
    if not np.isfinite(result).all():
        raise FloatingPointError("季度固定效应直接解产生非有限结果。")
    return result, nf + nd - int(components)


def _project(values: np.ndarray, firm: np.ndarray, day: np.ndarray, effects: str):
    if effects == "none":
        return values, 0
    _, fi = np.unique(firm, return_inverse=True)
    _, di = np.unique(day, return_inverse=True)
    if effects == "fq":
        return _demean(values, fi, int(fi.max()) + 1), int(fi.max()) + 1
    if effects == "date":
        return _demean(values, di, int(di.max()) + 1), int(di.max()) + 1
    return _absorb_both(values, fi, di)


def _stars(pvalue: float | None) -> str:
    return "" if pvalue is None else "***" if pvalue < .01 else "**" if pvalue < .05 else "*" if pvalue < .10 else ""


# 四、将投影后的解释变量和小批收益留在内存，不再产生逐模型残差文件。
def _fit_same_sample(cache: dict, group: dict, spec_id: int, full_coefficients: bool):
    controls, effects, _ = SPECS[spec_id]
    columns = [f"quarterLag1IndTop{cache['top_n']}", *controls]
    x_indices = [cache["signature"]["x_columns"].index(column) for column in columns]
    horizons = [outcome[0] for outcome in group["outcomes"]]
    keep = group["keep"]
    n, p, m = int(keep.sum()), len(columns), len(horizons)
    if n <= p + 1:
        raise ValueError("删除缺失和固定效应单例后，有效样本不足。")
    # 全控制解释变量约2.5GB；收益默认按每批5窗处理，前4和前5依次估计。
    x_memory = np.empty((n, p), dtype="float64")
    y_memory = np.empty((n, m), dtype="float64")
    firm_memory, day_memory = np.empty(n, dtype="int32"), np.empty(n, dtype="int32")
    a, b = np.zeros((p, p)), np.zeros((p, m))
    sum_x, sum_y, sum_yy = np.zeros(p), np.zeros(m), np.zeros(m)
    within_yy = np.zeros(m)
    blocks = []
    source_start = destination = cue_n = fe_rank = 0
    for info in cache["quarters"]:
        source_end = source_start + info["rows"]
        selected = keep[source_start:source_end]
        source_start = source_end
        if not selected.any():
            continue
        x, y, firm, day = _load_quarter_columns(cache, info["quarter"], x_indices, horizons)
        x, y, firm, day = x[selected], y[selected], firm[selected], day[selected]
        end = destination + len(y)
        cue_n += int(x[:, 0].sum())
        sum_x += x.sum(axis=0)
        sum_y += y.sum(axis=0)
        sum_yy += np.einsum("ij,ij->j", y, y)
        transformed, rank = _project(np.column_stack([x, y]), firm, day, effects)
        fe_rank += rank
        xr, yr = transformed[:, :p], transformed[:, p:]
        a += xr.T @ xr
        b += xr.T @ yr
        within_yy += np.einsum("ij,ij->j", yr, yr)
        x_memory[destination:end] = xr
        y_memory[destination:end] = yr
        firm_memory[destination:end], day_memory[destination:end] = firm, day
        blocks.append((destination, end))
        destination = end
        del x, y, firm, day, transformed, xr, yr
    if destination != n or not 0 < cue_n < n:
        raise ValueError("样本长度不一致，或提示变量在当前模型中没有变异。")
    mean_x, mean_y = sum_x / n, sum_y / n
    total_yy = sum_yy - sum_y * mean_y
    if effects == "none":
        # 无固定效应时保留显式常数，用全样本均值，不能季内中心化。
        a -= np.outer(sum_x, sum_x) / n
        b -= sum_x[:, None] * mean_y[None, :]
        within_yy = total_yy.copy()
    a = (a + a.T) / 2
    threshold = max(float(np.abs(np.diag(a)).max()), 1.0) * 1e-14
    active = np.flatnonzero(np.diag(a) > threshold)
    if 0 not in active:
        raise ValueError("行业提示变量被固定效应吸收，无法识别提示系数。")
    aa = a[np.ix_(active, active)]
    scale = np.sqrt(np.diag(aa))
    standardized = aa / scale[:, None] / scale[None, :]
    condition = float(np.linalg.cond(standardized))
    if not np.isfinite(condition) or condition >= 1e12:
        raise ValueError("解释变量不满秩或条件数过大，请检查控制变量共线性。")
    inverse = np.linalg.inv(standardized) / scale[:, None] / scale[None, :]
    beta = inverse @ b[active]
    k = len(active) + 1
    degrees = n - len(active) - fe_rank - int(effects == "none")
    if degrees <= 0 or np.any(total_yy <= 0) or np.any(within_yy <= 0):
        raise ValueError("残差自由度或因变量变异无效。")
    firms_all, days_all = np.unique(firm_memory), np.unique(day_memory)
    gf, gd, pa = len(firms_all), len(days_all), len(active)
    if min(gf, gd) < 2:
        raise ValueError("双向聚类至少需要两个公司簇和两个日期簇。")
    cue_position = int(np.flatnonzero(active == 0)[0])
    sse = np.zeros(m)
    if full_coefficients:
        firm_scores, date_scores = np.zeros((m, gf, pa)), np.zeros((m, gd, pa))
        observation_meat = np.zeros((m, pa, pa))
    else:
        # 图只需要提示系数。左右得分同时保留，与原完整协方差的双线性式一致。
        firm_left, firm_right = np.zeros((m, gf)), np.zeros((m, gf))
        date_left, date_right = np.zeros((m, gd)), np.zeros((m, gd))
        observation_product = np.zeros(m)
    for start, end in blocks:
        xr = x_memory[start:end] if pa == p else x_memory[start:end, active]
        yr = y_memory[start:end]
        if effects == "none":
            xr = xr - mean_x[active]
            yr = yr - mean_y
        errors = yr - xr @ beta
        sse += np.einsum("ij,ij->j", errors, errors)
        fi = np.searchsorted(firms_all, firm_memory[start:end])
        di = np.searchsorted(days_all, day_memory[start:end])
        if not full_coefficients:
            projected_left = xr @ inverse[cue_position, :]
            projected_right = xr @ inverse[:, cue_position]
        for outcome in range(m):
            if full_coefficients:
                score = xr * errors[:, [outcome]]
                observation_meat[outcome] += score.T @ score
                for column in range(pa):
                    firm_scores[outcome, :, column] += np.bincount(fi, weights=score[:, column], minlength=gf)
                    date_scores[outcome, :, column] += np.bincount(di, weights=score[:, column], minlength=gd)
                del score
            else:
                left = projected_left * errors[:, outcome]
                right = projected_right * errors[:, outcome]
                firm_left[outcome] += np.bincount(fi, weights=left, minlength=gf)
                firm_right[outcome] += np.bincount(fi, weights=right, minlength=gf)
                date_left[outcome] += np.bincount(di, weights=left, minlength=gd)
                date_right[outcome] += np.bincount(di, weights=right, minlength=gd)
                observation_product[outcome] += float(left @ right)
                del left, right
        del xr, yr, errors
    # 得分累计完成后先释放大数组，再逐窗口返回结果，便于入口立即保存。
    del x_memory, y_memory, firm_memory, day_memory
    gc.collect()
    # 公司簇跨季度累计后求外积；交集簇是已核验唯一的公司×日期。
    common_scale = (n - 1) / (n - k)
    for outcome, (horizon, missing, singletons) in enumerate(group["outcomes"]):
        if n + missing + singletons != cache["rows"]:
            raise ValueError("窗口缺失、单例和保留行数之和不等于缓存行数。")
        variance = np.full(pa, np.nan)
        if full_coefficients:
            meat = (gf / (gf - 1) * (firm_scores[outcome].T @ firm_scores[outcome])
                    + gd / (gd - 1) * (date_scores[outcome].T @ date_scores[outcome])
                    - n / (n - 1) * observation_meat[outcome]) * common_scale
            covariance = inverse @ meat @ inverse
            diagonal = np.diag((covariance + covariance.T) / 2)
            if not np.isfinite(diagonal).all():
                raise ValueError("完整双向聚类协方差含非有限值。")
            variance[:] = diagonal
        else:
            variance[cue_position] = (
                gf / (gf - 1) * (firm_left[outcome] @ firm_right[outcome])
                + gd / (gd - 1) * (date_left[outcome] @ date_right[outcome])
                - n / (n - 1) * observation_product[outcome]
            ) * common_scale
        diagnostics = {
            "n": n, "cue_n": cue_n, "missing": missing, "singletons": singletons,
            "fe_rank": fe_rank, "degrees": degrees, "gf": gf, "gd": gd, "k": k,
            "condition": condition, "total_yy": float(total_yy[outcome]),
            "within_yy": float(within_yy[outcome]), "sse": float(sse[outcome]),
            "constant": float(mean_y[outcome] - mean_x[active] @ beta[:, outcome]) if effects == "none" else None,
            "shared_windows": len(horizons),
        }
        yield _result_rows(cache, horizon, spec_id, columns, active,
                           beta[:, outcome], variance, diagnostics, full_coefficients)


def _result_rows(cache, horizon, spec_id, columns, active, beta, variance, info, full_coefficients):
    _, effects, description = SPECS[spec_id]
    cue_position = int(np.flatnonzero(active == 0)[0])
    if not np.isfinite(variance[cue_position]) or variance[cue_position] <= 0:
        raise ValueError("行业提示系数的双向聚类方差非有限或非正。")
    standard_errors = np.full(len(active), np.nan)
    positive = variance > 0
    standard_errors[positive] = np.sqrt(variance[positive])
    tvalues = beta / standard_errors
    pvalues = 2 * stats.t.sf(np.abs(tvalues), info["degrees"])
    beta_cue, se_cue = float(beta[cue_position]), float(standard_errors[cue_position])
    p_cue = float(pvalues[cue_position])
    summary = {
        "行业前几名": int(cache["top_n"]), "提示变量": columns[0],
        "窗口序号": int(horizon), "收益窗口": f"[t,t+{horizon}]",
        "因变量": RETURN_COLUMNS[horizon - 1], "规格": int(spec_id), "规格说明": description,
        "日期固定效应": "是" if effects in ("date", "both") else "否",
        "被提示公司×年度季度固定效应": "是" if effects in ("fq", "both") else "否",
        "线索系数": beta_cue * 100, "聚类标准误": se_cue * 100,
        "t值": float(tvalues[cue_position]), "p值": p_cue, "显著性": _stars(p_cue),
        "回归样本量": info["n"], "有效提示公司日": info["cue_n"],
        "被提示公司聚类数": info["gf"], "日期聚类数": info["gd"],
        "删除缺失观测数": info["missing"], "删除单例组观测数": info["singletons"],
        "固定效应秩": info["fe_rank"], "残差自由度": info["degrees"],
        "解释变量矩阵条件数": info["condition"],
        "求解方式": "季度吸收固定效应、内存累计和标准化求逆",
        "R方": 1 - info["sse"] / info["total_yy"],
        "组内R方": 1 - info["sse"] / info["within_yy"],
        "结果单位": "百分点", "输入签名": cache["input_id"], "模型状态": "完成",
        "常数项": info["constant"],
        "常数项说明": "原小数收益单位" if effects == "none" else "已由固定效应吸收，不单独报告",
        "有限样本修正": "是", "协方差修正参数数": info["k"],
        "置信区间分布": "残差自由度的双侧t分布，逐窗口区间",
        "与旧提示口径比较说明": cache["signature"]["comparison_note"],
        "估计口径版本": ENGINE_VERSION, "计算实现版本": COMPUTATION_VERSION,
        "共享投影窗口数": info["shared_windows"],
        "方差计算方式": "全部斜率双向聚类协方差" if full_coefficients else "提示系数左右影响得分双向聚类",
    }
    for level in (90, 95, 99):
        critical = float(stats.t.ppf((1 + level / 100) / 2, info["degrees"]))
        summary[f"{level}%置信区间下界"] = (beta_cue - critical * se_cue) * 100
        summary[f"{level}%置信区间上界"] = (beta_cue + critical * se_cue) * 100
    coefficients = []
    if full_coefficients:
        lookup = {int(column): pos for pos, column in enumerate(active)}
        for order, variable in enumerate(columns):
            pos = lookup.get(order)
            estimated = pos is not None
            se_valid = estimated and np.isfinite(standard_errors[pos])
            coefficient = float(beta[pos]) if estimated else None
            se = float(standard_errors[pos]) if se_valid else None
            probability = float(pvalues[pos]) if se_valid else None
            coefficients.append({
                "行业前几名": int(cache["top_n"]), "窗口序号": int(horizon),
                "收益窗口": summary["收益窗口"], "因变量": summary["因变量"],
                "规格": int(spec_id), "变量": variable, "变量顺序": order + 1,
                "变量类型": "线索事件" if order == 0 else "控制变量",
                "估计状态": "被固定效应吸收或无变异" if not estimated else "已估计" if se_valid else "双向聚类方差非正",
                "系数": coefficient, "聚类标准误": se,
                "系数（百分点）": coefficient * 100 if estimated else None,
                "聚类标准误（百分点）": se * 100 if se_valid else None,
                "t值": float(tvalues[pos]) if se_valid else None,
                "p值": probability, "显著性": _stars(probability),
                "回归样本量": info["n"], "输入签名": cache["input_id"],
                "估计口径版本": ENGINE_VERSION, "计算实现版本": COMPUTATION_VERSION,
            })
    return summary, coefficients


# 五、表格和图使用同一数学口径，按小批、相同样本组共享计算。
def _iter_model_batch(cache, horizons, spec_id, full_coefficients):
    controls, effects, _ = SPECS[spec_id]
    columns = [f"quarterLag1IndTop{cache['top_n']}", *controls]
    indices = [cache["signature"]["x_columns"].index(column) for column in columns]
    groups = _window_samples(cache, horizons, indices, effects)
    for group in groups:
        window_names = [outcome[0] for outcome in group["outcomes"]]
        print(f"行业前{cache['top_n']}名：规格{spec_id}，窗口{window_names}，有效样本{int(group['keep'].sum()):,}；内存估计。", flush=True)
        for result in _fit_same_sample(cache, group, spec_id, full_coefficients):
            yield result


def fit_model(cache: dict, horizon: int, spec_id: int = 6) -> tuple[dict, list[dict]]:
    """保留单模型接口；表格仍报告全部控制系数和标准误。"""
    if spec_id not in SPECS or horizon not in range(1, 51):
        raise ValueError("规格应为1至6，累计窗口终点应为1至50。")
    return next(_iter_model_batch(cache, [horizon], spec_id, True))


def iter_table_models(cache: dict, pending_models: list[tuple[int, int]]):
    """按规格合并两个收益窗口；每个返回模型均可立即保存。"""
    if len(pending_models) != len(set(pending_models)):
        raise ValueError("待估计的表格模型重复。")
    for horizon, spec_id in pending_models:
        if horizon not in (1, 10) or spec_id not in SPECS:
            raise ValueError("表格窗口仅支持1、10，规格仅支持1至6。")
    for spec_id in sorted({spec for _, spec in pending_models}):
        horizons = sorted(h for h, spec in pending_models if spec == spec_id)
        yield from _iter_model_batch(cache, horizons, spec_id, True)


def iter_cumulative_models(cache: dict, horizons, batch_size: int = 5):
    """图只估计提示系数方差；小批限制内存，不改变逐窗有效样本。"""
    horizons = list(horizons)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("窗口小批数必须为正整数。")
    if len(horizons) != len(set(horizons)) or any(type(h) is not int or not 1 <= h <= 50 for h in horizons):
        raise ValueError("累计窗口必须为不重复的1至50整数。")
    for start in range(0, len(horizons), batch_size):
        batch = horizons[start:start + batch_size]
        for summary, _ in _iter_model_batch(cache, batch, 6, False):
            yield summary


# 六、表2主规格与累计图同一窗口的结果应一致，避免悄悄混入口径。
def check_anchor(summary: dict, other: dict) -> None:
    for field in ("输入签名", "提示变量", "结果单位", "模型状态"):
        if str(summary.get(field)) != str(other.get(field)):
            raise ValueError(f"表2与累计图的{field}不同，不能复用或比较旧结果。")
    for field in ("行业前几名", "窗口序号", "规格", "回归样本量", "残差自由度"):
        if int(summary[field]) != int(other[field]):
            raise ValueError(f"表2与累计图的{field}不一致。")
    for field in ("线索系数", "聚类标准误"):
        if not np.isclose(float(summary[field]), float(other[field]), rtol=1e-8, atol=1e-10):
            raise ValueError(f"表2与累计图的{field}不一致，请检查输入或回归设置。")
