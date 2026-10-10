# -*- coding: utf-8 -*-
"""
分析师共同覆盖识别：Pattern 样本 + Full 样本优化版

核心判定逻辑保持不变：
1. 对每个 cue firm - recall firm 公司对；
2. 在 cue firm 公告日前 lookback_days 天窗口内；
3. 如果存在至少一个分析师同时覆盖 cue firm 和 recall firm；
4. 则把该公司对标记为 analyst-following overlap。

主要优化：
1. 路径不再写死，可通过 --root 指定项目根目录；
2. 把原来的 DataFrame.loc 日期窗口筛选，改成 numpy.searchsorted 快速定位窗口；
3. 对同一个「股票代码-公告日-窗口」的分析师集合做缓存；
4. Full 样本只读取一遍，同时处理多个 lookback 窗口；
5. Pattern 和 Full 共用一套函数，减少重复代码；
6. 输出文件名和原始代码保持一致。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 全局设置
# =============================================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

NS_PER_DAY = 24 * 60 * 60 * 1_000_000_000

FORECAST_RELATIVE_PATH = Path("中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv")
PATTERN_RELATIVE_PATH = Path("中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv")
FULL_RELATIVE_PATH = Path("中间结果/03_记忆矩阵/MemoryMatrixFull(封装优化版).csv")
OUTPUT_RELATIVE_ROOT = Path("中间结果/03_记忆矩阵/分析师共同覆盖")

OVERLAP_COLUMNS = [
    "Annodt",
    "Stkcd",
    "StkcdRecall",
    "same_analyst_overlap",
    "n_common_analysts",
    "lookback_days",
    "id_mode",
]


# =============================================================================
# 基础工具函数
# =============================================================================

def normalize_stock_code(x: object) -> str:
    """标准化股票代码，填充为 6 位数，并转换为字符串。"""
    if pd.isna(x):
        return ""

    s = str(x).strip()
    if s == "":
        return ""

    return f"{int(float(s)):06d}"


def resolve_project_root(root: str | Path | None) -> Path:
    """
    获取项目根目录。

    如果不传 --root，则默认使用当前运行目录。
    因此有两种推荐运行方式：
        1. cd 到 mmm复现数据包 后运行 python 本脚本.py；
        2. 在任意目录运行 python 本脚本.py --root "/你的/mmm复现数据包路径"。
    """
    if root is None:
        return Path.cwd().resolve()

    return Path(root).expanduser().resolve()


def require_file(path: Path, label: str, required: bool = True) -> bool:
    """
    检查文件是否存在。

    required=True 时，文件不存在就报错。
    required=False 时，文件不存在只提示并返回 False。
    """
    if path.exists():
        return True

    msg = (
        f"找不到{label}：{path}\n"
        f"请检查 --root 是否设置为项目根目录，或检查文件相对路径是否一致。"
    )

    if required:
        raise FileNotFoundError(msg)

    print(f"提示：{msg}")
    return False


def count_csv_data_rows(csv_path: Path) -> int:
    """
    快速统计 CSV 数据行数，不包含表头。

    用于在开始处理 Full 样本前估算总分块数。
    这里按二进制逐块读取换行符，避免把整个大 CSV 读进内存。
    """
    line_count = 0

    with csv_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            line_count += block.count(b"\n")

    # 减去表头行。若文件最后没有换行，也至少保证不返回负数。
    return max(line_count - 1, 0)


def ensure_output_dirs(output_root: Path) -> Tuple[Path, Path, Path]:
    """创建输出目录，并返回 split/team/compare 三个目录。"""
    output_split = output_root / "拆开AnanmID"
    output_team = output_root / "不拆开AnanmID"
    output_compare = output_root / "对比汇总"

    output_split.mkdir(parents=True, exist_ok=True)
    output_team.mkdir(parents=True, exist_ok=True)
    output_compare.mkdir(parents=True, exist_ok=True)

    return output_split, output_team, output_compare


def empty_overlap_result() -> pd.DataFrame:
    """返回固定列结构的空结果表。"""
    return pd.DataFrame(columns=OVERLAP_COLUMNS)


def make_pair_key_set(overlap_df: pd.DataFrame) -> set:
    """把 overlap 结果转成用于 split/team 对比的键集合。"""
    if overlap_df.empty:
        return set()

    return set(
        zip(
            overlap_df["Annodt"],
            overlap_df["Stkcd"],
            overlap_df["StkcdRecall"],
        )
    )


# =============================================================================
# 读取和清洗输入数据
# =============================================================================

def read_analyst_forecast(forecast_path: Path) -> pd.DataFrame:
    """
    读取并清洗分析师预测文件。

    保持原始逻辑：
    - Stkcd 标准化为 6 位；
    - InstitutionID 去掉末尾 .0；
    - Fenddt 重命名为 Accper；
    - 只保留 Accper <= 2025-12-31；
    - anndats = DeclareDate，如果 DeclareDate 缺失则用 Rptdt。
    """
    usecols = [
        "Stkcd",
        "InstitutionID",
        "AnanmID",
        "Ananm",
        "Fenddt",
        "Rptdt",
        "DeclareDate",
    ]

    df = pd.read_csv(
        forecast_path,
        encoding="utf-8",
        usecols=usecols,
        parse_dates=["Fenddt", "Rptdt", "DeclareDate"],
    )

    df["Stkcd"] = df["Stkcd"].apply(normalize_stock_code)

    df["InstitutionID"] = (
        df["InstitutionID"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df = df.rename(columns={"Fenddt": "Accper"})

    df = df.loc[
        df["Accper"] <= pd.Timestamp("2025-12-31")
    ].copy()

    df["anndats"] = df["DeclareDate"].fillna(df["Rptdt"])
    df = df.drop(columns=["Rptdt", "DeclareDate"])

    return df


def clean_memory_matrix(memory_df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗 MemoryMatrix。

    保持原始逻辑：
    - Stkcd 和 StkcdRecall 标准化为 6 位；
    - Annodt 转日期；
    - 按 Annodt, Stkcd, StkcdRecall 去重。
    """
    memory_df = memory_df.copy()

    memory_df["Stkcd"] = memory_df["Stkcd"].apply(normalize_stock_code)
    memory_df["StkcdRecall"] = memory_df["StkcdRecall"].apply(normalize_stock_code)
    memory_df["Annodt"] = pd.to_datetime(memory_df["Annodt"])

    memory_df = memory_df.drop_duplicates(
        ["Annodt", "Stkcd", "StkcdRecall"]
    ).copy()

    return memory_df


def read_memory_pattern(pattern_path: Path) -> pd.DataFrame:
    """读取并清洗 MemoryMatrixPattern。"""
    memory = pd.read_csv(
        pattern_path,
        encoding="utf-8",
        usecols=["Annodt", "Stkcd", "StkcdRecall"],
        parse_dates=["Annodt"],
    )

    return clean_memory_matrix(memory)


def iter_memory_full_chunks(
    full_path: Path,
    chunksize: int,
) -> Iterable[Tuple[int, pd.DataFrame]]:
    """
    分块读取并清洗 MemoryMatrixFull。

    注意：
    原始代码是在每个 chunk 内部 drop_duplicates，
    而不是对整个 Full 文件全局去重。
    这里保持这个逻辑不变。
    """
    reader = pd.read_csv(
        full_path,
        encoding="utf-8",
        usecols=["Annodt", "Stkcd", "StkcdRecall"],
        chunksize=chunksize,
    )

    for chunk_no, chunk in enumerate(reader, start=1):
        yield chunk_no, clean_memory_matrix(chunk)


# =============================================================================
# 构造分析师覆盖表
# =============================================================================

def build_team_coverage(af_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    构造不拆开 AnanmID 的覆盖表。
    对应原始代码中的 AF_Team。
    """
    af_team = af_forecast[
        ["Stkcd", "InstitutionID", "AnanmID", "Ananm", "anndats"]
    ].copy()

    af_team["analyst_key"] = (
        af_team["AnanmID"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
    )

    af_team = af_team.loc[
        (af_team["Stkcd"] != "")
        & (af_team["analyst_key"].notna())
        & (af_team["analyst_key"] != "")
        & (af_team["anndats"].notna())
    ].copy()

    af_team = af_team.drop_duplicates(["Stkcd", "analyst_key", "anndats"])
    af_team["id_mode"] = "team"

    return af_team


def build_split_coverage(af_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    构造拆开 AnanmID 的覆盖表。
    对应原始代码中的 AF_Split。
    """
    af_split = af_forecast[
        ["Stkcd", "InstitutionID", "AnanmID", "Ananm", "anndats"]
    ].copy()

    af_split["analyst_key"] = (
        af_split["AnanmID"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.split(",")
    )

    af_split = af_split.explode("analyst_key")

    af_split["analyst_key"] = (
        af_split["analyst_key"]
        .astype("string")
        .str.strip()
    )

    af_split = af_split.loc[
        (af_split["Stkcd"] != "")
        & (af_split["analyst_key"].notna())
        & (af_split["analyst_key"] != "")
        & (af_split["anndats"].notna())
    ].copy()

    af_split = af_split.drop_duplicates(["Stkcd", "analyst_key", "anndats"])
    af_split["id_mode"] = "split"

    return af_split


def build_fast_coverage_index(
    coverage_df: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    把覆盖表转换成可快速查询的索引。

    原始逻辑每次查询都做：
        coverage_df.loc[(anndats >= start_date) & (anndats < annodt)]

    优化后：
        1. 每只股票的 anndats 预先排序；
        2. 查询时用 np.searchsorted 快速定位日期窗口；
        3. 再取对应 analyst_key 集合。
    """
    coverage_index: Dict[str, Dict[str, np.ndarray]] = {}

    if coverage_df.empty:
        return coverage_index

    sorted_df = coverage_df.sort_values(["Stkcd", "anndats"], kind="mergesort")

    for stkcd, sub_df in sorted_df.groupby("Stkcd", sort=False):
        dates_ns = (
            sub_df["anndats"]
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
        )
        analysts = sub_df["analyst_key"].astype("string").to_numpy(dtype=object)

        coverage_index[stkcd] = {
            "dates_ns": dates_ns,
            "analysts": analysts,
        }

    return coverage_index


# =============================================================================
# 快速窗口查询和共同覆盖识别
# =============================================================================

def get_analyst_set_fast(
    coverage_index: Dict[str, Dict[str, np.ndarray]],
    stkcd: str,
    annodt_ns: int,
    lookback_days: int,
    cache: Dict[Tuple[str, int], frozenset],
) -> frozenset:
    """
    获取某股票在 [Annodt - lookback_days, Annodt) 内的分析师集合。

    与原始函数 get_analyst_set 逻辑等价：
        start_date = annodt - datetime.timedelta(days=lookback_days)
        tmp = coverage_df.loc[
            (coverage_df["anndats"] >= start_date)
            & (coverage_df["anndats"] < annodt)
        ]
        return set(tmp["analyst_key"])

    这里的 cache 是针对某一个 lookback_days、某一个 id_mode 的。
    所以 key 只需要放 stkcd 和 annodt_ns。
    """
    key = (stkcd, int(annodt_ns))

    cached = cache.get(key)
    if cached is not None:
        return cached

    stock_data = coverage_index.get(stkcd)

    if stock_data is None:
        result = frozenset()
        cache[key] = result
        return result

    dates_ns = stock_data["dates_ns"]
    analysts = stock_data["analysts"]

    start_ns = int(annodt_ns) - int(lookback_days) * NS_PER_DAY

    left = np.searchsorted(dates_ns, start_ns, side="left")
    right = np.searchsorted(dates_ns, int(annodt_ns), side="left")

    if right <= left:
        result = frozenset()
    else:
        result = frozenset(analysts[left:right])

    cache[key] = result
    return result


def identify_common_analyst_overlap_fast(
    memory_df: pd.DataFrame,
    coverage_index: Dict[str, Dict[str, np.ndarray]],
    lookback_days: int,
    id_mode: str,
    cache: Dict[Tuple[str, int], frozenset] | None = None,
    show_progress: bool = True,
    desc: str | None = None,
    ncols: int = 100,
    mininterval: float = 2.0,
) -> pd.DataFrame:
    """
    识别共同分析师覆盖。

    保持原始输出逻辑：
    - 只保存存在共同分析师的公司对；
    - same_analyst_overlap = 1；
    - n_common_analysts = 共同分析师数量；
    - lookback_days 和 id_mode 原样记录。
    """
    if cache is None:
        cache = {}

    if memory_df.empty:
        return empty_overlap_result()

    annodt_ns_array = (
        memory_df["Annodt"]
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )
    stkcd_array = memory_df["Stkcd"].to_numpy(dtype=object)
    recall_array = memory_df["StkcdRecall"].to_numpy(dtype=object)

    iterator = zip(annodt_ns_array, stkcd_array, recall_array)

    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(memory_df),
            desc=desc or f"{id_mode}, rolling {lookback_days}d",
            ncols=ncols,
            mininterval=mininterval,
        )

    result_list: List[dict] = []

    for annodt_ns, stkcd, stkcd_recall in iterator:
        cue_analysts = get_analyst_set_fast(
            coverage_index=coverage_index,
            stkcd=stkcd,
            annodt_ns=int(annodt_ns),
            lookback_days=lookback_days,
            cache=cache,
        )

        # 如果 cue firm 窗口内没有分析师，不可能有共同分析师。
        # 这只是提前跳过，不改变判断结果。
        if not cue_analysts:
            continue

        recall_analysts = get_analyst_set_fast(
            coverage_index=coverage_index,
            stkcd=stkcd_recall,
            annodt_ns=int(annodt_ns),
            lookback_days=lookback_days,
            cache=cache,
        )

        if not recall_analysts:
            continue

        common_analysts = cue_analysts & recall_analysts

        if len(common_analysts) > 0:
            result_list.append(
                {
                    "Annodt": pd.Timestamp(int(annodt_ns)),
                    "Stkcd": stkcd,
                    "StkcdRecall": stkcd_recall,
                    "same_analyst_overlap": 1,
                    "n_common_analysts": len(common_analysts),
                    "lookback_days": lookback_days,
                    "id_mode": id_mode,
                }
            )

    if not result_list:
        return empty_overlap_result()

    return pd.DataFrame(result_list, columns=OVERLAP_COLUMNS)


# =============================================================================
# 汇总和输出
# =============================================================================

def summarize_overlap_comparison(
    lookback_days: int,
    matrix_rows_checked: int,
    split_keys: set,
    team_keys: set,
) -> dict:
    """汇总 split/team 的 overlap 差异。"""
    both_keys = split_keys & team_keys
    split_only_keys = split_keys - team_keys
    team_only_keys = team_keys - split_keys

    return {
        "lookback_days": lookback_days,
        "matrix_rows_checked": matrix_rows_checked,
        "split_overlap_rows": len(split_keys),
        "team_overlap_rows": len(team_keys),
        "both_overlap_rows": len(both_keys),
        "split_only_rows": len(split_only_keys),
        "team_only_rows": len(team_only_keys),
        "split_only_share_among_split": (
            len(split_only_keys) / len(split_keys)
            if len(split_keys) > 0
            else np.nan
        ),
        "team_only_share_among_team": (
            len(team_only_keys) / len(team_keys)
            if len(team_keys) > 0
            else np.nan
        ),
    }


def read_overlap_keys_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    从已经写出的 overlap CSV 中读取三列 key，并做全局去重。

    这样 Full 主循环中不再需要长期维护巨大的 split_keys/team_keys set，
    可以明显降低内存压力。因为 overlap 输出文件相对 Full 输入小很多，
    最后再读这些输出文件来做汇总更划算。
    """
    key_cols = ["Annodt", "Stkcd", "StkcdRecall"]

    if (not csv_path.exists()) or csv_path.stat().st_size == 0:
        return pd.DataFrame(columns=key_cols)

    keys = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        usecols=key_cols,
        parse_dates=["Annodt"],
    )

    keys["Stkcd"] = keys["Stkcd"].apply(normalize_stock_code)
    keys["StkcdRecall"] = keys["StkcdRecall"].apply(normalize_stock_code)
    keys["Annodt"] = pd.to_datetime(keys["Annodt"])

    keys = keys.drop_duplicates(key_cols).copy()

    return keys


def summarize_overlap_from_output_files(
    lookback_days: int,
    matrix_rows_checked: int,
    split_file: Path,
    team_file: Path,
) -> dict:
    """
    从已经写出的 split/team overlap 文件计算对比汇总。

    等价于原来维护 split_keys/team_keys 的结果，
    但不会在 Full 主循环中一直占用大量内存。
    """
    key_cols = ["Annodt", "Stkcd", "StkcdRecall"]

    split_keys_df = read_overlap_keys_from_csv(split_file)
    team_keys_df = read_overlap_keys_from_csv(team_file)

    split_n = len(split_keys_df)
    team_n = len(team_keys_df)

    if split_n > 0 and team_n > 0:
        both_n = len(
            split_keys_df.merge(
                team_keys_df,
                on=key_cols,
                how="inner",
            )
        )
    else:
        both_n = 0

    split_only_n = split_n - both_n
    team_only_n = team_n - both_n

    return {
        "lookback_days": lookback_days,
        "matrix_rows_checked": matrix_rows_checked,
        "split_overlap_rows": split_n,
        "team_overlap_rows": team_n,
        "both_overlap_rows": both_n,
        "split_only_rows": split_only_n,
        "team_only_rows": team_only_n,
        "split_only_share_among_split": (
            split_only_n / split_n if split_n > 0 else np.nan
        ),
        "team_only_share_among_team": (
            team_only_n / team_n if team_n > 0 else np.nan
        ),
    }


def write_overlap_chunk(
    overlap_df: pd.DataFrame,
    output_path: Path,
    first_write: bool,
) -> None:
    """写出 overlap 结果；Full 样本分块时会追加写入。"""
    overlap_df.to_csv(
        output_path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# Pattern 样本
# =============================================================================

def run_pattern_sample(
    pattern_path: Path,
    output_split: Path,
    output_team: Path,
    output_compare: Path,
    split_index: Dict[str, Dict[str, np.ndarray]],
    team_index: Dict[str, Dict[str, np.ndarray]],
    lookback_days_list: Iterable[int],
    show_progress: bool,
    ncols: int,
    mininterval: float,
) -> pd.DataFrame:
    """运行 Pattern 样本分析。"""
    print("\n========== 开始处理 Pattern 样本 ==========")

    memory_pattern = read_memory_pattern(pattern_path)
    matrix_rows_checked = len(memory_pattern)

    summary_list: List[dict] = []

    for lookback_days in lookback_days_list:
        print(f"\nPattern样本：正在识别拆开 AnanmID 口径，窗口 = {lookback_days} 天")
        split_cache: Dict[Tuple[str, int], frozenset] = {}
        overlap_split = identify_common_analyst_overlap_fast(
            memory_df=memory_pattern,
            coverage_index=split_index,
            lookback_days=lookback_days,
            id_mode="split",
            cache=split_cache,
            show_progress=show_progress,
            desc=f"Pattern split {lookback_days}d",
            ncols=ncols,
            mininterval=mininterval,
        )

        print(f"Pattern样本：正在识别不拆开 AnanmID 口径，窗口 = {lookback_days} 天")
        team_cache: Dict[Tuple[str, int], frozenset] = {}
        overlap_team = identify_common_analyst_overlap_fast(
            memory_df=memory_pattern,
            coverage_index=team_index,
            lookback_days=lookback_days,
            id_mode="team",
            cache=team_cache,
            show_progress=show_progress,
            desc=f"Pattern team {lookback_days}d",
            ncols=ncols,
            mininterval=mininterval,
        )

        split_file = output_split / f"Pattern样本_分析师共同覆盖_滚动{lookback_days}天.csv"
        team_file = output_team / f"Pattern样本_分析师共同覆盖_滚动{lookback_days}天.csv"

        write_overlap_chunk(overlap_split, split_file, first_write=True)
        write_overlap_chunk(overlap_team, team_file, first_write=True)

        split_keys = make_pair_key_set(overlap_split)
        team_keys = make_pair_key_set(overlap_team)

        summary_list.append(
            summarize_overlap_comparison(
                lookback_days=lookback_days,
                matrix_rows_checked=matrix_rows_checked,
                split_keys=split_keys,
                team_keys=team_keys,
            )
        )

    summary = pd.DataFrame(summary_list)

    summary_path = output_compare / "Pattern样本_分析师共同覆盖_拆开与不拆开AnanmID对比汇总.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nPattern样本处理完成。汇总如下：")
    print(summary)
    print(f"Pattern样本汇总已保存：{summary_path}")

    return summary


# =============================================================================
# Full 样本
# =============================================================================

def run_full_sample(
    full_path: Path,
    output_split: Path,
    output_team: Path,
    output_compare: Path,
    split_index: Dict[str, Dict[str, np.ndarray]],
    team_index: Dict[str, Dict[str, np.ndarray]],
    lookback_days_list: List[int],
    chunksize: int,
    show_progress: bool,
    ncols: int,
    mininterval: float,
) -> pd.DataFrame:
    """
    运行 Full 样本分析。

    v4 输出优化：
    - 默认不显示 tqdm 进度条；
    - 每个 chunk 只打印简洁状态；
    - 打印 Annodt 日期范围；
    - 打印每个窗口/口径的耗时和 overlap 行数；
    - 打印每个 chunk 的总耗时。
    """
    print("========== 开始处理 Full 样本 ==========")

    total_rows_estimated = count_csv_data_rows(full_path)
    total_chunks_estimated = int(np.ceil(total_rows_estimated / chunksize)) if chunksize > 0 else np.nan

    print(f"Full样本预计总行数：{total_rows_estimated:,}")
    print(f"Full样本 chunksize：{chunksize:,}")
    print(f"Full样本预计分块数：{total_chunks_estimated:,}")

    matrix_rows_checked = 0
    first_write = {lb: True for lb in lookback_days_list}

    full_start_time = time.perf_counter()

    for chunk_no, memory_full_chunk in iter_memory_full_chunks(full_path, chunksize):
        chunk_start_time = time.perf_counter()

        chunk_rows = len(memory_full_chunk)
        matrix_rows_checked += chunk_rows

        if chunk_rows > 0:
            chunk_start_date = memory_full_chunk["Annodt"].min().date()
            chunk_end_date = memory_full_chunk["Annodt"].max().date()
            date_msg = f"Annodt = {chunk_start_date} 至 {chunk_end_date}"
        else:
            date_msg = "Annodt = 空分块"

        print(
            f"Full样本：第 {chunk_no}/{total_chunks_estimated} 块，"
            f"行数 = {chunk_rows:,}，{date_msg}"
        )

        # 缓存只在当前 chunk 内保留，避免 Full 样本下 cache 无限增长。
        split_cache_by_window = {lb: {} for lb in lookback_days_list}
        team_cache_by_window = {lb: {} for lb in lookback_days_list}

        for lookback_days in lookback_days_list:
            step_start_time = time.perf_counter()

            overlap_split = identify_common_analyst_overlap_fast(
                memory_df=memory_full_chunk,
                coverage_index=split_index,
                lookback_days=lookback_days,
                id_mode="split",
                cache=split_cache_by_window[lookback_days],
                show_progress=show_progress,
                desc=f"Full chunk {chunk_no} split {lookback_days}d",
                ncols=ncols,
                mininterval=mininterval,
            )

            split_elapsed = time.perf_counter() - step_start_time

            print(
                f"  {lookback_days}天 split 完成，"
                f"用时 {split_elapsed:.1f} 秒，"
                f"overlap = {len(overlap_split):,} 行"
            )

            step_start_time = time.perf_counter()

            overlap_team = identify_common_analyst_overlap_fast(
                memory_df=memory_full_chunk,
                coverage_index=team_index,
                lookback_days=lookback_days,
                id_mode="team",
                cache=team_cache_by_window[lookback_days],
                show_progress=show_progress,
                desc=f"Full chunk {chunk_no} team {lookback_days}d",
                ncols=ncols,
                mininterval=mininterval,
            )

            team_elapsed = time.perf_counter() - step_start_time

            print(
                f"  {lookback_days}天 team  完成，"
                f"用时 {team_elapsed:.1f} 秒，"
                f"overlap = {len(overlap_team):,} 行"
            )

            split_file = output_split / f"Full样本_分析师共同覆盖_滚动{lookback_days}天.csv"
            team_file = output_team / f"Full样本_分析师共同覆盖_滚动{lookback_days}天.csv"

            write_overlap_chunk(
                overlap_df=overlap_split,
                output_path=split_file,
                first_write=first_write[lookback_days],
            )

            write_overlap_chunk(
                overlap_df=overlap_team,
                output_path=team_file,
                first_write=first_write[lookback_days],
            )

            first_write[lookback_days] = False

        chunk_elapsed = time.perf_counter() - chunk_start_time
        total_elapsed = time.perf_counter() - full_start_time

        print(
            f"第 {chunk_no}/{total_chunks_estimated} 块完成，"
            f"本块总用时 {chunk_elapsed:.1f} 秒，"
            f"累计用时 {total_elapsed / 60:.1f} 分钟"
        )

        # 显式释放当前 chunk 对象引用，帮助长任务减少内存峰值。
        del memory_full_chunk

    print("Full样本主循环完成，正在基于已写出的 overlap 文件计算汇总...")

    summary_list = []

    for lookback_days in lookback_days_list:
        split_file = output_split / f"Full样本_分析师共同覆盖_滚动{lookback_days}天.csv"
        team_file = output_team / f"Full样本_分析师共同覆盖_滚动{lookback_days}天.csv"

        summary_list.append(
            summarize_overlap_from_output_files(
                lookback_days=lookback_days,
                matrix_rows_checked=matrix_rows_checked,
                split_file=split_file,
                team_file=team_file,
            )
        )

    summary = pd.DataFrame(summary_list)

    summary_path = output_compare / "Full样本_分析师共同覆盖_拆开与不拆开AnanmID对比汇总.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("Full样本处理完成。汇总如下：")
    print(summary)
    print(f"Full样本汇总已保存：{summary_path}")

    return summary


# =============================================================================
# 主流程
# =============================================================================

def run_analysis(
    root: Path,
    lookback_days_list: List[int],
    chunksize: int,
    run_pattern: bool,
    run_full: bool,
    show_progress: bool,
    ncols: int,
    mininterval: float,
) -> None:
    """完整运行分析流程。"""
    forecast_path = root / FORECAST_RELATIVE_PATH
    pattern_path = root / PATTERN_RELATIVE_PATH
    full_path = root / FULL_RELATIVE_PATH
    output_root = root / OUTPUT_RELATIVE_ROOT

    require_file(forecast_path, "分析师预测文件", required=True)

    if run_pattern:
        require_file(pattern_path, "MemoryMatrixPattern 文件", required=True)

    if run_full:
        require_file(full_path, "MemoryMatrixFull 文件", required=True)

    output_split, output_team, output_compare = ensure_output_dirs(output_root)

    print(f"项目根目录：{root}")
    print(f"分析师预测文件：{forecast_path}")
    print(f"输出目录：{output_root}")
    print(f"lookback_days：{lookback_days_list}")
    print(f"chunksize：{chunksize}")

    print("\n正在读取分析师预测文件...")
    af_forecast = read_analyst_forecast(forecast_path)

    print("正在构造不拆开 AnanmID 的覆盖表...")
    af_team = build_team_coverage(af_forecast)

    print("正在构造拆开 AnanmID 的覆盖表...")
    af_split = build_split_coverage(af_forecast)

    print("正在构造快速查询索引...")
    team_index = build_fast_coverage_index(af_team)
    split_index = build_fast_coverage_index(af_split)

    print(f"不拆开 AnanmID 覆盖股票数：{len(team_index)}")
    print(f"拆开 AnanmID 覆盖股票数：{len(split_index)}")

    if run_pattern:
        run_pattern_sample(
            pattern_path=pattern_path,
            output_split=output_split,
            output_team=output_team,
            output_compare=output_compare,
            split_index=split_index,
            team_index=team_index,
            lookback_days_list=lookback_days_list,
            show_progress=show_progress,
            ncols=ncols,
            mininterval=mininterval,
        )

    if run_full:
        run_full_sample(
            full_path=full_path,
            output_split=output_split,
            output_team=output_team,
            output_compare=output_compare,
            split_index=split_index,
            team_index=team_index,
            lookback_days_list=lookback_days_list,
            chunksize=chunksize,
            show_progress=show_progress,
            ncols=ncols,
            mininterval=mininterval,
        )

    print("\n全部处理完成。")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="识别 Pattern 样本和 Full 样本中的分析师共同覆盖。"
    )

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "项目根目录，例如：/Users/qiyuewu/Desktop/mmm复现数据包。"
            "如果不填，则默认使用当前运行目录。"
        ),
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        nargs="+",
        default=[45, 365],
        help="向前滚动窗口天数，默认：45 365。",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=10000000,
        help="Full 样本分块读取行数，默认 10000000。",
    )

    parser.add_argument(
        "--only-pattern",
        action="store_true",
        help="只运行 Pattern 样本。",
    )

    parser.add_argument(
        "--only-full",
        action="store_true",
        help="只运行 Full 样本。",
    )

    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="显示 tqdm 进度条。默认不显示，只打印简洁状态。",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="兼容旧命令；现在默认就不显示 tqdm 进度条。",
    )

    parser.add_argument(
        "--ncols",
        type=int,
        default=100,
        help="tqdm 进度条宽度，默认 100。",
    )

    parser.add_argument(
        "--mininterval",
        type=float,
        default=2.0,
        help="tqdm 最小刷新间隔秒数，默认 2 秒。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = resolve_project_root(args.root)

    if args.only_pattern and args.only_full:
        raise ValueError("--only-pattern 和 --only-full 不能同时使用。")

    run_pattern = not args.only_full
    run_full = not args.only_pattern

    try:
        run_analysis(
            root=root,
            lookback_days_list=args.lookback_days,
            chunksize=args.chunksize,
            run_pattern=run_pattern,
            run_full=run_full,
            show_progress=args.show_progress,
            ncols=args.ncols,
            mininterval=args.mininterval,
        )
    except Exception as exc:
        print(f"\n程序运行失败：{exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
