"""
按 Annodt 日度口径识别分析师共同覆盖。

这个脚本只对 MemoryMatrix 中实际出现的公告日和 cue-recall 公司对进行识别：

1. 对每个 Annodt，取 [Annodt - lookback_days, Annodt) 内的分析师覆盖；
2. 为该 Annodt 下 MemoryMatrix 涉及的股票构造活跃 analyst_key 集合；
3. 对每条 cue firm - recall firm 计算两个集合的交集；
4. 交集非空则输出 same_analyst_overlap = 1 和 n_common_analysts。

该方法等价于对每条 MemoryMatrix 观测逐条执行滚动窗口查询，但按 Annodt
批量组织计算，更容易在论文中表述为“在公告日构造日度分析师共同覆盖网络，
再合并至 cue-recall 公司对”。窗口右端不包含 Annodt 当天，避免使用公告日
当天及之后的信息。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - 未安装tqdm时不显示进度条。
    tqdm = None


NS_PER_DAY = 24 * 60 * 60 * 1_000_000_000 # 计算1天多少纳秒

FORECAST_RELATIVE_PATH = Path("中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv")
PATTERN_RELATIVE_PATH = Path("中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv")
FULL_RELATIVE_PATH = Path("中间结果/03_记忆矩阵/MemoryMatrixFull(封装优化版).csv")
DEFAULT_OUTPUT_RELATIVE_ROOT = Path("中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后")

OVERLAP_COLUMNS = [
    "Annodt",
    "Stkcd",
    "StkcdRecall",
    "same_analyst_overlap", # 是否有共同分析师覆盖
    "n_common_analysts", # 共同分析师的数量
    "lookback_days", # 向前看的窗口
    "id_mode", # 是否拆开分析的ID
]# 输出结果的列名

ID_MODE_DIR = {
    "split": "拆开AnanmID",
    "team": "不拆开AnanmID",
}

SAMPLE_LABEL = {
    "pattern": "Pattern样本",
    "full": "Full样本",
}


def normalize_stock_code(x: object) -> str:
    """
    将 CSMAR 股票代码统一标准化为 6 位字符串。

    原始 CSV 中的股票代码可能被读成整数、浮点数或字符串，例如 1、1.0、
    "000001"。后续与 MemoryMatrix、收益率或其它中间结果合并时，必须
    使用统一的 6 位字符串格式，否则同一股票可能无法匹配。
    """
    if pd.isna(x):
        return ""

    s = str(x).strip()
    if s == "":
        return ""

    return f"{int(float(s)):06d}"


def resolve_project_root(root: str | Path | None) -> Path:
    """解析项目根目录；未指定时使用当前运行目录。"""
    if root is None:
        return Path.cwd().resolve()

    return Path(root).expanduser().resolve()


def require_file(path: Path, label: str) -> None:
    """检查必要输入文件是否存在，缺失时抛出带路径的错误。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到{label}：{path}")


def read_analyst_forecast(forecast_path: Path) -> pd.DataFrame:
    """
    读取并清洗 CSMAR 分析师预测合并文件。

    关键口径：
    - Stkcd 标准化为 6 位字符串；
    - InstitutionID 去掉 CSV 读入后可能出现的 .0 后缀；
    - 只保留发布日期不晚于 2025-12-31 的报告
    - 分析师覆盖日期 anndats 优先使用 DeclareDate，缺失时使用 Rptdt。该处理遵循既有文献中以研究报告发布作为分析师信息进入市场时点的做法。


    返回的 DataFrame 仍保留 AnanmID 和 Ananm，后续由 split/team 两套函数
    分别构造 analyst_key。
    """
    usecols = [
        "Stkcd",
        "InstitutionID",
        "AnanmID",
        "Ananm",
        "Rptdt",
        "DeclareDate",
    ]

    df = pd.read_csv(
        forecast_path,
        encoding="utf-8",
        usecols=usecols,
        parse_dates=["Rptdt", "DeclareDate"],
    )

    df["Stkcd"] = df["Stkcd"].apply(normalize_stock_code)
    df["InstitutionID"] = (
        df["InstitutionID"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["anndats"] = df["DeclareDate"].fillna(df["Rptdt"])
    df = df.loc[
        df["anndats"].notna()
        & (df["anndats"] <= pd.Timestamp("2025-12-31"))
    ].copy()

    return df.drop(columns=["Rptdt", "DeclareDate"])


def normalize_team_key(x: object):
    """将联合署名团队标准化为与成员排列顺序无关的键。"""
    if pd.isna(x):
        return pd.NA

    members = {
        member.strip()
        for member in str(x).split(",")
        if member.strip()
    }

    if not members:
        return pd.NA

    # 按字符串排序，避免超长分析师编号转换为整数时出现精度或溢出问题
    return ",".join(sorted(members))



def build_team_coverage(af_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    构造“不拆开 AnanmID”的分析师覆盖表。

    CSMAR 中一条预测记录可能包含多个分析师 ID，以逗号分隔。team 口径
    不拆开这些 ID，而是把完整的 AnanmID 字符串视为一个 analyst_key。
    因此只有两家公司被同一个完整署名团队覆盖时，才会被识别为共同覆盖。

    输出列中最重要的是 Stkcd、analyst_key、anndats；同一股票、同一
    analyst_key、同一发布日期的重复记录会被去重。
    """
    af_team = af_forecast[["Stkcd", "InstitutionID", "AnanmID", "Ananm", "anndats"]].copy()
    af_team["analyst_key"] = (
        af_team["AnanmID"]
        .map(normalize_team_key)
        .astype("string")
    )

    af_team = af_team.loc[
        (af_team["Stkcd"] != "")
        & (af_team["analyst_key"].notna())
        & (af_team["analyst_key"] != "")
        & (af_team["anndats"].notna())
    ].copy()

    return af_team.drop_duplicates(["Stkcd", "analyst_key", "anndats"])


def build_split_coverage(af_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    构造“拆开 AnanmID”的分析师覆盖表。

    split 口径会把逗号分隔的 AnanmID 拆成多个个人 analyst_key。只要同一
    个人分析师在窗口内分别覆盖 cue firm 和 recall firm，就认为存在共同覆盖。
    这是更接近“个人分析师共同覆盖”的主口径，也通常会识别出比 team 口径
    更多的共同覆盖公司对。
    """
    af_split = af_forecast[["Stkcd", "InstitutionID", "AnanmID", "Ananm", "anndats"]].copy()
    af_split["analyst_key"] = (
        af_split["AnanmID"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.split(",")
    )
    af_split = af_split.explode("analyst_key")
    af_split["analyst_key"] = af_split["analyst_key"].astype("string").str.strip()

    af_split = af_split.loc[
        (af_split["Stkcd"] != "")
        & (af_split["analyst_key"].notna())
        & (af_split["analyst_key"] != "")
        & (af_split["anndats"].notna())
    ].copy()

    return af_split.drop_duplicates(["Stkcd", "analyst_key", "anndats"])


def build_fast_coverage_index(coverage_df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """
    将覆盖表转换为按股票分组的快速查询索引。

    对每只股票保存两个等长数组：
    - dates_ns：该股票被分析师覆盖的发布日期，转为纳秒整数并升序排列；
    - analysts：对应日期上的 analyst_key。

    后续查询某股票在 [Annodt - lookback_days, Annodt) 内的分析师集合时，
    可以用 np.searchsorted 在排序日期数组上直接定位左右边界，避免每条
    MemoryMatrix 观测都对完整覆盖表做 DataFrame 条件筛选。
    """
    coverage_index: Dict[str, Dict[str, np.ndarray]] = {}
    if coverage_df.empty:
        return coverage_index

    sorted_df = coverage_df.sort_values(["Stkcd", "anndats"], kind="mergesort")

    for stkcd, sub_df in sorted_df.groupby("Stkcd", sort=False): 
    # sub_df["anndats"]已经是严格升序排列的了，不需要再做任何排序！
    # 对每只股票，构造两个等长、严格对应的 numpy 数组：
        coverage_index[stkcd] = {
            "dates_ns": sub_df["anndats"].to_numpy(dtype="datetime64[ns]").astype("int64"),# 日期数组（纳秒整数）
            "analysts": sub_df["analyst_key"].astype("string").to_numpy(dtype=object),
        }# 分析师 ID 数组

    return coverage_index # 返回索引

# 当我们用np.searchsorted找到日期的左右边界left和right后
# analysts[left:right]就正好是窗口内所有分析师的 ID 数组



def get_analyst_set_fast(
    coverage_index: Dict[str, Dict[str, np.ndarray]],
    stkcd: str,
    annodt_ns: int,
    lookback_days: int,
    cache: Dict[Tuple[str, int], frozenset],
) -> frozenset: # frozenset不可变集合可以安全地作为字典的键，并且比普通set更轻量、更快
    """
    获取某股票在公告日前滚动窗口内的分析师集合。

    窗口定义为 [Annodt - lookback_days, Annodt)，左闭右开：
    - 包含窗口起点当天的覆盖记录；
    - 不包含 Annodt 当天的覆盖记录，避免公告日当天报告带来的前视偏误。

    cache 的键为 (Stkcd, Annodt_ns)。同一公告日下同一股票可能出现在大量
    cue-recall 公司对中，缓存可以避免重复做 searchsorted 和集合构造。
    """
    key = (stkcd, int(annodt_ns)) # 用元组作为字典的键，唯一标识一个 (股票，公告日) 组合
    cached = cache.get(key) 
# 先查缓存，这个cache字典是在identify_overlap_by_annodt_daily_sets函数中创建的，
# 每个窗口长度、每个数据块都有自己独立的缓存。处理完一个块后，缓存会被自动销毁，不会占用过多内存。
    if cached is not None:
        return cached
    stock_data = coverage_index.get(stkcd)

# 第二步：处理股票不存在的情况
# 如果索引中没有这只股票（说明它从来没有被任何分析师覆盖过）
    if stock_data is None:
        result = frozenset()
        cache[key] = result
        return result

# 取出股票的两个核心数组
    dates_ns = stock_data["dates_ns"]
    analysts = stock_data["analysts"]
    start_ns = int(annodt_ns) - int(lookback_days) * NS_PER_DAY # 计算左边界，公告日- 回溯窗口长度

    left = np.searchsorted(dates_ns, start_ns, side="left") 
    right = np.searchsorted(dates_ns, int(annodt_ns), side="left") # 查找左右边界的索引，左闭右开
    result = frozenset(analysts[left:right]) if right > left else frozenset() 
    # 切片并转成不可变集合
    cache[key] = result # 存入缓存并返回结果

    return result


def clean_memory_matrix(memory_df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗 MemoryMatrix 中用于共同覆盖识别的三列。

    只保留识别共同覆盖所需的 Annodt、Stkcd、StkcdRecall 逻辑：
    - Stkcd 和 StkcdRecall 标准化为 6 位字符串；
    - Annodt 转为 pandas 日期；
    - 按 Annodt + Stkcd + StkcdRecall 去重。

    Full 样本分块处理时，去重发生在每个 chunk 内。
    """
    memory_df = memory_df.copy()
    memory_df["Stkcd"] = memory_df["Stkcd"].apply(normalize_stock_code)
    memory_df["StkcdRecall"] = memory_df["StkcdRecall"].apply(normalize_stock_code)
    memory_df["Annodt"] = pd.to_datetime(memory_df["Annodt"])

    return memory_df.drop_duplicates(["Annodt", "Stkcd", "StkcdRecall"]).copy()


def read_memory_file(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    """
    一次性读取 Pattern 或小规模调试样本。

    max_rows 仅用于调试或抽样验证；正式生成结果时不应传入该参数。
    """
    read_kwargs = {
        "encoding": "utf-8",
        "usecols": ["Annodt", "Stkcd", "StkcdRecall"], # 只读取需要的三列数据
        "parse_dates": ["Annodt"],
    }
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows

    return clean_memory_matrix(pd.read_csv(path, **read_kwargs))


def iter_memory_chunks(
    path: Path,
    chunksize: int,
    max_rows: int | None = None,
) -> Iterable[Tuple[int, pd.DataFrame]]:
    """
    分块读取 Full 样本，避免一次性加载超大 MemoryMatrix。

    每个 chunk 都会调用 clean_memory_matrix，因此 chunk 内重复的
    Annodt-Stkcd-StkcdRecall 会被去除。
    """
    remaining = max_rows
    reader = pd.read_csv(
        path,
        encoding="utf-8",
        usecols=["Annodt", "Stkcd", "StkcdRecall"],
        parse_dates=["Annodt"],
        chunksize=chunksize,
    )

    for chunk_no, chunk in enumerate(reader, start=1):
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
            remaining -= len(chunk)

        yield chunk_no, clean_memory_matrix(chunk)


def progress_iter(iterable, total: int | None, desc: str, show_progress: bool):
    """
    根据参数包装 tqdm 进度条。

    如果环境没有安装 tqdm，则自动退化为普通迭代器，不影响脚本运行。
    """
    if show_progress and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, ncols=100, mininterval=2.0)
    return iterable


def identify_overlap_by_annodt_daily_sets(
    memory_df: pd.DataFrame,
    coverage_index: Dict[str, Dict[str, np.ndarray]],
    lookback_days: int,
    id_mode: str,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    按 Annodt 分组构造当日股票-分析师集合，再回填到 MemoryMatrix 公司对。

    对同一公告日 Annodt：
    1. 找出该日 MemoryMatrix 中所有出现过的股票，包括 cue firm 和 recall firm；
    2. 为这些股票分别查询 [Annodt - lookback_days, Annodt) 窗口内的 analyst_key 集合；
    3. 对该日每一条 cue-recall 公司对，计算两个集合的交集；
    4. 交集非空则输出一条共同覆盖结果。

    """
    if memory_df.empty:
        return pd.DataFrame(columns=OVERLAP_COLUMNS)

    result_list: List[dict] = []
    grouped = memory_df.groupby("Annodt", sort=False)
    iterator = progress_iter(
        grouped,
        total=memory_df["Annodt"].nunique(),
        desc=f"{id_mode} daily Annodt {lookback_days}d",
        show_progress=show_progress,
    )

    cache: Dict[Tuple[str, int], frozenset] = {}

    for annodt, day_df in iterator:
        annodt_ns = int(pd.Timestamp(annodt).to_datetime64().astype("datetime64[ns]").astype("int64"))
        needed_stocks = pd.unique(
            pd.concat([day_df["Stkcd"], day_df["StkcdRecall"]], ignore_index=True)
        )
        analyst_sets = {
            stkcd: get_analyst_set_fast(
                coverage_index=coverage_index,
                stkcd=stkcd,
                annodt_ns=annodt_ns,
                lookback_days=lookback_days,
                cache=cache,
            )
            for stkcd in needed_stocks
        }

        for row in day_df.itertuples(index=False):
            cue_analysts = analyst_sets.get(row.Stkcd, frozenset())
            if not cue_analysts:
                continue

            recall_analysts = analyst_sets.get(row.StkcdRecall, frozenset())
            if not recall_analysts:
                continue

            common_analysts = cue_analysts & recall_analysts
            if common_analysts:
                result_list.append(
                    {
                        "Annodt": pd.Timestamp(annodt),
                        "Stkcd": row.Stkcd,
                        "StkcdRecall": row.StkcdRecall,
                        "same_analyst_overlap": 1,
                        "n_common_analysts": len(common_analysts),
                        "lookback_days": lookback_days,
                        "id_mode": id_mode,
                    }
                )

    if not result_list:
        return pd.DataFrame(columns=OVERLAP_COLUMNS)

    return pd.DataFrame(result_list, columns=OVERLAP_COLUMNS)


def output_file_for(root: Path, sample: str, id_mode: str, lookback_days: int) -> Path:
    """根据样本、ID 口径和窗口长度生成输出文件路径。"""
    return (
        root
        / ID_MODE_DIR[id_mode]
        / f"{SAMPLE_LABEL[sample]}_分析师共同覆盖_滚动{lookback_days}天.csv"
    )


def ensure_output_dirs(output_root: Path) -> None:
    """创建 split/team 两类输出目录。"""
    for dirname in ID_MODE_DIR.values():
        (output_root / dirname).mkdir(parents=True, exist_ok=True)


def write_overlap(overlap_df: pd.DataFrame, output_path: Path, first_write: bool) -> None:
    """
    写出共同覆盖结果。

    Pattern 样本通常一次写完；Full 样本按 chunk 追加写入。first_write=True
    时覆盖旧文件并写入表头，之后的 chunk 使用追加模式且不重复写表头。
    """
    overlap_df.to_csv(
        output_path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
        encoding="utf-8-sig",
    )

def run_sample(
    sample: str,
    memory_path: Path,
    output_root: Path,
    coverage_indexes: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    lookback_days_list: List[int],
    id_modes: List[str],
    chunksize: int,
    max_rows: int | None,
    show_progress: bool,
) -> None:
    """
    运行一个样本的共同覆盖识别。

    Pattern 样本会一次性读入并处理；Full 样本会按 chunksize 分块读入、
    分块识别、分块追加写出。每个 lookback_days 会独立生成一套输出文件。
    如果同一输出文件已存在，当前运行会在第一个 chunk 覆盖该文件，后续
    chunk 再追加写入，确保得到的是本次运行的完整结果。
    """
    print(f"\n========== 处理 {SAMPLE_LABEL[sample]} ==========")
    is_full = sample == "full"

    for lookback_days in lookback_days_list:
        first_write = {id_mode: True for id_mode in id_modes}

        if is_full:
            chunk_iter = iter_memory_chunks(memory_path, chunksize=chunksize, max_rows=max_rows)
        else:
            memory_df = read_memory_file(memory_path, max_rows=max_rows)
            chunk_iter = [(1, memory_df)]

        total_checked = 0
        for chunk_no, memory_chunk in chunk_iter:
            total_checked += len(memory_chunk)
            print(
                f"{SAMPLE_LABEL[sample]}：窗口 {lookback_days} 天，第 {chunk_no} 块，"
                f"MemoryMatrix 行数 = {len(memory_chunk):,}"
            )

            for id_mode in id_modes:
                start = time.perf_counter()
                overlap = identify_overlap_by_annodt_daily_sets(
                    memory_df=memory_chunk,
                    coverage_index=coverage_indexes[id_mode],
                    lookback_days=lookback_days,
                    id_mode=id_mode,
                    show_progress=show_progress,
                )
                elapsed = time.perf_counter() - start

                output_path = output_file_for(output_root, sample, id_mode, lookback_days)
                write_overlap(overlap, output_path, first_write=first_write[id_mode])
                first_write[id_mode] = False

                print(
                    f"  {id_mode} 完成：overlap = {len(overlap):,} 行，"
                    f"用时 {elapsed:.1f} 秒，输出 {output_path}"
                )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="按 Annodt 日度集合口径识别共同分析师覆盖。"
    )
    parser.add_argument("--root", type=str, default=None, help="项目根目录；默认使用当前运行目录。")
    parser.add_argument(
        "--samples",
        nargs="+",
        choices=["pattern", "full"],
        default=["pattern"],
        help="处理样本，默认只处理 pattern；full 样本很大，需显式指定。",
    )
    parser.add_argument("--lookback-days", type=int, nargs="+", default=[365])
    parser.add_argument("--id-modes", nargs="+", choices=["split", "team"], default=["split", "team"])
    parser.add_argument("--chunksize", type=int, default=10_000_000)
    parser.add_argument("--max-rows", type=int, default=None, help="调试用：每个样本最多读取多少行。")
    parser.add_argument("--output-root", type=str, default=None, help="输出根目录；默认写入项目中间结果目录。")
    parser.add_argument("--no-progress", action="store_true", help="关闭 Annodt 分组进度条。")

    return parser.parse_args()


def main() -> None:
    """脚本入口：读取输入、构造覆盖索引、按样本生成共同覆盖明细表。"""
    args = parse_args()
    root = resolve_project_root(args.root)
    forecast_path = root / FORECAST_RELATIVE_PATH
    require_file(forecast_path, "分析师预测文件")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else root / DEFAULT_OUTPUT_RELATIVE_ROOT
    )
    ensure_output_dirs(output_root)

    print(f"项目根目录：{root}")
    print(f"分析师预测文件：{forecast_path}")
    print(f"输出目录：{output_root}")
    print(f"样本：{args.samples}")
    print(f"窗口：{args.lookback_days}")
    print(f"ID 口径：{args.id_modes}")
    print(f"进度条：{'关闭' if args.no_progress else '开启'}")

    print("\n正在读取并清洗分析师预测文件...")
    af_forecast = read_analyst_forecast(forecast_path)

    coverage_indexes: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    if "team" in args.id_modes:
        print("正在构造 team 覆盖索引...")
        coverage_indexes["team"] = build_fast_coverage_index(build_team_coverage(af_forecast))
    if "split" in args.id_modes:
        print("正在构造 split 覆盖索引...")
        coverage_indexes["split"] = build_fast_coverage_index(build_split_coverage(af_forecast))

    sample_paths = {
        "pattern": root / PATTERN_RELATIVE_PATH,
        "full": root / FULL_RELATIVE_PATH,
    }

    for sample in args.samples:
        require_file(sample_paths[sample], f"{SAMPLE_LABEL[sample]} MemoryMatrix 文件")
        run_sample(
            sample=sample,
            memory_path=sample_paths[sample],
            output_root=output_root,
            coverage_indexes=coverage_indexes,
            lookback_days_list=args.lookback_days,
            id_modes=args.id_modes,
            chunksize=args.chunksize,
            max_rows=args.max_rows,
            show_progress=not args.no_progress,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n程序运行失败：{exc}", file=sys.stderr)
        raise
