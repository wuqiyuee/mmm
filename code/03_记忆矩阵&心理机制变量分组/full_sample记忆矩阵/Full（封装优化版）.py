"""
构造 MemoryMatrixFull.csv —— 优化版

核心目标：
某公司今天公告时，市场可能会“联想到”哪些过去曾经和它同日公告的公司。

本脚本逻辑：
1. 读取 IAR_Rept.csv，仅保留 Stkcd、Accper、Annodt；
2. 按 Annodt 构造同日公告公司对；
3. 对每个真实公告日，取过去 364 天滚动窗口；
4. 计算 quarterLag0Alt-quarterLag4Alt；
5. 计算 nrFirmsAnnLag0Alt-nrFirmsAnnLag4Alt；
6. 输出 MemoryMatrixFull.csv。

相对上一版的主要优化：
1. pair_matrix 不再保留 AccperRecall，降低 1.4 亿行大表的内存压力；
2. build_memory_matrix 中不再重复排序 pair_matrix；
3. 每个 current_annodt 只处理“当天公告的 Stkcd”在过去一年内的历史共同公告关系；
4. 避免原来每个日期都对过去一年窗口里的所有公司做 groupby；
5. 保持核心计算逻辑和最终变量结构不变。

运行示例：
python Full（封装优化版）.py
python Full（封装优化版）.py --root "/Users/qiyuewu/Desktop/mmm复现数据包"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 常量设置
# =============================================================================

INPUT_RELATIVE_PATH = Path(
    "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv"
)

COMPANY_RELATIVE_PATH = Path(
    "input/CSMAR/公司文件(1990-12-10至2026-08-21)/TRD_Co.csv"
)

OUTPUT_RELATIVE_PATH = Path("中间结果/03_记忆矩阵/MemoryMatrixFull(封装优化版).csv")

A_SHARE_MARKETTYPES = {1, 4, 16, 32}  # 沪深A股
EXCLUDE_BACKDATED_CURRENT_EVENTS = True

BASE_COLS = ["Stkcd", "Accper", "Annodt"]
PAIR_COLS = ["Stkcd", "StkcdRecall"]

KEEP_COLS = [
    "Stkcd",
    "StkcdRecall",
    "Annodt",
    "Accper",
    "quarterLag0Alt",
    "quarterLag1Alt",
    "quarterLag2Alt",
    "quarterLag3Alt",
    "quarterLag4Alt",
    "nrFirmsAnnLag0Alt",
    "nrFirmsAnnLag1Alt",
    "nrFirmsAnnLag2Alt",
    "nrFirmsAnnLag3Alt",
    "nrFirmsAnnLag4Alt",
]


# =============================================================================
# 工具函数
# =============================================================================

def normalize_stock_code(x: object) -> str:
    """标准化股票代码：转成 6 位字符串。"""
    if pd.isna(x):
        return ""

    s = str(x).strip()

    if s == "":
        return ""

    return f"{int(float(s)):06d}"


def find_project_root(start: Path | None = None) -> Path:
    """
    自动寻找项目根目录。

    判断标准：某一层目录下存在 input/ 文件夹。
    """
    if start is None:
        start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

    start = Path(start).resolve()

    for candidate in [start, *start.parents]:
        if (candidate / "input").exists():
            return candidate

    return start


def get_project_paths(root: str | Path | None = None) -> tuple[Path, Path, Path]:
    """返回项目根目录、输入路径、输出路径。"""
    project_root = Path(root).expanduser().resolve() if root is not None else find_project_root()
    input_path = project_root / INPUT_RELATIVE_PATH
    output_path = project_root / OUTPUT_RELATIVE_PATH
    return project_root, input_path, output_path


def validate_input_path(input_path: Path) -> None:
    """检查输入文件是否存在。"""
    if not input_path.exists():
        raise FileNotFoundError(
            "找不到输入文件：\n"
            f"{input_path}\n\n"
            "请确认项目根目录是否正确，或使用：\n"
            "python Full（封装优化版）.py --root \"你的/mmm复现数据包/路径\""
        )


def print_dataframe_info(name: str, df: pd.DataFrame) -> None:
    """打印 DataFrame 基本信息，方便确认中间表规模。"""
    print(f"{name}: {df.shape[0]:,} 行 × {df.shape[1]:,} 列")


# =============================================================================
# 核心步骤 1：读取数据
# =============================================================================

def load_iar_rept(input_path: Path,company_path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """
    读取 IAR_Rept.csv。

    逻辑：
    - 只读取 Stkcd、Accper、Annodt；
    - Accper、Annodt 解析为日期；
    - Stkcd 标准化为 6 位字符串。
    """
    validate_input_path(input_path)

    iar_rept = pd.read_csv(
        input_path,
        encoding=encoding,
        usecols=BASE_COLS,
        parse_dates=["Accper", "Annodt"],
    )

    iar_rept["Stkcd"] = iar_rept["Stkcd"].apply(normalize_stock_code)

    # 去掉关键变量缺失的观测，避免后续无效 merge
    iar_rept = iar_rept.dropna(subset=["Stkcd", "Accper", "Annodt"]).copy()

    company = pd.read_csv(
        company_path,
        encoding=encoding,
        usecols=["Stkcd", "Markettype"],
    )
    company["Stkcd"] = company["Stkcd"].apply(normalize_stock_code)
    company["Markettype"] = pd.to_numeric(
        company["Markettype"], errors="raise"
    )

    if company["Stkcd"].duplicated().any():
        raise ValueError("公司文件的证券代码不唯一")

    iar_rept = iar_rept.merge(
        company,
        on="Stkcd",
        how="left",
        validate="m:1",
    )

    if iar_rept["Markettype"].isna().any():
        raise ValueError("部分公告证券未匹配到市场类型")

    iar_rept = (
        iar_rept.loc[
            iar_rept["Markettype"].isin(A_SHARE_MARKETTYPES)
        ]
        .drop(columns="Markettype")
        .copy()
    )

    # 未披露占位日期不是公告事件，不参与同日公司配对。
    n_undisclosed = int(iar_rept["Annodt"].eq(iar_rept["Accper"]).sum())
    print("沪深A股未披露占位记录数：", n_undisclosed)
    iar_rept = iar_rept.loc[
        iar_rept["Annodt"].ne(iar_rept["Accper"])
    ].copy()

    rows_before_dedup = len(iar_rept)

    iar_rept = (
        iar_rept
        .sort_values(
            ["Stkcd", "Annodt", "Accper"],
            kind="stable",
        )
        .drop_duplicates(
            ["Stkcd", "Annodt"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    print(
        "同公司同公告日去重删除记录数：",
        rows_before_dedup - len(iar_rept),
    )

    if iar_rept.duplicated(["Stkcd", "Annodt"]).any():
        raise ValueError("去重后仍存在同公司同公告日重复记录")

    return iar_rept


# =============================================================================
# 核心步骤 2：构造同日公告公司对
# =============================================================================

def build_announcement_pair_matrix(iar_rept: pd.DataFrame) -> pd.DataFrame:
    """
    构造同日公告公司对。

    对应原代码逻辑：
    1. 复制一份并改名为 recall side；
    2. 按 Annodt merge；
    3. 删除 Stkcd == StkcdRecall；
    4. 保留 Annodt、Stkcd、StkcdRecall、Accper。

    优化：
    - 不再保留 AccperRecall，因为后续计算和最终输出均未使用；
    - Stkcd / StkcdRecall 转为 category，降低内存；
    - 在这里完成排序，后面不再重复排序。
    """
    iar_rept_recall = iar_rept.rename(
        columns={
            "Stkcd": "StkcdRecall",
            "Accper": "AccperRecall",
        }
    )

    pair_matrix = iar_rept_recall.merge(
        iar_rept,
        on="Annodt",
        how="inner",
        sort=False,
        copy=False,
    )

    # 关键优化：不保留 AccperRecall
    pair_matrix = pair_matrix.loc[
        pair_matrix["Stkcd"] != pair_matrix["StkcdRecall"],
        ["Annodt", "Stkcd", "StkcdRecall", "Accper"],
    ].copy()

    pair_matrix = pair_matrix.sort_values(
        ["Annodt", "Stkcd", "StkcdRecall"]
    ).reset_index(drop=True)

    stock_categories = sorted(iar_rept["Stkcd"].dropna().unique().tolist())

    pair_matrix["Stkcd"] = pd.Categorical(
        pair_matrix["Stkcd"],
        categories=stock_categories,
        ordered=True,
    )

    pair_matrix["StkcdRecall"] = pd.Categorical(
        pair_matrix["StkcdRecall"],
        categories=stock_categories,
        ordered=True,
    )

    return pair_matrix


# =============================================================================
# 核心步骤 3：构造单个公告日的滚动窗口结果
# =============================================================================

def build_one_day_memory_matrix(
    pair_matrix: pd.DataFrame,
    current_annodt: pd.Timestamp,
    annodt_values: np.ndarray,
    window_days: int = 364,
) -> pd.DataFrame:
    """
    构造某一个公告日 current_annodt 的临时 MemoryMatrix。

    相对上一版的关键优化：
    - 先找出 current_annodt 当天公告的 Stkcd；
    - 再从过去一年窗口中只保留这些 Stkcd 的历史共同公告关系；
    - 避免对窗口内所有公司反复做 groupby。
    """
    current_annodt = pd.Timestamp(current_annodt)
    min_annodt = current_annodt - pd.Timedelta(days=window_days)
    max_annodt = current_annodt

    current_dt64 = max_annodt.to_datetime64()

    # -------------------------------------------------------------------------
    # 1. 找出当前公告日当天真实出现的 Stkcd
    # -------------------------------------------------------------------------
    cur_left = np.searchsorted(
        annodt_values,
        current_dt64,
        side="left",
    )

    cur_right = np.searchsorted(
        annodt_values,
        current_dt64,
        side="right",
    )

    if cur_left == cur_right:
        return pd.DataFrame(columns=KEEP_COLS)

    current_stkcds = pair_matrix.iloc[cur_left:cur_right]["Stkcd"].unique()

    if len(current_stkcds) == 0:
        return pd.DataFrame(columns=KEEP_COLS)

    # -------------------------------------------------------------------------
    # 2. 用 searchsorted 定位过去一年滚动窗口
    # -------------------------------------------------------------------------
    left = np.searchsorted(
        annodt_values,
        min_annodt.to_datetime64(),
        side="left",
    )

    right = np.searchsorted(
        annodt_values,
        max_annodt.to_datetime64(),
        side="right",
    )

    # -------------------------------------------------------------------------
    # 3. 关键优化：窗口内只保留当前公告公司的历史共同公告关系
    # -------------------------------------------------------------------------
    temp = pair_matrix.iloc[left:right]
    temp = temp[temp["Stkcd"].isin(current_stkcds)].copy()

    if temp.empty:
        return pd.DataFrame(columns=KEEP_COLS)

    # -------------------------------------------------------------------------
    # 4. 每个线索公司在窗口内最新的财报季度
    # -------------------------------------------------------------------------
    temp["maxAccper_byStkcd"] = (
        temp.groupby("Stkcd", sort=False, observed=True)["Accper"]
        .transform("max")
    )

    # 本次公告的财报期若早于此前已公告的最新财报期，
    # 本次不以该公司为提示公司生成记忆矩阵记录。
    # pair_matrix 不变，因此这条公告仍可参与未来的历史关系。
    if EXCLUDE_BACKDATED_CURRENT_EVENTS:
        backdated_firms = temp.loc[
            temp["Annodt"].eq(current_annodt)
            & temp["Accper"].lt(temp["maxAccper_byStkcd"]),
            "Stkcd",
        ].unique()

        if len(backdated_firms) > 0:
            temp = temp.loc[
                ~temp["Stkcd"].isin(backdated_firms)
            ].copy()

            if temp.empty:
                return pd.DataFrame(columns=KEEP_COLS)

    diff_days = (temp["maxAccper_byStkcd"] - temp["Accper"]).dt.days

    # -------------------------------------------------------------------------
    # 5. 判断历史共同公告属于 lag0-lag4 哪一类
    # -------------------------------------------------------------------------
    dummy_data = {
        "dummyQuarterLag0Alt": (
            temp["Accper"] == temp["maxAccper_byStkcd"]
        ).astype(np.int8)
    }

    for lag in range(1, 5):
        lower = (lag - 1) * 100
        upper = lag * 100

        dummy_data[f"dummyQuarterLag{lag}Alt"] = (
            (diff_days > lower) & (diff_days <= upper)
        ).astype(np.int8)

    dummy_df = pd.DataFrame(dummy_data, index=temp.index)
    temp = pd.concat([temp, dummy_df], axis=1)

    # -------------------------------------------------------------------------
    # 6. 公司对层面的 quarterLag 汇总
    # -------------------------------------------------------------------------
    dummy_cols = [f"dummyQuarterLag{lag}Alt" for lag in range(5)]

    quarter_lags = (
        temp.groupby(PAIR_COLS, sort=False, observed=True)[dummy_cols]
        .transform("max")
        .fillna(0)
        .astype(np.int8)
    )

    quarter_lags.columns = [f"quarterLag{lag}Alt" for lag in range(5)]
    temp = pd.concat([temp, quarter_lags], axis=1)

    # -------------------------------------------------------------------------
    # 7. 计算同一天公告的其他公司数量
    # -------------------------------------------------------------------------
    temp["nrFirmsAnnouncing"] = (
        temp.groupby(["Stkcd", "Annodt"], sort=False, observed=True)["StkcdRecall"]
        .transform("size")
    )

    # -------------------------------------------------------------------------
    # 8. 生成 nrFirmsAnnLag*_int
    # -------------------------------------------------------------------------
    int_data = {}
    nr_firms_arr = temp["nrFirmsAnnouncing"].to_numpy()

    for lag in range(5):
        lag_col = f"dummyQuarterLag{lag}Alt"
        int_col = f"nrFirmsAnnLag{lag}Alt_int"

        int_data[int_col] = np.where(
            temp[lag_col].to_numpy() == 1,
            nr_firms_arr,
            np.nan,
        )

    int_df = pd.DataFrame(int_data, index=temp.index)
    temp = pd.concat([temp, int_df], axis=1)

    # -------------------------------------------------------------------------
    # 9. 公司对层面的 nrFirmsAnnLag 汇总
    # -------------------------------------------------------------------------
    int_cols = [f"nrFirmsAnnLag{lag}Alt_int" for lag in range(5)]

    nr_firms_lags = (
        temp.groupby(PAIR_COLS, sort=False, observed=True)[int_cols]
        .transform("max")
    )

    nr_firms_lags.columns = [f"nrFirmsAnnLag{lag}Alt" for lag in range(5)]
    temp = pd.concat([temp, nr_firms_lags], axis=1)

    # -------------------------------------------------------------------------
    # 10. 每个 Stkcd-StkcdRecall 公司对只保留最近一条历史共同公告记录
    # -------------------------------------------------------------------------
    temp = (
        temp.sort_values(["Stkcd", "StkcdRecall", "Annodt"])
        .groupby(PAIR_COLS, sort=False, observed=True, group_keys=False)
        .tail(1)
        .copy()
    )

    if temp.empty:
        return pd.DataFrame(columns=KEEP_COLS)

    # -------------------------------------------------------------------------
    # 11. 替换为当前公告日
    #
    # 原代码是：
    # temp["maxAnnodt_byStkcd"] = temp.groupby("Stkcd")["Annodt"].transform("max")
    # temp["Annodt"] = temp["maxAnnodt_byStkcd"]
    # temp = temp[temp["Annodt"] == max_annodt]
    #
    # 由于本函数前面已经限定 Stkcd 必须是 current_annodt 当天公告的公司，
    # 所以这里可以直接赋值为 current_annodt。
    # -------------------------------------------------------------------------
    temp["Annodt"] = max_annodt

    return temp[KEEP_COLS].copy()


# =============================================================================
# 核心步骤 4：循环构造完整 MemoryMatrix
# =============================================================================

def build_memory_matrix(
    pair_matrix: pd.DataFrame,
    show_progress: bool = True,
    window_days: int = 364,
) -> pd.DataFrame:
    """
    构造完整 MemoryMatrix。

    优化：
    - pair_matrix 在 build_announcement_pair_matrix 中已经排序；
    - 此处不再对 1 亿多行大表重复 sort_values；
    - 仍然只遍历 pair_matrix 中真实存在的 Annodt。
    """
    memory_list: list[pd.DataFrame] = []

    if not pair_matrix["Annodt"].is_monotonic_increasing:
        pair_matrix = pair_matrix.sort_values(
            ["Annodt", "Stkcd", "StkcdRecall"]
        ).reset_index(drop=True)

    annodt_values = pair_matrix["Annodt"].to_numpy(dtype="datetime64[ns]")

    annodt_list = (
        pair_matrix["Annodt"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    iterator: Iterable[pd.Timestamp]

    if show_progress:
        iterator = tqdm(
            annodt_list,
            desc="Building MemoryMatrix",
            unit="date",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
        )
    else:
        iterator = annodt_list

    for current_annodt in iterator:
        one_day_matrix = build_one_day_memory_matrix(
            pair_matrix=pair_matrix,
            current_annodt=current_annodt,
            annodt_values=annodt_values,
            window_days=window_days,
        )

        if not one_day_matrix.empty:
            memory_list.append(one_day_matrix)

    if memory_list:
        memory_matrix = pd.concat(memory_list, ignore_index=True)
    else:
        memory_matrix = pd.DataFrame(columns=KEEP_COLS)

    memory_matrix = memory_matrix[KEEP_COLS].copy()

    memory_matrix = memory_matrix.sort_values(
        ["Stkcd", "Annodt", "StkcdRecall"]
    ).reset_index(drop=True)

    for col in [
        "quarterLag0Alt",
        "quarterLag1Alt",
        "quarterLag2Alt",
        "quarterLag3Alt",
        "quarterLag4Alt",
    ]:
        if col in memory_matrix.columns:
            memory_matrix[col] = memory_matrix[col].fillna(0).astype(np.int8)

    return memory_matrix


# =============================================================================
# 核心步骤 5：保存
# =============================================================================

def save_memory_matrix(
    memory_matrix: pd.DataFrame,
    output_path: Path,
    chunksize: int = 200_000,
) -> None:
    """
    分块保存 MemoryMatrixFull.csv，并显示保存进度条。

    参数
    ----
    memory_matrix:
        要保存的 DataFrame。
    output_path:
        输出 CSV 路径。
    chunksize:
        每次写入多少行。电脑容易卡死时建议 100_000 或 200_000。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 先写入临时文件，全部成功后再替换正式文件
    # 避免程序中途崩溃后留下一个看似完整但实际不完整的 CSV
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if tmp_path.exists():
        tmp_path.unlink()

    n_rows = len(memory_matrix)

    if n_rows == 0:
        raise ValueError(
            "记忆矩阵没有生成记录，请检查公告样本和同日公司配对。"
        )

    n_chunks = (n_rows + chunksize - 1) // chunksize

    print(f"准备分块保存：{n_rows:,} 行")
    print(f"每块行数：{chunksize:,}")
    print(f"预计块数：{n_chunks:,}")
    print(f"临时文件：{tmp_path}")

    with tqdm(
        total=n_rows,
        desc="Saving MemoryMatrix",
        unit="row",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as pbar:
        for start in range(0, n_rows, chunksize):
            end = min(start + chunksize, n_rows)

            chunk = memory_matrix.iloc[start:end]

            chunk.to_csv(
                tmp_path,
                mode="w" if start == 0 else "a",
                index=False,
                header=(start == 0),
                encoding="utf-8-sig" if start == 0 else "utf-8",
                date_format="%Y-%m-%d",
            )

            pbar.update(end - start)

    # 全部写完后，把临时文件改名为正式输出文件
    tmp_path.replace(output_path)

    print(f"保存完成：{output_path}")


# =============================================================================
# 总入口函数
# =============================================================================

def run_memory_matrix_pipeline(
    root: str | Path | None = None,
    encoding: str = "utf-8",
    show_progress: bool = True,
    window_days: int = 364,
) -> pd.DataFrame:
    """
    一键运行完整流程。
    """
    project_root, input_path, output_path = get_project_paths(root)

    print(f"项目根目录：{project_root}")
    print(f"输入文件：{input_path}")
    print(f"输出文件：{output_path}")

    iar_rept = load_iar_rept(
        input_path=input_path,
        company_path=project_root / COMPANY_RELATIVE_PATH,
        encoding=encoding,
    )
    print_dataframe_info("IAR_Rept", iar_rept)

    pair_matrix = build_announcement_pair_matrix(iar_rept)
    print_dataframe_info("IAR_Rept_merge", pair_matrix)

    memory_matrix = build_memory_matrix(
        pair_matrix=pair_matrix,
        show_progress=show_progress,
        window_days=window_days,
    )

    print_dataframe_info("MemoryMatrixFull", memory_matrix)

    # 保存前释放不再需要的大对象，降低卡死概率
    print("正在释放中间大表内存...")
    del pair_matrix
    del iar_rept

    import gc
    gc.collect()

    print("正在保存记忆矩阵...")
    save_memory_matrix(memory_matrix, output_path)
    print("保存成功！")

    print("最终变量列表：")
    for idx, col in enumerate(memory_matrix.columns, start=1):
        print(idx, col)

    return memory_matrix


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造 MemoryMatrixFull.csv 优化版")

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "项目根目录，例如：/Users/qiyuewu/Desktop/mmm复现数据包。"
            "默认自动寻找包含 input/ 的目录。"
        ),
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="输入 CSV 文件编码，默认 utf-8。",
    )

    parser.add_argument(
        "--window-days",
        type=int,
        default=364,
        help="滚动窗口天数，默认 364。",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示 tqdm 进度条。",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_memory_matrix_pipeline(
        root=args.root,
        encoding=args.encoding,
        show_progress=not args.no_progress,
        window_days=args.window_days,
    )