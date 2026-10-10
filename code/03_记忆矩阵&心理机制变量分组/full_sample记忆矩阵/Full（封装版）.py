"""
构造 MemoryMatrixFull.csv

核心目标：
某公司今天公告时，市场可能会“联想到”哪些过去曾经和它同日公告的公司。

本脚本逻辑：
1. 读取 IAR_Rept.csv，仅保留 Stkcd、Accper、Annodt；
2. 按 Annodt 构造同日公告公司对；
3. 对每个真实公告日，取过去 364 天滚动窗口；
4. 计算 quarterLag0Alt-quarterLag4Alt；
5. 计算 nrFirmsAnnLag0Alt-nrFirmsAnnLag4Alt；
6. 输出 MemoryMatrixFull.csv。

优化说明：
- 不改变原本计算逻辑；
- 用 searchsorted 定位滚动窗口，避免每轮扫描完整 pair_matrix；
- 将 5 次 groupby 合并成 1 次 groupby；
- 只把最终需要的变量放入 memory_list，降低内存压力；
- 路径自动寻找项目根目录，换电脑也能用。

运行示例：
python Full_encapsulated.py
python Full_encapsulated.py --root "/Users/qiyuewu/Desktop/mmm复现数据包"
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

OUTPUT_RELATIVE_PATH = Path("中间结果/03_记忆矩阵/MemoryMatrixFull.csv")

BASE_COLS = ["Stkcd", "Accper", "Annodt"]
PAIR_COLS = ["Stkcd", "StkcdRecall"]

KEEP_COLS = [
    "Stkcd",
    "StkcdRecall",
    "Annodt",'Accper',
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
    适合这种项目结构：
        mmm复现数据包/
        ├── input/
        ├── code/
        └── 中间结果/
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
            "python Full_encapsulated.py --root \"你的/mmm复现数据包/路径\""
        )


def print_dataframe_info(name: str, df: pd.DataFrame) -> None:
    """打印 DataFrame 基本信息，方便确认中间表规模。"""
    print(f"{name}: {df.shape[0]:,} 行 × {df.shape[1]:,} 列")


# =============================================================================
# 核心步骤 1：读取数据
# =============================================================================

def load_iar_rept(input_path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """
    读取 IAR_Rept.csv。

    逻辑保持原代码：
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
    4. 保留并排序 Annodt、Stkcd、StkcdRecall、Accper、AccperRecall。

    优化：
    - 保留原 merge 逻辑；
    - reset_index 方便后面用 iloc 窗口切片；
    - Stkcd / StkcdRecall 转为 category，降低内存并加快 groupby。
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

    pair_matrix = pair_matrix.loc[
        pair_matrix["Stkcd"] != pair_matrix["StkcdRecall"],
        ["Annodt", "Stkcd", "StkcdRecall", "Accper", "AccperRecall"],
    ].copy()

    pair_matrix = pair_matrix.sort_values(
        ["Annodt", "Stkcd", "StkcdRecall"]
    ).reset_index(drop=True)

    # 不改变取值，只压缩类型。ordered=True 是为了排序时仍按股票代码字典序。
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

    逻辑仍然是：
    - 取 [current_annodt - 364, current_annodt] 窗口；
    - 对窗口内公司对计算 quarterLag*Alt；
    - 计算 nrFirmsAnnLag*Alt；
    - 每个 Stkcd-StkcdRecall 只保留最近一条；
    - 只保留当前公告日 current_annodt 的结果。

    优化：
    - 用 searchsorted 取窗口，避免每轮扫描完整 pair_matrix；
    - 多个 lag 的 groupby 合并成一次。
    """
    current_annodt = pd.Timestamp(current_annodt)

    min_annodt = current_annodt - pd.Timedelta(days=window_days)
    max_annodt = current_annodt

    # -------------------------------------------------------------------------
    # 1. 当前日期的过去一年滚动窗口
    #    原逻辑：
    #    temp = pair_matrix[(Annodt >= min_annodt) & (Annodt <= max_annodt)]
    #
    #    优化：
    #    pair_matrix 已经按 Annodt 排序，所以用 searchsorted 定位窗口边界。
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

    temp = pair_matrix.iloc[left:right].copy()

    if temp.empty:
        return pd.DataFrame(columns=KEEP_COLS)

    # -------------------------------------------------------------------------
    # 2. 每个线索公司在窗口内最新的财报季度
    # -------------------------------------------------------------------------
    temp["maxAccper_byStkcd"] = (
        temp.groupby("Stkcd", sort=False, observed=True)["Accper"]
        .transform("max")
    )

    diff_days = (temp["maxAccper_byStkcd"] - temp["Accper"]).dt.days

    # -------------------------------------------------------------------------
    # 3. 逐行判断这条历史共同公告属于 lag0-lag4 哪一类
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
    # 4. 公司对层面的 quarterLag 汇总
    #    原逻辑：对 lag0-lag4 分别 groupby transform("max")
    #    优化：一次性 groupby 5 列
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
    # 5. 计算同一天公告的其他公司数量
    # -------------------------------------------------------------------------
    temp["nrFirmsAnnouncing"] = (
        temp.groupby(["Stkcd", "Annodt"], sort=False, observed=True)["StkcdRecall"]
        .transform("size")
    )

    # -------------------------------------------------------------------------
    # 6. 生成 nrFirmsAnnLag*_int
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
    # 7. 公司对层面的 nrFirmsAnnLag 汇总
    #    原逻辑：对 lag0-lag4 分别 groupby transform("max")
    #    优化：一次性 groupby 5 列
    # -------------------------------------------------------------------------
    int_cols = [f"nrFirmsAnnLag{lag}Alt_int" for lag in range(5)]

    nr_firms_lags = (
        temp.groupby(PAIR_COLS, sort=False, observed=True)[int_cols]
        .transform("max")
    )

    nr_firms_lags.columns = [f"nrFirmsAnnLag{lag}Alt" for lag in range(5)]

    temp = pd.concat([temp, nr_firms_lags], axis=1)

    # -------------------------------------------------------------------------
    # 8. 每个 Stkcd-StkcdRecall 公司对只保留最后一条历史共同公告记录
    # -------------------------------------------------------------------------
    temp = (
        temp.sort_values(["Stkcd", "StkcdRecall", "Annodt"])
        .groupby(PAIR_COLS, sort=False, observed=True, group_keys=False)
        .tail(1)
        .copy()
    )

    # -------------------------------------------------------------------------
    # 9. 把历史 Annodt 替换成线索公司当前最新公告日
    # -------------------------------------------------------------------------
    temp["maxAnnodt_byStkcd"] = (
        temp.groupby("Stkcd", sort=False, observed=True)["Annodt"]
        .transform("max")
    )

    temp["Annodt"] = temp["maxAnnodt_byStkcd"]

    # -------------------------------------------------------------------------
    # 10. 只保留当前循环日期的结果
    # -------------------------------------------------------------------------
    temp = temp[temp["Annodt"] == max_annodt].copy()

    if temp.empty:
        return pd.DataFrame(columns=KEEP_COLS)

    # 只返回最终需要的列，避免 memory_list 堆积临时变量
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

    逻辑保持原代码：
    - 只遍历 pair_matrix 中真实存在的 Annodt；
    - 每个 Annodt 构造一个滚动窗口结果；
    - 最后 concat。
    """
    memory_list: list[pd.DataFrame] = []

    # 确保 Annodt 有序，供 searchsorted 使用
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

    # 理论上这里已经只有 KEEP_COLS，保留这句是为了保险
    memory_matrix = memory_matrix[KEEP_COLS].copy()

    memory_matrix = memory_matrix.sort_values(
        ["Stkcd", "Annodt", "StkcdRecall"]
    ).reset_index(drop=True)

    # 类似 Stata compress：dummy 列压缩为 int8
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

def save_memory_matrix(memory_matrix: pd.DataFrame, output_path: Path) -> None:
    """保存 MemoryMatrixFull.csv。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    memory_matrix.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# 总入口函数
# =============================================================================

def run_memory_matrix_pipeline(
    root: str | Path | None = None,
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    一键运行完整流程。

    参数
    ----
    root:
        项目根目录。如果不传，会自动向上寻找包含 input/ 的目录。
    encoding:
        输入 CSV 编码，默认 utf-8。
    show_progress:
        是否显示 tqdm 进度条。

    返回
    ----
    pd.DataFrame
        构造好的 MemoryMatrix。
    """
    project_root, input_path, output_path = get_project_paths(root)

    print(f"项目根目录：{project_root}")
    print(f"输入文件：{input_path}")
    print(f"输出文件：{output_path}")

    iar_rept = load_iar_rept(input_path=input_path, encoding=encoding)
    print_dataframe_info("IAR_Rept", iar_rept)

    pair_matrix = build_announcement_pair_matrix(iar_rept)
    print_dataframe_info("IAR_Rept_merge", pair_matrix)

    memory_matrix = build_memory_matrix(
        pair_matrix=pair_matrix,
        show_progress=show_progress,
    )

    print_dataframe_info("MemoryMatrixFull", memory_matrix)

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
    parser = argparse.ArgumentParser(description="构造 MemoryMatrixFull.csv")

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
    )