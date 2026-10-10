"""
构造 111 与 110 两类记忆矩阵。

三位数字的含义固定为：
1. 当前真正触发提示的提示公司，按当前财报季度判断；
2. 历史共同公告中的提示公司，按历史财报季度判断；
3. 历史共同公告中的被提示公司，按历史财报季度判断。

本脚本主口径只保留 111 和 110：
- 111：当前提示端=1、历史提示端=1、历史被提示端=1；
- 110：当前提示端=1、历史提示端=1、历史被提示端=0。

与旧版矩阵的关键区别：当前提示事件单独来自所有当前季度为模式的公告，
而不是从已经筛选过的同日配对表反推。这样两组共享同一批当前提示事件。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


GROUPS = {
    "111": 1,  # 历史被提示端为模式季度
    "110": 0,  # 历史被提示端为非模式季度
}

IAR_RELATIVE_PATH = Path(
    "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv"
)
PATTERN_RELATIVE_PATH = Path(
    "中间结果/02_变量构造结果/6PatternFirmQtr/PatternFirmQtr.csv"
)
OUTPUT_RELATIVE_PATHS = {
    group: Path(f"中间结果/03_记忆矩阵/MemoryMatrix_{group}.csv")
    for group in GROUPS
}

BASE_COLS = ["Stkcd", "Accper", "Annodt"]
LAG_COLS = [f"quarterLag{lag}Alt" for lag in range(5)]
OUTPUT_COLS = [
    "pair_group",
    "Annodt",
    "currentAccper",
    "Stkcd",
    "StkcdRecall",
    "PatternCurrentCue",
    "PatternCueHistory",
    "PatternRecallHistory",
    *LAG_COLS,
]


def normalize_stock_code(value: object) -> str:
    """把证券代码标准化为六位字符串。"""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return f"{int(float(text)):06d}"
    except ValueError:
        return text.zfill(6)


def find_project_root(start: Path | None = None) -> Path:
    """从当前文件或传入目录向上寻找项目根目录。"""
    start_path = Path(start or __file__).resolve()
    search_start = start_path if start_path.is_dir() else start_path.parent
    for candidate in [search_start, *search_start.parents]:
        if (candidate / "input").exists() and (candidate / "code").exists():
            return candidate
    return search_start


def load_announcement_events(root: Path) -> pd.DataFrame:
    """
    读取公告记录并合并公司—财报季度模式标记。

    未匹配到模式标记、或匹配到但标记缺失的公告，均不参与111／110；
    不把这些公告当作标记为0的非模式报告。
    """
    iar_path = root / IAR_RELATIVE_PATH
    pattern_path = root / PATTERN_RELATIVE_PATH

    iar = pd.read_csv(
        iar_path,
        encoding="utf-8-sig",
        usecols=BASE_COLS,
        parse_dates=["Accper", "Annodt"],
    )
    iar["Stkcd"] = iar["Stkcd"].apply(normalize_stock_code)
    iar = iar.dropna(subset=BASE_COLS).copy()

    # 国泰安中 Annodt == Accper 表示有财务数据但未披露相关公告。
    # 这类记录不参与当前公告或历史共同公告配对。
    n_undisclosed = int(
        iar["Annodt"].eq(iar["Accper"]).sum()
    )
    print("未披露占位记录数：", n_undisclosed)

    iar = iar.loc[
        iar["Annodt"].ne(iar["Accper"])
    ].copy()

    iar = iar.loc[iar["Stkcd"].ne("")].copy()
    iar = iar.drop_duplicates(BASE_COLS).copy()



    pattern = pd.read_csv(
        pattern_path,
        encoding="utf-8-sig",
        usecols=["Stkcd", "Accper", "PatternFirmQtr"],
        parse_dates=["Accper"],
    )
    pattern["Stkcd"] = pattern["Stkcd"].apply(normalize_stock_code)
    pattern["PatternFirmQtr"] = pd.to_numeric(
        pattern["PatternFirmQtr"], errors="coerce"
    )

    n_unknown = int(pattern["PatternFirmQtr"].isna().sum())
    print("标记表中状态缺失的财报期数：", n_unknown)

    pattern = pattern.dropna(
        subset=["Stkcd", "Accper", "PatternFirmQtr"]
    ).copy()

    duplicate_marker = pattern.duplicated(["Stkcd", "Accper"]).sum()
    if duplicate_marker:
        raise ValueError(
            "PatternFirmQtr 在 Stkcd + Accper 层面存在重复，不能安全合并。"
        )

    matched = iar.merge(
        pattern,
        on=["Stkcd", "Accper"],
        how="left",
        validate="m:1",
        indicator=True,
    )
    unmatched = matched["_merge"].eq("left_only").sum()
    print(f"公告记录总数：{len(matched):,}")
    print(f"未取得可用模式标记的公告记录：{unmatched:,}")

    # PatternFirmQtr 上游当前只覆盖沪深A股；这里仅保留匹配到标记的公告。
    # 因此111和110目前都只包含沪深A股，包括110的非模式被提示端。
    # 本脚本未单独检查市场类型；更换上游标记表后需重新核查。

    events = matched.loc[matched["_merge"].eq("both")].drop(
        columns=["_merge"]
    )
    events["PatternFirmQtr"] = (
        events["PatternFirmQtr"].astype("int8")
    )
    return events


def build_current_cues(events: pd.DataFrame) -> pd.DataFrame:
    """提取当前公告时属于模式状态的提示公司事件。"""
    current = events.loc[
        events["PatternFirmQtr"].eq(1),
        ["Stkcd", "Annodt", "Accper"],
    ].rename(columns={"Accper": "currentAccper"})

    # 同一公司同一自然日若存在多个财报截止日，使用最新的财报截止日作为当前锚点。
    current = (
        current.sort_values(["Stkcd", "Annodt", "currentAccper"])
        .drop_duplicates(["Stkcd", "Annodt"], keep="last")
        .sort_values(["Annodt", "Stkcd"])
        .reset_index(drop=True)
    )
    print(f"当前模式提示事件数：{len(current):,}")
    return current


def build_historical_pairs(events: pd.DataFrame) -> pd.DataFrame:
    """
    构造历史同日公告的有向配对，并筛出历史提示端为模式的两类配对。

    被提示端使用自己的 Accper 判断 PatternRecallHistory，不能使用提示端的季度。
    """
    # 先在自连接前筛掉历史上不可能成为提示端的记录，避免对全部公告做
    # 无必要的同日笛卡尔配对；被提示端仍保留模式和非模式两类。
    cue_side = events.loc[
        events["PatternFirmQtr"].eq(1),
        ["Annodt", "Stkcd", "Accper", "PatternFirmQtr"],
    ].rename(
        columns={
            "Accper": "AccperCueHistory",
            "PatternFirmQtr": "PatternCueHistory",
        }
    )
    recall_side = events.loc[
        events["PatternFirmQtr"].isin([0, 1]),
        ["Annodt", "Stkcd", "Accper", "PatternFirmQtr"],
    ].rename(
        columns={
            "Stkcd": "StkcdRecall",
            "Accper": "AccperRecall",
            "PatternFirmQtr": "PatternRecallHistory",
        }
    )

    pairs = cue_side.merge(
        recall_side,
        on="Annodt",
        how="inner",
        sort=False,
    )
    pairs = pairs.loc[
        pairs["Stkcd"].ne(pairs["StkcdRecall"])
        & pairs["PatternCueHistory"].eq(1)
        & pairs["PatternRecallHistory"].isin([0, 1])
    ].copy()

    pairs["pair_group"] = np.where(
        pairs["PatternRecallHistory"].eq(1), "111", "110"
    )
    pairs = pairs.drop_duplicates(
        [
            "pair_group",
            "Annodt",
            "Stkcd",
            "StkcdRecall",
            "AccperCueHistory",
            "AccperRecall",
        ]
    )
    pairs = pairs.sort_values(
        ["Annodt", "Stkcd", "StkcdRecall"]
    ).reset_index(drop=True)

    print("历史配对数量（按组）：")
    print(pairs.groupby("pair_group").size().to_string())
    return pairs


def _empty_memory_frame() -> pd.DataFrame:
    """返回带有固定列结构的空记忆矩阵。"""
    return pd.DataFrame(columns=OUTPUT_COLS)


def iter_memory_by_current_date(
    current_cues: pd.DataFrame,
    historical_pairs: pd.DataFrame,
    window_days: int,
    show_progress: bool,
) -> Iterable[pd.DataFrame]:
    """按当前公告日逐块生成记忆矩阵，避免把全部结果一次性放入内存。"""
    if historical_pairs.empty or current_cues.empty:
        return

    historical_pairs = historical_pairs.sort_values(
        ["Annodt", "Stkcd", "StkcdRecall"]
    ).reset_index(drop=True)
    history_dates = historical_pairs["Annodt"].to_numpy(
        dtype="datetime64[ns]"
    )

    grouped_current = current_cues.groupby(
        "Annodt", sort=True, observed=True
    )
    iterator = grouped_current
    if show_progress:
        iterator = tqdm(
            grouped_current,
            total=grouped_current.ngroups,
            desc="构造当前模式提示日的历史关联",
            unit="日",
        )

    for current_date, current_day in iterator:
        current_date = pd.Timestamp(current_date)
        min_date = current_date - pd.Timedelta(days=window_days)

        left = np.searchsorted(
            history_dates,
            min_date.to_datetime64(),
            side="left",
        )
        right = np.searchsorted(
            history_dates,
            current_date.to_datetime64(),
            side="left",
        )
        if left == right:
            continue

        current_stocks = current_day["Stkcd"].unique()
        history = historical_pairs.iloc[left:right]
        history = history.loc[history["Stkcd"].isin(current_stocks)].copy()
        if history.empty:
            continue

        current_anchor = current_day[
            ["Stkcd", "currentAccper"]
        ].drop_duplicates("Stkcd")
        history = history.merge(
            current_anchor,
            on="Stkcd",
            how="inner",
            validate="many_to_one",
        )
        if history.empty:
            continue

        diff_days = (
            history["currentAccper"]
            - history["AccperCueHistory"]
        ).dt.days

        # 历史窗口已排除当前公告日，再按财报截止日差划分滞后。
        history["quarterLag0Alt"] = diff_days.eq(0).astype("int8")
        for lag in range(1, 5):
            lower = (lag - 1) * 100
            upper = lag * 100
            history[f"quarterLag{lag}Alt"] = (
                diff_days.gt(lower) & diff_days.le(upper)
            ).astype("int8")

        lag_values = (
            history.groupby(
                ["pair_group", "Stkcd", "StkcdRecall"],
                as_index=False,
                observed=True,
            )[LAG_COLS]
            .max()
        )
        lag_values["Annodt"] = current_date
        lag_values["PatternCurrentCue"] = 1
        lag_values["PatternCueHistory"] = 1
        lag_values["PatternRecallHistory"] = (
            lag_values["pair_group"].eq("111").astype("int8")
        )

        lag_values = lag_values.merge(
            current_anchor,
            on="Stkcd",
            how="left",
            validate="many_to_one",
        )
        lag_values = lag_values[OUTPUT_COLS]
        lag_values = lag_values.sort_values(
            ["pair_group", "Annodt", "Stkcd", "StkcdRecall"]
        ).reset_index(drop=True)

        duplicate_count = lag_values.duplicated(
            ["pair_group", "Annodt", "Stkcd", "StkcdRecall"]
        ).sum()
        if duplicate_count:
            raise ValueError("单日记忆矩阵中存在重复的组别—公司对。")

        yield lag_values


def write_group_outputs(
    current_cues: pd.DataFrame,
    historical_pairs: pd.DataFrame,
    output_paths: dict[str, Path],
    window_days: int,
    show_progress: bool,
) -> None:
    """逐块写出两组记忆矩阵。"""
    initialized = {group: False for group in output_paths}
    row_counts = {group: 0 for group in output_paths}

    for one_day in iter_memory_by_current_date(
        current_cues=current_cues,
        historical_pairs=historical_pairs,
        window_days=window_days,
        show_progress=show_progress,
    ):
        for group, output_path in output_paths.items():
            part = one_day.loc[one_day["pair_group"].eq(group)].copy()
            if part.empty:
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            part.to_csv(
                output_path,
                mode="a" if initialized[group] else "w",
                header=not initialized[group],
                index=False,
                encoding="utf-8-sig",
                date_format="%Y-%m-%d",
            )
            initialized[group] = True
            row_counts[group] += len(part)

    for group, output_path in output_paths.items():
        if not initialized[group]:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _empty_memory_frame().to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
            )
        print(f"{group} 记忆矩阵行数：{row_counts[group]:,}")
        print(f"输出文件：{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构造111与110两类记忆矩阵。"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="项目根目录；省略时自动寻找。",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=364,
        help="历史共同公告滚动窗口，默认364天。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示进度条。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root or find_project_root()).resolve()
    print(f"项目根目录：{root}")

    events = load_announcement_events(root)
    current_cues = build_current_cues(events)
    historical_pairs = build_historical_pairs(events)

    write_group_outputs(
        current_cues=current_cues,
        historical_pairs=historical_pairs,
        output_paths={
            group: root / relative_path
            for group, relative_path in OUTPUT_RELATIVE_PATHS.items()
        },
        window_days=args.window_days,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
