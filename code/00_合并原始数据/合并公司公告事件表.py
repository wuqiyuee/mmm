from __future__ import annotations

import gc
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.errors import ParserError

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class tqdm:
        def __init__(self, iterable: Iterable, desc: str = "", unit: str = ""):
            self.iterable = iterable
            self.desc = desc
            self.unit = unit
            self.total = len(iterable)
            self.current = 0

        def __iter__(self):
            for item in self.iterable:
                self.current += 1
                yield item

        def set_postfix_str(self, text: str) -> None:
            print(f"{self.desc}: {self.current}/{self.total} {self.unit} - {text}", flush=True)


ROOT = Path("/Users/qiyuewu/Desktop/mmm复现数据包")

INPUT_DIR = ROOT / "input/CSMAR/公司公告事件表（2002-03-07至2005-12-31）"
OUTPUT_PATH = ROOT / "中间结果/00_合并原始数据/CSMAR_公司公告事件表_2002_2025_合并结果.csv"

KEY_COL = "AnnouncementID"
STOCK_COL = "Symbol"
DATE_COL = "DeclareDate"


@dataclass
class MergeStats:
    file_count: int
    total_rows: int
    unique_announcement_ids: int
    duplicate_id_rows: int
    duplicate_id_count: int
    missing_announcement_id_rows: int
    output_path: Path
    duplicate_examples: list[dict[str, object]] = field(default_factory=list)


def normalize_stock_code(x: object) -> object:
    """把股票代码统一成 6 位字符串，无法识别时保留缺失。"""
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()
    if s == "":
        return pd.NA

    match = re.search(r"\d{1,6}", s)
    if match is None:
        return pd.NA

    return match.group(0).zfill(6)


def read_one_csv(file: Path) -> pd.DataFrame:
    """自动尝试常见编码读取单个 CSV。"""
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(file, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
        except ParserError:
            return read_csv_with_field_repair(file, encoding=enc)

    raise UnicodeDecodeError("csv", b"", 0, 1, f"无法识别文件编码：{file}")


def read_csv_with_field_repair(file: Path, encoding: str) -> pd.DataFrame:
    """
    修复少数公告标题中未转义英文引号导致的字段数异常。

    CSMAR 该表字段固定为 19 列；异常通常出现在 Title 中包含英文逗号，
    且逗号前有未转义英文引号，导致 csv 解析器把 Title 拆成多列。
    因此当一行字段数超过表头字段数时，把多出来的片段并回 Title。
    """
    rows: list[list[str]] = []
    repaired_rows = 0

    with file.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_len = len(header)
        title_idx = header.index("Title")
        tail_len = expected_len - title_idx - 1

        for line_no, row in enumerate(reader, start=2):
            if len(row) == expected_len:
                rows.append(row)
                continue

            if len(row) > expected_len:
                repaired_title = ",".join(row[title_idx: len(row) - tail_len])
                repaired_row = row[:title_idx] + [repaired_title] + row[len(row) - tail_len:]

                if len(repaired_row) != expected_len:
                    raise ValueError(
                        f"字段修复失败：{file} 第 {line_no} 行，"
                        f"原字段数 {len(row)}，修复后字段数 {len(repaired_row)}"
                    )

                rows.append(repaired_row)
                repaired_rows += 1
                continue

            raise ValueError(
                f"字段数少于表头：{file} 第 {line_no} 行，"
                f"表头字段数 {expected_len}，当前字段数 {len(row)}"
            )

    print(f"已修复字段数异常行：{file}，行数：{repaired_rows}", flush=True)
    return pd.DataFrame(rows, columns=header, dtype=str)


def find_announcement_files(input_dir: Path) -> list[Path]:
    """递归查找公司公告事件表 CSV。"""
    files = sorted(input_dir.rglob("ER_Announcement*.csv"))
    if not files:
        raise FileNotFoundError(f"没有找到 ER_Announcement*.csv：{input_dir}")
    return files


def update_duplicate_stats(
    df: pd.DataFrame,
    seen_ids: set[str],
    duplicate_ids: set[str],
    duplicate_examples: list[dict[str, object]],
    source_folder: str,
    source_file: str,
    max_examples: int = 20,
) -> tuple[int, int]:
    """
    更新 AnnouncementID 重复统计。

    返回：
    1. 当前文件中 AnnouncementID 缺失行数
    2. 当前文件中重复 AnnouncementID 行数
    """
    ids = df[KEY_COL].astype("string").str.strip()
    valid_mask = ids.notna() & (ids != "")
    missing_count = int((~valid_mask).sum())
    duplicate_rows = 0

    for idx, announcement_id in ids[valid_mask].items():
        announcement_id = str(announcement_id)

        if announcement_id in seen_ids:
            duplicate_rows += 1
            duplicate_ids.add(announcement_id)

            if len(duplicate_examples) < max_examples:
                row = df.loc[idx]
                duplicate_examples.append(
                    {
                        "AnnouncementID": announcement_id,
                        "Symbol": row.get(STOCK_COL, pd.NA),
                        "DeclareDate": row.get(DATE_COL, pd.NA),
                        "Title": row.get("Title", pd.NA),
                        "source_folder": source_folder,
                        "source_file": source_file,
                    }
                )
        else:
            seen_ids.add(announcement_id)

    return missing_count, duplicate_rows


def prepare_frame(df: pd.DataFrame, source_folder: str, source_file: str) -> pd.DataFrame:
    """标准化股票代码和日期，并添加来源字段。"""
    df = df.copy()

    if STOCK_COL in df.columns:
        df[STOCK_COL] = df[STOCK_COL].apply(normalize_stock_code)

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")

    df["source_folder"] = source_folder
    df["source_file"] = source_file

    return df


def check_duplicate_ids_in_output(output_path: Path, key_col: str = KEY_COL) -> dict[str, int]:
    """合并写出后，再从输出文件复核 AnnouncementID 重复情况。"""
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    total_rows = 0
    missing_rows = 0
    duplicate_rows = 0

    reader = pd.read_csv(
        output_path,
        encoding="utf-8-sig",
        dtype={key_col: str},
        usecols=[key_col],
        chunksize=500_000,
        low_memory=False,
    )

    for chunk in reader:
        total_rows += len(chunk)
        ids = chunk[key_col].astype("string").str.strip()
        valid_mask = ids.notna() & (ids != "")
        missing_rows += int((~valid_mask).sum())

        for announcement_id in ids[valid_mask]:
            announcement_id = str(announcement_id)
            if announcement_id in seen_ids:
                duplicate_rows += 1
                duplicate_ids.add(announcement_id)
            else:
                seen_ids.add(announcement_id)

    return {
        "total_rows": total_rows,
        "unique_announcement_ids": len(seen_ids),
        "duplicate_id_rows": duplicate_rows,
        "duplicate_id_count": len(duplicate_ids),
        "missing_announcement_id_rows": missing_rows,
    }


def merge_company_announcements(input_dir: Path = INPUT_DIR, output_path: Path = OUTPUT_PATH) -> MergeStats:
    """
    合并 CSMAR 公司公告事件表。

    注意：本函数只合并并检查重复，不会删除重复 AnnouncementID。
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = find_announcement_files(input_dir)

    first_write = True
    expected_columns: list[str] | None = None
    total_rows = 0
    missing_id_rows = 0
    duplicate_id_rows = 0
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    duplicate_examples: list[dict[str, object]] = []

    print(f"共找到 {len(files)} 个 CSV 文件，开始合并...", flush=True)

    progress = tqdm(files, desc="读取并合并公司公告事件表", unit="个")
    for file in progress:
        progress.set_postfix_str(f"{file.parent.name}/{file.name}")

        df = read_one_csv(file)

        if expected_columns is None:
            expected_columns = list(df.columns)
        elif list(df.columns) != expected_columns:
            raise ValueError(
                "CSV 字段不一致：\n"
                f"基准字段：{expected_columns}\n"
                f"当前文件：{file}\n"
                f"当前字段：{list(df.columns)}"
            )

        if KEY_COL not in df.columns:
            raise KeyError(f"{file} 缺少字段：{KEY_COL}")

        missing_count, duplicate_count = update_duplicate_stats(
            df=df,
            seen_ids=seen_ids,
            duplicate_ids=duplicate_ids,
            duplicate_examples=duplicate_examples,
            source_folder=file.parent.name,
            source_file=file.name,
        )

        df = prepare_frame(df, source_folder=file.parent.name, source_file=file.name)

        df.to_csv(
            output_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig",
        )

        total_rows += len(df)
        missing_id_rows += missing_count
        duplicate_id_rows += duplicate_count
        first_write = False

        del df
        gc.collect()

    return MergeStats(
        file_count=len(files),
        total_rows=total_rows,
        unique_announcement_ids=len(seen_ids),
        duplicate_id_rows=duplicate_id_rows,
        duplicate_id_count=len(duplicate_ids),
        missing_announcement_id_rows=missing_id_rows,
        output_path=output_path,
        duplicate_examples=duplicate_examples,
    )


def print_stats(stats: MergeStats) -> None:
    print("=" * 80, flush=True)
    print("合并完成", flush=True)
    print(f"读取 CSV 文件数：{stats.file_count:,}", flush=True)
    print(f"合并后总行数（保留重复）：{stats.total_rows:,}", flush=True)
    print(f"AnnouncementID 唯一值数量：{stats.unique_announcement_ids:,}", flush=True)
    print(f"AnnouncementID 缺失行数：{stats.missing_announcement_id_rows:,}", flush=True)
    print(f"AnnouncementID 重复行数：{stats.duplicate_id_rows:,}", flush=True)
    print(f"存在重复的 AnnouncementID 个数：{stats.duplicate_id_count:,}", flush=True)
    print(f"输出文件：{stats.output_path}", flush=True)

    if stats.duplicate_examples:
        print("=" * 80, flush=True)
        print("重复 AnnouncementID 示例（前 20 条重复出现的记录）：", flush=True)
        print(pd.DataFrame(stats.duplicate_examples).to_string(index=False), flush=True)


def main() -> None:
    stats = merge_company_announcements()
    print_stats(stats)

    print("=" * 80, flush=True)
    print("从合并后的输出文件复核重复值...", flush=True)
    check = check_duplicate_ids_in_output(stats.output_path)
    print(f"输出文件总行数：{check['total_rows']:,}", flush=True)
    print(f"输出文件 AnnouncementID 唯一值数量：{check['unique_announcement_ids']:,}", flush=True)
    print(f"输出文件 AnnouncementID 缺失行数：{check['missing_announcement_id_rows']:,}", flush=True)
    print(f"输出文件 AnnouncementID 重复行数：{check['duplicate_id_rows']:,}", flush=True)
    print(f"输出文件存在重复的 AnnouncementID 个数：{check['duplicate_id_count']:,}", flush=True)


if __name__ == "__main__":
    main()
