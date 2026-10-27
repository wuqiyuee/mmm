import gc
from pathlib import Path

import pandas as pd

# 根据脚本位置定位项目根目录，整个项目移动后无需修改路径。
ROOT = Path(__file__).resolve().parents[2]


try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class tqdm:
        def __init__(self, iterable, desc="", unit=""):
            self.iterable = iterable
            self.desc = desc
            self.unit = unit
            self.total = len(iterable)
            self.current = 0

        def __iter__(self):
            for item in self.iterable:
                self.current += 1
                yield item

        def set_postfix_str(self, text):
            print(f"{self.desc}: {self.current}/{self.total} {self.unit} - {text}", flush=True)


base_dir = ROOT / 'input/CSMAR/市场交易异常波动表(龙虎榜)2002-08-02至2026-05-25'

output_path = ROOT / '中间结果/00_合并原始数据/CSMAR_市场交易异常波动表_龙虎榜_2002-08-02_2026-05-25_合并结果.csv'

chunksize = 200_000

# 如需保留每行来自哪个原始文件，可改为 True。
add_source_file = False


def read_header(file_path):
    return pd.read_csv(file_path, encoding="utf-8-sig", nrows=0).columns.tolist()


csv_files = sorted(base_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"没有找到任何 CSV 文件，请检查文件夹路径：{base_dir}")

output_path.parent.mkdir(parents=True, exist_ok=True)

standard_columns = read_header(csv_files[0])
mismatch_files = []

for file in csv_files[1:]:
    columns = read_header(file)
    if columns != standard_columns:
        mismatch_files.append(file.name)

if mismatch_files:
    raise ValueError(
        "以下文件的表头与第一个 CSV 不一致，请先检查后再合并："
        + "、".join(mismatch_files)
    )

print(f"共找到 {len(csv_files)} 个 CSV 文件，开始分块合并...", flush=True)
print(f"输出文件：{output_path}", flush=True)

first_write = True
total_rows = 0

progress = tqdm(csv_files, desc="读取并合并 CSV 文件", unit="个")

for file in progress:
    progress.set_postfix_str(file.name)

    for chunk in pd.read_csv(
        file,
        encoding="utf-8-sig",
        dtype={"SecurityID": "string", "Symbol": "string"},
        chunksize=chunksize,
    ):
        if add_source_file:
            chunk["source_file"] = file.name

        chunk.to_csv(
            output_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig",
        )

        total_rows += len(chunk)
        first_write = False

        del chunk
        gc.collect()

print("合并完成", flush=True)
print(f"共合并 CSV 文件数：{len(csv_files)}", flush=True)
print(f"合并后总行数：{total_rows}", flush=True)
print(f"输出文件：{output_path}", flush=True)
