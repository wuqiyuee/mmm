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


base_dir = ROOT / 'input/CSMAR/股票历史日行情信息表(前复权)1996-01-01 至 2025-12-31'

output_path = (
    str(ROOT / '中间结果/00_合并原始数据/CSMAR_股票历史日行情信息表_前复权_1996_2025_合并结果.csv')
)

csv_files = sorted(
    file
    for folder in base_dir.iterdir()
    if folder.is_dir()
    for file in folder.glob("*.csv")
)

if not csv_files:
    raise FileNotFoundError("没有找到任何 CSV 文件，请检查文件夹路径。")

print(f"共找到 {len(csv_files)} 个 CSV 文件，开始边读取边合并...", flush=True)

first_write = True
total_rows = 0

progress = tqdm(csv_files, desc="读取并合并 CSV 文件", unit="个")

for file in progress:
    progress.set_postfix_str(f"{file.parent.name}/{file.name}")

    try:
        df = pd.read_csv(file, encoding="utf-8-sig", dtype={"Symbol": "string"})
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding="gb18030", dtype={"Symbol": "string"})

    df["source_folder"] = file.parent.name
    df["source_file"] = file.name

    df.to_csv(
        output_path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
        encoding="utf-8-sig",
    )

    total_rows += len(df)
    first_write = False

    del df
    gc.collect()

print("合并完成", flush=True)
print(f"共合并 {len(csv_files)} 个 CSV 文件", flush=True)
print(f"总行数：{total_rows}", flush=True)
print(f"输出文件：{output_path}", flush=True)
