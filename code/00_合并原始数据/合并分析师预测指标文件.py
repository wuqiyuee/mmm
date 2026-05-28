import gc
from pathlib import Path

import pandas as pd

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
            print(f"{self.desc}: {self.current}/{self.total} {self.unit} - {text}")


# 基础路径：CSMAR 分时间段下载的分析师预测指标文件
base_dir = Path(
    r"/Users/qiyuewu/Desktop/mmm复现数据包/input/CSMAR/分析师预测指标文件（2001-01-01至2025-12-31）"
)

# 合并后的输出文件放在中间结果文件夹下
output_path = '/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv'

# 只读取子文件夹中的 CSV，避免把根目录下已经生成的合并结果再次读进去
csv_files = sorted(
    file
    for folder in base_dir.iterdir()
    if folder.is_dir()
    for file in folder.glob("*.csv")
)

if not csv_files:
    raise FileNotFoundError("没有找到任何 CSV 文件，请检查文件夹路径。")

print(f"共找到 {len(csv_files)} 个 CSV 文件，开始边读取边合并...")

first_write = True
total_rows = 0

progress = tqdm(csv_files, desc="读取并合并 CSV 文件", unit="个")

for file in progress:
    progress.set_postfix_str(f"{file.parent.name}/{file.name}")

    try:
        df = pd.read_csv(file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding="gb18030")

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

print("合并完成")
print(f"共合并 {len(csv_files)} 个 CSV 文件")
print(f"总行数：{total_rows}")
print(f"输出文件：{output_path}")
