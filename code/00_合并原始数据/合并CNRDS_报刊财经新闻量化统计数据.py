from pathlib import Path
import pandas as pd

# CNRDS 报刊财经新闻量化统计原始 Excel 总目录
base_dir = Path(
    "/Users/qiyuewu/Desktop/mmm复现数据包/input/CNRDS/上市公司新闻舆情/财经新闻-CFND/报刊财经新闻/CFND_报刊财经新闻量化统计_ALL"
)

# 年份范围。1994-1999 用作 2000 年起 91 天滚动窗口的历史回看期。
years = range(1994, 2026)

# 输出文件
output_csv = Path('/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/00_合并原始数据/报刊财经新闻量化统计_1994_2025.csv')

dfs = []
missing_files = []

for year in years:
    year_dir = base_dir / str(year)

    if not year_dir.exists():
        missing_files.append(f"{year_dir} 不存在")
        continue

    # 先优先找类似 “报刊财经新闻量化统计-2000.xlsx” 这种文件
    excel_files = sorted(year_dir.glob("报刊财经新闻量化统计-*.xlsx"))

    # 万一文件名略有不同，就退一步读取该目录下全部 xlsx
    if not excel_files:
        excel_files = sorted(year_dir.glob("*.xlsx"))

    if not excel_files:
        missing_files.append(f"{year_dir} 下没有找到 xlsx 文件")
        continue

    for file in excel_files:
        try:
            df = pd.read_excel(
                file,
                dtype={"Scode": str},# 股票代码强制按【文本】读取（防止前面的0丢失）
                engine='openpyxl',# 用专业引擎读取新版Excel文件
                skiprows=[1]) # 跳过中文说明行

            # 清理列名：去掉列名前后的空格，删除看不见的乱码字符
            df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

            # 增加来源信息，后续排查方便
            df["year"] = year
            df["source_file"] = file.name

            dfs.append(df)
            print(f"已读取: {file} | 行数: {len(df)}")

        except Exception as e:
            print(f"读取失败: {file} | 错误: {e}")

if not dfs:
    raise ValueError("没有成功读取任何 Excel 文件，请检查路径或文件格式。")

# 合并
merged = pd.concat(dfs, ignore_index=True, sort=False)

# 可选：按日期、股票代码排序
sort_cols = [c for c in ["Date", "Scode"] if c in merged.columns]
if sort_cols:
    merged = merged.sort_values(sort_cols).reset_index(drop=True)

# 可选：去重
# merged = merged.drop_duplicates()

# 保存为 CSV
# utf-8-sig：Excel 打开中文更稳妥
merged.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("\n合并完成")
print(f"共读取文件数: {len(dfs)}")
print(f"合并后总行数: {len(merged)}")
print(f"输出文件: {output_csv}")

if missing_files:
    print("\n以下目录/文件有缺失：")
    for msg in missing_files:
        print("-", msg)

# merged.to_parquet(base_dir / "报刊财经新闻量化统计_1994_2024.parquet", index=False)
# print("Parquet 也已保存")
