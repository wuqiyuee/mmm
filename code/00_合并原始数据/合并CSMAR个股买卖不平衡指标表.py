import pandas as pd
import glob
import os
import gc

from pathlib import Path

# 根据脚本位置定位项目根目录，整个项目移动后无需修改路径。
ROOT = Path(__file__).resolve().parents[2]


# 基础路径
base_dir = str(ROOT / 'input/CSMAR/个股买卖不平衡指标表（日）2003-01-02至2026-08-31')

# 存储所有子文件夹合并后的 DataFrame
all_dfs = []

print("🚀 开始读取个股买卖不平衡指标表数据...")

# 获取 base_dir 下所有子文件夹，并排序
folders = sorted([
    f for f in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, f))
])

if not folders:
    print("❌ 没有找到任何子文件夹，请检查路径是否正确。")
    exit()

# 遍历每个时间段文件夹
for folder in folders:
    print(f"\n📁 正在处理文件夹: {folder}")
    folder_path = os.path.join(base_dir, folder)

    # 读取该文件夹下所有 csv 文件
    files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))

    if not files:
        print("  ⚠️ 该文件夹下没有找到 CSV 文件，跳过。")
        continue

    df_list = []
    total_files = len(files)

    for idx, f in enumerate(files, start=1):
        file_name = os.path.basename(f)
        print(f"  📄 正在读取 ({idx}/{total_files}): {file_name} ...")

        try:
            df = pd.read_csv(f, encoding='utf-8')
            df_list.append(df)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(f, encoding='gbk')
                df_list.append(df)
            except Exception as e:
                print(f"  ❌ 读取 {file_name} 失败: {e}")
        except Exception as e:
            print(f"  ❌ 读取 {file_name} 失败: {e}")

    # 合并当前文件夹下的所有文件
    if df_list:
        df_temp = pd.concat(df_list, ignore_index=True)
        print(f"✅ 文件夹 {folder} 合并完成，数据维度: {df_temp.shape}")
        all_dfs.append(df_temp)

print("\n⏳ 正在合并全样本数据...")

# 合并所有年份区间
if all_dfs:
    df_total = pd.concat(all_dfs, ignore_index=True)
    print(f"🎉 全样本合并成功！总数据维度: {df_total.shape}")
else:
    print("❌ 没有读取到任何数据，请检查文件夹和 CSV 文件。")
    exit()

# 释放内存
print("🧹 正在清理内存...")
try:
    del df_list, df_temp, all_dfs
except NameError:
    pass
gc.collect()

# 导出合并结果
output_path =str(ROOT / '中间结果/00_合并原始数据/CSMAR_个股买卖不平衡指标表_日_合并结果.csv')
print(f"💾 正在保存合并后的文件到: {output_path}")
df_total.to_csv(output_path, index=False, encoding='utf-8-sig')
print("🏁 CSV 文件保存完毕！")

print("\n" + "=" * 50)
print("🔍 数据结构检查：")
print("列名：", df_total.columns.tolist())
print("\n各列缺失值统计：\n", df_total.isnull().sum())
print("=" * 50)