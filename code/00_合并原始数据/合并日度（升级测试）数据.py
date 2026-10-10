import pandas as pd
from pathlib import Path
import gc


ROOT = Path('/Users/qiyuewu/Desktop/mmm复现数据包')

# 基础路径：直接指向“个股日度收益率（升级测试）”
base_dir = ROOT/'input/CSMAR/个股日度收益率（升级测试）1995-01-01至于1999-12-31'

# 是否导出合并结果
save_output = True

# 导出文件名（如果 save_output=True 时会用到）
output_file = ROOT /'中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv'
output_file.parent.mkdir(parents=True, exist_ok=True)

all_dfs = []

print("🚀 开始递归读取所有 CSV 文件...")

# 递归查找 base_dir 下所有 csv 文件
csv_files = sorted(base_dir.rglob('*.csv'))

# 避免把之前导出的合并文件再次读进去
csv_files = [f for f in csv_files if f.resolve() != output_file.resolve()]

if not csv_files:
    print("❌ 没有找到任何 CSV 文件，请检查路径是否正确。")
    raise SystemExit

print(f"📦 共找到 {len(csv_files)} 个 CSV 文件\n")

for i, file_path in enumerate(csv_files, start=1):
    print(f"📄 正在读取 ({i}/{len(csv_files)}): {file_path.relative_to(base_dir)}")
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        all_dfs.append(df)
        print(f"   ✅ 读取成功，维度: {df.shape}")
    except UnicodeDecodeError:
        print(f"   ⚠️ utf-8 读取失败，尝试用 gbk: {file_path.name}")
        try:
            df = pd.read_csv(file_path, encoding='gbk', low_memory=False)
            all_dfs.append(df)
            print(f"   ✅ 读取成功，维度: {df.shape}")
        except Exception as e:
            print(f"   ❌ 读取失败: {e}")
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")

print("\n⏳ 正在合并所有数据...")

if not all_dfs:
    print("❌ 没有成功读取到任何 CSV 文件。")
    raise SystemExit

df_total = pd.concat(all_dfs, ignore_index=True)
print(f"🎉 合并完成！总维度: {df_total.shape}")

print("\n🔍 数据结构检查：")
print("列名：")
print(df_total.columns.tolist())

print("\n各列缺失值统计：")
print(df_total.isnull().sum())

print("\n🧹 清理内存...")
del all_dfs
gc.collect()

if save_output:
    print(f"\n💾 正在保存到: {output_file}")
    df_total.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("✅ 保存完成！")
else:
    print("\nℹ️ 当前未导出文件。如需导出，把 save_output 改成 True。")