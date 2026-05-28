from pathlib import Path
import pandas as pd
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

ROOT = Path('/Users/qiyuewu/Desktop/mmm复现数据包')

_file = ROOT 
td_file = ROOT 

out_file = ROOT 
fig_file = ROOT 

out_file.parent.mkdir(parents=True, exist_ok=True)
fig_file.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 股票代码标准化
# =============================================================================

def normalize_stock_code(x):
    """
    把股票代码统一成 6 位字符串。

    例如：
    1          -> 000001
    000001     -> 000001
    000001.SZ  -> 000001
    600000.SH  -> 600000
    """

    if pd.isna(x):
        return pd.NA

    s = str(x).strip()

    if s == "":
        return pd.NA

    m = re.search(r"\d{1,6}", s)

    if m is None:
        return pd.NA

    return m.group(0).zfill(6)


# =============================================================================
# 自动读取单个 CSV
# =============================================================================

def read_one_csv(file, **read_csv_kwargs):
    """
    自动尝试常见编码读取单个 CSV 文件。
    """

    file = Path(file)

    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]

    for enc in encodings:
        try:
            df = pd.read_csv(
                file,
                encoding=enc,
                low_memory=False,
                **read_csv_kwargs
            )
            return df

        except UnicodeDecodeError:
            continue

    raise ValueError(f"无法识别文件编码：{file}")


# =============================================================================
# 合并文件夹下所有 CSV，并标准化股票代码和日期
# =============================================================================

def merge_csv_folder(
    input_folder,
    stock_cols=None,
    date_cols=None,
    add_source=True,
    output_name=None,
    output_dir="/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/00_合并原始数据",
    save=True,
    save_format="csv",
    **read_csv_kwargs
):
    """
    读取某个文件夹及其子文件夹下所有 CSV，合并成一个 DataFrame。

    功能：
    1. 自动递归读取文件夹下所有 CSV
    2. 自动尝试 utf-8-sig、utf-8、gbk、gb18030 编码
    3. 自动标准化股票代码列
    4. 自动转换日期列为 datetime
    5. 可选择保存为 parquet 或 csv

    参数
    ----
    input_folder : str or Path
        需要读取的文件夹路径。

    stock_cols : list
        需要标准化为 6 位股票代码的列名。
        例如 ["Stkcd"] 或 ["Stkcd", "Symbol"]。

    date_cols : list
        需要转换为日期格式的列名。
        例如 ["Trddt", "Listdt"]。

    add_source : bool
        是否添加来源文件名和来源文件夹。

    output_name : str
        输出文件名，不需要写后缀。
        例如 "TRD_Dalyr_raw"。

    output_dir : str or Path
        输出文件夹。

    save : bool
        是否保存合并结果。

    save_format : str
        "parquet" 或 "csv" 或 "both"。

    **read_csv_kwargs :
        传给 pd.read_csv 的其他参数，例如 usecols。

    返回
    ----
    df_all : DataFrame
        合并后的 DataFrame。
    """

    input_folder = Path(input_folder)
    output_dir = Path(output_dir)

    files = sorted(input_folder.rglob("*.csv"))

    if len(files) == 0:
        raise FileNotFoundError(f"没有找到 CSV 文件：{input_folder}")

    data = []

    for file in files:
        print(f"正在读取：{file}")

        df = read_one_csv(file, **read_csv_kwargs)

        if add_source:
            df["_source_file"] = file.name
            df["_source_folder"] = file.parent.name

        data.append(df)

    df_all = pd.concat(data, ignore_index=True)

    # 标准化股票代码
    if stock_cols is not None:
        for col in stock_cols:
            if col in df_all.columns:
                df_all[col] = df_all[col].apply(normalize_stock_code)
                print(f"已标准化股票代码列：{col}")

    # 转换日期格式
    if date_cols is not None:
        for col in date_cols:
            if col in df_all.columns:
                df_all[col] = pd.to_datetime(df_all[col], errors="coerce")
                print(f"已转换日期列：{col}")

    print("=" * 80)
    print(f"读取 CSV 文件数：{len(files)}")
    print(f"合并后总行数：{len(df_all)}")
    print(f"合并后总列数：{df_all.shape[1]}")

    # 保存文件
    if save:
        if output_name is None:
            output_name = input_folder.name

        output_dir.mkdir(parents=True, exist_ok=True)

        if save_format in ["parquet", "both"]:
            parquet_file = output_dir / f"{output_name}.parquet"
            df_all.to_parquet(parquet_file, index=False)
            print(f"已输出 parquet：{parquet_file}")

        if save_format in ["csv", "both"]:
            csv_file = output_dir / f"{output_name}.csv"
            df_all.to_csv(csv_file, index=False, encoding="utf-8-sig")
            print(f"已输出 CSV：{csv_file}")

    return df_all



def summary_row(s):
    s = pd.to_numeric(s, errors='coerce').dropna()

    return {
        'Mean': s.mean(),
        'p25': s.quantile(0.25),
        'p50': s.quantile(0.50),
        'p75': s.quantile(0.75),
        'Std.Dev.': s.std(),
        'Min': s.min(),
        'Max': s.max(),
        'N': int(s.shape[0])
    }