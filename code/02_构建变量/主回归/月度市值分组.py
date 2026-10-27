import pandas as pd
from pathlib import Path
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

ROOT = Path(__file__).resolve().parents[3]

TRD_Mnth_file = ROOT/'input/CSMAR/月个股回报率文件2000-01至2025-12/TRD_Mnth.csv'
out_file = ROOT/'中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv'


def normalize_stock_code(x):
    """标准化股票代码为 6 位字符串。"""
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()
    if s == '':
        return pd.NA

    m = re.search(r'\d{1,6}', s)
    if m is None:
        return pd.NA

    return m.group(0).zfill(6)


def read_and_prepare_data(input_file):
    """读取并整理月度市值数据。"""
    trd_mnth = pd.read_csv(
        input_file,
        encoding='utf-8',
        usecols=[
            'Stkcd',
            'Trdmnt',
            'Msmvosd',
            'Msmvttl',
            'Markettype',
        ],
    )

    trd_mnth['Stkcd'] = (
        trd_mnth['Stkcd']
        .apply(normalize_stock_code)
    )

    trd_mnth['Trdmnt'] = pd.to_datetime(
        trd_mnth['Trdmnt'],
        format='%Y-%m',
        errors='coerce',
    )

    trd_mnth['msizew'] = trd_mnth['Msmvosd'] * 1000
    trd_mnth['msize'] = trd_mnth['Msmvttl'] * 1000

    trd_mnth = trd_mnth.loc[
        trd_mnth['Markettype'].isin([1, 4])
    ].copy()

    trd_mnth['year'] = trd_mnth['Trdmnt'].dt.year
    trd_mnth['month'] = trd_mnth['Trdmnt'].dt.month

    return trd_mnth


def calculate_market_cap_statistics(trd_mnth):
    """计算月度市值分位数和年度平均市值。"""
    percentiles = {
        '90th': 0.90,
        '85th': 0.85,
        '95th': 0.95,
        '50th': 0.50,
        '25th': 0.25,
        '10th': 0.10,
    }

    for name, q in percentiles.items():
        trd_mnth[
            f'marketCap_{name}_percentile'
        ] = (
            trd_mnth.groupby('Trdmnt')['msize']
            .transform(lambda x: x.quantile(q))
        )

    trd_mnth['marketCapMean'] = (
        trd_mnth.groupby('year')['msize']
        .transform('mean')
    )

    return trd_mnth


def make_output_table(trd_mnth):
    """整理最终输出表。"""
    result = trd_mnth[[
        'Trdmnt',
        'marketCap_90th_percentile',
        'marketCap_85th_percentile',
        'marketCap_95th_percentile',
        'marketCap_50th_percentile',
        'marketCap_25th_percentile',
        'marketCap_10th_percentile',
        'marketCapMean',
    ]].copy()

    return result.drop_duplicates()


def main():
    trd_mnth = read_and_prepare_data(TRD_Mnth_file)

    trd_mnth = calculate_market_cap_statistics(trd_mnth)

    result = make_output_table(trd_mnth)

    out_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        out_file,
        index=False,
        encoding='utf-8-sig',
    )

    print(f'已输出月度市值分组数据：{out_file}')
    print(f'共{len(result)}行')


if __name__ == '__main__':
    main()


