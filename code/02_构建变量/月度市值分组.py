import pandas as pd
from pathlib import Path
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

ROOT = Path('/Users/qiyuewu/Desktop/mmm复现数据包')

TRD_Mnth_file = ROOT/'input/CSMAR/月个股回报率文件2000-01至2025-12/TRD_Mnth.csv'
out_file = ROOT/'中间结果/02_变量构造结果/月度市值分组.csv'

out_file.parent.mkdir(parents=True,exist_ok=True)

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

TRD_Mnth = pd.read_csv(
    TRD_Mnth_file,
    encoding='utf-8',
    usecols=['Stkcd','Trdmnt','Msmvosd','Msmvttl','Markettype']
)

TRD_Mnth['Stkcd'] = TRD_Mnth['Stkcd'].apply(normalize_stock_code)
TRD_Mnth['Trdmnt'] = pd.to_datetime(
    TRD_Mnth['Trdmnt'],
    format='%Y-%m',
    errors='coerce'
)
TRD_Mnth = TRD_Mnth.rename(columns={
    'Msmvosd':'msizew',
    'Msmvttl':'msize'
})#单位是千

TRD_Mnth1 = TRD_Mnth.loc[
    TRD_Mnth['Markettype'].isin([1,4])
]
# 表示对

TRD_Mnth1['year'] = TRD_Mnth1['Trdmnt'].dt.year
TRD_Mnth1['month'] = TRD_Mnth1['Trdmnt'].dt.month

percentiles = {
    "90th":0.90,
    "85th":0.85,
    "95th":0.95,
    "50th":0.50,
    "25th":0.25,
    "10th":0.10,
}

for name,q in percentiles.items():
    TRD_Mnth1[f"marketCap_{name}_percentile"]=(
        TRD_Mnth1.groupby('Trdmnt')['msize']
        .transform(lambda x: x.quantile(q))
    )

TRD_Mnth1['marketCapMean']=(
    TRD_Mnth1.groupby('year')['msize']
    .transform('mean')
)

marketCap_pctiles_prep = TRD_Mnth1[[
    'Trdmnt',
    "marketCap_90th_percentile",
    "marketCap_85th_percentile",
    "marketCap_95th_percentile",
    "marketCap_50th_percentile",
    "marketCap_25th_percentile",
    "marketCap_10th_percentile",
    "marketCapMean",
]].copy()

marketCap_pctiles_prep = marketCap_pctiles_prep.drop_duplicates()

marketCap_pctiles_prep.to_csv(
   out_file,
   index = False,
   encoding= 'utf-8-sig'
)


print(f'已输出月度市值分组数据：{out_file}')
print(f'共{len(marketCap_pctiles_prep)}行')