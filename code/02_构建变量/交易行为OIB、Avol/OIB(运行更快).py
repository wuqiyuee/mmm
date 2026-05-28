# =============================================================================
# CSMAR OIB 数据处理：基于 td 构造后续实际使用变量
#
# 最终保留变量：
#   Stkcd
#   Trddt
#   td
#   avolS
#   avol
#   OIBvol_avg_t_tp1 到 OIBvol_avg_t_tp50
#   OIBnum_avg_t_tp1
#   OIBnum_avg_t_tp10
#
# 最终列数：
#   5 + 50 + 2 = 57 列
#
# 说明：
#   1. OIBvol / OIBnum 仍基于小单；
#   2. avolS 基于小单成交量：
#        volS = B_Volume_S + S_Volume_S
#   3. avol 基于日度总成交股数：
#        vol = Dnshrtrd
#   4. 所有过去窗口和未来窗口仍严格基于 (Stkcd, td) 做 td-i / td+i；
#   5. 不生成 avolS_avg_t_tp*，因为后续代码没有使用。
#   6. 停牌 / 无日度交易记录的 volS 和 vol 设为 0；
#      正常交易但 OIB 或 Dnshrtrd 缺失仍保留 NaN。
#   7. 变量先在 full_panel 完整个股-市场交易日面板上构造，
#      后续回归样本由与清洗后的收益率 / 事件样本合并后确定。
#      因此 OIBvol_avg_t_tp10 可能使用 t 到 t+10 中未进入最终回归样本的日期，
#      这与先构造日度交易行为变量、再合并到回归样本的口径一致。
# =============================================================================

import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 基础设置
# =============================================================================

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

ROOT = Path('/Users/qiyuewu/Desktop/mmm复现数据包')

HF_BSImbalance_file = ROOT / '中间结果/00_合并原始数据/CSMAR_个股买卖不平衡指标表_日_合并结果.csv'
TRD_Dalyr_file = ROOT / '中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_2000_2025.csv'
panel_file = ROOT / '中间结果/01_清洗后样本/full_panel.csv'

out_file = ROOT / '中间结果/02_变量构造结果/交易行为OIB、Avol/OIB_CSMAR_prep.csv'
fig_file = ROOT / '描述性统计图/OIB_table1.png'

out_file.parent.mkdir(parents=True, exist_ok=True)
fig_file.parent.mkdir(parents=True, exist_ok=True)

# 后续实际使用：
# Figure 3 / 动态图：OIBvol_avg_t_tp1 到 OIBvol_avg_t_tp50
# 主要表格 / 稳健性：OIBnum_avg_t_tp1 和 OIBnum_avg_t_tp10
OIBVOL_HORIZONS = list(range(1, 51))
OIBNUM_HORIZONS = [1, 10]


# =============================================================================
# 工具函数
# =============================================================================

def print_memory(df, name):
    """打印 DataFrame 内存占用，方便检查优化效果。"""
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f'{name} 内存占用：{mem_mb:,.2f} MB')


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


def calc_oib(buy, sell):
    """
    OIB = (buy - sell) / (buy + sell) * 100

    当 buy + sell <= 0 或缺失时，设为 NaN。
    """
    buy = pd.to_numeric(buy, errors='coerce')
    sell = pd.to_numeric(sell, errors='coerce')

    denom = buy + sell
    out = (buy - sell) / denom * 100
    out = out.where(denom > 0)

    return out.astype('float32')


def check_unique_key(df, key_cols, name):
    """检查主键是否唯一。"""
    dup = df.duplicated(subset=key_cols).sum()
    if dup > 0:
        raise ValueError(f'{name} 存在 {dup} 条 {key_cols} 重复记录')


def make_forward_avg_by_td_dense(df, value_col, prefix, horizons, dtype=np.float32):
    """
    基于 (Stkcd, td) 精确计算未来窗口均值。

    对每一行 t：
        prefix_avg_t_tp1  = mean(value_t, value_{t+1})
        prefix_avg_t_tp10 = mean(value_t, ..., value_{t+10})
        prefix_avg_t_tp50 = mean(value_t, ..., value_{t+50})

    严格口径：
        只要窗口 [t, t+h] 内任意一个 td 不存在，或 value 缺失，
        该窗口均值就是 NaN。

    注意：
        这里不是按 DataFrame 行号 rolling，而是按 td 的数值位置计算。
    """
    horizons = sorted(set(int(h) for h in horizons))

    if len(horizons) == 0:
        return pd.DataFrame(index=df.index)

    if min(horizons) < 1:
        raise ValueError('horizons 必须从 1 开始')

    n = len(df)
    td_all = df['td'].to_numpy(dtype=np.int64, copy=False)

    value_all = (
        pd.to_numeric(df[value_col], errors='coerce')
        .to_numpy(dtype=np.float64, copy=False)
    )

    out = {
        f'{prefix}_avg_t_tp{h}': np.full(n, np.nan, dtype=dtype)
        for h in horizons
    }

    row_groups = df.groupby('Stkcd', sort=False, observed=True).indices

    for _, idx in row_groups.items():
        idx = np.asarray(idx, dtype=np.int64)

        td = td_all[idx]
        x = value_all[idx]

        min_td = int(td.min())
        max_td = int(td.max())
        span = max_td - min_td + 1

        pos = (td - min_td).astype(np.int64, copy=False)

        dense = np.full(span, np.nan, dtype=np.float64)
        dense[pos] = x

        valid = np.isfinite(dense)
        dense0 = np.where(valid, dense, 0.0)

        cs = np.empty(span + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(dense0, out=cs[1:])

        cv = np.empty(span + 1, dtype=np.int32)
        cv[0] = 0
        np.cumsum(valid.astype(np.int32), out=cv[1:])

        for h in horizons:
            w = h + 1
            col = f'{prefix}_avg_t_tp{h}'

            out_dense = np.full(span, np.nan, dtype=np.float64)

            if span >= w:
                sums = cs[w:] - cs[:-w]
                counts = cv[w:] - cv[:-w]

                vals = sums / w
                vals[counts != w] = np.nan

                # vals[j] 对应 dense[j] 到 dense[j+h] 的均值
                out_dense[:span - w + 1] = vals

            out[col][idx] = out_dense[pos].astype(dtype, copy=False)

    return pd.DataFrame(out, index=df.index)


def past_mean_by_td_dense(df, value_col, window=252, dtype=np.float32):
    """
    基于 (Stkcd, td) 精确计算过去 window 个 td 的均值，不含当天 t。

    对 td = k 的观测，使用：
        value_{k-1}, value_{k-2}, ..., value_{k-window}

    严格口径：
        过去 window 个 td 中任意一个 td 不存在，或 value 缺失，
        结果就是 NaN。
    """
    n = len(df)
    td_all = df['td'].to_numpy(dtype=np.int64, copy=False)

    value_all = (
        pd.to_numeric(df[value_col], errors='coerce')
        .to_numpy(dtype=np.float64, copy=False)
    )

    out = np.full(n, np.nan, dtype=dtype)

    row_groups = df.groupby('Stkcd', sort=False, observed=True).indices

    for _, idx in row_groups.items():
        idx = np.asarray(idx, dtype=np.int64)

        td = td_all[idx]
        x = value_all[idx]

        min_td = int(td.min())
        max_td = int(td.max())
        span = max_td - min_td + 1

        pos = (td - min_td).astype(np.int64, copy=False)

        dense = np.full(span, np.nan, dtype=np.float64)
        dense[pos] = x

        valid = np.isfinite(dense)
        dense0 = np.where(valid, dense, 0.0)

        cs = np.empty(span + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(dense0, out=cs[1:])

        cv = np.empty(span + 1, dtype=np.int32)
        cv[0] = 0
        np.cumsum(valid.astype(np.int32), out=cv[1:])

        out_dense = np.full(span, np.nan, dtype=np.float64)

        if span > window:
            # 对位置 p，过去 window 日为 [p-window, p-1]
            sums = cs[window:span] - cs[:span - window]
            counts = cv[window:span] - cv[:span - window]

            vals = sums / window
            vals[counts != window] = np.nan

            out_dense[window:] = vals

        out[idx] = out_dense[pos].astype(dtype, copy=False)

    return out


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


# =============================================================================
# 1. 读取 CSMAR 买卖不平衡数据
# =============================================================================

HF_BSImbalance = pd.read_csv(
    HF_BSImbalance_file,
    encoding='utf-8',
    dtype={'Stkcd': 'string'},
    usecols=[
        'Stkcd',
        'Trddt',
        'B_Num_S',
        'S_Num_S',
        'B_Volume_S',
        'S_Volume_S',
    ]
)

HF_BSImbalance['Stkcd'] = (
    HF_BSImbalance['Stkcd']
    .map(normalize_stock_code)
    .astype('string')
)

HF_BSImbalance['Trddt'] = pd.to_datetime(HF_BSImbalance['Trddt'])

numeric_cols = [
    'B_Num_S',
    'S_Num_S',
    'B_Volume_S',
    'S_Volume_S',
]

for c in numeric_cols:
    HF_BSImbalance[c] = pd.to_numeric(HF_BSImbalance[c], errors='coerce')

print_memory(HF_BSImbalance, 'HF_BSImbalance 原始读入后')


# =============================================================================
# 2. 处理负的小单成交量
# =============================================================================

print("=" * 80)
print("负成交量处理前：")
print("=" * 80)
print("B_Volume_S < 0 数量：", (HF_BSImbalance['B_Volume_S'] < 0).sum())
print("S_Volume_S < 0 数量：", (HF_BSImbalance['S_Volume_S'] < 0).sum())
print("B_Volume_S 缺失数量：", HF_BSImbalance['B_Volume_S'].isna().sum())
print("S_Volume_S 缺失数量：", HF_BSImbalance['S_Volume_S'].isna().sum())

HF_BSImbalance.loc[
    HF_BSImbalance['B_Volume_S'] < 0,
    'B_Volume_S'
] = np.nan

HF_BSImbalance.loc[
    HF_BSImbalance['S_Volume_S'] < 0,
    'S_Volume_S'
] = np.nan

print("\n" + "=" * 80)
print("负成交量处理后：")
print("=" * 80)
print("B_Volume_S < 0 数量：", (HF_BSImbalance['B_Volume_S'] < 0).sum())
print("S_Volume_S < 0 数量：", (HF_BSImbalance['S_Volume_S'] < 0).sum())
print("B_Volume_S 缺失数量：", HF_BSImbalance['B_Volume_S'].isna().sum())
print("S_Volume_S 缺失数量：", HF_BSImbalance['S_Volume_S'].isna().sum())


# =============================================================================
# 3. 计算 OIB
# =============================================================================

HF_BSImbalance['OIBvol'] = calc_oib(
    HF_BSImbalance['B_Volume_S'],
    HF_BSImbalance['S_Volume_S']
)

HF_BSImbalance['OIBnum'] = calc_oib(
    HF_BSImbalance['B_Num_S'],
    HF_BSImbalance['S_Num_S']
)

print("=" * 80)
print("OIB 异常值检查")
print("=" * 80)
print("|OIBvol| > 100 数量：", (HF_BSImbalance['OIBvol'].abs() > 100).sum())
print("|OIBnum| > 100 数量：", (HF_BSImbalance['OIBnum'].abs() > 100).sum())
print("OIBvol 为 inf 数量：", np.isinf(HF_BSImbalance['OIBvol']).sum())
print("OIBnum 为 inf 数量：", np.isinf(HF_BSImbalance['OIBnum']).sum())
print("OIBvol 缺失数量：", HF_BSImbalance['OIBvol'].isna().sum())
print("OIBnum 缺失数量：", HF_BSImbalance['OIBnum'].isna().sum())

# 后面需要：
#   B_Volume_S / S_Volume_S：计算小单 volS 和 avolS
#   OIBvol / OIBnum：计算未来窗口 OIB
#   OIBvol：计算 OIBvol_avg_t_tp*
#   OIBnum：计算 OIBnum_avg_t_tp1 和 OIBnum_avg_t_tp10
HF_BSImbalance = HF_BSImbalance[
    [
        'Stkcd',
        'Trddt',
        'B_Volume_S',
        'S_Volume_S',
        'OIBvol',
        'OIBnum',
    ]
].copy()

for c in [
    'B_Volume_S',
    'S_Volume_S',
    'OIBvol',
    'OIBnum',
]:
    HF_BSImbalance[c] = HF_BSImbalance[c].astype('float32')

print_memory(HF_BSImbalance, 'HF_BSImbalance 精简后')


# =============================================================================
# 4. 读取日度交易状态和总成交量
# =============================================================================

TRD_Dalyr_trade = pd.read_csv(
    TRD_Dalyr_file,
    encoding='utf-8',
    dtype={'Stkcd': 'string'},
    usecols=['Stkcd', 'Trddt', 'Markettype', 'Filling', 'Trdsta', 'Dnshrtrd']
)

TRD_Dalyr_trade['Stkcd'] = (
    TRD_Dalyr_trade['Stkcd']
    .map(normalize_stock_code)
    .astype('string')
)

TRD_Dalyr_trade['Trddt'] = pd.to_datetime(TRD_Dalyr_trade['Trddt'])

for c in ['Markettype', 'Filling', 'Trdsta', 'Dnshrtrd']:
    TRD_Dalyr_trade[c] = pd.to_numeric(TRD_Dalyr_trade[c], errors='coerce')

TRD_Dalyr_trade = TRD_Dalyr_trade.loc[
    TRD_Dalyr_trade['Markettype'].isin([1, 4, 16, 32])
    & TRD_Dalyr_trade['Filling'].isin([0, 1])
].copy()

check_unique_key(TRD_Dalyr_trade, ['Stkcd', 'Trddt'], 'TRD_Dalyr_trade')

TRD_Dalyr_trade['Filling'] = TRD_Dalyr_trade['Filling'].astype('int8')
TRD_Dalyr_trade['Markettype'] = TRD_Dalyr_trade['Markettype'].astype('int8')
TRD_Dalyr_trade['Trdsta'] = TRD_Dalyr_trade['Trdsta'].astype('float32')
TRD_Dalyr_trade['Dnshrtrd'] = TRD_Dalyr_trade['Dnshrtrd'].astype('float32')

print_memory(TRD_Dalyr_trade, 'TRD_Dalyr_trade 精简后')


# =============================================================================
# 5. 读取完整个股交易日面板，只读必要列
# =============================================================================

stock_day_panel = pd.read_csv(
    panel_file,
    encoding='utf-8',
    dtype={'Stkcd': 'string'},
    usecols=['Stkcd', 'Trddt', 'td']
)

stock_day_panel['Stkcd'] = (
    stock_day_panel['Stkcd']
    .map(normalize_stock_code)
    .astype('string')
)

stock_day_panel['Trddt'] = pd.to_datetime(stock_day_panel['Trddt'])
stock_day_panel['td'] = pd.to_numeric(stock_day_panel['td'], errors='coerce')

if stock_day_panel['td'].isna().any():
    raise ValueError('stock_day_panel 中存在 td 缺失，不能安全基于 td 计算窗口')

stock_day_panel['td'] = stock_day_panel['td'].astype('int32')

print_memory(stock_day_panel, 'stock_day_panel 只读取必要列后')

print("=" * 80)
print("重复值检查")
print("=" * 80)
print("stock_day_panel 整行重复数量：", stock_day_panel.duplicated().sum())
print(
    "stock_day_panel (Stkcd, Trddt) 重复数量：",
    stock_day_panel.duplicated(subset=['Stkcd', 'Trddt']).sum()
)
print(
    "HF_BSImbalance (Stkcd, Trddt) 重复数量：",
    HF_BSImbalance.duplicated(subset=['Stkcd', 'Trddt']).sum()
)

check_unique_key(stock_day_panel, ['Stkcd', 'Trddt'], 'stock_day_panel')
check_unique_key(HF_BSImbalance, ['Stkcd', 'Trddt'], 'HF_BSImbalance')

stock_day_panel = pd.merge(
    stock_day_panel,
    TRD_Dalyr_trade,
    on=['Stkcd', 'Trddt'],
    how='left',
    indicator='_merge_daily',
    validate='one_to_one'
)

print("=" * 80)
print("日度交易状态 merge 结果：")
print("=" * 80)
print(stock_day_panel['_merge_daily'].value_counts())
print(stock_day_panel['_merge_daily'].value_counts(normalize=True))

stock_day_panel['has_daily_record'] = (
    stock_day_panel['_merge_daily'].eq('both')
)
stock_day_panel = stock_day_panel.drop(columns=['_merge_daily'])

del TRD_Dalyr_trade
gc.collect()


# =============================================================================
# 6. 合并到完整个股交易日面板
# =============================================================================

HF_BSImbalance_td = pd.merge(
    stock_day_panel,
    HF_BSImbalance,
    on=['Stkcd', 'Trddt'],
    how='left',
    indicator=True,
    validate='one_to_one'
)

print("=" * 80)
print("merge 结果：")
print("=" * 80)
print(HF_BSImbalance_td['_merge'].value_counts())
print(HF_BSImbalance_td['_merge'].value_counts(normalize=True))

HF_BSImbalance_td = HF_BSImbalance_td.drop(columns=['_merge'])

del stock_day_panel, HF_BSImbalance
gc.collect()

HF_BSImbalance_td = (
    HF_BSImbalance_td
    .sort_values(['Stkcd', 'td'])
    .reset_index(drop=True)
)

check_unique_key(HF_BSImbalance_td, ['Stkcd', 'td'], 'HF_BSImbalance_td')

HF_BSImbalance_td['Stkcd'] = HF_BSImbalance_td['Stkcd'].astype('category')

print_memory(HF_BSImbalance_td, 'HF_BSImbalance_td merge 后')


# =============================================================================
# 7. 基于 td 计算未来平均 OIBvol 和 OIBnum
# =============================================================================

print("=" * 80)
print("开始计算未来平均 OIBvol / OIBnum，严格基于 td+i")
print("=" * 80)

oibvol_avg = make_forward_avg_by_td_dense(
    HF_BSImbalance_td,
    value_col='OIBvol',
    prefix='OIBvol',
    horizons=OIBVOL_HORIZONS,
    dtype=np.float32
)

oibnum_avg = make_forward_avg_by_td_dense(
    HF_BSImbalance_td,
    value_col='OIBnum',
    prefix='OIBnum',
    horizons=OIBNUM_HORIZONS,
    dtype=np.float32
)

HF_BSImbalance_td = pd.concat(
    [HF_BSImbalance_td, oibvol_avg, oibnum_avg],
    axis=1
)

del oibvol_avg, oibnum_avg
gc.collect()

print_memory(HF_BSImbalance_td, '加入 OIB 未来窗口均值后')


# =============================================================================
# 8. 计算 abnormal volume：小单口径 avolS 与日度总成交量口径 avol
# =============================================================================

print("=" * 80)
print("开始计算 abnormal volume：avolS 小单口径，avol 日度总成交量口径")
print("=" * 80)

no_daily_record = ~HF_BSImbalance_td['has_daily_record'].fillna(False)
filling_stop = HF_BSImbalance_td['Filling'].eq(1)
no_trade_day = no_daily_record | filling_stop

print("full_panel 中未匹配到日度交易记录数量：", no_daily_record.sum())
print("Filling = 1 停牌 / 暂停上市数量：", filling_stop.sum())
print("停牌或无日度交易记录数量：", no_trade_day.sum())

# 小单成交量：停牌 / 无交易日设为 0；正常交易但 OIB 缺失仍为 NaN。
raw_volS = (
    HF_BSImbalance_td['B_Volume_S'] + HF_BSImbalance_td['S_Volume_S']
).astype('float32')

zero_vol_day = HF_BSImbalance_td['Dnshrtrd'].eq(0)
volS_zero_day = no_trade_day | zero_vol_day

print(
    "正常交易但小单成交量缺失数量：",
    ((~volS_zero_day) & raw_volS.isna()).sum()
)

HF_BSImbalance_td['volS'] = raw_volS
HF_BSImbalance_td.loc[volS_zero_day, 'volS'] = 0.0

HF_BSImbalance_td['volS_lag252_mean'] = past_mean_by_td_dense(
    HF_BSImbalance_td,
    value_col='volS',
    window=252,
    dtype=np.float32
)

denom_s = HF_BSImbalance_td['volS_lag252_mean']

HF_BSImbalance_td['avolS'] = np.where(
    denom_s > 0,
    HF_BSImbalance_td['volS'] / denom_s,
    np.nan
).astype('float32')


# 日度总成交量：停牌 / 无交易日设为 0；正常交易但 Dnshrtrd 缺失仍为 NaN。
raw_vol = HF_BSImbalance_td['Dnshrtrd'].astype('float32')

print(
    "正常交易但 Dnshrtrd 缺失数量：",
    ((~no_trade_day) & raw_vol.isna()).sum()
)

HF_BSImbalance_td['vol'] = raw_vol
HF_BSImbalance_td.loc[no_trade_day, 'vol'] = 0.0

HF_BSImbalance_td['vol_lag252_mean'] = past_mean_by_td_dense(
    HF_BSImbalance_td,
    value_col='vol',
    window=252,
    dtype=np.float32
)

denom = HF_BSImbalance_td['vol_lag252_mean']

HF_BSImbalance_td['avol'] = np.where(
    denom > 0,
    HF_BSImbalance_td['vol'] / denom,
    np.nan
).astype('float32')


print("volS < 0 数量：", (HF_BSImbalance_td['volS'] < 0).sum())
print("avolS < 0 数量：", (HF_BSImbalance_td['avolS'] < 0).sum())
print("avolS 为 inf 数量：", np.isinf(HF_BSImbalance_td['avolS']).sum())

print("vol < 0 数量：", (HF_BSImbalance_td['vol'] < 0).sum())
print("avol < 0 数量：", (HF_BSImbalance_td['avol'] < 0).sum())
print("avol 为 inf 数量：", np.isinf(HF_BSImbalance_td['avol']).sum())


# 删除中间变量和原始成交量 / 状态列，最终只保留 avolS 与 avol
drop_after_avol = [
    'B_Volume_S',
    'S_Volume_S',
    'Markettype',
    'Filling',
    'Trdsta',
    'Dnshrtrd',
    'has_daily_record',
    'volS',
    'volS_lag252_mean',
    'vol',
    'vol_lag252_mean',
]

HF_BSImbalance_td = HF_BSImbalance_td.drop(
    columns=[c for c in drop_after_avol if c in HF_BSImbalance_td.columns]
)

gc.collect()

print_memory(HF_BSImbalance_td, '加入 avolS / avol 并删除中间变量后')


# =============================================================================
# 8. 最终只保留后续代码实际使用变量 + 新增 avol
# =============================================================================

final_cols = (
    ['Stkcd', 'Trddt', 'td', 'avolS', 'avol']
    + [f'OIBvol_avg_t_tp{i}' for i in OIBVOL_HORIZONS]
    + [f'OIBnum_avg_t_tp{i}' for i in OIBNUM_HORIZONS]
)

missing_cols = [c for c in final_cols if c not in HF_BSImbalance_td.columns]

if missing_cols:
    raise ValueError(f'最终需要保留的变量不存在：{missing_cols}')

HF_BSImbalance_td = HF_BSImbalance_td[final_cols].copy()

print("=" * 80)
print("最终保留变量：")
print("=" * 80)
print(HF_BSImbalance_td.columns.tolist())


# =============================================================================
# 9. 最终异常检查
# =============================================================================

print("=" * 80)
print("最终异常检查")
print("=" * 80)

print(
    "OIBvol_avg_t_tp1 inf 数量：",
    np.isinf(HF_BSImbalance_td['OIBvol_avg_t_tp1']).sum()
)

print(
    "OIBnum_avg_t_tp1 inf 数量：",
    np.isinf(HF_BSImbalance_td['OIBnum_avg_t_tp1']).sum()
)

print(
    "OIBnum_avg_t_tp10 inf 数量：",
    np.isinf(HF_BSImbalance_td['OIBnum_avg_t_tp10']).sum()
)

print(
    "avolS inf 数量：",
    np.isinf(HF_BSImbalance_td['avolS']).sum()
)

print(
    "avol inf 数量：",
    np.isinf(HF_BSImbalance_td['avol']).sum()
)

for v in [
    'OIBvol_avg_t_tp1',
    'OIBvol_avg_t_tp10',
    'OIBnum_avg_t_tp1',
    'OIBnum_avg_t_tp10',
    'avolS',
    'avol',
]:
    s = pd.to_numeric(HF_BSImbalance_td[v], errors='coerce')

    print("\n变量：", v)
    print("总行数：", len(s))
    print("非缺失数：", s.notna().sum())
    print("缺失数：", s.isna().sum())

    if v.startswith('OIB'):
        print("等于 100 的数量：", (s == 100).sum())
        print("等于 -100 的数量：", (s == -100).sum())
        print("大于 99 的数量：", (s > 99).sum())
        print("小于 -99 的数量：", (s < -99).sum())


# =============================================================================
# 10. 保存中间结果
# =============================================================================

print("=" * 80)
print("最终数据形状：", HF_BSImbalance_td.shape)
print("最终列数：", len(HF_BSImbalance_td.columns))
print_memory(HF_BSImbalance_td, '最终结果')

print("正在输出....")

HF_BSImbalance_td.to_csv(
    out_file,
    index=False,
    encoding='utf-8-sig'
)

print(f"结果已经保存到：{out_file}")


# =============================================================================
# 11. 生成类似 Table 1 的 OIB 描述性统计表格图片
# =============================================================================

vars_for_table = {
    'Retail OI in % over [t,t+1]': HF_BSImbalance_td['OIBvol_avg_t_tp1'],
    'Retail OI in % over [t,t+10]': HF_BSImbalance_td['OIBvol_avg_t_tp10'],
}

rows = {}

for name, s in vars_for_table.items():
    rows[name] = summary_row(s)

table1 = pd.DataFrame(rows).T

table1_print = table1.copy()

for col in ['Mean', 'p25', 'p50', 'p75', 'Std.Dev.', 'Min', 'Max']:
    table1_print[col] = table1_print[col].map(lambda x: f'{x:,.4f}')

table1_print['N'] = table1_print['N'].map(lambda x: f'{int(x):,}')

print("\nTable 1 风格描述性统计：")
print(table1_print)

show_df = table1_print.reset_index().rename(columns={'index': 'Variable'})

fig, ax = plt.subplots(figsize=(15, 2.8))
ax.axis('off')

ax.set_title(
    'Descriptive Statistics of Retail Order Imbalance',
    loc='left',
    pad=6
)

col_widths = [
    0.28,
    0.09,
    0.09,
    0.09,
    0.09,
    0.10,
    0.09,
    0.09,
    0.09,
]

tbl = ax.table(
    cellText=show_df.values,
    colLabels=show_df.columns,
    loc='upper center',
    cellLoc='center',
    colLoc='center',
    colWidths=col_widths
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

for i in range(len(show_df) + 1):
    cell = tbl[(i, 0)]
    cell.set_text_props(ha='left')

plt.subplots_adjust(
    left=0.02,
    right=0.98,
    top=0.82,
    bottom=0.08
)

plt.savefig(fig_file, dpi=300, bbox_inches='tight')
plt.close()

print(f'表格图片已保存：{fig_file}')


# =============================================================================
# 最终列说明
# =============================================================================
# 基础列：
#   Stkcd, Trddt, td, avolS, avol
#
# 其中：
#   avolS = 小单成交量 abnormal volume
#   avol  = 日度总成交股数 abnormal volume，基于 Dnshrtrd
#
# 未来 OIBvol：
#   OIBvol_avg_t_tp1 到 OIBvol_avg_t_tp50
#
# 未来 OIBnum：
#   OIBnum_avg_t_tp1, OIBnum_avg_t_tp10
#
# 合计：
#   5 + 50 + 2 = 57 列
# =============================================================================
