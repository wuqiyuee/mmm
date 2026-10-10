# =============================================================================
# CSMAR OIB 数据处理：基于 td 构造后续实际使用变量
#
# 最终分别输出两张表：
#
# 1. OIB_CSMAR_prep.csv
#    键变量：
#      Stkcd, Trddt, td
#    单日指标：
#      OIBvol, OIBnum, OIBamount, OIBorder
#    未来窗口指标：
#      OIBvol_avg_t_tp1 到 OIBvol_avg_t_tp50
#      OIBnum_avg_t_tp1, OIBnum_avg_t_tp10
#      OIBamount_avg_t_tp1, OIBamount_avg_t_tp10
#      OIBorder_avg_t_tp1, OIBorder_avg_t_tp10
#    共63列：
#      3 + 4 + 50 + 2 + 2 + 2 = 63
#
# 2. Avol_CSMAR_prep.csv
#    Stkcd, Trddt, td, avolS, avolS_min240,
#    volS_lag252_nvalid, avol
#    共7列。
#
# 说明：
#   1. 四个 OIB 指标均基于小单交易数据。
#   2. avolS 基于小单成交量：
#        volS = B_Volume_S + S_Volume_S
#   3. avol 基于日度总成交股数：
#        vol = Dnshrtrd
#   4. OIB 未来窗口、avol 和 avolS 保持严格口径；
#      avolS_min240 使用过去固定252日中至少240日有效的放宽口径。
#   5. 不生成 avolS_avg_t_tp*，因为后续代码没有使用。
#   6. 停牌 / 无日度交易记录的 volS 和 vol 设为 0；
#      正常交易但 OIB 或 Dnshrtrd 缺失仍保留 NaN。
# 7. OIB 与异常成交量采用不同的计算面板：
#    四个单日 OIB 只保留基础清洗后的真实交易日,
#    OIB 未来均值在窗口变量目标日资格索引上严格按照 td 计算，
#    窗口内任一交易日或 OIB 缺失，整个窗口结果均为缺失；
#
# 原始买卖数据
#     ↓
# 计算四个单日 OIB：OIBvol、OIBnum、OIBamount、OIBorder
#     ├─ 合并基础清洗样本索引 → 单日 OIB
#     └─ 合并窗口资格索引 → 严格计算 OIB_avg
#                               ↓
#                     再按基础样本筛选起始日 t

# 基础样本单日 OIB + 窗口面板 OIB_avg
#                     ↓
#           OIB_CSMAR_prep.csv

# 完整交易日面板计算 avol / avolS
#                     ↓
#          Avol_CSMAR_prep.csv
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
TRD_Dalyr_file = ROOT / '中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv'
panel_file = ROOT / '中间结果/01_清洗后样本/完整个股交易日面板索引_td.csv'
window_target_index_file = (
    ROOT / '中间结果/01_清洗后样本/'
    '窗口变量目标日资格索引_基础清洗含停牌填充_td.csv'
)

base_sample_index_file = (
    ROOT / '中间结果/01_清洗后样本/'
    '基础清洗后个股日度样本索引_td.csv'
)


OIB_AVOL_DIR = ROOT / "中间结果/02_变量构造结果/2交易行为OIB、Avol"

oib_out_file = OIB_AVOL_DIR / "OIB_CSMAR_prep.csv"
avol_out_file = OIB_AVOL_DIR / "Avol_CSMAR_prep.csv"

fig_file = ROOT / '图表结果/辅助分析/OIB_描述性统计.png'

# 后续实际使用：
# Figure 3 / 动态图：OIBvol_avg_t_tp1 到 OIBvol_avg_t_tp50
# 主要表格 / 稳健性：OIBnum_avg_t_tp1 和 OIBnum_avg_t_tp10
OIBVOL_HORIZONS = list(range(1, 51))
OIBNUM_HORIZONS = [1, 10]
OIBAMOUNT_HORIZONS = [1, 10]
OIBORDER_HORIZONS = [1, 10]

AVOL_WINDOW = 252
AVOLS_MIN_VALID_DAYS = 240  # 240 / 252 = 95.24%
OUTPUT_START_DATE = pd.Timestamp('2000-01-01')
OUTPUT_END_DATE = pd.Timestamp('2025-12-31')
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


def past_mean_by_td_dense(
        df,
        value_col, 
        window=252, 
        dtype=np.float32,
        min_valid=None,
        return_count=False,
):
    """
    基于 (Stkcd, td) 精确计算过去 window 个 td 的均值，不含当天 t。

    对 td = k 的观测，使用：
        value_{k-1}, value_{k-2}, ..., value_{k-window}

    计算规则：
    1. 必须已经具备完整的 window 个历史 td；
    2. min_valid=None 时，要求 window 个历史值全部有效；
    3. 指定 min_valid 时，至少有 min_valid 个有效值即可计算；
    4. 均值分母为窗口内实际有效观测数；
    5. return_count=True 时，同时返回窗口内有效观测数。
    """
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window 必须是正整数")

    if min_valid is None:
        min_valid = window

    if (
        not isinstance(min_valid, (int, np.integer))
        or not 1 <= min_valid <= window
    ):
        raise ValueError("min_valid 必须是 1 到 window 之间的整数")

    window = int(window)
    min_valid = int(min_valid)

    n = len(df)
    td_all = df['td'].to_numpy(dtype=np.int64, copy=False)

    value_all = (
        pd.to_numeric(df[value_col], errors='coerce')
        .to_numpy(dtype=np.float64, copy=False)
    )

    # 保存过去窗口均值
    out = np.full(n, np.nan, dtype=dtype)

    # 只有需要返回有效日数时才创建，避免不必要的内存占用
    count_out = (
        np.full(n, np.nan, dtype=np.float32)
        if return_count
        else None
    )

    row_groups = df.groupby('Stkcd', sort=False, observed=True).indices

    for _, idx in row_groups.items():
        idx = np.asarray(idx, dtype=np.int64)

        td = td_all[idx]
        x = value_all[idx]

        min_td = int(td.min())
        max_td = int(td.max())
        span = max_td - min_td + 1

        pos = (td - min_td).astype(np.int64, copy=False)

        # 构造连续交易日序号数组
        dense = np.full(span, np.nan, dtype=np.float64)
        dense[pos] = x

        valid = np.isfinite(dense)
        dense0 = np.where(valid, dense, 0.0)

        # 数值累计和
        cs = np.empty(span + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(dense0, out=cs[1:])

        # 有效观测数累计和
        cv = np.empty(span + 1, dtype=np.int32)
        cv[0] = 0
        np.cumsum(valid.astype(np.int32), out=cv[1:])

        out_dense = np.full(span, np.nan, dtype=np.float64)

        if return_count:
            count_dense = np.full(
                span,
                np.nan,
                dtype=np.float32,
            )

        if span > window:
            # 对位置 p，过去 window 日为 [p-window, p-1]
            # 因而不包含当天 p
            sums = cs[window:span] - cs[:span - window]
            counts = cv[window:span] - cv[:span - window]

            # 只有有效日数达到最低要求才计算均值
            eligible = counts >= min_valid
            vals = np.full(
                sums.shape,
                np.nan,
                dtype=np.float64,
            )

            # 分母使用实际有效日数
            vals[eligible] = (
                sums[eligible]
                / counts[eligible]
            )

            out_dense[window:] = vals

            if return_count:
                count_dense[window:] = counts

        out[idx] = out_dense[pos].astype(dtype, copy=False)
        if return_count:
                    count_out[idx] = count_dense[pos]

    if return_count:
        return out, count_out

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


def load_prepare_oib(file_path):
    """读取并整理 CSMAR 小单买卖不平衡数据。
    
    处理内容：
    1. 标准化股票代码和交易日期；
    2. 将原始指标转换为数值；
    3. 将负的小单成交量、成交金额、成交笔数、成交次数处理为缺失值；
    4. 计算 OIBvol 、 OIBnum 、 OIBamount 、 OIBorder;
    5. 删除不再使用的原始变量并压缩数据类型。
    """

# =============================================================================
# 1. 读取 CSMAR 买卖不平衡数据
# =============================================================================
    HF_BSImbalance = pd.read_csv(
        file_path,
        encoding='utf-8',
        dtype={'Stkcd': 'string'},
        usecols=[
            'Stkcd',
            'Trddt',
            'B_Num_S',
            'S_Num_S',
            'B_Volume_S',
            'S_Volume_S',
            'B_Amount_S',
            'S_Amount_S',
            'B_Order_S',
            'S_Order_S',
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
        'B_Amount_S',
        'S_Amount_S',
        'B_Order_S',
        'S_Order_S',
    ]

    for c in numeric_cols:
        HF_BSImbalance[c] = pd.to_numeric(HF_BSImbalance[c], errors='coerce')


    print_memory(HF_BSImbalance, 'HF_BSImbalance 原始读入后')


    # ==========================================================================
    # 2. 处理原始小单买卖指标中的负值
    # ==========================================================================

    print("=" * 80)
    print("负值处理前：")
    print("=" * 80)

    for c in numeric_cols:
        print(f"{c} < 0 数量：", (HF_BSImbalance[c] < 0).sum())
        print(f"{c} 缺失数量：", HF_BSImbalance[c].isna().sum())


    # 买卖次数、成交量、成交金额和成交笔数均不应为负
    for c in numeric_cols:
        HF_BSImbalance.loc[
            HF_BSImbalance[c] < 0,
            c,
        ] = np.nan


    print("\n" + "=" * 80)
    print("负值处理后：")
    print("=" * 80)

    for c in numeric_cols:
        print(f"{c} < 0 数量：", (HF_BSImbalance[c] < 0).sum())
        print(f"{c} 缺失数量：", HF_BSImbalance[c].isna().sum())
    # ==========================================================================
    # 3. 计算 OIB
    # ==========================================================================

    HF_BSImbalance['OIBvol'] = calc_oib(
        HF_BSImbalance['B_Volume_S'],
        HF_BSImbalance['S_Volume_S']
    )

    HF_BSImbalance['OIBnum'] = calc_oib(
        HF_BSImbalance['B_Num_S'],
        HF_BSImbalance['S_Num_S']
    )


    HF_BSImbalance['OIBamount'] = calc_oib(
        HF_BSImbalance['B_Amount_S'],
        HF_BSImbalance['S_Amount_S'],
    )

    HF_BSImbalance['OIBorder'] = calc_oib(
        HF_BSImbalance['B_Order_S'],
        HF_BSImbalance['S_Order_S'],
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
    #   B_Volume_S / S_Volume_S：
    #       计算 volS、avolS 和 avolS_min240；
    #   OIBvol / OIBnum / OIBamount / OIBorder：
    #       计算四类订单失衡指标的未来窗口均值。
    HF_BSImbalance = HF_BSImbalance[
        [
            'Stkcd',
            'Trddt',
            'B_Volume_S',
            'S_Volume_S',
            'OIBvol',
            'OIBnum',
            'OIBamount',
            'OIBorder',
        ]
    ].copy()

    for c in [
        'B_Volume_S',
        'S_Volume_S',
        'OIBvol',
        'OIBnum',
        'OIBamount',
        'OIBorder',
    ]:
        HF_BSImbalance[c] = HF_BSImbalance[c].astype('float32')

    print_memory(HF_BSImbalance, 'HF_BSImbalance 精简后')

    return HF_BSImbalance


def load_prepare_daily_trade(file_path):
    """
    读取并整理日度交易状态和总成交量数据。

    处理内容：
    1. 标准化股票代码和交易日期；
    2. 将交易状态和成交量转换为数值；
    3. 保留指定市场及有效填补状态；
    4. 检查股票-交易日主键；
    5. 压缩数据类型。
    """
    # ==========================================================================
    # 4. 读取日度交易状态和总成交量
    # ==========================================================================
    TRD_Dalyr_trade = pd.read_csv(
        file_path,
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

    return TRD_Dalyr_trade


# =============================================================================
# 5. 读取完整个股交易日面板，只读必要列
# =============================================================================
def load_stock_day_panel(file_path):
    """
    读取并整理完整股票交易日面板。
    
    处理内容：
    1. 读取股票代码、交易日期和统一交易日序号；
    2. 标准化股票代码和交易日期；
    3. 检查交易日序号是否缺失；
    4. 压缩交易日序号的数据类型。
    """
    stock_day_panel = pd.read_csv(
        file_path,
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

    return stock_day_panel

def build_full_panel(
        oib_file_path,
        daily_file_path,
        panel_file_path,
):
    """ 
    读取三个输入数据并构造完整股票交易日计算面板。

    处理内容：
    1. 读取买卖不平衡数据；
    2. 读取日度交易状态；
    3. 读取完整股票交易日面板；
    4. 检查股票—交易日主键；
    5. 合并日度状态和买卖不平衡数据；
    6. 按股票和统一交易日序号排序。
    """

    HF_BSImbalance = load_prepare_oib(oib_file_path)
    TRD_Dalyr_trade = load_prepare_daily_trade(daily_file_path)
    stock_day_panel = load_stock_day_panel(panel_file_path)

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

    return HF_BSImbalance_td


def calc_forward_oib_averages(
        df,
        oibvol_horizons,
        oibnum_horizons,
        oibamount_horizons,
        oiborder_horizons,
):
    """
    基于统一交易日序号构造未来 OIB 窗口均值。

    参数：
    df：已经按股票和交易日序号排序的完整计算面板；
    oibvol_horizons：成交量订单失衡的未来窗口；
    oibnum_horizons：买卖次数订单失衡的未来窗口；
    oibamount_horizons：成交金额订单失衡的未来窗口；
    oiborder_horizons：成交笔数订单失衡的未来窗口。

    严格口径：
    对每个起始日 t，计算从 t 至 t+h 的均值；
    窗口内任意一个交易日不存在或对应 OIB 缺失时，
    整个窗口均值设为缺失。
    """
    # =============================================================================
    # 7. 基于 td 计算未来平均 OIBvol 、 OIBnum 、 OIBamount 、 OIBorder
    # =============================================================================

    print("=" * 80)
    print(        
        "开始计算 OIBvol / OIBnum / OIBamount / OIBorder "
        "未来严格窗口均值"
    )
    print("=" * 80)

    oibvol_avg = make_forward_avg_by_td_dense(
        df,
        value_col='OIBvol',
        prefix='OIBvol',
        horizons=oibvol_horizons,
        dtype=np.float32
    )

    oibnum_avg = make_forward_avg_by_td_dense(
        df,
        value_col='OIBnum',
        prefix='OIBnum',
        horizons=oibnum_horizons,
        dtype=np.float32
    )

    oibamount_avg = make_forward_avg_by_td_dense(
    df,
    value_col='OIBamount',
    prefix='OIBamount',
    horizons=oibamount_horizons,
    dtype=np.float32,
)

    oiborder_avg = make_forward_avg_by_td_dense(
        df,
        value_col='OIBorder',
        prefix='OIBorder',
        horizons=oiborder_horizons,
        dtype=np.float32,
    )

    result = pd.concat(
        [df, oibvol_avg, oibnum_avg, oibamount_avg, oiborder_avg,],
        axis=1
    )

    del oibvol_avg, oibnum_avg, oibamount_avg, oiborder_avg
    gc.collect()

    print_memory(result, '加入 OIB 未来窗口均值后')

    return result


def calc_abnormal_volumes(
        df, 
        window,
        avols_min_valid_days,
):
    """
    计算异常小单成交量和异常总成交量。

    参数：
    df：包含成交量、交易状态和统一交易日序号的完整面板；
    window：计算历史平均成交量的窗口长度。
    avols_min_valid_days：avolS 的过去固定窗口中要求的最低有效日数，正式计算时取 240。

    返回：
    增加 avolS、avolS_min240、volS_lag252_nvalid 和 avol，
    并删除临时成交量变量后的面板。
    """

    HF_BSImbalance_td = df
    # =============================================================================
    # 8. 计算 abnormal volume：小单口径 avolS 与日度总成交量口径 avol
    # =============================================================================

    print("=" * 80)
    print("开始计算 abnormal volume：avolS 小单口径，avol 日度总成交量口径")
    print("=" * 80)

    no_daily_record = ~HF_BSImbalance_td['has_daily_record'].fillna(False)
    # 少量完整面板股票日未匹配到日行情且 OIB 表也无记录，视为无交易日，成交量记为 0。
    # 当前仅34个股票日未匹配到日行情，影响极为有限。
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

    # 完整面板已补齐股票首末交易日之间的日期，停牌、无记录及零成交日按 0 处理。
    # 因此，252 日窗口缺失主要来自正常交易日的小单数据缺失，而非交易日断档。
    # 一次计算放宽口径的历史均值和窗口内实际有效日数
    volS_mean_min_valid, volS_nvalid = past_mean_by_td_dense(
        HF_BSImbalance_td,
        value_col='volS',
        window=window,
        dtype=np.float32,
        min_valid=avols_min_valid_days,
        return_count=True,
    )

    # 保存过去固定窗口内的实际有效日数
    # 正式运行时 window=252，因此变量名保留 lag252
    HF_BSImbalance_td['volS_lag252_nvalid'] = (
        volS_nvalid.astype('float32', copy=False)
    )

    # 严格口径：只有过去 window 日全部有效时才保留历史均值
    HF_BSImbalance_td['volS_lag252_mean'] = np.where(
        volS_nvalid == window,
        volS_mean_min_valid,
        np.nan,
    ).astype('float32')

    # 严格口径的 avolS
    denom_s = HF_BSImbalance_td['volS_lag252_mean']

    HF_BSImbalance_td['avolS'] = np.where(
        denom_s > 0,
        HF_BSImbalance_td['volS'] / denom_s,
        np.nan,
    ).astype('float32')

    # 放宽口径的 avolS
    # 正式运行时要求过去252日中至少240日有效
    HF_BSImbalance_td['avolS_min240'] = np.where(
        volS_mean_min_valid > 0,
        (
            HF_BSImbalance_td['volS'].to_numpy(
                dtype=np.float32,
                copy=False,
            )
            / volS_mean_min_valid
        ),
        np.nan,
    ).astype('float32')

    del volS_mean_min_valid, volS_nvalid


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
        window=window,
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


    print(
    "avolS_min240 < 0 数量：",
    (HF_BSImbalance_td['avolS_min240'] < 0).sum()
)
    print(
        "avolS_min240 为 inf 数量：",
        np.isinf(HF_BSImbalance_td['avolS_min240']).sum()
    )
    print("vol < 0 数量：", (HF_BSImbalance_td['vol'] < 0).sum())
    print("avol < 0 数量：", (HF_BSImbalance_td['avol'] < 0).sum())
    print("avol 为 inf 数量：", np.isinf(HF_BSImbalance_td['avol']).sum())


    # 删除中间变量和原始成交量、状态列，
    # 最终保留 avolS、avolS_min240、volS_lag252_nvalid 和 avol
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

    print_memory(HF_BSImbalance_td, '加入 avolS / avolS_min240 / avol 并删除中间变量后')

    return HF_BSImbalance_td


def check_oib_output(df):
    """
    检查 OIB 输出表。

    缺失值属于允许情况，仅汇总报告；
    无穷值或超出 [-100, 100] 的数值视为异常并终止运行。
    """
    daily_cols = [
        'OIBvol',
        'OIBnum',
        'OIBamount',
        'OIBorder',
    ]

    avg_cols = (
        [f'OIBvol_avg_t_tp{i}' for i in OIBVOL_HORIZONS]
        + [f'OIBnum_avg_t_tp{i}' for i in OIBNUM_HORIZONS]
        + [f'OIBamount_avg_t_tp{i}' for i in OIBAMOUNT_HORIZONS]
        + [f'OIBorder_avg_t_tp{i}' for i in OIBORDER_HORIZONS]
    )

    oib_cols = daily_cols + avg_cols
    required_cols = ['Stkcd', 'Trddt', 'td'] + oib_cols

    if df.empty:
        raise ValueError('oib_output 为空')

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f'oib_output 缺少必要变量：{missing_cols}'
        )

    summary_rows = []
    total_inf = 0
    total_out_of_range = 0

    for col in oib_cols:
        s = pd.to_numeric(df[col], errors='coerce')

        missing_count = int(s.isna().sum())
        inf_count = int(
            s.isin([np.inf, -np.inf]).sum()
        )

        # 保留少量浮点误差容忍度
        out_of_range_count = int(
            (s.abs() > 100.0001).sum()
        )

        total_inf += inf_count
        total_out_of_range += out_of_range_count

        summary_rows.append(
            {
                '变量': col,
                '总行数': len(s),
                '非缺失数': int(s.notna().sum()),
                '缺失数': missing_count,
                '缺失率(%)': round(
                    missing_count / len(s) * 100,
                    4,
                ),
                '无穷值数': inf_count,
                '超出[-100,100]数': out_of_range_count,
            }
        )

    summary = pd.DataFrame(summary_rows)

    print('=' * 80)
    print('OIB 最终输出检查')
    print('=' * 80)
    print('日期范围：', df['Trddt'].min(), '至', df['Trddt'].max())
    print('输出形状：', df.shape)
    print('应有列数：', len(required_cols))
    print(summary.to_string(index=False))
    print_memory(df, 'oib_output')

    if total_inf > 0 or total_out_of_range > 0:
        raise ValueError(
            'oib_output 存在无穷值或超出 [-100, 100] 的数值'
        )


def check_avol_output(df):
    """
    检查异常成交量输出表。

    缺失值属于允许情况，仅汇总报告；
    无穷值、负值或窗口有效日数不符合构造规则时终止运行。
    """
    avol_cols = [
        'avolS',
        'avolS_min240',
        'volS_lag252_nvalid',
        'avol',
    ]

    required_cols = [
        'Stkcd',
        'Trddt',
        'td',
    ] + avol_cols

    if df.empty:
        raise ValueError('avol_output 为空')

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f'avol_output 缺少必要变量：{missing_cols}'
        )

    summary_rows = []
    total_inf = 0
    total_negative = 0

    for col in avol_cols:
        s = pd.to_numeric(df[col], errors='coerce')

        missing_count = int(s.isna().sum())
        inf_count = int(
            s.isin([np.inf, -np.inf]).sum()
        )
        negative_count = int((s < 0).sum())

        total_inf += inf_count
        total_negative += negative_count

        summary_rows.append(
            {
                '变量': col,
                '总行数': len(s),
                '非缺失数': int(s.notna().sum()),
                '缺失数': missing_count,
                '缺失率(%)': round(
                    missing_count / len(s) * 100,
                    4,
                ),
                '无穷值数': inf_count,
                '负值数': negative_count,
            }
        )

    summary = pd.DataFrame(summary_rows)

    nvalid = pd.to_numeric(
        df['volS_lag252_nvalid'],
        errors='coerce',
    )

    finite_nvalid = nvalid[
        nvalid.notna()
        & ~nvalid.isin([np.inf, -np.inf])
    ]

    # 有效日数必须位于0至252之间
    nvalid_out_of_range = int(
        (
            (finite_nvalid < 0)
            | (finite_nvalid > AVOL_WINDOW)
        ).sum()
    )

    # 有效日数应当为整数
    nvalid_non_integer = int(
        (
            (
                finite_nvalid
                - finite_nvalid.round()
            ).abs() > 1e-6
        ).sum()
    )

    # 严格口径非缺失时，窗口必须有252个有效日
    strict_rule_error = int(
        (
            df['avolS'].notna()
            & ~nvalid.eq(AVOL_WINDOW)
        ).sum()
    )

    # 放宽口径非缺失时，至少应有240个有效日
    relaxed_rule_error = int(
        (
            df['avolS_min240'].notna()
            & ~nvalid.ge(AVOLS_MIN_VALID_DAYS)
        ).sum()
    )

    # 严格口径有效时，放宽口径也必须有效
    strict_without_relaxed = int(
        (
            df['avolS'].notna()
            & df['avolS_min240'].isna()
        ).sum()
    )

    # 252日全部有效时，两种口径的数值应当相同
    both_nonmissing = (
        df['avolS'].notna()
        & df['avolS_min240'].notna()
        & nvalid.eq(AVOL_WINDOW)
    )

    strict_relaxed_difference = int(
        (
            both_nonmissing
            & (
                df['avolS']
                - df['avolS_min240']
            ).abs().gt(1e-5)
        ).sum()
    )

    print('=' * 80)
    print('Avol 最终输出检查')
    print('=' * 80)
    print('日期范围：', df['Trddt'].min(), '至', df['Trddt'].max())
    print('输出形状：', df.shape)
    print('应有列数：', len(required_cols))
    print(summary.to_string(index=False))

    print(
        '有效日数超出合法范围的数量：',
        nvalid_out_of_range,
    )
    print(
        '有效日数不是整数的数量：',
        nvalid_non_integer,
    )
    print(
        '严格口径有效日数不等于252的数量：',
        strict_rule_error,
    )
    print(
        '240日口径有效日数不足240的数量：',
        relaxed_rule_error,
    )
    print(
        '严格口径非缺失但240日口径缺失的数量：',
        strict_without_relaxed,
    )
    print(
        '252日完整时两个avolS口径数值不一致的数量：',
        strict_relaxed_difference,
    )

    print_memory(df, 'avol_output')

    logic_error_count = (
        nvalid_out_of_range
        + nvalid_non_integer
        + strict_rule_error
        + relaxed_rule_error
        + strict_without_relaxed
        + strict_relaxed_difference
    )

    if total_inf > 0 or total_negative > 0:
        raise ValueError(
            'avol_output 存在无穷值或负值'
        )

    if logic_error_count > 0:
        raise ValueError(
            'avol_output 的252日或240日窗口口径存在异常'
        )
# =============================================================================
# 11. 生成类似 Table 1 的 OIB 描述性统计表格图片
# =============================================================================
def save_oib_summary_figure(df, output_path):
    """
    生成并保存 OIB 描述性统计表格图片。

    参数：
    df：最终交易行为变量面板；
    output_path：图片保存路径。
    """

    vars_for_table = {
        'Retail OI in % over [t,t+1]': df['OIBvol_avg_t_tp1'],
        'Retail OI in % over [t,t+10]': df['OIBvol_avg_t_tp10'],
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

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'表格图片已保存：{output_path}')


def main():
    """分别构造并输出 OIB 与异常成交量变量。"""

    oib_out_file.parent.mkdir(parents=True, exist_ok=True)
    avol_out_file.parent.mkdir(parents=True, exist_ok=True)
    fig_file.parent.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # 1. 构造完整交易日面板
    # =====================================================================

    full_panel = build_full_panel(
        oib_file_path=HF_BSImbalance_file,
        daily_file_path=TRD_Dalyr_file,
        panel_file_path=panel_file,
    )

    # 提前保存单日 OIB，供后面的两个 OIB 面板使用
    oib_source = full_panel[
        ['Stkcd', 'Trddt', 'OIBvol', 'OIBnum','OIBamount','OIBorder',]
    ].copy()

    oib_source['Stkcd'] = oib_source['Stkcd'].astype('string')

    check_unique_key(
        oib_source,
        ['Stkcd', 'Trddt'],
        'oib_source',
    )

    # =====================================================================
    # 2. 在完整交易日面板上计算 avol、avolS
    # =====================================================================

    avol_result = calc_abnormal_volumes(
        df=full_panel,
        window=AVOL_WINDOW,
        avols_min_valid_days=AVOLS_MIN_VALID_DAYS,
    )

    avol_cols = [
        'Stkcd',
        'Trddt',
        'td',
        'avolS',
        'avolS_min240',
        'volS_lag252_nvalid',
        'avol',
    ]

    avol_output = avol_result.loc[
        avol_result['Trddt'].between(
            OUTPUT_START_DATE,
            OUTPUT_END_DATE,
        ),
        avol_cols,
    ].copy()

    avol_output = (
        avol_output
        .sort_values(['Stkcd', 'td'])
        .reset_index(drop=True)
    )

    check_unique_key(
        avol_output,
        ['Stkcd', 'Trddt'],
        'avol_output',
    )

    del avol_result, full_panel
    gc.collect()

    # =====================================================================
    # 3. 在窗口变量目标日资格索引上严格计算 OIB 未来均值
    # =====================================================================

    window_index = load_stock_day_panel(
        window_target_index_file
    )

    check_unique_key(
        window_index,
        ['Stkcd', 'Trddt'],
        'window_index',
    )

    oib_window_panel = pd.merge(
        window_index,
        oib_source,
        on=['Stkcd', 'Trddt'],
        how='left',
        validate='one_to_one',
    )

    oib_window_panel = (
        oib_window_panel
        .sort_values(['Stkcd', 'td'])
        .reset_index(drop=True)
    )

    check_unique_key(
        oib_window_panel,
        ['Stkcd', 'td'],
        'oib_window_panel',
    )

    oib_window_panel = calc_forward_oib_averages(
        df=oib_window_panel,
        oibvol_horizons=OIBVOL_HORIZONS,
        oibnum_horizons=OIBNUM_HORIZONS,
        oibamount_horizons=OIBAMOUNT_HORIZONS,
        oiborder_horizons=OIBORDER_HORIZONS,
    )

    oib_avg_cols = (
        [f'OIBvol_avg_t_tp{i}' for i in OIBVOL_HORIZONS]
        + [f'OIBnum_avg_t_tp{i}' for i in OIBNUM_HORIZONS]
        + [f'OIBamount_avg_t_tp{i}' for i in OIBAMOUNT_HORIZONS]
        + [f'OIBorder_avg_t_tp{i}' for i in OIBORDER_HORIZONS]
    )

    # =====================================================================
    # 4. 单日 OIB 按基础清洗样本保留
    # =====================================================================

    base_sample_index = load_stock_day_panel(
        base_sample_index_file
    )

    check_unique_key(
        base_sample_index,
        ['Stkcd', 'Trddt'],
        'base_sample_index',
    )

    oib_daily = pd.merge(
        base_sample_index,
        oib_source,
        on=['Stkcd', 'Trddt'],
        how='left',
        validate='one_to_one',
    )

    # 将单日 OIB 与 OIB 未来均值整理为一张 OIB 输出表
    oib_output = pd.merge(
        oib_daily,
        oib_window_panel[
            ['Stkcd', 'Trddt', 'td'] + oib_avg_cols
        ],
        on=['Stkcd', 'Trddt', 'td'],
        how='left',
        validate='one_to_one',
    )

    oib_cols = [
        'Stkcd',
        'Trddt',
        'td',
        'OIBvol',
        'OIBnum',
        'OIBamount',
        'OIBorder',
    ] + oib_avg_cols

    oib_output = oib_output.loc[
        oib_output['Trddt'].between(
            OUTPUT_START_DATE,
            OUTPUT_END_DATE,
        ),
        oib_cols,
    ].copy()

    oib_output = (
        oib_output
        .sort_values(['Stkcd', 'td'])
        .reset_index(drop=True)
    )

    check_unique_key(
        oib_output,
        ['Stkcd', 'Trddt'],
        'oib_output',
    )

    del oib_source, window_index
    del oib_window_panel, base_sample_index, oib_daily
    gc.collect()

    # =====================================================================
    # 5. 检查两张最终输出表
    # =====================================================================

    check_oib_output(oib_output)
    check_avol_output(avol_output)

    # =====================================================================
    # 6. 分别输出，不合并 OIB 和 Avol
    # =====================================================================
    print('正在分别输出 OIB 和 Avol……')

    oib_output.to_csv(
        oib_out_file,
        index=False,
        encoding='utf-8-sig',
    )

    avol_output.to_csv(
        avol_out_file,
        index=False,
        encoding='utf-8-sig',
    )

    print(f'OIB 结果已经保存到：{oib_out_file}')
    print(f'Avol 结果已经保存到：{avol_out_file}')
    print(f'OIB 输出形状：{oib_output.shape}')
    print(f'Avol 输出形状：{avol_output.shape}')

    save_oib_summary_figure(
        df=oib_output,
        output_path=fig_file,
    )



if __name__ == '__main__':
    main()
# =============================================================================
# 最终输出列说明
# =============================================================================
#
# 一、OIB_CSMAR_prep.csv
#
# 键变量：
#   Stkcd, Trddt, td
#
# 单日订单失衡指标：
#   OIBvol：小单成交量订单失衡
#   OIBnum：小单买卖次数订单失衡
#   OIBamount：小单成交金额订单失衡
#   OIBorder：小单成交笔数订单失衡
#
# 未来严格窗口均值：
#   OIBvol_avg_t_tp1 到 OIBvol_avg_t_tp50，共50列
#   OIBnum_avg_t_tp1, OIBnum_avg_t_tp10，共2列
#   OIBamount_avg_t_tp1, OIBamount_avg_t_tp10，共2列
#   OIBorder_avg_t_tp1, OIBorder_avg_t_tp10，共2列
#
# 总列数：
#   3个键变量 + 4个单日指标 + 50 + 2 + 2 + 2 = 63列
#
#
# 二、Avol_CSMAR_prep.csv
#
# 保留变量：
#   Stkcd, Trddt, td
#   avolS：严格252日异常小单成交量
#   avolS_min240：过去252日中至少240日有效的异常小单成交量
#   volS_lag252_nvalid：过去252日内小单成交量的实际有效日数
#   avol：严格252日异常总成交量，基于Dnshrtrd
#
# 总列数：
#   3个键变量 + 4个异常成交量相关变量 = 7列
# =============================================================================
