"""逐单元探索国泰安沪深日度三因子和五因子。

从上到下运行代码单元。起止日和候选口径在第一格选择。
各口径共用同一批原始数据，普通版始终参与对照。官方因子只用于计算误差。
逐年读行情和换组放在同一格，避免全量行情同时驻留内存。
"""


# %% 00. 导入库并选择日期、路径和候选口径
import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

source_file = globals().get('__file__')
running_as_script = (
    __name__ == '__main__'
    and source_file is not None
    and Path(sys.argv[0]).resolve() == Path(source_file).resolve()
)
if running_as_script:
    ROOT = Path(source_file).resolve().parents[4]
else:
    ROOT = Path('/Volumes/苹果资料/mmm复现数据包')
INPUT_DIR = ROOT / 'input/CSMAR'
RATE_FILE = (
    Path.home() / 'Desktop/课程学习/研一/研一下/FF-3/input'
    / '金融机构人民币存贷款基准利率调整表（1952-09-15至2025-05-20）/MBK_FINRMBDEPOSITLOAND.csv'
)
START = pd.Timestamp('2000-01-01')
END = pd.Timestamp('2026-08-19')

# 在交互窗口中可以改为其他日期，例如 pd.Timestamp('2025-01-01')。
sample_start = START
sample_end = END
input_dir = INPUT_DIR
rate_file = RATE_FILE
selected_specs = (
    '上一轮基准',
    '十二月长停牌',
    '十二月长停牌加六月股本变动',
    '全期保留权益符号',
)

SME_MERGER = pd.Timestamp('2021-04-06')
PROFIT_CHANGE = pd.Timestamp('2023-07-01')
MARKETS = (1, 4, 16, 32)  # 沪深主板、创业板、科创板
MAIN_BOARD = (1, 4)
CAP_COLUMNS = ['Dsmvosd', 'Dsmvtll', 'Dsmvtl2', 'Dsmvtl3']
WEIGHTS = {1: 'float_cap', 2: 'total_cap'}
FACTOR_NAMES = {
    'RiskPremium': '市场风险溢价', 'SMB': '规模', 'HML': '价值',
    'RMW': '盈利能力', 'CMA': '投资',
}
FACTORS = {3: ('RiskPremium', 'SMB', 'HML'), 5: tuple(FACTOR_NAMES)}
INDUSTRY_DATES = (
    ('P0201', pd.Timestamp.min, pd.Timestamp('2012-10-26')),
    ('P0207', pd.Timestamp('2012-10-26'), pd.Timestamp('2023-05-01')),
    ('P0221', pd.Timestamp('2023-05-01'), pd.Timestamp.max),
)


# 普通版的完整口径，以及每个候选相对普通版改动的规则。
ORDINARY = {
    'split_b_share_prices': True,
    'historical_main_board': True,
    'december_last_trade': True,
    'december_min_gap': 0,
    'june_restructure_cap': False,
    'profit_equity': 'before_2023',
    'ipo_from_offer_price': True,
    'five_calendar_cap': True,
}
SPEC_OVERRIDES = {
    '普通版': {},
    '上一轮基准': {
        'split_b_share_prices': False,
        'historical_main_board': False,
        'december_last_trade': False,
        'profit_equity': 'positive',
        'ipo_from_offer_price': False,
    },
    # 年末停牌填充且距最近成交至少 60 个开市日，才替换十二月市值。
    '十二月长停牌': {'december_min_gap': 60},
    '六月重大股本变动': {'june_restructure_cap': True},
    '十二月长停牌加六月股本变动': {
        'december_min_gap': 60,
        'june_restructure_cap': True,
    },
    '全期保留权益符号': {'profit_equity': 'signed'},
}


PERIODS = (
    ('2000—2025 年主样本', pd.Timestamp('2000-01-01'), pd.Timestamp('2025-12-31')),
    ('2023 年下半年至 2025 年', pd.Timestamp('2023-07-01'), pd.Timestamp('2025-12-31')),
    ('2026 年六月换组后', pd.Timestamp('2026-07-01'), pd.Timestamp('2026-08-19')),
)


if running_as_script:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=input_dir, help='国泰安原始数据目录')
    parser.add_argument('--rate-file', type=Path, default=rate_file, help='三个月定存基准利率文件')
    parser.add_argument('--start', type=pd.Timestamp, default=sample_start, help='起始日期，默认 2000-01-01')
    parser.add_argument('--end', type=pd.Timestamp, default=sample_end, help='终止日期，默认 2026-08-19')
    parser.add_argument(
        '--spec',
        nargs='+',
        choices=tuple(SPEC_OVERRIDES),
        default=selected_specs,
        help='指定一个或多个口径。始终保留普通版对照，省略时使用默认候选。',
    )
    args = parser.parse_args()
    input_dir = args.input_dir
    rate_file = args.rate_file
    sample_start = args.start
    sample_end = args.end
    selected_specs = args.spec

error = None
if not START <= sample_start <= sample_end <= END:
    error = '日期须在 2000-01-01 至 2026-08-19 之间，且起始日期不能晚于终止日期'
elif not input_dir.is_dir():
    error = f'找不到国泰安原始数据目录 {input_dir}'
elif not rate_file.is_file():
    error = f'找不到基准利率表 {rate_file}，请指定利率文件路径'
if error:
    if running_as_script:
        parser.error(error)
    raise ValueError(error)

selected_names = list(dict.fromkeys(('普通版', *selected_specs)))
specifications = {name: {**ORDINARY, **SPEC_OVERRIDES[name]} for name in selected_names}
print(f'比较口径  {"、".join(selected_names)}', flush=True)


def input_path(folder, filename):
    matches = sorted(folder.glob(f'*/{filename}'))
    if len(matches) != 1:
        raise ValueError(f'{folder} 应有一份 {filename}，找到 {len(matches)} 份')
    return matches[0]


# %% 01. 读取国泰安发布的因子对照表
official = {}
for model, file in ((3, 'STK_MKT_THRFACDAY.csv'), (5, 'STK_MKT_FIVEFACDAY.csv')):
    data = pd.read_csv(input_path(input_dir, file), parse_dates=['TradingDate'])
    subset = data['MarkettypeID'].eq('P9714')
    if model == 5:
        subset &= data['Portfolios'].eq(1)
    cols = [f'{factor}{i}' for factor in FACTORS[model] for i in WEIGHTS]
    official[model] = data.loc[subset, ['TradingDate', *cols]].rename(columns={'TradingDate': 'date'})
if not running_as_script:
    for model, table in official.items():
        print(f'国泰安{model}因子对照表：{table.shape}')


# %% 02. 整理交易日历并计算日度无风险利率
calendar = pd.read_csv(input_path(input_dir, 'CER_Cale.csv'), parse_dates=['Clddt'])
trading = calendar.loc[calendar['Markettype'].eq(1) & calendar['Statse'].eq('O'), 'Clddt']
trading = trading.drop_duplicates().sort_values()
days = pd.Series(np.arange(len(trading)), index=pd.DatetimeIndex(trading))
month_ends = trading.groupby([trading.dt.year, trading.dt.month]).max()
requested = trading.loc[trading.between(sample_start, sample_end)]
if requested.empty:
    raise ValueError('所选区间没有交易日')
rate = pd.read_csv(rate_file, parse_dates=['ChangeDate']).sort_values('ChangeDate')
rate = rate.loc[rate['ChangeDate'].le('2015-10-24'), ['ChangeDate', 'LumpFixed3Month']]
rate = pd.merge_asof(requested.to_frame('date'), rate, left_on='date', right_on='ChangeDate').set_index('date')
rf = np.expm1(np.log1p(rate['LumpFixed3Month'] / 100) / 365).rename('rf')
if rf.isna().any():
    raise ValueError('部分交易日缺少三个月定存基准利率')
if not running_as_script:
    print(f'所选区间交易日：{len(requested):,}')
    print(rf.head().to_string())


# %% 03. 合并年报并计算账面权益、营业利润和投资增长
balance_items = {
    'A003000000': 'equity', 'A002208000': 'tax_liability',
    'A001222000': 'tax_asset', 'A003112101': 'preferred_stock',
    'A001000000': 'assets',
}
income_items = {
    'B001101000': 'revenue', 'B001201000': 'cost',
    'B001209000': 'selling_expense', 'B001210000': 'admin_expense',
}
keys = ['Stkcd', 'year']
tables = []
for filename, columns, filter_col, filter_value in (
    ('FS_Combas.csv', balance_items, 'Typrep', 'A'),
    ('FS_Comins.csv', income_items, 'Typrep', 'A'),
    ('IAR_Rept.csv', {'Annodt': 'announced'}, 'Reptyp', 4),
):
    table = pd.read_csv(
        input_path(input_dir, filename),
        usecols=['Stkcd', 'Accper', filter_col, *columns],
        dtype={'Stkcd': 'int32', 'Accper': str},
    )
    keep = table['Accper'].str.fullmatch(r'\d{4}-12-31', na=False) & table[filter_col].eq(filter_value)
    table = table.loc[keep].copy()
    table['year'] = table['Accper'].str[:4].astype(int)
    tables.append(table.rename(columns=columns)[[*keys, *columns.values()]])

balance, income, reports = tables
income['has_income_statement'] = True
reports['announced'] = pd.to_datetime(reports['announced'], format='%Y-%m-%d', errors='coerce')
previous = balance[[*keys, 'assets']].rename(columns={'assets': 'previous_assets'}).copy()
previous['year'] += 1
data = balance.merge(previous, on=keys, how='left', validate='one_to_one')
data = data.merge(income, on=keys, how='left', validate='one_to_one')
data = data.merge(reports, on=keys, how='left', validate='one_to_one')
data['book'] = (
    data['equity'] + data['tax_liability'].fillna(0)
    - data['tax_asset'].fillna(0) - data['preferred_stock'].fillna(0)
)
# 经验复现口径不另扣利息。营收和成本必须有值，两项费用缺失按零处理。
data['operating_profit'] = (
    data['revenue'] - data['cost']
    - data['selling_expense'].fillna(0) - data['admin_expense'].fillna(0)
)
data['blank_financial_profit'] = (
    data['has_income_statement'].eq(True)
    & data[list(income_items.values())].isna().all(axis=1)
    & data['book'].gt(0)
)
data['inv'] = (
    data['assets'].where(data['assets'].gt(0))
    .div(data['previous_assets'].where(data['previous_assets'].gt(0))) - 1
)
financials = data[['Stkcd', 'year', 'announced', 'book', 'operating_profit', 'blank_financial_profit', 'inv']]
del data, balance, income, reports, tables, table, previous
if not running_as_script:
    print(f'年度财报：{financials.shape}')
    print(financials.head().to_string(index=False))


# %% 04. 整理股本变动、上市信息和行业分类
paths = sorted((input_dir / '专题研究系列-事件研究').glob('*/ER_Structure.csv'))
if len(paths) != 1:
    raise ValueError(f'股本变动事件表应有一份，实际找到 {len(paths)} 份')
shares = pd.read_csv(
    paths[0],
    usecols=['Symbol', 'ChangeDate', 'DeclareDate', 'TradableSharesB',
             'TradableSharesH', 'TradeSharesTotal', 'ChangeReasonID'],
    dtype={'Symbol': 'int32'},
).rename(columns={'Symbol': 'Stkcd', 'ChangeDate': 'date'})
for col in ('date', 'DeclareDate'):
    shares[col] = pd.to_datetime(shares[col], format='%Y-%m-%d', errors='coerce')
shares['has_b_shares'] = shares['TradableSharesB'].gt(0)
shares['cross_listed'] = shares['has_b_shares'] | shares['TradableSharesH'].gt(0)
shares = shares.dropna(subset=['date']).sort_values('date', kind='stable')

industry = pd.read_csv(
    input_path(input_dir, 'STK_INDUSTRYCLASS.csv'),
    usecols=['Symbol', 'IndustryClassificationID', 'ImplementDate', 'IndustryCode'],
    dtype=str,
).rename(columns={
    'Symbol': 'Stkcd', 'IndustryClassificationID': 'standard', 'ImplementDate': 'date',
})
industry['date'] = pd.to_datetime(industry['date'], errors='coerce')
industry = industry.dropna(subset=['Stkcd', 'date', 'IndustryCode']).copy()
industry['Stkcd'] = industry['Stkcd'].astype('int32')
industry['finance'] = industry['IndustryCode'].str[0].eq(
    industry['standard'].map({'P0201': 'I', 'P0207': 'J', 'P0221': 'J'})
)
industry = {
    key: group[['date', 'Stkcd', 'finance']].sort_values('date').reset_index(drop=True)
    for key, group in industry.groupby('standard')
}

companies = pd.read_csv(
    input_path(input_dir, 'TRD_Co.csv'), usecols=['Stkcd', 'Listdt', 'Ipoprc'],
    dtype={'Stkcd': 'int32'}, parse_dates=['Listdt'],
).drop_duplicates('Stkcd').set_index('Stkcd')
if not running_as_script:
    print(f'股本变动记录：{shares.shape}；上市公司信息：{companies.shape}')
    print('行业分类记录：', {standard: len(table) for standard, table in industry.items()})


# %% 05. 定义逐年使用的行情读取、行业匹配和组合分组
def mark_financials(rows, industry):
    """用相应日期生效的行业记录判断金融股，匹配不到时按当前复现规则保留。"""
    matches = []
    for standard, begin, end in INDUSTRY_DATES:
        sample = rows.loc[rows['date'].ge(begin) & rows['date'].lt(end), ['Stkcd', 'date']]
        if not sample.empty:
            sample = sample.rename_axis('row').reset_index().sort_values('date')
            matched = pd.merge_asof(sample, industry[standard], on='date', by='Stkcd')
            matches.append(matched.set_index('row')['finance'])
    if not matches:
        return pd.Series(False, index=rows.index)
    return pd.concat(matches).reindex(rows.index).eq(True)


def read_daily_years(input_dir, end_year, trading_days, shares, companies):
    """按原始下载分卷读取。保留停牌填充行供月底和市场权重计算。"""
    directory = input_dir / '个股日度收益率（升级测试）1995-01-01至2026'
    columns = ['Stkcd', 'Trddt', 'Markettype', 'Filling', 'Dretwd', 'Clsprc', *CAP_COLUMNS]
    types = {'Stkcd': 'int32', 'Markettype': 'int16', 'Filling': 'int8'}
    flags = shares[['Stkcd', 'date', 'has_b_shares', 'cross_listed']]
    for folder in sorted(directory.iterdir()):
        years = re.findall(r'\d{4}', folder.name)
        if not years or int(years[0]) > end_year or int(years[-1]) < 1998:
            continue
        chunks = []
        for file in sorted(folder.glob('TRDNEW_Dalyr*.csv')):
            for chunk in pd.read_csv(file, usecols=columns, dtype=types, chunksize=400_000):
                year = chunk['Trddt'].str[:4].astype('int16')
                keep = year.between(1998, end_year) & chunk['Markettype'].isin(MARKETS)
                if keep.any():
                    chunks.append(chunk.loc[keep].assign(year=year[keep]))
        if not chunks:
            continue
        data = pd.concat(chunks, ignore_index=True)
        del chunks
        for year, part in data.groupby('year', sort=True):
            daily = part.rename(columns={'Trddt': 'date'}).drop(columns='year')
            daily['date'] = pd.to_datetime(daily['date'], format='%Y-%m-%d')
            daily = daily.loc[daily['date'].isin(trading_days)].sort_values('date').reset_index(drop=True)
            if daily.duplicated(['Stkcd', 'date']).any():
                raise ValueError(f'{year} 年个股行情存在重复的股票和交易日')
            daily = pd.merge_asof(daily, flags, on='date', by='Stkcd')
            daily['total_base'] = daily['Dsmvtl2']
            daily['total_split'] = daily['Dsmvtl3'].where(daily['has_b_shares'].eq(True), daily['Dsmvtl2'])
            daily['total_split'] = daily['total_split'].fillna(daily['total_base'])
            for suffix in ('base', 'split'):
                daily[f'float_{suffix}'] = daily['Dsmvosd'] + (
                    daily[f'total_{suffix}'] - daily['Dsmvtll']
                ).clip(lower=0)
            issue_price = daily['Stkcd'].map(companies['Ipoprc'])
            listed_today = daily['date'].eq(daily['Stkcd'].map(companies['Listdt'])) & issue_price.gt(0)
            daily['offer_return'] = daily['Dretwd'].where(~listed_today, daily['Clsprc'].div(issue_price) - 1)
            yield int(year), daily


def last_actual_trades(previous, daily, cutoff):
    if pd.isna(cutoff):
        return previous
    cols = ['Stkcd', 'date', *CAP_COLUMNS, 'float_base', 'float_split']
    sample = daily.loc[daily['Filling'].eq(0) & daily['date'].le(cutoff), cols]
    if previous is not None:
        sample = pd.concat([previous.reset_index(), sample], ignore_index=True)
    return sample.sort_values('date').drop_duplicates('Stkcd', keep='last').set_index('Stkcd')


def form_portfolios(june, december, annual, shares, industry, days, june_end,
                    december_end, last_december, last_june, spec):
    """六月规模组和十二月特征组分别划分，再按证券代码取交集。"""
    suffix = 'split' if spec['split_b_share_prices'] else 'base'
    float_col = f'float_{suffix}'
    total_col = f'total_{suffix}'
    june = june.loc[june[float_col].gt(0)].set_index('Stkcd').copy()
    december = december.set_index('Stkcd').copy()
    june['float_cap'] = june[float_col]
    june['total_cap'] = june[total_col]

    if spec['june_restructure_cap']:
        # 停牌期间只考虑六月分组前已生效且已披露的增发、股改或吸收合并。
        eligible_events = shares.loc[
            shares['date'].le(june_end) & shares['DeclareDate'].le(june_end)
            & shares['ChangeReasonID'].isin(('P0704', 'P0712', 'P0716'))
        ]
        last_event = eligible_events.groupby('Stkcd')['date'].max().reindex(june.index)
        actual = last_june.reindex(june.index)
        replace = (
            june['Filling'].eq(1) & (days.at[june_end] - actual['date'].map(days)).ge(20)
            & last_event.gt(actual['date'])
        )
        total = actual['Dsmvtl2']
        if spec['split_b_share_prices']:
            total = actual['Dsmvtl3'].where(june['has_b_shares'].eq(True), total).fillna(total)
        floating = actual['Dsmvosd'] + (total - actual['Dsmvtll']).clip(lower=0)
        june['float_cap'] = june['float_cap'].mask(replace & floating.gt(0), floating)
        june['total_cap'] = june['total_cap'].mask(replace & total.gt(0), total)

    december['value_cap3'] = december[float_col]
    december['value_cap5'] = december[float_col]
    if spec['five_calendar_cap']:
        # 跨市场上市公司有上年 12 月 31 日股本明细时，以该流通股数乘上年末收盘价。
        year_end = pd.Timestamp(june_end.year - 1, 12, 31)
        year_end_shares = shares.loc[
            shares['date'].eq(year_end) & shares['DeclareDate'].le(june_end)
        ].set_index('Stkcd').reindex(december.index)
        cross_listed = december['cross_listed'].eq(True) | year_end_shares['cross_listed'].eq(True)
        shares_count = year_end_shares['TradeSharesTotal']
        exact = shares_count.where(cross_listed & shares_count.gt(0)) * december['Clsprc'] / 1000
        december['value_cap5'] = exact.fillna(december[float_col])

    if spec['december_last_trade']:
        actual = last_december.reindex(december.index)
        gap = days.at[december_end] - actual['date'].map(days)
        use_actual = december['Filling'].eq(1) & gap.ge(spec['december_min_gap']) & actual[float_col].notna()
        for col in ('value_cap3', 'value_cap5'):
            december[col] = december[col].mask(use_actual, actual[float_col])

    june_main = june['Markettype'].isin(MAIN_BOARD)
    december_main = december['Markettype'].isin(MAIN_BOARD)
    if spec['historical_main_board'] and june_end < SME_MERGER:
        # 两套断点都按六月换组时的板块状态选主板。中小板股票仍能入组。
        june_main &= ~june.index.to_series().between(2000, 3999)
        december_main &= ~december.index.to_series().between(2000, 3999)
    median = june.loc[june_main, 'float_cap'].median()
    if pd.isna(median):
        raise ValueError(f'{june_end.date()} 没有可用于规模断点的主板市值')
    june['size_group'] = np.where(june['float_cap'].ge(median), 'B', 'S')

    # 六月断点是交易日，非交易日公告映射到下一交易日后，此处的筛选结果相同。
    known = annual.loc[annual['announced'].le(june_end)].copy()
    is_financial = mark_financials(known.assign(date=december_end), industry)
    before_change = june_end.year < PROFIT_CHANGE.year
    use_signed = spec['profit_equity'] == 'signed' or (spec['profit_equity'] == 'before_2023' and before_change)
    book = known['book'].where(known['book'].ne(0) if use_signed else known['book'].gt(0))
    profit = known['operating_profit'].div(book)
    # 2023 年换组起，四项利润科目全空的正权益金融股以零作为排序值。
    if before_change:
        known['op'] = profit.mask(is_financial)
    else:
        known['op'] = profit.mask(is_financial & known['blank_financial_profit'], 0)
    december = december.join(known.set_index('Stkcd')[['book', 'op', 'inv']], validate='one_to_one')
    for signal, cap in (('bm3', 'value_cap3'), ('bm5', 'value_cap5')):
        december[signal] = december['book'].where(december['book'].gt(0)).div(
            december[cap].where(december[cap].gt(0)) * 1000
        )
    signals = ['bm3', 'bm5', 'op', 'inv']
    for signal in signals:
        values = december[signal]
        low, high = values.loc[december_main].quantile([0.30, 0.70])
        december[signal] = pd.Series(
            np.select([values.ge(high), values.lt(low), values.notna() & pd.notna(low)],
                      ['H', 'L', 'M'], default=''), index=values.index,
        ).replace('', np.nan)
    return june[['size_group', *WEIGHTS.values()]].join(december[signals], how='inner').reset_index()


# %% 06. 定义逐年使用的市值加权收益和因子公式
def weighted_mean_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    return_col: str,
    weight_col: str,
    value_name: str,
) -> pd.DataFrame:
    """按组计算市值加权收益。收益缺失或权重缺失、非正的观测不参与加权。"""
    cols = list(group_cols) + [return_col, weight_col]
    tmp = df.loc[:, cols].copy()
    tmp[return_col] = pd.to_numeric(tmp[return_col], errors="coerce")
    tmp[weight_col] = pd.to_numeric(tmp[weight_col], errors="coerce")

    valid = tmp[return_col].notna() & tmp[weight_col].notna() & tmp[weight_col].gt(0)
    tmp = tmp.loc[valid].copy()

    if tmp.empty:
        return pd.DataFrame(columns=list(group_cols) + [value_name, "n_stocks", "sum_weight"])

    tmp["_weighted_ret"] = tmp[return_col] * tmp[weight_col]
    out = (
        tmp.groupby(list(group_cols), observed=True, dropna=False)
        .agg(
            sum_weighted_ret=("_weighted_ret", "sum"),
            sum_weight=(weight_col, "sum"),
            n_stocks=(return_col, "count"),
        )
        .reset_index()
    )
    out[value_name] = out["sum_weighted_ret"] / out["sum_weight"]
    return out.loc[:, list(group_cols) + [value_name, "n_stocks", "sum_weight"]]


def make_ff_2x3_factors(
    legs: pd.DataFrame,
    date_col: str,
    size_col: str,
    char_col: str,
    return_col: str,
    smb_col: str,
    hml_col: str,
) -> pd.DataFrame:
    """由二乘三六组收益计算小盘减大盘和高特征组减低特征组。"""
    wide = legs.pivot_table(
        index=date_col,
        columns=[size_col, char_col],
        values=return_col,
        aggfunc="first",
    )

    required = [("S", "L"), ("S", "M"), ("S", "H"), ("B", "L"), ("B", "M"), ("B", "H")]
    for col in required:
        if col not in wide.columns:
            wide[col] = pd.NA

    out = pd.DataFrame(index=wide.index)
    out[smb_col] = (
        (wide[("S", "L")] + wide[("S", "M")] + wide[("S", "H")]) / 3
        - (wide[("B", "L")] + wide[("B", "M")] + wide[("B", "H")]) / 3
    )
    out[hml_col] = (
        (wide[("S", "H")] + wide[("B", "H")]) / 2
        - (wide[("S", "L")] + wide[("B", "L")]) / 2
    )

    return out.reset_index()


def market_returns(daily, previous_december, days, risk_free, spec):
    suffix = 'split' if spec['split_b_share_prices'] else 'base'
    return_col = 'offer_return' if spec['ipo_from_offer_price'] else 'Dretwd'
    cols = ['Stkcd', 'date', 'Filling', return_col, f'float_{suffix}', f'total_{suffix}']
    sample = pd.concat([previous_december[cols], daily[cols]], ignore_index=True).sort_values(['Stkcd', 'date'])
    stock = sample['Stkcd']
    number = sample['date'].map(days)
    adjacent = stock.eq(stock.shift()) & number.diff().eq(1)
    for source, target in (('float', 'lag_float'), ('total', 'lag_total')):
        weight = sample[f'{source}_{suffix}']
        # 上一开市日的市值可来自停牌填充。新上市等无前日记录时用当日市值。
        sample[target] = weight.shift().where(adjacent, weight)
    sample = sample.loc[sample['Filling'].eq(0) & sample['date'].isin(risk_free.index)]
    result = risk_free.to_frame()
    for i, weight in ((1, 'lag_float'), (2, 'lag_total')):
        name = f'Market{i}'
        value = weighted_mean_by_group(sample, ['date'], return_col, weight, name).set_index('date')[name]
        result[name] = value
        result[f'RiskPremium{i}'] = result[name] - result['rf']
    return result


def portfolio_returns(daily, labels, spec):
    """按固定六月权重计算六组日收益，再计算组合间收益差。"""
    data = daily.loc[daily['Filling'].eq(0) & daily['Dretwd'].notna()].merge(
        labels, on='Stkcd', how='inner', validate='many_to_one',
    )
    data['return'] = data['offer_return'] if spec['ipo_from_offer_price'] else data['Dretwd']
    drop_profit = data['date'].lt(PROFIT_CHANGE) & data['financial']
    data.loc[drop_profit, 'op'] = np.nan
    by_model = {3: [], 5: []}
    for i, weight in WEIGHTS.items():
        result = []
        for signal in ('bm3', 'bm5', 'op', 'inv'):
            sample = data.loc[data[signal].notna()]
            legs = weighted_mean_by_group(sample, ['date', 'size_group', signal], 'return', weight, 'portfolio_return')
            result.append(make_ff_2x3_factors(
                legs, 'date', 'size_group', signal, 'portfolio_return', f'size_{signal}', signal,
            ).set_index('date'))
        spreads = pd.concat(result, axis=1)
        three = pd.DataFrame({'SMB': spreads['size_bm3'], 'HML': spreads['bm3']})
        five = pd.DataFrame({
            'SMB': spreads[['size_bm5', 'size_op', 'size_inv']].mean(axis=1, skipna=False),
            'HML': spreads['bm5'], 'RMW': spreads['op'], 'CMA': -spreads['inv'],
        })
        by_model[3].append(three.add_suffix(str(i)))
        by_model[5].append(five.add_suffix(str(i)))
    return {model: pd.concat(parts, axis=1) for model, parts in by_model.items()}


# %% 07. 逐年读取行情、换组，计算各口径的市场和组合收益
yearly_factors = {name: {3: [], 5: []} for name in specifications}
prior_labels = {}
last_december = previous_december = None
seen = pd.DatetimeIndex([])
last_formation_year = None
last_formation_labels = {}
needs_june_history = any(spec['june_restructure_cap'] for spec in specifications.values())
for year, daily in read_daily_years(input_dir, sample_end.year, days.index, shares, companies):
    december_end = month_ends.get((year, 12), pd.NaT)
    dec = daily.loc[daily['date'].eq(december_end)]
    year_rf = rf.loc[rf.index.year == year]
    # 起始年前一年的六月分组供首年上半年使用。
    if year >= sample_start.year - 1 and previous_december is not None:
        june_end = month_ends.get((year, 6), pd.NaT)
        need_formation = year < sample_end.year or sample_end.month >= 7
        current_labels = {}
        if need_formation:
            june = daily.loc[daily['date'].eq(june_end)]
            if june.empty or previous_december.empty:
                raise ValueError(f'{year} 年组合缺少六月末或上年十二月末行情')
            last_june = last_actual_trades(last_december, daily, june_end) if needs_june_history else None
            annual = financials.loc[financials['year'].eq(year - 1)]
            for name, spec in specifications.items():
                current_labels[name] = form_portfolios(
                    june, previous_december, annual, shares, industry, days, june_end,
                    month_ends.at[(year - 1, 12)], last_december, last_june, spec,
                )
            if not running_as_script:
                last_formation_year = year
                last_formation_labels = current_labels

        # 当年日收益按上、下半年分别套用上一年和当年的六月分组。
        if not year_rf.empty:
            print(f'计算 {year} 年日度因子', flush=True)
            selected = daily.loc[daily['date'].isin(year_rf.index)].copy()
            has_trade = selected['Filling'].eq(0) & selected['Dretwd'].notna()
            seen = seen.union(pd.DatetimeIndex(selected.loc[has_trade, 'date'].unique()))
            selected['financial'] = mark_financials(
                selected.loc[selected['date'].lt(PROFIT_CHANGE) & selected['Filling'].eq(0)], industry,
            ).reindex(selected.index).eq(True)
            market_cache = {}
            for name, spec in specifications.items():
                market_key = spec['split_b_share_prices'], spec['ipo_from_offer_price']
                if market_key not in market_cache:
                    market_cache[market_key] = market_returns(daily, previous_december, days, year_rf, spec)
                market = market_cache[market_key]
                portfolios = {3: [], 5: []}
                for half, labels in ((False, prior_labels), (True, current_labels)):
                    part = selected.loc[selected['date'].dt.month.ge(7).eq(half)]
                    if not part.empty:
                        if name not in labels:
                            raise ValueError(f'{year} 年的日度收益缺少对应六月分组')
                        for model, factor in portfolio_returns(part, labels[name], spec).items():
                            portfolios[model].append(factor)
                for model in FACTORS:
                    returns = pd.concat(portfolios[model])
                    table = market.join(returns, validate='one_to_one').rename_axis('date').reset_index()
                    yearly_factors[name][model].append(table)
        prior_labels = current_labels

    # 无请求交易日的历史年份也更新十二月行情和最近实际成交。
    if not dec.empty:
        previous_december = dec
        last_december = last_actual_trades(last_december, daily, december_end)


if not running_as_script and last_formation_labels:
    print(f'{last_formation_year} 年六月换组后的股票数')
    for name, labels in last_formation_labels.items():
        print(f'{name}：{len(labels):,}')
    print(last_formation_labels['普通版'].head().to_string(index=False))


# %% 08. 检查日期覆盖并合并各口径的逐日因子
missing = pd.DatetimeIndex(requested).difference(seen)
if not missing.empty:
    raise ValueError(f'原始行情缺少 {len(missing)} 个交易日，首个为 {missing[0].date()}')
results = {
    name: {model: pd.concat(parts, ignore_index=True) for model, parts in models.items()}
    for name, models in yearly_factors.items()
}
if not running_as_script:
    print('普通版三因子前五个交易日')
    print(results['普通版'][3].head().to_string(index=False))
    print('普通版五因子前五个交易日')
    print(results['普通版'][5].head().to_string(index=False))


# %% 09. 按区间对照官方因子及各候选口径
def comparison_metrics(results, official, start, end):
    rows = []
    for model, table in results.items():
        current = table.loc[table['date'].between(start, end)]
        if current.empty:
            continue
        ref = official[model]
        joined = current.merge(ref, on='date', how='left', suffixes=('_calc', '_official'), validate='one_to_one')
        for factor in FACTORS[model]:
            for i in WEIGHTS:
                col = f'{factor}{i}'
                actual, target = joined[f'{col}_calc'], joined[f'{col}_official']
                error = (actual - target) * 10_000
                rows.append({
                    '模型': '三因子' if model == 3 else '五因子',
                    '因子': FACTOR_NAMES[factor], '加权': '流通市值' if i == 1 else '总市值',
                    '交易日数': int(error.count()), '相关系数': actual.corr(target),
                    '均方根误差（基点）': np.sqrt(error.pow(2).mean()),
                    '平均绝对误差（基点）': error.abs().mean(), '平均偏差（基点）': error.mean(),
                })
    return pd.DataFrame(rows)


intervals = []
for title, start, end in PERIODS:
    start = max(start, sample_start)
    end = min(end, sample_end)
    if start <= end:
        intervals.append((title, start, end))

requested_range = sample_start, sample_end
default_range = PERIODS[0][1], PERIODS[-1][2]
if requested_range != default_range and all((start, end) != requested_range for _, start, end in intervals):
    intervals.insert(0, ('所选区间', *requested_range))

print('\n每格依次为均方根误差（基点）／相关系数（有效配对交易日数）。')
comparisons = {}
for title, start, end in intervals:
    columns = []
    for name, factors in results.items():
        metrics = comparison_metrics(factors, official, start, end)
        if metrics.empty:
            continue
        metrics = metrics.set_index(['模型', '因子', '加权'])
        cells = [
            f'{error:.3f}／{corr:.6f}（{n_days:,}）' if pd.notna(error) else '—'
            for error, corr, n_days in zip(
                metrics['均方根误差（基点）'], metrics['相关系数'], metrics['交易日数'],
            )
        ]
        columns.append(pd.Series(cells, index=metrics.index, name=name))

    print(f'\n{title}  {start.date()} 至 {end.date()}')
    if columns:
        comparison = pd.concat(columns, axis=1).reset_index()
        comparisons[title] = comparison
        with pd.option_context('display.unicode.east_asian_width', True):
            print(comparison.to_string(index=False, na_rep='—'))
    else:
        print('该区间没有可对照的交易日。')
