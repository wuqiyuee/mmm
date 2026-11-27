"""重算并保存沪深日度三因子和五因子，供主回归的因子调整收益使用。"""

from pathlib import Path
import argparse
import re
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

INPUT_DIR = ROOT / 'input/CSMAR'
OUTPUT_DIR = ROOT / '中间结果/02_变量构造结果/因子调整收益率/日度因子'
OUTPUT_FILES = {
    3: OUTPUT_DIR / '日度三因子.csv',
    5: OUTPUT_DIR / '日度五因子.csv',
}
RATE_FILE = (
    Path.home() / 'Desktop/课程学习/研一/研一下/FF-3/input'
    / '金融机构人民币存贷款基准利率调整表（1952-09-15至2025-05-20）/MBK_FINRMBDEPOSITLOAND.csv'
)
START = pd.Timestamp('2000-01-01')
END = pd.Timestamp('2026-08-19')
MAIN_END = pd.Timestamp('2025-12-31')
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


def input_path(folder, filename):
    matches = sorted(folder.glob(f'*/{filename}'))
    if len(matches) != 1:
        raise ValueError(f'{folder} 应有一份 {filename}，找到 {len(matches)} 份')
    return matches[0]


def factor_return_columns(model):
    """按正式输出顺序返回指定模型的因子收益列名。"""
    return [f'{factor}{weight}' for factor in FACTORS[model] for weight in WEIGHTS]


def load_industry(input_dir):
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
    return {
        key: group[['date', 'Stkcd', 'finance']].sort_values('date').reset_index(drop=True)
        for key, group in industry.groupby('standard')
    }


def load_financials(input_dir):
    """只取年末合并报表，按实际公告日期保留分组时已经可用的年报。"""
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
    return data[['Stkcd', 'year', 'announced', 'book', 'operating_profit', 'blank_financial_profit', 'inv']]


def load_share_structure(input_dir):
    paths = sorted((input_dir / '专题研究系列-事件研究').glob('*/ER_Structure.csv'))
    if len(paths) != 1:
        raise ValueError(f'股本变动事件表应有一份，实际找到 {len(paths)} 份')
    shares = pd.read_csv(
        paths[0],
        usecols=['Symbol', 'ChangeDate', 'DeclareDate', 'TradableSharesB',
                 'TradableSharesH', 'TradeSharesTotal'],
        dtype={'Symbol': 'int32'},
    ).rename(columns={'Symbol': 'Stkcd', 'ChangeDate': 'date'})
    for col in ('date', 'DeclareDate'):
        shares[col] = pd.to_datetime(shares[col], format='%Y-%m-%d', errors='coerce')
    shares['has_b_shares'] = shares['TradableSharesB'].gt(0)
    shares['cross_listed'] = shares['has_b_shares'] | shares['TradableSharesH'].gt(0)
    return shares.dropna(subset=['date']).sort_values('date', kind='stable')


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
            daily_quotes = part.rename(columns={'Trddt': 'date'}).drop(columns='year')
            daily_quotes['date'] = pd.to_datetime(daily_quotes['date'], format='%Y-%m-%d')
            daily_quotes = daily_quotes.loc[daily_quotes['date'].isin(trading_days)]
            daily_quotes = daily_quotes.sort_values('date').reset_index(drop=True)
            if daily_quotes.duplicated(['Stkcd', 'date']).any():
                raise ValueError(f'{year} 年个股行情存在重复的股票和交易日')
            daily_quotes = pd.merge_asof(daily_quotes, flags, on='date', by='Stkcd')
            # 有 B 股时用分别计价的总市值，缺失时用原总市值。
            daily_quotes['daily_total_cap'] = daily_quotes['Dsmvtl3'].where(
                daily_quotes['has_b_shares'].eq(True), daily_quotes['Dsmvtl2']
            )
            daily_quotes['daily_total_cap'] = daily_quotes['daily_total_cap'].fillna(daily_quotes['Dsmvtl2'])
            daily_quotes['daily_float_cap'] = daily_quotes['Dsmvosd'] + (
                daily_quotes['daily_total_cap'] - daily_quotes['Dsmvtll']
            ).clip(lower=0)
            issue_price = daily_quotes['Stkcd'].map(companies['Ipoprc'])
            listed_today = daily_quotes['date'].eq(
                daily_quotes['Stkcd'].map(companies['Listdt'])
            ) & issue_price.gt(0)
            daily_quotes['offer_return'] = daily_quotes['Dretwd'].where(
                ~listed_today, daily_quotes['Clsprc'].div(issue_price) - 1
            )
            daily_quotes = daily_quotes.drop(columns=[*CAP_COLUMNS, 'has_b_shares'])
            yield int(year), daily_quotes


def last_actual_trades(previous, daily_quotes, cutoff):
    cols = ['Stkcd', 'date', 'daily_float_cap']
    sample = daily_quotes.loc[daily_quotes['Filling'].eq(0) & daily_quotes['date'].le(cutoff), cols]
    if previous is not None:
        sample = pd.concat([previous.reset_index(), sample], ignore_index=True)
    return sample.sort_values('date').drop_duplicates('Stkcd', keep='last').set_index('Stkcd')


def form_portfolios(
    june_quotes, december_quotes, annual_reports, shares, industry,
    trading_day_numbers, june_end, december_end, latest_december_trades,
):
    """六月规模组和十二月特征组分别划分，再按证券代码取交集。"""
    june_members = june_quotes.loc[june_quotes['daily_float_cap'].gt(0)].set_index('Stkcd').copy()
    december_features = december_quotes.set_index('Stkcd').copy()
    june_members['float_cap'] = june_members['daily_float_cap']
    june_members['total_cap'] = june_members['daily_total_cap']

    december_features['value_cap3'] = december_features['daily_float_cap']
    # 五因子对有年末股本明细的跨市场上市公司使用流通股数乘年末收盘价。
    year_end = pd.Timestamp(june_end.year - 1, 12, 31)
    year_end_shares = shares.loc[
        shares['date'].eq(year_end) & shares['DeclareDate'].le(june_end)
    ].set_index('Stkcd').reindex(december_features.index)
    cross_listed = december_features['cross_listed'].eq(True) | year_end_shares['cross_listed'].eq(True)
    shares_count = year_end_shares['TradeSharesTotal']
    exact = shares_count.where(cross_listed & shares_count.gt(0)) * december_features['Clsprc'] / 1000
    december_features['value_cap5'] = exact.fillna(december_features['daily_float_cap'])

    # 年末停牌填充且有有效的最近成交市值时，三因子和五因子均替换。
    actual = latest_december_trades.reindex(december_features.index)
    gap = trading_day_numbers.at[december_end] - actual['date'].map(trading_day_numbers)
    use_actual = december_features['Filling'].eq(1) & gap.ge(0) & actual['daily_float_cap'].notna()
    for col in ('value_cap3', 'value_cap5'):
        december_features[col] = december_features[col].mask(use_actual, actual['daily_float_cap'])

    june_main = june_members['Markettype'].isin(MAIN_BOARD)
    december_main = december_features['Markettype'].isin(MAIN_BOARD)
    if june_end < SME_MERGER:
        # 两套断点都按六月换组时的板块状态选主板。中小板股票仍能入组。
        june_main &= ~june_members.index.to_series().between(2000, 3999)
        december_main &= ~december_features.index.to_series().between(2000, 3999)
    median = june_members.loc[june_main, 'float_cap'].median()
    if pd.isna(median):
        raise ValueError(f'{june_end.date()} 没有可用于规模断点的主板市值')
    june_members['size_group'] = np.where(june_members['float_cap'].ge(median), 'B', 'S')

    # 六月断点是交易日，非交易日公告映射到下一交易日后，此处的筛选结果相同。
    available_reports = annual_reports.loc[annual_reports['announced'].le(june_end)].copy()
    is_financial = mark_financials(available_reports.assign(date=december_end), industry)
    before_change = june_end.year < PROFIT_CHANGE.year
    book = available_reports['book']
    profit_equity = book.where(book.ne(0) if before_change else book.gt(0))
    profitability = available_reports['operating_profit'].div(profit_equity)
    # 2023 年换组起，四项利润科目全空的正权益金融股以零作为排序值。
    if before_change:
        available_reports['op'] = profitability.mask(is_financial)
    else:
        available_reports['op'] = profitability.mask(
            is_financial & available_reports['blank_financial_profit'], 0
        )
    december_features = december_features.join(
        available_reports.set_index('Stkcd')[['book', 'op', 'inv']], validate='one_to_one'
    )
    for feature_name, cap in (('bm3', 'value_cap3'), ('bm5', 'value_cap5')):
        december_features[feature_name] = december_features['book'].where(december_features['book'].gt(0)).div(
            december_features[cap].where(december_features[cap].gt(0)) * 1000
        )
    # 保留排序数值，分组标签写入独立的列。
    feature_columns = ('bm3', 'bm5', 'op', 'inv')
    feature_group_columns = [f'{name}_group' for name in feature_columns]
    for feature_name in feature_columns:
        group_column = f'{feature_name}_group'
        values = december_features[feature_name]
        low, high = values.loc[december_main].quantile([0.30, 0.70])
        december_features[group_column] = pd.Series(
            np.select([values.ge(high), values.lt(low), values.notna() & pd.notna(low)],
                      ['H', 'L', 'M'], default=''), index=values.index,
        ).replace('', np.nan)
    return june_members[['size_group', *WEIGHTS.values()]].join(
        december_features[feature_group_columns], how='inner'
    ).reset_index()


def weighted_mean_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    return_col: str,
    weight_col: str,
    value_name: str,
) -> pd.DataFrame:
    """Calculate value-weighted returns by group.

    Missing returns and non-positive weights are excluded from both numerator
    and denominator.
    """
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


def make_ff_2x3_factors(group_returns: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """由六组收益计算规模收益差和特征收益差。"""
    wide_returns = group_returns.pivot_table(
        index='date',
        columns=['size_group', f'{feature_name}_group'],
        values='portfolio_return',
        aggfunc='first',
    )

    required_groups = (('S', 'L'), ('S', 'M'), ('S', 'H'), ('B', 'L'), ('B', 'M'), ('B', 'H'))
    for group_key in required_groups:
        if group_key not in wide_returns.columns:
            wide_returns[group_key] = pd.NA

    daily_spreads = pd.DataFrame(index=wide_returns.index)
    daily_spreads[f'size_{feature_name}'] = (
        (wide_returns[('S', 'L')] + wide_returns[('S', 'M')] + wide_returns[('S', 'H')]) / 3
        - (wide_returns[('B', 'L')] + wide_returns[('B', 'M')] + wide_returns[('B', 'H')]) / 3
    )
    daily_spreads[feature_name] = (
        (wide_returns[('S', 'H')] + wide_returns[('B', 'H')]) / 2
        - (wide_returns[('S', 'L')] + wide_returns[('B', 'L')]) / 2
    )

    return daily_spreads.rename_axis('date')


def market_returns(daily_quotes, previous_december_quotes, trading_day_numbers, risk_free_rates):
    cols = ['Stkcd', 'date', 'Filling', 'offer_return', 'daily_float_cap', 'daily_total_cap']
    sample = pd.concat(
        [previous_december_quotes[cols], daily_quotes[cols]], ignore_index=True
    ).sort_values(['Stkcd', 'date'])
    stock = sample['Stkcd']
    number = sample['date'].map(trading_day_numbers)
    adjacent = stock.eq(stock.shift()) & number.diff().eq(1)
    for source, target in (('daily_float_cap', 'lag_float'), ('daily_total_cap', 'lag_total')):
        weight = sample[source]
        # 上一开市日的市值可来自停牌填充。新上市等无前日记录时用当日市值。
        sample[target] = weight.shift().where(adjacent, weight)
    sample = sample.loc[sample['Filling'].eq(0) & sample['date'].isin(risk_free_rates.index)]
    result = risk_free_rates.to_frame()
    for i, weight in ((1, 'lag_float'), (2, 'lag_total')):
        name = f'Market{i}'
        value = weighted_mean_by_group(sample, ['date'], 'offer_return', weight, name).set_index('date')[name]
        result[name] = value
        result[f'RiskPremium{i}'] = result[name] - result['rf']
    return result


def portfolio_returns(daily_quotes, formation_labels):
    """按固定六月权重计算六组日收益，再计算组合间收益差。"""
    valid_quotes = daily_quotes['Filling'].eq(0) & daily_quotes['Dretwd'].notna()
    labeled_quotes = daily_quotes.loc[valid_quotes].merge(
        formation_labels, on='Stkcd', how='inner', validate='many_to_one',
    )
    exclude_profitability = (
        labeled_quotes['date'].lt(PROFIT_CHANGE) & labeled_quotes['financial']
    )
    labeled_quotes.loc[exclude_profitability, 'op_group'] = np.nan
    model_parts = {3: [], 5: []}
    for weight_method, weight_column in WEIGHTS.items():
        spread_parts = []
        for feature_name in ('bm3', 'bm5', 'op', 'inv'):
            group_column = f'{feature_name}_group'
            feature_quotes = labeled_quotes.loc[labeled_quotes[group_column].notna()]
            group_returns = weighted_mean_by_group(
                feature_quotes, ['date', 'size_group', group_column],
                'offer_return', weight_column, 'portfolio_return',
            )
            spread_parts.append(make_ff_2x3_factors(group_returns, feature_name))
        feature_spreads = pd.concat(spread_parts, axis=1)
        three_factors = pd.DataFrame({'SMB': feature_spreads['size_bm3'], 'HML': feature_spreads['bm3']})
        five_factors = pd.DataFrame({
            'SMB': feature_spreads[['size_bm5', 'size_op', 'size_inv']].mean(axis=1, skipna=False),
            'HML': feature_spreads['bm5'], 'RMW': feature_spreads['op'], 'CMA': -feature_spreads['inv'],
        })
        model_parts[3].append(three_factors.add_suffix(str(weight_method)))
        model_parts[5].append(five_factors.add_suffix(str(weight_method)))
    return {model: pd.concat(parts, axis=1) for model, parts in model_parts.items()}


def calculate_factors(input_dir, rate_file, start, end):
    """一次读入原始行情，逐年计算日度三因子和五因子。"""
    input_dir = Path(input_dir)
    calendar = pd.read_csv(input_path(input_dir, 'CER_Cale.csv'), parse_dates=['Clddt'])
    trading_dates = calendar.loc[calendar['Markettype'].eq(1) & calendar['Statse'].eq('O'), 'Clddt']
    trading_dates = trading_dates.drop_duplicates().sort_values()
    # 以交易日序号之差识别相邻交易日。
    trading_day_numbers = pd.Series(
        np.arange(len(trading_dates)), index=pd.DatetimeIndex(trading_dates)
    )
    month_ends = trading_dates.groupby([trading_dates.dt.year, trading_dates.dt.month]).max()
    requested_trading_dates = trading_dates.loc[trading_dates.between(start, end)]
    if requested_trading_dates.empty:
        raise ValueError('所选区间没有交易日')
    rate = pd.read_csv(rate_file, parse_dates=['ChangeDate']).sort_values('ChangeDate')
    rate = rate.loc[rate['ChangeDate'].le('2015-10-24'), ['ChangeDate', 'LumpFixed3Month']]
    rate = pd.merge_asof(
        requested_trading_dates.to_frame('date'), rate, left_on='date', right_on='ChangeDate'
    ).set_index('date')
    risk_free_rates = np.expm1(np.log1p(rate['LumpFixed3Month'] / 100) / 365).rename('rf')
    if risk_free_rates.isna().any():
        raise ValueError('部分交易日缺少三个月定存基准利率')

    financials = load_financials(input_dir)
    shares = load_share_structure(input_dir)
    industry = load_industry(input_dir)
    companies = pd.read_csv(
        input_path(input_dir, 'TRD_Co.csv'), usecols=['Stkcd', 'Listdt', 'Ipoprc'],
        dtype={'Stkcd': 'int32'}, parse_dates=['Listdt'],
    ).drop_duplicates('Stkcd').set_index('Stkcd')

    yearly_factors = {3: [], 5: []}
    prior_labels = None
    last_december_trades = previous_december = None
    covered_trade_dates = pd.DatetimeIndex([])
    for quote_year, daily_quotes in read_daily_years(
        input_dir, end.year, trading_day_numbers.index, shares, companies
    ):
        december_end = month_ends.get((quote_year, 12), pd.NaT)
        december_quotes = daily_quotes.loc[daily_quotes['date'].eq(december_end)]
        year_risk_free = risk_free_rates.loc[risk_free_rates.index.year == quote_year]
        # 起始年前一年的六月分组供首年上半年使用。
        if quote_year >= start.year - 1 and previous_december is not None:
            june_end = month_ends.get((quote_year, 6), pd.NaT)
            need_formation = quote_year < end.year or end.month >= 7
            current_labels = None
            if need_formation:
                june_quotes = daily_quotes.loc[daily_quotes['date'].eq(june_end)]
                if june_quotes.empty:
                    raise ValueError(f'{quote_year} 年组合缺少六月末行情')
                annual_reports = financials.loc[financials['year'].eq(quote_year - 1)]
                current_labels = form_portfolios(
                    june_quotes, previous_december, annual_reports, shares, industry,
                    trading_day_numbers, june_end, month_ends.at[(quote_year - 1, 12)],
                    last_december_trades,
                )

            # 当年上半年沿用上一轮标签，下半年使用当年六月标签。
            if not year_risk_free.empty:
                print(f'计算 {quote_year} 年日度因子', flush=True)
                selected_quotes = daily_quotes.loc[daily_quotes['date'].isin(year_risk_free.index)].copy()
                has_trade = selected_quotes['Filling'].eq(0) & selected_quotes['Dretwd'].notna()
                covered_trade_dates = covered_trade_dates.union(
                    pd.DatetimeIndex(selected_quotes.loc[has_trade, 'date'].unique())
                )
                selected_quotes['financial'] = mark_financials(
                    selected_quotes.loc[
                        selected_quotes['date'].lt(PROFIT_CHANGE) & selected_quotes['Filling'].eq(0)
                    ],
                    industry,
                ).reindex(selected_quotes.index).eq(True)
                year_market = market_returns(
                    daily_quotes, previous_december, trading_day_numbers, year_risk_free
                )
                year_portfolios = {3: [], 5: []}
                for second_half, formation_labels in ((False, prior_labels), (True, current_labels)):
                    part = selected_quotes.loc[selected_quotes['date'].dt.month.ge(7).eq(second_half)]
                    if not part.empty:
                        if formation_labels is None:
                            raise ValueError(f'{quote_year} 年的日度收益缺少对应六月分组')
                        for model, factor in portfolio_returns(part, formation_labels).items():
                            year_portfolios[model].append(factor)
                for model in FACTORS:
                    year_feature_factors = pd.concat(year_portfolios[model])
                    year_factor_table = year_market.join(
                        year_feature_factors, validate='one_to_one'
                    ).rename_axis('date').reset_index()
                    yearly_factors[model].append(year_factor_table)
            prior_labels = current_labels

        # 无请求交易日的历史年份也更新十二月行情和最近实际成交。
        if not december_quotes.empty:
            previous_december = december_quotes
            last_december_trades = last_actual_trades(last_december_trades, daily_quotes, december_end)

    missing = pd.DatetimeIndex(requested_trading_dates).difference(covered_trade_dates)
    if not missing.empty:
        raise ValueError(f'原始行情缺少 {len(missing)} 个交易日，首个为 {missing[0].date()}')
    results = {}  # 正式因子表按模型存放，交易日升序。
    for model, parts in yearly_factors.items():
        formal_columns = ['date', 'rf', *factor_return_columns(model)]
        combined_factors = pd.concat(parts, ignore_index=True)
        results[model] = combined_factors.loc[:, formal_columns].sort_values('date', ignore_index=True)
    return results


def load_official(input_dir):
    result = {}
    for model, file in ((3, 'STK_MKT_THRFACDAY.csv'), (5, 'STK_MKT_FIVEFACDAY.csv')):
        data = pd.read_csv(input_path(Path(input_dir), file), parse_dates=['TradingDate'])
        subset = data['MarkettypeID'].eq('P9714')
        if model == 5:
            subset &= data['Portfolios'].eq(1)
        cols = factor_return_columns(model)
        result[model] = data.loc[subset, ['TradingDate', *cols]].rename(columns={'TradingDate': 'date'})
    return result


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


def build_argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input-dir', type=Path, default=INPUT_DIR, help='国泰安原始数据目录')
    parser.add_argument('--rate-file', type=Path, default=RATE_FILE, help='三个月定存基准利率文件')
    parser.add_argument('--start', type=pd.Timestamp, default=START, help='起始日期，默认 2000-01-01')
    parser.add_argument('--end', type=pd.Timestamp, default=END, help='终止日期，默认 2026-08-19')
    return parser


def validate_arguments(parser, args):
    if not START <= args.start <= args.end <= END:
        parser.error('日期须在 2000-01-01 至 2026-08-19 之间，且起始日期不能晚于终止日期')
    if not args.input_dir.is_dir():
        parser.error(f'找不到国泰安原始数据目录 {args.input_dir}')
    if not args.rate_file.is_file():
        parser.error(f'找不到基准利率表 {args.rate_file}，请指定利率文件路径')


def save_daily_factors(results):
    """每张表一行一个交易日，保存小数形式的因子收益和无风险利率。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for model, path in OUTPUT_FILES.items():
            table = results[model]
            with path.open('x', encoding='utf-8-sig', newline='') as file:
                created.append(path)
                table.to_csv(file, index=False, date_format='%Y-%m-%d')
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    for path in created:
        print(f'已保存 {path}')


def main():
    parser = build_argument_parser(__doc__)
    args = parser.parse_args()
    validate_arguments(parser, args)
    for path in OUTPUT_FILES.values():
        if path.exists():
            parser.error(f'输出文件已存在，请先核对并移走旧文件，避免覆盖。{path}')
    results = calculate_factors(args.input_dir, args.rate_file, args.start, args.end)
    official = load_official(args.input_dir)
    for title, period_start, period_end in (
        ('主样本', START, MAIN_END),
        ('2023 年下半年至 2025 年', PROFIT_CHANGE, MAIN_END),
        ('2026 年上半年', pd.Timestamp('2026-01-01'), pd.Timestamp('2026-06-30')),
        ('2026 年六月换组后', pd.Timestamp('2026-07-01'), END),
    ):
        begin, end = max(args.start, period_start), min(args.end, period_end)
        if begin > end:
            continue
        comparison = comparison_metrics(results, official, begin, end)
        if not comparison.empty:
            print(f'\n{title}  {begin.date()} 至 {end.date()}')
            with pd.option_context('display.unicode.east_asian_width', True):
                print(comparison.to_string(index=False, float_format=lambda value: f'{value:.6f}'))
    save_daily_factors(results)


if __name__ == '__main__':
    main()
