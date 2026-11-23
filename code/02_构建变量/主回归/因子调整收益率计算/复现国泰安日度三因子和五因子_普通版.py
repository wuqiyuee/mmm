"""重算并保存沪深日度三因子和五因子，供主回归的因子调整收益使用。"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SHARED_CODE_DIR = ROOT / 'code/06_交易策略'
if str(SHARED_CODE_DIR) not in sys.path:
    sys.path.append(str(SHARED_CODE_DIR))

from 复现国泰安日度三因子和五因子 import input_path, load_industry
from 日度因子工具 import make_ff_2x3_factors, weighted_mean_by_group

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


@dataclass(frozen=True)
class Specification:
    """普通版只使用固定口径。探索版借此对照已试算的规则。"""

    name: str
    split_b_share_prices: bool = True
    historical_main_board: bool = True
    december_last_trade: bool = True
    december_min_gap: int = 0
    june_restructure_cap: bool = False
    profit_equity: str = 'before_2023'
    ipo_from_offer_price: bool = True
    five_calendar_cap: bool = True


ORDINARY = Specification('普通版')


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
                 'TradableSharesH', 'TradeSharesTotal', 'ChangeReasonID'],
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
    suffix = 'split' if spec.split_b_share_prices else 'base'
    float_col = f'float_{suffix}'
    total_col = f'total_{suffix}'
    june = june.loc[june[float_col].gt(0)].set_index('Stkcd').copy()
    december = december.set_index('Stkcd').copy()
    june['float_cap'] = june[float_col]
    june['total_cap'] = june[total_col]

    if spec.june_restructure_cap:
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
        if spec.split_b_share_prices:
            total = actual['Dsmvtl3'].where(june['has_b_shares'].eq(True), total).fillna(total)
        floating = actual['Dsmvosd'] + (total - actual['Dsmvtll']).clip(lower=0)
        june['float_cap'] = june['float_cap'].mask(replace & floating.gt(0), floating)
        june['total_cap'] = june['total_cap'].mask(replace & total.gt(0), total)

    december['value_cap3'] = december[float_col]
    december['value_cap5'] = december[float_col]
    if spec.five_calendar_cap:
        # 跨市场上市公司有上年 12 月 31 日股本明细时，以该流通股数乘上年末收盘价。
        year_end = pd.Timestamp(june_end.year - 1, 12, 31)
        year_end_shares = shares.loc[
            shares['date'].eq(year_end) & shares['DeclareDate'].le(june_end)
        ].set_index('Stkcd').reindex(december.index)
        cross_listed = december['cross_listed'].eq(True) | year_end_shares['cross_listed'].eq(True)
        shares_count = year_end_shares['TradeSharesTotal']
        exact = shares_count.where(cross_listed & shares_count.gt(0)) * december['Clsprc'] / 1000
        december['value_cap5'] = exact.fillna(december[float_col])

    if spec.december_last_trade:
        actual = last_december.reindex(december.index)
        gap = days.at[december_end] - actual['date'].map(days)
        use_actual = december['Filling'].eq(1) & gap.ge(spec.december_min_gap) & actual[float_col].notna()
        for col in ('value_cap3', 'value_cap5'):
            december[col] = december[col].mask(use_actual, actual[float_col])

    june_main = june['Markettype'].isin(MAIN_BOARD)
    december_main = december['Markettype'].isin(MAIN_BOARD)
    if spec.historical_main_board and june_end < SME_MERGER:
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
    use_signed = spec.profit_equity == 'signed' or (spec.profit_equity == 'before_2023' and before_change)
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


def market_returns(daily, previous_december, days, risk_free, spec):
    suffix = 'split' if spec.split_b_share_prices else 'base'
    return_col = 'offer_return' if spec.ipo_from_offer_price else 'Dretwd'
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
    data['return'] = data['offer_return'] if spec.ipo_from_offer_price else data['Dretwd']
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


def calculate_factors(input_dir, rate_file, start, end, specifications=(ORDINARY,)):
    """一次读入原始行情，逐年复算指定口径。官方因子表不参加计算。"""
    input_dir = Path(input_dir)
    specifications = tuple(specifications)
    calendar = pd.read_csv(input_path(input_dir, 'CER_Cale.csv'), parse_dates=['Clddt'])
    trading = calendar.loc[calendar['Markettype'].eq(1) & calendar['Statse'].eq('O'), 'Clddt']
    trading = trading.drop_duplicates().sort_values()
    days = pd.Series(np.arange(len(trading)), index=pd.DatetimeIndex(trading))
    month_ends = trading.groupby([trading.dt.year, trading.dt.month]).max()
    requested = trading.loc[trading.between(start, end)]
    if requested.empty:
        raise ValueError('所选区间没有交易日')
    rate = pd.read_csv(rate_file, parse_dates=['ChangeDate']).sort_values('ChangeDate')
    rate = rate.loc[rate['ChangeDate'].le('2015-10-24'), ['ChangeDate', 'LumpFixed3Month']]
    rate = pd.merge_asof(requested.to_frame('date'), rate, left_on='date', right_on='ChangeDate').set_index('date')
    rf = np.expm1(np.log1p(rate['LumpFixed3Month'] / 100) / 365).rename('rf')
    if rf.isna().any():
        raise ValueError('部分交易日缺少三个月定存基准利率')

    financials = load_financials(input_dir)
    shares = load_share_structure(input_dir)
    industry = load_industry(input_dir)
    companies = pd.read_csv(
        input_path(input_dir, 'TRD_Co.csv'), usecols=['Stkcd', 'Listdt', 'Ipoprc'],
        dtype={'Stkcd': 'int32'}, parse_dates=['Listdt'],
    ).drop_duplicates('Stkcd').set_index('Stkcd')

    results = {spec.name: {3: [], 5: []} for spec in specifications}
    prior_labels = {}
    last_december = previous_december = None
    seen = pd.DatetimeIndex([])
    needs_june_history = any(spec.june_restructure_cap for spec in specifications)
    for year, daily in read_daily_years(input_dir, end.year, days.index, shares, companies):
        december_end = month_ends.get((year, 12), pd.NaT)
        dec = daily.loc[daily['date'].eq(december_end)]
        year_rf = rf.loc[rf.index.year == year]
        if year >= start.year - 1 and previous_december is not None:
            june_end = month_ends.get((year, 6), pd.NaT)
            need_formation = year < end.year or end.month >= 7
            current_labels = {}
            if need_formation:
                june = daily.loc[daily['date'].eq(june_end)]
                if june.empty or previous_december.empty:
                    raise ValueError(f'{year} 年组合缺少六月末或上年十二月末行情')
                last_june = last_actual_trades(last_december, daily, june_end) if needs_june_history else None
                annual = financials.loc[financials['year'].eq(year - 1)]
                for spec in specifications:
                    current_labels[spec.name] = form_portfolios(
                        june, previous_december, annual, shares, industry, days, june_end,
                        month_ends.at[(year - 1, 12)], last_december, last_june, spec,
                    )

            if not year_rf.empty:
                print(f'计算 {year} 年日度因子', flush=True)
                selected = daily.loc[daily['date'].isin(year_rf.index)].copy()
                has_trade = selected['Filling'].eq(0) & selected['Dretwd'].notna()
                seen = seen.union(pd.DatetimeIndex(selected.loc[has_trade, 'date'].unique()))
                selected['financial'] = mark_financials(
                    selected.loc[selected['date'].lt(PROFIT_CHANGE) & selected['Filling'].eq(0)], industry,
                ).reindex(selected.index).eq(True)
                market_cache = {}
                for spec in specifications:
                    market_key = spec.split_b_share_prices, spec.ipo_from_offer_price
                    if market_key not in market_cache:
                        market_cache[market_key] = market_returns(daily, previous_december, days, year_rf, spec)
                    market = market_cache[market_key]
                    portfolios = {3: [], 5: []}
                    for half, labels in ((False, prior_labels), (True, current_labels)):
                        part = selected.loc[selected['date'].dt.month.ge(7).eq(half)]
                        if not part.empty:
                            if spec.name not in labels:
                                raise ValueError(f'{year} 年的日度收益缺少对应六月分组')
                            for model, factor in portfolio_returns(part, labels[spec.name], spec).items():
                                portfolios[model].append(factor)
                    for model in FACTORS:
                        returns = pd.concat(portfolios[model])
                        table = market.join(returns, validate='one_to_one').rename_axis('date').reset_index()
                        results[spec.name][model].append(table)
            prior_labels = current_labels

        if not dec.empty:
            previous_december = dec
            last_december = last_actual_trades(last_december, daily, december_end)

    missing = pd.DatetimeIndex(requested).difference(seen)
    if not missing.empty:
        raise ValueError(f'原始行情缺少 {len(missing)} 个交易日，首个为 {missing[0].date()}')
    return {
        name: {model: pd.concat(parts, ignore_index=True) for model, parts in models.items()}
        for name, models in results.items()
    }


def load_official(input_dir):
    result = {}
    for model, file in ((3, 'STK_MKT_THRFACDAY.csv'), (5, 'STK_MKT_FIVEFACDAY.csv')):
        data = pd.read_csv(input_path(Path(input_dir), file), parse_dates=['TradingDate'])
        subset = data['MarkettypeID'].eq('P9714')
        if model == 5:
            subset &= data['Portfolios'].eq(1)
        cols = [f'{factor}{i}' for factor in FACTORS[model] for i in WEIGHTS]
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
            columns = ['date', 'rf', *[f'{factor}{i}' for factor in FACTORS[model] for i in WEIGHTS]]
            table = results[model].loc[:, columns].sort_values('date')
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
    results = calculate_factors(args.input_dir, args.rate_file, args.start, args.end)[ORDINARY.name]
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
