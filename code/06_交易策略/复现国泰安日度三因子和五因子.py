"""按国泰安说明书复现日度三因子和二乘三五因子，并与官方日数据对照。"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd

from 日度因子工具 import make_ff_2x3_factors, weighted_mean_by_group

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'input/CSMAR'
RATE_FILE = (
    Path.home() / 'Desktop/课程学习/研一/研一下/FF-3/input'
    / '金融机构人民币存贷款基准利率调整表（1952-09-15至2025-05-20）/MBK_FINRMBDEPOSITLOAND.csv'
)
MARKETS = (1, 4, 16, 32)
MAIN_BOARD = (1, 4)
ST_STATES = (2, 3, 5, 6, 8, 9, 11, 12, 14, 15)
WEIGHTS = {1: 'lag_float', 2: 'lag_total'}
START_3 = pd.Timestamp('1996-07-01')
START_5 = pd.Timestamp('2004-07-01')


def input_path(folder, filename):
    matches = sorted(folder.glob(f'*/{filename}'))
    if len(matches) != 1:
        raise ValueError(f'{folder} 应有一份 {filename}，找到 {len(matches)} 份')
    return matches[0]


def load_financials(input_dir):
    """读取年末合并财务报表，计算分组所需的账面权益、盈利能力和投资。"""
    input_dir = Path(input_dir)
    keys = ["Stkcd", "year"]
    balance_items = {
        "A003000000": "equity",
        "A002208000": "deferred_tax_liability",
        "A001222000": "deferred_tax_asset",
        "A003112101": "preferred_stock",
        "A001000000": "assets",
    }
    income_items = {
        "B001101000": "revenue",
        "B001201000": "cost",
        "B001209000": "selling_expense",
        "B001210000": "admin_expense",
        "B001211101": "reported_interest",
    }
    source_cols = {
        "FS_Combas.csv": ["Typrep", *balance_items],
        "FS_Comins.csv": ["Typrep", *income_items],
        "FN_Fn051.csv": ["Typrep", "Sgnyea", "Fn05101", "Fn05102", "Fn05103"],
        "IAR_Rept.csv": ["Reptyp", "Annodt"],
    }
    tables = {}
    for filename, fields in source_cols.items():
        table = pd.read_csv(
            input_path(input_dir, filename),
            usecols=["Stkcd", "Accper", *fields],
            dtype={"Stkcd": str, "Accper": str},
        )
        table = table.loc[table["Accper"].str.fullmatch(r"\d{4}-12-31", na=False)].copy()
        table["Stkcd"] = table["Stkcd"].str.strip().str.zfill(6)
        table["year"] = table["Accper"].str[:4].astype(int)
        tables[filename] = table

    balance = tables["FS_Combas.csv"]
    balance = balance.loc[balance["Typrep"].eq("A"), [*keys, *balance_items]].rename(
        columns=balance_items
    )
    financials = tables["FS_Comins.csv"]
    financials = financials.loc[
        financials["Typrep"].eq("A"), [*keys, *income_items]
    ].rename(columns=income_items)

    # 附注只取当期已费用化利息。上期比较数可能在相应的六月分组之后才披露。
    notes = tables["FN_Fn051.csv"]
    notes = notes.loc[
        notes["Typrep"].eq(1)
        & notes["Sgnyea"].eq(1)
        & notes["Fn05101"].eq(1)
        & notes["Fn05103"].eq("CNY")
        & notes["Fn05102"].notna(),
        [*keys, "Fn05102"],
    ].rename(columns={"Fn05102": "note_interest"})

    reports = tables["IAR_Rept.csv"]
    reports = reports.loc[reports["Reptyp"].eq(4), [*keys, "Annodt"]].rename(
        columns={"Annodt": "announced"}
    )
    reports["announced"] = pd.to_datetime(
        reports["announced"], format="%Y-%m-%d", errors="coerce"
    )

    previous_assets = balance[[*keys, "assets"]].rename(columns={"assets": "previous_assets"})
    previous_assets["year"] += 1
    panel = balance.merge(previous_assets, on=keys, how="left", validate="one_to_one")
    panel = panel.merge(financials, on=keys, how="left", validate="one_to_one")
    panel = panel.merge(notes, on=keys, how="left", validate="one_to_one")
    panel = panel.merge(reports, on=keys, how="left", validate="one_to_one")

    # 递延税资产、负债和优先股的空科目按零处理，保留历史科目的原始正负号。
    panel["book"] = (
        panel["equity"]
        + panel["deferred_tax_liability"].fillna(0)
        - panel["deferred_tax_asset"].fillna(0)
        - panel["preferred_stock"].fillna(0)
    )
    # 2018 年之前优先使用附注，之后优先使用利润表，缺失时以另一来源补足。
    interest = panel["reported_interest"].where(
        panel["year"].ge(2018) | panel["note_interest"].isna(), panel["note_interest"]
    ).fillna(panel["note_interest"])
    # 分组环节按各财年实际年报公告日筛选。利润或投资的必需输入缺失时保留缺失。
    panel["op"] = (
        panel["revenue"] - panel["cost"] - panel["selling_expense"] - panel["admin_expense"]
        - interest
    ).div(panel["book"].where(panel["book"].gt(0)))
    panel["inv"] = (
        panel["assets"].where(panel["assets"].gt(0))
        .div(panel["previous_assets"].where(panel["previous_assets"].gt(0))) - 1
    )
    return panel[["Stkcd", "year", "book", "op", "inv", "announced"]]


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


def clean_five_factor_sample(daily, industry):
    # 沿用项目的行业标准切换日，按观察日匹配当时已生效的行业。
    eligible = daily.loc[daily['Trdsta'].notna() & ~daily['Trdsta'].isin(ST_STATES)]
    cutoffs = (pd.Timestamp.min, pd.Timestamp('2012-10-26'), pd.Timestamp('2023-05-01'), pd.Timestamp.max)
    pieces = []
    for standard, left, right in zip(('P0201', 'P0207', 'P0221'), cutoffs[:-1], cutoffs[1:]):
        part = eligible.loc[eligible['date'].ge(left) & eligible['date'].lt(right)]
        if part.empty:
            continue
        tagged = pd.merge_asof(part.sort_values('date'), industry[standard], on='date', by='Stkcd')
        pieces.append(tagged.loc[tagged['finance'].eq(False), eligible.columns])
    return pd.concat(pieces, ignore_index=True) if pieces else eligible.iloc[:0]


def form_portfolios(june, december, annual, signals, formation_date):
    label_columns = ['Stkcd', 'size_group', *signals]
    if june.empty or december.empty or annual.empty:
        return pd.DataFrame(columns=label_columns)
    june = june.loc[june['Dsmvosd'].gt(0), ['Stkcd', 'Markettype', 'Dsmvosd']].copy()
    median = june.loc[june['Markettype'].isin(MAIN_BOARD), 'Dsmvosd'].median()
    if pd.isna(median):
        return pd.DataFrame(columns=label_columns)
    june['size_group'] = np.where(june['Dsmvosd'].ge(median), 'B', 'S')
    snapshot = december[['Stkcd', 'Markettype', 'Dsmvosd']].merge(
        annual.loc[annual['announced'].le(formation_date), ['Stkcd', 'book', 'op', 'inv']],
        on='Stkcd', how='inner', validate='one_to_one',
    )
    # 股价表的市值单位为千元，财报为元。十二月断点单独计算。
    snapshot['bm'] = snapshot['book'].where(snapshot['book'].gt(0)).div(
        snapshot['Dsmvosd'].where(snapshot['Dsmvosd'].gt(0)) * 1000
    )
    main = snapshot.loc[snapshot['Markettype'].isin(MAIN_BOARD)]
    for signal in signals:
        low, high = main[signal].quantile([0.30, 0.70])
        values = snapshot[signal]
        snapshot[signal] = pd.Series(
            np.select([values.ge(high), values.lt(low), values.notna() & pd.notna(low)], ['H', 'L', 'M'], ''),
            index=snapshot.index,
        ).replace('', np.nan)
    return june[['Stkcd', 'size_group']].merge(
        snapshot[['Stkcd', *signals]], on='Stkcd', how='inner', validate='one_to_one'
    )


def read_daily_years(input_dir, first_year, last_year):
    root = input_dir / '个股日度收益率（升级测试）1995-01-01至2026'
    columns = ['Stkcd', 'Trddt', 'Markettype', 'Trdsta', 'Filling', 'Dretwd', 'Dsmvosd', 'Dsmvtll']
    dtypes = {'Stkcd': 'int32', 'Markettype': 'int16', 'Trdsta': 'float32', 'Filling': 'int8'}
    for folder in sorted(root.iterdir()):
        years = re.findall(r'\d{4}', folder.name)
        if not years or int(years[0]) > last_year or int(years[-1]) < first_year:
            continue
        print(f'读取 {folder.name}', flush=True)
        chunks = []
        for path in sorted(folder.glob('TRDNEW_Dalyr*.csv')):
            for chunk in pd.read_csv(path, usecols=columns, dtype=dtypes, chunksize=400_000):
                year = chunk['Trddt'].str.slice(0, 4).astype('int16')
                select = year.between(first_year, last_year) & chunk['Markettype'].isin(MARKETS)
                if select.any():
                    chunks.append(chunk.loc[select].assign(year=year[select]))
        if not chunks:
            continue
        partition = pd.concat(chunks, ignore_index=True)
        del chunks
        for year, daily in partition.groupby('year', sort=True):
            daily = daily.rename(columns={'Trddt': 'date'}).drop(columns='year')
            daily['date'] = pd.to_datetime(daily['date'], format='%Y-%m-%d')
            if daily.duplicated(['Stkcd', 'date']).any():
                raise ValueError(f'{year} 存在重复的股票交易日期')
            yield year, daily


def daily_returns(daily, previous_december, open_days):
    # 昨日必须是上一交易所开市日，且当日和上一开市日都有实际交易。
    valid = pd.concat([previous_december, daily], ignore_index=True)
    valid = valid.loc[valid['Filling'].eq(0)].sort_values(['Stkcd', 'date']).copy()
    day_number = valid['date'].map(open_days)
    grouped = valid.groupby('Stkcd', sort=False)
    adjacent = day_number.sub(day_number.groupby(valid['Stkcd'], sort=False).shift()).eq(1)
    valid['lag_float'] = grouped['Dsmvosd'].shift().where(adjacent)
    valid['lag_total'] = grouped['Dsmvtll'].shift().where(adjacent)
    keep = valid['date'].dt.year.eq(daily['date'].dt.year.iloc[0]) & valid['Dretwd'].notna()
    return valid.loc[keep, ['date', 'Stkcd', 'Trdsta', 'Dretwd', *WEIGHTS.values()]].copy()


def portfolio_factors(panel, weights, signals):
    factors = {}
    for signal in signals:
        eligible = panel.loc[panel[signal].notna()]
        legs = weighted_mean_by_group(
            eligible, ['date', 'size_group', signal], 'Dretwd', weights, 'return'
        )
        factors[signal] = make_ff_2x3_factors(
            legs, 'date', 'size_group', signal, 'return', f'SMB_{signal}', signal
        ).set_index('date')
    result = pd.concat(factors.values(), axis=1)
    # 按小盘减大盘计算投资分组的规模差，修正说明书中的加号。
    result['SMB'] = result[[f'SMB_{signal}' for signal in signals]].mean(axis=1, skipna=False)
    result['HML'] = result['bm']
    if 'op' in signals:
        result['RMW'] = result['op']
        result['CMA'] = -result['inv']
    return result[['SMB', 'HML', *(['RMW', 'CMA'] if 'op' in signals else [])]].reset_index()


def calculate_factors(input_dir, rate_file, start, end):
    cal = pd.read_csv(input_path(input_dir, 'CER_Cale.csv'), parse_dates=['Clddt'])
    dates = cal.loc[cal['Markettype'].eq(1) & cal['Statse'].eq('O'), 'Clddt'].sort_values()
    open_days = pd.Series(np.arange(len(dates)), index=pd.DatetimeIndex(dates))
    month_ends = dates.groupby([dates.dt.year, dates.dt.month]).max()
    dates = pd.DataFrame({'date': dates.loc[dates.between(start, end)].to_numpy()})
    rates = pd.read_csv(rate_file, parse_dates=['ChangeDate']).sort_values('ChangeDate')
    # 人民银行三个月基准利率自 2015-10-24 起为 1.10%。之后本地表的挂牌变动不纳入。
    rates = rates.loc[rates['ChangeDate'].le('2015-10-24'), ['ChangeDate', 'LumpFixed3Month']]
    dates = pd.merge_asof(dates, rates, left_on='date', right_on='ChangeDate')
    dates['rf'] = np.expm1(np.log1p(dates['LumpFixed3Month'] / 100) / 365)
    if dates['rf'].isna().any():
        raise ValueError('部分交易日没有可匹配的三个月定存基准利率')
    financials = load_financials(input_dir)
    financials['Stkcd'] = financials['Stkcd'].astype('int32')
    industry = load_industry(input_dir)
    dec = pd.DataFrame()
    previous_labels = {}
    by_model = {3: [], 5: []}
    for year, daily in read_daily_years(input_dir, start.year - 2, end.year):
        june_end = month_ends[year, 6]
        june = daily.loc[daily['date'].eq(june_end)]
        annual = financials.loc[financials['year'].eq(year - 1)]
        labels = {3: form_portfolios(june, dec, annual, ('bm',), june_end)}
        if year >= START_5.year:
            june_clean = clean_five_factor_sample(june, industry)
            dec_clean = clean_five_factor_sample(dec, industry) if not dec.empty else dec
            labels[5] = form_portfolios(june_clean, dec_clean, annual, ('bm', 'op', 'inv'), june_end)
        if year >= start.year:
            print(f'计算 {year} 年日度因子', flush=True)
            returns = daily_returns(daily, dec, open_days)
            year_dates = dates.loc[dates['date'].dt.year.eq(year), ['date', 'rf']]
            returns = returns.loc[returns['date'].between(start, end)]
            market = year_dates.copy()
            for number, weight in WEIGHTS.items():
                col = f'Market{number}'
                series = weighted_mean_by_group(returns, ['date'], 'Dretwd', weight, col)
                market = market.merge(series[['date', col]], on='date', how='left', validate='one_to_one')
                market[f'RiskPremium{number}'] = market[col] - market['rf']
            for model, signals in ((3, ('bm',)), (5, ('bm', 'op', 'inv'))):
                model_start = START_3 if model == 3 else START_5
                selected = market.loc[market['date'].ge(model_start)].copy()
                if selected.empty:
                    continue
                sample = returns.loc[returns['date'].ge(model_start)]
                if model == 5:
                    sample = clean_five_factor_sample(sample, industry)
                halves = []
                for second_half, tags in ((False, previous_labels), (True, labels)):
                    if model in tags and not tags[model].empty:
                        part = sample.loc[sample['date'].dt.month.ge(7).eq(second_half)]
                        halves.append(part.merge(tags[model], on='Stkcd', how='inner', validate='many_to_one'))
                if not halves:
                    raise ValueError(f'{year} 年 {model} 因子没有可用的六月分组')
                panel = pd.concat(halves, ignore_index=True)
                for number, weight in WEIGHTS.items():
                    pieces = portfolio_factors(panel, weight, signals).set_index('date').add_suffix(str(number))
                    selected = selected.merge(pieces, left_on='date', right_index=True, how='left', validate='one_to_one')
                by_model[model].append(selected)
        dec = daily.loc[daily['date'].eq(month_ends[year, 12])].copy()
        previous_labels = labels
    return {model: pd.concat(parts, ignore_index=True) for model, parts in by_model.items() if parts}


def compare_official(input_dir, results):
    names = {3: 'STK_MKT_THRFACDAY.csv', 5: 'STK_MKT_FIVEFACDAY.csv'}
    for model, self_built in results.items():
        official = pd.read_csv(input_path(input_dir, names[model]), parse_dates=['TradingDate'])
        official = official.loc[official['MarkettypeID'].eq('P9714')]
        if model == 5:
            official = official.loc[official['Portfolios'].eq(1)]
        factors = ['RiskPremium', 'SMB', 'HML', *(['RMW', 'CMA'] if model == 5 else [])]
        columns = [f'{key}{weight}' for key in factors for weight in WEIGHTS]
        joined = self_built.merge(
            official[['TradingDate', *columns]], left_on='date', right_on='TradingDate',
            suffixes=('_self', '_official'), how='left', validate='one_to_one',
        )
        rows = []
        for col in columns:
            pair = joined[[f'{col}_self', f'{col}_official']].dropna()
            reconstructed, benchmark = pair.iloc[:, 0], pair.iloc[:, 1]
            error = (reconstructed - benchmark) * 10_000
            rows.append({
                '因子': col, '交易日数': len(pair), '相关系数': reconstructed.corr(benchmark),
                '均方根误差（基点）': np.sqrt(error.pow(2).mean()),
                '平均绝对误差（基点）': error.abs().mean(),
                '平均偏差（基点）': error.mean(),
            })
        print(f'\n{model} 因子 | {joined.date.min().date()} 至 {joined.date.max().date()} | {len(joined):,} 个交易日')
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f'{value:.4f}'))


def compare_market(input_dir, self_built):
    columns = ['Trddt', 'Markettype', 'Cdretwdos', 'Cdretwdtl']
    official = pd.read_csv(input_path(input_dir, 'TRD_Cndalym.csv'), usecols=columns, parse_dates=['Trddt'])
    official = official.loc[official['Markettype'].eq(53)]
    aligned = self_built.merge(official, left_on='date', right_on='Trddt', how='left', validate='one_to_one')
    print('\n自算市场收益对照国泰安综合日市场回报率（未扣无风险利率）')
    for number, official_col in ((1, 'Cdretwdos'), (2, 'Cdretwdtl')):
        col = f'Market{number}'
        error = (aligned[col] - aligned[official_col]) * 10_000
        corr = aligned[col].corr(aligned[official_col])
        print(f'{col}  交易日数 {error.count()}  相关系数 {corr:.6f}  均方根误差 {np.sqrt(error.pow(2).mean()):.4f} 基点')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT, help='国泰安原始数据目录')
    parser.add_argument('--rate-file', type=Path, default=RATE_FILE, help='三个月定存基准利率文件')
    parser.add_argument('--start', type=pd.Timestamp, default=START_3, help='起始日期')
    parser.add_argument('--end', type=pd.Timestamp, default=pd.Timestamp('2025-12-31'), help='终止日期')
    args = parser.parse_args()
    if args.start < START_3 or args.end < args.start:
        parser.error('起始日期须在 1996-07-01 及以后，且不晚于终止日期')
    results = calculate_factors(args.input_dir, args.rate_file, args.start, args.end)
    if not results:
        parser.error('选定区间内没有可计算的交易日')
    compare_official(args.input_dir, results)
    compare_market(args.input_dir, results[3])


if __name__ == '__main__':
    main()
