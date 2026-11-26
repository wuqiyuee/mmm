"""逐单元查看沪深日度三因子和五因子的计算过程。

从上到下运行代码单元。第一格设置计算区间、数据路径和预览对象。
也可在命令行用 --preview-date 和 --preview-stock-code 指定预览对象。
"""

# %% 00. 导入库并设置日期、路径和预览对象
import argparse
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

source_file: str | None = globals().get('__file__')
running_as_script: bool = (
    __name__ == '__main__'
    and source_file is not None
    and Path(sys.argv[0]).resolve() == Path(source_file).resolve()
)
ROOT: Final[Path] = (
    Path(source_file).resolve().parents[4]
    if running_as_script and source_file is not None
    else Path('/Volumes/苹果资料/mmm复现数据包')
)
INPUT_DIR: Final[Path] = ROOT / 'input/CSMAR'
RATE_FILE: Final[Path] = (
    Path.home() / 'Desktop/课程学习/研一/研一下/FF-3/input'
    / '金融机构人民币存贷款基准利率调整表（1952-09-15至2025-05-20）'
    / 'MBK_FINRMBDEPOSITLOAND.csv'
)
START: Final[pd.Timestamp] = pd.Timestamp('2000-01-01')
END: Final[pd.Timestamp] = pd.Timestamp('2026-08-19')
MAIN_END: Final[pd.Timestamp] = pd.Timestamp('2025-12-31')
PREVIEW_ROWS: Final[int] = 5

sample_start: pd.Timestamp = START
sample_end: pd.Timestamp = END
input_dir: Path = INPUT_DIR
rate_file: Path = RATE_FILE
# 预览日期留空时使用所选区间的最后一个交易日。
preview_date: str | pd.Timestamp | None = None  # 例如 '2025-07-01'
preview_stock_code: str | int | None = None  # 例如 '000001'

SME_MERGER: Final[pd.Timestamp] = pd.Timestamp('2021-04-06')
PROFIT_CHANGE: Final[pd.Timestamp] = pd.Timestamp('2023-07-01')
MARKETS: Final[tuple[int, ...]] = (1, 4, 16, 32)  # 沪深主板、创业板和科创板
MAIN_BOARD: Final[tuple[int, ...]] = (1, 4)
CAP_COLUMNS: Final[tuple[str, ...]] = (
    'Dsmvosd', 'Dsmvtll', 'Dsmvtl2', 'Dsmvtl3',
)
WEIGHTS: Final[dict[int, str]] = {1: 'float_cap', 2: 'total_cap'}
FACTOR_NAMES: Final[dict[str, str]] = {
    'RiskPremium': '市场风险溢价',
    'SMB': '规模',
    'HML': '价值',
    'RMW': '盈利能力',
    'CMA': '投资',
}
FACTORS: Final[dict[int, tuple[str, ...]]] = {
    3: ('RiskPremium', 'SMB', 'HML'),
    5: tuple(FACTOR_NAMES),
}
INDUSTRY_DATES: Final[tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]] = (
    ('P0201', pd.Timestamp.min, pd.Timestamp('2012-10-26')),
    ('P0207', pd.Timestamp('2012-10-26'), pd.Timestamp('2023-05-01')),
    ('P0221', pd.Timestamp('2023-05-01'), pd.Timestamp.max),
)
PERIODS: Final[tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]] = (
    ('主样本', START, MAIN_END),
    ('2023 年下半年至 2025 年', PROFIT_CHANGE, MAIN_END),
    ('2026 年上半年', pd.Timestamp('2026-01-01'), pd.Timestamp('2026-06-30')),
    ('2026 年六月换组后', pd.Timestamp('2026-07-01'), END),
)

parser: argparse.ArgumentParser | None = None
if running_as_script:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-dir', type=Path, default=input_dir, help='国泰安原始数据目录'
    )
    parser.add_argument(
        '--rate-file', type=Path, default=rate_file, help='三个月定存基准利率文件'
    )
    parser.add_argument(
        '--start', type=pd.Timestamp, default=sample_start, help='计算起始日期'
    )
    parser.add_argument(
        '--end', type=pd.Timestamp, default=sample_end, help='计算终止日期'
    )
    parser.add_argument(
        '--preview-date', type=pd.Timestamp, default=preview_date,
        help='预览交易日，默认取计算区间内最后一个交易日',
    )
    parser.add_argument(
        '--preview-stock-code', default=preview_stock_code,
        help='预览股票代码，例如 000001',
    )
    arguments: argparse.Namespace = parser.parse_args()
    input_dir = arguments.input_dir
    rate_file = arguments.rate_file
    sample_start = arguments.start
    sample_end = arguments.end
    preview_date = arguments.preview_date
    preview_stock_code = arguments.preview_stock_code

configuration_error: str | None = None
if not START <= sample_start <= sample_end <= END:
    configuration_error = '日期须在 2000-01-01 至 2026-08-19 之间，且起始日期不能晚于终止日期'
elif not input_dir.is_dir():
    configuration_error = f'找不到国泰安原始数据目录 {input_dir}'
elif not rate_file.is_file():
    configuration_error = f'找不到基准利率表 {rate_file}，请指定利率文件路径'
if configuration_error is not None:
    if parser is not None:
        parser.error(configuration_error)
    raise ValueError(configuration_error)

trace_stock_code: int | None = None
if preview_stock_code is not None:
    stock_code_text: str = str(preview_stock_code).strip()
    if re.fullmatch(r'[0-9]{1,6}', stock_code_text) is None:
        stock_code_error: str = '预览股票代码须为一至六位数字，例如 000001'
        if parser is not None:
            parser.error(stock_code_error)
        raise ValueError(stock_code_error)
    trace_stock_code = int(stock_code_text)


def input_path(folder: Path, filename: str) -> Path:
    """在原始数据目录的直接子目录中定位唯一一份指定文件。"""
    matching_paths: list[Path] = sorted(folder.glob(f'*/{filename}'))
    if len(matching_paths) != 1:
        raise ValueError(
            f'{folder} 应有一份 {filename}，找到 {len(matching_paths)} 份'
        )
    return matching_paths[0]


def factor_return_columns(model: int) -> list[str]:
    """按正式输出顺序返回指定模型的因子收益列名。"""
    columns: list[str] = []
    factor_name: str
    weight_method: int
    for factor_name in FACTORS[model]:
        for weight_method in WEIGHTS:
            columns.append(f'{factor_name}{weight_method}')
    return columns


def stock_rows(
    table: pd.DataFrame,
    stock_code: int | None,
    *,
    column: str = 'Stkcd',
) -> pd.DataFrame:
    """按指定股票选取记录，兼容六位字符串和数值代码。"""
    if stock_code is None:
        return table
    if column in table.columns:
        codes: pd.Series | pd.Index = table[column]
    elif table.index.name == column:
        codes = table.index
    else:
        raise KeyError(f'预览表缺少股票代码列或同名索引 {column}')
    if pd.api.types.is_numeric_dtype(codes.dtype):
        return table.loc[codes == stock_code]
    stock_code_candidates: tuple[int | str, ...] = (
        stock_code, str(stock_code), f'{stock_code:06d}',
    )
    return table.loc[codes.isin(stock_code_candidates)]


def show_stage(
    title: str,
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    keys: str,
    reasons: str,
    before_rows: pd.DataFrame | None = None,
    after_rows: pd.DataFrame | None = None,
) -> None:
    """用完整阶段表的行列数和最多五行实际记录展示处理前后。"""
    print(f'\n【{title}】')
    print(f'合并键或筛选键：{keys}')
    print(f'处理口径：{reasons}')
    side: str
    table: pd.DataFrame
    preferred_rows: pd.DataFrame | None
    with pd.option_context('display.unicode.east_asian_width', True):
        for side, table, preferred_rows in (
            ('处理前', before, before_rows),
            ('处理后', after, after_rows),
        ):
            print(f'{side}：{table.shape[0]:,} 行 × {table.shape[1]:,} 列')
            sample_rows: pd.DataFrame = (
                table if preferred_rows is None else preferred_rows
            ).head(PREVIEW_ROWS)
            if not sample_rows.empty:
                print(sample_rows.to_string(
                    index=sample_rows.index.name is not None, na_rep='—'
                ))
            if preferred_rows is not None and len(sample_rows) < 3:
                print(f'符合预览条件的记录共 {len(sample_rows)} 行。')
                additional_rows: pd.DataFrame = table.head(
                    PREVIEW_ROWS + len(sample_rows)
                )
                if (
                    table.index.name is not None
                    and table.index.name in sample_rows.columns
                    and table.index.name not in table.columns
                ):
                    additional_rows = additional_rows.reset_index()
                shared_columns: pd.Index = sample_rows.columns.intersection(
                    additional_rows.columns
                )
                if not sample_rows.empty and len(shared_columns):
                    rows_to_compare: pd.DataFrame = pd.concat([
                        sample_rows.loc[:, shared_columns],
                        additional_rows.loc[:, shared_columns],
                    ], ignore_index=True)
                    repeated_rows: pd.Series = rows_to_compare.duplicated()
                    additional_rows = additional_rows.loc[
                        ~repeated_rows.iloc[len(sample_rows):].to_numpy()
                    ]
                if len(shared_columns):
                    additional_rows = additional_rows.loc[:, shared_columns]
                additional_rows = additional_rows.head(
                    PREVIEW_ROWS - len(sample_rows)
                )
                if not additional_rows.empty:
                    print('补充样例（原表其他记录）')
                    print(additional_rows.to_string(
                        index=additional_rows.index.name is not None,
                        na_rep='—',
                    ))
            elif sample_rows.empty:
                print('此侧无可展示记录。')


# %% 01. 整理交易日历并计算日度无风险利率
calendar_path: Path = input_path(input_dir, 'CER_Cale.csv')
calendar_source: pd.DataFrame = pd.read_csv(
    calendar_path, parse_dates=['Clddt']
)
calendar_market: pd.Series = calendar_source['Markettype'].eq(1)
calendar_open: pd.Series = calendar_source['Statse'].eq('O')
trading_dates: pd.Series = calendar_source.loc[
    calendar_market & calendar_open, 'Clddt'
].drop_duplicates().sort_values()
trading_day_numbers: pd.Series = pd.Series(
    np.arange(len(trading_dates)), index=pd.DatetimeIndex(trading_dates)
)
month_ends: pd.Series = trading_dates.groupby(
    [trading_dates.dt.year, trading_dates.dt.month]
).max()
requested_trading_dates: pd.Series = trading_dates.loc[
    trading_dates.between(sample_start, sample_end)
]
if requested_trading_dates.empty:
    raise ValueError('所选区间没有交易日')
trace_date: pd.Timestamp = pd.Timestamp(
    requested_trading_dates.iloc[-1] if preview_date is None else preview_date
)
if trace_date not in pd.DatetimeIndex(requested_trading_dates):
    preview_date_error: str = '预览日期须为所选计算区间内的交易日'
    if parser is not None:
        parser.error(preview_date_error)
    raise ValueError(preview_date_error)

preview_trading_dates: pd.Series = requested_trading_dates.loc[
    requested_trading_dates.le(trace_date)
].tail(PREVIEW_ROWS)
calendar_window: pd.Series = calendar_source['Clddt'].between(
    trace_date - pd.Timedelta(days=PREVIEW_ROWS - 1), trace_date
)
calendar_window_rows: pd.DataFrame = calendar_source.loc[
    calendar_window & calendar_market
].sort_values('Clddt').tail(PREVIEW_ROWS - 1)
calendar_other_market: pd.DataFrame = calendar_source.loc[
    ~calendar_market & calendar_source['Clddt'].eq(trace_date)
].head(1)
requested_table: pd.DataFrame = requested_trading_dates.to_frame('date')
print(f'预览交易日：{trace_date.date()}')
if trace_stock_code is not None:
    print(f'追踪股票：{trace_stock_code:06d}')
show_stage(
    f'交易日历 {calendar_path.name}',
    before=calendar_source,
    after=requested_table,
    keys='日历日期 Clddt、市场类型 Markettype。输出键为交易日 date。',
    reasons='选市场代码 1 的开市行，按日期去重和排序，再保留计算区间内的交易日。',
    before_rows=pd.concat([calendar_window_rows, calendar_other_market]),
    after_rows=preview_trading_dates.to_frame('date'),
)
del (
    calendar_source, calendar_market, calendar_open, calendar_window,
    calendar_window_rows, calendar_other_market,
)

rate_source: pd.DataFrame = pd.read_csv(
    rate_file, parse_dates=['ChangeDate']
).sort_values('ChangeDate')
eligible_rates: pd.DataFrame = rate_source.loc[
    rate_source['ChangeDate'].le('2015-10-24'),
    ['ChangeDate', 'LumpFixed3Month'],
]
matched_rates: pd.DataFrame = pd.merge_asof(
    requested_table, eligible_rates, left_on='date', right_on='ChangeDate'
).set_index('date')
risk_free_rates: pd.Series = np.expm1(
    np.log1p(matched_rates['LumpFixed3Month'] / 100) / 365
).rename('rf')
if risk_free_rates.isna().any():
    raise ValueError('部分交易日缺少三个月定存基准利率')
daily_rates: pd.DataFrame = matched_rates.assign(rf=risk_free_rates)
latest_rate_date: pd.Timestamp = matched_rates.at[trace_date, 'ChangeDate']
show_stage(
    f'三个月定存利率 {rate_file.name}',
    before=rate_source,
    after=daily_rates,
    keys='交易日 date 向前匹配最近的利率生效日 ChangeDate（含当日）。',
    reasons='沿用 2015-10-24 利率截点，按全年 365 天复利折算。后续使用日利率 rf。',
    before_rows=rate_source.loc[
        rate_source['ChangeDate'].le(latest_rate_date)
    ].tail(PREVIEW_ROWS),
    after_rows=daily_rates.loc[daily_rates.index.isin(preview_trading_dates)],
)
del rate_source, eligible_rates, matched_rates, daily_rates


# %% 02. 读取国泰安发布的因子对照表
official: dict[int, pd.DataFrame] = {}
official_model: int
official_filename: str
for official_model, official_filename in (
    (3, 'STK_MKT_THRFACDAY.csv'),
    (5, 'STK_MKT_FIVEFACDAY.csv'),
):
    official_source: pd.DataFrame = pd.read_csv(
        input_path(input_dir, official_filename), parse_dates=['TradingDate']
    )
    official_market: pd.Series = official_source['MarkettypeID'].eq('P9714')
    official_selection: pd.Series = official_market.copy()
    official_reason: str = '保留沪深指定市场 P9714 的日度因子。'
    if official_model == 5:
        official_selection &= official_source['Portfolios'].eq(1)
        official_reason += '五因子保留组合方式代码 1。'
    official_columns: list[str] = factor_return_columns(official_model)
    official_table: pd.DataFrame = official_source.loc[
        official_selection, ['TradingDate', *official_columns]
    ].rename(columns={'TradingDate': 'date'})
    official[official_model] = official_table
    official_date_window: pd.Series = official_source['TradingDate'].isin(
        preview_trading_dates
    )
    official_before_rows: pd.DataFrame = official_source.loc[
        official_market & official_date_window
    ].sort_values('TradingDate', ascending=False, kind='stable')
    official_after_rows: pd.DataFrame = official_table.loc[
        official_table['date'].isin(preview_trading_dates)
    ]
    show_stage(
        f'国泰安{official_model}因子对照表 {official_filename}',
        before=official_source,
        after=official_table,
        keys='交易日 TradingDate、市场 MarkettypeID。五因子另取组合方式 Portfolios。',
        reasons=official_reason,
        before_rows=official_before_rows,
        after_rows=official_after_rows,
    )
del (
    official_source, official_before_rows, official_after_rows,
    official_market, official_selection, official_date_window,
)


# %% 03. 合并年报并计算账面权益、营业利润和投资增长
balance_items: dict[str, str] = {
    'A003000000': 'equity',
    'A002208000': 'tax_liability',
    'A001222000': 'tax_asset',
    'A003112101': 'preferred_stock',
    'A001000000': 'assets',
}
income_items: dict[str, str] = {
    'B001101000': 'revenue',
    'B001201000': 'cost',
    'B001209000': 'selling_expense',
    'B001210000': 'admin_expense',
}
financial_keys: list[str] = ['Stkcd', 'year']
annual_tables: list[pd.DataFrame] = []
trace_fiscal_year: int = trace_date.year - (2 if trace_date.month < 7 else 1)
preview_first_fiscal_year: int = trace_fiscal_year - PREVIEW_ROWS + 1
fiscal_window_start: str = f'{preview_first_fiscal_year}-01-01'
fiscal_year_end: str = f'{trace_fiscal_year}-12-31'
report_filename: str
report_columns: dict[str, str]
report_type_column: str
report_type_value: str | int
for report_filename, report_columns, report_type_column, report_type_value in (
    ('FS_Combas.csv', balance_items, 'Typrep', 'A'),
    ('FS_Comins.csv', income_items, 'Typrep', 'A'),
    ('IAR_Rept.csv', {'Annodt': 'announced'}, 'Reptyp', 4),
):
    report_source: pd.DataFrame = pd.read_csv(
        input_path(input_dir, report_filename),
        usecols=['Stkcd', 'Accper', report_type_column, *report_columns],
        dtype={'Stkcd': 'int32', 'Accper': str},
    )
    annual_selection: pd.Series = (
        report_source['Accper'].str.fullmatch(r'\d{4}-12-31', na=False)
        & report_source[report_type_column].eq(report_type_value)
    )
    annual_report: pd.DataFrame = report_source.loc[annual_selection].copy()
    annual_report['year'] = annual_report['Accper'].str[:4].astype(int)
    annual_report = annual_report.rename(columns=report_columns)[
        [*financial_keys, *report_columns.values()]
    ]
    annual_tables.append(annual_report)
    report_before_rows: pd.DataFrame = stock_rows(
        report_source, trace_stock_code
    )
    report_after_rows: pd.DataFrame = stock_rows(
        annual_report, trace_stock_code
    )
    report_before_rows = report_before_rows.loc[
        report_before_rows['Accper'].between(
            fiscal_window_start, fiscal_year_end
        )
    ].sort_values('Accper', ascending=False, kind='stable').head(PREVIEW_ROWS)
    report_before_rows = report_before_rows.assign(
        保留=annual_selection.loc[report_before_rows.index]
    )
    show_stage(
        f'财务源表 {report_filename}',
        before=report_source,
        after=annual_report,
        keys=(
            f'股票代码 Stkcd、报表期 Accper、报表类型 {report_type_column}。'
            '年报合并键为 Stkcd、year。'
        ),
        reasons=(
            f'保留 12 月 31 日年报，且 {report_type_column} 为 '
            f'{report_type_value} 的记录。'
        ),
        before_rows=report_before_rows,
        after_rows=report_after_rows.loc[
            report_after_rows['year'].between(
                preview_first_fiscal_year, trace_fiscal_year
            )
        ].sort_values('year', ascending=False, kind='stable'),
    )
del report_source, report_before_rows, annual_selection, report_after_rows

annual_balance: pd.DataFrame
annual_income: pd.DataFrame
annual_announcements: pd.DataFrame
annual_balance, annual_income, annual_announcements = annual_tables
annual_income['has_income_statement'] = True
annual_announcements['announced'] = pd.to_datetime(
    annual_announcements['announced'], format='%Y-%m-%d', errors='coerce'
)
previous_assets: pd.DataFrame = annual_balance[
    [*financial_keys, 'assets']
].rename(columns={'assets': 'previous_assets'}).copy()
previous_assets['year'] += 1
merged_financials: pd.DataFrame = annual_balance.merge(
    previous_assets, on=financial_keys, how='left', validate='one_to_one'
).merge(
    annual_income, on=financial_keys, how='left', validate='one_to_one'
).merge(
    annual_announcements, on=financial_keys, how='left', validate='one_to_one'
)
financial_input_column_count: int = merged_financials.shape[1]
balance_preview: pd.DataFrame = stock_rows(annual_balance, trace_stock_code)
merged_preview: pd.DataFrame = stock_rows(merged_financials, trace_stock_code)
show_stage(
    '年度财务项合并',
    before=annual_balance,
    after=merged_financials,
    keys='股票代码 Stkcd、财年 year，一对一左连接。',
    reasons='以前移匹配的上一财年资产、当年利润表和公告日补充资产负债表。',
    before_rows=balance_preview.loc[
        balance_preview['year'].between(
            preview_first_fiscal_year, trace_fiscal_year
        )
    ].sort_values('year', ascending=False, kind='stable'),
    after_rows=merged_preview.loc[
        merged_preview['year'].between(
            preview_first_fiscal_year, trace_fiscal_year
        )
    ].sort_values('year', ascending=False, kind='stable'),
)

merged_financials['book'] = (
    merged_financials['equity'] + merged_financials['tax_liability'].fillna(0)
    - merged_financials['tax_asset'].fillna(0)
    - merged_financials['preferred_stock'].fillna(0)
)
# 复现口径中营收和成本必须有值，两项费用缺失按零处理。
merged_financials['operating_profit'] = (
    merged_financials['revenue'] - merged_financials['cost']
    - merged_financials['selling_expense'].fillna(0)
    - merged_financials['admin_expense'].fillna(0)
)
merged_financials['blank_financial_profit'] = (
    merged_financials['has_income_statement'].eq(True)
    & merged_financials[list(income_items.values())].isna().all(axis=1)
    & merged_financials['book'].gt(0)
)
merged_financials['inv'] = (
    merged_financials['assets'].where(merged_financials['assets'].gt(0))
    .div(merged_financials['previous_assets'].where(
        merged_financials['previous_assets'].gt(0)
    )) - 1
)
financials: pd.DataFrame = merged_financials[[
    'Stkcd', 'year', 'announced', 'book', 'operating_profit',
    'blank_financial_profit', 'inv',
]]
derived_financial_preview: pd.DataFrame = stock_rows(
    financials, trace_stock_code
)
show_stage(
    '账面权益、营业利润和资产增长构造',
    before=merged_financials.iloc[:, :financial_input_column_count],
    after=financials,
    keys='股票代码 Stkcd、财年 year。逐公司和财年计算。',
    reasons=(
        '按权益及递延税、优先股科目计算账面权益。营收和成本缺失时利润缺失，'
        '费用缺失按零处理。相邻两年资产均为正时计算增长率。'
    ),
    before_rows=merged_preview.loc[
        merged_preview['year'].between(
            preview_first_fiscal_year, trace_fiscal_year
        )
    ].sort_values('year', ascending=False, kind='stable').iloc[
        :, :financial_input_column_count
    ],
    after_rows=derived_financial_preview.loc[
        derived_financial_preview['year'].between(
            preview_first_fiscal_year, trace_fiscal_year
        )
    ].sort_values('year', ascending=False, kind='stable'),
)
del (
    merged_financials, annual_balance, annual_income, annual_announcements,
    annual_tables, annual_report, previous_assets, balance_preview,
    merged_preview, derived_financial_preview,
)


# %% 04. 整理股本变动、上市信息和行业分类
share_paths: list[Path] = sorted(
    (input_dir / '专题研究系列-事件研究').glob('*/ER_Structure.csv')
)
if len(share_paths) != 1:
    raise ValueError(f'股本变动事件表应有一份，实际找到 {len(share_paths)} 份')
share_source: pd.DataFrame = pd.read_csv(
    share_paths[0],
    usecols=[
        'Symbol', 'ChangeDate', 'DeclareDate', 'TradableSharesB',
        'TradableSharesH', 'TradeSharesTotal',
    ],
    dtype={'Symbol': 'int32'},
)
shares: pd.DataFrame = share_source.rename(
    columns={'Symbol': 'Stkcd', 'ChangeDate': 'date'}
)
share_date_column: str
for share_date_column in ('date', 'DeclareDate'):
    shares[share_date_column] = pd.to_datetime(
        shares[share_date_column], format='%Y-%m-%d', errors='coerce'
    )
shares['has_b_shares'] = shares['TradableSharesB'].gt(0)
shares['cross_listed'] = (
    shares['has_b_shares'] | shares['TradableSharesH'].gt(0)
)
shares = shares.dropna(subset=['date']).sort_values('date', kind='stable')
share_before_rows: pd.DataFrame = stock_rows(
    share_source, trace_stock_code, column='Symbol'
)
share_after_rows: pd.DataFrame = stock_rows(shares, trace_stock_code)
show_stage(
    f'股本变动事件 {share_paths[0].name}',
    before=share_source,
    after=shares,
    keys='证券代码 Symbol 和变动日期 ChangeDate，映射为 Stkcd 和 date。',
    reasons='排除无有效变动日期的记录，并标记是否有 B 股或跨市场上市。',
    before_rows=share_before_rows.loc[
        share_before_rows['ChangeDate'].le(trace_date.date().isoformat())
    ].sort_values('ChangeDate', ascending=False, kind='stable'),
    after_rows=share_after_rows.loc[
        share_after_rows['date'].le(trace_date)
    ].sort_values('date', ascending=False, kind='stable'),
)
del share_source, share_before_rows, share_after_rows

industry_source: pd.DataFrame = pd.read_csv(
    input_path(input_dir, 'STK_INDUSTRYCLASS.csv'),
    usecols=[
        'Symbol', 'IndustryClassificationID', 'ImplementDate', 'IndustryCode',
    ],
    dtype=str,
)
classified_industry: pd.DataFrame = industry_source.rename(columns={
    'Symbol': 'Stkcd',
    'IndustryClassificationID': 'standard',
    'ImplementDate': 'date',
})
classified_industry['date'] = pd.to_datetime(
    classified_industry['date'], errors='coerce'
)
classified_industry = classified_industry.dropna(
    subset=['Stkcd', 'date', 'IndustryCode']
).copy()
classified_industry['Stkcd'] = classified_industry['Stkcd'].astype('int32')
financial_industry_prefix: pd.Series = classified_industry['standard'].map(
    {'P0201': 'I', 'P0207': 'J', 'P0221': 'J'}
)
classified_industry['finance'] = classified_industry['IndustryCode'].str[0].eq(
    financial_industry_prefix
)
industry: dict[str, pd.DataFrame] = {}
industry_standard: str
industry_group: pd.DataFrame
for industry_standard, industry_group in classified_industry.groupby(
    'standard'
):
    industry[industry_standard] = industry_group[
        ['date', 'Stkcd', 'finance']
    ].sort_values('date').reset_index(drop=True)
industry_output: pd.DataFrame = (
    pd.concat(industry, names=['standard'])
    .reset_index(level='standard')
    .reset_index(drop=True)
    if industry
    else pd.DataFrame(columns=['standard', 'date', 'Stkcd', 'finance'])
)
industry_before_rows: pd.DataFrame = stock_rows(
    industry_source, trace_stock_code, column='Symbol'
)
industry_after_rows: pd.DataFrame = stock_rows(
    industry_output, trace_stock_code
)
show_stage(
    '上市公司行业分类 STK_INDUSTRYCLASS.csv',
    before=industry_source,
    after=industry_output,
    keys='公司代码、行业分类标准和行业记录生效日期。',
    reasons='排除缺失代码、生效日、行业代码或分类标准的记录，按各标准标记金融股。',
    before_rows=industry_before_rows.loc[
        industry_before_rows['ImplementDate'].le(trace_date.date().isoformat())
    ].sort_values('ImplementDate', ascending=False, kind='stable'),
    after_rows=industry_after_rows.loc[
        industry_after_rows['date'].le(trace_date)
    ].sort_values('date', ascending=False, kind='stable'),
)
del (
    industry_source, classified_industry, industry_output,
    industry_before_rows, industry_after_rows, financial_industry_prefix,
)

company_source: pd.DataFrame = pd.read_csv(
    input_path(input_dir, 'TRD_Co.csv'),
    usecols=['Stkcd', 'Listdt', 'Ipoprc'],
    dtype={'Stkcd': 'int32'},
    parse_dates=['Listdt'],
)
companies: pd.DataFrame = company_source.drop_duplicates(
    'Stkcd'
).set_index('Stkcd')
show_stage(
    '公司上市资料 TRD_Co.csv',
    before=company_source,
    after=companies,
    keys='股票代码 Stkcd，去重后用作索引。',
    reasons='同一代码保留第一条，保留上市日和发行价供上市首日收益率计算。',
    before_rows=stock_rows(company_source, trace_stock_code),
    after_rows=stock_rows(companies, trace_stock_code),
)
del company_source


# %% 05. 行情清理与六月换组
def stock_first_rows(
    table: pd.DataFrame, stock_code: int | None
) -> pd.DataFrame:
    """优先展示指定股票，再用同一张表的其他实际记录补齐。"""
    focus_rows: pd.DataFrame = stock_rows(table, stock_code).head(PREVIEW_ROWS)
    if stock_code is None or len(focus_rows) >= PREVIEW_ROWS:
        return focus_rows
    candidate_rows: pd.DataFrame = table.head(PREVIEW_ROWS + len(focus_rows))
    context_rows: pd.DataFrame = candidate_rows.loc[
        ~candidate_rows.index.isin(focus_rows.index)
    ].head(PREVIEW_ROWS - len(focus_rows))
    return pd.concat([focus_rows, context_rows])


def nearby_date_rows(table: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """从日度表取指定日及相邻交易日，最多五行。"""
    sorted_table: pd.DataFrame = table.sort_values('date', kind='stable')
    date_position: int = int(sorted_table['date'].searchsorted(date))
    centered_start: int = date_position - PREVIEW_ROWS // 2
    last_start: int = len(sorted_table) - PREVIEW_ROWS
    first_position: int = max(0, min(centered_start, last_start))
    return sorted_table.iloc[first_position:first_position + PREVIEW_ROWS]


def mark_financials(
    observations: pd.DataFrame, industry_by_standard: dict[str, pd.DataFrame]
) -> pd.Series:
    """按日期适用的行业标准识别金融股，未匹配的记录保留。"""
    finance_matches: list[pd.Series] = []
    industry_standard: str
    valid_from: pd.Timestamp
    valid_until: pd.Timestamp
    for industry_standard, valid_from, valid_until in INDUSTRY_DATES:
        period_rows: pd.DataFrame = observations.loc[
            observations['date'].ge(valid_from)
            & observations['date'].lt(valid_until),
            ['Stkcd', 'date'],
        ]
        if not period_rows.empty:
            period_rows = period_rows.rename_axis('row').reset_index()
            period_rows = period_rows.sort_values('date')
            matched_industry: pd.DataFrame = pd.merge_asof(
                period_rows,
                industry_by_standard[industry_standard],
                on='date',
                by='Stkcd',
            )
            finance_flags: pd.Series = matched_industry.set_index('row')[
                'finance'
            ]
            finance_matches.append(finance_flags)
    if not finance_matches:
        return pd.Series(False, index=observations.index)
    return pd.concat(finance_matches).reindex(observations.index).eq(True)


def read_daily_years(
    source_directory: Path,
    last_year: int,
    trading_days: pd.DatetimeIndex,
    share_structure: pd.DataFrame,
    listed_companies: pd.DataFrame,
    preview_day: pd.Timestamp,
    stock_code: int | None,
) -> Iterator[tuple[int, pd.DataFrame]]:
    """按下载分卷读取年度行情，保留停牌填充供权重和年末回填。"""
    quotes_directory: Path = (
        source_directory / '个股日度收益率（升级测试）1995-01-01至2026'
    )
    quote_columns: list[str] = [
        'Stkcd', 'Trddt', 'Markettype', 'Filling', 'Dretwd', 'Clsprc',
        *CAP_COLUMNS,
    ]
    quote_dtypes: dict[str, str] = {
        'Stkcd': 'int32', 'Markettype': 'int16', 'Filling': 'int8',
    }
    share_flags: pd.DataFrame = share_structure[
        ['Stkcd', 'date', 'has_b_shares', 'cross_listed']
    ]
    preview_year: int = preview_day.year
    preview_source_date: str = preview_day.strftime('%Y-%m-%d')

    download_folder: Path
    for download_folder in sorted(quotes_directory.iterdir()):
        folder_years: list[str] = re.findall(r'\d{4}', download_folder.name)
        if (
            not folder_years
            or int(folder_years[0]) > last_year
            or int(folder_years[-1]) < 1998
        ):
            continue
        preview_in_folder: bool = (
            int(folder_years[0]) <= preview_year <= int(folder_years[-1])
        )
        quote_parts: list[pd.DataFrame] = []
        source_day_parts: list[pd.DataFrame] = []
        quote_file: Path
        for quote_file in sorted(download_folder.glob('TRDNEW_Dalyr*.csv')):
            quote_chunk: pd.DataFrame
            for quote_chunk in pd.read_csv(
                quote_file,
                usecols=quote_columns,
                dtype=quote_dtypes,
                chunksize=400_000,
            ):
                quote_years: pd.Series = (
                    quote_chunk['Trddt'].str[:4].astype('int16')
                )
                keep_quotes: pd.Series = (
                    quote_years.between(1998, last_year)
                    & quote_chunk['Markettype'].isin(MARKETS)
                )
                if preview_in_folder:
                    source_day: pd.DataFrame = quote_chunk.loc[
                        quote_chunk['Trddt'].eq(preview_source_date)
                    ]
                    if not source_day.empty:
                        source_day_parts.append(source_day)
                if keep_quotes.any():
                    quote_parts.append(
                        quote_chunk.loc[keep_quotes].assign(
                            year=quote_years[keep_quotes]
                        )
                    )
        if not quote_parts:
            continue
        folder_quotes: pd.DataFrame = pd.concat(quote_parts, ignore_index=True)
        del quote_parts

        quote_year: int
        year_quotes: pd.DataFrame
        for quote_year, year_quotes in folder_quotes.groupby(
            'year', sort=True
        ):
            is_preview_year: bool = quote_year == preview_year
            if is_preview_year:
                original_quotes: pd.DataFrame = (
                    pd.concat(source_day_parts, ignore_index=True)
                    if source_day_parts
                    else pd.DataFrame(columns=quote_columns)
                )
                eligible_day_quotes: pd.DataFrame = year_quotes.loc[
                    year_quotes['Trddt'].eq(preview_source_date), quote_columns
                ]
                show_stage(
                    f'个股分卷读取和市场筛选（{preview_day.date()} 截面）',
                    before=original_quotes,
                    after=eligible_day_quotes,
                    keys='筛选字段 Trddt、Markettype；行情键 Stkcd、Trddt',
                    reasons=(
                        f'保留 1998 至 {last_year} 年及市场代码 {MARKETS}。'
                        '此处统计指定日已读取的实际记录。'
                    ),
                    before_rows=stock_first_rows(original_quotes, stock_code),
                    after_rows=stock_first_rows(
                        eligible_day_quotes, stock_code
                    ),
                )
                if stock_code is not None:
                    if stock_rows(original_quotes, stock_code).empty:
                        print(f'  指定股票 {stock_code:06d} 无当天原始行情。')
                    elif stock_rows(eligible_day_quotes, stock_code).empty:
                        print(f'  指定股票 {stock_code:06d} 当天未通过市场筛选。')

            calendar_quotes: pd.DataFrame = year_quotes.rename(
                columns={'Trddt': 'date'}
            ).drop(columns='year')
            calendar_quotes['date'] = pd.to_datetime(
                calendar_quotes['date'], format='%Y-%m-%d'
            )
            open_day_quotes: pd.DataFrame = (
                calendar_quotes.loc[calendar_quotes['date'].isin(trading_days)]
                .sort_values('date')
                .reset_index(drop=True)
            )
            del calendar_quotes
            if open_day_quotes.duplicated(['Stkcd', 'date']).any():
                raise ValueError(f'{quote_year} 年个股行情存在重复的股票和交易日')

            daily_quotes: pd.DataFrame = pd.merge_asof(
                open_day_quotes, share_flags, on='date', by='Stkcd'
            )
            del open_day_quotes
            # 有 B 股时用分别计价的总市值，缺失时用原总市值。
            daily_quotes['daily_total_cap'] = daily_quotes['Dsmvtl3'].where(
                daily_quotes['has_b_shares'].eq(True), daily_quotes['Dsmvtl2']
            )
            daily_quotes['daily_total_cap'] = (
                daily_quotes['daily_total_cap'].fillna(daily_quotes['Dsmvtl2'])
            )
            daily_quotes['daily_float_cap'] = daily_quotes['Dsmvosd'] + (
                daily_quotes['daily_total_cap'] - daily_quotes['Dsmvtll']
            ).clip(lower=0)
            issue_price: pd.Series = daily_quotes['Stkcd'].map(
                listed_companies['Ipoprc']
            )
            listed_today: pd.Series = daily_quotes['date'].eq(
                daily_quotes['Stkcd'].map(listed_companies['Listdt'])
            ) & issue_price.gt(0)
            daily_quotes['offer_return'] = daily_quotes['Dretwd'].where(
                ~listed_today, daily_quotes['Clsprc'].div(issue_price) - 1
            )
            if is_preview_year:
                cleaned_day_quotes: pd.DataFrame = daily_quotes.loc[
                    daily_quotes['date'].eq(preview_day)
                ]
                show_stage(
                    f'{quote_year} 年行情匹配交易日、股本和上市资料',
                    before=year_quotes,
                    after=daily_quotes,
                    keys=(
                        '交易日历按 date；股本按 Stkcd 向过去匹配最近 date；'
                        '上市资料按 Stkcd'
                    ),
                    reasons=(
                        '排除非交易日并检查股票日唯一性，保留停牌填充；'
                        '按匹配的股本口径调整市值，上市首日有有效招股价时重算收益率。'
                    ),
                    before_rows=stock_first_rows(
                        eligible_day_quotes, stock_code
                    ),
                    after_rows=stock_first_rows(
                        cleaned_day_quotes, stock_code
                    )[
                        [
                            'Stkcd', 'date', 'Markettype', 'Filling', 'Dretwd',
                            'has_b_shares', 'daily_total_cap',
                            'daily_float_cap',
                            'offer_return',
                        ]
                    ],
                )
            yield int(quote_year), daily_quotes


def last_actual_trades(
    prior_trades: pd.DataFrame | None,
    daily_quotes: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    *,
    show_preview: bool,
    stock_code: int | None,
) -> pd.DataFrame:
    """为每只股票保留截至给定日期最近的实际成交。"""
    trade_columns: list[str] = ['Stkcd', 'date', 'daily_float_cap']
    eligible_trades: pd.DataFrame = daily_quotes.loc[
        daily_quotes['Filling'].eq(0) & daily_quotes['date'].le(cutoff_date),
        trade_columns,
    ]
    if prior_trades is not None:
        eligible_trades = pd.concat(
            [prior_trades.reset_index(), eligible_trades], ignore_index=True
        )
    latest_trades: pd.DataFrame = (
        eligible_trades.sort_values('date')
        .drop_duplicates('Stkcd', keep='last')
        .set_index('Stkcd')
    )
    if show_preview:
        show_stage(
            f'截至 {cutoff_date.date()} 的最近实际成交',
            before=eligible_trades,
            after=latest_trades,
            keys='按 Stkcd 归组，date 决定最近一笔',
            reasons=(
                '合入以前年度保留的最近成交；本年只选截止日及之前的实际成交，'
                '每只股票取最后一行，供年末停牌市值回填。'
            ),
            before_rows=stock_first_rows(eligible_trades, stock_code),
            after_rows=stock_first_rows(
                latest_trades, stock_code
            ).reset_index(),
        )
    return latest_trades


def form_portfolios(
    june_quotes: pd.DataFrame,
    december_quotes: pd.DataFrame,
    annual_reports: pd.DataFrame,
    share_structure: pd.DataFrame,
    industry_by_standard: dict[str, pd.DataFrame],
    trading_day_numbers: pd.Series,
    june_end: pd.Timestamp,
    december_end: pd.Timestamp,
    latest_december_trades: pd.DataFrame,
    *,
    show_preview: bool,
    stock_code: int | None,
) -> pd.DataFrame:
    """计算六月规模组、年末特征组及两套固定六月权重。"""
    june_members: pd.DataFrame = (
        june_quotes.loc[june_quotes['daily_float_cap'].gt(0)]
        .set_index('Stkcd')
        .copy()
    )
    december_features: pd.DataFrame = december_quotes.set_index('Stkcd').copy()
    june_members['float_cap'] = june_members['daily_float_cap']
    june_members['total_cap'] = june_members['daily_total_cap']

    december_features['value_cap3'] = december_features['daily_float_cap']
    year_end_date: pd.Timestamp = pd.Timestamp(june_end.year - 1, 12, 31)
    year_end_shares: pd.DataFrame = (
        share_structure.loc[
            share_structure['date'].eq(year_end_date)
            & share_structure['DeclareDate'].le(june_end)
        ]
        .set_index('Stkcd')
        .reindex(december_features.index)
    )
    cross_listed: pd.Series = december_features['cross_listed'].eq(
        True
    ) | year_end_shares['cross_listed'].eq(True)
    tradable_shares: pd.Series = year_end_shares['TradeSharesTotal']
    exact_float_cap: pd.Series = (
        tradable_shares.where(cross_listed & tradable_shares.gt(0))
        * december_features['Clsprc']
        / 1000
    )
    december_features['value_cap5'] = exact_float_cap.fillna(
        december_features['daily_float_cap']
    )

    actual_trades: pd.DataFrame = latest_december_trades.reindex(
        december_features.index
    )
    trading_day_gap: pd.Series = trading_day_numbers.at[
        december_end
    ] - actual_trades['date'].map(trading_day_numbers)
    use_actual_trade: pd.Series = (
        december_features['Filling'].eq(1)
        & trading_day_gap.ge(0)
        & actual_trades['daily_float_cap'].notna()
    )
    cap_column: str
    for cap_column in ('value_cap3', 'value_cap5'):
        december_features[cap_column] = december_features[cap_column].mask(
            use_actual_trade, actual_trades['daily_float_cap']
        )

    june_main_board: pd.Series = june_members['Markettype'].isin(MAIN_BOARD)
    december_main_board: pd.Series = december_features['Markettype'].isin(
        MAIN_BOARD
    )
    if june_end < SME_MERGER:
        # 中小板股票只退出断点样本，满足其余条件时仍能入组。
        june_main_board &= ~june_members.index.to_series().between(2000, 3999)
        december_main_board &= ~december_features.index.to_series().between(
            2000, 3999
        )
    size_breakpoint: float = june_members.loc[
        june_main_board, 'float_cap'
    ].median()
    if pd.isna(size_breakpoint):
        raise ValueError(f'{june_end.date()} 没有可用于规模断点的主板市值')
    june_members['size_group'] = np.where(
        june_members['float_cap'].ge(size_breakpoint), 'B', 'S'
    )
    if show_preview:
        show_stage(
            f'{june_end.date()} 六月规模排序及固定权重',
            before=june_quotes,
            after=june_members,
            keys='六月截面以 Stkcd 为键；断点用主板流通市值',
            reasons=(
                f'六月流通市值须为正。主板流通市值中位数为 {size_breakpoint:g}，'
                '达到中位数归大盘，其余归小盘。'
            ),
            before_rows=stock_first_rows(june_quotes, stock_code)[
                [
                    'Stkcd', 'date', 'Markettype',
                    'daily_float_cap', 'daily_total_cap',
                ]
            ],
            after_rows=stock_first_rows(
                june_members, stock_code
            ).reset_index()[
                ['Stkcd', 'date', 'float_cap', 'total_cap', 'size_group']
            ],
        )

    # 六月断点是交易日。非交易日公告映射至下一交易日时，筛选结果相同。
    available_reports: pd.DataFrame = annual_reports.loc[
        annual_reports['announced'].le(june_end)
    ].copy()
    is_financial: pd.Series = mark_financials(
        available_reports.assign(date=december_end), industry_by_standard
    )
    before_profit_change: bool = june_end.year < PROFIT_CHANGE.year
    profit_equity: pd.Series = available_reports['book'].where(
        available_reports['book'].ne(0)
        if before_profit_change
        else available_reports['book'].gt(0)
    )
    profitability: pd.Series = available_reports['operating_profit'].div(
        profit_equity
    )
    if before_profit_change:
        available_reports['op'] = profitability.mask(is_financial)
    else:
        available_reports['op'] = profitability.mask(
            is_financial & available_reports['blank_financial_profit'], 0
        )
    if show_preview:
        show_stage(
            f'{june_end.date()} 换组可用年报与盈利能力',
            before=annual_reports,
            after=available_reports,
            keys='公告日不晚于六月换组日；行业按 Stkcd 和上年末 date 匹配',
            reasons=(
                '只取换组时已公开的上一财年年报。'
                '采用当期适用的权益条件与金融股盈利规则。'
            ),
            before_rows=stock_first_rows(annual_reports, stock_code),
            after_rows=stock_first_rows(available_reports, stock_code),
        )
    december_features = december_features.join(
        available_reports.set_index('Stkcd')[['book', 'op', 'inv']],
        validate='one_to_one',
    )

    feature_name: str
    for feature_name, cap_column in (
        ('bm3', 'value_cap3'), ('bm5', 'value_cap5')
    ):
        december_features[feature_name] = december_features['book'].where(
            december_features['book'].gt(0)
        ).div(december_features[cap_column].where(
            december_features[cap_column].gt(0)
        ) * 1000)

    if show_preview:
        show_stage(
            f'{december_end.date()} 年末市值与特征排序值',
            before=december_quotes,
            after=december_features,
            keys='按 Stkcd 匹配年末股本、最近实际成交和可用年报',
            reasons=(
                '五因子按跨市场股本资料修正流通市值；'
                '年末停牌且有有效成交市值时回填。'
                '账面权益及相应市值为正才计算账面市值比。'
            ),
            before_rows=stock_first_rows(december_quotes, stock_code)[
                [
                    'Stkcd', 'date', 'Filling', 'Clsprc',
                    'daily_float_cap', 'cross_listed',
                ]
            ],
            after_rows=stock_first_rows(
                december_features, stock_code
            ).reset_index()[
                [
                    'Stkcd', 'value_cap3', 'value_cap5', 'book',
                    'bm3', 'bm5', 'op', 'inv',
                ]
            ],
        )

    feature_columns: tuple[str, ...] = ('bm3', 'bm5', 'op', 'inv')
    for feature_name in feature_columns:
        feature_values: pd.Series = december_features[feature_name]
        lower_breakpoint: float
        upper_breakpoint: float
        lower_breakpoint, upper_breakpoint = feature_values.loc[
            december_main_board
        ].quantile([0.30, 0.70])
        december_features[feature_name] = pd.Series(
            np.select(
                [
                    feature_values.ge(upper_breakpoint),
                    feature_values.lt(lower_breakpoint),
                    feature_values.notna() & pd.notna(lower_breakpoint),
                ],
                ['H', 'L', 'M'],
                default='',
            ),
            index=feature_values.index,
        ).replace('', np.nan)
        if show_preview:
            print(
                f'  {feature_name} 的主板断点：'
                f'30% = {lower_breakpoint:g}，70% = {upper_breakpoint:g}'
            )

    june_weight_columns: list[str] = ['size_group', *WEIGHTS.values()]
    june_weights: pd.DataFrame = june_members[june_weight_columns]
    formation_labels: pd.DataFrame = june_weights.join(
        december_features[list(feature_columns)], how='inner'
    ).reset_index()
    if show_preview:
        show_stage(
            f'{june_end.date()} 六月换组最终标签（指定交易日实际采用）',
            before=june_weights,
            after=formation_labels,
            keys='六月规模、权重与上年末特征标签按 Stkcd 内连接',
            reasons=(
                '只保留两个截面都有的股票。按主板 30% 和 70% 断点分低、中、高组，'
                '单项特征缺失时相应标签为空。'
            ),
            before_rows=stock_first_rows(
                june_weights, stock_code
            ).reset_index(),
            after_rows=stock_first_rows(formation_labels, stock_code),
        )
        if (
            stock_code is not None
            and stock_rows(formation_labels, stock_code).empty
        ):
            print(f'  指定股票 {stock_code:06d} 无本轮六月换组标签。')
    return formation_labels


# %% 06. 市值加权、六组收益与因子公式
def weighted_mean_by_group(
    observations: pd.DataFrame,
    group_columns: Sequence[str],
    return_column: str,
    weight_column: str,
    result_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """计算加权收益，并返回计算实际使用的观测掩码。"""
    needed_columns: list[str] = [*group_columns, return_column, weight_column]
    weighting_rows: pd.DataFrame = observations.loc[:, needed_columns].copy()
    weighting_rows[return_column] = pd.to_numeric(
        weighting_rows[return_column], errors='coerce'
    )
    weighting_rows[weight_column] = pd.to_numeric(
        weighting_rows[weight_column], errors='coerce'
    )

    valid_weights: pd.Series = (
        weighting_rows[return_column].notna()
        & weighting_rows[weight_column].notna()
        & weighting_rows[weight_column].gt(0)
    )
    eligible_rows: pd.DataFrame = weighting_rows.loc[valid_weights].copy()
    result_columns: list[str] = [
        *group_columns, result_column, 'n_stocks', 'sum_weight'
    ]
    if eligible_rows.empty:
        return pd.DataFrame(columns=result_columns), valid_weights

    eligible_rows['weighted_return'] = (
        eligible_rows[return_column] * eligible_rows[weight_column]
    )
    grouped_returns: pd.DataFrame = (
        eligible_rows.groupby(list(group_columns), observed=True, dropna=False)
        .agg(
            sum_weighted_return=('weighted_return', 'sum'),
            sum_weight=(weight_column, 'sum'),
            n_stocks=(return_column, 'count'),
        )
        .reset_index()
    )
    grouped_returns[result_column] = (
        grouped_returns['sum_weighted_return'] / grouped_returns['sum_weight']
    )
    return grouped_returns.loc[:, result_columns], valid_weights


def make_ff_2x3_factors(
    group_returns: pd.DataFrame, feature_column: str
) -> pd.DataFrame:
    """由六组收益计算小盘减大盘和高特征组减低特征组。"""
    wide_returns: pd.DataFrame = group_returns.pivot_table(
        index='date',
        columns=['size_group', feature_column],
        values='portfolio_return',
        aggfunc='first',
    )
    required_groups: tuple[tuple[str, str], ...] = (
        ('S', 'L'), ('S', 'M'), ('S', 'H'),
        ('B', 'L'), ('B', 'M'), ('B', 'H'),
    )
    group_key: tuple[str, str]
    for group_key in required_groups:
        if group_key not in wide_returns.columns:
            wide_returns[group_key] = pd.NA

    daily_spreads: pd.DataFrame = pd.DataFrame(index=wide_returns.index)
    daily_spreads[f'size_{feature_column}'] = (
        (wide_returns[('S', 'L')] + wide_returns[('S', 'M')]
         + wide_returns[('S', 'H')]) / 3
        - (wide_returns[('B', 'L')] + wide_returns[('B', 'M')]
           + wide_returns[('B', 'H')]) / 3
    )
    daily_spreads[feature_column] = (
        (wide_returns[('S', 'H')] + wide_returns[('B', 'H')]) / 2
        - (wide_returns[('S', 'L')] + wide_returns[('B', 'L')]) / 2
    )
    return daily_spreads.rename_axis('date')


def market_returns(
    daily_quotes: pd.DataFrame,
    previous_december_quotes: pd.DataFrame,
    trading_day_numbers: pd.Series,
    risk_free_rates: pd.Series,
    preview_day: pd.Timestamp | None,
    stock_code: int | None,
) -> pd.DataFrame:
    """按前一开市日市值加权市场收益，并计算市场风险溢价。"""
    market_columns: list[str] = [
        'Stkcd', 'date', 'Filling', 'offer_return',
        'daily_float_cap', 'daily_total_cap',
    ]
    market_quotes: pd.DataFrame = pd.concat(
        [
            previous_december_quotes[market_columns],
            daily_quotes[market_columns],
        ],
        ignore_index=True,
    ).sort_values(['Stkcd', 'date'])
    stock_codes: pd.Series = market_quotes['Stkcd']
    day_numbers: pd.Series = market_quotes['date'].map(trading_day_numbers)
    previous_day_exists: pd.Series = (
        stock_codes.eq(stock_codes.shift()) & day_numbers.diff().eq(1)
    )
    cap_column: str
    weight_column: str
    for cap_column, weight_column in (
        ('daily_float_cap', 'lag_float'), ('daily_total_cap', 'lag_total')
    ):
        current_cap: pd.Series = market_quotes[cap_column]
        # 前一开市日市值可来自停牌填充，无相邻记录时使用当日市值。
        market_quotes[weight_column] = current_cap.shift().where(
            previous_day_exists, current_cap
        )

    eligible_market_quotes: pd.DataFrame = market_quotes.loc[
        market_quotes['Filling'].eq(0)
        & market_quotes['date'].isin(risk_free_rates.index)
    ]
    del market_quotes
    if preview_day is not None:
        market_day_sample: pd.DataFrame = stock_first_rows(
            eligible_market_quotes.loc[
                eligible_market_quotes['date'].eq(preview_day)
            ],
            stock_code,
        ).copy()
        market_day_sample['权重来源'] = np.where(
            previous_day_exists.reindex(market_day_sample.index),
            '上一开市日',
            '当日回退',
        )
        show_stage(
            '市场收益的前一开市日权重与交易筛选',
            before=daily_quotes,
            after=eligible_market_quotes,
            keys='Stkcd 相同且交易日序号相差 1；收益按 date 汇总',
            reasons=(
                '先接上年十二月末行情并构造权重，再保留计算区间内的实际交易。'
                '没有相邻记录用当日市值，前一开市日停牌填充市值仍可为权重。'
            ),
            before_rows=stock_first_rows(
                daily_quotes.loc[
                    daily_quotes['date'].eq(preview_day), market_columns
                ],
                stock_code,
            ),
            after_rows=market_day_sample,
        )

    daily_market: pd.DataFrame = risk_free_rates.to_frame()
    weight_method: int
    for weight_method, weight_column in ((1, 'lag_float'), (2, 'lag_total')):
        market_column: str = f'Market{weight_method}'
        grouped_market: pd.DataFrame
        valid_weights: pd.Series
        grouped_market, valid_weights = weighted_mean_by_group(
            eligible_market_quotes, ['date'], 'offer_return',
            weight_column, market_column,
        )
        daily_market[market_column] = grouped_market.set_index('date')[
            market_column
        ]
        daily_market[f'RiskPremium{weight_method}'] = (
            daily_market[market_column] - daily_market['rf']
        )
        if preview_day is not None:
            weight_sample: pd.DataFrame = stock_first_rows(
                eligible_market_quotes.loc[
                    eligible_market_quotes['date'].eq(preview_day),
                    ['Stkcd', 'date', 'offer_return', weight_column],
                ],
                stock_code,
            ).copy()
            weight_sample['参与加权'] = valid_weights.reindex(weight_sample.index)
            show_stage(
                f'市场组合收益（权重口径 {weight_method}）',
                before=eligible_market_quotes,
                after=grouped_market,
                keys='date；同一交易日股票汇总为一行',
                reasons=(
                    '排除收益率缺失、权重缺失或非正的观测，'
                    f'有效股票日观测 {int(valid_weights.sum()):,} 条。'
                    '加权市场收益减日度无风险利率得到风险溢价。'
                ),
                before_rows=weight_sample,
                after_rows=nearby_date_rows(grouped_market, preview_day),
            )
    return daily_market


def portfolio_returns(
    daily_quotes: pd.DataFrame,
    formation_labels: pd.DataFrame,
    preview_day: pd.Timestamp | None,
    stock_code: int | None,
) -> tuple[dict[int, pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    """按六月市值加权六组收益，计算因子并截取实际计算表。"""
    valid_quotes: pd.Series = (
        daily_quotes['Filling'].eq(0) & daily_quotes['Dretwd'].notna()
    )
    labeled_quotes: pd.DataFrame = daily_quotes.loc[valid_quotes].merge(
        formation_labels, on='Stkcd', how='inner', validate='many_to_one'
    )
    exclude_profitability: pd.Series = (
        labeled_quotes['date'].lt(PROFIT_CHANGE) & labeled_quotes['financial']
    )
    labeled_quotes.loc[exclude_profitability, 'op'] = np.nan

    formation_stock: pd.DataFrame = pd.DataFrame()
    observed_stock: pd.DataFrame = pd.DataFrame()
    if preview_day is not None:
        source_day_quotes: pd.DataFrame = daily_quotes.loc[
            daily_quotes['date'].eq(preview_day)
        ]
        labeled_day_quotes: pd.DataFrame = labeled_quotes.loc[
            labeled_quotes['date'].eq(preview_day)
        ]
        show_stage(
            '个股行情连接对应六月换组标签',
            before=daily_quotes,
            after=labeled_quotes,
            keys='按 Stkcd 多对一内连接；行情仍以 Stkcd、date 唯一',
            reasons=(
                '保留实际交易且原日收益率有值的行情，再保留能匹配六月标签的股票。'
                '制度切换日前的金融股不参加盈利能力组合。'
            ),
            before_rows=stock_first_rows(source_day_quotes, stock_code)[
                [
                    'Stkcd', 'date', 'Filling', 'Dretwd',
                    'offer_return', 'financial',
                ]
            ],
            after_rows=stock_first_rows(labeled_day_quotes, stock_code)[
                [
                    'Stkcd', 'date', 'size_group', 'bm3', 'bm5', 'op', 'inv',
                    'float_cap', 'total_cap', 'offer_return',
                ]
            ],
        )
        if stock_code is not None:
            formation_stock = stock_rows(formation_labels, stock_code)
            observed_stock = stock_rows(labeled_day_quotes, stock_code)
            input_stock: pd.DataFrame = stock_rows(
                source_day_quotes, stock_code
            )
            eligible_stock: pd.Series = valid_quotes.reindex(input_stock.index)
            if input_stock.empty:
                print(f'  指定股票 {stock_code:06d} 当日无行情记录。')
            elif not eligible_stock.any():
                print(
                    f'  指定股票 {stock_code:06d} 当日因填充标识或原始收益缺失'
                    '未进入组合计算。'
                )
            elif formation_stock.empty:
                print(f'  指定股票 {stock_code:06d} 无本轮六月分组标签。')

    model_parts: dict[int, list[pd.DataFrame]] = {3: [], 5: []}
    group_previews: list[pd.DataFrame] = []
    stock_group_previews: list[pd.DataFrame] = []
    if preview_day is not None:
        stock_labels_for_day: pd.DataFrame = (
            formation_stock if observed_stock.empty else observed_stock
        )
    weight_method: int
    weight_column: str
    for weight_method, weight_column in WEIGHTS.items():
        spread_parts: list[pd.DataFrame] = []
        feature_column: str
        for feature_column in ('bm3', 'bm5', 'op', 'inv'):
            feature_quotes: pd.DataFrame = labeled_quotes.loc[
                labeled_quotes[feature_column].notna()
            ]
            group_returns: pd.DataFrame
            valid_weights: pd.Series
            group_returns, valid_weights = weighted_mean_by_group(
                feature_quotes, ['date', 'size_group', feature_column],
                'offer_return', weight_column, 'portfolio_return',
            )
            daily_spreads: pd.DataFrame = make_ff_2x3_factors(
                group_returns, feature_column
            )
            spread_parts.append(daily_spreads)
            if preview_day is not None:
                day_groups: pd.DataFrame = (
                    group_returns.loc[group_returns['date'].eq(preview_day)]
                    .rename(columns={feature_column: 'feature_group'})
                    .assign(weight_method=weight_method, signal=feature_column)
                )
                group_previews.append(day_groups)
                weight_labels: pd.DataFrame = labeled_day_quotes[
                    [
                        'Stkcd', 'date', 'size_group', feature_column,
                        weight_column, 'offer_return',
                    ]
                ]
                stock_weight_sample: pd.DataFrame = stock_first_rows(
                    weight_labels, stock_code
                ).copy()
                stock_weight_sample['参与加权'] = valid_weights.reindex(
                    stock_weight_sample.index, fill_value=False
                )
                preferred_group: pd.DataFrame = day_groups.iloc[:0]
                if not stock_labels_for_day.empty:
                    effective_feature: str | float = stock_labels_for_day[
                        feature_column
                    ].iat[0]
                    if pd.notna(effective_feature):
                        preferred_group = day_groups.loc[
                            day_groups['size_group'].eq(
                                stock_labels_for_day['size_group'].iat[0]
                            )
                            & day_groups['feature_group'].eq(effective_feature)
                        ]
                    stock_group: pd.DataFrame = (
                        stock_labels_for_day[['size_group', feature_column]]
                        .rename(columns={feature_column: 'feature_group'})
                        .assign(
                            date=preview_day, weight_method=weight_method,
                            signal=feature_column,
                            formation_feature_group=formation_stock[
                                feature_column
                            ].iat[0],
                        )
                    )
                    measure: str
                    for measure in (
                        'portfolio_return', 'n_stocks', 'sum_weight'
                    ):
                        stock_group[measure] = (
                            preferred_group[measure].iat[0]
                            if not preferred_group.empty
                            else np.nan
                        )
                    stock_group['included'] = bool(valid_weights.reindex(
                        observed_stock.index, fill_value=False
                    ).any())
                    stock_group_previews.append(stock_group)

                preview_groups: pd.DataFrame = pd.concat([
                    preferred_group,
                    day_groups.loc[
                        ~day_groups.index.isin(preferred_group.index)
                    ],
                ]).rename(columns={'feature_group': feature_column})[
                    group_returns.columns
                ]
                show_stage(
                    f'{feature_column} 六组的加权收益（权重口径 {weight_method}）',
                    before=labeled_quotes,
                    after=group_returns,
                    keys=f'date、size_group、{feature_column}',
                    reasons=(
                        '使用固定六月市值，排除缺失特征、缺失收益及缺失或非正权重；'
                        '指定股票所在的有效组合优先展示。'
                    ),
                    before_rows=stock_weight_sample,
                    after_rows=preview_groups,
                )
                show_stage(
                    f'{feature_column} 六组收益差（权重口径 {weight_method}）',
                    before=group_returns,
                    after=daily_spreads,
                    keys='date；将规模组和特征组转为同日六列后比较',
                    reasons=(
                        '小盘三组均值减大盘三组均值；高特征两组均值减低特征两组均值。'
                        '所需组合收益缺失时，相应收益差保留缺失。'
                    ),
                    before_rows=preview_groups,
                    after_rows=nearby_date_rows(
                        daily_spreads.reset_index(), preview_day
                    ),
                )

        feature_spreads: pd.DataFrame = pd.concat(spread_parts, axis=1)
        three_factors: pd.DataFrame = pd.DataFrame({
            'SMB': feature_spreads['size_bm3'], 'HML': feature_spreads['bm3'],
        })
        five_factors: pd.DataFrame = pd.DataFrame({
            'SMB': feature_spreads[['size_bm5', 'size_op', 'size_inv']].mean(
                axis=1, skipna=False
            ),
            'HML': feature_spreads['bm5'],
            'RMW': feature_spreads['op'],
            'CMA': -feature_spreads['inv'],
        })
        model_parts[3].append(three_factors.add_suffix(str(weight_method)))
        model_parts[5].append(five_factors.add_suffix(str(weight_method)))
        if preview_day is not None:
            display_model_factors: pd.DataFrame = pd.concat([
                three_factors.add_suffix('_三因子'),
                five_factors.add_suffix('_五因子'),
            ], axis=1).reset_index()
            show_stage(
                f'特征收益差组合为三因子与五因子（权重口径 {weight_method}）',
                before=feature_spreads,
                after=display_model_factors,
                keys='交易日 date；三因子与五因子按各自公式计算',
                reasons=(
                    '五因子规模因子取三种排序所得规模收益差的均值，三者都须有效。'
                    '投资因子取低投资组减高投资组。'
                ),
                before_rows=nearby_date_rows(
                    feature_spreads.reset_index(), preview_day
                ),
                after_rows=nearby_date_rows(
                    display_model_factors, preview_day
                ),
            )
    model_results: dict[int, pd.DataFrame] = {}
    model_number: int
    model_tables: list[pd.DataFrame]
    for model_number, model_tables in model_parts.items():
        model_results[model_number] = pd.concat(model_tables, axis=1)
    return model_results, group_previews, stock_group_previews


# %% 07. 逐年计算并追踪指定交易日
trace_formation_year: int = trace_date.year - int(trace_date.month < 7)
if trace_stock_code is None:
    print(f'\n预览交易日：{trace_date.date()}；采用 {trace_formation_year} 年六月换组')
else:
    print(
        f'\n预览交易日：{trace_date.date()}；股票 {trace_stock_code:06d}；'
        f'采用 {trace_formation_year} 年六月换组'
    )

yearly_factors: dict[int, list[pd.DataFrame]] = {3: [], 5: []}
prior_labels: pd.DataFrame | None = None
previous_december: pd.DataFrame | None = None
last_december_trades: pd.DataFrame | None = None
covered_trade_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
group_preview_parts: list[pd.DataFrame] = []
stock_group_preview_parts: list[pd.DataFrame] = []

quote_year: int
daily_quotes: pd.DataFrame
for quote_year, daily_quotes in read_daily_years(
    input_dir, sample_end.year, trading_day_numbers.index, shares, companies,
    trace_date, trace_stock_code,
):
    december_end: pd.Timestamp = month_ends.get((quote_year, 12), pd.NaT)
    december_quotes: pd.DataFrame = daily_quotes.loc[
        daily_quotes['date'].eq(december_end)
    ]
    year_risk_free: pd.Series = risk_free_rates.loc[
        risk_free_rates.index.year == quote_year
    ]
    # 起始年前一年的六月分组供首年上半年使用。
    if (
        quote_year >= sample_start.year - 1
        and previous_december is not None
        and last_december_trades is not None
    ):
        june_end: pd.Timestamp = month_ends.get((quote_year, 6), pd.NaT)
        need_formation: bool = (
            quote_year < sample_end.year or sample_end.month >= 7
        )
        current_labels: pd.DataFrame | None = None
        if need_formation:
            june_quotes: pd.DataFrame = daily_quotes.loc[
                daily_quotes['date'].eq(june_end)
            ]
            if june_quotes.empty:
                raise ValueError(f'{quote_year} 年组合缺少六月末行情')
            annual_reports: pd.DataFrame = financials.loc[
                financials['year'].eq(quote_year - 1)
            ]
            current_labels = form_portfolios(
                june_quotes, previous_december, annual_reports, shares,
                industry, trading_day_numbers, june_end,
                month_ends.at[(quote_year - 1, 12)],
                last_december_trades,
                show_preview=(quote_year == trace_formation_year),
                stock_code=trace_stock_code,
            )

        if not year_risk_free.empty:
            print(f'计算 {quote_year} 年日度因子', flush=True)
            selected_quotes: pd.DataFrame = daily_quotes.loc[
                daily_quotes['date'].isin(year_risk_free.index)
            ].copy()
            has_trade: pd.Series = (
                selected_quotes['Filling'].eq(0)
                & selected_quotes['Dretwd'].notna()
            )
            covered_trade_dates = covered_trade_dates.union(pd.DatetimeIndex(
                selected_quotes.loc[has_trade, 'date'].unique()
            ))
            selected_quotes['financial'] = mark_financials(
                selected_quotes.loc[
                    selected_quotes['date'].lt(PROFIT_CHANGE)
                    & selected_quotes['Filling'].eq(0)
                ],
                industry,
            ).reindex(selected_quotes.index).eq(True)

            year_preview_day: pd.Timestamp | None = (
                trace_date if quote_year == trace_date.year else None
            )
            if year_preview_day is not None:
                selected_day_rows: pd.DataFrame = stock_first_rows(
                    selected_quotes.loc[
                        selected_quotes['date'].eq(trace_date)
                    ],
                    trace_stock_code,
                )[['Stkcd', 'date', 'Filling', 'Dretwd', 'financial']].copy()
                selected_day_rows['计入日期覆盖'] = has_trade.reindex(
                    selected_day_rows.index
                )
                show_stage(
                    '限定目标交易日区间并补充金融行业标记',
                    before=daily_quotes,
                    after=selected_quotes,
                    keys='date 匹配目标交易日；行业按 Stkcd、date 向过去匹配',
                    reasons=(
                        '只交给当年请求的交易日。盈利规则变更前对实际交易匹配行业。'
                        '实际成交且原收益有值的观测用于检查日期覆盖。'
                    ),
                    before_rows=stock_first_rows(
                        daily_quotes.loc[daily_quotes['date'].eq(trace_date)],
                        trace_stock_code,
                    )[['Stkcd', 'date', 'Filling', 'Dretwd']],
                    after_rows=selected_day_rows,
                )
            year_market: pd.DataFrame = market_returns(
                daily_quotes, previous_december, trading_day_numbers,
                year_risk_free, year_preview_day, trace_stock_code,
            )
            year_portfolios: dict[int, list[pd.DataFrame]] = {3: [], 5: []}
            # 当年上半年沿用上一轮标签，下半年使用当年六月标签。
            second_half: bool
            effective_labels: pd.DataFrame | None
            for second_half, effective_labels in (
                (False, prior_labels), (True, current_labels)
            ):
                half_quotes: pd.DataFrame = selected_quotes.loc[
                    selected_quotes['date'].dt.month.ge(7).eq(second_half)
                ]
                if half_quotes.empty:
                    continue
                if effective_labels is None:
                    raise ValueError(f'{quote_year} 年的日度收益缺少对应六月分组')
                half_preview_day: pd.Timestamp | None = (
                    trace_date
                    if year_preview_day is not None
                    and (trace_date.month >= 7) == second_half
                    else None
                )
                half_factors: dict[int, pd.DataFrame]
                half_group_preview: list[pd.DataFrame]
                half_stock_preview: list[pd.DataFrame]
                (
                    half_factors, half_group_preview, half_stock_preview
                ) = portfolio_returns(
                    half_quotes, effective_labels,
                    half_preview_day, trace_stock_code,
                )
                group_preview_parts.extend(half_group_preview)
                stock_group_preview_parts.extend(half_stock_preview)
                half_model: int
                half_table: pd.DataFrame
                for half_model, half_table in half_factors.items():
                    year_portfolios[half_model].append(half_table)

            model_number: int
            for model_number in FACTORS:
                year_feature_factors: pd.DataFrame = pd.concat(
                    year_portfolios[model_number]
                )
                year_factor_table: pd.DataFrame = year_market.join(
                    year_feature_factors, validate='one_to_one'
                ).rename_axis('date').reset_index()
                yearly_factors[model_number].append(year_factor_table)
                if year_preview_day is not None:
                    year_market_table: pd.DataFrame = year_market.reset_index()
                    show_stage(
                        f'{quote_year} 年市场收益与{model_number}因子组合收益合并',
                        before=year_market_table,
                        after=year_factor_table,
                        keys='date 一对一左连接市场与特征因子',
                        reasons=(
                            '保留市场侧请求交易日，特征因子对应日期或组合不足时'
                            '保留缺失。'
                        ),
                        before_rows=nearby_date_rows(
                            year_market_table, trace_date
                        ),
                        after_rows=nearby_date_rows(
                            year_factor_table, trace_date
                        ),
                    )
        prior_labels = current_labels

    # 无请求交易日的历史年份也更新十二月行情和最近实际成交。
    if not december_quotes.empty:
        previous_december = december_quotes
        last_december_trades = last_actual_trades(
            last_december_trades, daily_quotes, december_end,
            show_preview=(quote_year == trace_formation_year - 1),
            stock_code=trace_stock_code,
        )


# %% 08. 检查日期覆盖并合并逐日因子
TRACE_MODEL_NAMES: Final[dict[int, str]] = {3: '三因子', 5: '五因子'}
TRACE_WEIGHT_NAMES: Final[dict[int, str]] = {1: '流通市值', 2: '总市值'}
TRACE_FEATURE_NAMES: Final[dict[str, str]] = {
    'bm3': '三因子账面市值比',
    'bm5': '五因子账面市值比',
    'op': '盈利能力',
    'inv': '投资增长',
}

missing_dates: pd.DatetimeIndex = pd.DatetimeIndex(
    requested_trading_dates
).difference(covered_trade_dates)
if not missing_dates.empty:
    raise ValueError(
        f'原始行情缺少 {len(missing_dates)} 个交易日，'
        f'首个为 {missing_dates[0].date()}'
    )

results: dict[int, pd.DataFrame] = {}
result_model: int
model_year_tables: list[pd.DataFrame]
for result_model, model_year_tables in yearly_factors.items():
    combined_factors: pd.DataFrame = pd.concat(
        model_year_tables, ignore_index=True
    )
    results[result_model] = combined_factors
    trace_year_factors: pd.DataFrame = model_year_tables[0]
    annual_factors: pd.DataFrame
    for annual_factors in model_year_tables:
        if annual_factors['date'].eq(trace_date).any():
            trace_year_factors = annual_factors
            break

    formal_columns: list[str] = [
        'date', 'rf', *factor_return_columns(result_model)
    ]
    result_model_name: str = TRACE_MODEL_NAMES[result_model]
    show_stage(
        f'{result_model_name}分年拼接',
        before=trace_year_factors,
        after=combined_factors,
        keys='交易日 date。分年表沿行拼接，逐年核对交易日覆盖。',
        reasons=(
            f'处理前展示追踪日所在的分年表。汇总 {len(model_year_tables)} 张'
            f'分年表，样例显示正式口径的 {len(formal_columns)} 列。'
        ),
        before_rows=nearby_date_rows(
            trace_year_factors.loc[:, formal_columns], trace_date
        ),
        after_rows=nearby_date_rows(
            combined_factors.loc[:, formal_columns], trace_date
        ),
    )
    exact_factors: pd.DataFrame = combined_factors.loc[
        combined_factors['date'].eq(trace_date), formal_columns
    ]
    print(f'\n{trace_date.date()} 的最终{result_model_name}，收益为小数')
    print(exact_factors.to_string(index=False, na_rep='—'))


# %% 09. 查看指定交易日各组及追踪股票的组合收益
def label_group_preview(table: pd.DataFrame) -> pd.DataFrame:
    """为组合预览副本加上中文组名和列名。"""
    labels: pd.DataFrame = table.copy()
    labels['weight_method'] = labels['weight_method'].map(TRACE_WEIGHT_NAMES)
    labels['signal'] = labels['signal'].map(TRACE_FEATURE_NAMES)
    labels['size_group'] = labels['size_group'].map({'S': '小盘', 'B': '大盘'})
    group_names: dict[str, str] = {'L': '低', 'M': '中', 'H': '高'}
    labels['feature_group'] = labels['feature_group'].map(group_names)
    if 'included' in labels:
        labels['included'] = labels['included'].map({True: '是', False: '否'})
        labels['formation_feature_group'] = labels[
            'formation_feature_group'
        ].map(group_names)
    return labels.rename(
        columns={
            'date': '交易日',
            'weight_method': '加权',
            'signal': '排序特征',
            'size_group': '规模组',
            'feature_group': '特征组',
            'formation_feature_group': '六月原特征组',
            'n_stocks': '股票数',
            'sum_weight': '权重合计',
            'portfolio_return': '组合收益率（小数）',
            'included': '该股参与加权',
        }
    )


group_preview_table: pd.DataFrame = pd.DataFrame()
if group_preview_parts:
    observed_groups: pd.DataFrame = pd.concat(
        group_preview_parts, ignore_index=True
    )
    group_keys: list[str] = [
        'weight_method', 'signal', 'size_group', 'feature_group'
    ]
    group_grid_keys: pd.MultiIndex = pd.MultiIndex.from_product(
        [
            list(WEIGHTS), list(TRACE_FEATURE_NAMES),
            ['S', 'B'], ['L', 'M', 'H'],
        ],
        names=group_keys,
    )
    group_grid: pd.DataFrame = (
        observed_groups.set_index(group_keys)
        [['n_stocks', 'sum_weight', 'portfolio_return']]
        .reindex(group_grid_keys)
        .reset_index()
    )
    # 空格没有有效股票及权重，收益继续显示为缺失。
    group_grid['n_stocks'] = group_grid['n_stocks'].fillna(0).astype(int)
    group_grid['sum_weight'] = group_grid['sum_weight'].fillna(0)
    group_grid.insert(0, 'date', trace_date)
    group_preview_table = label_group_preview(group_grid)
    show_stage(
        '整理指定日的四种排序特征组合',
        before=observed_groups,
        after=group_preview_table,
        keys='交易日、加权口径、排序特征、规模组、特征组。',
        reasons='按两种权重补齐六格。空组股票数和权重显示为零，收益缺失。',
        before_rows=label_group_preview(observed_groups),
    )
    print(f'\n{trace_date.date()} 的全部组合格')
    with pd.option_context('display.unicode.east_asian_width', True):
        print(group_preview_table.to_string(index=False, na_rep='—'))

if trace_stock_code is not None:
    print(f'\n{trace_date.date()} 股票 {trace_stock_code:06d} 的对应组合')
    if stock_group_preview_parts:
        print('当日无有效行情时，按六月标签列出对应组供对照。')
        exact_stock_groups: pd.DataFrame = pd.concat(
            stock_group_preview_parts, ignore_index=True
        )
        with pd.option_context('display.unicode.east_asian_width', True):
            print(label_group_preview(exact_stock_groups).to_string(
                index=False, na_rep='—'
            ))
    else:
        print('该股当日没有进入日行情与六月分组的合并结果。')


# %% 10. 按交易日及区间对照国泰安公布的日度因子
def comparison_metrics(
    comparison_tables: dict[int, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """按区间计算同日配对的误差、相关性和配对交易日数。"""
    metric_rows: list[dict[str, object]] = []
    model: int
    comparison_table: pd.DataFrame
    factor_name: str
    weight_method: int
    for model, comparison_table in comparison_tables.items():
        selected_dates: pd.DataFrame = comparison_table.loc[
            comparison_table['date'].between(start, end)
        ]
        if selected_dates.empty:
            continue
        for factor_name in FACTORS[model]:
            for weight_method in WEIGHTS:
                factor_column: str = f'{factor_name}{weight_method}'
                calculated: pd.Series = selected_dates[f'{factor_column}_calc']
                reference_column: str = f'{factor_column}_official'
                reference: pd.Series = selected_dates[reference_column]
                difference: pd.Series = selected_dates[f'{factor_column}_bps']
                metric_rows.append({
                    '模型': TRACE_MODEL_NAMES[model],
                    '因子': FACTOR_NAMES[factor_name],
                    '加权': TRACE_WEIGHT_NAMES[weight_method],
                    '交易日数': int(difference.count()),
                    '相关系数': calculated.corr(reference),
                    '均方根误差（基点）': np.sqrt(difference.pow(2).mean()),
                    '平均绝对误差（基点）': difference.abs().mean(),
                    '平均偏差（基点）': difference.mean(),
                })
    return pd.DataFrame(metric_rows)


factor_comparison_tables: dict[int, pd.DataFrame] = {}
comparison_model: int
daily_factor_table: pd.DataFrame
for comparison_model, daily_factor_table in results.items():
    official_factor_table: pd.DataFrame = official[comparison_model]
    matched_factors: pd.DataFrame = daily_factor_table.merge(
        official_factor_table,
        on='date',
        how='left',
        suffixes=('_calc', '_official'),
        validate='one_to_one',
    )
    compared_column: str
    compared_columns: list[str] = factor_return_columns(comparison_model)
    for compared_column in compared_columns:
        matched_factors[f'{compared_column}_bps'] = (
            matched_factors[f'{compared_column}_calc']
            - matched_factors[f'{compared_column}_official']
        ) * 10_000
    factor_comparison_tables[comparison_model] = matched_factors

    sample_factor: str = compared_columns[0]
    reference_sample_columns: list[str] = [
        'date',
        f'{sample_factor}_calc',
        f'{sample_factor}_official',
        f'{sample_factor}_bps',
    ]
    show_stage(
        f'{TRACE_MODEL_NAMES[comparison_model]}对照发布值',
        before=daily_factor_table,
        after=matched_factors,
        keys='交易日 date，一对一左连接。',
        reasons=(
            f'发布表有 {len(official_factor_table):,} 行、'
            f'{official_factor_table.shape[1]} 列。保留所有复算交易日，'
            '缺少发布值时保留缺失。样例以流通市值加权的市场风险溢价'
            '展示复算值、官方值和基点差值。'
        ),
        before_rows=nearby_date_rows(
            daily_factor_table.loc[:, ['date', sample_factor]], trace_date
        ),
        after_rows=nearby_date_rows(
            matched_factors.loc[:, reference_sample_columns], trace_date
        ).rename(columns={
            'date': '交易日',
            f'{sample_factor}_calc': '复算值（小数）',
            f'{sample_factor}_official': '官方值（小数）',
            f'{sample_factor}_bps': '差值（基点）',
        }),
    )

comparisons: dict[str, pd.DataFrame] = {}
period_title: str
period_start: pd.Timestamp
period_end: pd.Timestamp
for period_title, period_start, period_end in PERIODS:
    comparison_start: pd.Timestamp = max(sample_start, period_start)
    comparison_end: pd.Timestamp = min(sample_end, period_end)
    if comparison_start > comparison_end:
        continue
    period_comparison: pd.DataFrame = comparison_metrics(
        factor_comparison_tables, comparison_start, comparison_end
    )
    if not period_comparison.empty:
        comparisons[period_title] = period_comparison
        print(
            f'\n{period_title}  {comparison_start.date()}'
            f' 至 {comparison_end.date()}'
        )
        with pd.option_context('display.unicode.east_asian_width', True):
            print(period_comparison.to_string(
                index=False, float_format='{:.6f}'.format
            ))
