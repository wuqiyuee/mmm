# Rptdt[报告日期] 盈利预测指标公布的时间，即报告的撰写日期。(没有缺失值)
# DeclareDate[发布日期] 是研究报告平台发布日期。（有少量的缺失值）
# 目前决定主要使用发布日期（发布之后被市场所接收到），如果发布日期是缺失值的话就使用报告日期

# 参照中文文献的做法使用股本进行调整
from pathlib import Path

import pandas as pd
import numpy as np

# 根据脚本位置定位项目根目录，整个项目移动后无需修改路径。
ROOT = Path(__file__).resolve().parents[4]


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# =============================================================================
# 配置区
# =============================================================================

# sue1、sue2 主口径：
# "cumulative" = 累计报告期
# "quarterly" = 独立单季度
SUE_MODE = "quarterly"

# 分析师预测窗口，同时计算两个口径
FORECAST_WINDOWS = (180, 365)

# 保留原有 sue3 字段时，默认指向哪个窗口
SUE3_PRIMARY_WINDOW = 365

# sue3使用实际公告日前第几个市场交易日的收盘价
SUE3_PRICE_LAG_TRADING_DAYS = 3

# 财报截止日向前匹配总股本时允许的最大自然日数
MAX_SHARE_SHIFT_DAYS = 10

# 财报截止日向前匹配股价时允许的最大自然日数
MAX_PRICE_SHIFT_DAYS = 10

VALID_SUE_MODES = {"cumulative", "quarterly"}

# 财报公告日期上限
START_DATE = '2000-01-01'
END_DATE = '2025-12-31'

if SUE_MODE not in VALID_SUE_MODES:
    raise ValueError(
        f"SUE_MODE只能是{VALID_SUE_MODES}，当前为：{SUE_MODE}"
    )

if SUE3_PRIMARY_WINDOW not in FORECAST_WINDOWS:
    raise ValueError(
        f"SUE3_PRIMARY_WINDOW必须属于{FORECAST_WINDOWS}"
    )



### 路径
FORECAST_PATH = ROOT / '中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv'

REPORT_PATH = ROOT / 'input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv'
TRADING_CALENDAR_PATH = ROOT / 'input/CSMAR/日历文件(1990-12-19至2026-08-20)/CER_Cale.csv'

TOTAL_SHARE_PATH = ROOT / '中间结果/00_合并原始数据/CSMAR_股票历史日行情信息表_前复权_1996_2025_合并结果.csv'


PRICE_PATH = ROOT / '中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv'


OUTPUT_PATH = ROOT / '中间结果/02_变量构造结果/4sue/sue.csv'


def normalize_stock_code(x:object) -> str:
    """标准化股票代码，填充为6位数，并转换为字符串"""
    if pd.isna(x):
        return ''
    s=str(x).strip()
    if s=='':
        return ''
    return f"{int(float(s)):06d}"

#===============================================================================
# Step 1. 读取预测数据 
#===============================================================================
# Rptdt[报告日期] 盈利预测指标公布的时间，即报告的撰写日期。
# DeclareDate[发布日期] 是研究报告平台发布日期。
# 目前决定主要使用发布日期，如果发布日期是缺失值的话就使用报告日期


def load_forecast_data(file_path, end_date):
    """
    读取并清洗分析师每股收益预测数据。

    处理内容：
        1. 标准化股票代码；
        2. 标准化机构编号；
        3. 将分析师预测的盈利指标到期日改名为Accper（方便后续合并）；
        4. 优先使用发布日期，缺失时使用报告日期；
        5. 删除预测值或必要日期缺失的记录。
    """

    forecast = pd.read_csv(
        file_path,
        encoding='utf-8',
        usecols=[
            'Stkcd',
            'InstitutionID',
            'AnanmID',
            'Ananm',
            'Feps',
            'Fenddt',
            'Rptdt',
            'DeclareDate'
        ]
    )

    forecast['Stkcd'] = (
        forecast['Stkcd']
        .apply(normalize_stock_code)
    )

    forecast['InstitutionID'] = (
        forecast['InstitutionID']
        .astype('string')
        .str.strip()
        .str.replace(r'\.0$', '', regex=True)
    )

    forecast['Fenddt'] = pd.to_datetime(
        forecast['Fenddt']
    )

    forecast = forecast.rename(
        columns={'Fenddt': 'Accper'}
    )

    forecast = forecast.loc[
        forecast['Accper'] <= pd.Timestamp(end_date)
    ].copy()

    forecast['Rptdt'] = pd.to_datetime(
        forecast['Rptdt']
    )

    forecast['DeclareDate'] = pd.to_datetime(
        forecast['DeclareDate']
    )

    # 优先采用发布日期，缺失时使用报告日期
    # 发布日期的缺失率仅 0.604%
    forecast['anndats'] = (
        forecast['DeclareDate']
        .fillna(forecast['Rptdt'])
    )

    forecast = forecast.drop(
        columns=['Rptdt', 'DeclareDate']
    )

    forecast['Feps'] = pd.to_numeric(
        forecast['Feps'],
        errors='coerce'
    )

    forecast = forecast.dropna(
        subset=['Feps', 'anndats', 'Accper']
    )

    return forecast.reset_index(drop=True)


#===============================================================================
# Step 2. 连接预测数据与真实数据 
#===============================================================================

# 根据 FORECAST_WINDOWS 构造实际公告日前1天至相应上限的预测样本

def load_report_data(
    file_path,
    end_date,
    exchange_codes=(1, 2)
):
    """
    读取并清洗年报、中报和季报基本情况数据。

    参数
    ----------
    file_path：
        财报数据文件路径。

    end_date：
        公告日期上限。

    exchange_codes：
        保留的交易所区域代码，默认保留上交所和深交所。

    返回
    ----------
    每家公司、每个报告期的实际财报数据。
    """

    report = pd.read_csv(
        file_path,
        encoding='utf-8',
        usecols=[
            'Stkcd',
            'Reptyp',
            'Accper',
            'Sctcd',
            'Annodt',
            'Erana',
            'Eranb'
        ]
    )

    report['Stkcd'] = (
        report['Stkcd']
        .apply(normalize_stock_code)
    )

    report['Accper'] = pd.to_datetime(
        report['Accper']
    )

    report['Annodt'] = pd.to_datetime(
        report['Annodt']
    )

    # 保留指定交易所
    report = report.loc[
        report['Sctcd'].isin(exchange_codes)
    ].copy()

    # 保留公告日期不超过样本截止日的记录
    report = report.loc[
        report['Annodt'] <= pd.Timestamp(end_date)
    ].copy()

    report = report.drop(
        columns=['Sctcd']
    )

    return report.reset_index(drop=True)


def select_latest_forecast(data, window_days):
    """
    在指定公告前窗口内构造每位分析师的最新预测。

    处理顺序：
        1. 筛选严格发生在公告日前的窗口内预测；
        2. 同一分析师同一预测日存在多个预测时，
           先对预测EPS取中位数；
        3. 再保留每位分析师日期最新的预测。
    """

    # 先筛选公告前指定窗口内的预测
    result = data.loc[
        data['dgap_days'].between(
            1,
            window_days,
            inclusive='both'
        )
    ].copy()

    analyst_keys = [
        'Stkcd',
        'Accper',
        'InstitutionID',
        'AnanmID'
    ]

    # 同一分析师同一预测日的分组键
    analyst_day_keys = (
        analyst_keys + ['anndats']
    )

    # 同一分析师同一天存在多个预测时，
    # 将Feps统一替换为当日预测中位数
    result['Feps'] = (
        result
        .groupby(
            analyst_day_keys,
            dropna=False
        )['Feps']
        .transform('median')
    )

    # 每个“分析师 × 预测日”只保留一行
    result = (
        result
        .sort_values(
            analyst_day_keys,
            kind='mergesort'
        )
        .drop_duplicates(
            subset=analyst_day_keys,
            keep='last'
        )
    )

    # 在公告前窗口内，
    # 保留每位分析师日期最新的预测
    result = (
        result
        .sort_values(
            analyst_keys + ['anndats'],
            kind='mergesort'
        )
        .drop_duplicates(
            subset=analyst_keys,
            keep='last'
        )
    )

    # 标记所属预测窗口
    result['forecast_window'] = window_days

    return result.reset_index(drop=True)


def prepare_forecast_window_samples(
    forecast,
    report,
    forecast_windows
):
    """
    将预测数据匹配至真实财报公告，
    并构造不同公告前窗口的最新分析师预测样本。
    """

    # 将预测匹配到对应报告期的真实公告
    merged = pd.merge(
        forecast,
        report,
        how='inner',
        on=['Stkcd', 'Accper']
    )

    # 计算预测日至真实公告日的自然日数
    # 大于0表示预测严格发生在公告日前
    merged['dgap_days'] = (
        merged['Annodt']
        - merged['anndats']
    ).dt.days

    # 分别构造不同预测窗口的样本
    forecast_samples = [
        select_latest_forecast(
            merged,
            window_days
        )
        for window_days in forecast_windows
    ]

    # 合并成长表
    # 同一条预测可能同时属于180天和365天窗口
    return pd.concat(
        forecast_samples,
        ignore_index=True
    )



def build_share_reference_dates(forecast_data):
    """
    构造sue3进行股本口径调整所需的日期表。

    日期包括：
        1. 分析师预测发布日期；
        2. 财报统计截止日期Accper。

    为复用后续日期匹配流程，
    两类日期统一存放在anndats列。
    """

    # 分析师预测发布日期
    forecast_dates = (
        forecast_data[['Stkcd', 'anndats']]
        .dropna()
        .drop_duplicates()
    )

    # 财报统计截止日期
    report_period_dates = (
        forecast_data[['Stkcd', 'Accper']]
        .dropna()
        .drop_duplicates()
        .rename(columns={'Accper': 'anndats'})
    )

    return (
        pd.concat(
            [
                forecast_dates,
                report_period_dates
            ],
            ignore_index=True
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

# 把所有的预测发布日期和财报截止日调整到最近的交易日（向前匹配）
# 为了提取调整因子（这里暂时参照中文文献的做法使用股本进行调整）
# CSMAR中公布了 每股收益计算的分母 使用的是财报截止日的普通股总股本 

def load_trading_calendar(
    file_path,
    market_type=1,
    open_state='O'
):
    """
    读取并清洗市场交易日历。

    参数
    ----------
    file_path：
        交易日历文件路径。

    market_type：
        需要保留的市场类型。

    open_state：
        代表正常开市交易的状态代码。

    返回
    ----------
    指定市场的正常交易日历。
    """

    calendar = pd.read_csv(
        file_path,
        encoding='utf-8',
        usecols=[
            'Clddt',
            'Markettype',
            'Statse'
        ]
    )

    calendar['Clddt'] = pd.to_datetime(
        calendar['Clddt']
    )

    calendar = calendar.loc[
        calendar['Markettype'].eq(market_type)
        & calendar['Statse'].eq(open_state)
    ].copy()

    return calendar.reset_index(drop=True)

def map_reference_dates_to_trading_days(
    reference_dates,
    trading_calendar
):
    """
    将股本基准日期向前映射到最近交易日。

    输入日期包括：
        1. 分析师预测发布日期；
        2. 实际财报公告日期。

    返回字段包括：
        anndats：原始股本基准日期；
        Clddt：向前匹配到的最近交易日；
        days_shift_back：向前调整的自然日数。
    """

    # 整理需要匹配的唯一股本基准日期
    unique_dates = (
        reference_dates[['anndats']]
        .dropna()
        .drop_duplicates()
        .sort_values('anndats')
        .reset_index(drop=True)
    )

    # 整理唯一交易日
    calendar_dates = (
        trading_calendar[['Clddt']]
        .dropna()
        .drop_duplicates()
        .sort_values('Clddt')
        .reset_index(drop=True)
    )

    # 交易日当天保留当天；
    # 非交易日向前匹配最近的交易日
    mapped_dates = pd.merge_asof(
        unique_dates,
        calendar_dates,
        left_on='anndats',
        right_on='Clddt',
        direction='backward'
    )

    # 计算原始日期向前移动的自然日数
    mapped_dates['days_shift_back'] = (
        mapped_dates['anndats']
        - mapped_dates['Clddt']
    ).dt.days

    # 输出匹配情况
    print("\n股本基准日期向前匹配最近交易日结果：")
    print(
        "原始唯一 anndats 数量：",
        unique_dates['anndats'].nunique()
    )
    print(
        "成功匹配数量：",
        mapped_dates['Clddt'].notna().sum()
    )
    print(
        "未匹配数量：",
        mapped_dates['Clddt'].isna().sum()
    )

    print("\n向前调整天数分布：")
    print(
        mapped_dates['days_shift_back']
        .value_counts(dropna=False)
        .sort_index()
    )

    unmatched = mapped_dates.loc[
        mapped_dates['Clddt'].isna()
    ]

    if len(unmatched) > 0:
        print("\n仍未匹配的 anndats：")
        print(unmatched)

    return mapped_dates





# 将分析师预测EPS调整到财报统计截止日股本口径

# sample code使用CFACSHR股份累计调整因子：
# 调整后预测EPS
# = 原预测EPS × 公告日调整因子 / 预测日调整因子

# 本脚本参照中文文献，使用总股本比例调整：
# 调整后预测EPS
# = 原预测EPS × 预测发布日期总股本 / 财报统计截止日总股本

# CSMAR说明书明确：中报和年报每股收益的分母
# 为报告期末或年度末普通股股份总数；
# 本脚本将该股本口径用于一季报和三季报。

# ① 按照 CSMAR 字段 每股收益eps=净利润/报告期末普通股总股数，应使用普通股总股数进行调整

# ② 查看 CSMAR 的股本变动文件，样本范围内只有2只股票的优先股取值不为0
# 股票：000509 日期：2008/12/31	股本变动原因：00008（其他）	总股数：250009885 优先股：106641
# （在此之前优先股为0，在此之后不知道哪天优先股变为0且再也没增加）
# 股票：600671 日期：2000/10/9	股本变动原因：30000	总股数：121778628 优先股：18900000
# （从上市首日起就一直有优先股，再次之后不知道哪天优先股变为0且再也没增加）
# 鉴于此，优先股的影响很小，决定使用总股本来进行调整

# ③ 对比了 CSMAR 数据库中的 股票历史日行情信息表(后复权)&股票历史日行情信息表(前复权)，这两张表上的总股数字段是一致的，故决定任意采用一个

def load_total_share_data(file_path):
    """
    读取并清洗股票日度总股本数据。

    参数
    ----------
    file_path：
        股票日度总股本数据文件路径。

    返回
    ----------
    包含股票代码、交易日期和总股本的数据。
    """

    total_share = pd.read_csv(
        file_path,
        encoding='utf-8',
        usecols=[
            'TradingDate',
            'Symbol',
            'TotalShare'
        ]
    )

    total_share['TradingDate'] = pd.to_datetime(
        total_share['TradingDate']
    )

    total_share = total_share.rename(
        columns={
            'Symbol': 'Stkcd',
            'TradingDate': 'Clddt'
        }
    )

    total_share['Stkcd'] = (
        total_share['Stkcd']
        .apply(normalize_stock_code)
    )

    return total_share

def attach_total_share_to_reference_dates(
    reference_dates,
    mapped_dates,
    total_share
):
    """
    为预测发布日期和实际财报公告日期匹配日度总股本。

    处理步骤：
        1. 合并基准日期对应的交易日；
        2. 按股票代码和交易日匹配总股本；
        3. 保留无法匹配总股本的基准日期。
    """

    result = reference_dates.copy()

    date_lookup = mapped_dates[
        ['anndats', 'Clddt', 'days_shift_back']
    ].copy()

    share_lookup = total_share[
        ['Stkcd', 'Clddt', 'TotalShare']
    ].copy()

    duplicate_date = date_lookup.duplicated(
        subset=['anndats']
    ).sum()

    if duplicate_date > 0:
        raise ValueError(
            f"交易日映射表存在{duplicate_date}条重复日期"
        )

    duplicate_share = share_lookup.duplicated(
        subset=['Stkcd', 'Clddt']
    ).sum()

    if duplicate_share > 0:
        raise ValueError(
            f"日度总股本表存在{duplicate_share}条重复记录"
        )

    # 匹配基准日期对应的交易日
    result = pd.merge(
        result,
        date_lookup,
        how='left',
        on='anndats',
        validate='many_to_one'
    )

    # 按股票代码和交易日匹配总股本
    result = pd.merge(
        result,
        share_lookup,
        how='left',
        on=['Stkcd', 'Clddt'],
        validate='many_to_one'
    )

    print("\n股本基准日期匹配总股本的情况：")
    print("基准日期记录数：", len(result))
    print(
        "未成功匹配总股本数量：",
        result['TotalShare'].isna().sum()
    )

    return result



def build_share_adjusted_forecast_median(
        forecast_data,
        share_reference_data,
        forecast_windows,
        primary_window
    ):
    """
    将分析师预测EPS调整到财报统计截止日股本口径，
    并计算不同预测窗口的预测中位数和有效预测数量。
    """

    result = forecast_data.copy()

    if primary_window not in forecast_windows:
        raise ValueError(
            "primary_window必须属于forecast_windows"
        )

    share_lookup = share_reference_data[
        ['Stkcd', 'anndats', 'TotalShare']
    ].copy()

    duplicate_share = share_lookup.duplicated(
        subset=['Stkcd', 'anndats']
    ).sum()

    if duplicate_share > 0:
        raise ValueError(
            f"股本日期表存在{duplicate_share}条重复记录"
        )

    # 匹配预测发布日期的总股本
    forecast_share = share_lookup.rename(
        columns={'TotalShare': 'TotalShare_forecast'}
    )

    result = pd.merge(
        result,
        forecast_share,
        how='left',
        on=['Stkcd', 'anndats'],
        validate='many_to_one'
    )

    # 匹配财报统计截止日的总股本
    report_period_share = share_lookup.rename(
        columns={
            'anndats': 'Accper',
            'TotalShare': 'TotalShare_accper'
        }
    )

    result = pd.merge(
        result,
        report_period_share,
        how='left',
        on=['Stkcd', 'Accper'],
        validate='many_to_one'
    )

    valid_share = (
        result['TotalShare_forecast'].gt(0)
        & result['TotalShare_accper'].gt(0)
    )

    # 将预测EPS换算到财报统计截止日股本口径
    result['new_Feps'] = (
        result['Feps']
        * result['TotalShare_forecast']
        / result['TotalShare_accper']
    ).where(valid_share)

    analyst_keys = [
        'Stkcd',
        'Accper',
        'InstitutionID',
        'AnanmID',
        'forecast_window'
    ]

    duplicate_forecast = result.duplicated(
        subset=analyst_keys
    ).sum()

    if duplicate_forecast > 0:
        raise ValueError(
            f"最新预测样本存在{duplicate_forecast}条重复记录"
        )

    forecast_median_long = (
        result
        .groupby(
            ['Stkcd', 'Accper', 'forecast_window'],
            as_index=False
        )
        .agg(
            forecast_eps_median=('new_Feps', 'median'),
            forecast_count=('new_Feps', 'count')
        )
    )

    forecast_median = forecast_median_long.pivot(
        index=['Stkcd', 'Accper'],
        columns='forecast_window',
        values=[
            'forecast_eps_median',
            'forecast_count'
        ]
    )

    forecast_median.columns = [
        f'{variable}_{int(window_days)}'
        for variable, window_days
        in forecast_median.columns.to_flat_index()
    ]

    forecast_median = forecast_median.reset_index()

    for window_days in forecast_windows:
        forecast_eps_median_column = (
            f'forecast_eps_median_{window_days}'
        )

        forecast_count_column = (
            f'forecast_count_{window_days}'
        )

        if (
            forecast_eps_median_column
            not in forecast_median.columns
        ):
            forecast_median[
                forecast_eps_median_column
            ] = np.nan

        if (
            forecast_count_column
            not in forecast_median.columns
        ):
            forecast_median[
                forecast_count_column
            ] = pd.NA

        forecast_median[forecast_count_column] = (
            forecast_median[
                forecast_count_column
            ].astype('Int64')
        )

    # 主预测窗口对应的预测中位数和预测数量
    forecast_median['forecast_eps_median'] = (
        forecast_median[
            f'forecast_eps_median_{primary_window}'
        ]
    )

    forecast_median['forecast_count'] = (
        forecast_median[
            f'forecast_count_{primary_window}'
        ]
    )

    return forecast_median


#===============================================================================
# Step 4. 合并公司财务数据
# 左连接保留没有分析师预测的财报，以继续计算sue1和sue2
#===============================================================================
# 经过一番检查发现，A股上市公司基本都满足总资产>=0的要求。

def merge_report_with_forecast_median(
    report_data,
    forecast_median_data,
    forecast_windows
):
    """
    将财报数据与不同窗口的分析师预测中位数合并。

    合并后仍保持：
        每家公司 × 每个财报截止日一行。

    没有分析师预测的财报记录仍然保留。
    """

    window_columns = [
        column
        for window_days in forecast_windows
        for column in (
            f'forecast_eps_median_{window_days}',
            f'forecast_count_{window_days}'
        )
    ]

    forecast_median_columns = [
        'Stkcd',
        'Accper',
        *window_columns,
        'forecast_eps_median',
        'forecast_count'
    ]

    missing_columns = [
        column
        for column in forecast_median_columns
        if column not in forecast_median_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"预测中位数数据缺少字段：{missing_columns}"
        )

    duplicate_report = report_data.duplicated(
        subset=['Stkcd', 'Accper']
    ).sum()

    if duplicate_report > 0:
        raise ValueError(
            f"财报数据存在{duplicate_report}条重复记录"
        )

    duplicate_forecast_median = (
        forecast_median_data.duplicated(
            subset=['Stkcd', 'Accper']
        ).sum()
    )

    if duplicate_forecast_median > 0:
        raise ValueError(
            "预测中位数数据存在"
            f"{duplicate_forecast_median}条重复记录"
        )

    result = pd.merge(
        report_data,
        forecast_median_data[
            forecast_median_columns
        ],
        how='left',
        on=['Stkcd', 'Accper'],
        validate='one_to_one'
    )

    return (
        result
        .sort_values(
            ['Stkcd', 'Reptyp', 'Accper'],
            kind='mergesort'
        )
        .reset_index(drop=True)
    )

#===============================================================================
# Step 5. 计算标准化的盈余惊喜
#===============================================================================
def prepare_sue_report_data(
    report_data,
    total_share,
    max_shift_days
):
    """
    整理SUE计算所需的财报基础数据，
    并匹配财报截止日对应的总股本。

    总股本仅用于sue1和sue2的历史每股收益调整。
    """

    if max_shift_days < 0:
        raise ValueError(
            "max_shift_days不能小于0"
        )

    result = report_data.copy()

    result['fyearq'] = result['Accper'].dt.year

    result = result.rename(
        columns={'Reptyp': 'fqtr'}
    )

    share_lookup = total_share[
        ['Stkcd', 'Clddt', 'TotalShare']
    ].copy()

    duplicate_share = share_lookup.duplicated(
        subset=['Stkcd', 'Clddt']
    ).sum()

    if duplicate_share > 0:
        raise ValueError(
            f"日度总股本表存在{duplicate_share}条重复记录"
        )

    # merge_asof要求按照日期合并键排序
    result = result.sort_values(
        ['Accper', 'Stkcd'],
        kind='mergesort'
    )

    share_lookup = share_lookup.sort_values(
        ['Clddt', 'Stkcd'],
        kind='mergesort'
    )

    # 财报截止日当天没有记录时，
    # 向前匹配最近一条总股本记录
    result = pd.merge_asof(
        result,
        share_lookup,
        by='Stkcd',
        left_on='Accper',
        right_on='Clddt',
        direction='backward'
    )

    result['days_shift_back'] = (
        result['Accper']
        - result['Clddt']
    ).dt.days

    print("\n使用财报截止日匹配总股本的情况：")
    print(
        "未成功匹配数量：",
        result['TotalShare'].isna().sum()
    )

    print("\n向前调整天数分布：")
    print(
        result['days_shift_back']
        .value_counts(dropna=False)
        .sort_index()
    )

    stale_share = (
        result['days_shift_back'] > max_shift_days
    )

    result.loc[
        stale_share,
        'TotalShare'
    ] = np.nan

    print(
        "剔除过期匹配后缺失总股本数量：",
        result['TotalShare'].isna().sum()
    )

    # 明确保留财报截止日总股本，
    # 供sue3换算到公告日前第3个交易日股本口径
    result['TotalShare_accper'] = pd.to_numeric(
        result['TotalShare'],
        errors='coerce'
    )

    result = result.drop(
        columns=[
            'Clddt',
            'days_shift_back'
        ]
    )

    return (
        result
        .sort_values(
            ['Stkcd', 'fqtr', 'fyearq'],
            kind='mergesort'
        )
        .reset_index(drop=True)
    )


def add_actual_eps_versions(data):
    """
    构造累计口径和独立单季度口径的实际每股收益。
    """

    result = data.copy()

    result['fqtr'] = pd.to_numeric(
        result['fqtr'],
        errors='coerce'
    ).astype('Int64')

    result['TotalShare'] = pd.to_numeric(
        result['TotalShare'],
        errors='coerce'
    )

    result['actual1_cum'] = pd.to_numeric(
        result['Erana'],
        errors='coerce'
    )

    result['actual2_cum'] = pd.to_numeric(
        result['Eranb'],
        errors='coerce'
    )

    # 按公司、年度、报告期排序
    result = result.sort_values(
        ['Stkcd', 'fyearq', 'fqtr', 'Accper'],
        kind='mergesort'
    ).reset_index(drop=True)

    annual_group = result.groupby(
        ['Stkcd', 'fyearq'],
        sort=False
    )

    previous_fqtr = (
        annual_group['fqtr']
        .shift(1)
    )

    previous_total_share = (
        annual_group['TotalShare']
        .shift(1)
    )

    is_q1 = result['fqtr'].eq(1).fillna(False)

    # 只允许紧邻报告期相减
    is_consecutive = (
        result['fqtr'].between(2, 4)
        & previous_fqtr.eq(result['fqtr'] - 1)
    ).fillna(False)

    for number in (1, 2):
        cumulative_column = f'actual{number}_cum'
        quarterly_column = f'actual{number}_qtr'

        previous_cumulative = (
            annual_group[cumulative_column]
            .shift(1)
        )

        result[quarterly_column] = np.nan

        # 一季度累计值就是独立一季度值
        result.loc[is_q1, quarterly_column] = (
            result.loc[is_q1, cumulative_column]
        )

        # 其他季度：先将上一报告期累计EPS换算到本期末股本口径，
        # 再从本期累计EPS中扣除
        valid_quarterly = (
            is_consecutive
            & result[cumulative_column].notna()
            & previous_cumulative.notna()
            & result['TotalShare'].gt(0)
            & previous_total_share.gt(0)
        )

        result.loc[
            valid_quarterly,
            quarterly_column
        ] = (
            result.loc[
                valid_quarterly,
                cumulative_column
            ]
            - previous_cumulative.loc[
                valid_quarterly
            ]
            * previous_total_share.loc[
                valid_quarterly
            ]
            / result.loc[
                valid_quarterly,
                'TotalShare'
            ]
        )

    return result.sort_values(
        ['Stkcd', 'fqtr', 'fyearq', 'Accper'],
        kind='mergesort'
    ).reset_index(drop=True)


def add_expected_eps_versions(data):
    """
    分别构造累计口径和独立单季度口径的上年同期预期每股收益。

    预期值：
        上年同报告类型实际EPS
        × 上年同期总股本
        ÷ 本期总股本
    """

    result = data.copy()

    result['TotalShare'] = pd.to_numeric(
        result['TotalShare'],
        errors='coerce'
    )

    # 按公司、报告类型、年度排序
    result = result.sort_values(
        ['Stkcd', 'fqtr', 'fyearq', 'Accper'],
        kind='mergesort'# 稳定排序，如果两行在排序键上完全一样，它们原来的相对顺序不会乱。
    ).reset_index(drop=True)

    same_quarter_group = result.groupby(
        ['Stkcd', 'fqtr'],# 按照股票 ✖️ 财报类型分组
        sort=False
    )

    previous_year = (
        same_quarter_group['fyearq']
        .shift(1)
    )

    previous_total_share = (
        same_quarter_group['TotalShare']
        .shift(1)
    )# 取上年总股本

    # 必须是连续两年的相同报告类型
    is_consecutive_year = previous_year.eq(
        result['fyearq'] - 1
    ).fillna(False)

    valid_share = (
        is_consecutive_year
        & previous_total_share.notna()
        & result['TotalShare'].notna()
        & previous_total_share.gt(0)
        & result['TotalShare'].gt(0)
    )

    for number in (1, 2):# sue1、sue2
        for version in ('cum', 'qtr'):# 累计、单季
            actual_column = f'actual{number}_{version}'
            expected_column = f'expected{number}_{version}'

            previous_actual = (
                same_quarter_group[actual_column]
                .shift(1)
            )# 取得上年同期的实际eps

            valid_observation = (
                valid_share
                & previous_actual.notna()
            )

            result[expected_column] = np.nan # 先创建空列

            result.loc[
                valid_observation,
                expected_column
            ] = (
                previous_actual.loc[valid_observation]
                * previous_total_share.loc[valid_observation]
                / result.loc[
                    valid_observation,
                    'TotalShare'
                ]
            )# 对上期eps进行股本调整

    return result


def select_eps_mode(data, sue_mode):
    """
    根据配置选择累计口径或独立单季度口径，
    同时保留所有明确口径字段。
    """

    result = data.copy()

    version_map = {
        'cumulative': 'cum',
        'quarterly': 'qtr'
    }

    version = version_map[sue_mode]

    for number in (1, 2):
        result[f'actual{number}'] = result[
            f'actual{number}_{version}'
        ]

        result[f'expected{number}'] = result[
            f'expected{number}_{version}'
        ]

    return result

def calculate_sue_versions(
    data,
    sue_mode,
    forecast_windows,
    sue3_primary_window
):
    """
    计算累计口径、独立单季度口径以及分析师预测口径的SUE。

    sue1、sue2：
        同时保留累计与独立单季度结果，
        主变量由sue_mode决定。

    sue3：
        同时保留不同预测窗口结果，
        主变量由sue3_primary_window决定。
    """

    result = data.copy()

    report_period_price = pd.to_numeric(
        result['report_period_price'],
        errors='coerce'
    )

    # 财报截止日价格必须大于0
    valid_report_period_price = (
        report_period_price.gt(0)
    )

    sue3_price = pd.to_numeric(
        result['sue3_price'],
        errors='coerce'
    )

    # 公告日前指定交易日价格必须大于0
    valid_sue3_price = sue3_price.gt(0)

    # 财报截止日总股本
    total_share_accper = pd.to_numeric(
        result['TotalShare_accper'],
        errors='coerce'
    )

    # 公告日前指定交易日总股本
    total_share_tminus3 = pd.to_numeric(
        result['TotalShare_tminus3'],
        errors='coerce'
    )

    # sue3所需的价格和两个时点总股本必须有效
    valid_sue3_inputs = (
        valid_sue3_price
        & total_share_accper.gt(0)
        & total_share_tminus3.gt(0)
    )

    # 分别计算sue1、sue2的累计和独立单季度口径
    for number in (1, 2):
        for version in ('cum', 'qtr'):
            actual_column = f'actual{number}_{version}'
            expected_column = f'expected{number}_{version}'
            sue_column = f'sue{number}_{version}'

            result[sue_column] = (
                (
                    result[actual_column]
                    - result[expected_column]
                )
                / report_period_price
            ).where(valid_report_period_price)

    # 根据配置选择主口径
    version_map = {
        'cumulative': 'cum',
        'quarterly': 'qtr'
    }

    selected_version = version_map[sue_mode]

    result['sue1'] = result[
        f'sue1_{selected_version}'
    ]

    result['sue2'] = result[
        f'sue2_{selected_version}'
    ]

    # 分别计算不同预测窗口下的sue3
    # sue3始终使用累计实际EPS，不受sue_mode影响
    # 分母使用实际公告日前指定交易日的收盘价
    for window_days in forecast_windows:
        forecast_median_column = (
            f'forecast_eps_median_{window_days}'
        )

        result[f'sue3_{window_days}'] = (
            (
                result['actual1_cum']
                - result[forecast_median_column]
            )
            * total_share_accper
            / total_share_tminus3
            / sue3_price
        ).where(valid_sue3_inputs)

    # 原有sue3字段指向主要预测窗口
    result['sue3'] = result[
        f'sue3_{sue3_primary_window}'
    ]

    return result



# 读取日度收盘价，分别构造：
# 1. sue1、sue2使用的财报截止日价格；
# 2. sue3使用的实际公告日前指定交易日价格。
def load_price_data(file_path):
    """
    读取并清洗日度收盘价数据。

    返回字段：
        Stkcd：股票代码；
        Trddt：交易日期；
        Clsprc：收盘价。
    """

    price = pd.read_csv(
        file_path,
        encoding='utf-8',
        usecols=[
            'Stkcd',
            'Trddt',
            'Clsprc'
        ]
    )

    price['Stkcd'] = (
        price['Stkcd']
        .apply(normalize_stock_code)
    )

    price['Trddt'] = pd.to_datetime(
        price['Trddt']
    )

    price['Clsprc'] = pd.to_numeric(
        price['Clsprc'],
        errors='coerce'
    )

    duplicate_price = price.duplicated(
        subset=['Stkcd', 'Trddt']
    ).sum()

    if duplicate_price > 0:
        raise ValueError(
            f"日度股价数据存在{duplicate_price}条重复记录"
        )

    return price.reset_index(drop=True)


def attach_report_period_price(
    data,
    price_data,
    max_shift_days
):
    """
    为每条财报匹配财报截止日前最近交易日的收盘价。

    该价格用于sue1和sue2的标准化分母。
    超过最大允许自然日数的匹配结果，
    将收盘价设置为空值。
    """

    if max_shift_days < 0:
        raise ValueError(
            "max_shift_days不能小于0"
        )

    result = data.copy()

    price_lookup = (
        price_data[
            ['Stkcd', 'Trddt', 'Clsprc']
        ]
        .rename(
            columns={
                'Trddt': 'report_price_date',
                'Clsprc': 'report_period_price'
            }
        )
        .copy()
    )

    duplicate_price = price_lookup.duplicated(
        subset=['Stkcd', 'report_price_date']
    ).sum()

    if duplicate_price > 0:
        raise ValueError(
            f"日度股价数据存在{duplicate_price}条重复记录"
        )

    result = result.sort_values(
        ['Accper', 'Stkcd'],
        kind='mergesort'
    )

    price_lookup = price_lookup.sort_values(
        ['report_price_date', 'Stkcd'],
        kind='mergesort'
    )

    result = pd.merge_asof(
        result,
        price_lookup,
        by='Stkcd',
        left_on='Accper',
        right_on='report_price_date',
        direction='backward'
    )

    result['report_price_shift_days'] = (
        result['Accper']
        - result['report_price_date']
    ).dt.days

    print("\n使用财报截止日匹配股价的情况：")
    print(
        "未成功匹配数量：",
        result['report_period_price'].isna().sum()
    )

    print("\n财报截止日股价向前调整天数分布：")
    print(
        result['report_price_shift_days']
        .value_counts(dropna=False)
        .sort_index()
    )

    stale_price = (
        result['report_price_shift_days']
        > max_shift_days
    )

    result.loc[
        stale_price,
        'report_period_price'
    ] = np.nan

    print(
        "剔除过期匹配后缺失股价数量：",
        result['report_period_price'].isna().sum()
    )

    result = result.drop(
        columns=['report_price_shift_days']
    )

    return (
        result
        .sort_values(
            ['Stkcd', 'Accper'],
            kind='mergesort'
        )
        .reset_index(drop=True)
    )


def attach_sue3_pre_announcement_price(
    data,
    price_data,
    total_share,
    trading_calendar,
    lag_trading_days
):
    """
    为sue3匹配实际公告日前指定市场交易日的收盘价。

    处理规则：
        1. 公告日在交易日时，使用该交易日作为公告交易日；
        2. 公告日在非交易日时，映射至下一交易日；
        3. 从公告交易日向前移动指定数量的市场交易日；
        4. 精确匹配该日期的个股收盘价。
    """

    if lag_trading_days < 1:
        raise ValueError(
            "lag_trading_days必须大于等于1"
        )

    if 'Annodt' not in data.columns:
        raise ValueError(
            "待匹配数据缺少Annodt字段"
        )

    result = data.copy()

    calendar_lookup = (
        trading_calendar[['Clddt']]
        .dropna()
        .drop_duplicates()
        .sort_values('Clddt')
        .reset_index(drop=True)
        .rename(
            columns={
                'Clddt': 'announcement_trading_date'
            }
        )
    )

    # 对每一个公告交易日，生成其前若干个市场交易日
    calendar_lookup['sue3_price_date'] = (
        calendar_lookup[
            'announcement_trading_date'
        ].shift(lag_trading_days)
    )

    announcement_date_lookup = (
        result[['Annodt']]
        .dropna()
        .drop_duplicates()
        .sort_values('Annodt')
        .reset_index(drop=True)
    )

    # 公告日在非交易日时，映射至下一交易日
    announcement_date_lookup = pd.merge_asof(
        announcement_date_lookup,
        calendar_lookup,
        left_on='Annodt',
        right_on='announcement_trading_date',
        direction='forward'
    )

    result = pd.merge(
        result,
        announcement_date_lookup,
        how='left',
        on='Annodt',
        validate='many_to_one'
    )

    price_lookup = (
        price_data[
            ['Stkcd', 'Trddt', 'Clsprc']
        ]
        .rename(
            columns={
                'Trddt': 'sue3_price_date',
                'Clsprc': 'sue3_price'
            }
        )
        .copy()
    )

    duplicate_price = price_lookup.duplicated(
        subset=['Stkcd', 'sue3_price_date']
    ).sum()

    if duplicate_price > 0:
        raise ValueError(
            f"日度股价数据存在{duplicate_price}条重复记录"
        )

    # 精确匹配公告日前指定市场交易日的股价
    result = pd.merge(
        result,
        price_lookup,
        how='left',
        on=['Stkcd', 'sue3_price_date'],
        validate='many_to_one'
    )

    # 在与sue3股价相同的日期匹配总股本
    share_lookup = (
        total_share[
            ['Stkcd', 'Clddt', 'TotalShare']
        ]
        .rename(
            columns={
                'Clddt': 'sue3_price_date',
                'TotalShare': 'TotalShare_tminus3'
            }
        )
        .copy()
    )

    share_lookup['TotalShare_tminus3'] = pd.to_numeric(
        share_lookup['TotalShare_tminus3'],
        errors='coerce'
    )

    duplicate_share = share_lookup.duplicated(
        subset=['Stkcd', 'sue3_price_date']
    ).sum()

    if duplicate_share > 0:
        raise ValueError(
            f"日度总股本表存在{duplicate_share}条重复记录"
        )

    result = pd.merge(
        result,
        share_lookup,
        how='left',
        on=['Stkcd', 'sue3_price_date'],
        validate='many_to_one'
    )

    print(
        f"\n使用公告日前第{lag_trading_days}个交易日"
        "匹配sue3股价的情况："
    )

    print(
        "无法映射公告交易日数量：",
        result[
            'announcement_trading_date'
        ].isna().sum()
    )

    print(
        "未成功匹配sue3股价数量：",
        result['sue3_price'].isna().sum()
    )

    print(
        "未成功匹配公告日前第"
        f"{lag_trading_days}个交易日总股本数量：",
        result['TotalShare_tminus3'].isna().sum()
    )

    return (
        result
        .sort_values(
            ['Stkcd', 'Accper'],
            kind='mergesort'
        )
        .reset_index(drop=True)
    )
# 与MMM原文代码的主要差异：
# ①预测窗口不同。MMM使用公告日前2—45天；
#   本脚本同时计算公告日前1—180天和1—365天。
# ②股价日期不同。MMM使用实际盈余公告日前第3个交易日的收盘价；
#   本脚本sue1、sue2使用财报截止日前最近交易日的收盘价。
#   sue3股价口径与MMM原文一致：
#   使用实际盈余公告日前第3个交易日的收盘价。
# ③sue3的股本口径处理不同。
#   MMM原文代码直接使用ACTUAL和VALUE，未在此处进行股本调整；
#   PEAD示例代码使用CFACSHR股份累计调整因子；
#   本脚本先将预测EPS换算至财报截止日股本口径，
#   再将实际EPS与预测EPS之差换算至公告日前第3个交易日股本口径。
#
# 本脚本内部不同SUE指标的股本调整基准：
# sue1、sue2：上年同期与本期财报截止日总股本；
# sue3：预测日→财报截止日→公告日前第3个交易日股本口径。

def save_sue_results(data, output_path):
    """
    保存SUE计算结果。

    参数
    ----------
    data：
        最终SUE结果数据。

    output_path：
        输出文件路径。
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        output_path,
        index=False,
        encoding='utf-8-sig'
    )

    print(
        f"已成功保存计算结果到：{output_path}"
    )

    return output_path





def main():
    """
    执行完整的SUE计算流程。

    处理顺序：
        1. 读取分析师预测和实际财报；
        2. 筛选公告日前预测；
        3. 匹配股本并调整分析师预测；sue3
        4. 构造累计和独立单季度EPS；sue1、sue2
        5. 分别匹配财报截止日股价和公告日前第3个交易日股价；
        6. 计算并保存全部SUE指标。
    """

    # -------------------------------------------------------------------------
    # 1. 读取分析师预测和实际财报
    # -------------------------------------------------------------------------
    forecast = load_forecast_data(
        file_path=FORECAST_PATH,
        end_date=END_DATE
    )

    report = load_report_data(
        file_path=REPORT_PATH,
        end_date=END_DATE,
        exchange_codes=(1, 2)
    )

    # -------------------------------------------------------------------------
    # 2. 构造不同公告前窗口的最新分析师预测
    # -------------------------------------------------------------------------
    forecast_samples = prepare_forecast_window_samples(
        forecast=forecast,
        report=report,
        forecast_windows=FORECAST_WINDOWS
    )

    del forecast

    # -------------------------------------------------------------------------
    # 3. 构造预测日和实际公告日的股本基准日期
    # -------------------------------------------------------------------------
    share_reference_dates = build_share_reference_dates(
        forecast_data=forecast_samples
    )

    trading_calendar = load_trading_calendar(
        file_path=TRADING_CALENDAR_PATH,
        market_type=1,
        open_state='O'
    )

    mapped_reference_dates = (
        map_reference_dates_to_trading_days(
            reference_dates=share_reference_dates,
            trading_calendar=trading_calendar
        )
    )

    # -------------------------------------------------------------------------
    # 4. 读取总股本并匹配至股本基准日期
    # -------------------------------------------------------------------------
    total_share = load_total_share_data(
        file_path=TOTAL_SHARE_PATH
    )

    share_reference_data = (
        attach_total_share_to_reference_dates(
            reference_dates=share_reference_dates,
            mapped_dates=mapped_reference_dates,
            total_share=total_share
        )
    )

    del mapped_reference_dates
    del share_reference_dates

    # -------------------------------------------------------------------------
    # 5. 调整分析师预测并计算预测中位数
    # -------------------------------------------------------------------------
    forecast_median = (
        build_share_adjusted_forecast_median(
            forecast_data=forecast_samples,
            share_reference_data=share_reference_data,
            forecast_windows=FORECAST_WINDOWS,
            primary_window=SUE3_PRIMARY_WINDOW
        )
    )

    del forecast_samples
    del share_reference_data

    # -------------------------------------------------------------------------
    # 6. 将实际财报与分析师预测中位数合并
    # -------------------------------------------------------------------------
    report_with_forecast = (
        merge_report_with_forecast_median(
            report_data=report,
            forecast_median_data=forecast_median,
            forecast_windows=FORECAST_WINDOWS
        )
    )

    del report
    del forecast_median

    # -------------------------------------------------------------------------
    # 7. 匹配财报截止日总股本
    # -------------------------------------------------------------------------
    sue = prepare_sue_report_data(
        report_data=report_with_forecast,
        total_share=total_share,
        max_shift_days=MAX_SHARE_SHIFT_DAYS
    )

    del report_with_forecast

    # -------------------------------------------------------------------------
    # 8. 构造累计和独立单季度EPS
    # -------------------------------------------------------------------------
    sue = add_actual_eps_versions(sue)
    sue = add_expected_eps_versions(sue)

    sue = sue.loc[
        sue['Annodt'].between(
                pd.Timestamp(START_DATE),
                pd.Timestamp(END_DATE),
                inclusive='both'
            )
    ].copy()

    sue = select_eps_mode(
        data=sue,
        sue_mode=SUE_MODE
    )

    # -------------------------------------------------------------------------
    # 9. 分别匹配财报截止日股价和公告日前股价
    # -------------------------------------------------------------------------
    price = load_price_data(
        file_path=PRICE_PATH
    )

    sue = attach_report_period_price(
        data=sue,
        price_data=price,
        max_shift_days=MAX_PRICE_SHIFT_DAYS
    )

    sue = attach_sue3_pre_announcement_price(
        data=sue,
        price_data=price,
        total_share=total_share,
        trading_calendar=trading_calendar,
        lag_trading_days=SUE3_PRICE_LAG_TRADING_DAYS
    )

    del total_share
    del price
    del trading_calendar

    # -------------------------------------------------------------------------
    # 10. 计算并保存全部SUE指标
    # -------------------------------------------------------------------------
    sue = calculate_sue_versions(
        data=sue,
        sue_mode=SUE_MODE,
        forecast_windows=FORECAST_WINDOWS,
        sue3_primary_window=SUE3_PRIMARY_WINDOW
    )

    # 只保留最终输出需要的变量
    output_columns = [
        'Stkcd',
        'fqtr',
        'Accper',
        'Annodt',
        'forecast_count_180',
        'forecast_count_365',
        'sue1_cum',
        'sue1_qtr',
        'sue2_cum',
        'sue2_qtr',
        'sue3_180',
        'sue3_365',
        'sue1',
        'sue2',
        'sue3'
    ]

    sue_output = sue.loc[:, output_columns].copy()

    save_sue_results(
        data=sue_output,
        output_path=OUTPUT_PATH
    )

    return sue_output


if __name__ == '__main__':
    main()

# 本次运行结果（全部财报记录：281,323条）
#
# sue3_180非缺失：32,031条
# - 年报：32,002条
# - 中报：27条
# - 三季报：2条
# - 一季报：0条
#
# sue3_365非缺失：39,223条
# - 年报：39,192条
# - 中报：29条
# - 三季报：2条
# - 一季报：0条
#
# 当前sue3主变量指向sue3_365。