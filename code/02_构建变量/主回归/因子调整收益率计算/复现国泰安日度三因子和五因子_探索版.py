"""复用同目录普通版的计算流程，在终端比较不同因子口径的分期误差。"""

from dataclasses import replace

import pandas as pd

from 复现国泰安日度三因子和五因子_普通版 import (
    ORDINARY,
    build_argument_parser,
    calculate_factors,
    comparison_metrics,
    load_official,
    validate_arguments,
)


# 同一套行情、财报和组合计算流程，仅替换需要比较的口径。
SPECIFICATIONS = {
    ORDINARY.name: ORDINARY,
    '上一轮基准': replace(
        ORDINARY,
        name='上一轮基准',
        split_b_share_prices=False,
        historical_main_board=False,
        december_last_trade=False,
        profit_equity='positive',
        ipo_from_offer_price=False,
    ),
    # 年末停牌填充且距最近实际成交至少 60 个开市日，才修正十二月市值。
    '十二月长停牌': replace(ORDINARY, name='十二月长停牌', december_min_gap=60),
    # 普通版的计算函数会核对六月停牌时长及重大股本变动的披露日和生效日。
    '六月重大股本变动': replace(
        ORDINARY, name='六月重大股本变动', june_restructure_cap=True
    ),
    '十二月长停牌加六月股本变动': replace(
        ORDINARY,
        name='十二月长停牌加六月股本变动',
        december_min_gap=60,
        june_restructure_cap=True,
    ),
    '全期保留权益符号': replace(ORDINARY, name='全期保留权益符号', profit_equity='signed'),
}
DEFAULT_SPECS = (
    '上一轮基准',
    '十二月长停牌',
    '十二月长停牌加六月股本变动',
    '全期保留权益符号',
)

PERIODS = (
    ('2000—2025 年主样本', pd.Timestamp('2000-01-01'), pd.Timestamp('2025-12-31')),
    ('2023 年下半年至 2025 年', pd.Timestamp('2023-07-01'), pd.Timestamp('2025-12-31')),
    ('2026 年六月换组后', pd.Timestamp('2026-07-01'), pd.Timestamp('2026-08-19')),
)


def print_comparison(title, start, end, results, official):
    """按模型、因子和加权方式并列展示误差、相关系数和实际配对日数。"""
    columns = []
    for name, factors in results.items():
        metrics = comparison_metrics(factors, official, start, end)
        if metrics.empty:
            continue
        metrics = metrics.set_index(['模型', '因子', '加权'])
        cells = [
            f'{error:.3f}／{corr:.6f}（{days:,}）' if pd.notna(error) else '—'
            for error, corr, days in zip(
                metrics['均方根误差（基点）'], metrics['相关系数'], metrics['交易日数'],
            )
        ]
        columns.append(pd.Series(cells, index=metrics.index, name=name))

    print(f'\n{title}  {start.date()} 至 {end.date()}')
    if columns:
        comparison = pd.concat(columns, axis=1).reset_index()
        with pd.option_context('display.unicode.east_asian_width', True):
            print(comparison.to_string(index=False, na_rep='—'))
    else:
        print('该区间没有可对照的交易日。')


def main():
    parser = build_argument_parser(__doc__)
    parser.add_argument(
        '--spec',
        nargs='+',
        choices=tuple(SPECIFICATIONS),
        default=DEFAULT_SPECS,
        help='指定一个或多个口径。始终保留普通版对照，省略时使用默认候选。',
    )
    args = parser.parse_args()
    validate_arguments(parser, args)

    selected_names = list(dict.fromkeys((ORDINARY.name, *args.spec)))
    specifications = [SPECIFICATIONS[name] for name in selected_names]
    print(f'比较口径  {"、".join(selected_names)}', flush=True)
    official = load_official(args.input_dir)
    results = calculate_factors(args.input_dir, args.rate_file, args.start, args.end, specifications)

    intervals = []
    for title, start, end in PERIODS:
        start = max(start, args.start)
        end = min(end, args.end)
        if start <= end:
            intervals.append((title, start, end))

    requested = args.start, args.end
    default_range = PERIODS[0][1], PERIODS[-1][2]
    if requested != default_range and all((start, end) != requested for _, start, end in intervals):
        intervals.insert(0, ('所选区间', *requested))

    print('\n每格依次为均方根误差（基点）／相关系数（有效配对交易日数）。')
    for title, start, end in intervals:
        print_comparison(title, start, end, results, official)


if __name__ == '__main__':
    main()
