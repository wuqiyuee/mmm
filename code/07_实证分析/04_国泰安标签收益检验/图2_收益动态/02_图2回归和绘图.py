"""图2第二步：使用国泰安标签调整收益，按窗口回归并绘图。

先运行同目录的收益准备脚本，生成正式位置的国泰安标签调整收益表。
本脚本直接读取沪深300最终回归面板及上述收益表，不读取临时试算数据。
每个窗口使用27项控制、被提示公司×年度季度和日期固定效应，
按公司和日期双向聚类，提示变量统一使用沪深300提示。
国泰安调整收益使用回归筛样前全样本、各窗口自身的1%和99%分位点缩尾。
回归样本直接来自正式最终面板，未额外加入历史试算中的填充日起点；
不同窗口分别筛选收益有效值，不额外设置市值1亿元或股价1元门槛。
"""
import os
import sys
sys.dont_write_bytecode = True
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[4]
PANEL = ROOT / '中间结果/04_合并数据/Pattern样本_最终回归面板_沪深300.csv'
RETURNS = ROOT / '中间结果/02_变量构造结果/1DGTW特征调整收益率/国泰安标签_沪深300_50窗调整收益.csv'
OUT = ROOT / '图表结果/04_国泰安标签收益检验/图2_收益动态/重新计算'
CONTROLS = ([f'ownEA_tm{i}' for i in range(10, 0, -1)] + ['ownEA']
            + [f'ownEA_tp{i}' for i in range(1, 11)]
            + ['ownOtherEventER', 'ownPerformanceForecastBrief',
               'aesIncrease', 'aesDecrease', 'logIncreaseAEV', 'logDecreaseAEV'])
OLD_RETURNS = [f'chAdjreturn_t_tp{i}_Recall_w' for i in (1, 3, 5)]
OFFICIAL_RETURNS = [f'chAdjOfficial_t_tp{i}_w' for i in range(1, 51)]


def absorb_fixed_effects(values, firm, date):
    """同一季度内吸收公司与日期固定效应，返回残差及聚类所需矩阵。"""
    _, firm_id = np.unique(firm, return_inverse=True)
    _, date_id = np.unique(date, return_inverse=True)
    firm_count, date_count = firm_id.max() + 1, date_id.max() + 1
    rows = np.arange(len(firm_id))
    firm_matrix = sparse.csr_matrix((np.ones(len(firm_id)), (firm_id, rows)),
                                    shape=(firm_count, len(firm_id)))
    date_matrix = sparse.csr_matrix((np.ones(len(firm_id)), (date_id, rows)),
                                    shape=(date_count, len(firm_id)))
    counts = sparse.csr_matrix((np.ones(len(firm_id)), (firm_id, date_id)),
                               shape=(firm_count, date_count))
    firm_sizes = np.bincount(firm_id)
    components, labels = connected_components((counts.T @ counts).tocsr(), directed=False)
    reference = np.unique(labels, return_index=True)[1]
    free_dates = np.setdiff1d(np.arange(date_count), reference)

    # 先减去公司均值，再求日期效应；每个相连部分保留一个参照日期。
    schur = np.diag(np.bincount(date_id)) - (counts.T @ counts.multiply(1 / firm_sizes[:, None])).toarray()
    schur = (schur + schur.T) / 2
    residual = values.copy()
    residual -= ((firm_matrix @ residual) / firm_sizes[:, None])[firm_id]
    date_effect = np.zeros((date_count, residual.shape[1]))
    date_effect[free_dates] = cho_solve(
        cho_factor(schur[np.ix_(free_dates, free_dates)], lower=True, check_finite=False),
        (date_matrix @ residual)[free_dates], check_finite=False)
    residual -= date_effect[date_id]
    residual += ((counts @ date_effect) / firm_sizes[:, None])[firm_id]
    return residual, firm_matrix, date_matrix, firm_count + date_count - components


def blocks(data, horizons):
    """逐季度构造各窗口样本，并吸收公司季度与日期固定效应。"""
    for quarter, part in data.groupby('季度', sort=True):
        firm = part['StkcdRecall'].to_numpy(dtype='int32')
        dates = part['Trddt'].to_numpy(dtype='datetime64[D]').astype('int32')
        controls = part[CONTROLS].to_numpy(dtype='float64')
        old = part[OLD_RETURNS].to_numpy(dtype='float64')
        cues = part[['quarterLag1HS300']].to_numpy(dtype='float64')
        y = part[[OFFICIAL_RETURNS[h - 1] for h in horizons]].to_numpy(dtype='float64')
        common = np.isfinite(controls).all(axis=1) & np.isfinite(old).all(axis=1)
        common &= np.isfinite(cues[:, 0])
        groups = {}
        for h in range(len(horizons)):
            pos = np.flatnonzero(common & np.isfinite(y[:, h]))
            while len(pos):
                _, f = np.unique(firm[pos], return_inverse=True)
                _, d = np.unique(dates[pos], return_inverse=True)
                keep = (np.bincount(f)[f] > 1) & (np.bincount(d)[d] > 1)
                if keep.all():
                    break
                pos = pos[keep]
            if len(pos):
                key = pos.tobytes()
                if key not in groups:
                    groups[key] = [pos, []]
                groups[key][1].append(h)
        for pos, hs in groups.values():
            hs = np.asarray(hs)
            values = np.column_stack([controls[pos], cues[pos], y[pos][:, hs]])
            v, fs, ds, rank = absorb_fixed_effects(values, firm[pos], dates[pos])
            yield quarter, firm, dates, cues, pos, hs, v, fs, ds, rank


def estimate(data, horizons):
    started = time.monotonic()
    nh = len(horizons)
    # 一、逐季度累计回归所需的矩阵，提示变量采用沪深300口径。
    cc = np.zeros((nh, 27, 27))
    cd = np.zeros((nh, 27, 1))
    cy = np.zeros((nh, 27))
    dd = np.zeros((nh, 1))
    dy = np.zeros((nh, 1))
    nobs = np.zeros(nh, dtype='int64')
    ranks = np.zeros(nh, dtype='int64')
    positive = np.zeros((nh, 1), dtype='int64')
    firms = np.sort(data['StkcdRecall'].unique().astype('int32'))
    lookup = np.full(int(firms.max()) + 1, -1, dtype='int32')
    lookup[firms] = np.arange(len(firms))
    ndmax = int(np.datetime64('2025-12-31').astype('int32')) + 1
    fseen = np.zeros((len(firms), nh), dtype=bool)
    dseen = np.zeros((ndmax, nh), dtype=bool)
    for q, firm, dates, cues, pos, hs, v, fs, ds, rank in blocks(data, horizons):
        c, cue, y = v[:, :27], v[:, 27:28], v[:, 28:]
        cc[hs] += c.T @ c
        cd[hs] += c.T @ cue
        cy[hs] += (c.T @ y).T
        dd[hs] += np.sum(cue * cue, axis=0)
        dy[hs] += (cue.T @ y).T
        nobs[hs] += len(pos)
        ranks[hs] += rank
        positive[hs] += cues[pos].sum(axis=0).astype('int64')
        fseen[np.ix_(lookup[np.unique(firm[pos])], hs)] = True
        dseen[np.ix_(np.unique(dates[pos]), hs)] = True
    gd = np.linalg.solve(cc, cd)
    gy = np.linalg.solve(cc, cy[:, :, None])[:, :, 0]
    denominator = dd - np.sum(cd * gd, axis=1)
    beta = (dy - np.sum(cd * gy[:, :, None], axis=1)) / denominator
    control_beta = gy[:, :, None] - gd * beta[:, None, :]

    # 二、重读相同样本，分别累计公司和日期得分，计算双向聚类标准误。
    fscores = np.zeros((len(firms), nh, 1))
    dscores = np.zeros((ndmax, nh, 1))
    inter = np.zeros((nh, 1))
    for q, firm, dates, cues, pos, hs, v, fs, ds, rank in blocks(data, horizons):
        c, cue, y = v[:, :27], v[:, 27:28], v[:, 28:]
        error = y[:, :, None] - np.einsum('np,hpc->nhc', c, control_beta[hs]) - cue[:, None, :] * beta[hs][None, :, :]
        residual_cue = cue[:, None, :] - np.einsum('np,hpc->nhc', c, gd[hs])
        score = residual_cue / denominator[hs][None, :, :] * error
        fscores[np.ix_(lookup[np.unique(firm[pos])], hs, np.arange(1))] += (fs @ score.reshape(len(pos), -1)).reshape(-1, len(hs), 1)
        dscores[np.ix_(np.unique(dates[pos]), hs, np.arange(1))] += (ds @ score.reshape(len(pos), -1)).reshape(-1, len(hs), 1)
        inter[hs] += np.sum(score * score, axis=0)
    nf, nd = fseen.sum(axis=0), dseen.sum(axis=0)
    variance = ((nobs - 1) / (nobs - 29))[:, None] * (nf[:, None] / (nf[:, None] - 1) * np.sum(fscores * fscores, axis=0) + nd[:, None] / (nd[:, None] - 1) * np.sum(dscores * dscores, axis=0) - nobs[:, None] / (nobs[:, None] - 1) * inter)
    if (variance <= 0).any() or not np.isfinite(variance).all():
        raise ValueError('聚类方差异常。')
    se = np.sqrt(variance)
    df = nobs - ranks - 28
    probability = 2 * stats.t.sf(np.abs(beta / se), df[:, None])

    # 三、整理每个窗口的系数、标准误和逐点置信区间。
    rows = []
    for h, horizon in enumerate(horizons):
        row = {'提示口径': '沪深300提示', '窗口': horizon,
               '系数_百分点': beta[h, 0] * 100, '标准误_百分点': se[h, 0] * 100,
               '显著性概率': probability[h, 0], '样本量': int(nobs[h]),
               '残差自由度': int(df[h]), '正提示日': int(positive[h, 0]),
               '公司聚类数': int(nf[h]), '日期聚类数': int(nd[h])}
        for level in [.90, .95, .99]:
            critical = stats.t.ppf((1 + level) / 2, df[h])
            row[f'{level:.0%}下界'] = (beta[h, 0] - critical * se[h, 0]) * 100
            row[f'{level:.0%}上界'] = (beta[h, 0] + critical * se[h, 0]) * 100
        rows.append(row)
    result = pd.DataFrame(rows)
    print(f'完成估计，用时 {time.monotonic() - started:.1f}秒', flush=True)
    return result


def draw(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = '/System/Library/Fonts/Hiragino Sans GB.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(), 'axes.unicode_minus': False, 'font.size': 11})
    result = table.loc[table['提示口径'].eq('沪深300提示')].copy()
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.subplots_adjust(left=.105, right=.975, bottom=.23, top=.76)
    fig.text(.105, .935, '图2  沪深300提示后的累计特征调整收益', fontsize=19, color='#202932')
    fig.text(.105, .885, '国泰安官方组合标签；沪深300最终回归面板', fontsize=12, color='#5c6773')
    h = result['窗口'].to_numpy()
    beta = result['系数_百分点'].to_numpy()
    for level, color, width in [(99, '#c5ccd2', .8), (95, '#8199ad', 1.6), (90, '#245c7c', 2.8)]:
        ax.vlines(h, result[f'{level}%下界'], result[f'{level}%上界'], color=color, linewidth=width, label=f'{level}%置信区间', zorder=2)
    significant = result['显著性概率'].to_numpy() < .1
    ax.scatter(h[~significant], beta[~significant], color='#20252b', s=23, zorder=4)
    ax.scatter(h[significant], beta[significant], color='#c83e42', s=29, zorder=5, label='红点：10%水平显著')
    ax.axhline(0, color='#68727b', linewidth=.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e4e8eb', linewidth=.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 51)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlabel('累计窗口终点（交易日；窗口为 [t,t+h]）', labelpad=10)
    ax.set_ylabel('提示变量系数（百分点）', labelpad=10)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.03), ncol=4, frameon=False, fontsize=10)
    note = (f'2005年4月8日—2025年12月31日；公司×年度季度、交易日期固定效应；公司和日期双向聚类。\n'
            f'27项控制，数行者分段新闻；国泰安收益按筛样前全样本逐窗1%和99%缩尾；基准使用滞后3日流通市值权重。\n'
            f'逐窗口样本量：{result["样本量"].min():,}—{result["样本量"].max():,}；置信区间为逐点区间。')
    fig.text(.105, .135, note, fontsize=9.5, color='#5c6773', linespacing=1.7, va='top')
    fig.savefig(OUT / '图2_国泰安标签_沪深300提示_正式面板.png', dpi=220, facecolor='white')
    plt.close(fig)


def main():
    if not RETURNS.is_file():
        raise FileNotFoundError(f'缺少国泰安调整收益表，请先运行同目录的收益准备脚本：{RETURNS}')
    panel_columns = ['StkcdRecall', 'Trddt', 'quarterLag1HS300', *CONTROLS, *OLD_RETURNS]
    types = {'StkcdRecall': 'int32', **{c: 'float32' for c in CONTROLS + OLD_RETURNS + ['quarterLag1HS300']}}
    data = pd.read_csv(PANEL, usecols=panel_columns, dtype=types, parse_dates=['Trddt'])
    data = data.loc[data['Trddt'].between('2005-04-08', '2025-12-31')].copy()
    returns = pd.read_csv(RETURNS, usecols=['StkcdRecall', 'Trddt', *OFFICIAL_RETURNS],
                          dtype={'StkcdRecall': 'int32'}, parse_dates=['Trddt'])
    data = data.merge(returns, on=['StkcdRecall', 'Trddt'], how='left', validate='one_to_one', indicator=True)
    if not data['_merge'].eq('both').all():
        raise ValueError('国泰安收益表未覆盖全部面板公司日期，可能尚未完整写出，请核对。')
    data = data.drop(columns='_merge')
    del returns
    if not data['quarterLag1HS300'].isin([0, 1]).all():
        raise ValueError('沪深300提示变量包含缺失或非零一数值。')
    data['季度'] = data['Trddt'].dt.to_period('Q')
    OUT.mkdir(parents=True, exist_ok=True)
    result = estimate(data, list(range(1, 51)))
    result.to_csv(OUT / '图2_官方标签_沪深300提示_结果.csv', index=False, encoding='utf-8-sig')
    draw(result)
    print(result.loc[result['窗口'].isin([1, 10, 20, 30, 50])].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
