"""官方年度标签图2：精确季度双固定效应、固定缩尾、两种提示并列核验。

本脚本恢复自历史计算记录，仅调整了缓存和结果路径；本次未重新运行。
"""
import os
import sys
sys.dont_write_bytecode = True
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

from pathlib import Path
import argparse
import importlib.util
import json
import time
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/Volumes/苹果资料/mmm复现数据包')
GRID = ROOT / 'work/主回归口径组合试算_20261026'
CACHE = ROOT / 'work/国泰安标签收益检验/图2'
OUT = ROOT / '图表结果/04_国泰安标签收益检验/图2_收益动态/重新计算'
OUT.mkdir(parents=True, exist_ok=True)
RAW_TYPE = np.dtype([('firm', 'i4'), ('date', 'i4'), ('port', 'i2'), ('raw', 'f4', (50,))])
SPEC = importlib.util.spec_from_file_location('exact_projection', GRID / 'diagnostics_20261027/02_estimate_diagnostics.py')
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)
CUTS = np.load(ROOT / 'work/图2_允许填充起始日_临时/原样本固定口径.npz')


def blocks(horizons, official):
    if official:
        benchmark = np.load(CACHE / 'official_benchmark.npz')
        sumw, sumwr = benchmark['sumw'], benchmark['sumwr']
        start = int(benchmark['start'])
    for path in sorted((GRID / 'enriched').glob('q*.npy')):
        block = np.load(path, mmap_mode='r')
        dates = block['date']
        # 每个块本来就是单季度，块内公司效应等价于公司×年度季度效应。
        quarter = (dates.astype('datetime64[D]').astype('datetime64[M]').astype('int64') + 1970 * 12 - 2005 * 12) // 3
        if not np.all(quarter == int(path.stem[1:])):
            raise ValueError('季度缓存编号与实际日期不符。')
        old = np.load(GRID / 'figure2_selected_20261027/returns' / path.name, mmap_mode='r')
        news = np.where((dates >= np.datetime64('2010-01-01').astype('int32'))[:, None], block['news'][:, 8:12], block['news'][:, 4:8])
        controls = np.column_stack([block['x'], news]).astype('float64')
        common = np.isfinite(controls).all(axis=1) & np.isfinite(old[:, [0, 2, 4]]).all(axis=1)
        cues = block['cues'][:, [0, 8]].astype('float64')
        if not official:
            y = old[:, np.array(horizons) - 1].astype('float64')
        else:
            rec = np.fromfile(CACHE / 'raw_target' / (path.stem + '.bin'), dtype=RAW_TYPE)
            rk = rec['date'].astype('int64') * 1000000 + rec['firm']
            target = dates.astype('int64') * 1000000 + block['firm']
            if len(np.unique(rk)) != len(rk):
                raise ValueError('原始目标日收益键重复。')
            ix = pd.Index(rk).get_indexer(target)
            ok = ix >= 0
            y = np.full((len(block), len(horizons)), np.nan, dtype='float64')
            rr = rec[ix[ok]]
            gid = (rr['date'].astype('int64') - start) * 126 + rr['port'].astype('int64')
            hi = np.array(horizons) - 1
            w = sumw[gid][:, hi]
            wr = sumwr[gid][:, hi]
            bench = np.divide(wr, w, out=np.full_like(wr, np.nan), where=w > 0)
            # 包括填充日起点，直接使用宽日度面板原始收益，禁止旧收益反推覆盖。
            y[ok] = rr['raw'][:, hi].astype('float64') - bench
            y = np.clip(y, CUTS['lower'][hi], CUTS['upper'][hi])
        groups = {}
        for h in range(len(horizons)):
            pos = np.flatnonzero(common & np.isfinite(y[:, h]))
            while len(pos):
                _, f = np.unique(block['firm'][pos], return_inverse=True)
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
            v, fs, ds, rank = CORE.project(values, block['firm'][pos], dates[pos])
            yield path.stem, block, pos, hs, v, fs, ds, rank


def estimate(horizons, official):
    started = time.monotonic()
    nh = len(horizons)
    cc = np.zeros((nh, 27, 27)); cd = np.zeros((nh, 27, 2)); cy = np.zeros((nh, 27))
    dd = np.zeros((nh, 2)); dy = np.zeros((nh, 2))
    nobs = np.zeros(nh, dtype='int64'); ranks = np.zeros(nh, dtype='int64')
    positive = np.zeros((nh, 2), dtype='int64'); filled = np.zeros(nh, dtype='int64')
    firms = np.load(GRID / 'enriched/global_firms.npy')
    lookup = np.full(int(firms.max()) + 1, -1, dtype='int32'); lookup[firms] = np.arange(len(firms))
    ndmax = int(np.datetime64('2025-12-31').astype('int32')) + 1
    fseen = np.zeros((len(firms), nh), dtype=bool); dseen = np.zeros((ndmax, nh), dtype=bool)
    for q, b, pos, hs, v, fs, ds, rank in blocks(horizons, official):
        c, cue, y = v[:, :27], v[:, 27:29], v[:, 29:]
        cc[hs] += c.T @ c; cd[hs] += c.T @ cue; cy[hs] += (c.T @ y).T
        dd[hs] += np.sum(cue * cue, axis=0); dy[hs] += (cue.T @ y).T
        nobs[hs] += len(pos); ranks[hs] += rank
        positive[hs] += b['cues'][pos][:, [0, 8]].sum(axis=0).astype('int64')
        filled[hs] += int(b['filled'][pos].sum())
        fseen[np.ix_(lookup[np.unique(b['firm'][pos])], hs)] = True
        dseen[np.ix_(np.unique(b['date'][pos]), hs)] = True
        if int(q[1:]) % 20 == 0 and hs[0] == 0:
            print(f'系数累计至 {q}，{time.monotonic() - started:.1f}秒', flush=True)
    gd = np.linalg.solve(cc, cd)
    gy = np.linalg.solve(cc, cy[:, :, None])[:, :, 0]
    denominator = dd - np.sum(cd * gd, axis=1)
    beta = (dy - np.sum(cd * gy[:, :, None], axis=1)) / denominator
    control_beta = gy[:, :, None] - gd * beta[:, None, :]
    fscores = np.zeros((len(firms), nh, 2)); dscores = np.zeros((ndmax, nh, 2)); inter = np.zeros((nh, 2))
    for q, b, pos, hs, v, fs, ds, rank in blocks(horizons, official):
        c, cue, y = v[:, :27], v[:, 27:29], v[:, 29:]
        # 两种提示分别回归，互相不作为控制变量。
        error = y[:, :, None] - np.einsum('np,hpc->nhc', c, control_beta[hs]) - cue[:, None, :] * beta[hs][None, :, :]
        residual_cue = cue[:, None, :] - np.einsum('np,hpc->nhc', c, gd[hs])
        score = residual_cue / denominator[hs][None, :, :] * error
        fscores[np.ix_(lookup[np.unique(b['firm'][pos])], hs, np.arange(2))] += (fs @ score.reshape(len(pos), -1)).reshape(-1, len(hs), 2)
        dscores[np.ix_(np.unique(b['date'][pos]), hs, np.arange(2))] += (ds @ score.reshape(len(pos), -1)).reshape(-1, len(hs), 2)
        inter[hs] += np.sum(score * score, axis=0)
        if int(q[1:]) % 20 == 0 and hs[0] == 0:
            print(f'聚类累计至 {q}，{time.monotonic() - started:.1f}秒', flush=True)
    nf, nd = fseen.sum(axis=0), dseen.sum(axis=0)
    variance = ((nobs - 1) / (nobs - 29))[:, None] * (nf[:, None] / (nf[:, None] - 1) * np.sum(fscores * fscores, axis=0) + nd[:, None] / (nd[:, None] - 1) * np.sum(dscores * dscores, axis=0) - nobs[:, None] / (nobs[:, None] - 1) * inter)
    if (variance <= 0).any() or not np.isfinite(variance).all():
        raise ValueError('聚类方差异常。')
    se = np.sqrt(variance)
    df = nobs - ranks - 28
    probability = 2 * stats.t.sf(np.abs(beta / se), df[:, None])
    rows = []
    for h, horizon in enumerate(horizons):
        for j, cue in enumerate(['市值90分位提示', '沪深300提示']):
            row = {'提示口径': cue, '窗口': horizon, '系数_百分点': beta[h, j] * 100, '标准误_百分点': se[h, j] * 100, '显著性概率': probability[h, j], '样本量': int(nobs[h]), '残差自由度': int(df[h]), '正提示日': int(positive[h, j]), '填充起始日': int(filled[h]), '公司聚类数': int(nf[h]), '日期聚类数': int(nd[h])}
            for level in [.90, .95, .99]:
                critical = stats.t.ppf((1 + level) / 2, df[h])
                row[f'{level:.0%}下界'] = (beta[h, j] - critical * se[h, j]) * 100
                row[f'{level:.0%}上界'] = (beta[h, j] + critical * se[h, j]) * 100
            rows.append(row)
    result = pd.DataFrame(rows)
    print(f'完成估计，用时 {time.monotonic() - started:.1f}秒', flush=True)
    return result


def draw(table):
    os.environ['MPLCONFIGDIR'] = str(CACHE / 'mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = '/System/Library/Fonts/Hiragino Sans GB.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(), 'axes.unicode_minus': False, 'font.size': 11})
    result = table.loc[table['提示口径'].eq('市值90分位提示')].copy()
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.subplots_adjust(left=.105, right=.975, bottom=.23, top=.76)
    fig.text(.105, .935, '图2  市值90分位提示后的累计特征调整收益', fontsize=19, color='#202932')
    fig.text(.105, .885, '国泰安官方组合标签；保留合格停牌及暂停上市填充日', fontsize=12, color='#5c6773')
    h = result['窗口'].to_numpy(); beta = result['系数_百分点'].to_numpy()
    for level, color, width in [(99, '#c5ccd2', .8), (95, '#8199ad', 1.6), (90, '#245c7c', 2.8)]:
        ax.vlines(h, result[f'{level}%下界'], result[f'{level}%上界'], color=color, linewidth=width, label=f'{level}%置信区间', zorder=2)
    significant = result['显著性概率'].to_numpy() < .1
    ax.scatter(h[~significant], beta[~significant], color='#20252b', s=23, zorder=4)
    ax.scatter(h[significant], beta[significant], color='#c83e42', s=29, zorder=5, label='红点：10%水平显著')
    ax.axhline(0, color='#68727b', linewidth=.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e4e8eb', linewidth=.6); ax.set_axisbelow(True)
    ax.set_xlim(0, 51); ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlabel('累计窗口终点（交易日；窗口为 [t,t+h]）', labelpad=10)
    ax.set_ylabel('提示变量系数（百分点）', labelpad=10)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.03), ncol=4, frameon=False, fontsize=10)
    note = (f'2005年4月8日—2025年12月31日；公司×年度季度、交易日期固定效应；公司和日期双向聚类。\n'
            f'27项控制，数行者分段新闻；沿用原50窗缩尾阈值；日度组合基准使用滞后3日流通市值权重。\n'
            f'逐窗口样本量：{result["样本量"].min():,}—{result["样本量"].max():,}；置信区间为逐点区间。')
    fig.text(.105, .135, note, fontsize=9.5, color='#5c6773', linespacing=1.7, va='top')
    fig.savefig(OUT / '图2_官方标签_市值90分位提示_含停牌填充日.png', dpi=220, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--anchor', action='store_true'); args = parser.parse_args()
    if args.anchor:
        result = estimate([1, 10, 50], False)
        result.to_csv(OUT / '原图2估计器核验.csv', index=False, encoding='utf-8-sig')
        # 正式目录的逐窗进度表当前只写至第38窗；使用已有完整50窗清单。
        reference = pd.read_csv(GRID / 'figure2_selected_20261027/数行者两段新闻/图2_选定规格_完整50窗口.csv').set_index('窗口')
        rows = result.loc[result['提示口径'].eq('沪深300提示')]
        for _, row in rows.iterrows():
            ref = reference.loc[row['窗口']]
            assert row['样本量'] == ref['样本量']
            assert np.isclose(row['系数_百分点'], ref['系数_百分点'], rtol=1e-7, atol=1e-9)
            assert np.isclose(row['标准误_百分点'], ref['标准误_百分点'], rtol=1e-7, atol=1e-9)
        print('原图2第1、10、50窗口的系数、标准误、样本量全部复现。', flush=True)
        result.to_csv(OUT / '原图2估计器核验.csv', index=False, encoding='utf-8-sig')
        return
    result = estimate(list(range(1, 51)), True)
    result.to_csv(OUT / '官方标签_两种提示_精确估计结果.csv', index=False, encoding='utf-8-sig')
    selected = result.loc[result['提示口径'].eq('市值90分位提示')]
    selected.to_csv(OUT / '图2_官方标签_市值90分位提示_结果.csv', index=False, encoding='utf-8-sig')
    draw(result)
    audit = {'估计器': '季度舒尔补精确双固定效应；两种提示分别回归；双向聚类', '缩尾': '原样本固定口径.npz：lower/upper，50窗口', '原始收益': '直接提取宽日度面板，包含填充日起点', '结果': result.loc[result['窗口'].isin([1, 10, 20, 30, 50])].to_dict('records')}
    (OUT / '绘图核验.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(result.loc[result['窗口'].isin([1, 10, 20, 30, 50])].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
