"""估计111提示当日至后50个交易日的单日特征调整收益系数。

研究期为2005—2025年，提示公司市值门槛为90分位；每个相对日使用
该日收益可用的观测，控制27项变量，吸收被提示公司×年度季度和事件
起点日期固定效应，按被提示公司和事件起点日期双向聚类。

正式输入：
1. 中间结果/04_合并数据/Pattern样本_单日收益回归基础面板.csv；
2. 中间结果/04_合并数据/CueEvents_111.csv；
3. 中间结果/02_变量构造结果/1DGTW特征调整收益率/
   DGTW特征调整收益率.csv。

复用“单日特征调整收益”目录下的日期固定效应脚本及公共估计器。
结果保存至图表结果/02_扩展分析/其他扩展检验/提示事件111_110比较/
111单日特征调整收益51日/，包括系数明细、系数图和运行口径核验记录。
结果已存在时停止执行，防止覆盖已完成结果。

归档前的一次性控制变量核验已完成，记录保存在现有核验结果中。
正式执行不再依赖当时使用的临时主回归快照。
"""
from __future__ import annotations

import os
from pathlib import Path

def find_project_root() -> Path:
    """从脚本位置向上寻找项目根目录，避免依赖机器上的绝对路径。"""
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / 'AGENTS.md').is_file()
            and (candidate / 'code').is_dir()
            and (candidate / '中间结果').is_dir()
        ):
            return candidate
    raise FileNotFoundError('未找到同时包含项目说明、代码和中间结果的项目根目录。')


ROOT = find_project_root()
WORK = ROOT / 'work/111单日特征调整收益51日'
os.environ['MPLCONFIGDIR'] = str(WORK / 'matplotlib')
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'

import gc
import importlib.util
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

START = time.monotonic()
SCRIPT = ROOT / 'code/07_实证分析/02_扩展分析/单日特征调整收益/图2扩展_单日特征调整收益_加入日期固定效应.py'
EVENTS = ROOT / '中间结果/04_合并数据/CueEvents_111.csv'
OUT = ROOT / '图表结果/02_扩展分析/其他扩展检验/提示事件111_110比较/111单日特征调整收益51日'
OUT.mkdir(parents=True, exist_ok=True)
RESULT = OUT / '111_单日特征调整收益_双固定效应_51日系数.csv'
FIGURE = OUT / '111_单日特征调整收益_双固定效应_51日系数图.png'
AUDIT = OUT / '111_单日特征调整收益_运行口径与核验.json'
if RESULT.exists() or FIGURE.exists():
    raise FileExistsError(
        '111单日收益结果已存在；如需重新估计，请先另存既有结果或更改本脚本的输出目录。'
    )

def log(message):
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    print(f'[{now:%H:%M:%S}] {message}', flush=True)

def describe(path):
    st = path.stat()
    return {'path': str(path), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}

def save_audit(audit):
    temporary = AUDIT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(AUDIT)

spec = importlib.util.spec_from_file_location('_daily111_existing_source', SCRIPT)
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)

audit = {
    'started': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'sample_years': [2005, 2025], 'relative_days': [0, 50],
    'outcome': '目标日实际所属DGTW组合调整后的单日收益；非累计收益、非累计收益差分',
    'cue': '111：当前提示端、历史提示端、历史被提示端的财报季度模式标记均为1',
    'market_cap_threshold': '提示公司事件前三个交易日总市值严格高于事件上月90分位',
    'fixed_effects': ['被提示公司×事件起点年度季度', '事件起点交易日期'],
    'clustering': ['被提示公司', '事件起点交易日期'],
    'sample_mode': '逐日因变量可用样本，递归删除固定效应单例',
    'inference': '沿用项目公共估计器的双向聚类有限样本修正；图为逐点置信区间',
    'source_files': {
        'neutral_base': describe(ds.NEUTRAL_BASE_PATH), 'daily_returns': describe(ds.DAILY_ADJUSTED_RETURN_PATH),
        '111_events': describe(EVENTS), 'data_functions': describe(SCRIPT),
        'analysis_script': describe(Path(__file__).resolve()),
    },
}
log('读取中性基础面板及111事件')
base = ds.read_base_panel(ds.NEUTRAL_BASE_PATH)
if not base[ds.CUE_COL].isin([0, 1]).all():
    raise AssertionError('基础面板原提示变量不是完整0/1。')
events = pd.read_csv(EVENTS, dtype={'StkcdRecall': 'string'}, parse_dates=['date'], encoding='utf-8-sig')
events.StkcdRecall = ds.normalize_stock_code_series(events.StkcdRecall)
if events.duplicated(['StkcdRecall', 'date']).any():
    raise AssertionError('111事件公司日键不唯一。')
if not events[ds.CUE_COL].eq(1).all():
    raise AssertionError('111事件表含非正向提示记录。')
event_index = pd.MultiIndex.from_frame(events[['StkcdRecall', 'date']])
base_index = pd.MultiIndex.from_frame(base[['StkcdRecall', 'Trddt']])
cue111 = pd.Series(np.ones(len(events), dtype='float32'), index=event_index).reindex(base_index, fill_value=0).to_numpy()
old_cue = base[ds.CUE_COL].to_numpy(copy=True)
base[ds.CUE_COL] = cue111
audit['base'] = {
    'rows': len(base), 'firms': int(base.StkcdRecall.nunique()), 'dates': int(base.Trddt.nunique()),
    'first_date': str(base.Trddt.min().date()), 'last_date': str(base.Trddt.max().date()),
    'cue111': int(cue111.sum()), 'previous_cue': int(old_cue.sum()),
    'zero_to_one': int(((old_cue == 0) & (cue111 == 1)).sum()),
    'one_to_zero': int(((old_cue == 1) & (cue111 == 0)).sum()),
    'source_event_rows': len(events),
    'research_period_event_rows': int(events.date.dt.year.between(2005, 2025).sum()),
}
log(f'基础面板{len(base):,}行，111提示{int(cue111.sum()):,}个公司日')
del events, event_index, base_index, cue111, old_cue
gc.collect()

# 归档前已核对主回归共同公司日的27项控制变量；正式估计仅依赖上列正式输入。
save_audit(audit)

log('读取目标日单日特征调整收益并建立交易日匹配索引')
daily, lower, upper = ds.read_daily_return_lookup()
ds.validate_shared_td_calendar(base, daily)
audit['daily_returns'] = {'rows': len(daily), 'winsor_p01': lower, 'winsor_p99': upper, 'missing': int(daily[ds.OUTCOME_VALUE_COL].isna().sum())}
log(f'日度收益{len(daily):,}行，共同缩尾阈值为{lower:.10f}、{upper:.10f}')
lookup = ds.DailyOutcomeLookup(daily)
del daily
gc.collect()
save_audit(audit)

rows = []
sample_label = '111提示；逐日可用样本'
base_label = ds.BASE_SOURCE_LABELS['neutral']
for h in ds.HORIZONS:
    tick = time.monotonic()
    log(f'开始第{h + 1}/51个回归：事件后第{h}个交易日')
    aligned = ds.attach_daily_outcome(base, lookup, h)
    valid = aligned[ds.OUTCOME_VALUE_COL].notna()
    target_dates = aligned.loc[valid, ds.OUTCOME_DATE_COL]
    row = ds.run_date_fe_regression(
        aligned, horizon=h, outcome=ds.OUTCOME_VALUE_COL,
        outcome_label=ds.OUTCOME_LABELS[h], rhs_cols=ds.RHS_COLS, cue_col=ds.CUE_COL,
    )
    row.update({
        '分析版本': '111_单日特征调整收益_双固定效应_51日_v1',
        '提示口径': '111', '提示事件来源': str(EVENTS), '市值门槛': '90分位',
        '被解释变量口径': ds.OUTCOME_DEFINITION, '被解释变量来源': ds.OUTCOME_SOURCE_DEFINITION,
        '目标日组合口径': ds.TARGET_PORTFOLIO_DEFINITION, '目标日对齐口径': ds.TARGET_ALIGNMENT_DEFINITION,
        '目标日缺失处理': ds.TARGET_MISSING_DEFINITION,
        '样本口径': sample_label, '基础面板口径': base_label, '研究期': '2005—2025年',
        '缩尾口径': ds.WINSOR_DEFINITION, '缩尾下界': lower, '缩尾上界': upper,
        '目标日有效收益观测数': int(valid.sum()),
        '目标日日期起点': target_dates.min().strftime('%Y-%m-%d'),
        '目标日日期终点': target_dates.max().strftime('%Y-%m-%d'),
        '删除单例前111提示数': int(aligned.loc[valid, ds.CUE_COL].sum()),
        '单点耗时秒': round(time.monotonic() - tick, 2),
    })
    rows.append(row)
    results = ds.save_results(rows, RESULT)
    log(f"第{h}日完成：系数{row['线索系数']:.6f}，标准误{row['聚类标准误']:.6f}，概率值{row['p值']:.4f}，样本{row['回归样本量']:,}，耗时{row['单点耗时秒']:.1f}秒")
    del aligned, valid, target_dates
    gc.collect()

assert len(results) == 51 and results['窗口序号'].tolist() == list(range(51))
assert np.isfinite(results[['线索系数', '聚类标准误', 'p值']].to_numpy()).all()
assert (results['聚类标准误'] > 0).all()
assert results['日期固定效应'].eq('是').all()
assert results['被提示公司×年度季度固定效应'].eq('是').all()
assert results['结果单位'].eq('百分点').all()

font_path = Path('/System/Library/Fonts/Supplemental/Songti.ttc')
font = FontProperties(fname=str(font_path))
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
fig, ax = plt.subplots(figsize=(13.0, 6.8))
x = results['窗口序号'].to_numpy()
y = results['线索系数'].to_numpy()
se = results['聚类标准误'].to_numpy()
for critical, label, color, width, order in [
    (2.5758293035489004, '99%置信区间', '#c6cdd2', 1.0, 1),
    (1.959963984540054, '95%置信区间', '#78909c', 1.9, 2),
    (1.6448536269514722, '90%置信区间', '#214d6b', 3.3, 3),
]:
    ax.vlines(x, y - critical * se, y + critical * se, colors=color, linewidth=width, label=label, zorder=order)
sig = results['p值'].to_numpy() < 0.1
ax.scatter(x[~sig], y[~sig], color='#172b3a', s=22, zorder=5, label='系数估计')
ax.scatter(x[sig], y[sig], color='#b53030', s=27, zorder=6, label='10%水平显著')
ax.axhline(0, color='#555555', linewidth=0.9)
ax.set_xlim(-1, 51)
ax.set_xticks([0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
ax.set_xlabel('提示发生后的交易日（0为提示当日）', fontproperties=font, fontsize=12, labelpad=10)
ax.set_ylabel('111提示系数（百分点）', fontproperties=font, fontsize=12, labelpad=10)
ax.set_title('111提示后的单日特征调整收益\n2005—2025年｜90分位门槛｜公司季度与事件起点日期固定效应', fontproperties=font, fontsize=15, pad=18)
ax.legend(loc='upper right', frameon=False, prop=FontProperties(fname=str(font_path), size=10), ncol=2)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#e6e9ec', linewidth=0.6, zorder=0)
fig.text(0.09, 0.025, '注：每个点分别回归目标日的单日收益；逐日可用样本；完整控制变量；按公司和事件起点日期双向聚类。区间为逐点置信区间。', fontproperties=font, fontsize=10, color='#4e5963')
fig.tight_layout(rect=[0.02, 0.07, 0.995, 0.99])
fig.savefig(FIGURE, dpi=240, bbox_inches='tight', facecolor='white')
plt.close(fig)
audit['validation'] = {'coefficient_points': len(results), 'horizons': results['窗口序号'].tolist(), 'all_finite': True, 'positive_standard_errors': True}
audit['summary'] = {
    'n_min': int(results['回归样本量'].min()), 'n_max': int(results['回归样本量'].max()),
    'significant_10pct_days': results.loc[results['p值'] < .1, '窗口序号'].astype(int).tolist(),
    'significant_5pct_days': results.loc[results['p值'] < .05, '窗口序号'].astype(int).tolist(),
    'elapsed_seconds': round(time.monotonic() - START, 1),
}
audit['finished'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
audit['outputs'] = {'figure': str(FIGURE), 'coefficients': str(RESULT)}
save_audit(audit)
log(f'完成51个单日回归，图片：{FIGURE}')
print(json.dumps(audit['summary'], ensure_ascii=False), flush=True)
