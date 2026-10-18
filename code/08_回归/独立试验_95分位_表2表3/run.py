"""独立重建95分位提示并生成规律公司样本表2、表3图片。

默认研究期2005—2025年，既有脚本、正式数据与90分位结果不变。
运行：/opt/miniconda3/bin/python -B code/08_回归/独立试验_95分位_表2表3/run.py
临时面板与残差块只存项目work/，正常结束后清理；结果保存到独立时间目录。
"""
from __future__ import annotations

import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '2'

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def protection_state():
    return {
        'code': {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'code').rglob('*.py')
            if MODULE_DIR not in p.parents
        },
        'data': {
            str(p.relative_to(ROOT)): [p.stat().st_size, p.stat().st_mtime_ns]
            for folder in ('input', '中间结果')
            for p in (ROOT/folder).rglob('*') if p.is_file()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--reuse-panel', action='store_true', help='使用已完成且审计通过的本次临时面板')
    parser.add_argument('--keep-work', action='store_true', help='保留临时目录供最后核验，默认完成后清理')
    args = parser.parse_args()
    (ROOT/'work').mkdir(exist_ok=True)
    work = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix='q95_tables_', dir=ROOT/'work'))
    if not work.is_relative_to(ROOT/'work') or work == ROOT/'work' or work.is_symlink():
        raise ValueError('临时目录必须是项目work/下的独立非符号链接子目录。')
    work.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output = args.output_dir.resolve() if args.output_dir else ROOT/'图表结果/独立检验/95分位_表2表3'/stamp
    output.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(work/'matplotlib_cache')
    os.environ['MPLBACKEND'] = 'Agg'
    before = protection_state()
    prepare = load_module('q95_tables_prepare', MODULE_DIR/'prepare.py')
    estimator = load_module('q95_tables_estimate', MODULE_DIR/'estimate.py')
    renderer = load_module('q95_tables_render', MODULE_DIR/'render.py')
    if args.reuse_panel:
        if not (work/'panel.parquet').exists() or not (work/'preparation_audit.json').exists():
            raise FileNotFoundError('复用面板需要完整面板和准备审计文件。')
        audit = json.loads((work/'preparation_audit.json').read_text())
    else:
        audit = prepare.prepare(work)
    results = estimator.estimate_all(work/'panel.parquet', work, output)
    results.setdefault('metadata', {}).update({
        'sample': '规律公司主样本', 'start_year': 2005, 'end_year': 2025,
        'cue_threshold_percentile': 95,
        'cue_capitalization': '提示公司事件前第3个交易日总市值',
        'breakpoint_month': '事件交易月的上一个月',
        'baseline_rules': '原有一期历史关系、规律公司条件与三轮关联排除均保留',
        'return_unit': '百分点，原小数收益乘100',
        'oib_unit': '百分点，沿用原数据单位，不再次乘100',
        'clustering': '被提示公司及交易日期双向聚类，有限样本修正',
        'significance': '双侧单次检验，未作跨规格多重检验校正',
    })
    (output/'regression_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    images = renderer.render_results(results, output)
    after = protection_state()
    if before != after:
        raise AssertionError('运行期间既有代码或正式输入发生变化，请检查文件保护记录。')
    (output/'preparation_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    (output/'file_protection.json').write_text(json.dumps({
        'before': before, 'after_matches_before': True,
        'new_scripts': {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in MODULE_DIR.glob('*.py')
        },
    }, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(output), 'images': [str(p) for p in images],
                      'existing_code_files_unchanged': len(before['code']),
                      'formal_data_files_unchanged': len(before['data'])}, ensure_ascii=False), flush=True)
    if not args.keep_work:
        shutil.rmtree(work)


if __name__ == '__main__':
    main()
