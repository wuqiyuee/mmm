"""独立估计95% cue口径的四个图2/图3扩展。

脚本只读取 work/q95_tables_run/panel.parquet 与日度收益源表，运行时动态
导入正式扩展模块并替换其基础面板读取函数和输出目录；正式代码与正式结果
不被改写。用法示例：

  /opt/miniconda3/bin/python code/07_实证分析/03_95分位检验/扩展分析/运行95分位扩展.py --which single
  /opt/miniconda3/bin/python code/07_实证分析/03_95分位检验/扩展分析/运行95分位扩展.py --which all
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

def _find_project_root() -> Path:
    for ancestor in Path(__file__).resolve().parents:
        if (
            (ancestor / "AGENTS.md").is_file()
            and (ancestor / "code").is_dir()
            and (ancestor / "中间结果").is_dir()
        ):
            return ancestor
    raise FileNotFoundError("找不到包含 AGENTS.md、code 和中间结果的项目根目录。")


ROOT = _find_project_root()
SCRATCH_DIR = ROOT / "work" / "extensions95"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(SCRATCH_DIR / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRATCH_DIR / "cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


Q95_PANEL = ROOT / "work/q95_tables_run/panel.parquet"
DGTW_PATH = ROOT / "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW特征调整收益率.csv"
BASE_KEY_PATH = ROOT / "中间结果/04_合并数据/Pattern样本_单日收益回归基础面板.csv"
OUT_ROOT = ROOT / "图表结果/03_95分位检验/扩展分析"

SOURCES = {
    "single": ROOT / "code/07_实证分析/02_扩展分析/单日特征调整收益/图2扩展_单日特征调整收益_加入日期固定效应.py",
    "twoday": ROOT / "code/07_实证分析/02_扩展分析/等长两日非重叠窗口收益/图2扩展_等长两日非重叠窗口特征调整收益_加入日期固定效应.py",
    "interval": ROOT / "code/07_实证分析/02_扩展分析/非重叠窗口收益/图2扩展_非重叠窗口特征调整收益_加入日期固定效应.py",
    "oib": ROOT / "code/07_实证分析/02_扩展分析/单日散户订单不平衡/图3扩展_单日散户订单不平衡_加入日期固定效应.py",
}

TASK_DIR_NAMES = {
    "single": "单日特征调整收益",
    "twoday": "等长两日非重叠窗口收益",
    "interval": "非重叠窗口收益",
    "oib": "单日散户订单不平衡",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_td_date_map() -> pd.DataFrame:
    """读取DGTW源表的交易日序号—日期映射；只保留约6千个唯一交易日。"""
    pieces = []
    for chunk in pd.read_csv(
        DGTW_PATH,
        usecols=["td", "Trddt"],
        dtype={"td": "int32"},
        parse_dates=["Trddt"],
        chunksize=500_000,
    ):
        pieces.append(chunk.drop_duplicates())
    mapping = pd.concat(pieces, ignore_index=True).drop_duplicates("td")
    if mapping["td"].duplicated().any():
        raise ValueError("交易日序号对应多个日期")
    return mapping.sort_values("td").reset_index(drop=True)


class Base95:
    """将独立95面板适配为四个扩展脚本需要的基础面板接口。"""

    def __init__(self):
        if not Q95_PANEL.exists():
            raise FileNotFoundError(Q95_PANEL)
        self.map = build_td_date_map()
        schema = pq.read_schema(Q95_PANEL)
        self.columns = set(schema.names)
        self.data: pd.DataFrame | None = None

    def read(self, _path: Path, *, mod) -> pd.DataFrame:
        if self.data is not None:
            return self.data.copy()
        if hasattr(mod, "CONTROL_COLS"):
            controls = list(mod.CONTROL_COLS)
        else:
            controls = list(mod.OWN_EA_COLS + mod.DISCLOSURE_COLS + mod.NEUTRAL_MEDIA_COLS)
        needed = ["firm", "day", "q", "cue95", *controls]
        missing = [c for c in needed if c not in self.columns]
        if missing:
            raise ValueError(f"95面板缺少字段: {missing}")
        data = pd.read_parquet(Q95_PANEL, columns=needed)
        # q95准备面板中的day是交易日期的Unix日序（由datetime64[D]转int32），
        # 而扩展脚本需要DGTW源表的交易日序td；先恢复日期，再按日期映射。
        data = data.rename(columns={"firm": "StkcdRecall", "day": "_day_code"})
        data["StkcdRecall"] = (
            data["StkcdRecall"].astype("int64").astype("string").str.zfill(6)
        )
        data["Trddt"] = pd.to_datetime(data["_day_code"].astype("int64"), unit="D")
        td_map = self.map.rename(columns={"Trddt": "_map_date"})
        data = data.merge(td_map, left_on="Trddt", right_on="_map_date", how="left", validate="many_to_one")
        data = data.drop(columns=["_day_code", "_map_date"])
        if data["td"].isna().any():
            raise ValueError("95面板存在无法映射的交易日序号")
        data["yearQuarter"] = data["Trddt"].dt.to_period("Q").dt.to_timestamp("Q").astype("string")
        data["cue95"] = pd.to_numeric(data["cue95"], errors="coerce").astype("float32")
        # 与正式读入函数完全相同的研究期及不变量筛选。
        data = data.loc[data["Trddt"].dt.year.between(mod.START_YEAR, mod.END_YEAR)].copy()
        rhs = ["cue95", *controls]
        before = len(data)
        data = data.dropna(subset=["StkcdRecall", "Trddt", "td", "yearQuarter", *rhs]).reset_index(drop=True)
        print(f"[95适配基础面板] 原始{before:,}行，右端与固定效应完整{len(data):,}行，删除{before-len(data):,}行", flush=True)
        self.data = data
        return data.copy()


def configure(mod, key: str, base: Base95, out: Path):
    # 让所有正式回归函数实际使用 cue95，同时避免写入正式结果目录。
    mod.CUE_COL = "cue95"
    if hasattr(mod, "CONTROL_COLS"):
        mod.RHS_COLS = [mod.CUE_COL, *mod.CONTROL_COLS]
    if hasattr(mod, "BASE_READ_COLS"):
        mod.BASE_READ_COLS = list(dict.fromkeys(mod.BASE_KEY_COLS + mod.RHS_COLS))
    mod.NEUTRAL_BASE_PATH = Q95_PANEL
    mod.RESULT_DIR = out
    mod.FIGURE_DIR = out
    mod.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mod.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if key == "oib":
        mod.read_base_panel = lambda path: (
            base.read(path, mod=mod),
            list(mod.NEUTRAL_MEDIA_COLS),
            [mod.CUE_COL, *list(mod.OWN_EA_COLS + mod.DISCLOSURE_COLS + mod.NEUTRAL_MEDIA_COLS)],
        )
    else:
        mod.read_base_panel = lambda path: base.read(path, mod=mod)


def run_one(key: str):
    out = OUT_ROOT / TASK_DIR_NAMES[key]
    out.mkdir(parents=True, exist_ok=True)
    mod = load_module(SOURCES[key], f"extensions95_{key}")
    base = Base95()
    configure(mod, key, base, out)
    # main()的参数由本包装脚本固定，避免使用正式脚本默认的90%结果路径。
    sys.argv = [str(SOURCES[key]), "--sample-mode", "available", "--base-source", "neutral"]
    print(f"\n===== 开始95分位扩展：{key} =====", flush=True)
    mod.main()
    print(f"===== 完成95分位扩展：{key}；输出 {out} =====", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["single", "twoday", "interval", "oib", "all"], default="all")
    args = parser.parse_args()
    keys = list(SOURCES) if args.which == "all" else [args.which]
    for key in keys:
        run_one(key)


if __name__ == "__main__":
    main()

