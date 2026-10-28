"""表2与选定图2共用的主回归口径和只读数据入口。

一行是被提示公司×收益起始交易日。当前复用已经核验的季度数据，
其中包含原真实日起点及新增合格填充日起点；不会改写正式最终面板。
季度缓存仍保存在原试算目录，须保留；缺失时明确报错，不自动重建。
本模块不调用临时试算脚本，导入时不读取数据，也不生成任何文件。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "work" / "主回归口径组合试算_20261026"
PANEL_DIR = DATA_DIR / "enriched"
RETURN_DIR = DATA_DIR / "figure2_selected_20261027" / "returns"

SPEC_ID = "沪深300_个人365天_数行者分段_允许填充_全期"
SPEC_DESCRIPTION = (
    "历史沪深300提示；个人365天共同覆盖排除；一级供应链及同行业排除；"
    "提示方保留金融和房地产；被提示方沿用非金融等基础资格、保留房地产，"
    "不追加市值和60个月交易史筛选；2005—2009年数行者报刊、"
    "2010—2025年数行者网络；允许合格填充日起点；保留全部年份"
)
CUE_COLUMN = "quarterLag1HS300"
CUE_DEFINITION = "hs300_split_365_keep_fin_re"
NEWS_SOURCE = "datago_segmented"
START_DATE = "2005-04-08"
END_DATE = "2025-12-31"
ALLOW_FILLED_STARTS = True
EXCLUDED_YEARS: tuple[int, ...] = ()

OWN_EA_COLUMNS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)
EVENT_COLUMNS = ["ownOtherEventER", "ownPerformanceForecastBrief"]
NEWS_COLUMNS = ["aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV"]
CONTROL_COLUMNS = OWN_EA_COLUMNS + EVENT_COLUMNS + NEWS_COLUMNS


def main_input_paths() -> tuple[list[Path], int]:
    """核对输入完成清单、季度文件集合和提示定义，返回季度路径及提示列号。"""
    manifests = [
        DATA_DIR / "panel" / "complete.json",
        PANEL_DIR / "complete.json",
        RETURN_DIR / "complete.json",
        DATA_DIR / "cues" / "cue_columns.json",
    ]
    missing = [str(path) for path in manifests if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "当前沪深300主口径输入不完整：\n" + "\n".join(missing)
            + "\n请单独准备或恢复这些数据；本入口不会退回旧面板或自动重建。"
        )
    preparation, panel, returns, cues = [
        json.loads(path.read_text(encoding="utf-8")) for path in manifests
    ]
    if preparation["controls"] != OWN_EA_COLUMNS + EVENT_COLUMNS:
        raise ValueError("缓存中23项公告控制的顺序与当前主口径不一致。")
    if returns["windows"] != 50 or returns["date_origin"] != "1970-01-01":
        raise ValueError("收益缓存的窗口数或日期原点不一致。")
    if panel["rows"] != returns["actual"] + returns["filled"]:
        raise ValueError("季度面板与收益缓存的总行数不一致。")
    cue_columns = cues["columns"]
    matches = [i for i, definition in enumerate(cue_columns)
               if definition["name"] == CUE_DEFINITION]
    if len(matches) != 1:
        raise ValueError("找不到唯一的历史沪深300、个人365天、保留行业提示列。")
    cue_index = matches[0]
    expected = {"definition": "hs300", "analyst_identity": "split",
                "lookback_days": 365, "drop_cue_financial_realestate": False}
    if any(cue_columns[cue_index][key] != value for key, value in expected.items()):
        raise ValueError("提示列名称与清单记录的实际构造口径冲突。")
    paths = sorted(path for path in PANEL_DIR.glob("q*.npy")
                   if path.stem[1:].isdigit())
    return_names = {path.name for path in RETURN_DIR.glob("q*.npy")
                    if path.stem[1:].isdigit()}
    if not paths or {path.name for path in paths} != return_names:
        raise ValueError("当前主口径的面板和收益季度文件不齐全或不一一对应。")
    return paths, cue_index


def select_news(block: np.ndarray, source: str = NEWS_SOURCE) -> np.ndarray:
    """每段使用对应来源的4项变化控制；原报刊仅供显式历史对照使用。"""
    if source == "datago_segmented":
        switch = block["date"] >= np.datetime64("2010-01-01").astype("int32")
        return np.where(switch[:, None], block["news"][:, 8:12], block["news"][:, 4:8])
    if source == "cnrds_paper":
        return block["news"][:, :4]
    raise ValueError(f"不支持的新闻口径：{source}")


def iter_main_blocks(
    source: str = NEWS_SOURCE,
    *,
    required_horizons: tuple[int, ...] = (1, 3, 5),
):
    """逐季度提供面板、50窗收益、27项控制、提示指标及共同资格掩码。

    基础数据已继承两日、十一日收益完整资格。本处按调用方指定的窗口
    终点要求收益及27项控制共同有效；表2传入(1, 10)，只检查两日、十一日。
    默认(1, 3, 5)保留图2及其他既有调用方的三短窗要求。各模型再检查
    自身收益并删除相应固定效应单例。
    收益沿用既有缩尾值，不重新计算基准、断点或缩尾阈值。
    """
    if not required_horizons or any(
        not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= 50
        for horizon in required_horizons
    ):
        raise ValueError("所需收益窗口终点必须为1至50的整数，且至少指定一个窗口。")
    return_columns = [horizon - 1 for horizon in required_horizons]
    paths, cue_index = main_input_paths()
    total_rows = 0
    for path in paths:
        block = np.load(path, mmap_mode="r")
        total_rows += len(block)
        returns = np.load(RETURN_DIR / path.name, mmap_mode="r")
        if returns.shape != (len(block), 50):
            raise ValueError(f"{path.name} 的面板与收益行数或窗口数不一致。")
        if not np.array_equal(returns[:, [0, 2, 4]], block["y"], equal_nan=True):
            raise ValueError(f"{path.name} 的收益行位置与面板不对应。")
        keys = block["date"].astype("int64") * 1_000_000 + block["firm"]
        if len(keys) and (np.diff(keys) <= 0).any():
            raise ValueError(f"{path.name} 存在重复或无序的公司日期键。")
        dates = block["date"].astype("datetime64[D]")
        months = dates.astype("datetime64[M]").astype("int64")
        quarter = (months + 1970 * 12 - 2005 * 12) // 3
        if (quarter != int(path.stem[1:])).any():
            raise ValueError(f"{path.name} 含有不属于该季度的观测。")
        news = select_news(block, source)
        controls = np.column_stack([block["x"], news])
        cue = block["cues"][:, cue_index]
        if not np.isin(cue, [0, 1]).all():
            raise ValueError(f"{path.name} 的沪深300提示并非0/1指标。")
        keep = np.isfinite(controls).all(axis=1)
        keep &= np.isfinite(returns[:, return_columns]).all(axis=1)
        keep &= (dates >= np.datetime64(START_DATE)) & (dates <= np.datetime64(END_DATE))
        if not ALLOW_FILLED_STARTS:
            keep &= block["filled"] == 0
        if EXCLUDED_YEARS:
            years = dates.astype("datetime64[Y]").astype("int64") + 1970
            keep &= ~np.isin(years, EXCLUDED_YEARS)
        if not np.isfinite(returns[:, [0, 9]][keep]).all():
            raise ValueError(f"{path.name} 不满足既有两日、十一日收益完整资格。")
        yield block, returns, controls, cue, keep
    expected_rows = json.loads((PANEL_DIR / "complete.json").read_text())["rows"]
    if total_rows != expected_rows:
        raise ValueError("实际季度总行数与完成清单不一致，请检查遗漏的季度文件。")
