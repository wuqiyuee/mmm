"""尚未迁移的提示系数时间变化扩展所用的旧季度输入入口。

当前表2和两个图2脚本直接读取带“筛样规格5或6”后缀的正式合并面板，
不使用本模块或“准备主回归数据.py”。
本模块仍保留旧的无筛样规格后缀面板、口径清单及季度输入约定，
供提示系数时间变化扩展使用；这些扩展的数据来源和筛样尚未迁移。
当前合并脚本不再生成本模块要求的旧面板及清单，不能仅替换文件名使用新面板。

一行代表被提示公司×收益起始交易日，公司与日期组成唯一键。
季度观测与50窗收益按相同文件名、相同行位置对应；只保存选定主规格。
基础资格、提示关系、公告控制、分段新闻和收益缩尾均沿用旧正式输入，
本模块按各调用方指定窗口计算共同资格，不执行数据准备或回归。
导入本模块不读取数据，不写文件。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# 一、旧扩展入口的位置、数据结构及既有研究口径；本轮不迁移输入来源。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "中间结果" / "04_合并数据" / "主回归_沪深300口径"
PANEL_DIR = DATA_DIR / "季度观测"
RETURN_DIR = DATA_DIR / "50窗收益"
MANIFEST_DIR = DATA_DIR / "口径清单"
INPUT_MANIFEST_PATH = MANIFEST_DIR / "主回归输入清单.json"
PREPARATION_SCRIPT = PROJECT_ROOT / "code" / "04_合并数据" / "准备主回归数据.py"
SOURCE_PANEL_PATH = PROJECT_ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板_沪深300.csv"
SOURCE_MANIFEST_PATH = SOURCE_PANEL_PATH.with_suffix(".口径.json")
SCHEMA_VERSION = 3
DATE_ORIGIN = "1970-01-01"
RETURN_WINDOW_COUNT = 50

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

# firm（被提示公司）；date（自日期原点起的天数）；filled（填充起点标志）；
# x（23项公告控制）；news（4项已按年份选源的新闻控制）；
# cues（当前沪深300提示，仅1列）；y（终点1、3、5的收益行位置核验值）。
PANEL_DTYPE = np.dtype([
    ("firm", "i4"), ("date", "i4"), ("filled", "u1"),
    ("x", "f4", (23,)), ("news", "f4", (4,)),
    ("cues", "u1", (1,)), ("y", "f4", (3,)),
])


# 二、核对正式完成清单和季度文件集合。
def main_input_paths() -> tuple[list[Path], int]:
    """返回正式季度观测路径和提示列号，保持现有调用方的接口。"""
    if not INPUT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"旧扩展入口缺少季度输入清单：{INPUT_MANIFEST_PATH}"
            f"\n请先核对旧面板和口径清单，再使用整理脚本：{PREPARATION_SCRIPT}"
            "\n当前表2和两个图2脚本不需要该季度输入。"
        )
    info = json.loads(INPUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "spec_id": SPEC_ID,
        "cue_definition": CUE_DEFINITION,
        "news_source": NEWS_SOURCE,
        "date_origin": DATE_ORIGIN,
        "windows": RETURN_WINDOW_COUNT,
        "announcement_columns": OWN_EA_COLUMNS + EVENT_COLUMNS,
        "news_columns": NEWS_COLUMNS,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "input_kind": "unified_hs300_panel",
        "source_panel": str(SOURCE_PANEL_PATH.relative_to(PROJECT_ROOT)),
        "source_manifest": str(SOURCE_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
    }
    if any(info.get(key) != value for key, value in expected.items()):
        raise ValueError("正式主回归输入的结构或口径与当前代码不一致，请核对准备脚本。")
    sources = info.get("来源文件", [])
    expected_sources = {str(path.relative_to(PROJECT_ROOT)) for path in [SOURCE_PANEL_PATH, SOURCE_MANIFEST_PATH]}
    if {item["路径"] for item in sources} != expected_sources or len(sources) != 2:
        raise ValueError("季度输入未记录统一沪深300面板及其口径清单，请重新整理。")
    for item in sources:
        path = PROJECT_ROOT / item["路径"]
        if (not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns)
                != (item["字节数"], item["修改时间纳秒"])):
            raise ValueError(f"沪深300正式来源已更新或缺失，请重新整理季度输入：{path}")
    if info["rows"] != info["actual"] + info["filled"]:
        raise ValueError("完成清单的真实日、填充日和总行数不一致。")
    paths = sorted(path for path in PANEL_DIR.glob("q*.npy")
                   if path.stem[1:].isdigit())
    panel_names = {path.name for path in paths}
    return_names = {path.name for path in RETURN_DIR.glob("q*.npy")
                    if path.stem[1:].isdigit()}
    if not paths or panel_names != return_names or panel_names != set(info["quarter_rows"]):
        raise ValueError("正式季度观测、收益文件与完成清单不一一对应。")
    return paths, 0


# 三、逐季度读取、核对行位置，再按调用方指定窗口筛选。
def iter_main_blocks(
    source: str = NEWS_SOURCE,
    *,
    required_horizons: tuple[int, ...] = (1, 3, 5),
):
    """提供季度观测、50窗收益、27项控制、提示及共同资格掩码。

    基础输入已具备两日、十一日收益共同完整资格。
    保留旧扩展调用方的默认终点(1,3,5)，再检查各自待估计窗口的收益。
    第h−1列对应从当日至第h个后续交易日的累计收益，单位为小数。
    本入口不重新计算收益基准或缩尾边界；收益由统一合并脚本缩尾后传入。
    """
    if source != NEWS_SOURCE:
        raise ValueError("正式主回归输入只保存选定的数行者分段新闻口径。")
    if not required_horizons or any(
        not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= RETURN_WINDOW_COUNT
        for horizon in required_horizons
    ):
        raise ValueError("所需收益窗口终点必须为1至50的整数，且至少指定一个窗口。")
    paths, cue_index = main_input_paths()
    info = json.loads(INPUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    return_columns = [horizon - 1 for horizon in required_horizons]
    total_rows = 0
    filled_rows = 0
    last_key = -1
    for path in paths:
        # 1. 数组结构、季度行数和三个收益锚点必须对应。
        block = np.load(path, mmap_mode="r", allow_pickle=False)
        returns = np.load(RETURN_DIR / path.name, mmap_mode="r", allow_pickle=False)
        if block.ndim != 1 or block.dtype != PANEL_DTYPE:
            raise ValueError(f"{path.name} 的季度观测字段或精度与正式结构不一致。")
        if len(block) != info["quarter_rows"][path.name]:
            raise ValueError(f"{path.name} 的行数与完成清单不一致。")
        if returns.shape != (len(block), RETURN_WINDOW_COUNT) or returns.dtype != np.dtype("float32"):
            raise ValueError(f"{path.name} 的收益行数、窗口数或精度不一致。")
        if not np.array_equal(returns[:, [0, 2, 4]], block["y"], equal_nan=True):
            raise ValueError(f"{path.name} 的收益行位置与季度观测不对应。")

        # 2. 文件按日期、公司升序，季度编号从2005年第一季度的0开始。
        keys = block["date"].astype("int64") * 1_000_000 + block["firm"]
        if len(keys) and ((np.diff(keys) <= 0).any() or keys[0] <= last_key):
            raise ValueError(f"{path.name} 存在重复或无序的公司日期键。")
        if len(keys):
            last_key = int(keys[-1])
        dates = block["date"].astype("datetime64[D]")
        months = dates.astype("datetime64[M]").astype("int64")
        quarter = (months + 1970 * 12 - 2005 * 12) // 3
        if (quarter != int(path.stem[1:])).any():
            raise ValueError(f"{path.name} 含有其他季度的观测。")

        # 3. 公告控制与已选源新闻按统一列顺序拼成27项控制。
        controls = np.column_stack([block["x"], block["news"]])
        cue = block["cues"][:, cue_index]
        if not np.isin(cue, [0, 1]).all() or not np.isin(block["filled"], [0, 1]).all():
            raise ValueError(f"{path.name} 的提示或填充标志不是0/1指标。")
        if not np.isfinite(returns[:, [0, 9]]).all():
            raise ValueError(f"{path.name} 不满足基础两日、十一日收益完整资格。")
        keep = np.isfinite(controls).all(axis=1)
        keep &= np.isfinite(returns[:, return_columns]).all(axis=1)
        keep &= (dates >= np.datetime64(START_DATE)) & (dates <= np.datetime64(END_DATE))
        if not ALLOW_FILLED_STARTS:
            keep &= block["filled"] == 0
        if EXCLUDED_YEARS:
            years = dates.astype("datetime64[Y]").astype("int64") + 1970
            keep &= ~np.isin(years, EXCLUDED_YEARS)
        total_rows += len(block)
        filled_rows += int(block["filled"].sum())
        yield block, returns, controls, cue, keep
    if total_rows != info["rows"] or filled_rows != info["filled"]:
        raise ValueError("实际季度总行数或填充日数量与完成清单不一致。")
