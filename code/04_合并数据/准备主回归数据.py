"""为尚未迁移的扩展分析核对旧正式面板并整理季度文件。

当前表2和两个图2直接读取带筛样规格后缀的正式合并面板，无需运行本文件。
本文件仍要求旧的无筛样规格后缀面板及对应口径清单；当前合并脚本不再生成它们。
使用前须先核对旧扩展的数据来源和筛样，不可直接把新面板改名后传入。
本文件只读旧正式面板及清单，不构造研究变量，
不读取其他上游数据或旧市值面板，也不执行其他项目脚本。
同名季度观测与50窗收益按相同行位置配对；不覆盖非空目标目录。
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = PROJECT_ROOT / "code/07_实证分析/90_公共模块"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
from 主回归口径 import (
    DATA_DIR, PANEL_DIR, RETURN_DIR, MANIFEST_DIR, INPUT_MANIFEST_PATH,
    SOURCE_PANEL_PATH, SOURCE_MANIFEST_PATH, SCHEMA_VERSION, DATE_ORIGIN,
    RETURN_WINDOW_COUNT, PANEL_DTYPE, SPEC_ID, SPEC_DESCRIPTION,
    CUE_COLUMN, CUE_DEFINITION, NEWS_SOURCE, START_DATE, END_DATE,
    OWN_EA_COLUMNS, EVENT_COLUMNS, NEWS_COLUMNS,
)

CHUNK_SIZE = 150_000
ANNOUNCEMENT_COLUMNS = OWN_EA_COLUMNS + EVENT_COLUMNS
RETURN_COLUMNS = [f"chAdjreturn_t_tp{h}_Recall_w" for h in range(1, 51)]


def iter_source_blocks():
    """按原面板公司、日期顺序只读并核对，不筛样或重算任何变量。"""
    columns = ["StkcdRecall", "Trddt", "Filling", "cueDefinition", CUE_COLUMN,
               *ANNOUNCEMENT_COLUMNS, *NEWS_COLUMNS, *RETURN_COLUMNS]
    types = {column: "float32" for column in ANNOUNCEMENT_COLUMNS + NEWS_COLUMNS + RETURN_COLUMNS}
    types.update({"StkcdRecall": "int32", "Filling": "float32", CUE_COLUMN: "float32",
                  "cueDefinition": "string"})
    last_key = -1
    for part in pd.read_csv(
        SOURCE_PANEL_PATH, usecols=columns, dtype=types,
        parse_dates=["Trddt"], chunksize=CHUNK_SIZE,
    ):
        if part.empty:
            continue
        if (part["Trddt"].isna().any()
                or not part["Trddt"].between(START_DATE, END_DATE).all()
                or not part["cueDefinition"].eq("hs300").fillna(False).all()
                or not part["Filling"].isin([0, 1]).all()
                or not part[CUE_COLUMN].isin([0, 1]).all()):
            raise ValueError("正式沪深300面板的日期、提示口径或0/1标志异常，请核对合并入口。")
        days = part["Trddt"].to_numpy("datetime64[D]").astype("int64")
        keys = part["StkcdRecall"].to_numpy("int64") * 100_000 + days
        if keys[0] <= last_key or (np.diff(keys) <= 0).any():
            raise ValueError("正式沪深300面板的公司日期键重复或未升序排列。")
        last_key = int(keys[-1])
        announcement = part[ANNOUNCEMENT_COLUMNS].to_numpy()
        news = part[NEWS_COLUMNS].to_numpy()
        returns = part[RETURN_COLUMNS].to_numpy()
        if (not np.isin(announcement, [0, 1]).all()
                or not np.isfinite(news).all()
                or not np.isin(news[:, :2], [0, 1]).all()
                or (news[:, 2:] < 0).any()
                or not np.isfinite(returns[:, [0, 9]]).all()
                or np.isinf(returns).any()):
            raise ValueError("正式面板未满足27项控制和两日、十一日收益共同资格；本文件不代为筛样。")
        yield part


def main():
    # 一、核对旧扩展要求的正式面板及口径清单；本轮不迁移输入来源。
    if not SOURCE_PANEL_PATH.is_file() or not SOURCE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"旧扩展入口缺少面板或口径清单：{SOURCE_PANEL_PATH}；{SOURCE_MANIFEST_PATH}"
            "\n当前合并脚本不再生成这套旧输入，请先核对扩展分析的数据来源。"
            "当前表2和两个图2无需运行本文件。"
        )
    source_info = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {
        "panel_format": 1, "cue_definition": "hs300", "news_source": NEWS_SOURCE,
        "allow_filled_starts": True, "singleton_filter": "per_regression",
        "start_date": START_DATE, "end_date": END_DATE,
    }
    config = source_info.get("config", {})
    expected_config = {
        "CUE_DEFINITION": "hs300", "DROP_ANALYST_OVERLAP": True,
        "ANALYST_ID_MODE": "split", "ANALYST_LOOKBACK_DAYS": 365,
        "EVENT_DATE_MODE": "next_if_nontrading",
    }
    if (any(source_info.get(key) != value for key, value in expected.items())
            or any(config.get(key) != value for key, value in expected_config.items())):
        raise ValueError("沪深300正式面板的构造口径与当前主回归不一致，请核对合并入口。")
    source_paths = [SOURCE_PANEL_PATH, SOURCE_MANIFEST_PATH]
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths}
    if before[SOURCE_PANEL_PATH] != (source_info["panel_size"], source_info["panel_mtime_ns"]):
        raise ValueError("沪深300面板与口径清单不属于同一次完整输出，请重新运行合并入口。")
    bounds = np.asarray([
        source_info["return_winsor_bounds"][f"chAdjreturn_t_tp{h}_Recall"]
        for h in range(1, 51)
    ], dtype="float64")
    if (bounds.shape != (50, 2) or not np.isfinite(bounds).all()
            or (bounds[:, 0] > bounds[:, 1]).any()):
        raise ValueError("正式面板的50窗缩尾边界缺失或异常。")
    if DATA_DIR.exists() and (not DATA_DIR.is_dir() or any(DATA_DIR.iterdir())):
        raise FileExistsError(f"季度目标目录非空，请先核对并另行保留原结果：{DATA_DIR}")

    # 二、只统计各季度行数；所有样本资格已在统一合并入口确定。
    quarter_rows = {}
    actual_count = filled_count = positive_count = 0
    for part in iter_source_blocks():
        quarter = (part["Trddt"].dt.year - 2005) * 4 + part["Trddt"].dt.quarter - 1
        for q, count in quarter.value_counts().items():
            name = f"q{int(q):02d}.npy"
            quarter_rows[name] = quarter_rows.get(name, 0) + int(count)
        filled_count += int(part["Filling"].sum())
        actual_count += int(part["Filling"].eq(0).sum())
        positive_count += int(part[CUE_COLUMN].sum())
    rows = actual_count + filled_count
    if (not rows or rows != source_info["rows"] or actual_count != source_info["actual"]
            or filled_count != source_info["filled"] or positive_count != source_info["positive"]):
        raise ValueError("正式面板行数、真实日、填充日或提示数量与口径清单不一致。")

    # 三、预分配季度文件，再逐块原样写入提示、控制及已缩尾收益。
    for directory in [PANEL_DIR, RETURN_DIR, MANIFEST_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    for name, count in quarter_rows.items():
        block = np.lib.format.open_memmap(PANEL_DIR / name, mode="w+", dtype=PANEL_DTYPE, shape=(count,))
        returns = np.lib.format.open_memmap(
            RETURN_DIR / name, mode="w+", dtype="float32", shape=(count, RETURN_WINDOW_COUNT),
        )
        block.flush()
        returns.flush()
        del block, returns
    offsets = {name: 0 for name in quarter_rows}
    for part in iter_source_blocks():
        record = np.zeros(len(part), dtype=PANEL_DTYPE)
        record["firm"] = part["StkcdRecall"].to_numpy("int32")
        record["date"] = part["Trddt"].to_numpy("datetime64[D]").astype("int32")
        record["filled"] = part["Filling"].to_numpy("uint8")
        record["cues"][:, 0] = part[CUE_COLUMN].to_numpy("uint8")
        record["x"] = part[ANNOUNCEMENT_COLUMNS].to_numpy("float32")
        record["news"] = part[NEWS_COLUMNS].to_numpy("float32")
        y = part[RETURN_COLUMNS].to_numpy("float32")
        record["y"] = y[:, [0, 2, 4]]
        quarter = ((part["Trddt"].dt.year - 2005) * 4 + part["Trddt"].dt.quarter - 1).to_numpy()
        for q in np.unique(quarter):
            name = f"q{int(q):02d}.npy"
            take = quarter == q
            begin = offsets[name]
            end = begin + int(take.sum())
            if end > quarter_rows[name]:
                raise ValueError(f"{name} 的写入行数超过预分配数量。")
            block = np.load(PANEL_DIR / name, mmap_mode="r+", allow_pickle=False)
            returns = np.load(RETURN_DIR / name, mmap_mode="r+", allow_pickle=False)
            block[begin:end] = record[take]
            returns[begin:end] = y[take]
            block.flush()
            returns.flush()
            offsets[name] = end
            del block, returns

    # 四、季度内共同排序并核对配对位置；全部完成后才写完成清单。
    figure2_rows = 0
    for name, count in sorted(quarter_rows.items()):
        if offsets[name] != count:
            raise ValueError(f"{name} 未完整写入。")
        block = np.load(PANEL_DIR / name, mmap_mode="r+", allow_pickle=False)
        returns = np.load(RETURN_DIR / name, mmap_mode="r+", allow_pickle=False)
        keys = block["date"].astype("int64") * 1_000_000 + block["firm"]
        order = np.argsort(keys)
        if (np.diff(keys[order]) <= 0).any():
            raise ValueError(f"{name} 存在重复公司日期键。")
        block[:] = block[order]
        returns[:] = returns[order]
        if not np.array_equal(block["y"], returns[:, [0, 2, 4]], equal_nan=True):
            raise ValueError(f"{name} 的观测与收益行位置不一致。")
        figure2_rows += int(np.isfinite(returns[:, [0, 2, 4]]).all(axis=1).sum())
        block.flush()
        returns.flush()
        del block, returns
    for path, state in before.items():
        if state != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError(f"分季度整理期间正式来源发生变化，未写完成清单：{path}")
    information = {
        "schema_version": SCHEMA_VERSION, "spec_id": SPEC_ID,
        "spec_description": SPEC_DESCRIPTION, "cue_column": CUE_COLUMN,
        "cue_definition": CUE_DEFINITION, "news_source": NEWS_SOURCE,
        "date_origin": DATE_ORIGIN, "windows": RETURN_WINDOW_COUNT,
        "announcement_columns": ANNOUNCEMENT_COLUMNS, "news_columns": NEWS_COLUMNS,
        "start_date": START_DATE, "end_date": END_DATE,
        "actual": actual_count, "filled": filled_count, "rows": rows,
        "quarter_rows": quarter_rows, "正提示公司日数": positive_count,
        "表2共同资格数_删除单例前": rows,
        "图2默认共同资格数_删除单例前": figure2_rows,
        "input_kind": "unified_hs300_panel",
        "source_panel": str(SOURCE_PANEL_PATH.relative_to(PROJECT_ROOT)),
        "source_manifest": str(SOURCE_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "来源文件": [{"路径": str(path.relative_to(PROJECT_ROOT)), "字节数": state[0], "修改时间纳秒": state[1]}
                     for path, state in before.items()],
        "缩尾口径": source_info["winsor_rule"],
        "收益单位": "小数；终点1至50对应当日至第1至50个后续交易日。",
        "整理方式": "只核对及分季度存储，不重算变量、不重新筛样、不预删固定效应单例。",
    }
    with INPUT_MANIFEST_PATH.open("x", encoding="utf-8") as handle:
        json.dump(information, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"沪深300正式面板已整理为季度输入：{DATA_DIR}；共{rows:,}行。", flush=True)


if __name__ == "__main__":
    main()
