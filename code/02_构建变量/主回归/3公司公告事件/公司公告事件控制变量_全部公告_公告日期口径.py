"""
根据 CSMAR 公司公告合并表，生成“全部公告”口径的公司公告控制变量。

日期口径
--------
1. 使用原始表的 DeclareDate（公告日期）。
2. 公告日期为交易日时记在当天。
3. 公告日期为周末或节假日时顺延到下一个交易日。
4. 最终控制表是一行“被提示公司 × 交易日”。

变量口径
--------
ownAnnouncementAll=1 表示公司在该交易日的公告合并表中至少有一条记录。
该变量包含年报、中报、季报、业绩预告、董事会决议及其他公告。
本脚本不根据 ClassID、ClassifyID 或 Title 删除任何公告。

输出
----
默认输出：
* 公司公告事件控制变量_全部公告_公告日期口径.csv
* 公司公告事件控制变量_全部公告_公告日期口径_审计.csv

增加 --save-detail 后输出：
* 公司公告事件明细_全部公告_公告日期口径.csv

控制变量表可以按 StkcdRecall + Trddt 一对一合并到日度主面板。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/qiyuewu/Desktop/mmm复现数据包")

INPUT_PATH = (
    ROOT
    / "中间结果/00_合并原始数据/CSMAR_公司公告事件表_2002_2025_合并结果.csv"
)
CALENDAR_PATH = ROOT / "中间结果/01_清洗后样本/市场交易日历_td.csv"
OUTPUT_DIR = ROOT / "中间结果/02_变量构造结果/5公司公告事件"

CONTROL_OUTPUT_NAME = "公司公告事件控制变量_全部公告_公告日期口径.csv"
AUDIT_OUTPUT_NAME = "公司公告事件控制变量_全部公告_公告日期口径_审计.csv"
DETAIL_OUTPUT_NAME = "公司公告事件明细_全部公告_公告日期口径.csv"

USECOLS = [
    "AnnouncementID",
    "InstitutionID",
    "Symbol",
    "SecurityID",
    "ShortName",
    "DeclareDate",
    "FullName",
    "Title",
    "ClassID",
    "ClassName",
    "ClassifyID",
    "ClassifyName",
]


def normalize_stock_code(series: pd.Series) -> pd.Series:
    """把股票代码统一为六位字符串，无法识别的值设为缺失。"""

    text = series.astype("string").str.strip()
    text = text.str.replace(r"\.0+$", "", regex=True)
    valid = text.str.fullmatch(r"\d{1,6}", na=False)
    return text.where(valid).str.zfill(6)


def load_calendar(calendar_path: Path) -> pd.DataFrame:
    """读取交易日历并检查日期唯一性。"""

    calendar = pd.read_csv(
        calendar_path,
        usecols=["Trddt"],
        dtype={"Trddt": "string"},
        encoding="utf-8-sig",
    )
    calendar["Trddt"] = pd.to_datetime(
        calendar["Trddt"], errors="raise"
    ).dt.normalize()
    calendar = (
        calendar.dropna()
        .drop_duplicates()
        .sort_values("Trddt")
        .reset_index(drop=True)
    )
    if calendar.empty:
        raise ValueError(f"交易日历为空：{calendar_path}")
    return calendar


def map_to_trading_day(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """把一批公告日期映射到当天或下一个交易日。"""

    calendar_start = calendar["Trddt"].min()
    calendar_end = calendar["Trddt"].max()
    in_calendar = frame["DeclareDate"].between(calendar_start, calendar_end)
    outside_count = int((~in_calendar).sum())
    inside = frame.loc[in_calendar].copy()

    if inside.empty:
        return inside.assign(Trddt=pd.NaT), outside_count

    date_map = (
        inside[["DeclareDate"]]
        .drop_duplicates()
        .sort_values("DeclareDate")
    )
    date_map = pd.merge_asof(
        date_map,
        calendar,
        left_on="DeclareDate",
        right_on="Trddt",
        direction="forward",
        allow_exact_matches=True,
    )
    if date_map["Trddt"].isna().any():
        raise ValueError("存在公告日期无法映射到交易日，请检查交易日历范围。")

    mapped = inside.merge(
        date_map,
        on="DeclareDate",
        how="left",
        validate="m:1",
    )
    return mapped, outside_count


def prepare_chunk(
    chunk: pd.DataFrame,
    source_row_start: int,
    calendar: pd.DataFrame,
    save_detail: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, int]]:
    """清洗一批公告，并生成控制变量所需的去重明细。"""

    frame = chunk.copy()
    frame["_source_row"] = np.arange(
        source_row_start,
        source_row_start + len(frame),
        dtype="int64",
    )
    frame["StkcdRecall"] = normalize_stock_code(frame["Symbol"])
    frame["DeclareDate"] = pd.to_datetime(
        frame["DeclareDate"], errors="coerce"
    ).dt.normalize()

    invalid_code = frame["StkcdRecall"].isna()
    invalid_date = frame["DeclareDate"].isna()
    usable = ~invalid_code & ~invalid_date
    mapped, outside_count = map_to_trading_day(
        frame.loc[usable].copy(),
        calendar,
    )

    id_text = mapped["AnnouncementID"].astype("string").str.strip()
    missing_id = id_text.isna() | id_text.eq("")
    mapped["AnnouncementID_key"] = id_text
    mapped.loc[missing_id, "AnnouncementID_key"] = (
        "__MISSING_ID_ROW_"
        + mapped.loc[missing_id, "_source_row"].astype("string")
    )

    dedup_key = ["StkcdRecall", "Trddt", "AnnouncementID_key"]
    mapped_rows_before_dedup = len(mapped)
    mapped = mapped.drop_duplicates(dedup_key).copy()

    control_events = mapped[
        ["StkcdRecall", "DeclareDate", "Trddt", "AnnouncementID_key"]
    ].copy()

    detail_events = None
    if save_detail:
        detail_cols = [
            "AnnouncementID",
            "InstitutionID",
            "Symbol",
            "SecurityID",
            "ShortName",
            "DeclareDate",
            "FullName",
            "Title",
            "ClassID",
            "ClassName",
            "ClassifyID",
            "ClassifyName",
            "StkcdRecall",
            "Trddt",
            "AnnouncementID_key",
        ]
        detail_events = mapped[detail_cols].copy()

    audit = {
        "读取行数": len(frame),
        "股票代码缺失或异常行数": int(invalid_code.sum()),
        "公告日期缺失或无法解析行数": int(invalid_date.sum()),
        "超出交易日历范围行数": outside_count,
        "块内重复公告行数": mapped_rows_before_dedup - len(control_events),
        "公告编号缺失行数": int(missing_id.sum()),
        "块内保留去重公告行数": len(control_events),
    }
    return control_events, detail_events, audit


def aggregate_controls(events: pd.DataFrame) -> pd.DataFrame:
    """聚合为被提示公司 × 交易日层级。"""

    events = events.drop_duplicates(
        ["StkcdRecall", "Trddt", "AnnouncementID_key"]
    ).copy()
    controls = (
        events.groupby(["StkcdRecall", "Trddt"], as_index=False)
        .agg(
            ownAnnouncementCount=("AnnouncementID_key", "nunique"),
            ownAnnouncementNaturalDateCount=("DeclareDate", "nunique"),
        )
    )
    controls["ownAnnouncementAll"] = np.int8(1)
    controls["date"] = controls["Trddt"]
    controls["ownAnnouncementCount"] = controls["ownAnnouncementCount"].astype(
        "int32"
    )
    controls["ownAnnouncementNaturalDateCount"] = controls[
        "ownAnnouncementNaturalDateCount"
    ].astype("int16")
    return controls[
        [
            "StkcdRecall",
            "date",
            "Trddt",
            "ownAnnouncementAll",
            "ownAnnouncementCount",
            "ownAnnouncementNaturalDateCount",
        ]
    ].sort_values(["StkcdRecall", "Trddt"])


def write_audit(
    audit_counts: dict[str, int],
    controls: pd.DataFrame,
    events: pd.DataFrame,
    calendar: pd.DataFrame,
    output_path: Path,
) -> None:
    """写入小型审计表，记录行数和关键口径。"""

    summary = {
        **audit_counts,
        "最终去重公告行数": int(len(events)),
        "公司交易日行数": int(len(controls)),
        "交易日历开始": calendar["Trddt"].min().date().isoformat(),
        "交易日历结束": calendar["Trddt"].max().date().isoformat(),
        "日期处理口径": "交易日当天，非交易日顺延到下一交易日",
        "公告分类处理": "不使用分类字段筛选，全部公告保留",
        "公告日层级唯一键": "StkcdRecall + Trddt + AnnouncementID_key",
        "公司日层级唯一键": "StkcdRecall + Trddt",
    }
    pd.DataFrame([summary]).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def build_controls(
    input_path: Path = INPUT_PATH,
    calendar_path: Path = CALENDAR_PATH,
    output_dir: Path = OUTPUT_DIR,
    chunksize: int = 300_000,
    save_detail: bool = False,
) -> pd.DataFrame:
    """从原始公告合并表生成全部公告控制变量。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    calendar = load_calendar(calendar_path)
    control_parts: list[pd.DataFrame] = []
    detail_parts: list[pd.DataFrame] = []
    audit_counts = {
        "读取行数": 0,
        "股票代码缺失或异常行数": 0,
        "公告日期缺失或无法解析行数": 0,
        "超出交易日历范围行数": 0,
        "块内重复公告行数": 0,
        "公告编号缺失行数": 0,
        "块内保留去重公告行数": 0,
    }

    reader = pd.read_csv(
        input_path,
        usecols=USECOLS,
        dtype={
            "AnnouncementID": "string",
            "InstitutionID": "string",
            "Symbol": "string",
            "SecurityID": "string",
            "DeclareDate": "string",
            "ClassID": "string",
            "ClassifyID": "string",
        },
        encoding="utf-8-sig",
        chunksize=chunksize,
        low_memory=False,
    )

    source_row_start = 2
    for chunk_number, chunk in enumerate(reader, start=1):
        controls, details, audit = prepare_chunk(
            chunk,
            source_row_start=source_row_start,
            calendar=calendar,
            save_detail=save_detail,
        )
        source_row_start += len(chunk)
        control_parts.append(controls)
        if details is not None:
            detail_parts.append(details)
        for key, value in audit.items():
            audit_counts[key] += value
        print(
            f"已处理数据块 {chunk_number}，保留去重公告行数：{len(controls)}",
            flush=True,
        )

    if not control_parts:
        raise RuntimeError("原始公告表没有可用于生成控制变量的记录。")

    events = pd.concat(control_parts, ignore_index=True)

    if events.empty:
        raise RuntimeError(
            "公告表读取成功，但所有记录清洗后均为空。"
            "请检查股票代码、公告日期和交易日历范围。"
        )

    events_before_global_dedup = len(events)

    events = events.drop_duplicates(
        ["StkcdRecall", "Trddt", "AnnouncementID_key"]
    ).reset_index(drop=True)
    audit_counts["跨块重复公告行数"] = (
        events_before_global_dedup - len(events)
    )
    controls = aggregate_controls(events)

    control_output_path = output_dir / CONTROL_OUTPUT_NAME
    audit_output_path = output_dir / AUDIT_OUTPUT_NAME
    controls.to_csv(
        control_output_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    write_audit(
        audit_counts,
        controls=controls,
        events=events,
        calendar=calendar,
        output_path=audit_output_path,
    )

    if save_detail:
        details = pd.concat(detail_parts, ignore_index=True)
        details = details.drop_duplicates(
            ["StkcdRecall", "Trddt", "AnnouncementID_key"]
        ).sort_values(["StkcdRecall", "Trddt", "AnnouncementID_key"])
        details.to_csv(
            output_dir / DETAIL_OUTPUT_NAME,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
        )

    print("=" * 80)
    print("全部公告控制变量生成完成")
    print("控制表：", control_output_path)
    print("审计表：", audit_output_path)
    if save_detail:
        print("明细表：", output_dir / DETAIL_OUTPUT_NAME)
    print("最终去重公告行数：", len(events))
    print("公司交易日行数：", len(controls))
    print("公司交易日重复数：", controls.duplicated(["StkcdRecall", "Trddt"]).sum())

    return controls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按公告日期口径生成全部公司公告控制变量。"
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--calendar", type=Path, default=CALENDAR_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument(
        "--save-detail",
        action="store_true",
        help="同时保存去重后的公告明细表。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_controls(
        input_path=args.input,
        calendar_path=args.calendar,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        save_detail=args.save_detail,
    )
