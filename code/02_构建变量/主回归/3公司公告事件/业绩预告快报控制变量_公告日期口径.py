"""
根据 CSMAR 的四张“业绩预告/业绩快报”表，生成公司—交易日层面的控制变量。

本脚本的日期口径
----------------
1. 旧版业绩预告简表：同时使用 PubliDate（预告日期）和
   FirstPubliDate（首次预告日期）。旧版简表把部分修正信息汇总在一行，
   因此这两个日期都要保留。
2. 新版业绩预告表、旧版业绩快报简表、新版业绩快报表：使用每一条记录的
   PubliDate（公告披露日）。新版表中的首次公告和后续修正公告均保留。
3. AccPeriod（业绩报告所属会计期间）只保留在明细中用于核查，绝不当作事件日期，
   也不按“公司×AccPeriod”把新版明细压缩成一条。
4. 公告日期为交易日时记在当天；公告日期为周末或节假日时顺延到下一个交易日。
   当前数据没有公告发布时间，所以无法进一步判断盘中或收盘后披露。
5. 最终控制表是一行“证券代码 × 交易日”：当日有业绩预告记 ownForecast=1，
   有业绩快报记 ownQuickReport=1；二者并集为 ownPerformanceForecastBrief=1。

输入
----
四张 CSV 会按文件名自动查找，不依赖带日期范围的中文文件夹名称：
* FIN_F_SimForecFin.csv   （旧版业绩预告简表）
* FIN_F_ForecFin.csv      （新版业绩预告表）
* FIN_F_QuiTraSimFin.csv  （旧版业绩快报简表）
* FIN_F_QuiTraFin.csv     （新版业绩快报表）

输出
----
默认写入“中间结果/02_变量构造结果/5公司公告事件”：

默认只生成：
* 业绩预告快报控制变量_公告日期口径.csv

只有运行时增加 --save-diagnostics 参数，
才额外生成：
* 业绩预告快报事件明细_公告日期口径.csv
* 业绩预告快报控制变量_公告日期口径_审计.csv
* 业绩预告快报事件_交易日历范围外.csv

示例
----
python 业绩预告快报控制变量_公告日期口径.py
python 业绩预告快报控制变量_公告日期口径.py --start-date 2005-01-01 --end-date 2025-12-31
python 业绩预告快报控制变量_公告日期口径.py --forecast-source forecast-only

默认保留新版业绩预告表 Source=0（业绩预告）和 Source=1（定期公告）两种来源。
只有在研究设计明确要求“仅独立业绩预告”时，才使用 --forecast-source forecast-only。
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


# ============================================================================
# 路径和输出设置
# ============================================================================

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "input/CSMAR/业绩预告&业绩快报"
CALENDAR_PATH = ROOT / "中间结果/01_清洗后样本/市场交易日历_td.csv"
OUTPUT_DIR = ROOT / "中间结果/02_变量构造结果/5公司公告事件"

DETAIL_OUTPUT_PATH = OUTPUT_DIR / "业绩预告快报事件明细_公告日期口径.csv"
CONTROL_OUTPUT_PATH = OUTPUT_DIR / "业绩预告快报控制变量_公告日期口径.csv"
AUDIT_OUTPUT_PATH = OUTPUT_DIR / "业绩预告快报控制变量_公告日期口径_审计.csv"
OUTSIDE_OUTPUT_PATH = OUTPUT_DIR / "业绩预告快报事件_交易日历范围外.csv"


# 文件名：
#   事件类别、要提取的公告日期字段。
# 旧版预告简表有两个日期字段；其余表只使用 PubliDate。
TABLE_SPECS = {
    "FIN_F_SimForecFin.csv": ("预告", ["PubliDate", "FirstPubliDate"]),
    "FIN_F_ForecFin.csv": ("预告", ["PubliDate"]),
    "FIN_F_QuiTraSimFin.csv": ("快报", ["PubliDate"]),
    "FIN_F_QuiTraFin.csv": ("快报", ["PubliDate"]),
}


# ============================================================================
# 基础函数
# ============================================================================

def normalize_stock_code(series: pd.Series) -> pd.Series:
    """把证券代码统一为六位字符串，无法识别的值设为缺失。"""

    result = series.astype("string").str.strip()
    # 兼容部分软件把 000001 读成 1.0 的情况。
    result = result.str.replace(r"\.0+$", "", regex=True)
    valid = result.str.fullmatch(r"\d{1,6}", na=False)
    return result.where(valid).str.zfill(6)


def find_csv(filename: str) -> Path:
    """按文件名递归查找输入表，并检查是否存在重复文件。"""

    matches = sorted(DATA_DIR.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"找不到 {filename}，请检查目录：{DATA_DIR}")
    if len(matches) > 1:
        raise ValueError(f"找到多个 {filename}，无法判断应使用哪一个：{matches}")
    return matches[0]


def parse_optional_date(value: str | None, name: str) -> pd.Timestamp | None:
    """解析命令行中的可选日期。"""

    if value is None:
        return None
    parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{name} 必须为 YYYY-MM-DD，当前是：{value}")
    return parsed.normalize()


def expand_date_fields(record: dict, date_columns: list[str]) -> list[tuple[str, str]]:
    """
    把一条记录中的多个日期字段展开为（字段名，日期文本）列表。

    这个小函数也用于自检：旧版预告的一条记录应该展开为 PubliDate 和
    FirstPubliDate 两项，而不是只保留其中一个。
    """

    return [
        (column, str(record[column]).strip())
        for column in date_columns
        if record.get(column) is not None and str(record[column]).strip() != ""
    ]


# ============================================================================
# CSV 读取
# ============================================================================

def read_required_columns(path: Path, date_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    只读取构造日期所需的前置字段，避免坏引号的说明文字影响整表读取。

    当前下载文件中，部分“说明”字段含有不规范的逗号/引号。若直接用
    pandas.read_csv 读取所有列，可能出现字段数不一致并漏掉一条记录。
    四张表的 StockCode、AccPeriod、PubliDate 等必要字段都在前几列，
    因此逐行读取时只解析这些字段；说明文字的续行会被跳过并计入审计。
    """

    with path.open(encoding="utf-8-sig", newline="") as file:
        header_line = file.readline()
        if not header_line:
            raise ValueError(f"文件为空：{path}")
        header = next(csv.reader([header_line]))

        wanted = ["StockCode", "AccPeriod", *date_columns]
        if "Source" in header:
            wanted.append("Source")

        missing = [column for column in wanted if column not in header]
        if missing:
            raise ValueError(f"{path.name} 缺少字段：{missing}")

        positions = [header.index(column) for column in wanted]
        required_count = max(positions) + 1

        records: list[list[str]] = []
        continuation_lines = 0
        malformed_record_lines = 0

        for line_number, line in enumerate(file, start=2):
            # 真实数据记录均以六位证券代码字段开头；说明文字续行不满足此条件。
            if not re.match(r'^"\d{6}",', line):
                continuation_lines += 1
                continue

            try:
                parsed = next(csv.reader([line.rstrip("\r\n")], strict=False))
            except csv.Error as exc:
                raise ValueError(
                    f"{path.name} 第{line_number}行 CSV 格式无法解析：{exc}"
                ) from exc

            if len(parsed) < required_count:
                malformed_record_lines += 1
                raise ValueError(
                    f"{path.name} 第{line_number}行必要字段不足："
                    f"需要至少{required_count}列，实际{len(parsed)}列"
                )

            records.append([parsed[position] for position in positions] + [line_number])

    if not records:
        raise ValueError(f"{path} 未读取到有效记录")

    frame = pd.DataFrame(records, columns=[*wanted, "源文件行号"])
    audit = {
        "文件": str(path.relative_to(DATA_DIR)),
        "文件原始记录数": len(frame),
        "说明文字续行数": continuation_lines,
        "必要字段格式异常行数": malformed_record_lines,
    }
    return frame, audit


def load_one_table(
    filename: str,
    event_type: str,
    date_columns: list[str],
    forecast_source: str,
) -> tuple[pd.DataFrame, dict]:
    """读取一张表，展开日期字段，并返回明细和审计信息。"""

    path = find_csv(filename)
    raw, audit = read_required_columns(path, date_columns)

    raw = raw.rename(columns={"StockCode": "StkcdRecall"})
    raw["StkcdRecall"] = normalize_stock_code(raw["StkcdRecall"])
    raw["来源文件"] = str(path.relative_to(DATA_DIR))
    if "Source" not in raw.columns:
        raw["Source"] = ""

    # Source=0 是业绩预告，Source=1 是定期公告。
    # 默认两种来源均保留；只有显式指定 forecast-only 才过滤 Source=0。
    source_filtered = 0
    if filename == "FIN_F_ForecFin.csv" and forecast_source == "forecast-only":
        source = raw["Source"].astype("string").str.strip()
        source_filtered = int((source != "0").sum())
        raw = raw.loc[source.eq("0")].copy()

    invalid_code = raw["StkcdRecall"].isna()
    raw_valid = raw.loc[~invalid_code].copy()

    # melt（纵向展开）：旧版预告的两种日期各占一行；其他表每条记录一行。
    long = raw_valid.melt(
        id_vars=[
            "StkcdRecall", "AccPeriod", "Source",
            "来源文件", "源文件行号",
        ],
        value_vars=date_columns,
        var_name="日期字段",
        value_name="日期原值",
    )

    text = long["日期原值"].astype("string").str.strip()
    parsed_date = pd.to_datetime(
        text, format="%Y-%m-%d", errors="coerce"
    ).dt.normalize()

    # 非空但无法解析的日期不能静默当作“没有公告”。
    invalid_date = text.notna() & text.ne("") & parsed_date.isna()
    if invalid_date.any():
        examples = long.loc[
            invalid_date,
            ["源文件行号", "日期字段", "日期原值"],
        ].head(10)
        raise ValueError(f"{filename} 存在无法解析的日期：\n{examples}")

    long["公告自然日"] = parsed_date
    long["事件类别"] = event_type
    long = long.dropna(subset=["公告自然日"]).copy()

    audit.update(
        {
            "事件类别": event_type,
            "代码缺失或异常行数": int(invalid_code.sum()),
            "日期为空或无法使用行数": int(len(raw_valid) * len(date_columns) - len(long)),
            "来源过滤删除行数": source_filtered,
            "展开后有效公告日期数": len(long),
        }
    )
    return long, audit


# ============================================================================
# 日期映射和控制变量
# ============================================================================

def load_calendar(path: Path) -> pd.DataFrame:
    """读取交易日历，返回升序且不重复的 Trddt 日期。"""

    calendar = pd.read_csv(path, usecols=["Trddt"], encoding="utf-8-sig")
    calendar["Trddt"] = pd.to_datetime(
        calendar["Trddt"], errors="raise"
    ).dt.normalize()
    calendar = calendar.dropna().drop_duplicates().sort_values("Trddt")
    if calendar.empty:
        raise ValueError(f"交易日历为空：{path}")
    return calendar.reset_index(drop=True)


def map_to_trading_day(
    detail: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    将公告自然日映射到当天或下一交易日。

    超出交易日历范围的记录单独返回，不强行映射到日历首日或末日。
    """

    start = calendar["Trddt"].min()
    end = calendar["Trddt"].max()
    in_range = detail["公告自然日"].between(start, end)
    outside = detail.loc[~in_range].copy()

    inside = detail.loc[in_range].copy()
    if inside.empty:
        raise ValueError("所有公告日期都超出交易日历范围，无法生成控制变量")

    # direction="forward"：交易日记当天，周末/节假日顺延至下一交易日。
    mapped = pd.merge_asof(
        inside.sort_values("公告自然日"),
        calendar,
        left_on="公告自然日",
        right_on="Trddt",
        direction="forward",
        allow_exact_matches=True,
    )
    if mapped["Trddt"].isna().any():
        raise ValueError("存在公告日期无法映射到交易日")

    audit = {
        "交易日历开始": start.date().isoformat(),
        "交易日历结束": end.date().isoformat(),
        "日历范围内日期数": len(mapped),
        "日历范围外日期数": len(outside),
    }
    return mapped, outside, audit


def build_controls(mapped_detail: pd.DataFrame) -> pd.DataFrame:
    """按公司×交易日生成预告、快报及二者并集的0/1变量。"""

    if mapped_detail.empty:
        raise ValueError("没有可用的公告明细")

    # 同一公司同一交易日同一类别，无论报告期、修正次数或来源如何，只记1次。
    daily = mapped_detail[
        ["StkcdRecall", "Trddt", "事件类别"]
    ].drop_duplicates()
    daily["发生事件"] = 1

    controls = daily.pivot_table(
        index=["StkcdRecall", "Trddt"],
        columns="事件类别",
        values="发生事件",
        aggfunc="max",
        fill_value=0,
    ).reindex(columns=["预告", "快报"], fill_value=0)

    controls = controls.rename(
        columns={
            "预告": "ownForecast",       # 当日是否有业绩预告
            "快报": "ownQuickReport",    # 当日是否有业绩快报
        }
    ).astype("int8")
    controls["ownPerformanceForecastBrief"] = controls.max(axis=1).astype("int8")

    result = controls.reset_index().rename_axis(columns=None)
    if result.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("控制表在证券代码×交易日层面不唯一")
    return result.sort_values(["StkcdRecall", "Trddt"]).reset_index(drop=True)


def apply_research_window(
    mapped_detail: pd.DataFrame,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    """在完成非交易日顺延后，再按研究期间筛选。"""

    result = mapped_detail
    if start_date is not None:
        result = result.loc[result["Trddt"] >= start_date]
    if end_date is not None:
        result = result.loc[result["Trddt"] <= end_date]
    return result.copy()


# ============================================================================
# 主流程
# ============================================================================

def run(
    start_date: str | None = None,
    end_date: str | None = None,
    forecast_source: str = "all",
    calendar_path: Path = CALENDAR_PATH,
    save_diagnostics: bool = False,  # 是否额外保存明细、审计和范围外记录
) -> None:
    """读取四张表、映射交易日并写出明细、控制变量和审计文件。"""

    start = parse_optional_date(start_date, "--start-date")
    end = parse_optional_date(end_date, "--end-date")
    if start is not None and end is not None and start > end:
        raise ValueError("--start-date 不能晚于 --end-date")

    detail_parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = []

    for filename, (event_type, date_columns) in TABLE_SPECS.items():
        print(f"正在读取：{filename}（{event_type}）")
        part, audit = load_one_table(
            filename,
            event_type,
            date_columns,
            forecast_source,
        )
        detail_parts.append(part)
        audit_rows.append(audit)

    detail = pd.concat(detail_parts, ignore_index=True)
    if detail.empty:
        raise ValueError("四张表没有可用的公告日期")

    calendar = load_calendar(calendar_path)
    mapped_detail, outside, date_audit = map_to_trading_day(detail, calendar)

    # 先映射交易日，再按研究窗口筛选；避免研究开始日是周末时漏掉顺延公告。
    mapped_for_window = apply_research_window(mapped_detail, start, end)
    if mapped_for_window.empty:
        raise ValueError("研究期间内没有可用公告，请检查起止日期")

    controls = build_controls(mapped_for_window)

    # 明细中同时保留公告自然日、映射后的交易日、报告期、日期字段和来源。
    detail_columns = [
        "StkcdRecall", "AccPeriod", "Source", "来源文件", "源文件行号",
        "日期字段", "日期原值", "公告自然日", "Trddt", "事件类别",
    ]
    mapped_detail = mapped_for_window[detail_columns].sort_values(
        ["StkcdRecall", "Trddt", "事件类别", "公告自然日"]
    )

    # 审计表把研究窗口筛选和交易日历范围外记录明确记录下来。
    audit_rows.append(
        {
            "文件": "合计",
            "事件类别": "预告+快报",
            "文件原始记录数": "",
            "说明文字续行数": "",
            "必要字段格式异常行数": "",
            "代码缺失或异常行数": "",
            "日期为空或无法使用行数": "",
            "来源过滤删除行数": "",
            "展开后有效公告日期数": len(detail),
            "研究窗口筛选后日期数": len(mapped_for_window),
            **date_audit,
        }
    )
    audit = pd.DataFrame(audit_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 控制变量是回归必需文件，因此始终生成
    controls.to_csv(
        CONTROL_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n生成完成。")
    print(f"公司—交易日行数：{len(controls):,}")
    print(f"控制变量：{CONTROL_OUTPUT_PATH}")

    # 以下三个文件仅在显式指定参数时生成
    if save_diagnostics:
        mapped_detail.to_csv(
            DETAIL_OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        audit.to_csv(
            AUDIT_OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        outside.to_csv(
            OUTSIDE_OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"公告明细：{DETAIL_OUTPUT_PATH}")
        print(f"审计表：{AUDIT_OUTPUT_PATH}")
        print(f"交易日历范围外记录：{OUTSIDE_OUTPUT_PATH}")
    else:
        print(
            "未生成公告明细、审计和交易日历范围外文件；"
            "如需生成，请增加 --save-diagnostics 参数。"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成业绩预告/业绩快报的公司×交易日控制变量"
    )
    parser.add_argument(
        "--start-date",
        help="研究开始日期，格式 YYYY-MM-DD；在交易日映射后筛选",
    )
    parser.add_argument(
        "--end-date",
        help="研究结束日期，格式 YYYY-MM-DD；在交易日映射后筛选",
    )
    parser.add_argument(
        "--forecast-source",
        choices=["all", "forecast-only"],
        default="all",
        help="新版业绩预告来源：all保留Source=0/1；forecast-only只保留Source=0",
    )
    parser.add_argument(
        "--calendar",
        type=Path,
        default=CALENDAR_PATH,
        help="交易日历CSV路径，默认使用项目中的市场交易日历",
    )
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="额外生成明细、审计和交易日历范围外文件；默认不生成",
    )
    args = parser.parse_args()

    run(
        start_date=args.start_date,
        end_date=args.end_date,
        forecast_source=args.forecast_source,
        calendar_path=args.calendar,
        save_diagnostics=args.save_diagnostics,
    )


if __name__ == "__main__":
    main()
