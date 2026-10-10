"""
合并 CSMAR“专题研究系列—事件研究”中的十类非定期报告事件。

本脚本的口径：
1. 每张表只使用字段说明中明确标注为“公告日期”的字段。
2. 同一条记录有多个公告日期时，把每个日期展开为一条公告明细，避免漏掉后续公告；
   同时按事件表配置标记首次公告日期。
3. 公告自然日若为交易日，事件记在当天；若为周末或节假日，顺延到下一个交易日。
4. 最终控制表是一行“被提示公司 × 交易日”，两种总体口径分别保存：
   ownOtherEventER（所有有效公告日期）和 ownOtherEventFirstER（仅首次公告日期）。
   两个控制表均可用 validate="m:1" 合并到日度主面板。

代码顶部的 OUTPUT_MODE 是输出开关，也可在运行时用 --output-mode 覆盖：
- all：只输出所有有效公告日期口径；
- first：只输出首次公告日期口径；
- both：同时输出两个文件（默认）。

两个主控制文件分别为：
- 专题研究事件控制变量_公告日期口径.csv：所有有效公告日期口径；
- 专题研究事件控制变量_首次公告日期口径.csv：仅首次公告日期口径。

两个文件都是事件日稀疏表；合并到完整日度主面板后，应将未匹配的控制变量填为0。

只有运行时增加 --save-diagnostics 参数，
才额外生成：
- 专题研究事件明细_公告日期口径.csv：保留每个日期字段和来源；
- 专题研究事件控制变量_公告日期口径_审计.csv：缺失代码、缺失日期和日期范围检查。

脚本不会把“事件发生日期”“签署日期”“审批日期”“股权登记日”等日期误当作公告日期。
研究期间可以运行时通过 --start-date 和 --end-date 指定；不指定时保留交易日历覆盖范围内的全部日期。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# 路径设置
# -----------------------------------------------------------------------------

ROOT = Path("/Users/qiyuewu/Desktop/mmm复现数据包")
EVENT_DIR = ROOT / "input/CSMAR/专题研究系列-事件研究"
CALENDAR_PATH = ROOT / "中间结果/01_清洗后样本/市场交易日历_td.csv"
OUTPUT_DIR = ROOT / "中间结果/02_变量构造结果/5公司公告事件"

DETAIL_OUTPUT_PATH = OUTPUT_DIR / "专题研究事件明细_公告日期口径.csv"
ALL_DATES_CONTROL_OUTPUT_PATH = OUTPUT_DIR / "专题研究事件控制变量_公告日期口径.csv"
FIRST_DATE_CONTROL_OUTPUT_PATH = OUTPUT_DIR / "专题研究事件控制变量_首次公告日期口径.csv"
AUDIT_OUTPUT_PATH = OUTPUT_DIR / "专题研究事件控制变量_公告日期口径_审计.csv"

# 输出开关："all"=仅所有日期，"first"=仅首次日期，"both"=两个文件都输出。
# 命令行参数 --output-mode 可以临时覆盖这里的设置。
OUTPUT_MODE = "both"
OUTPUT_MODE_CHOICES = ("all", "first", "both")


# -----------------------------------------------------------------------------
# 事件表配置
#
# 每个表的 date_cols 都是“公告披露日期”字段；first_date_col 是该类事件用于
# “仅首次公告日期”口径的首次公告字段。不要把同一表中的实际发生日期、签署日期、
# 审核日期、登记日期等混入这里，否则控制变量会提前或滞后到错误日期。
# -----------------------------------------------------------------------------

EVENT_SPECS = {
    "ER_MA": {
        "label": "兼并收购",
        "dummy": "ownEventER_MA",
        "date_cols": [
            "FirstDeclareDate",   # 首次公告日期
            "LatestDeclareDate",  # 最新公告日期
            "FinishDeclareDate",  # 完成公告日期
        ],
        "first_date_col": "FirstDeclareDate",
    },
    "ER_Private": {
        "label": "定向增发",
        "dummy": "ownEventER_Private",
        "date_cols": [
            "DeclareDate",        # 预案公告日
            "ListedDeclareDate",  # 上市公告书公布日期
        ],
        "first_date_col": "DeclareDate",
    },
    "ER_PublicOffering": {
        "label": "公开增发",
        "dummy": "ownEventER_PublicOffering",
        "date_cols": [
            "DeclareDate",        # 预案公告日
            "SEO1DeclareDate",    # 招股意向书公告日期
            "ListedDeclareDate",  # 上市公告书公布日期
        ],
        "first_date_col": "DeclareDate",
    },
    "ER_RightsIssue": {
        "label": "配股",
        "dummy": "ownEventER_RightsIssue",
        "date_cols": [
            "DeclareDate",        # 预案公告日
            "DeclareDate2",       # 配股说明书公告日期
            "ListedDeclareDate",  # 上市公告书公布日期
        ],
        "first_date_col": "DeclareDate",
    },
    "ER_ProfitSharing": {
        "label": "红利分配",
        "dummy": "ownEventER_ProfitSharing",
        "date_cols": [
            "DeclareDate",               # 预案公告日期
            "HoldersMeetingDeclareDate", # 股东大会公告发布日期
            "ImplementationDate",        # 方案实施公告日
        ],
        "first_date_col": "DeclareDate",
    },
    "ER_LADetail": {
        "label": "诉讼仲裁",
        "dummy": "ownEventER_LADetail",
        "date_cols": ["DeclareDate"],   # 公告日期
        "first_date_col": "DeclareDate",
    },
    "ER_Guarantee": {
        "label": "对外担保",
        "dummy": "ownEventER_Guarantee",
        "date_cols": ["DeclareDate"],   # 公告日期
        "first_date_col": "DeclareDate",
    },
    "ER_Violation": {
        "label": "违规",
        "dummy": "ownEventER_Violation",
        "date_cols": ["DeclareDate"],   # 公告日期
        "first_date_col": "DeclareDate",
    },
    "ER_Structure": {
        "label": "股本变动",
        "dummy": "ownEventER_Structure",
        "date_cols": ["DeclareDate"],   # 公告日期
        "first_date_col": "DeclareDate",
    },
    "ER_Schange": {
        "label": "董事长总经理变更",
        "dummy": "ownEventER_Schange",
        "date_cols": ["DeclareDate"],   # 公告日期
        "first_date_col": "DeclareDate",
    },
}


def normalize_stock_code(series: pd.Series) -> pd.Series:
    """将股票代码统一为六位字符串；空值或非数字代码设为缺失。"""

    s = series.astype("string").str.strip()
    # 兼容部分软件把代码读成 1.0、2.0 的情况。
    s = s.str.replace(r"\.0+$", "", regex=True)
    valid = s.str.fullmatch(r"\d{1,6}", na=False)
    return s.where(valid).str.zfill(6)


def find_event_csv(table_stem: str) -> Path:
    """按文件名查找事件表，避免依赖带中文日期范围的文件夹名称。"""

    matches = sorted(EVENT_DIR.rglob(f"{table_stem}.csv"))
    if not matches:
        raise FileNotFoundError(
            f"找不到 {table_stem}.csv，请检查目录：{EVENT_DIR}"
        )
    if len(matches) > 1:
        raise ValueError(f"找到多个 {table_stem}.csv，无法判断应使用哪一个：{matches}")
    return matches[0]


def load_calendar(calendar_path: Path) -> pd.DataFrame:
    """读取交易日历，并统一为日期（不含时分秒）。"""

    calendar = pd.read_csv(
        calendar_path,
        usecols=["Trddt"],
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


def read_one_event_table(table_stem: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    """读取一张事件表，并把多个公告日期字段展开成长表。"""

    path = find_event_csv(table_stem)
    date_cols = spec["date_cols"]
    first_date_col = spec["first_date_col"]
    if first_date_col not in date_cols:
        raise ValueError(
            f"{table_stem} 的首次公告字段 {first_date_col} 不在公告日期字段列表中。"
        )
    usecols = ["EventID", "Symbol", *date_cols]

    raw = pd.read_csv(
        path,
        usecols=usecols,
        dtype="string",
        encoding="utf-8-sig",
        low_memory=False,
    )

    raw["StkcdRecall"] = normalize_stock_code(raw["Symbol"])

    for col in date_cols:
        raw[col] = pd.to_datetime(
            raw[col], format="%Y-%m-%d", errors="coerce"
        ).dt.normalize()

    invalid_code = raw["StkcdRecall"].isna()
    no_valid_date = raw[date_cols].isna().all(axis=1)

    # melt（纵向展开）后，一条原始记录的每个有效公告日期各占一行。
    # 例如兼并收购一条记录的首次、最新、完成公告日会分别保留。
    long = raw.loc[~invalid_code].melt(
        id_vars=["StkcdRecall", "EventID"],
        value_vars=date_cols,
        var_name="日期字段",
        value_name="公告自然日",
    ).dropna(subset=["公告自然日"])
    long["是否首次公告"] = long["日期字段"].eq(first_date_col).astype("int8")

    long["事件类型"] = spec["label"]
    long["事件表"] = table_stem
    long["来源文件"] = str(path.relative_to(EVENT_DIR))

    # 同一个事件编号可能在表内重复出现，但只删除完全相同的日期记录。
    # 不能仅按 EventID 去重，因为诉讼等事件的后续公告可能共用同一个 EventID。
    long = long.drop_duplicates(
        ["事件表", "StkcdRecall", "EventID", "日期字段", "公告自然日"]
    )

    valid_code_raw = raw.loc[~invalid_code]

    audit = {
        "事件表": table_stem,
        "事件类型": spec["label"],
        "来源文件": str(path.relative_to(EVENT_DIR)),
        "首次公告日期字段": first_date_col,
        "原始行数": len(raw),
        "代码缺失或异常行数": int(invalid_code.sum()),
        "所有公告日期均缺失或无效行数": int(no_valid_date.sum()),
        "首次公告日期缺失或无效行数": int(valid_code_raw[first_date_col].isna().sum()),
        "展开后有效公告日期行数": len(long),
        "展开后首次公告日期行数": int(long["是否首次公告"].sum()),
    }
    return long, audit


def map_to_next_trading_day(
    events: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """把公告自然日映射到当天或下一个交易日，并返回日期范围审计。"""

    calendar_start = calendar["Trddt"].min()
    calendar_end = calendar["Trddt"].max()

    events = events.copy()
    in_calendar = events["公告自然日"].between(calendar_start, calendar_end)
    outside_count = int((~in_calendar).sum())
    events = events.loc[in_calendar].copy()

    if events.empty:
        raise ValueError("公告日期全部落在交易日历范围之外，无法生成控制变量。")

    # 只按日期做 merge_asof（向后寻找最近交易日）。
    # direction="forward" 表示非交易日顺延，allow_exact_matches=True
    # 表示交易日公告保留当天。
    events = pd.merge_asof(
        events.sort_values("公告自然日"),
        calendar,
        left_on="公告自然日",
        right_on="Trddt",
        direction="forward",
        allow_exact_matches=True,
    )

    if events["Trddt"].isna().any():
        raise ValueError("存在公告日期无法映射到交易日，请检查交易日历。")

    events["date"] = events["Trddt"]
    audit = {
        "交易日历开始": calendar_start.date().isoformat(),
        "交易日历结束": calendar_end.date().isoformat(),
        "日历范围外公告日期行数": outside_count,
    }
    return events, audit


def build_controls(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成明细表、所有日期控制表和首次日期控制表。"""

    # 明细表保留来源和日期字段，之后发现误识别时可以追溯到原始记录。
    detail_cols = [
        "StkcdRecall",
        "Trddt",
        "date",
        "公告自然日",
        "事件表",
        "事件类型",
        "日期字段",
        "是否首次公告",
        "EventID",
        "来源文件",
    ]
    detail = (
        events[detail_cols]
        .sort_values(["StkcdRecall", "Trddt", "事件表", "EventID", "日期字段"])
        .reset_index(drop=True)
    )

    # 同一公司同一交易日同一事件类型无论出现几条，只记为1。
    firm_day_type = (
        events[["StkcdRecall", "Trddt", "事件类型"]]
        .drop_duplicates()
        .assign(发生事件=1)
    )

    wide = firm_day_type.pivot_table(
        index=["StkcdRecall", "Trddt"],
        columns="事件类型",
        values="发生事件",
        aggfunc="max",
        fill_value=0,
    )

    # 按配置顺序生成十类事件虚拟变量。没有记录的类别补0。
    labels = []
    dummy_cols = []
    for spec in EVENT_SPECS.values():
        label = spec["label"]
        dummy = spec["dummy"]
        labels.append(label)
        dummy_cols.append(dummy)
        if label in wide.columns:
            wide[dummy] = wide[label].astype("int8")
        else:
            wide[dummy] = 0

    # ownOtherEventER（十类其他事件）是所有有效公告日期的十个类别变量并集。
    # 该变量保留原有口径，可作为主回归的笼统控制变量。
    wide["ownOtherEventER"] = wide[dummy_cols].max(axis=1).astype("int8")

    # ownOtherEventFirstER 只使用每类事件配置的首次公告日期。
    # 该变量用于“仅首次公告日期”口径的稳健性分析；不以其他日期替代缺失的首次日期。
    first_day = (
        events.loc[events["是否首次公告"].eq(1), ["StkcdRecall", "Trddt"]]
        .drop_duplicates()
        .assign(ownOtherEventFirstER=1)
        .set_index(["StkcdRecall", "Trddt"])
    )
    wide = wide.join(first_day, how="left")
    wide["ownOtherEventFirstER"] = (
        wide["ownOtherEventFirstER"].fillna(0).astype("int8")
    )

    if (wide["ownOtherEventFirstER"] > wide["ownOtherEventER"]).any():
        raise ValueError("仅首次公告日期口径不能超出所有有效公告日期口径。")

    # 统计同一公司—交易日涉及的不同事件编号，不把同一事件的多个日期字段重复计数。
    event_day_unique = events.drop_duplicates(
        ["StkcdRecall", "Trddt", "事件表", "EventID"]
    )
    event_counts = (
        event_day_unique.groupby(["StkcdRecall", "Trddt"], as_index=True)
        .size()
        .rename("ownOtherEventERCount")
    )
    wide = wide.join(event_counts, how="left")
    wide["ownOtherEventERCount"] = (
        wide["ownOtherEventERCount"].fillna(0).astype("int32")
    )

    # 先生成内部合并表，用于检查两个口径之间的包含关系。
    combined_controls = wide.reset_index()
    combined_controls = combined_controls[
        [
            "StkcdRecall",
            "Trddt",
            *dummy_cols,
            "ownOtherEventER",
            "ownOtherEventFirstER",
            "ownOtherEventERCount",
        ]
    ].sort_values(["StkcdRecall", "Trddt"]).reset_index(drop=True)

    if combined_controls.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("控制表在 StkcdRecall—Trddt 层面不唯一。")

    for column in ["ownOtherEventER", "ownOtherEventFirstER"]:
        if not combined_controls[column].isin([0, 1]).all():
            raise ValueError(f"{column} 不是二元变量。")

    # 所有日期文件保留原有列结构，避免影响既有合并代码。
    all_date_controls = combined_controls[
        [
            "StkcdRecall",
            "Trddt",
            *dummy_cols,
            "ownOtherEventER",
            "ownOtherEventERCount",
        ]
    ].copy()

    # 首次日期文件只保留真正发生首次公告的公司—交易日。
    # 合并到完整日度面板后，未匹配日期应将 ownOtherEventFirstER 填为0。
    first_date_controls = (
        combined_controls.loc[
            combined_controls["ownOtherEventFirstER"].eq(1),
            ["StkcdRecall", "Trddt", "ownOtherEventFirstER"],
        ]
        .sort_values(["StkcdRecall", "Trddt"])
        .reset_index(drop=True)
    )

    if first_date_controls.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("首次公告日期控制表在 StkcdRecall—Trddt 层面不唯一。")

    return detail, all_date_controls, first_date_controls


def parse_optional_date(value: str | None, argument_name: str) -> pd.Timestamp | None:
    """解析可选的研究起止日期。"""

    if value is None:
        return None
    parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{argument_name} 必须是 YYYY-MM-DD 格式，当前值为：{value}")
    return parsed.normalize()


def run(
    start_date: str | None = None,
    end_date: str | None = None,
    save_diagnostics: bool = False,  # 是否额外保存明细和审计文件
    output_mode: str = OUTPUT_MODE,  # all、first 或 both
) -> None:
    """读取十类事件、生成结果并写入配置好的中间结果目录。"""

    if output_mode not in OUTPUT_MODE_CHOICES:
        raise ValueError(
            f"output_mode 必须是 {OUTPUT_MODE_CHOICES} 之一，当前值为：{output_mode}"
        )

    calendar = load_calendar(CALENDAR_PATH)
    events_parts = []
    audit_parts = []

    for table_stem, spec in EVENT_SPECS.items():
        print(f"正在读取：{table_stem}（{spec['label']}）")
        part, audit = read_one_event_table(table_stem, spec)
        events_parts.append(part)
        audit_parts.append(audit)

    events = pd.concat(events_parts, ignore_index=True)
    if events.empty:
        raise ValueError("十类事件表没有可用的公司代码和公告日期。")

    events, date_audit = map_to_next_trading_day(events, calendar)

    # 研究期间按映射后的交易日筛选。这样周末公告顺延到研究期第一天时不会被误删。
    start = parse_optional_date(start_date, "--start-date")
    end = parse_optional_date(end_date, "--end-date")
    if start is not None and end is not None and start > end:
        raise ValueError("--start-date 不能晚于 --end-date。")

    before_window = len(events)
    first_before_window = int(events["是否首次公告"].sum())
    if start is not None:
        events = events.loc[events["Trddt"] >= start].copy()
    if end is not None:
        events = events.loc[events["Trddt"] <= end].copy()

    audit_parts.append(
        {
            "事件表": "合计",
            "事件类型": "十类事件",
            "来源文件": "",
            "首次公告日期字段": "按事件表配置",
            "原始行数": "",
            "代码缺失或异常行数": "",
            "所有公告日期均缺失或无效行数": "",
            "首次公告日期缺失或无效行数": "",
            "展开后有效公告日期行数": before_window,
            "展开后首次公告日期行数": first_before_window,
            "研究期间筛选后行数": len(events),
            "研究期间筛选后首次公告日期行数": int(events["是否首次公告"].sum()),
            **date_audit,
        }
    )
    audit = pd.DataFrame(audit_parts)

    if events.empty:
        raise ValueError("研究期间内没有可用事件，请检查 --start-date/--end-date。")

    detail, all_date_controls, first_date_controls = build_controls(events)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n生成完成。")
    print(f"输出模式：{output_mode}")

    if output_mode in {"all", "both"}:
        all_date_controls.to_csv(
            ALL_DATES_CONTROL_OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"所有公告日期口径行数：{len(all_date_controls):,}")
        print(f"所有公告日期口径：{ALL_DATES_CONTROL_OUTPUT_PATH}")

    if output_mode in {"first", "both"}:
        first_date_controls.to_csv(
            FIRST_DATE_CONTROL_OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"首次公告日期口径行数：{len(first_date_controls):,}")
        print(f"首次公告日期口径：{FIRST_DATE_CONTROL_OUTPUT_PATH}")

    # 明细和审计文件只有显式指定参数时才生成
    if save_diagnostics:
        detail.to_csv(
            DETAIL_OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        audit.to_csv(
            AUDIT_OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"公告明细：{DETAIL_OUTPUT_PATH}")
        print(f"审计表：{AUDIT_OUTPUT_PATH}")
    else:
        print("未生成公告明细和审计文件；如需生成，请增加 --save-diagnostics 参数。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并专题研究系列事件研究中的十类非定期报告事件"
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="可选，研究期间开始日期，格式 YYYY-MM-DD；不填则不设下限",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="可选，研究期间结束日期，格式 YYYY-MM-DD；不填则不设上限",
    )
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="额外生成公告明细和审计文件；默认不生成",
    )
    parser.add_argument(
        "--output-mode",
        choices=OUTPUT_MODE_CHOICES,
        default=OUTPUT_MODE,
        help=(
            "输出开关：all=只输出所有日期口径，first=只输出首次日期口径，"
            "both=同时输出两个文件；默认读取代码顶部的 OUTPUT_MODE"
        ),
    )
    args = parser.parse_args()

    run(
        start_date=args.start_date,
        end_date=args.end_date,
        save_diagnostics=args.save_diagnostics,
        output_mode=args.output_mode,
    )


if __name__ == "__main__":
    main()
