"""从国泰安指数取样变更记录还原沪深300历史成分区间。

直接读取原始变更文件，不依赖临时目录、旧区间表或其他脚本。
一行输出代表一只股票的一段连续在组期间：纳入日含、剔除日不含。
截至2025年12月31日仍在组的股票，结束日期留空。

初始和期末成员数按当前数据应为300；中途不强制恰好300。
例如原始记录在2009年12月29日调出两只，2010年1月4日净增两只，
故这段期间应保留298只的状态，不移动日期、不自行补成员。
变量名使用英文，注释和报错说明使用中文。
"""

from pathlib import Path

import pandas as pd


# 一、输入、输出和日期范围。
ROOT = Path(__file__).resolve().parents[4]
INPUT_DIR = ROOT / "input/CSMAR/指数取样变更文件(2002年-2025年)"
OUTPUT_PATH = ROOT / "中间结果/02_变量构造结果/9指数成分股/沪深300/沪深300_历史成分区间.csv"
INDEX_CODE = "000300"
START_DATE = pd.Timestamp("2005-04-08")
END_DATE = pd.Timestamp("2025-12-31")
EXPECTED_COUNT = 300

if OUTPUT_PATH.exists():
    raise FileExistsError(f"结果已存在，请先核对，避免覆盖：{OUTPUT_PATH}")


# 二、读取各卷变更记录，按指数代码选取沪深300。
# 01为变更日期，02股票代码，03股票简称，04新增或剔除，
# 05证券类型，06公告日期，07交易市场；成员资格使用变更日期。
source_cols = ["Indexcd"] + [f"Chgsmp{i:02d}" for i in range(1, 8)]
input_paths = sorted(INPUT_DIR.rglob("IDX_Chgsmp*.csv"))
if not input_paths:
    raise FileNotFoundError(f"找不到国泰安指数取样变更文件：{INPUT_DIR}")

parts = []
for path in input_paths:
    for chunk in pd.read_csv(path, usecols=source_cols, dtype="string",
                             encoding="utf-8-sig", chunksize=200_000):
        index_codes = chunk["Indexcd"].str.strip().str.zfill(6)
        selected = chunk.loc[index_codes.eq(INDEX_CODE)].copy()
        if not selected.empty:
            selected["SourceFile"] = str(path.relative_to(ROOT))
            parts.append(selected)
if not parts:
    raise ValueError("原始文件中没有沪深300指数000300的变更记录。")

changes = pd.concat(parts, ignore_index=True).rename(columns={
    "Chgsmp01": "ChangeDate", "Chgsmp02": "Stkcd", "Chgsmp03": "StockName",
    "Chgsmp04": "ChangeType", "Chgsmp05": "SecurityType",
    "Chgsmp06": "AnnouncementDate", "Chgsmp07": "MarketCode",
})
del parts, chunk, selected
for column in ["Indexcd", "Stkcd", "ChangeType", "SecurityType", "MarketCode"]:
    changes[column] = changes[column].str.strip()
changes["Indexcd"] = changes["Indexcd"].str.zfill(6)
changes["Stkcd"] = changes["Stkcd"].str.zfill(6)
if not changes["Stkcd"].str.fullmatch(r"\d{6}", na=False).all():
    raise ValueError("成分股票代码缺失或不是六位数字。")
if not changes["SecurityType"].eq("1").fillna(False).all():
    raise ValueError("沪深300记录出现非股票类证券或缺失证券类型，请核对。")
if not changes["ChangeType"].isin(["1", "2"]).all():
    raise ValueError("变动方式只能是1（新增）或2（剔除）。")
if not changes["MarketCode"].isin(["SSE", "SZSE"]).all():
    raise ValueError("沪深300变更记录出现上海、深圳以外的市场，请核对。")
for column in ["ChangeDate", "AnnouncementDate"]:
    changes[column] = pd.to_datetime(changes[column], format="%Y-%m-%d", errors="raise")
if changes["ChangeDate"].isna().any():
    raise ValueError("存在缺失的变更生效日期。")


# 三、合并内容一致的跨卷重复；冲突记录必须先核查。
event_cols = ["Indexcd", "Stkcd", "ChangeDate", "ChangeType"]
duplicated = changes.duplicated(event_cols, keep=False)
if duplicated.any():
    detail_cols = ["StockName", "SecurityType", "AnnouncementDate", "MarketCode"]
    versions = changes.loc[duplicated].groupby(event_cols, dropna=False)[detail_cols].nunique(dropna=False)
    if versions.gt(1).any().any():
        raise ValueError("同一股票、日期和变动方式存在不同原始版本，请核对各卷数据。")
    print(f"合并内容一致的跨卷重复记录：{changes.duplicated(event_cols).sum():,}条。", flush=True)
    changes = changes.drop_duplicates(event_cols).copy()
if changes.duplicated(["Stkcd", "ChangeDate"]).any():
    raise ValueError("同一股票同一天既新增又剔除，无法直接确定区间，请核对原始记录。")

changes = changes.loc[changes["ChangeDate"].le(END_DATE)].copy()
changes = changes.sort_values(["ChangeDate", "ChangeType", "Stkcd"]).reset_index(drop=True)
if changes.empty or changes["ChangeDate"].min() != START_DATE:
    raise ValueError("缺少从2005年4月8日开始的沪深300完整变更历史。")
initial = changes.loc[changes["ChangeDate"].eq(START_DATE)]
if len(initial) != EXPECTED_COUNT or not initial["ChangeType"].eq("1").all():
    raise ValueError("首日记录应为300只不同股票的新增记录，请核对原始文件。")
print(f"已读取沪深300变更记录：{len(changes):,}条。", flush=True)


# 四、逐个生效日期还原成员。字典保存“股票代码：本次纳入日”。
# 同日调整先调出再调入，保留原始日期。中间数量变化仅报告，不强制凑齐300。
active = {}
interval_rows = []
count_rows = []
for date, group in changes.groupby("ChangeDate", sort=True):
    additions = group.loc[group["ChangeType"].eq("1"), "Stkcd"].tolist()
    removals = group.loc[group["ChangeType"].eq("2"), "Stkcd"].tolist()
    missing_members = set(removals) - set(active)
    existing_members = set(additions) & set(active)
    if missing_members or existing_members:
        raise ValueError(
            f"{date:%Y-%m-%d}变更衔接异常：剔除非成员{sorted(missing_members)}；"
            f"重复纳入{sorted(existing_members)}。请核对原始记录。"
        )
    for stock in removals:
        interval_rows.append({
            "Stkcd": stock, "start_date": active.pop(stock),
            "end_date": date, "StillMemberAtEnd": 0,
        })
    for stock in additions:
        active[stock] = date
    count_rows.append({"start_date": date, "member_count": len(active)})

# 列出中途数量不是300的区间，结束日期不包含在该区间内。
counts = pd.DataFrame(count_rows)
counts["end_date"] = counts["start_date"].shift(-1).fillna(END_DATE + pd.Timedelta(days=1))
unusual_counts = counts.loc[counts["member_count"].ne(EXPECTED_COUNT)]
if not unusual_counts.empty:
    print("以下期间按原始变更记录还原出的成员数不是300，已保留该状态：", flush=True)
    print(unusual_counts[["start_date", "end_date", "member_count"]].rename(columns={
        "start_date": "起始日（含）", "end_date": "结束日（不含）", "member_count": "成员数",
    }).to_string(index=False), flush=True)

for stock, in_date in active.items():
    interval_rows.append({
        "Stkcd": stock, "start_date": in_date,
        "end_date": pd.NaT, "StillMemberAtEnd": 1,
    })


# 五、检查区间，并保持旧区间表的六个输出字段及排列顺序。
intervals = pd.DataFrame(interval_rows).sort_values(["Stkcd", "start_date"]).reset_index(drop=True)
intervals["Indexcd"] = INDEX_CODE
intervals["CoverageEndDate"] = END_DATE
intervals = intervals[["Stkcd", "start_date", "end_date", "StillMemberAtEnd", "Indexcd", "CoverageEndDate"]]
if intervals.duplicated(["Stkcd", "start_date"]).any():
    raise ValueError("存在重复的股票和纳入日期。")
closed = intervals["end_date"].notna()
if not intervals.loc[closed, "start_date"].lt(intervals.loc[closed, "end_date"]).all():
    raise ValueError("存在纳入日不早于剔除日的区间。")
previous_end = intervals.groupby("Stkcd")["end_date"].shift()
if (intervals["start_date"] < previous_end).any():
    raise ValueError("同一股票的连续在组区间发生重叠。")
if len(intervals) != int(changes["ChangeType"].eq("1").sum()):
    raise ValueError("区间条数与原始新增事件数不一致。")
if int(intervals["StillMemberAtEnd"].sum()) != EXPECTED_COUNT:
    raise ValueError("覆盖期末开放区间数量不是300条，请核对期末变更记录。")


# 六、检查通过后保存正式区间表。
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
intervals.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
print(f"沪深300历史成分区间已保存：{OUTPUT_PATH}", flush=True)
print(f"区间数{len(intervals):,}；股票数{intervals['Stkcd'].nunique():,}；期末成员数{len(active)}。", flush=True)
print("区间纳入日包含、剔除日不包含；开放区间只适用于数据覆盖截止日以内。", flush=True)
