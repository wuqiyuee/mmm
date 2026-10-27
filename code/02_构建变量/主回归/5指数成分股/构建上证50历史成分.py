"""从国泰安指数取样变更文件还原上证50历史成分。

处理顺序：读取各卷变更记录 → 校正已核实的日期 → 还原成员区间 → 展开交易日。
只使用指数代码000016，不把成份证券代码000016误当作指数代码筛选。

输入：
1. input/CSMAR/指数取样变更文件(2002年-2025年)/各卷/IDX_Chgsmp*.csv
2. 中间结果/01_清洗后样本/市场交易日历_td.csv

输出目录：中间结果/02_变量构造结果/9指数成分股/上证50/
1. 上证50_校正后成分变更记录.csv：每行是一只股票的一次调入或调出。
2. 上证50_历史成分区间.csv：每行是一次连续在组区间，纳入日含、剔除日不含。
3. 上证50_交易日成分名单.csv：每行是一只成分股票在一个市场交易日的记录。

核心字段：
- Stkcd：成分股票代码；后续用于匹配提示公司，不是被提示公司StkcdRecall。
- ChangeDateRaw / ChangeDate：原始变更日期 / 本脚本实际使用的生效日期。
- AnnouncementDate：变更公告日，仅用于追溯，不作为成分生效日。
- InDate / OutDate：纳入日 / 真实剔除日；样本截止仍在组内者的剔除日留空。
- StillMemberAtEnd：截至数据覆盖期末是否仍为成分股，1是、0否。
- CoverageEndDate：区间表数据覆盖截止日，不得将开放区间外推至该日之后。
- Trddt / td：市场交易日期 / 项目统一交易日序号。
- is_sse50：当日属于上证50，日度输出中恒为1。

口径备注：
1. 保留原始输入文件不动，只在内存中校正广州控股600098的剔除日期。
2. 按变更生效日判断成员，非交易日变更自然在下一个市场交易日体现。
   不以公告日提前纳入，也不对已有交易日生效日期额外后移。
3. 成员名单不因个股停牌、异常状态或行业而另行删减；这些属于后续回归筛选。
4. 日表只有正成员。只有在本表日期覆盖期内，左合并后未匹配才可补0。
5. 不读取或修改最终回归面板，不计算提示变量，也不自动估计回归。
6. 重新读取输出时，股票代码和指数代码应按字符串读取，保留前导0。
"""

from pathlib import Path

import pandas as pd


# 第一部分：输入、输出和日期设置
ROOT = Path(__file__).resolve().parents[4]
INPUT_DIR = ROOT / "input/CSMAR/指数取样变更文件(2002年-2025年)"
CALENDAR_PATH = ROOT / "中间结果/01_清洗后样本/市场交易日历_td.csv"
OUTPUT_DIR = ROOT / "中间结果/02_变量构造结果/9指数成分股/上证50"

CHANGES_PATH = OUTPUT_DIR / "上证50_校正后成分变更记录.csv"
INTERVALS_PATH = OUTPUT_DIR / "上证50_历史成分区间.csv"
DAILY_PATH = OUTPUT_DIR / "上证50_交易日成分名单.csv"

INDEX_CODE = "000016"
INDEX_START_DATE = pd.Timestamp("2004-01-02")
DATA_END_DATE = pd.Timestamp("2025-12-31")
DAILY_START_DATE = INDEX_START_DATE
DAILY_END_DATE = DATA_END_DATE
EXPECTED_MEMBER_COUNT = 50
CHUNK_SIZE = 200_000

# 默认不覆盖旧结果。需要重新生成时，由使用者明确改为True（允许覆盖）。
OVERWRITE_OUTPUT = False

CORRECTION_SOURCE = (
    "https://www.sse.com.cn/market/sseindex/diclosure/c/"
    "c_20150911_3984924.shtml"
)


def require_unique_key(table, columns, table_name):
    """发现重复键就停止，不静默保留第一条。"""
    duplicated = table.duplicated(columns, keep=False)
    if duplicated.any():
        examples = table.loc[duplicated, columns].head(10).to_string(index=False)
        raise ValueError(f"{table_name}存在重复键：\n{examples}")


# 第二部分：读取各卷，并只保留上证50的股票类记录
def read_changes():
    input_files = sorted(INPUT_DIR.rglob("IDX_Chgsmp*.csv"))
    if not input_files:
        raise FileNotFoundError(f"没有找到指数取样变更文件：{INPUT_DIR}")

    source_columns = [
        "Indexcd", "Chgsmp01", "Chgsmp02", "Chgsmp03",
        "Chgsmp04", "Chgsmp05", "Chgsmp06", "Chgsmp07",
    ]
    pieces = []
    for path in input_files:
        reader = pd.read_csv(
            path, usecols=source_columns, dtype="string",
            encoding="utf-8-sig", chunksize=CHUNK_SIZE,
        )
        for chunk in reader:
            index_codes = chunk["Indexcd"].str.strip().str.zfill(6)
            selected = chunk.loc[index_codes.eq(INDEX_CODE)].copy()
            if selected.empty:
                continue
            selected["SourceFile"] = str(path.relative_to(ROOT))
            pieces.append(selected)

    if not pieces:
        raise ValueError("输入文件中没有上证50指数000016的记录。")

    changes = pd.concat(pieces, ignore_index=True).rename(columns={
        "Chgsmp01": "ChangeDateRaw",
        "Chgsmp02": "Stkcd",
        "Chgsmp03": "StockName",
        "Chgsmp04": "ChangeType",
        "Chgsmp05": "SecurityType",
        "Chgsmp06": "AnnouncementDate",
        "Chgsmp07": "MarketCode",
    })
    changes["Indexcd"] = changes["Indexcd"].str.strip().str.zfill(6)
    changes["Stkcd"] = changes["Stkcd"].str.strip().str.zfill(6)
    for column in ["ChangeType", "SecurityType", "MarketCode"]:
        changes[column] = changes[column].str.strip()

    if not changes["Stkcd"].str.fullmatch(r"\d{6}", na=False).all():
        raise ValueError("存在无法识别为六位数字的成分股票代码。")
    if not changes["ChangeType"].isin(["1", "2"]).all():
        raise ValueError("变动方式必须为1（新增）或2（剔除）。")
    if not changes["SecurityType"].eq("1").all():
        raise ValueError("上证50记录中出现非股票类证券，请核查输入。")
    if not changes["MarketCode"].eq("SSE").all():
        raise ValueError("上证50记录中出现非上海证券交易所证券。")

    for column in ["ChangeDateRaw", "AnnouncementDate"]:
        changes[column] = pd.to_datetime(
            changes[column], format="%Y-%m-%d", errors="raise",
        )
    if changes["ChangeDateRaw"].isna().any():
        raise ValueError("存在缺失的变更生效日期。")
    changes["ChangeType"] = changes["ChangeType"].astype("int8")
    require_unique_key(
        changes, ["Indexcd", "Stkcd", "ChangeDateRaw"], "原始变更记录",
    )
    return changes


# 第三部分：只校正已用官方公告核实的一条日期，不改原始输入
def correct_verified_date(changes):
    changes = changes.copy()
    changes["ChangeDate"] = changes["ChangeDateRaw"]
    changes["DateCorrected"] = 0
    changes["CorrectionNote"] = ""
    changes["CorrectionSource"] = ""

    # 官方2006-10-30公告：工商银行2006-11-10纳入，广州控股同时调出。
    correction = (
        changes["Stkcd"].eq("600098")
        & changes["ChangeType"].eq(2)
        & changes["ChangeDateRaw"].eq(pd.Timestamp("2007-01-02"))
        & changes["AnnouncementDate"].eq(pd.Timestamp("2006-10-30"))
    )
    if correction.sum() != 1:
        raise ValueError(
            "广州控股日期校正必须恰好匹配一条原始记录；"
            "若数据库已更新，请先复核这项校正规则。"
        )
    changes.loc[correction, "ChangeDate"] = pd.Timestamp("2006-11-10")
    changes.loc[correction, "DateCorrected"] = 1
    changes.loc[correction, "CorrectionNote"] = (
        "依据上交所公告，广州控股应与工商银行纳入同日调出；"
        "剔除日由2007-01-02校正为2006-11-10。"
    )
    changes.loc[correction, "CorrectionSource"] = CORRECTION_SOURCE
    require_unique_key(
        changes, ["Indexcd", "Stkcd", "ChangeDate"], "校正后变更记录",
    )
    return changes.sort_values(["ChangeDate", "ChangeType", "Stkcd"]).reset_index(drop=True)


# 第四部分：按生效日还原成员区间，每批调整后必须恰有50只
def build_intervals(changes):
    history = changes.loc[changes["ChangeDate"].le(DATA_END_DATE)].copy()
    if history.empty or history["ChangeDate"].min() != INDEX_START_DATE:
        raise ValueError("缺少2004-01-02的上证50初始成分记录。")

    initial = history.loc[history["ChangeDate"].eq(INDEX_START_DATE)]
    if len(initial) != EXPECTED_MEMBER_COUNT or not initial["ChangeType"].eq(1).all():
        raise ValueError("初始记录必须是50只不同股票全部新增。")

    # 字典保存当前成员及其纳入日；同一天先检查再批量调出、调入。
    active = {}
    interval_rows = []
    for date, group in history.groupby("ChangeDate", sort=True):
        additions = group.loc[group["ChangeType"].eq(1), "Stkcd"].tolist()
        removals = group.loc[group["ChangeType"].eq(2), "Stkcd"].tolist()
        missing_members = set(removals) - set(active)
        existing_members = set(additions) & set(active)
        if missing_members or existing_members:
            raise ValueError(
                f"{date:%Y-%m-%d}变更衔接异常："
                f"剔除非成员{sorted(missing_members)}，"
                f"重复纳入{sorted(existing_members)}。"
            )

        for stock in removals:
            interval_rows.append({
                "Indexcd": INDEX_CODE, "Stkcd": stock,
                "InDate": active.pop(stock), "OutDate": date,
                "StillMemberAtEnd": 0,
            })
        for stock in additions:
            active[stock] = date
        if len(active) != EXPECTED_MEMBER_COUNT:
            raise ValueError(
                f"{date:%Y-%m-%d}调整后有{len(active)}只成分，"
                f"不是{EXPECTED_MEMBER_COUNT}只，请核查当日变更。"
            )

    for stock, in_date in active.items():
        interval_rows.append({
            "Indexcd": INDEX_CODE, "Stkcd": stock,
            "InDate": in_date, "OutDate": pd.NaT,
            "StillMemberAtEnd": 1,
        })
    intervals = pd.DataFrame(interval_rows).sort_values(["Stkcd", "InDate"])
    intervals["CoverageEndDate"] = DATA_END_DATE
    require_unique_key(intervals, ["Indexcd", "Stkcd", "InDate"], "历史成分区间")
    closed = intervals["OutDate"].notna()
    if not intervals.loc[closed, "InDate"].lt(intervals.loc[closed, "OutDate"]).all():
        raise ValueError("存在纳入日不早于剔除日的异常区间。")
    return intervals.reset_index(drop=True)


# 第五部分：利用项目交易日历展开，只保留属于上证50的股票—交易日
def build_daily_members(intervals):
    if not INDEX_START_DATE <= DAILY_START_DATE <= DAILY_END_DATE <= DATA_END_DATE:
        raise ValueError("日表输出范围必须位于2004-01-02至2025-12-31之间。")
    calendar = pd.read_csv(CALENDAR_PATH, usecols=["Trddt", "td"])
    calendar["Trddt"] = pd.to_datetime(
        calendar["Trddt"], format="%Y-%m-%d", errors="raise",
    )
    if calendar[["Trddt", "td"]].isna().any().any():
        raise ValueError("交易日历存在缺失日期或交易日序号。")
    calendar["td"] = pd.to_numeric(calendar["td"], errors="raise")
    if calendar["td"].mod(1).ne(0).any():
        raise ValueError("交易日序号必须是整数。")
    calendar["td"] = calendar["td"].astype("int64")
    calendar = calendar.drop_duplicates().sort_values("Trddt").reset_index(drop=True)
    require_unique_key(calendar, ["Trddt"], "交易日历日期")
    require_unique_key(calendar, ["td"], "交易日历序号")
    if calendar.empty:
        raise ValueError("市场交易日历为空。")
    if calendar["Trddt"].min() > DAILY_START_DATE or calendar["Trddt"].max() < DAILY_END_DATE:
        raise ValueError("市场交易日历没有覆盖完整的日表输出范围。")
    calendar = calendar.loc[
        calendar["Trddt"].between(DAILY_START_DATE, DAILY_END_DATE)
    ].copy()
    if calendar.empty or not calendar["td"].diff().iloc[1:].eq(1).all():
        raise ValueError("输出期交易日历为空或交易日序号不连续。")

    pieces = []
    for row in intervals.itertuples(index=False):
        in_interval = calendar["Trddt"].ge(row.InDate)
        if pd.notna(row.OutDate):
            in_interval &= calendar["Trddt"].lt(row.OutDate)
        block = calendar.loc[in_interval, ["Trddt", "td"]].copy()
        if block.empty:
            continue
        block["Stkcd"] = row.Stkcd
        pieces.append(block)
    if not pieces:
        raise ValueError("历史成分区间与输出期交易日没有交集。")

    daily = pd.concat(pieces, ignore_index=True)
    daily["Indexcd"] = INDEX_CODE
    daily["is_sse50"] = 1
    daily = daily[["Stkcd", "Trddt", "td", "Indexcd", "is_sse50"]]
    daily = daily.sort_values(["Stkcd", "td"]).reset_index(drop=True)
    require_unique_key(daily, ["Stkcd", "Trddt"], "交易日成分名单")
    counts = daily.groupby("Trddt")["Stkcd"].nunique().reindex(calendar["Trddt"])
    if not counts.eq(EXPECTED_MEMBER_COUNT).all():
        raise ValueError("部分市场交易日的成员数不是50只，停止输出。")
    return daily


# 第六部分：所有检查通过后才保存三张表；导入脚本不会执行本步骤
def main():
    targets = [CHANGES_PATH, INTERVALS_PATH, DAILY_PATH]
    existing = [path for path in targets if path.exists()]
    if existing and not OVERWRITE_OUTPUT:
        raise FileExistsError(
            "以下结果已存在，默认不覆盖：\n"
            + "\n".join(str(path) for path in existing)
            + "\n确认需要重新生成后，可将OVERWRITE_OUTPUT（允许覆盖）改为True。"
        )

    print("读取上证50变更记录……", flush=True)
    changes = correct_verified_date(read_changes())
    intervals = build_intervals(changes)
    daily = build_daily_members(intervals)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for table, path in zip([changes, intervals, daily], targets):
        table.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        print(f"已输出：{path}，{len(table):,}行。")
    print(
        f"日表覆盖{daily['Trddt'].min():%Y-%m-%d}至"
        f"{daily['Trddt'].max():%Y-%m-%d}，"
        f"共{daily['Trddt'].nunique():,}个交易日，每日50只成分。"
    )
    print("已保留原始变更日期及官方校正依据；原始输入文件未修改。")


if __name__ == "__main__":
    main()
