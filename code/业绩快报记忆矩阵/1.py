from pathlib import Path  # 路径处理工具
import pandas as pd      # 数据处理库

# 1. 项目根目录
root = Path(__file__).resolve().parents[2]

# 2. 业绩预告数据目录
data_dir = root / "input" / "CSMAR" / "业绩预告&业绩快报"

# 3. 旧版业绩预告简表路径
old_path = (
    data_dir
    / "业绩预告简表2005-03-31 至 2010-09-30"
    / "FIN_F_SimForecFin.csv"
)

# 4. 新版业绩预告表路径
new_path = (
    data_dir
    / "业绩预告表(季)2010-12-31 至 2026-12-31"
    / "FIN_F_ForecFin.csv"
)

# 5. 检查文件是否存在
print("旧版数据文件存在：", old_path.exists())
print("新版数据文件存在：", new_path.exists())



# 6. 读取旧版业绩预告表的关键字段
old_cols = [
    "StockCode",          # 证券代码
    "AccPeriod",          # 业绩所属报告期
    "PubliDate",          # 预告日期
    "FirstPubliDate",     # 首次预告日期
    "NumForecFinReport",  # 预告次数
]

old_df = pd.read_csv(
    old_path,
    usecols=old_cols,
    dtype="string",         # 先全部按文本读取，保留证券代码前面的零
    encoding="utf-8-sig",   # 兼容文件开头的编码标记
)

# 按上面列出的顺序排列字段
old_df = old_df[old_cols]

# 7. 查看读取结果
print("旧表的行数和列数：", old_df.shape)
print("旧表前5行：")
print(old_df.head())



# 8. 提取旧表中的首次预告信息
old_first = old_df[
    ["StockCode", "AccPeriod", "FirstPubliDate"]
].copy()

# 统一字段名称
old_first.columns = [
    "Stkcd",         # 公告公司代码，当前尚未区分提示公司和被提示公司
    "Accper",        # 业绩报告期
    "ForecastDate",  # 首次预告日期
]

# 9. 统一公司代码和日期格式
old_first["Stkcd"] = old_first["Stkcd"].str.strip().str.zfill(6)

old_first["Accper"] = pd.to_datetime(
    old_first["Accper"], format="%Y-%m-%d", errors="coerce"
)

old_first["ForecastDate"] = pd.to_datetime(
    old_first["ForecastDate"], format="%Y-%m-%d", errors="coerce"
)

# 10. 检查缺失和重复
print("旧表各字段缺失数：")
print(old_first.isna().sum())

print(
    "涉及公司×报告期重复的行数：",
    old_first.duplicated(["Stkcd", "Accper"], keep=False).sum()
)

print("整理后的旧表前5行：")
print(old_first.head())


# 11. 读取新版业绩预告表
new_cols = [
    "StockCode",          # 证券代码
    "Source",             # 业绩来源
    "AccPeriod",          # 业绩报告期
    "PubliDate",          # 预告日期
    "NumForecFinReport",  # 预告发布序号
]






import csv  # 逗号分隔文件读取工具
import re   # 正则匹配工具

# 逐行读取新版表，避免说明文字中的坏引号影响下一条记录
new_rows = []

with new_path.open(encoding="utf-8-sig", newline="") as file:
    # 打开new_path.open()文件， newline=""为换行参数符
    # with ... as file，打开文件，并把这个打开的文件对象命名为 file。
    # 当代码离开 with 代码块时，自动关闭文件。
    # 读取表头，确定所需字段的位置
    header = next(csv.reader([file.readline()]))
    # file.readline()表示从文件当前位置读取一行文本。
    positions = [header.index(col) for col in new_cols]
    # 只读取前面所列行的索引和数据

    for line_no, line in enumerate(file, start=2):
        # 数据记录以六位证券代码开头；其余为说明文字续行
        if not re.match(r'^"\d{6}",', line):
            continue

        # 每行单独解析，避免跨行合并
        row = next(csv.reader([line], strict=False))

        if len(row) <= max(positions):
            raise ValueError(f"源文件第{line_no}行缺少必要字段")

        # 按字段列表提取所需内容
        new_rows.append([row[position] for position in positions])

# 转换为数据表，所有字段先保留为文本
new_df = pd.DataFrame(
    new_rows,
    columns=new_cols,
    dtype="string",
)


# 12. 查看新版表的基本情况
print("新版表的行数和列数：", new_df.shape)
print("新版表前5行：")
print(new_df.head())

print("新版表的业绩来源分布：")
print(new_df["Source"].value_counts(dropna=False))

print("新版表的预告发布序号分布：")
print(new_df["NumForecFinReport"].value_counts(dropna=False))


# 13. 提取新版表中标记为首次预告的记录
new_first = new_df.loc[
    new_df["NumForecFinReport"].str.strip().eq("0"),
    ["StockCode", "AccPeriod", "PubliDate", "Source"],
].copy()

# 统一字段名称
new_first.columns = [
    "Stkcd",         # 公告公司代码
    "Accper",        # 业绩报告期
    "ForecastDate",  # 预告日期
    "Source",        # 业绩来源
]

# 14. 统一公司代码和日期格式
new_first["Stkcd"] = new_first["Stkcd"].str.strip().str.zfill(6)

new_first["Accper"] = pd.to_datetime(
    new_first["Accper"], format="%Y-%m-%d", errors="coerce"
)

new_first["ForecastDate"] = pd.to_datetime(
    new_first["ForecastDate"], format="%Y-%m-%d", errors="coerce"
)

# 15. 检查筛选结果
print("新版首次预告表的行数和列数：", new_first.shape)

print("新版首次预告表各字段缺失数：")
print(new_first.isna().sum())

print(
    "涉及公司×报告期重复的行数：",
    new_first.duplicated(["Stkcd", "Accper"], keep=False).sum()
)

print("新版首次预告表的来源分布：")
print(new_first["Source"].value_counts(dropna=False))


# 16. 提取公司×报告期重复的记录
dup_mask = new_first.duplicated(
    ["Stkcd", "Accper"], keep=False
)

new_dup = new_first.loc[dup_mask].sort_values(
    ["Stkcd", "Accper", "ForecastDate", "Source"]
)

# 17. 汇总每个重复组的记录数、日期数和来源数
dup_summary = new_dup.groupby(
    ["Stkcd", "Accper"]
).agg(
    记录条数=("ForecastDate", "size"),
    不同日期数=("ForecastDate", "nunique"),
    来源种类数=("Source", "nunique"),
)

print("涉及的公司×报告期组数：", len(dup_summary))

print("重复组的记录、日期和来源组合分布：")
print(dup_summary.value_counts().sort_index())

print("重复记录的前20行：")
print(new_dup.head(20).to_string(index=False))



# 18. 找出每个公司×报告期的最早预告日期
earliest_date = new_first.groupby(
    ["Stkcd", "Accper"]
)["ForecastDate"].transform("min")

earliest_rows = new_first.loc[
    new_first["ForecastDate"].eq(earliest_date)
].copy()

# 19. 合并最早日期相同的记录，保留来源组合
new_first_unique = earliest_rows.groupby(
    ["Stkcd", "Accper", "ForecastDate"],
    as_index=False,
).agg(
    FirstSource=(
        "Source",
        lambda x: "+".join(sorted(x.unique())),
    )
)

# 20. 检查整理结果
print("新版唯一首次预告表的行数和列数：")
print(new_first_unique.shape)

print("公司×报告期重复数：")
print(
    new_first_unique.duplicated(["Stkcd", "Accper"]).sum()
)

print("最早预告的来源组合分布：")
print(new_first_unique["FirstSource"].value_counts().sort_index())


# 21. 为旧版数据补充来源字段和版本标记
old_for_merge = old_first.copy()
old_for_merge["FirstSource"] = pd.NA
old_for_merge["DataVersion"] = "旧版"

# 为新版数据添加版本标记
new_for_merge = new_first_unique.copy()
new_for_merge["DataVersion"] = "新版"

# 22. 合并两版首次预告数据
forecast_first = pd.concat(
    [old_for_merge, new_for_merge],
    ignore_index=True,
)

# 按公司和报告期排序
forecast_first = forecast_first.sort_values(
    ["Stkcd", "Accper"]
).reset_index(drop=True)

# 23. 检查合并结果
print("合并后的首次预告表行数和列数：")
print(forecast_first.shape)

print("公司×报告期重复数：")
print(
    forecast_first.duplicated(["Stkcd", "Accper"]).sum()
)

print("关键字段缺失数：")
print(
    forecast_first[
        ["Stkcd", "Accper", "ForecastDate"]
    ].isna().sum()
)

print("各数据版本的记录数：")
print(forecast_first["DataVersion"].value_counts())


# 24. 提取预告目标年度
forecast_first["TargetYear"] = forecast_first["Accper"].dt.year

# 25. 根据目标报告期，识别预告对应的报告类型
forecast_first["TargetType"] = (
    forecast_first["Accper"]
    .dt.strftime("%m-%d")
    .map({
        "03-31": "一季度",
        "06-30": "半年度",
        "09-30": "前三季度",
        "12-31": "年度",
    })
)

# 26. 检查目标报告类型
print("预告目标报告类型分布：")
print(forecast_first["TargetType"].value_counts(dropna=False))

print("目标报告类型无法识别的记录数：")
print(forecast_first["TargetType"].isna().sum())

print("添加年度和报告类型后的前8行：")
print(
    forecast_first[
        ["Stkcd", "Accper", "ForecastDate", "TargetYear", "TargetType"]
    ].head(8).to_string(index=False)
)



# 27. 保留首次预告底表的五个字段
forecast_first = forecast_first[
    ["Stkcd", "Accper", "ForecastDate", "FirstSource", "DataVersion"]
].copy()

# 28. 读取正式财报的公司代码和公布日期
report_path = (
    root / "input" / "CSMAR"
    / "年、中、季报基本情况文件1990-12-31至2026-06-30"
    / "IAR_Rept.csv"
)

report_dates = pd.read_csv(
    report_path,
    usecols=["Stkcd", "Annodt"],
    dtype="string",
    encoding="utf-8-sig",
)

report_dates["Stkcd"] = (
    report_dates["Stkcd"].str.strip().str.zfill(6)
)
report_dates["Annodt"] = pd.to_datetime(
    report_dates["Annodt"],
    format="%Y-%m-%d",
    errors="coerce",
)

# 29. 每个“公司×财报公布日”只保留一条，避免合并增行
report_dates = (
    report_dates
    .dropna(subset=["Stkcd", "Annodt"])
    .drop_duplicates(["Stkcd", "Annodt"])
    .rename(columns={"Annodt": "ForecastDate"})
)
report_dates["SameDayReport"] = 1

# 30. 将财报同日标签匹配到首次预告底表
forecast_first = forecast_first.merge(
    report_dates,
    on=["Stkcd", "ForecastDate"],
    how="left",
    validate="many_to_one",
)

forecast_first["SameDayReport"] = (
    forecast_first["SameDayReport"].fillna(0).astype("Int8")
)

# 31. 检查合并是否增行，以及标签分布
print("添加财报同日标签后的行数和列数：")
print(forecast_first.shape)

print("财报同日标签分布：")
print(forecast_first["SameDayReport"].value_counts().sort_index())



# 32. 建立模式识别临时表
pattern_work = forecast_first.sort_values(
    ["Stkcd", "Accper"]
).reset_index(drop=True).copy()

# 33. 同一公司、同类目标报告期，跨年比较
same_type = pattern_work.groupby(
    ["Stkcd", pattern_work["Accper"].dt.month]
)

# 前两次同类预告对应的目标报告期
previous_period_1 = same_type["Accper"].shift(1)
previous_period_2 = same_type["Accper"].shift(2)

# 前两次同类预告的实际披露日期
previous_date_1 = same_type["ForecastDate"].shift(1)
previous_date_2 = same_type["ForecastDate"].shift(2)

current_year = pattern_work["Accper"].dt.year

# 必须恰好对应前一年、前两年，不能跳过缺失年份
history_complete = (
    (current_year - previous_period_1.dt.year).eq(1)
    & (current_year - previous_period_2.dt.year).eq(2)
)

# 历史预告必须在本次预告之前已经披露
history_available = (
    previous_date_1.lt(pattern_work["ForecastDate"])
    & previous_date_2.lt(pattern_work["ForecastDate"])
)

# 34. 标记是否具备连续三年的同类预告记录
pattern_work["HistoryEligible"] = (
    history_complete & history_available
).astype("Int8")

print("连续三年同类预告历史检查：")
print(
    pattern_work["HistoryEligible"].value_counts().sort_index()
)


# 35. 生成“同月第几个星期几”的规律编码
announce_date = pattern_work["ForecastDate"]

pattern_work["RuleMonthWeekday"] = (
    announce_date.dt.month.astype("string")
    + "_"
    + (((announce_date.dt.day - 1) // 7) + 1).astype("string")
    + "_"
    + (announce_date.dt.weekday + 1).astype("string")
)

# 36. 比较同公司、同类报告期的连续三次规律编码
same_type = pattern_work.groupby(
    ["Stkcd", pattern_work["Accper"].dt.month]
)

previous_rule_1 = same_type["RuleMonthWeekday"].shift(1)
previous_rule_2 = same_type["RuleMonthWeekday"].shift(2)

pattern_work["PatternMonthWeekday"] = (
    pattern_work["RuleMonthWeekday"].eq(previous_rule_1)
    & pattern_work["RuleMonthWeekday"].eq(previous_rule_2)
).astype("Int8")

# 历史不足的记录保留缺失
pattern_work.loc[
    pattern_work["HistoryEligible"].eq(0),
    "PatternMonthWeekday",
] = pd.NA

# 37. 检查本类规律的识别结果
print("同月第几个星期几规律的识别结果：")
print(
    pattern_work["PatternMonthWeekday"]
    .value_counts(dropna=False)
    .sort_index()
)

# 38. 生成固定公历月日编码
pattern_work["RuleMonthDay"] = (
    pattern_work["ForecastDate"].dt.strftime("%m-%d")
)

# 39. 比较连续三次同类预告的月日
same_type = pattern_work.groupby(
    ["Stkcd", pattern_work["Accper"].dt.month]
)

previous_rule_1 = same_type["RuleMonthDay"].shift(1)
previous_rule_2 = same_type["RuleMonthDay"].shift(2)

pattern_work["PatternMonthDay"] = (
    pattern_work["RuleMonthDay"].eq(previous_rule_1)
    & pattern_work["RuleMonthDay"].eq(previous_rule_2)
).astype("Int8")

# 历史不足的记录保留缺失
pattern_work.loc[
    pattern_work["HistoryEligible"].eq(0),
    "PatternMonthDay",
] = pd.NA

# 40. 检查固定公历月日规律
print("固定公历月日规律的识别结果：")
print(
    pattern_work["PatternMonthDay"]
    .value_counts(dropna=False)
    .sort_index()
)


# 41. 计算首次预告日相对目标报告期末的自然日间隔
pattern_work["RuleDaysFromAccper"] = (
    pattern_work["ForecastDate"] - pattern_work["Accper"]
).dt.days

# 42. 比较连续三次同类预告的日期间隔
same_type = pattern_work.groupby(
    ["Stkcd", pattern_work["Accper"].dt.month]
)

previous_rule_1 = same_type["RuleDaysFromAccper"].shift(1)
previous_rule_2 = same_type["RuleDaysFromAccper"].shift(2)

pattern_work["PatternDaysFromAccper"] = (
    pattern_work["RuleDaysFromAccper"].eq(previous_rule_1)
    & pattern_work["RuleDaysFromAccper"].eq(previous_rule_2)
).astype("Int8")

# 历史不足的记录保留缺失
pattern_work.loc[
    pattern_work["HistoryEligible"].eq(0),
    "PatternDaysFromAccper",
] = pd.NA

# 43. 检查固定自然日间隔规律
print("相对目标期末固定自然日间隔规律的识别结果：")
print(
    pattern_work["PatternDaysFromAccper"]
    .value_counts(dropna=False)
    .sort_index()
)


# 44. 计算公告日在当月属于第几个日历周
announce_date = pattern_work["ForecastDate"]

month_start = announce_date.dt.to_period("M").dt.to_timestamp()

# 月初星期编码：周日为0，周一至周六为1至6
month_start_weekday = (month_start.dt.weekday + 1) % 7

week_number = (
    (announce_date.dt.day + month_start_weekday - 1) // 7
) + 1

# 编码依次表示：月份、月内日历周序号、星期几
pattern_work["RuleMonthWeek"] = (
    announce_date.dt.month.astype("string")
    + "_"
    + week_number.astype("string")
    + "_"
    + (announce_date.dt.weekday + 1).astype("string")
)

# 45. 比较连续三次同类预告的规律编码
same_type = pattern_work.groupby(
    ["Stkcd", pattern_work["Accper"].dt.month]
)

previous_rule_1 = same_type["RuleMonthWeek"].shift(1)
previous_rule_2 = same_type["RuleMonthWeek"].shift(2)

pattern_work["PatternMonthWeek"] = (
    pattern_work["RuleMonthWeek"].eq(previous_rule_1)
    & pattern_work["RuleMonthWeek"].eq(previous_rule_2)
).astype("Int8")

pattern_work.loc[
    pattern_work["HistoryEligible"].eq(0),
    "PatternMonthWeek",
] = pd.NA

# 46. 检查本类规律
print("同月日历周与星期规律的识别结果：")
print(
    pattern_work["PatternMonthWeek"]
    .value_counts(dropna=False)
    .sort_index()
)


# 47. 计算预告日、目标期末所在周的起点
announce_date = pattern_work["ForecastDate"]
accper_date = pattern_work["Accper"]

announce_week_start = announce_date - pd.to_timedelta(
    (announce_date.dt.weekday + 1) % 7,
    unit="D",
)

accper_week_start = accper_date - pd.to_timedelta(
    (accper_date.dt.weekday + 1) % 7,
    unit="D",
)

# 两个日历周之间相隔的周数，保留正负号
week_gap = (
    (announce_week_start - accper_week_start).dt.days // 7
)

# 编码依次表示：相对周数、星期几
pattern_work["RuleWeekFromAccper"] = (
    week_gap.astype("string")
    + "_"
    + (announce_date.dt.weekday + 1).astype("string")
)

# 48. 比较连续三次同类预告的规律编码
same_type = pattern_work.groupby(
    ["Stkcd", pattern_work["Accper"].dt.month]
)

previous_rule_1 = same_type["RuleWeekFromAccper"].shift(1)
previous_rule_2 = same_type["RuleWeekFromAccper"].shift(2)

pattern_work["PatternWeekFromAccper"] = (
    pattern_work["RuleWeekFromAccper"].eq(previous_rule_1)
    & pattern_work["RuleWeekFromAccper"].eq(previous_rule_2)
).astype("Int8")

pattern_work.loc[
    pattern_work["HistoryEligible"].eq(0),
    "PatternWeekFromAccper",
] = pd.NA

# 49. 检查本类规律
print("相对目标期末日历周与星期规律的识别结果：")
print(
    pattern_work["PatternWeekFromAccper"]
    .value_counts(dropna=False)
    .sort_index()
)


# 50. 计算相对目标期末的同星期几出现序号
announce_date = pattern_work["ForecastDate"]

day_gap = (
    announce_date - pattern_work["Accper"]
).dt.days

weekday_order = 1 + (day_gap - 1) // 7

pattern_work["RuleWeekdayFromAccper"] = (
    weekday_order.astype("string")
    + "_"
    + (announce_date.dt.weekday + 1).astype("string")
)

# 51. 比较连续三次同类预告的规律编码
same_type = pattern_work.groupby(
    ["Stkcd", pattern_work["Accper"].dt.month]
)

previous_rule_1 = same_type["RuleWeekdayFromAccper"].shift(1)
previous_rule_2 = same_type["RuleWeekdayFromAccper"].shift(2)

pattern_work["PatternWeekdayFromAccper"] = (
    pattern_work["RuleWeekdayFromAccper"].eq(previous_rule_1)
    & pattern_work["RuleWeekdayFromAccper"].eq(previous_rule_2)
).astype("Int8")

pattern_work.loc[
    pattern_work["HistoryEligible"].eq(0),
    "PatternWeekdayFromAccper",
] = pd.NA

# 52. 检查本类规律
print("相对目标期末同星期几出现序号规律的识别结果：")
print(
    pattern_work["PatternWeekdayFromAccper"]
    .value_counts(dropna=False)
    .sort_index()
)


import numpy as np  # 数值计算库

# 53. 计算公告日是当月第几个非周末日
announce_date = pattern_work["ForecastDate"]

nonweekend_order = pd.Series(
    np.busday_count(
        announce_date.values.astype("datetime64[M]"),
        (announce_date + pd.Timedelta(days=1))
        .values.astype("datetime64[D]"),
        weekmask="1111100",
    ),
    index=pattern_work.index,
)

# 编码依次表示：月份、非周末日序号
# 周末披露的记录不适用本类规则
pattern_work["RuleNonWeekendMonth"] = (
    announce_date.dt.month.astype("string")
    + "_"
    + nonweekend_order.astype("string")
).where(announce_date.dt.weekday.lt(5), pd.NA)

# 54. 比较连续三次同类预告的规律编码
same_type = pattern_work.groupby(
    ["Stkcd", pattern_work["Accper"].dt.month]
)

previous_rule_1 = same_type["RuleNonWeekendMonth"].shift(1)
previous_rule_2 = same_type["RuleNonWeekendMonth"].shift(2)

pattern_work["PatternNonWeekendMonth"] = (
    pattern_work["RuleNonWeekendMonth"].eq(previous_rule_1)
    & pattern_work["RuleNonWeekendMonth"].eq(previous_rule_2)
).fillna(False).astype("Int8")

pattern_work.loc[
    pattern_work["HistoryEligible"].eq(0),
    "PatternNonWeekendMonth",
] = pd.NA

# 55. 检查本类规律
print("同月非周末日序号规律的识别结果：")
print(
    pattern_work["PatternNonWeekendMonth"]
    .value_counts(dropna=False)
    .sort_index()
)



# 56. 计算相对目标期末的非周末日间隔
announce_date = pattern_work["ForecastDate"]

earlier_date = pattern_work[["Accper", "ForecastDate"]].min(axis=1)
later_date = pattern_work[["Accper", "ForecastDate"]].max(axis=1)

# 不含较早日期，包含较晚日期
nonweekend_gap = pd.Series(
    np.busday_count(
        (earlier_date + pd.Timedelta(days=1))
        .values.astype("datetime64[D]"),
        (later_date + pd.Timedelta(days=1))
        .values.astype("datetime64[D]"),
        weekmask="1111100",
    ),
    index=pattern_work.index,
)

# 预告早于目标期末时，间隔取负
nonweekend_gap = nonweekend_gap.where(
    announce_date.ge(pattern_work["Accper"]),
    -nonweekend_gap,
)

# 周末发布的预告不生成这类规律编码
pattern_work["RuleNonWeekendFromAccper"] = (
    nonweekend_gap.astype("string")
    .where(announce_date.dt.weekday.lt(5), pd.NA)
)


# 57. 比较连续三年同类预告的规律编码
same_type = pattern_work.groupby(
    ["Stkcd", pattern_work["Accper"].dt.month]
)

previous_rule_1 = same_type["RuleNonWeekendFromAccper"].shift(1)
previous_rule_2 = same_type["RuleNonWeekendFromAccper"].shift(2)

pattern_work["PatternNonWeekendFromAccper"] = (
    pattern_work["RuleNonWeekendFromAccper"].eq(previous_rule_1)
    & pattern_work["RuleNonWeekendFromAccper"].eq(previous_rule_2)
).fillna(False).astype("Int8")

pattern_work.loc[
    pattern_work["HistoryEligible"].eq(0),
    "PatternNonWeekendFromAccper",
] = pd.NA


# 58. 查看识别结果
print("相对目标期末非周末日间隔规律的识别结果：")
print(
    pattern_work["PatternNonWeekendFromAccper"]
    .value_counts(dropna=False)
    .sort_index()
)


# 59. 读取交易日历
calendar_path = (
    root
    / "input"
    / "CSMAR"
    / "日历文件(1990-12-19至2026-08-20)"
    / "CER_Cale.csv"
)

calendar_raw = pd.read_csv(
    calendar_path,
    usecols=["Clddt", "Markettype", "Statse"],
    dtype="string",
    encoding="utf-8-sig",
)


# 60. 提取上证A股日历，保留开市日和休市日
trade_calendar = calendar_raw.loc[
    calendar_raw["Markettype"].str.strip().eq("1"),
    ["Clddt", "Statse"],
].copy()

trade_calendar["Clddt"] = pd.to_datetime(
    trade_calendar["Clddt"],
    format="%Y-%m-%d",
    errors="raise",
)
trade_calendar["Statse"] = trade_calendar["Statse"].str.strip()

trade_calendar = trade_calendar.sort_values(
    "Clddt"
).reset_index(drop=True)


# 61. 检查日期唯一性、状态和覆盖情况
print("交易日历的行数和列数：", trade_calendar.shape)

print(
    "日历日期重复数：",
    trade_calendar.duplicated("Clddt").sum(),
)

print("日历字段缺失数：")
print(trade_calendar.isna().sum())

print("开休市状态分布：")
print(trade_calendar["Statse"].value_counts(dropna=False))

print(
    "日历日期范围：",
    trade_calendar["Clddt"].min(),
    "至",
    trade_calendar["Clddt"].max(),
)

all_dates = pd.date_range(
    trade_calendar["Clddt"].min(),
    trade_calendar["Clddt"].max(),
)

print(
    "日历范围内缺少的自然日数：",
    len(all_dates.difference(trade_calendar["Clddt"])),
)

print(
    "预告日期未被日历覆盖的记录数：",
    (~pattern_work["ForecastDate"].isin(trade_calendar["Clddt"])).sum(),
)

print(
    "目标期末未被日历覆盖的记录数：",
    (~pattern_work["Accper"].isin(trade_calendar["Clddt"])).sum(),
)


# 62. 标记日历未覆盖的日期
forecast_outside = ~pattern_work["ForecastDate"].isin(
    trade_calendar["Clddt"]
)

accper_outside = ~pattern_work["Accper"].isin(
    trade_calendar["Clddt"]
)


# 63. 查看未覆盖日期的分布
print("未覆盖的预告日期及记录数：")
print(
    pattern_work.loc[forecast_outside, "ForecastDate"]
    .value_counts()
    .sort_index()
)

print("未覆盖的目标期末及记录数：")
print(
    pattern_work.loc[accper_outside, "Accper"]
    .value_counts()
    .sort_index()
)

print(
    "至少一项日期未覆盖的记录数：",
    (forecast_outside | accper_outside).sum(),
)

print("相关记录示例：")
print(
    pattern_work.loc[
        forecast_outside | accper_outside,
        ["Stkcd", "Accper", "ForecastDate", "HistoryEligible"],
    ]
    .sort_values(["ForecastDate", "Stkcd", "Accper"])
    .head(15)
    .to_string(index=False)
)


# 62. 标记日历未覆盖的日期
forecast_outside = ~pattern_work["ForecastDate"].isin(
    trade_calendar["Clddt"]
)

accper_outside = ~pattern_work["Accper"].isin(
    trade_calendar["Clddt"]
)


# 63. 查看未覆盖日期的分布
print("未覆盖的预告日期及记录数：")
print(
    pattern_work.loc[forecast_outside, "ForecastDate"]
    .value_counts()
    .sort_index()
)

print("未覆盖的目标期末及记录数：")
print(
    pattern_work.loc[accper_outside, "Accper"]
    .value_counts()
    .sort_index()
)

print(
    "至少一项日期未覆盖的记录数：",
    (forecast_outside | accper_outside).sum(),
)

print("相关记录示例：")
print(
    pattern_work.loc[
        forecast_outside | accper_outside,
        ["Stkcd", "Accper", "ForecastDate", "HistoryEligible"],
    ]
    .sort_values(["ForecastDate", "Stkcd", "Accper"])
    .head(15)
    .to_string(index=False)
)


# 64. 构造月内交易日序号
trade_calendar["IsTradingDay"] = (
    trade_calendar["Statse"].eq("O").astype("Int8")
)

trade_calendar["TradingDayOfMonth"] = (
    trade_calendar.groupby(
        trade_calendar["Clddt"].dt.to_period("M")
    )["IsTradingDay"]
    .cumsum()
    .astype("Int64")
    .where(trade_calendar["IsTradingDay"].eq(1), pd.NA)
)


# 65. 按预告日期匹配月内交易日序号
calendar_by_date = trade_calendar.set_index("Clddt")

announce_date = pattern_work["ForecastDate"]
month_start = announce_date.dt.to_period("M").dt.to_timestamp()

# 预告日和所在月月初都需要在日历范围内
pattern_work["CalendarCoveredMonth"] = (
    announce_date.isin(calendar_by_date.index)
    & month_start.isin(calendar_by_date.index)
).astype("Int8")

month_trading_order = announce_date.map(
    calendar_by_date["TradingDayOfMonth"]
)

# 编码表示“月份_月内交易日序号”
pattern_work["RuleTradingMonth"] = (
    announce_date.dt.month.astype("string")
    + "_"
    + month_trading_order.astype("string")
).where(pattern_work["CalendarCoveredMonth"].eq(1), pd.NA)


# 66. 比较连续三年同类预告的规律编码
same_type = pattern_work.groupby(
    ["Stkcd", pattern_work["Accper"].dt.month]
)

previous_rule_1 = same_type["RuleTradingMonth"].shift(1)
previous_rule_2 = same_type["RuleTradingMonth"].shift(2)

pattern_work["PatternTradingMonth"] = (
    pattern_work["RuleTradingMonth"].eq(previous_rule_1)
    & pattern_work["RuleTradingMonth"].eq(previous_rule_2)
).fillna(False).astype("Int8")


# 67. 检查三年窗口的日历是否全部覆盖
calendar_available = (
    pattern_work["CalendarCoveredMonth"].eq(1)
    & same_type["CalendarCoveredMonth"].shift(1).eq(1)
    & same_type["CalendarCoveredMonth"].shift(2).eq(1)
)

pattern_work.loc[
    pattern_work["HistoryEligible"].eq(0) | ~calendar_available,
    "PatternTradingMonth",
] = pd.NA


# 68. 查看识别结果
print("同月交易日序号规律的识别结果：")
print(
    pattern_work["PatternTradingMonth"]
    .value_counts(dropna=False)
    .sort_index()
)

print(
    "历史充分但三年窗口日历覆盖不足的记录数：",
    (
        pattern_work["HistoryEligible"].eq(1)
        & ~calendar_available
    ).sum(),
)



# 69. 构造截至每个自然日的累计交易日数
trade_calendar["CumTradingDays"] = (
    trade_calendar["IsTradingDay"].cumsum().astype("Int64")
)

# 新增字段后，重新建立日期索引
calendar_by_date = trade_calendar.set_index("Clddt")


# 70. 计算预告日相对目标期末的交易日间隔
forecast_cum = pattern_work["ForecastDate"].map(
    calendar_by_date["CumTradingDays"]
)

accper_cum = pattern_work["Accper"].map(
    calendar_by_date["CumTradingDays"]
)

trading_gap = forecast_cum - accper_cum

# 两个日期都需要在日历范围内
pattern_work["CalendarCoveredGap"] = (
    pattern_work["ForecastDate"].isin(calendar_by_date.index)
    & pattern_work["Accper"].isin(calendar_by_date.index)
).astype("Int8")

# 本类规律要求预告发布日为交易日
announce_is_trading = (
    pattern_work["ForecastDate"]
    .map(calendar_by_date["IsTradingDay"])
    .eq(1)
    .fillna(False)
)

pattern_work["RuleTradingFromAccper"] = (
    trading_gap.astype("string")
    .where(
        announce_is_trading
        & pattern_work["CalendarCoveredGap"].eq(1),
        pd.NA,
    )
)


# 71. 比较连续三年同类预告的规律编码
same_type = pattern_work.groupby(
    ["Stkcd", pattern_work["Accper"].dt.month]
)

previous_rule_1 = same_type["RuleTradingFromAccper"].shift(1)
previous_rule_2 = same_type["RuleTradingFromAccper"].shift(2)

pattern_work["PatternTradingFromAccper"] = (
    pattern_work["RuleTradingFromAccper"].eq(previous_rule_1)
    & pattern_work["RuleTradingFromAccper"].eq(previous_rule_2)
).fillna(False).astype("Int8")


# 72. 历史不足或三年中任一年日历未覆盖，保留缺失
calendar_available = (
    pattern_work["CalendarCoveredGap"].eq(1)
    & same_type["CalendarCoveredGap"].shift(1).eq(1)
    & same_type["CalendarCoveredGap"].shift(2).eq(1)
)

pattern_work.loc[
    pattern_work["HistoryEligible"].eq(0) | ~calendar_available,
    "PatternTradingFromAccper",
] = pd.NA


# 73. 查看识别结果
print("相对目标期末交易日间隔规律的识别结果：")
print(
    pattern_work["PatternTradingFromAccper"]
    .value_counts(dropna=False)
    .sort_index()
)

print(
    "历史充分但三年窗口日历覆盖不足的记录数：",
    (
        pattern_work["HistoryEligible"].eq(1)
        & ~calendar_available
    ).sum(),
)



# 74. 列出十类规律标签
pattern_cols = [
    "PatternMonthWeekday",          # 同月第几个星期几
    "PatternMonthDay",              # 固定公历月日
    "PatternDaysFromAccper",        # 相对期末自然日间隔
    "PatternMonthWeek",             # 同月日历周与星期
    "PatternWeekFromAccper",        # 相对期末日历周与星期
    "PatternWeekdayFromAccper",     # 相对期末同星期几出现序号
    "PatternNonWeekendMonth",       # 同月非周末日序号
    "PatternNonWeekendFromAccper",  # 相对期末非周末日间隔
    "PatternTradingMonth",          # 同月交易日序号
    "PatternTradingFromAccper",     # 相对期末交易日间隔
]

pattern_flags = pattern_work[pattern_cols]

# 是否至少有一类命中
has_pattern = pattern_flags.eq(1).any(axis=1)

# 是否十类都已完成判断
all_checked = pattern_flags.notna().all(axis=1)


# 75. 构造模式总标签
pattern_work["PatternForecastQtr"] = pd.Series(
    pd.NA,
    index=pattern_work.index,
    dtype="Int8",
)

# 十类都已判断，先记为0
pattern_work.loc[
    all_checked, "PatternForecastQtr"
] = 0

# 任意一类命中，总标签记为1
pattern_work.loc[
    has_pattern, "PatternForecastQtr"
] = 1

# 历史不足的记录统一保留缺失
pattern_work.loc[
    pattern_work["HistoryEligible"].eq(0),
    "PatternForecastQtr",
] = pd.NA


# 76. 查看总标签分布
print("业绩预告模式总标签分布：")
print(
    pattern_work["PatternForecastQtr"]
    .value_counts(dropna=False)
    .sort_index()
)

print(
    "历史充分但总标签仍无法判断的记录数：",
    (
        pattern_work["HistoryEligible"].eq(1)
        & pattern_work["PatternForecastQtr"].isna()
    ).sum(),
)



# 77. 提取模式预告明细：公司×目标报告期
pattern_forecast = pattern_work.loc[
    pattern_work["PatternForecastQtr"].eq(1),
    [
        "Stkcd",
        "Accper",
        "ForecastDate",
        "FirstSource",
        "DataVersion",
        "SameDayReport",
    ],
].copy()


# 78. 整理模式预告事件：公司×首次预告日
pattern_events = (
    pattern_forecast[["Stkcd", "ForecastDate"]]
    .drop_duplicates()
    .sort_values(["ForecastDate", "Stkcd"])
    .reset_index(drop=True)
)


# 79. 检查事件数量和共同公告情况
daily_pattern_count = pattern_events.groupby(
    "ForecastDate"
)["Stkcd"].nunique()

print("模式预告明细记录数：", len(pattern_forecast))
print("去重后的公司×预告日事件数：", len(pattern_events))
print("涉及公司数：", pattern_events["Stkcd"].nunique())

print(
    "至少两家公司共同预告的日期数：",
    daily_pattern_count.ge(2).sum(),
)

print(
    "这些共同公告日期涉及的公司事件数：",
    daily_pattern_count.loc[daily_pattern_count.ge(2)].sum(),
)


# 80. 计算理论上的公司对×日期记录数
expected_pair_events = (
    daily_pattern_count * (daily_pattern_count - 1) // 2
).sum()


# 81. 按相同预告日期，将事件表与自身连接
history_pairs = pattern_events.rename(
    columns={"Stkcd": "StkcdA"}
).merge(
    pattern_events.rename(columns={"Stkcd": "StkcdB"}),
    on="ForecastDate",
    how="inner",
)


# 82. 每对不同公司只保留代码从小到大的一条
history_pairs = (
    history_pairs.loc[
        history_pairs["StkcdA"] < history_pairs["StkcdB"],
        ["StkcdA", "StkcdB", "ForecastDate"],
    ]
    .sort_values(["ForecastDate", "StkcdA", "StkcdB"])
    .reset_index(drop=True)
)


# 83. 核对配对数量和唯一键
print("理论公司对×日期记录数：", expected_pair_events)
print("实际公司对×日期记录数：", len(history_pairs))

print(
    "公司对×日期重复数：",
    history_pairs.duplicated(
        ["StkcdA", "StkcdB", "ForecastDate"]
    ).sum(),
)

print(
    "跨日期去重后的公司对数：",
    len(history_pairs[["StkcdA", "StkcdB"]].drop_duplicates()),
)

print("历史共同预告公司对前10行：")
print(history_pairs.head(10).to_string(index=False))



# 84. 为历史关系生成两个匹配方向
forward_pairs = history_pairs.rename(
    columns={
        "StkcdA": "Stkcd",
        "StkcdB": "StkcdRecall",
        "ForecastDate": "MemoryDate",
    }
)

reverse_pairs = history_pairs.rename(
    columns={
        "StkcdB": "Stkcd",
        "StkcdA": "StkcdRecall",
        "ForecastDate": "MemoryDate",
    }
)


# 85. 合并两个方向
history_directed = pd.concat(
    [forward_pairs, reverse_pairs],
    ignore_index=True,
)

history_directed = (
    history_directed[
        ["Stkcd", "StkcdRecall", "MemoryDate"]
    ]
    .sort_values(["MemoryDate", "Stkcd", "StkcdRecall"])
    .reset_index(drop=True)
)


# 86. 检查方向记录
print("展开后的历史关系记录数：", len(history_directed))

print(
    "提示公司×被提示公司×历史日期重复数：",
    history_directed.duplicated(
        ["Stkcd", "StkcdRecall", "MemoryDate"]
    ).sum(),
)

print(
    "自身配对数：",
    history_directed["Stkcd"]
    .eq(history_directed["StkcdRecall"])
    .sum(),
)

print("展开后的前10行：")
print(history_directed.head(10).to_string(index=False))



# 87. 读取正式财报明细
report_detail = pd.read_csv(
    report_path,
    usecols=["Stkcd", "Accper", "Annodt", "Reptyp"],
    dtype="string",
    encoding="utf-8-sig",
)


# 88. 统一公司代码、日期和报告类型
report_detail["Stkcd"] = (
    report_detail["Stkcd"].str.strip().str.zfill(6)
)

for col in ["Accper", "Annodt"]:
    report_detail[col] = pd.to_datetime(
        report_detail[col],
        format="%Y-%m-%d",
        errors="coerce",
    )

report_detail["Reptyp"] = pd.to_numeric(
    report_detail["Reptyp"],
    errors="coerce",
).astype("Int8")


# 89. 检查财报明细
print("财报明细的行数和列数：", report_detail.shape)

print("财报明细各字段缺失数：")
print(report_detail.isna().sum())

print(
    "涉及公司×报告期重复的行数：",
    report_detail.duplicated(
        ["Stkcd", "Accper"], keep=False
    ).sum(),
)

print(
    "公告日等于报告期末的记录数：",
    report_detail["Annodt"].eq(report_detail["Accper"]).sum(),
)

print("财报类型分布：")
print(
    report_detail["Reptyp"]
    .value_counts(dropna=False)
    .sort_index()
)

print("财报明细前5行：")
print(report_detail.head().to_string(index=False))


# 90. 清理财报公告日期，说明书中提示了为占位符
report_clean = report_detail.loc[
    report_detail["Annodt"].notna()
    & report_detail["Annodt"].ne(report_detail["Accper"])
].copy()


# 91. 读取已有的正式财报模式标签
report_pattern_path = (
    root
    / "中间结果"
    / "02_变量构造结果"
    / "6PatternFirmQtr"
    / "PatternFirmQtr.csv"
)

report_pattern = pd.read_csv(
    report_pattern_path,
    usecols=["Stkcd", "Accper", "PatternFirmQtr"],
    dtype={
        "Stkcd": "string",
        "PatternFirmQtr": "Int8",
    },
    parse_dates=["Accper"],
    encoding="utf-8-sig",
)

report_pattern["Stkcd"] = (
    report_pattern["Stkcd"].str.strip().str.zfill(6)
)


# 92. 按公司×报告期一对一合并
report_clean = report_clean.merge(
    report_pattern,
    on=["Stkcd", "Accper"],
    how="left",
    validate="one_to_one",
    indicator=True,
)


# 93. 分别检查未匹配和已匹配但标签缺失的记录
print(
    "未匹配到财报模式表的记录数：",
    report_clean["_merge"].eq("left_only").sum(),
)

print(
    "已匹配但财报模式标签缺失的记录数：",
    (
        report_clean["_merge"].eq("both")
        & report_clean["PatternFirmQtr"].isna()
    ).sum(),
)

# 删除临时合并状态字段
report_clean = report_clean.drop(columns="_merge")

print("清理并合并后的财报明细行列数：", report_clean.shape)

print("正式财报模式标签分布：")
print(
    report_clean["PatternFirmQtr"]
    .value_counts(dropna=False)
    .sort_index()
)


# 98. 保留公司代码、目标报告期和首次预告日期
period_events = pattern_forecast[
    ["Stkcd", "Accper", "ForecastDate"]
].copy()


# 99. 按“同一目标报告期 + 同一预告日期”配对
period_pairs = period_events.merge(
    period_events,
    on=["Accper", "ForecastDate"],
    suffixes=("A", "B")
)

# 只保留代码较小的公司在前：排除自身配对和反向重复
period_pairs = period_pairs.loc[
    period_pairs["StkcdA"] < period_pairs["StkcdB"],
    ["StkcdA", "StkcdB", "Accper", "ForecastDate"]
].copy()

period_pairs = period_pairs.sort_values(
    ["ForecastDate", "Accper", "StkcdA", "StkcdB"]
).reset_index(drop=True)


# 100. 核对公司对数量与唯一性
period_company_count = period_events.groupby(
    ["Accper", "ForecastDate"]
)["Stkcd"].nunique()

expected_period_pairs = (
    period_company_count * (period_company_count - 1) // 2
).sum()

print("同日、同报告期的理论公司对数：", expected_period_pairs)
print("同日、同报告期的实际公司对数：", len(period_pairs))

print(
    "公司对×报告期重复数：",
    period_pairs.duplicated(
        ["StkcdA", "StkcdB", "Accper"]
    ).sum()
)

print("原来仅要求同日的公司对数：", len(history_pairs))
print("增加同报告期条件后减少的记录数：",
      len(history_pairs) - len(period_pairs))

print("同日、同报告期的历史公司对前10行：")
print(period_pairs.head(10).to_string(index=False))




# 101. 将每组公司对展开为两个方向
period_forward = period_pairs.rename(columns={
    "StkcdA": "Stkcd",
    "StkcdB": "StkcdRecall"
})

period_reverse = period_pairs.rename(columns={
    "StkcdB": "Stkcd",
    "StkcdA": "StkcdRecall"
})


# 102. 合并两个方向，将共同预告日命名为记忆形成日
period_directed = pd.concat(
    [period_forward, period_reverse],
    ignore_index=True
)

period_directed = period_directed.rename(columns={
    "ForecastDate": "MemoryDate"
})

period_directed = period_directed[
    ["Stkcd", "StkcdRecall", "Accper", "MemoryDate"]
].sort_values(
    ["MemoryDate", "Accper", "Stkcd", "StkcdRecall"]
).reset_index(drop=True)


# 103. 核对展开后的关系
print("理论双向关系记录数：", len(period_pairs) * 2)
print("实际双向关系记录数：", len(period_directed))

print(
    "提示公司×被提示公司×报告期重复数：",
    period_directed.duplicated(
        ["Stkcd", "StkcdRecall", "Accper"]
    ).sum()
)

print(
    "自身配对数：",
    (period_directed["Stkcd"]
     == period_directed["StkcdRecall"]).sum()
)

print("展开后的前10行：")
print(period_directed.head(10).to_string(index=False))



# 104. 提取可以作为提示事件的模式财报，保留报告期
cue_reports = report_clean.loc[
    report_clean["PatternFirmQtr"].eq(1).fillna(False),
    ["Stkcd", "Accper", "Reptyp", "Annodt"]
].copy()

cue_reports = cue_reports.rename(columns={
    "Annodt": "CueAnnodt"
})

print("可作为提示事件的模式财报数：", len(cue_reports))


# 105. 按“提示公司 + 同一报告期”匹配
# 每条历史关系最多匹配一份同公司、同报告期的财报
period_report_match = period_directed.merge(
    cue_reports,
    on=["Stkcd", "Accper"],
    how="left",
    validate="many_to_one",
    indicator=True
)

print("连接前记录数：", len(period_directed))
print("连接后记录数：", len(period_report_match))

print(
    "匹配到同报告期模式财报的关系数：",
    period_report_match["_merge"].eq("both").sum()
)

print(
    "未匹配到同报告期模式财报的关系数：",
    period_report_match["_merge"].eq("left_only").sum()
)


# 106. 检查时间顺序
print(
    "共同预告日等于财报公告日的记录数：",
    (period_report_match["MemoryDate"]
     == period_report_match["CueAnnodt"]).sum()
)

print(
    "共同预告日晚于财报公告日的记录数：",
    (period_report_match["MemoryDate"]
     > period_report_match["CueAnnodt"]).sum()
)


# 107. 仅保留成功匹配，且共同预告日严格早于财报公告日的关系
period_cue_links = period_report_match.loc[
    period_report_match["_merge"].eq("both")
    & (
        period_report_match["MemoryDate"]
        < period_report_match["CueAnnodt"]
    ),
    [
        "Stkcd", "StkcdRecall", "Accper",
        "MemoryDate", "CueAnnodt", "Reptyp"
    ]
].copy()

period_cue_links = period_cue_links.sort_values(
    ["CueAnnodt", "Stkcd", "StkcdRecall", "Accper"]
).reset_index(drop=True)

print("满足同报告期及时间顺序条件的关系数：", len(period_cue_links))

print(
    "提示公司×被提示公司×报告期重复数：",
    period_cue_links.duplicated(
        ["Stkcd", "StkcdRecall", "Accper"]
    ).sum()
)

print("匹配结果前10行：")
print(period_cue_links.head(10).to_string(index=False))


# 108. 用全部财报检查历史关系的覆盖情况
report_coverage = period_directed.merge(
    report_clean[["Stkcd", "Accper", "PatternFirmQtr"]],
    on=["Stkcd", "Accper"],
    how="left",
    validate="many_to_one",
    indicator=True
)


# 109. 区分模式、非模式、标签缺失和没有匹配记录
report_coverage["ReportStatus"] = (
    report_coverage["PatternFirmQtr"]
    .map({
        0: "非模式财报",
        1: "模式财报"
    })
    .fillna("财报模式标签缺失")
)

report_coverage.loc[
    report_coverage["_merge"].eq("left_only"),
    "ReportStatus"
] = "未匹配到财报记录"

print("历史关系对应的财报状态分布：")
print(report_coverage["ReportStatus"].value_counts())


# 110. 看看13942条关系实际来自多少份提示财报
trigger_report_count = period_cue_links[
    ["Stkcd", "Accper"]
].drop_duplicates().shape[0]

print("实际触发的提示公司×报告期数：", trigger_report_count)
print("涉及提示公司数：", period_cue_links["Stkcd"].nunique())
print("涉及被提示公司数：", period_cue_links["StkcdRecall"].nunique())


# 111. 提取全部正式财报
# 包括模式标签缺失、但财报日期有效的记录
all_cue_reports = report_clean[
    ["Stkcd", "Accper", "Reptyp", "Annodt", "PatternFirmQtr"]
].copy()

all_cue_reports = all_cue_reports.rename(columns={
    "Annodt": "CueAnnodt"
})

all_report_match = period_directed.merge(
    all_cue_reports,
    on=["Stkcd", "Accper"],
    how="inner",
    validate="many_to_one"
)

print("全部财报匹配后、日期筛选前的关系数：",
      len(all_report_match))


# 112. 仍要求共同预告日严格早于正式财报公告日
print(
    "共同预告日不早于财报公告日的关系数：",
    (all_report_match["MemoryDate"]
     >= all_report_match["CueAnnodt"]).sum()
)

period_cue_links_all = all_report_match.loc[
    all_report_match["MemoryDate"]
    < all_report_match["CueAnnodt"]
].copy()


# 113. 比较两种触发口径
comparison = pd.DataFrame({
    "触发口径": [
        "仅模式正式财报",
        "全部正式财报"
    ],
    "关系条数": [
        len(period_cue_links),
        len(period_cue_links_all)
    ],
    "提示公司×报告期数": [
        period_cue_links[
            ["Stkcd", "Accper"]
        ].drop_duplicates().shape[0],

        period_cue_links_all[
            ["Stkcd", "Accper"]
        ].drop_duplicates().shape[0]
    ],
    "提示公司×公告日数": [
        period_cue_links[
            ["Stkcd", "CueAnnodt"]
        ].drop_duplicates().shape[0],

        period_cue_links_all[
            ["Stkcd", "CueAnnodt"]
        ].drop_duplicates().shape[0]
    ]
})

print("两种财报触发口径的比较：")
print(comparison.to_string(index=False))


# 114. 提取不同的提示财报公告日期
# 全部财报版本包含严格版本，因此只需建立一张日期对照表
cue_dates = (
    period_cue_links_all[["CueAnnodt"]]
    .drop_duplicates()
    .sort_values("CueAnnodt")
    .reset_index(drop=True)
)

# 提取交易日
cue_open_dates = (
    trade_calendar.loc[
        trade_calendar["Statse"].eq("O"),
        ["Clddt"]
    ]
    .rename(columns={"Clddt": "CueTradingDate"})
    .sort_values("CueTradingDate")
    .reset_index(drop=True)
)


# 115. 保留交易日原日，非交易日寻找下一交易日
cue_date_map = pd.merge_asof(
    cue_dates,
    cue_open_dates,
    left_on="CueAnnodt",
    right_on="CueTradingDate",
    direction="forward",
    allow_exact_matches=True
)

# 日历起点之前的日期不能直接映射到首个交易日
cue_date_map.loc[
    cue_date_map["CueAnnodt"] < trade_calendar["Clddt"].min(),
    "CueTradingDate"
] = pd.NaT


# 116. 检查日期映射及日历覆盖情况
print("不同财报公告日期总数：", len(cue_date_map))

print(
    "保持原日期的日期数：",
    (cue_date_map["CueTradingDate"]
     == cue_date_map["CueAnnodt"]).sum()
)

print(
    "顺延至下一交易日的日期数：",
    (cue_date_map["CueTradingDate"]
     > cue_date_map["CueAnnodt"]).sum()
)

print(
    "未找到可用交易日的日期数：",
    cue_date_map["CueTradingDate"].isna().sum()
)

if cue_date_map["CueTradingDate"].isna().any():
    print("未映射的财报公告日期：")
    print(
        cue_date_map.loc[
            cue_date_map["CueTradingDate"].isna(),
            ["CueAnnodt"]
        ].to_string(index=False)
    )

print("日期映射前10行：")
print(cue_date_map.head(10).to_string(index=False))


# 117. 将提示交易日合回两版关系表
period_cue_links_trade = period_cue_links.merge(
    cue_date_map,
    on="CueAnnodt",
    how="left",
    validate="many_to_one"
)

period_cue_links_all_trade = period_cue_links_all.merge(
    cue_date_map,
    on="CueAnnodt",
    how="left",
    validate="many_to_one"
)


# 118. 单独保留尚未映射的关系
strict_unmapped = period_cue_links_trade.loc[
    period_cue_links_trade["CueTradingDate"].isna()
].copy()

all_unmapped = period_cue_links_all_trade.loc[
    period_cue_links_all_trade["CueTradingDate"].isna()
].copy()


# 119. 统计日历覆盖不足的影响
mapping_summary = pd.DataFrame({
    "触发口径": [
        "仅模式正式财报",
        "全部正式财报"
    ],
    "合并后关系数": [
        len(period_cue_links_trade),
        len(period_cue_links_all_trade)
    ],
    "未映射关系数": [
        len(strict_unmapped),
        len(all_unmapped)
    ],
    "未映射提示财报数": [
        strict_unmapped[
            ["Stkcd", "Accper"]
        ].drop_duplicates().shape[0],

        all_unmapped[
            ["Stkcd", "Accper"]
        ].drop_duplicates().shape[0]
    ]
})

print("交易日期映射的覆盖情况：")
print(mapping_summary.to_string(index=False))


# 120. 全部财报版：汇总每家公司每天收到的提示
memory_all = (
    period_cue_links_all_trade
    .dropna(subset=["CueTradingDate"])
    .groupby(
        ["StkcdRecall", "CueTradingDate"],
        as_index=False
    )
    .agg(nCueFirms=("Stkcd", "nunique"))
)


# 121. 模式财报版：采用相同方式汇总
memory_strict = (
    period_cue_links_trade
    .dropna(subset=["CueTradingDate"])
    .groupby(
        ["StkcdRecall", "CueTradingDate"],
        as_index=False
    )
    .agg(nPatternCueFirms=("Stkcd", "nunique"))
)


# 122. 将两版汇总结果合并
memory_matrix = memory_all.merge(
    memory_strict,
    on=["StkcdRecall", "CueTradingDate"],
    how="left",
    validate="one_to_one"
)

# 没有匹配到模式财报提示时，模式提示公司数记为0
memory_matrix["nPatternCueFirms"] = (
    memory_matrix["nPatternCueFirms"]
    .fillna(0)
    .astype(int)
)

# 当前表只收录全部财报版中存在提示的公司日
memory_matrix["MemoryForecast"] = 1

memory_matrix["MemoryForecastStrict"] = (
    memory_matrix["nPatternCueFirms"].gt(0).astype(int)
)

memory_matrix = memory_matrix.sort_values(
    ["CueTradingDate", "StkcdRecall"]
).reset_index(drop=True)


# 123. 检查汇总结果
print("全部财报版触发的公司×交易日数：", len(memory_all))
print("模式财报版触发的公司×交易日数：", len(memory_strict))
print("记忆汇总表的行数和列数：", memory_matrix.shape)

print(
    "被提示公司×交易日重复数：",
    memory_matrix.duplicated(
        ["StkcdRecall", "CueTradingDate"]
    ).sum()
)

print(
    "模式提示公司数超过全部提示公司数的行数：",
    (memory_matrix["nPatternCueFirms"]
     > memory_matrix["nCueFirms"]).sum()
)

print("记忆汇总表前10行：")
print(memory_matrix.head(10).to_string(index=False))





# 124. 将全部正式财报公告日映射到交易日
own_report_date_map = pd.merge_asof(
    report_clean[["Annodt"]]
    .drop_duplicates()
    .sort_values("Annodt"),

    cue_open_dates.rename(columns={
        "CueTradingDate": "ReportTradingDate"
    }),

    left_on="Annodt",
    right_on="ReportTradingDate",
    direction="forward",
    allow_exact_matches=True
)

own_report_date_map.loc[
    own_report_date_map["Annodt"]
    < trade_calendar["Clddt"].min(),
    "ReportTradingDate"
] = pd.NaT


# 125. 汇总每家公司每个交易日公布的自身财报数量
own_report_detail = report_clean[
    ["Stkcd", "Accper", "Annodt"]
].merge(
    own_report_date_map,
    on="Annodt",
    how="left",
    validate="many_to_one"
)

own_report_days = (
    own_report_detail
    .dropna(subset=["ReportTradingDate"])
    .groupby(
        ["Stkcd", "ReportTradingDate"],
        as_index=False
    )
    .agg(nOwnReports=("Accper", "nunique"))
    .rename(columns={
        "Stkcd": "StkcdRecall",
        "ReportTradingDate": "CueTradingDate"
    })
)


# 126. 给被提示公司日添加自身财报标签
memory_matrix = memory_matrix.merge(
    own_report_days,
    on=["StkcdRecall", "CueTradingDate"],
    how="left",
    validate="one_to_one"
)

memory_matrix["nOwnReports"] = (
    memory_matrix["nOwnReports"].fillna(0).astype(int)
)

memory_matrix["OwnReportDay"] = (
    memory_matrix["nOwnReports"].gt(0).astype(int)
)


# 127. 查看自身财报交易日重合情况
print("添加标签后的行数和列数：", memory_matrix.shape)

print("自身财报交易日重合标签分布：")
print(memory_matrix["OwnReportDay"].value_counts())

print(
    "严格版中自身财报交易日重合的公司日数：",
    (
        memory_matrix["MemoryForecastStrict"].eq(1)
        & memory_matrix["OwnReportDay"].eq(1)
    ).sum()
)



# 128. 读取项目现有的股票日度样本母表
daily_index_path = (
    root
    / "中间结果/01_清洗后样本"
    / "基础清洗后个股日度样本索引_td.csv"
)

daily_index = pd.read_csv(
    daily_index_path,
    usecols=["Stkcd", "Trddt", "td"],
    dtype={
        "Stkcd": "string",
        "td": "Int64"
    },
    parse_dates=["Trddt"]
)

daily_index["Stkcd"] = (
    daily_index["Stkcd"].str.strip().str.zfill(6)
)


# 129. 检查母表的唯一键、缺失和日期覆盖
print("日度母表的行数和列数：", daily_index.shape)

print("关键字段缺失数：")
print(daily_index[["Stkcd", "Trddt", "td"]].isna().sum())

print(
    "股票×交易日重复数：",
    daily_index.duplicated(["Stkcd", "Trddt"]).sum()
)

print("涉及公司数：", daily_index["Stkcd"].nunique())

print(
    "日度母表日期范围：",
    daily_index["Trddt"].min(),
    "至",
    daily_index["Trddt"].max()
)

print(
    "交易日历终点之后的公司日数：",
    (
        daily_index["Trddt"]
        > trade_calendar["Clddt"].max()
    ).sum()
)

print("日度母表前5行：")
print(daily_index.head().to_string(index=False))



# 130. 将记忆表的键改为日度母表使用的名称
# 此处 Stkcd 代表被提示公司
memory_for_merge = memory_matrix.rename(columns={
    "StkcdRecall": "Stkcd",
    "CueTradingDate": "Trddt"
})


# 131. 从大母表中提取可能匹配的公司日，减少合并开销
daily_event_keys = daily_index.loc[
    daily_index["Stkcd"].isin(memory_for_merge["Stkcd"])
    & daily_index["Trddt"].isin(memory_for_merge["Trddt"]),
    ["Stkcd", "Trddt", "td"]
]

memory_sample_match = memory_for_merge.merge(
    daily_event_keys,
    on=["Stkcd", "Trddt"],
    how="left",
    validate="one_to_one",
    indicator=True
)


# 132. 区分匹配成功、日期越界和范围内未匹配
daily_start = daily_index["Trddt"].min()
daily_end = daily_index["Trddt"].max()

memory_sample_match["MatchStatus"] = "母表日期范围内未匹配"

outside_dates = (
    memory_sample_match["Trddt"].lt(daily_start)
    | memory_sample_match["Trddt"].gt(daily_end)
)

memory_sample_match.loc[
    outside_dates, "MatchStatus"
] = "超出母表日期范围"

memory_sample_match.loc[
    memory_sample_match["_merge"].eq("both"),
    "MatchStatus"
] = "匹配成功"


# 133. 查看两版匹配情况，并保留匹配与未匹配明细
print(
    "连接前、后的触发公司日数：",
    len(memory_matrix),
    len(memory_sample_match)
)

print("全部财报版的匹配状态：")
print(memory_sample_match["MatchStatus"].value_counts())

print("模式财报版的匹配状态：")
print(
    memory_sample_match.loc[
        memory_sample_match["MemoryForecastStrict"].eq(1),
        "MatchStatus"
    ].value_counts()
)

memory_in_sample = memory_sample_match.loc[
    memory_sample_match["_merge"].eq("both")
].drop(columns=["_merge", "MatchStatus"]).copy()

memory_not_in_sample = memory_sample_match.loc[
    memory_sample_match["_merge"].eq("left_only")
].copy()


# 134. 检查未匹配事件的公司是否曾经进入日度母表
sample_firms = set(daily_index["Stkcd"].unique())

memory_not_in_sample = memory_not_in_sample.copy()

# UnmatchedReason：未匹配分类
memory_not_in_sample["UnmatchedReason"] = "公司在母表中，但该日无记录"

memory_not_in_sample.loc[
    ~memory_not_in_sample["Stkcd"].isin(sample_firms),
    "UnmatchedReason"
] = "公司从未进入母表"

print("全部财报版未匹配事件的分类：")
print(
    memory_not_in_sample["UnmatchedReason"]
    .value_counts(dropna=False)
)

print("模式财报版未匹配事件的分类：")
print(
    memory_not_in_sample.loc[
        memory_not_in_sample["MemoryForecastStrict"].eq(1),
        "UnmatchedReason"
    ].value_counts(dropna=False)
)


# 135. 检查样本内触发事件与自身财报是否重合
print("样本内全部财报版的自身财报重合分布：")
print(
    memory_in_sample["OwnReportDay"]
    .value_counts(dropna=False)
    .sort_index()
)

print("样本内模式财报版的自身财报重合分布：")
print(
    memory_in_sample.loc[
        memory_in_sample["MemoryForecastStrict"].eq(1),
        "OwnReportDay"
    ].value_counts(dropna=False).sort_index()
)



# 模式财报版未匹配事件的分类
print("模式财报版未匹配事件的分类：")
print(
    memory_not_in_sample.loc[
        memory_not_in_sample["MemoryForecastStrict"].eq(1),
        "UnmatchedReason"
    ].value_counts(dropna=False)
)

# OwnReportDay：自身财报交易日重合标记
print("样本内全部财报版的自身财报重合分布：")
print(
    memory_in_sample["OwnReportDay"]
    .value_counts(dropna=False)
    .sort_index()
)

print("样本内模式财报版的自身财报重合分布：")
print(
    memory_in_sample.loc[
        memory_in_sample["MemoryForecastStrict"].eq(1),
        "OwnReportDay"
    ].value_counts(dropna=False).sort_index()
)




# 136. 将四个记忆变量合入完整日度母表
# 此处 Stkcd 表示被提示公司，Trddt 表示交易日期
memory_daily = daily_index.merge(
    memory_in_sample[
        [
            "Stkcd",
            "Trddt",
            "nCueFirms",
            "nPatternCueFirms",
            "MemoryForecast",
            "MemoryForecastStrict"
        ]
    ],
    on=["Stkcd", "Trddt"],
    how="left",
    validate="one_to_one"
)

print("合并记忆变量后的行数和列数：")
print(memory_daily.shape)


# 137. 整理完整的自身财报日表
own_report_for_daily = own_report_days.rename(
    columns={
        "StkcdRecall": "Stkcd",
        "CueTradingDate": "Trddt"
    }
)[["Stkcd", "Trddt", "nOwnReports"]]

# 合入自身财报份数
memory_daily = memory_daily.merge(
    own_report_for_daily,
    on=["Stkcd", "Trddt"],
    how="left",
    validate="one_to_one"
)

# nOwnReports：当天匹配到的自身财报份数
memory_daily["nOwnReports"] = (
    memory_daily["nOwnReports"].fillna(0).astype("int64")
)

# OwnReportDay：当天是否匹配到自身财报
memory_daily["OwnReportDay"] = (
    memory_daily["nOwnReports"].gt(0).astype("int8")
)

print("合并自身财报标签后的行数和列数：")
print(memory_daily.shape)

print("全部财报版记忆标记分布：")
print(memory_daily["MemoryForecast"].value_counts(dropna=False))

print("模式财报版记忆标记分布：")
print(memory_daily["MemoryForecastStrict"].value_counts(dropna=False))



# 138. 按预告目标报告期检查识别覆盖情况
forecast_coverage = pattern_work.groupby("Accper").agg(
    首次预告数=("Stkcd", "size"),
    三年历史充分数=("HistoryEligible", "sum"),
    模式标签可判断数=("PatternForecastQtr", "count"),
    已识别模式数=("PatternForecastQtr", "sum")
)

forecast_coverage["模式标签无法判断数"] = (
    forecast_coverage["首次预告数"]
    - forecast_coverage["模式标签可判断数"]
)

print("最早12个目标报告期的覆盖情况：")
print(forecast_coverage.head(12).to_string())

print("最近8个目标报告期的覆盖情况：")
print(forecast_coverage.tail(8).to_string())



# 139. 添加主研究期标记
memory_daily["MemoryStudySample"] = (
    memory_daily["Trddt"].ge(pd.Timestamp("2008-01-01"))
    .astype("int8")
)

study_mask = memory_daily["MemoryStudySample"].eq(1)

# 仅填补研究期内的四个记忆变量
memory_fields = [
    "nCueFirms",
    "nPatternCueFirms",
    "MemoryForecast",
    "MemoryForecastStrict"
]

for col in memory_fields:
    memory_daily.loc[study_mask, col] = (
        memory_daily.loc[study_mask, col].fillna(0)
    )

    # 使用可保留缺失值的整数类型
    memory_daily[col] = memory_daily[col].astype("Int64")



    # 140. 检查主研究期的记忆指标
print("添加研究期标记后的行数和列数：")
print(memory_daily.shape)

print("研究期标记分布：")
print(memory_daily["MemoryStudySample"].value_counts().sort_index())

print("主样本全部财报版记忆标记分布：")
print(
    memory_daily.loc[study_mask, "MemoryForecast"]
    .value_counts(dropna=False).sort_index()
)

print("主样本模式财报版记忆标记分布：")
print(
    memory_daily.loc[study_mask, "MemoryForecastStrict"]
    .value_counts(dropna=False).sort_index()
)




# 141. 提取主样本内的触发公司日
memory_event_main = memory_daily.loc[
    study_mask & memory_daily["MemoryForecast"].eq(1),
    ["Stkcd", "Trddt", "MemoryForecastStrict", "OwnReportDay"]
]

print("主样本全部财报版的自身财报重合分布：")
print(
    memory_event_main["OwnReportDay"]
    .value_counts(dropna=False).sort_index()
)

print("主样本模式财报版的自身财报重合分布：")
print(
    memory_event_main.loc[
        memory_event_main["MemoryForecastStrict"].eq(1),
        "OwnReportDay"
    ].value_counts(dropna=False).sort_index()
)




# 142. 读取当日特征调整收益率
dgtw_path = (
    str(root / '中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW特征调整收益率.csv')
)

dgtw_daily = pd.read_csv(
    dgtw_path,
    usecols=["Stkcd", "Trddt", "chAdj_ret_t"],
    dtype={"Stkcd": "string"},
    parse_dates=["Trddt"],
    encoding="utf-8-sig"
)

print("当日特征调整收益表的行数和列数：")
print(dgtw_daily.shape)

print("股票×交易日重复数：")
print(dgtw_daily.duplicated(["Stkcd", "Trddt"]).sum())

print("收益表日期范围：")
print(
    dgtw_daily["Trddt"].min(),
    "至",
    dgtw_daily["Trddt"].max()
)

print("各字段缺失数：")
print(dgtw_daily.isna().sum())



# 143. 合并当日特征调整收益率
# 此处 Stkcd 表示被提示公司
memory_daily = memory_daily.merge(
    dgtw_daily,
    on=["Stkcd", "Trddt"],
    how="left",
    validate="one_to_one"
)

print("合并当日收益后的行数和列数：")
print(memory_daily.shape)


# 144. 提取主样本触发事件及其当日收益
memory_event_main = memory_daily.loc[
    memory_daily["MemoryStudySample"].eq(1)
    & memory_daily["MemoryForecast"].eq(1),
    [
        "Stkcd", "Trddt", "MemoryForecastStrict",
        "OwnReportDay", "chAdj_ret_t"
    ]
].copy()

# ReturnMatchStatus：收益匹配状态
memory_event_main["ReturnMatchStatus"] = "收益日期范围内未匹配"

memory_event_main.loc[
    memory_event_main["chAdj_ret_t"].notna(),
    "ReturnMatchStatus"
] = "匹配成功"

memory_event_main.loc[
    ~memory_event_main["Trddt"].between(
        dgtw_daily["Trddt"].min(),
        dgtw_daily["Trddt"].max()
    ),
    "ReturnMatchStatus"
] = "超出收益日期范围"

print("全部财报版触发事件的收益匹配状态：")
print(memory_event_main["ReturnMatchStatus"].value_counts())

print("模式财报版触发事件的收益匹配状态：")
print(
    memory_event_main.loc[
        memory_event_main["MemoryForecastStrict"].eq(1),
        "ReturnMatchStatus"
    ].value_counts()
)


# 145. 收益有效样本标记
memory_daily["ReturnSample"] = (
    memory_daily["MemoryStudySample"].eq(1)
    & memory_daily["chAdj_ret_t"].notna()
).astype("int8")

# 收益有效、且不处于自身财报日的样本标记
memory_daily["ReturnSampleNoOwn"] = (
    memory_daily["ReturnSample"].eq(1)
    & memory_daily["OwnReportDay"].eq(0)
).astype("int8")


# 146. 汇总收益分析样本
sample_check = []

for label, flag in [
    ("保留自身财报日", "ReturnSample"),
    ("排除自身财报日", "ReturnSampleNoOwn")
]:
    use = memory_daily[flag].eq(1)

    sample_check.append({
        "样本口径": label,
        "公司日数": int(use.sum()),
        "全部版触发数": int(
            memory_daily.loc[use, "MemoryForecast"].sum()
        ),
        "严格版触发数": int(
            memory_daily.loc[use, "MemoryForecastStrict"].sum()
        )
    })

print("两种收益分析样本口径：")
print(pd.DataFrame(sample_check).to_string(index=False))




# 147. 建立收益描述统计工作表
reaction_work = memory_daily.loc[
    memory_daily["ReturnSampleNoOwn"].eq(1),
    ["MemoryForecast", "MemoryForecastStrict", "chAdj_ret_t"]
].copy()

# MemoryGroup：记忆触发组别
# 0：未识别到触发；1：仅全部版触发；2：严格版触发
reaction_work["MemoryGroup"] = (
    reaction_work["MemoryForecast"].astype("int8")
)

reaction_work.loc[
    reaction_work["MemoryForecastStrict"].eq(1),
    "MemoryGroup"
] = 2




# 148. 按触发组别统计当日特征调整收益率
ret_summary = reaction_work.groupby(
    "MemoryGroup"
)["chAdj_ret_t"].agg(
    公司日数="size",
    平均收益="mean",
    收益中位数="median",
    收益标准差="std"
)

ret_summary = ret_summary.rename(index={
    0: "未识别到触发",
    1: "仅全部版触发",
    2: "严格版触发"
})
ret_summary.index.name = "触发组别"

return_stat_cols = ["平均收益", "收益中位数", "收益标准差"]
ret_summary[return_stat_cols] = (
    ret_summary[return_stat_cols] * 100
)

print("当日特征调整收益描述统计，收益单位为百分比：")
print(ret_summary.round(4).to_string())






# 149. 读取两个累计特征调整收益率字段
dgtw_windows = pd.read_csv(
    dgtw_path,
    usecols=[
        "Stkcd",
        "Trddt",
        "chAdj_ret_t_t1",
        "chAdj_ret_t_t10"
    ],
    dtype={"Stkcd": "string"},
    parse_dates=["Trddt"],
    encoding="utf-8-sig"
)

print("两个累计收益字段的缺失数：")
print(
    dgtw_windows[
        ["chAdj_ret_t_t1", "chAdj_ret_t_t10"]
    ].isna().sum()
)

# 150. 合入完整日度记忆表
memory_daily = memory_daily.merge(
    dgtw_windows,
    on=["Stkcd", "Trddt"],
    how="left",
    validate="one_to_one"
)

print("合并累计收益后的行数和列数：")
print(memory_daily.shape)




# 151. 建立市场交易日序号
open_dates = (
    cue_open_dates[["CueTradingDate"]]
    .sort_values("CueTradingDate")
    .reset_index(drop=True)
)
open_dates["TradeNo"] = open_dates.index

own_report_seq = own_report_days[
    ["StkcdRecall", "CueTradingDate"]
].merge(
    open_dates,
    on="CueTradingDate",
    how="left",
    validate="many_to_one"
)

assert own_report_seq["TradeNo"].notna().all(), \
    "存在未能定位到交易日历的自身财报日"




# 152. 生成窗口内自身财报标签
window_parts = []

for k in range(11):
    part = own_report_seq[["StkcdRecall", "TradeNo"]].copy()

    part["TradeNo"] = part["TradeNo"] - k
    part["OwnReportIn2D"] = int(k <= 1)

    window_parts.append(part)

own_window_marks = pd.concat(window_parts, ignore_index=True)

# 同一个窗口可能包含多份财报，标签取最大值
own_window_marks = own_window_marks.groupby(
    ["StkcdRecall", "TradeNo"],
    as_index=False
)["OwnReportIn2D"].max()

own_window_marks["OwnReportIn11D"] = 1
own_window_marks = own_window_marks.loc[
    own_window_marks["TradeNo"].ge(0)
]

# 将交易日序号映射回真实日期
own_window_marks = own_window_marks.merge(
    open_dates,
    on="TradeNo",
    how="left",
    validate="many_to_one"
).rename(columns={
    "StkcdRecall": "Stkcd",
    "CueTradingDate": "Trddt"
})[
    ["Stkcd", "Trddt", "OwnReportIn2D", "OwnReportIn11D"]
]

print("自身财报窗口标签表的行数和列数：")
print(own_window_marks.shape)

print("公司×起始交易日重复数：")
print(own_window_marks.duplicated(["Stkcd", "Trddt"]).sum())




# 153. 合入自身财报窗口标签
n_before = len(memory_daily)

memory_daily = memory_daily.merge(
    own_window_marks,
    on=["Stkcd", "Trddt"],
    how="left",
    validate="one_to_one"
)

assert len(memory_daily) == n_before, "连接后行数发生变化"


# 154. 仅在日历完整覆盖的窗口内，将缺失标签补0
# 2日窗口包含当日及后1个交易日
# 11日窗口包含当日及后10个交易日

calendar_start = open_dates["CueTradingDate"].iloc[0]

for column, forward_days in [
    ("OwnReportIn2D", 1),
    ("OwnReportIn11D", 10)
]:
    last_complete_start = (
        open_dates["CueTradingDate"].iloc[-(forward_days + 1)]
    )

    complete_window = memory_daily["Trddt"].between(
        calendar_start,
        last_complete_start
    )

    memory_daily.loc[complete_window, column] = (
        memory_daily.loc[complete_window, column].fillna(0)
    )

    memory_daily[column] = memory_daily[column].astype("Int8")


print("合并自身财报窗口标签后的行数和列数：")
print(memory_daily.shape)

print("2日窗口自身财报标签分布：")
print(memory_daily["OwnReportIn2D"].value_counts(dropna=False))

print("11日窗口自身财报标签分布：")
print(memory_daily["OwnReportIn11D"].value_counts(dropna=False))

# 155. 建立2日累计收益样本：当日至后1个交易日
memory_daily["ReturnSample2DNoOwn"] = (
    memory_daily["MemoryStudySample"].eq(1)
    & memory_daily["chAdj_ret_t_t1"].notna()
    & memory_daily["OwnReportIn2D"].eq(0)
).fillna(False).astype("int8")


# 156. 建立11日累计收益样本：当日至后10个交易日
memory_daily["ReturnSample11DNoOwn"] = (
    memory_daily["MemoryStudySample"].eq(1)
    & memory_daily["chAdj_ret_t_t10"].notna()
    & memory_daily["OwnReportIn11D"].eq(0)
).fillna(False).astype("int8")


# 157. 比较两个窗口的样本量及触发公司日数
window_summary = []

for sample_column, sample_name in [
    ("ReturnSample2DNoOwn", "2日窗口，排除窗口内自身财报"),
    ("ReturnSample11DNoOwn", "11日窗口，排除窗口内自身财报")
]:
    sample_mask = memory_daily[sample_column].eq(1)

    window_summary.append({
        "样本口径": sample_name,
        "公司日数": int(sample_mask.sum()),
        "全部版触发数": int((
            sample_mask
            & memory_daily["MemoryForecast"].eq(1)
        ).sum()),
        "严格版触发数": int((
            sample_mask
            & memory_daily["MemoryForecastStrict"].eq(1)
        ).sum())
    })

print("添加累计收益样本标记后的行数和列数：")
print(memory_daily.shape)

print("两种累计收益窗口的样本情况：")
print(pd.DataFrame(window_summary).to_string(index=False))





# 158. 主回归样本：研究期内，对应累计收益有效
memory_daily["ReturnSample2D"] = (
    memory_daily["MemoryStudySample"].eq(1)
    & memory_daily["chAdj_ret_t_t1"].notna()
).fillna(False).astype("int8")

memory_daily["ReturnSample11D"] = (
    memory_daily["MemoryStudySample"].eq(1)
    & memory_daily["chAdj_ret_t_t10"].notna()
).fillna(False).astype("int8")


# 159. 对比主回归与稳健性样本
sample_summary = []

for column, name in [
    ("ReturnSample2D", "2日主回归：保留自身财报"),
    ("ReturnSample2DNoOwn", "2日稳健性：排除窗口内自身财报"),
    ("ReturnSample11D", "11日主回归：保留自身财报"),
    ("ReturnSample11DNoOwn", "11日稳健性：排除窗口内自身财报")
]:
    sample_mask = memory_daily[column].eq(1)

    sample_summary.append({
        "样本口径": name,
        "公司日数": int(sample_mask.sum()),
        "全部版触发数": int((
            sample_mask
            & memory_daily["MemoryForecast"].eq(1)
        ).sum()),
        "严格版触发数": int((
            sample_mask
            & memory_daily["MemoryForecastStrict"].eq(1)
        ).sum())
    })

print("添加主回归样本标记后的行数和列数：")
print(memory_daily.shape)

print("主回归与稳健性样本比较：")
print(pd.DataFrame(sample_summary).to_string(index=False))




# 160. 生成自身财报前后10个交易日的控制长表
date_by_trade_no = open_dates.set_index(
    "TradeNo"
)["CueTradingDate"]

own_control_parts = []

for k in range(-10, 11):
    part = own_report_seq[
        ["StkcdRecall", "TradeNo"]
    ].copy()

    # 自身财报日 = 起始日 + k
    # 因此：起始日 = 自身财报日 - k
    part["Trddt"] = (
        part["TradeNo"] - k
    ).map(date_by_trade_no)

    part["RelativeDay"] = k

    # 去掉超出交易日历范围的起始日
    part = part.dropna(subset=["Trddt"])

    own_control_parts.append(
        part[["StkcdRecall", "Trddt", "RelativeDay"]]
    )

own_control_long = pd.concat(
    own_control_parts,
    ignore_index=True
).rename(columns={"StkcdRecall": "Stkcd"})

own_control_long["RelativeDay"] = (
    own_control_long["RelativeDay"].astype("int8")
)

# 长表只记录确认存在自身财报的情况
own_control_long["OwnReport"] = 1
own_control_long["OwnReport"] = (
    own_control_long["OwnReport"].astype("int8")
)


# 161. 检查长表
print("自身财报逐日控制长表的行数和列数：")
print(own_control_long.shape)

print("公司×起始交易日×相对日重复数：")
print(
    own_control_long.duplicated(
        ["Stkcd", "Trddt", "RelativeDay"]
    ).sum()
)

print("各相对交易日的记录数：")
print(
    own_control_long.groupby("RelativeDay")
    .size()
    .sort_index()
)

print("控制长表前5行：")
print(own_control_long.head().to_string(index=False))