# from pathlib import Path  # 处理文件路径
# from zipfile import ZipFile  # 读取压缩包

# # 数行者原始数据目录
# data_dir = Path(
#     "/Users/qiyuewu/Desktop/mmm复现数据包/input/DATAGO"
# )

# # 公司关系表压缩包
# relation_zip = (
#     data_dir
#     / "网络新闻相关上市公司表（2009-01-01~2025-12-31）"
#     / "网络新闻相关上市公司表（2009-01-01~2013-12-31）.zip"
# )

# # 情感结果表压缩包
# sentiment_zip = (
#     data_dir
#     / "网络新闻情感量化结果数据表（2009-01-01~2025-12-31）"
#     / "网络新闻情感量化结果数据表（2009-01-01~2013-12-31）.zip"
# )

# # 分别查看两个压缩包
# for zip_path in [relation_zip, sentiment_zip]:
#     print("\n压缩包：", zip_path.name)

#     with ZipFile(zip_path) as archive:
#         for file_name in archive.namelist():
#             print("  ", file_name)


# import io  # 在内存中读取文件内容
# import pandas as pd  # 处理表格数据

# # 一、查看公司关系表

# relation_file = (
#     "网络新闻相关上市公司表_2009-01-01~2013-12-31_1.xlsx"
# )

# with ZipFile(relation_zip) as archive:
#     relation_preview = pd.read_excel(
#         io.BytesIO(archive.read(relation_file)),
#         usecols=[
#             "news_id",             # 新闻编号
#             "pub_date",            # 发布时间
#             "stkcd",               # 股票代码
#             "is_ment_in_title",     # 标题是否提及该公司
#             "is_ment_in_content",   # 正文是否提及该公司
#         ],
#         dtype={
#             "news_id": "string",
#             "stkcd": "string",
#         },
#         nrows=5,
#     )

# print("\n公司关系表前5行：")
# print(relation_preview.to_string(index=False))


# # 二、查看情感结果表

# sentiment_file = (
#     "网络新闻情感量化结果数据表_2009-01-01~2013-12-31_1.xlsx"
# )

# with ZipFile(sentiment_zip) as archive:
#     sentiment_preview = pd.read_excel(
#         io.BytesIO(archive.read(sentiment_file)),
#         usecols=[
#             "news_id",                      # 新闻编号
#             "pub_date",                     # 发布时间
#             "overall_senti_score_adjust",   # 调整后的整体情感得分
#         ],
#         dtype={
#             "news_id": "string",
#         },
#         nrows=5,
#     )

# print("\n情感结果表前5行：")
# print(sentiment_preview.to_string(index=False))



# # 一、从发布时间提取自然日期
# for table in [relation_preview, sentiment_preview]:
#     table["date"] = (
#         pd.to_datetime(table["pub_date"], errors="raise")
#         .dt.normalize()
#     )

# # 二、按“新闻编号＋自然日期”合并情感
# merged_preview = relation_preview.merge(
#     sentiment_preview[
#         [
#             "news_id",
#             "date",
#             "overall_senti_score_adjust",
#         ]
#     ],
#     on=["news_id", "date"],
#     how="left",               # 保留全部公司关系
#     validate="many_to_one",    # 多条公司关系只能匹配一条新闻情感
#     indicator=True,           # 标记是否匹配成功
# )

# # 三、查看合并结果
# print("\n合并情感后的前5条公司关系：")
# print(
#     merged_preview[
#         [
#             "news_id",
#             "date",
#             "stkcd",
#             "overall_senti_score_adjust",
#             "_merge",
#         ]
#     ].to_string(index=False)
# )

# print("\n匹配情况：")
# print(merged_preview["_merge"].value_counts())


# # 一、同一篇新闻对同一家公司只保留一条关系
# # 标题、正文提及标记取最大值：只要出现过1，就保留1。
# # 情感得分在上一步已通过多对一匹配检查，因此取第一条即可。

# news_demo = (
#     merged_preview
#     .groupby(
#         ["news_id", "date", "stkcd"],
#         as_index=False,
#     )
#     .agg({
#         "is_ment_in_title": "max",
#         "is_ment_in_content": "max",
#         "overall_senti_score_adjust": "first",
#     })
# )

# # 二、把情感得分转成四类新闻计数标记
# score = pd.to_numeric(
#     news_demo["overall_senti_score_adjust"],
#     errors="coerce",
# )

# news_demo["Posnews_All"] = score.gt(0).astype(int)       # 正面
# news_demo["Neunews_All"] = score.eq(0).astype(int)       # 中性
# news_demo["Negnews_All"] = score.lt(0).astype(int)       # 负面
# news_demo["Unclassifiednews_All"] = score.isna().astype(int)  # 未分类

# # 每条去重后的新闻—公司关系计为一篇相关新闻
# news_demo["Allnews_All"] = 1

# # 标题和正文提及篇数分别统计
# news_demo["Newsnum_Title"] = news_demo["is_ment_in_title"]
# news_demo["Newsnum_Cont"] = news_demo["is_ment_in_content"]

# # 三、按公司和自然日加总
# count_cols = [
#     "Newsnum_Title",          # 标题提及篇数
#     "Newsnum_Cont",           # 正文提及篇数
#     "Posnews_All",            # 正面篇数
#     "Neunews_All",            # 中性篇数
#     "Negnews_All",            # 负面篇数
#     "Unclassifiednews_All",   # 未分类篇数
#     "Allnews_All",            # 全部相关新闻篇数
# ]

# daily_demo = (
#     news_demo
#     .groupby(["stkcd", "date"], as_index=False)[count_cols]
#     .sum()
#     .sort_values(["date", "stkcd"])
# )

# print("\n公司×自然日新闻统计：")
# print(daily_demo.to_string(index=False))



# # 一、找出情感压缩包中的全部数据工作簿
# # 通过文件名前缀筛选，排除字段说明等文件。

# sentiment_parts = []

# with ZipFile(sentiment_zip) as archive:
#     sentiment_files = sorted(
#         name
#         for name in archive.namelist()
#         if Path(name).name.startswith("网络新闻情感量化结果数据表_")
#         and name.lower().endswith(".xlsx")
#     )

#     if not sentiment_files:
#         raise ValueError("压缩包中没有找到情感数据工作簿。")

#     # 二、逐个读取，只保留必要的三列
#     for file_name in sentiment_files:
#         print("\n正在读取：", file_name, flush=True)

#         part = pd.read_excel(
#             io.BytesIO(archive.read(file_name)),
#             usecols=[
#                 "news_id",                     # 新闻编号
#                 "pub_date",                    # 发布时间
#                 "overall_senti_score_adjust",  # 调整后的整体情感得分
#             ],
#             dtype={"news_id": "string"},
#         )

#         print("本分片行数：", len(part), flush=True)
#         sentiment_parts.append(part)

# # 三、拼接这一日期段的全部情感记录
# sentiment = pd.concat(sentiment_parts, ignore_index=True)
# sentiment_parts.clear()
# del part

# print("\n拼接后情感记录数：", len(sentiment))

# # 四、规范新闻编号和自然日期
# sentiment["news_id"] = sentiment["news_id"].str.strip()

# sentiment["date"] = (
#     pd.to_datetime(sentiment["pub_date"], errors="raise")
#     .dt.normalize()
# )

# invalid_key = (
#     sentiment["news_id"].isna()
#     | sentiment["news_id"].eq("")
#     | sentiment["date"].isna()
# )

# if invalid_key.any():
#     raise ValueError("情感表存在缺失的新闻编号或日期，请先核查。")

# # 五、规范情感得分；无法识别的分数保留为缺失
# score_col = "overall_senti_score_adjust"

# sentiment[score_col] = (
#     pd.to_numeric(sentiment[score_col], errors="coerce")
#     .replace([float("inf"), -float("inf")], float("nan"))
# )

# # 六、检查同一篇新闻是否存在冲突分数
# # 先删除“新闻编号、日期、分数都相同”的重复记录。
# sentiment = sentiment[
#     ["news_id", "date", score_col]
# ].drop_duplicates()

# conflict = sentiment.duplicated(
#     ["news_id", "date"],
#     keep=False,
# )

# if conflict.any():
#     print("\n存在冲突的记录示例：")
#     print(sentiment.loc[conflict].head(10).to_string(index=False))
#     raise ValueError("同一新闻、同一自然日存在不同情感分数。")

# print("去重后新闻记录数：", len(sentiment))
# print("情感得分缺失数：", sentiment[score_col].isna().sum())

# print("\n清洗后的情感表前5行：")
# print(sentiment.head().to_string(index=False))



# # 一、读取2009—2013年的全部公司关系分片
# relation_parts = []

# with ZipFile(relation_zip) as archive:
#     relation_files = sorted(
#         name
#         for name in archive.namelist()
#         if Path(name).name.startswith("网络新闻相关上市公司表_")
#         and name.lower().endswith(".xlsx")
#     )

#     if not relation_files:
#         raise ValueError("没有找到公司关系数据工作簿。")

#     for file_name in relation_files:
#         print("\n正在读取：", file_name, flush=True)

#         part = pd.read_excel(
#             io.BytesIO(archive.read(file_name)),
#             usecols=[
#                 "news_id",             # 新闻编号
#                 "pub_date",            # 发布时间
#                 "stkcd",               # 股票代码
#                 "is_ment_in_title",     # 标题提及标记
#                 "is_ment_in_content",   # 正文提及标记
#             ],
#             dtype={
#                 "news_id": "string",
#                 "stkcd": "string",
#             },
#         )

#         print("本分片行数：", len(part), flush=True)
#         relation_parts.append(part)

# relations = pd.concat(relation_parts, ignore_index=True)
# relation_parts.clear()
# del part

# print("\n拼接后公司关系行数：", len(relations))

# # 二、规范新闻编号、股票代码和自然日期
# relations["news_id"] = relations["news_id"].str.strip()
# relations["stkcd"] = relations["stkcd"].str.strip().str.zfill(6)

# relations["date"] = (
#     pd.to_datetime(relations["pub_date"], errors="raise")
#     .dt.normalize()
# )

# invalid_key = (
#     relations["news_id"].isna()
#     | relations["news_id"].eq("")
#     | relations["stkcd"].isna()
#     | ~relations["stkcd"].str.fullmatch(r"\d{6}", na=False)
#     | relations["date"].isna()
# )

# if invalid_key.any():
#     raise ValueError("公司关系表存在无效的新闻编号、股票代码或日期。")

# # 三、检查标题、正文提及标记
# flag_cols = ["is_ment_in_title", "is_ment_in_content"]

# for col in flag_cols:
#     relations[col] = pd.to_numeric(relations[col], errors="raise")

#     if not relations[col].isin([0, 1]).all():
#         raise ValueError(f"{col} 存在缺失或非0、1的取值。")

# # 四、跨全部分片去重
# # 同一篇新闻对同一家公司只保留一条关系；
# # 提及标记取最大值，保留任一重复记录中出现过的1。
# before = len(relations)

# relations = (
#     relations
#     .groupby(["news_id", "date", "stkcd"], as_index=False)[flag_cols]
#     .max()
# )

# print("去重后公司关系行数：", len(relations))
# print("合并的重复关系行数：", before - len(relations))

# # 五、匹配已经清洗好的完整情感表
# news = relations.merge(
#     sentiment,
#     on=["news_id", "date"],
#     how="left",                # 保留全部公司关系
#     validate="many_to_one",    # 检查每条关系最多匹配一条情感
#     indicator=True,
# )

# print("\n完整数据的情感匹配情况：")
# print(news["_merge"].value_counts())

# print("\n匹配后明细前5行：")
# print(news.head().to_string(index=False))


# # 一、生成每条新闻—公司关系的计数标记
# score = news["overall_senti_score_adjust"]

# news["Posnews_All"] = score.gt(0).astype(int)       # 正面新闻
# news["Neunews_All"] = score.eq(0).astype(int)       # 中性新闻
# news["Negnews_All"] = score.lt(0).astype(int)       # 负面新闻
# news["Unclassifiednews_All"] = score.isna().astype(int)  # 未分类新闻

# news["Allnews_All"] = 1  # 每条唯一关系对应一篇相关新闻

# news["Newsnum_Title"] = news["is_ment_in_title"]
# news["Newsnum_Cont"] = news["is_ment_in_content"]

# # 二、聚合到公司×自然日
# count_cols = [
#     "Newsnum_Title",          # 标题提及篇数
#     "Newsnum_Cont",           # 正文提及篇数
#     "Posnews_All",            # 正面篇数
#     "Neunews_All",            # 中性篇数
#     "Negnews_All",            # 负面篇数
#     "Unclassifiednews_All",   # 未分类篇数
#     "Allnews_All",            # 全部相关新闻篇数
# ]

# daily_news = (
#     news
#     .groupby(["stkcd", "date"], as_index=False)[count_cols]
#     .sum()
#     .sort_values(["stkcd", "date"])
#     .reset_index(drop=True)
# )

# # 三、检查四类新闻是否加总为全部新闻
# sentiment_cols = [
#     "Posnews_All",
#     "Neunews_All",
#     "Negnews_All",
#     "Unclassifiednews_All",
# ]

# if not daily_news[sentiment_cols].sum(axis=1).eq(
#     daily_news["Allnews_All"]
# ).all():
#     raise ValueError("情感分类篇数与全部新闻篇数不一致。")

# # 四、检查聚合前后的新闻—公司关系总数是否守恒
# if daily_news["Allnews_All"].sum() != len(news):
#     raise ValueError("聚合前后的新闻—公司关系总数不一致。")

# # 五、查看聚合结果
# print("\n2009—2013年公司日聚合结果：")
# print("聚合后公司日行数：", len(daily_news))
# print("涉及公司数：", daily_news["stkcd"].nunique())
# print("最早日期：", daily_news["date"].min())
# print("最晚日期：", daily_news["date"].max())

# print(
#     "新闻—公司关系总数：",
#     daily_news["Allnews_All"].sum(),
# )
# print(
#     "其中未分类关系数：",
#     daily_news["Unclassifiednews_All"].sum(),
# )

# print("\n聚合结果前10行：")
# print(daily_news.head(10).to_string(index=False))



# # 一、统一股票代码和日期字段名称
# daily_news = daily_news.rename(
#     columns={
#         "stkcd": "Scode",  # 股票代码
#         "date": "Date",    # 自然日期
#     }
# )

# # 公司简称没有从原始表读取，暂留空；不影响新闻计数计算。
# daily_news["Coname"] = ""

# # 日期所属年份
# daily_news["year"] = daily_news["Date"].dt.year

# # 二、整理最终列顺序
# daily_news = daily_news[
#     [
#         "Scode",                  # 股票代码
#         "Coname",                 # 公司简称，暂为空
#         "Date",                   # 自然日期
#         "Newsnum_Title",           # 标题提及篇数
#         "Newsnum_Cont",            # 正文提及篇数
#         "Posnews_All",             # 正面篇数
#         "Neunews_All",             # 中性篇数
#         "Negnews_All",             # 负面篇数
#         "Unclassifiednews_All",    # 未分类篇数
#         "Allnews_All",             # 全部相关新闻篇数
#         "year",                   # 年份
#     ]
# ].copy()

# # 三、设置输出路径
# project_dir = data_dir.parent.parent

# output_dir = project_dir / "中间结果/00_合并原始数据"
# output_dir.mkdir(parents=True, exist_ok=True)

# output_path = (
#     output_dir / "DATAGO_网络新闻量化统计_2009_2013.csv"
# )

# # 四、保存公司日新闻计数表
# daily_news.to_csv(
#     output_path,
#     index=False,
#     encoding="utf-8-sig",  # 便于表格软件识别中文
#     date_format="%Y-%m-%d",
# )

# print("\n已保存：", output_path)
# print("输出行列数：", daily_news.shape)


# from pathlib import Path  # 处理路径
# from zipfile import ZipFile  # 读取压缩包
# import io  # 在内存中读取工作簿
# import sqlite3  # 标准库中的轻量级数据库模块
# import pandas as pd  # 处理表格

# # 一、项目路径
# project_dir = Path(
#     "/Users/qiyuewu/Desktop/mmm复现数据包"
# )

# data_dir = project_dir / "input/DATAGO"

# # 二、2014—2018年的两个输入压缩包
# relation_zip = (
#     data_dir
#     / "网络新闻相关上市公司表（2009-01-01~2025-12-31）"
#     / "网络新闻相关上市公司表（2014-01-01~2018-12-31）.zip"
# )

# sentiment_zip = (
#     data_dir
#     / "网络新闻情感量化结果数据表（2009-01-01~2025-12-31）"
#     / "网络新闻情感量化结果数据表（2014-01-01~2018-12-31）.zip"
# )

# # 三、临时数据库放在项目工作目录中
# work_dir = project_dir / "work"
# work_dir.mkdir(parents=True, exist_ok=True)

# database_path = work_dir / "网络新闻_2014_2018_处理中.sqlite"

# connection = sqlite3.connect(database_path)

# # 四、建立两张明细表
# connection.executescript("""
# CREATE TABLE IF NOT EXISTS sentiment (
#     news_id TEXT NOT NULL,
#     date TEXT NOT NULL,
#     score REAL,
#     PRIMARY KEY (news_id, date)
# );

# CREATE TABLE IF NOT EXISTS relations (
#     news_id TEXT NOT NULL,
#     date TEXT NOT NULL,
#     stkcd TEXT NOT NULL,
#     is_ment_in_title INTEGER NOT NULL,
#     is_ment_in_content INTEGER NOT NULL,
#     PRIMARY KEY (news_id, date, stkcd)
# );
# """)

# connection.commit()

# print("临时数据库已就绪：", database_path)




# # 一、逐个读取2014—2018年的情感工作簿
# score_col = "overall_senti_score_adjust"

# with ZipFile(sentiment_zip) as archive:
#     sentiment_files = sorted(
#         name
#         for name in archive.namelist()
#         if Path(name).name.startswith("网络新闻情感量化结果数据表_")
#         and name.lower().endswith(".xlsx")
#     )

#     if not sentiment_files:
#         raise ValueError("没有找到情感数据工作簿。")

#     print("情感数据工作簿数量：", len(sentiment_files))

#     for file_name in sentiment_files:
#         print("\n正在读取：", file_name, flush=True)

#         part = pd.read_excel(
#             io.BytesIO(archive.read(file_name)),
#             usecols=["news_id", "pub_date", score_col],
#             dtype={"news_id": "string"},
#         )

#         print("本分片原始行数：", len(part), flush=True)

#         # 二、规范新闻编号和自然日期
#         part["news_id"] = part["news_id"].str.strip()

#         part["date"] = (
#             pd.to_datetime(part["pub_date"], errors="raise")
#             .dt.normalize()
#         )

#         invalid_key = (
#             part["news_id"].isna()
#             | part["news_id"].eq("")
#             | part["date"].isna()
#         )

#         if invalid_key.any():
#             raise ValueError("存在缺失的新闻编号或日期。")

#         if not part["date"].between(
#             pd.Timestamp("2014-01-01"),
#             pd.Timestamp("2018-12-31"),
#         ).all():
#             raise ValueError("出现超出2014—2018年范围的发布日期。")

#         # 三、规范得分，缺失得分继续保留
#         part["score"] = (
#             pd.to_numeric(part[score_col], errors="coerce")
#             .replace(
#                 [float("inf"), -float("inf")],
#                 float("nan"),
#             )
#         )

#         part["date"] = part["date"].dt.strftime("%Y-%m-%d")
#         part = part[["news_id", "date", "score"]].drop_duplicates()

#         # 四、检查当前工作簿内部的分数冲突
#         conflict = part.duplicated(
#             ["news_id", "date"],
#             keep=False,
#         )

#         if conflict.any():
#             print(part.loc[conflict].head(10).to_string(index=False))
#             raise ValueError("当前工作簿中，同一新闻、同一日期存在冲突分数。")

#         # 五、当前工作簿先写入本批暂存表
#         part.to_sql(
#             "sentiment_batch",
#             connection,
#             if_exists="replace",
#             index=False,
#             chunksize=50_000,
#         )

#         # 六、与数据库中已有的情感记录比较
#         # “IS NOT”可以同时比较数值和缺失值。
#         conflicts = connection.execute("""
#             SELECT
#                 incoming.news_id,
#                 incoming.date,
#                 existing.score,
#                 incoming.score
#             FROM sentiment_batch AS incoming
#             JOIN sentiment AS existing
#                 ON incoming.news_id = existing.news_id
#                AND incoming.date = existing.date
#             WHERE incoming.score IS NOT existing.score
#             LIMIT 5
#         """).fetchall()

#         if conflicts:
#             print("冲突示例：新闻编号、日期、已有分数、本批分数")
#             for row in conflicts:
#                 print(row)
#             raise ValueError("不同工作簿之间存在冲突的情感分数。")

#         # 七、通过冲突检查后，插入尚不存在的记录
#         cursor = connection.execute("""
#             INSERT OR IGNORE INTO sentiment (news_id, date, score)
#             SELECT news_id, date, score
#             FROM sentiment_batch
#         """)

#         connection.commit()
#         print("本分片新增记录数：", cursor.rowcount, flush=True)

#         del part

# # 八、清理本批暂存表，保留正式情感表
# connection.execute("DROP TABLE IF EXISTS sentiment_batch")
# connection.commit()

# # 九、查看这一日期段的导入结果
# total, missing = connection.execute("""
#     SELECT COUNT(*), SUM(score IS NULL)
#     FROM sentiment
# """).fetchone()

# if total == 0:
#     raise ValueError("导入后情感表为空。")

# print("\n2014—2018年情感表导入完成")
# print("数据库中的新闻日记录数：", total)
# print("其中情感得分缺失数：", missing)



# # 一、定义公司关系的入库规则
# # 相同“新闻编号＋日期＋股票代码”只保留一条。
# # 标题、正文标记取最大值，保留重复记录中出现过的1。

# insert_relation_sql = """
# INSERT INTO relations (
#     news_id,
#     date,
#     stkcd,
#     is_ment_in_title,
#     is_ment_in_content
# )
# VALUES (?, ?, ?, ?, ?)
# ON CONFLICT (news_id, date, stkcd)
# DO UPDATE SET
#     is_ment_in_title = MAX(
#         relations.is_ment_in_title,
#         excluded.is_ment_in_title
#     ),
#     is_ment_in_content = MAX(
#         relations.is_ment_in_content,
#         excluded.is_ment_in_content
#     )
# """

# # 二、逐个读取公司关系工作簿
# with ZipFile(relation_zip) as archive:
#     relation_files = sorted(
#         name
#         for name in archive.namelist()
#         if Path(name).name.startswith("网络新闻相关上市公司表_")
#         and name.lower().endswith(".xlsx")
#     )

#     if not relation_files:
#         raise ValueError("没有找到公司关系数据工作簿。")

#     print("公司关系数据工作簿数量：", len(relation_files))

#     for file_name in relation_files:
#         print("\n正在读取：", file_name, flush=True)

#         part = pd.read_excel(
#             io.BytesIO(archive.read(file_name)),
#             usecols=[
#                 "news_id",
#                 "pub_date",
#                 "stkcd",
#                 "is_ment_in_title",
#                 "is_ment_in_content",
#             ],
#             dtype={
#                 "news_id": "string",
#                 "stkcd": "string",
#             },
#         )

#         print("本分片原始行数：", len(part), flush=True)

#         # 三、规范新闻编号、股票代码和日期
#         part["news_id"] = part["news_id"].str.strip()
#         part["stkcd"] = part["stkcd"].str.strip().str.zfill(6)

#         part["date"] = (
#             pd.to_datetime(part["pub_date"], errors="raise")
#             .dt.normalize()
#         )

#         invalid_key = (
#             part["news_id"].isna()
#             | part["news_id"].eq("")
#             | ~part["stkcd"].str.fullmatch(r"\d{6}", na=False)
#             | part["date"].isna()
#         )

#         if invalid_key.any():
#             raise ValueError("存在无效的新闻编号、股票代码或日期。")

#         if not part["date"].between(
#             pd.Timestamp("2014-01-01"),
#             pd.Timestamp("2018-12-31"),
#         ).all():
#             raise ValueError("出现超出2014—2018年范围的发布日期。")

#         # 四、检查两个提及标记
#         flag_cols = ["is_ment_in_title", "is_ment_in_content"]

#         for col in flag_cols:
#             part[col] = pd.to_numeric(part[col], errors="raise")

#             if not part[col].isin([0, 1]).all():
#                 raise ValueError(f"{col} 存在缺失或非0、1的取值。")

#         part["date"] = part["date"].dt.strftime("%Y-%m-%d")

#         # 五、先合并当前工作簿内部的重复关系
#         part = (
#             part
#             .groupby(
#                 ["news_id", "date", "stkcd"],
#                 as_index=False,
#             )[flag_cols]
#             .max()
#         )

#         # 六、逐条传给数据库
#         # 显式转为整数，保证两个提及标记按整数存储。
#         records = (
#             (
#                 news_id,
#                 date,
#                 stkcd,
#                 int(title_flag),
#                 int(content_flag),
#             )
#             for news_id, date, stkcd, title_flag, content_flag
#             in part[
#                 [
#                     "news_id",
#                     "date",
#                     "stkcd",
#                     "is_ment_in_title",
#                     "is_ment_in_content",
#                 ]
#             ].itertuples(index=False, name=None)
#         )

#         # 一个工作簿成功后提交；出错则撤销本工作簿的写入。
#         with connection:
#             connection.executemany(insert_relation_sql, records)

#         print("本分片处理完成。", flush=True)
#         del part

# # 七、检查完整公司关系表与情感表的匹配情况
# total, matched, unmatched = connection.execute("""
#     SELECT
#         COUNT(*),
#         COALESCE(SUM(s.news_id IS NOT NULL), 0),
#         COALESCE(SUM(s.news_id IS NULL), 0)
#     FROM relations AS r
#     LEFT JOIN sentiment AS s
#         ON r.news_id = s.news_id
#        AND r.date = s.date
# """).fetchone()

# print("\n2014—2018年公司关系表导入完成")
# print("去重后新闻—公司关系数：", total)
# print("匹配到情感记录的关系数：", matched)
# print("未匹配到情感记录的关系数：", unmatched)



# # 一、合并情感，并按“公司＋自然日”聚合
# # r代表公司关系表，s代表情感表。

# daily_sql = """
# SELECT
#     r.stkcd AS Scode,
#     '' AS Coname,
#     r.date AS Date,

#     SUM(r.is_ment_in_title) AS Newsnum_Title,
#     SUM(r.is_ment_in_content) AS Newsnum_Cont,

#     SUM(CASE WHEN s.score > 0 THEN 1 ELSE 0 END) AS Posnews_All,
#     SUM(CASE WHEN s.score = 0 THEN 1 ELSE 0 END) AS Neunews_All,
#     SUM(CASE WHEN s.score < 0 THEN 1 ELSE 0 END) AS Negnews_All,
#     SUM(CASE WHEN s.score IS NULL THEN 1 ELSE 0 END)
#         AS Unclassifiednews_All,

#     COUNT(*) AS Allnews_All,
#     CAST(SUBSTR(r.date, 1, 4) AS INTEGER) AS year

# FROM relations AS r
# LEFT JOIN sentiment AS s
#     ON r.news_id = s.news_id
#    AND r.date = s.date

# GROUP BY r.stkcd, r.date
# ORDER BY r.stkcd, r.date
# """

# # 二、取得聚合前的关系总数，用于核对
# expected_relations = connection.execute(
#     "SELECT COUNT(*) FROM relations"
# ).fetchone()[0]

# if expected_relations == 0:
#     raise ValueError("公司关系表为空，请先完成导入。")

# # 三、设置输出路径
# output_dir = project_dir / "中间结果/00_合并原始数据"
# output_dir.mkdir(parents=True, exist_ok=True)

# output_path = (
#     output_dir / "DATAGO_网络新闻量化统计_2014_2018.csv"
# )

# # 先写到工作目录，检查通过后再移到正式位置。
# pending_path = (
#     work_dir / "DATAGO_网络新闻量化统计_2014_2018_待完成.csv"
# )

# # 四、分批读取聚合结果并写出
# total_rows = 0
# total_relations = 0
# total_unclassified = 0
# first = True

# sentiment_cols = [
#     "Posnews_All",
#     "Neunews_All",
#     "Negnews_All",
#     "Unclassifiednews_All",
# ]

# with pending_path.open(
#     "w",
#     encoding="utf-8-sig",
#     newline="",
# ) as output_file:

#     for chunk in pd.read_sql_query(
#         daily_sql,
#         connection,
#         chunksize=100_000,
#     ):
#         if chunk.empty:
#             continue

#         # 检查四类新闻是否加总为全部新闻。
#         if not chunk[sentiment_cols].sum(axis=1).eq(
#             chunk["Allnews_All"]
#         ).all():
#             raise ValueError("情感分类篇数与全部新闻篇数不一致。")

#         chunk.to_csv(
#             output_file,
#             index=False,
#             header=first,
#         )

#         first = False
#         total_rows += len(chunk)
#         total_relations += int(chunk["Allnews_All"].sum())
#         total_unclassified += int(
#             chunk["Unclassifiednews_All"].sum()
#         )

#         print("已写出公司日行数：", total_rows, flush=True)

# # 五、检查聚合前后计数是否一致
# if total_rows == 0:
#     raise ValueError("聚合结果为空。")

# if total_relations != expected_relations:
#     raise ValueError("聚合前后的新闻—公司关系总数不一致。")

# # 六、检查通过后，保存为正式结果
# pending_path.replace(output_path)

# print("\n2014—2018年公司日聚合完成")
# print("输出行列数：", (total_rows, 11))
# print("新闻—公司关系总数：", total_relations)
# print("其中未分类关系数：", total_unclassified)
# print("已保存：", output_path)



# from pathlib import Path
# import sqlite3

# # 一、项目目录
# project_dir = Path("/Users/qiyuewu/Desktop/mmm复现数据包")
# data_dir = project_dir / "input/DATAGO"
# work_dir = project_dir / "work"

# work_dir.mkdir(exist_ok=True)

# # 二、本次处理的两个压缩包
# relation_zip = (
#     data_dir
#     / "网络新闻相关上市公司表（2009-01-01~2025-12-31）"
#     / "网络新闻相关上市公司表（2019-01-01~2023-12-31）.zip"
# )

# sentiment_zip = (
#     data_dir
#     / "网络新闻情感量化结果数据表（2009-01-01~2025-12-31）"
#     / "网络新闻情感量化结果数据表（2019-01-01~2023-12-31）.zip"
# )

# # 检查压缩包是否存在
# for zip_path in [relation_zip, sentiment_zip]:
#     if not zip_path.is_file():
#         raise FileNotFoundError(f"找不到压缩包：{zip_path}")

# # 三、连接本日期段专用的临时数据库
# database_path = work_dir / "网络新闻_2019_2023_处理中.sqlite"
# connection = sqlite3.connect(database_path)

# # 四、建立情感表与公司关系表
# connection.executescript("""
# CREATE TABLE IF NOT EXISTS sentiment (
#     news_id TEXT NOT NULL,
#     date TEXT NOT NULL,
#     score REAL,
#     PRIMARY KEY (news_id, date)
# );

# CREATE TABLE IF NOT EXISTS relations (
#     news_id TEXT NOT NULL,
#     date TEXT NOT NULL,
#     stkcd TEXT NOT NULL,
#     is_ment_in_title INTEGER NOT NULL,
#     is_ment_in_content INTEGER NOT NULL,
#     PRIMARY KEY (news_id, date, stkcd)
# );
# """)

# connection.commit()

# print("临时数据库已就绪：", database_path)


# import io
# import pandas as pd
# from zipfile import ZipFile

# # 一、逐个读取情感数据工作簿
# with ZipFile(sentiment_zip) as zip_file:

#     # 只选择数据工作簿，排除字段说明等文件
#     sentiment_files = sorted(
#         name
#         for name in zip_file.namelist()
#         if Path(name).name.startswith("网络新闻情感量化结果数据表_")
#         and name.endswith(".xlsx")
#         and not name.startswith("__MACOSX/")
#     )

#     if not sentiment_files:
#         raise ValueError("压缩包中没有找到情感数据工作簿")

#     print("情感数据工作簿数：", len(sentiment_files))

#     for number, file_name in enumerate(sentiment_files, start=1):

#         print(f"\n正在读取第 {number}/{len(sentiment_files)} 个工作簿：")
#         print(Path(file_name).name)

#         sentiment_df = pd.read_excel(
#             io.BytesIO(zip_file.read(file_name)),
#             usecols=[
#                 "news_id",
#                 "pub_date",
#                 "overall_senti_score_adjust",
#             ],
#             dtype={"news_id": "string"},
#             engine="openpyxl",
#         )

#         print("本分片行数：", len(sentiment_df))

#         # 二、清洗新闻编号、自然日和情感分数
#         sentiment_df["news_id"] = (
#             sentiment_df["news_id"].str.strip()
#         )

#         if (
#             sentiment_df["news_id"].isna().any()
#             or sentiment_df["news_id"].eq("").any()
#         ):
#             raise ValueError("存在缺失或空白的新闻编号")

#         sentiment_df["date"] = pd.to_datetime(
#             sentiment_df["pub_date"],
#             errors="coerce",
#         ).dt.strftime("%Y-%m-%d")

#         if sentiment_df["date"].isna().any():
#             raise ValueError("存在无法识别的发布日期")

#         if not sentiment_df["date"].between(
#             "2019-01-01", "2023-12-31"
#         ).all():
#             raise ValueError("存在超出2019—2023年范围的日期")

#         sentiment_df["score"] = pd.to_numeric(
#             sentiment_df["overall_senti_score_adjust"],
#             errors="raise",
#         )

#         sentiment_df = sentiment_df[
#             ["news_id", "date", "score"]
#         ]

#         # 三、把当前工作簿放进暂存表
#         sentiment_df.to_sql(
#             "sentiment_batch",
#             connection,
#             if_exists="replace",
#             index=False,
#             chunksize=50_000,
#         )

#         # 四、检查当前工作簿内部是否存在分数冲突
#         # 同一新闻、同一天：不同有效分数，或缺失与有效值并存，均报错
#         conflict = connection.execute("""
#             SELECT news_id, date
#             FROM sentiment_batch
#             GROUP BY news_id, date
#             HAVING COUNT(DISTINCT score) > 1
#                 OR (
#                     COUNT(score) > 0
#                     AND COUNT(score) < COUNT(*)
#                 )
#             LIMIT 1
#         """).fetchone()

#         if conflict is not None:
#             raise ValueError(
#                 f"当前工作簿内存在情感分数冲突：{conflict}"
#             )

#         # 五、检查是否与前面已导入工作簿的分数冲突
#         conflict = connection.execute("""
#             SELECT
#                 b.news_id,
#                 b.date,
#                 b.score,
#                 s.score
#             FROM sentiment_batch AS b
#             INNER JOIN sentiment AS s
#                 ON b.news_id = s.news_id
#                 AND b.date = s.date
#             WHERE b.score IS NOT s.score
#             LIMIT 1
#         """).fetchone()

#         if conflict is not None:
#             raise ValueError(
#                 f"当前工作簿与已导入数据存在情感分数冲突：{conflict}"
#             )

#         # 六、写入正式情感表；相同新闻日的重复记录只保留一条
#         connection.execute("""
#             INSERT OR IGNORE INTO sentiment (
#                 news_id, date, score
#             )
#             SELECT news_id, date, score
#             FROM sentiment_batch
#         """)

#         # 当前工作簿处理完毕，删除暂存表
#         connection.execute("DROP TABLE sentiment_batch")
#         connection.commit()

#         print("本分片已核对并导入")

#         del sentiment_df

# # 七、汇总数据库中的情感记录
# record_count, missing_count = connection.execute("""
#     SELECT
#         COUNT(*),
#         COALESCE(SUM(score IS NULL), 0)
#     FROM sentiment
# """).fetchone()

# print("\n2019—2023年情感表导入完成")
# print("数据库中的新闻日记录数：", record_count)
# print("其中情感得分缺失数：", missing_count)


# import io
# import pandas as pd
# from zipfile import ZipFile

# # 一、逐个读取公司关系数据工作簿
# with ZipFile(relation_zip) as zip_file:

#     relation_files = sorted(
#         name
#         for name in zip_file.namelist()
#         if Path(name).name.startswith(
#             "网络新闻相关上市公司表_2019-01-01~2023-12-31_"
#         )
#         and name.endswith(".xlsx")
#         and not name.startswith("__MACOSX/")
#     )

#     if len(relation_files) != 40:
#         raise ValueError(
#             f"预期有40个公司关系数据工作簿，实际找到{len(relation_files)}个"
#         )

#     print("公司关系数据工作簿数：", len(relation_files))

#     for number, file_name in enumerate(relation_files, start=1):

#         print(f"\n正在读取第 {number}/{len(relation_files)} 个工作簿：")
#         print(Path(file_name).name)

#         relation_df = pd.read_excel(
#             io.BytesIO(zip_file.read(file_name)),
#             usecols=[
#                 "news_id",
#                 "pub_date",
#                 "stkcd",
#                 "is_ment_in_title",
#                 "is_ment_in_content",
#             ],
#             dtype={
#                 "news_id": "string",
#                 "stkcd": "string",
#             },
#             engine="openpyxl",
#         )

#         print("本分片原始行数：", len(relation_df))

#         # 二、检查新闻编号和股票代码
#         for column in ["news_id", "stkcd"]:
#             relation_df[column] = relation_df[column].str.strip()

#             if (
#                 relation_df[column].isna().any()
#                 or relation_df[column].eq("").any()
#             ):
#                 raise ValueError(f"{column}列存在缺失或空白")

#         # 股票代码只允许1—6位数字，再补足六位
#         if not relation_df["stkcd"].str.fullmatch(r"[0-9]{1,6}").all():
#             raise ValueError("存在无法识别的股票代码")

#         relation_df["stkcd"] = relation_df["stkcd"].str.zfill(6)

#         # 三、提取自然日并检查日期范围
#         relation_df["date"] = pd.to_datetime(
#             relation_df["pub_date"],
#             errors="coerce",
#         ).dt.strftime("%Y-%m-%d")

#         if relation_df["date"].isna().any():
#             raise ValueError("存在无法识别的发布日期")

#         if not relation_df["date"].between(
#             "2019-01-01", "2023-12-31"
#         ).all():
#             raise ValueError("存在超出2019—2023年范围的日期")

#         # 四、检查标题、正文提及标记，只允许0或1
#         mention_columns = [
#             "is_ment_in_title",
#             "is_ment_in_content",
#         ]

#         for column in mention_columns:
#             relation_df[column] = pd.to_numeric(
#                 relation_df[column],
#                 errors="raise",
#             )

#             if not relation_df[column].isin([0, 1]).all():
#                 raise ValueError(f"{column}列存在缺失或非0/1取值")

#             relation_df[column] = relation_df[column].astype("int64")

#         # 五、合并当前工作簿内部的重复新闻—公司关系
#         # 同一新闻、同一天、同一公司只保留一条，提及标记取最大值
#         relation_df = (
#             relation_df.groupby(
#                 ["news_id", "date", "stkcd"],
#                 as_index=False,
#                 sort=False,
#             )[mention_columns]
#             .max()
#         )

#         # 六、导入数据库
#         # 如果前面的工作簿已有相同关系，仍将两个提及标记取最大值
#         with connection:
#             connection.executemany("""
#                 INSERT INTO relations (
#                     news_id,
#                     date,
#                     stkcd,
#                     is_ment_in_title,
#                     is_ment_in_content
#                 )
#                 VALUES (?, ?, ?, ?, ?)

#                 ON CONFLICT (news_id, date, stkcd)
#                 DO UPDATE SET
#                     is_ment_in_title = MAX(
#                         relations.is_ment_in_title,
#                         excluded.is_ment_in_title
#                     ),
#                     is_ment_in_content = MAX(
#                         relations.is_ment_in_content,
#                         excluded.is_ment_in_content
#                     )
#             """, relation_df.itertuples(index=False, name=None))

#         print("本分片处理完成")

#         del relation_df

# # 七、检查公司关系能否匹配到情感记录
# relation_count, matched_count, unmatched_count = connection.execute("""
#     SELECT
#         COUNT(*),
#         COUNT(s.news_id),
#         COUNT(*) - COUNT(s.news_id)
#     FROM relations AS r
#     LEFT JOIN sentiment AS s
#         ON r.news_id = s.news_id
#         AND r.date = s.date
# """).fetchone()

# print("\n2019—2023年公司关系表导入完成")
# print("去重后新闻—公司关系数：", relation_count)
# print("匹配到情感记录的关系数：", matched_count)
# print("未匹配到情感记录的关系数：", unmatched_count)



# from pathlib import Path
# import sqlite3
# import pandas as pd

# # 一、连接2019—2023年的数据库
# project_dir = Path("/Users/qiyuewu/Desktop/mmm复现数据包")

# database_path = (
#     project_dir / "work/网络新闻_2019_2023_处理中.sqlite"
# )

# if not database_path.is_file():
#     raise FileNotFoundError(f"找不到数据库：{database_path}")

# connection = sqlite3.connect(database_path)

# expected_relations = connection.execute(
#     "SELECT COUNT(*) FROM relations"
# ).fetchone()[0]

# if expected_relations != 39_260_745:
#     raise ValueError(
#         f"数据库关系数应为39260745，实际为{expected_relations}"
#     )

# # 二、设置输出位置
# output_dir = project_dir / "中间结果/00_合并原始数据"
# output_dir.mkdir(parents=True, exist_ok=True)

# output_path = (
#     output_dir / "DATAGO_网络新闻量化统计_2019_2023.csv"
# )

# # 先写到待完成文件，核对通过后再移到正式位置
# pending_path = (
#     project_dir
#     / "work/DATAGO_网络新闻量化统计_2019_2023_待完成.csv"
# )

# # 三、合并情感，并按公司和自然日聚合
# daily_sql = """
# SELECT
#     r.stkcd AS Scode,
#     '' AS Coname,
#     r.date AS Date,

#     SUM(r.is_ment_in_title) AS Newsnum_Title,
#     SUM(r.is_ment_in_content) AS Newsnum_Cont,

#     SUM(CASE WHEN s.score > 0 THEN 1 ELSE 0 END)
#         AS Posnews_All,

#     SUM(CASE WHEN s.score = 0 THEN 1 ELSE 0 END)
#         AS Neunews_All,

#     SUM(CASE WHEN s.score < 0 THEN 1 ELSE 0 END)
#         AS Negnews_All,

#     SUM(CASE WHEN s.score IS NULL THEN 1 ELSE 0 END)
#         AS Unclassifiednews_All,

#     COUNT(*) AS Allnews_All,

#     CAST(SUBSTR(r.date, 1, 4) AS INTEGER) AS year

# FROM relations AS r
# LEFT JOIN sentiment AS s
#     ON r.news_id = s.news_id
#     AND r.date = s.date

# GROUP BY r.stkcd, r.date
# ORDER BY r.stkcd, r.date
# """

# # 四、每次取出十万行公司日结果，逐批写出
# total_rows = 0
# total_relations = 0
# total_unclassified = 0

# first_chunk = True
# earliest_date = None
# latest_date = None

# sentiment_columns = [
#     "Posnews_All",
#     "Neunews_All",
#     "Negnews_All",
#     "Unclassifiednews_All",
# ]

# print("开始公司日聚合，第一批结果输出前需要等待数据库计算。")

# with pending_path.open(
#     "w", encoding="utf-8-sig", newline=""
# ) as output_file:

#     for chunk in pd.read_sql_query(
#         daily_sql,
#         connection,
#         chunksize=100_000,
#     ):

#         if chunk.empty:
#             continue

#         # 检查四类新闻加总是否等于全部新闻
#         category_total = chunk[sentiment_columns].sum(axis=1)

#         if not category_total.eq(chunk["Allnews_All"]).all():
#             raise ValueError("情感分类合计与全部新闻数量不一致")

#         # 检查日期是否处于本日期段内
#         if not chunk["Date"].between(
#             "2019-01-01", "2023-12-31"
#         ).all():
#             raise ValueError("聚合结果存在超出2019—2023年的日期")

#         chunk.to_csv(
#             output_file,
#             index=False,
#             header=first_chunk,
#         )

#         first_chunk = False

#         total_rows += len(chunk)
#         total_relations += int(chunk["Allnews_All"].sum())
#         total_unclassified += int(
#             chunk["Unclassifiednews_All"].sum()
#         )

#         chunk_start = chunk["Date"].min()
#         chunk_end = chunk["Date"].max()

#         if earliest_date is None or chunk_start < earliest_date:
#             earliest_date = chunk_start

#         if latest_date is None or chunk_end > latest_date:
#             latest_date = chunk_end

#         print("已写出公司日行数：", total_rows)

# connection.close()

# # 五、核对聚合前后的关系总数
# if total_rows == 0:
#     raise ValueError("聚合结果为空")

# if total_relations != expected_relations:
#     raise ValueError(
#         f"聚合前关系数为{expected_relations}，"
#         f"聚合后加总为{total_relations}，两者不一致"
#     )

# # 此前情感全部匹配且没有缺失，因此未分类应为零
# if total_unclassified != 0:
#     raise ValueError(
#         f"预期未分类关系数为0，实际为{total_unclassified}"
#     )

# # 六、核对通过，保存为正式文件
# pending_path.replace(output_path)

# print("\n2019—2023年公司日聚合完成")
# print("输出行列数：", (total_rows, 11))
# print("最早日期：", earliest_date)
# print("最晚日期：", latest_date)
# print("新闻—公司关系总数：", total_relations)
# print("其中未分类关系数：", total_unclassified)
# print("已保存：", output_path)


from pathlib import Path
import sqlite3

# 一、项目目录
project_dir = Path("/Users/qiyuewu/Desktop/mmm复现数据包")
data_dir = project_dir / "input/DATAGO"
work_dir = project_dir / "work"

work_dir.mkdir(exist_ok=True)

# 二、2024—2025年的两个压缩包
relation_zip = (
    data_dir
    / "网络新闻相关上市公司表（2009-01-01~2025-12-31）"
    / "网络新闻相关上市公司表（2024-01-01~2025-12-31）.zip"
)

sentiment_zip = (
    data_dir
    / "网络新闻情感量化结果数据表（2009-01-01~2025-12-31）"
    / "网络新闻情感量化结果数据表（2024-01-01~2025-12-31）.zip"
)

for zip_path in [relation_zip, sentiment_zip]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"找不到压缩包：{zip_path}")

# 三、连接本日期段专用的数据库
database_path = (
    work_dir / "网络新闻_2024_2025_处理中.sqlite"
)

connection = sqlite3.connect(database_path)

# 四、建立情感表和公司关系表
connection.executescript("""
CREATE TABLE IF NOT EXISTS sentiment (
    news_id TEXT NOT NULL,
    date TEXT NOT NULL,
    score REAL,
    PRIMARY KEY (news_id, date)
);

CREATE TABLE IF NOT EXISTS relations (
    news_id TEXT NOT NULL,
    date TEXT NOT NULL,
    stkcd TEXT NOT NULL,
    is_ment_in_title INTEGER NOT NULL,
    is_ment_in_content INTEGER NOT NULL,
    PRIMARY KEY (news_id, date, stkcd)
);
""")

connection.commit()

print("临时数据库已就绪：", database_path)


import io
import pandas as pd
from zipfile import ZipFile

# # 一、找出2024—2025年的情感数据工作簿
# with ZipFile(sentiment_zip) as zip_file:

#     sentiment_files = sorted(
#         name
#         for name in zip_file.namelist()
#         if Path(name).name.startswith(
#             "网络新闻情感量化结果数据表_2024-01-01~2025-12-31_"
#         )
#         and name.endswith(".xlsx")
#         and not name.startswith("__MACOSX/")
#     )

#     if len(sentiment_files) != 15:
#         raise ValueError(
#             f"预期有15个情感数据工作簿，实际找到{len(sentiment_files)}个"
#         )

#     print("情感数据工作簿数：", len(sentiment_files))

#     for number, file_name in enumerate(sentiment_files, start=1):

#         print(f"\n正在读取第 {number}/{len(sentiment_files)} 个工作簿：")
#         print(Path(file_name).name)

#         sentiment_df = pd.read_excel(
#             io.BytesIO(zip_file.read(file_name)),
#             usecols=[
#                 "news_id",
#                 "pub_date",
#                 "overall_senti_score_adjust",
#             ],
#             dtype={"news_id": "string"},
#             engine="openpyxl",
#         )

#         print("本分片行数：", len(sentiment_df))

#         # 二、清洗并检查新闻编号
#         sentiment_df["news_id"] = (
#             sentiment_df["news_id"].str.strip()
#         )

#         if (
#             sentiment_df["news_id"].isna().any()
#             or sentiment_df["news_id"].eq("").any()
#         ):
#             raise ValueError("存在缺失或空白的新闻编号")

#         # 三、提取自然日并检查日期范围
#         sentiment_df["date"] = pd.to_datetime(
#             sentiment_df["pub_date"],
#             errors="coerce",
#         ).dt.strftime("%Y-%m-%d")

#         if sentiment_df["date"].isna().any():
#             raise ValueError("存在无法识别的发布日期")

#         if not sentiment_df["date"].between(
#             "2024-01-01", "2025-12-31"
#         ).all():
#             raise ValueError("存在超出2024—2025年范围的日期")

#         # 四、将情感分数转为数值
#         sentiment_df["score"] = pd.to_numeric(
#             sentiment_df["overall_senti_score_adjust"],
#             errors="raise",
#         )

#         sentiment_df = sentiment_df[
#             ["news_id", "date", "score"]
#         ]

#         # 五、将当前工作簿存入暂存表，准备检查重复记录
#         sentiment_df.to_sql(
#             "sentiment_batch",
#             connection,
#             if_exists="replace",
#             index=False,
#             chunksize=50_000,
#         )

#         # 检查当前工作簿内部：
#         # 同一新闻、同一天是否有不同分数，
#         # 或同时存在缺失分数与有效分数
#         conflict = connection.execute("""
#             SELECT news_id, date
#             FROM sentiment_batch
#             GROUP BY news_id, date
#             HAVING COUNT(DISTINCT score) > 1
#                 OR (
#                     COUNT(score) > 0
#                     AND COUNT(score) < COUNT(*)
#                 )
#             LIMIT 1
#         """).fetchone()

#         if conflict is not None:
#             raise ValueError(
#                 f"当前工作簿内存在情感分数冲突：{conflict}"
#             )

#         # 六、检查是否与前面已经导入的工作簿存在分数冲突
#         conflict = connection.execute("""
#             SELECT
#                 b.news_id,
#                 b.date,
#                 b.score,
#                 s.score
#             FROM sentiment_batch AS b
#             INNER JOIN sentiment AS s
#                 ON b.news_id = s.news_id
#                 AND b.date = s.date
#             WHERE b.score IS NOT s.score
#             LIMIT 1
#         """).fetchone()

#         if conflict is not None:
#             raise ValueError(
#                 f"当前工作簿与已导入数据存在情感分数冲突：{conflict}"
#             )

#         # 七、写入正式情感表
#         # 已经存在且分数一致的新闻日记录不再重复写入
#         connection.execute("""
#             INSERT OR IGNORE INTO sentiment (
#                 news_id, date, score
#             )
#             SELECT news_id, date, score
#             FROM sentiment_batch
#         """)

#         connection.execute("DROP TABLE sentiment_batch")
#         connection.commit()

#         print("本分片已核对并导入")

#         del sentiment_df

# # 八、统计导入后的记录数及情感分数缺失数
# record_count, missing_count = connection.execute("""
#     SELECT
#         COUNT(*),
#         COALESCE(SUM(score IS NULL), 0)
#     FROM sentiment
# """).fetchone()

# print("\n2024—2025年情感表导入完成")
# print("数据库中的新闻日记录数：", record_count)
# print("其中情感得分缺失数：", missing_count)


import io
import pandas as pd
from zipfile import ZipFile

# 一、找出2024—2025年的公司关系数据工作簿
with ZipFile(relation_zip) as zip_file:

    relation_files = sorted(
        name
        for name in zip_file.namelist()
        if Path(name).name.startswith(
            "网络新闻相关上市公司表_2024-01-01~2025-12-31_"
        )
        and name.endswith(".xlsx")
        and not name.startswith("__MACOSX/")
    )

    if len(relation_files) != 33:
        raise ValueError(
            f"预期有33个公司关系数据工作簿，实际找到{len(relation_files)}个"
        )

    print("公司关系数据工作簿数：", len(relation_files))

    for number, file_name in enumerate(relation_files, start=1):

        print(f"\n正在读取第 {number}/{len(relation_files)} 个工作簿：")
        print(Path(file_name).name)

        relation_df = pd.read_excel(
            io.BytesIO(zip_file.read(file_name)),
            usecols=[
                "news_id",
                "pub_date",
                "stkcd",
                "is_ment_in_title",
                "is_ment_in_content",
            ],
            dtype={
                "news_id": "string",
                "stkcd": "string",
            },
            engine="openpyxl",
        )

        print("本分片原始行数：", len(relation_df))

        # 二、检查新闻编号和股票代码
        for column in ["news_id", "stkcd"]:
            relation_df[column] = relation_df[column].str.strip()

            if (
                relation_df[column].isna().any()
                or relation_df[column].eq("").any()
            ):
                raise ValueError(f"{column}列存在缺失或空白")

        # 股票代码只允许1—6位数字，再补足六位
        if not relation_df["stkcd"].str.fullmatch(r"[0-9]{1,6}").all():
            raise ValueError("存在无法识别的股票代码")

        relation_df["stkcd"] = relation_df["stkcd"].str.zfill(6)

        # 三、提取自然日并检查日期范围
        relation_df["date"] = pd.to_datetime(
            relation_df["pub_date"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

        if relation_df["date"].isna().any():
            raise ValueError("存在无法识别的发布日期")

        if not relation_df["date"].between(
            "2024-01-01", "2025-12-31"
        ).all():
            raise ValueError("存在超出2024—2025年范围的日期")

        # 四、检查标题、正文提及标记，只允许0或1
        mention_columns = [
            "is_ment_in_title",
            "is_ment_in_content",
        ]

        for column in mention_columns:
            relation_df[column] = pd.to_numeric(
                relation_df[column],
                errors="raise",
            )

            if not relation_df[column].isin([0, 1]).all():
                raise ValueError(f"{column}列存在缺失或非0/1取值")

            relation_df[column] = relation_df[column].astype("int64")

        # 五、合并当前工作簿内部的重复关系
        # 同一新闻、同一天、同一公司只保留一条
        # 标题和正文提及标记分别取最大值
        relation_df = (
            relation_df.groupby(
                ["news_id", "date", "stkcd"],
                as_index=False,
                sort=False,
            )[mention_columns]
            .max()
        )

        # 六、导入数据库
        # 与前面工作簿中的关系重复时，提及标记仍取最大值
        with connection:
            connection.executemany("""
                INSERT INTO relations (
                    news_id,
                    date,
                    stkcd,
                    is_ment_in_title,
                    is_ment_in_content
                )
                VALUES (?, ?, ?, ?, ?)

                ON CONFLICT (news_id, date, stkcd)
                DO UPDATE SET
                    is_ment_in_title = MAX(
                        relations.is_ment_in_title,
                        excluded.is_ment_in_title
                    ),
                    is_ment_in_content = MAX(
                        relations.is_ment_in_content,
                        excluded.is_ment_in_content
                    )
            """, relation_df.itertuples(index=False, name=None))

        print("本分片处理完成")

        del relation_df

# 七、检查公司关系能否匹配到情感记录
relation_count, matched_count, unmatched_count = connection.execute("""
    SELECT
        COUNT(*),
        COUNT(s.news_id),
        COUNT(*) - COUNT(s.news_id)
    FROM relations AS r
    LEFT JOIN sentiment AS s
        ON r.news_id = s.news_id
        AND r.date = s.date
""").fetchone()

print("\n2024—2025年公司关系表导入完成")
print("去重后新闻—公司关系数：", relation_count)
print("匹配到情感记录的关系数：", matched_count)
print("未匹配到情感记录的关系数：", unmatched_count)