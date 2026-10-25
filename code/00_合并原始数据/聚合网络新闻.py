"""将数行者网络新闻聚合为2009—2025年的公司×自然日计数表。

顺序：逐工作簿读取 → 情感去重 → 公司关系去重 → 匹配 → 公司日聚合 → 拼接。
发布时间只取自然日。关系按新闻编号＋自然日＋公司去重，两个提及标记取最大值。
整篇新闻的调整后分数大于0、等于0、小于0分别归为正面、中性、负面；
缺失或未匹配归为未分类。同一篇涉及多家公司时分别计数，标题与正文不能相加。
公司简称留空，只输出有新闻的公司日，不补零、不移日期、不构造滚动媒体指标。
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import io
import sqlite3

import pandas as pd


# 一、运行设置：
PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_DIR / "input/DATAGO"
OUTPUT_DIR = PROJECT_DIR / "中间结果/00_合并原始数据"
WORK_DIR = PROJECT_DIR / "work"

# True：从原始包重新构建四段；False：只核对、拼接已有四份分段表。
# 重新构建始终使用新临时数据库，不自动复用以前的数据库。
REBUILD_FROM_RAW = True

# True：同时核对本次已确认的数据包基准；原始下载包更新后可改为False。
# 此设置不影响唯一键、日期范围、分类合计和聚合前后守恒检查。
CHECK_BASELINE = True
CHUNK_SIZE = 100_000

PERIODS = [
    dict(period="2009_2013", start="2009-01-01", end="2013-12-31",
         sentiment_files=2, relation_files=3, rows=928_831, relations=2_683_553),
    dict(period="2014_2018", start="2014-01-01", end="2018-12-31",
         sentiment_files=4, relation_files=11, rows=2_295_770, relations=10_484_778),
    dict(period="2019_2023", start="2019-01-01", end="2023-12-31",
         sentiment_files=14, relation_files=40, rows=4_318_698, relations=39_260_745),
    dict(period="2024_2025", start="2024-01-01", end="2025-12-31",
         sentiment_files=15, relation_files=33, rows=2_856_066, relations=32_218_552),
]

COLUMNS = [
    "Scode", "Coname", "Date", "Newsnum_Title", "Newsnum_Cont",
    "Posnews_All", "Neunews_All", "Negnews_All", "Unclassifiednews_All",
    "Allnews_All", "year",
]
COUNT_COLUMNS = COLUMNS[3:10]
SENTIMENT_COLUMNS = COLUMNS[5:9]
MENTION_COLUMNS = ["is_ment_in_title", "is_ment_in_content"]
READ_TYPES = {"Scode": "string", "Coname": "string", "Date": "string",
              "year": "int64", **{name: "int64" for name in COUNT_COLUMNS}}


def data_files(archive, prefix, expected_count=None):
    """只选择本日期段的数据工作簿，并检查分片编号连续、不重复。"""
    files = [
        name for name in archive.namelist()
        if Path(name).name.startswith(prefix)
        and name.lower().endswith(".xlsx")
        and not name.startswith("__MACOSX/")
    ]
    if not files:
        raise ValueError(f"没有找到数据工作簿：{prefix}")
    numbers = [int(Path(name).stem.rsplit("_", 1)[1]) for name in files]
    if sorted(numbers) != list(range(1, len(files) + 1)):
        raise ValueError(f"工作簿分片编号不连续或重复：{prefix}")
    if expected_count is not None and len(files) != expected_count:
        raise ValueError(f"{prefix}应有{expected_count}个工作簿，实际{len(files)}个")
    return [name for _, name in sorted(zip(numbers, files))]


def clean_news_keys(frame, start_date, end_date):
    """两类原始表共用的新闻编号、自然日清洗。"""
    frame["news_id"] = frame["news_id"].str.strip()
    if frame["news_id"].isna().any() or frame["news_id"].eq("").any():
        raise ValueError("存在缺失或空白的新闻编号")
    frame["date"] = pd.to_datetime(
        frame["pub_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    if frame["date"].isna().any():
        raise ValueError("存在无法识别的发布日期")
    if not frame["date"].between(start_date, end_date).all():
        raise ValueError(f"存在超出{start_date}至{end_date}的发布日期")


def build_database(sentiment_zip, relation_zip, database_path,
                   start_date, end_date, expected_files=None):
    """逐工作簿构建新的明细数据库；拒绝静默复用已经存在的数据库。"""
    database_path = Path(database_path)
    if database_path.exists():
        raise FileExistsError(f"重建需要新的数据库，已有文件予以保留：{database_path}")
    for path in [sentiment_zip, relation_zip]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"找不到原始压缩包：{path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript("""
            CREATE TABLE sentiment (
                news_id TEXT NOT NULL,
                date TEXT NOT NULL,
                score REAL,
                PRIMARY KEY (news_id, date)
            );
            CREATE TABLE relations (
                news_id TEXT NOT NULL,
                date TEXT NOT NULL,
                stkcd TEXT NOT NULL,
                is_ment_in_title INTEGER NOT NULL,
                is_ment_in_content INTEGER NOT NULL,
                PRIMARY KEY (news_id, date, stkcd)
            );
        """)

        # 二、导入情感。相同新闻日只能有一种分数，缺失与有效值也视为冲突。
        date_range = f"{start_date}~{end_date}"
        with ZipFile(sentiment_zip) as archive:
            files = data_files(
                archive, f"网络新闻情感量化结果数据表_{date_range}_",
                None if expected_files is None else expected_files[0],
            )
            for number, name in enumerate(files, 1):
                print(f"情感表 {number}/{len(files)}：{Path(name).name}", flush=True)
                frame = pd.read_excel(
                    io.BytesIO(archive.read(name)),
                    usecols=["news_id", "pub_date", "overall_senti_score_adjust"],
                    dtype={"news_id": "string"}, engine="openpyxl",
                )
                clean_news_keys(frame, start_date, end_date)
                frame["score"] = pd.to_numeric(
                    frame["overall_senti_score_adjust"], errors="raise"
                )
                if frame["score"].isin([float("inf"), -float("inf")]).any():
                    raise ValueError("情感分数存在无穷值，请核查原始记录")
                frame = frame[["news_id", "date", "score"]]
                frame.to_sql("sentiment_batch", connection, if_exists="replace",
                             index=False, chunksize=50_000)
                conflict = connection.execute("""
                    SELECT news_id, date FROM sentiment_batch
                    GROUP BY news_id, date
                    HAVING COUNT(DISTINCT score) > 1
                        OR (COUNT(score) > 0 AND COUNT(score) < COUNT(*))
                    LIMIT 1
                """).fetchone()
                if conflict is not None:
                    raise ValueError(f"同一工作簿内存在情感分数冲突：{conflict}")
                conflict = connection.execute("""
                    SELECT b.news_id, b.date, b.score, s.score
                    FROM sentiment_batch AS b
                    JOIN sentiment AS s
                        ON b.news_id = s.news_id AND b.date = s.date
                    WHERE b.score IS NOT s.score
                    LIMIT 1
                """).fetchone()
                if conflict is not None:
                    raise ValueError(f"不同工作簿之间存在情感分数冲突：{conflict}")
                connection.execute("""
                    INSERT OR IGNORE INTO sentiment (news_id, date, score)
                    SELECT news_id, date, score FROM sentiment_batch
                """)
                connection.execute("DROP TABLE sentiment_batch")
                connection.commit()
                del frame
        sentiment_count, missing_count = connection.execute("""
            SELECT COUNT(*), COALESCE(SUM(score IS NULL), 0) FROM sentiment
        """).fetchone()
        if sentiment_count == 0:
            raise ValueError("导入后的情感表为空")
        print(f"情感表导入完成：{sentiment_count}条，分数缺失{missing_count}条", flush=True)

        # 三、导入公司关系。先检查空代码再补零；分片内外均按关系键取标记最大值。
        with ZipFile(relation_zip) as archive:
            files = data_files(
                archive, f"网络新闻相关上市公司表_{date_range}_",
                None if expected_files is None else expected_files[1],
            )
            for number, name in enumerate(files, 1):
                print(f"公司关系表 {number}/{len(files)}：{Path(name).name}", flush=True)
                frame = pd.read_excel(
                    io.BytesIO(archive.read(name)),
                    usecols=["news_id", "pub_date", "stkcd", *MENTION_COLUMNS],
                    dtype={"news_id": "string", "stkcd": "string"}, engine="openpyxl",
                )
                clean_news_keys(frame, start_date, end_date)
                frame["stkcd"] = frame["stkcd"].str.strip()
                if not frame["stkcd"].str.fullmatch(r"[0-9]{1,6}", na=False).all():
                    raise ValueError("存在缺失、空白或无效的股票代码")
                frame["stkcd"] = frame["stkcd"].str.zfill(6)
                for column in MENTION_COLUMNS:
                    frame[column] = pd.to_numeric(frame[column], errors="raise")
                    if not frame[column].isin([0, 1]).all():
                        raise ValueError(f"{column}存在缺失或非0/1取值")
                    frame[column] = frame[column].astype("int64")
                frame = frame.groupby(
                    ["news_id", "date", "stkcd"], as_index=False, sort=False
                )[MENTION_COLUMNS].max()
                with connection:
                    connection.executemany("""
                        INSERT INTO relations (
                            news_id, date, stkcd, is_ment_in_title, is_ment_in_content
                        ) VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (news_id, date, stkcd) DO UPDATE SET
                            is_ment_in_title = MAX(
                                relations.is_ment_in_title, excluded.is_ment_in_title
                            ),
                            is_ment_in_content = MAX(
                                relations.is_ment_in_content, excluded.is_ment_in_content
                            )
                    """, frame.itertuples(index=False, name=None))
                del frame
        relations, matched = connection.execute("""
            SELECT COUNT(*), COUNT(s.news_id)
            FROM relations AS r LEFT JOIN sentiment AS s
                ON r.news_id = s.news_id AND r.date = s.date
        """).fetchone()
        if relations == 0:
            raise ValueError("导入后的公司关系表为空")
        print(f"公司关系导入完成：{relations}条，匹配{matched}条，"
              f"未匹配{relations - matched}条", flush=True)
    finally:
        connection.close()


def check_daily_chunk(frame, start_date, end_date, previous_key):
    """检查公司日表的列、计数口径和跨批次唯一键，返回本批最后一个键。"""
    if list(frame.columns) != COLUMNS:
        raise ValueError("公司日表的列名或列顺序不一致")
    if not frame["Scode"].astype("string").str.fullmatch(r"[0-9]{6}", na=False).all():
        raise ValueError("公司日表存在无效股票代码")
    dates = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="raise")
    if not dates.between(start_date, end_date).all():
        raise ValueError("公司日表存在缺失日期或越界日期")
    if not frame["year"].eq(dates.dt.year).all():
        raise ValueError("公司日表的年份与日期不一致")
    if frame[COUNT_COLUMNS].lt(0).any().any() or frame["Allnews_All"].le(0).any():
        raise ValueError("公司日新闻计数不符合非负、全部新闻大于零的要求")
    if not frame[SENTIMENT_COLUMNS].sum(axis=1).eq(frame["Allnews_All"]).all():
        raise ValueError("四类情感新闻加总与全部新闻数不一致")
    for column in ["Newsnum_Title", "Newsnum_Cont"]:
        if frame[column].gt(frame["Allnews_All"]).any():
            raise ValueError("标题或正文提及篇数超过全部新闻数")
    keys = pd.MultiIndex.from_frame(frame[["Scode", "Date"]])
    if not keys.is_monotonic_increasing or not keys.is_unique:
        raise ValueError("公司日存在重复或排序异常")
    if previous_key is not None and keys[0] <= previous_key:
        raise ValueError("公司日在批次交界处存在重复或倒序")
    return keys[-1]


def aggregate_database(database_path, output_path, start_date, end_date):
    """只读明细数据库，分批写出公司日表，并核对关系数守恒。"""
    database_path, output_path = Path(database_path), Path(output_path)
    if not database_path.is_file():
        raise FileNotFoundError(f"找不到数据库：{database_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = output_path.with_name(f"{output_path.stem}_待完成.csv")
    connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    stats = dict(rows=0, relations=0, unclassified=0, min_date=None, max_date=None)
    previous_key = None
    try:
        expected_relations = connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        if expected_relations == 0:
            raise ValueError("公司关系表为空")
        # 控制排序、分组时的内存占用。
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute("PRAGMA cache_size = -65536")
        daily_sql = """
            SELECT r.stkcd AS Scode, '' AS Coname, r.date AS Date,
                SUM(r.is_ment_in_title) AS Newsnum_Title,
                SUM(r.is_ment_in_content) AS Newsnum_Cont,
                SUM(CASE WHEN s.score > 0 THEN 1 ELSE 0 END) AS Posnews_All,
                SUM(CASE WHEN s.score = 0 THEN 1 ELSE 0 END) AS Neunews_All,
                SUM(CASE WHEN s.score < 0 THEN 1 ELSE 0 END) AS Negnews_All,
                SUM(CASE WHEN s.score IS NULL THEN 1 ELSE 0 END) AS Unclassifiednews_All,
                COUNT(*) AS Allnews_All,
                CAST(SUBSTR(r.date, 1, 4) AS INTEGER) AS year
            FROM relations AS r LEFT JOIN sentiment AS s
                ON r.news_id = s.news_id AND r.date = s.date
            GROUP BY r.stkcd, r.date ORDER BY r.stkcd, r.date
        """
        print(f"开始公司日聚合：{start_date}至{end_date}", flush=True)
        with pending_path.open("w", encoding="utf-8-sig", newline="") as handle:
            for frame in pd.read_sql_query(daily_sql, connection, chunksize=CHUNK_SIZE):
                if frame.empty:
                    continue
                previous_key = check_daily_chunk(frame, start_date, end_date, previous_key)
                frame.to_csv(handle, index=False, header=(stats["rows"] == 0))
                stats["rows"] += len(frame)
                stats["relations"] += int(frame["Allnews_All"].sum())
                stats["unclassified"] += int(frame["Unclassifiednews_All"].sum())
                low, high = frame["Date"].min(), frame["Date"].max()
                stats["min_date"] = low if stats["min_date"] is None else min(low, stats["min_date"])
                stats["max_date"] = high if stats["max_date"] is None else max(high, stats["max_date"])
                print("已写出公司日行数：", stats["rows"], flush=True)
        if stats["rows"] == 0 or stats["relations"] != expected_relations:
            raise ValueError("聚合结果为空或聚合前后的关系总数不一致")
        pending_path.replace(output_path)
        return stats
    finally:
        connection.close()


def merge_outputs(part_paths, output_path, periods, expected_stats=None, check_baseline=True):
    """按互不重叠的日期段纵向拼接；保留各段内部的公司、日期顺序。"""
    if not periods or len(part_paths) != len(periods):
        raise ValueError("分段文件与日期段设置不对应")
    previous_end = None
    for path, period in zip(part_paths, periods):
        if period["start"] > period["end"]:
            raise ValueError("日期段的起止日期颠倒")
        if previous_end is not None and period["start"] <= previous_end:
            raise ValueError("日期段重叠或未按时间排列")
        previous_end = period["end"]
        if not Path(path).is_file():
            raise FileNotFoundError(f"找不到分段文件：{path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = output_path.with_name(f"{output_path.stem}_待完成.csv")
    total = dict(rows=0, relations=0, unclassified=0, min_date=None, max_date=None)
    with pending_path.open("w", encoding="utf-8-sig", newline="") as handle:
        for path, period in zip(part_paths, periods):
            print("正在拼接：", Path(path).name, flush=True)
            part = dict(rows=0, relations=0, unclassified=0)
            previous_key = None
            with pd.read_csv(path, encoding="utf-8-sig", dtype=READ_TYPES,
                             chunksize=CHUNK_SIZE) as reader:
                for frame in reader:
                    if frame.empty:
                        continue
                    previous_key = check_daily_chunk(frame, period["start"], period["end"], previous_key)
                    frame.to_csv(handle, index=False, header=(total["rows"] == 0))
                    part["rows"] += len(frame)
                    part["relations"] += int(frame["Allnews_All"].sum())
                    part["unclassified"] += int(frame["Unclassifiednews_All"].sum())
                    low, high = frame["Date"].min(), frame["Date"].max()
                    total["min_date"] = low if total["min_date"] is None else min(low, total["min_date"])
                    total["max_date"] = high if total["max_date"] is None else max(high, total["max_date"])
                    total["rows"] += len(frame)
            if part["rows"] == 0:
                raise ValueError(f"{period['period']}的分段结果为空")
            if expected_stats is not None:
                for key in ["rows", "relations", "unclassified"]:
                    if part[key] != expected_stats[period["period"]][key]:
                        raise ValueError(f"{period['period']}拼接前后的{key}不一致")
            if check_baseline and (
                part["rows"] != period["rows"] or part["relations"] != period["relations"]
                or part["unclassified"] != 0
            ):
                raise ValueError(f"{period['period']}与本批数据基准不一致：{part}")
            total["relations"] += part["relations"]
            total["unclassified"] += part["unclassified"]
            print(f"本段核对通过：{part}", flush=True)
    if check_baseline and (
        total["rows"] != sum(item["rows"] for item in periods)
        or total["relations"] != sum(item["relations"] for item in periods)
    ):
        raise ValueError("完整表与各日期段的基准加总不一致")
    pending_path.replace(output_path)
    return total


def main():
    """所有正式输出先在工作目录生成并核对，通过后再替换。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="网络新闻_构建_", dir=WORK_DIR) as temporary:
        stage_dir = Path(temporary)
        part_paths, expected_stats = [], {}
        for period in PERIODS:
            file_name = f"DATAGO_网络新闻量化统计_{period['period']}.csv"
            if REBUILD_FROM_RAW:
                date_range = f"{period['start']}~{period['end']}"
                sentiment_zip = (
                    INPUT_DIR / "网络新闻情感量化结果数据表（2009-01-01~2025-12-31）"
                    / f"网络新闻情感量化结果数据表（{date_range}）.zip"
                )
                relation_zip = (
                    INPUT_DIR / "网络新闻相关上市公司表（2009-01-01~2025-12-31）"
                    / f"网络新闻相关上市公司表（{date_range}）.zip"
                )
                database_path = stage_dir / f"网络新闻_{period['period']}.sqlite"
                expected_files = (
                    (period["sentiment_files"], period["relation_files"])
                    if CHECK_BASELINE else None
                )
                build_database(sentiment_zip, relation_zip, database_path,
                               period["start"], period["end"], expected_files)
                output_path = stage_dir / file_name
                expected_stats[period["period"]] = aggregate_database(
                    database_path, output_path, period["start"], period["end"]
                )
                # 只删除本次新建、已完成聚合的临时库，释放空间。
                database_path.unlink()
            else:
                output_path = OUTPUT_DIR / file_name
            part_paths.append(output_path)

        full_path = stage_dir / "DATAGO_网络新闻量化统计_2009_2025.csv"
        total = merge_outputs(part_paths, full_path, PERIODS,
                              expected_stats if REBUILD_FROM_RAW else None, CHECK_BASELINE)
        if REBUILD_FROM_RAW:
            for path in part_paths:
                path.replace(OUTPUT_DIR / path.name)
        full_path.replace(OUTPUT_DIR / full_path.name)

    print("\n2009—2025年完整表处理完成")
    print("输出行列数：", (total["rows"], len(COLUMNS)))
    print("最早日期：", total["min_date"])
    print("最晚日期：", total["max_date"])
    print("新闻—公司关系总数：", total["relations"])
    print("其中未分类关系数：", total["unclassified"])
    print("公司＋自然日唯一性检查：通过")
    print("已保存：", OUTPUT_DIR / "DATAGO_网络新闻量化统计_2009_2025.csv")


if __name__ == "__main__":
    main()
