"""
将 DATAGO 报刊新闻明细聚合为 公司-自然日新闻表。

输入只使用“新闻相关上市公司表”和“新闻情感量化数据表”的六个分段包，
不读取覆盖相同年份的 fullpacked_data（全期打包数据）包，也不使用网络新闻表。

输出的一行代表一家公司在一个自然日的新闻统计，核心字段与现有
``计算AES和AEV.py`` 的输入兼容。文章情感分类默认使用
``overall_senti_score_adjust`` 的符号：正数为正面、负数为负面、零为中性；
分类规则通过 ``--neutral-band`` 可调整。DATAGO 官方聚合表的具体三分类阈值
未在本地指南中公开，因此本脚本将采用的规则明确写入命令行输出，而不宣称
与官方聚合表逐行完全一致。未匹配情感的新闻会保留在 ``Allnews_All``，并
单独计入 ``Unclassifiednews_All``；由于现有 AES/AEV 脚本不使用该字段，正式
运行前应查看分段已分类新闻率，必要时用 ``--max-unclassified-rate`` 设置失败阈值。
日期筛选、质量阈值及最终新闻篇数摘要均采用所选输出日期范围；正式结果仅在
全部分段成功后替换，临时文件在项目 work（临时工作目录）内自动清理。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
from pathlib import Path
import re
import tempfile
from typing import Iterable, Iterator, Sequence
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELATION_DIR = (
    ROOT
    / "input"
    / "DATAGO"
    / "新闻相关上市公司表（1998-01-01~2025-12-31）"
)
DEFAULT_SENTIMENT_DIR = (
    ROOT
    / "input"
    / "DATAGO"
    / "新闻情感量化数据表（1998-01-01~2025-12-31）"
)
DEFAULT_OUTPUT = (
    ROOT
    / "中间结果"
    / "00_合并原始数据"
    / "DATAGO_报刊新闻量化统计_1998_2025.csv"
)

DEFAULT_SCORE_COLUMN = "overall_senti_score_adjust"
RELATION_COLUMNS = [
    "news_id",
    "pub_date",
    "stkcd",
    "is_ment_in_title",
    "is_ment_in_content",
]
SENTIMENT_BASE_COLUMNS = ["news_id", "pub_date"]
DOCUMENTATION_WORKBOOKS = {"《报刊新闻量化舆情数据库》字段说明.xlsx"}
OUTPUT_COLUMNS = [
    "Scode",
    "Coname",
    "Date",
    "Newsnum_Title",
    "Newsnum_Cont",
    "Posnews_All",
    "Neunews_All",
    "Negnews_All",
    "Unclassifiednews_All",
    "Allnews_All",
    "year",
]
SEGMENT_PATTERN = re.compile(
    r"[（(](?P<start>\d{4}-\d{2}-\d{2})[~～](?P<end>\d{4}-\d{2}-\d{2})[）)]\.zip$"
)


@dataclass(frozen=True)
class SegmentPair:
    """同一日期段的公司关系表和情感表。"""

    start: pd.Timestamp
    end: pd.Timestamp
    relation_path: Path
    sentiment_path: Path


def normalize_stock_codes(values: pd.Series) -> pd.Series:
    """将股票代码统一为六位字符串。"""

    extracted = (
        values.astype("string")
        .str.strip()
        .str.extract(r"(\d+)", expand=False)
    )
    return extracted.str.zfill(6)


def normalize_dates(
    values: pd.Series,
    field_name: str = "发布日期",
) -> pd.Series:
    """把发布日期归一到自然日；缺失或无法解析的日期直接报错。"""

    # DATAGO 文件中可能同时出现 YYYY-MM-DD 和带时分秒的日期；
    # format="mixed" 避免 pandas 按首行格式解析后误删另一种格式。
    try:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:  # 兼容较旧 pandas 版本
        parsed = pd.to_datetime(values, errors="coerce")
    invalid = parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].head(3).tolist()
        raise ValueError(
            f"{field_name} 存在 {int(invalid.sum())} 条缺失或无法解析的日期，"
            f"原值示例: {examples}"
        )
    return parsed.dt.normalize()


def parse_output_date_range(
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """解析输出日期边界，拒绝缺失日期及倒置的日期范围。"""

    start = pd.Timestamp(start_date) if start_date is not None else None
    end = pd.Timestamp(end_date) if end_date is not None else None
    if (start is not None and pd.isna(start)) or (
        end is not None and pd.isna(end)
    ):
        raise ValueError("输出起止日期不能是缺失日期")
    start = start.normalize() if start is not None else None
    end = end.normalize() if end is not None else None
    if start is not None and end is not None and end < start:
        raise ValueError("输出结束日期不能早于开始日期")
    return start, end


def classify_sentiment(
    scores: pd.Series,
    neutral_band: float = 0.0,
) -> pd.Series:
    """
    按连续情感分数生成文章级三分类。

    ``neutral_band=0`` 时，分数大于0为正面、小于0为负面、等于0为中性。
    缺失或无法转换的分数标为 ``unclassified``（未分类）。
    """

    band = float(neutral_band)
    if not np.isfinite(band) or band < 0:
        raise ValueError("neutral_band 必须是非负有限数")

    numeric = pd.to_numeric(scores, errors="coerce")
    result = pd.Series("unclassified", index=scores.index, dtype="string")
    # 无穷值不是有效的模型分数；不要把它们误当成极端正面/负面。
    valid = numeric.notna() & np.isfinite(numeric)
    result.loc[valid & (numeric > band)] = "positive"
    result.loc[valid & (numeric < -band)] = "negative"
    result.loc[valid & (numeric.abs() <= band)] = "neutral"
    return result


def _required_columns_error(table_name: str, columns: Iterable[str]) -> ValueError:
    missing = ", ".join(sorted(set(columns)))
    return ValueError(f"{table_name} 缺少必要字段: {missing}")


def prepare_sentiment(
    sentiment: pd.DataFrame,
    score_column: str = DEFAULT_SCORE_COLUMN,
    neutral_band: float = 0.0,
) -> pd.DataFrame:
    """清洗并校验新闻级情感表，返回一条新闻一条记录。"""

    required = [*SENTIMENT_BASE_COLUMNS, score_column]
    missing = [column for column in required if column not in sentiment.columns]
    if missing:
        raise _required_columns_error("情感表", missing)

    out = sentiment[required].copy()
    out["news_id"] = out["news_id"].astype("string").str.strip()
    out["pub_date"] = normalize_dates(out["pub_date"], "情感表发布日期")
    out[score_column] = pd.to_numeric(out[score_column], errors="coerce")
    out[score_column] = out[score_column].where(
        np.isfinite(out[score_column])
    )
    out = out.dropna(subset=["news_id"])

    key = ["news_id", "pub_date"]
    grouped = out.groupby(key, dropna=False)[score_column].agg(
        nonmissing_nunique=lambda values: values.dropna().nunique(),
        has_missing=lambda values: values.isna().any(),
    )
    conflicts = grouped[
        (grouped["nonmissing_nunique"] > 1)
        | (
            grouped["has_missing"]
            & (grouped["nonmissing_nunique"] > 0)
        )
    ]
    if not conflicts.empty:
        examples = list(conflicts.index[:3])
        raise ValueError(
            "同一 news_id + pub_date 存在冲突的情感分数，示例: "
            f"{examples}"
        )

    # 允许完全相同的重复行，但不允许静默覆盖不同分数。
    out = out.drop_duplicates(key, keep="last")
    out["sentiment_class"] = classify_sentiment(
        out[score_column], neutral_band=neutral_band
    )
    return out[["news_id", "pub_date", score_column, "sentiment_class"]]


def prepare_relations(relations: pd.DataFrame) -> pd.DataFrame:
    """清洗公司关系表，并在新闻-公司层面去重。"""

    missing = [column for column in RELATION_COLUMNS if column not in relations.columns]
    if missing:
        raise _required_columns_error("公司关系表", missing)

    out = relations[RELATION_COLUMNS].copy()
    out["news_id"] = out["news_id"].astype("string").str.strip()
    out["pub_date"] = normalize_dates(out["pub_date"], "关系表发布日期")
    out["stkcd"] = normalize_stock_codes(out["stkcd"])
    out = out.dropna(subset=["news_id", "stkcd"])

    for column in ["is_ment_in_title", "is_ment_in_content"]:
        numeric = pd.to_numeric(out[column], errors="coerce").fillna(0)
        invalid = ~numeric.isin([0, 1])
        if invalid.any():
            examples = numeric.loc[invalid].head(3).tolist()
            raise ValueError(
                f"{column} 包含非 0/1 标记，示例: {examples}"
            )
        out[column] = numeric.gt(0).astype("int8")

    key = ["news_id", "pub_date", "stkcd"]
    # 同一新闻-公司若在重复记录中出现过一次标题/正文提及，保留该标记。
    out = (
        out.groupby(key, as_index=False, dropna=False)[
            ["is_ment_in_title", "is_ment_in_content"]
        ]
        .max()
    )
    return out


def aggregate_daily_news(
    relations: pd.DataFrame,
    sentiment: pd.DataFrame,
    score_column: str = DEFAULT_SCORE_COLUMN,
    neutral_band: float = 0.0,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """聚合所选日期的公司日统计，附带同范围的独立新闻篇数摘要。"""

    start, end = parse_output_date_range(start_date, end_date)
    rel = prepare_relations(relations)
    sent = prepare_sentiment(
        sentiment,
        score_column=score_column,
        neutral_band=neutral_band,
    )

    # 在关联和文章级摘要之前筛选，避免把范围外新闻计入输出摘要。
    if start is not None:
        rel = rel.loc[rel["pub_date"] >= start]
    if end is not None:
        rel = rel.loc[rel["pub_date"] <= end]
    if rel.empty:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)
        result.attrs["sentiment_total_articles"] = 0
        result.attrs["sentiment_unclassified_articles"] = 0
        return result

    merged = rel.merge(
        sent[["news_id", "pub_date", "sentiment_class"]],
        on=["news_id", "pub_date"],
        how="left",
        validate="many_to_one",
    )
    # 没有匹配到情感表的新闻保留在总新闻数中，并单独计入未分类。
    merged["sentiment_class"] = merged["sentiment_class"].fillna("unclassified")

    # 用新闻编号作为计数对象，避免 ment_times（提及次数）被误当成新闻篇数。
    merged["title_news_id"] = merged["news_id"].where(
        merged["is_ment_in_title"].eq(1)
    )
    merged["content_news_id"] = merged["news_id"].where(
        merged["is_ment_in_content"].eq(1)
    )
    for label in ["positive", "neutral", "negative", "unclassified"]:
        merged[f"{label}_news_id"] = merged["news_id"].where(
            merged["sentiment_class"].eq(label)
        )

    grouped = (
        merged.groupby(["stkcd", "pub_date"], as_index=False, dropna=False)
        .agg(
            Newsnum_Title=("title_news_id", "nunique"),
            Newsnum_Cont=("content_news_id", "nunique"),
            Posnews_All=("positive_news_id", "nunique"),
            Neunews_All=("neutral_news_id", "nunique"),
            Negnews_All=("negative_news_id", "nunique"),
            Unclassifiednews_All=("unclassified_news_id", "nunique"),
            Allnews_All=("news_id", "nunique"),
        )
    )

    grouped["Scode"] = grouped.pop("stkcd")
    grouped["Coname"] = ""
    grouped["Date"] = grouped.pop("pub_date")
    grouped["year"] = grouped["Date"].dt.year.astype("int16")

    count_columns = [
        "Newsnum_Title",
        "Newsnum_Cont",
        "Posnews_All",
        "Neunews_All",
        "Negnews_All",
        "Unclassifiednews_All",
        "Allnews_All",
    ]
    grouped[count_columns] = grouped[count_columns].astype("int64")
    classified_total = grouped[
        ["Posnews_All", "Neunews_All", "Negnews_All", "Unclassifiednews_All"]
    ].sum(axis=1)
    if not classified_total.eq(grouped["Allnews_All"]).all():
        raise AssertionError("情感分类计数与全部新闻数不一致")

    article_status = merged[
        ["news_id", "pub_date", "sentiment_class"]
    ].drop_duplicates()
    result = grouped[OUTPUT_COLUMNS].sort_values(["Scode", "Date"]).reset_index(
        drop=True
    )
    result.attrs["sentiment_total_articles"] = int(len(article_status))
    result.attrs["sentiment_unclassified_articles"] = int(
        article_status["sentiment_class"].eq("unclassified").sum()
    )
    return result


def parse_segment_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """解析分段包日期；fullpacked_data 和说明包返回 None。"""

    if "fullpacked_data" in path.name:
        return None
    match = SEGMENT_PATTERN.search(path.name)
    if match is None:
        return None
    start = pd.Timestamp(match.group("start"))
    end = pd.Timestamp(match.group("end"))
    if end < start:
        raise ValueError(f"压缩包日期范围倒置: {path}")
    return start, end


def discover_segment_archives(directory: Path) -> dict[tuple[pd.Timestamp, pd.Timestamp], Path]:
    """发现一个 DATAGO 表的分段包，并排除全期包。"""

    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    found: dict[tuple[pd.Timestamp, pd.Timestamp], Path] = {}
    for path in sorted(directory.glob("*.zip")):
        date_range = parse_segment_range(path)
        if date_range is None:
            continue
        if date_range in found:
            raise ValueError(f"同一日期段存在多个压缩包: {date_range}")
        found[date_range] = path
    if not found:
        raise FileNotFoundError(f"未发现分段压缩包: {directory}")
    return found


def pair_segment_archives(
    relation_dir: Path = DEFAULT_RELATION_DIR,
    sentiment_dir: Path = DEFAULT_SENTIMENT_DIR,
) -> list[SegmentPair]:
    """按日期段配对公司关系包和情感包，并检查日期段连续且不重叠。"""

    relation = discover_segment_archives(relation_dir)
    sentiment = discover_segment_archives(sentiment_dir)
    if set(relation) != set(sentiment):
        raise ValueError(
            "公司关系表和情感表的日期段不一致: "
            f"关系表独有={sorted(set(relation) - set(sentiment))}, "
            f"情感表独有={sorted(set(sentiment) - set(relation))}"
        )

    pairs = [
        SegmentPair(start, end, relation[(start, end)], sentiment[(start, end)])
        for start, end in sorted(relation)
    ]
    for previous, current in zip(pairs, pairs[1:]):
        if current.start <= previous.end:
            raise ValueError(
                "分段日期重叠: "
                f"{previous.start.date()}~{previous.end.date()} 与 "
                f"{current.start.date()}~{current.end.date()}"
            )
        expected_start = previous.end + pd.Timedelta(days=1)
        if current.start != expected_start:
            raise ValueError(
                "分段日期存在缺口: "
                f"上一段结束于 {previous.end.date()}，"
                f"下一段从 {current.start.date()} 开始"
            )
    return pairs


def validate_segment_dates(
    table: pd.DataFrame,
    pair: SegmentPair,
    table_name: str,
) -> None:
    """确认压缩包内部的发布日期没有越过文件名声明的日期段。"""

    archive_name = (
        pair.relation_path.name
        if table_name == "关系表"
        else pair.sentiment_path.name
    )
    dates = normalize_dates(
        table["pub_date"], f"{table_name} {archive_name} 的发布日期"
    )
    outside = (dates < pair.start) | (dates > pair.end)
    if outside.any():
        examples = dates.loc[outside].head(3).dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(
            f"{table_name} {archive_name} 含有超出声明日期段的记录，"
            f"示例: {examples}"
        )


def iter_excel_members(
    archive_path: Path,
    required_columns: Sequence[str],
) -> Iterator[pd.DataFrame]:
    """只跳过已知说明工作簿；任何数据分片读取失败或缺列均报错。"""

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not member.filename.lower().endswith(".xlsx"):
                continue
            if Path(member.filename).name in DOCUMENTATION_WORKBOOKS:
                continue
            try:
                payload = archive.read(member)
                header = pd.read_excel(io.BytesIO(payload), nrows=0)
                missing = set(required_columns).difference(header.columns)
                if missing:
                    raise _required_columns_error("数据分片", missing)
                frame = pd.read_excel(
                    io.BytesIO(payload), usecols=list(required_columns)
                )
            except Exception as error:
                raise ValueError(
                    f"读取数据分片失败: {archive_path} / {member.filename}: {error}"
                ) from error
            yield frame


def read_segment_table(
    archive_path: Path,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    """读取一个分段包内的所有数据工作簿并拼接。"""

    frames = list(iter_excel_members(archive_path, required_columns))
    if not frames:
        raise ValueError(
            f"压缩包中没有包含所需字段的工作簿: {archive_path}"
        )
    return pd.concat(frames, ignore_index=True)


def aggregate_segment_archives(
    output_path: Path = DEFAULT_OUTPUT,
    relation_dir: Path = DEFAULT_RELATION_DIR,
    sentiment_dir: Path = DEFAULT_SENTIMENT_DIR,
    score_column: str = DEFAULT_SCORE_COLUMN,
    neutral_band: float = 0.0,
    start_date: str | None = None,
    end_date: str | None = None,
    max_unclassified_rate: float | None = None,
) -> dict[str, int | float]:
    """按所选日期分段聚合，成功后原子替换正式结果并返回同范围摘要。

    新闻篇数按新闻编号与自然日去重，不累加公司日篇数。
    分段数分别报告输入、实际处理及产生输出的分段数量。
    """

    pairs = pair_segment_archives(relation_dir, sentiment_dir)
    start, end = parse_output_date_range(start_date, end_date)
    if max_unclassified_rate is not None and not (
        np.isfinite(max_unclassified_rate)
        and 0 <= max_unclassified_rate <= 1
    ):
        raise ValueError("max_unclassified_rate 必须在 [0, 1] 内")

    selected_pairs = [
        pair
        for pair in pairs
        if (start is None or pair.end >= start)
        and (end is None or pair.start <= end)
    ]
    if not selected_pairs:
        raise ValueError("输出日期范围与输入分段没有交集")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = ROOT / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    if work_dir.stat().st_dev != output_path.parent.stat().st_dev:
        raise ValueError(
            "正式输出与项目临时工作目录不在同一文件系统，无法原子替换；"
            "请将输出设置在项目所在文件系统。"
        )

    total_rows = 0
    total_articles = 0
    total_unclassified = 0
    output_segments = 0

    # 临时结果始终位于项目临时工作目录；异常退出时自动清理。
    with tempfile.TemporaryDirectory(prefix="datago_news_", dir=work_dir) as temp_dir:
        pending_output = Path(temp_dir) / output_path.name
        for pair in selected_pairs:
            sentiment = read_segment_table(
                pair.sentiment_path,
                [*SENTIMENT_BASE_COLUMNS, score_column],
            )
            relations = read_segment_table(pair.relation_path, RELATION_COLUMNS)
            validate_segment_dates(sentiment, pair, "情感表")
            validate_segment_dates(relations, pair, "关系表")
            daily = aggregate_daily_news(
                relations,
                sentiment,
                score_column=score_column,
                neutral_band=neutral_band,
                start_date=start,
                end_date=end,
            )
            if daily.empty:
                continue

            segment_articles = int(daily.attrs["sentiment_total_articles"])
            segment_unclassified = int(
                daily.attrs["sentiment_unclassified_articles"]
            )
            segment_rate = segment_unclassified / segment_articles
            print(
                f"{pair.start.date()}~{pair.end.date()}（所选输出日期内）: "
                f"独立新闻篇数={segment_articles}; "
                f"已分类新闻率={1 - segment_rate:.4%}; "
                f"未分类新闻率={segment_rate:.4%}"
            )
            if (
                max_unclassified_rate is not None
                and segment_rate > max_unclassified_rate
            ):
                raise ValueError(
                    f"{pair.start.date()}~{pair.end.date()} 所选输出日期内的"
                    f"未分类新闻率 {segment_rate:.4%} 超过阈值 "
                    f"{max_unclassified_rate:.4%}"
                )

            first = output_segments == 0
            daily.to_csv(
                pending_output,
                mode="w" if first else "a",
                header=first,
                index=False,
                encoding="utf-8-sig" if first else "utf-8",
                date_format="%Y-%m-%d",
            )
            output_segments += 1
            total_rows += len(daily)
            total_articles += segment_articles
            total_unclassified += segment_unclassified

        if output_segments == 0:
            raise ValueError("筛选后没有可写出的新闻记录")
        summary = {
            "input_segment_count": len(pairs),
            "processed_segment_count": len(selected_pairs),
            "segment_count": output_segments,
            "output_rows": total_rows,
            "total_news_articles": total_articles,
            "unclassified_news_articles": total_unclassified,
            "unclassified_rate": total_unclassified / total_articles,
        }
        # 同一文件系统内替换；此前的任何失败都不会覆盖原正式结果。
        pending_output.replace(output_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 DATAGO 报刊新闻明细聚合为公司-自然日统计表"
    )
    parser.add_argument("--relation-dir", type=Path, default=DEFAULT_RELATION_DIR)
    parser.add_argument("--sentiment-dir", type=Path, default=DEFAULT_SENTIMENT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--score-column",
        default=DEFAULT_SCORE_COLUMN,
        choices=[
            "overall_senti_score_adjust",
            "overall_senti_score",
            "content_senti_score",
            "content_senti_score_ref",
            "wgt_content_senti_score",
            "wgt_content_senti_score_ref",
        ],
    )
    parser.add_argument(
        "--neutral-band",
        type=float,
        default=0.0,
        help="情感分数绝对值不超过该值时归为中性，默认0",
    )
    parser.add_argument("--start-date", help="可选的输出起始自然日")
    parser.add_argument("--end-date", help="可选的输出结束自然日")
    parser.add_argument(
        "--max-unclassified-rate",
        type=float,
        default=None,
        help="可选；某分段在所选输出日期内的未分类新闻率超限时中止，范围[0,1]",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查分段配对，不读取工作簿正文或写出结果",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pairs = pair_segment_archives(args.relation_dir, args.sentiment_dir)
    print(f"发现 {len(pairs)} 个报刊输入分段，已排除 fullpacked_data 全期包")
    for pair in pairs:
        print(
            f"{pair.start.date()}~{pair.end.date()}: "
            f"{pair.relation_path.name} + {pair.sentiment_path.name}"
        )
    print(
        f"情感字段={args.score_column}; "
        f"中性区间=[{-args.neutral_band}, {args.neutral_band}]"
    )
    if args.max_unclassified_rate is None and not args.check_only:
        print(
            "警告：未设置未分类新闻率阈值；下游 AES/AEV 不会使用 "
            "Unclassifiednews_All。"
        )
    if args.check_only:
        return

    summary = aggregate_segment_archives(
        output_path=args.output,
        relation_dir=args.relation_dir,
        sentiment_dir=args.sentiment_dir,
        score_column=args.score_column,
        neutral_band=args.neutral_band,
        start_date=args.start_date,
        end_date=args.end_date,
        max_unclassified_rate=args.max_unclassified_rate,
    )
    print(f"已写出: {args.output}")
    print(summary)


if __name__ == "__main__":
    main()
