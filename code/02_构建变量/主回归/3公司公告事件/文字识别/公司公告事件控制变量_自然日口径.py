# 识别公司的公告事件，不做日期调整

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path("/Users/qiyuewu/Desktop/mmm复现数据包")

INPUT_PATH = ROOT / "中间结果/00_合并原始数据/CSMAR_公司公告事件表_2002_2025_合并结果.csv"
OUTPUT_PATH = ROOT / "中间结果/02_变量构造结果/公司公告事件/公司公告事件控制变量_同日口径.csv"

USECOLS = [
    "AnnouncementID",
    "Symbol",
    "DeclareDate",
    "Title",
    "ClassID",
    "ClassName",
    "ClassifyID",
    "ClassifyName",
]

# CSMAR 公司公告事件表 ClassID:
# 1=IPO, 2=定期报告, 3=增发融资, 4=配股融资, 5=可转债融资, 6=其他融资,
# 7=股权变动或并购重组, 8=经营交易, 9=股东大会, 10=股权质押冻结,
# 11=诉讼仲裁, 12=董事会与监事会, 13=突发事件, 14=违规,
# 15=董监高变动, 16=捐赠, 99=其他。
MAJOR_ANNOUNCEMENT_BROAD_CLASS_CODES = {"1", "3", "4", "5", "6", "7", "8", "10", "11", "13", "14", "15", "16"}
MAJOR_ANNOUNCEMENT_NARROW_CLASS_CODES = {"3", "4", "5", "6", "7", "8", "10", "11", "13", "14", "15"}

REPORT_TITLE_PATTERN = (
    r"年度报告|年报|半年度报告|半年报|季度报告|季报|中期报告|定期报告|财务报告"
)

MAJOR_ANNOUNCEMENT_BROAD_TITLE_PATTERN = (
    r"IPO|首次公开发行|上市公告|增发|配股|可转债|融资|募集|债券|"
    r"收购|并购|重组|资产购买|资产出售|股权转让|权益变动|"
    r"合同|中标|项目|投资|交易|担保|委托理财|"
    r"质押|冻结|诉讼|仲裁|突发|风险|违规|处罚|谴责|立案|调查|"
    r"董事|监事|高管|辞职|任职|聘任|捐赠"
)

MAJOR_ANNOUNCEMENT_NARROW_TITLE_PATTERN = (
    r"增发|配股|可转债|融资|募集|债券|"
    r"收购|并购|重组|资产购买|资产出售|股权转让|权益变动|"
    r"合同|中标|项目|投资|交易|担保|委托理财|"
    r"质押|冻结|诉讼|仲裁|突发|风险|违规|处罚|谴责|立案|调查|"
    r"董事|监事|高管|辞职|任职|聘任"
)

PERFORMANCE_FORECAST_BRIEF_TITLE_PATTERN = r"业绩(?:预告|预报|快报)"
PERFORMANCE_FORECAST_BRIEF_EXCLUDE_TITLE_PATTERN = (
    r"问询函|关注函|回复|专项说明|审核报告|核查意见|盈利预测|"
    r"业绩承诺|业绩补偿|承诺实现|实现情况|差异|监管函"
)


def normalize_stock_code(x: object) -> str:
    """标准化股票代码，填充为 6 位字符串。"""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s == "":
        return ""
    match = re.search(r"\d{1,6}", s)
    if match is None:
        return ""
    return match.group(0).zfill(6)


def contains_class_code(s: pd.Series, codes: Iterable[str]) -> pd.Series:
    """判断 ClassID 是否包含某组逗号分隔的分类编码。"""
    pattern = r"(?:^|,)(?:" + "|".join(re.escape(code) for code in sorted(codes)) + r")(?:,|$)"
    clean = (
        s.fillna("")
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
    )
    return clean.str.contains(pattern, regex=True)


def join_unique(s: pd.Series) -> str:
    """将组内非缺失唯一值合并为分号分隔字符串。"""
    values = []
    seen = set()
    for value in s.dropna():
        text = str(value).strip()
        if text == "" or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return ";".join(values)


def prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """清洗一个公告数据块并生成公告事件标记。"""
    df = chunk.copy()
    df["StkcdRecall"] = df["Symbol"].apply(normalize_stock_code)
    df["DeclareDate"] = pd.to_datetime(df["DeclareDate"], errors="coerce")

    title = df["Title"].fillna("").astype(str)
    is_performance_forecast_brief = (
        title.str.contains(PERFORMANCE_FORECAST_BRIEF_TITLE_PATTERN, regex=True)
        & ~title.str.contains(PERFORMANCE_FORECAST_BRIEF_EXCLUDE_TITLE_PATTERN, regex=True)
    )
    is_regular_report = (
        contains_class_code(df["ClassID"], {"2"})
        | title.str.contains(REPORT_TITLE_PATTERN, regex=True)
    ) & ~is_performance_forecast_brief

    major_broad_by_class = contains_class_code(df["ClassID"], MAJOR_ANNOUNCEMENT_BROAD_CLASS_CODES)
    major_narrow_by_class = contains_class_code(df["ClassID"], MAJOR_ANNOUNCEMENT_NARROW_CLASS_CODES)
    major_broad_by_title = title.str.contains(MAJOR_ANNOUNCEMENT_BROAD_TITLE_PATTERN, regex=True)
    major_narrow_by_title = title.str.contains(MAJOR_ANNOUNCEMENT_NARROW_TITLE_PATTERN, regex=True)

    df["ownAnnouncementAll"] = (~is_regular_report).astype("int8")
    df["ownMajorAnnouncementBroad"] = ((~is_regular_report) & (major_broad_by_class | major_broad_by_title)).astype("int8")
    df["ownMajorAnnouncementNarrow"] = ((~is_regular_report) & (major_narrow_by_class | major_narrow_by_title)).astype("int8")
    df["ownPerformanceForecastBrief"] = is_performance_forecast_brief.astype("int8")

    df = df.loc[
        df["StkcdRecall"].ne("")
        & df["DeclareDate"].notna()
        & df["ownAnnouncementAll"].eq(1)
    ].copy()

    df["date"] = df["DeclareDate"]
    df["Trddt"] = df["DeclareDate"]

    df["ownMajorAnnouncementBroadID"] = np.where(df["ownMajorAnnouncementBroad"].eq(1), df["AnnouncementID"], pd.NA)
    df["ownMajorAnnouncementNarrowID"] = np.where(df["ownMajorAnnouncementNarrow"].eq(1), df["AnnouncementID"], pd.NA)
    df["ownPerformanceForecastBriefID"] = np.where(
        df["ownPerformanceForecastBrief"].eq(1),
        df["AnnouncementID"],
        pd.NA,
    )

    return df


def aggregate_controls(events: pd.DataFrame) -> pd.DataFrame:
    """聚合到 StkcdRecall-date 层级，保证后续主面板可 1:1 合并。"""
    events = events.drop_duplicates(["StkcdRecall", "date", "AnnouncementID"]).copy()

    controls = (
        events
        .groupby(["StkcdRecall", "date", "Trddt"], as_index=False)
        .agg(
            ownAnnouncementAll=("ownAnnouncementAll", "max"),
            ownMajorAnnouncementBroad=("ownMajorAnnouncementBroad", "max"),
            ownMajorAnnouncementNarrow=("ownMajorAnnouncementNarrow", "max"),
            ownPerformanceForecastBrief=("ownPerformanceForecastBrief", "max"),
            ownAnnouncementCount=("AnnouncementID", "nunique"),
            ownMajorAnnouncementBroadCount=("ownMajorAnnouncementBroadID", "nunique"),
            ownMajorAnnouncementNarrowCount=("ownMajorAnnouncementNarrowID", "nunique"),
            ownPerformanceForecastBriefCount=("ownPerformanceForecastBriefID", "nunique"),
            AnnouncementIDExample=("AnnouncementID", "first"),
            ClassIDExample=("ClassID", "first"),
            ClassNameExample=("ClassName", "first"),
            ClassifyIDExample=("ClassifyID", "first"),
            ClassifyNameExample=("ClassifyName", "first"),
            TitleExample=("Title", "first"),
        )
    )

    for col in ["ownAnnouncementAll", "ownMajorAnnouncementBroad", "ownMajorAnnouncementNarrow", "ownPerformanceForecastBrief"]:
        controls[col] = controls[col].astype("int8")

    for col in ["ownAnnouncementCount", "ownMajorAnnouncementBroadCount", "ownMajorAnnouncementNarrowCount", "ownPerformanceForecastBriefCount"]:
        controls[col] = controls[col].astype("int16")

    controls = controls.sort_values(["StkcdRecall", "date"]).reset_index(drop=True)
    return controls


def build_controls(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """按同日口径构造公司公告事件控制变量。"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = []
    reader = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        usecols=USECOLS,
        dtype={
            "AnnouncementID": "string",
            "Symbol": "string",
            "ClassID": "string",
            "ClassifyID": "string",
        },
        chunksize=chunksize,
        low_memory=False,
    )

    for i, chunk in enumerate(reader, start=1):
        cleaned = prepare_chunk(chunk)
        if not cleaned.empty:
            parts.append(cleaned)
        print(f"已处理数据块 {i}，保留非财报公告行数：{len(cleaned)}", flush=True)

    if parts:
        events = pd.concat(parts, ignore_index=True)
        controls = aggregate_controls(events)
    else:
        controls = pd.DataFrame()

    controls.to_csv(output_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    print("=" * 80)
    print("公司公告事件控制变量生成完成：同日口径")
    print("输出文件：", output_path)
    print("输出行数：", len(controls))
    if not controls.empty:
        print("StkcdRecall-date 重复数：", controls.duplicated(["StkcdRecall", "date"]).sum())
        print(
            controls[
                [
                    "ownAnnouncementAll",
                    "ownMajorAnnouncementBroad",
                    "ownMajorAnnouncementNarrow",
                    "ownPerformanceForecastBrief",
                ]
            ].sum()
        )

    return controls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造公司公告事件控制变量：同日口径")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_controls(input_path=args.input, output_path=args.output, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
