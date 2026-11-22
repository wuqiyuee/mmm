"""合并原始总市值构造日度行业名次，再匹配基础样本资格。

以锐思补缺行业面板的股票、交易日为输出范围。基础样本资格在排名后匹配，
未进入基础样本的股票也可以有名次。无法排名的名次输出为 NA（缺失值）。
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


# 一、输入、输出及已核对的行业区间
ROOT = Path(__file__).resolve().parents[4]
INDUSTRY_PATH = (
    ROOT / "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv"
)
MARKET_PATH = ROOT / "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv"
BASE_SAMPLE_PATH = ROOT / "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv"
OUTPUT_PATH = ROOT / "中间结果/02_变量构造结果/行业提示资格/日度行业市值排名面板.csv"

CHUNK_SIZE = 400_000
A_SHARE_MARKETS = (1, 4, 16, 32)
KEYS = ["Stkcd", "Trddt"]
INDUSTRY_COLUMNS = ["Stkcd", "td", "Trddt", "IndustryCode"]
MARKET_COLUMNS = ["Stkcd", "Trddt", "Markettype", "Filling", "Dsmvtll"]
OUTPUT_COLUMNS = INDUSTRY_COLUMNS + ["industryGroup", "size", "industryRank", "baseSampleEligible"]

# 起止日期都包含在内。争议期间保留基础行业记录，排名时跳过。
DISPUTED_INTERVALS = [
    ("000780", "2007-04-26", "2008-04-18"),  # 平庄能源
    ("000861", "2010-09-07", "2010-11-16"),  # 海印股份
    ("002445", "2010-07-13", "2011-07-04"),  # 中南重工
    ("600772", "2004-12-29", "2006-11-29"),  # 中油龙昌
]

# 保留原行业排名脚本中已核对的两处修正，仅在原行业代码吻合时使用。
VERIFIED_INDUSTRY_OVERRIDES = [
    ("000780", "2002-03-06", "2007-04-25", "C0", "C01"),  # 草原兴发主营为食品加工
    ("000809", "2010-11-15", "2010-11-15", "C8", "C81"),  # 公告当日已启用医药分类
]


# 二、统一股票代码、日期并按年份暂存，保证同日股票在同一批排名
def stage_by_year(path, columns, temporary, label):
    parts = {}
    reader = pd.read_csv(
        path, usecols=columns, dtype="string", chunksize=CHUNK_SIZE, encoding="utf-8-sig"
    )
    for number, chunk in enumerate(reader):
        if chunk.empty:
            continue
        codes = chunk["Stkcd"].str.strip().str.replace(r"\.0+$", "", regex=True)
        if not codes.str.fullmatch(r"[0-9]{1,6}", na=False).all():
            raise ValueError(f"{path.name} 存在无法识别的股票代码")
        chunk["Stkcd"] = codes.str.zfill(6)
        chunk["Trddt"] = pd.to_datetime(chunk["Trddt"], format="%Y-%m-%d", errors="raise")
        if chunk["Trddt"].isna().any():
            raise ValueError(f"{path.name} 存在缺失的交易日期")
        for year, rows in chunk.groupby(chunk["Trddt"].dt.year, sort=False):
            part = temporary / f"{label}_{year}_{number}.parquet"
            rows.to_parquet(part, index=False)
            parts.setdefault(int(year), []).append(part)
    if not parts:
        raise ValueError(f"{path.name} 没有可读取的记录")
    return parts


def read_year(parts, year, columns):
    paths = parts.get(year, [])
    if paths:
        return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return pd.DataFrame(columns=columns).astype({"Stkcd": "string", "Trddt": "datetime64[ns]"})


def main():
    work_dir = ROOT / "work"
    work_dir.mkdir(exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="日度行业市值排名_", dir=work_dir) as directory:
        temporary = Path(directory)
        staged_output = temporary / OUTPUT_PATH.name
        industry_parts = stage_by_year(INDUSTRY_PATH, INDUSTRY_COLUMNS, temporary, "industry")
        market_parts = stage_by_year(MARKET_PATH, MARKET_COLUMNS, temporary, "market")
        base_parts = stage_by_year(BASE_SAMPLE_PATH, KEYS, temporary, "base")

        with staged_output.open("w", encoding="utf-8-sig", newline="") as output:
            for number, year in enumerate(sorted(industry_parts)):
                # 三、保留行业面板的股票日，合并原始当日总市值
                industry = read_year(industry_parts, year, INDUSTRY_COLUMNS)
                industry["td"] = industry["td"].astype("int64")
                calendar = industry[["Trddt", "td"]].drop_duplicates()
                if not calendar["td"].is_unique or not calendar["Trddt"].is_unique:
                    raise ValueError(f"{year} 年行业面板的交易日期与交易日序号不一致")

                market = read_year(market_parts, year, MARKET_COLUMNS)
                market_type = pd.to_numeric(market["Markettype"], errors="coerce")
                market["Filling"] = pd.to_numeric(market["Filling"], errors="coerce")
                # 保留项目使用的 A 股行情。填充标志 0 为真实记录，1 为允许的填充记录。
                market = market.loc[
                    market_type.isin(A_SHARE_MARKETS) & market["Filling"].isin([0, 1])
                ].copy()
                # 同股票同日优先保留真实记录，同类记录保留源文件最后一条。
                market = market.sort_values("Filling", ascending=False, kind="stable")
                market = market.drop_duplicates(KEYS, keep="last")
                market["size"] = pd.to_numeric(market["Dsmvtll"], errors="coerce").astype("float64") * 1000
                panel = industry.merge(market[KEYS + ["size"]], on=KEYS, how="left", validate="one_to_one")

                # 四、归并行业和计算名次，资格标记在排名完成后才匹配
                codes = panel["IndustryCode"].str.strip().str.upper()
                old_standard = panel["Trddt"].lt("2012-10-26")
                valid_group = codes.str.fullmatch(r"[A-Z][0-9]{2}", na=False)
                valid_old_code = codes.str.fullmatch(r"[A-Z][0-9]{4}|M", na=False)
                panel["industryGroup"] = codes.str[:3].where(valid_group | (old_standard & valid_old_code))

                for code, start, end, original, group in VERIFIED_INDUSTRY_OVERRIDES:
                    selected = panel["Stkcd"].eq(code) & panel["Trddt"].between(start, end)
                    panel.loc[selected & codes.eq(original), "industryGroup"] = group

                size = panel["size"]
                can_rank = panel["industryGroup"].notna() & size.gt(0) & np.isfinite(size)
                for code, start, end in DISPUTED_INTERVALS:
                    disputed = panel["Stkcd"].eq(code) & panel["Trddt"].between(start, end)
                    can_rank &= ~disputed

                ranked = panel.loc[can_rank, ["td", "industryGroup", "size", "Stkcd"]].sort_values(
                    ["td", "industryGroup", "size", "Stkcd"], ascending=[True, True, False, True]
                )
                ranks = ranked.groupby(["td", "industryGroup"], sort=False).cumcount() + 1
                panel["industryRank"] = ranks.astype("Int32")

                base_sample = read_year(base_parts, year, KEYS)
                base_sample["baseSampleEligible"] = 1
                panel = panel.merge(base_sample, on=KEYS, how="left", validate="one_to_one")
                panel["baseSampleEligible"] = panel["baseSampleEligible"].fillna(0).astype("int8")

                panel.sort_values(["td", "Stkcd"], inplace=True)
                panel.to_csv(
                    output, columns=OUTPUT_COLUMNS, index=False, header=(number == 0),
                    date_format="%Y-%m-%d", na_rep="NA",
                )
                print(f"{year} 年写入 {len(panel):,} 行，{len(ranked):,} 行有行业名次", flush=True)

        staged_output.replace(OUTPUT_PATH)
    print(f"日度行业市值排名面板已生成：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
