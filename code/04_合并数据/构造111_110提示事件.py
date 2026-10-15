"""
把 111 与 110 记忆矩阵转成可合并到日度面板的提示事件表。

输入是“构造111_110记忆矩阵.py”输出的两张公司对—当前公告日表。
本脚本只完成表2需要的后续步骤：

1. 把公告自然日映射到事件交易日；
2. 合并提示公司公告前第3个交易日的市值；
3. 按市值断点识别大市值提示公司；
4. 删除同行业、供应链和共同分析师覆盖的公司对；
5. 构造 quarterLag1size90plus；
6. 折叠到“被提示公司 × 事件交易日”。

两组使用完全相同的辅助表和筛选口径，只更换输入记忆矩阵。
输出事件表只保留提示变量等于1的日期，后续合并到日度面板时将未匹配日期填为0。
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


GROUPS = ("111", "110")

MEMORY_RELATIVE_PATHS = {
    group: Path(f"中间结果/03_记忆矩阵/MemoryMatrix_{group}.csv")
    for group in GROUPS
}
EVENT_OUTPUT_RELATIVE_PATHS = {
    group: Path(
        f"中间结果/04_合并数据/CueEvents_{group}.csv"
    )
    for group in GROUPS
}

CALENDAR_RELATIVE_PATH = Path(
    "中间结果/01_清洗后样本/市场交易日历_td.csv"
)
MARKET_PANEL_RELATIVE_PATH = Path(
    "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv"
)
MARKET_BREAKPOINT_RELATIVE_PATH = Path(
    "中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv"
)
INDUSTRY_RELATIVE_PATH = Path(
    "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv"
)
SUPPLY_CHAIN_RELATIVE_PATH = Path(
    "input/CSMAR/供应链网络关系指标表2001-12-31至2025-12-31/SC_NetworkRelationsIndex.csv"
)
ANALYST_RELATIVE_ROOT = Path(
    "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后"
)

EVENT_DATE_MODE = "next_if_nontrading"
MARKETCAP_BREAKPOINT_MODE = "prev_month"
ANALYST_ID_MODE = "split"
ANALYST_LOOKBACK_DAYS = 365
CHUNK_SIZE = 1_000_000


def normalize_stock_code(value: object) -> str:
    """把证券代码标准化为六位字符串。"""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return f"{int(float(text)):06d}"
    except ValueError:
        return text.zfill(6)


def find_project_root(start: Path | None = None) -> Path:
    """从当前文件或传入目录向上寻找项目根目录。"""
    start_path = Path(start or __file__).resolve()
    search_start = start_path if start_path.is_dir() else start_path.parent
    for candidate in [search_start, *search_start.parents]:
        if (candidate / "input").exists() and (candidate / "code").exists():
            return candidate
    return search_start


def load_calendar(root: Path) -> pd.DataFrame:
    calendar = pd.read_csv(
        root / CALENDAR_RELATIVE_PATH,
        encoding="utf-8-sig",
        usecols=["Trddt", "td"],
        parse_dates=["Trddt"],
    )
    calendar["td"] = pd.to_numeric(calendar["td"], errors="coerce")
    calendar = calendar.dropna(subset=["Trddt", "td"])
    calendar = calendar.sort_values("Trddt").drop_duplicates("Trddt")
    return calendar


def load_market_panel(root: Path) -> pd.DataFrame:
    market = pd.read_csv(
        root / MARKET_PANEL_RELATIVE_PATH,
        encoding="utf-8-sig",
        usecols=["Stkcd", "td", "size", "sizew"],
        dtype={"Stkcd": "string"},
    )
    market["Stkcd"] = market["Stkcd"].apply(normalize_stock_code)
    market["td"] = pd.to_numeric(market["td"], errors="coerce")
    market = market.rename(
        columns={
            "td": "td_tm3",
            "size": "mktcap_tm3",
            "sizew": "float_mktcap_tm3",
        }
    )
    market = market.dropna(subset=["Stkcd", "td_tm3"])
    duplicate_count = market.duplicated(["Stkcd", "td_tm3"]).sum()
    if duplicate_count:
        raise ValueError(
            "DGTW日度面板在 Stkcd + td 层面存在重复，不能匹配市值。"
        )
    return market


def load_market_breakpoints(root: Path) -> pd.DataFrame:
    breakpoints = pd.read_csv(
        root / MARKET_BREAKPOINT_RELATIVE_PATH,
        encoding="utf-8-sig",
    )
    breakpoints["Trdmnt"] = pd.to_datetime(
        breakpoints["Trdmnt"], errors="coerce"
    )
    breakpoints = breakpoints.rename(
        columns={"Trdmnt": "Trdmnt_breakpoint"}
    )
    required = {
        "Trdmnt_breakpoint",
        "marketCap_90th_percentile",
    }
    missing = required - set(breakpoints.columns)
    if missing:
        raise ValueError(f"市值断点表缺少字段：{sorted(missing)}")
    breakpoints = breakpoints.dropna(
        subset=["Trdmnt_breakpoint", "marketCap_90th_percentile"]
    ).drop_duplicates("Trdmnt_breakpoint")
    return breakpoints


def load_industry(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    industry = pd.read_csv(
        root / INDUSTRY_RELATIVE_PATH,
        encoding="utf-8-sig",
        usecols=["Stkcd", "td", "IndustryCode"],
        dtype={"Stkcd": "string", "IndustryCode": "string"},
    )
    industry["Stkcd"] = industry["Stkcd"].apply(normalize_stock_code)
    industry["td"] = pd.to_numeric(industry["td"], errors="coerce")
    industry["IndustryCode"] = industry["IndustryCode"].str.strip()
    industry = industry.dropna(
        subset=["Stkcd", "td", "IndustryCode"]
    ).drop_duplicates(["Stkcd", "td"])

    cue_industry = industry.rename(
        columns={
            "td": "event_td",
            "IndustryCode": "cue_industry_code",
        }
    )
    recall_industry = industry.rename(
        columns={
            "Stkcd": "StkcdRecall",
            "td": "event_td",
            "IndustryCode": "recall_industry_code",
        }
    )
    return cue_industry, recall_industry


def load_vertical_pairs(root: Path) -> pd.DataFrame:
    supply = pd.read_csv(
        root / SUPPLY_CHAIN_RELATIVE_PATH,
        encoding="utf-8-sig",
        usecols=[
            "Symbol",
            "EndDate",
            "StateTypeCode",
            "PSCBusinessRelation",
            "PSCSymbol",
        ],
        dtype={
            "Symbol": "string",
            "StateTypeCode": "string",
            "PSCBusinessRelation": "string",
            "PSCSymbol": "string",
        },
        parse_dates=["EndDate"],
    )
    for col in [
        "Symbol",
        "StateTypeCode",
        "PSCBusinessRelation",
        "PSCSymbol",
    ]:
        supply[col] = supply[col].str.strip()

    supply["yearCue"] = supply["EndDate"].dt.year
    is_a_share_pair = (
        supply["StateTypeCode"].eq("1")
        & supply["PSCSymbol"].str.fullmatch(
            r"[0368][0-9]{5}", na=False
        )
        & supply["PSCBusinessRelation"].isin(["1", "2"])
    )
    supply = supply.loc[is_a_share_pair].copy()
    supply["Symbol"] = supply["Symbol"].apply(normalize_stock_code)
    supply["PSCSymbol"] = supply["PSCSymbol"].apply(normalize_stock_code)

    supply["upstream_stkcd"] = np.where(
        supply["PSCBusinessRelation"].eq("1"),
        supply["Symbol"],
        supply["PSCSymbol"],
    )
    supply["downstream_stkcd"] = np.where(
        supply["PSCBusinessRelation"].eq("1"),
        supply["PSCSymbol"],
        supply["Symbol"],
    )

    supply = supply.loc[
        supply["upstream_stkcd"].ne(supply["downstream_stkcd"]),
        ["yearCue", "upstream_stkcd", "downstream_stkcd"],
    ].drop_duplicates()

    forward = supply.rename(
        columns={
            "upstream_stkcd": "Stkcd",
            "downstream_stkcd": "StkcdRecall",
        }
    )[["yearCue", "Stkcd", "StkcdRecall"]]
    reverse = supply.rename(
        columns={
            "downstream_stkcd": "Stkcd",
            "upstream_stkcd": "StkcdRecall",
        }
    )[["yearCue", "Stkcd", "StkcdRecall"]]

    return (
        pd.concat([forward, reverse], ignore_index=True)
        .drop_duplicates()
        .assign(vertical_related=1)
    )


def load_analyst_overlap(root: Path) -> pd.DataFrame:
    folder = {
        "split": "拆开AnanmID",
        "team": "不拆开AnanmID",
    }
    if ANALYST_ID_MODE not in folder:
        raise ValueError("ANALYST_ID_MODE 必须是 split 或 team。")

    path = (
        root
        / ANALYST_RELATIVE_ROOT
        / folder[ANALYST_ID_MODE]
        / f"Full样本_分析师共同覆盖_滚动{ANALYST_LOOKBACK_DAYS}天.csv"
    )
    overlap = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=[
            "Annodt",
            "Stkcd",
            "StkcdRecall",
            "same_analyst_overlap",
            "n_common_analysts",
        ],
        dtype={
            "Stkcd": "string",
            "StkcdRecall": "string",
            "same_analyst_overlap": "Int64",
            "n_common_analysts": "Int64",
        },
        parse_dates=["Annodt"],
    )
    overlap["Stkcd"] = overlap["Stkcd"].apply(normalize_stock_code)
    overlap["StkcdRecall"] = overlap["StkcdRecall"].apply(
        normalize_stock_code
    )
    return overlap.drop_duplicates(
        ["Annodt", "Stkcd", "StkcdRecall"]
    )


def combine_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """把多个读取块折叠到被提示公司×事件交易日。"""
    if not parts:
        return pd.DataFrame(
            columns=["StkcdRecall", "event_Trddt", "quarterLag1size90plus"]
        )
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(
            ["StkcdRecall", "event_Trddt"],
            as_index=False,
        )["quarterLag1size90plus"]
        .max()
    )


def process_memory_chunk(
    memory_chunk: pd.DataFrame,
    calendar: pd.DataFrame,
    market: pd.DataFrame,
    breakpoints: pd.DataFrame,
    cue_industry: pd.DataFrame,
    recall_industry: pd.DataFrame,
    vertical_pairs: pd.DataFrame,
    analyst_overlap: pd.DataFrame,
) -> pd.DataFrame:
    """处理一块记忆矩阵并返回可折叠的提示事件。"""
    mm = memory_chunk.copy()
    mm["Stkcd"] = mm["Stkcd"].apply(normalize_stock_code)
    mm["StkcdRecall"] = mm["StkcdRecall"].apply(normalize_stock_code)
    mm = mm.loc[
        mm["Stkcd"].ne("") & mm["StkcdRecall"].ne("")
    ].copy()
    if mm.empty:
        return pd.DataFrame(
            columns=["StkcdRecall", "event_Trddt", "quarterLag1size90plus"]
        )

    annodt_key = (
        mm[["Annodt"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Annodt")
    )
    calendar_key = calendar.rename(
        columns={"Trddt": "event_Trddt", "td": "event_td"}
    )
    annodt_key = pd.merge_asof(
        annodt_key,
        calendar_key,
        left_on="Annodt",
        right_on="event_Trddt",
        direction="forward",
        allow_exact_matches=(EVENT_DATE_MODE == "next_if_nontrading"),
    )
    mm = mm.merge(
        annodt_key,
        on="Annodt",
        how="left",
        validate="m:1",
    )
    mm["event_Trddt"] = pd.to_datetime(
        mm["event_Trddt"], errors="coerce"
    )
    mm["event_td"] = pd.to_numeric(mm["event_td"], errors="coerce")
    mm["td_tm3"] = mm["event_td"] - 3

    mm = mm.merge(
        market[["Stkcd", "td_tm3", "mktcap_tm3", "float_mktcap_tm3"]],
        on=["Stkcd", "td_tm3"],
        how="left",
        validate="m:1",
    )

    if MARKETCAP_BREAKPOINT_MODE == "prev_month":
        mm["Trdmnt_for_breakpoint"] = (
            mm["event_Trddt"].dt.to_period("M")
            .sub(1)
            .dt.to_timestamp()
        )
    elif MARKETCAP_BREAKPOINT_MODE == "same_month":
        mm["Trdmnt_for_breakpoint"] = (
            mm["event_Trddt"].dt.to_period("M").dt.to_timestamp()
        )
    else:
        raise ValueError(
            "MARKETCAP_BREAKPOINT_MODE 必须是 prev_month 或 same_month。"
        )

    mm = mm.merge(
        breakpoints,
        left_on="Trdmnt_for_breakpoint",
        right_on="Trdmnt_breakpoint",
        how="left",
        validate="m:1",
    )
    mm = mm.dropna(
        subset=["mktcap_tm3", "marketCap_90th_percentile"]
    ).copy()
    if mm.empty:
        return pd.DataFrame(
            columns=["StkcdRecall", "event_Trddt", "quarterLag1size90plus"]
        )

    key_cols = ["Annodt", "Stkcd", "StkcdRecall"]
    mm_key = mm[key_cols + ["event_td"]].drop_duplicates()
    mm_key = mm_key.merge(
        cue_industry[["Stkcd", "event_td", "cue_industry_code"]],
        on=["Stkcd", "event_td"],
        how="left",
        validate="m:1",
    )
    mm_key = mm_key.merge(
        recall_industry[
            ["StkcdRecall", "event_td", "recall_industry_code"]
        ],
        on=["StkcdRecall", "event_td"],
        how="left",
        validate="m:1",
    )
    mm_key["same_industry"] = (
        mm_key["cue_industry_code"].notna()
        & mm_key["recall_industry_code"].notna()
        & (
            mm_key["cue_industry_code"]
            == mm_key["recall_industry_code"]
        )
    )
    mm_key = mm_key.loc[~mm_key["same_industry"]].copy()

    mm_key["yearCue"] = mm_key["Annodt"].dt.year
    mm_key = mm_key.merge(
        vertical_pairs,
        on=["yearCue", "Stkcd", "StkcdRecall"],
        how="left",
        validate="m:1",
    )
    mm_key = mm_key.loc[mm_key["vertical_related"].isna()].copy()

    mm_key = mm_key.merge(
        analyst_overlap,
        on=["Annodt", "Stkcd", "StkcdRecall"],
        how="left",
        validate="m:1",
    )

    # 固定排除已识别存在共同分析师覆盖的公司对。
    mm_key = mm_key.loc[
        mm_key["same_analyst_overlap"].isna()
    ].copy()

    mm_key_keep = mm_key[key_cols].drop_duplicates()
    mm = mm.merge(
        mm_key_keep,
        on=key_cols,
        how="inner",
        validate="m:1",
    )
    if mm.empty:
        return pd.DataFrame(
            columns=["StkcdRecall", "event_Trddt", "quarterLag1size90plus"]
        )

    mm["quarterLag1Alt"] = pd.to_numeric(
        mm["quarterLag1Alt"], errors="coerce"
    ).fillna(0).astype("int8")
    mm["size90plus"] = (
        mm["mktcap_tm3"] > mm["marketCap_90th_percentile"]
    ).astype("int8")
    mm["quarterLag1size90plus"] = (
        mm["quarterLag1Alt"] * mm["size90plus"]
    ).astype("int8")

    return (
        mm.groupby(
            ["StkcdRecall", "event_Trddt"],
            as_index=False,
        )["quarterLag1size90plus"]
        .max()
    )


def run_group(
    root: Path,
    group: str,
    calendar: pd.DataFrame,
    market: pd.DataFrame,
    breakpoints: pd.DataFrame,
    cue_industry: pd.DataFrame,
    recall_industry: pd.DataFrame,
    vertical_pairs: pd.DataFrame,
    analyst_overlap: pd.DataFrame,
) -> None:
    input_path = root / MEMORY_RELATIVE_PATHS[group]
    output_path = root / EVENT_OUTPUT_RELATIVE_PATHS[group]
    if not input_path.exists():
        raise FileNotFoundError(f"找不到{group}记忆矩阵：{input_path}")

    usecols = ["Annodt", "Stkcd", "StkcdRecall", "quarterLag1Alt"]
    reader = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        usecols=usecols,
        dtype={"Stkcd": "string", "StkcdRecall": "string"},
        parse_dates=["Annodt"],
        chunksize=CHUNK_SIZE,
    )

    with input_path.open(encoding="utf-8-sig") as input_file:
        total_rows = sum(1 for _ in input_file) - 1
    total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    parts: list[pd.DataFrame] = []

    for chunk in tqdm(
        reader,
        total=total_chunks,
        desc=f"处理{group}提示事件",
        unit="块",
    ):
        part = process_memory_chunk(
            memory_chunk=chunk,
            calendar=calendar,
            market=market,
            breakpoints=breakpoints,
            cue_industry=cue_industry,
            recall_industry=recall_industry,
            vertical_pairs=vertical_pairs,
            analyst_overlap=analyst_overlap,
        )
        if not part.empty:
            parts.append(part)
        if len(parts) >= 25:
            parts = [combine_parts(parts)]
            gc.collect()
        del chunk, part
        gc.collect()

    collapsed = combine_parts(parts)
    collapsed = collapsed.loc[
        collapsed["quarterLag1size90plus"].eq(1)
    ].copy()
    collapsed = collapsed.rename(columns={"event_Trddt": "date"})
    collapsed = collapsed[[
        "StkcdRecall",
        "date",
        "quarterLag1size90plus",
    ]]
    collapsed["StkcdRecall"] = collapsed["StkcdRecall"].apply(
        normalize_stock_code
    )
    collapsed["date"] = pd.to_datetime(
        collapsed["date"], errors="coerce"
    )
    collapsed = collapsed.dropna(subset=["StkcdRecall", "date"])
    collapsed = collapsed.drop_duplicates(
        ["StkcdRecall", "date"]
    ).sort_values(["StkcdRecall", "date"])

    duplicate_count = collapsed.duplicated(
        ["StkcdRecall", "date"]
    ).sum()
    if duplicate_count:
        raise ValueError(f"{group}事件表存在重复的被提示公司—日期。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    collapsed.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    print(f"{group}提示事件数：{len(collapsed):,}")
    print(f"输出文件：{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把111与110记忆矩阵转成提示事件表。"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="项目根目录；省略时自动寻找。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root or find_project_root()).resolve()
    print(f"项目根目录：{root}")

    if EVENT_DATE_MODE not in {
        "next_if_nontrading",
        "strict_next_trading",
    }:
        raise ValueError(
            "事件日期参数只能是 "
            "'next_if_nontrading'（仅非交易日顺延）或 "
            "'strict_next_trading'（严格使用下一交易日）。"
            f"当前值为：{EVENT_DATE_MODE!r}"
        )

    calendar = load_calendar(root)
    market = load_market_panel(root)
    breakpoints = load_market_breakpoints(root)
    cue_industry, recall_industry = load_industry(root)
    vertical_pairs = load_vertical_pairs(root)
    analyst_overlap = load_analyst_overlap(root)

    for group in GROUPS:
        run_group(
            root=root,
            group=group,
            calendar=calendar,
            market=market,
            breakpoints=breakpoints,
            cue_industry=cue_industry,
            recall_industry=recall_industry,
            vertical_pairs=vertical_pairs,
            analyst_overlap=analyst_overlap,
        )


if __name__ == "__main__":
    main()
