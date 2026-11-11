"""第一阶段：准备所有提示口径共用的候选公司对。

输入为正式记忆矩阵、提示公司日度市值、月度市值断点、供应链关系、
分析师共同覆盖结果和盈余惊喜。输出一行代表提示公司×事件交易日×被提示公司。
同行业排除及主提示资格判断留在第二阶段，第一阶段保留市值和断点缺失记录。
本文件包含本阶段所需的全部计算函数，不导入或执行其他项目脚本。
导入本文件不读取数据；直接执行时由主函数按顺序完成计算和保存。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


START_YEAR = 2005
END_YEAR = 2025
ANALYST_ID_MODE = "split"
ANALYST_LOOKBACK_DAYS = 365
DROP_ANALYST_OVERLAP = True
MARKETCAP_BREAKPOINT_MODE = "prev_month"
CHUNK_SIZE = 250_000

OUTPUT_ROOT_NAME = "三阶段合并"
OUTPUT_NAME = "候选公司对基础表.csv"
SUMMARY_NAME = "候选公司对审计摘要.json"


def find_root() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "AGENTS.md").is_file() and (folder / "中间结果").is_dir():
            return folder
    raise FileNotFoundError("找不到包含AGENTS.md和中间结果的项目根目录。")


def build_input_paths(root: Path) -> dict:
    """第一阶段固定使用的正式输入文件。"""
    return {
        "MemoryMatrixPattern_path": root / "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv",
        "TRD_Dalyr_DGTW_path": root / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv",
        "marketcap_group_path": root / "中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv",
        "SC_NetWorkRelationsIndex_path": root / "input/CSMAR/供应链网络关系指标表2001-12-31至2025-12-31/SC_NetworkRelationsIndex.csv",
        "analyst_forecast_path": root / "中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv",
        "sue_path": root / "中间结果/02_变量构造结果/4sue/sue.csv",
    }


def output_dir(root: Path) -> Path:
    return root / "中间结果/04_合并数据" / OUTPUT_ROOT_NAME


def save_summary(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def normalize_stock_code(x: object) -> str:
    """标准化股票代码，填充为6位数，并转换为字符串"""
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s == '':
        return ''
    return f"{int(float(s)):06d}"


def union_pipe_values(values):
    """合并以竖线分隔的来源标识，去重后按升序保留。"""
    tokens = set()
    for value in values.dropna():
        tokens.update(token for token in str(value).split("|") if token)
    return "|".join(sorted(tokens))


def load_pattern_matrix_with_marketcap(matrix_path, daily_panel_path):
    """读取公司对记忆矩阵，合并提示公司事件前第三个交易日的市值。"""

    mm = pd.read_csv(
        matrix_path,
        encoding="utf-8",
        parse_dates=["Annodt", "event_Trddt"],
    )

    mm["Stkcd"] = mm["Stkcd"].apply(normalize_stock_code)
    mm["StkcdRecall"] = mm["StkcdRecall"].apply(normalize_stock_code)

    # 直接使用上游生成的事件交易日与交易日序号。
    required_event_cols = {
        "event_Trddt", "event_td", "currentAccpersLag1",
        "lag1AnnualAnchor", "lag1Q1Anchor",
    }
    missing_event_cols = required_event_cols - set(mm.columns)

    if missing_event_cols:
        raise ValueError(
            f"记忆矩阵缺少事件日期或报告来源字段：{missing_event_cols}。"
            "请先按保留全部合格报告的口径重建记忆矩阵。"
        )

    if mm.duplicated(["Stkcd", "Annodt", "StkcdRecall"]).any():
        raise ValueError("记忆矩阵在提示公司×自然公告日×被提示公司层面不唯一。")

    mm["event_Trddt"] = pd.to_datetime(
        mm["event_Trddt"], errors="coerce"
    )
    mm["event_td"] = pd.to_numeric(
        mm["event_td"], errors="coerce"
    )
    mm["td_tm3"] = mm["event_td"] - 3

    # 读取个股日度市值。
    mktcap = pd.read_csv(
        daily_panel_path,
        encoding="utf-8",
        usecols=["Stkcd", "td", "size", "sizew"],
        dtype={"Stkcd": "string"},
    )

    mktcap["Stkcd"] = mktcap["Stkcd"].apply(normalize_stock_code)
    mktcap["td"] = pd.to_numeric(mktcap["td"], errors="coerce")

    mktcap = mktcap.rename(
        columns={
            "td": "td_tm3",
            "size": "mktcap_tm3",
            "sizew": "float_mktcap_tm3",
        }
    )

    # 同一提示公司在同一事件日可能对应多个被提示公司。
    # 先提取需要匹配的唯一键，减少重复匹配。
    mktcap_key = (
        mm[["Stkcd", "td_tm3"]]
        .drop_duplicates()
        .merge(
            mktcap[
                ["Stkcd", "td_tm3", "mktcap_tm3", "float_mktcap_tm3"]
            ],
            how="left",
            on=["Stkcd", "td_tm3"],
            validate="m:1",
        )
    )

    # 将提示公司的市值合并回每条公司对记录。
    mm = mm.merge(
        mktcap_key,
        how="left",
        on=["Stkcd", "td_tm3"],
        validate="m:1",
    )

    print("合并事件前第三个交易日市值后行数：", len(mm))
    print("提示公司总市值缺失数：", mm["mktcap_tm3"].isna().sum())
    print("提示公司总市值缺失比例：", mm["mktcap_tm3"].isna().mean())

    return mm


def merge_monthly_marketcap_breakpoints(
    mm,
    breakpoint_path,
    breakpoint_mode,
    drop_missing=True,
):
    """合并市值断点；市值模式删除必要市值缺失，沪深300模式保留。"""

    marketcap_group = pd.read_csv(
        breakpoint_path,
        encoding="utf-8",
    )

    mm["event_Trddt"] = pd.to_datetime(mm["event_Trddt"])
    marketcap_group["Trdmnt"] = pd.to_datetime(
        marketcap_group["Trdmnt"]
    )

    # 月份依据是提示事件交易日。
    event_month = mm["event_Trddt"].dt.to_period("M")

    if breakpoint_mode == "prev_month":
        # 主口径：使用事件所在月份的上月断点。
        lookup_month = event_month.sub(1)
    elif breakpoint_mode == "same_month":
        # 原文对照口径：使用事件所在月份的当月断点。
        lookup_month = event_month
    else:
        raise ValueError(
            "市值断点月份只能设置为 "
            "'prev_month'（上月）或 'same_month'（当月）"
        )

    mm["Trdmnt_for_breakpoint"] = lookup_month.dt.to_timestamp()

    marketcap_group_merge = marketcap_group.rename(
        columns={"Trdmnt": "Trdmnt_breakpoint"}
    )

    # 每条公司对记录匹配一个月份；右表每个月份应只有一行。
    mm = mm.merge(
        marketcap_group_merge,
        left_on="Trdmnt_for_breakpoint",
        right_on="Trdmnt_breakpoint",
        how="left",
        validate="m:1",
    )

    missing_month = mm["Trdmnt_breakpoint"].isna()

    print("合并月度断点后的公司对记录数：", len(mm))
    print("未匹配到断点月份的记录数：", missing_month.sum())
    print(
        mm.loc[
            missing_month,
            ["event_Trddt", "Trdmnt_for_breakpoint"],
        ].head()
    )

    # 保留原有筛选口径：下面四个字段任意一个缺失就删除。
    required_columns = [
        "mktcap_tm3",
        "marketCap_90th_percentile",
        "marketCap_50th_percentile",
        "marketCap_10th_percentile",
    ]

    print("各筛选字段的缺失数：")
    print(mm[required_columns].isna().sum())

    rows_before_filter = len(mm)
    if drop_missing:
        mm = mm.dropna(subset=required_columns).copy()
    else:
        print("沪深300模式：市值或断点缺失不影响主提示资格，保留这些公司对。")

    print(
        "因总市值或必要断点缺失删除的公司对记录数：",
        rows_before_filter - len(mm),
    )
    print("筛选后剩余公司对记录数：", len(mm))

    return mm


def filter_supply_chain_pairs(mm_key, supply_chain_path):
    """按原公告年份匹配供应链关系，排除任一方向存在关系的公司对。"""

    mm_key["yearCue"] = mm_key["Annodt"].dt.year

    SC_NetWorkRelationsIndex = pd.read_csv(
        supply_chain_path,
        encoding="utf-8",
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

    SC_NetWorkRelationsIndex["yearCue"] = (
        SC_NetWorkRelationsIndex["EndDate"].dt.year
    )
    SC_NetWorkRelationsIndex["Symbol"] = (
        SC_NetWorkRelationsIndex["Symbol"].apply(
            normalize_stock_code
        )
    )

    # 保留原筛选规则。
    # 关联方代码先按原始六位格式筛选，避免补零纳入其它市场代码。
    is_a_share_pair = (
        SC_NetWorkRelationsIndex["StateTypeCode"].eq("1")
        & SC_NetWorkRelationsIndex["PSCSymbol"].str.fullmatch(
            r"[0368][0-9]{5}", na=False
        )
        & SC_NetWorkRelationsIndex["PSCBusinessRelation"].isin(
            ["1", "2"]
        )
    )

    supply_chain = SC_NetWorkRelationsIndex.loc[
        is_a_share_pair
    ].copy()

    # 沿用原脚本的方向转换规则。
    supply_chain["upstream_stkcd"] = np.where(
        supply_chain["PSCBusinessRelation"].eq("1"),
        supply_chain["Symbol"],
        supply_chain["PSCSymbol"],
    )

    supply_chain["downstream_stkcd"] = np.where(
        supply_chain["PSCBusinessRelation"].eq("1"),
        supply_chain["PSCSymbol"],
        supply_chain["Symbol"],
    )

    # 删除自身配对，并对年度有向关系去重。
    supply_chain = supply_chain.loc[
        supply_chain["upstream_stkcd"]
        != supply_chain["downstream_stkcd"],
        ["yearCue", "upstream_stkcd", "downstream_stkcd"],
    ].drop_duplicates()

    # 任一方向有关系，就同时生成两个匹配方向。
    vertical_pairs_forward = supply_chain.rename(
        columns={
            "upstream_stkcd": "Stkcd",
            "downstream_stkcd": "StkcdRecall",
        }
    )[["yearCue", "Stkcd", "StkcdRecall"]]

    vertical_pairs_reverse = supply_chain.rename(
        columns={
            "downstream_stkcd": "Stkcd",
            "upstream_stkcd": "StkcdRecall",
        }
    )[["yearCue", "Stkcd", "StkcdRecall"]]

    vertical_pairs = (
        pd.concat(
            [vertical_pairs_forward, vertical_pairs_reverse],
            ignore_index=True,
        )
        .drop_duplicates()
        .assign(vertical_related=1)
    )

    mm_key = mm_key.merge(
        vertical_pairs,
        on=["yearCue", "Stkcd", "StkcdRecall"],
        how="left",
        validate="m:1",
    )

    is_related = mm_key["vertical_related"].eq(1)

    print("供应链过滤前公司对记录数：", len(mm_key))
    print("已识别的供应链公司对记录数：", is_related.sum())
    print("已识别的供应链公司对记录比例：", is_related.mean())
    print("双向公司对×年度查找表记录数：", len(vertical_pairs))

    # 保留未匹配到供应链关系的记录。
    mm_key = mm_key.loc[
        mm_key["vertical_related"].isna()
    ].copy()

    mm_key = mm_key.drop(
        columns=["yearCue", "vertical_related"],
        errors="ignore",
    )

    print("供应链过滤后公司对记录数：", len(mm_key))

    return mm_key


def filter_analyst_overlap_pairs(
    mm_key,
    overlap_root,
    id_mode,
    lookback_days,
    drop_overlap,
    matrix_path,
    forecast_path,
):
    """读取覆盖结果，拒绝使用早于当前记忆矩阵或分析师输入的旧结果。"""

    id_mode_folder = {
        "split": "拆开AnanmID",
        "team": "不拆开AnanmID",
    }

    if id_mode not in id_mode_folder:
        raise ValueError(
            "分析师识别方式只能是 "
            "'split'（个人）或 'team'（团队字符串）"
        )

    analyst_overlap_path = (
        Path(overlap_root)
        / id_mode_folder[id_mode]
        / f"Pattern样本_分析师共同覆盖_滚动{lookback_days}天.csv"
    )

    # 该结果只保存“有共同覆盖”的公司对，未匹配不能区分零覆盖和未计算。
    # 因此重建矩阵或分析师输入后，必须先重新生成覆盖结果，不能静默沿用旧文件。
    for input_path in [Path(matrix_path), Path(forecast_path)]:
        if analyst_overlap_path.stat().st_mtime_ns < input_path.stat().st_mtime_ns:
            raise ValueError(
                f"共同分析师覆盖结果早于输入文件：{input_path}。"
                "请先基于当前记忆矩阵重新运行分析师共同覆盖识别，再合并样本。"
            )

    analyst_overlap = pd.read_csv(
        analyst_overlap_path,
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

    analyst_overlap["Stkcd"] = analyst_overlap["Stkcd"].apply(
        normalize_stock_code
    )
    analyst_overlap["StkcdRecall"] = (
        analyst_overlap["StkcdRecall"].apply(normalize_stock_code)
    )

    # 保留原规则：同一匹配键重复时保留首条。
    analyst_overlap = analyst_overlap.drop_duplicates(
        ["Annodt", "Stkcd", "StkcdRecall"]
    )

    mm_key = mm_key.merge(
        analyst_overlap,
        on=["Annodt", "Stkcd", "StkcdRecall"],
        how="left",
        validate="m:1",
    )

    # 来源表只保存存在共同覆盖的公司对。
    # 沿用原判断：标记非缺失，视为匹配到了共同覆盖记录。
    matched_overlap = mm_key["same_analyst_overlap"].notna()

    print("共同分析师覆盖文件：", analyst_overlap_path)
    print("分析师过滤前公司对记录数：", len(mm_key))
    print("匹配到共同覆盖的记录数：", matched_overlap.sum())
    print("匹配到共同覆盖的记录比例：", matched_overlap.mean())

    if drop_overlap:
        mm_key = mm_key.loc[
            mm_key["same_analyst_overlap"].isna()
        ].copy()

        mm_key = mm_key.drop(
            columns=["same_analyst_overlap", "n_common_analysts"],
            errors="ignore",
        )
    else:
        mm_key["same_analyst_overlap"] = (
            mm_key["same_analyst_overlap"]
            .fillna(0)
            .astype("int8")
        )
        mm_key["n_common_analysts"] = (
            mm_key["n_common_analysts"]
            .fillna(0)
            .astype("int16")
        )

    print("分析师过滤后公司对记录数：", len(mm_key))

    return mm_key


def apply_pair_filters(mm, mm_key):
    """按原公告日和公司对，保留通过各轮过滤的完整记录。"""

    key_cols = ["Annodt", "Stkcd", "StkcdRecall"]

    # 在去重之前统计，才能看出原筛选键表是否有重复。
    duplicate_count = mm_key.duplicated(key_cols).sum()
    print("去重前多出的重复筛选键记录数：", duplicate_count)

    mm_key_keep = mm_key[key_cols].drop_duplicates()
    print("去重后的合格筛选键数量：", len(mm_key_keep))

    rows_before_filter = len(mm)

    # 只使用名单筛选行，完整矩阵中的变量继续保留。
    mm = mm.merge(
        mm_key_keep,
        on=key_cols,
        how="inner",
        validate="1:1",
    )

    print("应用公司对过滤前记录数：", rows_before_filter)
    print("应用公司对过滤后记录数：", len(mm))
    print(
        "三轮公司对过滤合计删除记录数：",
        rows_before_filter - len(mm),
    )

    return mm


def deduplicate_pattern_event_pairs(mm):
    """自然日筛选完成后，合并映射到同一交易日的相同公司对。"""
    keys = ["Stkcd", "event_Trddt", "StkcdRecall"]
    mm = mm.copy()
    mm["currentAccpersLag1"] = mm["currentAccpersLag1"].fillna("")
    mm["sourceAnnodts"] = mm["Annodt"].dt.strftime("%Y-%m-%d")
    duplicate_rows = mm.duplicated(keys, keep=False)
    print("同一公司对交易日上多出的自然日记录数：", mm.duplicated(keys).sum())

    if not duplicate_rows.any():
        return mm

    # 相同公司对交易日只有一份市值权重与月度断点。
    # 二元关系和机制取并集；历史共同公告公司数、财报期距离沿用上游最大值口径。
    max_cols = [
        col for col in mm.columns
        if col.startswith(("quarterLag", "lag", "nrFirmsAnnLag", "daysSinceOverlapLag"))
    ]
    union_cols = ["currentAccpersLag1", "sourceAnnodts"]
    stable_cols = [
        col for col in mm.columns
        if col not in keys + max_cols + union_cols + ["Annodt"]
    ]
    duplicate_groups = mm.loc[duplicate_rows].groupby(keys, dropna=False)
    conflicts = duplicate_groups[stable_cols].nunique(dropna=False).gt(1).any()
    conflict_cols = conflicts.index[conflicts].tolist()
    if conflict_cols:
        raise ValueError(
            "同一公司对交易日的权重或事件属性不一致，不能直接合并："
            f"{conflict_cols}"
        )

    rules = {col: "first" for col in stable_cols}
    rules.update({col: "max" for col in max_cols})
    rules.update({col: union_pipe_values for col in union_cols})
    rules["Annodt"] = "min"
    combined = duplicate_groups.agg(rules).reset_index()
    result = pd.concat([mm.loc[~duplicate_rows], combined], ignore_index=True)
    if result.duplicated(keys).any():
        raise ValueError("合并自然日来源后，公司对交易日仍然不唯一。")
    return result


def merge_current_cue_sue(mm, sue_path):
    """仅匹配真实支持滞后一期关系的报告，每家公司对内按不同报告等权。"""

    source_col = "currentAccpersLag1"
    if source_col not in mm.columns:
        raise ValueError("记忆矩阵缺少当前报告来源，请先重建记忆矩阵。")

    mm[source_col] = mm[source_col].fillna("").astype(str)
    lag1 = mm["quarterLag1Alt"].eq(1)
    if (lag1 & mm[source_col].eq("")).any():
        raise ValueError("存在滞后一期关系却没有支持报告截止期，请检查记忆矩阵。")
    if (~lag1 & mm[source_col].ne("")).any():
        raise ValueError("无滞后一期关系的记录不应保留该关系的支持报告。")

    sue = pd.read_csv(
        sue_path,
        encoding="utf-8",
        usecols=["Stkcd", "Accper", "sue1", "sue2", "sue3"],
        parse_dates=["Accper"],
    )

    sue["Stkcd"] = sue["Stkcd"].apply(normalize_stock_code)
    if sue.duplicated(["Stkcd", "Accper"]).any():
        raise ValueError("盈余惊喜表在股票代码×财报截止期层面不唯一。")

    # 相同公司及支持报告组合只匹配一次，再回填到所有被提示公司。
    source_keys = ["Stkcd", source_col]
    report_sue = mm.loc[lag1, source_keys].drop_duplicates()
    report_sue["Accper"] = report_sue[source_col].str.split("|", regex=False)
    report_sue = report_sue.explode("Accper")
    report_sue["Accper"] = pd.to_datetime(
        report_sue["Accper"], format="%Y-%m-%d", errors="raise"
    )
    report_sue = report_sue.drop_duplicates(source_keys + ["Accper"])
    report_sue = report_sue.merge(
        sue,
        on=["Stkcd", "Accper"],
        how="left",
        validate="m:1",
    )

    # 每个非缺失报告值等权；两份报告均缺失时仍然缺失。
    # 年报及一季报另保留一组值，供检查合成盈余惊喜的来源。
    sue_cols = ["sue1", "sue2", "sue3"]
    agg_specs = {col: (col, "mean") for col in sue_cols}
    agg_specs["nrCurrentReportsLag1"] = ("Accper", "nunique")
    for label, quarter in [("Annual", 4), ("Q1", 1)]:
        for col in sue_cols:
            out_col = f"{col}{label}Anchor"
            report_sue[out_col] = report_sue[col].where(
                report_sue["Accper"].dt.quarter.eq(quarter)
            )
            agg_specs[out_col] = (out_col, "mean")

    source_sue = report_sue.groupby(source_keys, as_index=False).agg(**agg_specs)
    mm = mm.merge(source_sue, on=source_keys, how="left", validate="m:1")
    mm["nrCurrentReportsLag1"] = mm["nrCurrentReportsLag1"].fillna(0).astype("int16")
    mm["lag1CueReportSources"] = [
        "|".join(f"{stock}@{date}" for date in dates.split("|") if date)
        for stock, dates in zip(mm["Stkcd"], mm[source_col])
    ]

    print("合并当前盈余惊喜后的公司对记录数：", len(mm))
    print("由多份不同报告支持的公司对记录数：", mm["nrCurrentReportsLag1"].gt(1).sum())
    print("各盈余惊喜字段缺失的公司对记录数：")
    print(mm[["sue1", "sue2", "sue3"]].isna().sum())

    return mm


def write_candidate_pairs(root: Path, mm: pd.DataFrame) -> dict:
    """保存第一阶段表和审计摘要。"""

    required_keys = ["Stkcd", "event_Trddt", "StkcdRecall"]
    root_out = output_dir(root)
    table_path = root_out / OUTPUT_NAME
    summary_path = root_out / SUMMARY_NAME
    root_out.mkdir(parents=True, exist_ok=True)
    mm.to_csv(table_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    summary = {
        "stage": 1,
        "grain": "Stkcd × event_Trddt × StkcdRecall",
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "analyst_id_mode": ANALYST_ID_MODE,
        "analyst_lookback_days": ANALYST_LOOKBACK_DAYS,
        "drop_analyst_overlap": DROP_ANALYST_OVERLAP,
        "marketcap_breakpoint_mode": MARKETCAP_BREAKPOINT_MODE,
        "rows": int(len(mm)),
        "columns": list(mm.columns),
        "key_duplicate_count": int(mm.duplicated(required_keys).sum()),
        "output_path": str(table_path),
        "next_stage": "02_构造提示事件.py",
    }
    save_summary(summary_path, summary)
    return {"table_path": table_path, "summary_path": summary_path, "rows": len(mm)}


def main() -> dict:
    root = find_root()
    paths = build_input_paths(root)

    # 1. 读取记忆矩阵，合并事件前第3个交易日市值，再限制研究期。
    mm = load_pattern_matrix_with_marketcap(
        matrix_path=paths["MemoryMatrixPattern_path"],
        daily_panel_path=paths["TRD_Dalyr_DGTW_path"],
    )
    mm = mm.loc[
        mm["event_Trddt"].dt.year.between(START_YEAR, END_YEAR)
    ].copy()

    # 2. 合并月度断点；缺失记录保留到第二阶段按口径处理。
    mm = merge_monthly_marketcap_breakpoints(
        mm=mm,
        breakpoint_path=paths["marketcap_group_path"],
        breakpoint_mode=MARKETCAP_BREAKPOINT_MODE,
        drop_missing=False,
    )

    # 3. 排除供应链和共同分析师覆盖公司对。
    mm_key = mm[["Annodt", "event_td", "Stkcd", "StkcdRecall"]].copy()
    mm_key = filter_supply_chain_pairs(
        mm_key=mm_key,
        supply_chain_path=paths["SC_NetWorkRelationsIndex_path"],
    )
    mm_key = filter_analyst_overlap_pairs(
        mm_key=mm_key,
        overlap_root=root / "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后",
        id_mode=ANALYST_ID_MODE,
        lookback_days=ANALYST_LOOKBACK_DAYS,
        drop_overlap=DROP_ANALYST_OVERLAP,
        matrix_path=paths["MemoryMatrixPattern_path"],
        forecast_path=paths["analyst_forecast_path"],
    )
    mm = apply_pair_filters(mm=mm, mm_key=mm_key)

    # 4. 合并映射到同一事件交易日的自然日来源，并匹配当前报告盈余惊喜。
    mm = deduplicate_pattern_event_pairs(mm)
    mm = merge_current_cue_sue(mm=mm, sue_path=paths["sue_path"])
    if mm.duplicated(["Stkcd", "event_Trddt", "StkcdRecall"]).any():
        raise ValueError("候选公司对基础表在公司对×事件交易日层面重复。")

    # 5. 保存统一候选表及原有审计摘要。
    result = write_candidate_pairs(root, mm)
    print(f"第一阶段完成：{result['table_path']}")
    print(f"候选公司对行数：{result['rows']:,}")
    return result


if __name__ == "__main__":
    main()
