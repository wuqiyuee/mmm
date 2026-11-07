"""构建行业前几名提示面板；只由本任务入口显式调用。

提示资格：在事件前第3个交易日，同一可比行业内总市值排名不超过给定名次。
排名参照池允许金融公司，被提示公司仍使用原非金融基础样本资格。
历史关系来自现有记忆矩阵；本模块不重建记忆矩阵、不运行回归。
公司对行业排除使用新归并行业，因此本结果不等于旧结果只改一项门槛。
未来提示标记仅使用本次最终基准样本内事件，输出末端可能缺少后续覆盖；
未匹配标记为0沿用旧程序含义，不代表全市场未来没有提示。

旧合并脚本仅作为函数来源。内部沿用其历史工作列，正式输出全部改为
IndTop4（行业前4名）或 IndTop5（行业前5名），不输出沪深300名义结果。
"""

import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_original_helpers(root):
    """只导入旧函数定义；禁写字节码，不调用旧主函数。"""
    script = Path(root) / "code/04_合并数据/合并pattern_firms的数据.py"
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("industry_top_original_helpers", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法读取原合并脚本：{script}")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers


def stop_with_audit(data, audit_dir, filename, message):
    """只在新任务目录保存需要核对的记录，不把未知值补成0。"""
    path = Path(audit_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    raise ValueError(f"{message}。核对清单：{path}")


def read_memory_matrix(matrix_path, start_year, end_year, chunk_size, helpers, audit_dir):
    """分块读取研究期事件，保留历史机制和支持报告字段。"""
    pieces = []
    start = pd.Timestamp(start_year, 1, 1)
    end = pd.Timestamp(end_year, 12, 31)
    for chunk in pd.read_csv(
        matrix_path,
        dtype={"Stkcd": "string", "StkcdRecall": "string", "currentAccpersLag1": "string"},
        parse_dates=["Annodt", "event_Trddt"],
        encoding="utf-8-sig",
        chunksize=chunk_size,
    ):
        selected = chunk.loc[chunk["event_Trddt"].between(start, end)].copy()
        lag1 = pd.to_numeric(selected["quarterLag1Alt"], errors="coerce")
        invalid_lag = ~lag1.isin([0, 1])
        if invalid_lag.any():
            stop_with_audit(selected.loc[invalid_lag], audit_dir, "一期关系标记非零一.csv", "一期记忆关系标记必须为已知的0或1")
        # 本实验所有主提示和机制只使用一期关系，历史共同公告数量已在矩阵中。
        selected = selected.loc[lag1.eq(1)].copy()
        selected["quarterLag1Alt"] = 1
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError("指定年份没有一期记忆矩阵事件，不能构建行业提示面板。")
    mm = pd.concat(pieces, ignore_index=True)
    required = {
        "Annodt", "event_Trddt", "event_td", "Stkcd", "StkcdRecall",
        "quarterLag1Alt", "currentAccpersLag1", "lag1AnnualAnchor", "lag1Q1Anchor",
    }
    missing = required - set(mm.columns)
    if missing:
        raise ValueError(f"记忆矩阵缺少必要字段：{sorted(missing)}")
    for col in ["Stkcd", "StkcdRecall"]:
        mm[col] = mm[col].apply(helpers.normalize_stock_code)
    event_td = pd.to_numeric(mm["event_td"], errors="coerce")
    invalid_td = ~np.isfinite(event_td) | event_td.mod(1).ne(0)
    if invalid_td.any():
        stop_with_audit(mm.loc[invalid_td], audit_dir, "事件交易日序号无效.csv", "事件交易日序号必须是有限整数")
    mm["event_td"] = event_td.astype("int64")
    if mm[["Annodt", "event_Trddt", "event_td"]].isna().any().any():
        raise ValueError("记忆矩阵事件日期或交易日序号缺失。")
    if mm[["Stkcd", "StkcdRecall"]].eq("").any().any():
        raise ValueError("记忆矩阵存在空股票代码。")
    if mm.duplicated(["Annodt", "Stkcd", "StkcdRecall"]).any():
        raise ValueError("记忆矩阵在自然公告日公司对层面不唯一。")
    if mm["Stkcd"].eq(mm["StkcdRecall"]).any():
        raise ValueError("记忆矩阵存在公司与自身配对。")
    return mm


def read_cache_rows(keys, cache_dir, columns):
    """按已知交易日期逐年读取缓存，只保留本次需要的公司日。"""
    keys = keys[["Stkcd", "td", "Trddt"]].drop_duplicates()
    if keys.duplicated(["Stkcd", "td"]).any():
        raise ValueError("待匹配公司日对应多个交易日期。")
    parts = []
    for year, needed in keys.groupby(keys["Trddt"].dt.year):
        path = Path(cache_dir) / f"{int(year)}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"缺少已完成的年度缓存：{path}")
        data = pd.read_parquet(path, columns=columns)
        data["Stkcd"] = data["Stkcd"].astype("string").str.zfill(6)
        if data.duplicated(["Stkcd", "td"]).any():
            raise ValueError(f"年度缓存公司日键重复：{path}")
        matched = needed.merge(data, on=["Stkcd", "td", "Trddt"], how="left", validate="1:1", indicator=True)
        parts.append(matched)
        del data
    return pd.concat(parts, ignore_index=True)


def merge_rank_information(mm, calendar, rank_cache, audit_dir):
    """事件前第3个交易日排名与权重来自含金融公司的同一数据源。"""
    event_dates = mm[["event_td", "event_Trddt"]].drop_duplicates()
    checked = event_dates.merge(
        calendar.rename(columns={"td": "event_td", "Trddt": "calendar_date"}),
        on="event_td", how="left", validate="1:1",
    )
    bad_dates = checked.loc[checked["event_Trddt"].ne(checked["calendar_date"])]
    if not bad_dates.empty:
        stop_with_audit(bad_dates, audit_dir, "事件交易日与日历不一致.csv", "事件交易日与统一日历不一致")
    mm["td_tm3"] = mm["event_td"] - 3
    wanted = mm[["Stkcd", "td_tm3"]].drop_duplicates().rename(columns={"td_tm3": "td"})
    wanted = wanted.merge(calendar, on="td", how="left", validate="m:1")
    if wanted["Trddt"].isna().any():
        stop_with_audit(wanted.loc[wanted["Trddt"].isna()], audit_dir, "排名日期缺少交易日历.csv", "排名日不在统一交易日历中")
    ranked = read_cache_rows(
        wanted, rank_cache["rank_dir"],
        ["Stkcd", "td", "Trddt", "size", "sizew", "IndustryCode", "industryGroup", "industryStatus", "industrySourceStandard", "industryRank", "industryRankMax", "industryN", "rankingEligible"],
    )
    missing = ranked["_merge"].ne("both")
    unknown = ~ranked["rankingEligible"].isin([0, 1])
    eligible = ranked["rankingEligible"].eq(1)
    # 行业待定可以暂时没有名次；完成公司对筛选后才检查它是否影响实际提示。
    # industryN 只计已确定行业的公司；最差可能名次可超过这个数字。
    rank = ranked["industryRank"]
    rank_max = ranked["industryRankMax"]
    known_industry = ranked["industryStatus"].eq("valid").fillna(False)
    incomplete_bounds = rank.isna() ^ rank_max.isna()
    invalid_bounds = rank.notna() & (
        rank.lt(1) | rank_max.lt(rank)
        | (known_industry & (ranked["industryN"].isna() | rank.gt(ranked["industryN"])))
    )
    invalid_rank = eligible & (
        incomplete_bounds | invalid_bounds.fillna(False)
        | ranked["size"].isna() | ranked["size"].le(0)
    )
    bad = missing | unknown | invalid_rank.fillna(False)
    if bad.any():
        stop_with_audit(ranked.loc[bad], audit_dir, "提示公司排名缺失或无效.csv", "提示公司排名数据不完整，不能将未知提示资格补成0")
    ranked = ranked.drop(columns="_merge").rename(columns={
        "td": "td_tm3", "Trddt": "rankingDate", "size": "mktcap_tm3",
        "sizew": "float_mktcap_tm3", "IndustryCode": "rankingIndustryCode",
        "industryGroup": "rankingIndustryGroup", "industryStatus": "rankingIndustryStatus",
        "industrySourceStandard": "rankingIndustrySourceStandard",
    })
    # 已知基础资格为0的记录允许排名缺失，但其主提示资格必须为0。
    mm = mm.merge(ranked, on=["Stkcd", "td_tm3"], how="left", validate="m:1")
    return mm


def filter_recalled_event_eligibility(mm, base_sample_path, chunk_size, helpers, audit_dir):
    """先按原基础索引筛被提示方事件日，避免核对最终不可能使用的行业。"""
    base_sample_path = Path(base_sample_path)
    if not base_sample_path.is_file():
        raise FileNotFoundError(f"缺少原被提示公司基础资格索引：{base_sample_path}")
    wanted = mm[["StkcdRecall", "event_td", "event_Trddt"]].drop_duplicates().rename(
        columns={"StkcdRecall": "Stkcd", "event_td": "td", "event_Trddt": "expected_date"}
    )
    if wanted.duplicated(["Stkcd", "td"]).any():
        raise ValueError("被提示公司同一事件交易日序号对应多个日期。")
    pieces = []
    source_rows = 0
    for chunk in pd.read_csv(
        base_sample_path, usecols=["Stkcd", "td", "Trddt"],
        dtype={"Stkcd": "string"}, chunksize=chunk_size, encoding="utf-8-sig",
    ):
        source_rows += len(chunk)
        chunk["Stkcd"] = chunk["Stkcd"].apply(helpers.normalize_stock_code)
        chunk["Trddt"] = pd.to_datetime(chunk["Trddt"], errors="coerce")
        td = pd.to_numeric(chunk["td"], errors="coerce")
        invalid = chunk["Stkcd"].eq("") | chunk["Trddt"].isna() | ~np.isfinite(td) | td.mod(1).ne(0)
        if invalid.any():
            stop_with_audit(chunk.loc[invalid], audit_dir, "被提示公司基础索引键无效.csv", "原基础资格索引含缺失或无效键，不能把它解释为不具备资格")
        chunk["td"] = td.astype("int64")
        # 先按代码和序号匹配，再核对日期；防止日期冲突被三键连接静默丢掉。
        matched = chunk.merge(wanted, on=["Stkcd", "td"], how="inner", validate="m:1")
        bad_date = matched["Trddt"].ne(matched["expected_date"])
        if bad_date.any():
            stop_with_audit(matched.loc[bad_date], audit_dir, "被提示公司基础资格日期冲突.csv", "原基础索引日期与待匹配事件日期不一致")
        if not matched.empty:
            pieces.append(matched[["Stkcd", "td", "Trddt"]])
    if source_rows == 0:
        raise ValueError("原被提示公司基础资格索引为空，不能把全部事件判为不合格。")
    matched_keys = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["Stkcd", "td", "Trddt"])
    duplicate = matched_keys.duplicated(["Stkcd", "td"], keep=False)
    if duplicate.any():
        stop_with_audit(matched_keys.loc[duplicate], audit_dir, "被提示公司基础资格键冲突.csv", "原基础索引在本次事件键上存在重复或冲突记录")
    before = len(mm)
    keep = matched_keys.rename(columns={"Stkcd": "StkcdRecall", "td": "event_td", "Trddt": "event_Trddt"})
    filtered = mm.iloc[0:0].copy() if keep.empty else mm.merge(
        keep, on=["StkcdRecall", "event_td", "event_Trddt"], how="inner", validate="m:1",
    )
    counts = {
        "source_path": str(base_sample_path),
        "pair_rows_before": before, "pair_rows_kept": len(filtered),
        "pair_rows_removed": before - len(filtered),
        "requested_recalled_firm_days": len(wanted),
        "qualified_recalled_firm_days": len(matched_keys),
        "unqualified_recalled_firm_days": len(wanted) - len(matched_keys),
    }
    counts_path = Path(audit_dir) / "被提示公司基础资格匹配统计.json"
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    counts_path.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"按被提示方原基础资格保留公司对{len(filtered):,}条，删除{before - len(filtered):,}条。", flush=True)
    return filtered, counts


def filter_event_industries(mm, rank_cache, audit_dir):
    """事件日可确定同业时排除，可确定异业时保留，关系不明时核对。"""
    cue_keys = mm[["Stkcd", "event_td", "event_Trddt"]].rename(columns={"event_td": "td", "event_Trddt": "Trddt"})
    recall_keys = mm[["StkcdRecall", "event_td", "event_Trddt"]].rename(columns={"StkcdRecall": "Stkcd", "event_td": "td", "event_Trddt": "Trddt"})
    wanted = pd.concat([cue_keys, recall_keys], ignore_index=True).drop_duplicates()
    industry = read_cache_rows(
        wanted, rank_cache["industry_dir"],
        ["Stkcd", "td", "Trddt", "IndustryCode", "industryGroup", "industryStatus", "industrySourceStandard"],
    )
    valid = industry["industryStatus"].eq("valid") & industry["industryGroup"].notna()
    coarse = (
        industry["industryStatus"].eq("unresolved_coarse")
        & industry["IndustryCode"].str.fullmatch(r"C[0-9]", na=False)
    )
    # 粗码能限定可能的行业分支；其他不明代码没有可可靠判断的分支。
    industry["industryBranch"] = industry["industryGroup"].str[:2].where(valid)
    industry.loc[coarse, "industryBranch"] = industry.loc[coarse, "IndustryCode"]
    bad = industry["_merge"].ne("both") | (industry["industryStatus"].eq("valid") & ~valid)
    if bad.any():
        stop_with_audit(industry.loc[bad], audit_dir, "事件日公司对行业待核对.csv", "事件日行业记录缺失或已标记有效却没有行业大类")
    industry = industry.drop(columns=["_merge", "Trddt"])
    cue_industry = industry.rename(columns={"td": "event_td", "IndustryCode": "cueIndustryCode", "industryGroup": "cueIndustryGroup", "industryStatus": "cueIndustryStatus", "industrySourceStandard": "cueIndustrySourceStandard", "industryBranch": "cueIndustryBranch"})
    recall_industry = industry.rename(columns={"Stkcd": "StkcdRecall", "td": "event_td", "IndustryCode": "recallIndustryCode", "industryGroup": "recallIndustryGroup", "industryStatus": "recallIndustryStatus", "industrySourceStandard": "recallIndustrySourceStandard", "industryBranch": "recallIndustryBranch"})
    mm = mm.merge(cue_industry, on=["Stkcd", "event_td"], how="left", validate="m:1")
    mm = mm.merge(recall_industry, on=["StkcdRecall", "event_td"], how="left", validate="m:1")
    both_valid = mm["cueIndustryStatus"].eq("valid") & mm["recallIndustryStatus"].eq("valid")
    different_branch = (
        mm["cueIndustryBranch"].notna() & mm["recallIndustryBranch"].notna()
        & mm["cueIndustryBranch"].ne(mm["recallIndustryBranch"])
    ).fillna(False)
    uncertain = (~both_valid & ~different_branch).fillna(True)
    if uncertain.any():
        stop_with_audit(mm.loc[uncertain], audit_dir, "事件日公司对行业待核对.csv", "事件日公司对可能同业也可能异业，无法确定是否应排除")
    same_industry = (both_valid & mm["cueIndustryGroup"].eq(mm["recallIndustryGroup"])).fillna(False)
    print(f"归并行业相同的公司对：{int(same_industry.sum()):,}；过滤前：{len(mm):,}。", flush=True)
    return mm.loc[~same_industry].copy()


def check_cue_rank_bounds(mm, top_ns, audit_dir):
    """仅对最终需要的线索检查名次上下界是否跨越任一前N门槛。"""
    eligible = mm["rankingEligible"].eq(1)
    unknown = mm["industryRank"].isna() | mm["industryRankMax"].isna()
    for top_n in top_ns:
        unknown |= mm["industryRank"].le(top_n) & mm["industryRankMax"].gt(top_n)
    unknown = eligible & unknown.fillna(True)
    if unknown.any():
        columns = [
            "Stkcd", "StkcdRecall", "event_Trddt", "rankingDate", "rankingIndustryCode",
            "rankingIndustryGroup", "rankingIndustryStatus",
            "industryRank", "industryRankMax", "industryN", "mktcap_tm3",
        ]
        stop_with_audit(mm.loc[unknown, columns].drop_duplicates(), audit_dir, "提示公司行业名次待核对.csv", "实际线索的行业排名不确定，可能改变前四或前五资格")


def add_industry_cue_flags(mm, top_n):
    """主资格用行业排名，真实规模组只作辅助；不叠加原90分位门槛。"""
    eligible = mm["rankingEligible"].eq(1)
    mm["mainCueEligible"] = (
        eligible & mm["industryRank"].le(top_n).fillna(False)
    ).astype("int8")
    mm["size90plus"] = (eligible & (mm["mktcap_tm3"] > mm["marketCap_90th_percentile"])).astype("int8")
    mm["size50_90"] = (eligible & (mm["mktcap_tm3"] > mm["marketCap_50th_percentile"]) & (mm["mktcap_tm3"] <= mm["marketCap_90th_percentile"])).astype("int8")
    mm["size10_50"] = (eligible & (mm["mktcap_tm3"] >= mm["marketCap_10th_percentile"]) & (mm["mktcap_tm3"] <= mm["marketCap_50th_percentile"])).astype("int8")
    mm["size10minus"] = (eligible & (mm["mktcap_tm3"] < mm["marketCap_10th_percentile"])).astype("int8")
    # size90plus（旧90分位后缀）仅在以下季度/机制工作列中代表本次主组。
    # 独立的size90plus（真实市值分组）仍保留原含义，不进入正式公司对输出。
    for suffix, eligible_col in [
        ("size90plus", "mainCueEligible"), ("size50_90", "size50_90"),
        ("size10_50", "size10_50"), ("size10minus", "size10minus"),
    ]:
        mm[f"quarterLag1{suffix}"] = (mm["quarterLag1Alt"] * mm[eligible_col]).astype("int8")
    return mm


def format_industry_output(data, top_n):
    """保存前将主提示、全部机制和加权变量统一改为行业排名名称。"""
    tag = f"IndTop{top_n}"
    renamed = {col: col.replace("size90plus", tag).replace("S90p", tag) for col in data.columns}
    if len(set(renamed.values())) != len(renamed):
        raise ValueError("行业提示变量重命名后出现重复列。")
    output = data.rename(columns=renamed, copy=False)
    if any("HS300" in col or "S90p" in col or "size90plus" in col for col in output.columns):
        raise ValueError("正式输出残留其他主提示口径名称。")
    output["cueDefinition"] = "industry_top"
    output["industryTopN"] = top_n
    return output


def make_cue_events(mm, top_n, helpers):
    """复用历史机制、盈余惊喜和干扰聚合，返回未改名的内部事件表。"""
    mm = add_industry_cue_flags(mm, top_n)
    mm = helpers.add_strong_memory_cues(mm, top_mover_mode="any")
    mm = helpers.add_sue_cue_inputs(mm)
    mm = helpers.add_marketcap_and_interference_inputs(mm)
    mm, cue_keys = helpers.prepare_pattern_cue_pairs(mm)
    pair_columns = [
        "Stkcd", "event_Trddt", "event_td", "StkcdRecall", "sourceAnnodts",
        "rankingDate", "td_tm3", "rankingIndustryCode", "rankingIndustryGroup", "rankingIndustryStatus", "rankingIndustrySourceStandard",
        "industryRank", "industryRankMax", "industryN", "rankingEligible",
        "mktcap_tm3", "float_mktcap_tm3", "cueIndustryCode", "cueIndustryGroup",
        "recallIndustryCode", "recallIndustryGroup", "cueIndustrySourceStandard", "recallIndustrySourceStandard", "quarterLag1Alt", "quarterLag1size90plus",
        "currentAccpersLag1", "lag1AnnualAnchor", "lag1Q1Anchor",
    ]
    pair_details = mm.loc[mm["quarterLag1size90plus"].eq(1), pair_columns].copy()
    pair_details = pair_details.rename(columns={"event_Trddt": "Trddt", "event_td": "td"})
    collapsed = helpers.collapse_pattern_cues(mm, denominator_mode="nonmissing_sue")
    drop_cols = ["event_Trddt"] + [col for col in collapsed.columns if col.startswith(("sumSue", "sumMarketCap", "sumNrFirmsAnn"))]
    events = collapsed.drop(columns=drop_cols, errors="ignore").copy()
    return events, cue_keys, pair_details


def load_recalled_panel(paths, base_sample_path, start_year, end_year, chunk_size, helpers):
    """先用周边日期生成自身公告窗口，再保留原非金融基础资格。"""
    daily = helpers.load_pattern_daily_panel(paths["TRD_Dalyr_DGTW_path"])
    daily = helpers.add_own_earnings_announcement_flag(daily, paths["IAR_Rept_path"], event_date_mode="next_if_nontrading")
    daily = helpers.add_own_ea_windows(daily, window=10)
    start = pd.Timestamp(start_year, 1, 1)
    end = pd.Timestamp(end_year, 12, 31)
    daily = daily.loc[daily["Trddt"].between(start, end)].copy()
    # 原宽面板恢复过不合格日期索引。明确用原基础样本键保持被提示方资格。
    base_parts = []
    for chunk in pd.read_csv(base_sample_path, usecols=["Stkcd", "td", "Trddt"], dtype={"Stkcd": "string"}, parse_dates=["Trddt"], chunksize=chunk_size):
        selected = chunk.loc[chunk["Trddt"].between(start, end), ["Stkcd", "td", "Trddt"]].copy()
        if not selected.empty:
            base_parts.append(selected)
    if not base_parts:
        raise ValueError("指定年份没有原非金融基础样本记录。")
    base = pd.concat(base_parts, ignore_index=True).rename(columns={"Stkcd": "StkcdRecall"})
    base["StkcdRecall"] = base["StkcdRecall"].apply(helpers.normalize_stock_code)
    if base.duplicated(["StkcdRecall", "td"]).any():
        raise ValueError("原基础样本索引的公司日键重复。")
    daily = daily.merge(base, on=["StkcdRecall", "td", "Trddt"], how="inner", validate="1:1")
    daily = helpers.merge_own_event_controls(daily, paths["performance_forecast_controls_path"], paths["other_event_controls_path"])
    daily = helpers.merge_news_controls(daily, paths["AES_AEV_controls_path"])
    return daily


def finish_regression_panel(daily, events, event_path, paths, cue_keys, helpers):
    """只构造与筛选回归面板，不估计回归。"""
    # 直接传内存事件表，避免旧函数从盘读时只接受旧口径标记。
    daily = helpers.merge_pattern_cue_events(daily, event_path, cue_events=events)
    daily = helpers.fill_pattern_cue_indicators(daily)
    daily = helpers.merge_adjusted_returns(daily, paths["adjusted_return_path"])
    daily = helpers.add_interference_groups(daily)
    daily = helpers.add_sue_cue_quantiles(daily)
    daily = helpers.add_recalled_firm_size_group(daily)
    winsor_limits, winsor_source_cols = helpers.compute_pattern_winsor_limits(daily, p=0.01)
    daily = helpers.prepare_pattern_baseline_inputs(daily, winsor_limits)
    daily = helpers.filter_pattern_baseline_sample(daily)
    if daily.empty:
        raise ValueError("原基准筛样后没有观测，未保存空回归面板。")
    daily = daily.sort_values(["StkcdRecall", "Trddt"]).reset_index(drop=True)
    reach, reach_summary = helpers.summarize_pattern_cue_reach(daily, cue_keys)
    daily = helpers.add_remaining_pattern_winsor_cols(daily, winsor_source_cols, winsor_limits)
    daily = helpers.add_future_pattern_cue_flags(daily)
    daily = helpers.clean_pattern_final_columns(daily)
    front = ["StkcdRecall", "Trddt"]
    daily = daily[front + [col for col in daily.columns if col not in front]]
    return daily, reach, reach_summary


def build_panels(root, rank_cache, output_dir, start_year, end_year, top_ns=(4, 5), chunk_size=400000):
    """生成独立行业排名提示事件、公司对和最终回归面板，返回保存路径。"""
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    allowed_output = root / "中间结果/04_合并数据/行业排名提示检验"
    allowed_work = root / "work/行业排名提示检验"
    if not any(output_dir == allowed or allowed in output_dir.parents for allowed in [allowed_output, allowed_work]):
        raise ValueError("输出目录必须位于本任务独立结果目录或本任务工作目录内。")
    if start_year > end_year or chunk_size <= 0:
        raise ValueError("年份顺序或读取分块大小无效。")
    top_ns = tuple(dict.fromkeys(top_ns))
    if not top_ns or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in top_ns):
        raise ValueError("行业名次必须是正整数。")
    if int(rank_cache["start_year"]) != start_year or int(rank_cache["end_year"]) != end_year:
        raise ValueError("排名缓存覆盖参数与面板年份不一致。")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = allowed_work / "面板核对清单"
    helpers = load_original_helpers(root)
    paths = helpers.build_pattern_paths(root, cue_definition="size90")
    # 只读取原输入路径；不使用旧函数提供的任何输出路径。
    base_sample_path = root / "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv"
    calendar = pd.read_csv(paths["TRD_Cale_td_path"], usecols=["td", "Trddt"], parse_dates=["Trddt"])
    if calendar["td"].isna().any() or calendar["Trddt"].isna().any() or calendar["td"].duplicated().any() or calendar["Trddt"].duplicated().any():
        raise ValueError("统一交易日历存在缺失或重复键。")

    print("读取原记忆矩阵，匹配事件前第3个交易日行业排名。", flush=True)
    mm = read_memory_matrix(paths["MemoryMatrixPattern_path"], start_year, end_year, chunk_size, helpers, audit_dir)
    matrix_rows = len(mm)
    mm = merge_rank_information(mm, calendar, rank_cache, audit_dir)
    # 资格明确为0的公司对不能触发主提示或辅助规模提示；日度左连接会恢复0值。
    # 保留所有合格排名，不只保留前4/5名，以便按原函数构造辅助规模组。
    ineligible_pair_rows = int(mm["rankingEligible"].eq(0).sum())
    mm = mm.loc[mm["rankingEligible"].eq(1)].copy()
    print(f"删除排名基础资格明确不合格的公司对：{ineligible_pair_rows:,}。", flush=True)
    if mm.empty:
        raise ValueError("研究期一期公司对的提示方均不具备排名基础资格。")
    # 与后半段主面板使用同一份基础资格索引，提前删掉最终不会进入面板的事件。
    # 行业未知不会触发本筛选；合格被提示方仍进入后续严格行业核对。
    mm, recalled_eligibility_counts = filter_recalled_event_eligibility(
        mm, base_sample_path, chunk_size, helpers, audit_dir,
    )
    if mm.empty:
        raise ValueError("一期公司对中没有事件日具备原基础资格的被提示公司。")
    mm_key = mm[["Annodt", "event_td", "Stkcd", "StkcdRecall"]].copy()
    mm_key = helpers.filter_supply_chain_pairs(mm_key, paths["SC_NetWorkRelationsIndex_path"])
    # 此原函数只读缓存，并核对缓存不早于矩阵/分析师源；缺失或过期即停止。
    # 不调用分析师识别脚本，不创建或更新旧分析师目录。
    mm_key = helpers.filter_analyst_overlap_pairs(
        mm_key,
        overlap_root=root / "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后",
        id_mode="split", lookback_days=365, drop_overlap=True,
        matrix_path=paths["MemoryMatrixPattern_path"], forecast_path=paths["analyst_forecast_path"],
    )
    mm = helpers.apply_pair_filters(mm, mm_key)
    del mm_key
    if mm.empty:
        raise ValueError("公司对筛选后没有记录，未生成行业提示面板。")
    # 独立于行业的公司对筛选先执行，避免为最终不会使用的记录误停。
    mm = filter_event_industries(mm, rank_cache, audit_dir)
    if mm.empty:
        raise ValueError("行业筛选后没有公司对，未生成行业提示面板。")
    check_cue_rank_bounds(mm, top_ns, audit_dir)
    mm = helpers.merge_monthly_marketcap_breakpoints(mm, paths["marketcap_group_path"], breakpoint_mode="prev_month", drop_missing=False)
    mm = helpers.deduplicate_pattern_event_pairs(mm)
    mm = helpers.merge_current_cue_sue(mm, paths["sue_path"])

    results = {}
    reference_keys = None
    for top_n in top_ns:
        print(f"开始构建行业前{top_n}名提示面板。", flush=True)
        folder = output_dir / f"行业前{top_n}名"
        folder.mkdir(parents=True, exist_ok=True)
        event_path = folder / f"行业前{top_n}名_提示事件.csv"
        pair_path = folder / f"行业前{top_n}名_合格提示公司对.csv"
        panel_path = folder / f"行业前{top_n}名_最终回归面板.csv"
        events, cue_keys, pair_details = make_cue_events(mm.copy(), top_n, helpers)
        daily = load_recalled_panel(paths, base_sample_path, start_year, end_year, chunk_size, helpers)
        daily, reach, reach_summary = finish_regression_panel(daily, events, event_path, paths, cue_keys, helpers)
        final_keys = pd.MultiIndex.from_frame(daily[["StkcdRecall", "Trddt"]])
        if not final_keys.is_unique:
            raise ValueError("最终回归面板的被提示公司与交易日键重复。")
        if reference_keys is None:
            reference_keys = final_keys
        elif not reference_keys.equals(final_keys):
            only_first = reference_keys.difference(final_keys).to_frame(index=False).assign(出现位置="第一套面板独有")
            only_current = final_keys.difference(reference_keys).to_frame(index=False).assign(出现位置=f"行业前{top_n}名面板独有")
            stop_with_audit(pd.concat([only_first, only_current], ignore_index=True), audit_dir, "不同名次最终样本键不一致.csv", "不同名次口径的最终被提示公司日样本不一致")
        in_sample = daily[["StkcdRecall", "Trddt"]].assign(in_main_sample=1)
        pair_details = pair_details.merge(in_sample, on=["StkcdRecall", "Trddt"], how="left", validate="m:1")
        pair_details["in_main_sample"] = pair_details["in_main_sample"].fillna(0).astype("int8")
        for table, path in [(events, event_path), (pair_details, pair_path), (daily, panel_path)]:
            format_industry_output(table, top_n).to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        reach_path = folder / f"行业前{top_n}名_提示范围.csv"
        reach.reset_index().to_csv(reach_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        summary = {
            "cue_definition": "industry_top", "industry_top_n": top_n,
            "start_year": start_year, "end_year": end_year,
            "ranking_rule": "事件前第3个交易日，同日归并行业内总市值降序排名",
            "ranking_bounds": "industryRank为已确定同行下的最佳可能名次，industryRankMax包含可能同业竞争者后的最差可能名次；industryN只计已确定行业的公司，不一定是全部同行数",
            "ineligible_cue_rule": "排名基础资格已知为0的公司对不触发主提示或辅助规模提示，聚合前删除，日度左连接恢复无提示0值",
            "pair_filter_rule": "事件日归并行业排除，并排除原一级供应链和共同分析师关系",
            "recalled_sample_rule": "原非金融基础样本资格，再沿用原短窗口/长窗口基准样本筛选",
            "future_cue_rule": "只以本次最终基准样本内事件构造未来提示；输出末端可能缺少后续覆盖，未匹配补0不代表全市场未来没有提示",
            "comparison_note": "同时采用归并行业排除与含金融的提示公司市值来源，不是旧结果仅更换提示门槛",
            "lag1_matrix_rows_in_period": matrix_rows, "pair_rows_after_filters": len(mm),
            "known_ineligible_pair_rows_removed": ineligible_pair_rows,
            "recalled_event_eligibility_filter": recalled_eligibility_counts,
            "eligible_cue_pairs": len(pair_details), "panel_rows": len(daily),
            "panel_firms": int(daily["StkcdRecall"].nunique()),
            "cue_firm_days": int(daily["quarterLag1size90plus"].sum()),
            "event_path": event_path, "pair_path": pair_path, "panel_path": panel_path,
            "reach_path": reach_path, "reach_summary": reach_summary.to_dict(),
            "rank_manifest_path": Path(rank_cache["manifest_path"]),
        }
        summary_path = folder / "构造口径与核对摘要.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        summary["summary_path"] = summary_path
        results[f"IndTop{top_n}"] = summary
        del daily, events, cue_keys, pair_details, reach, reach_summary, in_sample
        gc.collect()
        print(f"行业前{top_n}名面板已保存：{panel_path}", flush=True)
    return {"output_dir": output_dir, "panels": results}
