"""第二阶段：独立构造四种口径的提示事件。

输入第一阶段的公司对明细，输出被提示公司×事件交易日的提示事件和合格公司对。
市值90分位、沪深300、行业前4名和行业前5名使用相同聚合、盈余惊喜分母和机制算法。
所需计算实现均在本文件；仅导入标准库和数据处理库，不加载任何项目脚本。
行业口径只读取正式日度资格面板，核对完成摘要后匹配资格、市值和事件日行业。
内部主提示字段保留历史名称，实际口径由cueDefinition（提示口径）记录。
直接执行时才读取数据与写出结果；导入本文件不会启动计算。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TOP_NS = (2, 4, 5, 10)
INDUSTRY_CUE_POLICIES = ("unclean", "clean")
SUE_DENOMINATOR_MODE = "nonmissing_sue"
TOP_MOVER_MODE = "any"
START_YEAR = 2005
END_YEAR = 2025
STAGE_ROOT_NAME = "三阶段合并"
INPUT_NAME = "候选公司对基础表.csv"


def find_root() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "AGENTS.md").is_file() and (folder / "中间结果").is_dir():
            return folder
    raise FileNotFoundError("找不到包含AGENTS.md和中间结果的项目根目录。")


def build_input_paths(root: Path) -> dict:
    """本阶段的正式输入路径，不调用其他项目脚本。"""
    return {
        "TRD_Cale_td_path": root / "中间结果/01_清洗后样本/市场交易日历_td.csv",
        "hs300_membership_path": root / "中间结果/02_变量构造结果/9指数成分股/沪深300/沪深300_历史成分区间.csv",
        "industry_path": root / "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv",
        "industry_rank_path": root / "中间结果/02_变量构造结果/行业提示资格/日度行业市值排名面板.csv",
        "eligibility_path": root / "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv",
    }


def stage_root(root: Path) -> Path:
    return root / "中间结果/04_合并数据" / STAGE_ROOT_NAME


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


def add_sue_denominator_inputs(data):
    """为三种 SUE 构造各自的有效 cue 数量和有效市值分母。"""
    for sue_id in [1, 2, 3]:
        valid_cue = (
            data["quarterLag1size90plus"].eq(1)
            & data[f"sue{sue_id}"].notna()
        )
        data[f"validSue{sue_id}CueLag1S90p"] = valid_cue.astype("int8")
        data[f"validSue{sue_id}CueMarketCapLag1S90p"] = np.where(
            valid_cue,
            data["mktcap_tm3"],
            0.0,
        )
    return data


def sue_denominator_agg_specs():
    """返回三种 SUE 的非缺失 cue 数量和市值加总规则。"""
    specs = {}
    for sue_id in [1, 2, 3]:
        specs[f"sumSue{sue_id}ValidCueCountLag1S90p"] = (
            f"validSue{sue_id}CueLag1S90p",
            "sum",
        )
        specs[f"sumSue{sue_id}ValidCueMarketCapLag1S90p"] = (
            f"validSue{sue_id}CueMarketCapLag1S90p",
            "sum",
        )
    return specs


def calculate_sue_cue_averages(collapsed, denominator_mode):
    """按指定分母口径计算三种 SUE 的等权和市值加权平均。"""
    valid_modes = {"nonmissing_sue", "all_cues"}
    if denominator_mode not in valid_modes:
        raise ValueError(
            "SUE_DENOMINATOR_MODE 只能是 "
            f"{valid_modes}，当前是 {denominator_mode}"
        )

    for sue_id in [1, 2, 3]:
        if denominator_mode == "nonmissing_sue":
            ew_denominator = collapsed[
                f"sumSue{sue_id}ValidCueCountLag1S90p"
            ]
            vw_denominator = collapsed[
                f"sumSue{sue_id}ValidCueMarketCapLag1S90p"
            ]
        else:
            ew_denominator = collapsed["nrCuesLag1S90p"]
            vw_denominator = collapsed["sumMarketCap_Minus3Lag1S90p"]

        collapsed[f"sue{sue_id}CueLag1S90pEW"] = (
            collapsed[f"sumSue{sue_id}Lag1S90p"]
            .div(ew_denominator)
            .where(ew_denominator.ne(0), np.nan)
        )
        collapsed[f"sue{sue_id}CueLag1S90pVW"] = (
            collapsed[f"sumSue{sue_id}Lag1XmarketCapS90p"]
            .div(vw_denominator)
            .where(vw_denominator.ne(0), np.nan)
        )

    return collapsed


def merge_hs300_membership(mm, membership_path):
    """匹配历史沪深300资格，返回矩阵及可识别成员资格的起止日期。"""
    membership_path = Path(membership_path)
    if not membership_path.is_file():
        raise FileNotFoundError(
            f"找不到沪深300历史成分区间文件：{membership_path}。"
            "请在主函数的 hs300_membership_path（历史成分文件路径）参数中指定现有文件。"
        )

    members = pd.read_csv(membership_path, dtype={"Stkcd": "string"})
    required = {"Stkcd", "start_date", "end_date", "CoverageEndDate"}
    missing = required - set(members.columns)
    if missing:
        raise ValueError(f"沪深300历史成分文件缺少字段：{missing}")
    if members.empty:
        raise ValueError("沪深300历史成分文件为空，不能判断成员资格。")

    members["Stkcd"] = members["Stkcd"].apply(normalize_stock_code).astype("string")
    members["start_date"] = pd.to_datetime(members["start_date"], errors="raise")
    members["end_date"] = pd.to_datetime(members["end_date"], errors="raise")
    if members["Stkcd"].eq("").any() or members["start_date"].isna().any():
        raise ValueError("沪深300历史成分文件存在缺失证券代码或区间起始日。")
    if "Indexcd" in members.columns:
        index_codes = members["Indexcd"].apply(normalize_stock_code)
        if not index_codes.eq("000300").all():
            raise ValueError("历史成分文件包含非沪深300指数的记录。")

    # 结束日缺失表示截至文件覆盖末日仍在成分股中。
    coverage = pd.to_datetime(members["CoverageEndDate"], errors="raise")
    if coverage.isna().any() or coverage.nunique() != 1:
        raise ValueError("历史成分文件的覆盖截止日缺失或不一致。")
    coverage_start = members["start_date"].min()
    coverage_end = coverage.iloc[0]
    if coverage_end < coverage_start:
        raise ValueError("历史成分文件的覆盖截止日早于起始日。")
    open_end = coverage_end + pd.Timedelta(days=1)
    members["end_date"] = members["end_date"].fillna(open_end)
    members = members.sort_values(["Stkcd", "start_date"])
    previous_end = members.groupby("Stkcd")["end_date"].shift()
    if (
        members["end_date"].le(members["start_date"]).any()
        or members["start_date"].lt(previous_end).any()
    ):
        raise ValueError("沪深300历史成分区间长度无效，或同一公司的区间相互重叠。")

    if mm["event_Trddt"].isna().any():
        raise ValueError("记忆矩阵存在缺失事件交易日，不能判断沪深300资格。")
    in_coverage = mm["event_Trddt"].between(coverage_start, coverage_end)
    print("沪深300事件构造排除覆盖期外的公司对记录数：", (~in_coverage).sum())
    mm = mm.loc[in_coverage].copy()
    events = mm[["Stkcd", "event_Trddt"]].drop_duplicates().copy()
    matches = pd.merge_asof(
        events.sort_values("event_Trddt"),
        members[["Stkcd", "start_date", "end_date"]].sort_values("start_date"),
        by="Stkcd",
        left_on="event_Trddt",
        right_on="start_date",
        direction="backward",
        allow_exact_matches=True,
    )
    matches["isHS300"] = (
        matches["start_date"].notna()
        & matches["event_Trddt"].lt(matches["end_date"])
    ).astype("int8")
    mm = mm.merge(
        matches[["Stkcd", "event_Trddt", "isHS300"]],
        on=["Stkcd", "event_Trddt"],
        how="left",
        validate="m:1",
    )
    print("历史沪深300成员资格成立的公司对记录数：", mm["isHS300"].sum())
    return mm, coverage_start, coverage_end


def filter_same_industry_pairs(mm, industry_path):
    """匹配事件日双方行业，删除行业均已知且相同的公司对记录。"""

    mm_key = mm[
        ["Annodt", "event_td", "Stkcd", "StkcdRecall"]
    ].copy()

    mm_key["Stkcd"] = mm_key["Stkcd"].apply(normalize_stock_code)
    mm_key["StkcdRecall"] = mm_key["StkcdRecall"].apply(
        normalize_stock_code
    )

    daily_industry = pd.read_csv(
        industry_path,
        encoding="utf-8",
        usecols=["Stkcd", "td", "IndustryCode"],
        dtype={
            "Stkcd": "string",
            "IndustryCode": "string",
        },
    )

    daily_industry["Stkcd"] = daily_industry["Stkcd"].apply(
        normalize_stock_code
    )
    daily_industry["td"] = pd.to_numeric(
        daily_industry["td"], errors="coerce"
    )

    # 保留原规则：删除缺失字段记录，再对重复键保留首条。
    daily_industry = (
        daily_industry
        .dropna(subset=["Stkcd", "td", "IndustryCode"])
        .drop_duplicates(["Stkcd", "td"])
    )

    cue_industry = daily_industry.rename(
        columns={
            "td": "event_td",
            "IndustryCode": "cue_industry_code",
        }
    )

    recall_industry = daily_industry.rename(
        columns={
            "Stkcd": "StkcdRecall",
            "td": "event_td",
            "IndustryCode": "recall_industry_code",
        }
    )

    # 匹配提示公司在提示事件交易日的行业。
    mm_key = mm_key.merge(
        cue_industry[
            ["Stkcd", "event_td", "cue_industry_code"]
        ],
        on=["Stkcd", "event_td"],
        how="left",
        validate="m:1",
    )

    # 匹配被提示公司在同一提示事件交易日的行业。
    mm_key = mm_key.merge(
        recall_industry[
            ["StkcdRecall", "event_td", "recall_industry_code"]
        ],
        on=["StkcdRecall", "event_td"],
        how="left",
        validate="m:1",
    )

    industry_missing = mm_key[
        ["cue_industry_code", "recall_industry_code"]
    ].isna().any(axis=1)

    mm_key["same_industry"] = (
        mm_key["cue_industry_code"].notna()
        & mm_key["recall_industry_code"].notna()
        & (
            mm_key["cue_industry_code"]
            == mm_key["recall_industry_code"]
        )
    )

    print("同行业过滤前公司对记录数：", len(mm_key))
    print("至少一方行业缺失的记录数：", industry_missing.sum())
    print("同行业公司对记录数：", mm_key["same_industry"].sum())
    print("同行业公司对记录比例：", mm_key["same_industry"].mean())

    mm_key = mm_key.loc[~mm_key["same_industry"]].copy()

    print("同行业过滤后公司对记录数：", len(mm_key))

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


def add_size_groups_and_basic_cues(mm, cue_definition="size90"):
    """保留真实规模分组，并按开关构造主提示资格与季度关系的交互变量。"""

    # 使用已经并入的提示公司总市值和月度分位点。
    mm["size90plus"] = (
        mm["mktcap_tm3"] > mm["marketCap_90th_percentile"]
    ).astype(int)

    if cue_definition == "size90":
        mm["mainCueEligible"] = mm["size90plus"]

    elif cue_definition == "size95":
        mm["mainCueEligible"] = (
            mm["mktcap_tm3"] > mm["marketCap_95th_percentile"]
        ).astype("int8")

    elif cue_definition == "hs300":
        if "isHS300" not in mm.columns:
            raise ValueError("尚未匹配历史沪深300成分资格。")
        mm["mainCueEligible"] = mm["isHS300"]

    else:
        raise ValueError("主提示资格只能选择市值90分位、市值95分位或沪深300。")
    mm["size50_90"] = (
        (mm["mktcap_tm3"] > mm["marketCap_50th_percentile"])
        & (mm["mktcap_tm3"] <= mm["marketCap_90th_percentile"])
    ).astype(int)

    mm["size10_50"] = (
        (mm["mktcap_tm3"] >= mm["marketCap_10th_percentile"])
        & (mm["mktcap_tm3"] <= mm["marketCap_50th_percentile"])
    ).astype(int)

    mm["size10minus"] = (
        mm["mktcap_tm3"] < mm["marketCap_10th_percentile"]
    ).astype(int)

    # 主组内部沿用旧计算列名，实际资格由上面的开关决定。
    # 沪深300模式最终输出时，所有主组列统一改为沪深300名称。
    # 辅助规模组始终按真实市值划分，与沪深300主组可能重叠。
    # 当前构造和后续汇总均使用上一季度关系。
    max_lag = 1

    for i in range(1, max_lag + 1):
        lag_col = f"quarterLag{i}Alt"

        if lag_col not in mm.columns:
            print(f"跳过：{lag_col} 不在当前表中")
            continue

        mm[f"quarterLag{i}size90plus"] = (
            mm[lag_col] * mm["mainCueEligible"]
        ).astype("int8")

        mm[f"quarterLag{i}size50_90"] = (
            mm[lag_col] * mm["size50_90"]
        ).astype("int8")

        mm[f"quarterLag{i}size10_50"] = (
            mm[lag_col] * mm["size10_50"]
        ).astype("int8")

        mm[f"quarterLag{i}size10minus"] = (
            mm[lag_col] * mm["size10minus"]
        ).astype("int8")

    return mm


def add_strong_memory_cues(mm, top_mover_mode="any"):
    """将历史机制标记与所选主提示资格、真实中高市值资格分别相乘。"""

    max_lag = 1

    # 左侧为内部工作列后缀，右侧为实际使用的资格标记。
    size_groups = {
        "size90plus": "mainCueEligible",
        "size50_90": "size50_90",
    }

    # 一、历史注意力和成交量标记与规模组交互。
    attention_volume_suffixes = [
        "hiAEV",
        "hiavol",
        "hiavolS",
    ]

    for i in range(1, max_lag + 1):
        for size, eligibility_col in size_groups.items():
            for suffix in attention_volume_suffixes:
                src = f"lag{i}{suffix}"
                dst = f"lag{i}{suffix}{size}"

                if src in mm.columns and eligibility_col in mm.columns:
                    mm[dst] = (
                        mm[src] * mm[eligibility_col]
                    ).astype("int8")

    # 二、历史盈余惊喜标记与规模组交互。
    sue_suffixes = [
        "hi_sue1_abs",
        "hi_sue2_abs",
        "hi_sue3_abs",
        "sue1pos",
        "sue2pos",
        "sue3pos",
        "sue1neg",
        "sue2neg",
        "sue3neg",
        # 双方同为极端正向或极端负向盈余惊喜。
        "hi_sue1_pos",
        "hi_sue2_pos",
        "hi_sue3_pos",
        "hi_sue1_neg",
        "hi_sue2_neg",
        "hi_sue3_neg",
    ]

    for i in range(1, max_lag + 1):
        for size, eligibility_col in size_groups.items():
            for suffix in sue_suffixes:
                src = f"lag{i}{suffix}"
                dst = f"lag{i}{suffix}{size}"

                if src in mm.columns and eligibility_col in mm.columns:
                    mm[dst] = (
                        mm[src] * mm[eligibility_col]
                    ).astype("int8")

    # 三、按所选历史价格异动口径，与规模组交互。
    top_mover_sources = {
        "any": "anyTopMover",
        "both": "topMover",
    }
    top_mover_labels = {
        "any": "任意一家进入价格异动组",
        "both": "两家公司同时进入价格异动组",
    }

    if top_mover_mode not in top_mover_sources:
        raise ValueError(
            "历史价格异动口径只能设置为 any（任意一家）"
            "或 both（两家同时）"
        )

    source_suffix = top_mover_sources[top_mover_mode]

    for i in range(1, max_lag + 1):
        src = f"lag{i}{source_suffix}"

        # 所选口径缺少来源列时明确报错，避免静默跳过。
        if src not in mm.columns:
            raise ValueError(
                f"所选价格异动口径缺少来源列：{src}，"
                "请检查上游记忆矩阵。"
            )

        for size, eligibility_col in size_groups.items():
            # 输出列名保持统一，具体含义由所选口径决定。
            dst = f"lag{i}topMover{size}"

            if eligibility_col in mm.columns:
                mm[dst] = (
                    mm[src] * mm[eligibility_col]
                ).astype("int8")

    print(
        "历史价格异动口径：",
        top_mover_labels[top_mover_mode],
    )

    return mm


def add_sue_cue_inputs(mm):
    """准备三种当前盈余惊喜的加总分子及非缺失分母输入。"""

    max_lag = 1

    for sue_col in ["sue1", "sue2", "sue3"]:
        for i in range(1, max_lag + 1):
            lag_col = f"quarterLag{i}Alt"

            # 等权平均所需的逐行分子。
            mm[f"{sue_col}Lag{i}S90p"] = (
                mm[sue_col]
                * mm[lag_col]
                * mm["mainCueEligible"]
            )

            # 市值加权平均所需的逐行分子。
            mm[f"{sue_col}Lag{i}XmarketCapS90p"] = (
                mm[sue_col]
                * mm[lag_col]
                * mm["mktcap_tm3"]
                * mm["mainCueEligible"]
            )

    # 始终准备各自的非缺失提示数量和有效市值。
    # 后续聚合并计算平均时，再按配置选择分母口径。
    mm = add_sue_denominator_inputs(mm)

    # 来源诊断单独保留，不改变每个提示公司在主盈余惊喜指标中只有一票的口径。
    mm["lag1CueReportSourcesS90p"] = mm["lag1CueReportSources"].where(
        mm["quarterLag1size90plus"].eq(1), ""
    )
    for anchor in ["Annual", "Q1"]:
        source_cue = mm["quarterLag1size90plus"].eq(1) & mm[f"lag1{anchor}Anchor"].eq(1)
        mm[f"lag1{anchor}Anchorsize90plus"] = source_cue.astype("int8")
        for sue_id in [1, 2, 3]:
            source_sue = f"sue{sue_id}{anchor}Anchor"
            mm[f"{source_sue}Lag1S90p"] = mm[source_sue].where(source_cue)
            mm[f"{source_sue}ValidCueLag1S90p"] = (
                source_cue & mm[source_sue].notna()
            ).astype("int8")

    return mm


def add_marketcap_and_interference_inputs(mm):
    """准备所选主组和中高市值组的市值及历史共同公告数量汇总输入。"""

    max_lag = 1

    # 一、提示公司市值：供后续市值加权平均作分母。
    for i in range(1, max_lag + 1):
        mm[f"marketCap_Minus3Lag{i}S90p"] = (
            mm["mktcap_tm3"]
            * mm[f"quarterLag{i}size90plus"]
        )

        mm[f"marketCap_Minus3Lag{i}S5090"] = (
            mm["mktcap_tm3"]
            * mm[f"quarterLag{i}size50_90"]
        )

    # 二、历史共同公告数量：供后续等权平均作分子。
    # 来源变量已包含对应季度的关系条件。
    for i in range(1, max_lag + 1):
        nr_col = f"nrFirmsAnnLag{i}Alt"

        mm[f"nrFirmsAnnLag{i}S90p"] = (
            mm[nr_col] * mm["mainCueEligible"]
        )

        mm[f"nrFirmsAnnLag{i}S5090"] = (
            mm[nr_col] * mm["size50_90"]
        )

    # 三、历史数量乘提示公司市值：供市值加权平均作分子。
    for i in range(1, max_lag + 1):
        nr_col = f"nrFirmsAnnLag{i}Alt"

        mm[f"nrFirmsAnnLag{i}XmarketCapS90p"] = (
            mm[nr_col]
            * mm["mktcap_tm3"]
            * mm["mainCueEligible"]
        )

        mm[f"nrFirmsAnnLag{i}XmarketCapS5090"] = (
            mm[nr_col]
            * mm["mktcap_tm3"]
            * mm["size50_90"]
        )

    return mm


def prepare_pattern_cue_pairs(mm):
    """返回排序后的完整矩阵，以及供后续统计使用的精简配对表。"""

    sort_cols = [
        "Stkcd",
        "Annodt",
        "StkcdRecall",
    ]

    sort_cols_existing = [
        col for col in sort_cols
        if col in mm.columns
    ]

    mm = (
        mm.sort_values(sort_cols_existing)
        .reset_index(drop=True)
    )

    # 只提取上一季度关系成立、且满足所选主提示资格的配对。
    cue_pairs = (
        mm.loc[
            mm["quarterLag1size90plus"].eq(1),
            ["Stkcd", "event_Trddt", "StkcdRecall"],
        ]
        .rename(columns={"event_Trddt": "Trddt"})
        .dropna(subset=["Stkcd", "StkcdRecall", "Trddt"])
        .drop_duplicates(["Stkcd", "Trddt", "StkcdRecall"])
        .copy()
    )

    return mm, cue_pairs


def build_pattern_cue_agg_specs(data_columns):
    """根据输入字段，构造求和与最大值聚合规则。"""

    agg_dict = {
        # 所选主组和三个真实市值辅助组的提示数量。
        "nrCuesLag1S90p": ("quarterLag1size90plus", "sum"),
        "nrCuesLag1S5090": ("quarterLag1size50_90", "sum"),
        "nrCuesLag1S1050": ("quarterLag1size10_50", "sum"),
        "nrCuesLag1S10m": ("quarterLag1size10minus", "sum"),

        # 盈余惊喜的等权分子。
        "sumSue1Lag1S90p": ("sue1Lag1S90p", "sum"),
        "sumSue2Lag1S90p": ("sue2Lag1S90p", "sum"),
        "sumSue3Lag1S90p": ("sue3Lag1S90p", "sum"),

        # 盈余惊喜的市值加权分子。
        "sumSue1Lag1XmarketCapS90p": (
            "sue1Lag1XmarketCapS90p", "sum"
        ),
        "sumSue2Lag1XmarketCapS90p": (
            "sue2Lag1XmarketCapS90p", "sum"
        ),
        "sumSue3Lag1XmarketCapS90p": (
            "sue3Lag1XmarketCapS90p", "sum"
        ),

        # 市值分母。
        "sumMarketCap_Minus3Lag1S90p": (
            "marketCap_Minus3Lag1S90p", "sum"
        ),
        "sumMarketCap_Minus3Lag1S5090": (
            "marketCap_Minus3Lag1S5090", "sum"
        ),

        # 历史共同公告数量的等权分子。
        "sumNrFirmsAnnLag1S90p": (
            "nrFirmsAnnLag1S90p", "sum"
        ),
        "sumNrFirmsAnnLag1S5090": (
            "nrFirmsAnnLag1S5090", "sum"
        ),

        # 历史共同公告数量的市值加权分子。
        "sumNrFirmsAnnLag1XmarketCapS90p": (
            "nrFirmsAnnLag1XmarketCapS90p", "sum"
        ),
        "sumNrFirmsAnnLag1XmarketCapS5090": (
            "nrFirmsAnnLag1XmarketCapS5090", "sum"
        ),
    }

    # 加入三种盈余惊喜各自的有效提示数量和有效市值。
    agg_dict.update(sue_denominator_agg_specs())

    # 所选主提示的实际报告来源及来源专属盈余惊喜，供后续核查。
    agg_dict["cueReportSourcesLag1S90p"] = (
        "lag1CueReportSourcesS90p", union_pipe_values
    )
    for anchor in ["Annual", "Q1"]:
        flag = f"lag1{anchor}Anchorsize90plus"
        agg_dict[f"nrCuesLag1{anchor}S90p"] = (flag, "sum")
        for sue_id in [1, 2, 3]:
            prefix = f"sue{sue_id}{anchor}Anchor"
            agg_dict[f"sumSue{sue_id}{anchor}AnchorLag1S90p"] = (
                f"{prefix}Lag1S90p", "sum"
            )
            agg_dict[f"sumSue{sue_id}{anchor}AnchorValidCueLag1S90p"] = (
                f"{prefix}ValidCueLag1S90p", "sum"
            )

    max_cols = [
        # 基础提示标记。
        "quarterLag1size90plus",
        "quarterLag1size50_90",
        "quarterLag1size10_50",
        "quarterLag1size10minus",
        "lag1AnnualAnchorsize90plus",
        "lag1Q1Anchorsize90plus",

        # 历史注意力和成交量标记。
        "lag1hiAEVsize90plus",
        "lag1hiavolsize90plus",
        "lag1hiavolSsize90plus",
        "lag1hiAEVsize50_90",
        "lag1hiavolsize50_90",
        "lag1hiavolSsize50_90",

        # 历史高绝对盈余惊喜。
        "lag1hi_sue1_abssize90plus",
        "lag1hi_sue2_abssize90plus",
        "lag1hi_sue3_abssize90plus",
        "lag1hi_sue1_abssize50_90",
        "lag1hi_sue2_abssize50_90",
        "lag1hi_sue3_abssize50_90",

        # 历史双方正向盈余惊喜。
        "lag1sue1possize90plus",
        "lag1sue2possize90plus",
        "lag1sue3possize90plus",
        "lag1sue1possize50_90",
        "lag1sue2possize50_90",
        "lag1sue3possize50_90",

        # 历史双方负向盈余惊喜。
        "lag1sue1negsize90plus",
        "lag1sue2negsize90plus",
        "lag1sue3negsize90plus",
        "lag1sue1negsize50_90",
        "lag1sue2negsize50_90",
        "lag1sue3negsize50_90",

        # 历史双方极端正向盈余惊喜。
        "lag1hi_sue1_possize90plus",
        "lag1hi_sue2_possize90plus",
        "lag1hi_sue3_possize90plus",
        "lag1hi_sue1_possize50_90",
        "lag1hi_sue2_possize50_90",
        "lag1hi_sue3_possize50_90",

        # 历史双方极端负向盈余惊喜。
        "lag1hi_sue1_negsize90plus",
        "lag1hi_sue2_negsize90plus",
        "lag1hi_sue3_negsize90plus",
        "lag1hi_sue1_negsize50_90",
        "lag1hi_sue2_negsize50_90",
        "lag1hi_sue3_negsize50_90",

        # 历史价格异动，具体口径由配置决定。
        "lag1topMoversize90plus",
        "lag1topMoversize50_90",
    ]

    for col in max_cols:
        if col in data_columns:
            agg_dict[col] = (col, "max")

    return agg_dict


def collapse_pattern_cues(mm, denominator_mode):
    """将公司对记录聚合为被提示公司与事件交易日层面的提示表。"""

    df = mm.copy()
    group_keys = ["StkcdRecall", "event_Trddt"]
    if df.duplicated(["Stkcd", "event_Trddt", "StkcdRecall"]).any():
        raise ValueError("公司对交易日尚未去重，不能进行提示数量或市值权重求和。")

    # 一、按原顺序整理聚合工作表。
    df = df.sort_values(
        ["StkcdRecall", "event_Trddt", "mktcap_tm3", "Stkcd"]
    ).reset_index(drop=True)

    # 二、构造规则并检查必需字段。
    agg_dict = build_pattern_cue_agg_specs(df.columns)

    needed_cols = set(group_keys)
    for _, value in agg_dict.items():
        needed_cols.add(value[0])

    missing_cols = [
        col for col in needed_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"以下变量缺失，无法聚合：{missing_cols}"
        )

    # 三、按被提示公司与事件交易日聚合。
    collapsed = (
        df.groupby(group_keys, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    # 四、按传入的分母口径计算盈余惊喜均值。
    collapsed = calculate_sue_cue_averages(
        collapsed,
        denominator_mode=denominator_mode,
    )
    print("盈余惊喜聚合分母口径：", denominator_mode)

    # 来源分列仅作诊断，始终按该来源非缺失的提示公司等权。
    # 同一公司支持两种来源时，在各自来源列各出现一次，在主指标仍只出现一次。
    for anchor in ["Annual", "Q1"]:
        for sue_id in [1, 2, 3]:
            numerator = collapsed[f"sumSue{sue_id}{anchor}AnchorLag1S90p"]
            denominator = collapsed[f"sumSue{sue_id}{anchor}AnchorValidCueLag1S90p"]
            collapsed[f"sue{sue_id}{anchor}AnchorCueLag1S90pEW"] = (
                numerator.div(denominator).where(denominator.ne(0), np.nan)
            )

    # 五、计算历史共同公告数量的市值加权均值。
    collapsed["nrFirmsAnnLag1VWS90p"] = np.where(
        collapsed["sumMarketCap_Minus3Lag1S90p"] != 0,
        collapsed["sumNrFirmsAnnLag1XmarketCapS90p"]
        / collapsed["sumMarketCap_Minus3Lag1S90p"],
        np.nan,
    )

    collapsed["nrFirmsAnnLag1VWS5090"] = np.where(
        collapsed["sumMarketCap_Minus3Lag1S5090"] != 0,
        collapsed["sumNrFirmsAnnLag1XmarketCapS5090"]
        / collapsed["sumMarketCap_Minus3Lag1S5090"],
        np.nan,
    )

    # 六、计算历史共同公告数量的等权均值。
    collapsed["nrFirmsAnnLag1EWS90p"] = np.where(
        collapsed["nrCuesLag1S90p"] != 0,
        collapsed["sumNrFirmsAnnLag1S90p"]
        / collapsed["nrCuesLag1S90p"],
        np.nan,
    )

    collapsed["nrFirmsAnnLag1EWS5090"] = np.where(
        collapsed["nrCuesLag1S5090"] != 0,
        collapsed["sumNrFirmsAnnLag1S5090"]
        / collapsed["nrCuesLag1S5090"],
        np.nan,
    )

    # 七、保留供后续日度面板合并使用的日期别名。
    collapsed["date"] = collapsed["event_Trddt"]

    # 八、保留原有诊断输出。
    print("聚合前表结构：", df.shape)
    print("聚合后表结构：", collapsed.shape)
    print(
        "聚合后重复键数量：",
        collapsed.duplicated(group_keys).sum(),
    )

    print("\n核心变量合计：")
    check_cols = [
        "nrCuesLag1S90p",
        "nrCuesLag1S5090",
        "nrCuesLag1S1050",
        "nrCuesLag1S10m",
        "quarterLag1size90plus",
        "lag1hiAEVsize90plus",
        "lag1hiavolsize90plus",
        "lag1hi_sue1_possize90plus",
        "lag1hi_sue1_negsize90plus",
        "lag1hi_sue2_possize90plus",
        "lag1hi_sue2_negsize90plus",
        "lag1topMoversize90plus",
    ]
    check_cols = [
        col for col in check_cols
        if col in collapsed.columns
    ]
    print(
        collapsed[check_cols]
        .sum()
        .sort_values(ascending=False)
    )

    collapsed = (
        collapsed.sort_values(["StkcdRecall", "date"])
        .reset_index(drop=True)
    )

    return collapsed


def stop_with_audit(data, audit_dir, filename, message):
    """只在新任务目录保存需要核对的记录，不把未知值补成0。"""
    path = Path(audit_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    raise ValueError(f"{message}。核对清单：{path}")


def load_industry_daily_panel(root, wanted, chunk_size=250_000):
    """分块读取日度行业市值排名面板，只保留所需公司日。"""
    panel_path = Path(root) / "中间结果/02_变量构造结果/行业提示资格/日度行业市值排名面板.csv"
    columns = ["Stkcd", "td", "Trddt", "IndustryCode", "industryGroup", "size", "industryRank", "baseSampleEligible"]
    if not panel_path.is_file():
        raise FileNotFoundError(f"缺少日度行业市值排名面板：{panel_path}")
    if pd.read_csv(panel_path, nrows=0, encoding="utf-8-sig").columns.tolist() != columns:
        raise ValueError("日度行业市值排名面板字段顺序不一致。")
    wanted = wanted[["Stkcd", "td", "Trddt"]].drop_duplicates().copy()
    wanted["Stkcd"] = wanted["Stkcd"].astype("string")
    wanted["td"] = pd.to_numeric(wanted["td"], errors="raise").astype("int64")
    wanted["Trddt"] = pd.to_datetime(wanted["Trddt"], errors="raise")
    if wanted.duplicated(["Stkcd", "td"]).any():
        raise ValueError("待匹配行业公司日键重复。")
    parts = []
    previous_key = None
    previous_date = None
    dtype = {"Stkcd": "string", "IndustryCode": "string", "industryGroup": "string"}
    for chunk in pd.read_csv(panel_path, dtype=dtype, encoding="utf-8-sig", chunksize=chunk_size):
        if chunk.empty:
            continue
        chunk["Trddt"] = pd.to_datetime(chunk["Trddt"], errors="raise")
        for column in ["td", "size", "industryRank", "baseSampleEligible"]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        if chunk["Stkcd"].isna().any() or chunk["td"].isna().any() or chunk["Trddt"].isna().any():
            raise ValueError("日度行业市值排名面板存在无效键。")
        chunk["td"] = chunk["td"].astype("int64")
        same_td = chunk["td"].eq(chunk["td"].shift())
        unordered = chunk["td"].lt(chunk["td"].shift()) | (same_td & chunk["Stkcd"].le(chunk["Stkcd"].shift()))
        first_key = (int(chunk["td"].iloc[0]), str(chunk["Stkcd"].iloc[0]))
        if unordered.any() or (previous_key is not None and first_key <= previous_key):
            raise ValueError("日度行业市值排名面板未按交易日序号和股票代码升序保存。")
        dates = chunk[["td", "Trddt"]].drop_duplicates()
        if dates["td"].duplicated().any() or (previous_key is not None and first_key[0] == previous_key[0] and chunk["Trddt"].iloc[0] != previous_date):
            raise ValueError("日度行业市值排名面板交易日序号对应多个日期。")
        if (chunk["baseSampleEligible"].notna() & ~chunk["baseSampleEligible"].isin([0, 1])).any():
            raise ValueError("日度行业市值排名面板基础样本资格不是0、1。")
        rank = chunk["industryRank"]
        if (rank.notna() & (rank.lt(1) | rank.mod(1).ne(0))).any():
            raise ValueError("日度行业市值排名面板行业名次无效。")
        previous_key = (int(chunk["td"].iloc[-1]), str(chunk["Stkcd"].iloc[-1]))
        previous_date = chunk["Trddt"].iloc[-1]
        matched = chunk.merge(wanted, on=["Stkcd", "td", "Trddt"], how="inner", validate="1:1")
        if not matched.empty:
            parts.append(matched)
    return pd.concat(parts, ignore_index=True) if parts else wanted.iloc[:0].reindex(columns=columns)

def load_industry_daily_for_candidates(root, candidates, calendar):
    """将排名日及事件日双方的键合并后，只扫描一次正式资格文件。"""
    rank_keys = candidates[["Stkcd", "event_td"]].drop_duplicates().rename(columns={"event_td": "td"})
    rank_keys["td"] = rank_keys["td"] - 3
    rank_keys = rank_keys.merge(calendar, on="td", how="left", validate="m:1")
    cue_keys = candidates[["Stkcd", "event_td", "event_Trddt"]].rename(columns={"event_td": "td", "event_Trddt": "Trddt"})
    recall_keys = candidates[["StkcdRecall", "event_td", "event_Trddt"]].rename(columns={"StkcdRecall": "Stkcd", "event_td": "td", "event_Trddt": "Trddt"})
    wanted = pd.concat([rank_keys, cue_keys, recall_keys], ignore_index=True).drop_duplicates()
    return load_industry_daily_panel(root, wanted)


def match_industry_daily_rows(keys, daily, columns):
    """从已读取的正式资格行中按公司、交易序号和日期匹配。"""
    keys = keys[["Stkcd", "td", "Trddt"]].drop_duplicates()
    if keys.duplicated(["Stkcd", "td"]).any():
        raise ValueError("待匹配公司日对应多个交易日期。")
    return keys.merge(daily[columns], on=["Stkcd", "td", "Trddt"], how="left", validate="1:1", indicator=True)


def merge_rank_information(mm, calendar, industry_daily, audit_dir):
    """匹配提示公司事件前第三个交易日的名次、市值和基础资格。"""
    mm = mm.copy()
    mm["td_tm3"] = mm["event_td"] - 3
    wanted = mm[["Stkcd", "td_tm3"]].drop_duplicates().rename(columns={"td_tm3": "td"})
    wanted = wanted.merge(calendar, on="td", how="left", validate="m:1")
    if wanted["Trddt"].isna().any():
        stop_with_audit(wanted.loc[wanted["Trddt"].isna()], audit_dir, "排名日期缺少交易日历.csv", "排名日不在统一交易日历中")
    ranked = match_industry_daily_rows(wanted, industry_daily, ["Stkcd", "td", "Trddt", "size", "IndustryCode", "industryGroup", "industryRank", "baseSampleEligible"])
    bad = ranked["_merge"].ne("both")
    if bad.any():
        stop_with_audit(ranked.loc[bad], audit_dir, "提示公司排名缺失或无效.csv", "提示公司事件前第三个交易日没有对应行业面板记录")
    ranked = ranked.drop(columns="_merge").rename(columns={"td":"td_tm3", "Trddt":"rankingDate", "size":"mktcap_tm3", "IndustryCode":"rankingIndustryCode", "industryGroup":"rankingIndustryGroup", "industryRank":"rankingIndustryRank", "baseSampleEligible":"rankingBaseSampleEligible"})
    return mm.merge(ranked, on=["Stkcd", "td_tm3"], how="left", validate="m:1")

def filter_event_industries(mm, industry_daily, audit_dir):
    """优先比较事件日二级行业，只有二级行业不足时才比较大类。"""
    event_keys = ["event_td", "event_Trddt"]
    event_dates = {"event_td": "td", "event_Trddt": "Trddt"}
    cue_keys = mm[["Stkcd", *event_keys]].rename(columns=event_dates)
    recall_keys = mm[["StkcdRecall", *event_keys]].rename(
        columns={"StkcdRecall": "Stkcd", **event_dates}
    )
    wanted = pd.concat([cue_keys, recall_keys], ignore_index=True).drop_duplicates()
    industry = match_industry_daily_rows(
        wanted, industry_daily, ["Stkcd", "td", "Trddt", "IndustryCode", "industryGroup"],
    ).drop(columns="_merge")
    industry["industryGroup"] = industry["industryGroup"].astype("string")
    industry["IndustryCode"] = industry["IndustryCode"].astype("string").str.strip().str.upper()
    coarse = (
        industry["industryGroup"].isna()
        & industry["Trddt"].lt("2012-10-26")
        & industry["IndustryCode"].str.fullmatch(r"C[0-9]", na=False)
    )
    industry["branch"] = industry["industryGroup"].str[:2].where(
        ~coarse, industry["IndustryCode"]
    )
    industry = industry.rename(columns={"td": "event_td", "Trddt": "event_Trddt"})
    cue = industry.rename(columns={
        "IndustryCode": "cueIndustryCode", "industryGroup": "cueIndustryGroup", "branch": "cueBranch",
    })
    recall = industry.rename(columns={
        "Stkcd": "StkcdRecall", "IndustryCode": "recallIndustryCode",
        "industryGroup": "recallIndustryGroup", "branch": "recallBranch",
    })
    mm = mm.merge(cue, on=["Stkcd", *event_keys], how="left", validate="m:1")
    mm = mm.merge(recall, on=["StkcdRecall", *event_keys], how="left", validate="m:1")

    both_fine = mm["cueIndustryGroup"].notna() & mm["recallIndustryGroup"].notna()
    same_group = mm["cueIndustryGroup"].eq(mm["recallIndustryGroup"]).fillna(False)
    both_branches = mm["cueBranch"].notna() & mm["recallBranch"].notna()
    different_branch = both_branches & mm["cueBranch"].ne(mm["recallBranch"]).fillna(False)
    same = both_fine & same_group
    uncertain = ~both_fine & ~different_branch
    audit_path = Path(audit_dir) / "事件日公司对行业待核对.csv"
    if uncertain.any() or audit_path.exists():
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit = mm.loc[uncertain, [
            "Stkcd", "StkcdRecall", *event_keys, "cueIndustryCode", "recallIndustryCode",
            "cueIndustryGroup", "recallIndustryGroup", "cueBranch", "recallBranch",
        ]].copy()
        audit["reason"] = np.where(
            both_branches.loc[uncertain],
            "同一大类且至少一方二级行业不明",
            "事件日行业记录缺失或无法识别",
        )
        audit.to_csv(audit_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    print(
        f"明确同行剔除：{int(same.sum()):,} 条，无法确认异业剔除：{int(uncertain.sum()):,} 条，"
        f"筛选前：{len(mm):,} 条。",
        flush=True,
    )
    return mm.loc[~(same | uncertain)].copy()


def check_cue_rank_bounds(mm, top_ns, audit_dir):
    """检查提示公司事件前第三个交易日的名次。"""
    unknown = mm["rankingIndustryRank"].isna()
    if unknown.any():
        stop_with_audit(mm.loc[unknown], audit_dir, "提示公司行业名次待核对.csv", "提示公司行业名次缺失")


def add_industry_cue_flags(mm, top_n, cue_policy="unclean"):
    if cue_policy not in INDUSTRY_CUE_POLICIES:
        raise ValueError(f"未知提示公司清洗口径：{cue_policy}")
    rank = pd.to_numeric(mm["rankingIndustryRank"], errors="coerce")
    main = rank.le(int(top_n)).fillna(False)
    if cue_policy == "clean":
        main &= mm["rankingBaseSampleEligible"].eq(1)
    mm["mainCueEligible"] = main.astype("int8")
    eligible = mm["mainCueEligible"].eq(1)
    mm["size90plus"] = (eligible & (mm["mktcap_tm3"] > mm["marketCap_90th_percentile"])).astype("int8")
    mm["size50_90"] = (eligible & (mm["mktcap_tm3"] > mm["marketCap_50th_percentile"]) & (mm["mktcap_tm3"] <= mm["marketCap_90th_percentile"])).astype("int8")
    mm["size10_50"] = (eligible & (mm["mktcap_tm3"] >= mm["marketCap_10th_percentile"]) & (mm["mktcap_tm3"] <= mm["marketCap_50th_percentile"])).astype("int8")
    mm["size10minus"] = (eligible & (mm["mktcap_tm3"] < mm["marketCap_10th_percentile"])).astype("int8")
    for suffix, col in [("size90plus","mainCueEligible"),("size50_90","size50_90"),("size10_50","size10_50"),("size10minus","size10minus")]:
        mm[f"quarterLag1{suffix}"] = (mm["quarterLag1Alt"] * mm[col]).astype("int8")
    return mm


def read_candidates(root: Path) -> pd.DataFrame:
    """读取公司对，明确保留证券代码前导零。"""
    path = stage_root(root) / INPUT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"缺少第一阶段输出：{path}")
    data = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"Stkcd": "string", "StkcdRecall": "string"},
        parse_dates=["Annodt", "event_Trddt", "Trdmnt_for_breakpoint"],
    )
    for column in ["Stkcd", "StkcdRecall"]:
        data[column] = data[column].apply(normalize_stock_code).astype("string")
    key = ["Stkcd", "event_Trddt", "StkcdRecall"]
    if data.duplicated(key).any():
        raise ValueError("第一阶段输出在公司对×事件交易日层面重复。")
    return data


def prepare_definition_candidates(
    root: Path,
    candidates: pd.DataFrame,
    definition: str,
    top_n: int | None = None,
    industry_daily: pd.DataFrame | None = None,
    cue_policy: str = "unclean",
) -> pd.DataFrame:
    """只改变本阶段主提示资格与行业筛选，返回公司对明细。"""
    if definition not in {"size90", "size95", "hs300", "industry_top"}:
        raise ValueError("提示口径只能为市值90分位、市值95分位、沪深300或行业排名。")
    paths = build_input_paths(root)
    mm = candidates.copy()
    if definition == "industry_top":
        if top_n is None:
            raise ValueError("行业排名口径必须指定每行业入选名额。")
        calendar = pd.read_csv(
            paths["TRD_Cale_td_path"],
            usecols=["td", "Trddt"],
            parse_dates=["Trddt"],
        )
        if industry_daily is None:
            industry_daily = load_industry_daily_for_candidates(root, candidates, calendar)
        audit_dir = stage_root(root) / "提示事件" / f"行业前{top_n}名" / "核对清单"
        # 行业分支的排名市值与权重统一来自含金融公司的排名源。
        # 先移除第一阶段的同名市值列，避免合并后出现重名后缀。
        mm = mm.drop(columns=["mktcap_tm3", "float_mktcap_tm3"], errors="ignore")
        mm = merge_rank_information(mm, calendar, industry_daily, audit_dir)
        # 欣泰退于2017-08-28终止上市；经核对，剔除当天作为提示公司的候选公司对。
        delisted_cue_event = (
            mm["Stkcd"].eq("300372")
            & mm["event_Trddt"].eq(pd.Timestamp("2017-08-28"))
        )
        print(f"剔除欣泰退在2017-08-28作为提示公司的候选公司对：{int(delisted_cue_event.sum()):,}条。", flush=True)
        mm = mm.loc[~delisted_cue_event].copy()
        # 提前应用第三阶段已有的被提示公司日基础资格，避免检查不会进入日度面板的事件。
        recalled_event_columns = ["StkcdRecall", "event_td", "event_Trddt"]
        recalled_event_keys = pd.MultiIndex.from_frame(mm[recalled_event_columns].drop_duplicates())
        recalled_base_parts = []
        for chunk in pd.read_csv(
            paths["eligibility_path"],
            usecols=["Stkcd", "td", "Trddt"],
            dtype={"Stkcd": "string"},
            parse_dates=["Trddt"],
            chunksize=250_000,
            encoding="utf-8-sig",
        ):
            chunk["Stkcd"] = chunk["Stkcd"].apply(normalize_stock_code)
            chunk = chunk.rename(columns={
                "Stkcd": "StkcdRecall", "td": "event_td", "Trddt": "event_Trddt",
            })
            selected = chunk.loc[
                pd.MultiIndex.from_frame(chunk[recalled_event_columns]).isin(recalled_event_keys),
                recalled_event_columns,
            ].copy()
            if not selected.empty:
                recalled_base_parts.append(selected)
        if not recalled_base_parts:
            raise ValueError("行业口径没有符合基础样本资格的被提示公司事件日。")
        recalled_base = pd.concat(recalled_base_parts, ignore_index=True)
        if recalled_base.duplicated(["StkcdRecall", "event_td"]).any():
            raise ValueError("被提示公司基础样本索引的公司日键重复。")
        rows_before_base_filter = len(mm)
        mm = mm.merge(recalled_base, on=recalled_event_columns, how="inner", validate="m:1")
        print(f"剔除被提示公司事件日不在基础样本索引中的候选公司对：{rows_before_base_filter - len(mm):,}条。", flush=True)
        mm = filter_event_industries(mm, industry_daily, audit_dir)
    else:
        if definition == "hs300":
            mm, _, _ = merge_hs300_membership(mm, paths["hs300_membership_path"])
        if industry_daily is None:
            calendar = pd.read_csv(
                paths["TRD_Cale_td_path"],
                usecols=["td", "Trddt"],
                parse_dates=["Trddt"],
            )
            industry_daily = load_industry_daily_for_candidates(root, candidates, calendar)
        audit_dir = stage_root(root) / "提示事件" / definition / "核对清单"
        mm = filter_event_industries(mm, industry_daily, audit_dir)
        if definition in {"size90", "size95"}:
            required_columns = [
                "mktcap_tm3",
                "marketCap_90th_percentile",
                "marketCap_50th_percentile",
                "marketCap_10th_percentile",
            ]

            if definition == "size95":
                required_columns.append("marketCap_95th_percentile")

            mm = mm.dropna(subset=required_columns).copy()
    return mm


def add_cue_variables(
    mm: pd.DataFrame,
    definition: str,
    top_n: int | None,
    cue_policy: str = "unclean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if definition == "industry_top":
        mm = add_industry_cue_flags(mm, top_n=top_n, cue_policy=cue_policy)
    else:
        mm = add_size_groups_and_basic_cues(mm, cue_definition=definition)
    mm = add_strong_memory_cues(mm, top_mover_mode=TOP_MOVER_MODE)
    mm = add_sue_cue_inputs(mm)
    mm = add_marketcap_and_interference_inputs(mm)
    mm, cue_pairs = prepare_pattern_cue_pairs(mm)
    events = collapse_pattern_cues(mm, denominator_mode=SUE_DENOMINATOR_MODE)
    return events, cue_pairs


def add_definition_metadata(
    events: pd.DataFrame,
    cue_pairs: pd.DataFrame,
    definition: str,
    top_n: int | None,
    cue_policy: str = "unclean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """统一不同口径的提示事件和合格公司对字段。"""
    events = events.copy()
    events["cueDefinition"] = definition
    events["industryCuePolicy"] = cue_policy if definition == "industry_top" else pd.NA
    events["industryTopN"] = int(top_n) if top_n is not None else pd.NA
    events["cueColumn"] = "quarterLag1size90plus"
    events["date"] = pd.to_datetime(events["date"])

    cue_pairs = cue_pairs.copy()
    cue_pairs["cueDefinition"] = definition
    cue_pairs["industryCuePolicy"] = cue_policy if definition == "industry_top" else pd.NA
    cue_pairs["industryTopN"] = int(top_n) if top_n is not None else pd.NA
    cue_pairs = cue_pairs.rename(columns={"event_Trddt": "Trddt"})
    return events, cue_pairs


def write_definition_output(
    root: Path,
    events: pd.DataFrame,
    cue_pairs: pd.DataFrame,
    definition: str,
    top_n: int | None,
    cue_policy: str = "unclean",
) -> dict:
    """保存单一口径的提示事件、合格公司对和审计摘要。"""
    label = f"行业前{top_n}名_提示公司{'已清洗' if cue_policy == 'clean' else '未清洗'}" if definition == "industry_top" else definition
    folder = stage_root(root) / "提示事件" / label
    folder.mkdir(parents=True, exist_ok=True)
    event_path = folder / "提示事件.csv"
    pair_path = folder / "合格提示公司对.csv"
    summary_path = folder / "提示事件审计摘要.json"
    events.to_csv(event_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    cue_pairs.to_csv(pair_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    summary = {
        "stage": 2,
        "cue_definition": definition,
        "cue_label": (
            f"行业前{top_n}名"
            if definition == "industry_top"
            else {
                "size90": "市值90分位",
                "size95": "市值95分位",
                "hs300": "沪深300",
            }[definition]
        ),
        "marketcap_quantile": {
            "size90": 0.90,
            "size95": 0.95,
        }.get(definition),
        "industry_top_n": top_n,
        "industry_cue_cleaned": cue_policy == "clean" if definition == "industry_top" else None,
        "grain": "StkcdRecall × date",
        "rows": int(len(events)),
        "eligible_pair_rows": int(len(cue_pairs)),
        "key_duplicate_count": int(events.duplicated(["StkcdRecall", "date"]).sum()),
        "event_path": str(event_path),
        "pair_path": str(pair_path),
        "next_stage": "03_合并提示最终面板.py",
    }
    save_summary(summary_path, summary)
    return {
        "cue_definition": definition,
        "industry_top_n": top_n,
        "event_path": event_path,
        "pair_path": pair_path,
        "summary_path": summary_path,
        "rows": len(events),
    }


def build_one_definition(
    root: Path,
    candidates: pd.DataFrame,
    definition: str,
    top_n: int | None = None,
    industry_daily: pd.DataFrame | None = None,
    cue_policy: str = "unclean",
) -> dict:
    mm = prepare_definition_candidates(root, candidates, definition, top_n, industry_daily, cue_policy)
    events, cue_pairs = add_cue_variables(mm, definition, top_n, cue_policy)
    events, cue_pairs = add_definition_metadata(events, cue_pairs, definition, top_n, cue_policy)
    return write_definition_output(root, events, cue_pairs, definition, top_n, cue_policy)


def main() -> dict:
    root = find_root()
    candidates = read_candidates(root)
    # 所有口径共用同一份事件日行业和事件前三个交易日排名行。
    paths = build_input_paths(root)
    calendar = pd.read_csv(
        paths["TRD_Cale_td_path"],
        usecols=["td", "Trddt"],
        parse_dates=["Trddt"],
    )
    industry_daily = load_industry_daily_for_candidates(root, candidates, calendar)
    results = []
    for definition, top_n, cue_policy in [
        ("size90", None, "unclean"),
        ("size95", None, "unclean"),
        ("hs300", None, "unclean"),
    ] + [("industry_top", number, policy) for number in TOP_NS for policy in INDUSTRY_CUE_POLICIES]:
        print(f"构造提示事件：{definition}，行业名额：{top_n}，提示公司：{cue_policy}", flush=True)
        results.append(build_one_definition(root, candidates, definition, top_n, industry_daily, cue_policy))
    index_path = stage_root(root) / "提示事件" / "本次运行结果.json"
    save_summary(index_path, {"stage": 2, "results": results})
    return {"results": results, "index_path": index_path}


if __name__ == "__main__":
    main()
