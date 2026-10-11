
import pandas as pd
import numpy as np
import gc
from dateutil.relativedelta import * 
from pathlib import Path
import datetime 

#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)


#===============================================================================
## 工具函数
#===============================================================================

def normalize_stock_code(x: object) -> str:
    """标准化股票代码，填充为6位数，并转换为字符串"""
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s == '':
        return ''
    return f"{int(float(s)):06d}"


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


# =============================================================================
## 路径设置
# =============================================================================

def build_pattern_paths(root):
    """根据项目根目录，构造模式公司脚本所需的文件路径。"""
    root = Path(root)

    relative_paths = {
        # 输入文件
        "MemoryMatrixPattern_path":
            "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv",
        "TRD_Cale_td_path":
            "中间结果/01_清洗后样本/市场交易日历_td.csv",
        "TRD_Dalyr_DGTW_path":
            "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv",
        "marketcap_group_path":
            "中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv",
        "industry_path":
            "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv",
        "sue_path":
            "中间结果/02_变量构造结果/4sue/sue.csv",
        "SC_NetWorkRelationsIndex_path":
            "input/CSMAR/供应链网络关系指标表2001-12-31至2025-12-31/SC_NetworkRelationsIndex.csv",
        "IAR_Rept_path":
            "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv",
        "AES_AEV_controls_path":
            "中间结果/02_变量构造结果/3报刊新闻_AES_AEV/报刊新闻_AES_AEV_真实交易日控制变量.csv",
        "adjusted_return_path":
            "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW特征调整收益率.csv",
        "performance_forecast_controls_path":
            "中间结果/02_变量构造结果/5公司公告事件/业绩预告快报控制变量_公告日期口径.csv",
        "other_event_controls_path":
            "中间结果/02_变量构造结果/5公司公告事件/专题研究事件控制变量_公告日期口径.csv",

        # 输出文件
        "sample_intermediate_out_path":
            "中间结果/04_合并数据/Sample_intermediate_pattern.csv",
        "cue_events_out_path":
            "中间结果/04_合并数据/Pattern_cue_events_after_collapse.csv",
        "output_path":
            "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
    }

    return {
        name: root / relative_path
        for name, relative_path in relative_paths.items()
    }


ROOT = Path("/Users/qiyuewu/Desktop/mmm复现数据包")
paths = build_pattern_paths(ROOT)

# 分步封装期间，保留原变量名，供后面的代码继续使用。
MemoryMatrixPattern_path = paths["MemoryMatrixPattern_path"]
TRD_Cale_td_path = paths["TRD_Cale_td_path"]
TRD_Dalyr_DGTW_path = paths["TRD_Dalyr_DGTW_path"]
marketcap_group_path = paths["marketcap_group_path"]
industry_path = paths["industry_path"]
sue_path = paths["sue_path"]
SC_NetWorkRelationsIndex_path = paths["SC_NetWorkRelationsIndex_path"]
IAR_Rept_path = paths["IAR_Rept_path"]
AES_AEV_controls_path = paths["AES_AEV_controls_path"]
adjusted_return_path = paths["adjusted_return_path"]
performance_forecast_controls_path = paths["performance_forecast_controls_path"]
other_event_controls_path = paths["other_event_controls_path"]
sample_intermediate_out_path = paths["sample_intermediate_out_path"]
cue_events_out_path = paths["cue_events_out_path"]
output_path = paths["output_path"]

# =============================================================================
# 参数构造与校验
# =============================================================================

def build_pattern_config(
    *,
    sue_denominator_mode,
    drop_analyst_overlap,
    analyst_id_mode,
    analyst_lookback_days,
    marketcap_breakpoint_mode,
    event_date_mode,
    top_mover_mode="any",
):
    """集中构造并校验模式公司脚本的运行参数。"""

    config = {
        "SUE_DENOMINATOR_MODE": sue_denominator_mode,
        "DROP_ANALYST_OVERLAP": drop_analyst_overlap,
        "ANALYST_ID_MODE": analyst_id_mode,
        "ANALYST_LOOKBACK_DAYS": analyst_lookback_days,
        "MARKETCAP_BREAKPOINT_MODE": marketcap_breakpoint_mode,
        "EVENT_DATE_MODE": event_date_mode,
        "TOP_MOVER_MODE": top_mover_mode,
    }

    choice_rules = {
        "SUE_DENOMINATOR_MODE": (
            "盈余惊喜分母口径",
            ("nonmissing_sue", "all_cues"),
        ),
        "ANALYST_ID_MODE": (
            "分析师识别口径",
            ("split", "team"),
        ),
        "MARKETCAP_BREAKPOINT_MODE": (
            "市值断点月份",
            ("prev_month", "same_month"),
        ),
        "EVENT_DATE_MODE": (
            "自身公告事件日口径",
            ("next_if_nontrading", "strict_next_trading"),
        ),
        "TOP_MOVER_MODE": (
            "历史价格异动口径",
            ("any", "both"),
        ),
    }

    for name, (label, allowed_values) in choice_rules.items():
        if config[name] not in allowed_values:
            raise ValueError(
                f"{label}只能取 {allowed_values}，"
                f"当前设置为 {config[name]!r}"
            )

    if not isinstance(drop_analyst_overlap, bool):
        raise ValueError("是否删除共同分析师覆盖公司对，必须设置为布尔值")

    if (
        not isinstance(analyst_lookback_days, int)
        or isinstance(analyst_lookback_days, bool)
        or analyst_lookback_days <= 0
    ):
        raise ValueError("分析师回溯天数必须为正整数")

    return config


# 今后调整口径，集中修改下面的参数值。
config = build_pattern_config(
    # 每种盈余惊喜使用自身非缺失提示公司数量／市值作分母；
    # 若改为全部提示公司作分母，使用 "all_cues"。
    sue_denominator_mode="nonmissing_sue",

    # 删除有共同分析师覆盖的公司对。
    drop_analyst_overlap=True,

    # 按个人分析师识别；按原始团队字符串识别则使用 "team"。
    analyst_id_mode="split",

    # 选择对应回溯窗口的已有分析师共同覆盖文件。
    analyst_lookback_days=365,

    # 使用上月市值断点；使用当月断点则设置为 "same_month"。
    marketcap_breakpoint_mode="prev_month",

    # 自身公告：交易日保留，非交易日映射至下一交易日。
    # 交易日也顺延则设置为 "strict_next_trading"。
    event_date_mode="next_if_nontrading",
    
    # 历史共同公告时，任意一家进入价格异动组。
    # 要求两家公司同时进入，则设置为 "both"。
    top_mover_mode="any",
)

# 分步封装期间，保留原变量名供后面的代码使用。
SUE_DENOMINATOR_MODE = config["SUE_DENOMINATOR_MODE"]
DROP_ANALYST_OVERLAP = config["DROP_ANALYST_OVERLAP"]
ANALYST_ID_MODE = config["ANALYST_ID_MODE"]
ANALYST_LOOKBACK_DAYS = config["ANALYST_LOOKBACK_DAYS"]
MARKETCAP_BREAKPOINT_MODE = config["MARKETCAP_BREAKPOINT_MODE"]
EVENT_DATE_MODE = config["EVENT_DATE_MODE"]

# 分析师目录暂时保留，后续封装读取函数时再统一整理。
ANALYST_OVERLAP_ROOT = (
    ROOT / "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后"
)

# =============================================================================
# 读取记忆矩阵并合并提示公司市值
# =============================================================================

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
    required_event_cols = {"event_Trddt", "event_td"}
    missing_event_cols = required_event_cols - set(mm.columns)

    if missing_event_cols:
        raise ValueError(
            f"记忆矩阵缺少事件交易日字段：{missing_event_cols}"
        )

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


mm = load_pattern_matrix_with_marketcap(
    matrix_path=paths["MemoryMatrixPattern_path"],
    daily_panel_path=paths["TRD_Dalyr_DGTW_path"],
)

# 合并市值排名数据

# 本文在复现过程中发现，原始代码在识别“大市值 cue 公司”时可能存在前视信息使用问题。
# 具体而言，代码先基于 CRSP 月度数据按 yearMonth 计算 NYSE 股票市值分位数断点，
# 随后将公告日 ANNDATS_ACT 所在月份的市值断点合并至日度样本，并以公告日前第 3 个交易日的
# 公司市值 marketCap_Minus3 是否超过当月 marketCap_90th_percentile 来定义大市值 cue 公司。
# 由于月度市值断点基于月末市值计算，对于发生在月初或月中的公告而言，当月月末断点在公告日时尚不可观测，
# 因此该处理可能引入前视偏差。为避免使用未来信息，
# 本文在复现中改用公告日前最近一个已知月末，即上月末的市值分位数断点，作为大市值公司划分标准。

# =============================================================================
# 合并月度市值断点并筛选缺失记录
# =============================================================================

def merge_monthly_marketcap_breakpoints(
    mm,
    breakpoint_path,
    breakpoint_mode,
):
    """按事件月份合并市值断点，保留市值及必要断点完整的公司对记录。"""

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
    mm = mm.dropna(subset=required_columns).copy()

    print(
        "因总市值或必要断点缺失删除的公司对记录数：",
        rows_before_filter - len(mm),
    )
    print("筛选后剩余公司对记录数：", len(mm))

    return mm


mm = merge_monthly_marketcap_breakpoints(
    mm=mm,
    breakpoint_path=paths["marketcap_group_path"],
    breakpoint_mode=config["MARKETCAP_BREAKPOINT_MODE"],
)

# =============================================================================
# 合并提示公司当前财报的盈余惊喜
# =============================================================================

def merge_current_cue_sue(mm, sue_path, report_path):
    """通过提示公司的原公告日找到报告，再合并该报告的盈余惊喜。"""

    sue = pd.read_csv(
        sue_path,
        encoding="utf-8",
        usecols=["Stkcd", "Accper", "sue1", "sue2", "sue3"],
        parse_dates=["Accper"],
    )

    sue["Stkcd"] = sue["Stkcd"].apply(normalize_stock_code)

    # 用提示公司代码和原公告日，查找对应的财报截止日。
    current_accper = pd.read_csv(
        report_path,
        usecols=["Stkcd", "Annodt", "Accper"],
        parse_dates=["Annodt", "Accper"],
        dtype={"Stkcd": "string"},
    )

    current_accper["Stkcd"] = current_accper["Stkcd"].apply(
        normalize_stock_code
    )

    # 保留原规则：
    # 同一公司同一公告日有多份报告时，选择最早的财报截止日。
    current_accper = (
        current_accper
        .sort_values(["Stkcd", "Annodt", "Accper"])
        .groupby(["Stkcd", "Annodt"], as_index=False)
        .head(1)
        .rename(columns={"Accper": "currentAccper"})
    )

    mm = mm.merge(
        current_accper,
        on=["Stkcd", "Annodt"],
        how="left",
        validate="m:1",
    )

    # 用提示公司代码和所选财报截止日匹配盈余惊喜。
    mm = mm.merge(
        sue,
        left_on=["Stkcd", "currentAccper"],
        right_on=["Stkcd", "Accper"],
        how="left",
        validate="m:1",
    )

    # 保留所选截止日，删除右表临时引入的同义字段。
    mm = mm.drop(columns=["Accper"], errors="ignore")

    print("合并当前盈余惊喜后的公司对记录数：", len(mm))
    print(
        "所选财报截止日缺失的公司对记录数：",
        mm["currentAccper"].isna().sum(),
    )
    print("各盈余惊喜字段缺失的公司对记录数：")
    print(mm[["sue1", "sue2", "sue3"]].isna().sum())

    return mm


mm = merge_current_cue_sue(
    mm=mm,
    sue_path=paths["sue_path"],
    report_path=paths["IAR_Rept_path"],
)

#===============================================================================
# 筛选符合条件的 cue-recall 公司对
#===============================================================================

# =============================================================================
# 构造公司对筛选键表，并排除已识别的同行业公司对
# =============================================================================

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


mm_key = filter_same_industry_pairs(
    mm=mm,
    industry_path=paths["industry_path"],
)

# 删除上下游垂直相关公司对
#（这里仅排除一级供应链关系，后续如果需要的话还可以排除二级供应链关系作为稳健性检验）

# =============================================================================
# 排除已识别的一级供应链公司对
# =============================================================================

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


mm_key = filter_supply_chain_pairs(
    mm_key=mm_key,
    supply_chain_path=paths["SC_NetWorkRelationsIndex_path"],
)

# =============================================================================
# 合并共同分析师覆盖，并按开关过滤公司对
# =============================================================================

def filter_analyst_overlap_pairs(
    mm_key,
    overlap_root,
    id_mode,
    lookback_days,
    drop_overlap,
):
    """读取已有共同分析师覆盖结果，并按原规则处理公司对。"""

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


mm_key = filter_analyst_overlap_pairs(
    mm_key=mm_key,
    overlap_root=ANALYST_OVERLAP_ROOT,
    id_mode=config["ANALYST_ID_MODE"],
    lookback_days=config["ANALYST_LOOKBACK_DAYS"],
    drop_overlap=config["DROP_ANALYST_OVERLAP"],
)


# =============================================================================
# 将合格公司对名单应用到完整记忆矩阵
# =============================================================================

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


mm = apply_pair_filters(
    mm=mm,
    mm_key=mm_key,
)

#===============================================================================
# 构造 cue dummy: 大公司cue、强记忆cue
#===============================================================================

def add_size_groups_and_basic_cues(mm):
    """划分提示公司的规模组，并构造季度关系与规模组的交互变量。"""

    # 使用已经并入的提示公司总市值和月度分位点。
    mm["size90plus"] = (
        mm["mktcap_tm3"] > mm["marketCap_90th_percentile"]
    ).astype(int)

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

    # 当前构造和后续汇总均使用上一季度关系。
    max_lag = 1

    for i in range(1, max_lag + 1):
        lag_col = f"quarterLag{i}Alt"

        if lag_col not in mm.columns:
            print(f"跳过：{lag_col} 不在当前表中")
            continue

        mm[f"quarterLag{i}size90plus"] = (
            mm[lag_col] * mm["size90plus"]
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


mm = add_size_groups_and_basic_cues(mm)

# ============================================================
# 构造历史机制标记与提示公司规模组的交互变量
# ============================================================

def add_strong_memory_cues(mm, top_mover_mode="any"):
    """将已有的上一季度历史机制标记与当前提示公司规模组相乘。"""

    max_lag = 1

    size_groups = [
        "size90plus",
        "size50_90",
    ]

    # 一、历史注意力和成交量标记与规模组交互。
    attention_volume_suffixes = [
        "hiAEV",
        "hiavol",
        "hiavolS",
    ]

    for i in range(1, max_lag + 1):
        for size in size_groups:
            for suffix in attention_volume_suffixes:
                src = f"lag{i}{suffix}"
                dst = f"lag{i}{suffix}{size}"

                if src in mm.columns and size in mm.columns:
                    mm[dst] = (
                        mm[src] * mm[size]
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
        for size in size_groups:
            for suffix in sue_suffixes:
                src = f"lag{i}{suffix}"
                dst = f"lag{i}{suffix}{size}"

                if src in mm.columns and size in mm.columns:
                    mm[dst] = (
                        mm[src] * mm[size]
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

        for size in size_groups:
            # 输出列名保持统一，具体含义由所选口径决定。
            dst = f"lag{i}topMover{size}"

            if size in mm.columns:
                mm[dst] = (
                    mm[src] * mm[size]
                ).astype("int8")

    print(
        "历史价格异动口径：",
        top_mover_labels[top_mover_mode],
    )

    return mm


mm = add_strong_memory_cues(
    mm=mm,
    top_mover_mode=config["TOP_MOVER_MODE"],
)

# ============================================================
# 准备大市值提示事件的盈余惊喜分子及有效分母输入
# ============================================================

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
                * mm["size90plus"]
            )

            # 市值加权平均所需的逐行分子。
            mm[f"{sue_col}Lag{i}XmarketCapS90p"] = (
                mm[sue_col]
                * mm[lag_col]
                * mm["mktcap_tm3"]
                * mm["size90plus"]
            )

    # 始终准备各自的非缺失提示数量和有效市值。
    # 后续聚合并计算平均时，再按配置选择分母口径。
    mm = add_sue_denominator_inputs(mm)

    return mm


mm = add_sue_cue_inputs(mm)
# ============================================================
# 准备市值分母和历史共同公告数量分子
# ============================================================

def add_marketcap_and_interference_inputs(mm):
    """准备两个规模组的市值及历史共同公告数量汇总输入。"""

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
            mm[nr_col] * mm["size90plus"]
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
            * mm["size90plus"]
        )

        mm[f"nrFirmsAnnLag{i}XmarketCapS5090"] = (
            mm[nr_col]
            * mm["mktcap_tm3"]
            * mm["size50_90"]
        )

    return mm


mm = add_marketcap_and_interference_inputs(mm)
# ============================================================
# 聚合前排序，并提取基准大市值提示配对表
# ============================================================

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

    # 只提取上一季度关系成立的大市值提示配对。
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


mm, cue_pairs = prepare_pattern_cue_pairs(mm)
# ============================================================
# Collapse data
# 当前 mm 层级：Stkcd × event_Trddt × StkcdRecall
# 目标层级：StkcdRecall × event_Trddt
# ============================================================


# ============================================================
# 构造提示事件的聚合规则
# ============================================================

def build_pattern_cue_agg_specs(data_columns):
    """根据输入字段，构造求和与最大值聚合规则。"""

    agg_dict = {
        # 四个规模组的提示数量。
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

    max_cols = [
        # 基础提示标记。
        "quarterLag1size90plus",
        "quarterLag1size50_90",
        "quarterLag1size10_50",
        "quarterLag1size10minus",

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


# ============================================================
# 聚合到被提示公司与事件交易日，并计算各类均值
# ============================================================

def collapse_pattern_cues(mm, denominator_mode):
    """将公司对记录聚合为被提示公司与事件交易日层面的提示表。"""

    df = mm.copy()
    group_keys = ["StkcdRecall", "event_Trddt"]

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


collapsed = collapse_pattern_cues(
    mm=mm,
    denominator_mode=config["SUE_DENOMINATOR_MODE"],
)

# ============================================================
# 精简提示事件表并保存，供后续日度面板合并
# ============================================================

def save_pattern_cue_events(collapsed, output_path):
    """删除聚合中间列，保存并返回精简提示事件表。"""

    output_path = Path(output_path)

    # 日期别名已在上游生成，保留日期别名供后续合并。
    # 各类加总列只用于计算均值，完成计算后删除。
    panel_drop_cols = (
        ["event_Trddt"]
        + [
            col for col in collapsed.columns
            if col.startswith(
                ("sumSue", "sumMarketCap", "sumNrFirmsAnn")
            )
        ]
    )

    cue_events_for_panel = (
        collapsed
        .drop(columns=panel_drop_cols, errors="ignore")
        .copy()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cue_events_for_panel.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print(
        "\n已保存用于后续主面板合并的提示事件表：",
        output_path,
    )
    print("保存行数：", len(cue_events_for_panel))
    print("保存列数：", cue_events_for_panel.shape[1])
    print("已删除的中间变量：")
    print(panel_drop_cols)

    return cue_events_for_panel


cue_events_for_panel = save_pattern_cue_events(
    collapsed=collapsed,
    output_path=paths["cue_events_out_path"],
)

del mm, mm_key,  collapsed
gc.collect()


#===============================================================================
# 从股票日度数据开始构建主样本，再合并 cue 事件数据
#===============================================================================

# ============================================================
# 读取日度主面板并统一收益变量名称
# ============================================================

def load_pattern_daily_panel(daily_panel_path):
    """读取个股日度面板，标准化股票代码并重命名收益变量。"""

    forward_return_cols = [
        f"raw_ret_t_t{i}"
        for i in range(1, 51)
    ]

    base_return_cols = [
        "raw_ret_tm3",
        "raw_ret_tm2",
        "raw_ret_tm1",
        "raw_ret_t",
    ]

    daily_panel = pd.read_csv(
        daily_panel_path,
        encoding="utf-8",
        usecols=(
            [
                "Stkcd",
                "td",
                "Trddt",
                "year",
                "mktcap_tm3",
            ]
            + base_return_cols
            + forward_return_cols
        ),
        dtype=(
            {
                "Stkcd": "string",
                "td": "int32",
                "year": "float32",
                "mktcap_tm3": "float32",
            }
            | {
                column: "float32"
                for column in base_return_cols + forward_return_cols
            }
        ),
        parse_dates=["Trddt"],
    )

    daily_panel["Stkcd"] = (
        daily_panel["Stkcd"]
        .apply(normalize_stock_code)
    )

    # 日度面板在后续全部按被提示公司命名。
    daily_panel = (
        daily_panel
        .rename(columns={"Stkcd": "StkcdRecall"})
        .drop(columns=["IndustryCode"], errors="ignore")
    )

    raw_return_rename = {
        "raw_ret_tm3": "return_tm3_Recall",
        "raw_ret_tm2": "return_tm2_Recall",
        "raw_ret_tm1": "return_tm1_Recall",
        "raw_ret_t": "return_t_Recall",
    }

    raw_return_rename.update({
        f"raw_ret_t_t{i}": f"return_t_tp{i}_Recall"
        for i in range(1, 51)
    })

    daily_panel = daily_panel.rename(
        columns=raw_return_rename
    )

    return daily_panel


TRD_Dalyr = load_pattern_daily_panel(
    TRD_Dalyr_DGTW_path
)
# ============================================================
# 将自身公告日映射到事件交易日，并生成 ownEA
# ============================================================

def add_own_earnings_announcement_flag(
    daily_panel,
    report_path,
    event_date_mode,
):
    """将自身公告日期映射到交易日，并合并 ownEA 标记。"""

    if event_date_mode not in {
        "next_if_nontrading",
        "strict_next_trading",
    }:
        raise ValueError(
            "自身公告事件日口径设置无效："
            f"{event_date_mode!r}"
        )

    report = pd.read_csv(
        report_path,
        usecols=["Stkcd", "Accper", "Annodt"],
        parse_dates=["Accper", "Annodt"],
    )

    report["Stkcd"] = (
        report["Stkcd"]
        .apply(normalize_stock_code)
    )

    report = report.rename(
        columns={"Stkcd": "StkcdRecall"}
    )

    calendar = (
        daily_panel[["Trddt", "td"]]
        .drop_duplicates()
        .sort_values("Trddt")
    )

    announcement_dates = (
        report[["Annodt"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Annodt")
    )

    announcement_dates = pd.merge_asof(
        announcement_dates,
        calendar,
        left_on="Annodt",
        right_on="Trddt",
        direction="forward",
        allow_exact_matches=(
            event_date_mode == "next_if_nontrading"
        ),
    )

    own_ea = (
        report
        .merge(
            announcement_dates,
            on="Annodt",
            how="left",
            validate="many_to_one",
        )
        .dropna(subset=["Trddt"])
        [["StkcdRecall", "Trddt"]]
        .drop_duplicates()
    )

    own_ea["ownEA"] = 1

    daily_panel = daily_panel.merge(
        own_ea,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="one_to_one",
    )

    daily_panel["ownEA"] = (
        daily_panel["ownEA"]
        .fillna(0)
        .astype("int8")
    )

    print(
        "ownEA为1的数量：",
        (daily_panel["ownEA"] == 1).sum(),
    )
    print(
        "ownEA为1的比例：",
        (daily_panel["ownEA"] == 1).sum()
        / len(daily_panel),
    )

    unmatched_own_ea = (
        own_ea
        .merge(
            daily_panel[["StkcdRecall", "Trddt"]],
            on=["StkcdRecall", "Trddt"],
            how="left",
            indicator=True,
        )
        .loc[lambda data: data["_merge"].eq("left_only")]
        .copy()
    )

    print(
        "未匹配自身公告事件数：",
        len(unmatched_own_ea),
    )
    print(
        unmatched_own_ea["Trddt"]
        .dt.year
        .value_counts()
        .sort_index()
    )

    return daily_panel


TRD_Dalyr = add_own_earnings_announcement_flag(
    daily_panel=TRD_Dalyr,
    report_path=IAR_Rept_path,
    event_date_mode=config["EVENT_DATE_MODE"],
)
# ============================================================
# 构造自身公告日前后交易日窗口控制变量
# ============================================================

def add_own_ea_windows(daily_panel, window=10):
    """生成自身公告日前后若干个交易日的控制变量。"""

    if (
        not isinstance(window, int)
        or isinstance(window, bool)
        or window <= 0
    ):
        raise ValueError("自身公告窗口长度必须为正整数")

    key_cols = ["StkcdRecall", "td"]

    # 日度主面板必须在被提示公司—交易日序号层面唯一。
    duplicate_count = daily_panel.duplicated(key_cols).sum()
    print(
        "日度主面板中 StkcdRecall-td 重复数：",
        duplicate_count,
    )

    if duplicate_count != 0:
        raise ValueError(
            "日度主面板中 StkcdRecall-td 不唯一，"
            "不能构造自身公告窗口。"
        )

    # 提取自身公告发生的公司—交易日序号。
    own_ea_td = (
        daily_panel.loc[
            daily_panel["ownEA"].eq(1),
            ["StkcdRecall", "td"],
        ]
        .drop_duplicates()
        .copy()
    )

    print("ownEA 事件数：", len(own_ea_td))

    for i in range(1, window + 1):
        # ownEA_tp{i}：当前日之后第 i 个交易日发生自身公告。
        future_ea = own_ea_td.copy()
        future_ea["td"] = future_ea["td"] - i
        future_ea = (
            future_ea
            .drop_duplicates(["StkcdRecall", "td"])
            .assign(**{f"ownEA_tp{i}": 1})
        )

        daily_panel = daily_panel.merge(
            future_ea,
            on=["StkcdRecall", "td"],
            how="left",
            validate="one_to_one",
        )

        daily_panel[f"ownEA_tp{i}"] = (
            daily_panel[f"ownEA_tp{i}"]
            .fillna(0)
            .astype("int8")
        )

        # ownEA_tm{i}：当前日之前第 i 个交易日发生自身公告。
        past_ea = own_ea_td.copy()
        past_ea["td"] = past_ea["td"] + i
        past_ea = (
            past_ea
            .drop_duplicates(["StkcdRecall", "td"])
            .assign(**{f"ownEA_tm{i}": 1})
        )

        daily_panel = daily_panel.merge(
            past_ea,
            on=["StkcdRecall", "td"],
            how="left",
            validate="one_to_one",
        )

        daily_panel[f"ownEA_tm{i}"] = (
            daily_panel[f"ownEA_tm{i}"]
            .fillna(0)
            .astype("int8")
        )

    print("ownEA 前后窗口变量生成完成。")

    own_ea_window_cols = (
        [f"ownEA_tp{i}" for i in range(1, window + 1)]
        + ["ownEA"]
        + [f"ownEA_tm{i}" for i in range(1, window + 1)]
    )

    print(daily_panel[own_ea_window_cols].sum())

    return daily_panel


TRD_Dalyr = add_own_ea_windows(
    daily_panel=TRD_Dalyr,
    window=10,
)

# ============================================================
# 合并自身业绩预告和专题事件控制变量
# ============================================================

def merge_own_event_controls(
    daily_panel,
    performance_forecast_path,
    other_event_path,
):
    """读取并合并被提示公司的业绩预告和专题事件控制变量。"""

    performance_forecast_controls = pd.read_csv(
        performance_forecast_path,
        encoding="utf-8-sig",
        usecols=[
            "StkcdRecall",
            "Trddt",
            "ownPerformanceForecastBrief",
        ],
        dtype={"StkcdRecall": "string"},
        parse_dates=["Trddt"],
    )

    other_event_controls = pd.read_csv(
        other_event_path,
        encoding="utf-8-sig",
        usecols=[
            "StkcdRecall",
            "Trddt",
            "ownOtherEventER",
        ],
        dtype={"StkcdRecall": "string"},
        parse_dates=["Trddt"],
    )

    control_tables = [
        (
            "业绩预告快报",
            performance_forecast_controls,
            "ownPerformanceForecastBrief",
        ),
        (
            "专题事件",
            other_event_controls,
            "ownOtherEventER",
        ),
    ]

    for controls_name, controls_df, value_col in control_tables:
        controls_df["StkcdRecall"] = (
            controls_df["StkcdRecall"]
            .apply(normalize_stock_code)
        )

        missing_key = (
            controls_df["StkcdRecall"].eq("")
            | controls_df["Trddt"].isna()
        ).sum()

        duplicate_key = controls_df.duplicated(
            ["StkcdRecall", "Trddt"]
        ).sum()

        print(
            f"{controls_name}控制表缺失键数：{missing_key}，"
            f"重复键数：{duplicate_key}"
        )

        if missing_key != 0 or duplicate_key != 0:
            raise ValueError(
                f"{controls_name}控制表在 "
                "StkcdRecall-Trddt 层面存在缺失或重复，"
                "请先检查。"
            )

        controls_df[value_col] = (
            pd.to_numeric(
                controls_df[value_col],
                errors="raise",
            )
            .fillna(0)
            .astype("int8")
        )

    # 两张控制表已经按真实交易日整理，直接按公司和交易日合并。
    daily_panel = daily_panel.merge(
        performance_forecast_controls,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="one_to_one",
    )

    daily_panel = daily_panel.merge(
        other_event_controls,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="one_to_one",
    )

    control_cols = [
        "ownPerformanceForecastBrief",
        "ownOtherEventER",
    ]

    daily_panel[control_cols] = (
        daily_panel[control_cols]
        .fillna(0)
        .astype("int8")
    )

    # 保持后续回归使用的变量名称。
    daily_panel["ownMajorDisclosure"] = (
        daily_panel["ownOtherEventER"]
        .astype("int8")
    )

    daily_panel["ownPerformanceForecast"] = (
        daily_panel["ownPerformanceForecastBrief"]
        .astype("int8")
    )

    print(
        "ownMajorDisclosure=1 数量：",
        daily_panel["ownMajorDisclosure"].sum(),
    )
    print(
        "ownPerformanceForecast=1 数量：",
        daily_panel["ownPerformanceForecast"].sum(),
    )
    print(
        "ownMajorDisclosure 与 ownPerformanceForecast "
        "同时为1数量：",
        (
            daily_panel["ownMajorDisclosure"].eq(1)
            & daily_panel["ownPerformanceForecast"].eq(1)
        ).sum(),
    )

    return daily_panel


TRD_Dalyr = merge_own_event_controls(
    daily_panel=TRD_Dalyr,
    performance_forecast_path=performance_forecast_controls_path,
    other_event_path=other_event_controls_path,
)


#===============================================================================
# 合并上游已构造的真实交易日新闻控制变量
#===============================================================================

AES_AEV_controls = pd.read_csv(
    AES_AEV_controls_path,
    encoding='utf-8-sig',
    usecols=[
        'Stkcd',
        'date',
        'AES',
        'AEV',
        'aesIncrease',
        'aesDecrease',
        'logIncreaseAEV',
        'logDecreaseAEV',
    ],
    dtype={'Stkcd': 'string'},
    parse_dates=['date']
)

AES_AEV_controls['Stkcd'] = AES_AEV_controls['Stkcd'].apply(
    normalize_stock_code
)

AES_AEV_controls = AES_AEV_controls.rename(columns={
    'Stkcd': 'StkcdRecall',
    'date': 'Trddt'
})

TRD_Dalyr['Trddt'] = pd.to_datetime(TRD_Dalyr['Trddt'])
AES_AEV_controls['Trddt'] = pd.to_datetime(AES_AEV_controls['Trddt'])

# 检查上游控制表是否在 股票-真实交易日 层面唯一
dup_news = AES_AEV_controls.duplicated(['StkcdRecall', 'Trddt']).sum()
print('真实交易日新闻控制表中 StkcdRecall-Trddt 重复数：', dup_news)

if dup_news != 0:
    raise ValueError(
        '真实交易日新闻控制表在 StkcdRecall-Trddt 层面不唯一，请先检查上游构造。'
    )

news_numeric_cols = [
    'AES',
    'AEV',
    'aesIncrease',
    'aesDecrease',
    'logIncreaseAEV',
    'logDecreaseAEV',
]
AES_AEV_controls[news_numeric_cols] = AES_AEV_controls[news_numeric_cols].apply(
    pd.to_numeric,
    errors='raise',
)

if (AES_AEV_controls['AEV'].dropna() < 0).any():
    raise ValueError('真实交易日新闻控制表中的 AEV 不应出现负数。')

print('真实交易日新闻控制表行数：', len(AES_AEV_controls))

# 正式合并
TRD_Dalyr = TRD_Dalyr.merge(
    AES_AEV_controls,
    on=['StkcdRecall', 'Trddt'],
    how='left',
    validate='1:1'
)
del AES_AEV_controls
gc.collect()

print('合并真实交易日 AES/AEV 控制变量后 TRD_Dalyr 表结构：', TRD_Dalyr.shape)
for news_col in news_numeric_cols:
    print(f'{news_col} 非缺失数量：', TRD_Dalyr[news_col].notna().sum())

print("媒体控制变量汇总：")
print(TRD_Dalyr[
    [
        "aesIncrease",
        "aesDecrease",
        "logIncreaseAEV",
        "logDecreaseAEV"
    ]
].describe())

#===============================================================================
# 合并 collapse 后的 cue 事件表
# 当前主面板：TRD_Dalyr，层级 StkcdRecall × Trddt
# cue 表：cue_events_for_panel，层级 StkcdRecall × date
#===============================================================================

try:
    cue_panel = cue_events_for_panel.copy()
except NameError:
    cue_panel = pd.read_csv(
        cue_events_out_path,
        encoding="utf-8-sig",
        parse_dates=["date"],
        dtype={"StkcdRecall": "string"}
    )
    cue_panel["StkcdRecall"] = cue_panel["StkcdRecall"].apply(normalize_stock_code)

TRD_Dalyr["Trddt"] = pd.to_datetime(TRD_Dalyr["Trddt"])
cue_panel["date"] = pd.to_datetime(cue_panel["date"])

# 检查主面板唯一性
dup_main = TRD_Dalyr.duplicated(["StkcdRecall", "Trddt"]).sum()
print("TRD_Dalyr 中 StkcdRecall-Trddt 重复数：", dup_main)

if dup_main != 0:
    raise ValueError("TRD_Dalyr 中 StkcdRecall-Trddt 不唯一，请先检查主面板。")

# 检查 cue 表唯一性
dup_cue = cue_panel.duplicated(["StkcdRecall", "date"]).sum()
print("cue_panel 中 StkcdRecall-date 重复数：", dup_cue)

if dup_cue != 0:
    raise ValueError("cue_panel 中 StkcdRecall-date 不唯一，请检查 collapse 步骤。")

print("cue 事件总数：", len(cue_panel))

# 正式合并
TRD_Dalyr = TRD_Dalyr.merge(
    cue_panel,
    left_on=["StkcdRecall", "Trddt"],
    right_on=["StkcdRecall", "date"],
    how="left",
    validate="1:1"
)
matched_cue_count = TRD_Dalyr["date"].notna().sum()
print("cue 能匹配到日度主面板数量：", matched_cue_count)
print("cue 匹配不到日度主面板数量：", len(cue_panel) - matched_cue_count)
del cue_panel
TRD_Dalyr = TRD_Dalyr.drop(columns=["date"], errors="ignore")

print("合并 cue 事件后的 TRD_Dalyr 表结构：", TRD_Dalyr.shape)
print("合并后 StkcdRecall-Trddt 重复数：",
      TRD_Dalyr.duplicated(["StkcdRecall", "Trddt"]).sum())

#===============================================================================
# 没有 cue 的日期，将 cue dummy 缺失值设为 0
#===============================================================================

cue_dummy_cols = [
    "quarterLag1size90plus",
    "quarterLag1size50_90",
    "quarterLag1size10_50",
    "quarterLag1size10minus",

    "lag1hiAEVsize90plus",
    "lag1hiavolsize90plus",
    "lag1hiavolSsize90plus",
    "lag1hiAEVsize50_90",
    "lag1hiavolsize50_90",
    "lag1hiavolSsize50_90",

    "lag1hi_sue1_abssize90plus",
    "lag1hi_sue2_abssize90plus",
    "lag1hi_sue3_abssize90plus",
    "lag1hi_sue1_abssize50_90",
    "lag1hi_sue2_abssize50_90",
    "lag1hi_sue3_abssize50_90",

    "lag1sue1possize90plus",
    "lag1sue2possize90plus",
    "lag1sue3possize90plus",
    "lag1sue1possize50_90",
    "lag1sue2possize50_90",
    "lag1sue3possize50_90",

    "lag1sue1negsize90plus",
    "lag1sue2negsize90plus",
    "lag1sue3negsize90plus",
    "lag1sue1negsize50_90",
    "lag1sue2negsize50_90",
    "lag1sue3negsize50_90",

    "lag1hi_sue1_possize90plus",
    "lag1hi_sue2_possize90plus",
    "lag1hi_sue3_possize90plus",
    "lag1hi_sue1_possize50_90",
    "lag1hi_sue2_possize50_90",
    "lag1hi_sue3_possize50_90",

    "lag1hi_sue1_negsize90plus",
    "lag1hi_sue2_negsize90plus",
    "lag1hi_sue3_negsize90plus",
    "lag1hi_sue1_negsize50_90",
    "lag1hi_sue2_negsize50_90",
    "lag1hi_sue3_negsize50_90",

    "lag1topMoversize90plus",
    "lag1topMoversize50_90",
]

cue_dummy_cols = [c for c in cue_dummy_cols if c in TRD_Dalyr.columns]

TRD_Dalyr[cue_dummy_cols] = (
    TRD_Dalyr[cue_dummy_cols]
    .fillna(0)
    .astype("int8")
)

print("cue dummy 缺失值已设为 0")
print(TRD_Dalyr[cue_dummy_cols].sum().sort_values(ascending=False).head(20))

cue_count_cols = [
    "nrCuesLag1S90p",
    "nrCuesLag1S5090",
    "nrCuesLag1S1050",
    "nrCuesLag1S10m",
]

cue_count_cols = [c for c in cue_count_cols if c in TRD_Dalyr.columns]

TRD_Dalyr[cue_count_cols] = (
    TRD_Dalyr[cue_count_cols]
    .fillna(0)
)

#===============================================================================
# 合并特征调整收益率
#===============================================================================
adj = pd.read_csv(
    adjusted_return_path,
    encoding="utf-8-sig",
    dtype=(
        {"Stkcd": "string", "portfolioNumber": "float32"}
        | {
            column: "float32"
            for column in (
                ["chAdj_ret_tm3", "chAdj_ret_tm2", "chAdj_ret_tm1", "chAdj_ret_t"]
                + [f"chAdj_ret_t_t{i}" for i in range(1, 51)]
            )
        }
    ),
    parse_dates=["Trddt"]
)

adj["Stkcd"] = adj["Stkcd"].apply(normalize_stock_code)

adj = adj.rename(columns={
    "Stkcd": "StkcdRecall",
    "chAdj_ret_tm3": "chAdjreturn_tm3_Recall",
    "chAdj_ret_tm2": "chAdjreturn_tm2_Recall",
    "chAdj_ret_tm1": "chAdjreturn_tm1_Recall",
    "chAdj_ret_t": "chAdjreturn_t_Recall",
})

rename_forward = {
    f"chAdj_ret_t_t{i}": f"chAdjreturn_t_tp{i}_Recall"
    for i in range(1, 51)
}
adj = adj.rename(columns=rename_forward)

keep_adj_cols = (
    ["StkcdRecall", "Trddt", "portfolioNumber"]
    + [c for c in adj.columns if c.startswith("chAdjreturn_")]
)

adj = adj[keep_adj_cols].copy()

dup_adj = adj.duplicated(["StkcdRecall", "Trddt"]).sum()
print("DGTW特征调整收益率 StkcdRecall-Trddt 重复数：", dup_adj)
if dup_adj != 0:
    raise ValueError("DGTW特征调整收益率在 StkcdRecall-Trddt 层面不唯一。")

TRD_Dalyr = TRD_Dalyr.merge(
    adj,
    on=["StkcdRecall", "Trddt"],
    how="left",
    validate="1:1",
)
del adj
gc.collect()

print("合并特征调整收益率后表结构：", TRD_Dalyr.shape)
print("chAdjreturn_t_Recall 缺失比例：", TRD_Dalyr["chAdjreturn_t_Recall"].isna().mean())
print("chAdjreturn_t_tp1_Recall 缺失比例：", TRD_Dalyr["chAdjreturn_t_tp1_Recall"].isna().mean())
print("chAdjreturn_t_tp10_Recall 缺失比例：", TRD_Dalyr["chAdjreturn_t_tp10_Recall"].isna().mean())

#===============================================================================
# 最终整理样本并进行缩尾
#===============================================================================

# 构造 interference 高低组虚拟变量
# interference 衡量同一个 cue 场景中同时出现的其他公告数量

# 只是兜底的写法，防止没有year这一列
if "year" not in TRD_Dalyr.columns:
    TRD_Dalyr["year"] = TRD_Dalyr["Trddt"].dt.year

def by_year_xtile_2(df, value_col, year_col="year"):
    """
    按年份把变量分成低/高两组：低组=1，高组=2。
    """
    def _split_one_year(s):
        nonmissing = s.dropna()

        # 如果该年份没有有效值，返回缺失
        if nonmissing.empty:
            return pd.Series(np.nan, index=s.index)

        # 如果该年份有效取值不足以分成两组，则全部设为缺失
        # 避免 pandas qcut 因重复断点报错
        if nonmissing.nunique() < 2:
            return pd.Series(np.nan, index=s.index)

        out = pd.Series(np.nan, index=s.index)

        try:
            out.loc[nonmissing.index] = pd.qcut(
                nonmissing,
                q=2,
                labels=[1, 2],
                duplicates="drop"
            ).astype(float)
        except ValueError:
            out.loc[nonmissing.index] = np.nan

        return out

    return df.groupby(year_col, group_keys=False)[value_col].apply(_split_one_year)

# 当前只对 lag1 构造，但循环写法可以扩展到其他 lag
for i in range(1, 2):
    # (1) 等权版本：对所有 cue 等权平均
    for suffix in ["S90p", "S5090"]:
        value_col = f"nrFirmsAnnLag{i}EW{suffix}"
        group_col = f"interferenceGroupEWLag{i}{suffix}"
        low_col = f"lowInterferenceEWLag{i}{suffix}"
        high_col = f"highInterferenceEWLag{i}{suffix}"

        TRD_Dalyr[group_col] = by_year_xtile_2(TRD_Dalyr, value_col)
        TRD_Dalyr[low_col] = (TRD_Dalyr[group_col] == 1).astype("int8")
        TRD_Dalyr[high_col] = (TRD_Dalyr[group_col] == 2).astype("int8")

    # (2) 市值加权版本：按照提示公司市值给 cue 加权
    for suffix in ["S90p", "S5090"]:
        value_col = f"nrFirmsAnnLag{i}VW{suffix}"
        group_col = f"interferenceGroupVWLag{i}{suffix}"
        low_col = f"lowInterferenceVWLag{i}{suffix}"
        high_col = f"highInterferenceVWLag{i}{suffix}"

        TRD_Dalyr[group_col] = by_year_xtile_2(TRD_Dalyr, value_col)
        TRD_Dalyr[low_col] = (TRD_Dalyr[group_col] == 1).astype("int8")
        TRD_Dalyr[high_col] = (TRD_Dalyr[group_col] == 2).astype("int8")


# 把 cue 的盈余惊喜变量分成五分位
def xtile_nq(series, q):
    """
    按分位点阈值分组，重复值很多时，允许某些组为空，不把整列设为缺失。
    """
    out = pd.Series(np.nan, index=series.index)
    nonmissing = series.dropna()

    if nonmissing.empty:
        return out

    probs = [k / q for k in range(1, q)]
    cutoffs = nonmissing.quantile(probs).to_numpy()

    values = nonmissing.to_numpy()
    groups = np.searchsorted(cutoffs, values, side="right") + 1

    out.loc[nonmissing.index] = groups
    return out


# 当前只处理 lag1，但循环可以扩展到其他 lag
for i in range(1, 2):
    for sue_id in [1, 2, 3]:
        TRD_Dalyr[f"sue{sue_id}CueLag{i}VWS90p_quintile"] = xtile_nq(
            TRD_Dalyr[f"sue{sue_id}CueLag{i}S90pVW"],
            q=5
        )# 大市值 cue 的市值加权盈余惊喜五分位

        TRD_Dalyr[f"sue{sue_id}CueLag{i}EWS90p_quintile"] = xtile_nq(
            TRD_Dalyr[f"sue{sue_id}CueLag{i}S90pEW"],
            q=5
        )# 大市值 cue 的等权盈余惊喜五分位

TRD_Dalyr["marketCap_Minus3_Recall"] = TRD_Dalyr["mktcap_tm3"]

TRD_Dalyr["marketCap_Minus3_Recall_medGr"] = by_year_xtile_2(
    TRD_Dalyr,
    "marketCap_Minus3_Recall",
    year_col="year"
) # 按年份把被提示公司市值分成高低两组

TRD_Dalyr["smallFirm"] = (
    TRD_Dalyr["marketCap_Minus3_Recall_medGr"] == 1
).astype("int8") # 如果被提示公司市值低于当年中位数，则 smallFirm = 1


# 缩尾方式是：全样本 1% winsorize（缩尾），不是按年份缩尾。
# 也就是把变量低于第 1 百分位的值替换为第 1 百分位，把高于第 99 百分位的值替换为第 99 百分位，
# 原变量保留，新生成 _w 变量。
print("开始缩尾......")
def winsorize_limits(s, p=0.01):
    """
    只计算缩尾阈值，不立刻生成缩尾后的整列。
    """
    lower = s.quantile(p)
    upper = s.quantile(1 - p)
    return lower, upper


def add_winsor_col(df, col):
    """
    使用预先在全样本上算好的缩尾阈值生成 _w 变量。
    """
    lower, upper = winsor_limits[col]
    df[f"{col}_w"] = df[col].clip(lower=lower, upper=upper)


# 对主要因变量进行 1% winsorize
dep_vars = [
    "chAdjreturn_tm3_Recall",
    "chAdjreturn_tm2_Recall",
    "chAdjreturn_tm1_Recall",
    "chAdjreturn_t_Recall",
]

dep_vars += [
    f"chAdjreturn_t_tp{i}_Recall"
    for i in range(1, 51)
]

# 对 cue 盈余惊喜变量进行 1% winsorize
sue_winsor_cols = []
for i in range(1, 2):
    for sue_id in [1, 2, 3]:
        for weight in ["VW", "EW"]:
            sue_winsor_cols.append(f"sue{sue_id}CueLag{i}S90p{weight}")

# 原文缩尾变量清单：收益率窗口 + cue 盈余惊喜
winsor_source_cols = [
    col
    for col in dep_vars + sue_winsor_cols
    if col in TRD_Dalyr.columns
]

# 注意：阈值仍然在筛基准样本前的全样本上计算
winsor_limits = {}
for col in winsor_source_cols:
    winsor_limits[col] = winsorize_limits(TRD_Dalyr[col], p=0.01)

# 筛基准样本前，只先生成后面马上要用的两个因变量缩尾列
for col in ["chAdjreturn_t_tp1_Recall", "chAdjreturn_t_tp10_Recall"]:
    if col in winsor_limits:
        add_winsor_col(TRD_Dalyr, col)

# 生成日历季度和年度-季度固定效应变量
TRD_Dalyr["quarter"] = TRD_Dalyr["Trddt"].dt.quarter
TRD_Dalyr["yearQuarter"] = TRD_Dalyr["Trddt"].dt.to_period("Q")



# 按被提示公司和日期排序，并把核心标识变量放在前面
front_cols = ["StkcdRecall", "Trddt"]
other_cols = [c for c in TRD_Dalyr.columns if c not in front_cols]

TRD_Dalyr = (
    TRD_Dalyr[front_cols + other_cols]
    .sort_values(["StkcdRecall", "Trddt"])
    .reset_index(drop=True)
)

def baseline_regression_sample_mask(df, dep_col, rhs_cols, fe_cols, cluster_cols=None):
    """
    1. 回归变量、固定效应变量、聚类变量不能缺失；
    2. 吸收固定效应后，删除 singleton（单例组），即固定效应组内只有 1 条观测的组。
    """
    cluster_cols = cluster_cols or []
    needed_cols = [dep_col] + rhs_cols + fe_cols + cluster_cols

    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少基准回归变量，不能复刻 e(sample)：{missing_cols}")

    mask = df[needed_cols].notna().all(axis=1)

    fe_key = (
        df.loc[mask, fe_cols]
        .astype(str)
        .agg("×".join, axis=1)
    )

    fe_count = fe_key.value_counts()
    keep_index = fe_key.loc[fe_key.map(fe_count) > 1].index

    out = pd.Series(False, index=df.index)
    out.loc[keep_index] = True
    return out


# 准备控制变量
own_ea_cols = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)

own_disclosure_cols = [
    "ownMajorDisclosure",
    "ownPerformanceForecast",
]

baseline_rhs = (
    ["quarterLag1size90plus"]
    + own_ea_cols
    + own_disclosure_cols
    + ["aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV"]
)

# [t, t+1] 基准回归样本
mask_tp1 = baseline_regression_sample_mask(
    TRD_Dalyr,
    dep_col="chAdjreturn_t_tp1_Recall_w",
    rhs_cols=baseline_rhs,
    fe_cols=["StkcdRecall", "yearQuarter"],
    cluster_cols=["StkcdRecall", "Trddt"],
)

TRD_Dalyr = TRD_Dalyr.loc[mask_tp1].copy()


# [t, t+10] 基准回归样本
mask_tp10 = baseline_regression_sample_mask(
    TRD_Dalyr,
    dep_col="chAdjreturn_t_tp10_Recall_w",
    rhs_cols=baseline_rhs,
    fe_cols=["StkcdRecall", "yearQuarter"],
    cluster_cols=["StkcdRecall", "Trddt"],
)

TRD_Dalyr = TRD_Dalyr.loc[mask_tp10].copy()

#===============================================================================
# 统计：一次提示公司的公告会提示多少家被提示公司
#===============================================================================

# 一、提取最终回归样本中的“被提示公司 × 交易日”
firm_days_in_sample = (
    TRD_Dalyr[
        ["StkcdRecall", "Trddt"]
    ]
    .drop_duplicates()
    .copy()
)

# 二、使用记忆矩阵折叠前提取的基准大市值提示事件配对表。

# 三、只保留确实存在于最终回归样本中的被提示公司日
cue_pairs_final_sample = cue_pairs.merge(
    firm_days_in_sample,
    on=["StkcdRecall", "Trddt"],
    how="inner",
    validate="many_to_one",
)

# 四、检查“提示公司 × 日期 × 被提示公司”是否唯一
duplicate_count = cue_pairs_final_sample.duplicated(
    ["Stkcd", "Trddt", "StkcdRecall"]
).sum()

if duplicate_count != 0:
    raise ValueError(
        "提示公司、日期和被提示公司的组合存在重复，请先检查配对明细。"
    )

# 五、按“提示公司 × 公告交易日”统计被提示公司数量
nr_firms_cued = (
    cue_pairs_final_sample
    .groupby(
        ["Stkcd", "Trddt"],
        observed=True,
    )["StkcdRecall"]
    .nunique()
    .rename("nrFirmsCued")
)

# 六、生成与原文 summarize, detail 类似的描述性统计
nr_firms_cued_summary = pd.Series(
    {
        "提示公司公告次数": nr_firms_cued.shape[0],
        "均值": nr_firms_cued.mean(),
        "第25百分位": nr_firms_cued.quantile(0.25),
        "中位数": nr_firms_cued.median(),
        "第75百分位": nr_firms_cued.quantile(0.75),
        "标准差": nr_firms_cued.std(ddof=1),
        "最小值": nr_firms_cued.min(),
        "最大值": nr_firms_cued.max(),
    },
    name="每次公告提示的公司数量",
)

print("\n一次提示公司公告会提示多少家被提示公司：")
print(nr_firms_cued_summary)




print("基准样本筛选后补生成其余缩尾变量......")

for col in winsor_source_cols:
    w_col = f"{col}_w"
    if col in TRD_Dalyr.columns and w_col not in TRD_Dalyr.columns:
        add_winsor_col(TRD_Dalyr, col)

print("最终同时进入两个基准回归的样本量：", len(TRD_Dalyr))
print("最终 cue 数量：", TRD_Dalyr["quarterLag1size90plus"].sum())

#===============================================================================
# 构造 future cue 虚拟变量：cue_tp1 到 cue_tp10
# cue_tp{i}=1 表示当前日后第 i 个交易日会发生基准 cue
#===============================================================================

# 当前主面板唯一层级应为 StkcdRecall × td
dup_n = TRD_Dalyr.duplicated(["StkcdRecall", "td"]).sum()
print("构造 future cue 前 StkcdRecall-td 重复数：", dup_n)

if dup_n != 0:
    raise ValueError("TRD_Dalyr 中 StkcdRecall-td 不唯一，不能构造 future cue。")

# 提取真实 cue 发生日
cue_days = (
    TRD_Dalyr.loc[
        TRD_Dalyr["quarterLag1size90plus"].eq(1),
        ["StkcdRecall", "td"]
    ]
    .drop_duplicates()
    .copy()
)

print("基准 cue 事件数：", len(cue_days))

for i in range(1, 11):
    # 如果 cue 发生在 td = T，
    # 那么 td = T - i 的那一行应标记 cue_tp{i}=1
    future_cue = cue_days.copy()
    future_cue["td"] = future_cue["td"] - i

    future_cue = (
        future_cue
        .drop_duplicates(["StkcdRecall", "td"])
        .assign(**{f"cue_tp{i}": 1})
    )

    TRD_Dalyr = TRD_Dalyr.merge(
        future_cue,
        on=["StkcdRecall", "td"],
        how="left",
        validate="1:1"
    )

    TRD_Dalyr[f"cue_tp{i}"] = (
        TRD_Dalyr[f"cue_tp{i}"]
        .fillna(0)
        .astype("int8")
    )

print("future cue 虚拟变量生成完成。")
print(TRD_Dalyr[[f"cue_tp{i}" for i in range(1, 11)]].sum())

#===============================================================================
# 删除最终输出不需要的中间变量
#===============================================================================

# 确认原始收益率已经按原文命名，避免误删 raw_ret_* 里的核心收益变量
expected_return_cols = (
    ["return_tm3_Recall", "return_tm2_Recall", "return_tm1_Recall", "return_t_Recall"]
    + [f"return_t_tp{i}_Recall" for i in range(1, 51)]
)

missing_return_cols = [
    c for c in expected_return_cols
    if c not in TRD_Dalyr.columns
]

if missing_return_cols:
    raise ValueError(
        "先把 raw_ret_* 重命名为 return_*_Recall，再删除 raw_ret_* 辅助列。"
        f" 当前缺少：{missing_return_cols[:10]}"
    )

drop_cols_final = [
    # 新闻控制变量直接使用上游构造结果，最终保留四个变化控制变量
    "AES",
    "AEV",

    # smallFirm 和 interference 高低组的中间分组变量
    "marketCap_Minus3_Recall_medGr",
    "interferenceGroupEWLag1S90p",
    "interferenceGroupEWLag1S5090",
    "interferenceGroupVWLag1S90p",
    "interferenceGroupVWLag1S5090",

    # A 股自身公告控制变量的来源字段，最终只保留 ownMajorDisclosure / ownPerformanceForecast
    "ownPerformanceForecastBrief",
    "ownOtherEventER",

    # 日度面板和 DGTW 构造中的辅助字段
    "ret",
    "retn",
    "Markettype",
    "size",
    "sizew",
    "trade_days_1m",
    "trade_days_1y",
    "mktcap_tm3",
    "float_mktcap_tm3",
    "mktcap_tm3_valid",
    "float_mktcap_tm3_valid",
    "portfolioNumber",
]

# raw_ret_* 中剩下的应只是不再需要的 nobs / valid_full 等辅助字段
drop_cols_final += [
    c for c in TRD_Dalyr.columns
    if c.startswith("raw_ret_")
]

drop_cols_final = list(dict.fromkeys(drop_cols_final))
drop_cols_existing = [c for c in drop_cols_final if c in TRD_Dalyr.columns]

print("最终输出前删除中间变量数量：", len(drop_cols_existing))
print(drop_cols_existing)

TRD_Dalyr = TRD_Dalyr.drop(columns=drop_cols_existing)

#===============================================================================
# 变量中文标签
#===============================================================================

variable_labels = {
    "StkcdRecall": "被提示公司股票代码",
    "Trddt": "交易日期",
    "td": "交易日序号",
    "marketCap_Minus3_Recall": "被提示公司 t-3 日总市值",

    "return_t_Recall": "被提示公司 t 日原始收益率",
    "ownEA": "自身盈余公告日虚拟变量，等于1表示被提示公司在 t 日发布盈余公告",

    "aesIncrease": "新闻情绪上升虚拟变量，使用上游真实交易日控制表口径",
    "aesDecrease": "新闻情绪下降虚拟变量，使用上游真实交易日控制表口径",
    "logIncreaseAEV": "新闻覆盖量上升幅度的对数值，log(1+上升量)",
    "logDecreaseAEV": "新闻覆盖量下降幅度的对数值，log(1+下降量)",
    "ownMajorDisclosure": "被提示公司专题研究事件控制变量，十类专题事件的并集",
    "ownPerformanceForecast": "被提示公司业绩预告或快报控制变量",

    "nrCuesLag1S90p": "lag1 大市值提示公司数量，提示公司市值高于90分位",
    "nrCuesLag1S5090": "lag1 中高市值提示公司数量，提示公司市值位于50-90分位",
    "nrCuesLag1S1050": "lag1 中低市值提示公司数量，提示公司市值位于10-50分位",
    "nrCuesLag1S10m": "lag1 小市值提示公司数量，提示公司市值低于10分位",

    "quarterLag1size90plus": "基准 cue 虚拟变量，等于1表示存在 lag1 大市值提示事件",
    "quarterLag1size50_90": "中高市值 cue 虚拟变量，等于1表示存在50-90分位市值提示事件",
    "quarterLag1size10_50": "中低市值 cue 虚拟变量，等于1表示存在10-50分位市值提示事件",
    "quarterLag1size10minus": "小市值 cue 虚拟变量，等于1表示存在10分位以下市值提示事件",

    "lowInterferenceEWLag1S90p": "低干扰大市值 cue 虚拟变量，等权 interference 分组为低组",
    "highInterferenceEWLag1S90p": "高干扰大市值 cue 虚拟变量，等权 interference 分组为高组",
    "lowInterferenceVWLag1S90p": "低干扰大市值 cue 虚拟变量，市值加权 interference 分组为低组",
    "highInterferenceVWLag1S90p": "高干扰大市值 cue 虚拟变量，市值加权 interference 分组为高组",

    "nrFirmsAnnLag1EWS90p": "lag1 大市值 cue 的共同公告公司数量，按 cue 等权平均",
    "nrFirmsAnnLag1EWS5090": "lag1 中高市值 cue 的共同公告公司数量，按 cue 等权平均",
    "nrFirmsAnnLag1VWS90p": "lag1 大市值 cue 的共同公告公司数量，按提示公司市值加权平均",
    "nrFirmsAnnLag1VWS5090": "lag1 中高市值 cue 的共同公告公司数量，按提示公司市值加权平均",

    "smallFirm": "小市值被提示公司虚拟变量，等于1表示被提示公司 t-3 市值低于当年中位数",
    "quarter": "交易日期所属日历季度",
    "yearQuarter": "交易日期所属年度-季度",
    "year": "交易日期所属年份",
}

for i in range(1, 4):
    variable_labels[f"return_tm{i}_Recall"] = f"被提示公司 t-{i} 日原始收益率"
    variable_labels[f"chAdjreturn_tm{i}_Recall"] = f"被提示公司 t-{i} 日特征调整收益率"

for i in range(1, 51):
    variable_labels[f"return_t_tp{i}_Recall"] = f"被提示公司 [t, t+{i}] 窗口原始累计收益率"
    variable_labels[f"chAdjreturn_t_tp{i}_Recall"] = (
        f"被提示公司 [t, t+{i}] 窗口特征调整累计收益率"
    )

for i in range(1, 11):
    variable_labels[f"ownEA_tp{i}"] = f"未来自身公告控制变量，等于1表示被提示公司在 t+{i} 日发布盈余公告"
    variable_labels[f"ownEA_tm{i}"] = f"过去自身公告控制变量，等于1表示被提示公司在 t-{i} 日发布盈余公告"
    variable_labels[f"cue_tp{i}"] = f"未来 cue 控制变量，等于1表示被提示公司在 t+{i} 日发生基准 cue"

for sue_id in [1, 2, 3]:
    variable_labels[f"sue{sue_id}CueLag1S90pVW"] = (
        f"lag1 大市值提示公司的 SUE{sue_id} 盈余惊喜，按提示公司市值加权平均"
    )
    variable_labels[f"sue{sue_id}CueLag1S90pEW"] = (
        f"lag1 大市值提示公司的 SUE{sue_id} 盈余惊喜，按 cue 等权平均"
    )
    variable_labels[f"sue{sue_id}CueLag1VWS90p_quintile"] = (
        f"lag1 大市值提示公司市值加权 SUE{sue_id} 盈余惊喜五分位"
    )
    variable_labels[f"sue{sue_id}CueLag1EWS90p_quintile"] = (
        f"lag1 大市值提示公司等权 SUE{sue_id} 盈余惊喜五分位"
    )

print("正在输出最终文件......")

TRD_Dalyr.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d"
)

print("最终文件输出成功!")
