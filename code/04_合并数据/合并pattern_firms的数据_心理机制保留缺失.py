"""模式公司日度面板：保留心理机制无法判断时的缺失值。

阅读顺序：
    1. 先找到文件末尾的 main（主流程函数），看四个阶段按什么顺序执行。
    2. 想了解某一步时，再按函数名找到上方的定义；每个函数开头都说明输入、处理和输出。
    3. 函数里的中文注释按处理顺序排列，可以边看代码边想象表格怎样变化。
    4. def（定义函数）组织处理步骤，return（返回）把结果交回主流程。
       单独定义函数不会读写数据；直接运行本文件才会调用主流程。

主要表名：
    mm（记忆矩阵）：一行是“提示公司×原公告日×被提示公司”。
    events（历史公告特征表）：一行是“模式公司×原公告日”。
    conditions（历史机制条件表）：一行是同一原公告日的一对公司。
    cue_events（提示事件表）：一行是“被提示公司×事件交易日”。
    panel（日度面板）：一行是“被提示公司×交易日”。
    Stkcd（提示公司代码），StkcdRecall（被提示公司代码）。

常见写法：
    表["列名"]：取出一列；表["新列"] = 计算结果：新增或覆盖一列。
    .merge（合并）：按指定键把外部变量接到当前表。
    .groupby（分组）：同一家公司、日期等归为一组，再计算数量、均值等。
    .astype（转换数据类型）：例如把判断的真假转换成1和0。
    .loc（按条件取行列）：在满足条件的记录上取值或赋值。
    int8（普通8位整数）用于没有缺失的标记，Int8（可空8位整数）允许保留未知值。

与同目录原合并脚本的主要区别：
    从原始公告特征重建机制条件，保留“有提示但条件无法判断”的缺失，
    并单独输出到“中间结果/04_合并数据/心理机制保留缺失/”。

沿用当前默认研究口径：
    使用上月市值断点；剔除同行业、供应链、过去365天共同分析师覆盖公司对；
    分析师编号拆为个人；各盈余惊喜均值使用自身非缺失提示作为分母；
    历史价格异动条件是任意一家公司进入前100名。
    Annodt（公告日期）已考虑收盘后披露，交易日公告保留当天，
    仅非交易日公告映射到下一交易日；新闻特征仍按原公告自然日合并。
    364天回溯窗口、财报截止日季度距离、基准筛样顺序和缩尾口径保持原样。

机制缺失规则：
    无对应规模提示，机制提示标记为0；连续机制指标仍保留缺失。
    有提示且条件已知，满足为1，确认不满足为0；无法判断为缺失。
    同时要求多个条件时，任意一个已知为0，就能确定结果为0。
    汇总多条记录时：有1则1，全0才为0，没有1但仍有未知则保留缺失。
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd


# =============================================================================
# 路径设置：需要搬动项目时，先检查这里
# =============================================================================

root = Path(__file__).resolve().parents[2]
matrix_path = root / "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv"
daily_path = root / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv"
calendar_path = root / "中间结果/01_清洗后样本/市场交易日历_td.csv"
report_path = root / "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv"
pattern_path = root / "中间结果/02_变量构造结果/6PatternFirmQtr/PatternFirmQtr.csv"
breakpoint_path = root / "中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv"
industry_path = root / "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv"
supply_path = root / "input/CSMAR/供应链网络关系指标表2001-12-31至2025-12-31/SC_NetworkRelationsIndex.csv"
analyst_path = root / "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后/拆开AnanmID/Pattern样本_分析师共同覆盖_滚动365天.csv"
sue_path = root / "中间结果/02_变量构造结果/4sue/sue.csv"
news_feature_path = root / "中间结果/02_变量构造结果/3报刊新闻_AES_AEV/报刊新闻_AES_AEV.csv"
volume_path = root / "中间结果/02_变量构造结果/2交易行为OIB、Avol/Avol_CSMAR_prep.csv"
forecast_path = root / "中间结果/02_变量构造结果/5公司公告事件/业绩预告快报控制变量_公告日期口径.csv"
other_event_path = root / "中间结果/02_变量构造结果/5公司公告事件/专题研究事件控制变量_公告日期口径.csv"
# 四个回归控制使用数行者分段新闻；上方历史高新闻覆盖特征仍使用独立来源。
news_control_path = root / "中间结果/02_变量构造结果/3分段新闻_AES_AEV/DATAGO_分段新闻_真实交易日控制变量.csv"
adjusted_return_path = root / "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW特征调整收益率.csv"

output_dir = root / "中间结果/04_合并数据/心理机制保留缺失"
cue_output = output_dir / "Pattern_cue_events_after_collapse_心理机制保留缺失.csv"
final_output = output_dir / "Pattern样本_最终回归面板_心理机制保留缺失.csv"

# =============================================================================
# 公共列名清单：只存列名和对应关系，不存数据表
# =============================================================================

# 公司对匹配键：原公告日期、提示公司代码、被提示公司代码。
PAIR_KEYS = ["Annodt", "Stkcd", "StkcdRecall"]

# 四个规模组的基础提示标记，以及各组的提示数量。
BASE_CUE_COLS = [
    "quarterLag1size90plus", "quarterLag1size50_90",
    "quarterLag1size10_50", "quarterLag1size10minus",
]
COUNT_COLS = ["nrCuesLag1S90p", "nrCuesLag1S5090", "nrCuesLag1S1050", "nrCuesLag1S10m"]

# 每行两项：最终机制变量的前缀、需要读取的历史机制列。
# 最后一行保留原变量名称，但来源使用“任意一家公司价格异动”。
MECHANISM_SOURCES = [
    ("lag1hiAEV", "lag1hiAEV"),
    ("lag1hiavol", "lag1hiavol"),
    ("lag1hiavolS", "lag1hiavolS"),
    ("lag1hi_sue1_abs", "lag1hi_sue1_abs"),
    ("lag1hi_sue2_abs", "lag1hi_sue2_abs"),
    ("lag1hi_sue3_abs", "lag1hi_sue3_abs"),
    ("lag1sue1pos", "lag1sue1pos"),
    ("lag1sue2pos", "lag1sue2pos"),
    ("lag1sue3pos", "lag1sue3pos"),
    ("lag1sue1neg", "lag1sue1neg"),
    ("lag1sue2neg", "lag1sue2neg"),
    ("lag1sue3neg", "lag1sue3neg"),
    ("lag1hi_sue1_pos", "lag1hi_sue1_pos"),
    ("lag1hi_sue2_pos", "lag1hi_sue2_pos"),
    ("lag1hi_sue3_pos", "lag1hi_sue3_pos"),
    ("lag1hi_sue1_neg", "lag1hi_sue1_neg"),
    ("lag1hi_sue2_neg", "lag1hi_sue2_neg"),
    ("lag1hi_sue3_neg", "lag1hi_sue3_neg"),
    ("lag1topMover", "lag1anyTopMover"),
]

# 将同一组机制分别用于大市值提示和中等市值提示，形成两份列名清单。
LARGE_MECHANISM_COLS = [prefix + "size90plus" for prefix, source in MECHANISM_SOURCES]

MEDIUM_MECHANISM_COLS = [prefix + "size50_90" for prefix, source in MECHANISM_SOURCES]

MECHANISM_COLS = LARGE_MECHANISM_COLS + MEDIUM_MECHANISM_COLS

# 每行五项：干扰均值列、年度分组列、低组标记、高组标记、对应的基础提示列。
INTERFERENCE_SPECS = [
    (
        "nrFirmsAnnLag1EWS90p", "interferenceGroupEWLag1S90p",
        "lowInterferenceEWLag1S90p", "highInterferenceEWLag1S90p",
        "quarterLag1size90plus",
    ),
    (
        "nrFirmsAnnLag1VWS90p", "interferenceGroupVWLag1S90p",
        "lowInterferenceVWLag1S90p", "highInterferenceVWLag1S90p",
        "quarterLag1size90plus",
    ),
    (
        "nrFirmsAnnLag1EWS5090", "interferenceGroupEWLag1S5090",
        "lowInterferenceEWLag1S5090", "highInterferenceEWLag1S5090",
        "quarterLag1size50_90",
    ),
    (
        "nrFirmsAnnLag1VWS5090", "interferenceGroupVWLag1S5090",
        "lowInterferenceVWLag1S5090", "highInterferenceVWLag1S5090",
        "quarterLag1size50_90",
    ),
]

# 每行两项：提示盈余惊喜均值列、对应的全样本五分位组列。
SUE_QUINTILES = [
    ("sue1CueLag1S90pVW", "sue1CueLag1VWS90p_quintile"),
    ("sue1CueLag1S90pEW", "sue1CueLag1EWS90p_quintile"),
    ("sue2CueLag1S90pVW", "sue2CueLag1VWS90p_quintile"),
    ("sue2CueLag1S90pEW", "sue2CueLag1EWS90p_quintile"),
    ("sue3CueLag1S90pVW", "sue3CueLag1VWS90p_quintile"),
    ("sue3CueLag1S90pEW", "sue3CueLag1EWS90p_quintile"),
]

# =============================================================================
# 小工具：股票代码、分位分组和保留缺失的机制汇总
# =============================================================================

def normalize_stock_code(value):
    """股票代码统一成六位字符串；空值沿用原口径转成空字符串。"""
    if pd.isna(value) or str(value).strip() == "":
        return ""
    return f"{int(float(str(value).strip())):06d}"


def xtile_by_group(values, n):
    """沿用原分位分组算法；缺失不参与计算，并列值不拆分。"""
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.dropna()
    count = len(valid)
    if count == 0:
        return result
    sorted_values = np.sort(valid.to_numpy(dtype="float64"))
    cutpoints = []
    for k in range(1, n):
        position, remainder = divmod(count * k, n)
        if remainder == 0:
            cutpoint = (sorted_values[position - 1] + sorted_values[position]) / 2
        else:
            cutpoint = sorted_values[position]
        cutpoints.append(cutpoint)
    result.loc[valid.index] = np.searchsorted(
        cutpoints, valid.to_numpy(dtype="float64"), side="left"
    ) + 1
    return result


def aggregate_flags(data, keys, columns):
    """多条机制记录取并集：有1则1，全0则0，其余保留缺失。"""
    grouped = data.groupby(keys, sort=False, dropna=False)
    maximum = grouped[columns].max()
    valid_count = grouped[columns].count()
    total_count = grouped.size()
    has_unknown = valid_count.lt(total_count, axis=0)
    # 普通最大值跳过缺失；把“0和未知”重新恢复成未知。
    result = maximum.mask(maximum.ne(1) & has_unknown)
    return result.astype("Int8").reset_index()

# =============================================================================
# 读取并补充公司对记忆矩阵
# =============================================================================

def prepare_memory_matrix(
    matrix_path,
    daily_path,
    breakpoint_path,
    sue_path,
    report_path,
):
    """读取记忆矩阵，合并提示公司市值、上月断点和当前盈余惊喜。
    输入：公司对记忆矩阵，以及市值、断点、财报、盈余惊喜文件。
    处理：按公司和事件前三日匹配市值，按上月匹配断点，按原公告匹配当前报告。
    输出：仍为“提示公司×原公告日×被提示公司”；删除必要市值字段缺失的行。
    新增变量：事件前三日市值、上月市值断点、当前报告截止日及三种盈余惊喜。
    """

    # 1. 读取记忆矩阵，合并提示公司事件前三个交易日的市值
    # 本段前后都是：提示公司 × 原公告日 × 被提示公司。
    mm = pd.read_csv(
        matrix_path, encoding="utf-8", parse_dates=["Annodt", "event_Trddt"]
    )
    mm["Stkcd"] = mm["Stkcd"].apply(normalize_stock_code)
    mm["StkcdRecall"] = mm["StkcdRecall"].apply(normalize_stock_code)
    mm["event_Trddt"] = pd.to_datetime(mm["event_Trddt"], errors="coerce")
    mm["event_td"] = pd.to_numeric(mm["event_td"], errors="coerce")
    mm["td_tm3"] = mm["event_td"] - 3

    mktcap = pd.read_csv(
        daily_path, encoding="utf-8", usecols=["Stkcd", "td", "size", "sizew"],
        dtype={"Stkcd": "string"},
    )
    mktcap["Stkcd"] = mktcap["Stkcd"].apply(normalize_stock_code)
    mktcap["td"] = pd.to_numeric(mktcap["td"], errors="coerce")
    mktcap = mktcap.rename(columns={
        "td": "td_tm3", "size": "mktcap_tm3", "sizew": "float_mktcap_tm3",
    })
    mm = mm.merge(mktcap, on=["Stkcd", "td_tm3"], how="left", validate="m:1")
    del mktcap
    print("合并提示公司市值后的公司对记录数：", len(mm))

    # 2. 合并上月市值断点，删除必要市值字段缺失的记录
    marketcap_group = pd.read_csv(
        breakpoint_path, encoding="utf-8", parse_dates=["Trdmnt"]
    )
    mm["Trdmnt_for_breakpoint"] = (
        mm["event_Trddt"].dt.to_period("M") - 1
    ).dt.to_timestamp()
    marketcap_group = marketcap_group.rename(columns={"Trdmnt": "Trdmnt_breakpoint"})
    mm = mm.merge(
        marketcap_group,
        left_on="Trdmnt_for_breakpoint", right_on="Trdmnt_breakpoint",
        how="left", validate="m:1",
    )
    mm = mm.dropna(subset=[
        "mktcap_tm3", "marketCap_90th_percentile",
        "marketCap_50th_percentile", "marketCap_10th_percentile",
    ]).copy()
    del marketcap_group
    print("保留市值和必要断点完整的公司对记录数：", len(mm))

    # 3. 合并提示公司当前公告对应的盈余惊喜
    # 同一提示公司同一原公告日有多份报告时，沿用原规则选最早财报截止日。
    sue = pd.read_csv(
        sue_path, encoding="utf-8",
        usecols=["Stkcd", "Accper", "sue1", "sue2", "sue3"], parse_dates=["Accper"],
    )
    sue["Stkcd"] = sue["Stkcd"].apply(normalize_stock_code)
    current_accper = pd.read_csv(
        report_path, usecols=["Stkcd", "Annodt", "Accper"],
        dtype={"Stkcd": "string"}, parse_dates=["Annodt", "Accper"],
    )
    current_accper["Stkcd"] = current_accper["Stkcd"].apply(normalize_stock_code)
    current_accper = current_accper.sort_values(["Stkcd", "Annodt", "Accper"])
    current_accper = current_accper.groupby(["Stkcd", "Annodt"], as_index=False).head(1)
    current_accper = current_accper.rename(columns={"Accper": "currentAccper"})
    mm = mm.merge(current_accper, on=["Stkcd", "Annodt"], how="left", validate="m:1")
    mm = mm.merge(
        sue, left_on=["Stkcd", "currentAccper"], right_on=["Stkcd", "Accper"],
        how="left", validate="m:1",
    )
    mm = mm.drop(columns=["Accper"], errors="ignore")
    del sue, current_accper

    return mm


# =============================================================================
# 剔除其他关联关系
# =============================================================================

def filter_related_pairs(
    mm,
    industry_path,
    supply_path,
    analyst_path,
):
    """依次排除同行业、供应链和共同分析师覆盖的公司对。
    输入：已补充市值及当前盈余惊喜的公司对记忆矩阵。
    处理：先在精简名单上筛选，再按原公告日、提示公司、被提示公司匹配回完整矩阵。
    输出：行数可能减少；保留下来的公司对仍带有完整原始变量。
    口径：行业使用事件交易日，供应链使用原公告年份，分析师使用过去365天个人编号。
    """

    # 1. 剔除同行业公司对
    # 先在精简名单上筛选，最后把名单合并回完整记忆矩阵。
    mm_key = mm[["Annodt", "event_td", "Stkcd", "StkcdRecall"]].copy()
    industry = pd.read_csv(
        industry_path, encoding="utf-8", usecols=["Stkcd", "td", "IndustryCode"],
        dtype={"Stkcd": "string", "IndustryCode": "string"},
    )
    industry["Stkcd"] = industry["Stkcd"].apply(normalize_stock_code)
    industry["td"] = pd.to_numeric(industry["td"], errors="coerce")
    industry = industry.dropna(subset=["Stkcd", "td", "IndustryCode"])
    industry = industry.drop_duplicates(["Stkcd", "td"])
    cue_industry = industry.rename(columns={
        "td": "event_td", "IndustryCode": "cue_industry_code",
    })
    recall_industry = industry.rename(columns={
        "Stkcd": "StkcdRecall", "td": "event_td", "IndustryCode": "recall_industry_code",
    })
    mm_key = mm_key.merge(cue_industry, on=["Stkcd", "event_td"], how="left", validate="m:1")
    mm_key = mm_key.merge(
        recall_industry, on=["StkcdRecall", "event_td"], how="left", validate="m:1"
    )
    # 双方行业均已知且相同才删除；行业无法识别的公司对继续保留。
    same_industry = (
        mm_key["cue_industry_code"].notna()
        & mm_key["recall_industry_code"].notna()
        & mm_key["cue_industry_code"].eq(mm_key["recall_industry_code"])
    )
    mm_key = mm_key.loc[~same_industry].copy()
    del industry, cue_industry, recall_industry
    print("剔除同行业后公司对记录数：", len(mm_key))

    # 2. 剔除存在任一方向供应链关系的公司对
    # 使用原公告年份；关联方代码先按原始六位格式筛选，再识别关系。
    mm_key["yearCue"] = mm_key["Annodt"].dt.year
    supply = pd.read_csv(
        supply_path, encoding="utf-8",
        usecols=["Symbol", "EndDate", "StateTypeCode", "PSCBusinessRelation", "PSCSymbol"],
        dtype={
            "Symbol": "string", "StateTypeCode": "string",
            "PSCBusinessRelation": "string", "PSCSymbol": "string",
        },
        parse_dates=["EndDate"],
    )
    supply["yearCue"] = supply["EndDate"].dt.year
    supply["Symbol"] = supply["Symbol"].apply(normalize_stock_code)
    keep_supply = (
        supply["StateTypeCode"].eq("1")
        & supply["PSCSymbol"].str.fullmatch(r"[0368][0-9]{5}", na=False)
        & supply["PSCBusinessRelation"].isin(["1", "2"])
    )
    supply = supply.loc[keep_supply].copy()
    supply["upstream_stkcd"] = np.where(
        supply["PSCBusinessRelation"].eq("1"), supply["Symbol"], supply["PSCSymbol"]
    )
    supply["downstream_stkcd"] = np.where(
        supply["PSCBusinessRelation"].eq("1"), supply["PSCSymbol"], supply["Symbol"]
    )
    supply = supply.loc[
        supply["upstream_stkcd"].ne(supply["downstream_stkcd"]),
        ["yearCue", "upstream_stkcd", "downstream_stkcd"],
    ].drop_duplicates()
    forward_pairs = supply.rename(columns={
        "upstream_stkcd": "Stkcd", "downstream_stkcd": "StkcdRecall",
    })
    reverse_pairs = supply.rename(columns={
        "downstream_stkcd": "Stkcd", "upstream_stkcd": "StkcdRecall",
    })
    vertical_pairs = pd.concat([forward_pairs, reverse_pairs], ignore_index=True).drop_duplicates()
    vertical_pairs["vertical_related"] = 1
    mm_key = mm_key.merge(
        vertical_pairs, on=["yearCue", "Stkcd", "StkcdRecall"], how="left", validate="m:1"
    )
    mm_key = mm_key.loc[mm_key["vertical_related"].isna()].copy()
    mm_key = mm_key.drop(columns=["yearCue", "vertical_related"])
    del supply, forward_pairs, reverse_pairs, vertical_pairs
    print("剔除供应链关系后公司对记录数：", len(mm_key))

    # 3. 剔除过去365天存在共同分析师覆盖的公司对
    # 来源表仅保存共同覆盖配对，沿用“标记非缺失即匹配”口径。
    analyst = pd.read_csv(
        analyst_path, encoding="utf-8-sig",
        usecols=["Annodt", "Stkcd", "StkcdRecall", "same_analyst_overlap", "n_common_analysts"],
        dtype={
            "Stkcd": "string", "StkcdRecall": "string",
            "same_analyst_overlap": "Int64", "n_common_analysts": "Int64",
        },
        parse_dates=["Annodt"],
    )
    analyst["Stkcd"] = analyst["Stkcd"].apply(normalize_stock_code)
    analyst["StkcdRecall"] = analyst["StkcdRecall"].apply(normalize_stock_code)
    analyst = analyst.drop_duplicates(PAIR_KEYS)
    mm_key = mm_key.merge(analyst, on=PAIR_KEYS, how="left", validate="m:1")
    mm_key = mm_key.loc[mm_key["same_analyst_overlap"].isna(), PAIR_KEYS].drop_duplicates()
    mm = mm.merge(mm_key, on=PAIR_KEYS, how="inner", validate="1:1")
    del analyst, mm_key
    print("完成三轮公司对筛选，剩余记录数：", len(mm))

    return mm


# =============================================================================
# 读取完整历史公告及特征
# =============================================================================

def read_history_features(
    report_path,
    pattern_path,
    calendar_path,
    news_feature_path,
    volume_path,
    sue_path,
):
    """读取历史模式公司公告，并合并新闻、成交量、盈余惊喜。
    输入：财报公告、模式公司季度名单、交易日历和三个公告特征文件。
    处理：确定完整历史公告样本，映射事件交易日，再按各来源的对应键左连接。
    输出：一行代表“模式公司×原公告日”；外部合并不增加行数。
    保留缺失：新闻和成交量缺失不填0，正负无穷值按缺失处理。
    """

    # 1. 读取历史公告，确定机制分组使用的公告样本
    # 本段得到：模式公司 × 原公告日。不能提前筛成上面留下的公司对。
    events = pd.read_csv(
        report_path, encoding="utf-8", usecols=["Stkcd", "Accper", "Annodt"],
        parse_dates=["Accper", "Annodt"],
    )
    events["Stkcd"] = events["Stkcd"].apply(normalize_stock_code)
    events = events.loc[events["Annodt"].ne(events["Accper"])].copy()
    # 历史公告沿用原记忆矩阵规则：同公司同公告日保留最晚财报截止日。
    events = events.sort_values(["Stkcd", "Annodt", "Accper"], kind="stable")
    events = events.drop_duplicates(["Stkcd", "Annodt"], keep="last").reset_index(drop=True)
    pattern_quarters = pd.read_csv(pattern_path, encoding="utf-8", parse_dates=["Accper"])
    pattern_quarters["Stkcd"] = pattern_quarters["Stkcd"].apply(normalize_stock_code)
    events = events.merge(pattern_quarters, on=["Stkcd", "Accper"], how="inner", validate="1:1")
    if events.empty:
        raise ValueError("公告表与模式公司季度表合并后没有匹配记录。")
    events = events.loc[events["PatternFirmQtr"].eq(1)].copy()
    events["date"] = events["Annodt"]
    del pattern_quarters

    calendar = pd.read_csv(
        calendar_path, encoding="utf-8", usecols=["Trddt", "td"], parse_dates=["Trddt"]
    ).sort_values("Trddt")
    if calendar[["Trddt", "td"]].isna().any().any():
        raise ValueError("交易日历的日期或序号存在缺失。")
    if calendar["Trddt"].duplicated().any() or calendar["td"].duplicated().any():
        raise ValueError("交易日历的日期或序号存在重复。")
    announcement_dates = events[["Annodt"]].dropna().drop_duplicates().sort_values("Annodt")
    date_map = pd.merge_asof(
        announcement_dates,
        calendar.rename(columns={"Trddt": "event_Trddt", "td": "event_td"}),
        left_on="Annodt", right_on="event_Trddt", direction="forward", allow_exact_matches=True,
    )
    events = events.merge(date_map, on="Annodt", how="left", validate="m:1")
    if events["event_Trddt"].isna().any():
        raise ValueError("部分公告无法映射到交易日，请检查交易日历范围。")
    events["Trddt"] = events["event_Trddt"]
    del calendar, announcement_dates, date_map

    # 2. 合并历史公告的新闻、成交量和盈余惊喜
    # 新闻按原公告自然日；成交量按事件交易日；盈余惊喜按财报截止日。
    # 来源缺失和无穷值保留为缺失，不在这里填0。
    news = pd.read_csv(
        news_feature_path, encoding="utf-8-sig", usecols=["Stkcd", "date", "AEV"],
        dtype={"Stkcd": "string"}, parse_dates=["date"],
    )
    news["Stkcd"] = news["Stkcd"].apply(normalize_stock_code)
    news["AEV"] = pd.to_numeric(news["AEV"], errors="raise").replace([np.inf, -np.inf], np.nan)
    events = events.merge(news, on=["Stkcd", "date"], how="left", validate="m:1")
    del news

    volume = pd.read_csv(
        volume_path, encoding="utf-8-sig", usecols=["Stkcd", "Trddt", "td", "avol", "avolS"],
        dtype={"Stkcd": "string"}, parse_dates=["Trddt"],
    )
    volume["Stkcd"] = volume["Stkcd"].apply(normalize_stock_code)
    for col in ["td", "avol", "avolS"]:
        volume[col] = pd.to_numeric(volume[col], errors="raise").replace([np.inf, -np.inf], np.nan)
    events = events.merge(volume, on=["Stkcd", "Trddt"], how="left", validate="m:1")
    wrong_td = events["td"].notna() & events["event_td"].ne(events["td"])
    if wrong_td.any():
        raise ValueError("异常成交量来源与公告事件的交易日序号不一致。")
    del volume

    earnings = pd.read_csv(
        sue_path, encoding="utf-8-sig", usecols=["Stkcd", "Accper", "sue1", "sue2", "sue3"],
        dtype={"Stkcd": "string"}, parse_dates=["Accper"],
    )
    earnings["Stkcd"] = earnings["Stkcd"].apply(normalize_stock_code)
    for col in ["sue1", "sue2", "sue3"]:
        earnings[col] = pd.to_numeric(earnings[col], errors="raise").replace([np.inf, -np.inf], np.nan)
    events = events.merge(earnings, on=["Stkcd", "Accper"], how="left", validate="m:1")
    del earnings

    return events


# =============================================================================
# 生成历史价格异动标记
# =============================================================================

def add_top_mover_flags(events, daily_path):
    """给历史公告合并当天的价格异动标记。
    输入：历史公告特征表，以及全市场个股日度数据。
    处理：逐日剔除有效样本中规模最低30%的股票，再按绝对收益率选前100名。
    输出：公告表行数不变，增加价格异动标记；来源无效时保留缺失。
    注意：排名使用所需交易日的全市场股票，完成排名后才保留事件所需公司。
    """

    # 按块读取以控制内存；每个所需交易日保留全市场股票参加排名。
    event_keys = events[["Stkcd", "Trddt"]].drop_duplicates()
    needed_dates = events["Trddt"].dropna().unique()
    pieces = []
    for chunk in pd.read_csv(
        daily_path, encoding="utf-8-sig", usecols=["Stkcd", "Trddt", "ret", "size"],
        dtype={"Stkcd": "string"}, parse_dates=["Trddt"], chunksize=500_000,
    ):
        selected = chunk.loc[chunk["Trddt"].isin(needed_dates)].copy()
        if not selected.empty:
            pieces.append(selected)

    if pieces:
        mover_daily = pd.concat(pieces, ignore_index=True)
        mover_daily["Stkcd"] = mover_daily["Stkcd"].apply(normalize_stock_code)
        if mover_daily.duplicated(["Stkcd", "Trddt"]).any():
            raise ValueError("价格异动来源存在重复公司日，无法唯一排名。")
        for col in ["ret", "size"]:
            mover_daily[col] = pd.to_numeric(mover_daily[col], errors="raise")
            mover_daily[col] = mover_daily[col].replace([np.inf, -np.inf], np.nan)
        # 来源有效但未进入前100名为0；收益率或规模缺失则为未知。
        known = mover_daily[["ret", "size"]].notna().all(axis=1)
        mover_daily["topMover"] = pd.Series(pd.NA, index=mover_daily.index, dtype="Int8")
        mover_daily.loc[known, "topMover"] = 0
        eligible = mover_daily.loc[known, ["Stkcd", "Trddt", "ret", "size"]].copy()
        size_p30 = eligible.groupby("Trddt")["size"].transform(lambda s: s.quantile(0.30))
        eligible = eligible.loc[eligible["size"].gt(size_p30)].copy()
        eligible["abs_ret"] = eligible["ret"].abs()
        eligible = eligible.sort_values(
            ["Trddt", "abs_ret", "size", "Stkcd"], ascending=[True, False, False, True]
        )
        rank = eligible.groupby("Trddt").cumcount() + 1
        mover_daily.loc[eligible.index[rank.le(100)], "topMover"] = 1
        top_mover = event_keys.merge(
            mover_daily[["Stkcd", "Trddt", "topMover"]],
            on=["Stkcd", "Trddt"], how="left", validate="1:1",
        )
        del mover_daily, eligible, known, size_p30, rank
    else:
        top_mover = event_keys.copy()
        top_mover["topMover"] = pd.Series(pd.NA, index=top_mover.index, dtype="Int8")

    events = events.merge(top_mover, on=["Stkcd", "Trddt"], how="left", validate="m:1")
    del pieces, top_mover, event_keys
    gc.collect()

    return events


# =============================================================================
# 构造同日公司对历史机制条件
# =============================================================================

def build_history_conditions(events):
    """从单家公司公告，构造同日公告公司对的历史机制条件。
    输入：一行一家公司公告的特征表。
    处理：按公告年份分组，生成高新闻、高成交量、极端及正负盈余标记，再同日配对。
    输出：一行代表“提示公司×原公告日×被提示公司”；配对会增加行数，删除自身配对。
    关键规则：可空布尔值保留“真、假、未知”，同时满足和任一满足分别使用交集与并集。
    """

    # 1. 生成历史公告特征标记
    # 年份取原公告日年份；缺失不参与分位点计算。
    # 可空布尔类型有“真、假、未知”三种状态，输出时再转成1、0、缺失。
    flags = events[["Stkcd", "Annodt", "Accper", "topMover"]].copy()
    flags["topMover"] = flags["topMover"].astype("boolean")
    announcement_year = events["Annodt"].dt.year
    attention_fields = [
        ("AEV", "hi_AEV"), ("avol", "hi_avol"), ("avolS", "hi_avolS"),
    ]
    for value_col, flag_col in attention_fields:
        groups = events[value_col].groupby(announcement_year).transform(lambda s: xtile_by_group(s, 2))
        flags[flag_col] = groups.eq(2).astype("boolean")

    # 三种盈余惊喜分别生成：绝对值最高十分位、正向、负向标记。
    for k in [1, 2, 3]:
        values = events[f"sue{k}"]
        groups = values.abs().groupby(announcement_year).transform(lambda s: xtile_by_group(s, 10))
        flags[f"hi_sue{k}_abs"] = groups.eq(10).astype("boolean")
        flags[f"sue{k}_pos"] = values.gt(0).astype("boolean").where(values.notna(), pd.NA)
        flags[f"sue{k}_neg"] = values.lt(0).astype("boolean").where(values.notna(), pd.NA)
    del events, announcement_year, values, groups

    # 2. 把同一原公告日的公司两两配对，生成机制条件
    # 表结构：提示公司 × 原公告日 × 被提示公司；保留有方向的公司对。
    if flags.duplicated(["Stkcd", "Annodt"]).any():
        raise ValueError("同一公司同一公告日存在多条记录，不能构造公司对。")
    recall_flags = flags.rename(columns={
        col: col + "Recall" for col in flags.columns if col != "Annodt"
    })
    pairs = recall_flags.merge(flags, on="Annodt", how="inner", validate="m:m")
    pairs = pairs.loc[pairs["Stkcd"].ne(pairs["StkcdRecall"])].copy()
    pairs = pairs.sort_values(["Annodt", "Stkcd", "StkcdRecall"]).reset_index(drop=True)
    conditions = pairs[["Annodt", "Stkcd", "StkcdRecall", "Accper"]].copy()

    # 每行依次为：生成的机制列、提示公司标记、被提示公司标记。
    memory_fields = [
        ("lag1hiAEV", "hi_AEV", "hi_AEVRecall"),
        ("lag1hiavol", "hi_avol", "hi_avolRecall"),
        ("lag1hiavolS", "hi_avolS", "hi_avolSRecall"),
        ("lag1hi_sue1_abs", "hi_sue1_abs", "hi_sue1_absRecall"),
        ("lag1hi_sue2_abs", "hi_sue2_abs", "hi_sue2_absRecall"),
        ("lag1hi_sue3_abs", "hi_sue3_abs", "hi_sue3_absRecall"),
        ("lag1sue1pos", "sue1_pos", "sue1_posRecall"),
        ("lag1sue2pos", "sue2_pos", "sue2_posRecall"),
        ("lag1sue3pos", "sue3_pos", "sue3_posRecall"),
        ("lag1sue1neg", "sue1_neg", "sue1_negRecall"),
        ("lag1sue2neg", "sue2_neg", "sue2_negRecall"),
        ("lag1sue3neg", "sue3_neg", "sue3_negRecall"),
        ("lag1topMover", "topMover", "topMoverRecall"),
    ]
    for output_col, cue_col, recall_col in memory_fields:
        # 同时满足：任一已知为假即可判为假；真与未知仍为未知。
        conditions[output_col] = pairs[cue_col] & pairs[recall_col]

    # 任意一家发生价格异动：有一个为真即可判为真。
    conditions["lag1anyTopMover"] = pairs["topMover"] | pairs["topMoverRecall"]

    # 把“双方极端正向”和“双方极端负向”分开，避免多层循环。
    for k in [1, 2, 3]:
        both_extreme = pairs[f"hi_sue{k}_abs"] & pairs[f"hi_sue{k}_absRecall"]
        conditions[f"lag1hi_sue{k}_pos"] = (
            both_extreme & pairs[f"sue{k}_pos"] & pairs[f"sue{k}_posRecall"]
        )
        conditions[f"lag1hi_sue{k}_neg"] = (
            both_extreme & pairs[f"sue{k}_neg"] & pairs[f"sue{k}_negRecall"]
        )
    del flags, recall_flags, pairs, both_extreme
    gc.collect()

    return conditions


# =============================================================================
# 重建上一季度机制
# =============================================================================

def rebuild_lag1_mechanisms(mm, conditions):
    """用原始公告特征重新构造上一季度机制，覆盖矩阵里的对应历史机制列。
    输入：通过公司对筛选的矩阵，以及同日公告公司对历史机制表。
    处理：逐个当前公告日，在含首尾的364天窗口中定位上一季度记录并汇总。
    输出：原矩阵公司对行数不变，更新历史机制列；基础提示关系不改变。
    季度距离：相对提示公司全部历史伙伴中的最大财报截止日，取大于0且不超过100天。
    必要检查：重建的基础季度关系必须与旧矩阵一致，否则停止，避免混入口径不同的结果。
    """

    mechanism_history_cols = [col for col in conditions.columns if col.startswith("lag1")]

    # 只更新原矩阵已有公司对的机制列；基础提示关系和样本名单不重建。
    if mm[PAIR_KEYS].isna().any().any() or mm.duplicated(PAIR_KEYS).any():
        raise ValueError("记忆矩阵的公司对公告日键存在缺失或重复。")
    conditions = conditions.sort_values("Annodt").reset_index(drop=True)
    history_dates = conditions["Annodt"].to_numpy(dtype="datetime64[ns]")
    rebuilt = []
    target_dates = mm.groupby("Annodt", sort=True)

    for position, (current_date, target_rows) in enumerate(target_dates, start=1):
        # 日期已排序，定位起止行以减少扫描；窗口仍为含首尾的364天回溯。
        window_start = current_date - pd.Timedelta(days=364)
        start = history_dates.searchsorted(np.datetime64(window_start), side="left")
        stop = history_dates.searchsorted(np.datetime64(current_date), side="right")
        window = conditions.iloc[start:stop]

        # 在提示公司的全部历史伙伴上找最大财报截止日，再筛所需公司对。
        maximum_accper = window.groupby("Stkcd")["Accper"].max()
        target = target_rows[PAIR_KEYS + ["quarterLag1Alt"]].copy()
        history = window.merge(
            target[["Stkcd", "StkcdRecall"]],
            on=["Stkcd", "StkcdRecall"], how="inner", validate="m:1",
        )
        quarter_distance = (history["Stkcd"].map(maximum_accper) - history["Accper"]).dt.days
        in_previous_quarter = quarter_distance.gt(0) & quarter_distance.le(100)
        history["_rebuilt_relation"] = in_previous_quarter.astype("boolean")
        for col in mechanism_history_cols:
            history[col] = history[col] & history["_rebuilt_relation"]

        summary = aggregate_flags(
            history, ["Stkcd", "StkcdRecall"], ["_rebuilt_relation"] + mechanism_history_cols
        )
        summary["_has_history"] = 1
        target = target.merge(summary, on=["Stkcd", "StkcdRecall"], how="left", validate="1:1")
        missing_history = target["_has_history"].isna()
        different_relation = target["_rebuilt_relation"].ne(target["quarterLag1Alt"]).fillna(True)
        if (missing_history | different_relation).any():
            raise ValueError(
                f"{current_date:%Y-%m-%d}的旧矩阵与重建历史不一致，请核对公告样本和季度口径。"
            )
        rebuilt.append(target[PAIR_KEYS + mechanism_history_cols])
        if position == 1 or position % 100 == 0 or position == target_dates.ngroups:
            print(f"已重建上一季度机制：{position}/{target_dates.ngroups}个公告日", flush=True)

    if not rebuilt:
        raise ValueError("筛选后的公司对为空，无法构建机制面板。")
    updates = pd.concat(rebuilt, ignore_index=True)
    mm = mm.drop(columns=mechanism_history_cols, errors="ignore")
    mm = mm.merge(updates, on=PAIR_KEYS, how="left", validate="1:1")
    del conditions, history_dates, rebuilt, updates, target_dates
    del window, history, target, summary, maximum_accper, target_rows
    gc.collect()

    return mm


# =============================================================================
# 生成规模和机制提示标记
# =============================================================================

def add_size_and_mechanism_cues(mm):
    """生成四个规模组的基础提示标记，以及两个规模组的机制提示标记。
    输入：已更新历史机制的公司对矩阵。
    处理：逐行比较提示公司市值和上月断点，再与上一季度关系、历史机制条件结合。
    输出：行数不变，新增规模组、基础提示标记和机制提示标记。
    缺失规则：无该规模提示为0；有提示但机制条件未知，继续保留缺失。
    """

    # 1. 划分提示公司规模，构造四类基础提示标记
    # 市值比较边界与原脚本一致；本段不改变公司对行数。
    # 取出两列逐行比较；严格大于断点为真，再把真假转为1和0。
    mm["size90plus"] = (
        mm["mktcap_tm3"] > mm["marketCap_90th_percentile"]
    ).astype("int8")
    mm["size50_90"] = (
        (mm["mktcap_tm3"] > mm["marketCap_50th_percentile"])
        & (mm["mktcap_tm3"] <= mm["marketCap_90th_percentile"])
    ).astype("int8")
    mm["size10_50"] = (
        (mm["mktcap_tm3"] >= mm["marketCap_10th_percentile"])
        & (mm["mktcap_tm3"] <= mm["marketCap_50th_percentile"])
    ).astype("int8")
    mm["size10minus"] = (
        mm["mktcap_tm3"] < mm["marketCap_10th_percentile"]
    ).astype("int8")
    # 两个0/1条件相乘：上一季度关系成立、且当前提示公司属于大市值组，才取1。
    mm["quarterLag1size90plus"] = (
        mm["quarterLag1Alt"] * mm["size90plus"]
    ).astype("int8")
    mm["quarterLag1size50_90"] = (mm["quarterLag1Alt"] * mm["size50_90"]).astype("int8")
    mm["quarterLag1size10_50"] = (mm["quarterLag1Alt"] * mm["size10_50"]).astype("int8")
    mm["quarterLag1size10minus"] = (mm["quarterLag1Alt"] * mm["size10minus"]).astype("int8")


    # 2. 把历史机制条件与大中规模提示结合
    # 每行左列是输出变量前缀，右列是历史条件来源。
    # 价格异动沿用“任意一家发生异动”，最终变量名保持原样。
    large_cue = mm["quarterLag1size90plus"].astype("boolean")
    medium_cue = mm["quarterLag1size50_90"].astype("boolean")
    for output_prefix, source_col in MECHANISM_SOURCES:
        historical_condition = mm[source_col].astype("boolean")
        large_col = output_prefix + "size90plus"
        medium_col = output_prefix + "size50_90"
        # 无该规模提示为0；有提示但历史条件未知时保留缺失。
        mm[large_col] = (historical_condition & large_cue).astype("Int8")
        mm[medium_col] = (historical_condition & medium_cue).astype("Int8")
    del large_cue, medium_cue, historical_condition

    return mm


# =============================================================================
# 汇总提示事件与连续机制指标
# =============================================================================

def collapse_nullable_cues(mm):
    """把公司对记录汇总为被提示公司的日度提示事件。
    输入：已生成基础提示和机制提示的公司对矩阵。
    处理：按被提示公司和事件交易日求和、取机制并集，再计算盈余惊喜及干扰均值。
    输出一：提示事件表，一行代表“被提示公司×事件交易日”，行数通常减少。
    输出二：基准大市值提示配对表，供最终统计每个提示公司事件日覆盖的公司数。
    分母：各盈余指标使用自身非缺失提示数及市值；任意实际提示的干扰数量未知则均值未知。
    """

    # 1. 准备均值所需的分子和分母
    # 各盈余惊喜分别使用自身非缺失提示的数量和市值作为分母。
    for k in [1, 2, 3]:
        value = mm[f"sue{k}"]
        valid_cue = mm["quarterLag1size90plus"].eq(1) & value.notna()
        mm[f"sue{k}Lag1S90p"] = value * mm["quarterLag1Alt"] * mm["size90plus"]
        mm[f"sue{k}Lag1XmarketCapS90p"] = value * mm["quarterLag1Alt"] * mm["mktcap_tm3"] * mm["size90plus"]
        mm[f"validSue{k}CueLag1S90p"] = valid_cue.astype("int8")
        mm[f"validSue{k}CueMarketCapLag1S90p"] = np.where(valid_cue, mm["mktcap_tm3"], 0.0)

    # 历史共同公告数沿用已有变量，分别准备两类规模组的分子和市值分母。
    mm["marketCap_Minus3Lag1S90p"] = mm["mktcap_tm3"] * mm["quarterLag1size90plus"]
    mm["marketCap_Minus3Lag1S5090"] = mm["mktcap_tm3"] * mm["quarterLag1size50_90"]
    mm["nrFirmsAnnLag1S90p"] = mm["nrFirmsAnnLag1Alt"] * mm["size90plus"]
    mm["nrFirmsAnnLag1S5090"] = mm["nrFirmsAnnLag1Alt"] * mm["size50_90"]
    mm["nrFirmsAnnLag1XmarketCapS90p"] = mm["nrFirmsAnnLag1Alt"] * mm["mktcap_tm3"] * mm["size90plus"]
    mm["nrFirmsAnnLag1XmarketCapS5090"] = mm["nrFirmsAnnLag1Alt"] * mm["mktcap_tm3"] * mm["size50_90"]
    mm = mm.sort_values(["Stkcd", "Annodt", "StkcdRecall"]).reset_index(drop=True)

    # 保留基准提示配对，用于统计每个提示公司事件日覆盖的公司数。
    cue_pairs = mm.loc[
        mm["quarterLag1size90plus"].eq(1), ["Stkcd", "event_Trddt", "StkcdRecall"]
    ].rename(columns={"event_Trddt": "Trddt"})
    cue_pairs = cue_pairs.dropna().drop_duplicates(["Stkcd", "Trddt", "StkcdRecall"])

    # 2. 汇总成“被提示公司 × 事件交易日”的提示事件表
    # 先求和再计算均值；基础提示取最大值，机制标记采用保留缺失的并集。
    day_keys = ["StkcdRecall", "event_Trddt"]
    mm = mm.sort_values(["StkcdRecall", "event_Trddt", "mktcap_tm3", "Stkcd"]).reset_index(drop=True)
    sum_rules = {
        "nrCuesLag1S90p": ("quarterLag1size90plus", "sum"),
        "nrCuesLag1S5090": ("quarterLag1size50_90", "sum"),
        "nrCuesLag1S1050": ("quarterLag1size10_50", "sum"),
        "nrCuesLag1S10m": ("quarterLag1size10minus", "sum"),
        "sumMarketCap_Minus3Lag1S90p": ("marketCap_Minus3Lag1S90p", "sum"),
        "sumMarketCap_Minus3Lag1S5090": ("marketCap_Minus3Lag1S5090", "sum"),
        "sumNrFirmsAnnLag1S90p": ("nrFirmsAnnLag1S90p", "sum"),
        "sumNrFirmsAnnLag1S5090": ("nrFirmsAnnLag1S5090", "sum"),
        "sumNrFirmsAnnLag1XmarketCapS90p": ("nrFirmsAnnLag1XmarketCapS90p", "sum"),
        "sumNrFirmsAnnLag1XmarketCapS5090": ("nrFirmsAnnLag1XmarketCapS5090", "sum"),
    }
    for k in [1, 2, 3]:
        sum_rules[f"sumSue{k}Lag1S90p"] = (f"sue{k}Lag1S90p", "sum")
        sum_rules[f"sumSue{k}Lag1XmarketCapS90p"] = (f"sue{k}Lag1XmarketCapS90p", "sum")
        sum_rules[f"sumSue{k}ValidCueCountLag1S90p"] = (f"validSue{k}CueLag1S90p", "sum")
        sum_rules[f"sumSue{k}ValidCueMarketCapLag1S90p"] = (f"validSue{k}CueMarketCapLag1S90p", "sum")

    collapsed = mm.groupby(day_keys, dropna=False).agg(**sum_rules).reset_index()
    basic_flags = mm.groupby(day_keys, dropna=False)[BASE_CUE_COLS].max().reset_index()
    mechanism_flags = aggregate_flags(mm, day_keys, MECHANISM_COLS)
    collapsed = collapsed.merge(basic_flags, on=day_keys, how="left", validate="1:1")
    collapsed = collapsed.merge(mechanism_flags, on=day_keys, how="left", validate="1:1")

    for k in [1, 2, 3]:
        valid_count = collapsed[f"sumSue{k}ValidCueCountLag1S90p"].replace(0, np.nan)
        valid_marketcap = collapsed[f"sumSue{k}ValidCueMarketCapLag1S90p"].replace(0, np.nan)
        collapsed[f"sue{k}CueLag1S90pEW"] = collapsed[f"sumSue{k}Lag1S90p"] / valid_count
        collapsed[f"sue{k}CueLag1S90pVW"] = collapsed[f"sumSue{k}Lag1XmarketCapS90p"] / valid_marketcap

    # 干扰均值分别计算等权、市值加权；分母为0时保留缺失。
    collapsed["nrFirmsAnnLag1EWS90p"] = (
        collapsed["sumNrFirmsAnnLag1S90p"] / collapsed["nrCuesLag1S90p"].replace(0, np.nan)
    )
    collapsed["nrFirmsAnnLag1VWS90p"] = (
        collapsed["sumNrFirmsAnnLag1XmarketCapS90p"]
        / collapsed["sumMarketCap_Minus3Lag1S90p"].replace(0, np.nan)
    )
    collapsed["nrFirmsAnnLag1EWS5090"] = (
        collapsed["sumNrFirmsAnnLag1S5090"] / collapsed["nrCuesLag1S5090"].replace(0, np.nan)
    )
    collapsed["nrFirmsAnnLag1VWS5090"] = (
        collapsed["sumNrFirmsAnnLag1XmarketCapS5090"]
        / collapsed["sumMarketCap_Minus3Lag1S5090"].replace(0, np.nan)
    )

    # 任意实际提示的历史数量未知，该规模组的干扰均值就不能确定。
    # 先按公司日汇总未知标记，再按键合并回去，直接展示匹配过程。
    interference_groups = [
        ("quarterLag1size90plus", "nrFirmsAnnLag1EWS90p", "nrFirmsAnnLag1VWS90p"),
        ("quarterLag1size50_90", "nrFirmsAnnLag1EWS5090", "nrFirmsAnnLag1VWS5090"),
    ]
    for cue_col, ew_col, vw_col in interference_groups:
        actual_cues = mm.loc[mm[cue_col].eq(1), day_keys + ["nrFirmsAnnLag1Alt"]].copy()
        actual_cues["_unknown_interference"] = actual_cues["nrFirmsAnnLag1Alt"].isna()
        unknown = actual_cues.groupby(day_keys)["_unknown_interference"].any().reset_index()
        collapsed = collapsed.merge(unknown, on=day_keys, how="left", validate="1:1")
        unknown_mask = collapsed["_unknown_interference"].eq(True)
        collapsed.loc[unknown_mask, [ew_col, vw_col]] = np.nan
        collapsed = collapsed.drop(columns=["_unknown_interference"])

    collapsed["date"] = collapsed["event_Trddt"]
    collapsed = collapsed.sort_values(["StkcdRecall", "date"]).reset_index(drop=True)
    intermediate_cols = [
        col for col in collapsed.columns
        if col.startswith(("sumSue", "sumMarketCap", "sumNrFirmsAnn"))
    ]
    cue_events = collapsed.drop(columns=["event_Trddt"] + intermediate_cols).copy()
    del mm, collapsed, basic_flags, mechanism_flags, actual_cues, unknown, value, valid_cue
    gc.collect()

    return cue_events, cue_pairs


# =============================================================================
# 读取日度主面板
# =============================================================================

def load_daily_panel(daily_path):
    """读取被提示公司日度面板，并统一原始收益率名称。
    输入：已有的个股日度数据文件。
    处理：保留必要列、统一股票代码，把公司代码改为被提示公司代码。
    输出：一行代表“被提示公司×交易日”；读取不筛样，收益率缺失保留。
    新名称：原始收益率统一为原始收益率前缀及被提示公司后缀，与调整收益区分。
    """

    # 表结构：被提示公司 × 交易日。收益率沿用单精度以控制内存。
    raw_return_cols = ["raw_ret_tm3", "raw_ret_tm2", "raw_ret_tm1", "raw_ret_t"]
    raw_return_cols += [f"raw_ret_t_t{i}" for i in range(1, 51)]
    daily_dtypes = {"Stkcd": "string", "td": "int32", "year": "float32", "mktcap_tm3": "float32"}
    for col in raw_return_cols:
        daily_dtypes[col] = "float32"
    panel = pd.read_csv(
        daily_path, encoding="utf-8",
        usecols=["Stkcd", "td", "Trddt", "year", "mktcap_tm3"] + raw_return_cols,
        dtype=daily_dtypes, parse_dates=["Trddt"],
    )
    panel["Stkcd"] = panel["Stkcd"].apply(normalize_stock_code)
    return_names = {
        "Stkcd": "StkcdRecall",
        "raw_ret_tm3": "return_tm3_Recall", "raw_ret_tm2": "return_tm2_Recall",
        "raw_ret_tm1": "return_tm1_Recall", "raw_ret_t": "return_t_Recall",
    }
    for i in range(1, 51):
        return_names[f"raw_ret_t_t{i}"] = f"return_t_tp{i}_Recall"
    panel = panel.rename(columns=return_names)
    if panel["StkcdRecall"].eq("").any() or panel[["StkcdRecall", "Trddt", "td"]].isna().any().any():
        raise ValueError("日度面板的公司、交易日或交易日序号存在缺失。")
    if panel.duplicated(["StkcdRecall", "Trddt"]).any() or panel.duplicated(["StkcdRecall", "td"]).any():
        raise ValueError("日度面板存在重复公司日。")

    return panel


# =============================================================================
# 合并自身财报公告窗口
# =============================================================================

def add_own_announcement_controls(panel, report_path):
    """合并被提示公司自身财报公告及前后十个市场交易日标记。
    输入：被提示公司日度面板，以及财报公告文件。
    处理：交易日公告使用当天，非交易日公告顺延，再按市场交易日序号平移窗口。
    输出：日度面板行数不变，新增自身公告及窗口标记；没有事件为0。
    方向：后续第1日标记为1表示下一市场交易日公告，之前第1日标记为1表示上一市场交易日已公告。
    """

    # 自身财报公告只把非交易日公告顺延，交易日公告使用当天。
    own_reports = pd.read_csv(
        report_path, usecols=["Stkcd", "Annodt"],
        dtype={"Stkcd": "string"}, parse_dates=["Annodt"],
    )
    own_reports["Stkcd"] = own_reports["Stkcd"].apply(normalize_stock_code)
    own_reports = own_reports.rename(columns={"Stkcd": "StkcdRecall"})
    calendar = panel[["Trddt", "td"]].drop_duplicates().sort_values("Trddt")
    announcement_dates = own_reports[["Annodt"]].dropna().drop_duplicates().sort_values("Annodt")
    announcement_dates = pd.merge_asof(
        announcement_dates, calendar, left_on="Annodt", right_on="Trddt",
        direction="forward", allow_exact_matches=True,
    )
    own_reports = own_reports.merge(announcement_dates, on="Annodt", how="left", validate="m:1")
    own_reports = own_reports.dropna(subset=["Trddt"])[["StkcdRecall", "Trddt"]].drop_duplicates()
    own_reports["ownEA"] = 1
    panel = panel.merge(own_reports, on=["StkcdRecall", "Trddt"], how="left", validate="1:1")
    panel["ownEA"] = panel["ownEA"].fillna(0).astype("int8")

    own_ea_days = panel.loc[panel["ownEA"].eq(1), ["StkcdRecall", "td"]].copy()
    for i in range(1, 11):
        # 当前日的第i个后续市场交易日会公告：公告日序号减i再匹配。
        future_announcement = own_ea_days.copy()
        future_announcement["td"] = future_announcement["td"] - i
        future_announcement[f"ownEA_tp{i}"] = 1
        panel = panel.merge(
            future_announcement, on=["StkcdRecall", "td"], how="left", validate="1:1"
        )
        panel[f"ownEA_tp{i}"] = panel[f"ownEA_tp{i}"].fillna(0).astype("int8")

        # 当前日的第i个之前市场交易日已公告：公告日序号加i再匹配。
        past_announcement = own_ea_days.copy()
        past_announcement["td"] = past_announcement["td"] + i
        past_announcement[f"ownEA_tm{i}"] = 1
        panel = panel.merge(
            past_announcement, on=["StkcdRecall", "td"], how="left", validate="1:1"
        )
        panel[f"ownEA_tm{i}"] = panel[f"ownEA_tm{i}"].fillna(0).astype("int8")
    del own_reports, calendar, announcement_dates, own_ea_days, future_announcement, past_announcement

    return panel


# =============================================================================
# 合并其他自身事件与新闻
# =============================================================================

def merge_other_event_controls(
    panel,
    forecast_path,
    other_event_path,
    news_control_path,
):
    """合并自身业绩预告快报、专题事件和新闻控制变量。
    输入：日度面板和三类已经按真实交易日整理的控制变量表。
    处理：均按“被提示公司代码×交易日”左连接，并检查右表唯一性。
    输出：行数不变，增加两类自身事件标记和新闻控制列。
    缺失规则：没有自身事件填0，新闻控制变量继续保留缺失。
    """

    # 两类自身事件没有记录时填0；新闻控制变量的缺失继续保留。
    forecast = pd.read_csv(
        forecast_path, encoding="utf-8-sig",
        usecols=["StkcdRecall", "Trddt", "ownPerformanceForecastBrief"],
        dtype={"StkcdRecall": "string"}, parse_dates=["Trddt"],
    )
    forecast["StkcdRecall"] = forecast["StkcdRecall"].apply(normalize_stock_code)
    if forecast["StkcdRecall"].eq("").any() or forecast[["StkcdRecall", "Trddt"]].isna().any().any():
        raise ValueError("业绩预告快报控制表存在缺失键。")
    forecast["ownPerformanceForecastBrief"] = (
        pd.to_numeric(forecast["ownPerformanceForecastBrief"], errors="raise").fillna(0).astype("int8")
    )
    panel = panel.merge(forecast, on=["StkcdRecall", "Trddt"], how="left", validate="1:1")
    panel["ownPerformanceForecastBrief"] = panel["ownPerformanceForecastBrief"].fillna(0).astype("int8")
    del forecast

    other_events = pd.read_csv(
        other_event_path, encoding="utf-8-sig", usecols=["StkcdRecall", "Trddt", "ownOtherEventER"],
        dtype={"StkcdRecall": "string"}, parse_dates=["Trddt"],
    )
    other_events["StkcdRecall"] = other_events["StkcdRecall"].apply(normalize_stock_code)
    if other_events["StkcdRecall"].eq("").any() or other_events[["StkcdRecall", "Trddt"]].isna().any().any():
        raise ValueError("专题事件控制表存在缺失键。")
    other_events["ownOtherEventER"] = (
        pd.to_numeric(other_events["ownOtherEventER"], errors="raise").fillna(0).astype("int8")
    )
    panel = panel.merge(other_events, on=["StkcdRecall", "Trddt"], how="left", validate="1:1")
    panel["ownOtherEventER"] = panel["ownOtherEventER"].fillna(0).astype("int8")
    del other_events

    news_cols = ["AES", "AEV", "aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV"]
    news = pd.read_csv(
        news_control_path, encoding="utf-8-sig", usecols=["Stkcd", "date"] + news_cols,
        dtype={"Stkcd": "string"}, parse_dates=["date"],
    )
    news["Stkcd"] = news["Stkcd"].apply(normalize_stock_code)
    news = news.rename(columns={"Stkcd": "StkcdRecall", "date": "Trddt"})
    news["Trddt"] = pd.to_datetime(news["Trddt"], errors="coerce")
    if news["StkcdRecall"].eq("").any() or news[["StkcdRecall", "Trddt"]].isna().any().any():
        raise ValueError("新闻控制表存在缺失键。")
    news[news_cols] = news[news_cols].apply(pd.to_numeric, errors="raise")
    if news["AEV"].dropna().lt(0).any():
        raise ValueError("新闻控制表的新闻数量不能为负数。")
    panel = panel.merge(news, on=["StkcdRecall", "Trddt"], how="left", validate="1:1")
    del news

    return panel


# =============================================================================
# 合并提示事件并处理机制缺失
# =============================================================================

def merge_nullable_cues(panel, cue_events):
    """把提示事件表合并到完整日度面板，并处理结构性缺失。
    输入：被提示公司日度面板，以及被提示公司事件日提示表。
    处理：按被提示公司和交易日左连接，基础提示及数量的空值填0。
    输出：日度行数不变，新增提示数量、提示机制、盈余均值及干扰均值。
    机制规则：无对应规模提示才补0，有提示但无法判断的机制值仍保留缺失。
    连续变量：本函数不对盈余惊喜均值、干扰均值填0。
    """

    panel = panel.merge(
        cue_events.rename(columns={"date": "Trddt"}),
        on=["StkcdRecall", "Trddt"], how="left", validate="1:1",
    )
    del cue_events
    panel[BASE_CUE_COLS] = panel[BASE_CUE_COLS].fillna(0).astype("int8")
    panel[COUNT_COLS] = panel[COUNT_COLS].fillna(0).astype("int64")

    # 大市值提示：没有该提示才补0；有提示但机制未知的记录保留缺失。
    no_large_cue = panel["quarterLag1size90plus"].eq(0)
    for col in LARGE_MECHANISM_COLS:
        panel[col] = panel[col].astype("Int8")
        panel.loc[no_large_cue, col] = 0

    # 中等市值提示采用同样规则；连续变量在本段都不填0。
    no_medium_cue = panel["quarterLag1size50_90"].eq(0)
    for col in MEDIUM_MECHANISM_COLS:
        panel[col] = panel[col].astype("Int8")
        panel.loc[no_medium_cue, col] = 0
    del no_large_cue, no_medium_cue

    return panel


# =============================================================================
# 合并特征调整收益率
# =============================================================================

def merge_adjusted_returns(panel, adjusted_return_path):
    """合并特征调整收益率。
    输入：日度面板，以及上游已经构造好的特征调整收益率文件。
    处理：统一公司代码和收益率名称，按被提示公司与交易日左连接。
    输出：行数不变，增加调整收益列，与原始收益率同时保留；缺失不填0。
    """

    # 原始收益率与特征调整收益率并存；未匹配到的收益率继续缺失。
    adjusted_cols = ["chAdj_ret_tm3", "chAdj_ret_tm2", "chAdj_ret_tm1", "chAdj_ret_t"]
    adjusted_cols += [f"chAdj_ret_t_t{i}" for i in range(1, 51)]
    adjusted_dtypes = {"Stkcd": "string", "portfolioNumber": "float32"}
    for col in adjusted_cols:
        adjusted_dtypes[col] = "float32"
    adjusted = pd.read_csv(
        adjusted_return_path, encoding="utf-8-sig", dtype=adjusted_dtypes, parse_dates=["Trddt"]
    )
    adjusted["Stkcd"] = adjusted["Stkcd"].apply(normalize_stock_code)
    adjusted_names = {
        "Stkcd": "StkcdRecall",
        "chAdj_ret_tm3": "chAdjreturn_tm3_Recall", "chAdj_ret_tm2": "chAdjreturn_tm2_Recall",
        "chAdj_ret_tm1": "chAdjreturn_tm1_Recall", "chAdj_ret_t": "chAdjreturn_t_Recall",
    }
    for i in range(1, 51):
        adjusted_names[f"chAdj_ret_t_t{i}"] = f"chAdjreturn_t_tp{i}_Recall"
    adjusted = adjusted.rename(columns=adjusted_names)
    keep_adjusted_cols = [col for col in adjusted.columns if col.startswith("chAdjreturn_")]
    adjusted = adjusted[["StkcdRecall", "Trddt", "portfolioNumber"] + keep_adjusted_cols]
    panel = panel.merge(adjusted, on=["StkcdRecall", "Trddt"], how="left", validate="1:1")
    del adjusted
    gc.collect()

    return panel


# =============================================================================
# 构造机制及规模分组
# =============================================================================

def add_mechanism_groups(panel):
    """生成干扰高低组、提示盈余惊喜五分位和被提示公司规模组。
    输入：已合并提示及收益的完整日度面板。
    处理：干扰和被提示公司规模按年度分组；提示盈余惊喜按全样本分五组。
    输出：行数不变，增加各类分组及相应0/1标记。
    缺失规则：有提示但干扰未知时，高组和低组都缺失；无该规模提示时两组均为0。
    """

    # 干扰按年度分组。每行依次列出：来源值、分组列、低组列、高组列、对应提示列。
    for value_col, group_col, low_col, high_col, cue_col in INTERFERENCE_SPECS:
        panel[group_col] = panel.groupby("year")[value_col].transform(
            lambda s: xtile_by_group(s, 2).astype("float64")
        )
        # 先设缺失；组别已知才赋0或1；无提示的公司日两列均为0。
        known_group = panel[group_col].notna()
        panel[low_col] = panel[group_col].eq(1).astype("Int8").where(known_group, pd.NA)
        panel[high_col] = panel[group_col].eq(2).astype("Int8").where(known_group, pd.NA)
        panel.loc[panel[cue_col].eq(0), [low_col, high_col]] = 0
    del known_group

    # 提示盈余惊喜五分位沿用全样本分组，不改为年度分组。
    for value_col, group_col in SUE_QUINTILES:
        panel[group_col] = xtile_by_group(panel[value_col], 5).astype("float64")

    panel["marketCap_Minus3_Recall"] = panel["mktcap_tm3"]
    panel["marketCap_Minus3_Recall_medGr"] = panel.groupby("year")["marketCap_Minus3_Recall"].transform(
        lambda s: xtile_by_group(s, 2).astype("float64")
    )
    panel["smallFirm"] = panel["marketCap_Minus3_Recall_medGr"].eq(1).astype("int8")

    return panel


# =============================================================================
# 缩尾并筛选基准样本
# =============================================================================

def prepare_baseline_sample(panel):
    """计算缩尾变量，并依次保留两个基准回归的有效样本。
    输入：分组完成的完整日度面板。
    处理：先用筛样前全样本算1%与99%阈值，再依次筛后1日和后10日基准回归样本。
    输出：增加缩尾列、季度标识，行数可能减少；每轮删除公司×年季度中的单例组。
    注意：机制变量不参与这一步的非缺失筛样，各机制回归仍自行选择有效样本。
    """

    # 阈值在筛样前确定：低于1%分位点、高于99%分位点的值分别压到边界。
    return_cols = [
        "chAdjreturn_tm3_Recall", "chAdjreturn_tm2_Recall",
        "chAdjreturn_tm1_Recall", "chAdjreturn_t_Recall",
    ]
    return_cols += [f"chAdjreturn_t_tp{i}_Recall" for i in range(1, 51)]
    sue_mean_cols = [value_col for value_col, group_col in SUE_QUINTILES]
    winsor_cols = [col for col in return_cols + sue_mean_cols if col in panel.columns]
    winsor_limits = {}
    for col in winsor_cols:
        winsor_limits[col] = (panel[col].quantile(0.01), panel[col].quantile(0.99))

    # 先生成基准筛样要用的两个缩尾因变量。
    baseline_outcomes = ["chAdjreturn_t_tp1_Recall", "chAdjreturn_t_tp10_Recall"]
    for col in baseline_outcomes:
        lower, upper = winsor_limits[col]
        panel[col + "_w"] = panel[col].clip(lower=lower, upper=upper)

    panel["quarter"] = panel["Trddt"].dt.quarter
    panel["yearQuarter"] = panel["Trddt"].dt.to_period("Q")
    front_cols = ["StkcdRecall", "Trddt"]
    other_cols = [col for col in panel.columns if col not in front_cols]
    panel = panel[front_cols + other_cols].sort_values(front_cols).reset_index(drop=True)

    own_ea_cols = [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    own_ea_cols += ["ownEA"] + [f"ownEA_tp{i}" for i in range(1, 11)]
    baseline_rhs = ["quarterLag1size90plus"] + own_ea_cols + [
        "ownOtherEventER", "ownPerformanceForecastBrief",
        "aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV",
    ]
    for outcome in baseline_outcomes:
        # 沿用顺序：先筛后1日窗口，再在剩余样本中筛后10日窗口。
        needed_cols = [outcome + "_w"] + baseline_rhs + ["StkcdRecall", "yearQuarter", "Trddt"]
        panel = panel.dropna(subset=needed_cols).copy()
        # 固定效应为“被提示公司×年季度”，该组仅剩一行时删除。
        group_size = panel.groupby(["StkcdRecall", "yearQuarter"])["Trddt"].transform("size")
        panel = panel.loc[group_size.gt(1)].copy()
    del group_size

    # 机制变量不在基准筛样字段中；各机制回归仍使用各自的有效样本。
    for col in winsor_cols:
        if col + "_w" not in panel.columns:
            lower, upper = winsor_limits[col]
            panel[col + "_w"] = panel[col].clip(lower=lower, upper=upper)
    print("同时进入两个基准回归的样本量：", len(panel))

    return panel


# =============================================================================
# 统计提示范围并生成未来提示
# =============================================================================

def summarize_and_add_future_cues(panel, cue_pairs):
    """统计各提示公司事件日的覆盖范围，并生成未来十个市场交易日提示标记。
    输入：最终基准日度样本，以及聚合前保留的基准大市值提示配对表。
    处理：先把配对限制到最终样本，再统计覆盖范围和按市场交易日序号平移事件。
    输出：日度面板行数不变，新增未来提示标记；另返回覆盖数量与描述统计。
    未来提示只依据筛样后仍保留的事件，不使用已被删除的提示事件。
    """

    firm_days = panel[["StkcdRecall", "Trddt"]].drop_duplicates()
    cue_pairs_final = cue_pairs.merge(firm_days, on=["StkcdRecall", "Trddt"], how="inner", validate="m:1")
    nr_firms_cued = cue_pairs_final.groupby(["Stkcd", "Trddt"], observed=True)["StkcdRecall"].nunique()
    # 下方沿用原描述统计名称；“公告次数”实际按“提示公司×事件交易日”计数。
    # 多个原公告日映射到同一交易日时，这里会合并为同一个提示公司事件日。
    reach_summary = pd.Series({
        "提示公司公告次数": len(nr_firms_cued),
        "均值": nr_firms_cued.mean(), "第25百分位": nr_firms_cued.quantile(0.25),
        "中位数": nr_firms_cued.median(), "第75百分位": nr_firms_cued.quantile(0.75),
        "标准差": nr_firms_cued.std(ddof=1), "最小值": nr_firms_cued.min(), "最大值": nr_firms_cued.max(),
    }, name="每次公告提示的公司数量")
    print(reach_summary)

    # 未来提示只用最终基准样本中保留的事件，并按市场交易日序号平移。
    cue_days = panel.loc[panel["quarterLag1size90plus"].eq(1), ["StkcdRecall", "td"]].drop_duplicates()
    for i in range(1, 11):
        future_cue = cue_days.copy()
        future_cue["td"] = future_cue["td"] - i
        future_cue[f"cue_tp{i}"] = 1
        panel = panel.merge(future_cue, on=["StkcdRecall", "td"], how="left", validate="1:1")
        panel[f"cue_tp{i}"] = panel[f"cue_tp{i}"].fillna(0).astype("int8")
    del cue_pairs, firm_days, cue_pairs_final, cue_days, future_cue

    return panel, nr_firms_cued, reach_summary


# =============================================================================
# 检查并整理最终面板
# =============================================================================

def check_and_clean_final_panel(panel):
    """检查机制缺失口径，并删除最终交付不需要的中间列。
    输入：已完成筛样和未来提示标记的日度面板。
    处理：检查公司日唯一性、0/1标记和未知值规则，再清理辅助字段。
    输出：行数不变，列数减少；保留原始及调整收益率、回归变量和机制标记。
    本函数只返回整理后的表；实际保存位置及保存语句放在主流程末尾。
    """

    if panel.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("最终面板存在重复被提示公司日。")
    for col in MECHANISM_COLS:
        if not panel[col].dropna().isin([0, 1]).all():
            raise ValueError(f"机制标记含非0或1的值：{col}")
        cue_col = "quarterLag1size90plus" if col in LARGE_MECHANISM_COLS else "quarterLag1size50_90"
        no_cue = panel[cue_col].eq(0)
        if (panel[col].eq(1) & panel[cue_col].ne(1)).any():
            raise ValueError(f"机制提示为1时，对应规模提示必须也为1：{col}")
        if panel.loc[no_cue, col].isna().any() or not panel.loc[no_cue, col].eq(0).all():
            raise ValueError(f"无对应规模提示时，机制标记必须为0：{col}")
        unknown_count = (panel[cue_col].eq(1) & panel[col].isna()).sum()
        print(f"{col}：有提示但机制未知{unknown_count}条")

    for value_col, group_col, low_col, high_col, cue_col in INTERFERENCE_SPECS:
        unknown = panel[cue_col].eq(1) & panel[value_col].isna()
        if panel.loc[unknown, [low_col, high_col]].notna().any().any():
            raise ValueError("有提示但干扰未知时，高低干扰标记都应保留缺失。")

    expected_returns = ["return_tm3_Recall", "return_tm2_Recall", "return_tm1_Recall", "return_t_Recall"]
    expected_returns += [f"return_t_tp{i}_Recall" for i in range(1, 51)]
    if any(col not in panel.columns for col in expected_returns):
        raise ValueError("核心原始收益率列不完整，请先检查收益率重命名。")

    drop_cols = [
        "AES", "AEV", "marketCap_Minus3_Recall_medGr",
        "interferenceGroupEWLag1S90p", "interferenceGroupEWLag1S5090",
        "interferenceGroupVWLag1S90p", "interferenceGroupVWLag1S5090",
        "ret", "retn", "Markettype", "size", "sizew", "trade_days_1m", "trade_days_1y",
        "mktcap_tm3", "float_mktcap_tm3", "mktcap_tm3_valid", "float_mktcap_tm3_valid", "portfolioNumber",
    ]
    drop_cols += [col for col in panel.columns if col.startswith("raw_ret_")]
    panel = panel.drop(columns=drop_cols, errors="ignore")

    return panel

# =============================================================================
# 主流程：先从这里阅读，再按函数名查找各步骤的具体处理
# =============================================================================

def main():
    """按四个阶段构造面板，保存两个数据文件，并返回便于查看的结果。"""

    # 第一阶段：准备公司对记忆矩阵，并排除其他关联关系。
    # 输入和输出都保持“提示公司×原公告日×被提示公司”的层级。
    mm = prepare_memory_matrix(
        matrix_path=matrix_path,
        daily_path=daily_path,
        breakpoint_path=breakpoint_path,
        sue_path=sue_path,
        report_path=report_path,
    )
    mm = filter_related_pairs(
        mm=mm,
        industry_path=industry_path,
        supply_path=supply_path,
        analyst_path=analyst_path,
    )

    # 第二阶段：从原始公告重建机制，再汇总成被提示公司的提示事件表。
    # 历史公告中，每家公司每个原公告日一条，再据此构造同日公告公司对。
    events = read_history_features(
        report_path=report_path,
        pattern_path=pattern_path,
        calendar_path=calendar_path,
        news_feature_path=news_feature_path,
        volume_path=volume_path,
        sue_path=sue_path,
    )
    events = add_top_mover_flags(
        events=events,
        daily_path=daily_path,
    )
    conditions = build_history_conditions(events=events)
    del events

    # 用历史机制条件更新原矩阵；随后加入规模条件和提示标记。
    mm = rebuild_lag1_mechanisms(mm=mm, conditions=conditions)
    del conditions
    mm = add_size_and_mechanism_cues(mm=mm)

    # 这里的逗号表示函数返回两张表：提示事件表、基准大市值提示配对表。
    cue_events, cue_pairs = collapse_nullable_cues(mm=mm)
    del mm
    gc.collect()

    # 提示事件表已经汇总为“被提示公司×事件交易日”，单独保存。
    output_dir.mkdir(parents=True, exist_ok=True)
    cue_events.to_csv(
        cue_output,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    print("已保存提示事件表：", cue_output)

    # 第三阶段：从完整日度面板出发，依次合并自身事件、新闻、提示和收益。
    # 左连接始终保留“被提示公司×交易日”的行粒度。
    panel = load_daily_panel(daily_path=daily_path)
    panel = add_own_announcement_controls(
        panel=panel,
        report_path=report_path,
    )
    panel = merge_other_event_controls(
        panel=panel,
        forecast_path=forecast_path,
        other_event_path=other_event_path,
        news_control_path=news_control_path,
    )
    panel = merge_nullable_cues(panel=panel, cue_events=cue_events)
    del cue_events
    panel = merge_adjusted_returns(
        panel=panel,
        adjusted_return_path=adjusted_return_path,
    )

    # 第四阶段：构造分组、筛选基准样本、补充未来提示标记，最后保存。
    # 缩尾阈值在基准筛样之前计算；机制缺失不参与基准样本的非缺失筛选。
    panel = add_mechanism_groups(panel=panel)
    panel = prepare_baseline_sample(panel=panel)
    panel, nr_firms_cued, reach_summary = summarize_and_add_future_cues(
        panel=panel,
        cue_pairs=cue_pairs,
    )
    del cue_pairs
    panel = check_and_clean_final_panel(panel=panel)

    panel.to_csv(
        final_output,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    print("最终面板已保存：", final_output)
    print("最终表结构：", panel.shape)

    # 返回值只供查看，不会额外生成文件。
    return {
        "daily_panel": panel,  # 最终被提示公司日度面板
        "nr_firms_cued": nr_firms_cued,  # 各提示公司公告覆盖的被提示公司数量
        "nr_firms_cued_summary": reach_summary,  # 上述数量的描述统计
    }


# 直接运行本文件时，调用主流程；被其他脚本导入时，不执行数据处理。
if __name__ == "__main__":
    results = main()
