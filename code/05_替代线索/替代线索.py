# 本脚本基于模式公司的记忆矩阵，构造非盈余公告类线索：
# 其他重大公告；业绩预告、业绩快报；

from pathlib import Path
import numpy as np
import pandas as pd

#===============================================================================
# 路径设置
#===============================================================================
ROOT = Path("/Users/qiyuewu/Desktop/mmm复现数据包")

MEMORY_PATH = ROOT / "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv"
ALT_EVENT_PATH = ROOT / "中间结果/02_变量构造结果/公司公告事件/公司公告事件明细_下一交易日口径.csv"
DAILY_PATH = ROOT / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv"
SIZE_BP_PATH = ROOT / "中间结果/02_变量构造结果/月度市值分组.csv"

OUTPUT_DIR = ROOT / "中间结果/05_替代线索"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALT_CUES_OUTPUT_PATH = OUTPUT_DIR / "替代线索_两类公告.csv"

#===============================================================================
# 函数工具
#===============================================================================
def normalize_stock_code(x):
    if pd.isna(x):
        return ""
    return str(x).strip().zfill(6)

def safe_qcut_2(s):
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    valid = s.dropna()

    if len(valid) < 2 or valid.nunique() < 2:
        print("警告：某一年有效干扰度不足或没有差异，低/高干扰分组设为缺失。")
        return out

    cut = pd.qcut(valid, q=2, labels=False, duplicates="drop")

    if pd.Series(cut, index=valid.index).nunique(dropna=True) < 2:
        print("警告：某一年分位点重复，改用排序后分组。")
        cut = pd.qcut(valid.rank(method="first"), q=2, labels=False)

    out.loc[valid.index] = pd.Series(cut, index=valid.index).astype(float) + 1
    return out



# 读取记忆矩阵
memory = pd.read_csv(
    MEMORY_PATH,
    dtype={"Stkcd": "string", "StkcdRecall": "string"},
    parse_dates=["Annodt", "event_Trddt"],
    encoding="utf-8-sig",
)

memory["Stkcd"] = memory["Stkcd"].apply(normalize_stock_code)
memory["StkcdRecall"] = memory["StkcdRecall"].apply(normalize_stock_code)
memory["yearQuarter"] = memory["event_Trddt"].dt.to_period("Q").astype(str)


# 读取两类替代公告事件
event = pd.read_csv(
    ALT_EVENT_PATH,
    dtype={"StkcdRecall": "string"},
    parse_dates=["date", "Trddt"],
    encoding="utf-8-sig",
)

event["Stkcd"] = event["StkcdRecall"].apply(normalize_stock_code)
event["yearQuarter"] = event["date"].dt.to_period("Q").astype(str)

event = event[
    event["altForecast"].fillna(0).eq(1)
    | event["altMajorNarrowOther"].fillna(0).eq(1)
].copy()

if event.empty:
    raise RuntimeError(
        "公告明细表中没有 altForecast 或 altMajorNarrowOther 事件。"
        "请先确认上游明细表是否已生成，以及两类公告分类变量是否正常。"
    )

event = (
    event
    .groupby(["Stkcd", "date", "yearQuarter"], as_index=False)
    .agg(
        altForecast=("altForecast", "max"),
        altMajorNarrowOther=("altMajorNarrowOther", "max"),
    )
)

event = event[
    ["Stkcd", "date", "yearQuarter", "altForecast", "altMajorNarrowOther"]
].drop_duplicates(["Stkcd", "date"])

# 把替代公告接入记忆矩阵
alt_memory = event.merge(
    memory,
    on=["Stkcd", "yearQuarter"],
    how="inner",
    validate="m:m",
)

print("替代公告接入记忆矩阵后行数：", len(alt_memory))


# 过滤不合格事件
alt_memory = alt_memory[
    alt_memory["date"].ne(alt_memory["event_Trddt"])
].copy() # 如果和原本的盈余公告在同一天，删除
# 考虑到检验的窗口为[t,t+1]两天和[t,t+10]11天，都会把公告后一天考虑进去，所以主检验暂时不做向后调整

alt_memory = alt_memory[
    ~(
        alt_memory["quarterLag0Alt"].fillna(0).eq(1)
        & alt_memory["quarterLag1Alt"].fillna(0).eq(0)
        & (alt_memory["date"] < alt_memory["event_Trddt"])
    )
].copy() # 如果发生在记忆编码之前，则不能作为触发已形成记忆的线索事件

# 合并线索公司 t-3 市值并识别大市值线索
daily = pd.read_csv(
    DAILY_PATH,
    usecols=["Stkcd", "td", "Trddt", "size"],
    dtype={"Stkcd": "string"},
    parse_dates=["Trddt"],
    encoding="utf-8",
)

daily["Stkcd"] = daily["Stkcd"].apply(normalize_stock_code)
daily["td"] = pd.to_numeric(daily["td"], errors="coerce")

date_td = daily[["Trddt", "td"]].drop_duplicates().rename(
    columns={"Trddt": "date", "td": "alt_td"}
)

alt_memory = alt_memory.merge(date_td, on="date", how="left", validate="m:1")
alt_memory["td_tm3"] = alt_memory["alt_td"] - 3

mktcap = daily.rename(columns={"td": "td_tm3", "size": "mktcap_tm3"})[
    ["Stkcd", "td_tm3", "mktcap_tm3"]
].drop_duplicates(["Stkcd", "td_tm3"])

alt_memory = alt_memory.merge(
    mktcap,
    on=["Stkcd", "td_tm3"],
    how="left",
    validate="m:1",
)

size_bp = pd.read_csv(SIZE_BP_PATH, parse_dates=["Trdmnt"], encoding="utf-8-sig")

alt_memory["Trdmnt_for_breakpoint"] = (
    alt_memory["date"].dt.to_period("M").sub(1).dt.to_timestamp()
)

alt_memory = alt_memory.merge(
    size_bp.rename(columns={"Trdmnt": "Trdmnt_for_breakpoint"}),
    on="Trdmnt_for_breakpoint",
    how="left",
    validate="m:1",
)

alt_memory = alt_memory.dropna(
    subset=["mktcap_tm3", "marketCap_90th_percentile"]
).copy()

alt_memory["size90plus"] = (
    alt_memory["mktcap_tm3"] > alt_memory["marketCap_90th_percentile"]
).astype("int8")

# 生成两类替代线索变量
event_specs = {
    "cueMajorNarrowOther": "altMajorNarrowOther",
    "cueForecast": "altForecast",
}

feature_map = {
    "lag1hiAEVs90p": "lag1hiAEV",
    "lag1hiavols90p": "lag1hiavol",
    "lag1hiavolSs90p": "lag1hiavolS",

    "lag1hi_sue1_abss90p": "lag1hi_sue1_abs",
    "lag1hi_sue2_abss90p": "lag1hi_sue2_abs",
    "lag1hi_sue3_abss90p": "lag1hi_sue3_abs",

    "lag1hi_sue1_poss90p": "lag1hi_sue1_pos",
    "lag1hi_sue1_negs90p": "lag1hi_sue1_neg",
    "lag1hi_sue2_poss90p": "lag1hi_sue2_pos",
    "lag1hi_sue2_negs90p": "lag1hi_sue2_neg",
    "lag1hi_sue3_poss90p": "lag1hi_sue3_pos",
    "lag1hi_sue3_negs90p": "lag1hi_sue3_neg",

    "lag1sue1poss90p": "lag1sue1pos",
    "lag1sue2poss90p": "lag1sue2pos",
    "lag1sue3poss90p": "lag1sue3pos",
    "lag1sue1negs90p": "lag1sue1neg",
    "lag1sue2negs90p": "lag1sue2neg",
    "lag1sue3negs90p": "lag1sue3neg",

    "lag1topMovers90p": "lag1topMover",
}
for prefix, event_col in event_specs.items():
    alt_memory[f"{prefix}_lag1s90p"] = (
        alt_memory[event_col]
        * alt_memory["quarterLag1Alt"].fillna(0)
        * alt_memory["size90plus"]
    ).astype("int8")
# altForecast == 1：线索公司当天发生业绩预告/快报
# quarterLag1Alt == 1：线索公司和被提示公司有 lag1 记忆
# size90plus == 1：线索公司是大市值公司
# 三个条件都满足，cueForecast_lag1s90p = 1，大市值线索公司发生业绩预告/快报，
# 并且该线索公司与被提示公司存在上一财报期共同公告记忆。

    for suffix, src in feature_map.items():
        alt_memory[f"{prefix}_{suffix}"] = (
            alt_memory[event_col]
            * alt_memory[src].fillna(0)
            * alt_memory["size90plus"]
        ).astype("int8")
# e.g.大市值线索公司发生业绩预告/快报，并且它与被提示公司在上一财报期共同公告记忆中属于
# “双方高媒体关注”的强记忆关系、历史记忆形成时双方都是 top mover、历史记忆形成时双方 SUE1 都为正等。    

# 生成低/高干扰度变量
alt_memory["year"] = alt_memory["date"].dt.year

for prefix in event_specs:
    base = f"{prefix}_lag1s90p"
    score = f"{prefix}_nrFirmsAnnLag1s90p"
    group = f"{prefix}_interferenceGroupLag1s90p"

    alt_memory[score] = np.where(
        alt_memory[base].eq(1),
        alt_memory["nrFirmsAnnLag1Alt"],
        np.nan,
    )
# 只有当这一行确实是一个有效替代线索（满足3个条件）时，才把历史共同公告拥挤度 nrFirmsAnnLag1Alt 填进去；
# 如果不是有效线索，就填缺失值。

    alt_memory[group] = (
        alt_memory.groupby("year")[score]
        .transform(safe_qcut_2)
    ) # 分成高低两组

    alt_memory[f"{prefix}_lowInterfLag1s90p"] = (
        alt_memory[group].astype("float").eq(1)
    ).astype("int8")

    alt_memory[f"{prefix}_highInterfLag1s90p"] = (
        alt_memory[group].astype("float").eq(2)
    ).astype("int8")


# 折叠到被提示公司 × 日期
cue_cols = [
    c for c in alt_memory.columns
    if c.startswith("cueMajorNarrowOther_") or c.startswith("cueForecast_")
]

alt_cues = (
    alt_memory
    .groupby(["StkcdRecall", "date"], as_index=False)[cue_cols]
    .max()
)# 以最大值的方式折叠

print(alt_cues.shape)
print(alt_cues[cue_cols].sum().sort_values(ascending=False).head(20))


alt_cues.to_csv(
    ALT_CUES_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d",
)

print("替代线索表已保存：", ALT_CUES_OUTPUT_PATH)
print("输出行数：", len(alt_cues))
print("输出列数：", alt_cues.shape[1])