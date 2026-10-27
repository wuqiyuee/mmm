# 在 A 股市场中，Filling in {0,1} 相比完整市场开市时间线少了 34 个股票-日期观测；
# 其中 22 条可由停牌表直接解释，表现为市场开市但个股停牌，且升级表未补为 Filling=1；
# 其余 12 条未被现有停退牌表覆盖，主要集中在 2000–2002 年，较可能与早期历史停牌记录缺失或数据库边界处理有关。

# Stkcd    Trddt        Markettype in_old
#600849 2000-06-30           1       0
#600849 2000-09-12           1       0
#600849 2000-09-13           1       0
#600018 2001-03-26           1       0
#600849 2001-04-27           1       0
#600018 2001-06-25           1       0
#600849 2001-12-31           1       0
#600018 2002-05-31           1       0
#600018 2002-06-03           1       0
#600849 2002-06-28           1       0
#600849 2002-12-30           1       0
#600849 2002-12-31           1       0
#600018 2006-10-20           1       0
#600018 2006-10-23           1       0
#600018 2006-10-24           1       0
#600018 2006-10-25           1       0
#002473 2022-05-05           4       0
#002473 2022-05-06           4       0
#002473 2022-05-09           4       0
#002473 2022-05-10           4       0
#002473 2022-05-11           4       0
#002473 2022-05-12           4       0
#002473 2022-05-13           4       0
#002473 2022-05-16           4       0
#002473 2022-05-17           4       0
#002473 2022-05-18           4       0
#002473 2022-05-19           4       0
#002473 2022-05-20           4       0
#002473 2022-05-23           4       0
#002473 2022-05-24           4       0
#002473 2022-05-25           4       0
#002473 2022-05-26           4       0
#002473 2022-05-27           4       0
#002473 2022-05-30           4       0

import pandas as pd

from pathlib import Path

# 根据脚本位置定位项目根目录，整个项目移动后无需修改路径。
ROOT = Path(__file__).resolve().parents[4]


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 300)

NEW_PATH = str(ROOT / '下载数据/CSMAR/个股日度收益率（升级测试）/CSMAR_日个股回报率(升级测试)_2000_2025.csv')
CALE_PATH = str(ROOT / '下载数据/CSMAR/日历文件2000-01-01至2025-12-31/TRD_Cale.csv')
SUSP_PATH = str(ROOT / '下载数据/CSMAR/交易停复牌数据表2000-01-01至2025-12-31/TSR_Stkstat_backup.csv')

A_MARKETS = [1, 4, 16, 32]

def read_csv_auto(path, usecols=None):
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols, low_memory=False)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"读取失败: {path}\n最后报错: {last_err}")

def norm_stkcd(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)

# ========= 1) 读升级表、市场日历、停退牌表 =========
new = read_csv_auto(
    NEW_PATH,
    usecols=["Stkcd", "Trddt", "Markettype", "Filling", "Trdsta", "Dnshrtrd", "Dnvaltrd", "Clsprc"]
)
cale = read_csv_auto(
    CALE_PATH,
    usecols=["Markettype", "Clddt", "State"]
)
susp = read_csv_auto(
    SUSP_PATH,
    usecols=["Stkcd", "Stknme", "Type", "Suspdate", "Resmdate", "Reason"]
)

# ========= 2) 标准化 =========
new["Stkcd"] = norm_stkcd(new["Stkcd"])
new["Trddt"] = pd.to_datetime(new["Trddt"], errors="coerce")
new["Filling"] = new["Filling"].astype(str).str.strip()

cale["Clddt"] = pd.to_datetime(cale["Clddt"], errors="coerce")
cale["State"] = cale["State"].astype(str).str.strip().str.upper()

susp["Stkcd"] = norm_stkcd(susp["Stkcd"])
susp["Suspdate"] = pd.to_datetime(susp["Suspdate"], errors="coerce")
susp["Resmdate"] = pd.to_datetime(susp["Resmdate"], errors="coerce")

new = new.dropna(subset=["Stkcd", "Trddt", "Markettype", "Filling"]).copy()
cale = cale.dropna(subset=["Markettype", "Clddt", "State"]).copy()
susp = susp.dropna(subset=["Stkcd", "Suspdate", "Type"]).copy()

new = new.loc[new["Markettype"].isin(A_MARKETS)].copy()
cale = cale.loc[cale["Markettype"].isin(A_MARKETS)].copy()

# ========= 3) 重新构造那 34 条 =========
new_open = (
    new.loc[new["Filling"].isin(["0", "1"]), ["Stkcd", "Trddt", "Markettype"]]
    .drop_duplicates()
    .copy()
)

stock_range = (
    new.groupby(["Stkcd", "Markettype"], as_index=False)
    .agg(first_dt=("Trddt", "min"), last_dt=("Trddt", "max"))
)

open_cale = (
    cale.loc[cale["State"] == "O", ["Markettype", "Clddt"]]
    .rename(columns={"Clddt": "Trddt"})
    .drop_duplicates()
    .copy()
)

market_open = stock_range.merge(open_cale, on="Markettype", how="left")
market_open = market_open.loc[
    (market_open["Trddt"] >= market_open["first_dt"]) &
    (market_open["Trddt"] <= market_open["last_dt"])
].copy()

market_open = market_open[["Stkcd", "Trddt", "Markettype"]].drop_duplicates()

cmp = market_open.merge(
    new_open,
    on=["Stkcd", "Trddt", "Markettype"],
    how="left",
    indicator=True
)

missing34 = cmp.loc[cmp["_merge"] == "left_only", ["Stkcd", "Trddt", "Markettype"]].copy()
missing34 = missing34.sort_values(["Trddt", "Markettype", "Stkcd"]).reset_index(drop=True)

print("=" * 100)
print("A股 market_open_timeline 里有、但升级表 Filling in {0,1} 没有的 34 条")
print("=" * 100)
print(missing34.to_string(index=False))

# ========= 4) 直接与停退牌表匹配 =========
# 检验规则：看 Trddt 是否落在 [Suspdate, Resmdate) 区间
# 若 Resmdate 为空，则视为从 Suspdate 起一直生效
m = missing34.merge(
    susp[["Stkcd", "Stknme", "Type", "Suspdate", "Resmdate", "Reason"]],
    on="Stkcd",
    how="left"
)

mask = (
    m["Suspdate"].notna() &
    (m["Trddt"] >= m["Suspdate"]) &
    (
        m["Resmdate"].isna() |
        (m["Trddt"] < m["Resmdate"])
    )
)

hits = m.loc[mask].copy()
hits = hits.sort_values(["Trddt", "Markettype", "Stkcd", "Suspdate"]).reset_index(drop=True)

print("\n" + "=" * 100)
print("这 34 条中，落在停退牌区间内的记录")
print("=" * 100)
if len(hits) == 0:
    print("0 条：没有任何一条落在停退牌区间内。")
else:
    print(hits[["Stkcd", "Stknme", "Trddt", "Markettype", "Type", "Suspdate", "Resmdate", "Reason"]].to_string(index=False))

# ========= 5) 汇总 =========
hit_keys = hits[["Stkcd", "Trddt", "Markettype"]].drop_duplicates().copy()

summary = (
    hit_keys.merge(missing34, on=["Stkcd", "Trddt", "Markettype"], how="right", indicator=True)
)

matched_cnt = int((summary["_merge"] == "both").sum())
unmatched_cnt = int((summary["_merge"] == "right_only").sum())

print("\n" + "=" * 100)
print("匹配汇总")
print("=" * 100)
print(f"34 条里，被停退牌表解释到的键数: {matched_cnt}")
print(f"34 条里，仍未被停退牌表解释到的键数: {unmatched_cnt}")

if len(hits) > 0:
    type_tab = (
        hits[["Stkcd", "Trddt", "Markettype", "Type"]]
        .drop_duplicates()
        ["Type"]
        .value_counts()
        .sort_index()
    )
    print("\n按 Type 分布：")
    print(type_tab)

# ========= 6) 未匹配到的，再看最近的停退牌事件 =========
unmatched = missing34.merge(
    hit_keys.assign(hit=1),
    on=["Stkcd", "Trddt", "Markettype"],
    how="left"
)
unmatched = unmatched.loc[unmatched["hit"].isna(), ["Stkcd", "Trddt", "Markettype"]].copy()

print("\n" + "=" * 100)
print("未被停退牌表解释到的明细")
print("=" * 100)
if len(unmatched) == 0:
    print("0 条：全部都能被停退牌表解释。")
else:
    print(unmatched.to_string(index=False))

    near = unmatched.merge(
        susp[["Stkcd", "Stknme", "Type", "Suspdate", "Resmdate", "Reason"]],
        on="Stkcd",
        how="left"
    ).copy()

    near["dist_to_susp"] = (near["Trddt"] - near["Suspdate"]).dt.days.abs()
    near["dist_to_resm"] = (near["Trddt"] - near["Resmdate"]).dt.days.abs()

    # 取每条缺失键最近的一条停退牌事件
    near["score"] = near[["dist_to_susp", "dist_to_resm"]].min(axis=1)
    near_best = (
        near.sort_values(["Stkcd", "Trddt", "score", "Suspdate"])
        .groupby(["Stkcd", "Trddt", "Markettype"], as_index=False)
        .first()
    )

    print("\n" + "=" * 100)
    print("每条未匹配键最近的停退牌事件（帮助判断是不是边界问题）")
    print("=" * 100)
    print(
        near_best[["Stkcd", "Stknme", "Trddt", "Markettype", "Type", "Suspdate", "Resmdate", "score", "Reason"]]
        .to_string(index=False)
    )