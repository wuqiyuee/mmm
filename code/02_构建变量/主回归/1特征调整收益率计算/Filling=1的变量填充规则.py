#对于 Filling=1：
# Stkcd / Trddt / Markettype：只是定位这条补齐记录是谁、哪天、哪个市场
# Trdsta：保留状态码，不缺失
# Dretwd：通常填 0
# Dsmvosd / Dsmvtll：通常沿用前一有效记录的冻结值

import pandas as pd
import numpy as np

from pathlib import Path

# 根据脚本位置定位项目根目录，整个项目移动后无需修改路径。
ROOT = Path(__file__).resolve().parents[4]


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

NEW_PATH = str(ROOT / '下载数据/CSMAR/个股日度收益率（升级测试）/CSMAR_日个股回报率(升级测试)_2000_2025.csv')
SUSP_PATH = str(ROOT / '下载数据/CSMAR/交易停复牌数据表2000-01-01至2025-12-31/TSR_Stkstat_backup.csv')

# 可选：只看A股
FILTER_MARKETS = [1, 4, 16, 32]
# FILTER_MARKETS = None

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

# =========================
# 1) 读取
# =========================
usecols = [
    "Stkcd", "Trddt", "Markettype", "Filling",
    "Trdsta",
    "Opnprc", "Hiprc", "Loprc", "Clsprc",
    "Dnshrtrd", "Dnvaltrd",
    "Dsmvosd", "Dsmvtll", "Dsmvtl2", "Dsmvtl3",
    "Dretwd", "Dretnd",
    "Adjprcwd", "Adjprcnd",
    "Capchgdt", "Ahshrtrd_D", "Ahvaltrd_D",
    "PreClosePrice", "ChangeRatio", "LimitDown", "LimitUp", "LimitStatus"
]
df = read_csv_auto(NEW_PATH, usecols=usecols)

df["Stkcd"] = norm_stkcd(df["Stkcd"])
df["Trddt"] = pd.to_datetime(df["Trddt"], errors="coerce")
df["Filling"] = df["Filling"].astype(str).str.strip()

if FILTER_MARKETS is not None:
    df = df.loc[df["Markettype"].isin(FILTER_MARKETS)].copy()

df = df.dropna(subset=["Stkcd", "Trddt", "Markettype", "Filling"]).copy()
df = df.sort_values(["Stkcd", "Markettype", "Trddt"]).reset_index(drop=True)

f1 = df.loc[df["Filling"] == "1"].copy()

print("=" * 100)
print("基础规模")
print("=" * 100)
print(f"样本总行数: {len(df):,}")
print(f"Filling=1 行数: {len(f1):,}")

# =========================
# 2) 先看“固定值型填充”
# =========================
print("\n" + "=" * 100)
print("A. 固定值型填充检查")
print("=" * 100)

fixed_cols = ["Dnshrtrd", "Dnvaltrd", "Dretwd", "Dretnd", "Trdsta", "ChangeRatio", "LimitStatus"]
rows = []
for c in fixed_cols:
    if c not in f1.columns:
        continue
    s = pd.to_numeric(f1[c], errors="coerce")
    rows.append({
        "column": c,
        "missing_rate": s.isna().mean(),
        "zero_rate_among_all_rows": (s.fillna(np.nan) == 0).mean() if s.notna().any() else np.nan,
        "nonzero_count": int(((s != 0) & s.notna()).sum()) if s.notna().any() else 0
    })
print(pd.DataFrame(rows).to_string(index=False))

# =========================
# 3) 看是否“沿用上一条/下一条”
# =========================
check_cols = [
    "Opnprc", "Hiprc", "Loprc", "Clsprc",
    "Adjprcwd", "Adjprcnd",
    "PreClosePrice",
    "Dsmvosd", "Dsmvtll", "Dsmvtl2", "Dsmvtl3",
    "LimitDown", "LimitUp"
]
check_cols = [c for c in check_cols if c in df.columns]

g = df.groupby(["Stkcd", "Markettype"], group_keys=False)

for c in check_cols:
    df[f"prev_{c}"] = g[c].shift(1)
    df[f"next_{c}"] = g[c].shift(-1)

f1 = df.loc[df["Filling"] == "1"].copy()

def safe_equal(a, b):
    return a.notna() & b.notna() & (a == b)

rows = []
for c in check_cols:
    s = pd.to_numeric(f1[c], errors="coerce")
    prev = pd.to_numeric(f1[f"prev_{c}"], errors="coerce")
    nxt = pd.to_numeric(f1[f"next_{c}"], errors="coerce")

    eq_prev = safe_equal(s, prev)
    eq_next = safe_equal(s, nxt)

    rows.append({
        "column": c,
        "nonmissing_rate": 1 - s.isna().mean(),
        "equal_prev_rate_in_all_f1": eq_prev.mean(),
        "equal_prev_rate_in_nonmissing": eq_prev.sum() / s.notna().sum() if s.notna().sum() else np.nan,
        "equal_next_rate_in_all_f1": eq_next.mean(),
        "equal_next_rate_in_nonmissing": eq_next.sum() / s.notna().sum() if s.notna().sum() else np.nan,
    })

print("\n" + "=" * 100)
print("B. 是否沿用上一条/下一条数值")
print("=" * 100)
print(pd.DataFrame(rows).sort_values("equal_prev_rate_in_nonmissing", ascending=False).to_string(index=False))

# =========================
# 4) 只看“前后都相等”的情形
# =========================
rows = []
for c in check_cols:
    s = pd.to_numeric(f1[c], errors="coerce")
    prev = pd.to_numeric(f1[f"prev_{c}"], errors="coerce")
    nxt = pd.to_numeric(f1[f"next_{c}"], errors="coerce")

    both_same = s.notna() & prev.notna() & nxt.notna() & (s == prev) & (s == nxt)
    rows.append({
        "column": c,
        "both_prev_and_next_equal_rate": both_same.mean(),
        "both_prev_and_next_equal_rate_in_nonmissing": both_same.sum() / s.notna().sum() if s.notna().sum() else np.nan
    })

print("\n" + "=" * 100)
print("C. 当前值是否同时等于前值和后值")
print("=" * 100)
print(pd.DataFrame(rows).sort_values("both_prev_and_next_equal_rate_in_nonmissing", ascending=False).to_string(index=False))

# =========================
# 5) 停牌/暂停上市命中率
# =========================
susp = read_csv_auto(
    SUSP_PATH,
    usecols=["Stkcd", "Type", "Suspdate", "Resmdate"]
)
susp["Stkcd"] = norm_stkcd(susp["Stkcd"])
susp["Suspdate"] = pd.to_datetime(susp["Suspdate"], errors="coerce")
susp["Resmdate"] = pd.to_datetime(susp["Resmdate"], errors="coerce")
susp = susp.dropna(subset=["Stkcd", "Suspdate", "Type"]).copy()
susp = susp.loc[susp["Type"].isin([1, 3])].copy()

m = f1[["Stkcd", "Trddt", "Markettype"]].merge(
    susp[["Stkcd", "Type", "Suspdate", "Resmdate"]],
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

hit = m.loc[mask, ["Stkcd", "Trddt", "Markettype", "Type"]].drop_duplicates()

print("\n" + "=" * 100)
print("D. Filling=1 落在停牌/暂停上市区间的比例")
print("=" * 100)
print(f"Filling=1 总键数: {f1[['Stkcd','Trddt','Markettype']].drop_duplicates().shape[0]:,}")
print(f"命中停牌/暂停上市区间的键数: {hit[['Stkcd','Trddt','Markettype']].drop_duplicates().shape[0]:,}")
print(f"占比: {hit[['Stkcd','Trddt','Markettype']].drop_duplicates().shape[0] / f1[['Stkcd','Trddt','Markettype']].drop_duplicates().shape[0]:.4%}")
if len(hit) > 0:
    print("\nType 分布：")
    print(hit["Type"].value_counts().sort_index())

# =========================
# 6) 抽样看若干 Filling=1 前后窗口
# =========================
sample_cols = [
    "Stkcd", "Markettype", "Trddt", "Filling", "Trdsta",
    "Opnprc", "Hiprc", "Loprc", "Clsprc",
    "Dnshrtrd", "Dnvaltrd", "Dretwd", "Dretnd",
    "Adjprcwd", "Adjprcnd",
    "Dsmvosd", "Dsmvtll", "Dsmvtl2", "Dsmvtl3",
    "PreClosePrice", "LimitDown", "LimitUp"
]
sample_cols = [c for c in sample_cols if c in df.columns]

print("\n" + "=" * 100)
print("E. 随机抽 10 个 Filling=1，查看前后各 2 条记录")
print("=" * 100)

sample_idx = f1.sample(min(10, len(f1)), random_state=1).index.tolist()
for idx in sample_idx:
    r = df.loc[idx]
    sub = df.loc[max(0, idx-2):min(len(df)-1, idx+2), sample_cols].copy()
    print("\n" + "-" * 100)
    print(f"Stkcd={r['Stkcd']}, Markettype={r['Markettype']}, Trddt={str(r['Trddt'].date())}")
    print(sub.to_string(index=False))