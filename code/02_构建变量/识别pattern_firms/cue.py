import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)#不自动转行
pd.set_option('display.width', 1000)#设置很长的虚拟宽度

# ==========================================
# 1. 文件路径
# ==========================================
pattern_obs_path = "/Users/qiyuewu/Desktop/mmm复现数据包/pattern_output/pattern_observations.csv"
reg_sample_path = "/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/DGTW_final_sample.csv"

out_sample_path = "/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/DGTW_final_sample_with_cue.csv"
out_summary_path = "/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/cue_summary_two_rows.csv"

# ==========================================
# 2. 小工具：自动识别列名
# ==========================================
def pick_col(df, candidates, name):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"找不到{name}列。候选列名={candidates}，实际列名={list(df.columns)[:50]}")

# ==========================================
# 3. 读取 pattern observations，并构造 cue_daily
# ==========================================
print("正在读取 pattern observations ...")
po = pd.read_csv(pattern_obs_path, low_memory=False)

po_firm_col = pick_col(
    po,
    ["Stkcd6", "Stkcd", "stkcd", "firm", "firm_id", "code"],
    "pattern observations 的公司代码"
)
po_accper_col = pick_col(
    po,
    ["Accper", "accper", "report_date", "fiscal_date"],
    "pattern observations 的期末日"
)
po_actudt_col = pick_col(
    po,
    ["Actudt", "actudt", "Annodt", "annodt", "ea_date", "event_date"],
    "pattern observations 的公告日"
)

if "is_pattern_observation" not in po.columns:
    raise ValueError("pattern_observations.csv 里没有 is_pattern_observation 列。")

po[po_accper_col] = pd.to_datetime(po[po_accper_col], errors="coerce")
po[po_actudt_col] = pd.to_datetime(po[po_actudt_col], errors="coerce")

po = po.loc[po["is_pattern_observation"] == 1].copy()
po = po.dropna(subset=[po_firm_col, po_accper_col, po_actudt_col])

# 按公司内的 Accper 顺序，取上一期的公告日
po = po.sort_values([po_firm_col, po_accper_col])
po["prev_actudt"] = po.groupby(po_firm_col)[po_actudt_col].shift(1)

base = po[[po_firm_col, po_accper_col, po_actudt_col, "prev_actudt"]].dropna().copy()

# 上一期“同日公告” -> 形成记忆关联
pairs = base.merge(
    base,
    on="prev_actudt",
    suffixes=("_i", "_j")
)

# 去掉自己和自己
pairs = pairs.loc[pairs[f"{po_firm_col}_i"] != pairs[f"{po_firm_col}_j"]].copy()

# cueing firm j 在 t = Actudt_j 公告 -> 对 firm i 形成 cue
cue_daily = (
    pairs[[f"{po_firm_col}_i", f"{po_actudt_col}_j", f"{po_firm_col}_j"]]
    .rename(columns={
        f"{po_firm_col}_i": "firm_i",
        f"{po_actudt_col}_j": "t",
        f"{po_firm_col}_j": "cueing_firm"
    })
    .groupby(["firm_i", "t"], as_index=False)
    .agg(Number_of_cues=("cueing_firm", "nunique"))
)

cue_daily["Cue"] = 1

print("cue_daily 构造完成。")
print(cue_daily.head())

# ==========================================
# 4. 读取进入回归的样本，并合并 cue 变量
# ==========================================
print("正在读取回归样本 DGTW_final_sample ...")
reg = pd.read_csv(reg_sample_path, low_memory=False)

reg_firm_col = pick_col(
    reg,
    ["Stkcd6", "Stkcd", "stkcd", "firm", "firm_id", "code", "permno"],
    "回归样本的公司代码"
)
reg_date_col = pick_col(
    reg,
    ["date", "Date", "trddt", "Trddt", "Clddt", "Actudt", "event_date", "t"],
    "回归样本的日期"
)

reg[reg_date_col] = pd.to_datetime(reg[reg_date_col], errors="coerce")
reg = reg.dropna(subset=[reg_firm_col, reg_date_col]).copy()

# 保留原始列名，同时构造 merge key
reg["firm_i"] = reg[reg_firm_col]
reg["t"] = reg[reg_date_col]

# 左连接到“进入回归的样本”
reg = reg.merge(cue_daily, on=["firm_i", "t"], how="left")

# Cue 在回归样本全体 firm-days 上定义；无 cue 的记为 0
reg["Cue"] = reg["Cue"].fillna(0).astype(int)

# Number_of_cues 只在 Cue=1 的样本上有意义；其他设为缺失
reg.loc[reg["Cue"] == 0, "Number_of_cues"] = np.nan

# ==========================================
# 5. 按 MMM Table 1 口径做两行描述统计
#    - Cue (dummy): 用全部回归样本 firm-days
#    - Number of cues: 只在 Cue=1 的回归样本 firm-days
# ==========================================
def summarize_series(s: pd.Series):
    s = s.dropna()
    return pd.Series({
        "Mean": s.mean(),
        "p25": s.quantile(0.25),
        "p50": s.quantile(0.50),
        "p75": s.quantile(0.75),
        "Std. Dev.": s.std(ddof=1),
        "Min": s.min(),
        "Max": s.max(),
        "N": int(s.shape[0])
    })

summary = pd.DataFrame({
    "Cue (dummy)": summarize_series(reg["Cue"]),
    "Number of cues": summarize_series(reg["Number_of_cues"])
}).T

summary_fmt = summary.copy()
for c in ["Mean", "p25", "p50", "p75", "Std. Dev.", "Min", "Max"]:
    summary_fmt[c] = summary_fmt[c].map(lambda x: f"{x:.4f}")
summary_fmt["N"] = summary_fmt["N"].map(lambda x: f"{x:,}")

print("\n=== MMM风格两行描述统计（基于进入回归的样本） ===")
print(summary_fmt)

# ==========================================
# 6. 保存结果
# ==========================================
reg.to_csv(out_sample_path, index=False, encoding="utf-8-sig")
summary.to_csv(out_summary_path, encoding="utf-8-sig")

print(f"\n已保存带 Cue 变量的回归样本到: {out_sample_path}")
print(f"已保存两行描述统计到: {out_summary_path}")

# ==========================================
# 7. 额外打印三个最关键的数字
# ==========================================
print("\n=== 三个最关键的检查量 ===")
print("Cue mean =", reg["Cue"].mean())
print("Number_of_cues mean | Cue=1 =", reg.loc[reg["Cue"] == 1, "Number_of_cues"].mean())
print("N(Cue=1 firm-days) =", reg.loc[reg["Cue"] == 1].shape[0])

print("\n=== Number_of_cues 最大的前30个 firm-days ===")
print(
    reg.loc[reg["Cue"] == 1]
      .sort_values("Number_of_cues", ascending=False)[["firm_i","t","Number_of_cues"]]
      .head(30)
      .to_string(index=False)
)
print(reg.loc[reg["Cue"] == 1].groupby("t")["Number_of_cues"].max().sort_values(ascending=False).head(30))