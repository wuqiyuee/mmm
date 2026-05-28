# 筛选A股市场的股票：清洗金融行业、上市未满一年、ST状态的观测值，用于计算DGTW的分组标签
# 清洗的方式为：
# ① 将个股填充至完整序列,计算两个指标（过去1年交易的天数和过去1个月交易天数）
# ② 删除金融行业、上市未满一年、ST观测值
# ③ 再用td匹配回完整面板上
#
# （不合格的日度观测值除了Stkcd、Trddt和td以外的其他变量都会变成NaN）
# 使用shift+rolling计算动量指标，只要窗口期内有一条缺失值就mom指标就会为缺失值
# (要求窗口期内不能有一条不合格的观测值！！！！)
#
# 4️⃣再在6月份的年度数据上筛选掉过去1年交易的天数不足120天的。

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import logging
import sys

ROOT = Path("/Users/qiyuewu/Desktop/mmm复现数据包")
sys.path.append(str(ROOT / "code"))
from mytools import normalize_stock_code

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
logging.getLogger("matplotlib.font_manager").disabled = True

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# =============================================================================
# 路径设置
# =============================================================================

TRD_Dalyr_path = ROOT/'中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_2000_2025.csv'
industry_path = ROOT / "input/CSMAR/上市公司行业分类表/STK_INDUSTRYCLASS.csv"
TRD_Co_path = ROOT / "input/CSMAR/公司文件/TRD_Co.csv"
TRD_Cale_td_path = ROOT / "中间结果/01_清洗后样本/TRD_Cale_td.csv"


out_file = ROOT / "中间结果/01_清洗后样本/TRD_Dalyr_DGTW.csv"
full_panel_path = ROOT/"中间结果/01_清洗后样本/full_panel.csv"
out_file.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# 读取日度数据、保留 A 股市场数据、标准化股票代码和日期
# =========================================================

TRD_Dalyr = pd.read_csv(
    TRD_Dalyr_path,
    encoding = 'utf-8',
    usecols = ['Stkcd','Trddt',
               "Markettype","Filling",'Trdsta',
               'Dnshrtrd','Dretwd','Dretnd','Dsmvosd','Dsmvtll'],
    parse_dates=['Trddt']
)

TRD_Dalyr['Stkcd'] = TRD_Dalyr['Stkcd'].apply(normalize_stock_code)
TRD_Dalyr["year"] = TRD_Dalyr["Trddt"].dt.year

for col in ['Dnshrtrd','Dretwd','Dretnd','Dsmvosd','Dsmvtll']:
    TRD_Dalyr[col] = pd.to_numeric(TRD_Dalyr[col], errors="coerce")

TRD_Dalyr = TRD_Dalyr.rename(
    columns={
        'Dretwd':'ret',
        'Dretnd':'retn',
    }
)

TRD_Dalyr['sizew'] = TRD_Dalyr['Dsmvosd']*1000
TRD_Dalyr['size'] = TRD_Dalyr['Dsmvtll']*1000

TRD_Dalyr = TRD_Dalyr.drop(columns=['Dsmvosd','Dsmvtll'])

# 筛选 A 股市场的股票，包括主板、科创板、中小板等。
TRD_Dalyr = TRD_Dalyr.loc[
    (TRD_Dalyr["Markettype"].isin([1, 4, 16, 32])) &
    (TRD_Dalyr["Filling"].isin([0, 1]))
    # & (TRD_Dalyr["year"].between(2003, 2025))
].copy()

# =========================================================
# 按照市场统一的交易日完整填充个股交易日历
# =========================================================
TRD_Cale_td = pd.read_csv(
    TRD_Cale_td_path,
    encoding='utf-8',
)
TRD_Cale_td['Trddt'] = pd.to_datetime(TRD_Cale_td['Trddt'])
TRD_Dalyr_td = pd.merge(
    TRD_Dalyr,
    TRD_Cale_td[['Trddt','td']],
    on=['Trddt'],
    how='left'
)
print("没有匹配到td的观测数字：",TRD_Dalyr_td['td'].isna().sum())# 未匹配上的为0
TRD_Dalyr_td = TRD_Dalyr_td.dropna(subset=['td']).copy()
TRD_Dalyr_td['td'] = TRD_Dalyr_td['td'].astype(int)

TRD_Dalyr_td = (
    TRD_Dalyr_td
    .sort_values(["Stkcd", "td"])
    .reset_index(drop=True)
)

# 先检查是否存在同一股票同一交易日的重复观测
dup_n = TRD_Dalyr_td.duplicated(["Stkcd", "td"]).sum()
print("Stkcd-td 重复观测数：", dup_n)# 重复值为0

if dup_n > 0:
    print(
        TRD_Dalyr_td.loc[
            TRD_Dalyr_td.duplicated(["Stkcd", "td"], keep=False),
            ["Stkcd", "Trddt", "td"]
        ].head(30)
    )

    # 如果确认只是重复记录，可以保留最后一条
    TRD_Dalyr_td = (
        TRD_Dalyr_td
        .sort_values(["Stkcd", "td", "Trddt"])
        .drop_duplicates(["Stkcd", "td"], keep="last")
        .reset_index(drop=True)
    )

# 保存原始列顺序
orig_cols = TRD_Dalyr_td.columns.tolist()
#  TRD_Dalyr3_td.columns取TRD_Dalyr3_td的列名，默认格式为index，.tolist()转为列表

# 每只股票自己的 td 起点和终点
td_range = (
    TRD_Dalyr_td
    .groupby("Stkcd", as_index=False)["td"]
    .agg(td_min="min", td_max="max")
)

# 构造完整 Stkcd-td 面板
lengths = (td_range["td_max"] - td_range["td_min"] + 1).astype(int).to_numpy()

full_panel = pd.DataFrame({
    "Stkcd": np.repeat(td_range["Stkcd"].to_numpy(), lengths),
    # td_range["Stkcd"].to_numpy(),把td_range里面的Stkcd这一列转换为numpy数组
    # np.repeat(数组，重复次数)：把数组里的每个元素，按对应次数重复，最后拼接成一个大数组
    # 也就是把Stkcd重复lengths这么多次
    "td": np.concatenate([ # 把列表里的所有小数组，首位拼接成一个完整的大数组
        np.arange(lo, hi + 1, dtype=np.int64) # 生成从起始td到结束td的连续整数序列
        for lo, hi in zip(td_range["td_min"], td_range["td_max"]) 
        # zip是python内置函数，作用是把两个列表的元素按位置一一配对，得到每只股票的(起始td，结束td)
        # 所以lo，hi对应的就是每只股票的的起始td和结束td
    ])
})
# Stkcd 重复起始日期长度那么多次
# 按照起始日期填充序列
# 两列拼起来就构成了完整序列

# 用交易日历把 td 转回 Trddt
full_panel = pd.merge(
    full_panel,
    TRD_Cale_td[["td", "Trddt"]],
    on="td",
    how="left"
)

full_panel.to_csv(
    full_panel_path,
    encoding='utf-8',
    index=False
)

# 原表中删掉 Trddt，避免合并后出现 Trddt_x / Trddt_y
# full_panel 里的 Trddt 是交易日历给出的完整交易日期为左表，保证了td连续
TRD_Dalyr1 = pd.merge(
    full_panel,
    TRD_Dalyr_td.drop(columns=["Trddt"]),
    on=["Stkcd", "td"],
    how="left"
)

# 恢复列顺序
TRD_Dalyr1 = TRD_Dalyr1[orig_cols]


print("补齐前样本数：", len(TRD_Dalyr))
print("补齐后样本数：", len(TRD_Dalyr1))
print("新增补齐观测数：", len(TRD_Dalyr1) - len(TRD_Dalyr))

###  检查补齐后每只股票内部 td 是否连续

tmp = TRD_Dalyr1.sort_values(["Stkcd", "td"]).copy()
tmp["td_gap"] = tmp.groupby("Stkcd")["td"].diff()

print(tmp["td_gap"].value_counts(dropna=False).head(20))

gap = tmp.loc[
    tmp["td_gap"].notna() & (tmp["td_gap"] != 1)
].copy()

print("td 不连续的观测数：", len(gap))
print("td 不连续的股票数：", gap["Stkcd"].nunique())

if len(gap) > 0:
    print(gap[["Stkcd", "Trddt", "td", "td_gap"]].head(30))

print()

# =========================================================
# 在完整面板上计算一些变量
# =========================================================

TRD_Dalyr1= (
    TRD_Dalyr1
    .sort_values(["Stkcd", "td"])
    .reset_index(drop=True)
)

TRD_Dalyr1["has_volume"] = (
    TRD_Dalyr1["Dnshrtrd"].fillna(0).gt(0)
).astype("int8")

TRD_Dalyr1 = TRD_Dalyr1.drop(columns=['Dnshrtrd'])

TRD_Dalyr1 = (
    TRD_Dalyr1
    .sort_values(["Stkcd", "Trddt"])
    .reset_index(drop=True)
    .copy()
)

TRD_Dalyr1["trade_days_1m"] = (
    TRD_Dalyr1
    .groupby("Stkcd")
    .rolling(
        window="30D",
        on="Trddt",
        min_periods=15,
        closed="right"
    )["has_volume"]
    .sum()
    .reset_index(level=0, drop=True)# 恢复0级索引（stkcd列）为普通列，并且把它删除
    .to_numpy()
)

TRD_Dalyr1["trade_days_1y"] = (
    TRD_Dalyr1
    .groupby("Stkcd")
    .rolling(
        window="365D",
        on="Trddt",
        min_periods=120,
        closed="right"
    )["has_volume"]
    .sum()
    .reset_index(level=0, drop=True)
    .to_numpy()
)

TRD_Dalyr1["trade_days_1m"] = TRD_Dalyr1["trade_days_1m"].astype("Int16")
TRD_Dalyr1["trade_days_1y"] = TRD_Dalyr1["trade_days_1y"].astype("Int16")

# =========================================================
# 按“当期行业”剔除金融行业股票：使用 STK_INDUSTRYCLASS
# =========================================================
industry = pd.read_csv(
    industry_path,
    encoding='utf-8',
    usecols=["ImplementDate", "IndustryClassificationID", "Symbol", "IndustryCode"],
    parse_dates=["ImplementDate"]
)

industry = industry.dropna(subset=["ImplementDate", "Symbol", "IndustryCode"]).copy()

industry["IndustryCode"] = (
    industry["IndustryCode"]
    .astype(str)
    .str.strip()
    .str.upper()
)

industry = industry.loc[
    industry["IndustryClassificationID"].isin(["P0201", "P0207", "P0221"])
].copy()

industry["is_fin"] = (
    ((industry["IndustryClassificationID"] == "P0201") & industry["IndustryCode"].str.startswith("I")) |
    ((industry["IndustryClassificationID"] == "P0207") & industry["IndustryCode"].str.startswith("J")) |
    ((industry["IndustryClassificationID"] == "P0221") & industry["IndustryCode"].str.startswith("J"))
).astype(int)

industry = industry.rename(columns={
    "Symbol": "Stkcd",
    "IndustryClassificationID": "std_id"
})

industry['Stkcd'] = industry['Stkcd'].apply(normalize_stock_code)

industry = (
    industry
    .sort_values(["Stkcd", "std_id", "ImplementDate"])
    .drop_duplicates(["Stkcd", "std_id", "ImplementDate"], keep="last")
)

TRD_Dalyr1["std_id"] = np.select(
    [
        TRD_Dalyr1["Trddt"] < pd.Timestamp("2012-10-26"),
        (TRD_Dalyr1["Trddt"] >= pd.Timestamp("2012-10-26")) & (TRD_Dalyr1["Trddt"] < pd.Timestamp("2023-05-01")),
        TRD_Dalyr1["Trddt"] >= pd.Timestamp("2023-05-01")
    ],
    ["P0201", "P0207", "P0221"],
    default="P0201"
)

left = (
    TRD_Dalyr1
    .sort_values(["Trddt", "Stkcd", "std_id"])
    .reset_index(drop=True)
    .copy()
)

right = (
    industry[["ImplementDate", "Stkcd", "std_id", "IndustryCode", "is_fin"]]
    .dropna(subset=["ImplementDate"])
    .sort_values(["ImplementDate", "Stkcd", "std_id"])
    .reset_index(drop=True)
    .copy()
)

TRD_Dalyr_fin = pd.merge_asof(
    left,
    right,
    left_on="Trddt",
    right_on="ImplementDate",
    by=["Stkcd", "std_id"],
    direction="backward",
    allow_exact_matches=True
)

# 检查行业表匹配情况
match_success = TRD_Dalyr_fin[TRD_Dalyr_fin["is_fin"].notna()].copy()
match_fail = TRD_Dalyr_fin[TRD_Dalyr_fin["is_fin"].isna()].copy()

print("=" * 80)
print("1. 匹配总体情况")
print(f"   总观测数：{len(TRD_Dalyr_fin)}")
print(f"   匹配成功：{len(match_success)}（占比 {len(match_success) / len(TRD_Dalyr_fin):.2%}）")
print(f"   匹配失败：{len(match_fail)}（占比 {len(match_fail) / len(TRD_Dalyr_fin):.2%}）")
print("=" * 80)

if len(match_fail) > 0:
    print("\n   --- 涉及股票 ---")
    print(f"   匹配失败的股票共 {match_fail['Stkcd'].nunique()} 只")
    print("   前20只失败次数最多的股票：")
    print(match_fail["Stkcd"].value_counts().head(20).to_string())

original_len = len(TRD_Dalyr_fin)

fin_obs = TRD_Dalyr_fin["is_fin"].fillna(0).eq(1).sum()
fin_stocks = TRD_Dalyr_fin.loc[
    TRD_Dalyr_fin["is_fin"].fillna(0).eq(1),
    "Stkcd"
].nunique()

# is_fin=1的是金融企业。is_fin=0或者是NaN的3只股票是非金融企业（手动校准过了）

TRD_Dalyr2 = TRD_Dalyr_fin.loc[
    TRD_Dalyr_fin["is_fin"].fillna(0) != 1
].copy()

print(f"按当期行业口径识别出曾在某些时点属于金融行业的股票共 {fin_stocks} 只")
print(f"删除的金融行业 股票-日期 观测共 {fin_obs} 条")
print(f"已从日度收益文件中按当期行业剔除金融股，样本从 {original_len} 下降为 {len(TRD_Dalyr2)}")

TRD_Dalyr2 = TRD_Dalyr2.drop(columns=["std_id", "ImplementDate", "is_fin"]).copy()

# =========================================================
# 删除上市不满 1 年的日度观测值
# =========================================================
TRD_Co = pd.read_csv(
    TRD_Co_path,
    encoding='utf-8',
    usecols=["Stkcd", "Listdt"],
    parse_dates=["Listdt"]
)

TRD_Co['Stkcd'] = TRD_Co['Stkcd'].apply(normalize_stock_code)

TRD_Co_check = TRD_Co[["Stkcd", "Listdt"]].copy()

dup_stk_mask = TRD_Co_check.duplicated(subset=["Stkcd"], keep=False)
dup_stk = TRD_Co_check.loc[dup_stk_mask].sort_values(["Stkcd", "Listdt"])

print("TRD_Co 总行数：", len(TRD_Co_check))
print("TRD_Co 股票数：", TRD_Co_check["Stkcd"].nunique())
print("Stkcd 重复行数：", len(dup_stk))
print("Stkcd 重复股票数：", dup_stk["Stkcd"].nunique())

if len(dup_stk) > 0:
    print("\n重复 Stkcd 示例：")
    print(dup_stk.head(50))

print("TRD_Co 类型：")
print(TRD_Co.dtypes)

TRD_Dalyr2 = pd.merge(
    TRD_Dalyr2,
    TRD_Co,
    how="left",
    on="Stkcd"
)

TRD_Dalyr2['Listdt'] = pd.to_datetime(TRD_Dalyr2['Listdt'])

TRD_Dalyr2["Listdt_1y"] = TRD_Dalyr2["Listdt"] + pd.DateOffset(years=1)

TRD_Dalyr3 = TRD_Dalyr2.loc[
    TRD_Dalyr2["Trddt"] >= TRD_Dalyr2["Listdt_1y"]
].copy()

TRD_Dalyr3 = TRD_Dalyr3.drop(columns=["Listdt_1y"])

print(f"\n未匹配上市日期观测数：{TRD_Dalyr2['Listdt'].isna().sum()}")
print(f"未匹配上市日期股票数：{TRD_Dalyr2.loc[TRD_Dalyr2['Listdt'].isna(), 'Stkcd'].nunique()}")
print(f"删除上市不满一年观测值前的样本数：{len(TRD_Dalyr2)}")
print(f"删除上市不满一年观测值后的样本数：{len(TRD_Dalyr3)}")
print(f"删除上市不满一年观测值共：{len(TRD_Dalyr2) - len(TRD_Dalyr3)} 条")

# =========================================================
# 删除 ST 等异常交易状态的观测值
# =========================================================

TRD_Dalyr4 = TRD_Dalyr3.loc[
    TRD_Dalyr3["Trdsta"].isin([1, 4, 7, 10, 13])
].copy()

print(f"\n删除 ST 等异常交易状态观测值前的样本数：{len(TRD_Dalyr3)}")
print(f"删除 ST 等异常交易状态观测值后的样本数：{len(TRD_Dalyr4)}")
print(f"删除 ST 等异常交易状态观测值共：{len(TRD_Dalyr3) - len(TRD_Dalyr4)} 条")
# =========================================================
# 最后 merge 回完整面板前，先删除清洗后完全没有有效观测的股票
# 例如：从头到尾都是金融行业的股票
# =========================================================
valid_stocks = TRD_Dalyr4["Stkcd"].drop_duplicates()

full_panel_valid = full_panel.loc[
    full_panel["Stkcd"].isin(valid_stocks)
].copy()

TRD_Dalyr5 = pd.merge(
    full_panel_valid,
    TRD_Dalyr4.drop(columns=['Trddt']),
    how='left',
    on=['Stkcd','td']
)

TRD_Dalyr5 = TRD_Dalyr5.drop(columns=['Trdsta','Filling','has_volume','Listdt'])
nan_ret_count = (TRD_Dalyr5['ret'].isna()).sum()
TRD_Dalyr5.to_csv(out_file, index=False, encoding="utf-8")
print()