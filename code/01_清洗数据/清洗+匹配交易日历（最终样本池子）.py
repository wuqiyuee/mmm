# 这段代码以 CSMAR 个股日度收益率数据为基础，首先筛选 A 股市场样本，并对股票代码和交易日期进行标准化处理。
# ① 根据不同历史时期适用的行业分类标准，为每条个股日度观测匹配其当期行业信息，并据此识别和剔除金融行业股票。
# ② 合并公司上市日期信息，删除上市未满一年的股票日度观测。
# ③ 删除 ST、PT 的日度观测值。
# ④ 删除停牌或暂停上市的日度观测值。


# “剔除过去12个月交易记录少于120天；剔除最近1个月交易记录少于15天”？（暂时没剔除）
# “对于已分配DGTW分组标签的股票，本文在固定市场交易日窗口内计算特征调整收益率。
#  若股票在事件窗口内存在停牌，本文进一步要求窗口内有效交易日比例不低于70%。
# 对于仅在窗口起点和终点存在交易、窗口中间长期停牌的观测，虽然其特征调整收益率可被机械计算，
# 但其价格发现过程高度不连续，异常收益主要集中于复牌日，因而在主样本中予以剔除。”

# 在完成基础样本清理后，代码读取市场交易日历，筛选正常开市交易日，并按照时间顺序生成统一的市场交易日序号 td。
# 最后，代码将该交易日序号匹配到清理后的个股日度样本中，保留成功匹配交易日历的观测，
# 并输出包含股票代码 Stkcd、交易日期 Trddt 和市场交易日序号 td 的 CSV 文件。

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
TRD_Cale_path = ROOT / "input/CSMAR/日历文件2000-01-01至2025-12-31/TRD_Cale.csv"

out_file = ROOT / "中间结果/01_清洗后样本/TRD_Dalyr_td.csv"
out_file.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# 读取日度数据、保留 A 股市场数据、标准化股票代码和日期
# =========================================================

TRD_Dalyr = pd.read_csv(
    TRD_Dalyr_path,
    encoding = 'utf-8',
    usecols = ["Stkcd", "Trddt", "Markettype", "Filling", "Trdsta"],
    parse_dates=['Trddt']
)

TRD_Dalyr['Stkcd'] = TRD_Dalyr['Stkcd'].apply(normalize_stock_code)
TRD_Dalyr["year"] = TRD_Dalyr["Trddt"].dt.year

# 筛选 A 股市场的股票，包括主板、科创板、中小板等。
TRD_Dalyr = TRD_Dalyr.loc[
    (TRD_Dalyr["Markettype"].isin([1, 4, 16, 32])) &
    (TRD_Dalyr["Filling"].isin([0, 1]))
    # & (TRD_Dalyr["year"].between(2003, 2025))
].copy()

print(f"筛选 A 股市场后样本数：{len(TRD_Dalyr)}")

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

TRD_Dalyr["std_id"] = np.select(
    [
        TRD_Dalyr["Trddt"] < pd.Timestamp("2012-10-26"),
        (TRD_Dalyr["Trddt"] >= pd.Timestamp("2012-10-26")) & (TRD_Dalyr["Trddt"] < pd.Timestamp("2023-05-01")),
        TRD_Dalyr["Trddt"] >= pd.Timestamp("2023-05-01")
    ],
    ["P0201", "P0207", "P0221"],
    default="P0201"
)

left = (
    TRD_Dalyr
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

TRD_Dalyr1 = TRD_Dalyr_fin.loc[
    TRD_Dalyr_fin["is_fin"].fillna(0) != 1
].copy()

print(f"按当期行业口径识别出曾在某些时点属于金融行业的股票共 {fin_stocks} 只")
print(f"删除的金融行业 股票-日期 观测共 {fin_obs} 条")
print(f"已从日度收益文件中按当期行业剔除金融股，样本从 {original_len} 下降为 {len(TRD_Dalyr1)}")

TRD_Dalyr1 = TRD_Dalyr1.drop(columns=["std_id", "ImplementDate", "is_fin"]).copy()

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

TRD_Dalyr1 = pd.merge(
    TRD_Dalyr1,
    TRD_Co,
    how="left",
    on="Stkcd"
)

TRD_Dalyr1['Listdt'] = pd.to_datetime(TRD_Dalyr1['Listdt'])

TRD_Dalyr1["Listdt_1y"] = TRD_Dalyr1["Listdt"] + pd.DateOffset(years=1)

TRD_Dalyr2 = TRD_Dalyr1.loc[
    TRD_Dalyr1["Trddt"] >= TRD_Dalyr1["Listdt_1y"]
].copy()

TRD_Dalyr2 = TRD_Dalyr2.drop(columns=["Listdt_1y"])

print(f"\n未匹配上市日期观测数：{TRD_Dalyr1['Listdt'].isna().sum()}")
print(f"未匹配上市日期股票数：{TRD_Dalyr1.loc[TRD_Dalyr1['Listdt'].isna(), 'Stkcd'].nunique()}")
print(f"删除上市不满一年观测值前的样本数：{len(TRD_Dalyr1)}")
print(f"删除上市不满一年观测值后的样本数：{len(TRD_Dalyr2)}")
print(f"删除上市不满一年观测值共：{len(TRD_Dalyr1) - len(TRD_Dalyr2)} 条")

# =========================================================
# 删除 ST 等异常交易状态的观测值
# =========================================================

TRD_Dalyr3 = TRD_Dalyr2.loc[
    TRD_Dalyr2["Trdsta"].isin([1, 4, 7, 10, 13])
].copy()

print(f"\n删除 ST 等异常交易状态观测值前的样本数：{len(TRD_Dalyr2)}")
print(f"删除 ST 等异常交易状态观测值后的样本数：{len(TRD_Dalyr3)}")
print(f"删除 ST 等异常交易状态观测值共：{len(TRD_Dalyr2) - len(TRD_Dalyr3)} 条")

# =========================================================
# 删除停牌或者暂停上市的观测值
# =========================================================

TRD_Dalyr4 = TRD_Dalyr3.loc[
    TRD_Dalyr3["Filling"] == 0
].copy()

print(f"\n删除停牌或者暂停上市的观测值前的样本数：{len(TRD_Dalyr3)}")
print(f"删除停牌或者暂停上市的观测值后的样本数：{len(TRD_Dalyr4)}")
print(f"删除停牌或者暂停上市的观测值共：{len(TRD_Dalyr3) - len(TRD_Dalyr4)} 条")

# =========================================================
# 匹配交易日历
# =========================================================

TRD_Cale = pd.read_csv(
    TRD_Cale_path,
    encoding='utf-8',
    parse_dates=["Clddt"]
)

TRD_Cale["year"] = TRD_Cale["Clddt"].dt.year
TRD_Cale = TRD_Cale.loc[
    (TRD_Cale["Markettype"] == 1) &
    (TRD_Cale["State"] == "O")
    # & (TRD_Cale["year"].between(2003, 2025))
].copy()

TRD_Cale = TRD_Cale.rename(columns={"Clddt": "Trddt"})
TRD_Cale = TRD_Cale.sort_values(["Trddt"]).reset_index(drop=True)
TRD_Cale["td"] = np.arange(1, len(TRD_Cale) + 1)

# 这里是只筛选了A股市场开市的日期！！！
TRD_Cale.to_csv(
    '/Users/qiyuewu/Desktop/mmm复现数据包/中间结果/01_清洗后样本/TRD_Cale_td.csv',
    index=False,
    encoding='utf-8'
)

TRD_Dalyr5 = pd.merge(
    TRD_Dalyr4,
    TRD_Cale[["Trddt", "td"]],
    on=["Trddt"],
    how="left"
)

print(f"\n没有匹配到交易序号的观测数：{TRD_Dalyr5['td'].isna().sum()}")

TRD_Dalyr5 = TRD_Dalyr5.sort_values(["Stkcd", "Trddt"]).reset_index(drop=True)
TRD_Dalyr_td = TRD_Dalyr5[["Stkcd", "Trddt", "td"]].copy()
TRD_Dalyr_td = TRD_Dalyr_td.dropna(subset=["td"]).copy()
TRD_Dalyr_td["td"] = TRD_Dalyr_td["td"].astype(int)

# =========================================================
# 检查每只股票内部 td 是否连续
# =========================================================

tmp = TRD_Dalyr_td.sort_values(["Stkcd", "td"]).copy()
tmp["td_gap"] = tmp.groupby("Stkcd")["td"].diff()
print(tmp["td_gap"].value_counts(dropna=False).head(20))
gap = tmp.loc[
    tmp["td_gap"].notna() & (tmp["td_gap"] != 1)
].copy()
print("td 不连续的观测数：", len(gap))
print("td 不连续的股票数：", gap["Stkcd"].nunique())
print(gap[["Stkcd", "Trddt", "td", "td_gap"]].head(30))

# =========================================================
# 输出结果
# =========================================================

print("正在输出：TRD_Dalyr_td")

TRD_Dalyr_td.to_csv(
    out_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"已输出：TRD_Dalyr_td，共 {len(TRD_Dalyr_td)} 条观测值")