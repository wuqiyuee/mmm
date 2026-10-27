import gc
from pathlib import Path
import re

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# =============================================================================
# 路径设置
# =============================================================================

ROOT = Path(__file__).resolve().parents[4]

# 公司文件用于识别股票所属市场，避免把 B 股或北交所股票混入沪深 A 股样本。
TRD_CO_FILE = ROOT / "input/CSMAR/公司文件(1990-12-10至2026-08-21)/TRD_Co.csv"

# 已经合并好的 CSMAR 市场交易异常波动表（龙虎榜）明细数据。
ABNORMAL_FILE = ROOT / "中间结果/00_合并原始数据/CSMAR_市场交易异常波动表_龙虎榜_2002-08-02_2026-05-25_合并结果.csv"

# 输出文件夹：TopMover.csv 和变量说明统一放在同名文件夹下。
OUT_DIR = ROOT / "中间结果/02_变量构造结果/7TopMover"

# 输出：股票-日期层面的 Top Mover dummy。
OUT_FILE = OUT_DIR / "TopMover.csv"

# 输出：TopMover.csv 的变量说明。使用 txt，便于直接阅读。
DESC_FILE = OUT_DIR / "TopMover_变量说明.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 口径设置
# =============================================================================

# 沪深 A 股市场类型：
# 1  = 上证 A 股市场（不含科创板）
# 4  = 深证 A 股市场（不含创业板）
# 16 = 创业板
# 32 = 科创板
HS_A_MARKET_TYPES = {"1", "4", "16", "32"}

# 用于复现 Top Mover 的单日价格异动龙虎榜编码：
# P8401：日价格涨幅偏离值达 7%以上的证券
# P8402：日价格跌幅偏离值达 7%以上的证券
# P8422：有价格涨跌幅限制的日收盘价格涨幅达到 15%的证券
# P8423：有价格涨跌幅限制的日收盘价格跌幅达到 15%的证券
TOP_MOVER_INFO_CODES = {"P8401", "P8402", "P8422", "P8423"}

CHUNKSIZE = 300_000


# =============================================================================
# 工具函数
# =============================================================================

def normalize_stock_code(x):
    """把股票代码统一成 6 位字符串。"""
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()
    if s == "":
        return pd.NA

    m = re.search(r"\d{1,6}", s)
    if m is None:
        return pd.NA

    return m.group(0).zfill(6)


def read_hs_a_stock_codes():
    """从公司文件中读取沪深 A 股代码集合。"""
    co = pd.read_csv(
        TRD_CO_FILE,
        encoding="utf-8-sig",
        dtype={"Stkcd": "string", "Markettype": "string"},
        usecols=["Stkcd", "Markettype"],
        low_memory=False,
    )

    co["Stkcd"] = co["Stkcd"].apply(normalize_stock_code)
    co["Markettype"] = co["Markettype"].astype("string").str.strip()

    hs_a = co.loc[
        co["Markettype"].isin(HS_A_MARKET_TYPES) & co["Stkcd"].notna(),
        "Stkcd",
    ].drop_duplicates()

    return set(hs_a.tolist())


def build_top_mover():
    """按沪深 A 股 + 单日价格异动口径，构造股票-日期层面的 topMover dummy。"""
    hs_a_codes = read_hs_a_stock_codes()
    print(f"沪深 A 股代码数：{len(hs_a_codes):,}", flush=True)

    usecols = [
        "Symbol",
        "TradingDate",
        "InformationCode",
        "SecurityTypeID",
        "Duration",
    ]

    pieces = []
    total_rows = 0
    kept_rows_before_dedup = 0

    for chunk in pd.read_csv(
        ABNORMAL_FILE,
        encoding="utf-8-sig",
        dtype={
            "Symbol": "string",
            "TradingDate": "string",
            "InformationCode": "string",
            "SecurityTypeID": "string",
            "Duration": "string",
        },
        usecols=usecols,
        chunksize=CHUNKSIZE,
        low_memory=False,
    ):
        total_rows += len(chunk)

        # 标准化关键字段，保证筛选和去重稳定。
        chunk["Symbol"] = chunk["Symbol"].apply(normalize_stock_code)
        chunk["TradingDate"] = chunk["TradingDate"].astype("string").str.strip()
        chunk["InformationCode"] = chunk["InformationCode"].astype("string").str.strip()
        chunk["SecurityTypeID"] = chunk["SecurityTypeID"].astype("string").str.strip()
        chunk["Duration"] = chunk["Duration"].astype("string").str.strip()

        mask = (
            chunk["Symbol"].isin(hs_a_codes)
            & chunk["SecurityTypeID"].eq("S0101")
            & chunk["Duration"].eq("1")
            & chunk["InformationCode"].isin(TOP_MOVER_INFO_CODES)
        )

        selected = chunk.loc[mask, ["Symbol", "TradingDate"]].copy()
        kept_rows_before_dedup += len(selected)

        if not selected.empty:
            pieces.append(selected)

        del chunk, selected
        gc.collect()

    if pieces:
        top_mover = pd.concat(pieces, ignore_index=True)
    else:
        top_mover = pd.DataFrame(columns=["Symbol", "TradingDate"])

    # 关键步骤：龙虎榜原表是营业部明细层面，同一股票-日期会有多行。
    # 这里压缩到股票-日期唯一，才能对应原代码 TopMovers_prep.dta 的结构。
    top_mover = top_mover.drop_duplicates(subset=["Symbol", "TradingDate"])
    top_mover = top_mover.rename(columns={"Symbol": "Stkcd", "TradingDate": "Trddt"})
    top_mover["topMover"] = 1

    top_mover["Trddt"] = pd.to_datetime(
        top_mover["Trddt"], errors="coerce"
    )
    top_mover = top_mover.dropna(subset=["Stkcd", "Trddt"])
    top_mover = top_mover.sort_values(["Trddt", "Stkcd"]).reset_index(drop=True)
    top_mover["Trddt"] = top_mover["Trddt"].dt.strftime("%Y-%m-%d")

    # 最终只保留直接可 merge 的窄表结构。
    top_mover = top_mover[["Stkcd", "Trddt", "topMover"]]

    if top_mover.duplicated(subset=["Stkcd", "Trddt"]).any():
        raise ValueError("输出中仍存在重复的 Stkcd-Trddt，请检查去重逻辑。")

    top_mover.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    desc_text = """TopMover.csv 变量说明

文件结构：股票-日期层面的 Top Mover dummy 表。
唯一键：Stkcd + Trddt。

构造口径：
1. 从 TRD_Co.csv 中保留沪深 A 股：Markettype in (1, 4, 16, 32)。
   1=上证A股市场（不含科创板）；4=深证A股市场（不含创业板）；16=创业板；32=科创板。
2. 从 CSMAR 市场交易异常波动表（龙虎榜）中保留：
   SecurityTypeID = S0101；
   Duration = 1；
   InformationCode in (P8401, P8402, P8422, P8423)。
3. 按 Stkcd-Trddt 去重，把营业部明细层面的多行压缩成股票-日期唯一。
4. 生成 topMover = 1。未出现在 TopMover.csv 中的股票-日期，在后续合并后可设为 0。

变量说明：
Stkcd
  含义：证券代码，统一为 6 位字符串。
  来源/生成方式：来自 BT_AbnormalFlucture.Symbol，并按 6 位股票代码标准化。

Trddt
  含义：交易日期，即交易所市场交易日。
  来源/生成方式：来自 BT_AbnormalFlucture.TradingDate，重命名为 Trddt，格式为 YYYY-MM-DD；仅在对应股票触发龙虎榜单日价格异动规则时出现。

topMover
  含义：Top Mover 虚拟变量，取值恒为 1。
  来源/生成方式：沪深 A 股且满足单日价格异动龙虎榜编码的股票-日期，去重后生成。

涉及的 InformationCode：
P8401：日价格涨幅偏离值达 7%以上的证券。
P8402：日价格跌幅偏离值达 7%以上的证券。
P8422：有价格涨跌幅限制的日收盘价格涨幅达到 15%的证券。
P8423：有价格涨跌幅限制的日收盘价格跌幅达到 15%的证券。
"""
    DESC_FILE.write_text(desc_text, encoding="utf-8-sig")

    daily_counts = top_mover.groupby("Trddt").size()
    print("TopMover 构造完成", flush=True)
    print(f"读取龙虎榜明细行数：{total_rows:,}", flush=True)
    print(f"筛选后去重前明细行数：{kept_rows_before_dedup:,}", flush=True)
    print(f"输出股票-日期行数：{len(top_mover):,}", flush=True)

    if not top_mover.empty:
        print(f"日期范围：{top_mover['Trddt'].min()} 至 {top_mover['Trddt'].max()}", flush=True)
        print(f"有 TopMover 的交易日数：{daily_counts.shape[0]:,}", flush=True)
        print(f"平均每天 TopMover 数：{daily_counts.mean():.2f}", flush=True)
        print(f"每天 TopMover 数中位数：{daily_counts.median():.0f}", flush=True)
        print(f"每天 TopMover 数最小/最大：{daily_counts.min()} / {daily_counts.max()}", flush=True)

    print(f"输出文件：{OUT_FILE}", flush=True)
    print(f"变量说明：{DESC_FILE}", flush=True)


if __name__ == "__main__":
    build_top_mover()
