import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

ownership_path = (
    ROOT
    / "input/CSMAR/中国上市公司股权性质文件(2003-12-31至2025-12-31)"
    / "EN_EquityNatureAll.csv"
)


def build_ownership_table(path):
    ownership = pd.read_csv(
        path,
        usecols=["Symbol", "EndDate", "EquityNatureID"],
        dtype="string",
        encoding="utf-8-sig",
    )

    # 1. 统一证券代码和日期
    ownership["Symbol"] = (
        ownership["Symbol"]
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .str.zfill(6)
    )

    ownership["EndDate"] = pd.to_datetime(
        ownership["EndDate"],
        errors="coerce",
    )

    # 2. 修正原始文件中已识别的一条字段错位记录
    bad_row = (
        ownership["Symbol"].eq("600986")
        & ownership["EndDate"].eq(pd.Timestamp("2016-12-31"))
        & ownership["EquityNatureID"].eq("民营")
    )
    ownership.loc[bad_row, "EquityNatureID"] = "2"

    # 3. 清理股权性质编码
    nature_code = (
        ownership["EquityNatureID"]
        .str.strip()
        .str.replace("，", ",", regex=False)
        .str.replace(r"\s+", "", regex=True)
        .replace("", pd.NA)
    )

    valid_code = nature_code.str.fullmatch(
        r"[1-4](,[1-4])*",
        na=False,
    )

    invalid_code = nature_code.notna() & ~valid_code
    if invalid_code.any():
        raise ValueError("发现无法识别的股权性质编码。")

    # 4. 国资参与标识：编码中只要包含1，就认定存在国资参与
    ownership["state_involved_lag1"] = pd.NA
    ownership.loc[valid_code, "state_involved_lag1"] = (
        nature_code.loc[valid_code]
        .str.contains(r"(?:^|,)1(?:,|$)")
        .astype("int8")
    )
    ownership["state_involved_lag1"] = ownership[
        "state_involved_lag1"
    ].astype("Int8")

    # 严格口径：只比较纯国企1和纯民营2
    ownership["SOE_strict_lag1"] = (
        nature_code.map({"1": 1, "2": 0}).astype("Int8")
    )

    # 混合股权性质标记
    ownership["mixed_ownership_lag1"] = (
        nature_code.str.contains(",", na=False).astype("Int8")
    )

    # 5. 事件年度t匹配t-1年末股权性质
    ownership["event_year"] = (
        ownership["EndDate"].dt.year + 1
    ).astype("Int64")

    ownership = ownership.rename(
        columns={
            "Symbol": "StkcdRecall",
            "EquityNatureID": "EquityNatureID_lag1",
            "EndDate": "ownership_end_date",
        }
    )

    keep_cols = [
        "StkcdRecall",
        "event_year",
        "ownership_end_date",
        "EquityNatureID_lag1",
        "state_involved_lag1",
        "SOE_strict_lag1",
        "mixed_ownership_lag1",
    ]
    ownership = ownership[keep_cols]

    # 公司×事件年度必须唯一
    if ownership.duplicated(
        ["StkcdRecall", "event_year"]
    ).any():
        raise ValueError("股权性质表的公司×年度不唯一。")

    return ownership
