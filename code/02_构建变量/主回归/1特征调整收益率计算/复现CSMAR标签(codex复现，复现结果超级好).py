"""从原始数据重建国泰安 DGTW（股票特征基准）标签，并与官方表比较。

直接运行本脚本即可，不依赖项目的其他清洗脚本或中间结果。
依赖 pandas（表格处理库）、numpy（数值计算库）。

主要口径：
1. 六月末流通市值；上年末合并年报账面权益 / 上年末流通市值。
   流通市值取A股市值+B股人民币市值+按A股价格计值的其他境外流通股，
   不直接把升级日表仅含A股的市值作为公司全部流通市值。
   六月剔除节假日填充；十二月保留节假日填充，优先使用12月31日记录。
   六月停牌但当月有成交，且月表末次价格与填充价相同时，A股部分取月表市值。
   价格不同时保留日表值，避免用除息前的月表市值覆盖除息后的技术填充。
   停牌缩股后仍沿用原价且尚未复牌时，保留末次真实交易市值，直到出现新价。
   职工股上市后持续停牌且价格不变时，A股部分同样延续末次成交市值。
   六月整月停牌且股改限售股上市后仍沿用旧价时，A股部分延续末次成交市值。
   年报披露后回填至12月31日的股权激励股本，停牌同价时也沿用末次成交市值。
2. 行业按上年末匹配；形成年度早于2023用2012版证监会分类，否则用协会分类。
3. 动量为上年7月至当年6月参考价月收益的复合值，至少6个有交易月份。
   从日表含红利/不含红利收益还原参考价收益；普通除权事件用原始价格和
   红利、配股方案精确化。月收益先保留6位，再逐月复合。
4. 全体三指标有效股票按规模、行业调整账面比、动量逐级降序等数量分组。
   余数依次分给前组，与已观察到的官方组人数安排一致；并列值按股票代码排序。
   这与说明书“主板断点”的文字存在差异，不声称是官方内部源代码。
5. 行业内采用样本标准差；附属财务科目缺失补0，主权益缺失不补0。
   市值范围和标准差口径经官方组合数值核对，实际匹配率以运行结果为准。
6. 早年月收益补齐时，上市首月沿原始月表的空收益口径处理。新代码在形成
   窗口已有至少6个交易月时，不再衔接旧码；不足门槛时才补旧码收益。
7. 逐日以真实价格、红利和配股方案精确化观察因子；不匹配的复杂对价保留
   观察值。剔除增发引入的额外因子，纠正引用倒数第二实收价的旧日收益。
   特定股东重整方案采用交易所参考价；原月表空收益继续保留。
   无除权事件的停牌技术日若改变了填充价，复牌收益须接回最后真实收盘价。
   先流通股东股改、后全体年度转增且同日除权时，按登记顺序和股本衔接恢复两段收益。
8. 本地财报会被后来更正覆盖；已核对历史公告版本的财报科目在计算权益前恢复。
   保留官方样本期内已经披露的更正，不把所有历史年度机械改成首次年报。
   所有恢复项列明原始公告来源，不使用官方标签反推或覆盖指标。

官方分配表只在自主构造标签完成后读取，不参与筛选样本、计算指标或分组。
脚本不套用研究样本的剔金融、剔特殊处理股票、上市满一年等额外条件。
运行时读取较大的日度文件，会逐文件显示进度；采用分块读取控制内存。
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================================
# 0. 路径与可调整口径
# ============================================================================

ROOT = Path(__file__).resolve().parents[4]  # 项目根目录。
raw_dir = ROOT / "input/CSMAR"
output_dir = ROOT / "中间结果/02_变量构造结果/DGTW/CSMAR标签复现"

start_formation_year = 2000
end_formation_year = 2025
exclude_bse = True
min_trading_months = 6
industry_std_ddof = 1  # 1为样本标准差；0为总体标准差。
daily_chunk_size = 300_000
reference_price_tolerance = 0.000005  # 日复利因子相对容差；兼容原始日收益的6位舍入。
misaligned_price_tolerance = 0.000001  # 仅纠正能由倒数第二个真实收盘价解释的旧日收益。

monthly_path = raw_dir / "月个股回报率文件2000-01至2025-12/TRD_Mnth.csv"
daily_dir = raw_dir / "个股日度收益率（升级测试）1995-01-01至2026"
balance_sheet_path = raw_dir / "资产负债表1990-12-31至2026-06-30/FS_Combas.csv"
industry_path = raw_dir / "上市公司行业分类表1990年12月10日—2026年4月20日/STK_INDUSTRYCLASS.csv"
company_path = raw_dir / "公司文件(1990-12-10至2026-08-21)/TRD_Co.csv"
share_structure_path = raw_dir / "专题研究系列-事件研究/股本变动事件表1990-12-07 至 2026-09-22/ER_Structure.csv"
dividend_path = raw_dir / "专题研究系列-事件研究/红利分配事件表1991-02-10 至 2026-09-11/ER_ProfitSharing.csv"
rights_issue_path = raw_dir / "专题研究系列-事件研究/配股事件表1990-03-01 至 2025-06-26/ER_RightsIssue.csv"
official_dir = raw_dir / "DGTW（2000至2025）"
official_assignment_path = official_dir / "DGTW股票分配表/STK_MKT_DGTWASSINGMENTS.csv"
official_maxima_path = official_dir / "DGTW分位数表2000至2025/STK_MKT_DGTWBREAKPOINTS.csv"


# ============================================================================
# 工具函数：仅保留代码标准化、唯一键检查、五分组与比较汇总
# ============================================================================

def normalize_stock_codes(column):
    result = column.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if not result.str.fullmatch(r"\d{1,6}", na=False).all():
        raise ValueError("股票代码存在空值或非数字，请检查原始表")
    return result.str.zfill(6)


def check_unique_key(table, keys, table_name):
    if table[keys].isna().any().any():
        raise ValueError(f"{table_name}的关键字段存在空值：{keys}")
    duplicate_rows = table.duplicated(keys, keep=False)
    if duplicate_rows.any():
        examples = table.loc[duplicate_rows, keys].head(10).to_string(index=False)
        raise ValueError(f"{table_name}的唯一键重复，不能静默保留最后一行：\n{examples}")


def assign_descending_quintiles(table, group_keys, sort_column):
    group_numbers = pd.Series(pd.NA, index=table.index, dtype="Int8")
    for _, subtable in table.groupby(group_keys, sort=False):
        sorted_rows = subtable.sort_values([sort_column, "Stkcd"], ascending=[False, True])
        for sequence_number, row_index in enumerate(np.array_split(sorted_rows.index.to_numpy(), 5), start=1):
            group_numbers.loc[row_index] = sequence_number
    if group_numbers.isna().any():
        raise ValueError(f"{sort_column}分组失败，存在未分配组号的股票")
    return group_numbers


def summarize_matches(table, year):
    in_both = table["样本来源"].eq("both")  # 两表共有。
    has_official = ~table["样本来源"].eq("left_only")  # 非仅重建。
    has_rebuilt = ~table["样本来源"].eq("right_only")  # 非仅官方。
    size_match = table["规模一致"].fillna(False)
    bm_match = table["账面比一致"].fillna(False)
    momentum_match = table["动量一致"].fillna(False)
    all_match = table["全部一致"].fillna(False)

    def safe_ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else np.nan

    return {
        "形成年度": year,
        "官方股票数": int(has_official.sum()),
        "重建股票数": int(has_rebuilt.sum()),
        "共有股票数": int(in_both.sum()),
        "仅官方股票数": int(table["样本来源"].eq("right_only").sum()),
        "仅重建股票数": int(table["样本来源"].eq("left_only").sum()),
        "官方样本覆盖率": safe_ratio(in_both.sum(), has_official.sum()),
        "重建样本被官方覆盖率": safe_ratio(in_both.sum(), has_rebuilt.sum()),
        "共有样本规模一致率": safe_ratio(size_match.sum(), in_both.sum()),
        "共有样本账面比一致率": safe_ratio(bm_match.sum(), in_both.sum()),
        "共有样本动量一致率": safe_ratio(momentum_match.sum(), in_both.sum()),
        "共有样本全部一致率": safe_ratio(all_match.sum(), in_both.sum()),
        "规模一致时账面比一致率": safe_ratio((size_match & bm_match).sum(), size_match.sum()),
        "前两维一致时动量一致率": safe_ratio(all_match.sum(), (size_match & bm_match).sum()),
        "全部一致股票数": int(all_match.sum()),
        "全部一致数占官方比例": safe_ratio(all_match.sum(), has_official.sum()),
    }


def main():
    # main（主流程）：所有结果只在直接运行脚本时生成。
    if not 1996 <= start_formation_year <= end_formation_year <= 2025:
        raise ValueError("当前原始文件支持的形成年度设置范围为1996至2025")
    if not 1 <= min_trading_months <= 12 or industry_std_ddof not in [0, 1]:
        raise ValueError("月份门槛应为1至12，标准差自由度应为0或1")
    if not np.isfinite(reference_price_tolerance) or reference_price_tolerance < 0:
        raise ValueError("理论参考价容差应为非负有限数")
    if not np.isfinite(misaligned_price_tolerance) or misaligned_price_tolerance < 0:
        raise ValueError("错位前价容差应为非负有限数")
    for path in [monthly_path, daily_dir, balance_sheet_path, industry_path, company_path, share_structure_path,
                 dividend_path, rights_issue_path, official_assignment_path, official_maxima_path]:
        if not path.exists():
            raise FileNotFoundError(f"缺少输入：{path}")

    market_types = [1, 4, 16, 32] if exclude_bse else [1, 4, 16, 32, 64]
    market_value_markets = market_types + [2, 8]  # B股只用于还原市值，不进入A股排序样本。
    is_not_bse = int(exclude_bse)
    start_month = pd.Timestamp(start_formation_year - 1, 7, 1)
    end_month = pd.Timestamp(end_formation_year, 6, 1)
    start_date = f"{start_formation_year - 1}-01-01"  # 包含上年12月市值所需数据。
    end_date = f"{end_formation_year}-12-31"
    stage_samples = {}  # 仅保存各构造阶段的股票年键，供最后解释样本差异。

    companies = pd.read_csv(
        company_path, encoding="utf-8-sig", dtype="string",
        usecols=["Stkcd", "FormerCode", "Listdt"],
    ).drop_duplicates()
    companies["Stkcd"] = normalize_stock_codes(companies["Stkcd"])
    check_unique_key(companies, ["Stkcd"], "公司基本资料")
    # FormerCode（曾用代码）：仅接受明确、单一的六位旧代码。
    companies["FormerCode"] = companies["FormerCode"].str.strip().str.extract(r"^(\d{6})$", expand=False)
    companies["Listdt"] = pd.to_datetime(companies["Listdt"], errors="coerce")  # 新代码上市日期。
    code_aliases = companies.dropna(subset=["FormerCode"]).rename(columns={"Stkcd": "new_code"})
    code_change_events = code_aliases.loc[code_aliases["Listdt"].between(start_month, end_month + pd.offsets.MonthEnd(0))].copy()
    code_change_events["change_month"] = code_change_events["Listdt"].dt.strftime("%Y-%m")  # 换码月份。
    code_change_day_keys = pd.concat([
        code_change_events[["new_code", "change_month"]].rename(columns={"new_code": "Stkcd"}),
        code_change_events[["FormerCode", "change_month"]].rename(columns={"FormerCode": "Stkcd"}),
    ], ignore_index=True).drop_duplicates()
    code_change_index = pd.MultiIndex.from_frame(code_change_day_keys.rename(columns={"change_month": "Trdmnt"}))

    # ========================================================================
    # 1. 读取原始月收益；不使用项目已清洗研究样本
    # ========================================================================
    print("1/7 读取原始月收益……", flush=True)
    # Mretwd为含红利月收益，Ndaytrd为月实际交易天数，Markettype为市场类型。
    monthly_data = pd.read_csv(
        monthly_path, encoding="utf-8-sig",
        usecols=["Stkcd", "Trdmnt", "Mretwd", "Ndaytrd", "Markettype", "Msmvosd", "Mclsprc"],
        dtype={"Stkcd": "string"}, parse_dates=["Trdmnt"],
    )
    monthly_data = monthly_data.loc[monthly_data["Markettype"].isin(market_types)].copy()
    monthly_data["Stkcd"] = normalize_stock_codes(monthly_data["Stkcd"])
    monthly_data = monthly_data.drop_duplicates()
    check_unique_key(monthly_data, ["Stkcd", "Trdmnt"], "原始月收益表")
    if monthly_data.empty:
        raise ValueError("原始月收益表没有所选市场股票")
    first_original_month = monthly_data["Trdmnt"].min()
    monthly_data["月收益来自日表"] = 0  # 后续包含早年补数、日月冲突纠正及必要的换码衔接。
    monthly_data["缺失日收益数"] = 0
    monthly_data["跨代码补充月份"] = 0
    monthly_data = monthly_data.loc[monthly_data["Trdmnt"].between(start_month, end_month)].copy()
    june_monthly_values = monthly_data.loc[
        monthly_data["Trdmnt"].dt.month.eq(6) & monthly_data["Ndaytrd"].gt(0) & monthly_data["Msmvosd"].gt(0),
        ["Stkcd", "Trdmnt", "Msmvosd", "Mclsprc"],
    ].rename(columns={"Msmvosd": "monthly_me_a", "Mclsprc": "monthly_price"})
    june_monthly_values["monthly_me_a"] *= 1000  # 月表A股流通市值，转为元。
    monthly_data = monthly_data.drop(columns=["Msmvosd", "Mclsprc"])

    # 红利字段均为每10股；配股比例已经是每股比例，不再除以10。
    # 老记录存在股东对象标记不一致；年度Q0103/Q0105另列候选，由日收益验证。
    dividends = pd.read_csv(
        dividend_path, encoding="utf-8-sig", dtype={"Symbol": "string"},
        usecols=["Symbol", "ExDividendDate", "RecordDate", "Object", "Term", "Currency",
                 "BonusRatio", "ConversionRatio", "DividentBT", "DistributionBaseShares", "AddShares"],
    ).rename(columns={"Symbol": "Stkcd"})
    dividends["Stkcd"] = normalize_stock_codes(dividends["Stkcd"])
    dividends["Trddt"] = pd.to_datetime(dividends["ExDividendDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    # 先统计全部对象和币种的方案，防止筛选后漏掉同日另一项分配。
    same_day_plan_counts = dividends.drop_duplicates([
        "Stkcd", "Trddt", "RecordDate", "Object", "Term", "Currency", "BonusRatio", "ConversionRatio",
        "DividentBT", "DistributionBaseShares", "AddShares",
    ]).groupby(["Stkcd", "Trddt"]).size()
    dividends = dividends.loc[
        dividends["Trddt"].between(start_date, end_date)
        & (dividends["Currency"].isna() | dividends["Currency"].isin(["人民币元", "人民币", "CNY"]))
    ].copy()
    dividend_columns = ["BonusRatio", "ConversionRatio", "DividentBT"]
    dividends[dividend_columns] = dividends[dividend_columns].fillna(0) / 10
    ordinary_recipients = dividends["Object"].isin(["Q0101", "Q0102", "Q0104", "Q0106"])
    historical_annual_recipients = dividends["Term"].eq("P2702") & dividends["Object"].isin(["Q0103", "Q0105"])
    plan_tables = []
    for suffix, included_rows in [("_o", ordinary_recipients), ("", ordinary_recipients | historical_annual_recipients)]:
        subtable = dividends.loc[included_rows].drop_duplicates(["Stkcd", "Trddt", "RecordDate"] + dividend_columns)
        subtable = subtable.groupby(["Stkcd", "Trddt", "RecordDate"], dropna=False, as_index=False)[dividend_columns].max()
        subtable["mult"] = 1 + subtable["BonusRatio"] + subtable["ConversionRatio"]
        subtable = subtable.groupby(["Stkcd", "Trddt"], as_index=False).agg(
            cash=("DividentBT", "sum"), mult=("mult", "prod"),
        ).rename(columns={"cash": "cash" + suffix, "mult": "mult" + suffix})
        plan_tables.append(subtable)
    rights_issues = pd.read_csv(
        rights_issue_path, encoding="utf-8-sig", dtype={"Symbol": "string"},
        usecols=["Symbol", "ExRightDate", "ProjectActualize", "SecurityType", "PlacingRatio",
                 "ShareHoldersAllotmentRatio", "Price"],
    ).rename(columns={"Symbol": "Stkcd"})
    rights_issues["Stkcd"] = normalize_stock_codes(rights_issues["Stkcd"])
    rights_issues["Trddt"] = pd.to_datetime(rights_issues["ExRightDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    rights_issues = rights_issues.loc[
        rights_issues["ProjectActualize"].eq("P6301") & rights_issues["SecurityType"].eq("A")
        & rights_issues["Trddt"].between(start_date, end_date)
    ].drop_duplicates(["Stkcd", "Trddt", "PlacingRatio", "ShareHoldersAllotmentRatio", "Price"]).copy()
    # 两个原始比例字段偶有差异，分别计算理论候选，再按日观察因子核验。
    rights_issues["alt_ratio"] = rights_issues["ShareHoldersAllotmentRatio"].fillna(rights_issues["PlacingRatio"])
    rights_issues["rights_cost"] = rights_issues["PlacingRatio"] * rights_issues["Price"]
    rights_issues["alt_cost"] = rights_issues["alt_ratio"] * rights_issues["Price"]
    rights_issues = rights_issues.groupby(["Stkcd", "Trddt"], as_index=False).agg(
        rights_ratio=("PlacingRatio", "sum"), rights_cost=("rights_cost", "sum"),
        alt_ratio=("alt_ratio", "sum"), alt_cost=("alt_cost", "sum"),
    )
    ordinary_plans = plan_tables[0].merge(
        rights_issues[["Stkcd", "Trddt", "rights_ratio", "rights_cost"]],
        on=["Stkcd", "Trddt"], how="outer", validate="one_to_one",
    )
    ordinary_plans[["cash_o", "rights_ratio", "rights_cost"]] = ordinary_plans[["cash_o", "rights_ratio", "rights_cost"]].fillna(0)
    ordinary_plans["mult_o"] = ordinary_plans["mult_o"].fillna(1)
    ordinary_plans = ordinary_plans.rename(columns={"rights_ratio": "ratio_o", "rights_cost": "cost_o"})
    extended_plans = plan_tables[1].merge(rights_issues, on=["Stkcd", "Trddt"], how="outer", validate="one_to_one")
    extended_plans[["cash", "rights_ratio", "rights_cost", "alt_ratio", "alt_cost"]] = extended_plans[
        ["cash", "rights_ratio", "rights_cost", "alt_ratio", "alt_cost"]
    ].fillna(0)
    extended_plans["mult"] = extended_plans["mult"].fillna(1)
    ex_right_events = extended_plans.merge(ordinary_plans, on=["Stkcd", "Trddt"], how="outer", validate="one_to_one")
    ex_right_events["has_plan"] = ex_right_events["mult"].notna() | ex_right_events["mult_o"].notna()
    ex_right_events["legacy_cash"] = (
        ex_right_events["cash"].gt(0) & ex_right_events["cash_o"].isna()
        & ex_right_events["mult"].eq(1) & ex_right_events["rights_ratio"].eq(0)
    )
    special_plans = dividends.loc[
        dividends["Object"].eq("Q0105") & dividends["Term"].eq("P2799")
        & dividends["DividentBT"].eq(0) & (dividends["BonusRatio"] + dividends["ConversionRatio"]).gt(0),
        ["Stkcd", "Trddt"],
    ].drop_duplicates()
    special_plans["special"] = True
    ex_right_events = ex_right_events.merge(special_plans, on=["Stkcd", "Trddt"], how="outer", validate="one_to_one")
    share_events = pd.read_csv(
        share_structure_path, encoding="utf-8-sig", dtype={"Symbol": "string"},
        usecols=["Symbol", "ChangeDate", "EventID", "ChangeReasonID", "Total"],
    ).rename(columns={"Symbol": "Stkcd", "ChangeDate": "Trddt"})
    share_events["Stkcd"] = normalize_stock_codes(share_events["Stkcd"])
    share_events = share_events.sort_values(["Stkcd", "Trddt", "EventID"])

    # 两段同日除权但登记日先后不同，不能因观察收益漏掉首段就舍弃完整方案。
    # 只接受先流通股东股改、后全体年度纯转增，并用原始总股本核对两段衔接。
    plan_columns = ["Stkcd", "Trddt", "RecordDate", "Object", "Term"] + dividend_columns + [
        "DistributionBaseShares", "AddShares",
    ]
    original_plans = dividends[plan_columns].drop_duplicates()
    pure_capitalization = (
        original_plans["ConversionRatio"].gt(0) & original_plans["BonusRatio"].eq(0)
        & original_plans["DividentBT"].eq(0)
    )
    first_stage = original_plans.loc[
        pure_capitalization & original_plans["Object"].eq("Q0104") & original_plans["Term"].eq("P2799")
    ]
    second_stage = original_plans.loc[
        pure_capitalization & original_plans["Object"].eq("Q0101") & original_plans["Term"].eq("P2702")
    ]
    two_stage_candidates = first_stage.merge(second_stage, on=["Stkcd", "Trddt"], suffixes=("_先", "_后"))
    two_stage_candidates = two_stage_candidates.loc[
        (pd.to_datetime(two_stage_candidates["RecordDate_先"], errors="coerce")
         < pd.to_datetime(two_stage_candidates["RecordDate_后"], errors="coerce"))
        & (pd.to_datetime(two_stage_candidates["RecordDate_后"], errors="coerce")
           < pd.to_datetime(two_stage_candidates["Trddt"], errors="coerce"))
    ]
    two_stage_adjustments = []
    for _, plan in two_stage_candidates.iterrows():
        stock, ex_right_date = plan["Stkcd"], plan["Trddt"]
        same_day_rights_issues = rights_issues.loc[rights_issues["Stkcd"].eq(stock) & rights_issues["Trddt"].eq(ex_right_date)]
        same_day_share_records = share_events.loc[share_events["Stkcd"].eq(stock) & share_events["Trddt"].eq(ex_right_date)]
        if same_day_plan_counts.get((stock, ex_right_date), 0) != 2 or not same_day_rights_issues.empty:
            continue
        if not same_day_share_records["ChangeReasonID"].str.contains(r"(?:^|,)P0712(?:,|$)", na=False).any():
            continue
        previous_share_record = share_events.loc[share_events["Stkcd"].eq(stock) & share_events["Trddt"].lt(ex_right_date)].tail(1)
        if previous_share_record.empty:
            continue
        total_shares_before, total_shares_after = previous_share_record["Total"].iloc[0], same_day_share_records["Total"].iloc[-1]
        first_base_shares, first_added_shares = plan["DistributionBaseShares_先"], plan["AddShares_先"]
        second_base_shares, second_added_shares = plan["DistributionBaseShares_后"], plan["AddShares_后"]
        if not all(np.isfinite(x) and x > 0 for x in [total_shares_before, total_shares_after, first_base_shares, first_added_shares, second_base_shares, second_added_shares]):
            continue
        # 新增股份字段含小数，总股本为整数；四项衔接均允许最多2股差额。
        share_differences = [
            first_added_shares - first_base_shares * plan["ConversionRatio_先"],
            second_added_shares - second_base_shares * plan["ConversionRatio_后"],
            total_shares_before + first_added_shares - second_base_shares,
            second_base_shares + second_added_shares - total_shares_after,
        ]
        if not all(np.isfinite(x) and abs(x) <= 2 for x in share_differences):
            continue
        two_stage_adjustments.append({
            "Stkcd": stock, "Trddt": ex_right_date,
            "first_mult": 1 + plan["ConversionRatio_先"],
            "annual_mult": 1 + plan["ConversionRatio_后"],
        })
    two_stage_adjustments = pd.DataFrame(
        two_stage_adjustments, columns=["Stkcd", "Trddt", "first_mult", "annual_mult"],
    ).astype({"Stkcd": "string", "Trddt": "string", "first_mult": "float64", "annual_mult": "float64"})
    # 字典列表构造的键默认是对象类型，须与原始表统一，避免后续按代码近邻合并报错。
    check_unique_key(two_stage_adjustments, ["Stkcd", "Trddt"], "两段顺序转增方案")
    ex_right_events = ex_right_events.merge(two_stage_adjustments, on=["Stkcd", "Trddt"], how="left", validate="one_to_one")

    issue_events = share_events.loc[
        share_events["ChangeReasonID"].str.contains("P0703|P0704", na=False), ["Stkcd", "Trddt"],
    ].drop_duplicates()
    issue_events["issue"] = True
    ex_right_events = ex_right_events.merge(issue_events, on=["Stkcd", "Trddt"], how="outer", validate="one_to_one")
    reform_events = share_events.loc[
        share_events["ChangeReasonID"].str.contains("P0712|P0757", na=False), ["Stkcd", "Trddt"],
    ].drop_duplicates().rename(columns={"Trddt": "reform_date"})
    reform_events["reform_date"] = pd.to_datetime(reform_events["reform_date"])
    return_events = pd.concat([
        dividends[["Stkcd", "Trddt"]], rights_issues[["Stkcd", "Trddt"]],
        reform_events.assign(Trddt=reform_events["reform_date"].dt.strftime("%Y-%m-%d"))[["Stkcd", "Trddt"]],
    ], ignore_index=True).dropna().drop_duplicates()
    return_events["return_event_date"] = pd.to_datetime(return_events["Trddt"])

    # ========================================================================
    # 2. 分块读日表：取六月/十二月月末快照，补月表起点之前的收益
    # ========================================================================
    print("2/7 提取月末市值，并按除权参考价重建月收益……", flush=True)
    daily_files = []
    for path in sorted(daily_dir.glob("*/TRDNEW_Dalyr*.csv")):
        year = [int(x) for x in re.findall(r"\d{4}", path.parent.name)]
        if year and max(year) >= start_formation_year - 1 and min(year) <= end_formation_year:
            daily_files.append(path)
    if not daily_files:
        raise FileNotFoundError("没有找到所需年度的日度原始文件")

    snapshot_parts, actual_trade_snapshot_parts, supplement_return_parts, code_change_return_parts, daily_monthly_return_parts = [], [], [], [], []
    last_actual_prices, second_last_actual_prices, last_actual_dates = {}, {}, {}  # 跨读取块与年度衔接。
    last_calendar_prices, last_calendar_fillings, last_calendar_gross_returns = {}, {}, {}  # 包括停牌记录，用于识别技术填充。
    technical_fill_corrections = 0
    daily_columns = ["Stkcd", "Trddt", "Markettype", "Filling", "Dsmvosd", "Dretwd", "Dretnd",
              "Dnshrtrd", "Clsprc", "PreClosePrice"]
    # Dsmvosd为流通市值，单位千元；Dretwd为含红利日收益；Dnshrtrd为成交股数。
    # Filling（填充标识）：0正常记录，1非节假日填充，2节假日填充。
    for sequence_number, path in enumerate(daily_files, start=1):
        print(f"  日表 {sequence_number}/{len(daily_files)}：{path.parent.name}/{path.name}", flush=True)
        for chunk in pd.read_csv(
            path, usecols=daily_columns, encoding="utf-8-sig", chunksize=daily_chunk_size,
            dtype={"Stkcd": "string", "Trddt": "string"},
        ):
            chunk = chunk.loc[
                chunk["Markettype"].isin(market_value_markets)
                & chunk["Trddt"].between(start_date, end_date)
            ].copy()
            if chunk.empty:
                continue
            chunk["Stkcd"] = normalize_stock_codes(chunk["Stkcd"])
            chunk["Trdmnt"] = chunk["Trddt"].str.slice(0, 7)

            month = chunk["Trddt"].str.slice(5, 7)
            month_end_rows = chunk.loc[
                (month.eq("06") & chunk["Filling"].isin([0, 1]))
                | (month.eq("12") & chunk["Filling"].isin([0, 1, 2])),
                ["Stkcd", "Trddt", "Trdmnt", "Markettype", "Dsmvosd", "Filling", "Clsprc"],
            ].copy()
            # 每块只留各股票各月最后一天；合并所有块后再取全市场统一月末。
            month_end_rows = month_end_rows.sort_values("Trddt").groupby(["Stkcd", "Trdmnt"]).tail(1)
            snapshot_parts.append(month_end_rows)

            chunk = chunk.loc[chunk["Markettype"].isin(market_types)].copy()
            chunk = chunk.sort_values(["Stkcd", "Trddt"])
            chunk["calendar_gross"] = (1 + chunk["Dretnd"]) / (1 - chunk["Dretwd"] + chunk["Dretnd"])
            calendar_groups = chunk.groupby("Stkcd", sort=False)
            first_calendar_rows = chunk.index[calendar_groups.cumcount().eq(0)]
            last_calendar_rows = calendar_groups.tail(1)
            for source_column, previous_column, last_values_across_chunks in [
                ("Clsprc", "prev_calendar_price", last_calendar_prices),
                ("Filling", "prev_calendar_filling", last_calendar_fillings),
                ("calendar_gross", "prev_calendar_gross", last_calendar_gross_returns),
            ]:
                chunk[previous_column] = calendar_groups[source_column].shift()
                chunk.loc[first_calendar_rows, previous_column] = chunk.loc[first_calendar_rows, "Stkcd"].map(last_values_across_chunks)
                last_values_across_chunks.update(dict(zip(last_calendar_rows["Stkcd"], last_calendar_rows[source_column])))
            daily_trade_rows = chunk.loc[chunk["Filling"].eq(0)].sort_values(["Stkcd", "Trddt"]).copy()
            if not daily_trade_rows.empty:
                # 半年内只留末次真实交易，用于识别停牌缩股后价格未同步调整的快照。
                daily_trade_rows["halfyear"] = daily_trade_rows["Trddt"].str.slice(0, 4) + np.where(
                    daily_trade_rows["Trddt"].str.slice(5, 7).le("06"), "-06", "-12",
                )
                actual_trade_snapshot_parts.append(daily_trade_rows.groupby(["Stkcd", "halfyear"], sort=False).tail(1)[
                    ["Stkcd", "halfyear", "Trddt", "Clsprc", "Dsmvosd"]
                ].copy())
                if daily_trade_rows["Dretwd"].lt(-1).any():
                    raise ValueError("日表存在小于-100%的收益率")
                daily_trade_rows["trade_date"] = pd.to_datetime(daily_trade_rows["Trddt"])
                daily_trade_rows["prev_real"] = daily_trade_rows.groupby("Stkcd")["Clsprc"].shift()
                daily_trade_rows["prev2_real"] = daily_trade_rows.groupby("Stkcd")["Clsprc"].shift(2)
                daily_trade_rows["prev_date"] = daily_trade_rows.groupby("Stkcd")["trade_date"].shift()
                row_in_chunk = daily_trade_rows.groupby("Stkcd").cumcount()
                first_rows, second_rows = daily_trade_rows.index[row_in_chunk.eq(0)], daily_trade_rows.index[row_in_chunk.eq(1)]
                daily_trade_rows.loc[first_rows, "prev_real"] = daily_trade_rows.loc[first_rows, "Stkcd"].map(last_actual_prices)
                daily_trade_rows.loc[first_rows, "prev2_real"] = daily_trade_rows.loc[first_rows, "Stkcd"].map(second_last_actual_prices)
                daily_trade_rows.loc[second_rows, "prev2_real"] = daily_trade_rows.loc[second_rows, "Stkcd"].map(last_actual_prices)
                daily_trade_rows.loc[first_rows, "prev_date"] = daily_trade_rows.loc[first_rows, "Stkcd"].map(last_actual_dates)
                last_rows = daily_trade_rows.groupby("Stkcd", sort=False).tail(1)
                second_last_actual_prices.update(dict(zip(last_rows["Stkcd"], last_rows["prev_real"])))
                last_actual_prices.update(dict(zip(last_rows["Stkcd"], last_rows["Clsprc"])))
                last_actual_dates.update(dict(zip(last_rows["Stkcd"], last_rows["trade_date"])))
                daily_trade_rows = daily_trade_rows.merge(ex_right_events, on=["Stkcd", "Trddt"], how="left", validate="many_to_one")
                daily_trade_rows = pd.merge_asof(
                    daily_trade_rows.sort_values(["trade_date", "Stkcd"]), reform_events.sort_values(["reform_date", "Stkcd"]),
                    left_on="trade_date", right_on="reform_date", by="Stkcd", direction="backward",
                )
                daily_trade_rows = pd.merge_asof(
                    daily_trade_rows.sort_values(["trade_date", "Stkcd"]),
                    return_events[["Stkcd", "return_event_date"]].sort_values(["return_event_date", "Stkcd"]),
                    left_on="trade_date", right_on="return_event_date", by="Stkcd", direction="backward",
                )
                # 观察参考价因子：(1+不含红利收益)/(1-含红利收益+不含红利收益)。
                daily_trade_rows["观察复利"] = (1 + daily_trade_rows["Dretnd"]) / (1 - daily_trade_rows["Dretwd"] + daily_trade_rows["Dretnd"])
                actual_price_gross_return = (daily_trade_rows["Clsprc"] / daily_trade_rows["prev_real"]).where(daily_trade_rows["prev_real"].gt(0), daily_trade_rows["观察复利"])
                close_to_actual_price = (actual_price_gross_return - daily_trade_rows["观察复利"]).abs().le(
                    reference_price_tolerance * np.maximum(1, daily_trade_rows["观察复利"].abs())
                )
                daily_trade_rows["选定复利"] = daily_trade_rows["观察复利"].where(~close_to_actual_price, actual_price_gross_return)
                theoretical_candidates = pd.DataFrame({
                    "普通方案": daily_trade_rows["Clsprc"] * (daily_trade_rows["mult_o"] + daily_trade_rows["ratio_o"])
                    / (daily_trade_rows["prev_real"] - daily_trade_rows["cash_o"] + daily_trade_rows["cost_o"]),
                    "年度扩展方案": daily_trade_rows["Clsprc"] * (daily_trade_rows["mult"] + daily_trade_rows["rights_ratio"])
                    / (daily_trade_rows["prev_real"] - daily_trade_rows["cash"] + daily_trade_rows["rights_cost"]),
                    "备选配股比例": daily_trade_rows["Clsprc"] * (daily_trade_rows["mult"] + daily_trade_rows["alt_ratio"])
                    / (daily_trade_rows["prev_real"] - daily_trade_rows["cash"] + daily_trade_rows["alt_cost"]),
                })
                candidate_differences = theoretical_candidates.sub(daily_trade_rows["观察复利"], axis=0).abs().fillna(np.inf)
                closest_candidate = pd.Series(
                    theoretical_candidates.to_numpy()[np.arange(len(daily_trade_rows)), candidate_differences.to_numpy().argmin(axis=1)],
                    index=daily_trade_rows.index,
                )
                candidate_is_valid = (closest_candidate - daily_trade_rows["观察复利"]).abs().le(
                    reference_price_tolerance * np.maximum(1, daily_trade_rows["观察复利"].abs())
                )
                has_plan = daily_trade_rows["has_plan"].fillna(False).astype(bool)
                daily_trade_rows.loc[has_plan, "选定复利"] = closest_candidate.where(candidate_is_valid, daily_trade_rows["观察复利"])
                # 股本衔接已确认的两段转增：仅年度部分匹配时，补回遗漏的流通股东股改对价。
                annual_only_gross_return = daily_trade_rows["Clsprc"] * daily_trade_rows["annual_mult"] / daily_trade_rows["prev_real"]
                missing_first_stage = (
                    daily_trade_rows["first_mult"].gt(1) & daily_trade_rows["annual_mult"].gt(1) & daily_trade_rows["prev_real"].gt(0)
                    & (annual_only_gross_return - daily_trade_rows["观察复利"]).abs().le(
                        reference_price_tolerance * np.maximum(1, daily_trade_rows["观察复利"].abs())
                    )
                )
                daily_trade_rows.loc[missing_first_stage, "选定复利"] = annual_only_gross_return * daily_trade_rows["first_mult"]
                # 增发新股不是原持有人取得的送股，不计入收益；同日真实分红另按方案处理。
                issue_without_dividend = daily_trade_rows["issue"].fillna(False).astype(bool) & ~has_plan
                daily_trade_rows.loc[issue_without_dividend, "选定复利"] = actual_price_gross_return
                intervening_reform = daily_trade_rows["reform_date"].gt(daily_trade_rows["prev_date"]) & daily_trade_rows["reform_date"].le(daily_trade_rows["trade_date"])
                misaligned_previous_price = (
                    (daily_trade_rows["Clsprc"] / daily_trade_rows["prev2_real"] - daily_trade_rows["观察复利"]).abs().le(
                        misaligned_price_tolerance * np.maximum(1, daily_trade_rows["观察复利"].abs())
                    ) & ~has_plan & ~intervening_reform & ~close_to_actual_price
                )
                daily_trade_rows.loc[misaligned_previous_price, "选定复利"] = actual_price_gross_return
                # 仅向特定对象分配的非年度重整，不能把全部转增视为原股东所得。
                # 若同日还有普通股东方案，仍使用上面的普通方案验证结果。
                special_restructuring = daily_trade_rows["special"].fillna(False).astype(bool) & ~has_plan
                daily_trade_rows.loc[special_restructuring, "选定复利"] = daily_trade_rows["Clsprc"] / daily_trade_rows["PreClosePrice"]
                # 老年度纯现金方案有时未体现在升级日收益中，按原始派现金额恢复。
                historical_cash_dividend = (
                    daily_trade_rows["legacy_cash"].fillna(False).astype(bool)
                    & np.isfinite(theoretical_candidates["年度扩展方案"]) & theoretical_candidates["年度扩展方案"].gt(0)
                )
                daily_trade_rows.loc[historical_cash_dividend, "选定复利"] = theoretical_candidates["年度扩展方案"]
                # 少量早年记录在停牌技术日改变填充价，复牌收益又引用该填充价。
                # 过滤停牌行后不能丢掉这段价格变动；两段均能由原始价格解释才接回真实前价。
                # 停牌期间有分红、配股或股改时排除，避免抹去真实除权；例如000408的2000-08-24。
                no_intervening_ex_rights = ~daily_trade_rows["return_event_date"].gt(daily_trade_rows["prev_date"])
                technical_fill_return = (
                    daily_trade_rows["prev_calendar_filling"].eq(1)
                    & daily_trade_rows["prev_calendar_price"].gt(0) & daily_trade_rows["prev_real"].gt(0)
                    & daily_trade_rows["prev_calendar_price"].ne(daily_trade_rows["prev_real"])
                    & daily_trade_rows["PreClosePrice"].eq(daily_trade_rows["prev_real"])
                    & (daily_trade_rows["Clsprc"] / daily_trade_rows["prev_calendar_price"] - daily_trade_rows["观察复利"]).abs().le(
                        reference_price_tolerance * np.maximum(1, daily_trade_rows["观察复利"].abs())
                    )
                    & (daily_trade_rows["prev_calendar_price"] / daily_trade_rows["prev_real"] - daily_trade_rows["prev_calendar_gross"]).abs().le(
                        reference_price_tolerance * np.maximum(1, daily_trade_rows["prev_calendar_gross"].abs())
                    )
                    & no_intervening_ex_rights & ~has_plan & ~intervening_reform & ~close_to_actual_price & ~misaligned_previous_price & ~issue_without_dividend
                )
                daily_trade_rows.loc[technical_fill_return, "选定复利"] = actual_price_gross_return
                technical_fill_corrections += int(technical_fill_return.sum())
                daily_trade_rows = daily_trade_rows.loc[
                    daily_trade_rows["Trdmnt"].between(start_month.strftime("%Y-%m"), end_month.strftime("%Y-%m"))
                ].copy()
                daily_trade_rows["有交易日"] = daily_trade_rows["Dnshrtrd"].gt(0).astype(int)
                daily_trade_rows["缺失日收益"] = daily_trade_rows["观察复利"].isna().astype(int)
                daily_monthly_return_parts.append(daily_trade_rows.groupby(["Stkcd", "Trdmnt"], as_index=False).agg(
                    月参考复利=("选定复利", "prod"),
                    日交易天数=("有交易日", "sum"),
                    日收益缺失数=("缺失日收益", "sum"),
                ))
            supplement_rows = chunk.loc[
                (chunk["Trddt"] < first_original_month.strftime("%Y-%m-%d"))
                & (chunk["Trddt"] >= start_month.strftime("%Y-%m-%d"))
                & chunk["Filling"].eq(0),
                ["Stkcd", "Trddt", "Trdmnt", "Dretwd", "Dnshrtrd"],
            ].copy()
            if not supplement_rows.empty:
                supplement_return_parts.append(supplement_rows)

            if len(code_change_index) and not daily_trade_rows.empty:
                chunk_index = pd.MultiIndex.from_frame(daily_trade_rows[["Stkcd", "Trdmnt"]])
                code_change_days = daily_trade_rows.loc[
                    chunk_index.isin(code_change_index),
                    ["Stkcd", "Trddt", "Trdmnt", "Dnshrtrd", "观察复利", "选定复利"],
                ].copy()
                if not code_change_days.empty:
                    code_change_return_parts.append(code_change_days)

    print(f"  修复停牌技术填充收益引用：{technical_fill_corrections:,}条", flush=True)
    if not snapshot_parts:
        raise ValueError("日表没有六月或十二月的市值记录")
    snapshots = pd.concat(snapshot_parts, ignore_index=True).drop_duplicates()
    # 六月按最后交易日，十二月允许12月31日节假日记录；周末无记录时取最近记录日。
    # 已提前退出面板的股票不取该股票更早日期替代统一快照日。
    common_month_end_dates = snapshots.groupby("Trdmnt")["Trddt"].transform("max")
    snapshots = snapshots.loc[snapshots["Trddt"].eq(common_month_end_dates)].copy()
    check_unique_key(snapshots, ["Stkcd", "Trdmnt"], "日度月末市值快照")
    snapshots["Trddt"] = pd.to_datetime(snapshots["Trddt"])
    snapshots["Trdmnt"] = pd.to_datetime(snapshots["Trdmnt"])
    snapshots["year"] = snapshots["Trdmnt"].dt.year  # 快照日所在自然年。
    required_snapshots = (
        snapshots["Trdmnt"].dt.month.eq(6) & snapshots["year"].between(start_formation_year, end_formation_year)
    ) | (
        snapshots["Trdmnt"].dt.month.eq(12) & snapshots["year"].between(start_formation_year - 1, end_formation_year - 1)
    )
    snapshots = snapshots.loc[required_snapshots].copy()
    snapshots["me_a"] = snapshots["Dsmvosd"] * 1000  # 升级日表A股流通市值，元。
    b_share_market_values = snapshots.loc[snapshots["Markettype"].isin([2, 8]), ["Stkcd", "Trddt", "me_a"]].rename(
        columns={"Stkcd": "ABSymbol", "me_a": "me_b"},
    )
    # 数值核对表明，升级日表的B股市值已折成人民币；不能再乘一次汇率。
    # 月表B股市值仍以美元或港币计价，不能直接代替这里的日表数值。
    snapshots = snapshots.loc[snapshots["Markettype"].isin(market_types)].copy()

    share_structure = pd.read_csv(
        share_structure_path, encoding="utf-8-sig",
        usecols=["Symbol", "ChangeDate", "DeclareDate", "EventID", "ChangeReasonID", "ABSymbol", "TradableSharesA",
                 "TradableSharesB", "TradeSharesTotal"],
        dtype={"Symbol": "string", "ABSymbol": "string"}, parse_dates=["ChangeDate", "DeclareDate"],
    ).rename(columns={"Symbol": "Stkcd"})
    share_structure["Stkcd"] = normalize_stock_codes(share_structure["Stkcd"])
    share_structure["ABSymbol"] = share_structure["ABSymbol"].str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    # 同日可能记录多次股本事件，取事件编号最后的一次。
    share_structure = share_structure.sort_values(["ChangeDate", "EventID"]).drop_duplicates(["Stkcd", "ChangeDate"], keep="last")
    inherited_share_records = code_aliases.merge(share_structure, left_on="FormerCode", right_on="Stkcd", how="inner")
    inherited_share_records["Stkcd"] = inherited_share_records["new_code"]
    share_structure = pd.concat([share_structure, inherited_share_records[share_structure.columns]], ignore_index=True)
    share_structure = share_structure.drop_duplicates(["Stkcd", "ChangeDate"], keep="first")
    split_events = share_structure.loc[share_structure["ChangeReasonID"].eq("P0710"), ["Stkcd", "ChangeDate"]].rename(
        columns={"ChangeDate": "shrink_date"},
    )
    snapshots = pd.merge_asof(
        snapshots.sort_values(["Trddt", "Stkcd"]), share_structure.sort_values(["ChangeDate", "Stkcd"]),
        left_on="Trddt", right_on="ChangeDate", by="Stkcd", direction="backward",
    ).merge(b_share_market_values, on=["ABSymbol", "Trddt"], how="left", validate="many_to_one")
    valid_share_structure = snapshots["TradableSharesA"].gt(0)
    other_overseas_shares = (
        snapshots["TradeSharesTotal"] - snapshots["TradableSharesA"] - snapshots["TradableSharesB"].fillna(0)
    )
    invalid_share_structure = (valid_share_structure & snapshots["TradeSharesTotal"].isna()) | other_overseas_shares.lt(0)
    if invalid_share_structure.any():
        examples = snapshots.loc[invalid_share_structure, ["Stkcd", "Trddt", "ChangeDate"]].head(10).to_string(index=False)
        raise ValueError(f"股本记录存在全部流通股缺失或境外流通股数为负：\n{examples}")
    # 包含H股及其他境外流通股，不能只取H股字段；例如S股也需要计入。
    snapshots["me_overseas"] = (
        snapshots["me_a"] * other_overseas_shares / snapshots["TradableSharesA"]
    ).where(valid_share_structure, 0)
    missing_b_share_values = snapshots["TradableSharesB"].gt(0) & snapshots["me_b"].isna()
    if missing_b_share_values.any():
        examples = snapshots.loc[missing_b_share_values, ["Stkcd", "Trddt", "ABSymbol"]].head(10).to_string(index=False)
        raise ValueError(f"存在B股流通股本但缺少同期B股人民币市值：\n{examples}")
    snapshots["me_b"] = snapshots["me_b"].fillna(0)
    snapshots["me"] = snapshots["me_a"] + snapshots["me_b"] + snapshots["me_overseas"]
    actual_trade_snapshots = pd.concat(actual_trade_snapshot_parts, ignore_index=True).drop_duplicates()
    actual_trade_snapshots = actual_trade_snapshots.sort_values("Trddt").drop_duplicates(["Stkcd", "halfyear"], keep="last")
    actual_trade_snapshots["actual_date"] = pd.to_datetime(actual_trade_snapshots["Trddt"])
    actual_trade_snapshots["actual_me_a"] = actual_trade_snapshots["Dsmvosd"] * 1000
    actual_trade_snapshots = actual_trade_snapshots.rename(columns={"Clsprc": "actual_price"})
    snapshots = pd.merge_asof(
        snapshots.sort_values(["Trddt", "Stkcd"]),
        actual_trade_snapshots[["Stkcd", "actual_date", "actual_me_a", "actual_price"]].sort_values(["actual_date", "Stkcd"]),
        left_on="Trddt", right_on="actual_date", by="Stkcd", direction="backward",
    )
    snapshots = pd.merge_asof(
        snapshots.sort_values(["Trddt", "Stkcd"]), split_events.sort_values(["shrink_date", "Stkcd"]),
        left_on="Trddt", right_on="shrink_date", by="Stkcd", direction="backward",
    )
    shrink_without_new_price = (
        snapshots["Filling"].eq(1) & snapshots["shrink_date"].gt(snapshots["actual_date"])
        & snapshots["Clsprc"].eq(snapshots["actual_price"])
        & snapshots["actual_me_a"].gt(snapshots["me_a"])
    )
    # P0710包含拆细与缩股；同价下市值下降才属于本规则要保护的缩股情形。
    # 不能只改股数而保留缩股前价格：等待新的成交价或技术除权价后再使用新口径。
    # 原始公告可核对600705的2008年重组及2012年恢复上市说明：
    # https://static.cninfo.com.cn/finalpage/2008-10-28/45718875.PDF
    # https://static.cninfo.com.cn/finalpage/2012-08-23/61464936.PDF
    snapshots.loc[shrink_without_new_price, "me"] += (
        snapshots.loc[shrink_without_new_price, "actual_me_a"] - snapshots.loc[shrink_without_new_price, "me_a"]
    )
    print(f"  保护缩股后价格未同步调整的市值快照：{shrink_without_new_price.sum():,}条", flush=True)
    employee_listing_suspended = (
        snapshots["Filling"].eq(1)
        & snapshots["ChangeReasonID"].str.contains("P0760", regex=False, na=False)
        & snapshots["ChangeDate"].gt(snapshots["actual_date"])
        & snapshots["Clsprc"].eq(snapshots["actual_price"])
        & snapshots["actual_me_a"].gt(0) & snapshots["me_a"].gt(snapshots["actual_me_a"])
    )
    # 职工股上市但持续停牌、价格仍沿用末次成交价时，延续末次成交的A股市值口径。
    # 只认最新股本事件中的P0760；其他股本事件不能沿用更早的职工股上市保护。
    # 600538年报记载2005-12-27职工股上市，当时股票仍持续停牌：
    # https://static.cninfo.com.cn/finalpage/2009-06-03/53042412.PDF
    snapshots.loc[employee_listing_suspended, "me"] += (
        snapshots.loc[employee_listing_suspended, "actual_me_a"] - snapshots.loc[employee_listing_suspended, "me_a"]
    )
    print(f"  延续职工股上市停牌期间的末次成交市值：{employee_listing_suspended.sum():,}条", flush=True)
    year_end_incentive_backfill = (
        snapshots["Trdmnt"].dt.month.eq(12) & snapshots["Filling"].eq(1)
        & snapshots["ChangeReasonID"].str.contains("P0717", regex=False, na=False)
        & snapshots["ChangeDate"].dt.month.eq(12) & snapshots["ChangeDate"].dt.day.eq(31)
        & snapshots["DeclareDate"].gt(snapshots["Trddt"])
        & snapshots["ChangeDate"].gt(snapshots["actual_date"])
        & snapshots["Clsprc"].eq(snapshots["actual_price"]) & snapshots["actual_me_a"].gt(0)
        & ~(shrink_without_new_price | employee_listing_suspended)
    )
    # 年报披露的股权激励汇总有时回填至12月31日，停牌填充值随之变动。
    # 仅限该类年末回填，不将所有停牌或所有高管股变动回退到末次成交。
    # 例如300186：2014-12-26末次成交市值与原2014-12月表一致，12月31日回填增加股数，
    # 相应P0717记录的公告日期为2015-04-28；这里只替换A股部分。
    snapshots.loc[year_end_incentive_backfill, "me"] += (
        snapshots.loc[year_end_incentive_backfill, "actual_me_a"] - snapshots.loc[year_end_incentive_backfill, "me_a"]
    )
    print(f"  修复停牌期间年末股权激励回填市值：{year_end_incentive_backfill.sum():,}条", flush=True)
    snapshots = snapshots.merge(june_monthly_values, on=["Stkcd", "Trdmnt"], how="left", validate="one_to_one")
    june_reform_unlock_suspended = (
        snapshots["Trdmnt"].dt.month.eq(6) & snapshots["Filling"].eq(1)
        & snapshots["monthly_me_a"].isna()
        & snapshots["ChangeReasonID"].str.contains("P0757", regex=False, na=False)
        & snapshots["ChangeDate"].gt(snapshots["actual_date"])
        & snapshots["Clsprc"].eq(snapshots["actual_price"])
        & snapshots["actual_me_a"].gt(0) & snapshots["me_a"].gt(snapshots["actual_me_a"])
        & ~(shrink_without_new_price | employee_listing_suspended | year_end_incentive_backfill)
    )
    # P0757为股权分置股份上市；只处理六月整月无成交、填充值仍用旧价的情形。
    # 例如000582于2012-02-01股改限售股上市，六月仍停牌且沿用1月20日价格。
    # 仅延续A股市值；有B股时，已匹配的B股人民币市值继续单独计入。
    snapshots.loc[june_reform_unlock_suspended, "me"] += (
        snapshots.loc[june_reform_unlock_suspended, "actual_me_a"] - snapshots.loc[june_reform_unlock_suspended, "me_a"]
    )
    print(f"  延续六月股改解禁后仍停牌的末次成交市值：{june_reform_unlock_suspended.sum():,}条", flush=True)
    june_suspended_with_trades = (
        snapshots["Trdmnt"].dt.month.eq(6) & snapshots["Filling"].eq(1)
        & snapshots["monthly_me_a"].gt(0)
        & snapshots["monthly_price"].eq(snapshots["Clsprc"])
        & ~(shrink_without_new_price | employee_listing_suspended | year_end_incentive_backfill)
    )
    # 价格不相同可能来自停牌期间除息；此时保留日表的除息后技术填充值。
    # 仅替换A股部分，整月无成交不跨月回溯。十二月仍取日快照，不能机械套用六月替换。
    snapshots.loc[june_suspended_with_trades, "me"] += (
        snapshots.loc[june_suspended_with_trades, "monthly_me_a"] - snapshots.loc[june_suspended_with_trades, "me_a"]
    )
    snapshots = snapshots.loc[np.isfinite(snapshots["me"]) & snapshots["me"].gt(0)].copy()

    if supplement_return_parts:
        supplement_daily = pd.concat(supplement_return_parts, ignore_index=True).drop_duplicates()
        check_unique_key(supplement_daily, ["Stkcd", "Trddt"], "早年日收益表")
        if supplement_daily["Dretwd"].lt(-1).any():
            raise ValueError("早年日表存在小于-100%的收益率")
        supplement_daily["gross"] = 1 + supplement_daily["Dretwd"].fillna(0)  # 单日复利因子。
        supplement_daily["有交易"] = supplement_daily["Dnshrtrd"].gt(0).astype(int)
        supplement_daily["缺失日收益数"] = supplement_daily["Dretwd"].isna().astype(int)
        supplement_monthly = supplement_daily.groupby(["Stkcd", "Trdmnt"], as_index=False).agg(
            Mretwd=("gross", "prod"),
            Ndaytrd=("有交易", "sum"),
            缺失日收益数=("缺失日收益数", "sum"),
        )
        supplement_monthly["Mretwd"] = supplement_monthly["Mretwd"] - 1
        supplement_monthly["Trdmnt"] = pd.to_datetime(supplement_monthly["Trdmnt"])
        supplement_monthly = supplement_monthly.merge(companies[["Stkcd", "Listdt"]], on="Stkcd", how="left", validate="many_to_one")
        # 原始月表的上市首月收益为空，不能把日表相对发行价的首日涨幅带入。
        first_listing_month = supplement_monthly["Trdmnt"].dt.to_period("M").eq(supplement_monthly["Listdt"].dt.to_period("M"))
        supplement_monthly.loc[first_listing_month, "Mretwd"] = np.nan
        supplement_monthly = supplement_monthly.drop(columns="Listdt")
        supplement_monthly["月收益来自日表"] = 1
        supplement_monthly["跨代码补充月份"] = 0
        monthly_data = pd.concat([monthly_data, supplement_monthly], ignore_index=True)

    if daily_monthly_return_parts:
        # 月表只提供交易月数和上市首月空值口径；有效收益按日参考价重新构造。
        daily_monthly_returns = pd.concat(daily_monthly_return_parts, ignore_index=True).groupby(
            ["Stkcd", "Trdmnt"], as_index=False,
        ).agg(月参考复利=("月参考复利", "prod"),
              日交易天数=("日交易天数", "sum"), 日收益缺失数=("日收益缺失数", "sum"))
        daily_monthly_returns["Trdmnt"] = pd.to_datetime(daily_monthly_returns["Trdmnt"])
        monthly_data = monthly_data.merge(daily_monthly_returns, on=["Stkcd", "Trdmnt"], how="left", validate="one_to_one")
        can_rebuild = monthly_data["Mretwd"].notna() & monthly_data["月参考复利"].gt(0) & np.isfinite(monthly_data["月参考复利"])
        monthly_data.loc[can_rebuild, "Mretwd"] = (monthly_data.loc[can_rebuild, "月参考复利"] - 1).round(6)
        monthly_data.loc[can_rebuild, "月收益来自日表"] = 1
        monthly_data.loc[can_rebuild, "缺失日收益数"] = monthly_data.loc[can_rebuild, "日收益缺失数"]
        print(f"  按参考价重建月收益：{can_rebuild.sum():,}个股票月", flush=True)
        monthly_data = monthly_data.drop(columns=["日交易天数", "日收益缺失数", "月参考复利"])

    # 在换码后的首个形成年度，补入同一证券旧代码下的前置月份。
    # 不改写换码前的历史标签代码；新代码已有的原始月份优先保留。
    code_change_daily = pd.concat(code_change_return_parts, ignore_index=True).drop_duplicates() if code_change_return_parts else pd.DataFrame()
    if not code_change_daily.empty:
        check_unique_key(code_change_daily, ["Stkcd", "Trddt"], "换码月份日收益")
    for _, change_event in code_change_events.iterrows():
        new_code, old_code = change_event["new_code"], change_event["FormerCode"]
        code_change_month = pd.Timestamp(change_event["change_month"])
        formation_year = code_change_month.year + int(code_change_month.month >= 7)
        window_start = pd.Timestamp(formation_year - 1, 7, 1)
        window_end = pd.Timestamp(formation_year, 6, 1)
        native_months = monthly_data.loc[
            monthly_data["Stkcd"].eq(new_code) & monthly_data["Trdmnt"].between(window_start, window_end)
        ]
        # 新证券本身已达交易月数门槛时，保留原生窗口，不额外接入旧证券收益。
        if native_months["Ndaytrd"].gt(0).sum() >= min_trading_months:
            continue
        old_code_months = monthly_data.loc[
            monthly_data["Stkcd"].eq(old_code) & monthly_data["Trdmnt"].ge(window_start) & monthly_data["Trdmnt"].lt(code_change_month)
        ].copy()
        change_day_rows = code_change_daily.loc[
            code_change_daily["Stkcd"].isin([new_code, old_code]) & code_change_daily["Trdmnt"].eq(change_event["change_month"])
        ].copy() if not code_change_daily.empty else pd.DataFrame()
        # 有些北交所旧代码已被数据商整体回写为新代码，不应再次桥接。
        has_old_daily_rows = not change_day_rows.empty and change_day_rows["Stkcd"].eq(old_code).any()
        if old_code_months.empty and not has_old_daily_rows:
            continue
        old_code_months["Stkcd"] = new_code
        old_code_months["跨代码补充月份"] = 1
        existing_new_code_months = monthly_data.loc[monthly_data["Stkcd"].eq(new_code), "Trdmnt"]
        old_code_months = old_code_months.loc[~old_code_months["Trdmnt"].isin(existing_new_code_months)]
        monthly_data = pd.concat([monthly_data, old_code_months], ignore_index=True)
        if not change_day_rows.empty:
            check_unique_key(change_day_rows, ["Trddt"], f"{old_code}转{new_code}的换码交易日")
            code_change_gross_return = change_day_rows["选定复利"].prod()
            if not np.isfinite(code_change_gross_return) or code_change_gross_return <= 0:
                raise ValueError(f"{old_code}转{new_code}的换码月份无法还原有效参考价收益")
            code_change_month_row = pd.DataFrame([{
                "Stkcd": new_code, "Trdmnt": code_change_month,
                "Mretwd": round(code_change_gross_return - 1, 6),
                "Ndaytrd": int(change_day_rows["Dnshrtrd"].gt(0).sum()),
                "月收益来自日表": 1, "缺失日收益数": int(change_day_rows["观察复利"].isna().sum()),
                "跨代码补充月份": 1,
            }])
            monthly_data = monthly_data.loc[~(monthly_data["Stkcd"].eq(new_code) & monthly_data["Trdmnt"].eq(code_change_month))]
            monthly_data = pd.concat([monthly_data, code_change_month_row], ignore_index=True)
        print(f"  收益代码衔接：{old_code}→{new_code}，形成年度{formation_year}", flush=True)
    check_unique_key(monthly_data, ["Stkcd", "Trdmnt"], "补齐后的月收益表")

    required_months = pd.date_range(start_month, end_month, freq="MS")
    missing_months = required_months.difference(pd.DatetimeIndex(monthly_data["Trdmnt"].unique()))
    if len(missing_months):
        raise ValueError(f"原始数据整体缺少以下月份：{missing_months.strftime('%Y-%m').tolist()}")
    if monthly_data["Mretwd"].lt(-1).any():
        raise ValueError("月表存在小于-100%的收益率")

    # ========================================================================
    # 3. 上年7月至当年6月动量：缺失收益补0，但缺行不增加有效交易月数
    # ========================================================================
    print("3/7 计算过去十二个月动量……", flush=True)
    monthly_data["formation_year"] = np.where(
        monthly_data["Trdmnt"].dt.month.ge(7),
        monthly_data["Trdmnt"].dt.year + 1, monthly_data["Trdmnt"].dt.year,
    )  # formation_year（形成年度）。
    monthly_data["gross"] = 1 + monthly_data["Mretwd"].fillna(0)
    monthly_data["有交易月份"] = monthly_data["Ndaytrd"].gt(0).astype(int)
    monthly_data["缺失月收益数"] = monthly_data["Mretwd"].isna().astype(int)
    momentum_data = monthly_data.groupby(["Stkcd", "formation_year"], as_index=False).agg(
        mom=("gross", "prod"),  # mom（复合动量），单位为小数收益率。
        observed_months=("Trdmnt", "size"),  # 实际存在记录的月份数。
        valid_months=("有交易月份", "sum"),  # 实际有交易的月份数。
        missing_return_months=("缺失月收益数", "sum"),
        daily_derived_months=("月收益来自日表", "sum"),
        missing_daily_returns=("缺失日收益数", "sum"),
        code_linked_months=("跨代码补充月份", "sum"),  # 用曾用代码补充或重建的月份数。
    )
    momentum_data["mom"] = momentum_data["mom"] - 1
    stage_samples["存在动量月份"] = momentum_data[["Stkcd", "formation_year"]].copy()
    momentum_data = momentum_data.loc[momentum_data["valid_months"].ge(min_trading_months)].copy()
    stage_samples["交易月份达标"] = momentum_data[["Stkcd", "formation_year"]].copy()

    june_data = snapshots.loc[snapshots["Trdmnt"].dt.month.eq(6)].rename(columns={
        "year": "formation_year", "me": "size_june",  # 六月末流通市值，元。
        "Trddt": "formation_date",  # 形成日。
        "Filling": "june_filling",  # 形成日填充标识。
    })
    june_data = june_data.loc[june_data["formation_year"].between(start_formation_year, end_formation_year)].copy()
    december_data = snapshots.loc[snapshots["Trdmnt"].dt.month.eq(12)].rename(columns={
        "year": "fiscal_year", "me": "me_dec",  # 上年自然年、年末流通市值。
        "Trddt": "dec_date",  # 上年末市值快照日。
    })
    check_unique_key(june_data, ["Stkcd", "formation_year"], "六月市值表")
    check_unique_key(december_data, ["Stkcd", "fiscal_year"], "十二月市值表")
    missing_june_years = sorted(set(range(start_formation_year, end_formation_year + 1)) - set(june_data["formation_year"]))
    missing_december_years = sorted(set(range(start_formation_year - 1, end_formation_year)) - set(december_data["fiscal_year"]))
    if missing_june_years or missing_december_years:
        raise ValueError(f"日表快照年度覆盖不足：缺六月{missing_june_years}，缺十二月{missing_december_years}")
    stage_samples["存在六月末市值"] = june_data[["Stkcd", "formation_year"]].copy()
    stage_samples["存在上年末市值"] = december_data[["Stkcd", "fiscal_year"]].rename(columns={"fiscal_year": "formation_year"})
    stage_samples["存在上年末市值"]["formation_year"] += 1

    # ========================================================================
    # 4. 上年末合并年报账面权益及原始账面市值比
    # ========================================================================
    print("4/7 计算账面权益、原始账面市值比……", flush=True)
    account_columns = ["A003000000", "A003112101", "A001222000", "A002208000"]
    # 依次为所有者权益合计、优先股、递延所得税资产、递延所得税负债。
    financials = pd.read_csv(
        balance_sheet_path, encoding="utf-8-sig", usecols=["Stkcd", "Accper", "Typrep"] + account_columns,
        dtype={"Stkcd": "string"}, parse_dates=["Accper"],
    )
    financials = financials.loc[
        financials["Typrep"].eq("A")
        & financials["Accper"].dt.month.eq(12) & financials["Accper"].dt.day.eq(31)
        & financials["Accper"].dt.year.between(start_formation_year - 1, end_formation_year - 1)
    ].copy().drop_duplicates()
    financials["Stkcd"] = normalize_stock_codes(financials["Stkcd"])
    check_unique_key(financials, ["Stkcd", "Accper"], "合并年报")
    financials["fiscal_year"] = financials["Accper"].dt.year
    financials["statement_version_source"] = "本地合并年报"  # 财报版本与核对来源。
    # 本地数据下载于后续更正之后，以下科目取已核对的历史公告版本，不按标签补数。
    # 如2025年及以前已有更正，保留该更正；不能把2026年的追溯值与首次值简单二选一。
    # 新元科技2024首次年报权益仅3,139,494.33元，扣递延所得税资产后为负，须自然剔除。
    historical_statement_versions = [
        # 上海临港2026年追溯更正把原权益改列金融负债；恢复形成期已公布的科目。
        ("600848", 2022, 27_417_253_490.19, 0.0, 617_873_197.71, 9_195_082.22,
         "https://static.cninfo.com.cn/finalpage/2023-04-14/1216402918.PDF"),
        ("600848", 2023, 31_980_638_713.17, 0.0, 927_081_551.16, 88_316_900.04,
         "https://static.cninfo.com.cn/finalpage/2024-04-13/1219596576.PDF"),
        ("600848", 2024, 33_841_350_248.87, 0.0, 1_250_482_226.64, 78_551_088.37,
         "https://static.cninfo.com.cn/finalpage/2025-04-12/1223073233.PDF"),
        # 新元科技保留2025年已披露的前期差错更正，剔除2026年之后的追溯值。
        ("300472", 2022, 670_121_620.22, 0.0, 80_592_444.46, 190_974.69,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=11082531&stockid=300472"),
        ("300472", 2023, 443_658_309.05, 0.0, 65_212_747.02, 3_459.69,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=11082538&stockid=300472"),
        ("300472", 2024, 3_139_494.33, 0.0, 6_298_611.37, 0.0,
         "https://disc.static.szse.cn/download/disc/disk03/finalpage/2025-04-29/6ccc7684-5f59-4fde-8939-0f458279418d.PDF"),
        # 太极股份2026年专项核查追溯更正2022—2024年，恢复各年首次公告的科目。
        ("002368", 2022, 4_491_810_880.84, 0.0, 158_238_208.80, 0.0,
         "https://stockn.xueqiu.com/SZ002368/20230406777560.pdf"),
        ("002368", 2023, 5_488_689_673.75, 0.0, 180_614_601.31, 2_220_751.85,
         "https://static.cninfo.com.cn/finalpage/2024-04-18/1219648035.PDF"),
        ("002368", 2024, 5_601_376_299.79, 0.0, 214_697_828.46, 946_131.48,
         "https://static.cninfo.com.cn/finalpage/2025-04-02/1222981389.PDF"),
        # 康佳2026年追溯更正2022—2024年，四科目均使用对应年度首次年报。
        ("000016", 2022, 8_461_372_961.92, 0.0, 1_190_454_750.62, 89_535_936.97,
         "https://static.cninfo.com.cn/finalpage/2023-03-28/1216238393.PDF"),
        ("000016", 2023, 5_906_622_864.14, 0.0, 1_426_573_982.16, 185_026_165.27,
         "https://static.cninfo.com.cn/finalpage/2024-04-02/1219494193.PDF"),
        ("000016", 2024, 2_234_651_683.79, 0.0, 1_392_239_301.87, 133_299_175.48,
         "https://static.cninfo.com.cn/finalpage/2025-04-15/1223095858.PDF"),
        # 长方集团2026年再次更正2020年报，恢复2021年首次披露的合并报表。
        ("300301", 2020, 1_157_136_266.08, 0.0, 142_002_578.29, 8_818_359.42,
         "https://static.cninfo.com.cn/finalpage/2021-04-27/1209818019.PDF"),
        # 福能东方2026年更正追溯到2020年，原权益和原递延所得税资产均需恢复。
        ("300173", 2020, 1_620_126_615.67, 0.0, 24_894_583.20, 10_861_710.09,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=7041615&stockid=300173"),
        ("300173", 2021, 1_345_035_568.72, 0.0, 51_955_245.55, 7_557_477.02,
         "https://static.cninfo.com.cn/finalpage/2022-04-23/1213062484.PDF"),
        # 同德化工2026年追溯更正2015—2024年，恢复各年原始科目。
        ("002360", 2015, 1_069_156_951.01, 0.0, 16_615_313.24, 0.0,
         "https://file.finance.sina.com.cn/211.154.219.97%3A9494/MRGG/CNSESZ_STOCK/2016/2016-3/2016-03-29/2281555.PDF"),
        ("002360", 2016, 1_121_818_349.89, 0.0, 13_637_906.28, 0.0,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=3203280&stockid=002360"),
        ("002360", 2017, 1_091_538_854.47, 0.0, 13_110_356.29, 0.0,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=4189896&stockid=002360"),
        ("002360", 2018, 1_076_068_523.47, 0.0, 12_643_685.35, 0.0,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=5145069&stockid=002360"),
        ("002360", 2019, 1_185_395_467.05, 0.0, 12_240_650.05, 1_042_549.35,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=6070370&stockid=002360"),
        ("002360", 2020, 1_313_400_064.63, 0.0, 11_442_364.76, 1_006_841.88,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=7036687&stockid=002360"),
        ("002360", 2021, 1_482_394_248.40, 0.0, 11_267_394.37, 1_276_839.12,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=7973280&stockid=002360"),
        ("002360", 2022, 1_751_797_341.56, 0.0, 17_258_513.52, 1_954_764.12,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=9068188&stockid=002360"),
        ("002360", 2023, 2_179_038_681.67, 0.0, 17_635_072.20, 384_494.67,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=9975046&stockid=002360"),
        ("002360", 2024, 1_996_763_990.80, 0.0, 18_993_129.90, 363_106.79,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=11018207&stockid=002360"),
        # 贵州百灵2026年更正涉及收入、权益及税项，须同时恢复原始权益与递延税资产。
        # 2019年须保留2021年已经披露的更正，2021年须保留2023年已经披露的更正。
        ("002424", 2019, 4_136_140_262.52, 0.0, 32_123_172.09, 19_100_427.95,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=7210871&stockid=002424"),
        # 2020四科目由首次年报和2026年更正前列共同支持；2023年追溯的完整四科目尚待核齐。
        ("002424", 2020, 3_920_278_826.78, 0.0, 46_894_661.08, 21_715_849.96,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=7210884&stockid=002424"),
        ("002424", 2021, 3_877_149_234.16, 0.0, 66_843_071.60, 28_809_781.27,
         "https://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESZ_STOCK/2023/2023-4/2023-04-29/9184383.PDF"),
        # 2022保留准则解释16号及2025年更正；以公司公告、权益分项与年报勾稽的版本为准。
        ("002424", 2022, 3_750_855_407.60, 0.0, 125_934_057.61, 33_038_443.00,
         "https://static.cninfo.com.cn/finalpage/2025-04-29/1223391337.PDF"),
        # 2023及2024已与2025年披露版本一致，保留本地科目，不能恢复为2023首次年报。
        # 安记食品2026年追溯更正2015—2024年，以下为已独立核对的原始科目。
        ("603696", 2015, 671_695_308.51, 0.0, 3_449_599.38, 0.0,
         "https://file.finance.sina.com.cn/211.154.219.97%3A9494/MRGG/CNSESH_STOCK/2016/2016-4/2016-04-19/2364315.PDF"),
        ("603696", 2016, 698_065_337.48, 0.0, 5_005_393.60, 0.0,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=3234120&stockid=603696"),
        ("603696", 2017, 720_406_229.98, 0.0, 5_895_319.75, 0.0,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=4409242&stockid=603696"),
        ("603696", 2018, 646_689_888.84, 0.0, 5_055_355.01, 0.0,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=5149042&stockid=603696"),
        ("603696", 2019, 591_089_966.56, 0.0, 4_236_020.85, 403_079.77,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=6046956&stockid=603696"),
        ("603696", 2020, 565_616_535.55, 0.0, 4_157_544.51, 367_659.67,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=7076924&stockid=603696"),
        # 2021采用原表完整四科目；后续解释16号的税项与权益调整在账面权益中抵消。
        ("603696", 2021, 548_748_872.76, 0.0, 3_217_211.59, 155_427.27,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=8027970&stockid=603696"),
        # 2022保留2023年解释16号比较列，调整后账面权益与首次年报相同。
        ("603696", 2022, 537_289_545.26, 0.0, 3_930_986.28, 5_900.89,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10030077&stockid=603696"),
        ("603696", 2023, 557_279_347.22, 0.0, 3_668_952.41, 3_415.59,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10030077&stockid=603696"),
        ("603696", 2024, 578_874_920.22, 0.0, 3_178_738.79, 9_292.16,
         "https://static.cninfo.com.cn/finalpage/2025-04-22/1223193223.PDF"),
        # 至纯科技2026年追溯调整权益和递延税项，四科目使用截至2025年已披露版本。
        ("603690", 2021, 4_290_450_735.98, 0.0, 53_394_479.66, 25_585_361.26,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=8246125&stockid=603690"),
        # 2023年报比较列较2022年修订稿权益增加0.01元，沿用较晚披露的比较列。
        ("603690", 2022, 4_709_256_301.70, 0.0, 127_303_912.77, 66_474_164.06,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10148983&stockid=603690"),
        ("603690", 2023, 5_066_427_383.47, 0.0, 130_745_674.20, 90_144_708.39,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10148983&stockid=603690"),
        ("603690", 2024, 4_926_936_542.27, 0.0, 147_978_187.16, 94_953_585.29,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=11051449&stockid=603690"),
        # 沐邦高科撤销2026年追溯值，但2024保留2025年8月已经披露的更正。
        ("603398", 2023, 742_434_999.47, 0.0, 39_371_250.89, 15_822_459.45,
         "https://static.cninfo.com.cn/finalpage/2025-04-30/1223426841.PDF"),
        ("603398", 2024, 947_348_421.70, 0.0, 95_435_827.87, 14_030_324.19,
         "https://static.cninfo.com.cn/finalpage/2025-08-28/1224600459.PDF"),
        # 诺泰生物2026年4月才追溯2021—2024；2023/2024的权益与税资调整相互抵消。
        ("688076", 2021, 1_808_768_493.78, 0.0, 16_634_178.34, 4_263_139.87,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=8114963&stockid=688076"),
        ("688076", 2022, 1_924_886_706.90, 0.0, 28_129_171.81, 10_773_073.73,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=9034084&stockid=688076"),
        ("688076", 2023, 2_189_303_674.75, 0.0, 46_646_365.92, 10_164_846.68,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10939637&stockid=688076"),
        ("688076", 2024, 2_682_316_631.52, 0.0, 88_265_182.53, 8_609_277.38,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10939637&stockid=688076"),
        # 司尔特2026年再次更正；2022保留2025年5月已公布的更正报表，不回首次值。
        ("002538", 2022, 5_297_379_808.74, 0.0, 32_873_997.55, 197_369_378.96,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=11127642&stockid=002538"),
        ("002538", 2023, 5_207_438_771.10, 0.0, 41_931_380.46, 188_664_450.34,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=11127637&stockid=002538"),
        ("002538", 2024, 5_312_438_226.74, 0.0, 30_821_776.87, 179_173_332.33,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=11127637&stockid=002538"),
        # 湘邮科技2026年4月更正2023/2024权益；2025半年报仍确认2024原报四科目。
        ("600476", 2023, 131_296_770.26, 0.0, 7_231_217.25, 7_677_981.75,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=9927613&stockid=600476"),
        ("600476", 2024, 141_380_192.25, 0.0, 6_937_875.99, 3_263_305.49,
         "https://static.cninfo.com.cn/finalpage/2025-08-27/1224581083.PDF"),
        # 神雾节能2026年更正同时影响少数权益和税资，不能只恢复归母利润减少额。
        ("000820", 2024, 222_932_175.44, 0.0, 26_684_197.04, 380_063.28,
         "https://static.cninfo.com.cn/finalpage/2025-04-29/1223385090.PDF"),
        # 苏交科2026年1月追溯更正2023/2024，原年报两列与更正公告前列一致。
        ("300284", 2023, 8_725_994_071.42, 0.0, 651_591_288.88, 12_280_814.78,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10875047&stockid=300284"),
        ("300284", 2024, 8_795_764_212.04, 0.0, 732_165_714.46, 1_647_837.68,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10875047&stockid=300284"),
        # 恒实科技2026年4月才更正2024年报；原年报与更正公告的调整前列一致。
        ("300513", 2024, 1_465_699_535.34, 0.0, 58_361_778.13, 6_461_681.46,
         "https://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESZ_STOCK/2025/2025-4/2025-04-29/11087678.PDF"),
        # 赢时胜2026年4月更正2024年报；原年报递延所得税负债期末为空，按0处理。
        ("300377", 2024, 2_542_311_171.39, 0.0, 30_592_818.80, 0.0,
         "https://disc.static.szse.cn/disc/disk03/finalpage/2025-04-28/9c2ab972-dd8a-4ebf-867d-be2fb096e0c5.PDF"),
        # 汉嘉设计2026年4月两次更正2024年报，恢复2025年4月原合并报表。
        ("300746", 2024, 1_034_438_940.67, 0.0, 72_028_036.40, 17_825_432.17,
         "https://static.cninfo.com.cn/finalpage/2025-04-24/1223241254.PDF"),
        # 中设股份2026年4月追溯更正2022—2024年；权益和递延税资须同时恢复。
        # 2022保留2023年租赁所得税政策重述，三个科目的政策调整在账面权益中抵消。
        ("002883", 2022, 730_565_521.31, 0.0, 39_312_294.75, 2_664_941.34,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10026276&stockid=002883"),
        ("002883", 2023, 775_023_475.43, 0.0, 47_599_130.19, 2_003_687.29,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10026276&stockid=002883"),
        ("002883", 2024, 677_862_154.45, 0.0, 36_090_534.22, 408_811.41,
         "https://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESZ_STOCK/2025/2025-4/2025-04-25/10956992.PDF"),
        # 东珠生态2026年4月追溯更正，以下使用原合并年报的完整四科目。
        ("603359", 2018, 2_662_592_232.07, 0.0, 14_220_820.00, 0.0,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=5209141&stockid=603359"),
        # 2019税资取2019年末余额，不误用2020年新收入准则调整后的年初数。
        ("603359", 2019, 3_001_771_166.33, 0.0, 15_031_282.45, 0.0,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=6078281&stockid=603359"),
        ("603359", 2020, 3_301_421_219.42, 0.0, 30_420_516.82, 0.0,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=7051264&stockid=603359"),
        ("603359", 2021, 3_705_010_957.03, 0.0, 53_507_477.95, 0.0,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=8134144&stockid=603359"),
        # 2022保留2023年租赁所得税政策比较列，政策调整在账面权益公式中完全抵消。
        ("603359", 2022, 3_709_919_758.48, 0.0, 69_421_055.38, 962_080.97,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10059956&stockid=603359"),
        ("603359", 2023, 3_377_079_160.24, 0.0, 125_258_151.21, 778_105.85,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10059956&stockid=603359"),
        ("603359", 2024, 2_733_960_641.13, 0.0, 125_091_464.59, 594_130.73,
         "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=11042226&stockid=603359"),
        # 传化智联2026年4月更正2022—2024年，权益和两项递延税均发生变化。
        # 2022保留2023年报比较列的政策及同一控制合并调整，不退回2022首次年报。
        ("002010", 2022, 18_460_992_062.22, 0.0, 350_585_821.09, 574_197_209.49,
         "https://static.cninfo.com.cn/finalpage/2024-04-23/1219739992.PDF"),
        ("002010", 2023, 18_973_008_334.14, 0.0, 397_721_332.05, 694_527_006.57,
         "https://static.cninfo.com.cn/finalpage/2024-04-23/1219739992.PDF"),
        ("002010", 2024, 18_835_594_423.61, 0.0, 279_792_425.97, 559_631_231.36,
         "https://static.cninfo.com.cn/finalpage/2025-04-22/1223185711.PDF"),
        # 庚星股份2026年4月更正2021—2024年；2022使用原合并年报的科目。
        # 2021保留2022年报期初列已包含的较早更正，不倒回2021首次年报。
        ("600753", 2021, 305_425_709.50, 0.0, 20_439_537.67, 0.0,
         "https://dataclouds.cninfo.com.cn/shgonggao/2023/2023-04-29/1f5d7296e5c811ed81d2fa163e26e5de.pdf"),
        ("600753", 2022, 320_890_670.55, 0.0, 18_476_092.70, 0.0,
         "https://dataclouds.cninfo.com.cn/shgonggao/2023/2023-04-29/1f5d7296e5c811ed81d2fa163e26e5de.pdf"),
        ("600753", 2023, 235_643_065.60, 0.0, 33_704_334.79, 14_239_501.39,
         "https://static.cninfo.com.cn/finalpage/2025-04-26/1223327065.PDF"),
        # 2024恢复后账面权益仍为负，后续按统一正账面比条件剔除。
        ("600753", 2024, -3_376_857.69, 0.0, 4_711_743.86, 7_656_718.45,
         "https://static.cninfo.com.cn/finalpage/2025-04-26/1223327065.PDF"),
        # 因赛集团2026年4月更正2024年，原报与更正公告前列的权益及税资一致。
        ("300781", 2024, 735_200_390.39, 0.0, 5_016_959.19, 1_698_139.42,
         "https://static.cninfo.com.cn/finalpage/2025-04-08/1223021739.PDF"),
        # 迪瑞医疗2026年更正2023年报，使用原合并总权益，不能误用归母权益。
        ("300396", 2023, 2_039_311_406.64, 0.0, 13_525_366.28, 27_097_100.79,
         "https://static.cninfo.com.cn/finalpage/2024-04-19/1219662428.PDF"),
        # 2024四科目保留2025年半年报比较列，与2025年原年报一致。
        ("300396", 2024, 2_048_769_327.92, 0.0, 16_136_487.80, 25_381_443.06,
         "https://static.cninfo.com.cn/finalpage/2025-08-26/1224563804.PDF"),
        # 瀚川智能2026年4月追溯更正2022—2024年，恢复2025年底前已披露科目。
        # 2022保留2023修订年报比较列，2023及2024使用2025年7月修订年报。
        ("688022", 2022, 1_022_726_473.84, 0.0, 39_322_615.55, 14_068_157.35,
         "https://dataclouds.cninfo.com.cn/shgonggao/2024/2024-07-09/8cd7d6133d2411ef8945fa163e26e5de.pdf"),
        ("688022", 2023, 1_766_163_349.53, 0.0, 71_449_111.94, 11_705_850.25,
         "https://static.cninfo.com.cn/finalpage/2025-07-12/1224146481.PDF"),
        ("688022", 2024, 642_110_086.18, 0.0, 70_893_467.86, 8_229_144.24,
         "https://static.cninfo.com.cn/finalpage/2025-07-12/1224146481.PDF"),
        # 嘉事堂2026年3月更正2017—2024年报，以下四科目来自更正前合并年报。
        # 总权益包含少数股东权益；原报及随后比较列的两项递延税未因此更正。
        ("002462", 2017, 3_197_177_191.41, 0.0, 16_824_716.50, 120_520.22,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=5056272&stockid=002462"),
        ("002462", 2018, 3_732_704_925.65, 0.0, 21_333_481.72, 344_657.00,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=6178871&stockid=002462"),
        ("002462", 2019, 4_922_864_497.71, 0.0, 31_303_282.22, 11_938_194.39,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=6178871&stockid=002462"),
        ("002462", 2020, 5_324_031_026.96, 0.0, 56_362_381.22, 37_258_701.33,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=7930746&stockid=002462"),
        ("002462", 2021, 5_656_557_639.51, 0.0, 52_565_900.85, 29_945_248.53,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=9033726&stockid=002462"),
        ("002462", 2022, 5_921_604_115.16, 0.0, 68_149_359.56, 11_240_512.97,
         "https://static.cninfo.com.cn/finalpage/2024-04-29/1219868890.PDF"),
        ("002462", 2023, 6_091_512_685.68, 0.0, 92_970_994.94, 4_301_177.67,
         "https://static.cninfo.com.cn/finalpage/2024-04-29/1219868890.PDF"),
        ("002462", 2024, 6_064_840_635.68, 0.0, 73_030_743.21, 759_140.26,
         "https://disc.static.szse.cn/disc/disk03/finalpage/2025-08-30/6de93b40-89dd-4d73-89fb-d6ce6e4044b7.PDF"),
        # 新动力2026年8月再次追溯更正2022—2024年，三个补充科目原表均为空。
        # 2022保留2025年4月已生效更正，不使用2022首次年报的权益。
        ("300152", 2022, 396_766_087.09, 0.0, 0.0, 0.0,
         "https://static.cninfo.com.cn/finalpage/2026-08-28/1225518583.PDF"),
        ("300152", 2023, 321_395_514.70, 0.0, 0.0, 0.0,
         "https://static.cninfo.com.cn/finalpage/2025-06-23/1223953081.PDF"),
        ("300152", 2024, 240_584_095.68, 0.0, 0.0, 0.0,
         "https://static.cninfo.com.cn/finalpage/2025-06-23/1223953081.PDF"),
        # 宝馨科技保留2023年准则解释16号重述，权益与递延税调整在本脚本公式中抵消。
        # 2024年差错更正仅影响现金流量等披露，未再更改资产负债表四科目。
        ("002514", 2022, 1_425_300_508.28, 0.0, 80_068_345.88, 2_077_930.83,
         "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10174281&stockid=002514"),
        # 宝胜股份2026年才更正2014、2015权益；原年报与2026年补充公告前列相互验证。
        # 2026-03-14最初公告有年份错表，不能将其2016期末数误用于2015年。
        ("600973", 2014, 2_137_000_610.12, 0.0, 54_325_623.88, 774_843.40,
         "https://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESH_STOCK/2015/2015-3/2015-03-31/1686892.PDF"),
        ("600973", 2015, 2_302_038_809.08, 0.0, 68_724_051.67, 774_843.40,
         "https://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESH_STOCK/2016/2016-4/2016-04-23/2398935.PDF"),
    ]
    for stock_code, fiscal_year, total_equity, preferred_equity, deferred_tax_assets, deferred_tax_liabilities, source in historical_statement_versions:
        matched_rows = financials["Stkcd"].eq(stock_code) & financials["fiscal_year"].eq(fiscal_year)
        financials.loc[matched_rows, account_columns] = [total_equity, preferred_equity, deferred_tax_assets, deferred_tax_liabilities]
        financials.loc[matched_rows, "statement_version_source"] = source
    stage_samples["存在上年合并年报"] = financials[["Stkcd", "fiscal_year"]].rename(columns={"fiscal_year": "formation_year"})
    stage_samples["存在上年合并年报"]["formation_year"] += 1
    financials["supplement_missing_count"] = financials[account_columns[1:]].isna().sum(axis=1)
    # 上列记录补充科目的缺失个数；不把缺失的所有者权益补成0。
    financials["be"] = (
        financials["A003000000"] - financials["A003112101"].fillna(0)
        + financials["A002208000"].fillna(0) - financials["A001222000"].fillna(0)
    )  # be（账面权益），单位元。
    book_to_market_data = financials.merge(
        december_data[["Stkcd", "fiscal_year", "me_dec", "dec_date"]],
        on=["Stkcd", "fiscal_year"], how="inner", validate="one_to_one",
    )
    book_to_market_data["bm"] = book_to_market_data["be"] / book_to_market_data["me_dec"]  # bm（原始账面市值比）。
    book_to_market_data = book_to_market_data.loc[np.isfinite(book_to_market_data["bm"]) & book_to_market_data["bm"].gt(0)].copy()
    book_to_market_data["formation_year"] = book_to_market_data["fiscal_year"] + 1
    stage_samples["账面市值比为正"] = book_to_market_data[["Stkcd", "formation_year"]].copy()

    # ========================================================================
    # 5. 匹配原始行业历史：上年末时点；通过曾用代码补充更换代码公司的历史
    # ========================================================================
    print("5/7 匹配上年末行业，并计算行业调整账面比……", flush=True)
    industries = pd.read_csv(
        industry_path, encoding="utf-8-sig", dtype="string",
        usecols=["Symbol", "IndustryClassificationID", "ImplementDate", "IndustryCode"],
    ).rename(columns={"Symbol": "Stkcd", "IndustryClassificationID": "industry_standard"})
    # industry_standard（行业分类标准）；IndustryCode（行业代码）。
    industries = industries.loc[industries["industry_standard"].isin(["P0207", "P0221"])].copy()
    industries["Stkcd"] = normalize_stock_codes(industries["Stkcd"])
    industries["IndustryCode"] = industries["IndustryCode"].str.strip().str.upper()
    industries["ImplementDate"] = pd.to_datetime(industries["ImplementDate"], errors="raise")
    industries = industries.dropna(subset=["ImplementDate", "IndustryCode"])
    industries = industries.loc[industries["IndustryCode"].ne("")].drop_duplicates().copy()
    industry_keys = ["Stkcd", "industry_standard", "ImplementDate"]
    check_unique_key(industries, industry_keys, "原始行业历史")
    industries["industry_source"] = "原始行业历史"  # 行业信息来源。

    inherited_industry_records = code_aliases.merge(industries, left_on="FormerCode", right_on="Stkcd", how="inner")
    inherited_industry_records["Stkcd"] = inherited_industry_records["new_code"]
    inherited_industry_records["industry_source"] = "原始行业历史经曾用代码衔接"
    # 同代码同标准同日若已有本身记录，优先本身记录；继承只用于补历史。
    industries = pd.concat([industries, inherited_industry_records[industries.columns]], ignore_index=True)
    industries = industries.drop_duplicates(industry_keys, keep="first")

    book_to_market_data["industry_standard"] = pd.Series(
        np.where(book_to_market_data["formation_year"].lt(2023), "P0207", "P0221"),
        index=book_to_market_data.index, dtype="string",
    )
    book_to_market_data = pd.merge_asof(
        book_to_market_data.sort_values(["Accper", "Stkcd", "industry_standard"]),
        industries.sort_values(["ImplementDate", "Stkcd", "industry_standard"]),
        left_on="Accper", right_on="ImplementDate",
        by=["Stkcd", "industry_standard"], direction="backward", allow_exact_matches=True,
    )
    print(f"  正账面比股票年中缺行业：{book_to_market_data['IndustryCode'].isna().sum():,}", flush=True)
    book_to_market_data = book_to_market_data.dropna(subset=["IndustryCode"]).copy()
    stage_samples["匹配上年末行业"] = book_to_market_data[["Stkcd", "formation_year"]].copy()
    industry_groups = book_to_market_data.groupby(["formation_year", "IndustryCode"])
    book_to_market_data["industry_firms"] = industry_groups["Stkcd"].transform("nunique")  # 行业内公司数。
    book_to_market_data["industry_bm"] = industry_groups["be"].transform("sum") / industry_groups["me_dec"].transform("sum")
    # industry_bm（行业基准账面比）为总账面权益除以总流通市值。
    book_to_market_data["log_bm_gap"] = np.log(book_to_market_data["bm"]) - np.log(book_to_market_data["industry_bm"])
    book_to_market_data["industry_std"] = (
        book_to_market_data.groupby(["formation_year", "IndustryCode"])["log_bm_gap"]
        .transform("std", ddof=industry_std_ddof)
    )  # 行业内对数偏差标准差。
    book_to_market_data = book_to_market_data.loc[
        book_to_market_data["industry_firms"].gt(1) & book_to_market_data["industry_std"].gt(0)
    ].copy()
    book_to_market_data["bm_adj"] = book_to_market_data["log_bm_gap"] / book_to_market_data["industry_std"]
    # bm_adj（行业调整账面市值比）；先在账面比有效样本内计算，再取三指标交集。
    stage_samples["行业调整有效"] = book_to_market_data.loc[np.isfinite(book_to_market_data["bm_adj"]), ["Stkcd", "formation_year"]].copy()

    # ========================================================================
    # 6. 三指标交集 → 逐级等数量五分组 → 三位组合代码
    # ========================================================================
    print("6/7 形成年度特征表并逐级分组……", flush=True)
    labels = june_data[["Stkcd", "formation_year", "formation_date", "Markettype", "size_june", "june_filling"]].merge(
        momentum_data, on=["Stkcd", "formation_year"], how="inner", validate="one_to_one",
    ).merge(
        book_to_market_data[["Stkcd", "formation_year", "Accper", "dec_date", "be", "me_dec", "bm", "bm_adj",
                "IndustryCode", "industry_standard", "industry_source", "ImplementDate",
                "industry_firms", "industry_bm", "industry_std", "supplement_missing_count",
                "statement_version_source"]],
        on=["Stkcd", "formation_year"], how="inner", validate="one_to_one",
    )
    finite_values = np.isfinite(labels[["size_june", "bm", "bm_adj", "mom"]]).all(axis=1)
    labels = labels.loc[finite_values].reset_index(drop=True)
    if labels.empty:
        raise ValueError("三个指标交集为空，请检查输入日期与行业匹配")
    check_unique_key(labels, ["Stkcd", "formation_year"], "年度排序特征表")
    labels["MarketValue"] = assign_descending_quintiles(labels, ["formation_year"], "size_june")
    labels["BooktoMarket"] = assign_descending_quintiles(labels, ["formation_year", "MarketValue"], "bm_adj")
    labels["Momentum"] = assign_descending_quintiles(labels, ["formation_year", "MarketValue", "BooktoMarket"], "mom")
    # 上面三列依次是规模组、账面比组、动量组，都是1高、5低。
    labels["dgtw_port"] = (
        labels["MarketValue"].astype(str) + labels["BooktoMarket"].astype(str) + labels["Momentum"].astype(str)
    )  # 三位组合代码，例如111、235、555。
    labels["portfolioNumber"] = (
        (labels["MarketValue"].astype("int16") - 1) * 25
        + (labels["BooktoMarket"].astype("int16") - 1) * 5 + labels["Momentum"].astype("int16")
    )  # 便于合并的1至125编号；与原项目低值为1的编号含义不同，不能混用。
    labels["IsNotBSE"] = is_not_bse  # 是否剔除北交所。
    labels["dateBeg"] = pd.to_datetime(labels["formation_year"].astype(str) + "-07-01")
    labels["dateEnd"] = pd.to_datetime((labels["formation_year"] + 1).astype(str) + "-06-30")
    labels = labels.sort_values(["formation_year", "Stkcd"]).reset_index(drop=True)

    # ========================================================================
    # 7. 至此才读取官方结果；比较覆盖率、各层标签、条件匹配及组合最大值
    # ========================================================================
    print("7/7 与官方结果比较……", flush=True)
    official_labels = pd.read_csv(official_assignment_path, encoding="utf-8-sig", dtype={"Symbol": "string"})
    official_labels = official_labels.rename(columns={"Symbol": "Stkcd", "TradingYear": "formation_year"})
    official_labels["Stkcd"] = normalize_stock_codes(official_labels["Stkcd"])
    official_labels = official_labels.loc[
        official_labels["IsNotBSE"].eq(is_not_bse)
        & official_labels["formation_year"].between(start_formation_year, end_formation_year)
    ].copy()
    check_unique_key(official_labels, ["Stkcd", "formation_year"], "官方分配表所选口径")
    if official_labels.empty:
        raise ValueError("官方分配表在所选年份及北交所口径下没有可比较记录")
    # 不把官方尚未提供的年份放进比较分母，重建标签本身仍完整保留。
    comparable_years = official_labels["formation_year"].unique()
    comparison = labels.loc[labels["formation_year"].isin(comparable_years)].merge(
        official_labels[["Stkcd", "formation_year", "ShortName", "MarketValue", "BooktoMarket", "Momentum", "IndustryCode"]],
        on=["Stkcd", "formation_year"], how="outer", validate="one_to_one",
        suffixes=("_重建", "_官方"), indicator="样本来源",
    )
    in_both = comparison["样本来源"].eq("both")
    for field, match_flag in [("MarketValue", "规模一致"), ("BooktoMarket", "账面比一致"), ("Momentum", "动量一致")]:
        comparison[match_flag] = (
            comparison[f"{field}_重建"].eq(comparison[f"{field}_官方"])
            .astype("boolean").where(in_both, pd.NA)
        )
    comparison["行业一致"] = (
        comparison["IndustryCode_重建"].eq(comparison["IndustryCode_官方"])
        .astype("boolean").where(in_both, pd.NA)
    )
    comparison["全部一致"] = (comparison["规模一致"] & comparison["账面比一致"] & comparison["动量一致"]).where(in_both, pd.NA)
    # 此时才对差异股票附加构造阶段标志，绝不将官方入选范围用于前面的分组。
    comparison_index = pd.MultiIndex.from_frame(comparison[["Stkcd", "formation_year"]])
    for stage, stock_year_keys in stage_samples.items():
        comparison[stage] = comparison_index.isin(pd.MultiIndex.from_frame(stock_year_keys))
    comparison["首个未通过步骤"] = np.select(
        [~comparison["存在六月末市值"], ~comparison["存在动量月份"], ~comparison["交易月份达标"],
         ~comparison["存在上年合并年报"], ~comparison["存在上年末市值"], ~comparison["账面市值比为正"],
         ~comparison["匹配上年末行业"], ~comparison["行业调整有效"]],
        ["缺六月末有效流通市值", "缺动量窗口记录", "有交易月份不足", "缺上年合并年报",
         "缺上年末有效流通市值", "账面市值比非正或缺失", "缺上年末行业", "行业公司数或标准差无效"],
        default=np.where(comparison["样本来源"].eq("right_only"), "排序指标非有限或合并缺失", "已生成标签"),
    )
    summary_rows = [summarize_matches(subtable, int(year)) for year, subtable in comparison.groupby("formation_year", sort=True)]
    summary_rows.append(summarize_matches(comparison, "合计"))
    annual_match_summary = pd.DataFrame(summary_rows)

    portfolio_keys = ["formation_year", "IsNotBSE", "MarketValue", "BooktoMarket", "Momentum"]
    rebuilt_maxima = labels.groupby(portfolio_keys, as_index=False).agg(
        MarketValueMax=("size_june", "max"),  # 组合最大流通市值。
        AdjbtmMax=("bm_adj", "max"),  # 组合最大行业调整账面比。
        BtmMax=("bm", "max"),  # 组合最大原始账面比。
        MomMax=("mom", "max"),  # 组合最大动量。
        重建组合股票数=("Stkcd", "size"),
    )
    official_maxima = pd.read_csv(official_maxima_path, encoding="utf-8-sig").rename(columns={"TradingYear": "formation_year"})
    official_maxima = official_maxima.loc[
        official_maxima["IsNotBSE"].eq(is_not_bse)
        & official_maxima["formation_year"].between(start_formation_year, end_formation_year)
    ].copy()
    check_unique_key(official_maxima, portfolio_keys, "官方组合最大值表")
    maxima_comparison = rebuilt_maxima.loc[rebuilt_maxima["formation_year"].isin(official_maxima["formation_year"].unique())].merge(
        official_maxima, on=portfolio_keys, how="outer", validate="one_to_one",
        suffixes=("_重建", "_官方"), indicator="组合来源",
    )
    for field in ["MarketValueMax", "AdjbtmMax", "BtmMax", "MomMax"]:
        maxima_comparison[f"{field}_差值"] = maxima_comparison[f"{field}_重建"] - maxima_comparison[f"{field}_官方"]

    # 只有运行到此处才创建结果目录、写入结果，不覆盖项目原来的标签文件。
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{start_formation_year}_{end_formation_year}_剔北{is_not_bse}"
    labels.to_csv(output_dir / f"重建形成期标签_{suffix}.csv", index=False, encoding="utf-8-sig")
    comparison.sort_values(["formation_year", "Stkcd"]).to_csv(
        output_dir / f"逐股标签对照_{suffix}.csv", index=False, encoding="utf-8-sig",
    )
    annual_match_summary.to_csv(output_dir / f"年度匹配率_{suffix}.csv", index=False, encoding="utf-8-sig")
    maxima_comparison.to_csv(output_dir / f"组合最大值对照_{suffix}.csv", index=False, encoding="utf-8-sig")

    print("\n年度匹配结果，比例列以0至1表示：")
    print(annual_match_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n重建标签共{len(labels):,}行；官方仅在最后用于比较，没有参与构造。")
    print(f"输出目录：{output_dir}")
    print("参考价重建仍受日收益精度、历史股本和财报版本影响；以本次对照结果为准。")


if __name__ == "__main__":
    main()
