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
原始目录 = ROOT / "input/CSMAR"
输出目录 = ROOT / "中间结果/02_变量构造结果/DGTW/CSMAR标签复现"

起始形成年度 = 2000
结束形成年度 = 2025
剔除北交所 = True
最少有交易月份 = 6
行业标准差自由度 = 1  # 1为样本标准差；0为总体标准差。
日表分块行数 = 300_000
理论参考价容差 = 0.000005  # 日复利因子相对容差；兼容原始日收益的6位舍入。
错位前价容差 = 0.000001  # 仅纠正能由倒数第二个真实收盘价解释的旧日收益。

月表路径 = 原始目录 / "月个股回报率文件2000-01至2025-12/TRD_Mnth.csv"
日表目录 = 原始目录 / "个股日度收益率（升级测试）1995-01-01至2026"
财报路径 = 原始目录 / "资产负债表1990-12-31至2026-06-30/FS_Combas.csv"
行业路径 = 原始目录 / "上市公司行业分类表1990年12月10日—2026年4月20日/STK_INDUSTRYCLASS.csv"
公司路径 = 原始目录 / "公司文件(1990-12-10至2026-08-21)/TRD_Co.csv"
股本路径 = 原始目录 / "专题研究系列-事件研究/股本变动事件表1990-12-07 至 2026-09-22/ER_Structure.csv"
红利路径 = 原始目录 / "专题研究系列-事件研究/红利分配事件表1991-02-10 至 2026-09-11/ER_ProfitSharing.csv"
配股路径 = 原始目录 / "专题研究系列-事件研究/配股事件表1990-03-01 至 2025-06-26/ER_RightsIssue.csv"
官方目录 = 原始目录 / "DGTW（2000至2025）"
官方分配路径 = 官方目录 / "DGTW股票分配表/STK_MKT_DGTWASSINGMENTS.csv"
官方最大值路径 = 官方目录 / "DGTW分位数表2000至2025/STK_MKT_DGTWBREAKPOINTS.csv"


# ============================================================================
# 工具函数：仅保留代码标准化、唯一键检查、五分组与比较汇总
# ============================================================================

def 标准化代码(列):
    结果 = 列.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if not 结果.str.fullmatch(r"\d{1,6}", na=False).all():
        raise ValueError("股票代码存在空值或非数字，请检查原始表")
    return 结果.str.zfill(6)


def 检查唯一键(表, 键, 表名):
    if 表[键].isna().any().any():
        raise ValueError(f"{表名}的关键字段存在空值：{键}")
    重复 = 表.duplicated(键, keep=False)
    if 重复.any():
        示例 = 表.loc[重复, 键].head(10).to_string(index=False)
        raise ValueError(f"{表名}的唯一键重复，不能静默保留最后一行：\n{示例}")


def 降序五分组(表, 上层键, 排序字段):
    组号 = pd.Series(pd.NA, index=表.index, dtype="Int8")
    for _, 子表 in 表.groupby(上层键, sort=False):
        排序后 = 子表.sort_values([排序字段, "Stkcd"], ascending=[False, True])
        for 序号, 索引 in enumerate(np.array_split(排序后.index.to_numpy(), 5), start=1):
            组号.loc[索引] = 序号
    if 组号.isna().any():
        raise ValueError(f"{排序字段}分组失败，存在未分配组号的股票")
    return 组号


def 汇总匹配(表, 年份):
    共有 = 表["样本来源"].eq("both")  # 两表共有。
    有官方 = ~表["样本来源"].eq("left_only")  # 非仅重建。
    有重建 = ~表["样本来源"].eq("right_only")  # 非仅官方。
    规模同 = 表["规模一致"].fillna(False)
    账面同 = 表["账面比一致"].fillna(False)
    动量同 = 表["动量一致"].fillna(False)
    全同 = 表["全部一致"].fillna(False)

    def 比率(分子, 分母):
        return float(分子 / 分母) if 分母 else np.nan

    return {
        "形成年度": 年份,
        "官方股票数": int(有官方.sum()),
        "重建股票数": int(有重建.sum()),
        "共有股票数": int(共有.sum()),
        "仅官方股票数": int(表["样本来源"].eq("right_only").sum()),
        "仅重建股票数": int(表["样本来源"].eq("left_only").sum()),
        "官方样本覆盖率": 比率(共有.sum(), 有官方.sum()),
        "重建样本被官方覆盖率": 比率(共有.sum(), 有重建.sum()),
        "共有样本规模一致率": 比率(规模同.sum(), 共有.sum()),
        "共有样本账面比一致率": 比率(账面同.sum(), 共有.sum()),
        "共有样本动量一致率": 比率(动量同.sum(), 共有.sum()),
        "共有样本全部一致率": 比率(全同.sum(), 共有.sum()),
        "规模一致时账面比一致率": 比率((规模同 & 账面同).sum(), 规模同.sum()),
        "前两维一致时动量一致率": 比率(全同.sum(), (规模同 & 账面同).sum()),
        "全部一致股票数": int(全同.sum()),
        "全部一致数占官方比例": 比率(全同.sum(), 有官方.sum()),
    }


def main():
    # main（主流程）：所有结果只在直接运行脚本时生成。
    if not 1996 <= 起始形成年度 <= 结束形成年度 <= 2025:
        raise ValueError("当前原始文件支持的形成年度设置范围为1996至2025")
    if not 1 <= 最少有交易月份 <= 12 or 行业标准差自由度 not in [0, 1]:
        raise ValueError("月份门槛应为1至12，标准差自由度应为0或1")
    if not np.isfinite(理论参考价容差) or 理论参考价容差 < 0:
        raise ValueError("理论参考价容差应为非负有限数")
    if not np.isfinite(错位前价容差) or 错位前价容差 < 0:
        raise ValueError("错位前价容差应为非负有限数")
    for 路径 in [月表路径, 日表目录, 财报路径, 行业路径, 公司路径, 股本路径,
                 红利路径, 配股路径, 官方分配路径, 官方最大值路径]:
        if not 路径.exists():
            raise FileNotFoundError(f"缺少输入：{路径}")

    市场类型 = [1, 4, 16, 32] if 剔除北交所 else [1, 4, 16, 32, 64]
    市值读取市场 = 市场类型 + [2, 8]  # B股只用于还原市值，不进入A股排序样本。
    北交所口径 = int(剔除北交所)
    最早月份 = pd.Timestamp(起始形成年度 - 1, 7, 1)
    最晚月份 = pd.Timestamp(结束形成年度, 6, 1)
    最早日 = f"{起始形成年度 - 1}-01-01"  # 包含上年12月市值所需数据。
    最晚日 = f"{结束形成年度}-12-31"
    阶段样本 = {}  # 仅保存各构造阶段的股票年键，供最后解释样本差异。

    公司 = pd.read_csv(
        公司路径, encoding="utf-8-sig", dtype="string",
        usecols=["Stkcd", "FormerCode", "Listdt"],
    ).drop_duplicates()
    公司["Stkcd"] = 标准化代码(公司["Stkcd"])
    检查唯一键(公司, ["Stkcd"], "公司基本资料")
    # FormerCode（曾用代码）：仅接受明确、单一的六位旧代码。
    公司["FormerCode"] = 公司["FormerCode"].str.strip().str.extract(r"^(\d{6})$", expand=False)
    公司["Listdt"] = pd.to_datetime(公司["Listdt"], errors="coerce")  # 新代码上市日期。
    别名 = 公司.dropna(subset=["FormerCode"]).rename(columns={"Stkcd": "new_code"})
    换码事件 = 别名.loc[别名["Listdt"].between(最早月份, 最晚月份 + pd.offsets.MonthEnd(0))].copy()
    换码事件["change_month"] = 换码事件["Listdt"].dt.strftime("%Y-%m")  # 换码月份。
    换码日键 = pd.concat([
        换码事件[["new_code", "change_month"]].rename(columns={"new_code": "Stkcd"}),
        换码事件[["FormerCode", "change_month"]].rename(columns={"FormerCode": "Stkcd"}),
    ], ignore_index=True).drop_duplicates()
    换码索引 = pd.MultiIndex.from_frame(换码日键.rename(columns={"change_month": "Trdmnt"}))

    # ========================================================================
    # 1. 读取原始月收益；不使用项目已清洗研究样本
    # ========================================================================
    print("1/7 读取原始月收益……", flush=True)
    # Mretwd为含红利月收益，Ndaytrd为月实际交易天数，Markettype为市场类型。
    月表 = pd.read_csv(
        月表路径, encoding="utf-8-sig",
        usecols=["Stkcd", "Trdmnt", "Mretwd", "Ndaytrd", "Markettype", "Msmvosd", "Mclsprc"],
        dtype={"Stkcd": "string"}, parse_dates=["Trdmnt"],
    )
    月表 = 月表.loc[月表["Markettype"].isin(市场类型)].copy()
    月表["Stkcd"] = 标准化代码(月表["Stkcd"])
    月表 = 月表.drop_duplicates()
    检查唯一键(月表, ["Stkcd", "Trdmnt"], "原始月收益表")
    if 月表.empty:
        raise ValueError("原始月收益表没有所选市场股票")
    原始月表首月 = 月表["Trdmnt"].min()
    月表["月收益来自日表"] = 0  # 后续包含早年补数、日月冲突纠正及必要的换码衔接。
    月表["缺失日收益数"] = 0
    月表["跨代码补充月份"] = 0
    月表 = 月表.loc[月表["Trdmnt"].between(最早月份, 最晚月份)].copy()
    六月月表市值 = 月表.loc[
        月表["Trdmnt"].dt.month.eq(6) & 月表["Ndaytrd"].gt(0) & 月表["Msmvosd"].gt(0),
        ["Stkcd", "Trdmnt", "Msmvosd", "Mclsprc"],
    ].rename(columns={"Msmvosd": "monthly_me_a", "Mclsprc": "monthly_price"})
    六月月表市值["monthly_me_a"] *= 1000  # 月表A股流通市值，转为元。
    月表 = 月表.drop(columns=["Msmvosd", "Mclsprc"])

    # 红利字段均为每10股；配股比例已经是每股比例，不再除以10。
    # 老记录存在股东对象标记不一致；年度Q0103/Q0105另列候选，由日收益验证。
    红利 = pd.read_csv(
        红利路径, encoding="utf-8-sig", dtype={"Symbol": "string"},
        usecols=["Symbol", "ExDividendDate", "RecordDate", "Object", "Term", "Currency",
                 "BonusRatio", "ConversionRatio", "DividentBT", "DistributionBaseShares", "AddShares"],
    ).rename(columns={"Symbol": "Stkcd"})
    红利["Stkcd"] = 标准化代码(红利["Stkcd"])
    红利["Trddt"] = pd.to_datetime(红利["ExDividendDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    # 先统计全部对象和币种的方案，防止筛选后漏掉同日另一项分配。
    同日原始方案数 = 红利.drop_duplicates([
        "Stkcd", "Trddt", "RecordDate", "Object", "Term", "Currency", "BonusRatio", "ConversionRatio",
        "DividentBT", "DistributionBaseShares", "AddShares",
    ]).groupby(["Stkcd", "Trddt"]).size()
    红利 = 红利.loc[
        红利["Trddt"].between(最早日, 最晚日)
        & (红利["Currency"].isna() | 红利["Currency"].isin(["人民币元", "人民币", "CNY"]))
    ].copy()
    红利科目 = ["BonusRatio", "ConversionRatio", "DividentBT"]
    红利[红利科目] = 红利[红利科目].fillna(0) / 10
    普通对象 = 红利["Object"].isin(["Q0101", "Q0102", "Q0104", "Q0106"])
    历史年度对象 = 红利["Term"].eq("P2702") & 红利["Object"].isin(["Q0103", "Q0105"])
    方案表 = []
    for 后缀, 纳入 in [("_o", 普通对象), ("", 普通对象 | 历史年度对象)]:
        子表 = 红利.loc[纳入].drop_duplicates(["Stkcd", "Trddt", "RecordDate"] + 红利科目)
        子表 = 子表.groupby(["Stkcd", "Trddt", "RecordDate"], dropna=False, as_index=False)[红利科目].max()
        子表["mult"] = 1 + 子表["BonusRatio"] + 子表["ConversionRatio"]
        子表 = 子表.groupby(["Stkcd", "Trddt"], as_index=False).agg(
            cash=("DividentBT", "sum"), mult=("mult", "prod"),
        ).rename(columns={"cash": "cash" + 后缀, "mult": "mult" + 后缀})
        方案表.append(子表)
    配股 = pd.read_csv(
        配股路径, encoding="utf-8-sig", dtype={"Symbol": "string"},
        usecols=["Symbol", "ExRightDate", "ProjectActualize", "SecurityType", "PlacingRatio",
                 "ShareHoldersAllotmentRatio", "Price"],
    ).rename(columns={"Symbol": "Stkcd"})
    配股["Stkcd"] = 标准化代码(配股["Stkcd"])
    配股["Trddt"] = pd.to_datetime(配股["ExRightDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    配股 = 配股.loc[
        配股["ProjectActualize"].eq("P6301") & 配股["SecurityType"].eq("A")
        & 配股["Trddt"].between(最早日, 最晚日)
    ].drop_duplicates(["Stkcd", "Trddt", "PlacingRatio", "ShareHoldersAllotmentRatio", "Price"]).copy()
    # 两个原始比例字段偶有差异，分别计算理论候选，再按日观察因子核验。
    配股["alt_ratio"] = 配股["ShareHoldersAllotmentRatio"].fillna(配股["PlacingRatio"])
    配股["rights_cost"] = 配股["PlacingRatio"] * 配股["Price"]
    配股["alt_cost"] = 配股["alt_ratio"] * 配股["Price"]
    配股 = 配股.groupby(["Stkcd", "Trddt"], as_index=False).agg(
        rights_ratio=("PlacingRatio", "sum"), rights_cost=("rights_cost", "sum"),
        alt_ratio=("alt_ratio", "sum"), alt_cost=("alt_cost", "sum"),
    )
    普通方案 = 方案表[0].merge(
        配股[["Stkcd", "Trddt", "rights_ratio", "rights_cost"]],
        on=["Stkcd", "Trddt"], how="outer", validate="one_to_one",
    )
    普通方案[["cash_o", "rights_ratio", "rights_cost"]] = 普通方案[["cash_o", "rights_ratio", "rights_cost"]].fillna(0)
    普通方案["mult_o"] = 普通方案["mult_o"].fillna(1)
    普通方案 = 普通方案.rename(columns={"rights_ratio": "ratio_o", "rights_cost": "cost_o"})
    扩展方案 = 方案表[1].merge(配股, on=["Stkcd", "Trddt"], how="outer", validate="one_to_one")
    扩展方案[["cash", "rights_ratio", "rights_cost", "alt_ratio", "alt_cost"]] = 扩展方案[
        ["cash", "rights_ratio", "rights_cost", "alt_ratio", "alt_cost"]
    ].fillna(0)
    扩展方案["mult"] = 扩展方案["mult"].fillna(1)
    除权事件 = 扩展方案.merge(普通方案, on=["Stkcd", "Trddt"], how="outer", validate="one_to_one")
    除权事件["has_plan"] = 除权事件["mult"].notna() | 除权事件["mult_o"].notna()
    除权事件["legacy_cash"] = (
        除权事件["cash"].gt(0) & 除权事件["cash_o"].isna()
        & 除权事件["mult"].eq(1) & 除权事件["rights_ratio"].eq(0)
    )
    特殊方案 = 红利.loc[
        红利["Object"].eq("Q0105") & 红利["Term"].eq("P2799")
        & 红利["DividentBT"].eq(0) & (红利["BonusRatio"] + 红利["ConversionRatio"]).gt(0),
        ["Stkcd", "Trddt"],
    ].drop_duplicates()
    特殊方案["special"] = True
    除权事件 = 除权事件.merge(特殊方案, on=["Stkcd", "Trddt"], how="outer", validate="one_to_one")
    日内股本 = pd.read_csv(
        股本路径, encoding="utf-8-sig", dtype={"Symbol": "string"},
        usecols=["Symbol", "ChangeDate", "EventID", "ChangeReasonID", "Total"],
    ).rename(columns={"Symbol": "Stkcd", "ChangeDate": "Trddt"})
    日内股本["Stkcd"] = 标准化代码(日内股本["Stkcd"])
    日内股本 = 日内股本.sort_values(["Stkcd", "Trddt", "EventID"])

    # 两段同日除权但登记日先后不同，不能因观察收益漏掉首段就舍弃完整方案。
    # 只接受先流通股东股改、后全体年度纯转增，并用原始总股本核对两段衔接。
    方案字段 = ["Stkcd", "Trddt", "RecordDate", "Object", "Term"] + 红利科目 + [
        "DistributionBaseShares", "AddShares",
    ]
    原始方案 = 红利[方案字段].drop_duplicates()
    纯转增 = (
        原始方案["ConversionRatio"].gt(0) & 原始方案["BonusRatio"].eq(0)
        & 原始方案["DividentBT"].eq(0)
    )
    先段 = 原始方案.loc[
        纯转增 & 原始方案["Object"].eq("Q0104") & 原始方案["Term"].eq("P2799")
    ]
    后段 = 原始方案.loc[
        纯转增 & 原始方案["Object"].eq("Q0101") & 原始方案["Term"].eq("P2702")
    ]
    两段候选 = 先段.merge(后段, on=["Stkcd", "Trddt"], suffixes=("_先", "_后"))
    两段候选 = 两段候选.loc[
        (pd.to_datetime(两段候选["RecordDate_先"], errors="coerce")
         < pd.to_datetime(两段候选["RecordDate_后"], errors="coerce"))
        & (pd.to_datetime(两段候选["RecordDate_后"], errors="coerce")
           < pd.to_datetime(两段候选["Trddt"], errors="coerce"))
    ]
    两段补全 = []
    for _, 方案 in 两段候选.iterrows():
        股票, 除权日 = 方案["Stkcd"], 方案["Trddt"]
        同日配股 = 配股.loc[配股["Stkcd"].eq(股票) & 配股["Trddt"].eq(除权日)]
        同日股本 = 日内股本.loc[日内股本["Stkcd"].eq(股票) & 日内股本["Trddt"].eq(除权日)]
        if 同日原始方案数.get((股票, 除权日), 0) != 2 or not 同日配股.empty:
            continue
        if not 同日股本["ChangeReasonID"].str.contains(r"(?:^|,)P0712(?:,|$)", na=False).any():
            continue
        前次股本 = 日内股本.loc[日内股本["Stkcd"].eq(股票) & 日内股本["Trddt"].lt(除权日)].tail(1)
        if 前次股本.empty:
            continue
        变动前总股本, 变动后总股本 = 前次股本["Total"].iloc[0], 同日股本["Total"].iloc[-1]
        基数先, 新增先 = 方案["DistributionBaseShares_先"], 方案["AddShares_先"]
        基数后, 新增后 = 方案["DistributionBaseShares_后"], 方案["AddShares_后"]
        if not all(np.isfinite(x) and x > 0 for x in [变动前总股本, 变动后总股本, 基数先, 新增先, 基数后, 新增后]):
            continue
        # 新增股份字段含小数，总股本为整数；四项衔接均允许最多2股差额。
        股数差额 = [
            新增先 - 基数先 * 方案["ConversionRatio_先"],
            新增后 - 基数后 * 方案["ConversionRatio_后"],
            变动前总股本 + 新增先 - 基数后,
            基数后 + 新增后 - 变动后总股本,
        ]
        if not all(np.isfinite(x) and abs(x) <= 2 for x in 股数差额):
            continue
        两段补全.append({
            "Stkcd": 股票, "Trddt": 除权日,
            "first_mult": 1 + 方案["ConversionRatio_先"],
            "annual_mult": 1 + 方案["ConversionRatio_后"],
        })
    两段补全 = pd.DataFrame(
        两段补全, columns=["Stkcd", "Trddt", "first_mult", "annual_mult"],
    ).astype({"Stkcd": "string", "Trddt": "string", "first_mult": "float64", "annual_mult": "float64"})
    # 字典列表构造的键默认是对象类型，须与原始表统一，避免后续按代码近邻合并报错。
    检查唯一键(两段补全, ["Stkcd", "Trddt"], "两段顺序转增方案")
    除权事件 = 除权事件.merge(两段补全, on=["Stkcd", "Trddt"], how="left", validate="one_to_one")

    增发事件 = 日内股本.loc[
        日内股本["ChangeReasonID"].str.contains("P0703|P0704", na=False), ["Stkcd", "Trddt"],
    ].drop_duplicates()
    增发事件["issue"] = True
    除权事件 = 除权事件.merge(增发事件, on=["Stkcd", "Trddt"], how="outer", validate="one_to_one")
    股改事件 = 日内股本.loc[
        日内股本["ChangeReasonID"].str.contains("P0712|P0757", na=False), ["Stkcd", "Trddt"],
    ].drop_duplicates().rename(columns={"Trddt": "reform_date"})
    股改事件["reform_date"] = pd.to_datetime(股改事件["reform_date"])
    收益事件 = pd.concat([
        红利[["Stkcd", "Trddt"]], 配股[["Stkcd", "Trddt"]],
        股改事件.assign(Trddt=股改事件["reform_date"].dt.strftime("%Y-%m-%d"))[["Stkcd", "Trddt"]],
    ], ignore_index=True).dropna().drop_duplicates()
    收益事件["return_event_date"] = pd.to_datetime(收益事件["Trddt"])

    # ========================================================================
    # 2. 分块读日表：取六月/十二月月末快照，补月表起点之前的收益
    # ========================================================================
    print("2/7 提取月末市值，并按除权参考价重建月收益……", flush=True)
    日文件 = []
    for 路径 in sorted(日表目录.glob("*/TRDNEW_Dalyr*.csv")):
        年份 = [int(x) for x in re.findall(r"\d{4}", 路径.parent.name)]
        if 年份 and max(年份) >= 起始形成年度 - 1 and min(年份) <= 结束形成年度:
            日文件.append(路径)
    if not 日文件:
        raise FileNotFoundError("没有找到所需年度的日度原始文件")

    快照片段, 实交易快照片段, 补充收益片段, 换码收益片段, 日月收益片段 = [], [], [], [], []
    上次实收价, 上上次实收价, 上次实交易日 = {}, {}, {}  # 跨读取块与年度衔接。
    上次日历价, 上次日历填充, 上次日历复利 = {}, {}, {}  # 包括停牌记录，用于识别技术填充。
    停牌技术收益修正数 = 0
    日字段 = ["Stkcd", "Trddt", "Markettype", "Filling", "Dsmvosd", "Dretwd", "Dretnd",
              "Dnshrtrd", "Clsprc", "PreClosePrice"]
    # Dsmvosd为流通市值，单位千元；Dretwd为含红利日收益；Dnshrtrd为成交股数。
    # Filling（填充标识）：0正常记录，1非节假日填充，2节假日填充。
    for 序号, 路径 in enumerate(日文件, start=1):
        print(f"  日表 {序号}/{len(日文件)}：{路径.parent.name}/{路径.name}", flush=True)
        for 块 in pd.read_csv(
            路径, usecols=日字段, encoding="utf-8-sig", chunksize=日表分块行数,
            dtype={"Stkcd": "string", "Trddt": "string"},
        ):
            块 = 块.loc[
                块["Markettype"].isin(市值读取市场)
                & 块["Trddt"].between(最早日, 最晚日)
            ].copy()
            if 块.empty:
                continue
            块["Stkcd"] = 标准化代码(块["Stkcd"])
            块["Trdmnt"] = 块["Trddt"].str.slice(0, 7)

            月份 = 块["Trddt"].str.slice(5, 7)
            月末 = 块.loc[
                (月份.eq("06") & 块["Filling"].isin([0, 1]))
                | (月份.eq("12") & 块["Filling"].isin([0, 1, 2])),
                ["Stkcd", "Trddt", "Trdmnt", "Markettype", "Dsmvosd", "Filling", "Clsprc"],
            ].copy()
            # 每块只留各股票各月最后一天；合并所有块后再取全市场统一月末。
            月末 = 月末.sort_values("Trddt").groupby(["Stkcd", "Trdmnt"]).tail(1)
            快照片段.append(月末)

            块 = 块.loc[块["Markettype"].isin(市场类型)].copy()
            块 = 块.sort_values(["Stkcd", "Trddt"])
            块["calendar_gross"] = (1 + 块["Dretnd"]) / (1 - 块["Dretwd"] + 块["Dretnd"])
            日历分组 = 块.groupby("Stkcd", sort=False)
            日历首行 = 块.index[日历分组.cumcount().eq(0)]
            日历末行 = 日历分组.tail(1)
            for 原列, 前列, 跨块末值 in [
                ("Clsprc", "prev_calendar_price", 上次日历价),
                ("Filling", "prev_calendar_filling", 上次日历填充),
                ("calendar_gross", "prev_calendar_gross", 上次日历复利),
            ]:
                块[前列] = 日历分组[原列].shift()
                块.loc[日历首行, 前列] = 块.loc[日历首行, "Stkcd"].map(跨块末值)
                跨块末值.update(dict(zip(日历末行["Stkcd"], 日历末行[原列])))
            日月 = 块.loc[块["Filling"].eq(0)].sort_values(["Stkcd", "Trddt"]).copy()
            if not 日月.empty:
                # 半年内只留末次真实交易，用于识别停牌缩股后价格未同步调整的快照。
                日月["halfyear"] = 日月["Trddt"].str.slice(0, 4) + np.where(
                    日月["Trddt"].str.slice(5, 7).le("06"), "-06", "-12",
                )
                实交易快照片段.append(日月.groupby(["Stkcd", "halfyear"], sort=False).tail(1)[
                    ["Stkcd", "halfyear", "Trddt", "Clsprc", "Dsmvosd"]
                ].copy())
                if 日月["Dretwd"].lt(-1).any():
                    raise ValueError("日表存在小于-100%的收益率")
                日月["trade_date"] = pd.to_datetime(日月["Trddt"])
                日月["prev_real"] = 日月.groupby("Stkcd")["Clsprc"].shift()
                日月["prev2_real"] = 日月.groupby("Stkcd")["Clsprc"].shift(2)
                日月["prev_date"] = 日月.groupby("Stkcd")["trade_date"].shift()
                块内序号 = 日月.groupby("Stkcd").cumcount()
                首行, 次行 = 日月.index[块内序号.eq(0)], 日月.index[块内序号.eq(1)]
                日月.loc[首行, "prev_real"] = 日月.loc[首行, "Stkcd"].map(上次实收价)
                日月.loc[首行, "prev2_real"] = 日月.loc[首行, "Stkcd"].map(上上次实收价)
                日月.loc[次行, "prev2_real"] = 日月.loc[次行, "Stkcd"].map(上次实收价)
                日月.loc[首行, "prev_date"] = 日月.loc[首行, "Stkcd"].map(上次实交易日)
                末行 = 日月.groupby("Stkcd", sort=False).tail(1)
                上上次实收价.update(dict(zip(末行["Stkcd"], 末行["prev_real"])))
                上次实收价.update(dict(zip(末行["Stkcd"], 末行["Clsprc"])))
                上次实交易日.update(dict(zip(末行["Stkcd"], 末行["trade_date"])))
                日月 = 日月.merge(除权事件, on=["Stkcd", "Trddt"], how="left", validate="many_to_one")
                日月 = pd.merge_asof(
                    日月.sort_values(["trade_date", "Stkcd"]), 股改事件.sort_values(["reform_date", "Stkcd"]),
                    left_on="trade_date", right_on="reform_date", by="Stkcd", direction="backward",
                )
                日月 = pd.merge_asof(
                    日月.sort_values(["trade_date", "Stkcd"]),
                    收益事件[["Stkcd", "return_event_date"]].sort_values(["return_event_date", "Stkcd"]),
                    left_on="trade_date", right_on="return_event_date", by="Stkcd", direction="backward",
                )
                # 观察参考价因子：(1+不含红利收益)/(1-含红利收益+不含红利收益)。
                日月["观察复利"] = (1 + 日月["Dretnd"]) / (1 - 日月["Dretwd"] + 日月["Dretnd"])
                实价复利 = (日月["Clsprc"] / 日月["prev_real"]).where(日月["prev_real"].gt(0), 日月["观察复利"])
                接近实价 = (实价复利 - 日月["观察复利"]).abs().le(
                    理论参考价容差 * np.maximum(1, 日月["观察复利"].abs())
                )
                日月["选定复利"] = 日月["观察复利"].where(~接近实价, 实价复利)
                理论候选 = pd.DataFrame({
                    "普通方案": 日月["Clsprc"] * (日月["mult_o"] + 日月["ratio_o"])
                    / (日月["prev_real"] - 日月["cash_o"] + 日月["cost_o"]),
                    "年度扩展方案": 日月["Clsprc"] * (日月["mult"] + 日月["rights_ratio"])
                    / (日月["prev_real"] - 日月["cash"] + 日月["rights_cost"]),
                    "备选配股比例": 日月["Clsprc"] * (日月["mult"] + 日月["alt_ratio"])
                    / (日月["prev_real"] - 日月["cash"] + 日月["alt_cost"]),
                })
                候选差异 = 理论候选.sub(日月["观察复利"], axis=0).abs().fillna(np.inf)
                最接近候选 = pd.Series(
                    理论候选.to_numpy()[np.arange(len(日月)), 候选差异.to_numpy().argmin(axis=1)],
                    index=日月.index,
                )
                候选可信 = (最接近候选 - 日月["观察复利"]).abs().le(
                    理论参考价容差 * np.maximum(1, 日月["观察复利"].abs())
                )
                有方案 = 日月["has_plan"].fillna(False).astype(bool)
                日月.loc[有方案, "选定复利"] = 最接近候选.where(候选可信, 日月["观察复利"])
                # 股本衔接已确认的两段转增：仅年度部分匹配时，补回遗漏的流通股东股改对价。
                仅年度复利 = 日月["Clsprc"] * 日月["annual_mult"] / 日月["prev_real"]
                两段转增遗漏 = (
                    日月["first_mult"].gt(1) & 日月["annual_mult"].gt(1) & 日月["prev_real"].gt(0)
                    & (仅年度复利 - 日月["观察复利"]).abs().le(
                        理论参考价容差 * np.maximum(1, 日月["观察复利"].abs())
                    )
                )
                日月.loc[两段转增遗漏, "选定复利"] = 仅年度复利 * 日月["first_mult"]
                # 增发新股不是原持有人取得的送股，不计入收益；同日真实分红另按方案处理。
                增发无分红 = 日月["issue"].fillna(False).astype(bool) & ~有方案
                日月.loc[增发无分红, "选定复利"] = 实价复利
                期间股改 = 日月["reform_date"].gt(日月["prev_date"]) & 日月["reform_date"].le(日月["trade_date"])
                引用错位 = (
                    (日月["Clsprc"] / 日月["prev2_real"] - 日月["观察复利"]).abs().le(
                        错位前价容差 * np.maximum(1, 日月["观察复利"].abs())
                    ) & ~有方案 & ~期间股改 & ~接近实价
                )
                日月.loc[引用错位, "选定复利"] = 实价复利
                # 仅向特定对象分配的非年度重整，不能把全部转增视为原股东所得。
                # 若同日还有普通股东方案，仍使用上面的普通方案验证结果。
                特殊重整 = 日月["special"].fillna(False).astype(bool) & ~有方案
                日月.loc[特殊重整, "选定复利"] = 日月["Clsprc"] / 日月["PreClosePrice"]
                # 老年度纯现金方案有时未体现在升级日收益中，按原始派现金额恢复。
                历史现金 = (
                    日月["legacy_cash"].fillna(False).astype(bool)
                    & np.isfinite(理论候选["年度扩展方案"]) & 理论候选["年度扩展方案"].gt(0)
                )
                日月.loc[历史现金, "选定复利"] = 理论候选["年度扩展方案"]
                # 少量早年记录在停牌技术日改变填充价，复牌收益又引用该填充价。
                # 过滤停牌行后不能丢掉这段价格变动；两段均能由原始价格解释才接回真实前价。
                # 停牌期间有分红、配股或股改时排除，避免抹去真实除权；例如000408的2000-08-24。
                无期间除权 = ~日月["return_event_date"].gt(日月["prev_date"])
                停牌技术收益 = (
                    日月["prev_calendar_filling"].eq(1)
                    & 日月["prev_calendar_price"].gt(0) & 日月["prev_real"].gt(0)
                    & 日月["prev_calendar_price"].ne(日月["prev_real"])
                    & 日月["PreClosePrice"].eq(日月["prev_real"])
                    & (日月["Clsprc"] / 日月["prev_calendar_price"] - 日月["观察复利"]).abs().le(
                        理论参考价容差 * np.maximum(1, 日月["观察复利"].abs())
                    )
                    & (日月["prev_calendar_price"] / 日月["prev_real"] - 日月["prev_calendar_gross"]).abs().le(
                        理论参考价容差 * np.maximum(1, 日月["prev_calendar_gross"].abs())
                    )
                    & 无期间除权 & ~有方案 & ~期间股改 & ~接近实价 & ~引用错位 & ~增发无分红
                )
                日月.loc[停牌技术收益, "选定复利"] = 实价复利
                停牌技术收益修正数 += int(停牌技术收益.sum())
                日月 = 日月.loc[
                    日月["Trdmnt"].between(最早月份.strftime("%Y-%m"), 最晚月份.strftime("%Y-%m"))
                ].copy()
                日月["有交易日"] = 日月["Dnshrtrd"].gt(0).astype(int)
                日月["缺失日收益"] = 日月["观察复利"].isna().astype(int)
                日月收益片段.append(日月.groupby(["Stkcd", "Trdmnt"], as_index=False).agg(
                    月参考复利=("选定复利", "prod"),
                    日交易天数=("有交易日", "sum"),
                    日收益缺失数=("缺失日收益", "sum"),
                ))
            补充 = 块.loc[
                (块["Trddt"] < 原始月表首月.strftime("%Y-%m-%d"))
                & (块["Trddt"] >= 最早月份.strftime("%Y-%m-%d"))
                & 块["Filling"].eq(0),
                ["Stkcd", "Trddt", "Trdmnt", "Dretwd", "Dnshrtrd"],
            ].copy()
            if not 补充.empty:
                补充收益片段.append(补充)

            if len(换码索引) and not 日月.empty:
                本块索引 = pd.MultiIndex.from_frame(日月[["Stkcd", "Trdmnt"]])
                换码日 = 日月.loc[
                    本块索引.isin(换码索引),
                    ["Stkcd", "Trddt", "Trdmnt", "Dnshrtrd", "观察复利", "选定复利"],
                ].copy()
                if not 换码日.empty:
                    换码收益片段.append(换码日)

    print(f"  修复停牌技术填充收益引用：{停牌技术收益修正数:,}条", flush=True)
    if not 快照片段:
        raise ValueError("日表没有六月或十二月的市值记录")
    快照 = pd.concat(快照片段, ignore_index=True).drop_duplicates()
    # 六月按最后交易日，十二月允许12月31日节假日记录；周末无记录时取最近记录日。
    # 已提前退出面板的股票不取该股票更早日期替代统一快照日。
    统一月末日 = 快照.groupby("Trdmnt")["Trddt"].transform("max")
    快照 = 快照.loc[快照["Trddt"].eq(统一月末日)].copy()
    检查唯一键(快照, ["Stkcd", "Trdmnt"], "日度月末市值快照")
    快照["Trddt"] = pd.to_datetime(快照["Trddt"])
    快照["Trdmnt"] = pd.to_datetime(快照["Trdmnt"])
    快照["year"] = 快照["Trdmnt"].dt.year  # 快照日所在自然年。
    所需快照 = (
        快照["Trdmnt"].dt.month.eq(6) & 快照["year"].between(起始形成年度, 结束形成年度)
    ) | (
        快照["Trdmnt"].dt.month.eq(12) & 快照["year"].between(起始形成年度 - 1, 结束形成年度 - 1)
    )
    快照 = 快照.loc[所需快照].copy()
    快照["me_a"] = 快照["Dsmvosd"] * 1000  # 升级日表A股流通市值，元。
    B股市值 = 快照.loc[快照["Markettype"].isin([2, 8]), ["Stkcd", "Trddt", "me_a"]].rename(
        columns={"Stkcd": "ABSymbol", "me_a": "me_b"},
    )
    # 数值核对表明，升级日表的B股市值已折成人民币；不能再乘一次汇率。
    # 月表B股市值仍以美元或港币计价，不能直接代替这里的日表数值。
    快照 = 快照.loc[快照["Markettype"].isin(市场类型)].copy()

    股本 = pd.read_csv(
        股本路径, encoding="utf-8-sig",
        usecols=["Symbol", "ChangeDate", "DeclareDate", "EventID", "ChangeReasonID", "ABSymbol", "TradableSharesA",
                 "TradableSharesB", "TradeSharesTotal"],
        dtype={"Symbol": "string", "ABSymbol": "string"}, parse_dates=["ChangeDate", "DeclareDate"],
    ).rename(columns={"Symbol": "Stkcd"})
    股本["Stkcd"] = 标准化代码(股本["Stkcd"])
    股本["ABSymbol"] = 股本["ABSymbol"].str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    # 同日可能记录多次股本事件，取事件编号最后的一次。
    股本 = 股本.sort_values(["ChangeDate", "EventID"]).drop_duplicates(["Stkcd", "ChangeDate"], keep="last")
    继承股本 = 别名.merge(股本, left_on="FormerCode", right_on="Stkcd", how="inner")
    继承股本["Stkcd"] = 继承股本["new_code"]
    股本 = pd.concat([股本, 继承股本[股本.columns]], ignore_index=True)
    股本 = 股本.drop_duplicates(["Stkcd", "ChangeDate"], keep="first")
    拆细缩股 = 股本.loc[股本["ChangeReasonID"].eq("P0710"), ["Stkcd", "ChangeDate"]].rename(
        columns={"ChangeDate": "shrink_date"},
    )
    快照 = pd.merge_asof(
        快照.sort_values(["Trddt", "Stkcd"]), 股本.sort_values(["ChangeDate", "Stkcd"]),
        left_on="Trddt", right_on="ChangeDate", by="Stkcd", direction="backward",
    ).merge(B股市值, on=["ABSymbol", "Trddt"], how="left", validate="many_to_one")
    股本有效 = 快照["TradableSharesA"].gt(0)
    其他境外股数 = (
        快照["TradeSharesTotal"] - 快照["TradableSharesA"] - 快照["TradableSharesB"].fillna(0)
    )
    股本异常 = (股本有效 & 快照["TradeSharesTotal"].isna()) | 其他境外股数.lt(0)
    if 股本异常.any():
        示例 = 快照.loc[股本异常, ["Stkcd", "Trddt", "ChangeDate"]].head(10).to_string(index=False)
        raise ValueError(f"股本记录存在全部流通股缺失或境外流通股数为负：\n{示例}")
    # 包含H股及其他境外流通股，不能只取H股字段；例如S股也需要计入。
    快照["me_overseas"] = (
        快照["me_a"] * 其他境外股数 / 快照["TradableSharesA"]
    ).where(股本有效, 0)
    B股缺失 = 快照["TradableSharesB"].gt(0) & 快照["me_b"].isna()
    if B股缺失.any():
        示例 = 快照.loc[B股缺失, ["Stkcd", "Trddt", "ABSymbol"]].head(10).to_string(index=False)
        raise ValueError(f"存在B股流通股本但缺少同期B股人民币市值：\n{示例}")
    快照["me_b"] = 快照["me_b"].fillna(0)
    快照["me"] = 快照["me_a"] + 快照["me_b"] + 快照["me_overseas"]
    实交易快照 = pd.concat(实交易快照片段, ignore_index=True).drop_duplicates()
    实交易快照 = 实交易快照.sort_values("Trddt").drop_duplicates(["Stkcd", "halfyear"], keep="last")
    实交易快照["actual_date"] = pd.to_datetime(实交易快照["Trddt"])
    实交易快照["actual_me_a"] = 实交易快照["Dsmvosd"] * 1000
    实交易快照 = 实交易快照.rename(columns={"Clsprc": "actual_price"})
    快照 = pd.merge_asof(
        快照.sort_values(["Trddt", "Stkcd"]),
        实交易快照[["Stkcd", "actual_date", "actual_me_a", "actual_price"]].sort_values(["actual_date", "Stkcd"]),
        left_on="Trddt", right_on="actual_date", by="Stkcd", direction="backward",
    )
    快照 = pd.merge_asof(
        快照.sort_values(["Trddt", "Stkcd"]), 拆细缩股.sort_values(["shrink_date", "Stkcd"]),
        left_on="Trddt", right_on="shrink_date", by="Stkcd", direction="backward",
    )
    缩股未形成新价 = (
        快照["Filling"].eq(1) & 快照["shrink_date"].gt(快照["actual_date"])
        & 快照["Clsprc"].eq(快照["actual_price"])
        & 快照["actual_me_a"].gt(快照["me_a"])
    )
    # P0710包含拆细与缩股；同价下市值下降才属于本规则要保护的缩股情形。
    # 不能只改股数而保留缩股前价格：等待新的成交价或技术除权价后再使用新口径。
    # 原始公告可核对600705的2008年重组及2012年恢复上市说明：
    # https://static.cninfo.com.cn/finalpage/2008-10-28/45718875.PDF
    # https://static.cninfo.com.cn/finalpage/2012-08-23/61464936.PDF
    快照.loc[缩股未形成新价, "me"] += (
        快照.loc[缩股未形成新价, "actual_me_a"] - 快照.loc[缩股未形成新价, "me_a"]
    )
    print(f"  保护缩股后价格未同步调整的市值快照：{缩股未形成新价.sum():,}条", flush=True)
    职工股上市仍停牌 = (
        快照["Filling"].eq(1)
        & 快照["ChangeReasonID"].str.contains("P0760", regex=False, na=False)
        & 快照["ChangeDate"].gt(快照["actual_date"])
        & 快照["Clsprc"].eq(快照["actual_price"])
        & 快照["actual_me_a"].gt(0) & 快照["me_a"].gt(快照["actual_me_a"])
    )
    # 职工股上市但持续停牌、价格仍沿用末次成交价时，延续末次成交的A股市值口径。
    # 只认最新股本事件中的P0760；其他股本事件不能沿用更早的职工股上市保护。
    # 600538年报记载2005-12-27职工股上市，当时股票仍持续停牌：
    # https://static.cninfo.com.cn/finalpage/2009-06-03/53042412.PDF
    快照.loc[职工股上市仍停牌, "me"] += (
        快照.loc[职工股上市仍停牌, "actual_me_a"] - 快照.loc[职工股上市仍停牌, "me_a"]
    )
    print(f"  延续职工股上市停牌期间的末次成交市值：{职工股上市仍停牌.sum():,}条", flush=True)
    年末激励回填 = (
        快照["Trdmnt"].dt.month.eq(12) & 快照["Filling"].eq(1)
        & 快照["ChangeReasonID"].str.contains("P0717", regex=False, na=False)
        & 快照["ChangeDate"].dt.month.eq(12) & 快照["ChangeDate"].dt.day.eq(31)
        & 快照["DeclareDate"].gt(快照["Trddt"])
        & 快照["ChangeDate"].gt(快照["actual_date"])
        & 快照["Clsprc"].eq(快照["actual_price"]) & 快照["actual_me_a"].gt(0)
        & ~(缩股未形成新价 | 职工股上市仍停牌)
    )
    # 年报披露的股权激励汇总有时回填至12月31日，停牌填充值随之变动。
    # 仅限该类年末回填，不将所有停牌或所有高管股变动回退到末次成交。
    # 例如300186：2014-12-26末次成交市值与原2014-12月表一致，12月31日回填增加股数，
    # 相应P0717记录的公告日期为2015-04-28；这里只替换A股部分。
    快照.loc[年末激励回填, "me"] += (
        快照.loc[年末激励回填, "actual_me_a"] - 快照.loc[年末激励回填, "me_a"]
    )
    print(f"  修复停牌期间年末股权激励回填市值：{年末激励回填.sum():,}条", flush=True)
    快照 = 快照.merge(六月月表市值, on=["Stkcd", "Trdmnt"], how="left", validate="one_to_one")
    六月股改解禁仍停牌 = (
        快照["Trdmnt"].dt.month.eq(6) & 快照["Filling"].eq(1)
        & 快照["monthly_me_a"].isna()
        & 快照["ChangeReasonID"].str.contains("P0757", regex=False, na=False)
        & 快照["ChangeDate"].gt(快照["actual_date"])
        & 快照["Clsprc"].eq(快照["actual_price"])
        & 快照["actual_me_a"].gt(0) & 快照["me_a"].gt(快照["actual_me_a"])
        & ~(缩股未形成新价 | 职工股上市仍停牌 | 年末激励回填)
    )
    # P0757为股权分置股份上市；只处理六月整月无成交、填充值仍用旧价的情形。
    # 例如000582于2012-02-01股改限售股上市，六月仍停牌且沿用1月20日价格。
    # 仅延续A股市值；有B股时，已匹配的B股人民币市值继续单独计入。
    快照.loc[六月股改解禁仍停牌, "me"] += (
        快照.loc[六月股改解禁仍停牌, "actual_me_a"] - 快照.loc[六月股改解禁仍停牌, "me_a"]
    )
    print(f"  延续六月股改解禁后仍停牌的末次成交市值：{六月股改解禁仍停牌.sum():,}条", flush=True)
    六月停牌有成交 = (
        快照["Trdmnt"].dt.month.eq(6) & 快照["Filling"].eq(1)
        & 快照["monthly_me_a"].gt(0)
        & 快照["monthly_price"].eq(快照["Clsprc"])
        & ~(缩股未形成新价 | 职工股上市仍停牌 | 年末激励回填)
    )
    # 价格不相同可能来自停牌期间除息；此时保留日表的除息后技术填充值。
    # 仅替换A股部分，整月无成交不跨月回溯。十二月仍取日快照，不能机械套用六月替换。
    快照.loc[六月停牌有成交, "me"] += (
        快照.loc[六月停牌有成交, "monthly_me_a"] - 快照.loc[六月停牌有成交, "me_a"]
    )
    快照 = 快照.loc[np.isfinite(快照["me"]) & 快照["me"].gt(0)].copy()

    if 补充收益片段:
        补充日表 = pd.concat(补充收益片段, ignore_index=True).drop_duplicates()
        检查唯一键(补充日表, ["Stkcd", "Trddt"], "早年日收益表")
        if 补充日表["Dretwd"].lt(-1).any():
            raise ValueError("早年日表存在小于-100%的收益率")
        补充日表["gross"] = 1 + 补充日表["Dretwd"].fillna(0)  # 单日复利因子。
        补充日表["有交易"] = 补充日表["Dnshrtrd"].gt(0).astype(int)
        补充日表["缺失日收益数"] = 补充日表["Dretwd"].isna().astype(int)
        补充月表 = 补充日表.groupby(["Stkcd", "Trdmnt"], as_index=False).agg(
            Mretwd=("gross", "prod"),
            Ndaytrd=("有交易", "sum"),
            缺失日收益数=("缺失日收益数", "sum"),
        )
        补充月表["Mretwd"] = 补充月表["Mretwd"] - 1
        补充月表["Trdmnt"] = pd.to_datetime(补充月表["Trdmnt"])
        补充月表 = 补充月表.merge(公司[["Stkcd", "Listdt"]], on="Stkcd", how="left", validate="many_to_one")
        # 原始月表的上市首月收益为空，不能把日表相对发行价的首日涨幅带入。
        上市首月 = 补充月表["Trdmnt"].dt.to_period("M").eq(补充月表["Listdt"].dt.to_period("M"))
        补充月表.loc[上市首月, "Mretwd"] = np.nan
        补充月表 = 补充月表.drop(columns="Listdt")
        补充月表["月收益来自日表"] = 1
        补充月表["跨代码补充月份"] = 0
        月表 = pd.concat([月表, 补充月表], ignore_index=True)

    if 日月收益片段:
        # 月表只提供交易月数和上市首月空值口径；有效收益按日参考价重新构造。
        日月收益 = pd.concat(日月收益片段, ignore_index=True).groupby(
            ["Stkcd", "Trdmnt"], as_index=False,
        ).agg(月参考复利=("月参考复利", "prod"),
              日交易天数=("日交易天数", "sum"), 日收益缺失数=("日收益缺失数", "sum"))
        日月收益["Trdmnt"] = pd.to_datetime(日月收益["Trdmnt"])
        月表 = 月表.merge(日月收益, on=["Stkcd", "Trdmnt"], how="left", validate="one_to_one")
        可重建 = 月表["Mretwd"].notna() & 月表["月参考复利"].gt(0) & np.isfinite(月表["月参考复利"])
        月表.loc[可重建, "Mretwd"] = (月表.loc[可重建, "月参考复利"] - 1).round(6)
        月表.loc[可重建, "月收益来自日表"] = 1
        月表.loc[可重建, "缺失日收益数"] = 月表.loc[可重建, "日收益缺失数"]
        print(f"  按参考价重建月收益：{可重建.sum():,}个股票月", flush=True)
        月表 = 月表.drop(columns=["日交易天数", "日收益缺失数", "月参考复利"])

    # 在换码后的首个形成年度，补入同一证券旧代码下的前置月份。
    # 不改写换码前的历史标签代码；新代码已有的原始月份优先保留。
    换码日表 = pd.concat(换码收益片段, ignore_index=True).drop_duplicates() if 换码收益片段 else pd.DataFrame()
    if not 换码日表.empty:
        检查唯一键(换码日表, ["Stkcd", "Trddt"], "换码月份日收益")
    for _, 事件 in 换码事件.iterrows():
        新码, 旧码 = 事件["new_code"], 事件["FormerCode"]
        换码月 = pd.Timestamp(事件["change_month"])
        形成年 = 换码月.year + int(换码月.month >= 7)
        窗口起点 = pd.Timestamp(形成年 - 1, 7, 1)
        窗口终点 = pd.Timestamp(形成年, 6, 1)
        原生月份 = 月表.loc[
            月表["Stkcd"].eq(新码) & 月表["Trdmnt"].between(窗口起点, 窗口终点)
        ]
        # 新证券本身已达交易月数门槛时，保留原生窗口，不额外接入旧证券收益。
        if 原生月份["Ndaytrd"].gt(0).sum() >= 最少有交易月份:
            continue
        旧月份 = 月表.loc[
            月表["Stkcd"].eq(旧码) & 月表["Trdmnt"].ge(窗口起点) & 月表["Trdmnt"].lt(换码月)
        ].copy()
        事件日 = 换码日表.loc[
            换码日表["Stkcd"].isin([新码, 旧码]) & 换码日表["Trdmnt"].eq(事件["change_month"])
        ].copy() if not 换码日表.empty else pd.DataFrame()
        # 有些北交所旧代码已被数据商整体回写为新代码，不应再次桥接。
        有旧日记录 = not 事件日.empty and 事件日["Stkcd"].eq(旧码).any()
        if 旧月份.empty and not 有旧日记录:
            continue
        旧月份["Stkcd"] = 新码
        旧月份["跨代码补充月份"] = 1
        已有新码月份 = 月表.loc[月表["Stkcd"].eq(新码), "Trdmnt"]
        旧月份 = 旧月份.loc[~旧月份["Trdmnt"].isin(已有新码月份)]
        月表 = pd.concat([月表, 旧月份], ignore_index=True)
        if not 事件日.empty:
            检查唯一键(事件日, ["Trddt"], f"{旧码}转{新码}的换码交易日")
            换码复利 = 事件日["选定复利"].prod()
            if not np.isfinite(换码复利) or 换码复利 <= 0:
                raise ValueError(f"{旧码}转{新码}的换码月份无法还原有效参考价收益")
            换码月行 = pd.DataFrame([{
                "Stkcd": 新码, "Trdmnt": 换码月,
                "Mretwd": round(换码复利 - 1, 6),
                "Ndaytrd": int(事件日["Dnshrtrd"].gt(0).sum()),
                "月收益来自日表": 1, "缺失日收益数": int(事件日["观察复利"].isna().sum()),
                "跨代码补充月份": 1,
            }])
            月表 = 月表.loc[~(月表["Stkcd"].eq(新码) & 月表["Trdmnt"].eq(换码月))]
            月表 = pd.concat([月表, 换码月行], ignore_index=True)
        print(f"  收益代码衔接：{旧码}→{新码}，形成年度{形成年}", flush=True)
    检查唯一键(月表, ["Stkcd", "Trdmnt"], "补齐后的月收益表")

    所需月份 = pd.date_range(最早月份, 最晚月份, freq="MS")
    缺失月份 = 所需月份.difference(pd.DatetimeIndex(月表["Trdmnt"].unique()))
    if len(缺失月份):
        raise ValueError(f"原始数据整体缺少以下月份：{缺失月份.strftime('%Y-%m').tolist()}")
    if 月表["Mretwd"].lt(-1).any():
        raise ValueError("月表存在小于-100%的收益率")

    # ========================================================================
    # 3. 上年7月至当年6月动量：缺失收益补0，但缺行不增加有效交易月数
    # ========================================================================
    print("3/7 计算过去十二个月动量……", flush=True)
    月表["formation_year"] = np.where(
        月表["Trdmnt"].dt.month.ge(7),
        月表["Trdmnt"].dt.year + 1, 月表["Trdmnt"].dt.year,
    )  # formation_year（形成年度）。
    月表["gross"] = 1 + 月表["Mretwd"].fillna(0)
    月表["有交易月份"] = 月表["Ndaytrd"].gt(0).astype(int)
    月表["缺失月收益数"] = 月表["Mretwd"].isna().astype(int)
    动量 = 月表.groupby(["Stkcd", "formation_year"], as_index=False).agg(
        mom=("gross", "prod"),  # mom（复合动量），单位为小数收益率。
        observed_months=("Trdmnt", "size"),  # 实际存在记录的月份数。
        valid_months=("有交易月份", "sum"),  # 实际有交易的月份数。
        missing_return_months=("缺失月收益数", "sum"),
        daily_derived_months=("月收益来自日表", "sum"),
        missing_daily_returns=("缺失日收益数", "sum"),
        code_linked_months=("跨代码补充月份", "sum"),  # 用曾用代码补充或重建的月份数。
    )
    动量["mom"] = 动量["mom"] - 1
    阶段样本["存在动量月份"] = 动量[["Stkcd", "formation_year"]].copy()
    动量 = 动量.loc[动量["valid_months"].ge(最少有交易月份)].copy()
    阶段样本["交易月份达标"] = 动量[["Stkcd", "formation_year"]].copy()

    六月 = 快照.loc[快照["Trdmnt"].dt.month.eq(6)].rename(columns={
        "year": "formation_year", "me": "size_june",  # 六月末流通市值，元。
        "Trddt": "formation_date",  # 形成日。
        "Filling": "june_filling",  # 形成日填充标识。
    })
    六月 = 六月.loc[六月["formation_year"].between(起始形成年度, 结束形成年度)].copy()
    十二月 = 快照.loc[快照["Trdmnt"].dt.month.eq(12)].rename(columns={
        "year": "fiscal_year", "me": "me_dec",  # 上年自然年、年末流通市值。
        "Trddt": "dec_date",  # 上年末市值快照日。
    })
    检查唯一键(六月, ["Stkcd", "formation_year"], "六月市值表")
    检查唯一键(十二月, ["Stkcd", "fiscal_year"], "十二月市值表")
    缺六月年 = sorted(set(range(起始形成年度, 结束形成年度 + 1)) - set(六月["formation_year"]))
    缺十二月年 = sorted(set(range(起始形成年度 - 1, 结束形成年度)) - set(十二月["fiscal_year"]))
    if 缺六月年 or 缺十二月年:
        raise ValueError(f"日表快照年度覆盖不足：缺六月{缺六月年}，缺十二月{缺十二月年}")
    阶段样本["存在六月末市值"] = 六月[["Stkcd", "formation_year"]].copy()
    阶段样本["存在上年末市值"] = 十二月[["Stkcd", "fiscal_year"]].rename(columns={"fiscal_year": "formation_year"})
    阶段样本["存在上年末市值"]["formation_year"] += 1

    # ========================================================================
    # 4. 上年末合并年报账面权益及原始账面市值比
    # ========================================================================
    print("4/7 计算账面权益、原始账面市值比……", flush=True)
    科目 = ["A003000000", "A003112101", "A001222000", "A002208000"]
    # 依次为所有者权益合计、优先股、递延所得税资产、递延所得税负债。
    财报 = pd.read_csv(
        财报路径, encoding="utf-8-sig", usecols=["Stkcd", "Accper", "Typrep"] + 科目,
        dtype={"Stkcd": "string"}, parse_dates=["Accper"],
    )
    财报 = 财报.loc[
        财报["Typrep"].eq("A")
        & 财报["Accper"].dt.month.eq(12) & 财报["Accper"].dt.day.eq(31)
        & 财报["Accper"].dt.year.between(起始形成年度 - 1, 结束形成年度 - 1)
    ].copy().drop_duplicates()
    财报["Stkcd"] = 标准化代码(财报["Stkcd"])
    检查唯一键(财报, ["Stkcd", "Accper"], "合并年报")
    财报["fiscal_year"] = 财报["Accper"].dt.year
    财报["statement_version_source"] = "本地合并年报"  # 财报版本与核对来源。
    # 本地数据下载于后续更正之后，以下科目取已核对的历史公告版本，不按标签补数。
    # 如2025年及以前已有更正，保留该更正；不能把2026年的追溯值与首次值简单二选一。
    # 新元科技2024首次年报权益仅3,139,494.33元，扣递延所得税资产后为负，须自然剔除。
    历史财报版本核对 = [
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
    for 代码, 财年, 权益, 优先股, 税资, 税负, 来源 in 历史财报版本核对:
        命中 = 财报["Stkcd"].eq(代码) & 财报["fiscal_year"].eq(财年)
        财报.loc[命中, 科目] = [权益, 优先股, 税资, 税负]
        财报.loc[命中, "statement_version_source"] = 来源
    阶段样本["存在上年合并年报"] = 财报[["Stkcd", "fiscal_year"]].rename(columns={"fiscal_year": "formation_year"})
    阶段样本["存在上年合并年报"]["formation_year"] += 1
    财报["supplement_missing_count"] = 财报[科目[1:]].isna().sum(axis=1)
    # 上列记录补充科目的缺失个数；不把缺失的所有者权益补成0。
    财报["be"] = (
        财报["A003000000"] - 财报["A003112101"].fillna(0)
        + 财报["A002208000"].fillna(0) - 财报["A001222000"].fillna(0)
    )  # be（账面权益），单位元。
    账面表 = 财报.merge(
        十二月[["Stkcd", "fiscal_year", "me_dec", "dec_date"]],
        on=["Stkcd", "fiscal_year"], how="inner", validate="one_to_one",
    )
    账面表["bm"] = 账面表["be"] / 账面表["me_dec"]  # bm（原始账面市值比）。
    账面表 = 账面表.loc[np.isfinite(账面表["bm"]) & 账面表["bm"].gt(0)].copy()
    账面表["formation_year"] = 账面表["fiscal_year"] + 1
    阶段样本["账面市值比为正"] = 账面表[["Stkcd", "formation_year"]].copy()

    # ========================================================================
    # 5. 匹配原始行业历史：上年末时点；通过曾用代码补充更换代码公司的历史
    # ========================================================================
    print("5/7 匹配上年末行业，并计算行业调整账面比……", flush=True)
    行业 = pd.read_csv(
        行业路径, encoding="utf-8-sig", dtype="string",
        usecols=["Symbol", "IndustryClassificationID", "ImplementDate", "IndustryCode"],
    ).rename(columns={"Symbol": "Stkcd", "IndustryClassificationID": "industry_standard"})
    # industry_standard（行业分类标准）；IndustryCode（行业代码）。
    行业 = 行业.loc[行业["industry_standard"].isin(["P0207", "P0221"])].copy()
    行业["Stkcd"] = 标准化代码(行业["Stkcd"])
    行业["IndustryCode"] = 行业["IndustryCode"].str.strip().str.upper()
    行业["ImplementDate"] = pd.to_datetime(行业["ImplementDate"], errors="raise")
    行业 = 行业.dropna(subset=["ImplementDate", "IndustryCode"])
    行业 = 行业.loc[行业["IndustryCode"].ne("")].drop_duplicates().copy()
    行业键 = ["Stkcd", "industry_standard", "ImplementDate"]
    检查唯一键(行业, 行业键, "原始行业历史")
    行业["industry_source"] = "原始行业历史"  # 行业信息来源。

    继承行业 = 别名.merge(行业, left_on="FormerCode", right_on="Stkcd", how="inner")
    继承行业["Stkcd"] = 继承行业["new_code"]
    继承行业["industry_source"] = "原始行业历史经曾用代码衔接"
    # 同代码同标准同日若已有本身记录，优先本身记录；继承只用于补历史。
    行业 = pd.concat([行业, 继承行业[行业.columns]], ignore_index=True)
    行业 = 行业.drop_duplicates(行业键, keep="first")

    账面表["industry_standard"] = pd.Series(
        np.where(账面表["formation_year"].lt(2023), "P0207", "P0221"),
        index=账面表.index, dtype="string",
    )
    账面表 = pd.merge_asof(
        账面表.sort_values(["Accper", "Stkcd", "industry_standard"]),
        行业.sort_values(["ImplementDate", "Stkcd", "industry_standard"]),
        left_on="Accper", right_on="ImplementDate",
        by=["Stkcd", "industry_standard"], direction="backward", allow_exact_matches=True,
    )
    print(f"  正账面比股票年中缺行业：{账面表['IndustryCode'].isna().sum():,}", flush=True)
    账面表 = 账面表.dropna(subset=["IndustryCode"]).copy()
    阶段样本["匹配上年末行业"] = 账面表[["Stkcd", "formation_year"]].copy()
    行业组 = 账面表.groupby(["formation_year", "IndustryCode"])
    账面表["industry_firms"] = 行业组["Stkcd"].transform("nunique")  # 行业内公司数。
    账面表["industry_bm"] = 行业组["be"].transform("sum") / 行业组["me_dec"].transform("sum")
    # industry_bm（行业基准账面比）为总账面权益除以总流通市值。
    账面表["log_bm_gap"] = np.log(账面表["bm"]) - np.log(账面表["industry_bm"])
    账面表["industry_std"] = (
        账面表.groupby(["formation_year", "IndustryCode"])["log_bm_gap"]
        .transform("std", ddof=行业标准差自由度)
    )  # 行业内对数偏差标准差。
    账面表 = 账面表.loc[
        账面表["industry_firms"].gt(1) & 账面表["industry_std"].gt(0)
    ].copy()
    账面表["bm_adj"] = 账面表["log_bm_gap"] / 账面表["industry_std"]
    # bm_adj（行业调整账面市值比）；先在账面比有效样本内计算，再取三指标交集。
    阶段样本["行业调整有效"] = 账面表.loc[np.isfinite(账面表["bm_adj"]), ["Stkcd", "formation_year"]].copy()

    # ========================================================================
    # 6. 三指标交集 → 逐级等数量五分组 → 三位组合代码
    # ========================================================================
    print("6/7 形成年度特征表并逐级分组……", flush=True)
    标签 = 六月[["Stkcd", "formation_year", "formation_date", "Markettype", "size_june", "june_filling"]].merge(
        动量, on=["Stkcd", "formation_year"], how="inner", validate="one_to_one",
    ).merge(
        账面表[["Stkcd", "formation_year", "Accper", "dec_date", "be", "me_dec", "bm", "bm_adj",
                "IndustryCode", "industry_standard", "industry_source", "ImplementDate",
                "industry_firms", "industry_bm", "industry_std", "supplement_missing_count",
                "statement_version_source"]],
        on=["Stkcd", "formation_year"], how="inner", validate="one_to_one",
    )
    有限值 = np.isfinite(标签[["size_june", "bm", "bm_adj", "mom"]]).all(axis=1)
    标签 = 标签.loc[有限值].reset_index(drop=True)
    if 标签.empty:
        raise ValueError("三个指标交集为空，请检查输入日期与行业匹配")
    检查唯一键(标签, ["Stkcd", "formation_year"], "年度排序特征表")
    标签["MarketValue"] = 降序五分组(标签, ["formation_year"], "size_june")
    标签["BooktoMarket"] = 降序五分组(标签, ["formation_year", "MarketValue"], "bm_adj")
    标签["Momentum"] = 降序五分组(标签, ["formation_year", "MarketValue", "BooktoMarket"], "mom")
    # 上面三列依次是规模组、账面比组、动量组，都是1高、5低。
    标签["dgtw_port"] = (
        标签["MarketValue"].astype(str) + 标签["BooktoMarket"].astype(str) + 标签["Momentum"].astype(str)
    )  # 三位组合代码，例如111、235、555。
    标签["portfolioNumber"] = (
        (标签["MarketValue"].astype("int16") - 1) * 25
        + (标签["BooktoMarket"].astype("int16") - 1) * 5 + 标签["Momentum"].astype("int16")
    )  # 便于合并的1至125编号；与原项目低值为1的编号含义不同，不能混用。
    标签["IsNotBSE"] = 北交所口径  # 是否剔除北交所。
    标签["dateBeg"] = pd.to_datetime(标签["formation_year"].astype(str) + "-07-01")
    标签["dateEnd"] = pd.to_datetime((标签["formation_year"] + 1).astype(str) + "-06-30")
    标签 = 标签.sort_values(["formation_year", "Stkcd"]).reset_index(drop=True)

    # ========================================================================
    # 7. 至此才读取官方结果；比较覆盖率、各层标签、条件匹配及组合最大值
    # ========================================================================
    print("7/7 与官方结果比较……", flush=True)
    官方 = pd.read_csv(官方分配路径, encoding="utf-8-sig", dtype={"Symbol": "string"})
    官方 = 官方.rename(columns={"Symbol": "Stkcd", "TradingYear": "formation_year"})
    官方["Stkcd"] = 标准化代码(官方["Stkcd"])
    官方 = 官方.loc[
        官方["IsNotBSE"].eq(北交所口径)
        & 官方["formation_year"].between(起始形成年度, 结束形成年度)
    ].copy()
    检查唯一键(官方, ["Stkcd", "formation_year"], "官方分配表所选口径")
    if 官方.empty:
        raise ValueError("官方分配表在所选年份及北交所口径下没有可比较记录")
    # 不把官方尚未提供的年份放进比较分母，重建标签本身仍完整保留。
    可比年份 = 官方["formation_year"].unique()
    对照 = 标签.loc[标签["formation_year"].isin(可比年份)].merge(
        官方[["Stkcd", "formation_year", "ShortName", "MarketValue", "BooktoMarket", "Momentum", "IndustryCode"]],
        on=["Stkcd", "formation_year"], how="outer", validate="one_to_one",
        suffixes=("_重建", "_官方"), indicator="样本来源",
    )
    共有 = 对照["样本来源"].eq("both")
    for 字段, 标记 in [("MarketValue", "规模一致"), ("BooktoMarket", "账面比一致"), ("Momentum", "动量一致")]:
        对照[标记] = (
            对照[f"{字段}_重建"].eq(对照[f"{字段}_官方"])
            .astype("boolean").where(共有, pd.NA)
        )
    对照["行业一致"] = (
        对照["IndustryCode_重建"].eq(对照["IndustryCode_官方"])
        .astype("boolean").where(共有, pd.NA)
    )
    对照["全部一致"] = (对照["规模一致"] & 对照["账面比一致"] & 对照["动量一致"]).where(共有, pd.NA)
    # 此时才对差异股票附加构造阶段标志，绝不将官方入选范围用于前面的分组。
    对照索引 = pd.MultiIndex.from_frame(对照[["Stkcd", "formation_year"]])
    for 阶段, 股票年 in 阶段样本.items():
        对照[阶段] = 对照索引.isin(pd.MultiIndex.from_frame(股票年))
    对照["首个未通过步骤"] = np.select(
        [~对照["存在六月末市值"], ~对照["存在动量月份"], ~对照["交易月份达标"],
         ~对照["存在上年合并年报"], ~对照["存在上年末市值"], ~对照["账面市值比为正"],
         ~对照["匹配上年末行业"], ~对照["行业调整有效"]],
        ["缺六月末有效流通市值", "缺动量窗口记录", "有交易月份不足", "缺上年合并年报",
         "缺上年末有效流通市值", "账面市值比非正或缺失", "缺上年末行业", "行业公司数或标准差无效"],
        default=np.where(对照["样本来源"].eq("right_only"), "排序指标非有限或合并缺失", "已生成标签"),
    )
    汇总行 = [汇总匹配(子表, int(年份)) for 年份, 子表 in 对照.groupby("formation_year", sort=True)]
    汇总行.append(汇总匹配(对照, "合计"))
    年度匹配 = pd.DataFrame(汇总行)

    组合键 = ["formation_year", "IsNotBSE", "MarketValue", "BooktoMarket", "Momentum"]
    重建最大值 = 标签.groupby(组合键, as_index=False).agg(
        MarketValueMax=("size_june", "max"),  # 组合最大流通市值。
        AdjbtmMax=("bm_adj", "max"),  # 组合最大行业调整账面比。
        BtmMax=("bm", "max"),  # 组合最大原始账面比。
        MomMax=("mom", "max"),  # 组合最大动量。
        重建组合股票数=("Stkcd", "size"),
    )
    官方最大值 = pd.read_csv(官方最大值路径, encoding="utf-8-sig").rename(columns={"TradingYear": "formation_year"})
    官方最大值 = 官方最大值.loc[
        官方最大值["IsNotBSE"].eq(北交所口径)
        & 官方最大值["formation_year"].between(起始形成年度, 结束形成年度)
    ].copy()
    检查唯一键(官方最大值, 组合键, "官方组合最大值表")
    最大值对照 = 重建最大值.loc[重建最大值["formation_year"].isin(官方最大值["formation_year"].unique())].merge(
        官方最大值, on=组合键, how="outer", validate="one_to_one",
        suffixes=("_重建", "_官方"), indicator="组合来源",
    )
    for 字段 in ["MarketValueMax", "AdjbtmMax", "BtmMax", "MomMax"]:
        最大值对照[f"{字段}_差值"] = 最大值对照[f"{字段}_重建"] - 最大值对照[f"{字段}_官方"]

    # 只有运行到此处才创建结果目录、写入结果，不覆盖项目原来的标签文件。
    输出目录.mkdir(parents=True, exist_ok=True)
    后缀 = f"{起始形成年度}_{结束形成年度}_剔北{北交所口径}"
    标签.to_csv(输出目录 / f"重建形成期标签_{后缀}.csv", index=False, encoding="utf-8-sig")
    对照.sort_values(["formation_year", "Stkcd"]).to_csv(
        输出目录 / f"逐股标签对照_{后缀}.csv", index=False, encoding="utf-8-sig",
    )
    年度匹配.to_csv(输出目录 / f"年度匹配率_{后缀}.csv", index=False, encoding="utf-8-sig")
    最大值对照.to_csv(输出目录 / f"组合最大值对照_{后缀}.csv", index=False, encoding="utf-8-sig")

    print("\n年度匹配结果，比例列以0至1表示：")
    print(年度匹配.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n重建标签共{len(标签):,}行；官方仅在最后用于比较，没有参与构造。")
    print(f"输出目录：{输出目录}")
    print("参考价重建仍受日收益精度、历史股本和财报版本影响；以本次对照结果为准。")


if __name__ == "__main__":
    main()
