"""准备行业前4、前5公司日度资格面板。

一行代表公司×交易日；唯一键为Stkcd（公司代码）与td（市场交易日序号）。
当日总市值按同行业降序排列，相同市值按公司代码升序排列。
前4、前5资格均为当日资格，主合并再匹配事件前第3个交易日，不提前平移。
保留金融公司及既有市场、上市年限、交易状态、行业归并与人工修正规则。
0表示明确不入选，1表示明确入选，空值表示资料不足或名次区间跨越门槛。
行业底表独有的公司日保留行业资料，其排名资格和前4、前5资格保持空值。

输入全部来自正式原始数据或正式中间结果；不读取记忆矩阵，不调用项目脚本。
默认2005—2025年，并用市场交易日历向前补足3个交易日；参数不取决于事件。
输出为正式变量目录内的一份日度面板、构造摘要及必要核对清单。
分块暂存仅在项目临时工作目录中；本文件供直接运行，不作为模块导入。
普通版与探索版运算、参数、字段、排序和写出方式相同，探索版增加分段查看。
探索版可在代码编辑器中按单元顺序运行；循环后保留最后一年数据供查看。
"""

from pathlib import Path
import gc
import json
import tempfile

import numpy as np
import pandas as pd


# %% 一、集中设置年份和分块大小
START_YEAR = 2005
END_YEAR = 2025
CHUNK_SIZE = 400_000
VERSION = "industry_daily_eligibility_v1"
A_SHARE_MARKETS = (1, 4, 16, 32)
ALLOWED_STATUSES = (1, 4, 7, 10, 13)
AUDIT_COLUMNS = [
    "stage", "issue", "Stkcd", "td", "Trddt", "IndustryCode",
    "industryGroup", "details",
]
INDUSTRY_COLUMNS = [
    "Stkcd", "td", "Trddt", "IndustryCode", "industryGroup", "industryStatus",
    "industrySourceStandard",
]
RANK_COLUMNS = [
    "Stkcd", "td", "Trddt", "size", "sizew", "IndustryCode", "industryGroup",
    "industryStatus", "industrySourceStandard", "industryRank", "industryRankMax",
    "industryN", "rankingEligible",
]
PANEL_COLUMNS = RANK_COLUMNS + ["isIndTop4", "isIndTop5"]

# 已核对的行业人工修正；限定原行业代码，依据和说明随构造摘要保存。
INDUSTRY_OVERRIDES = [
    {
        "Stkcd": "000780",
        "start": "2002-03-06",
        "end": "2007-04-25",
        "industryGroup": "C01",
        "onlyIfIndustryCode": "C0",
        "source": "https://static.cninfo.com.cn/finalpage/2002-03-06/550159.html",
        "note": (
            "草原兴发2001年报在2002年3月6日公开，主营为肉羊、肉鸡产品生产加工销售；"
            "据主营及旧版分类定义人工推定为C01食品加工业，非报告直接列明的代码。"
            "仅细化原C0，截止2007年4月26日资产交割前一天，不延用至煤炭业务期间。"
            "交割日期核对：https://static.cninfo.com.cn/finalpage/2007-08-17/30838419.PDF"
        ),
    },
    {
        "Stkcd": "000809",
        "start": "2010-11-15",
        "end": "2010-11-15",
        "industryGroup": "C81",
        "onlyIfIndustryCode": "C8",
        "source": "https://epaper.cs.com.cn/zgzqb/images/2010-11/15/B004/ZQBXP0041115C.pdf",
        "note": (
            "中汇医药2010年11月15日公告明确当日启用医药、生物制品分类；"
            "锐思2001版行业记录将C81生效日记为次日。"
            "2009年报明确主营为中药材及中成药加工业，对应旧版C8110，归并为C81。"
            "年报：https://static.cninfo.com.cn/finalpage/2010-03-24/57721726.PDF"
            "仅修正当天C8记录，不提前其他公司或其他日期的锐思生效日。"
        ),
    },
]


# %% 二、必要的小工具：键值检查、行业修复和名次区间
def find_root():
    """从脚本所在目录向上定位项目，移动项目后无需修改绝对路径。"""
    for folder in Path(__file__).resolve().parents:
        if (folder / "AGENTS.md").is_file() and (folder / "中间结果").is_dir():
            return folder
    raise FileNotFoundError("找不到包含项目说明和中间结果的项目根目录。")


def _source_paths(root):
    return {
        "calendar": root / "中间结果/01_清洗后样本/市场交易日历_td.csv",
        "industry": root / "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板.csv",
        "daily": root / "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv",
        "company": root / "input/CSMAR/公司文件(1990-12-10至2026-08-21)/TRD_Co.csv",
        "resset": root / "input/RESSET/RESSET 股票_行业变更历史/RESSET_INDHIS_1.csv",
    }



def _source_state(paths):
    state = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"行业排名所需文件不存在：{path}")
        stat = path.stat()
        state[name] = {
            "path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return state


def _stock_codes(values):
    codes = values.astype("string").str.strip()
    codes = codes.str.replace(r"\.0+$", "", regex=True)
    valid = codes.str.fullmatch(r"[0-9]{1,6}", na=False)
    return codes.where(valid).str.zfill(6)


def _clean_industry_codes(values):
    codes = values.astype("string").str.strip().str.upper()
    return codes.mask(codes.isin(["", "NAN", "NONE", "<NA>", "NAT"]))


def _prepare_overrides(overrides):
    """接受字典列表，或每行依次为代码、起始日、终止日、行业大类的列表。"""
    result = []
    for row in overrides or []:
        if isinstance(row, dict):
            values = [row.get(k) for k in ["Stkcd", "start", "end", "industryGroup"]]
        else:
            values = list(row)
        if len(values) != 4:
            raise ValueError("行业人工修正每行必须包含 Stkcd、start、end、industryGroup。")
        code = _stock_codes(pd.Series([values[0]])).iloc[0]
        start = pd.to_datetime(values[1], errors="coerce")
        end = pd.to_datetime(values[2], errors="coerce")
        group = str(values[3]).strip().upper()
        if pd.isna(code) or pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError(f"行业人工修正的公司或日期区间无效：{values}")
        valid_group = pd.Series([group]).str.fullmatch(r"[A-Z][0-9]{2}|M").iloc[0]
        if not valid_group or (group == "M" and end >= pd.Timestamp("2012-10-26")):
            raise ValueError(f"人工修正必须明确到行业大类，M 仅适用于旧版：{values}")
        item = {
            "Stkcd": str(code), "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"), "industryGroup": group,
        }
        if isinstance(row, dict):
            original_code = row.get("onlyIfIndustryCode")
            if original_code is not None:
                original_code = str(original_code).strip().upper()
                if not original_code:
                    raise ValueError("人工修正限定的原行业代码不能为空。")
                item["onlyIfIndustryCode"] = original_code
            for field in ["source", "note"]:
                if row.get(field) is not None:
                    item[field] = str(row[field])
        for previous in result:
            overlap = (previous["Stkcd"] == item["Stkcd"]
                       and previous["start"] <= item["end"]
                       and item["start"] <= previous["end"])
            same_scope = (
                previous.get("onlyIfIndustryCode") is None
                or item.get("onlyIfIndustryCode") is None
                or previous["onlyIfIndustryCode"] == item["onlyIfIndustryCode"]
            )
            if overlap and same_scope and previous["industryGroup"] != item["industryGroup"]:
                raise ValueError(f"同一公司存在互相冲突的行业人工修正：{previous}；{item}")
        result.append(item)
    return sorted(result, key=lambda x: (x["Stkcd"], x["start"], x["end"], x["industryGroup"]))



def _append_audit(rows, audit_path, stage, issue, details):
    """追加核对记录，允许年度内容问题一次覆盖整个研究期间。"""
    audit = rows.copy()
    audit["stage"] = stage
    audit["issue"] = issue
    audit["details"] = details
    for column in AUDIT_COLUMNS:
        if column not in audit:
            audit[column] = pd.NA
    audit[AUDIT_COLUMNS].to_csv(
        audit_path, mode="a", header=not audit_path.exists(),
        index=False, encoding="utf-8-sig",
    )
    return len(audit)


def _stop_with_audit(rows, audit_path, stage, issue, details):
    """结构问题仍立即阻止后续处理。"""
    _append_audit(rows, audit_path, stage, issue, details)
    raise ValueError(f"{issue}。{details} 核对清单：{audit_path}")


def _empty_industry():
    frame = pd.DataFrame({c: pd.Series(dtype="string") for c in INDUSTRY_COLUMNS})
    frame["td"] = pd.Series(dtype="int64")
    frame["Trddt"] = pd.Series(dtype="datetime64[ns]")
    return frame


def _empty_rank():
    frame = pd.DataFrame({c: pd.Series(dtype="string") for c in RANK_COLUMNS})
    frame["td"] = pd.Series(dtype="int64")
    frame["Trddt"] = pd.Series(dtype="datetime64[ns]")
    frame["size"] = pd.Series(dtype="float64")
    frame["sizew"] = pd.Series(dtype="float64")
    frame["industryRank"] = pd.Series(dtype="Int32")
    frame["industryRankMax"] = pd.Series(dtype="Int32")
    frame["industryN"] = pd.Series(dtype="Int32")
    frame["rankingEligible"] = pd.Series(dtype="int8")
    return frame


def _load_calendar(path):
    calendar = pd.read_csv(path, usecols=["Trddt", "td"], parse_dates=["Trddt"])
    calendar["td"] = pd.to_numeric(calendar["td"], errors="coerce")
    if calendar.isna().any().any() or calendar["Trddt"].duplicated().any() or calendar["td"].duplicated().any():
        raise ValueError("市场交易日历存在缺失日期、缺失序号或重复键。")
    if calendar["td"].mod(1).ne(0).any():
        raise ValueError("市场交易日序号必须为整数。")
    calendar["td"] = calendar["td"].astype("int64")
    calendar = calendar.sort_values("Trddt").reset_index(drop=True)
    if calendar["td"].diff().dropna().ne(1).any():
        raise ValueError("统一市场交易日序号不连续，不能用于公告前第3个交易日匹配。")
    return calendar



def _resset_repair_intervals(path):
    """按三套分类分别整理已生效、已公开且尚未取消的行业修复区间。"""
    columns = [
        "StkCd", "IndClsStd", "EffDt", "CancelDt", "InfoPubDt",
        "IndCd", "IndCd1", "IndCd2", "IndCd3", "IndCd4",
    ]
    history = pd.read_csv(path, usecols=columns, dtype="string", encoding="utf-8-sig")
    history = history.loc[history["IndClsStd"].isin(["1", "22", "44"])].copy()
    history["Stkcd"] = _stock_codes(history["StkCd"])
    history["repairStd"] = history["IndClsStd"].astype("string")
    for column in ["EffDt", "CancelDt", "InfoPubDt"]:
        history[column] = pd.to_datetime(history[column], errors="coerce")
    history = history.dropna(subset=["Stkcd", "EffDt"])
    history["availableDate"] = history[["EffDt", "InfoPubDt"]].max(axis=1)

    candidates = history[["IndCd", "IndCd1", "IndCd2", "IndCd3", "IndCd4"]].copy()
    for column in candidates:
        values = _clean_industry_codes(candidates[column])
        # 上市公司协会部分代码为 CG37 等，按项目既有规则转为 C37。
        association_code = history["repairStd"].eq("44") & values.str.fullmatch(r"[A-Z]{2}[0-9]{2}", na=False)
        values = values.where(~association_code, values.str[0] + values.str[-2:])
        valid = values.str.fullmatch(r"[A-Z][0-9]{2}(?:[0-9]{2})?", na=False)
        valid |= history["repairStd"].eq("1") & values.eq("M").fillna(False)
        candidates[column] = values.where(valid).str[:3]
    # 一条历史记录给出两个互相冲突的大类时，不任意挑选其中一个。
    history["repairGroup"] = candidates.bfill(axis=1).iloc[:, 0]
    history.loc[candidates.nunique(axis=1).ne(1), "repairGroup"] = pd.NA
    # 公布日期未知时保留这条生效记录作为阻断区间，而非删除后沿用更老细分类。
    history.loc[history["InfoPubDt"].isna(), "repairGroup"] = pd.NA

    intervals = []
    for (code, standard), stock in history.groupby(["Stkcd", "repairStd"], sort=False):
        boundaries = set(stock["availableDate"].dropna().tolist())
        boundaries.update((stock["CancelDt"].dropna() + pd.Timedelta(days=1)).tolist())
        for date in sorted(boundaries):
            active = stock.loc[
                stock["availableDate"].le(date)
                & (stock["CancelDt"].isna() | stock["CancelDt"].ge(date))
            ]
            group = pd.NA
            if not active.empty:
                active = active.loc[active["EffDt"].eq(active["EffDt"].max())]
                if active["InfoPubDt"].notna().all():
                    active = active.loc[active["InfoPubDt"].eq(active["InfoPubDt"].max())]
                    if active["repairGroup"].notna().all() and active["repairGroup"].nunique() == 1:
                        group = active["repairGroup"].iloc[0]
            # 取消后写入空区间，防止后续近邻匹配沿用已经取消的旧行业。
            intervals.append({"Stkcd": code, "repairStd": standard, "repairDate": date, "repairGroup": group})
    result = pd.DataFrame(intervals, columns=["Stkcd", "repairStd", "repairDate", "repairGroup"])
    result["Stkcd"] = result["Stkcd"].astype("string")
    result["repairStd"] = result["repairStd"].astype("string")
    result["repairGroup"] = result["repairGroup"].astype("string")
    result["repairDate"] = pd.to_datetime(result["repairDate"])
    return result.sort_values(["repairDate", "Stkcd", "repairStd"]).reset_index(drop=True)


def _group_industries(frame, repairs, overrides):
    frame = frame.copy().reset_index(drop=True)
    codes = _clean_industry_codes(frame["IndustryCode"])
    frame["IndustryCode"] = codes
    frame["industrySourceStandard"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    old = frame["Trddt"].lt(pd.Timestamp("2012-10-26"))
    valid_three = codes.str.fullmatch(r"[A-Z][0-9]{2}", na=False)
    valid_old_detail = old & codes.str.fullmatch(r"[A-Z][0-9]{4}", na=False)
    valid_old_m = old & codes.eq("M").fillna(False)
    coarse = old & codes.str.fullmatch(r"C[0-9]", na=False)
    valid = valid_three | valid_old_detail | valid_old_m
    frame["industryGroup"] = codes.str[:3].where(valid).astype("string")
    frame["industryStatus"] = pd.Series("invalid_code", index=frame.index, dtype="string")
    frame.loc[codes.isna(), "industryStatus"] = "unresolved_missing"
    frame.loc[coarse, "industryStatus"] = "unresolved_coarse"
    frame.loc[valid, "industryStatus"] = "valid"

    needs_repair = coarse | codes.isna()
    if needs_repair.any() and not repairs.empty:
        left = frame.loc[needs_repair, ["Stkcd", "Trddt", "IndustryCode"]].copy()
        left["_row"] = left.index
        # 通常只匹配当期标准；旧版代码未经映射不能混入新版行业排名。
        left["repairStd"] = "1"
        left.loc[left["Trddt"].ge(pd.Timestamp("2012-10-26")), "repairStd"] = "22"
        left.loc[left["Trddt"].ge(pd.Timestamp("2023-05-01")), "repairStd"] = "44"
        left["repairStd"] = left["repairStd"].astype("string")
        matched = pd.merge_asof(
            left.sort_values(["Trddt", "Stkcd", "repairStd"]), repairs,
            left_on="Trddt", right_on="repairDate", by=["Stkcd", "repairStd"],
            direction="backward", allow_exact_matches=True,
        )
        # 原代码全部缺失可补齐；非空粗码必须与细码同分支。
        same_branch = matched["repairGroup"].str[:2].eq(matched["IndustryCode"]).fillna(False)
        use_repair = matched["repairGroup"].notna() & (
            matched["IndustryCode"].isna() | same_branch
        )
        indices = matched.loc[use_repair, "_row"].astype("int64")
        frame.loc[indices, "industryGroup"] = matched.loc[use_repair, "repairGroup"].to_numpy()
        frame.loc[indices, "industryStatus"] = "valid"
        frame.loc[indices, "industrySourceStandard"] = (
            "RESSET_" + matched.loc[use_repair, "repairStd"]
        ).to_numpy()

        # 已核对的单独例外：招商港口001872的G55（水上运输业）在2012版
        # 与协会版对应一致。2012版记录公布于2012-12-31、2013-01-01生效；
        # 协会版记录2023-06-30才公布，不提前使用这条未来记录。
        # 当前标准必须完全没有当日可匹配记录：若存在取消、冲突或公布日
        # 未知留下的空区间，repairDate仍非空，不能绕过这些阻断记录。
        verified_gap = (
            matched["Stkcd"].eq("001872") & matched["IndustryCode"].isna()
            & matched["repairStd"].eq("44")
            & matched["Trddt"].between("2023-05-01", "2023-06-29")
            & matched["repairDate"].isna()
        )
        if verified_gap.any():
            query = matched.loc[verified_gap, ["Stkcd", "Trddt", "_row"]].copy()
            previous = repairs.loc[
                repairs["Stkcd"].eq("001872") & repairs["repairStd"].eq("22")
            ]
            if not previous.empty:
                verified = pd.merge_asof(
                    query.sort_values(["Trddt", "Stkcd"]), previous,
                    left_on="Trddt", right_on="repairDate", by="Stkcd",
                    direction="backward", allow_exact_matches=True,
                )
                verified = verified.loc[verified["repairGroup"].eq("G55").fillna(False)]
                indices = verified["_row"].astype("int64")
                frame.loc[indices, "industryGroup"] = "G55"
                frame.loc[indices, "industryStatus"] = "valid"
                frame.loc[indices, "industrySourceStandard"] = "RESSET_22_verified_G55_to_44"

    frame.loc[valid, "industrySourceStandard"] = "CSMAR"

    for item in overrides:
        selected = (frame["Stkcd"].eq(item["Stkcd"])
                    & frame["Trddt"].between(item["start"], item["end"]))
        if item.get("onlyIfIndustryCode") is not None:
            selected &= frame["IndustryCode"].eq(item["onlyIfIndustryCode"]).fillna(False)
        frame.loc[selected, "industryGroup"] = item["industryGroup"]
        frame.loc[selected, "industryStatus"] = "valid"
        frame.loc[selected, "industrySourceStandard"] = "manual"
    return frame[INDUSTRY_COLUMNS]



def make_daily_eligibility(daily, industry):
    """取行情排名与完整行业底表的键并集，生成可直接合并的当日资格。"""
    value_columns = [
        "Stkcd", "td", "Trddt", "size", "sizew",
        "industryRank", "industryRankMax", "industryN", "rankingEligible",
    ]
    output = industry.merge(
        daily[value_columns], on=["Stkcd", "td", "Trddt"],
        how="outer", validate="one_to_one", sort=False,
    )
    output["industryStatus"] = output["industryStatus"].fillna("unresolved_missing")
    for column in ["Stkcd", "IndustryCode", "industryGroup", "industryStatus", "industrySourceStandard"]:
        output[column] = output[column].astype("string")
    output["td"] = output["td"].astype("int64")
    for column in ["size", "sizew"]:
        output[column] = output[column].astype("float64")
    for column in ["industryRank", "industryRankMax", "industryN"]:
        output[column] = output[column].astype("Int32")
    output["rankingEligible"] = output["rankingEligible"].astype("Int8")
    eligible = output["rankingEligible"].eq(1).fillna(False)
    for top_n in (4, 5):
        flag = pd.Series(pd.NA, index=output.index, dtype="Int8")
        flag.loc[output["rankingEligible"].eq(0).fillna(False)] = 0
        known_rank = (
            eligible & output["industryRank"].notna()
            & output["industryRankMax"].notna()
        )
        flag.loc[known_rank & output["industryRank"].gt(top_n).fillna(False)] = 0
        flag.loc[known_rank & output["industryRankMax"].le(top_n).fillna(False)] = 1
        output[f"isIndTop{top_n}"] = flag
    output = output[PANEL_COLUMNS].sort_values(["td", "Stkcd"], kind="mergesort").reset_index(drop=True)
    if output.duplicated(["Stkcd", "td"]).any():
        raise ValueError("日度资格面板的公司与交易日序号不唯一。")
    if (output["isIndTop4"].eq(1) & output["isIndTop5"].ne(1)).fillna(False).any():
        raise ValueError("行业前4资格必须属于行业前5资格。")
    return output


# %% 三、定位正式输入、输出与日期覆盖范围
if type(START_YEAR) is not int or type(END_YEAR) is not int or START_YEAR > END_YEAR:
    raise ValueError("起止年份必须为整数，且起始年份不能晚于结束年份。")
if type(CHUNK_SIZE) is not int or CHUNK_SIZE < 1:
    raise ValueError("每次读取行数必须为正整数。")

ROOT = find_root()
paths = _source_paths(ROOT)
sources = _source_state(paths)
overrides = _prepare_overrides(INDUSTRY_OVERRIDES)
calendar = _load_calendar(paths["calendar"])
study_start = pd.Timestamp(year=START_YEAR, month=1, day=1)
finish = pd.Timestamp(year=END_YEAR, month=12, day=31)
study_calendar = calendar.loc[calendar["Trddt"].between(study_start, finish)]
if (
    study_calendar.empty
    or not calendar["Trddt"].dt.year.eq(START_YEAR).any()
    or not calendar["Trddt"].dt.year.eq(END_YEAR).any()
):
    raise ValueError("市场交易日历没有覆盖指定研究年份。")
first_study_td = int(study_calendar["td"].iloc[0])
first_rank_td = first_study_td - 3
rank_dates = calendar.loc[calendar["td"].eq(first_rank_td), "Trddt"]
if len(rank_dates) != 1:
    raise ValueError("市场交易日历无法向前提供3个交易日，请核对日历覆盖范围。")
begin = rank_dates.iloc[0]
years = list(range(begin.year, END_YEAR + 1))

output_dir = ROOT / "中间结果/02_变量构造结果/行业提示资格"
output_path = output_dir / "行业前4前5_日度资格面板.csv"
summary_path = output_dir / "日度资格面板_构造摘要.json"
audit_path = output_dir / "日度资格面板_核对清单.csv"
output_dir.mkdir(parents=True, exist_ok=True)
# 构造中移除完成摘要；旧面板只在新结果全部检查通过后替换。
summary_path.unlink(missing_ok=True)
pd.DataFrame(columns=AUDIT_COLUMNS).to_csv(audit_path, index=False, encoding="utf-8-sig")

work_dir = ROOT / "work/行业提示资格"
work_dir.mkdir(parents=True, exist_ok=True)
temporary_manager = tempfile.TemporaryDirectory(prefix="日度资格_", dir=work_dir)
temporary = Path(temporary_manager.name)
industry_dir = temporary / "行业归并"
industry_dir.mkdir()
staged_panel = temporary / output_path.name
industry_parts = {year: [] for year in years}
daily_parts = {year: [] for year in years}
print(f"研究年份：{START_YEAR}—{END_YEAR}；日度资格覆盖：{begin:%Y-%m-%d}至{finish:%Y-%m-%d}。", flush=True)

repairs = _resset_repair_intervals(paths["resset"])
company = pd.read_csv(paths["company"], usecols=["Stkcd", "Listdt"], dtype="string", encoding="utf-8-sig")
company["Stkcd"] = _stock_codes(company["Stkcd"])
company["Listdt"] = pd.to_datetime(company["Listdt"], errors="coerce")
company = company.drop_duplicates()
if company["Stkcd"].isna().any() or company["Stkcd"].duplicated().any():
    _stop_with_audit(company, audit_path, "company", "上市日期表存在无效或冲突公司键", "请核对公司文件，不自动选择某个上市日期。")
company["listedOneYearDate"] = company["Listdt"] + pd.DateOffset(years=1)


# %% 四、按年暂存完整行业底表，保留尚未确定的行业状态
reader = pd.read_csv(
    paths["industry"], usecols=["Stkcd", "td", "Trddt", "IndustryCode"],
    dtype="string", chunksize=CHUNK_SIZE, encoding="utf-8-sig",
)
for part_number, chunk in enumerate(reader):
    chunk["Trddt"] = pd.to_datetime(chunk["Trddt"], errors="coerce")
    chunk = chunk.loc[chunk["Trddt"].between(begin, finish)].copy()
    if chunk.empty:
        continue
    chunk["Stkcd"] = _stock_codes(chunk["Stkcd"])
    chunk["td"] = pd.to_numeric(chunk["td"], errors="coerce")
    invalid = chunk["Stkcd"].isna() | chunk["td"].isna() | chunk["td"].mod(1).ne(0)
    if invalid.any():
        _stop_with_audit(chunk.loc[invalid], audit_path, "industry", "行业底表存在无效键", "不能用无效公司或交易日序号参与排名。")
    chunk["td"] = chunk["td"].astype("int64")
    check = chunk.merge(calendar.rename(columns={"td": "calendar_td"}), on="Trddt", how="left", validate="many_to_one")
    invalid = check["calendar_td"].isna() | check["td"].ne(check["calendar_td"])
    if invalid.any():
        _stop_with_audit(check.loc[invalid], audit_path, "industry", "行业底表与统一市场日历不一致", "请先核对行业底表和市场日历。")
    chunk = _group_industries(chunk, repairs, overrides)
    for year, yearly in chunk.groupby(chunk["Trddt"].dt.year):
        part = temporary / f"industry_{year}_{part_number}.parquet"
        yearly.to_parquet(part, index=False)
        industry_parts[int(year)].append(part)

for year in years:
    parts = industry_parts[year]
    industry = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else _empty_industry()
    industry = industry.drop_duplicates()
    conflict = industry.duplicated(["Stkcd", "td"], keep=False)
    if conflict.any():
        _stop_with_audit(industry.loc[conflict], audit_path, "industry", "行业底表同一公司交易日存在冲突记录", "不能任意保留某一个行业。")
    industry = industry.sort_values(["td", "Stkcd"]).reset_index(drop=True)
    industry.to_parquet(industry_dir / f"{year}.parquet", index=False)


# %% 五、按年暂存原始行情，保留源文件行号供跨块去重
daily_columns = ["Stkcd", "Trddt", "Markettype", "Trdsta", "Filling", "Dsmvtll", "Dsmvosd"]
reader = pd.read_csv(paths["daily"], usecols=daily_columns, dtype="string", chunksize=CHUNK_SIZE, encoding="utf-8-sig")
source_offset = 0
for part_number, chunk in enumerate(reader):
    chunk["_source_order"] = np.arange(source_offset, source_offset + len(chunk), dtype="int64")
    source_offset += len(chunk)
    chunk["Trddt"] = pd.to_datetime(chunk["Trddt"], errors="coerce")
    for column in ["Markettype", "Trdsta", "Filling", "Dsmvtll", "Dsmvosd"]:
        # 使用普通浮点缺失值，避免可空布尔运算把市值缺失留成未知资格。
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("float64")
    a_share = chunk["Markettype"].isin(A_SHARE_MARKETS)
    if (a_share & chunk["Trddt"].isna()).any():
        _stop_with_audit(chunk.loc[a_share & chunk["Trddt"].isna()], audit_path, "daily", "A股行情存在无法识别的日期", "无法判断该记录是否影响当前排名期间。")
    chunk = chunk.loc[chunk["Trddt"].between(begin, finish)].copy()
    if chunk.empty:
        continue
    unknown = chunk["Markettype"].isna() | (chunk["Markettype"].isin(A_SHARE_MARKETS) & chunk["Filling"].isna())
    if unknown.any():
        _stop_with_audit(chunk.loc[unknown], audit_path, "daily", "行情市场类型或填充标志缺失", "无法确认是否属于排名竞争者，不能静默排除。")
    chunk = chunk.loc[chunk["Markettype"].isin(A_SHARE_MARKETS) & chunk["Filling"].isin([0, 1])].copy()
    if chunk.empty:
        continue
    chunk["Stkcd"] = _stock_codes(chunk["Stkcd"])
    if chunk["Stkcd"].isna().any():
        _stop_with_audit(chunk.loc[chunk["Stkcd"].isna()], audit_path, "daily", "行情公司代码无效", "不能静默删除可能的行业竞争者。")
    chunk = chunk.merge(calendar, on="Trddt", how="left", validate="many_to_one")
    if chunk["td"].isna().any():
        _stop_with_audit(chunk.loc[chunk["td"].isna()], audit_path, "daily", "行情日期无法匹配统一交易日历", "请核对日历覆盖范围。")
    chunk["td"] = chunk["td"].astype("int64")
    for year, yearly in chunk.groupby(chunk["Trddt"].dt.year):
        part = temporary / f"daily_{year}_{part_number}.parquet"
        yearly.to_parquet(part, index=False)
        daily_parts[int(year)].append(part)


# %% 六、汇集全年行情计算日度名次，并将完整资格面板按年追加写入暂存文件
summaries = []
ranking_issue_count = 0
ranking_issue_counts = {}
problem_years = []
panel_rows = 0
first_output = True
last_output_td = None

# 只打开一次带字节顺序标记的文件，年度之间不会重复插入文件头。
with staged_panel.open("w", encoding="utf-8-sig", newline="") as panel_handle:
    for year in years:
        print(f"准备{year}年行业前4、前5日度资格。", flush=True)
        parts = daily_parts[year]
        industry = pd.read_parquet(industry_dir / f"{year}.parquet")
        if parts:
            daily = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            before_dedup = len(daily)
            # 填充标志降序使 0 排在最后；同一优先级保留源文件最后一行。
            daily = daily.sort_values(["Stkcd", "td", "Filling", "_source_order"], ascending=[True, True, False, True], kind="mergesort")
            daily = daily.drop_duplicates(["Stkcd", "td"], keep="last")
            daily = daily.merge(company[["Stkcd", "listedOneYearDate"]], on="Stkcd", how="left", validate="many_to_one")
            daily = daily.merge(industry.drop(columns="Trddt"), on=["Stkcd", "td"], how="left", validate="one_to_one")
            daily["industryStatus"] = daily["industryStatus"].fillna("unresolved_missing")
            daily["size"] = daily["Dsmvtll"] * 1000.0
            daily["sizew"] = daily["Dsmvosd"] * 1000.0
            old_enough = daily["Trddt"].ge(daily["listedOneYearDate"])
            status_allowed = daily["Trdsta"].isin(ALLOWED_STATUSES)
            unknown_listing = daily["listedOneYearDate"].isna() & (status_allowed | daily["Trdsta"].isna())
            unknown_status = old_enough & daily["Trdsta"].isna()
            base_eligible = old_enough & status_allowed
            missing_size = base_eligible & (daily["size"].isna() | ~np.isfinite(daily["size"]))
            eligible = base_eligible & daily["size"].gt(0)
            unknown_industry = eligible & daily["industryStatus"].ne("valid")
            # 上市资格和市值不足时无法计算可靠的排名区间，继续汇总后停止。
            year_issue_count = 0
            checks = [
                (unknown_listing | unknown_status, "上市日期或交易状态不足以判断排名资格",
                 "已知不合格的公司日保留零资格；资格未知的潜在竞争者必须先核对。"),
                (missing_size, "合格竞争者总市值缺失或无穷",
                 "不能把市值未知的公司静默排除出行业排名。"),
            ]
            for mask, issue, details in checks:
                if mask.any():
                    count = _append_audit(daily.loc[mask], audit_path, "ranking", issue, details)
                    year_issue_count += count
                    ranking_issue_counts[issue] = ranking_issue_counts.get(issue, 0) + count
            if year_issue_count:
                ranking_issue_count += year_issue_count
                problem_years.append(year)
                print(f"{year}年发现{year_issue_count:,}条核对记录，不生成该年排名，继续核对后续年份。", flush=True)
                continue
            if unknown_industry.any():
                _append_audit(
                    daily.loc[unknown_industry], audit_path, "ranking", "合格竞争者行业大类待定",
                    "保留竞争资格和原行业代码，并计算可能影响的名次范围；"
                    "实际提示资格跨越门槛或同业关系不确定时继续核对。",
                )
            daily["rankingEligible"] = eligible.astype("int8")
            daily["industryRank"] = pd.Series(pd.NA, index=daily.index, dtype="Int32")
            daily["industryRankMax"] = pd.Series(pd.NA, index=daily.index, dtype="Int32")
            daily["industryN"] = pd.Series(pd.NA, index=daily.index, dtype="Int32")
            ranked = daily.loc[eligible].sort_values(
                ["td", "size", "Stkcd"], ascending=[True, False, True], kind="mergesort",
            ).copy()
            known = ranked["industryStatus"].eq("valid")
            coarse = ranked["industryStatus"].eq("unresolved_coarse")
            other_unknown = ~known & ~coarse

            # 已确定行业的最好名次只计确定的同行；最坏名次还计入市值更高、
            # 或市值相同且代码靠前、可能属于该细行业的待定竞争者。
            # 完全缺失或非法行业可能属于任意分支，不按字符串前缀推断。
            ranked["_branch"] = ranked["industryGroup"].str[:2].where(known)
            ranked.loc[coarse, "_branch"] = ranked.loc[coarse, "IndustryCode"]
            ranked["_branch"] = ranked["_branch"].fillna("__unknown__")
            ranked["_coarse"] = coarse.astype("int32")
            ranked["_other_unknown"] = other_unknown.astype("int32")
            ranked["_other_before"] = ranked.groupby("td", sort=False)["_other_unknown"].cumsum()
            branch_groups = ranked.groupby(["td", "_branch"], sort=False)
            ranked["_coarse_before"] = branch_groups["_coarse"].cumsum()
            ranked["_branch_before"] = branch_groups.cumcount()
            ranked["_day_before"] = ranked.groupby("td", sort=False).cumcount()
            known_groups = ranked.loc[known].groupby(["td", "industryGroup"], sort=False)
            known_rank = known_groups.cumcount() + 1
            daily.loc[known_rank.index, "industryRank"] = known_rank.astype("Int32")
            daily.loc[known_rank.index, "industryRankMax"] = (
                known_rank + ranked.loc[known_rank.index, "_coarse_before"]
                + ranked.loc[known_rank.index, "_other_before"]
            ).astype("Int32")
            # industryN 为已确定同行数；存在待定竞争者时不代表完整行业人数。
            daily.loc[known_rank.index, "industryN"] = known_groups["Stkcd"].transform("size").astype("Int32")

            # 待定公司的最好名次保守设为1，最坏名次计入所有可能同行。
            # 粗码仅与同分支及完全未知者比较；其他未知可能属于任何行业。
            daily.loc[ranked.index[~known], "industryRank"] = 1
            daily.loc[ranked.index[coarse], "industryRankMax"] = (
                1 + ranked.loc[coarse, "_branch_before"] + ranked.loc[coarse, "_other_before"]
            ).astype("Int32")
            daily.loc[ranked.index[other_unknown], "industryRankMax"] = (
                1 + ranked.loc[other_unknown, "_day_before"]
            ).astype("Int32")

        else:
            daily = _empty_rank()
            before_dedup = 0
            eligible = pd.Series(False, index=daily.index)
            unknown_industry = pd.Series(False, index=daily.index)
            ranked = daily

        output = make_daily_eligibility(daily, industry)
        if not output.empty:
            if last_output_td is not None and int(output["td"].min()) <= last_output_td:
                raise ValueError("年度日度面板没有按连续递增的交易日序号写出。")
            last_output_td = int(output["td"].max())
            output.to_csv(panel_handle, index=False, header=first_output, date_format="%Y-%m-%d")
            first_output = False
            panel_rows += len(output)
        summaries.append({
            "year": year, "daily_rows": len(daily), "industry_rows": len(industry),
            "panel_rows": len(output),
            "removed_duplicate_rows": before_dedup - len(daily),
            "ranking_rows": int(eligible.sum()),
            "industry_groups": int(ranked["industryGroup"].nunique()),
            "unresolved_industry_rows": int(unknown_industry.sum()),
            "top4_rows": int(output["isIndTop4"].eq(1).sum()),
            "top5_rows": int(output["isIndTop5"].eq(1).sum()),
            "unknown_top4_rows": int(output["isIndTop4"].isna().sum()),
            "unknown_top5_rows": int(output["isIndTop5"].isna().sum()),
        })
        gc.collect()


# %% 七、全部检查通过后发布正式面板与构造摘要，清理暂存分块
if _source_state(paths) != sources:
    temporary_manager.cleanup()
    raise ValueError("构造期间正式输入发生变化，未发布面板，请重新准备。")
if ranking_issue_count:
    temporary_manager.cleanup()
    raise ValueError(
        f"全期间存在{ranking_issue_count:,}条排名资格或总市值问题，"
        f"涉及年份{problem_years}；请查看核对清单：{audit_path}"
    )
available_years = {
    item["year"] for item in summaries
    if item["daily_rows"] > 0 and item["industry_rows"] > 0
}
missing_years = [year for year in years if year not in available_years]
if missing_years:
    temporary_manager.cleanup()
    raise ValueError(
        f"这些年份缺少行情或行业底表：{missing_years}；未发布跨全期的完成面板。"
    )
if panel_rows == 0:
    temporary_manager.cleanup()
    raise ValueError("指定期间没有日度面板记录，未发布空面板。")

staged_panel.replace(output_path)
panel_stat = output_path.stat()
summary = {
    "complete": True,
    "version": VERSION,
    "start_year": START_YEAR,
    "end_year": END_YEAR,
    "begin": begin.strftime("%Y-%m-%d"),
    "finish": finish.strftime("%Y-%m-%d"),
    "buffer_trading_days": 3,
    "rows": panel_rows,
    "columns": PANEL_COLUMNS,
    "key_order": ["td", "Stkcd"],
    "panel_file": {"size": panel_stat.st_size, "mtime_ns": panel_stat.st_mtime_ns},
    "sources": sources,
    "industry_overrides": overrides,
    "years": summaries,
    "rules": {
        "ranking": "当日同行业总市值降序；总市值相同按公司代码升序。",
        "date_match": "资格为当日值；事件前第3个交易日的匹配由主合并完成。",
        "pool": "包含金融公司；沿用既有市场、上市年限和交易状态规则。",
        "unknown": "排名区间跨越门槛或资料不足保留空值；明确不入选才记0。",
        "industry_only": "行业底表与行情排名取键并集；仅有行业资料时排名资格保持空值。",
    },
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
temporary_manager.cleanup()
print(f"已保存日度资格面板：{output_path}；共{panel_rows:,}行。", flush=True)
print(f"构造摘要：{summary_path}；核对清单：{audit_path}", flush=True)


# %% 八、查看最后一年结构与资格分布
print(output.head(10).to_string(index=False))
print(pd.DataFrame(summaries).to_string(index=False))
print(output[["rankingEligible", "isIndTop4", "isIndTop5"]].dtypes)
