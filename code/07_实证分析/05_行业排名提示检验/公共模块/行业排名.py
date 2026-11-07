"""构造含金融公司的日度行业市值排名，供独立行业提示检验读取。

只在调用 build_rank_cache（构造排名缓存）时读取数据、写入缓存；导入本文件
不会运行。所有运行时文件均限定在项目 work（临时工作目录）内的 cache_dir
（缓存目录）。本模块不修改既有清洗结果，也不调用既有清洗脚本。

旧版行业取“字母加两位数字”，M（综合类）单独保留。原国泰安行业缺失，
或旧版 C0、C8 等粗码缺少细分时，使用同标准、已生效且已公布的锐思记录
补充；粗码与锐思冲突时保留国泰安粗码并标记待定。对行业已确定的竞争者，
输出考虑待定竞争者前后的最好、最坏名次；待定公司本身保留资格和原代码。
不自动回退其他分类标准。001872在2023年5月1日至6月29日仅允许用已核对
对应关系的2012版G55补缺，且须当时有效、已公布，当前标准没有可匹配记录。
上市日期、交易状态或市值无法确定时逐年汇总后停止；结构问题立即停止。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "industry_rank_v7_verified_g55_exception"
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


def _source_paths(root):
    return {
        "calendar": root / "中间结果/01_清洗后样本/市场交易日历_td.csv",
        "industry": root / "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板.csv",
        "daily": root / "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv",
        "company": root / "input/CSMAR/公司文件(1990-12-10至2026-08-21)/TRD_Co.csv",
        "resset": root / "input/RESSET/RESSET 股票_行业变更历史/RESSET_INDHIS_1.csv",
        "matrix": root / "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv",
        "module": Path(__file__).resolve(),
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


def _cache_setup(root, cache_dir, start_year, end_year, overrides):
    root = Path(root).resolve()
    cache_dir = Path(cache_dir).resolve()
    if not cache_dir.is_relative_to(root / "work"):
        raise ValueError("行业排名的 cache_dir（缓存目录）必须位于当前项目 work 内。")
    start_year, end_year = int(start_year), int(end_year)
    if start_year > end_year:
        raise ValueError("排名起始年份不能晚于结束年份。")
    params = {
        "start_year": start_year, "end_year": end_year,
        "overrides": _prepare_overrides(overrides),
    }
    result = {
        "rank_dir": cache_dir / "行业排名_逐年",
        "industry_dir": cache_dir / "行业归并_逐年",
        "audit_path": cache_dir / "行业排名_核对清单.csv",
        "manifest_path": cache_dir / "行业排名_缓存清单.json",
        "start_year": start_year, "end_year": end_year,
    }
    return root, cache_dir, params, result


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


def _required_ranking_start(matrix_path, calendar, start_year, end_year, chunk_size):
    """只读扫描当期一期关系，确定最早真正需要的排名日期。"""
    start = pd.Timestamp(year=start_year, month=1, day=1)
    finish = pd.Timestamp(year=end_year, month=12, day=31)
    first_event_td = None
    columns = ["event_Trddt", "event_td", "quarterLag1Alt"]
    for chunk in pd.read_csv(
        matrix_path, usecols=columns, dtype="string",
        chunksize=int(chunk_size), encoding="utf-8-sig",
    ):
        chunk["event_Trddt"] = pd.to_datetime(chunk["event_Trddt"], errors="coerce")
        if chunk["event_Trddt"].isna().any():
            raise ValueError("无法准备排名缓冲：记忆矩阵存在无法识别的事件日期，不能确定其是否影响当前期间。")
        chunk = chunk.loc[chunk["event_Trddt"].between(start, finish)].copy()
        if chunk.empty:
            continue
        lag1 = pd.to_numeric(chunk["quarterLag1Alt"], errors="coerce")
        if (~lag1.isin([0, 1])).any():
            raise ValueError("无法准备排名缓冲：研究期内的一期记忆关系标记不是已知的0或1。")
        chunk = chunk.loc[lag1.eq(1)].copy()
        if chunk.empty:
            continue
        event_td = pd.to_numeric(chunk["event_td"], errors="coerce").astype("float64")
        if (~np.isfinite(event_td) | event_td.mod(1).ne(0)).any():
            raise ValueError("无法准备排名缓冲：一期记忆关系的事件交易日序号不是有限整数。")
        chunk["event_td"] = event_td.astype("int64")
        checked = chunk[["event_Trddt", "event_td"]].drop_duplicates().merge(
            calendar.rename(columns={"Trddt": "calendar_date", "td": "event_td"}),
            on="event_td", how="left", validate="many_to_one",
        )
        if checked["calendar_date"].isna().any() or checked["event_Trddt"].ne(checked["calendar_date"]).any():
            raise ValueError("无法准备排名缓冲：一期事件日期与统一市场交易日历不一致。")
        chunk_first = int(chunk["event_td"].min())
        first_event_td = chunk_first if first_event_td is None else min(first_event_td, chunk_first)
    if first_event_td is None:
        raise ValueError("无法准备排名缓冲：指定年份没有一期记忆关系事件。")
    first_rank_td = first_event_td - 3
    rank_dates = calendar.loc[calendar["td"].eq(first_rank_td), "Trddt"]
    if len(rank_dates) != 1:
        raise ValueError(f"无法准备排名缓冲：统一日历没有最早事件前第3个交易日，所需序号为{first_rank_td}。")
    begin = rank_dates.iloc[0]
    if begin < pd.Timestamp(year=start_year - 1, month=1, day=1):
        raise ValueError("无法准备排名缓冲：必要排名日期早于年度缓存支持范围，请核对市场交易日历。")
    first_event_date = calendar.loc[calendar["td"].eq(first_event_td), "Trddt"].iloc[0]
    coverage = {
        "begin": begin.strftime("%Y-%m-%d"),
        "finish": finish.strftime("%Y-%m-%d"),
        "first_event_date": first_event_date.strftime("%Y-%m-%d"),
        "first_event_td": first_event_td, "first_rank_td": first_rank_td,
    }
    return begin, coverage


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


def load_rank_cache(root, cache_dir, start_year, end_year, overrides=None):
    """只读验证完成标记、来源文件、口径参数及逐年结果，返回缓存路径。"""
    root, _, params, result = _cache_setup(root, cache_dir, start_year, end_year, overrides)
    path = result["manifest_path"]
    if not path.is_file():
        raise FileNotFoundError(f"没有完整的行业排名缓存，请先执行构造排名步骤：{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("version") != VERSION:
        raise ValueError("行业排名缓存未完成，或算法版本已经变化，请重新构造。")
    if manifest.get("parameters") != params:
        raise ValueError("行业排名缓存的年份或人工修正与当前参数不同，请重新构造。")
    if manifest.get("sources") != _source_state(_source_paths(root)):
        raise ValueError("行业排名的来源文件或模块发生变化，请重新构造缓存。")
    coverage = manifest.get("coverage", {})
    if not all(key in coverage for key in ["begin", "finish", "first_event_date", "first_event_td", "first_rank_td"]):
        raise ValueError("行业排名缓存缺少实际事件缓冲范围，请重新构造。")
    expected = []
    for year in range(params["start_year"] - 1, params["end_year"] + 1):
        expected.extend([result["rank_dir"] / f"{year}.parquet", result["industry_dir"] / f"{year}.parquet"])
    outputs = manifest.get("outputs", {})
    for output in expected:
        if not output.is_file():
            raise FileNotFoundError(f"行业排名缓存缺少年份文件：{output}")
        stat = output.stat()
        if outputs.get(str(output)) != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
            raise ValueError(f"行业排名缓存文件与完成清单不符，请重新构造：{output}")
    result["coverage"] = coverage
    return result


def build_rank_cache(root, cache_dir, start_year, end_year, chunk_size=400000, overrides=None):
    """逐年构造行业表和排名表；每一年先汇集全部块，再做完整日度截面排名。"""
    root, cache_dir, params, result = _cache_setup(root, cache_dir, start_year, end_year, overrides)
    if int(chunk_size) <= 0:
        raise ValueError("读取块大小必须为正整数。")
    paths = _source_paths(root)
    sources = _source_state(paths)
    # 缓冲准备只读进行；失败时不清空上一次核对清单或缓存完成标记。
    calendar = _load_calendar(paths["calendar"])
    begin, coverage = _required_ranking_start(
        paths["matrix"], calendar, params["start_year"], params["end_year"], chunk_size,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    result["rank_dir"].mkdir(exist_ok=True)
    result["industry_dir"].mkdir(exist_ok=True)
    # 构造期间没有完成标记；失败后不会把不完整或旧参数缓存误认为可用。
    result["manifest_path"].unlink(missing_ok=True)
    pd.DataFrame(columns=AUDIT_COLUMNS).to_csv(result["audit_path"], index=False, encoding="utf-8-sig")
    audit_path = result["audit_path"]
    years = list(range(params["start_year"] - 1, params["end_year"] + 1))
    finish = pd.Timestamp(year=years[-1], month=12, day=31)
    print(f"最早一期事件日：{coverage['first_event_date']}；实际排名起始日：{coverage['begin']}。")
    repairs = _resset_repair_intervals(paths["resset"])

    company = pd.read_csv(paths["company"], usecols=["Stkcd", "Listdt"], dtype="string", encoding="utf-8-sig")
    company["Stkcd"] = _stock_codes(company["Stkcd"])
    company["Listdt"] = pd.to_datetime(company["Listdt"], errors="coerce")
    company = company.drop_duplicates()
    if company["Stkcd"].isna().any() or company["Stkcd"].duplicated().any():
        _stop_with_audit(company, audit_path, "company", "上市日期表存在无效或冲突公司键", "请核对公司文件，不自动选择某个上市日期。")
    company["listedOneYearDate"] = company["Listdt"] + pd.DateOffset(years=1)

    summaries = []
    ranking_issue_count = 0
    ranking_issue_counts = {}
    problem_years = []
    with tempfile.TemporaryDirectory(prefix="行业排名_分块_", dir=cache_dir) as temporary:
        temporary = Path(temporary)
        industry_parts = {year: [] for year in years}
        daily_parts = {year: [] for year in years}

        # 一、行业底表分年。无法明确的行业先保留状态，实际需要它时再严格核对。
        reader = pd.read_csv(
            paths["industry"], usecols=["Stkcd", "td", "Trddt", "IndustryCode"],
            dtype="string", chunksize=int(chunk_size), encoding="utf-8-sig",
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
            chunk = _group_industries(chunk, repairs, params["overrides"])
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
            industry.to_parquet(result["industry_dir"] / f"{year}.parquet", index=False)

        # 二、源行情分年。保留原文件行号，去重在汇集完整年份后进行，不能分块去重。
        daily_columns = ["Stkcd", "Trddt", "Markettype", "Trdsta", "Filling", "Dsmvtll", "Dsmvosd"]
        reader = pd.read_csv(paths["daily"], usecols=daily_columns, dtype="string", chunksize=int(chunk_size), encoding="utf-8-sig")
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

        # 三、完整年份内统一去重、判断资格，再按“交易日×行业”整体排名。
        for year in years:
            print(f"构造 {year} 年含金融行业市值排名。")
            parts = daily_parts[year]
            industry = pd.read_parquet(result["industry_dir"] / f"{year}.parquet")
            rank_path = result["rank_dir"] / f"{year}.parquet"
            # 当前年度核对失败时，不留下上一次运行的旧排名冒充本次结果。
            rank_path.unlink(missing_ok=True)
            if not parts:
                _empty_rank().to_parquet(rank_path, index=False)
                summaries.append({"year": year, "source_rows": 0, "ranking_rows": 0, "industry_rows": len(industry)})
                continue
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
            output = daily[RANK_COLUMNS].sort_values(["td", "Stkcd"]).reset_index(drop=True)
            if output.duplicated(["Stkcd", "td"]).any():
                raise AssertionError("年度排名输出的公司交易日键不唯一。")
            output.to_parquet(rank_path, index=False)
            summaries.append({
                "year": year, "source_rows": len(output),
                "removed_duplicate_rows": before_dedup - len(output),
                "ranking_rows": int(eligible.sum()), "industry_rows": len(industry),
                "industry_groups": int(ranked["industryGroup"].nunique()),
                "unresolved_industry_rows": int(unknown_industry.sum()),
            })

    if _source_state(paths) != sources:
        raise ValueError("构造期间来源文件发生变化，未写入完成标记，请重新构造。")
    if ranking_issue_count:
        print("全期间年度核对已完成，各类问题记录数：", flush=True)
        for issue, count in ranking_issue_counts.items():
            print(f"  {issue}：{count:,}条。", flush=True)
        raise ValueError(
            f"全部年份核对完成，共{ranking_issue_count:,}条核对记录（按问题累计），"
            f"涉及年份：{problem_years}。问题年份未生成排名，未写入缓存完成标记，"
            f"不能继续构造面板。完整核对清单：{audit_path}"
        )
    outputs = {}
    for year in years:
        for directory in [result["rank_dir"], result["industry_dir"]]:
            output = directory / f"{year}.parquet"
            stat = output.stat()
            outputs[str(output)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    manifest = {
        "complete": True, "version": VERSION, "parameters": params,
        "sources": sources, "outputs": outputs, "years": summaries,
        "coverage": coverage,
        "rules": {
            "includes_financial": True, "market_types": list(A_SHARE_MARKETS),
            "allowed_statuses": list(ALLOWED_STATUSES), "listing_age": "one_calendar_year",
            "deduplication": "Filling=0 first; last source row within equal priority",
            "ranking": "total market cap descending, stock code ascending; rank interval includes possible unknown competitors",
            "buffer": "first in-period lag1 event trading index minus three; complete eligible cross-sections retained",
            "coarse_industry": "CSMAR first; same-standard effective and published RESSET history; no future fill",
            "verified_standard_exception": "001872 only, CSMAR missing, 2023-05-01 to 2023-06-29: publicly available and active RESSET 2012 G55 mapped to association G55; no generic cross-standard fallback",
        },
    }
    result["manifest_path"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    result["coverage"] = coverage
    return result
