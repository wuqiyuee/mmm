"""第二阶段探索：独立、逐单元构造提示事件。

用于VS Code（代码编辑器）的交互式窗口。所有计算代码与必要小工具都在本文件中，
不导入普通版、旧合并脚本或行业模块。按顺序运行单元可查看公司对筛选、主提示、
机制变量、聚合和保存后的数据结构。每次检查一种口径，默认市值90分位。
"""

# %% 00. 导入数据处理库、定义参数和本文件内必要工具
from pathlib import Path
import json

import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT_HINT = Path("/Volumes/苹果资料/mmm复现数据包")


TOP_NS = (4, 5)
SUE_DENOMINATOR_MODE = "nonmissing_sue"
TOP_MOVER_MODE = "any"
START_YEAR = 2005
END_YEAR = 2025
RANK_CACHE_DIR_NAME = "中间结果/04_合并数据/行业排名提示检验/排名缓存"
STAGE_ROOT_NAME = "三阶段合并"
INPUT_NAME = "候选公司对基础表.csv"
VERSION = "industry_rank_v7_verified_g55_exception"


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


def find_root() -> Path:
    for folder in [PROJECT_ROOT_HINT, *PROJECT_ROOT_HINT.parents]:
        if (folder / "AGENTS.md").is_file() and (folder / "中间结果").is_dir():
            return folder
    raise FileNotFoundError("找不到包含AGENTS.md和中间结果的项目根目录。")


def build_input_paths(root: Path) -> dict:
    """本阶段的正式输入路径，不调用其他项目脚本。"""
    return {
        "TRD_Cale_td_path": root / "中间结果/01_清洗后样本/市场交易日历_td.csv",
        "hs300_membership_path": root / "中间结果/02_变量构造结果/9指数成分股/沪深300/沪深300_历史成分区间.csv",
        "industry_path": root / "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv",
    }


def stage_root(root: Path) -> Path:
    return root / "中间结果/04_合并数据" / STAGE_ROOT_NAME


def save_summary(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def normalize_stock_code(x: object) -> str:
    """标准化股票代码，填充为6位数，并转换为字符串"""
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s == '':
        return ''
    return f"{int(float(s)):06d}"


def union_pipe_values(values):
    """合并以竖线分隔的来源标识，去重后按升序保留。"""
    tokens = set()
    for value in values.dropna():
        tokens.update(token for token in str(value).split("|") if token)
    return "|".join(sorted(tokens))


def sue_denominator_agg_specs():
    """返回三种 SUE 的非缺失 cue 数量和市值加总规则。"""
    specs = {}
    for sue_id in [1, 2, 3]:
        specs[f"sumSue{sue_id}ValidCueCountLag1S90p"] = (
            f"validSue{sue_id}CueLag1S90p",
            "sum",
        )
        specs[f"sumSue{sue_id}ValidCueMarketCapLag1S90p"] = (
            f"validSue{sue_id}CueMarketCapLag1S90p",
            "sum",
        )
    return specs


def build_pattern_cue_agg_specs(data_columns):
    """根据输入字段，构造求和与最大值聚合规则。"""

    agg_dict = {
        # 所选主组和三个真实市值辅助组的提示数量。
        "nrCuesLag1S90p": ("quarterLag1size90plus", "sum"),
        "nrCuesLag1S5090": ("quarterLag1size50_90", "sum"),
        "nrCuesLag1S1050": ("quarterLag1size10_50", "sum"),
        "nrCuesLag1S10m": ("quarterLag1size10minus", "sum"),

        # 盈余惊喜的等权分子。
        "sumSue1Lag1S90p": ("sue1Lag1S90p", "sum"),
        "sumSue2Lag1S90p": ("sue2Lag1S90p", "sum"),
        "sumSue3Lag1S90p": ("sue3Lag1S90p", "sum"),

        # 盈余惊喜的市值加权分子。
        "sumSue1Lag1XmarketCapS90p": (
            "sue1Lag1XmarketCapS90p", "sum"
        ),
        "sumSue2Lag1XmarketCapS90p": (
            "sue2Lag1XmarketCapS90p", "sum"
        ),
        "sumSue3Lag1XmarketCapS90p": (
            "sue3Lag1XmarketCapS90p", "sum"
        ),

        # 市值分母。
        "sumMarketCap_Minus3Lag1S90p": (
            "marketCap_Minus3Lag1S90p", "sum"
        ),
        "sumMarketCap_Minus3Lag1S5090": (
            "marketCap_Minus3Lag1S5090", "sum"
        ),

        # 历史共同公告数量的等权分子。
        "sumNrFirmsAnnLag1S90p": (
            "nrFirmsAnnLag1S90p", "sum"
        ),
        "sumNrFirmsAnnLag1S5090": (
            "nrFirmsAnnLag1S5090", "sum"
        ),

        # 历史共同公告数量的市值加权分子。
        "sumNrFirmsAnnLag1XmarketCapS90p": (
            "nrFirmsAnnLag1XmarketCapS90p", "sum"
        ),
        "sumNrFirmsAnnLag1XmarketCapS5090": (
            "nrFirmsAnnLag1XmarketCapS5090", "sum"
        ),
    }

    # 加入三种盈余惊喜各自的有效提示数量和有效市值。
    agg_dict.update(sue_denominator_agg_specs())

    # 所选主提示的实际报告来源及来源专属盈余惊喜，供后续核查。
    agg_dict["cueReportSourcesLag1S90p"] = (
        "lag1CueReportSourcesS90p", union_pipe_values
    )
    for anchor in ["Annual", "Q1"]:
        flag = f"lag1{anchor}Anchorsize90plus"
        agg_dict[f"nrCuesLag1{anchor}S90p"] = (flag, "sum")
        for sue_id in [1, 2, 3]:
            prefix = f"sue{sue_id}{anchor}Anchor"
            agg_dict[f"sumSue{sue_id}{anchor}AnchorLag1S90p"] = (
                f"{prefix}Lag1S90p", "sum"
            )
            agg_dict[f"sumSue{sue_id}{anchor}AnchorValidCueLag1S90p"] = (
                f"{prefix}ValidCueLag1S90p", "sum"
            )

    max_cols = [
        # 基础提示标记。
        "quarterLag1size90plus",
        "quarterLag1size50_90",
        "quarterLag1size10_50",
        "quarterLag1size10minus",
        "lag1AnnualAnchorsize90plus",
        "lag1Q1Anchorsize90plus",

        # 历史注意力和成交量标记。
        "lag1hiAEVsize90plus",
        "lag1hiavolsize90plus",
        "lag1hiavolSsize90plus",
        "lag1hiAEVsize50_90",
        "lag1hiavolsize50_90",
        "lag1hiavolSsize50_90",

        # 历史高绝对盈余惊喜。
        "lag1hi_sue1_abssize90plus",
        "lag1hi_sue2_abssize90plus",
        "lag1hi_sue3_abssize90plus",
        "lag1hi_sue1_abssize50_90",
        "lag1hi_sue2_abssize50_90",
        "lag1hi_sue3_abssize50_90",

        # 历史双方正向盈余惊喜。
        "lag1sue1possize90plus",
        "lag1sue2possize90plus",
        "lag1sue3possize90plus",
        "lag1sue1possize50_90",
        "lag1sue2possize50_90",
        "lag1sue3possize50_90",

        # 历史双方负向盈余惊喜。
        "lag1sue1negsize90plus",
        "lag1sue2negsize90plus",
        "lag1sue3negsize90plus",
        "lag1sue1negsize50_90",
        "lag1sue2negsize50_90",
        "lag1sue3negsize50_90",

        # 历史双方极端正向盈余惊喜。
        "lag1hi_sue1_possize90plus",
        "lag1hi_sue2_possize90plus",
        "lag1hi_sue3_possize90plus",
        "lag1hi_sue1_possize50_90",
        "lag1hi_sue2_possize50_90",
        "lag1hi_sue3_possize50_90",

        # 历史双方极端负向盈余惊喜。
        "lag1hi_sue1_negsize90plus",
        "lag1hi_sue2_negsize90plus",
        "lag1hi_sue3_negsize90plus",
        "lag1hi_sue1_negsize50_90",
        "lag1hi_sue2_negsize50_90",
        "lag1hi_sue3_negsize50_90",

        # 历史价格异动，具体口径由配置决定。
        "lag1topMoversize90plus",
        "lag1topMoversize50_90",
    ]

    for col in max_cols:
        if col in data_columns:
            agg_dict[col] = (col, "max")

    return agg_dict


def _source_paths(root):
    """只校验既有排名缓存的来源元数据，不加载或执行来源脚本。"""
    return {
        "calendar": root / "中间结果/01_清洗后样本/市场交易日历_td.csv",
        "industry": root / "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板.csv",
        "daily": root / "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv",
        "company": root / "input/CSMAR/公司文件(1990-12-10至2026-08-21)/TRD_Co.csv",
        "resset": root / "input/RESSET/RESSET 股票_行业变更历史/RESSET_INDHIS_1.csv",
        "matrix": root / "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv",
        "module": root / "code/07_实证分析/05_行业排名提示检验/公共模块/行业排名.py",
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
    if not cache_dir.is_relative_to(root / "中间结果"):
        raise ValueError("行业排名的正式缓存目录必须位于当前项目中间结果内。")
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


def stop_with_audit(data, audit_dir, filename, message):
    """只在新任务目录保存需要核对的记录，不把未知值补成0。"""
    path = Path(audit_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    raise ValueError(f"{message}。核对清单：{path}")


def read_cache_rows(keys, cache_dir, columns):
    """按已知交易日期逐年读取缓存，只保留本次需要的公司日。"""
    keys = keys[["Stkcd", "td", "Trddt"]].drop_duplicates()
    if keys.duplicated(["Stkcd", "td"]).any():
        raise ValueError("待匹配公司日对应多个交易日期。")
    parts = []
    for year, needed in keys.groupby(keys["Trddt"].dt.year):
        path = Path(cache_dir) / f"{int(year)}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"缺少已完成的年度缓存：{path}")
        data = pd.read_parquet(path, columns=columns)
        data["Stkcd"] = data["Stkcd"].astype("string").str.zfill(6)
        if data.duplicated(["Stkcd", "td"]).any():
            raise ValueError(f"年度缓存公司日键重复：{path}")
        matched = needed.merge(data, on=["Stkcd", "td", "Trddt"], how="left", validate="1:1", indicator=True)
        parts.append(matched)
        del data
    return pd.concat(parts, ignore_index=True)


# %% 01. 读取第一阶段公司对，保留证券代码前导零
PROJECT_ROOT = find_root()
paths = build_input_paths(PROJECT_ROOT)
candidate_path = stage_root(PROJECT_ROOT) / INPUT_NAME
if not candidate_path.is_file():
    raise FileNotFoundError(f"缺少第一阶段输出：{candidate_path}")
candidates = pd.read_csv(
    candidate_path,
    encoding="utf-8-sig",
    dtype={"Stkcd": "string", "StkcdRecall": "string"},
    parse_dates=["Annodt", "event_Trddt", "Trdmnt_for_breakpoint"],
)
for column in ["Stkcd", "StkcdRecall"]:
    candidates[column] = candidates[column].apply(normalize_stock_code).astype("string")
if candidates.duplicated(["Stkcd", "event_Trddt", "StkcdRecall"]).any():
    raise ValueError("第一阶段输出在公司对×事件交易日层面重复。")
print("表层级：提示公司 × 事件交易日 × 被提示公司")
print("候选公司对：", candidates.shape)
display(candidates.head())


# %% 02. 选择当前口径；行业排名口径还需将名额设为4或5
CURRENT_DEFINITION = "size90"
CURRENT_TOP_N = None
if CURRENT_DEFINITION not in {"size90", "hs300", "industry_top"}:
    raise ValueError("提示口径只能为市值90分位、沪深300或行业排名。")
if CURRENT_DEFINITION == "industry_top" and CURRENT_TOP_N is None:
    raise ValueError("行业排名口径必须指定每行业入选名额。")
mm = candidates.copy()


# %% 03. 合并当前口径资格，筛选事件日同行业公司对
if CURRENT_DEFINITION == "industry_top":
    rank_cache = load_rank_cache(
        root=PROJECT_ROOT,
        cache_dir=PROJECT_ROOT / RANK_CACHE_DIR_NAME,
        start_year=START_YEAR,
        end_year=END_YEAR,
        overrides=INDUSTRY_OVERRIDES,
    )
    calendar = pd.read_csv(
        paths["TRD_Cale_td_path"],
        usecols=["td", "Trddt"], parse_dates=["Trddt"],
    )
    audit_dir = stage_root(PROJECT_ROOT) / "提示事件" / f"行业前{CURRENT_TOP_N}名" / "核对清单"
    mm = mm.drop(columns=["mktcap_tm3", "float_mktcap_tm3"], errors="ignore")
    event_dates = mm[["event_td", "event_Trddt"]].drop_duplicates()
    checked = event_dates.merge(
        calendar.rename(columns={"td": "event_td", "Trddt": "calendar_date"}),
        on="event_td", how="left", validate="1:1",
    )
    bad_dates = checked.loc[checked["event_Trddt"].ne(checked["calendar_date"])]
    if not bad_dates.empty:
        stop_with_audit(bad_dates, audit_dir, "事件交易日与日历不一致.csv", "事件交易日与统一日历不一致")
    mm["td_tm3"] = mm["event_td"] - 3
    wanted = mm[["Stkcd", "td_tm3"]].drop_duplicates().rename(columns={"td_tm3": "td"})
    wanted = wanted.merge(calendar, on="td", how="left", validate="m:1")
    if wanted["Trddt"].isna().any():
        stop_with_audit(wanted.loc[wanted["Trddt"].isna()], audit_dir, "排名日期缺少交易日历.csv", "排名日不在统一交易日历中")
    ranked = read_cache_rows(
        wanted, rank_cache["rank_dir"],
        ["Stkcd", "td", "Trddt", "size", "sizew", "IndustryCode", "industryGroup", "industryStatus", "industrySourceStandard", "industryRank", "industryRankMax", "industryN", "rankingEligible"],
    )
    missing = ranked["_merge"].ne("both")
    unknown = ~ranked["rankingEligible"].isin([0, 1])
    eligible = ranked["rankingEligible"].eq(1)
    # 行业待定可以暂时没有名次；完成公司对筛选后才检查它是否影响实际提示。
    # industryN 只计已确定行业的公司；最差可能名次可超过这个数字。
    rank = ranked["industryRank"]
    rank_max = ranked["industryRankMax"]
    known_industry = ranked["industryStatus"].eq("valid").fillna(False)
    incomplete_bounds = rank.isna() ^ rank_max.isna()
    invalid_bounds = rank.notna() & (
        rank.lt(1) | rank_max.lt(rank)
        | (known_industry & (ranked["industryN"].isna() | rank.gt(ranked["industryN"])))
    )
    invalid_rank = eligible & (
        incomplete_bounds | invalid_bounds.fillna(False)
        | ranked["size"].isna() | ranked["size"].le(0)
    )
    bad = missing | unknown | invalid_rank.fillna(False)
    if bad.any():
        stop_with_audit(ranked.loc[bad], audit_dir, "提示公司排名缺失或无效.csv", "提示公司排名数据不完整，不能将未知提示资格补成0")
    ranked = ranked.drop(columns="_merge").rename(columns={
        "td": "td_tm3", "Trddt": "rankingDate", "size": "mktcap_tm3",
        "sizew": "float_mktcap_tm3", "IndustryCode": "rankingIndustryCode",
        "industryGroup": "rankingIndustryGroup", "industryStatus": "rankingIndustryStatus",
        "industrySourceStandard": "rankingIndustrySourceStandard",
    })
    # 已知基础资格为0的记录允许排名缺失，但其主提示资格必须为0。
    mm = mm.merge(ranked, on=["Stkcd", "td_tm3"], how="left", validate="m:1")
    mm = mm.loc[mm["rankingEligible"].eq(1)].copy()
    cue_keys = mm[["Stkcd", "event_td", "event_Trddt"]].rename(columns={"event_td": "td", "event_Trddt": "Trddt"})
    recall_keys = mm[["StkcdRecall", "event_td", "event_Trddt"]].rename(columns={"StkcdRecall": "Stkcd", "event_td": "td", "event_Trddt": "Trddt"})
    wanted = pd.concat([cue_keys, recall_keys], ignore_index=True).drop_duplicates()
    industry = read_cache_rows(
        wanted, rank_cache["industry_dir"],
        ["Stkcd", "td", "Trddt", "IndustryCode", "industryGroup", "industryStatus", "industrySourceStandard"],
    )
    valid = industry["industryStatus"].eq("valid") & industry["industryGroup"].notna()
    coarse = (
        industry["industryStatus"].eq("unresolved_coarse")
        & industry["IndustryCode"].str.fullmatch(r"C[0-9]", na=False)
    )
    # 粗码能限定可能的行业分支；其他不明代码没有可可靠判断的分支。
    industry["industryBranch"] = industry["industryGroup"].str[:2].where(valid)
    industry.loc[coarse, "industryBranch"] = industry.loc[coarse, "IndustryCode"]
    bad = industry["_merge"].ne("both") | (industry["industryStatus"].eq("valid") & ~valid)
    if bad.any():
        stop_with_audit(industry.loc[bad], audit_dir, "事件日公司对行业待核对.csv", "事件日行业记录缺失或已标记有效却没有行业大类")
    industry = industry.drop(columns=["_merge", "Trddt"])
    cue_industry = industry.rename(columns={"td": "event_td", "IndustryCode": "cueIndustryCode", "industryGroup": "cueIndustryGroup", "industryStatus": "cueIndustryStatus", "industrySourceStandard": "cueIndustrySourceStandard", "industryBranch": "cueIndustryBranch"})
    recall_industry = industry.rename(columns={"Stkcd": "StkcdRecall", "td": "event_td", "IndustryCode": "recallIndustryCode", "industryGroup": "recallIndustryGroup", "industryStatus": "recallIndustryStatus", "industrySourceStandard": "recallIndustrySourceStandard", "industryBranch": "recallIndustryBranch"})
    mm = mm.merge(cue_industry, on=["Stkcd", "event_td"], how="left", validate="m:1")
    mm = mm.merge(recall_industry, on=["StkcdRecall", "event_td"], how="left", validate="m:1")
    both_valid = mm["cueIndustryStatus"].eq("valid") & mm["recallIndustryStatus"].eq("valid")
    different_branch = (
        mm["cueIndustryBranch"].notna() & mm["recallIndustryBranch"].notna()
        & mm["cueIndustryBranch"].ne(mm["recallIndustryBranch"])
    ).fillna(False)
    uncertain = (~both_valid & ~different_branch).fillna(True)
    if uncertain.any():
        stop_with_audit(mm.loc[uncertain], audit_dir, "事件日公司对行业待核对.csv", "事件日公司对可能同业也可能异业，无法确定是否应排除")
    same_industry = (both_valid & mm["cueIndustryGroup"].eq(mm["recallIndustryGroup"])).fillna(False)
    print(f"归并行业相同的公司对：{int(same_industry.sum()):,}；过滤前：{len(mm):,}。", flush=True)
    mm = mm.loc[~same_industry].copy()
    top_ns = (int(CURRENT_TOP_N),)
    eligible = mm["rankingEligible"].eq(1)
    unknown = mm["industryRank"].isna() | mm["industryRankMax"].isna()
    for top_n in top_ns:
        unknown |= mm["industryRank"].le(top_n) & mm["industryRankMax"].gt(top_n)
    unknown = eligible & unknown.fillna(True)
    if unknown.any():
        columns = [
            "Stkcd", "StkcdRecall", "event_Trddt", "rankingDate", "rankingIndustryCode",
            "rankingIndustryGroup", "rankingIndustryStatus",
            "industryRank", "industryRankMax", "industryN", "mktcap_tm3",
        ]
        stop_with_audit(mm.loc[unknown, columns].drop_duplicates(), audit_dir, "提示公司行业名次待核对.csv", "实际线索的行业排名不确定，可能改变前四或前五资格")
else:
    if CURRENT_DEFINITION == "hs300":
        membership_path = paths["hs300_membership_path"]
        membership_path = Path(membership_path)
        if not membership_path.is_file():
            raise FileNotFoundError(
                f"找不到沪深300历史成分区间文件：{membership_path}。"
                "请在主函数的 hs300_membership_path（历史成分文件路径）参数中指定现有文件。"
            )

        members = pd.read_csv(membership_path, dtype={"Stkcd": "string"})
        required = {"Stkcd", "start_date", "end_date", "CoverageEndDate"}
        missing = required - set(members.columns)
        if missing:
            raise ValueError(f"沪深300历史成分文件缺少字段：{missing}")
        if members.empty:
            raise ValueError("沪深300历史成分文件为空，不能判断成员资格。")

        members["Stkcd"] = members["Stkcd"].apply(normalize_stock_code).astype("string")
        members["start_date"] = pd.to_datetime(members["start_date"], errors="raise")
        members["end_date"] = pd.to_datetime(members["end_date"], errors="raise")
        if members["Stkcd"].eq("").any() or members["start_date"].isna().any():
            raise ValueError("沪深300历史成分文件存在缺失证券代码或区间起始日。")
        if "Indexcd" in members.columns:
            index_codes = members["Indexcd"].apply(normalize_stock_code)
            if not index_codes.eq("000300").all():
                raise ValueError("历史成分文件包含非沪深300指数的记录。")

        # 结束日缺失表示截至文件覆盖末日仍在成分股中。
        coverage = pd.to_datetime(members["CoverageEndDate"], errors="raise")
        if coverage.isna().any() or coverage.nunique() != 1:
            raise ValueError("历史成分文件的覆盖截止日缺失或不一致。")
        coverage_start = members["start_date"].min()
        coverage_end = coverage.iloc[0]
        if coverage_end < coverage_start:
            raise ValueError("历史成分文件的覆盖截止日早于起始日。")
        open_end = coverage_end + pd.Timedelta(days=1)
        members["end_date"] = members["end_date"].fillna(open_end)
        members = members.sort_values(["Stkcd", "start_date"])
        previous_end = members.groupby("Stkcd")["end_date"].shift()
        if (
            members["end_date"].le(members["start_date"]).any()
            or members["start_date"].lt(previous_end).any()
        ):
            raise ValueError("沪深300历史成分区间长度无效，或同一公司的区间相互重叠。")

        if mm["event_Trddt"].isna().any():
            raise ValueError("记忆矩阵存在缺失事件交易日，不能判断沪深300资格。")
        in_coverage = mm["event_Trddt"].between(coverage_start, coverage_end)
        print("沪深300事件构造排除覆盖期外的公司对记录数：", (~in_coverage).sum())
        mm = mm.loc[in_coverage].copy()
        events = mm[["Stkcd", "event_Trddt"]].drop_duplicates().copy()
        matches = pd.merge_asof(
            events.sort_values("event_Trddt"),
            members[["Stkcd", "start_date", "end_date"]].sort_values("start_date"),
            by="Stkcd",
            left_on="event_Trddt",
            right_on="start_date",
            direction="backward",
            allow_exact_matches=True,
        )
        matches["isHS300"] = (
            matches["start_date"].notna()
            & matches["event_Trddt"].lt(matches["end_date"])
        ).astype("int8")
        mm = mm.merge(
            matches[["Stkcd", "event_Trddt", "isHS300"]],
            on=["Stkcd", "event_Trddt"],
            how="left",
            validate="m:1",
        )
        print("历史沪深300成员资格成立的公司对记录数：", mm["isHS300"].sum())
    industry_path = paths["industry_path"]
    mm_key = mm[
        ["Annodt", "event_td", "Stkcd", "StkcdRecall"]
    ].copy()

    mm_key["Stkcd"] = mm_key["Stkcd"].apply(normalize_stock_code)
    mm_key["StkcdRecall"] = mm_key["StkcdRecall"].apply(
        normalize_stock_code
    )

    daily_industry = pd.read_csv(
        industry_path,
        encoding="utf-8",
        usecols=["Stkcd", "td", "IndustryCode"],
        dtype={
            "Stkcd": "string",
            "IndustryCode": "string",
        },
    )

    daily_industry["Stkcd"] = daily_industry["Stkcd"].apply(
        normalize_stock_code
    )
    daily_industry["td"] = pd.to_numeric(
        daily_industry["td"], errors="coerce"
    )

    # 保留原规则：删除缺失字段记录，再对重复键保留首条。
    daily_industry = (
        daily_industry
        .dropna(subset=["Stkcd", "td", "IndustryCode"])
        .drop_duplicates(["Stkcd", "td"])
    )

    cue_industry = daily_industry.rename(
        columns={
            "td": "event_td",
            "IndustryCode": "cue_industry_code",
        }
    )

    recall_industry = daily_industry.rename(
        columns={
            "Stkcd": "StkcdRecall",
            "td": "event_td",
            "IndustryCode": "recall_industry_code",
        }
    )

    # 匹配提示公司在提示事件交易日的行业。
    mm_key = mm_key.merge(
        cue_industry[
            ["Stkcd", "event_td", "cue_industry_code"]
        ],
        on=["Stkcd", "event_td"],
        how="left",
        validate="m:1",
    )

    # 匹配被提示公司在同一提示事件交易日的行业。
    mm_key = mm_key.merge(
        recall_industry[
            ["StkcdRecall", "event_td", "recall_industry_code"]
        ],
        on=["StkcdRecall", "event_td"],
        how="left",
        validate="m:1",
    )

    industry_missing = mm_key[
        ["cue_industry_code", "recall_industry_code"]
    ].isna().any(axis=1)

    mm_key["same_industry"] = (
        mm_key["cue_industry_code"].notna()
        & mm_key["recall_industry_code"].notna()
        & (
            mm_key["cue_industry_code"]
            == mm_key["recall_industry_code"]
        )
    )

    print("同行业过滤前公司对记录数：", len(mm_key))
    print("至少一方行业缺失的记录数：", industry_missing.sum())
    print("同行业公司对记录数：", mm_key["same_industry"].sum())
    print("同行业公司对记录比例：", mm_key["same_industry"].mean())

    mm_key = mm_key.loc[~mm_key["same_industry"]].copy()

    print("同行业过滤后公司对记录数：", len(mm_key))
    key_cols = ["Annodt", "Stkcd", "StkcdRecall"]

    # 在去重之前统计，才能看出原筛选键表是否有重复。
    duplicate_count = mm_key.duplicated(key_cols).sum()
    print("去重前多出的重复筛选键记录数：", duplicate_count)

    mm_key_keep = mm_key[key_cols].drop_duplicates()
    print("去重后的合格筛选键数量：", len(mm_key_keep))

    rows_before_filter = len(mm)

    # 只使用名单筛选行，完整矩阵中的变量继续保留。
    mm = mm.merge(
        mm_key_keep,
        on=key_cols,
        how="inner",
        validate="1:1",
    )

    print("应用公司对过滤前记录数：", rows_before_filter)
    print("应用公司对过滤后记录数：", len(mm))
    print(
        "三轮公司对过滤合计删除记录数：",
        rows_before_filter - len(mm),
    )
    if CURRENT_DEFINITION == "size90":
        mm = mm.dropna(subset=[
            "mktcap_tm3", "marketCap_90th_percentile",
            "marketCap_50th_percentile", "marketCap_10th_percentile",
        ]).copy()

print("当前口径：", CURRENT_DEFINITION, "行业名额：", CURRENT_TOP_N)
print("口径筛选后的公司对：", mm.shape)
display(mm.head())


# %% 04. 构造主提示资格和真实市值辅助组
if CURRENT_DEFINITION == "industry_top":
    top_n = int(CURRENT_TOP_N)
    eligible = mm["rankingEligible"].eq(1)
    mm["mainCueEligible"] = (
        eligible & mm["industryRank"].le(top_n).fillna(False)
    ).astype("int8")
    mm["size90plus"] = (eligible & (mm["mktcap_tm3"] > mm["marketCap_90th_percentile"])).astype("int8")
    mm["size50_90"] = (eligible & (mm["mktcap_tm3"] > mm["marketCap_50th_percentile"]) & (mm["mktcap_tm3"] <= mm["marketCap_90th_percentile"])).astype("int8")
    mm["size10_50"] = (eligible & (mm["mktcap_tm3"] >= mm["marketCap_10th_percentile"]) & (mm["mktcap_tm3"] <= mm["marketCap_50th_percentile"])).astype("int8")
    mm["size10minus"] = (eligible & (mm["mktcap_tm3"] < mm["marketCap_10th_percentile"])).astype("int8")
    # size90plus（旧90分位后缀）仅在以下季度/机制工作列中代表本次主组。
    # 独立的size90plus（真实市值分组）仍保留原含义，不进入正式公司对输出。
    for suffix, eligible_col in [
        ("size90plus", "mainCueEligible"), ("size50_90", "size50_90"),
        ("size10_50", "size10_50"), ("size10minus", "size10minus"),
    ]:
        mm[f"quarterLag1{suffix}"] = (mm["quarterLag1Alt"] * mm[eligible_col]).astype("int8")
else:
    cue_definition = CURRENT_DEFINITION
    mm["size90plus"] = (
        mm["mktcap_tm3"] > mm["marketCap_90th_percentile"]
    ).astype(int)

    if cue_definition == "size90":
        mm["mainCueEligible"] = mm["size90plus"]
    elif cue_definition == "hs300":
        if "isHS300" not in mm.columns:
            raise ValueError("尚未匹配历史沪深300成分资格，不能构造沪深300提示。")
        mm["mainCueEligible"] = mm["isHS300"]
    else:
        raise ValueError("提示公司口径只能是 size90（市值90分位）或 hs300（沪深300）。")

    mm["size50_90"] = (
        (mm["mktcap_tm3"] > mm["marketCap_50th_percentile"])
        & (mm["mktcap_tm3"] <= mm["marketCap_90th_percentile"])
    ).astype(int)

    mm["size10_50"] = (
        (mm["mktcap_tm3"] >= mm["marketCap_10th_percentile"])
        & (mm["mktcap_tm3"] <= mm["marketCap_50th_percentile"])
    ).astype(int)

    mm["size10minus"] = (
        mm["mktcap_tm3"] < mm["marketCap_10th_percentile"]
    ).astype(int)

    # 主组内部沿用旧计算列名，实际资格由上面的开关决定。
    # 沪深300模式最终输出时，所有主组列统一改为沪深300名称。
    # 辅助规模组始终按真实市值划分，与沪深300主组可能重叠。
    # 当前构造和后续汇总均使用上一季度关系。
    max_lag = 1

    for i in range(1, max_lag + 1):
        lag_col = f"quarterLag{i}Alt"

        if lag_col not in mm.columns:
            print(f"跳过：{lag_col} 不在当前表中")
            continue

        mm[f"quarterLag{i}size90plus"] = (
            mm[lag_col] * mm["mainCueEligible"]
        ).astype("int8")

        mm[f"quarterLag{i}size50_90"] = (
            mm[lag_col] * mm["size50_90"]
        ).astype("int8")

        mm[f"quarterLag{i}size10_50"] = (
            mm[lag_col] * mm["size10_50"]
        ).astype("int8")

        mm[f"quarterLag{i}size10minus"] = (
            mm[lag_col] * mm["size10minus"]
        ).astype("int8")
print("主提示资格成立的公司对数：", int(mm["mainCueEligible"].sum()))
display(mm[["Stkcd", "event_Trddt", "StkcdRecall", "mainCueEligible", "quarterLag1size90plus"]].head())


# %% 05. 历史注意力、成交量、盈余惊喜和价格异动机制
top_mover_mode = TOP_MOVER_MODE
max_lag = 1

# 左侧为内部工作列后缀，右侧为实际使用的资格标记。
size_groups = {
    "size90plus": "mainCueEligible",
    "size50_90": "size50_90",
}

# 一、历史注意力和成交量标记与规模组交互。
attention_volume_suffixes = [
    "hiAEV",
    "hiavol",
    "hiavolS",
]

for i in range(1, max_lag + 1):
    for size, eligibility_col in size_groups.items():
        for suffix in attention_volume_suffixes:
            src = f"lag{i}{suffix}"
            dst = f"lag{i}{suffix}{size}"

            if src in mm.columns and eligibility_col in mm.columns:
                mm[dst] = (
                    mm[src] * mm[eligibility_col]
                ).astype("int8")

# 二、历史盈余惊喜标记与规模组交互。
sue_suffixes = [
    "hi_sue1_abs",
    "hi_sue2_abs",
    "hi_sue3_abs",
    "sue1pos",
    "sue2pos",
    "sue3pos",
    "sue1neg",
    "sue2neg",
    "sue3neg",
    # 双方同为极端正向或极端负向盈余惊喜。
    "hi_sue1_pos",
    "hi_sue2_pos",
    "hi_sue3_pos",
    "hi_sue1_neg",
    "hi_sue2_neg",
    "hi_sue3_neg",
]

for i in range(1, max_lag + 1):
    for size, eligibility_col in size_groups.items():
        for suffix in sue_suffixes:
            src = f"lag{i}{suffix}"
            dst = f"lag{i}{suffix}{size}"

            if src in mm.columns and eligibility_col in mm.columns:
                mm[dst] = (
                    mm[src] * mm[eligibility_col]
                ).astype("int8")

# 三、按所选历史价格异动口径，与规模组交互。
top_mover_sources = {
    "any": "anyTopMover",
    "both": "topMover",
}
top_mover_labels = {
    "any": "任意一家进入价格异动组",
    "both": "两家公司同时进入价格异动组",
}

if top_mover_mode not in top_mover_sources:
    raise ValueError(
        "历史价格异动口径只能设置为 any（任意一家）"
        "或 both（两家同时）"
    )

source_suffix = top_mover_sources[top_mover_mode]

for i in range(1, max_lag + 1):
    src = f"lag{i}{source_suffix}"

    # 所选口径缺少来源列时明确报错，避免静默跳过。
    if src not in mm.columns:
        raise ValueError(
            f"所选价格异动口径缺少来源列：{src}，"
            "请检查上游记忆矩阵。"
        )

    for size, eligibility_col in size_groups.items():
        # 输出列名保持统一，具体含义由所选口径决定。
        dst = f"lag{i}topMover{size}"

        if eligibility_col in mm.columns:
            mm[dst] = (
                mm[src] * mm[eligibility_col]
            ).astype("int8")

print(
    "历史价格异动口径：",
    top_mover_labels[top_mover_mode],
)
print("历史机制构造后：", mm.shape)
display(mm.head())


# %% 06. 当前盈余惊喜分子、有效分母和来源诊断
max_lag = 1

for sue_col in ["sue1", "sue2", "sue3"]:
    for i in range(1, max_lag + 1):
        lag_col = f"quarterLag{i}Alt"

        # 等权平均所需的逐行分子。
        mm[f"{sue_col}Lag{i}S90p"] = (
            mm[sue_col]
            * mm[lag_col]
            * mm["mainCueEligible"]
        )

        # 市值加权平均所需的逐行分子。
        mm[f"{sue_col}Lag{i}XmarketCapS90p"] = (
            mm[sue_col]
            * mm[lag_col]
            * mm["mktcap_tm3"]
            * mm["mainCueEligible"]
        )

# 始终准备各自的非缺失提示数量和有效市值。
# 后续聚合并计算平均时，再按配置选择分母口径。
for sue_id in [1, 2, 3]:
    valid_cue = (
        mm["quarterLag1size90plus"].eq(1)
        & mm[f"sue{sue_id}"].notna()
    )
    mm[f"validSue{sue_id}CueLag1S90p"] = valid_cue.astype("int8")
    mm[f"validSue{sue_id}CueMarketCapLag1S90p"] = np.where(
        valid_cue,
        mm["mktcap_tm3"],
        0.0,
    )

# 来源诊断单独保留，不改变每个提示公司在主盈余惊喜指标中只有一票的口径。
mm["lag1CueReportSourcesS90p"] = mm["lag1CueReportSources"].where(
    mm["quarterLag1size90plus"].eq(1), ""
)
for anchor in ["Annual", "Q1"]:
    source_cue = mm["quarterLag1size90plus"].eq(1) & mm[f"lag1{anchor}Anchor"].eq(1)
    mm[f"lag1{anchor}Anchorsize90plus"] = source_cue.astype("int8")
    for sue_id in [1, 2, 3]:
        source_sue = f"sue{sue_id}{anchor}Anchor"
        mm[f"{source_sue}Lag1S90p"] = mm[source_sue].where(source_cue)
        mm[f"{source_sue}ValidCueLag1S90p"] = (
            source_cue & mm[source_sue].notna()
        ).astype("int8")
print("盈余惊喜输入构造后：", mm.shape)
display(mm.head())


# %% 07. 市值分母和历史共同公告数量输入
max_lag = 1

# 一、提示公司市值：供后续市值加权平均作分母。
for i in range(1, max_lag + 1):
    mm[f"marketCap_Minus3Lag{i}S90p"] = (
        mm["mktcap_tm3"]
        * mm[f"quarterLag{i}size90plus"]
    )

    mm[f"marketCap_Minus3Lag{i}S5090"] = (
        mm["mktcap_tm3"]
        * mm[f"quarterLag{i}size50_90"]
    )

# 二、历史共同公告数量：供后续等权平均作分子。
# 来源变量已包含对应季度的关系条件。
for i in range(1, max_lag + 1):
    nr_col = f"nrFirmsAnnLag{i}Alt"

    mm[f"nrFirmsAnnLag{i}S90p"] = (
        mm[nr_col] * mm["mainCueEligible"]
    )

    mm[f"nrFirmsAnnLag{i}S5090"] = (
        mm[nr_col] * mm["size50_90"]
    )

# 三、历史数量乘提示公司市值：供市值加权平均作分子。
for i in range(1, max_lag + 1):
    nr_col = f"nrFirmsAnnLag{i}Alt"

    mm[f"nrFirmsAnnLag{i}XmarketCapS90p"] = (
        mm[nr_col]
        * mm["mktcap_tm3"]
        * mm["mainCueEligible"]
    )

    mm[f"nrFirmsAnnLag{i}XmarketCapS5090"] = (
        mm[nr_col]
        * mm["mktcap_tm3"]
        * mm["size50_90"]
    )
print("聚合输入构造后：", mm.shape)
display(mm.head())


# %% 08. 排序并提取合格提示公司对
sort_cols = [
    "Stkcd",
    "Annodt",
    "StkcdRecall",
]

sort_cols_existing = [
    col for col in sort_cols
    if col in mm.columns
]

mm = (
    mm.sort_values(sort_cols_existing)
    .reset_index(drop=True)
)

# 只提取上一季度关系成立、且满足所选主提示资格的配对。
cue_pairs = (
    mm.loc[
        mm["quarterLag1size90plus"].eq(1),
        ["Stkcd", "event_Trddt", "StkcdRecall"],
    ]
    .rename(columns={"event_Trddt": "Trddt"})
    .dropna(subset=["Stkcd", "StkcdRecall", "Trddt"])
    .drop_duplicates(["Stkcd", "Trddt", "StkcdRecall"])
    .copy()
)
print("合格提示公司对：", cue_pairs.shape)
display(cue_pairs.head())


# %% 09. 聚合到被提示公司×事件交易日，计算均值
denominator_mode = SUE_DENOMINATOR_MODE
df = mm.copy()
group_keys = ["StkcdRecall", "event_Trddt"]
if df.duplicated(["Stkcd", "event_Trddt", "StkcdRecall"]).any():
    raise ValueError("公司对交易日尚未去重，不能进行提示数量或市值权重求和。")

# 一、按原顺序整理聚合工作表。
df = df.sort_values(
    ["StkcdRecall", "event_Trddt", "mktcap_tm3", "Stkcd"]
).reset_index(drop=True)

# 二、构造规则并检查必需字段。
agg_dict = build_pattern_cue_agg_specs(df.columns)

needed_cols = set(group_keys)
for _, value in agg_dict.items():
    needed_cols.add(value[0])

missing_cols = [
    col for col in needed_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"以下变量缺失，无法聚合：{missing_cols}"
    )

# 三、按被提示公司与事件交易日聚合。
collapsed = (
    df.groupby(group_keys, dropna=False)
    .agg(**agg_dict)
    .reset_index()
)

# 四、按传入的分母口径计算盈余惊喜均值。
valid_modes = {"nonmissing_sue", "all_cues"}
if denominator_mode not in valid_modes:
    raise ValueError(
        "SUE_DENOMINATOR_MODE 只能是 "
        f"{valid_modes}，当前是 {denominator_mode}"
    )

for sue_id in [1, 2, 3]:
    if denominator_mode == "nonmissing_sue":
        ew_denominator = collapsed[
            f"sumSue{sue_id}ValidCueCountLag1S90p"
        ]
        vw_denominator = collapsed[
            f"sumSue{sue_id}ValidCueMarketCapLag1S90p"
        ]
    else:
        ew_denominator = collapsed["nrCuesLag1S90p"]
        vw_denominator = collapsed["sumMarketCap_Minus3Lag1S90p"]

    collapsed[f"sue{sue_id}CueLag1S90pEW"] = (
        collapsed[f"sumSue{sue_id}Lag1S90p"]
        .div(ew_denominator)
        .where(ew_denominator.ne(0), np.nan)
    )
    collapsed[f"sue{sue_id}CueLag1S90pVW"] = (
        collapsed[f"sumSue{sue_id}Lag1XmarketCapS90p"]
        .div(vw_denominator)
        .where(vw_denominator.ne(0), np.nan)
    )
print("盈余惊喜聚合分母口径：", denominator_mode)

# 来源分列仅作诊断，始终按该来源非缺失的提示公司等权。
# 同一公司支持两种来源时，在各自来源列各出现一次，在主指标仍只出现一次。
for anchor in ["Annual", "Q1"]:
    for sue_id in [1, 2, 3]:
        numerator = collapsed[f"sumSue{sue_id}{anchor}AnchorLag1S90p"]
        denominator = collapsed[f"sumSue{sue_id}{anchor}AnchorValidCueLag1S90p"]
        collapsed[f"sue{sue_id}{anchor}AnchorCueLag1S90pEW"] = (
            numerator.div(denominator).where(denominator.ne(0), np.nan)
        )

# 五、计算历史共同公告数量的市值加权均值。
collapsed["nrFirmsAnnLag1VWS90p"] = np.where(
    collapsed["sumMarketCap_Minus3Lag1S90p"] != 0,
    collapsed["sumNrFirmsAnnLag1XmarketCapS90p"]
    / collapsed["sumMarketCap_Minus3Lag1S90p"],
    np.nan,
)

collapsed["nrFirmsAnnLag1VWS5090"] = np.where(
    collapsed["sumMarketCap_Minus3Lag1S5090"] != 0,
    collapsed["sumNrFirmsAnnLag1XmarketCapS5090"]
    / collapsed["sumMarketCap_Minus3Lag1S5090"],
    np.nan,
)

# 六、计算历史共同公告数量的等权均值。
collapsed["nrFirmsAnnLag1EWS90p"] = np.where(
    collapsed["nrCuesLag1S90p"] != 0,
    collapsed["sumNrFirmsAnnLag1S90p"]
    / collapsed["nrCuesLag1S90p"],
    np.nan,
)

collapsed["nrFirmsAnnLag1EWS5090"] = np.where(
    collapsed["nrCuesLag1S5090"] != 0,
    collapsed["sumNrFirmsAnnLag1S5090"]
    / collapsed["nrCuesLag1S5090"],
    np.nan,
)

# 七、保留供后续日度面板合并使用的日期别名。
collapsed["date"] = collapsed["event_Trddt"]

# 八、保留原有诊断输出。
print("聚合前表结构：", df.shape)
print("聚合后表结构：", collapsed.shape)
print(
    "聚合后重复键数量：",
    collapsed.duplicated(group_keys).sum(),
)

print("\n核心变量合计：")
check_cols = [
    "nrCuesLag1S90p",
    "nrCuesLag1S5090",
    "nrCuesLag1S1050",
    "nrCuesLag1S10m",
    "quarterLag1size90plus",
    "lag1hiAEVsize90plus",
    "lag1hiavolsize90plus",
    "lag1hi_sue1_possize90plus",
    "lag1hi_sue1_negsize90plus",
    "lag1hi_sue2_possize90plus",
    "lag1hi_sue2_negsize90plus",
    "lag1topMoversize90plus",
]
check_cols = [
    col for col in check_cols
    if col in collapsed.columns
]
print(
    collapsed[check_cols]
    .sum()
    .sort_values(ascending=False)
)

collapsed = (
    collapsed.sort_values(["StkcdRecall", "date"])
    .reset_index(drop=True)
)
events = collapsed
print("聚合后提示事件：", events.shape)
display(events.head())


# %% 10. 统一字段并检查输出粒度
events = events.copy()
events["cueDefinition"] = CURRENT_DEFINITION
events["industryTopN"] = int(CURRENT_TOP_N) if CURRENT_TOP_N is not None else pd.NA
events["cueColumn"] = "quarterLag1size90plus"
events["date"] = pd.to_datetime(events["date"])

cue_pairs = cue_pairs.copy()
cue_pairs["cueDefinition"] = CURRENT_DEFINITION
cue_pairs["industryTopN"] = int(CURRENT_TOP_N) if CURRENT_TOP_N is not None else pd.NA
cue_pairs = cue_pairs.rename(columns={"event_Trddt": "Trddt"})
print("提示事件主键重复数：", events.duplicated(["StkcdRecall", "date"]).sum())
display(events.head(20))


# %% 11. 保存当前口径的相同提示事件、合格公司对与摘要
label = f"行业前{CURRENT_TOP_N}名" if CURRENT_DEFINITION == "industry_top" else CURRENT_DEFINITION
folder = stage_root(PROJECT_ROOT) / "提示事件" / label
folder.mkdir(parents=True, exist_ok=True)
event_path = folder / "提示事件.csv"
pair_path = folder / "合格提示公司对.csv"
summary_path = folder / "提示事件审计摘要.json"
events.to_csv(event_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
cue_pairs.to_csv(pair_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
summary = {
    "stage": 2,
    "cue_definition": CURRENT_DEFINITION,
    "industry_top_n": CURRENT_TOP_N,
    "grain": "StkcdRecall × date",
    "rows": int(len(events)),
    "eligible_pair_rows": int(len(cue_pairs)),
    "key_duplicate_count": int(events.duplicated(["StkcdRecall", "date"]).sum()),
    "event_path": str(event_path),
    "pair_path": str(pair_path),
    "next_stage": "03_合并提示最终面板.py",
}
save_summary(summary_path, summary)
print("提示事件输出：", event_path)
print("合格提示公司对输出：", pair_path)
# %%
