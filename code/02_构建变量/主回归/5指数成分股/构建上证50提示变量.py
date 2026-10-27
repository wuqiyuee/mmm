"""构造上证50提示变量，单独输出，不覆盖模式公司最终回归面板。

目标：只将“提示公司总市值超过90分位”替换为“事件日属于上证50”。
历史关系直接使用当前模式公司记忆矩阵中的滞后一期标记；行业、供应链、
共同分析师过滤复用正式合并脚本，不读取提示公司市值或月度市值断点。

处理顺序：
1. 读取记忆矩阵，保留2005—2025年事件交易日及滞后一期关系。
2. 按提示公司代码和事件交易日，匹配已经生成的上证50交易日成分名单。
3. 按自然公告日公司对，排除同行业、供应链和共同分析师覆盖关系。
4. 合并同一公司对交易日的多个自然公告来源，再汇总提示公司数量。
5. 以现有主面板的被提示公司—日期键为左表，合并新提示标记并补0。

输出目录：中间结果/02_变量构造结果/9指数成分股/上证50/
- 上证50_合格提示公司对.csv：一行是一家提示公司在某交易日对一家被提示
  公司的合格提示关系；保留自然公告来源及是否进入当前主样本。
- 上证50_提示变量_被提示公司日度.csv：一行对应原主面板的一家被提示公司
  在一个交易日的观测，只保留键、新提示标记及原90分位标记供后续比较。

核心字段：
- Stkcd：提示公司代码；上证50成员身份匹配这一列，不是StkcdRecall。
- StkcdRecall：被提示公司代码，不限制其是否属于上证50。
- Trddt / td：事件交易日期 / 项目统一交易日序号。
- quarterLag1sse50：至少一家合格上证50提示公司时为1，否则为0。
- nrCuesLag1SSE50：当日不同的合格上证50提示公司数量，重复来源不重复计数。
- quarterLag1size90plus：原主面板的90分位提示标记，原样保留、不覆盖。
- sourceAnnodts：构成该公司对交易日的自然公告日期，以竖线连接。
- in_main_sample：该合格公司对的被提示公司—日期是否在当前主样本中。

重要口径：
1. 使用现有event_Trddt（事件交易日），不把公告再向后移动一天。
2. 不额外要求当前日有至少两家模式公司或至少两家上证50公司公告。
3. 在自然公告日层面完成过滤后才按交易日去重，避免混淆过滤日期。
4. 保留原主面板研究期内全部观测，包括没有新提示的0值行；不重新筛收益、
   控制变量或固定效应，也不估计回归、不修改其他脚本。
5. 提示变量值缺失补0的前提是上证50名单完整覆盖研究期交易日。
6. 读取输出时，股票代码应按字符串读取，保留前导0。
"""

import importlib.util
from pathlib import Path

import pandas as pd


# 第一部分：路径和研究期。默认不覆盖已存在的结果。
ROOT = Path(__file__).resolve().parents[4]
FILTER_SCRIPT = ROOT / "code/04_合并数据/合并pattern_firms的数据.py"
MATRIX_PATH = ROOT / "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv"
CALENDAR_PATH = ROOT / "中间结果/01_清洗后样本/市场交易日历_td.csv"
PANEL_PATH = ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"
INDUSTRY_PATH = ROOT / "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv"
SUPPLY_CHAIN_PATH = (
    ROOT / "input/CSMAR/供应链网络关系指标表2001-12-31至2025-12-31/SC_NetworkRelationsIndex.csv"
)
FORECAST_PATH = ROOT / "中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv"
ANALYST_ROOT = ROOT / "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后"
ANALYST_PATH = ANALYST_ROOT / "拆开AnanmID/Pattern样本_分析师共同覆盖_滚动365天.csv"

OUTPUT_DIR = ROOT / "中间结果/02_变量构造结果/9指数成分股/上证50"
MEMBERSHIP_PATH = OUTPUT_DIR / "上证50_交易日成分名单.csv"
PAIR_OUTPUT_PATH = OUTPUT_DIR / "上证50_合格提示公司对.csv"
FLAG_OUTPUT_PATH = OUTPUT_DIR / "上证50_提示变量_被提示公司日度.csv"

START_DATE = pd.Timestamp("2005-01-01")
END_DATE = pd.Timestamp("2025-12-31")
CHUNK_SIZE = 200_000
OVERWRITE_OUTPUT = False


def normalize_codes(values):
    """统一六位股票代码；无效代码报错，不静默当作非成员。"""
    codes = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if not codes.str.fullmatch(r"\d{6}", na=False).all():
        raise ValueError("存在缺失或非六位数字股票代码，请核查输入。")
    return codes


def require_unique_key(table, columns, name):
    duplicated = table.duplicated(columns, keep=False)
    if duplicated.any():
        examples = table.loc[duplicated, columns].head(10).to_string(index=False)
        raise ValueError(f"{name}存在重复键：\n{examples}")


def load_filter_functions():
    """仅导入正式合并脚本中的函数，不调用其主流程。"""
    specification = importlib.util.spec_from_file_location("pattern_pair_filters", FILTER_SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


# 第二部分：核对上证50成员名单的日期覆盖和交易日序号。
def load_calendar_and_membership():
    calendar = pd.read_csv(CALENDAR_PATH, usecols=["Trddt", "td"])
    calendar["Trddt"] = pd.to_datetime(calendar["Trddt"], errors="raise")
    calendar["td"] = pd.to_numeric(calendar["td"], errors="raise")
    if calendar[["Trddt", "td"]].isna().any().any() or calendar["td"].mod(1).ne(0).any():
        raise ValueError("交易日历存在缺失日期或非整数交易日序号。")
    calendar["td"] = calendar["td"].astype("int64")
    calendar = calendar.drop_duplicates().sort_values("Trddt").reset_index(drop=True)
    require_unique_key(calendar, ["Trddt"], "交易日历日期")
    require_unique_key(calendar, ["td"], "交易日历序号")
    if calendar.empty or calendar["Trddt"].min() > START_DATE or calendar["Trddt"].max() < END_DATE:
        raise ValueError("交易日历没有完整覆盖研究期。")
    calendar = calendar.loc[calendar["Trddt"].between(START_DATE, END_DATE)].copy()
    if calendar.empty or not calendar["td"].diff().iloc[1:].eq(1).all():
        raise ValueError("研究期交易日历为空或交易日序号不连续。")

    members = pd.read_csv(
        MEMBERSHIP_PATH, dtype={"Stkcd": "string", "Indexcd": "string"},
        usecols=["Stkcd", "Trddt", "td", "Indexcd", "is_sse50"],
    )
    members["Stkcd"] = normalize_codes(members["Stkcd"])
    members["Trddt"] = pd.to_datetime(members["Trddt"], errors="raise")
    if members["Trddt"].isna().any():
        raise ValueError("上证50名单存在缺失日期。")
    if not members["Indexcd"].eq("000016").all() or not members["is_sse50"].eq(1).all():
        raise ValueError("成员名单必须只包含指数000016的正成员记录。")
    members = members.loc[members["Trddt"].between(START_DATE, END_DATE)].copy()
    require_unique_key(members, ["Stkcd", "Trddt"], "上证50成员名单")
    counts = members.groupby("Trddt")["Stkcd"].nunique().reindex(calendar["Trddt"])
    if not counts.eq(50).all():
        raise ValueError("上证50名单未完整覆盖研究期，或部分交易日不是50只成员。")
    members = members.merge(
        calendar.rename(columns={"td": "calendar_td"}),
        on="Trddt", how="left", validate="m:1",
    )
    if not members["td"].eq(members["calendar_td"]).all():
        raise ValueError("上证50名单与项目交易日历的日期或序号不一致。")
    members = members[["Stkcd", "Trddt", "is_sse50"]].rename(
        columns={"Trddt": "event_Trddt"}
    )
    return calendar, members


# 第三部分：筛选滞后一期关系，再按提示公司匹配上证50身份。
def read_sse50_candidates(calendar, members):
    columns = ["Annodt", "event_Trddt", "event_td", "Stkcd", "StkcdRecall", "quarterLag1Alt"]
    pieces = []
    for number, chunk in enumerate(pd.read_csv(
        MATRIX_PATH, usecols=columns, encoding="utf-8-sig", chunksize=CHUNK_SIZE,
        dtype={"Stkcd": "string", "StkcdRecall": "string", "quarterLag1Alt": "Int8"},
        parse_dates=["Annodt", "event_Trddt"],
    ), start=1):
        lag1 = chunk.loc[chunk["quarterLag1Alt"].eq(1)].copy()
        if lag1[["Annodt", "event_Trddt", "event_td"]].isna().any().any():
            raise ValueError("滞后一期记忆关系存在缺失事件日期或交易日序号。")
        lag1 = lag1.loc[lag1["event_Trddt"].between(START_DATE, END_DATE)].copy()
        if not lag1.empty:
            pieces.append(lag1)
        if number % 10 == 0:
            print(f"已读取记忆矩阵{number}个数据块。", flush=True)
    if not pieces:
        print("研究期内没有滞后一期记忆关系，将保留原主样本并生成全0新标记。", flush=True)
        return pd.DataFrame(columns=columns + ["is_sse50"])

    candidates = pd.concat(pieces, ignore_index=True)
    candidates["Stkcd"] = normalize_codes(candidates["Stkcd"])
    candidates["StkcdRecall"] = normalize_codes(candidates["StkcdRecall"])
    require_unique_key(candidates, ["Stkcd", "Annodt", "StkcdRecall"], "记忆矩阵自然公告日公司对")
    if candidates["Stkcd"].eq(candidates["StkcdRecall"]).any():
        raise ValueError("记忆矩阵存在提示公司与被提示公司相同的自配对。")
    candidates = candidates.merge(
        calendar.rename(columns={"Trddt": "event_Trddt", "td": "calendar_td"}),
        on="event_Trddt", how="left", validate="m:1",
    )
    if not candidates["event_td"].eq(candidates["calendar_td"]).all():
        raise ValueError("记忆矩阵的事件日期与项目交易日序号不一致。")
    if candidates["Annodt"].gt(candidates["event_Trddt"]).any():
        raise ValueError("存在事件交易日早于自然公告日的记录。")
    candidates = candidates.drop(columns="calendar_td")
    candidates["event_td"] = candidates["event_td"].astype("int64")

    print(f"研究期滞后一期候选公司对：{len(candidates):,}条。", flush=True)
    candidates = candidates.merge(
        members, on=["Stkcd", "event_Trddt"], how="inner", validate="m:1",
    )
    print(f"提示公司属于事件日上证50的候选公司对：{len(candidates):,}条。", flush=True)
    return candidates


# 第四部分：只复用公司对排除函数；自然日过滤完成后才合并交易日来源。
def filter_and_collapse_pairs(candidates, helpers):
    # 明确空结果的列和类型，后续仍可按日期、代码及序号与原主样本合并。
    empty_pairs = pd.DataFrame({
        "Stkcd": pd.Series(dtype="string"),
        "Trddt": pd.Series(dtype="datetime64[ns]"),
        "StkcdRecall": pd.Series(dtype="string"),
        "td": pd.Series(dtype="int64"),
        "sourceAnnodts": pd.Series(dtype="string"),
        "quarterLag1Alt": pd.Series(dtype="int8"),
        "is_sse50": pd.Series(dtype="int8"),
    })
    if candidates.empty:
        print("没有上证50候选公司对，合格提示公司对表为空。", flush=True)
        return empty_pairs
    keys = helpers.filter_same_industry_pairs(candidates, INDUSTRY_PATH)
    keys = helpers.filter_supply_chain_pairs(keys, SUPPLY_CHAIN_PATH)
    keys = helpers.filter_analyst_overlap_pairs(
        mm_key=keys, overlap_root=ANALYST_ROOT, id_mode="split", lookback_days=365,
        drop_overlap=True, matrix_path=MATRIX_PATH, forecast_path=FORECAST_PATH,
    )
    if keys.empty:
        print("公司对过滤后没有合格关系，将生成全0新标记。", flush=True)
        return empty_pairs
    accepted = helpers.apply_pair_filters(candidates, keys)
    accepted["sourceAnnodts"] = accepted["Annodt"].dt.strftime("%Y-%m-%d")
    pair_keys = ["Stkcd", "event_Trddt", "StkcdRecall"]
    pairs = accepted.groupby(pair_keys, as_index=False).agg(
        td=("event_td", "first"),
        sourceAnnodts=("sourceAnnodts", helpers.union_pipe_values),
    ).rename(columns={"event_Trddt": "Trddt"})
    pairs["quarterLag1Alt"] = 1
    pairs["is_sse50"] = 1
    require_unique_key(pairs, ["Stkcd", "Trddt", "StkcdRecall"], "合格提示公司对交易日")
    print(f"过滤后合格提示公司对交易日：{len(pairs):,}条。", flush=True)
    return pairs


# 第五部分：原主面板只读取键和原提示标记，保留原来的全部0值观察。
def read_main_sample(calendar):
    columns = ["StkcdRecall", "Trddt", "td", "quarterLag1size90plus"]
    pieces = []
    for number, chunk in enumerate(pd.read_csv(
        PANEL_PATH, usecols=columns, encoding="utf-8-sig", chunksize=CHUNK_SIZE,
        dtype={"StkcdRecall": "string"}, parse_dates=["Trddt"],
    ), start=1):
        if chunk["Trddt"].isna().any():
            raise ValueError("原主面板存在缺失交易日期。")
        selected = chunk.loc[chunk["Trddt"].between(START_DATE, END_DATE)].copy()
        if not selected.empty:
            selected["StkcdRecall"] = normalize_codes(selected["StkcdRecall"])
            pieces.append(selected)
        if number % 10 == 0:
            print(f"已读取原主面板{number}个数据块。", flush=True)
    if not pieces:
        raise ValueError("原主面板在研究期内没有观测。")
    sample = pd.concat(pieces, ignore_index=True)
    require_unique_key(sample, ["StkcdRecall", "Trddt"], "原主面板公司日期")
    if not sample["quarterLag1size90plus"].isin([0, 1]).all():
        raise ValueError("原主面板的90分位提示标记存在缺失或非0/1值。")
    sample = sample.merge(
        calendar.rename(columns={"td": "calendar_td"}),
        on="Trddt", how="left", validate="m:1",
    )
    if not sample["td"].eq(sample["calendar_td"]).all():
        raise ValueError("原主面板日期或交易日序号与项目日历不一致。")
    sample = sample.drop(columns="calendar_td")
    sample["td"] = sample["td"].astype("int64")
    sample["quarterLag1size90plus"] = sample["quarterLag1size90plus"].astype("int8")
    return sample


def merge_flags(sample, pairs):
    key_columns = ["StkcdRecall", "Trddt", "td"]
    lookup = sample[key_columns].assign(in_main_sample=1)
    pairs = pairs.merge(lookup, on=key_columns, how="left", validate="m:1")
    pairs["in_main_sample"] = pairs["in_main_sample"].fillna(0).astype("int8")

    # 每个公司对交易日已经唯一，因此不同来源不会重复增加提示公司数量。
    events = pairs.groupby(key_columns, as_index=False).agg(
        nrCuesLag1SSE50=("Stkcd", "nunique"),
    )
    flags = sample.merge(events, on=key_columns, how="left", validate="1:1")
    flags["nrCuesLag1SSE50"] = flags["nrCuesLag1SSE50"].fillna(0).astype("int16")
    if not flags["nrCuesLag1SSE50"].between(0, 50).all():
        raise ValueError("合格提示公司数量不在0至50之间。")
    flags["quarterLag1sse50"] = flags["nrCuesLag1SSE50"].gt(0).astype("int8")
    if len(flags) != len(sample):
        raise ValueError("合并新提示标记改变了原主样本行数。")
    require_unique_key(flags, ["StkcdRecall", "Trddt"], "新提示标记表")
    flags = flags[[
        "StkcdRecall", "Trddt", "td", "quarterLag1sse50",
        "nrCuesLag1SSE50", "quarterLag1size90plus",
    ]]
    pairs = pairs[[
        "Stkcd", "StkcdRecall", "Trddt", "td", "sourceAnnodts",
        "quarterLag1Alt", "is_sse50", "in_main_sample",
    ]].sort_values(["StkcdRecall", "Trddt", "Stkcd"]).reset_index(drop=True)
    return flags, pairs


def print_comparison(flags, pairs):
    old = flags["quarterLag1size90plus"].eq(1)
    new = flags["quarterLag1sse50"].eq(1)
    print(f"原主样本观测数保持为：{len(flags):,}。")
    print(f"原90分位提示日：{int(old.sum()):,}；上证50提示日：{int(new.sum()):,}。")
    print(f"两者都有：{int((old & new).sum()):,}。")
    print(f"仅原90分位有：{int((old & ~new).sum()):,}；仅上证50有：{int((~old & new).sum()):,}。")
    print(f"合格提示公司对中进入当前主样本的记录：{int(pairs['in_main_sample'].sum()):,}。")
    yearly = flags.assign(
        year=flags["Trddt"].dt.year, both=(old & new).astype("int8"),
    ).groupby("year").agg(
        观测数=("StkcdRecall", "size"),
        原90分位提示日=("quarterLag1size90plus", "sum"),
        上证50提示日=("quarterLag1sse50", "sum"),
        两者都有=("both", "sum"),
    )
    print("各年度提示日对照：\n" + yearly.to_string())


# 第六部分：只有直接运行本脚本时，才读取数据并输出两张独立结果。
def main():
    targets = [PAIR_OUTPUT_PATH, FLAG_OUTPUT_PATH]
    existing = [path for path in targets if path.exists()]
    if existing and not OVERWRITE_OUTPUT:
        raise FileExistsError(
            "以下结果已存在，默认不覆盖：\n"
            + "\n".join(str(path) for path in existing)
            + "\n确认重新生成后，可将OVERWRITE_OUTPUT（允许覆盖）改为True。"
        )
    for path in [
        FILTER_SCRIPT, MATRIX_PATH, CALENDAR_PATH, PANEL_PATH, INDUSTRY_PATH,
        SUPPLY_CHAIN_PATH, FORECAST_PATH, ANALYST_PATH, MEMBERSHIP_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"缺少必要输入：{path}")
    for source in [MATRIX_PATH, FORECAST_PATH]:
        if ANALYST_PATH.stat().st_mtime_ns < source.stat().st_mtime_ns:
            raise ValueError("共同分析师覆盖结果早于输入，请先基于当前输入重新识别共同覆盖。")
    for source in [MATRIX_PATH, ANALYST_PATH]:
        if PANEL_PATH.stat().st_mtime_ns < source.stat().st_mtime_ns:
            raise ValueError("原主面板早于记忆矩阵或共同分析师结果，请先确认并更新主面板。")

    helpers = load_filter_functions()
    calendar, members = load_calendar_and_membership()
    candidates = read_sse50_candidates(calendar, members)
    pairs = filter_and_collapse_pairs(candidates, helpers)
    print("读取原主面板的公司—日期键；不读取收益和控制变量……", flush=True)
    sample = read_main_sample(calendar)
    flags, pairs = merge_flags(sample, pairs)
    print_comparison(flags, pairs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for table, path in [(pairs, PAIR_OUTPUT_PATH), (flags, FLAG_OUTPUT_PATH)]:
        table.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        print(f"已输出：{path}，{len(table):,}行。")
    print("原主面板及原始输入文件均未修改；本脚本不执行回归。")


if __name__ == "__main__":
    main()
