"""第三阶段：逐单元检查提示事件与最终日度面板的合并。

本文件直接展开各项计算，只导入标准库及第三方库，不加载普通版或其他项目脚本。
按 # %% 分段运行可依次检查表结构，参数、处理顺序、字段和正式输出与普通版一致。
CSMAR（国泰安数据库）的Annodt（公告日期）只在非交易日映射至下一交易日。
"""

# %% 00. 导入库、设置路径和研究参数
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
from IPython.display import display


PROJECT_ROOT = Path("/Volumes/苹果资料/mmm复现数据包")
if not (PROJECT_ROOT / "AGENTS.md").is_file() or not (PROJECT_ROOT / "中间结果").is_dir():
    raise FileNotFoundError("请将PROJECT_ROOT（项目根目录）设为包含AGENTS.md和中间结果的目录。")
START_YEAR = 2005
END_YEAR = 2025
REGRESSION_SAMPLE_SPEC = 5
CHUNK_SIZE = 250_000
STAGE_ROOT_NAME = "三阶段合并"
EVENT_ROOT_NAME = "提示事件"
FINAL_ROOT_NAME = "最终面板"

# 本次使用的收益来源。
RETURN_DEFINITION = "self_built_labels"

RETURN_SETTINGS = {
    "self_built_labels": (
        "自算标签收益",
        "DGTW特征调整收益率.csv",
    ),
    "official_labels": (
        "官方标签收益",
        "官方标签_特征调整收益率.csv",
    ),
}

if RETURN_DEFINITION not in RETURN_SETTINGS:
    raise ValueError("被解释变量口径只能选择自算标签收益或官方标签收益。")

RETURN_LABEL, RETURN_FILENAME = RETURN_SETTINGS[RETURN_DEFINITION]


def build_stage_paths(root: Path) -> dict:
    """列出第三阶段直接读取的正式数据文件。"""
    return {
        "TRD_Dalyr_DGTW_path": root / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv",
        "eligibility_path": root / "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv",
        "IAR_Rept_path": root / "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv",
        "performance_forecast_controls_path": root / "中间结果/02_变量构造结果/5公司公告事件/业绩预告快报控制变量_公告日期口径.csv",
        "other_event_controls_path": root / "中间结果/02_变量构造结果/5公司公告事件/专题研究事件控制变量_公告日期口径.csv",
        "AES_AEV_controls_path": root / "中间结果/02_变量构造结果/3分段新闻_AES_AEV/DATAGO_分段新闻_真实交易日控制变量.csv",
        "adjusted_return_path": (
            root
            / "中间结果/02_变量构造结果/1DGTW特征调整收益率"
            / RETURN_FILENAME
        ),
    }


def stage_root(root: Path) -> Path:
    return root / "中间结果/04_合并数据" / STAGE_ROOT_NAME


def save_summary(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# %% 01. 本文件内的股票代码、分位分组和样本筛选小工具
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


def xtile_by_group(s, n):
    """按原文默认分位规则分组，并列值不拆分。"""
    if n < 2:
        raise ValueError("分组数量至少为2。")

    out = pd.Series(
        pd.NA,
        index=s.index,
        dtype="Int64",
    )

    non_missing = s.dropna()
    m = len(non_missing)

    if m == 0:
        return out

    values = non_missing.to_numpy(dtype="float64")
    sorted_values = np.sort(values)
    cutpoints = np.empty(n - 1, dtype="float64")

    for k in range(1, n):
        lower_index, remainder = divmod(m * k, n)

        if remainder == 0:
            cutpoints[k - 1] = (
                sorted_values[lower_index - 1]
                + sorted_values[lower_index]
            ) / 2
        else:
            cutpoints[k - 1] = sorted_values[lower_index]

    groups = (
        np.searchsorted(cutpoints, values, side="left") + 1
    )

    out.loc[non_missing.index] = groups
    return out


def by_year_xtile_2(df, value_col, year_col="year"):
    """按年份分为低组1、高组2，缺失保留。"""
    return (
        df.groupby(year_col)[value_col]
        .transform(
            lambda s: xtile_by_group(s, n=2).astype("float64")
        )
    )


def xtile_nq(series, q):
    """对整列按原文规则进行分位分组，缺失保留。"""
    return xtile_by_group(series, n=q).astype("float64")


def winsorize_limits(s, p=0.01):
    """
    只计算缩尾阈值，不立刻生成缩尾后的整列。
    """
    print(
        f"[缩尾阈值] {s.name}：开始计算下侧分位点。",
        flush=True,
    )
    lower = s.quantile(p)
    print(
        f"[缩尾阈值] {s.name}：下侧分位点完成，开始计算上侧分位点。",
        flush=True,
    )
    upper = s.quantile(1 - p)
    print(
        f"[缩尾阈值] {s.name}：上下侧分位点均已完成。",
        flush=True,
    )
    return lower, upper


def add_winsor_col(df, col, winsor_limits):
    """使用传入的全样本缩尾阈值，生成缩尾后的变量。"""

    print(f"[生成缩尾列] {col}：开始。", flush=True)
    lower, upper = winsor_limits[col]
    df[f"{col}_w"] = df[col].clip(
        lower=lower,
        upper=upper,
    )
    print(f"[生成缩尾列] {col}：完成。", flush=True)


def baseline_regression_sample_mask(
    df,
    dep_col,
    rhs_cols,
    fe_cols,
    cluster_cols=None,
    regression_sample_spec=5,
):
    """检查必要变量，并按选定回归规格递归删除固定效应单例。"""

    if regression_sample_spec not in (5, 6):
        raise ValueError("合并阶段回归样本筛选规格只能为5或6。")

    cluster_cols = cluster_cols or []

    # 第一组固定效应：被提示公司×年度季度。
    fe_groups = [fe_cols]

    # 规格6另加日期固定效应，两组分别检查。
    if regression_sample_spec == 6:
        fe_groups.append(["Trddt"])

    # 一、检查所需列是否存在。
    fe_columns = []
    for group_cols in fe_groups:
        fe_columns.extend(group_cols)

    numeric_cols = [dep_col] + rhs_cols
    needed_cols = list(dict.fromkeys(
        numeric_cols + fe_columns + cluster_cols
    ))

    missing_cols = [
        column for column in needed_cols
        if column not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"缺少回归样本筛选所需变量：{missing_cols}")

    # 二、检查本轮回归必要变量是否完整、数值是否有效。
    valid = df[needed_cols].notna().all(axis=1)

    for column in numeric_cols:
        valid &= np.isfinite(
            df[column].to_numpy(dtype="float64")
        )

    # 只复制判断单例所需的列，保留原行索引。
    sample_cols = list(dict.fromkeys([dep_col] + fe_columns))
    sample = df.loc[valid, sample_cols].copy()

    # 三、分别检查各组固定效应，递归删除单例。
    total_removed = 0

    while not sample.empty:
        singleton = np.zeros(len(sample), dtype=bool)

        for group_cols in fe_groups:
            group_size = (
                sample.groupby(
                    group_cols,
                    observed=True,
                    sort=False,
                )[dep_col]
                .transform("size")
                .to_numpy()
            )
            singleton |= group_size == 1

        removed = int(singleton.sum())

        if removed == 0:
            break

        total_removed += removed
        sample = sample.loc[~singleton]

    # 四、返回与输入表逐行对应的保留标记。
    result = pd.Series(False, index=df.index)
    result.loc[sample.index] = True

    print(
        f"回归样本筛选规格（{regression_sample_spec}）："
        f"删除单例观测{total_removed:,}条，"
        f"保留{len(sample):,}条。",
        flush=True,
    )

    return result


# %% 02. 选择要检查的提示口径并读取事件表
CURRENT_DEFINITION = "size90"
CURRENT_TOP_N = None
paths = build_stage_paths(PROJECT_ROOT)
EVENT_ROOT = stage_root(PROJECT_ROOT) / EVENT_ROOT_NAME
CURRENT_LABEL = (
    f"行业前{CURRENT_TOP_N}名"
    if CURRENT_DEFINITION == "industry_top"
    else CURRENT_DEFINITION
)
EVENT_PATH = EVENT_ROOT / CURRENT_LABEL / "提示事件.csv"
if not EVENT_PATH.is_file():
    raise FileNotFoundError(f"缺少第二阶段提示事件表：{EVENT_PATH}")
# 检查所选收益文件是否存在。
if not paths["adjusted_return_path"].is_file():
    raise FileNotFoundError(
        f"缺少所选基础调整收益：{paths['adjusted_return_path']}"
    )

# 读取第二阶段提示事件的口径摘要。
EVENT_SUMMARY_PATH = EVENT_PATH.with_name("提示事件审计摘要.json")

if not EVENT_SUMMARY_PATH.is_file():
    raise FileNotFoundError(
        f"缺少第二阶段口径摘要：{EVENT_SUMMARY_PATH}"
    )

event_info = json.loads(
    EVENT_SUMMARY_PATH.read_text(encoding="utf-8")
)

# 新增字段必须由修改后的第二阶段实际生成。
required_fields = {"cue_label", "marketcap_quantile"}
missing_fields = required_fields - event_info.keys()

if missing_fields:
    raise ValueError(
        f"第二阶段摘要缺少字段：{sorted(missing_fields)}。"
        "请先用修改后的第二阶段生成对应口径。"
    )

# 核对摘要与当前选择是否对应。
if event_info["cue_definition"] != CURRENT_DEFINITION:
    raise ValueError("事件摘要的提示口径与当前选择不一致。")

if event_info["industry_top_n"] != CURRENT_TOP_N:
    raise ValueError("事件摘要的行业名额与当前选择不一致。")

if Path(event_info["event_path"]).resolve() != EVENT_PATH.resolve():
    raise ValueError("事件摘要对应的事件文件与当前选择不一致。")

# 后续直接使用上游记录的中文名称和分位门槛。
CUE_LABEL = event_info["cue_label"]
MARKETCAP_QUANTILE = event_info["marketcap_quantile"]

events = pd.read_csv(
    EVENT_PATH,
    encoding="utf-8-sig",
    parse_dates=["date"],
    dtype={"StkcdRecall": "string"},
)
events["StkcdRecall"] = events["StkcdRecall"].apply(normalize_stock_code)
if events.duplicated(["StkcdRecall", "date"]).any():
    raise ValueError("提示事件表在被提示公司×事件日层面重复。")
print("当前口径：", CURRENT_DEFINITION, "行业名额：", CURRENT_TOP_N)
print("提示事件表：", events.shape)
display(events.head())


# %% 03. 读取被提示公司日度底表并重命名收益字段
daily_panel_path = paths["TRD_Dalyr_DGTW_path"]
forward_return_cols = [
    f"raw_ret_t_t{i}"
    for i in range(1, 51)
]

base_return_cols = [
    "raw_ret_tm3",
    "raw_ret_tm2",
    "raw_ret_tm1",
    "raw_ret_t",
]

daily = pd.read_csv(
    daily_panel_path,
    encoding="utf-8",
    usecols=(
        [
            "Stkcd",
            "td",
            "Trddt",
            "Filling",
            "year",
            "mktcap_tm3",
        ]
        + base_return_cols
        + forward_return_cols
    ),
    dtype=(
        {
            "Stkcd": "string",
            "td": "int32",
            "Filling": "float32",
            "year": "float32",
            "mktcap_tm3": "float32",
        }
        | {
            column: "float32"
            for column in base_return_cols + forward_return_cols
        }
    ),
    parse_dates=["Trddt"],
)

daily["Stkcd"] = (
    daily["Stkcd"]
    .apply(normalize_stock_code)
)

# 日度面板在后续全部按被提示公司命名。
daily = (
    daily
    .rename(columns={"Stkcd": "StkcdRecall"})
    .drop(columns=["IndustryCode"], errors="ignore")
)

raw_return_rename = {
    "raw_ret_tm3": "return_tm3_Recall",
    "raw_ret_tm2": "return_tm2_Recall",
    "raw_ret_tm1": "return_tm1_Recall",
    "raw_ret_t": "return_t_Recall",
}

raw_return_rename.update({
    f"raw_ret_t_t{i}": f"return_t_tp{i}_Recall"
    for i in range(1, 51)
})

daily = daily.rename(
    columns=raw_return_rename
)

print("日度底表：", daily.shape)
display(daily.head())


# %% 04. 分块读取基础样本资格，并与日度底表一对一合并
base_path = paths["eligibility_path"]
base_parts = []
for chunk in pd.read_csv(
    base_path,
    usecols=["Stkcd", "td", "Trddt"],
    dtype={"Stkcd": "string"},
    parse_dates=["Trddt"],
    chunksize=CHUNK_SIZE,
    encoding="utf-8-sig",
):
    selected = chunk.loc[
        chunk["Trddt"].dt.year.between(START_YEAR, END_YEAR),
        ["Stkcd", "td", "Trddt"],
    ].copy()
    if not selected.empty:
        base_parts.append(selected)
if not base_parts:
    raise ValueError("指定年份没有被提示公司基础样本记录。")
base = pd.concat(base_parts, ignore_index=True)
base["StkcdRecall"] = base["Stkcd"].apply(normalize_stock_code)
base = base.drop(columns=["Stkcd"])
if base.duplicated(["StkcdRecall", "td"]).any():
    raise ValueError("被提示公司基础样本索引的公司日键重复。")
daily = daily.merge(
    base,
    on=["StkcdRecall", "td", "Trddt"],
    how="inner",
    validate="1:1",
)

print("资格筛选后：", daily.shape)
print("主键重复数：", daily.duplicated(["StkcdRecall", "Trddt"]).sum())
display(daily.head())


# %% 05. 标记自身公告，只将非交易日公告映射至下一交易日
# CSMAR（国泰安数据库）的Annodt（公告日期）已考虑收盘后公告，不再整体后移。
report_path = paths["IAR_Rept_path"]
event_date_mode = "next_if_nontrading"
calendar_path = None
window = 0
if event_date_mode not in {
    "next_if_nontrading",
    "strict_next_trading",
}:
    raise ValueError(
        "自身公告事件日口径设置无效："
        f"{event_date_mode!r}"
    )

report = pd.read_csv(
    report_path,
    usecols=["Stkcd", "Accper", "Annodt"],
    parse_dates=["Accper", "Annodt"],
)

report["Stkcd"] = (
    report["Stkcd"]
    .apply(normalize_stock_code)
)

report = report.rename(
    columns={"Stkcd": "StkcdRecall"}
)

calendar_source = (
    pd.read_csv(calendar_path, usecols=["Trddt", "td"], parse_dates=["Trddt"])
    if calendar_path is not None else daily[["Trddt", "td"]]
)
calendar = (
    calendar_source
    .drop_duplicates()
    .sort_values("Trddt")
)
if calendar_path is not None and (calendar.empty or calendar[["Trddt", "td"]].isna().any().any()
        or calendar.duplicated("Trddt").any() or calendar.duplicated("td").any()
        or not np.all(np.diff(calendar["td"].to_numpy()) == 1)):
    raise ValueError("公告映射需要日期、序号唯一且连续的正式市场日历。")

announcement_dates = (
    report[["Annodt"]]
    .dropna()
    .drop_duplicates()
    .sort_values("Annodt")
)

announcement_dates = pd.merge_asof(
    announcement_dates,
    calendar,
    left_on="Annodt",
    right_on="Trddt",
    direction="forward",
    allow_exact_matches=(
        event_date_mode == "next_if_nontrading"
    ),
)

own_ea = (
    report
    .merge(
        announcement_dates,
        on="Annodt",
        how="left",
        validate="many_to_one",
    )
    .dropna(subset=["Trddt"])
    [["StkcdRecall", "Trddt", "td"]]
    .drop_duplicates()
)

own_ea["ownEA"] = 1

daily = daily.merge(
    own_ea[["StkcdRecall", "Trddt", "ownEA"]],
    on=["StkcdRecall", "Trddt"],
    how="left",
    validate="one_to_one",
)

daily["ownEA"] = (
    daily["ownEA"]
    .fillna(0)
    .astype("int8")
)

# 主回归直接使用全部公告的市场日序号，不因公司停牌而丢失窗口内公告。
# 默认window=0（窗口为零）保留其他调用方先标记、再单独构造窗口的接口。
if window:
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ValueError("公告窗口必须为非负整数。")
    event_keys = pd.MultiIndex.from_frame(own_ea[["StkcdRecall", "td"]])
    for offset in range(1, window + 1):
        for prefix, sign in [("tm", -1), ("tp", 1)]:
            target_keys = pd.MultiIndex.from_arrays([
                daily["StkcdRecall"], daily["td"] + sign * offset,
            ])
            daily[f"ownEA_{prefix}{offset}"] = target_keys.isin(event_keys).astype("int8")

print(
    "ownEA为1的数量：",
    (daily["ownEA"] == 1).sum(),
)
print(
    "ownEA为1的比例：",
    (daily["ownEA"] == 1).sum()
    / len(daily),
)

unmatched_own_ea = (
    own_ea
    .merge(
        daily[["StkcdRecall", "Trddt"]],
        on=["StkcdRecall", "Trddt"],
        how="left",
        indicator=True,
    )
    .loc[lambda data: data["_merge"].eq("left_only")]
    .copy()
)

print(
    "未匹配自身公告事件数：",
    len(unmatched_own_ea),
)
print(
    unmatched_own_ea["Trddt"]
    .dt.year
    .value_counts()
    .sort_index()
)

display(daily[["StkcdRecall", "Trddt", "ownEA"]].head())


# %% 06. 构造自身公告前后10个交易日窗口
window = 10
if (
    not isinstance(window, int)
    or isinstance(window, bool)
    or window <= 0
):
    raise ValueError("自身公告窗口长度必须为正整数")

key_cols = ["StkcdRecall", "td"]

# 日度主面板必须在被提示公司—交易日序号层面唯一。
duplicate_count = daily.duplicated(key_cols).sum()
print(
    "日度主面板中 StkcdRecall-td 重复数：",
    duplicate_count,
)

if duplicate_count != 0:
    raise ValueError(
        "日度主面板中 StkcdRecall-td 不唯一，"
        "不能构造自身公告窗口。"
    )

# 提取自身公告发生的公司—交易日序号。
own_ea_td = (
    daily.loc[
        daily["ownEA"].eq(1),
        ["StkcdRecall", "td"],
    ]
    .drop_duplicates()
    .copy()
)

print("ownEA 事件数：", len(own_ea_td))

for i in range(1, window + 1):
    # ownEA_tp{i}：当前日之后第 i 个交易日发生自身公告。
    future_ea = own_ea_td.copy()
    future_ea["td"] = future_ea["td"] - i
    future_ea = (
        future_ea
        .drop_duplicates(["StkcdRecall", "td"])
        .assign(**{f"ownEA_tp{i}": 1})
    )

    daily = daily.merge(
        future_ea,
        on=["StkcdRecall", "td"],
        how="left",
        validate="one_to_one",
    )

    daily[f"ownEA_tp{i}"] = (
        daily[f"ownEA_tp{i}"]
        .fillna(0)
        .astype("int8")
    )

    # ownEA_tm{i}：当前日之前第 i 个交易日发生自身公告。
    past_ea = own_ea_td.copy()
    past_ea["td"] = past_ea["td"] + i
    past_ea = (
        past_ea
        .drop_duplicates(["StkcdRecall", "td"])
        .assign(**{f"ownEA_tm{i}": 1})
    )

    daily = daily.merge(
        past_ea,
        on=["StkcdRecall", "td"],
        how="left",
        validate="one_to_one",
    )

    daily[f"ownEA_tm{i}"] = (
        daily[f"ownEA_tm{i}"]
        .fillna(0)
        .astype("int8")
    )

print("ownEA 前后窗口变量生成完成。")

own_ea_window_cols = (
    [f"ownEA_tp{i}" for i in range(1, window + 1)]
    + ["ownEA"]
    + [f"ownEA_tm{i}" for i in range(1, window + 1)]
)

print(daily[own_ea_window_cols].sum())

print("公告窗口合并后：", daily.shape)
display(daily.head())


# %% 07. 合并业绩预告快报和其他公司事件
performance_forecast_path = paths["performance_forecast_controls_path"]
other_event_path = paths["other_event_controls_path"]
performance_forecast_controls = pd.read_csv(
    performance_forecast_path,
    encoding="utf-8-sig",
    usecols=[
        "StkcdRecall",
        "Trddt",
        "ownPerformanceForecastBrief",
    ],
    dtype={"StkcdRecall": "string"},
    parse_dates=["Trddt"],
)

other_event_controls = pd.read_csv(
    other_event_path,
    encoding="utf-8-sig",
    usecols=[
        "StkcdRecall",
        "Trddt",
        "ownOtherEventER",
    ],
    dtype={"StkcdRecall": "string"},
    parse_dates=["Trddt"],
)

control_tables = [
    (
        "业绩预告快报",
        performance_forecast_controls,
        "ownPerformanceForecastBrief",
    ),
    (
        "专题事件",
        other_event_controls,
        "ownOtherEventER",
    ),
]

for controls_name, controls_df, value_col in control_tables:
    controls_df["StkcdRecall"] = (
        controls_df["StkcdRecall"]
        .apply(normalize_stock_code)
    )

    missing_key = (
        controls_df["StkcdRecall"].eq("")
        | controls_df["Trddt"].isna()
    ).sum()

    duplicate_key = controls_df.duplicated(
        ["StkcdRecall", "Trddt"]
    ).sum()

    print(
        f"{controls_name}控制表缺失键数：{missing_key}，"
        f"重复键数：{duplicate_key}"
    )

    if missing_key != 0 or duplicate_key != 0:
        raise ValueError(
            f"{controls_name}控制表在 "
            "StkcdRecall-Trddt 层面存在缺失或重复，"
            "请先检查。"
        )

    controls_df[value_col] = (
        pd.to_numeric(
            controls_df[value_col],
            errors="raise",
        )
        .fillna(0)
        .astype("int8")
    )

# 两张控制表已经按真实交易日整理，直接按公司和交易日合并。
daily = daily.merge(
    performance_forecast_controls,
    on=["StkcdRecall", "Trddt"],
    how="left",
    validate="one_to_one",
)

daily = daily.merge(
    other_event_controls,
    on=["StkcdRecall", "Trddt"],
    how="left",
    validate="one_to_one",
)

control_cols = [
    "ownPerformanceForecastBrief",
    "ownOtherEventER",
]

daily[control_cols] = (
    daily[control_cols]
    .fillna(0)
    .astype("int8")
)

    # 直接使用来源变量进行统计和后续回归。
print(
    "专题事件控制变量为1的数量：",
    daily["ownOtherEventER"].sum(),
)
print(
    "业绩预告或快报控制变量为1的数量：",
    daily["ownPerformanceForecastBrief"].sum(),
)
print(
    "两类自身事件控制变量同时为1的数量：",
    (
        daily["ownOtherEventER"].eq(1)
        & daily["ownPerformanceForecastBrief"].eq(1)
    ).sum(),
)

print("公司事件合并后：", daily.shape)
display(daily.head())


# %% 08. 合并真实交易日新闻控制变量
news_path = paths["AES_AEV_controls_path"]
news_controls = pd.read_csv(
    news_path,
    encoding="utf-8-sig",
    usecols=[
        "Stkcd",
        "date",
        "AES",
        "AEV",
        "aesIncrease",
        "aesDecrease",
        "logIncreaseAEV",
        "logDecreaseAEV",
    ],
    dtype={"Stkcd": "string"},
    parse_dates=["date"],
)

news_controls["Stkcd"] = (
    news_controls["Stkcd"]
    .apply(normalize_stock_code)
)

news_controls = news_controls.rename(
    columns={
        "Stkcd": "StkcdRecall",
        "date": "Trddt",
    }
)

daily["Trddt"] = pd.to_datetime(
    daily["Trddt"],
    errors="coerce",
)
news_controls["Trddt"] = pd.to_datetime(
    news_controls["Trddt"],
    errors="coerce",
)

news_key_missing = (
    news_controls["StkcdRecall"].eq("")
    | news_controls["StkcdRecall"].isna()
    | news_controls["Trddt"].isna()
).sum()

news_key_duplicate = news_controls.duplicated(
    ["StkcdRecall", "Trddt"]
).sum()

print(
    "新闻控制表缺失键数：",
    news_key_missing,
    "，重复键数：",
    news_key_duplicate,
)

if news_key_missing != 0 or news_key_duplicate != 0:
    raise ValueError(
        "新闻控制表在 StkcdRecall-Trddt 层面存在缺失或重复。"
    )

panel_key_missing = (
    daily["StkcdRecall"].eq("")
    | daily["StkcdRecall"].isna()
    | daily["Trddt"].isna()
).sum()

panel_key_duplicate = daily.duplicated(
    ["StkcdRecall", "Trddt"]
).sum()

if panel_key_missing != 0 or panel_key_duplicate != 0:
    raise ValueError(
        "日度主面板在 StkcdRecall-Trddt 层面存在缺失或重复。"
    )

news_numeric_cols = [
    "AES",
    "AEV",
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]

news_controls[news_numeric_cols] = (
    news_controls[news_numeric_cols]
    .apply(pd.to_numeric, errors="raise")
)

if (news_controls["AEV"].dropna() < 0).any():
    raise ValueError(
        "真实交易日新闻控制表中的 AEV 不应出现负数。"
    )

# 统计新闻表中无法匹配日度主面板的公司—交易日。
panel_keys = (
    daily[["StkcdRecall", "Trddt"]]
    .drop_duplicates()
)

unmatched_news = (
    news_controls[["StkcdRecall", "Trddt"]]
    .merge(
        panel_keys,
        on=["StkcdRecall", "Trddt"],
        how="left",
        indicator=True,
    )
    .loc[lambda data: data["_merge"].eq("left_only")]
)

unmatched_rate = (
    len(unmatched_news) / len(news_controls)
    if len(news_controls) != 0
    else np.nan
)

print("真实交易日新闻控制表行数：", len(news_controls))
print("无法匹配日度主面板的新闻键数：", len(unmatched_news))
print("无法匹配比例：", unmatched_rate)

daily = daily.merge(
    news_controls,
    on=["StkcdRecall", "Trddt"],
    how="left",
    validate="one_to_one",
)

print(
    "合并真实交易日新闻控制变量后表结构：",
    daily.shape,
)

for news_col in news_numeric_cols:
    print(
        f"{news_col} 非缺失数量：",
        daily[news_col].notna().sum(),
    )

print("媒体控制变量汇总：")
print(
    daily[
        [
            "aesIncrease",
            "aesDecrease",
            "logIncreaseAEV",
            "logDecreaseAEV",
        ]
    ].describe()
)

display(daily.head())


# %% 09. 将提示事件左合并到日度主面板
cue_panel = events.copy()
daily["Trddt"] = pd.to_datetime(
    daily["Trddt"]
)
cue_panel["date"] = pd.to_datetime(
    cue_panel["date"]
)

# 检查日度主面板唯一性。
duplicate_main = daily.duplicated(
    ["StkcdRecall", "Trddt"]
).sum()

print("日度主面板重复键数量：", duplicate_main)

if duplicate_main != 0:
    raise ValueError(
        "日度主面板在被提示公司与交易日层面不唯一。"
    )

# 检查聚合事件表唯一性。
duplicate_cue = cue_panel.duplicated(
    ["StkcdRecall", "date"]
).sum()

print("提示事件表重复键数量：", duplicate_cue)

if duplicate_cue != 0:
    raise ValueError(
        "提示事件表在被提示公司与事件交易日层面不唯一。"
    )

print("提示事件表记录数：", len(cue_panel))

daily = daily.merge(
    cue_panel,
    left_on=["StkcdRecall", "Trddt"],
    right_on=["StkcdRecall", "date"],
    how="left",
    validate="one_to_one",
)

matched_cue_count = (
    daily["date"].notna().sum()
)

print(
    "能匹配到日度主面板的事件表记录数：",
    matched_cue_count,
)
print(
    "无法匹配到日度主面板的事件表记录数：",
    len(cue_panel) - matched_cue_count,
)

del cue_panel

# 主面板已有交易日期，删除合并带入的日期别名。
daily = daily.drop(
    columns=["date"],
    errors="ignore",
)

print("合并提示事件后的表结构：", daily.shape)
print(
    "合并后被提示公司与交易日重复键数量：",
    daily.duplicated(
        ["StkcdRecall", "Trddt"]
    ).sum(),
)

display(daily.head())


# %% 10. 提示标记和提示次数缺失填零，保留连续变量缺失
agg_specs = build_pattern_cue_agg_specs(
    daily.columns
)

cue_dummy_cols = [
    col
    for col, (_, aggregation) in agg_specs.items()
    if aggregation == "max"
]

daily[cue_dummy_cols] = (
    daily[cue_dummy_cols]
    .fillna(0)
    .astype("int8")
)

print("提示标记缺失值已设为0")
print(
    daily[cue_dummy_cols]
    .sum()
    .sort_values(ascending=False)
    .head(20)
)

cue_count_cols = [
    "nrCuesLag1S90p",
    "nrCuesLag1S5090",
    "nrCuesLag1S1050",
    "nrCuesLag1S10m",
    "nrCuesLag1AnnualS90p",
    "nrCuesLag1Q1S90p",
]

cue_count_cols = [
    col for col in cue_count_cols
    if col in daily.columns
]

daily[cue_count_cols] = (
    daily[cue_count_cols]
    .fillna(0)
)
if "cueReportSourcesLag1S90p" in daily.columns:
    daily["cueReportSourcesLag1S90p"] = (
        daily["cueReportSourcesLag1S90p"].fillna("")
    )

print("提示日数量：", int(daily["quarterLag1size90plus"].sum()))
display(daily[["StkcdRecall", "Trddt", "quarterLag1size90plus", "nrCuesLag1S90p"]].head(20))


# %% 11. 合并特征调整收益率
adjusted_return_path = paths["adjusted_return_path"]
base_return_cols = [
    "chAdj_ret_tm3",
    "chAdj_ret_tm2",
    "chAdj_ret_tm1",
    "chAdj_ret_t",
]

forward_return_cols = [
    f"chAdj_ret_t_t{i}"
    for i in range(1, 51)
]

adjusted_returns = pd.read_csv(
    adjusted_return_path,
    encoding="utf-8-sig",
    dtype=(
        {
            "Stkcd": "string",
            "portfolioNumber": "float32",
        }
        | {
            column: "float32"
            for column in (
                base_return_cols + forward_return_cols
            )
        }
    ),
    parse_dates=["Trddt"],
)

adjusted_returns["Stkcd"] = (
    adjusted_returns["Stkcd"]
    .apply(normalize_stock_code)
)

return_rename = {
    "Stkcd": "StkcdRecall",
    "chAdj_ret_tm3": "chAdjreturn_tm3_Recall",
    "chAdj_ret_tm2": "chAdjreturn_tm2_Recall",
    "chAdj_ret_tm1": "chAdjreturn_tm1_Recall",
    "chAdj_ret_t": "chAdjreturn_t_Recall",
}

return_rename.update({
    f"chAdj_ret_t_t{i}": f"chAdjreturn_t_tp{i}_Recall"
    for i in range(1, 51)
})

adjusted_returns = adjusted_returns.rename(
    columns=return_rename
)

keep_adj_cols = (
    ["StkcdRecall", "Trddt", "portfolioNumber"]
    + [
        col for col in adjusted_returns.columns
        if col.startswith("chAdjreturn_")
    ]
)

adjusted_returns = (
    adjusted_returns[keep_adj_cols].copy()
)

duplicate_count = adjusted_returns.duplicated(
    ["StkcdRecall", "Trddt"]
).sum()

print(
    "特征调整收益率表重复键数量：",
    duplicate_count,
)

if duplicate_count != 0:
    raise ValueError(
        "特征调整收益率在被提示公司与交易日层面不唯一。"
    )

daily = daily.merge(
    adjusted_returns,
    on=["StkcdRecall", "Trddt"],
    how="left",
    validate="one_to_one",
)

del adjusted_returns
gc.collect()

print(
    "合并特征调整收益率后表结构：",
    daily.shape,
)

check_cols = [
    "chAdjreturn_t_Recall",
    "chAdjreturn_t_tp1_Recall",
    "chAdjreturn_t_tp10_Recall",
]

for col in check_cols:
    print(
        f"{col} 缺失比例：",
        daily[col].isna().mean(),
    )

display(daily.head())


# %% 12. 构造年度干扰高低组
if "year" not in daily.columns:
    daily["year"] = daily["Trddt"].dt.year

# 当前使用上一季度关系对应的干扰指标。
for i in range(1, 2):
    for weight in ["EW", "VW"]:
        for suffix in ["S90p", "S5090"]:
            value_col = (
                f"nrFirmsAnnLag{i}{weight}{suffix}"
            )
            group_col = (
                f"interferenceGroup{weight}Lag{i}{suffix}"
            )
            low_col = (
                f"lowInterference{weight}Lag{i}{suffix}"
            )
            high_col = (
                f"highInterference{weight}Lag{i}{suffix}"
            )

            daily[group_col] = by_year_xtile_2(
                daily,
                value_col,
                year_col="year",
            )

            daily[low_col] = (
                daily[group_col] == 1
            ).astype("int8")

            daily[high_col] = (
                daily[group_col] == 2
            ).astype("int8")

print("干扰分组后：", daily.shape)
display(daily.head())


# %% 13. 构造提示盈余惊喜五分位组
for i in range(1, 2):
    for sue_id in [1, 2, 3]:
        for weight in ["VW", "EW"]:
            value_col = (
                f"sue{sue_id}CueLag{i}S90p{weight}"
            )
            group_col = (
                f"sue{sue_id}CueLag{i}{weight}S90p_quintile"
            )

            daily[group_col] = xtile_nq(
                daily[value_col],
                q=5,
            )

print("盈余惊喜分组后：", daily.shape)
display(daily.head())


# %% 14. 构造被提示公司的年度规模组
daily["marketCap_Minus3_Recall"] = (
    daily["mktcap_tm3"]
)

daily["marketCap_Minus3_Recall_medGr"] = (
    by_year_xtile_2(
        daily,
        "marketCap_Minus3_Recall",
        year_col="year",
    )
)

daily["smallFirm"] = (
    daily["marketCap_Minus3_Recall_medGr"] == 1
).astype("int8")

print("规模分组后：", daily.shape)
display(daily.head())


# %% 15. 在筛样前计算全样本1%和99%缩尾阈值
p = 0.01
dep_vars = [
    "chAdjreturn_tm3_Recall",
    "chAdjreturn_tm2_Recall",
    "chAdjreturn_tm1_Recall",
    "chAdjreturn_t_Recall",
]

dep_vars += [
    f"chAdjreturn_t_tp{i}_Recall"
    for i in range(1, 51)
]

# 提示公司盈余惊喜：三种指标，各有市值加权和等权口径。
sue_winsor_cols = []
for i in range(1, 2):
    for sue_id in [1, 2, 3]:
        for weight in ["VW", "EW"]:
            sue_winsor_cols.append(
                f"sue{sue_id}CueLag{i}S90p{weight}"
            )

# 只为当前表中实际存在的变量计算阈值。
winsor_source_cols = [
    col
    for col in dep_vars + sue_winsor_cols
    if col in daily.columns
]

print(
    f"[缩尾阈值] 开始计算，共 {len(winsor_source_cols)} 个变量。",
    flush=True,
)
winsor_limits = {}
for col in winsor_source_cols:
    winsor_limits[col] = winsorize_limits(
        daily[col],
        p=p,
    )

print("[缩尾阈值] 所有变量计算完成。", flush=True)

display(pd.DataFrame(winsor_limits, index=["下限", "上限"]).T.head(20))


# %% 16. 生成基准因变量缩尾列和年度季度标识
print("[基准变量] 开始生成两个基准因变量的缩尾列。", flush=True)
# 使用已经算好的阈值，先生成两个基准因变量的缩尾列。
for col in [
    "chAdjreturn_t_tp1_Recall",
    "chAdjreturn_t_tp10_Recall",
]:
    if col in winsor_limits:
        add_winsor_col(
            df=daily,
            col=col,
            winsor_limits=winsor_limits,
        )

print("[基准变量] 两个基准缩尾列处理完成，开始生成季度标识。", flush=True)
# 生成日历季度和年度—季度固定效应标识。
daily["quarter"] = daily["Trddt"].dt.quarter
daily["yearQuarter"] = (
    daily["Trddt"].dt.to_period("Q")
)
print("[基准变量] 季度及年度季度标识生成完成。", flush=True)

print(
    "[基准变量] 基准变量准备完成，整表整理将在样本筛选后进行。",
    flush=True,
)

display(daily.head())


# %% 17. 按规格5依次筛选两日和十一日基准回归样本
regression_sample_spec = REGRESSION_SAMPLE_SPEC
own_ea_cols = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)

# 被提示公司自身专题事件、业绩预告或快报。
own_disclosure_cols = [
    "ownOtherEventER",
    "ownPerformanceForecastBrief",
]

# 两个基准回归使用相同的解释变量。
baseline_rhs = (
    ["quarterLag1size90plus"]
    + own_ea_cols
    + own_disclosure_cols
    + [
        "aesIncrease",
        "aesDecrease",
        "logIncreaseAEV",
        "logDecreaseAEV",
    ]
)

# 先筛选事件日至后第1个交易日的样本，
# 再在剩余样本中筛选事件日至后第10个交易日的样本。
for dep_col in [
    "chAdjreturn_t_tp1_Recall_w",
    "chAdjreturn_t_tp10_Recall_w",
]:
    print(
        f"[基准筛样] {dep_col}：开始，当前行数 {len(daily)}。",
        flush=True,
    )
    sample_mask = baseline_regression_sample_mask(
        df=daily,
        dep_col=dep_col,
        rhs_cols=baseline_rhs,
        fe_cols=["StkcdRecall", "yearQuarter"],
        cluster_cols=["StkcdRecall", "Trddt"],
        regression_sample_spec=regression_sample_spec,
    )

    print(
        f"[基准筛样] {dep_col}：掩码计算完成，开始筛选并复制整表。",
        flush=True,
    )
    daily = daily.loc[sample_mask].copy()
    print(
        f"[基准筛样] {dep_col}：整表筛选完成，剩余 {len(daily)} 行。",
        flush=True,
    )

if daily.empty:
    raise ValueError("当前口径筛样后没有观测。")
daily = daily.sort_values(["StkcdRecall", "Trddt"]).reset_index(drop=True)
print("基准回归样本：", daily.shape)
display(daily.head())


# %% 18. 用筛样前阈值补齐其余缩尾列
print("基准样本筛选后补生成其余缩尾变量......", flush=True)

for col in winsor_source_cols:
    w_col = f"{col}_w"

    if (
        col in daily.columns
        and w_col not in daily.columns
    ):
        add_winsor_col(
            df=daily,
            col=col,
            winsor_limits=winsor_limits,
        )

print(
    "最终同时进入两个基准回归的样本量：",
    len(daily),
)
print(
    "最终提示事件数量：",
    daily["quarterLag1size90plus"].sum(),
)

display(daily.head())


# %% 19. 根据筛样后提示日构造未来10个交易日提示标记
duplicate_count = daily.duplicated(
    ["StkcdRecall", "td"]
).sum()

print(
    "构造未来提示标记前的重复记录数：",
    duplicate_count,
)

if duplicate_count != 0:
    raise ValueError(
        "被提示公司与交易日序号的组合不唯一，"
        "不能构造未来提示标记。"
    )

# 提取最终基准样本中保留的提示事件日。
cue_days = (
    daily.loc[
        daily["quarterLag1size90plus"].eq(1),
        ["StkcdRecall", "td"],
    ]
    .drop_duplicates()
    .copy()
)

print("基准提示事件数：", len(cue_days))

for i in range(1, 11):
    # 将事件日序号减去偏移量，匹配到事件发生前的日期。
    future_cue = cue_days.copy()
    future_cue["td"] = future_cue["td"] - i

    future_cue = (
        future_cue
        .drop_duplicates(["StkcdRecall", "td"])
        .assign(**{f"cue_tp{i}": 1})
    )

    daily = daily.merge(
        future_cue,
        on=["StkcdRecall", "td"],
        how="left",
        validate="1:1",
    )

    daily[f"cue_tp{i}"] = (
        daily[f"cue_tp{i}"]
        .fillna(0)
        .astype("int8")
    )

print("未来提示事件标记生成完成。")
print(
    daily[
        [f"cue_tp{i}" for i in range(1, 11)]
    ].sum()
)

display(daily.head())


# %% 20. 保留核心收益和回归变量，删除辅助列
expected_return_cols = (
    [
        "return_tm3_Recall",
        "return_tm2_Recall",
        "return_tm1_Recall",
        "return_t_Recall",
    ]
    + [
        f"return_t_tp{i}_Recall"
        for i in range(1, 51)
    ]
)

missing_return_cols = [
    col
    for col in expected_return_cols
    if col not in daily.columns
]

if missing_return_cols:
    raise ValueError(
        "请先完成核心原始收益率列的重命名，再清理辅助列。"
        f" 当前缺少：{missing_return_cols[:10]}"
    )

drop_cols_final = [
    # 新闻水平指标，最终保留四个新闻变化控制变量。
    "AES",
    "AEV",

    # 规模和干扰分组的中间列，最终保留对应标记。
    "marketCap_Minus3_Recall_medGr",
    "interferenceGroupEWLag1S90p",
    "interferenceGroupEWLag1S5090",
    "interferenceGroupVWLag1S90p",
    "interferenceGroupVWLag1S5090",

    # 日度面板和特征调整收益率构造中的辅助字段。
    "ret",
    "retn",
    "Markettype",
    "size",
    "sizew",
    "trade_days_1m",
    "trade_days_1y",
    "mktcap_tm3",
    "float_mktcap_tm3",
    "mktcap_tm3_valid",
    "float_mktcap_tm3_valid",
    "portfolioNumber",
]

# 删除原始收益率旧前缀下剩余的辅助字段。
drop_cols_final += [
    col
    for col in daily.columns
    if col.startswith("raw_ret_")
]

drop_cols_final = list(dict.fromkeys(drop_cols_final))

drop_cols_existing = [
    col
    for col in drop_cols_final
    if col in daily.columns
]

print("最终输出前删除中间变量数量：", len(drop_cols_existing))
print(drop_cols_existing)

daily = daily.drop(columns=drop_cols_existing)

print("最终面板：", daily.shape)
print("最终主键重复数：", daily.duplicated(["StkcdRecall", "Trddt"]).sum())
display(daily.head(20))


# %% 21. 按提示口径和被解释变量口径保存最终面板及摘要
daily = daily.copy()
daily["cueDefinition"] = CURRENT_DEFINITION
daily["industryTopN"] = (
    int(CURRENT_TOP_N)
    if CURRENT_TOP_N is not None
    else pd.NA
)

# 不同提示口径、不同收益来源分别保存。
folder = (
    stage_root(PROJECT_ROOT)
    / FINAL_ROOT_NAME
    / CURRENT_LABEL
    / RETURN_LABEL
)
folder.mkdir(parents=True, exist_ok=True)

output_path = folder / (
    f"最终回归面板_筛样规格{REGRESSION_SAMPLE_SPEC}.csv"
)
summary_path = output_path.with_suffix(".json")

daily.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d",
)

# 将上游口径与本次收益来源一并记录。
summary = {
    "stage": 3,
    "cue_definition": CURRENT_DEFINITION,
    "cue_label": CUE_LABEL,
    "marketcap_quantile": MARKETCAP_QUANTILE,
    "industry_top_n": CURRENT_TOP_N,
    "cue_column": "quarterLag1size90plus",
    "return_definition": RETURN_DEFINITION,
    "return_label": RETURN_LABEL,
    "return_source_path": str(paths["adjusted_return_path"]),
    "regression_sample_spec": REGRESSION_SAMPLE_SPEC,
    "grain": "StkcdRecall × Trddt",
    "rows": int(len(daily)),
    "firms": int(daily["StkcdRecall"].nunique()),
    "key_duplicate_count": int(
        daily.duplicated(["StkcdRecall", "Trddt"]).sum()
    ),
    "event_path": str(EVENT_PATH),
    "event_summary_path": str(EVENT_SUMMARY_PATH),
    "output_path": str(output_path),
}
save_summary(summary_path, summary)

print("提示口径：", CUE_LABEL)
print("被解释变量口径：", RETURN_LABEL)
print("最终面板输出：", output_path)
print("口径摘要输出：", summary_path)
# %%
