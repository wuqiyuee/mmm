"""第三阶段：把统一提示事件表合并到日度面板并生成最终回归面板。

所有计算实现在本文件内，仅导入标准库和第三方库，不加载其他项目脚本。
各口径按同一顺序处理日度基础面板、自身公告和公司事件、新闻、提示事件、
特征调整收益、干扰和盈余惊喜分组、缩尾、基准样本筛选及未来提示变量。
参数、输出路径及筛样规格沿用三阶段设计，主提示资格由第二阶段确定。
CSMAR（国泰安数据库）的Annodt（公告日期）只在非交易日映射至下一交易日。
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd


START_YEAR = 2005
END_YEAR = 2025
REGRESSION_SAMPLE_SPEC = 5
CHUNK_SIZE = 250_000
STAGE_ROOT_NAME = "三阶段合并"
EVENT_ROOT_NAME = "提示事件"
FINAL_ROOT_NAME = "最终面板"



def find_root() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "AGENTS.md").is_file() and (folder / "中间结果").is_dir():
            return folder
    raise FileNotFoundError("找不到包含AGENTS.md和中间结果的项目根目录。")


def build_stage_paths(root: Path) -> dict:
    """列出第三阶段直接读取的正式数据文件。"""
    return {
        "TRD_Dalyr_DGTW_path": root / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv",
        "eligibility_path": root / "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv",
        "IAR_Rept_path": root / "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv",
        "performance_forecast_controls_path": root / "中间结果/02_变量构造结果/5公司公告事件/业绩预告快报控制变量_公告日期口径.csv",
        "other_event_controls_path": root / "中间结果/02_变量构造结果/5公司公告事件/专题研究事件控制变量_公告日期口径.csv",
        "AES_AEV_controls_path": root / "中间结果/02_变量构造结果/3分段新闻_AES_AEV/DATAGO_分段新闻_真实交易日控制变量.csv",
        "adjusted_return_path": root / "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW特征调整收益率.csv",
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


def load_pattern_daily_panel(daily_panel_path):
    """读取个股日度面板，标准化股票代码并重命名收益变量。"""

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

    daily_panel = pd.read_csv(
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

    daily_panel["Stkcd"] = (
        daily_panel["Stkcd"]
        .apply(normalize_stock_code)
    )

    # 日度面板在后续全部按被提示公司命名。
    daily_panel = (
        daily_panel
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

    daily_panel = daily_panel.rename(
        columns=raw_return_rename
    )

    return daily_panel


def add_own_earnings_announcement_flag(
    daily_panel,
    report_path,
    event_date_mode,
    calendar_path=None,
    window=0,
):
    """将自身公告日期映射到交易日，并合并 ownEA 标记。"""

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
        if calendar_path is not None else daily_panel[["Trddt", "td"]]
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

    daily_panel = daily_panel.merge(
        own_ea[["StkcdRecall", "Trddt", "ownEA"]],
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="one_to_one",
    )

    daily_panel["ownEA"] = (
        daily_panel["ownEA"]
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
                    daily_panel["StkcdRecall"], daily_panel["td"] + sign * offset,
                ])
                daily_panel[f"ownEA_{prefix}{offset}"] = target_keys.isin(event_keys).astype("int8")

    print(
        "ownEA为1的数量：",
        (daily_panel["ownEA"] == 1).sum(),
    )
    print(
        "ownEA为1的比例：",
        (daily_panel["ownEA"] == 1).sum()
        / len(daily_panel),
    )

    unmatched_own_ea = (
        own_ea
        .merge(
            daily_panel[["StkcdRecall", "Trddt"]],
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

    return daily_panel


def add_own_ea_windows(daily_panel, window=10):
    """生成自身公告日前后若干个交易日的控制变量。"""

    if (
        not isinstance(window, int)
        or isinstance(window, bool)
        or window <= 0
    ):
        raise ValueError("自身公告窗口长度必须为正整数")

    key_cols = ["StkcdRecall", "td"]

    # 日度主面板必须在被提示公司—交易日序号层面唯一。
    duplicate_count = daily_panel.duplicated(key_cols).sum()
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
        daily_panel.loc[
            daily_panel["ownEA"].eq(1),
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

        daily_panel = daily_panel.merge(
            future_ea,
            on=["StkcdRecall", "td"],
            how="left",
            validate="one_to_one",
        )

        daily_panel[f"ownEA_tp{i}"] = (
            daily_panel[f"ownEA_tp{i}"]
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

        daily_panel = daily_panel.merge(
            past_ea,
            on=["StkcdRecall", "td"],
            how="left",
            validate="one_to_one",
        )

        daily_panel[f"ownEA_tm{i}"] = (
            daily_panel[f"ownEA_tm{i}"]
            .fillna(0)
            .astype("int8")
        )

    print("ownEA 前后窗口变量生成完成。")

    own_ea_window_cols = (
        [f"ownEA_tp{i}" for i in range(1, window + 1)]
        + ["ownEA"]
        + [f"ownEA_tm{i}" for i in range(1, window + 1)]
    )

    print(daily_panel[own_ea_window_cols].sum())

    return daily_panel


def merge_own_event_controls(
    daily_panel,
    performance_forecast_path,
    other_event_path,
):
    """读取并合并被提示公司的业绩预告和专题事件控制变量。"""

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
    daily_panel = daily_panel.merge(
        performance_forecast_controls,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="one_to_one",
    )

    daily_panel = daily_panel.merge(
        other_event_controls,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="one_to_one",
    )

    control_cols = [
        "ownPerformanceForecastBrief",
        "ownOtherEventER",
    ]

    daily_panel[control_cols] = (
        daily_panel[control_cols]
        .fillna(0)
        .astype("int8")
    )

        # 直接使用来源变量进行统计和后续回归。
    print(
        "专题事件控制变量为1的数量：",
        daily_panel["ownOtherEventER"].sum(),
    )
    print(
        "业绩预告或快报控制变量为1的数量：",
        daily_panel["ownPerformanceForecastBrief"].sum(),
    )
    print(
        "两类自身事件控制变量同时为1的数量：",
        (
            daily_panel["ownOtherEventER"].eq(1)
            & daily_panel["ownPerformanceForecastBrief"].eq(1)
        ).sum(),
    )

    return daily_panel


def merge_news_controls(daily_panel, news_path):
    """读取并合并真实交易日新闻控制变量。"""

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

    daily_panel["Trddt"] = pd.to_datetime(
        daily_panel["Trddt"],
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
        daily_panel["StkcdRecall"].eq("")
        | daily_panel["StkcdRecall"].isna()
        | daily_panel["Trddt"].isna()
    ).sum()

    panel_key_duplicate = daily_panel.duplicated(
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
        daily_panel[["StkcdRecall", "Trddt"]]
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

    daily_panel = daily_panel.merge(
        news_controls,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="one_to_one",
    )

    print(
        "合并真实交易日新闻控制变量后表结构：",
        daily_panel.shape,
    )

    for news_col in news_numeric_cols:
        print(
            f"{news_col} 非缺失数量：",
            daily_panel[news_col].notna().sum(),
        )

    print("媒体控制变量汇总：")
    print(
        daily_panel[
            [
                "aesIncrease",
                "aesDecrease",
                "logIncreaseAEV",
                "logDecreaseAEV",
            ]
        ].describe()
    )

    return daily_panel


def merge_pattern_cue_events(
    daily_panel,
    cue_events_path,
    cue_events=None,
):
    """合并提示事件指标，保持被提示公司与交易日层面的行粒度。"""

    # 传入事件表时直接使用；未传入时读取已保存文件。
    if cue_events is None:
        cue_panel = pd.read_csv(
            cue_events_path,
            encoding="utf-8-sig",
            parse_dates=["date"],
            dtype={"StkcdRecall": "string"},
        )
        cue_panel["StkcdRecall"] = (
            cue_panel["StkcdRecall"]
            .apply(normalize_stock_code)
        )

        # 从已保存的沪深300事件表读取时，恢复内部计算列名。
        # 以季度关系或历史机制开头的列使用规模后缀，其余汇总列使用主组后缀。
        if "cueDefinition" in cue_panel.columns:
            definitions = cue_panel["cueDefinition"].dropna().unique()
            if len(definitions) != 1 or definitions[0] not in {"size90", "hs300"}:
                raise ValueError("提示事件表没有唯一、有效的提示口径标记。")
            if definitions[0] == "hs300":
                rename_columns = {}
                for col in cue_panel.columns:
                    if "HS300" in col:
                        suffix = "size90plus" if col.startswith(("quarterLag", "lag")) else "S90p"
                        rename_columns[col] = col.replace("HS300", suffix)
                cue_panel = cue_panel.rename(columns=rename_columns)
    else:
        cue_panel = cue_events.copy()

    daily_panel["Trddt"] = pd.to_datetime(
        daily_panel["Trddt"]
    )
    cue_panel["date"] = pd.to_datetime(
        cue_panel["date"]
    )

    # 检查日度主面板唯一性。
    duplicate_main = daily_panel.duplicated(
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

    daily_panel = daily_panel.merge(
        cue_panel,
        left_on=["StkcdRecall", "Trddt"],
        right_on=["StkcdRecall", "date"],
        how="left",
        validate="one_to_one",
    )

    matched_cue_count = (
        daily_panel["date"].notna().sum()
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
    daily_panel = daily_panel.drop(
        columns=["date"],
        errors="ignore",
    )

    print("合并提示事件后的表结构：", daily_panel.shape)
    print(
        "合并后被提示公司与交易日重复键数量：",
        daily_panel.duplicated(
            ["StkcdRecall", "Trddt"]
        ).sum(),
    )

    return daily_panel


def fill_pattern_cue_indicators(daily_panel):
    """将提示标记和提示数量的缺失填0，保留连续指标的缺失。"""

    # 当前聚合规则中的最大值项，全部是0/1提示标记。
    # 复用同一份清单，且只处理当前表中已有的列。
    agg_specs = build_pattern_cue_agg_specs(
        daily_panel.columns
    )

    cue_dummy_cols = [
        col
        for col, (_, aggregation) in agg_specs.items()
        if aggregation == "max"
    ]

    daily_panel[cue_dummy_cols] = (
        daily_panel[cue_dummy_cols]
        .fillna(0)
        .astype("int8")
    )

    print("提示标记缺失值已设为0")
    print(
        daily_panel[cue_dummy_cols]
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
        if col in daily_panel.columns
    ]

    daily_panel[cue_count_cols] = (
        daily_panel[cue_count_cols]
        .fillna(0)
    )
    if "cueReportSourcesLag1S90p" in daily_panel.columns:
        daily_panel["cueReportSourcesLag1S90p"] = (
            daily_panel["cueReportSourcesLag1S90p"].fillna("")
        )

    return daily_panel


def merge_adjusted_returns(
    daily_panel,
    adjusted_return_path,
):
    """读取上游特征调整收益率，并合并到日度主面板。"""

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

    daily_panel = daily_panel.merge(
        adjusted_returns,
        on=["StkcdRecall", "Trddt"],
        how="left",
        validate="one_to_one",
    )

    del adjusted_returns
    gc.collect()

    print(
        "合并特征调整收益率后表结构：",
        daily_panel.shape,
    )

    check_cols = [
        "chAdjreturn_t_Recall",
        "chAdjreturn_t_tp1_Recall",
        "chAdjreturn_t_tp10_Recall",
    ]

    for col in check_cols:
        print(
            f"{col} 缺失比例：",
            daily_panel[col].isna().mean(),
        )

    return daily_panel


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


def add_interference_groups(daily_panel):
    """按年份划分干扰指标高低组，并生成对应标记。"""

    if "year" not in daily_panel.columns:
        daily_panel["year"] = daily_panel["Trddt"].dt.year

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

                daily_panel[group_col] = by_year_xtile_2(
                    daily_panel,
                    value_col,
                    year_col="year",
                )

                daily_panel[low_col] = (
                    daily_panel[group_col] == 1
                ).astype("int8")

                daily_panel[high_col] = (
                    daily_panel[group_col] == 2
                ).astype("int8")

    return daily_panel


def xtile_nq(series, q):
    """对整列按原文规则进行分位分组，缺失保留。"""
    return xtile_by_group(series, n=q).astype("float64")


def add_sue_cue_quantiles(daily_panel):
    """将三种提示盈余惊喜均值分别划分为五分位组。"""

    for i in range(1, 2):
        for sue_id in [1, 2, 3]:
            for weight in ["VW", "EW"]:
                value_col = (
                    f"sue{sue_id}CueLag{i}S90p{weight}"
                )
                group_col = (
                    f"sue{sue_id}CueLag{i}{weight}S90p_quintile"
                )

                daily_panel[group_col] = xtile_nq(
                    daily_panel[value_col],
                    q=5,
                )

    return daily_panel


def add_recalled_firm_size_group(daily_panel):
    """按年度划分被提示公司规模组，并生成小公司标记。"""

    # 保留最终输出使用的被提示公司市值名称。
    daily_panel["marketCap_Minus3_Recall"] = (
        daily_panel["mktcap_tm3"]
    )

    daily_panel["marketCap_Minus3_Recall_medGr"] = (
        by_year_xtile_2(
            daily_panel,
            "marketCap_Minus3_Recall",
            year_col="year",
        )
    )

    daily_panel["smallFirm"] = (
        daily_panel["marketCap_Minus3_Recall_medGr"] == 1
    ).astype("int8")

    return daily_panel


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


def compute_pattern_winsor_limits(daily_panel, p=0.01):
    """在筛选基准样本前，计算全样本缩尾阈值及对应变量清单。"""

    # 调整后收益率：事件前各日、事件当日及后续累计窗口。
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
        if col in daily_panel.columns
    ]

    print(
        f"[缩尾阈值] 开始计算，共 {len(winsor_source_cols)} 个变量。",
        flush=True,
    )
    winsor_limits = {}
    for col in winsor_source_cols:
        winsor_limits[col] = winsorize_limits(
            daily_panel[col],
            p=p,
        )

    print("[缩尾阈值] 所有变量计算完成。", flush=True)
    return winsor_limits, winsor_source_cols


def prepare_pattern_baseline_inputs(daily_panel, winsor_limits):
    """准备基准回归所需的缩尾因变量和季度标识，并整理表结构。"""

    print("[基准变量] 开始生成两个基准因变量的缩尾列。", flush=True)
    # 使用已经算好的阈值，先生成两个基准因变量的缩尾列。
    for col in [
        "chAdjreturn_t_tp1_Recall",
        "chAdjreturn_t_tp10_Recall",
    ]:
        if col in winsor_limits:
            add_winsor_col(
                df=daily_panel,
                col=col,
                winsor_limits=winsor_limits,
            )

    print("[基准变量] 两个基准缩尾列处理完成，开始生成季度标识。", flush=True)
    # 生成日历季度和年度—季度固定效应标识。
    daily_panel["quarter"] = daily_panel["Trddt"].dt.quarter
    daily_panel["yearQuarter"] = (
        daily_panel["Trddt"].dt.to_period("Q")
    )
    print("[基准变量] 季度及年度季度标识生成完成。", flush=True)

    print(
        "[基准变量] 基准变量准备完成，整表整理将在样本筛选后进行。",
        flush=True,
    )

    return daily_panel


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


def filter_pattern_baseline_sample(
    daily_panel,
    regression_sample_spec=5,
):
    """按选定规格，依次筛选两日和十一日基准回归样本。"""

    # 被提示公司自身财报公告及前后交易日标记。
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
            f"[基准筛样] {dep_col}：开始，当前行数 {len(daily_panel)}。",
            flush=True,
        )
        sample_mask = baseline_regression_sample_mask(
            df=daily_panel,
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
        daily_panel = daily_panel.loc[sample_mask].copy()
        print(
            f"[基准筛样] {dep_col}：整表筛选完成，剩余 {len(daily_panel)} 行。",
            flush=True,
        )

    return daily_panel


def add_remaining_pattern_winsor_cols(
    daily_panel,
    winsor_source_cols,
    winsor_limits,
):
    """使用筛选前计算的阈值，补生成其余缩尾列。"""

    print("基准样本筛选后补生成其余缩尾变量......", flush=True)

    for col in winsor_source_cols:
        w_col = f"{col}_w"

        if (
            col in daily_panel.columns
            and w_col not in daily_panel.columns
        ):
            add_winsor_col(
                df=daily_panel,
                col=col,
                winsor_limits=winsor_limits,
            )

    print(
        "最终同时进入两个基准回归的样本量：",
        len(daily_panel),
    )
    print(
        "最终提示事件数量：",
        daily_panel["quarterLag1size90plus"].sum(),
    )

    return daily_panel


def add_future_pattern_cue_flags(daily_panel):
    """根据筛选后保留的基准提示事件，生成未来十个交易日标记。"""

    # 检查被提示公司与交易日序号的组合是否唯一。
    duplicate_count = daily_panel.duplicated(
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
        daily_panel.loc[
            daily_panel["quarterLag1size90plus"].eq(1),
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

        daily_panel = daily_panel.merge(
            future_cue,
            on=["StkcdRecall", "td"],
            how="left",
            validate="1:1",
        )

        daily_panel[f"cue_tp{i}"] = (
            daily_panel[f"cue_tp{i}"]
            .fillna(0)
            .astype("int8")
        )

    print("未来提示事件标记生成完成。")
    print(
        daily_panel[
            [f"cue_tp{i}" for i in range(1, 11)]
        ].sum()
    )

    return daily_panel


def clean_pattern_final_columns(daily_panel):
    """确认核心原始收益率列齐全，再删除最终输出不需要的中间列。"""

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
        if col not in daily_panel.columns
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
        for col in daily_panel.columns
        if col.startswith("raw_ret_")
    ]

    drop_cols_final = list(dict.fromkeys(drop_cols_final))

    drop_cols_existing = [
        col
        for col in drop_cols_final
        if col in daily_panel.columns
    ]

    print("最终输出前删除中间变量数量：", len(drop_cols_existing))
    print(drop_cols_existing)

    return daily_panel.drop(columns=drop_cols_existing)


def read_event_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少第二阶段提示事件表：{path}")
    events = pd.read_csv(
        path,
        encoding="utf-8-sig",
        parse_dates=["date"],
        dtype={"StkcdRecall": "string"},
    )
    events["StkcdRecall"] = events["StkcdRecall"].apply(normalize_stock_code)
    key = ["StkcdRecall", "date"]
    if events.duplicated(key).any():
        raise ValueError(f"提示事件表在被提示公司×事件日层面重复：{path}")
    return events


def load_recalled_daily_panel(root: Path, paths: dict) -> pd.DataFrame:
    daily = load_pattern_daily_panel(paths["TRD_Dalyr_DGTW_path"])
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
    daily = add_own_earnings_announcement_flag(
        daily_panel=daily,
        report_path=paths["IAR_Rept_path"],
        event_date_mode="next_if_nontrading",
    )
    daily = add_own_ea_windows(daily_panel=daily, window=10)
    daily = merge_own_event_controls(
        daily_panel=daily,
        performance_forecast_path=paths["performance_forecast_controls_path"],
        other_event_path=paths["other_event_controls_path"],
    )
    daily = merge_news_controls(
        daily_panel=daily,
        news_path=paths["AES_AEV_controls_path"],
    )
    return daily


def add_final_panel_features(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    event_path: Path,
    paths: dict,
    definition: str,
) -> pd.DataFrame:
    """按原文顺序把提示事件和最终回归变量加入日度面板。"""
    daily = merge_pattern_cue_events(
        daily_panel=daily,
        cue_events_path=event_path,
        cue_events=events,
    )
    daily = fill_pattern_cue_indicators(daily)
    daily = merge_adjusted_returns(
        daily_panel=daily,
        adjusted_return_path=paths["adjusted_return_path"],
    )
    daily = add_interference_groups(daily)
    daily = add_sue_cue_quantiles(daily)
    daily = add_recalled_firm_size_group(daily)
    winsor_limits, winsor_source_cols = compute_pattern_winsor_limits(
        daily_panel=daily,
        p=0.01,
    )
    daily = prepare_pattern_baseline_inputs(
        daily_panel=daily,
        winsor_limits=winsor_limits,
    )
    daily = filter_pattern_baseline_sample(
        daily_panel=daily,
        regression_sample_spec=REGRESSION_SAMPLE_SPEC,
    )
    if daily.empty:
        raise ValueError(f"{definition}筛样后没有观测。")
    daily = daily.sort_values(["StkcdRecall", "Trddt"]).reset_index(drop=True)
    daily = add_remaining_pattern_winsor_cols(
        daily_panel=daily,
        winsor_source_cols=winsor_source_cols,
        winsor_limits=winsor_limits,
    )
    daily = add_future_pattern_cue_flags(daily)
    daily = clean_pattern_final_columns(daily)
    return daily


def write_final_panel(
    root: Path,
    daily: pd.DataFrame,
    event_path: Path,
    definition: str,
    top_n: int | None,
) -> dict:
    """补充口径元数据并保存最终面板和审计摘要。"""
    daily = daily.copy()
    daily["cueDefinition"] = definition
    daily["industryTopN"] = int(top_n) if top_n is not None else pd.NA

    label = f"行业前{top_n}名" if definition == "industry_top" else definition
    folder = stage_root(root) / FINAL_ROOT_NAME / label
    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder / f"最终回归面板_筛样规格{REGRESSION_SAMPLE_SPEC}.csv"
    summary_path = folder / "最终面板审计摘要.json"
    daily.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    summary = {
        "stage": 3,
        "cue_definition": definition,
        "industry_top_n": top_n,
        "regression_sample_spec": REGRESSION_SAMPLE_SPEC,
        "grain": "StkcdRecall × Trddt",
        "rows": int(len(daily)),
        "firms": int(daily["StkcdRecall"].nunique()),
        "key_duplicate_count": int(daily.duplicated(["StkcdRecall", "Trddt"]).sum()),
        "event_path": str(event_path),
        "output_path": str(output_path),
    }
    save_summary(summary_path, summary)
    return {
        "cue_definition": definition,
        "industry_top_n": top_n,
        "output_path": output_path,
        "summary_path": summary_path,
        "rows": len(daily),
    }


def finish_one_definition(
    root: Path,
    paths: dict,
    event_path: Path,
    definition: str,
    top_n: int | None,
) -> dict:
    events = read_event_table(event_path)
    daily = load_recalled_daily_panel(root, paths)
    daily = add_final_panel_features(
        daily=daily,
        events=events,
        event_path=event_path,
        paths=paths,
        definition=definition,
    )
    return write_final_panel(root, daily, event_path, definition, top_n)


def main() -> dict:
    root = find_root()
    paths = build_stage_paths(root)
    event_root = stage_root(root) / EVENT_ROOT_NAME
    definitions = [
        ("size90", None, event_root / "size90/提示事件.csv"),
        ("hs300", None, event_root / "hs300/提示事件.csv"),
        ("industry_top", 4, event_root / "行业前4名/提示事件.csv"),
        ("industry_top", 5, event_root / "行业前5名/提示事件.csv"),
    ]
    results = [
        finish_one_definition(root, paths, event_path, definition, top_n)
        for definition, top_n, event_path in definitions
    ]
    index_path = stage_root(root) / FINAL_ROOT_NAME / "本次运行结果.json"
    save_summary(index_path, {"stage": 3, "results": results})
    return {"results": results, "index_path": index_path}


if __name__ == "__main__":
    main()
