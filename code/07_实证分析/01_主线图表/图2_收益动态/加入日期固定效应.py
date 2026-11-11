"""图2：收益动态和长期反转——加入日期固定效应版本。

直接读取选定筛样规格的沪深300正式合并面板，
沿用面板中已有的缩尾收益，各窗口分别处理自身收益缺失。

吸收被提示公司×年度季度及交易日期两组固定效应，
递归删除单例，按被提示公司、交易日期双向聚类。
沿用当前有限样本修正和统计检验自由度计算规则。

估计50个累计窗口，结果单位为百分点，
绘制99%、95%、90%置信区间，红点表示10%水平显著。
每个窗口保存前检查口径和数值完整性。

续算时核验输入面板的路径、大小、修改时间以及脚本、公共算法和依赖版本。
上述信息记入现有结果表；旧结果缺少核验信息时须从头估计。
导入本文件不会读取回归数据或写入结果文件。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components


def find_project_root() -> Path:
    """按项目说明、代码和中间结果三个标记，从当前文件的祖先目录寻根。"""
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "AGENTS.md").is_file()
            and (candidate / "code").is_dir()
            and (candidate / "中间结果").is_dir()
        ):
            return candidate
    raise FileNotFoundError("无法找到包含项目说明、代码和中间结果的项目根目录。")


ROOT = find_project_root()
MODULE_DIR = ROOT / "code" / "07_实证分析" / "90_公共模块"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from 图2图3扩展_日期固定效应回归公共模块 import two_way_cluster_covariance

# 一、输入面板：与表2读取同一份沪深300最终面板。
# 选择输入面板的回归样本筛选规格：5或6。
# 这是合并阶段的筛样选择，不是本脚本的回归规格；本脚本始终使用双固定效应。
INPUT_SAMPLE_SPEC = 5

if INPUT_SAMPLE_SPEC not in (5, 6):
    raise ValueError("输入面板的筛样规格只能选择5或6。")

INPUT_PANEL_PATH = ROOT / (
    "中间结果/04_合并数据/"
    f"Pattern样本_最终回归面板_沪深300_筛样规格{INPUT_SAMPLE_SPEC}.csv"
)

# 二、提示变量、新闻来源及研究口径说明。
CUE_COLUMN = "quarterLag1HS300"
NEWS_SOURCE = "datago_segmented"

SPEC_ID = (
    "沪深300_个人365天_数行者分段_允许填充_全期"
    f"_筛样规格{INPUT_SAMPLE_SPEC}"
)

SPEC_DESCRIPTION = (
    "历史沪深300提示；个人365天共同覆盖排除；一级供应链及同行业排除；"
    "提示方保留金融和房地产；被提示方沿用非金融等基础资格、保留房地产，"
    "不追加市值和60个月交易史筛选；2005—2009年数行者报刊、"
    "2010—2025年数行者网络；允许合格填充日起点；保留全部年份"
)

# 三、控制变量。
# 21项自身财报公告控制。
OWN_EA_COLUMNS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)

# 2项其他公告控制。
EVENT_COLUMNS = [
    "ownOtherEventER",
    "ownPerformanceForecastBrief",
]

# 4项新闻控制。
NEWS_COLUMNS = [
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]

# 完整控制变量，共27项。
CONTROL_COLUMNS = OWN_EA_COLUMNS + EVENT_COLUMNS + NEWS_COLUMNS

# 四、收益窗口：终点1至50，对应两日至五十一日累计收益。
HORIZONS = list(range(1, 51))
OUTCOME_COLS = {
    horizon: f"chAdjreturn_t_tp{horizon}_Recall_w"
    for horizon in HORIZONS
}

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# 五、输出位置：结果和图片均标明输入面板的筛样规格。
OUTPUT_DIR = ROOT / "图表结果" / "01_主线图表" / "图2_收益动态"
OUTPUT_TAG = f"沪深300_筛样规格{INPUT_SAMPLE_SPEC}"

RESULT_PATH = OUTPUT_DIR / (
    f"图2_收益动态和长期反转_加入日期固定效应_{OUTPUT_TAG}_结果.csv"
)

OUTPUT_PATH = OUTPUT_DIR / (
    f"figure2_收益动态和长期反转_加入日期固定效应_{OUTPUT_TAG}.png"
)


RESULT_SPECIFICATION = {
    "口径标识": SPEC_ID,
    "提示变量": CUE_COLUMN,
    "新闻来源": NEWS_SOURCE,
    "表2规格": "(6)",
    "日期固定效应": "是",
    "被提示公司×年度季度固定效应": "是",
    "控制变量": "是",
    "结果单位": "百分点",
    "有限样本修正": "是",
}
RUN_INFORMATION_COLUMNS = (
    "输入面板路径", "输入面板字节数", "输入面板修改时间纳秒", "计算版本",
)
NUMERIC_RESULT_FIELDS = [
    "线索系数", "聚类标准误", "置信区间下界", "置信区间上界",
    "t值", "p值", "回归样本量", "解释变量矩阵条件数", "R方",
    "残差自由度", "固定效应实际秩",
]
RESULT_COMPLETION_FIELDS = NUMERIC_RESULT_FIELDS + ["系数求解方法"]

# 从最长到最短绘制三档区间，保留原配色和线宽。
ERROR_BAR_SPECS = (
    {"label": "99%置信区间（1%水平）", "confidence_level": 0.99,
     "color": "#bdbdbd", "line_width": 0.8, "zorder": 1},
    {"label": "95%置信区间（5%水平）", "confidence_level": 0.95,
     "color": "#78909c", "line_width": 1.6, "zorder": 2},
    {"label": "90%置信区间（10%水平）", "confidence_level": 0.90,
     "color": "#1f4e79", "line_width": 2.8, "zorder": 3},
)


def significance_stars(p_value):
    """根据双侧检验的 p 值生成显著性星号。"""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def drop_fe_singletons_iteratively(reg, fe_code_cols):
    """
    递归删除固定效应单例组。

    当存在两组固定效应时，删除一组中的单例观测可能使另一组
    新产生单例，因此需要反复检查，直到不再存在单例组。
    """
    if not fe_code_cols:
        return reg, 0

    total_removed = 0

    while True:
        singleton = np.zeros(len(reg), dtype=bool)

        for col in fe_code_cols:
            codes = reg[col].to_numpy(
                dtype=np.int64,
                copy=False,
            )
            counts = np.bincount(codes)
            singleton |= counts[codes] == 1

        removed_now = int(singleton.sum())

        if removed_now == 0:
            break

        total_removed += removed_now
        reg = reg.loc[~singleton].copy()

    return reg, total_removed


def exact_fixed_effect_rank(reg, fe_code_cols):
    """计算实际吸收秩；双固定效应扣除公司季度—日期图的连通分量。"""
    if not fe_code_cols:
        return 0
    if len(fe_code_cols) == 1:
        return int(reg[fe_code_cols[0]].nunique())
    left, left_groups = pd.factorize(reg[fe_code_cols[0]], sort=False)
    right, right_groups = pd.factorize(reg[fe_code_cols[1]], sort=False)
    counts = sparse.csr_matrix(
        (np.ones(len(reg)), (left, right)),
        shape=(len(left_groups), len(right_groups)),
    )
    components = connected_components(
        (counts.T @ counts).tocsr(), directed=False, return_labels=False,
    )
    return len(left_groups) + len(right_groups) - int(components)


def figure2_consistent_inference(
    model, result, clusters, reg, fe_code_cols, *, include_diagnostics=False,
):
    """点估计保留原估计器，另按图2的修正及精确自由度计算全部斜率推断。

    显式常数＋提示＋27控制时，聚类协方差分母为N−29，不在这个分母
    再扣吸收固定效应。双侧概率则使用N−实际固定效应秩−28的t分布。
    不直接沿用计量库可能随版本变化的吸收自由度协方差修正。
    """
    names = result.params.index
    absorbed = model.absorbed_exog
    if not absorbed.columns.equals(names):
        raise ValueError("吸收后解释变量与已估计参数的列顺序不一致。")
    x = absorbed.to_numpy(dtype="float64", copy=False)
    xpx = x.T @ x
    xpx_condition = float(np.linalg.cond(xpx)) if np.isfinite(xpx).all() else np.inf
    if not np.isfinite(xpx_condition) or xpx_condition >= 1e12:
        raise ValueError("吸收后的解释变量矩阵不满秩或条件数过大。")
    # 复用正式图2公共函数，分块累计得分；不另生成一份全样本伪逆。
    covariance = two_way_cluster_covariance(
        x, result.wresids.to_numpy(dtype="float64", copy=False),
        clusters["firm"].to_numpy(dtype="int64", copy=False),
        clusters["date"].to_numpy(dtype="int64", copy=False),
        np.linalg.inv(xpx),
    )
    diagonal = pd.Series(np.diag(covariance), index=names)
    reported = [name for name in names if name != "const"]
    if not np.isfinite(diagonal[reported]).all() or diagonal[CUE_COLUMN] <= 0:
        raise ValueError("聚类方差非有限，或沪深300提示系数的聚类方差非正。")
    fe_rank = exact_fixed_effect_rank(reg, fe_code_cols)
    slope_count = len(names) - 1
    degrees = len(reg) - fe_rank - (slope_count if fe_code_cols else len(names))
    if degrees <= 0:
        raise ValueError("回归残差自由度必须为正数。")
    # 双向聚类对个别控制项可能给出非正方差，保留系数并在明细中明确标记。
    variances = diagonal.loc[reported]
    standard_errors = np.sqrt(variances.where(variances > 0))
    t_values = result.params.loc[reported] / standard_errors
    inference = pd.DataFrame({
        "标准误": standard_errors,
        "t值": t_values,
        "p值": 2 * stats.t.sf(np.abs(t_values), degrees),
    })
    # 原有调用默认仍返回三项；图2可额外取得实际矩阵条件数。
    if include_diagnostics:
        return inference, degrees, fe_rank, xpx_condition
    return inference, degrees, fe_rank


def read_figure2_panel(horizons: list[int]) -> pd.DataFrame:
    """直接读取选定的正式面板，各累计窗口分别处理自身收益缺失。"""

    if not horizons or not set(horizons).issubset(HORIZONS):
        raise ValueError("待估计窗口必须属于1至50。")

    if not INPUT_PANEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到正式合并面板：{INPUT_PANEL_PATH}"
        )

    # 两个基础窗口用于检查面板是否已完成合并阶段筛样。
    baseline_return_columns = [
        OUTCOME_COLS[1],
        OUTCOME_COLS[10],
    ]

    # 读取待估计窗口，同时保留两个基础窗口用于检查。
    return_columns = list(dict.fromkeys(
        baseline_return_columns
        + [OUTCOME_COLS[horizon] for horizon in horizons]
    ))

    numeric_columns = [
        CUE_COLUMN,
        *CONTROL_COLUMNS,
        *return_columns,
    ]

    read_columns = [
        "StkcdRecall",
        "Trddt",
        "Filling",
        "cueDefinition",
        *numeric_columns,
    ]

    column_types = {
        column: "float32"
        for column in numeric_columns
    }
    column_types.update({
        "StkcdRecall": "int32",
        "Filling": "float32",
        "cueDefinition": "string",
    })

    print(
        f"正在读取筛样规格（{INPUT_SAMPLE_SPEC}）面板："
        f"{INPUT_PANEL_PATH}",
        flush=True,
    )

    data = pd.read_csv(
        INPUT_PANEL_PATH,
        usecols=read_columns,
        dtype=column_types,
        encoding="utf-8-sig",
    )

    if data.empty:
        raise ValueError("正式合并面板没有观测。")

    data["Trddt"] = pd.to_datetime(
        data["Trddt"],
        errors="raise",
    )

    # 检查日期、唯一键和提示公司口径。
    if data["Trddt"].isna().any():
        raise ValueError("正式面板中存在缺失日期。")

    if data.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("被提示公司与日期的组合不唯一。")

    if not data["cueDefinition"].eq("hs300").fillna(False).all():
        raise ValueError("输入面板不是统一的沪深300提示口径。")

    # 提示、27项控制和两个基础收益窗口应当完整。
    required_columns = [
        CUE_COLUMN,
        *CONTROL_COLUMNS,
        *baseline_return_columns,
    ]

    for column in required_columns:
        if not np.isfinite(data[column].to_numpy()).all():
            raise ValueError(
                f"正式面板的 {column} 存在缺失或无穷值，"
                "请核对合并阶段的回归样本筛选。"
            )

    for column in [CUE_COLUMN, "Filling"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} 必须全部为0或1。")

    # 其余收益窗口允许缺失，由各次回归分别删除。
    # 无穷值属于异常，直接提示核对。
    for column in return_columns:
        if np.isinf(data[column].to_numpy()).any():
            raise ValueError(f"{column} 存在无穷值，请核对收益数据。")

    data.drop(columns=["cueDefinition"], inplace=True)
    data["Filling"] = data["Filling"].astype("uint8")
    data["StkcdRecall"] = data["StkcdRecall"].astype("category")
    data["yearQuarter"] = (
        data["Trddt"].dt.to_period("Q").astype("category")
    )

    print(
        f"图2正式面板读取完成：{len(data):,}条观测。",
        flush=True,
    )

    return data

def run_date_fe_regression(
    data: pd.DataFrame,
    horizon: int,
) -> dict:
    """按当前规格（6）直接估计一个累计收益窗口。"""
    outcome = OUTCOME_COLS[horizon]
    x_columns = [CUE_COLUMN] + CONTROL_COLUMNS

    needed = [
        outcome,
        "StkcdRecall",
        "Trddt",
        "Filling",
        "yearQuarter",
        *x_columns,
    ]

    reg = data[needed].dropna().copy()

    if reg.empty:
        raise ValueError(
            f"窗口[t,t+{horizon}]没有可用回归观测。"
        )

    # 一、构造公司季度和日期两组固定效应编号。
    reg["_fe_firm_yearquarter"] = (
        reg.groupby(
            ["StkcdRecall", "yearQuarter"],
            observed=True,
            sort=False,
        )
        .ngroup()
        .astype("int64")
    )

    reg["_fe_date"] = pd.factorize(
        reg["Trddt"],
        sort=False,
    )[0].astype("int64")

    fe_code_columns = [
        "_fe_firm_yearquarter",
        "_fe_date",
    ]

    # 二、递归删除固定效应单例组。
    reg, singleton_removed = drop_fe_singletons_iteratively(
        reg,
        fe_code_columns,
    )

    if reg.empty:
        raise ValueError(
            f"窗口[t,t+{horizon}]删除单例后没有剩余观测。"
        )

    absorb = pd.DataFrame(
        {
            column: pd.Categorical(reg[column])
            for column in fe_code_columns
        },
        index=reg.index,
    )

    # 三、构造公司、日期双向聚类编号。
    clusters = pd.DataFrame(
        {
            "firm": pd.factorize(
                reg["StkcdRecall"],
                sort=False,
            )[0],
            "date": pd.factorize(
                reg["Trddt"],
                sort=False,
            )[0],
        },
        index=reg.index,
    )

    # 四、显式加入常数项，吸收两组固定效应。
    X = reg[x_columns].astype("float64")
    X.insert(0, "const", 1.0)

    model = AbsorbingLS(
        dependent=reg[outcome].astype("float64"),
        exog=X,
        absorb=absorb,
        drop_absorbed=True,
    )

    result = model.fit(
        cov_type="unadjusted",
        debiased=False,
        method="hdfe",
        absorb_options={
            "options": {"tol": 1e-12}
        },
    )

    if CUE_COLUMN not in result.params.index:
        raise ValueError(
            f"窗口[t,t+{horizon}]的提示变量被固定效应完全吸收。"
        )

    # 五、沿用当前双向聚类修正和统计推断规则。
    inference, degrees, fe_rank, condition = (
        figure2_consistent_inference(
            model,
            result,
            clusters,
            reg,
            fe_code_columns,
            include_diagnostics=True,
        )
    )

    coefficient = float(result.params[CUE_COLUMN]) * 100
    standard_error = float(
        inference.loc[CUE_COLUMN, "标准误"]
    ) * 100
    t_value = float(inference.loc[CUE_COLUMN, "t值"])
    p_value = float(inference.loc[CUE_COLUMN, "p值"])
    critical_value = float(stats.t.ppf(0.975, degrees))

    # 六、整理当前窗口结果，收益系数统一为百分点。
    row = {
        **RESULT_SPECIFICATION,
        "口径说明": SPEC_DESCRIPTION,
        "窗口序号": horizon,
        "事件窗口": f"[t, t+{horizon}]",
        "因变量": outcome,
        "线索系数": coefficient,
        "聚类标准误": standard_error,
        "置信区间下界": (
            coefficient - critical_value * standard_error
        ),
        "置信区间上界": (
            coefficient + critical_value * standard_error
        ),
        "t值": t_value,
        "p值": p_value,
        "显著性": significance_stars(p_value),
        "回归样本量": int(result.nobs),
        "被提示公司聚类数": int(reg["StkcdRecall"].nunique()),
        "日期聚类数": int(reg["Trddt"].nunique()),
        "删除单例组观测数": singleton_removed,
        "正提示公司日数": int(reg[CUE_COLUMN].sum()),
        "填充起始日数": int(reg["Filling"].sum()),
        "残差自由度": degrees,
        "固定效应实际秩": fe_rank,
        "解释变量矩阵条件数": condition,
        "系数求解方法": "吸收最小二乘（图2本地估计）",
        "R方": float(result.rsquared),
    }

    del result, model, absorb, clusters, X, reg, inference
    gc.collect()

    return row


def current_run_information() -> dict[str, str]:
    """读取本次输入文件属性和计算版本，不读取面板内容或生成额外文件。"""
    if not INPUT_PANEL_PATH.is_file():
        raise FileNotFoundError(f"找不到正式合并面板：{INPUT_PANEL_PATH}")
    panel_stat = INPUT_PANEL_PATH.stat()
    calculation = {
        "本脚本": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "公共聚类算法": hashlib.sha256(
            (MODULE_DIR / "图2图3扩展_日期固定效应回归公共模块.py").read_bytes()
        ).hexdigest(),
        "依赖版本": {
            name: version(name)
            for name in ["numpy", "pandas", "scipy", "linearmodels", "pyhdfe"]
        },
    }
    calculation_version = hashlib.sha256(
        json.dumps(calculation, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "输入面板路径": str(INPUT_PANEL_PATH.resolve()),
        "输入面板字节数": str(panel_stat.st_size),
        "输入面板修改时间纳秒": str(panel_stat.st_mtime_ns),
        "计算版本": calculation_version,
    }


def validate_result_specification(results: pd.DataFrame) -> None:
    """在筛除不完整窗口之前检查每一行，拒绝旧口径或混合口径续算。"""
    for column, expected in RESULT_SPECIFICATION.items():
        if column not in results or not results[column].astype(str).eq(expected).all():
            raise ValueError(
                f"已有图2结果的“{column}”与当前表2口径不一致，不能续算。"
                "请不带 --resume（续算参数）从头运行；旧文件不会在检查时被删除。"
            )
    for column, expected in current_run_information().items():
        if column not in results or not results[column].astype(str).eq(expected).all():
            raise ValueError(
                f"已有图2结果缺少“{column}”或与当前输入、计算版本不一致，不能续算。"
                "请不带 --resume（续算参数）从头运行。"
            )


def validate_result_rows(results: pd.DataFrame) -> None:
    """检查一批完整窗口，可用于保存中间进度及最终绘图。"""
    validate_result_specification(results)
    horizons = pd.to_numeric(results["窗口序号"], errors="raise")
    if not horizons.isin(HORIZONS).all() or horizons.duplicated().any():
        raise ValueError("图2包含无效或重复窗口。")
    values = results[NUMERIC_RESULT_FIELDS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("图2结果包含缺失或非有限数值。")
    if not (results["聚类标准误"] > 0).all() or not (results["残差自由度"] > 0).all():
        raise ValueError("图2结果的标准误或残差自由度必须为正数。")
    if not results["p值"].between(0, 1).all():
        raise ValueError("图2结果包含无效p值。")
    if not (
        (results["置信区间下界"] <= results["线索系数"])
        & (results["线索系数"] <= results["置信区间上界"])
    ).all():
        raise ValueError("图2结果包含无效置信区间。")
    if results["系数求解方法"].isna().any():
        raise ValueError("图2结果缺少系数求解方法。")


def save_results(rows: list[dict]) -> pd.DataFrame:
    """检查口径和数值完整性，再保存逐窗口结果。"""
    results = (
        pd.DataFrame(rows)
        .sort_values("窗口序号")
        .reset_index(drop=True)
    )

    validate_result_rows(results)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = ROOT / "work" / "图2结果暂存"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_dir / (RESULT_PATH.name + ".tmp")

    results.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
    )
    temporary_path.replace(RESULT_PATH)

    return results

def load_completed_rows(resume: bool) -> list[dict]:
    """仅接受同口径、同输入和同计算版本；缺失诊断的窗口重新估计。"""
    if not resume or not RESULT_PATH.exists():
        return []
    # 纳秒修改时间必须按字符串读取，避免大整数转为浮点数后损失精度。
    existing = pd.read_csv(
        RESULT_PATH,
        dtype={column: "string" for column in RUN_INFORMATION_COLUMNS},
    )
    if existing.empty:
        return []
    validate_result_specification(existing)
    if "窗口序号" not in existing:
        raise ValueError("已有图2结果缺少窗口序号，无法续算。")
    horizons = pd.to_numeric(existing["窗口序号"], errors="raise")
    if not horizons.isin(HORIZONS).all():
        raise ValueError("已有图2结果的窗口序号必须为1至50的整数。")
    existing["窗口序号"] = horizons.astype(int)
    for column in RESULT_COMPLETION_FIELDS:
        if column not in existing:
            existing[column] = np.nan
    numeric = existing[NUMERIC_RESULT_FIELDS].apply(pd.to_numeric, errors="coerce")
    existing[NUMERIC_RESULT_FIELDS] = numeric
    complete = (
        np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
        & existing["系数求解方法"].notna()
    )
    incomplete_horizons = existing.loc[~complete, "窗口序号"].tolist()
    if incomplete_horizons:
        print(f"以下已有窗口的统计或诊断字段不完整，将重新估计：{incomplete_horizons}")
    completed = (
        existing.loc[complete]
        .drop_duplicates("窗口序号", keep="last")
        .sort_values("窗口序号")
    )
    if not completed.empty:
        validate_result_rows(completed)
    return completed.to_dict("records")


def validate_results(results: pd.DataFrame) -> None:
    """最终绘图必须有顺序完整的50个窗口。"""
    validate_result_rows(results)
    if results["窗口序号"].tolist() != HORIZONS:
        raise ValueError("日期固定效应版图2未包含顺序完整的50个窗口。")


def draw_figure2_date_fe(
    results: pd.DataFrame,
) -> None:
    """绘制加入交易日期固定效应的收益动态图。"""
    plot_data = (
        results
        .sort_values(
            "窗口序号"
        )
        .reset_index(
            drop=True
        )
    )

    font_path = Path(
        "/System/Library/Fonts/"
        "Hiragino Sans GB.ttc"
    )

    if font_path.exists():
        font_manager.fontManager.addfont(
            str(font_path)
        )

        font_name = (
            font_manager
            .FontProperties(
                fname=str(
                    font_path
                )
            )
            .get_name()
        )

        plt.rcParams[
            "font.sans-serif"
        ] = [
            font_name
        ]

    plt.rcParams[
        "axes.unicode_minus"
    ] = False

    figure, axis = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    x_values = plot_data[
        "窗口序号"
    ].to_numpy(
        dtype=np.float64,
    )

    coefficients = plot_data[
        "线索系数"
    ].to_numpy(
        dtype=np.float64,
    )

    standard_errors = plot_data[
        "聚类标准误"
    ].to_numpy(
        dtype=np.float64,
    )

    degrees = plot_data["残差自由度"].to_numpy(dtype=np.float64)

    for spec in ERROR_BAR_SPECS:
        critical_values = stats.t.ppf((1 + spec["confidence_level"]) / 2, degrees)
        interval_radius = critical_values * standard_errors
        axis.vlines(
            x_values,
            coefficients
            - interval_radius,
            coefficients
            + interval_radius,
            colors=spec[
                "color"
            ],
            linewidth=spec[
                "line_width"
            ],
            label=spec[
                "label"
            ],
            zorder=spec[
                "zorder"
            ],
        )

    # 与图中90%置信区间一致，判断10%水平显著性
    significant = plot_data["p值"].to_numpy(dtype=np.float64) < 0.10

    # 显著用红色，其余用黑色
    point_colors = np.where(significant, "#d62728", "#1a1a1a")

    axis.scatter(
        x_values,
        coefficients,
        color=point_colors,
        s=16,
        zorder=4,
    )

    # 只添加红点图例，不增加实际数据点
    axis.scatter(
        [],
        [],
        color="#d62728",
        s=16,
        label="红点：10%水平显著",
    )

    axis.axhline(
        0,
        color="#333333",
        linewidth=0.8,
    )

    axis.set_title(
        "收益动态和长期反转：历史沪深300提示\n"
        "个人365天排除；数行者分段新闻；合格填充日起点；全部年份\n"
        "固定效应：被提示公司×年度季度；交易日期（事件起点）",
        fontsize=12,
    )

    axis.set_xlabel(
        "事件窗口"
    )

    axis.set_ylabel(
        "提示变量系数（百分点）"
    )

    ticks = [
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ]

    axis.set_xticks(
        ticks
    )

    axis.set_xticklabels(
        [
            f"[t,t+{tick}]"
            for tick in ticks
        ]
    )

    axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=9,
        handlelength=1.4,
    )

    axis.spines[
        "top"
    ].set_visible(False)

    axis.spines[
        "right"
    ].set_visible(False)

    axis.tick_params(
        direction="out"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""
    parser = argparse.ArgumentParser(description="按当前表2主规格估计并绘制图2。")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "仅在输入面板、研究设置和计算版本一致且诊断完整时续算；"
            "面板重新生成或计算代码改变后，请从头运行。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """读取正式面板，逐窗口估计、保存并绘图。"""
    args = parse_args()

    run_information = current_run_information()
    existing_rows = load_completed_rows(resume=args.resume)
    completed = {
        int(row["窗口序号"])
        for row in existing_rows
    }
    pending = [
        horizon
        for horizon in HORIZONS
        if horizon not in completed
    ]
    rows = existing_rows.copy()

    if pending:
        data = read_figure2_panel(pending)

        for position, horizon in enumerate(pending, start=1):
            print(
                f"正在运行加入日期固定效应的图2：[t, t+{horizon}]"
                f"（待估计窗口 {position}/{len(pending)}）"
            )

            row = run_date_fe_regression(data, horizon)
            row.update(run_information)

            # 先检查并保存，再更新内存中的已完成记录。
            results = save_results(rows + [row])
            rows.append(row)

            print(
                f"窗口 {horizon} 完成："
                f"系数={row['线索系数']:.6f}，"
                f"标准误={row['聚类标准误']:.6f}"
            )

            data.drop(
                columns=[OUTCOME_COLS[horizon]],
                inplace=True,
            )
            gc.collect()

        del data
        gc.collect()

    else:
        print("已有结果包含全部50个完整窗口，无需重新估计。")
        results = (
            pd.DataFrame(rows)
            .sort_values("窗口序号")
            .reset_index(drop=True)
        )

    validate_results(results)
    draw_figure2_date_fe(results)

    print(f"日期固定效应版图2结果已保存：{RESULT_PATH}")
    print(f"日期固定效应版图2已保存：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
