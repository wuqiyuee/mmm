"""独立估计表2收益回归，可直接执行本文件。

直接读取顶部选定筛样规格的沪深300正式合并面板。
提示、控制变量、收益缩尾和合并阶段两轮回归样本筛选，
由合并脚本完成；具体研究口径见本文件顶部设置。

对所选面板估计两日、十一日两个收益窗口，
每个窗口包含六个递进回归规格，共12个模型。
第（6）列为公司×年度季度和日期双固定效应主规格。
输入面板的筛样规格选择与这六个回归规格分别设置。

各模型按自身固定效应递归删除单例，
沿用当前双向聚类协方差修正及统计检验自由度计算规则。
收益使用正式面板中已有的缩尾值。

模型汇总和系数明细文件名均包含提示口径及筛样规格。
本文件负责估计与导出，绘图由同目录的绘制脚本完成。
仅在直接运行或调用主函数时读取研究面板并估计。
"""

from __future__ import annotations

from pathlib import Path
import gc
import sys

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components


# 一、定位项目，设置正式面板路径并导入公共双向聚类协方差算法
def find_project_root() -> Path:
    """从当前文件的祖先目录寻找同时含项目说明、代码和中间结果的根目录。"""
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "AGENTS.md").is_file()
            and (candidate / "code").is_dir()
            and (candidate / "中间结果").is_dir()
        ):
            return candidate
    raise FileNotFoundError("无法找到包含项目说明、代码和中间结果的项目根目录。")


PROJECT_ROOT = find_project_root()
COMMON_DIR = PROJECT_ROOT / "code" / "07_实证分析" / "90_公共模块"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
# 正式输入：直接使用合并脚本生成的沪深300最终面板。
# 选择输入面板的回归样本筛选规格：5或6。
INPUT_SAMPLE_SPEC = 5

if INPUT_SAMPLE_SPEC not in (5, 6):
    raise ValueError("输入面板的筛样规格只能选择5或6。")

INPUT_PANEL_PATH = PROJECT_ROOT / (
    "中间结果/04_合并数据/"
    f"Pattern样本_最终回归面板_沪深300_筛样规格{INPUT_SAMPLE_SPEC}.csv"
)

# 提示变量及当前研究口径说明。
CUE_COLUMN = "quarterLag1HS300"

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

# 21项自身财报公告控制：公告前10日、当日、公告后10日。
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

from 图2图3扩展_日期固定效应回归公共模块 import two_way_cluster_covariance


# 二、设置表2输出路径、六个回归规格和两个收益窗口
OUTPUT_DIR = (
    PROJECT_ROOT
    / "图表结果"
    / "01_主线图表"
    / "表2_收益回归"
)

OUTPUT_TAG = f"沪深300_筛样规格{INPUT_SAMPLE_SPEC}"

TABLE2_RESULTS_PATH = OUTPUT_DIR / (
    f"表2_模型汇总_{OUTPUT_TAG}.csv"
)

TABLE2_COEFFICIENTS_PATH = OUTPUT_DIR / (
    f"表2_系数明细_{OUTPUT_TAG}.csv"
)


# 六个回归规格使用本文件上方定义的控制变量。
table2_specifications = [
    {
        "规格": "(1)",
        "规格说明": "无控制变量、无固定效应",
        "控制变量": [],
        "固定效应": [],
    },
    {
        "规格": "(2)",
        "规格说明": "自身盈余公告控制、无固定效应",
        "控制变量": OWN_EA_COLUMNS,
        "固定效应": [],
    },
    {
        "规格": "(3)",
        "规格说明": "完整控制变量、无固定效应",
        "控制变量": CONTROL_COLUMNS,
        "固定效应": [],
    },
    {
        "规格": "(4)",
        "规格说明": "完整控制变量、日期固定效应",
        "控制变量": CONTROL_COLUMNS,
        "固定效应": ["date"],
    },
    {
        "规格": "(5)",
        "规格说明": "完整控制变量、公司×年度季度固定效应",
        "控制变量": CONTROL_COLUMNS,
        "固定效应": ["firm_yearquarter"],
    },
    {
        "规格": "(6)",
        "规格说明": "完整控制变量、日期和公司×年度季度固定效应",
        "控制变量": CONTROL_COLUMNS,
        "固定效应": ["date", "firm_yearquarter"],
    },
]

# 每项依次为：窗口终点、因变量列名、窗口标签。
# 终点1对应两日累计收益，终点10对应十一日累计收益。
# 读取收益列、检查收益完整性和运行回归均使用这一份设置。
table2_outcomes = [
    (
        1,
        "chAdjreturn_t_tp1_Recall_w",
        "[t, t+1]",
    ),
    (
        10,
        "chAdjreturn_t_tp10_Recall_w",
        "[t, t+10]",
    ),
]


# 三、表2单模型估计与结果整理；当前两个图2在各自脚本内独立估计
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


def build_table2_coefficient_rows(
    result,
    inference,
    x_cols,
    cue_col,
    outcome,
    window_label,
    specification,
    fe_names,
):
    """整理表2单个模型中线索变量和全部控制变量的估计结果。"""
    rows = []

    for variable_order, variable in enumerate(x_cols, start=1):
        estimated = variable in result.params.index

        if estimated:
            beta = float(result.params[variable])
            standard_error = float(inference.loc[variable, "标准误"])
            t_value = float(inference.loc[variable, "t值"])
            p_value = float(inference.loc[variable, "p值"])
            stars = significance_stars(p_value)
            status = "已估计" if np.isfinite(standard_error) else "双向聚类方差非正"
        else:
            beta = np.nan
            standard_error = np.nan
            t_value = np.nan
            p_value = np.nan
            stars = ""
            status = "被固定效应吸收"

        rows.append({
            "口径标识": SPEC_ID,
            "提示变量": CUE_COLUMN,
            "收益窗口": window_label,
            "因变量": outcome,
            "规格": specification["规格"],
            "规格说明": specification["规格说明"],
            "变量": variable,
            "变量顺序": variable_order,
            "变量类型": (
                "线索事件"
                if variable == cue_col
                else "控制变量"
            ),
            "估计状态": status,
            "系数": beta,
            "聚类标准误": standard_error,
            "系数（百分点）": beta * 100,
            "聚类标准误（百分点）": standard_error * 100,
            "t值": t_value,
            "p值": p_value,
            "显著性": stars,
            "日期固定效应": (
                "是" if "date" in fe_names else "否"
            ),
            "被提示公司×年度季度固定效应": (
                "是"
                if "firm_yearquarter" in fe_names
                else "否"
            ),
            "回归样本量": int(result.nobs),
            "R方": float(result.rsquared),
        })

    return rows


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


def run_return_regression(
    data,
    outcome,
    window_label,
    specification,
    *,
    include_diagnostics=False,
):
    """
    运行表2中的一个收益回归。

    标准误按被提示公司和交易日期双向聚类。
    """ 
    cue_col = CUE_COLUMN
    control_cols = specification["控制变量"]
    fe_names = specification["固定效应"]

    x_cols = [cue_col] + control_cols # 解释变量=提示列+控制列

    needed = (
        [outcome, "StkcdRecall", "Trddt", "Filling"] # 保留填充标记供报告样本组成
        + x_cols
    )

    if "firm_yearquarter" in fe_names:
        needed.append("yearQuarter")

    needed = list(dict.fromkeys(needed))

    # 删除本次回归所需变量中存在缺失值的观测
    reg = data[needed].dropna().copy()

    # ------------------------------------------------------------
    # 构造固定效应编号
    # ------------------------------------------------------------
    fe_code_cols = []

    if "firm_yearquarter" in fe_names:
        reg["_fe_firm_yearquarter"] = (
            reg.groupby(
                ["StkcdRecall", "yearQuarter"],
                observed=True, # 如果分组变量是分类变量，只处理数据中实际出现的组合，不生成没有出现过的公司—季度组合。
                sort=False,# 不按照公司代码和季度排序编号，基本按照组合在数据中首次出现的顺序分配编号。编号顺序不会影响回归结果。
            )
            .ngroup() # 相同组合会得到相同编号，不同组合得到不同编号
            .astype("int64")
        )
        fe_code_cols.append("_fe_firm_yearquarter")# 把新生成的固定效应列名加入列表

    # 构造日期固定效应
    if "date" in fe_names:
        reg["_fe_date"] = pd.factorize(
            reg["Trddt"],
            sort=False,
        )[0].astype("int64")
        fe_code_cols.append("_fe_date")

    # 删除高维固定效应回归的单例组
    reg, singleton_removed = drop_fe_singletons_iteratively(
        reg,
        fe_code_cols,
    )

    # ------------------------------------------------------------
    # 构造需要吸收的固定效应
    # ------------------------------------------------------------
    absorb = None

    if fe_code_cols:
        absorb = pd.DataFrame(
            {
                col: pd.Categorical(reg[col])
                for col in fe_code_cols
            },
            index=reg.index,
        )

    # ------------------------------------------------------------
    # 构造被提示公司和日期双向聚类编号
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 构造解释变量
    # ------------------------------------------------------------
    X = reg[x_cols].astype("float64")

    # 显式加入常数项
    X.insert(0, "const", 1.0)

    model = AbsorbingLS(
        dependent=reg[outcome].astype("float64"),
        exog=X,
        absorb=absorb,
        drop_absorbed=True,
    )

    fit_options = {"cov_type": "unadjusted", "debiased": False}

    if fe_code_cols:
        fit_options["method"] = "hdfe"

        # 仅多维固定效应设置迭代收敛容差
        if len(fe_code_cols) > 1:
            fit_options["absorb_options"] = {
                "options": {"tol": 1e-12}
            }

    result = model.fit(**fit_options)

    if cue_col not in result.params.index:
        raise ValueError(
            f"{outcome} 的 {specification['规格']} 中，"
            f"{cue_col} 被固定效应完全吸收。"
        )

    beta = float(result.params[cue_col])
    inference, degrees, fe_rank, xpx_condition = figure2_consistent_inference(
        model, result, clusters, reg, fe_code_cols, include_diagnostics=True,
    )
    standard_error = float(inference.loc[cue_col, "标准误"])
    t_value = float(inference.loc[cue_col, "t值"])
    p_value = float(inference.loc[cue_col, "p值"])

    regression_row = {
        "口径标识": SPEC_ID,
        "口径说明": SPEC_DESCRIPTION,
        "提示变量": CUE_COLUMN,
        "主规格": "是" if specification["规格"] == "(6)" else "否",
        "收益窗口": window_label,
        "因变量": outcome,
        "规格": specification["规格"],
        "规格说明": specification["规格说明"],

        # 原始收益率使用小数单位
        "线索系数": beta,
        "聚类标准误": standard_error,

        # 乘100后转换为百分点
        "线索系数（百分点）": beta * 100,
        "聚类标准误（百分点）": standard_error * 100,

        "t值": t_value,
        "p值": p_value,
        "显著性": significance_stars(p_value),
        "回归样本量": int(result.nobs),
        "被提示公司聚类数": int(reg["StkcdRecall"].nunique()),
        "日期聚类数": int(reg["Trddt"].nunique()),
        "删除单例组观测数": singleton_removed,
        "正提示公司日数": int(reg[cue_col].sum()),
        "填充起始日数": int(reg["Filling"].sum()),
        "残差自由度": degrees,
        "固定效应实际秩": fe_rank,
        "R方": float(result.rsquared),
    }

    # 仅图2请求诊断时添加字段，表2原有导出列保持不变。
    if include_diagnostics:
        regression_row["解释变量矩阵条件数"] = xpx_condition
        regression_row["系数求解方法"] = "吸收最小二乘（复用表2估计器）"

    coefficient_rows = build_table2_coefficient_rows(
        result=result,
        inference=inference,
        x_cols=x_cols,
        cue_col=cue_col,
        outcome=outcome,
        window_label=window_label,
        specification=specification,
        fe_names=fe_names,
    )

    del result, model, absorb, clusters, X, reg
    gc.collect()

    return regression_row, coefficient_rows


# 四、直接读取正式合并面板，组装表2回归数据
def load_selected_main_panel() -> pd.DataFrame:
    """读取合并阶段已经筛好的样本，不重新缩尾或改变筛样口径。"""

    if not INPUT_PANEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到正式合并面板：{INPUT_PANEL_PATH}"
        )

    # 只读取表2需要的两个收益窗口及控制变量。
    return_columns = [
        column for _, column, _ in table2_outcomes
    ]

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

    # 沿用此前读取时的单精度数值类型。
    column_types = {
        column: "float32"
        for column in numeric_columns
    }
    column_types.update({
        "StkcdRecall": "int32",
        "Filling": "float32",
        "cueDefinition": "string",
    })

    print(f"正在读取正式合并面板：{INPUT_PANEL_PATH}", flush=True)

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

    # 面板应已完成两窗及完整控制变量的共同资格筛选。
    # 出现异常时提示核对合并步骤，不在这里静默删样本。
    for column in numeric_columns:
        if not np.isfinite(data[column].to_numpy()).all():
            raise ValueError(
                f"正式面板的 {column} 存在缺失或无穷值，"
                "请核对合并阶段的回归样本筛选。"
            )

    for column in [CUE_COLUMN, "Filling"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} 必须全部为0或1。")

    # 生成回归需要的分类标识。
    data.drop(columns=["cueDefinition"], inplace=True)
    data["Filling"] = data["Filling"].astype("uint8")
    data["StkcdRecall"] = data["StkcdRecall"].astype("category")
    data["yearQuarter"] = (
        data["Trddt"].dt.to_period("Q").astype("category")
    )

    print(f"表2正式面板读取完成：{len(data):,}条观测。", flush=True)

    return data

# 五、按两个窗口、六个规格依次估计，保存表2结果
def main() -> list[Path]:
    """估计选定主样本的两窗口六规格，共12模型；不夹入旧口径完整样本。"""
    # 1. 读取当前选定的模式关系主样本。
    print(f"当前正式表2口径：{SPEC_DESCRIPTION}")
    data = load_selected_main_panel()
    sample_name = "pattern"
    sample_label = "A. 沪深300提示主样本（模式关系）"

    # 2. 每个收益窗口依次运行六个规格，汇总模型信息和全部系数。
    model_rows = []
    coefficient_rows = []
    for _, outcome, window_label in table2_outcomes:
        for specification in table2_specifications:
            print(
                f"正在运行表2：{sample_label}，"
                f"{window_label}，"
                f"规格 {specification['规格']}"
            )

            model_row, model_coefficient_rows = run_return_regression(
                data=data,
                outcome=outcome,
                window_label=window_label,
                specification=specification,
            )
            model_rows.append(
                {
                    "样本": sample_name,
                    "样本说明": sample_label,
                    **model_row,
                }
            )
            coefficient_rows.extend(
                {
                    "样本": sample_name,
                    "样本说明": sample_label,
                    **row,
                }
                for row in model_coefficient_rows
            )

    del data
    gc.collect()
    table2_results = pd.DataFrame(model_rows)
    table2_coefficients = pd.DataFrame(coefficient_rows)

    # 3. 沿用原路径、列顺序、单位和编码，保存两张结果表。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table2_results.to_csv(
        TABLE2_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    table2_coefficients.to_csv(
        TABLE2_COEFFICIENTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n表2：收益回归结果")
    print(table2_results.to_string(index=False))
    print(f"\n表2结构化结果已保存：{TABLE2_RESULTS_PATH}")
    print(
        f"表2全部系数明细已保存："
        f"{TABLE2_COEFFICIENTS_PATH}"
    )

    return [TABLE2_RESULTS_PATH, TABLE2_COEFFICIENTS_PATH]


if __name__ == "__main__":
    main()
