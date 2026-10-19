"""独立估计表2收益回归，可直接执行本文件；本模块不绘图。

输入是模式公司和完整样本最终回归面板，输入一行代表一家被提示公司
在一个交易日的观测，唯一键为被提示公司与交易日。
沿用2005—2025年筛选、原字段类型、逐模型剔除缺失、递归剔除固定效应
单例、显式常数项、双向聚类与有限样本修正。两个窗口各运行六种原有规格。
收益直接使用来源面板已缩尾字段，不新增缩尾或其他样本筛选。

表2不使用订单失衡和表1专用字段；来源脚本中的订单失衡一对一左合并
不改变表2所用字段与行样本，因此本模块直接读取表2必要字段。
模型汇总一行代表“样本×收益窗口×规格”，系数明细再按变量展开。
输出集中到项目根目录下“图表结果／01_主线图表／表2_收益回归”，
保留原模型汇总与系数明细文件名、列名、单位和导出编码。
只在调用主函数或直接执行时读取面板并估计；导入本模块不读取面板或写文件。
"""

from __future__ import annotations

from pathlib import Path
import gc

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


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
PATTERN_PATH = PROJECT_ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"
FULL_PATH = PROJECT_ROOT / "中间结果/04_合并数据/Full样本_最终回归面板.csv"
OUTPUT_DIR = PROJECT_ROOT / "图表结果" / "01_主线图表" / "表2_收益回归"
TABLE2_RESULTS_PATH = OUTPUT_DIR / "表2_模型汇总.csv"
TABLE2_COEFFICIENTS_PATH = OUTPUT_DIR / "表2_系数明细.csv"
START_YEAR = 2005
END_YEAR = 2025


# 被提示公司的盈余公告列
own_ea_cols = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)


# 新闻列
media_cols = [
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]


# 收益列
return_cols = [
    "chAdjreturn_t_tp1_Recall_w",
    "chAdjreturn_t_tp10_Recall_w",
]


def significance_stars(p_value):
    """根据双侧检验的 p 值生成显著性星号。"""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


# 控制其他重大事件
table2_disclosure_cols = [
    "ownOtherEventER",
    "ownPerformanceForecastBrief",
]

# 完整控制变量：自身盈余公告列、其他重大事件列、媒体变量
table2_full_controls = (
    own_ea_cols
    + table2_disclosure_cols
    + media_cols
)

# 六个递进回归规格
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
        "控制变量": own_ea_cols,
        "固定效应": [],
    },
    {
        "规格": "(3)",
        "规格说明": "完整控制变量、无固定效应",
        "控制变量": table2_full_controls,
        "固定效应": [],
    },
    {
        "规格": "(4)",
        "规格说明": "完整控制变量、日期固定效应",
        "控制变量": table2_full_controls,
        "固定效应": ["date"],
    },
    {
        "规格": "(5)",
        "规格说明": "完整控制变量、公司×年度季度固定效应",
        "控制变量": table2_full_controls,
        "固定效应": ["firm_yearquarter"],
    },
    {
        "规格": "(6)",
        "规格说明": "完整控制变量、日期和公司×年度季度固定效应",
        "控制变量": table2_full_controls,
        "固定效应": ["date", "firm_yearquarter"],
    },
]

# 两个收益窗口；每个窗口依次运行上述六种规格
table2_outcomes = [
    (
        "chAdjreturn_t_tp1_Recall_w",
        "[t, t+1]",
    ),
    (
        "chAdjreturn_t_tp10_Recall_w",
        "[t, t+10]",
    ),
]


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
            standard_error = float(result.std_errors[variable])
            t_value = float(result.tstats[variable])
            p_value = float(result.pvalues[variable])
            stars = significance_stars(p_value)
            status = "已估计"
        else:
            beta = np.nan
            standard_error = np.nan
            t_value = np.nan
            p_value = np.nan
            stars = ""
            status = "被固定效应吸收"

        rows.append({
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


def run_return_regression(
    data,
    outcome,
    window_label,
    specification,
):
    """
    运行表2中的一个收益回归。

    标准误按被提示公司和交易日期双向聚类。
    """ 
    cue_col = "quarterLag1size90plus"
    control_cols = specification["控制变量"]
    fe_names = specification["固定效应"]

    x_cols = [cue_col] + control_cols # 解释变量=提示列+控制列

    needed = (
        [outcome, "StkcdRecall", "Trddt"] # outcome就是因变量
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

    result = model.fit(
        cov_type="clustered",
        clusters=clusters,
        debiased=True,
    )

    if cue_col not in result.params.index:
        raise ValueError(
            f"{outcome} 的 {specification['规格']} 中，"
            f"{cue_col} 被固定效应完全吸收。"
        )

    beta = float(result.params[cue_col])
    standard_error = float(result.std_errors[cue_col])
    t_value = float(result.tstats[cue_col])
    p_value = float(result.pvalues[cue_col])

    regression_row = {
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
        "R方": float(result.rsquared),
    }

    coefficient_rows = build_table2_coefficient_rows(
        result=result,
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


def run_table2_sample(
    data,
    sample_name,
    sample_label,
):
    model_rows = []
    coefficient_rows = []

    for outcome, window_label in table2_outcomes:
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

    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(coefficient_rows),
    )


def load_table2_panel(input_path: Path, sample_label: str) -> pd.DataFrame:
    """按来源脚本的类型、研究期和唯一键口径读取一家被提示公司一天一行的面板。"""
    input_cols = list(dict.fromkeys(
        [
            "StkcdRecall",
            "Trddt",
            "year",
            "yearQuarter",
            "quarterLag1size90plus",
        ]
        + return_cols
        + table2_full_controls
    ))
    numeric_cols = [
        col for col in input_cols
        if col not in {"StkcdRecall", "Trddt", "yearQuarter"}
    ]
    dtype = {"StkcdRecall": "string", "yearQuarter": "string"}
    dtype.update({col: "float32" for col in numeric_cols})

    print(f"读取{sample_label}最终回归面板……")
    data = pd.read_csv(
        input_path,
        usecols=input_cols,
        dtype=dtype,
        parse_dates=["Trddt"],
    )
    data["StkcdRecall"] = data["StkcdRecall"].str.strip().str.zfill(6)
    if START_YEAR is not None:
        data = data.loc[data["year"] >= START_YEAR].copy()
    if END_YEAR is not None:
        data = data.loc[data["year"] <= END_YEAR].copy()
    if data.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError(f"{sample_label}面板的被提示公司与交易日不是唯一键。")

    # 来源脚本在表1合并完成后把模式公司标识转成分类类型；完整样本保持字符串。
    if input_path == PATTERN_PATH:
        data["StkcdRecall"] = data["StkcdRecall"].astype("category")
        data["yearQuarter"] = data["yearQuarter"].astype("category")
    print(f"{sample_label}样本行数：{len(data):,}")
    return data


def main() -> list[Path]:
    """独立估计两类样本共24个模型，返回模型汇总和系数明细路径。"""
    main = load_table2_panel(PATTERN_PATH, "模式公司")
    # 运行模式公司样本
    (
        table2_pattern_results,
        table2_pattern_coefficients,
    ) = run_table2_sample(
        data=main,
        sample_name="pattern",
        sample_label="A. 模式公司主样本",
    )

    # 按来源脚本顺序先释放模式公司面板，再读取完整样本。
    del main
    gc.collect()

    full = load_table2_panel(FULL_PATH, "完整样本")
    # 在全样本中按相同顺序重复两个窗口、六种规格
    (
        table2_full_results,
        table2_full_coefficients,
    ) = run_table2_sample(
        data=full,
        sample_name="full",
        sample_label="B. 完整样本",
    )

    del full
    gc.collect()

    # 合并两类样本的24个回归结果，供后续核对和绘制论文表格
    table2_results = pd.concat(
        [
            table2_pattern_results,
            table2_full_results,
        ],
        ignore_index=True,
    )

    table2_coefficients = pd.concat(
        [
            table2_pattern_coefficients,
            table2_full_coefficients,
        ],
        ignore_index=True,
    )

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
