"""表4规模异质性的独立估计入口（两日收益扩展口径）。

来源：code/07_实证分析/99_历史与待核对/原综合脚本/回归.py（原回归脚本）第1399—1911行独立运行准备区。
本任务延续当前项目扩展：因变量为[t,t+1]两日特征调整收益的既有缩尾列，
与论文表4原窗口的关系由项目研究设计另行解释；此处不改成十日收益。
输入：规律公告公司最终回归面板。只读取本任务字段，不构造表1主面板。
口径：2005—2025年，全部、小型、大型被提示公司三组；四种提示公司规模线索同时入模；
逐模型删除所需字段缺失，递归删除公司季度固定效应单例；保留常数项，
吸收被提示公司×年度季度固定效应，按被提示公司和实际日期双向聚类并作有限样本修正。
直接使用原先已缩尾的收益列，不再进行额外缩尾；数值结果同时保留原小数及百分点列。
输出：项目根目录/图表结果/01_主线图表/表4_规模异质性/下两张数值表，
完整系数图片由本任务绘图入口存同目录。导入不读面板、不创建结果目录。
"""

from __future__ import annotations

from pathlib import Path
import gc

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


def find_project_root() -> Path:
    """沿祖先目录寻找项目标识，不依赖脚本固定层数。"""
    for candidate in Path(__file__).resolve().parents:
        if ((candidate / "AGENTS.md").is_file()
                and (candidate / "code").is_dir()
                and (candidate / "中间结果").is_dir()):
            return candidate
    raise FileNotFoundError("未找到含项目说明、代码及中间结果目录的项目根目录。")


PROJECT_ROOT = find_project_root()
PATTERN_PATH = PROJECT_ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"
OUTPUT_DIR = PROJECT_ROOT / "图表结果/01_主线图表/表4_规模异质性"
RESULTS_PATH = OUTPUT_DIR / "表4_规模异质性结果.csv"
COEFFICIENTS_PATH = OUTPUT_DIR / "表4_规模异质性系数明细.csv"

START_YEAR = 2005
END_YEAR = 2025


# =============================================================================
# 控制变量
# =============================================================================

own_ea_cols = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)

disclosure_cols = [
    "ownOtherEventER",
    "ownPerformanceForecastBrief",
]

media_cols = [
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]


# =============================================================================
# 公共函数
# =============================================================================

def significance_stars(p_value):
    """根据双侧检验的p值添加显著性星号。"""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def drop_fe_singletons_iteratively(
    reg,
    fe_code_cols,
):
    """递归删除固定效应单例组。"""
    if not fe_code_cols:
        return reg, 0

    total_removed = 0

    while True:
        singleton = np.zeros(
            len(reg),
            dtype=bool,
        )

        for col in fe_code_cols:
            codes = reg[col].to_numpy(
                dtype=np.int64,
                copy=False,
            )

            counts = np.bincount(codes)

            singleton |= (
                counts[codes] == 1
            )

        removed_now = int(
            singleton.sum()
        )

        if removed_now == 0:
            break

        total_removed += removed_now

        reg = reg.loc[
            ~singleton
        ].copy()

    return reg, total_removed

# =============================================================================
# Table 4：按照公司规模进行横截面切分
# =============================================================================

table4_outcome = "chAdjreturn_t_tp1_Recall_w"

table4_cue_cols = [
    "quarterLag1size90plus",
    "quarterLag1size50_90",
    "quarterLag1size10_50",
    "quarterLag1size10minus",
]

table4_control_cols = (
    own_ea_cols
    + disclosure_cols
    + media_cols
)

table4_input_cols = list(dict.fromkeys(
    [
        "StkcdRecall",
        "Trddt",
        "year",
        "yearQuarter",
        "smallFirm",
        table4_outcome,
    ]
    + table4_cue_cols
    + table4_control_cols
))

table4_numeric_cols = [
    col
    for col in table4_input_cols
    if col not in {
        "StkcdRecall",
        "Trddt",
        "yearQuarter",
    }
]

table4_dtype = {
    "StkcdRecall": "string",
    "yearQuarter": "string",
}

table4_dtype.update({
    col: "float32"
    for col in table4_numeric_cols
})



table4_specifications = [
    {
        "规格": "(1)",
        "样本": "全部规律公告公司样本",
        "smallFirm取值": None,
    },
    {
        "规格": "(2)",
        "样本": "小型被提示公司",
        "smallFirm取值": 1,
    },
    {
        "规格": "(3)",
        "样本": "大型被提示公司",
        "smallFirm取值": 0,
    },
]


def run_table4_regression(data, specification):
    x_cols = (
        table4_cue_cols
        + table4_control_cols
    )

    small_firm_value = specification[
        "smallFirm取值"
    ]

    if small_firm_value is None:
        sample = data
    else:
        sample = data.loc[
            data["smallFirm"].eq(
                small_firm_value
            )
        ]

    needed = list(dict.fromkeys(
        [
            table4_outcome,
            "StkcdRecall",
            "Trddt",
            "yearQuarter",
        ]
        + x_cols
    ))

    reg = sample[needed].dropna().copy()

    # 构造被提示公司×年度季度固定效应
    reg["_fe_firm_yearquarter"] = (
        reg.groupby(
            [
                "StkcdRecall",
                "yearQuarter",
            ],
            observed=True,
            sort=False,
        )
        .ngroup()
        .astype("int64")
    )

    reg, singleton_removed = (
        drop_fe_singletons_iteratively(
            reg,
            ["_fe_firm_yearquarter"],
        )
    )

    absorb = pd.DataFrame(
        {
            "firm_yearquarter":
                pd.Categorical(
                    reg[
                        "_fe_firm_yearquarter"
                    ]
                ),
        },
        index=reg.index,
    )

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

    X = reg[x_cols].astype("float64")

    # 加入常数项
    X.insert(0, "const", 1.0)

    model = AbsorbingLS(
        dependent=reg[
            table4_outcome
        ].astype("float64"),
        exog=X,
        absorb=absorb,
        drop_absorbed=True,
    )

    result = model.fit(
        cov_type="clustered",
        clusters=clusters,
        debiased=True,
    )

    model_info = {
        "规格": specification["规格"],
        "样本": specification["样本"],
        "因变量": table4_outcome,
        "控制变量": "是",
        "被提示公司×年度季度固定效应": "是",
        "日期固定效应": "否",
        "回归样本量": int(result.nobs),
        "被提示公司聚类数": int(
            reg["StkcdRecall"].nunique()
        ),
        "日期聚类数": int(
            reg["Trddt"].nunique()
        ),
        "删除单例组观测数":
            singleton_removed,
        "R方": float(result.rsquared),
    }

    coefficient_rows = []

    for variable_order, variable in enumerate(
        x_cols,
        start=1,
    ):
        estimated = (
            variable in result.params.index
        )

        if estimated:
            beta = float(
                result.params[variable]
            )
            standard_error = float(
                result.std_errors[variable]
            )
            t_value = float(
                result.tstats[variable]
            )
            p_value = float(
                result.pvalues[variable]
            )
            stars = significance_stars(
                p_value
            )
            status = "已估计"
        else:
            beta = np.nan
            standard_error = np.nan
            t_value = np.nan
            p_value = np.nan
            stars = ""
            status = "被固定效应吸收"

        coefficient_rows.append({
            **model_info,
            "变量": variable,
            "变量顺序": variable_order,
            "变量类型": (
                "规模线索变量"
                if variable
                in table4_cue_cols
                else "控制变量"
            ),
            "估计状态": status,

            # 原始收益率采用小数单位
            "系数": beta,
            "聚类标准误": standard_error,

            # 转换为百分点展示
            "系数（百分点）": (
                beta * 100
                if pd.notna(beta)
                else np.nan
            ),
            "聚类标准误（百分点）": (
                standard_error * 100
                if pd.notna(standard_error)
                else np.nan
            ),

            "t值": t_value,
            "p值": p_value,
            "显著性": stars,
        })

    del (
        result,
        model,
        absorb,
        clusters,
        X,
        reg,
    )
    gc.collect()

    return model_info, coefficient_rows



def main() -> list[Path]:
    """独立读取本任务面板、估计三个规格，并返回两张数值表的路径。"""
    print("读取表4所需规律公告公司样本……")

    table4_data = pd.read_csv(
        PATTERN_PATH,
        usecols=table4_input_cols,
        dtype=table4_dtype,
        parse_dates=["Trddt"],
    )

    table4_data["StkcdRecall"] = (
        table4_data["StkcdRecall"]
        .str.strip()
        .str.zfill(6)
    )

    if START_YEAR is not None:
        table4_data = table4_data.loc[
            table4_data["year"] >= START_YEAR
        ].copy()

    if END_YEAR is not None:
        table4_data = table4_data.loc[
            table4_data["year"] <= END_YEAR
        ].copy()

    if table4_data.duplicated(
        ["StkcdRecall", "Trddt"]
    ).any():
        raise ValueError(
            "表4样本的被提示公司和交易日期不是唯一键。"
        )

    table4_data["StkcdRecall"] = (
        table4_data["StkcdRecall"]
        .astype("category")
    )

    table4_data["yearQuarter"] = (
        table4_data["yearQuarter"]
        .astype("category")
    )


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table4_model_rows = []
    table4_coefficient_rows = []

    for specification in table4_specifications:
        print(
            f"正在运行表4："
            f"{specification['规格']}，"
            f"{specification['样本']}"
        )

        model_info, coefficient_rows = (
            run_table4_regression(
                table4_data,
                specification,
            )
        )

        table4_model_rows.append(model_info)
        table4_coefficient_rows.extend(
            coefficient_rows
        )

    table4_results = pd.DataFrame(
        table4_model_rows
    )

    table4_coefficients = pd.DataFrame(
        table4_coefficient_rows
    )

    # 只提取表4关注的四个规模线索变量
    table4_core_results = (
        table4_coefficients.loc[
            table4_coefficients[
                "变量"
            ].isin(table4_cue_cols)
        ]
        .copy()
    )

    table4_core_results.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    table4_coefficients.to_csv(
        COEFFICIENTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n表4：规模异质性核心结果")
    print(
        table4_core_results.to_string(
            index=False
        )
    )

    print(
        f"\n表4核心结果已保存："
        f"{RESULTS_PATH}"
    )

    print(
        f"表4全部系数明细已保存："
        f"{COEFFICIENTS_PATH}"
    )
    del table4_data
    gc.collect()
    return [RESULTS_PATH, COEFFICIENTS_PATH]


if __name__ == "__main__":
    main()
