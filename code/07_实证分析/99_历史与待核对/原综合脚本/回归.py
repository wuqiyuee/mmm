# 本脚本按论文表格顺序运行描述性统计、主收益回归、订单不平衡回归、机制检验、替代 cue、交易策略和稳健性检验。

from pathlib import Path
import gc

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS # 吸收固定效应
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"
font_manager.fontManager.addfont(font_path)
table_font = font_manager.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False

#===============================================================================
# 路径设置
#===============================================================================

ROOT = Path(__file__).resolve().parents[4]
TABLE_IMAGE_DIR = (
    ROOT
    / "图表结果/论文复现/表格"
    / "01_正文表"
    / "表1_描述性统计"
)

PATTERN_PATH = ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"
FULL_PATH = (
    ROOT / "中间结果/04_合并数据/"
    "Full样本_最终回归面板.csv"
)
OIB_PATH = ROOT / "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv"


TABLE1_IMAGE_PATH = (
    TABLE_IMAGE_DIR / "table1_描述性统计.png"
)

TABLE2_RESULTS_PATH = (
    ROOT / "中间结果/08_回归结果/"
    "表2_模型汇总.csv"
)

TABLE2_COEFFICIENTS_PATH = (
    ROOT / "中间结果/08_回归结果/"
    "表2_系数明细.csv"
)

TABLE2_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

TABLE3_RESULTS_PATH = (
    ROOT / "中间结果/08_回归结果/"
    "表3_散户订单失衡结果.csv"
)

TABLE3_COEFFICIENTS_PATH = (
    ROOT / "中间结果/08_回归结果/"
    "表3_系数明细.csv"
)

APPENDIX_A2_RESULTS_PATH = (
    ROOT / "中间结果/08_回归结果/"
    "附表A2_OIBnum_买卖次数订单失衡结果.csv"
)

APPENDIX_A2_COEFFICIENTS_PATH = (
    ROOT / "中间结果/08_回归结果/"
    "附表A2_OIBnum_买卖次数订单失衡系数明细.csv"
)

TABLE3_RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE4_RESULTS_PATH = (
    ROOT / "中间结果/08_回归结果/"
    "表4_规模异质性结果.csv"
)

TABLE4_COEFFICIENTS_PATH = (
    ROOT / "中间结果/08_回归结果/"
    "表4_规模异质性系数明细.csv"
)




#===============================================================================
# 一些参数设置
#===============================================================================
# 默认研究期：2005—2025年（含首尾年）
START_YEAR = 2005
END_YEAR = 2025

#===============================================================================
# Table 1:描述性统计
#===============================================================================

# 被提示公司的盈余公告列
own_ea_cols = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)


# 其余重大事件列
disclosure_cols = [
    "ownOtherEventER",
    "ownPerformanceForecastBrief",
]

# 新闻列
media_cols = [
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]

# 盈余惊喜列
sue_cols = [
    "sue1CueLag1S90pEW_w",
    "sue1CueLag1S90pVW_w",
    "sue2CueLag1S90pEW_w",
    "sue2CueLag1S90pVW_w",
    "sue3CueLag1S90pEW_w",
    "sue3CueLag1S90pVW_w",
]

# 收益列
return_cols = [
    "chAdjreturn_t_tp1_Recall_w",
    "chAdjreturn_t_tp10_Recall_w",
]

main_cols = list(dict.fromkeys(
    [
        "StkcdRecall",
        "Trddt",
        "year",
        "yearQuarter",
        "quarterLag1size90plus",
        "nrCuesLag1S90p",
    ]
    + return_cols
    + own_ea_cols
    + disclosure_cols
    + media_cols
    + sue_cols
))

# 数字列
numeric_cols = [
    c for c in main_cols
    if c not in {"StkcdRecall", "Trddt", "yearQuarter"}
]

main_dtype = {"StkcdRecall": "string", "yearQuarter": "string"}
main_dtype.update({c: "float32" for c in numeric_cols})

print("读取模式公司最终回归面板……")
main = pd.read_csv(
    PATTERN_PATH,
    usecols=main_cols,
    dtype=main_dtype,
    parse_dates=["Trddt"],
)

main["StkcdRecall"] = main["StkcdRecall"].str.strip().str.zfill(6)

if START_YEAR is not None:
    main = main.loc[main["year"] >= START_YEAR].copy()
if END_YEAR is not None:
    main = main.loc[main["year"] <= END_YEAR].copy()

if main.duplicated(["StkcdRecall", "Trddt"]).any():
    raise ValueError("模式公司面板的 StkcdRecall + Trddt 不是唯一键。")

print(f"模式公司样本行数：{len(main):,}")

# 订单不平衡变量
oib_raw_cols = [
    "OIBvol_avg_t_tp1",
    "OIBvol_avg_t_tp10",
]

print("读取订单失衡数据……")
oib = pd.read_csv(
    OIB_PATH,
    usecols=["Stkcd", "Trddt"] + oib_raw_cols,
    dtype={
        "Stkcd": "string",
        "OIBvol_avg_t_tp1": "float32",
        "OIBvol_avg_t_tp10": "float32",
    },
    parse_dates=["Trddt"],
)

oib["Stkcd"] = oib["Stkcd"].str.strip().str.zfill(6)
oib = oib.rename(columns={"Stkcd": "StkcdRecall"})

if oib.duplicated(["StkcdRecall", "Trddt"]).any():
    raise ValueError("订单失衡表的 StkcdRecall + Trddt 不是唯一键。")

# 合并回归面板数据与订单不平衡数据
main = main.merge(
    oib,
    on=["StkcdRecall", "Trddt"],
    how="left",
    validate="one_to_one",
)

# 删除订单不平衡数据来缓解内存压力
del oib
gc.collect()

# 在合并后的模式公司主样本上对订单不平衡指标进行1%缩尾
def winsorize_1pct(series):
    valid = pd.to_numeric(series, errors="coerce")
    lower = valid.quantile(0.01)
    upper = valid.quantile(0.99)
    return valid.clip(lower=lower, upper=upper)


oib_w_cols = []
for col in oib_raw_cols:
    w_col = f"{col}_w"
    main[w_col] = winsorize_1pct(main[col]).astype("float32")
    oib_w_cols.append(w_col)

# 缩尾后删除原变量
main = main.drop(columns=oib_raw_cols)

# 合并完成后再转为分类变量，减少固定效应和聚类变量内存
main["StkcdRecall"] = main["StkcdRecall"].astype("category")
main["yearQuarter"] = main["yearQuarter"].astype("category")


def summary_statistics(series):
    """计算均值、四分位数、样本标准差、最小值、最大值和有效样本量。"""
    x = pd.to_numeric(series, errors="coerce").dropna()
    q = x.quantile([0.25, 0.50, 0.75])

    return {
        "均值": x.mean(),
        "第25百分位": q.loc[0.25],
        "中位数": q.loc[0.50],
        "第75百分位": q.loc[0.75],
        "标准差": x.std(ddof=1),
        "最小值": x.min(),
        "最大值": x.max(),
        "样本量": int(x.shape[0]),
    }


rhs_cols = (
    ["quarterLag1size90plus"]
    + own_ea_cols
    + disclosure_cols
    + media_cols
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


def build_oib_regression_info(
    result,
    outcome,
    cue_col,
    firm_cluster_count,
    date_cluster_count,
    singleton_removed,
):
    """整理表3或附表A.2中单个订单失衡模型的汇总结果。"""
    beta = float(result.params[cue_col])
    standard_error = float(result.std_errors[cue_col])
    t_value = float(result.tstats[cue_col])
    p_value = float(result.pvalues[cue_col])

    return {
        "因变量": outcome,
        "线索系数": beta,
        "聚类标准误": standard_error,
        "单位": "百分点",
        "t值": t_value,
        "p值": p_value,
        "显著性": significance_stars(p_value),
        "控制变量": "是",
        "日期固定效应": "否",
        "被提示公司×年度季度固定效应": "是",
        "回归样本量": int(result.nobs),
        "被提示公司聚类数": int(firm_cluster_count),
        "日期聚类数": int(date_cluster_count),
        "删除单例组观测数": int(singleton_removed),
        "R方": float(result.rsquared),
    }


def build_oib_coefficient_rows(
    result,
    x_cols,
    cue_col,
    outcome,
):
    """整理订单失衡模型中线索变量和全部控制变量的估计结果。"""
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
            "因变量": outcome,
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
            "单位": "百分点",
            "t值": t_value,
            "p值": p_value,
            "显著性": stars,
            "控制变量": "是",
            "日期固定效应": "否",
            "被提示公司×年度季度固定效应": "是",
            "回归样本量": int(result.nobs),
            "R方": float(result.rsquared),
        })

    return rows


def run_oib_regression(data, outcome):
    """
    吸收被提示公司×年度季度固定效应，
    标准误按被提示公司和日期双向聚类。
    """
    # data:合并订单失衡指标后的模式公司主面板
    # outcome:本次回归使用的订单不平衡因变量名称

    cue_col = "quarterLag1size90plus"

    needed = (
        [outcome, "StkcdRecall", "Trddt", "yearQuarter"]
        + rhs_cols
    )
    
    # 只保留回归所需变量，并删除其中任意变量缺失的观测
    reg = data[needed].dropna().copy()

    # 构造被提示公司×年度季度固定效应编号
    fe_codes = (
        reg.groupby(
            ["StkcdRecall", "yearQuarter"],
            observed=True,# observed=True（只处理实际出现的类别组合）：避免为不存在的公司季度组合建立分组；
            sort=False,
        )
        .ngroup() # ngroup()（生成分组编号）：将每个固定效应组转换为整数。
        .to_numpy(dtype=np.int64)
    )

    # 删除固定效应单例组：“某家公司在某个年度季度内只有一条有效回归数据”，
    # 即使不主动删除，固定效应处理后它对系数估计通常也没有作用；
    # 单例组通常不改变系数，但可能改变样本量、自由度修正和最终报告的聚类标准误
    # 同时控制个体（企业、行业、地级市、省份）固定效应和时间固定效应
    fe_counts = np.bincount(fe_codes)
    keep = fe_counts[fe_codes] > 1
    singleton_removed = int((~keep).sum())

    # 因此，固定效应单例组实际只删除：
    # 1日订单失衡回归：60条观测，占非缺失样本的 \(0.00052642\%\)；
    # 10日订单失衡回归：103条观测，占非缺失样本的 \(0.00095776\%\)。



    reg = reg.loc[keep].copy()
    fe_codes = fe_codes[keep]

    # 声明需要吸收的固定效应：被提示公司×年度季度
    absorb = pd.DataFrame(
        {"firm_yearquarter": pd.Categorical(fe_codes)},
        index=reg.index,
    )

    # 构造双向聚类编号
    clusters = pd.DataFrame(
        {
            "firm": pd.factorize(reg["StkcdRecall"], sort=False)[0],
            "date": pd.factorize(reg["Trddt"], sort=False)[0],
        },
        index=reg.index,
    )

# 建立回归模型，包括被解释变量、解释变量、吸收变量（公司✖️年度季度固定效应）
    model = AbsorbingLS(
        dependent=reg[outcome].astype("float64"),
        exog=reg[rhs_cols].astype("float64"),
        absorb=absorb,
        drop_absorbed=True,
    )

# 回归系数照常计算，但计算标准误时，要考虑同一公司和同一天的观测可能彼此相关。
    result = model.fit(
        cov_type="clustered",
        clusters=clusters,
        debiased=True, # debiased=True（有限样本自由度修正）：对聚类标准误作自由度调整。
    )

# 提取模型汇总和全部解释变量的估计结果
    regression_info = build_oib_regression_info(
        result=result,
        outcome=outcome,
        cue_col=cue_col,
        firm_cluster_count=reg["StkcdRecall"].nunique(),
        date_cluster_count=reg["Trddt"].nunique(),
        singleton_removed=singleton_removed,
    )

    coefficient_rows = build_oib_coefficient_rows(
        result=result,
        x_cols=rhs_cols,
        cue_col=cue_col,
        outcome=outcome,
    )

    # 对被解释变量（OIB）进行描述性统计
    outcome_stats = summary_statistics(reg[outcome])

    # 清除不需要的表单，缓解内存压力
    del result, model, absorb, clusters, reg
    gc.collect()

    return outcome_stats, regression_info, coefficient_rows


# ------------------------------------------------------------------
# 生成 表1 描述性统计
# ------------------------------------------------------------------

description_rows = {}

# 当前收益率是小数，乘100后以百分比展示，与原文单位更接近
for col in return_cols:
    description_rows[f"{col}_百分比"] = summary_statistics(main[col] * 100)

description_rows["quarterLag1size90plus"] = summary_statistics(
    main["quarterLag1size90plus"]
)

cue_sample = main["quarterLag1size90plus"].eq(1)

description_rows["nrCuesLag1S90p"] = summary_statistics(
    main.loc[cue_sample, "nrCuesLag1S90p"]
)

# 三套标准化未预期盈余的等权和市值加权版本全部统计
for col in sue_cols:
    description_rows[col] = summary_statistics(main.loc[cue_sample, col])

# 订单失衡必须按各自回归实际样本统计
regression_rows = []

for outcome in oib_w_cols:
    stats, regression_info, _ = run_oib_regression(main, outcome)
    description_rows[outcome] = stats
    regression_rows.append(regression_info)

description_table = pd.DataFrame.from_dict(
    description_rows,
    orient="index",
)

regression_check = pd.DataFrame(regression_rows)


def export_table1_image(table, output_path):
    """把描述性统计结果输出为原文风格的三线表图片。"""
    rows = [
        ("chAdjreturn_t_tp1_Recall_w_百分比", "特征调整收益率（%），[t, t+1]"),
        ("chAdjreturn_t_tp10_Recall_w_百分比", "特征调整收益率（%），[t, t+10]"),
        ("OIBvol_avg_t_tp1_w", "小单成交量订单失衡（%），[t, t+1]"),
        ("OIBvol_avg_t_tp10_w", "小单成交量订单失衡（%），[t, t+10]"),
        ("quarterLag1size90plus", "线索事件（虚拟变量）"),
        ("nrCuesLag1S90p", "线索数量"),
        ("sue1CueLag1S90pEW_w", "盈余惊喜1（等权）"),
        ("sue1CueLag1S90pVW_w", "盈余惊喜1（市值加权）"),
        ("sue2CueLag1S90pEW_w", "盈余惊喜2（等权）"),
        ("sue2CueLag1S90pVW_w", "盈余惊喜2（市值加权）"),
        ("sue3CueLag1S90pEW_w", "盈余惊喜3（等权）"),
        ("sue3CueLag1S90pVW_w", "盈余惊喜3（市值加权）"),
    ]

    stat_cols = [
        "均值", "第25百分位", "中位数", "第75百分位",
        "标准差", "最小值", "最大值", "样本量",
    ]
    headers = [
        "变量", "均值", "p25", "p50", "p75",
        "标准差", "最小值", "最大值", "样本量",
    ]

    cell_text = []
    for variable, label in rows:
        values = [label]
        for stat in stat_cols:
            value = table.loc[variable, stat]
            if pd.isna(value):
                values.append("-")
            elif stat == "样本量":
                values.append(f"{int(value):,}")
            else:
                values.append(f"{value:.4f}")
        cell_text.append(values)

    fig, ax = plt.subplots(figsize=(16, 8.8))
    ax.axis("off")
    ax.text(0.015, 0.965, "表 1", fontsize=18,
            fontweight="bold", fontproperties=table_font)
    ax.text(0.015, 0.915, "描述性统计", fontsize=18,
            fontweight="bold", fontproperties=table_font)

    image_table = ax.table(
        cellText=cell_text, colLabels=headers,
        cellLoc="right", colLoc="right",
        colWidths=[0.33, 0.078, 0.068, 0.068, 0.068,
                   0.078, 0.078, 0.078, 0.10],
        bbox=[0.015, 0.18, 0.97, 0.67],
    )
    image_table.auto_set_font_size(False)
    image_table.set_fontsize(11.5)

    last_row = len(cell_text)
    for (row, col), cell in image_table.get_celld().items():
        cell.set_facecolor("white")
        cell.get_text().set_fontproperties(table_font)
        cell.get_text().set_ha("left" if col == 0 else "right")
        cell.visible_edges = "BT" if row == 0 else ("B" if row == last_row else "")
        cell.set_linewidth(1.0 if row in {0, last_row} else 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


export_table1_image(description_table, TABLE1_IMAGE_PATH)
print(f"表1图片已输出：{TABLE1_IMAGE_PATH}")


print("\n订单失衡回归检查：")
print(regression_check.to_string(index=False))


#===============================================================================
# Table 2：收益回归主结果——模式公司样本
#===============================================================================

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


# 运行模式公司样本
(
    table2_pattern_results,
    table2_pattern_coefficients,
) = run_table2_sample(
    data=main,
    sample_name="pattern",
    sample_label="A. 模式公司主样本",
)

# 表1和模式公司样本回归已经完成，释放内存后再读取完整样本
del main
gc.collect()

# 全样本只读取表2回归实际需要的变量
table2_input_cols = list(dict.fromkeys(
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

table2_numeric_cols = [
    col for col in table2_input_cols
    if col not in {"StkcdRecall", "Trddt", "yearQuarter"}
]

full_dtype = {
    "StkcdRecall": "string",
    "yearQuarter": "string",
}
full_dtype.update({col: "float32" for col in table2_numeric_cols})

print("读取完整样本最终回归面板……")
full = pd.read_csv(
    FULL_PATH,
    usecols=table2_input_cols,
    dtype=full_dtype,
    parse_dates=["Trddt"],
)

full["StkcdRecall"] = full["StkcdRecall"].str.strip().str.zfill(6)

if START_YEAR is not None:
    full = full.loc[full["year"] >= START_YEAR].copy()
if END_YEAR is not None:
    full = full.loc[full["year"] <= END_YEAR].copy()

if full.duplicated(["StkcdRecall", "Trddt"]).any():
    raise ValueError("完整样本面板的 StkcdRecall + Trddt 不是唯一键。")

print(f"完整样本行数：{len(full):,}")

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


# =============================================================================
# Table 3：散户订单失衡结果
# =============================================================================

# -----------------------------------------------------------------
# 1. 读取表3回归需要的规律公告公司样本
# -----------------------------------------------------------------

table3_panel_cols = list(dict.fromkeys(
    [
        "StkcdRecall",
        "Trddt",
        "year",
        "yearQuarter",
    ]
    + rhs_cols
))

table3_numeric_cols = [
    col
    for col in table3_panel_cols
    if col not in {
        "StkcdRecall",
        "Trddt",
        "yearQuarter",
    }
]

table3_dtype = {
    "StkcdRecall": "string",
    "yearQuarter": "string",
}

table3_dtype.update({
    col: "float32"
    for col in table3_numeric_cols
}) # update() 是 Python 字典的方法，用来把一组“键—值”添加到已有字典中。

print("读取表3所需规律公告公司样本……")

table3_data = pd.read_csv(
    PATTERN_PATH,
    usecols=table3_panel_cols,
    dtype=table3_dtype,
    parse_dates=["Trddt"],
)

table3_data["StkcdRecall"] = (
    table3_data["StkcdRecall"]
    .str.strip()
    .str.zfill(6)
)

if START_YEAR is not None:
    table3_data = table3_data.loc[
        table3_data["year"] >= START_YEAR
    ].copy()

if END_YEAR is not None:
    table3_data = table3_data.loc[
        table3_data["year"] <= END_YEAR
    ].copy()

if table3_data.duplicated(
    ["StkcdRecall", "Trddt"]
).any():
    raise ValueError(
        "表3主样本的公司和日期不是唯一键。"
    )

# -----------------------------------------------------------------
# 2. 读取订单失衡数据
# -----------------------------------------------------------------

table3_oib_raw_cols = [
    # 表3：成交量口径
    "OIBvol_avg_t_tp1",
    "OIBvol_avg_t_tp10",

    # 附表A.2：小单买卖次数口径
    "OIBnum_avg_t_tp1",
    "OIBnum_avg_t_tp10",
]

table3_oib_dtype = {
    "Stkcd": "string",
}

table3_oib_dtype.update({
    col: "float32"
    for col in table3_oib_raw_cols
})

print("读取表3订单失衡数据……")

table3_oib = pd.read_csv(
    OIB_PATH,
    usecols=[
        "Stkcd",
        "Trddt",
    ] + table3_oib_raw_cols,
    dtype=table3_oib_dtype,
    parse_dates=["Trddt"],
)

table3_oib["Stkcd"] = (
    table3_oib["Stkcd"]
    .str.strip()
    .str.zfill(6)
)

table3_oib = table3_oib.rename(
    columns={
        "Stkcd": "StkcdRecall",
    }
)

if table3_oib.duplicated(
    ["StkcdRecall", "Trddt"]
).any():
    raise ValueError(
        "订单失衡表的公司和日期不是唯一键。"
    )

# -----------------------------------------------------------------
# 3. 合并规律公告公司样本与订单失衡数据
# -----------------------------------------------------------------

table3_data = table3_data.merge(
    table3_oib,
    on=["StkcdRecall", "Trddt"],
    how="left",
    validate="one_to_one",
)

del table3_oib
gc.collect()

# -----------------------------------------------------------------
# 4. 对订单失衡变量进行全样本1%缩尾
# -----------------------------------------------------------------

table3_oib_w_cols = []

for col in table3_oib_raw_cols:
    w_col = f"{col}_w"

    table3_data[w_col] = (
        winsorize_1pct(table3_data[col])
        .astype("float32")
    )

    table3_oib_w_cols.append(w_col)

# 缩尾完成后不再保留原始列
table3_data = table3_data.drop(
    columns=table3_oib_raw_cols
)

table3_data["StkcdRecall"] = (
    table3_data["StkcdRecall"]
    .astype("category")
)

table3_data["yearQuarter"] = (
    table3_data["yearQuarter"]
    .astype("category")
)

# -----------------------------------------------------------------
# 5. 表3：成交量口径订单失衡
# -----------------------------------------------------------------

table3_volume_outcomes = [
    (
        "OIBvol_avg_t_tp1_w",
        "[t, t+1]",
    ),
    (
        "OIBvol_avg_t_tp10_w",
        "[t, t+10]",
    ),
]

table3_rows = []
table3_coefficient_rows = []

for outcome, window_label in table3_volume_outcomes:
    _, regression_info, coefficient_rows = run_oib_regression(
        table3_data,
        outcome,
    )

    model_metadata = {
        "结果表": "表3",
        "衡量口径": "成交量",
        "订单失衡窗口": window_label,
        "散户识别方法": "A股小单口径",
    }

    table3_rows.append({
        **model_metadata,
        **regression_info,
    })

    table3_coefficient_rows.extend(
        {
            **model_metadata,
            **row,
        }
        for row in coefficient_rows
    )

table3_results = pd.DataFrame(table3_rows)
table3_coefficients = pd.DataFrame(
    table3_coefficient_rows
)

table3_results.to_csv(
    TABLE3_RESULTS_PATH,
    index=False,
    encoding="utf-8-sig",
)

table3_coefficients.to_csv(
    TABLE3_COEFFICIENTS_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("\n表3：散户订单失衡结果")
print(table3_results.to_string(index=False))
print(f"\n表3结果已保存：{TABLE3_RESULTS_PATH}")
print(
    f"表3全部系数明细已保存："
    f"{TABLE3_COEFFICIENTS_PATH}"
)

# 不覆盖本节基准结果的日期固定效应版本由
# “表3_散户订单失衡_加入日期固定效应.py”单独估计并保存。

# -----------------------------------------------------------------
# 6. 附表A.2：小单买卖次数订单失衡（基准规格）
# -----------------------------------------------------------------
# 本节吸收被提示公司×年度季度固定效应，按被提示公司和交易日期双向聚类。
# 三种订单失衡口径并加入交易日期固定效应的本项目扩展由
# “附表A2-A4_三种订单失衡指标.py”单独估计并保存。

appendix_a2_outcomes = [
    (
        "OIBnum_avg_t_tp1_w",
        "[t, t+1]",
    ),
    (
        "OIBnum_avg_t_tp10_w",
        "[t, t+10]",
    ),
]

appendix_a2_rows = []
appendix_a2_coefficient_rows = []

for outcome, window_label in appendix_a2_outcomes:
    _, regression_info, coefficient_rows = run_oib_regression(
        table3_data,
        outcome,
    )

    model_metadata = {
        "结果表": "附表A.2",
        "订单失衡指标": "OIBnum",
        "衡量口径": "买卖次数",
        "分析口径": "基准规格",
        "订单失衡窗口": window_label,
        "散户识别方法": "A股小单口径",
    }

    appendix_a2_rows.append({
        **model_metadata,
        **regression_info,
    })

    appendix_a2_coefficient_rows.extend(
        {
            **model_metadata,
            **row,
        }
        for row in coefficient_rows
    )

appendix_a2_results = pd.DataFrame(
    appendix_a2_rows
)

appendix_a2_coefficients = pd.DataFrame(
    appendix_a2_coefficient_rows
)

appendix_a2_results.to_csv(
    APPENDIX_A2_RESULTS_PATH,
    index=False,
    encoding="utf-8-sig",
)

appendix_a2_coefficients.to_csv(
    APPENDIX_A2_COEFFICIENTS_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("\n附表A.2：小单买卖次数订单失衡结果（基准规格）")
print(appendix_a2_results.to_string(index=False))
print(
    f"\n附表A.2结果已保存："
    f"{APPENDIX_A2_RESULTS_PATH}"
)
print(
    f"附表A.2全部系数明细已保存："
    f"{APPENDIX_A2_COEFFICIENTS_PATH}"
)

del table3_data
gc.collect()



# %% 表4独立运行准备区
 
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


# =============================================================================
# 路径
# =============================================================================

ROOT = Path(__file__).resolve().parents[4]

PATTERN_PATH = (
    ROOT
    / "中间结果/04_合并数据/"
    / "Pattern样本_最终回归面板.csv"
)

TABLE4_RESULTS_PATH = (
    ROOT
    / "中间结果/08_回归结果/"
    / "表4_规模异质性结果.csv"
)

TABLE4_COEFFICIENTS_PATH = (
    ROOT
    / "中间结果/08_回归结果/"
    / "表4_规模异质性系数明细.csv"
)

TABLE4_RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 样本期设置
# =============================================================================

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
    TABLE4_RESULTS_PATH,
    index=False,
    encoding="utf-8-sig",
)

table4_coefficients.to_csv(
    TABLE4_COEFFICIENTS_PATH,
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
    f"{TABLE4_RESULTS_PATH}"
)

print(
    f"表4全部系数明细已保存："
    f"{TABLE4_COEFFICIENTS_PATH}"
)
8
del table4_data
gc.collect()

# %%
