"""表3与附表A.2的订单失衡纯计算函数。

来源：code/07_实证分析/99_历史与待核对/原综合脚本/回归.py（原回归脚本）第222—466行。
保留原来的1%和99%分位缩尾、逐模型完整样本、公司×年度季度单例删除、
吸收固定效应以及公司与实际交易日期双向聚类和有限样本修正。
本模块不定位数据、不读取面板、不创建目录、不写出结果。
仅将原函数依赖的控制变量列表改为显式参数，便于两个任务独立调用。
"""

import gc

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS  # 吸收固定效应回归器。

def winsorize_1pct(series):
    valid = pd.to_numeric(series, errors="coerce")
    lower = valid.quantile(0.01)
    upper = valid.quantile(0.99)
    return valid.clip(lower=lower, upper=upper)



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


def run_oib_regression(data, outcome, rhs_cols):
    """
    吸收被提示公司×年度季度固定效应，
    标准误按被提示公司和日期双向聚类。

    控制变量和线索变量列表由任务显式传入，不依赖其他表先运行后留下的全局变量。
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
    fe_counts = np.bincount(fe_codes)
    keep = fe_counts[fe_codes] > 1
    singleton_removed = int((~keep).sum())

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


