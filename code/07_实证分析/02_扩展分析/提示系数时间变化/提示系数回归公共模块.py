"""时间变化脚本共用的当前主回归样本及年度估计入口。

复用正式主口径的季度数据、历史沪深300提示及分段新闻控制。
收益使用既有缩尾值；订单失衡先在范围内的主口径公司日全样本缩尾，
再筛选共同有效资格和本次因变量，不填补缺失订单失衡。
导入时不读取数据、不创建目录、不估计、不写结果。
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[4]
COMMON_DIR = PROJECT_ROOT / "code/07_实证分析/90_公共模块"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from 主回归口径 import (
    ALLOW_FILLED_STARTS, CONTROL_COLUMNS, CUE_COLUMN, END_DATE, EXCLUDED_YEARS,
    SPEC_DESCRIPTION, SPEC_ID, START_DATE, iter_main_blocks, main_input_paths,
)


def print_progress(stage, completed, total, detail=""):
    """按处理数量每约20%报告一次，并报告开始、结束；比例不代表耗时。"""
    if total <= 0:
        return
    if completed not in (0, total):
        if completed * 5 // total == (completed - 1) * 5 // total:
            return
    message = f"  {stage}：{completed}/{total}（{completed / total:.0%}）"
    if detail:
        message += f"；{detail}"
    print(message, flush=True)


def load_main_sample(horizon, start_date, end_date, outcome_type="收益"):
    """读取当前主口径；窗口编号1对应包含当天的两日窗口。"""
    if not isinstance(horizon, int) or not 1 <= horizon <= 50:
        raise ValueError("窗口终点必须为1—50的整数。")
    start = max(pd.Timestamp(start_date), pd.Timestamp(START_DATE))
    end = min(pd.Timestamp(end_date), pd.Timestamp(END_DATE))
    if start > end:
        raise ValueError("所选日期范围与当前主回归日期范围没有交集。")
    if outcome_type not in {"收益", "交易机制"}:
        raise ValueError("因变量类型只能为收益或交易机制。")

    print(f"读取{outcome_type}样本：{start.date()}至{end.date()}", flush=True)
    pieces = []
    dependent_variable = f"chAdjreturn_t_tp{horizon}_Recall_w"
    total_blocks = len(main_input_paths()[0])
    print_progress("读取季度样本", 0, total_blocks)
    for number, (block, returns, controls, cue, common) in enumerate(
        iter_main_blocks(), start=1
    ):
        dates = pd.to_datetime(block["date"], unit="D", origin="unix")
        keep = (dates >= start) & (dates <= end)
        if not ALLOW_FILLED_STARTS:
            keep &= block["filled"] == 0
        if EXCLUDED_YEARS:
            keep &= ~dates.year.isin(EXCLUDED_YEARS)
        # 收益可直接保留共同有效行；订单失衡须先保留缩尾前的公司日总体。
        if outcome_type == "收益":
            keep &= common & np.isfinite(returns[:, horizon - 1])
        positions = np.flatnonzero(keep)
        if not len(positions):
            print_progress("读取季度样本", number, total_blocks)
            continue
        frame = pd.DataFrame(controls[positions], columns=CONTROL_COLUMNS)
        frame["StkcdRecall"] = block["firm"][positions].astype("int64")
        frame["Trddt"] = dates[positions]
        frame["Filling"] = block["filled"][positions]
        frame[CUE_COLUMN] = cue[positions]
        if outcome_type == "收益":
            frame[dependent_variable] = returns[positions, horizon - 1]
        else:
            frame["_main_valid"] = common[positions]
        pieces.append(frame)
        print_progress("读取季度样本", number, total_blocks)

    if not pieces:
        raise ValueError("当前主回归口径在所选日期范围内没有有效观测。")
    data = pd.concat(pieces, ignore_index=True)
    del pieces
    keys = ["StkcdRecall", "Trddt"]
    if data.duplicated(keys).any():
        raise ValueError("主口径季度数据存在重复的被提示公司日期键。")

    if outcome_type == "交易机制":
        # 先合并与统一缩尾；此时未因控制缺失、逐年或固定效应单例删样本。
        print("正在合并订单失衡数据并统一缩尾……", flush=True)
        raw_outcome = f"OIBvol_avg_t_tp{horizon}"
        dependent_variable = f"{raw_outcome}_w"
        oib_path = PROJECT_ROOT / (
            "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv"
        )
        oib = pd.read_csv(
            oib_path, usecols=["Stkcd", "Trddt", raw_outcome],
            dtype={"Stkcd": "string", raw_outcome: "float64"},
            encoding="utf-8-sig",
        )
        oib["Trddt"] = pd.to_datetime(oib["Trddt"], errors="raise")
        oib = oib.loc[oib["Trddt"].between(start, end)].copy()
        codes = (
            oib["Stkcd"].str.strip().str.replace(r"\.0+$", "", regex=True)
        )
        if not codes.str.fullmatch(r"\d{1,6}", na=False).all():
            raise ValueError("订单失衡源表存在缺失或无效公司代码。")
        oib["StkcdRecall"] = codes.astype("int64")
        if oib[keys].isna().any().any() or oib.duplicated(keys).any():
            raise ValueError("订单失衡源表存在缺失或重复公司日期键。")
        data = data.merge(
            oib[keys + [raw_outcome]], on=keys, how="left", validate="one_to_one"
        )
        del oib
        raw_values = data[raw_outcome].replace([np.inf, -np.inf], np.nan)
        if not raw_values.notna().any():
            raise ValueError("当前主样本合并后没有有效订单失衡观测。")
        lower, upper = raw_values.quantile([0.01, 0.99]).tolist()
        data[dependent_variable] = raw_values.clip(lower, upper)
        print(
            f"订单失衡缩尾完成：样本{len(data):,}行；"
            f"1%、99%界限为{lower:.6g}、{upper:.6g}", flush=True,
        )
        valid = data["_main_valid"] & np.isfinite(data[dependent_variable])
        data = data.loc[valid].drop(columns=["_main_valid", raw_outcome]).copy()

    if data.empty:
        raise ValueError("共同资格及本次因变量筛选后没有有效回归观测。")
    data["yearQuarter"] = data["Trddt"].dt.to_period("Q").astype("category")
    print(
        f"有效主样本{len(data):,}行，其中合格填充日起点"
        f"{int(data['Filling'].sum()):,}行", flush=True,
    )
    return data.reset_index(drop=True)


def estimate_main_specification(
    data, dependent_variable, window_label, significance_level, scale=100
):
    """分年调用主规格：双固定效应、公司日期双聚类及正式主回归推断。"""
    if not 0 < significance_level < 1:
        raise ValueError("显著性水平必须介于0和1之间。")
    # 只载入正式估计器的函数定义，不调用其主流程或写结果。
    estimator_path = PROJECT_ROOT / "code/07_实证分析/01_主线图表/表2_收益回归/估计回归.py"
    spec = importlib.util.spec_from_file_location("_time_variation_table2", estimator_path)
    estimator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(estimator)

    numeric_cols = [dependent_variable, CUE_COLUMN] + CONTROL_COLUMNS
    needed = ["StkcdRecall", "Trddt", "yearQuarter", "Filling"] + numeric_cols
    reg = data[needed].dropna().copy()
    reg = reg.loc[np.isfinite(reg[numeric_cols].to_numpy()).all(axis=1)].copy()
    if reg.empty or reg.duplicated(["StkcdRecall", "Trddt"]).any():
        raise ValueError("年度回归样本为空，或存在重复被提示公司日期键。")
    reg["_fe_firm_quarter"] = reg.groupby(
        ["StkcdRecall", "yearQuarter"], observed=True, sort=False
    ).ngroup()
    reg["_fe_date"] = pd.factorize(reg["Trddt"], sort=False)[0]
    fe_cols = ["_fe_firm_quarter", "_fe_date"]
    reg, singleton_removed = estimator.drop_fe_singletons_iteratively(reg, fe_cols)
    if reg.empty:
        raise ValueError("递归删除固定效应单例后没有年度回归观测。")
    reg = reg.reset_index(drop=True)
    if reg[CUE_COLUMN].nunique() != 2:
        raise ValueError("年度样本没有同时保留提示与未提示观测。")

    # 年度内恒定的控制项已被固定效应吸收，避免与显式常数重复。
    active_controls = [col for col in CONTROL_COLUMNS if reg[col].nunique() > 1]
    design = reg[[CUE_COLUMN] + active_controls].astype("float64").copy()
    design.insert(0, "const", 1.0)
    absorb = pd.DataFrame({col: pd.Categorical(reg[col]) for col in fe_cols})
    clusters = pd.DataFrame({
        "firm": pd.factorize(reg["StkcdRecall"], sort=False)[0],
        "date": pd.factorize(reg["Trddt"], sort=False)[0],
    })
    model = AbsorbingLS(
        dependent=reg[dependent_variable].astype("float64"), exog=design,
        absorb=absorb, drop_absorbed=True,
    )
    result = model.fit(
        cov_type="unadjusted", debiased=False, method="hdfe",
        absorb_options={"options": {"tol": 1e-12}}, use_cache=False,
    )
    if CUE_COLUMN not in result.params.index:
        raise ValueError("提示变量被固定效应或控制变量吸收，无法单独估计。")
    inference, degrees, fe_rank = estimator.figure2_consistent_inference(
        model, result, clusters, reg, fe_cols,
    )
    coefficient = float(result.params[CUE_COLUMN]) * scale
    standard_error = float(inference.loc[CUE_COLUMN, "标准误"]) * scale
    pvalue = float(inference.loc[CUE_COLUMN, "p值"])
    half_width = float(stats.t.ppf(1 - significance_level / 2, degrees)) * standard_error
    if not np.isfinite([coefficient, standard_error, pvalue, half_width]).all():
        raise ValueError("年度回归系数或推断结果包含缺失、无穷值。")
    omitted = [col for col in CONTROL_COLUMNS if col not in result.params.index]
    return {
        "口径标识": SPEC_ID,
        "提示变量": CUE_COLUMN,
        "观测数": int(result.nobs),
        "公司数": int(reg["StkcdRecall"].nunique()),
        "交易日期数": int(reg["Trddt"].nunique()),
        "提示观测数": int(reg[CUE_COLUMN].eq(1).sum()),
        "提示日期数": int(reg.loc[reg[CUE_COLUMN].eq(1), "Trddt"].nunique()),
        "填充起始日数": int(reg["Filling"].sum()),
        "删除单例观测数": singleton_removed,
        "保留控制变量数": len(CONTROL_COLUMNS) - len(omitted),
        "省略控制变量": omitted,
        "残差自由度": degrees,
        "固定效应实际秩": fe_rank,
        "系数_百分点": coefficient,
        "标准误_百分点": standard_error,
        "置信下限_百分点": coefficient - half_width,
        "置信上限_百分点": coefficient + half_width,
        "显著性概率": pvalue,
        "达到显著性阈值": pvalue < significance_level,
        "状态": "成功",
    }
