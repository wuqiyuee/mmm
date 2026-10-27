"""逐年估计累计特征调整收益的提示系数，并保存年度系数图。

每年分别吸收被提示公司×年度季度、交易日期固定效应，
采用被提示公司和交易日期双向聚类标准误，递归删除固定效应单例。
源收益以小数存储，展示系数、标准误和置信区间时乘100，单位为百分点。
本脚本打印回归结果，并将年度系数图保存为图片。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from linearmodels.panel.utility import AbsorbingEffectError


# 1. 集中设置路径、窗口、年份和显著性水平
root = Path(__file__).resolve().parents[4]
image_dir = root / "图表结果/02_扩展分析/提示系数时间变化"
input_path = root / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"

horizon = 4
start_year = 2005
end_year = 2025
significance_level = 0.10

confidence_level = 1 - significance_level
dependent_variable = f"chAdjreturn_t_tp{horizon}_Recall_w"
cue = "quarterLag1size90plus"
window_label = f"[t,t+{horizon}]"

# 2. 保留原回归的27个控制变量
own_ea_cols = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)
disclosure_cols = ["ownOtherEventER", "ownPerformanceForecastBrief"]
media_cols = ["aesIncrease", "aesDecrease", "logIncreaseAEV", "logDecreaseAEV"]
control_cols = own_ea_cols + disclosure_cols + media_cols


def main():
    if not isinstance(horizon, int) or not 1 <= horizon <= 50:
        raise ValueError("窗口终点必须为1—50的整数；所选窗口还必须在主面板中存在。")
    if start_year > end_year:
        raise ValueError("起始年份不能晚于结束年份。")
    if not 0 < significance_level < 1:
        raise ValueError("显著性水平必须介于0和1之间。")

    years = list(range(start_year, end_year + 1))
    numeric_cols = [dependent_variable, cue] + control_cols
    needed_cols = ["StkcdRecall", "Trddt"] + numeric_cols
    keys = ["StkcdRecall", "Trddt"]

    # 3. 检查字段，并分块读取年度回归所需样本
    header_cols = pd.read_csv(
        input_path, nrows=0, encoding="utf-8-sig"
    ).columns
    missing_cols = [col for col in needed_cols if col not in header_cols]
    if missing_cols:
        raise ValueError(f"主面板缺少以下字段：{missing_cols}")

    print(f"回归年份：{start_year}—{end_year}年")
    print(f"累计特征调整收益窗口：{window_label}")
    print(f"显著性水平：{significance_level:.0%}")
    print(f"置信水平：{confidence_level:.0%}")
    print("控制变量数量：", len(control_cols))

    read_types = {col: "float64" for col in numeric_cols}
    read_types["StkcdRecall"] = "string"
    year_parts = {year: [] for year in years}

    reader = pd.read_csv(
        input_path,
        usecols=needed_cols,
        dtype=read_types,
        chunksize=300_000,
        encoding="utf-8-sig",
    )

    for block_number, chunk in enumerate(reader, start=1):
        print(f"正在读取第 {block_number} 个数据块……", flush=True)
        dates = pd.to_datetime(chunk["Trddt"], errors="coerce")
        if dates.isna().any():
            raise ValueError("主面板交易日期存在缺失或无法解析的值。")
        chunk["Trddt"] = dates
        chunk = chunk.loc[dates.dt.year.isin(years)].copy()
        if chunk.empty:
            continue

        chunk["StkcdRecall"] = (
            chunk["StkcdRecall"].str.strip()
            .str.replace(r"\.0+$", "", regex=True)
            .str.zfill(6)
        )
        invalid_codes = (
            chunk["StkcdRecall"].notna()
            & ~chunk["StkcdRecall"].str.fullmatch(r"\d{6}", na=False)
        )
        if invalid_codes.any():
            raise ValueError("主面板被提示公司代码不能规范为6位数字。")
        if not chunk[cue].dropna().isin([0, 1]).all():
            raise ValueError("提示变量存在0、1以外的取值。")

        valid = (
            chunk["StkcdRecall"].notna()
            & np.isfinite(chunk[numeric_cols].to_numpy()).all(axis=1)
        )
        chunk = chunk.loc[valid, needed_cols].copy()
        for year, part in chunk.groupby(chunk["Trddt"].dt.year, sort=False):
            year_parts[int(year)].append(part.copy())

    # 4. 逐年整理固定效应、递归删除单例并回归
    annual_rows = []
    for year in years:
        print(f"\n========== 正在处理{year}年 ==========")
        # 每个年份先保留完整结果字段，失败年份也有明确的缺失结果。
        row = {
            "年份": year,
            "观测数": 0,
            "公司数": 0,
            "交易日期数": 0,
            "提示观测数": 0,
            "提示日期数": 0,
            "删除单例观测数": 0,
            "保留控制变量数": 0,
            "系数_百分点": np.nan,
            "标准误_百分点": np.nan,
            "置信下限_百分点": np.nan,
            "置信上限_百分点": np.nan,
            "显著性概率": np.nan,
            "达到显著性阈值": np.nan,
            "状态": "没有有效样本",
        }
        annual_rows.append(row)
        parts = year_parts.pop(year)
        if not parts:
            print("没有有效样本。")
            continue

        reg = pd.concat(parts, ignore_index=True)
        del parts
        try:
            duplicate_count = int(reg.duplicated(keys).sum())
            if duplicate_count:
                raise ValueError(f"发现{duplicate_count}条重复公司日期记录。")

            quarter = reg["Trddt"].dt.to_period("Q")
            reg["_fe_firm_quarter"] = reg.groupby(
                ["StkcdRecall", quarter], sort=False, observed=True
            ).ngroup()
            reg["_fe_date"] = pd.factorize(reg["Trddt"], sort=False)[0]

            singleton_removed = 0
            while not reg.empty:
                singleton = np.zeros(len(reg), dtype=bool)
                for col in ["_fe_firm_quarter", "_fe_date"]:
                    codes = reg[col].to_numpy(dtype="int64")
                    counts = np.bincount(codes)
                    singleton |= counts[codes] == 1
                if not singleton.any():
                    break
                singleton_removed += int(singleton.sum())
                reg = reg.loc[~singleton].copy()

            row["删除单例观测数"] = singleton_removed
            if reg.empty:
                raise ValueError("删除固定效应单例组后，没有剩余样本。")
            reg = reg.reset_index(drop=True)
            row.update({
                "观测数": len(reg),
                "公司数": reg["StkcdRecall"].nunique(),
                "交易日期数": reg["Trddt"].nunique(),
                "提示观测数": int(reg[cue].eq(1).sum()),
                "提示日期数": reg.loc[reg[cue].eq(1), "Trddt"].nunique(),
            })
            if reg[cue].nunique() != 2:
                raise ValueError("年度样本没有同时保留提示和未提示观测。")

            absorb = pd.DataFrame({
                "公司季度": pd.Categorical(reg["_fe_firm_quarter"]),
                "日期": pd.Categorical(reg["_fe_date"]),
            }, index=reg.index)
            clusters = pd.DataFrame({
                "公司": pd.factorize(reg["StkcdRecall"], sort=False)[0],
                "日期": reg["_fe_date"].to_numpy(),
            }, index=reg.index)

            # 恒定控制变量本来就被固定效应吸收，先排除以免与常数项重复。
            constant_controls = [
                col for col in control_cols if reg[col].nunique() <= 1
            ]
            active_controls = [
                col for col in control_cols if col not in constant_controls
            ]
            # 保持原顺序：控制变量在前，提示变量在后。
            X = reg[active_controls + [cue]].astype("float64").copy()
            X.insert(0, "const", 1.0)

            print("观测数：", len(reg))
            print("公司数：", row["公司数"])
            print("交易日期数：", row["交易日期数"])
            print("删除单例组观测数：", singleton_removed)
            print("正在估计年度回归……", flush=True)

            model = AbsorbingLS(
                dependent=reg[dependent_variable],
                exog=X,
                absorb=absorb,
                drop_absorbed=True,
            )
            result = model.fit(
                cov_type="clustered",
                clusters=clusters,
                debiased=True,
                use_cache=False,
            )
            if cue not in result.params.index:
                raise ValueError("提示变量被固定效应或控制变量吸收，不能单独估计。")

            cue_variance = float(result.cov.loc[cue, cue])
            if not np.isfinite(cue_variance) or cue_variance <= 0:
                raise ValueError("提示系数的双向聚类方差非正或无效。")

            # 5. 收益以小数存储，展示时换算成百分点
            coefficient = float(result.params[cue]) * 100
            standard_error = float(result.std_errors[cue]) * 100
            ci_low, ci_high = (
                result.conf_int(level=confidence_level).loc[cue] * 100
            ).tolist()
            pvalue = float(result.pvalues[cue])
            if not np.isfinite([
                coefficient, standard_error, ci_low, ci_high, pvalue
            ]).all():
                raise ValueError("年度回归结果包含缺失或无穷值。")

            omitted_controls = [
                col for col in control_cols if col not in result.params.index
            ]
            row.update({
                "观测数": int(result.nobs),
                "保留控制变量数": len(control_cols) - len(omitted_controls),
                "系数_百分点": coefficient,
                "标准误_百分点": standard_error,
                "置信下限_百分点": ci_low,
                "置信上限_百分点": ci_high,
                "显著性概率": pvalue,
                "达到显著性阈值": pvalue < significance_level,
                "状态": "成功",
            })
            print("提示系数（百分点）：", coefficient)
            print("双向聚类标准误（百分点）：", standard_error)
            print(
                f"{confidence_level:.0%}置信区间（百分点）：",
                [ci_low, ci_high],
            )
            print("显著性概率：", pvalue)
            print(f"是否达到{significance_level:.0%}显著性阈值：", pvalue < significance_level)
            print("因恒定、固定效应或共线而省略的控制变量：", omitted_controls)
        except (ValueError, np.linalg.LinAlgError, AbsorbingEffectError) as error:
            row["状态"] = f"未成功：{error}"
            print(f"{year}年未成功：{error}")

    # 6. 汇总年度结果；所有年份均失败时给出提示
    annual_results = pd.DataFrame(annual_rows).sort_values("年份").reset_index(drop=True)
    print(f"\n{start_year}—{end_year}年年度回归结果：")
    print(annual_results.to_string(index=False))
    success = annual_results["状态"].eq("成功")
    print("\n成功估计的年份数：", int(success.sum()))
    plot_data = annual_results.loc[success].copy()
    if plot_data.empty:
        print("没有成功估计的年度结果，无法绘制系数图。")
        return annual_results

    # 7. 保存年度系数图，文件名体现样本区间、回归类型、窗口和置信水平
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    error_lower = plot_data["系数_百分点"] - plot_data["置信下限_百分点"]
    error_upper = plot_data["置信上限_百分点"] - plot_data["系数_百分点"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(
        plot_data["年份"],
        plot_data["系数_百分点"],
        yerr=np.vstack([error_lower, error_upper]),
        fmt="o",
        color="#2166AC",
        ecolor="#8FAFCB",
        markersize=5,
        elinewidth=1.2,
        capsize=3,
        label=f"年度系数及{confidence_level:.0%}双向聚类置信区间",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(years)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlabel("年份")
    ax.set_ylabel("提示系数（百分点）")
    ax.set_title(f"{window_label}累计特征调整收益：年度提示系数变化")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    image_dir.mkdir(parents=True, exist_ok=True)
    figure_path = image_dir / (
        "年度_主回归_"
        f"窗口{window_label}_{confidence_level:.0%}置信区间.png"
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("图片已保存：", figure_path)
    return annual_results


if __name__ == "__main__":
    annual_results = main()
