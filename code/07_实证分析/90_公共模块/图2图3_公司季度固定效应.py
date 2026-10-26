"""图2与图3公司季度固定效应的公共计算入口。

迁移自原图表目录的公司季度合并绘图脚本，保留迁移时工作区已有修改。
逐窗口读取进度、估计、缩尾、样本期、双向聚类、有限样本修正和三套
置信区间绘图均沿用来源脚本。主函数的图号必须显式指定为2或3，
只计算并绘制所选图，不会触发另一图的估计；导入时不读取研究面板或写文件。
图片、数值结果和逐窗口进度集中到相应主线图表任务目录。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from tqdm.auto import tqdm
import gc
import hashlib
import json
from importlib.metadata import version


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
pattern_sample_PATH = ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"
oib_path = ROOT / "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv"
# 仍可在代码中设置仅重画；单图入口也提供显式的仅重画参数。
PLOT_ONLY = False
START_YEAR = 2005
END_YEAR = 2025


RESULT_FIELDS = [
    "window", "coef", "se", "ci_low", "ci_high", "nobs"
]


def load_progress(path, source_paths, curve_name):
    """核对计算版本，并恢复已经保存的窗口。"""
    input_info = []

    for source in source_paths:
        source = Path(source)
        file_stat = source.stat()
        input_info.append({
            "path": str(source.resolve()),
            "size": file_stat.st_size,
            "mtime_ns": file_stat.st_mtime_ns,
        })

    # 根据脚本内容、输入文件元信息和依赖版本生成校验标记
    config = {
        "curve": curve_name,
        "script_hash": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "inputs": input_info,
        "versions": {
            name: version(name)
            for name in [
                "numpy", "pandas", "scipy", "linearmodels", "pyhdfe"
            ]
        },
    }

    signature = hashlib.sha256(
        json.dumps(
            config, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()

    if not path.exists():
        return [], signature

    saved = pd.read_csv(
        path,
        dtype={"signature": "string"},
        float_precision="round_trip",
    )

    if saved.empty or list(saved.columns) != RESULT_FIELDS + ["signature"]:
        raise ValueError(f"进度表为空或格式不正确，请检查：{path}")

    if not saved["signature"].eq(signature).fillna(False).all():
        raise ValueError(
            f"进度表与当前输入文件、脚本或依赖版本不一致：{path}\n"
            "请更换进度表文件名，开始新一轮计算。"
        )

    values = saved[RESULT_FIELDS].apply(
        pd.to_numeric, errors="raise"
    )
    windows = values["window"]

    valid = (
        np.isfinite(values.to_numpy(dtype=float)).all()
        and windows.between(1, 50).all()
        and windows.mod(1).eq(0).all()
        and not windows.duplicated().any()
        and values["se"].ge(0).all()
        and values["nobs"].gt(0).all()
        and values["nobs"].mod(1).eq(0).all()
        and values["ci_low"].le(values["coef"]).all()
        and values["ci_high"].ge(values["coef"]).all()
    )

    if not valid:
        raise ValueError(f"进度表存在异常结果，请检查：{path}")

    values = values.astype({"window": "int64", "nobs": "int64"})
    values = values.sort_values("window")

    return values.to_dict("records"), signature


def save_progress(rows, path, signature):
    """先写临时文件，写完后再替换正式进度表。"""
    table = pd.DataFrame(rows, columns=RESULT_FIELDS)
    table = table.sort_values("window")
    table["signature"] = signature

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
    )
    temporary_path.replace(path)

def draw_dynamic_figure(table, title, output_path):
    """根据已经保存的回归结果绘图。"""
    # 延迟初始化绘图与字体，导入公共计算模块时不创建绘图文件。
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    font_manager.fontManager.addfont(font_path)
    zh_font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.sans-serif"] = [zh_font]
    plt.rcParams["axes.unicode_minus"] = False

    table = table.sort_values("window")

    windows = table["window"].to_numpy(dtype=float)
    coefficients = table["coef"].to_numpy(dtype=float)
    standard_errors = table["se"].to_numpy(dtype=float)

    interval_specs = [
        # 从最长到最短依次叠画，95%区间保留原回归结果
        (
            coefficients - 2.5758293035489004 * standard_errors,
            coefficients + 2.5758293035489004 * standard_errors,
            "#bdbdbd", 0.8, "99%置信区间", 1,
        ),
        (
            table["ci_low"].to_numpy(dtype=float),
            table["ci_high"].to_numpy(dtype=float),
            "#78909c", 1.6, "95%置信区间", 2,
        ),
        (
            coefficients - 1.6448536269514722 * standard_errors,
            coefficients + 1.6448536269514722 * standard_errors,
            "#1f4e79", 2.8, "90%置信区间", 3,
        ),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for lower, upper, color, width, label, order in interval_specs:
        ax.vlines(
            windows,
            lower,
            upper,
            colors=color,
            linewidth=width,
            label=label,
            zorder=order,
        )

    # 按图中90%置信区间判断10%水平显著性
    significant = (
        np.abs(coefficients)
        > 1.6448536269514722 * standard_errors
    )

    # 显著用红色，其余用黑色
    point_colors = np.where(significant, "#d62728", "#1a1a1a")

    ax.scatter(
        windows,
        coefficients,
        color=point_colors,
        s=16,
        zorder=4,
    )

    ax.axhline(0, color="#333333", linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("事件窗口")
    ax.set_ylabel("提示变量系数（百分点）")

    ticks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"[t,t+{tick}]" for tick in ticks])

    # 添加红点的图例说明
    ax.scatter(
        [],
        [],
        color="#d62728",
        s=16,
        label="红点：10%水平显著",
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=9,
        handlelength=1.4,
    )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"图已保存到：{output_path}")


def load_pattern_panel(include_returns: bool, horizons=None):
    """沿用原样本口径；图2按待估窗口读取收益，图3只读取共同变量。"""
    all_cols = pd.read_csv(pattern_sample_PATH, nrows=0).columns.tolist()

    # 未指定时读取全部窗口；续算时只读取待完成窗口。
    if horizons is None:
        horizons = range(1, 51)

    y_cols = [
        f"chAdjreturn_t_tp{i}_Recall_w"
        for i in horizons
    ]

    # 自己的财报公告日
    own_e_cols = [
        col for col in all_cols
        if col.startswith("ownEA")
    ]

    # 被提示公司自身公告事件控制变量
    disclosure_cols = [
        "ownOtherEventER",              # 自身专题事件
        "ownPerformanceForecastBrief",  # 自身业绩预告或快报
    ]

    # 控制变量 = 自己的财报公告日+重大事件+新闻指标
    control_cols = (
        own_e_cols
        + disclosure_cols
        + [
            "aesIncrease",
            "aesDecrease",
            "logIncreaseAEV",
            "logDecreaseAEV",
        ]
    )

    # 解释变量
    base_cols = [
        "quarterLag1size90plus",
        "StkcdRecall",       
        "Trddt",            
        "yearQuarter",
    ]

    # 只读取需要的变量
    usecols = list(dict.fromkeys((y_cols if include_returns else []) + base_cols + control_cols))

    missing = [col for col in usecols if col not in all_cols]
    if missing:
        raise ValueError(f"CSV 中缺少变量: {missing}")

    print("正在读取模式公司样本数据......")
    df = pd.read_csv(pattern_sample_PATH, usecols=usecols)
    df["Trddt"] = pd.to_datetime(df["Trddt"], errors="raise")
    df = df.loc[
        df["Trddt"].dt.year.between(START_YEAR, END_YEAR)
    ]
    print("模式公司数据读取成功！")

    return df, control_cols


def run_figure2(fig2_result_path, fig2_progress_path, fig2_out_path):
    """先核对进度，再读取待完成窗口所需的数据，并保留原估计口径。"""
    results, fig2_signature = load_progress(
        fig2_progress_path,
        [pattern_sample_PATH],
        "图2：原始小数收益率",
    )

    done2 = {int(row["window"]) for row in results}
    todo2 = [i for i in range(1, 51) if i not in done2]

    print(f"图2已恢复 {len(done2)} 个窗口，待计算 {len(todo2)} 个。")

    # 全部完成时不读取大面板，直接使用已有结果继续整理和绘图。
    if todo2:
        df, control_cols = load_pattern_panel(
            include_returns=True,
            horizons=todo2,
        )

    pbar = tqdm(
        todo2,
        total=len(todo2),
        desc="正在回归窗口",
        unit="个窗口",
        dynamic_ncols=True,
    )

    for i in pbar:
        pbar.set_postfix_str(f"当前窗口 {i}/50")
        y_col = f"chAdjreturn_t_tp{i}_Recall_w" #被解释变量

        # 用于回归的变量
        reg_cols = [y_col, "quarterLag1size90plus", "StkcdRecall", "Trddt", "yearQuarter"] + control_cols
        # 逐列检查缺失，保留当前窗口全部必要变量非缺失的行。
        valid_rows = np.ones(len(df), dtype=bool)
        for col in reg_cols:
            valid_rows &= df[col].notna().to_numpy()

        reg_df = df.loc[valid_rows, reg_cols]
        del valid_rows

        # 固定效应
        # “公司×年度季度”组
        # pd.Categorical(）将前面的“公司×年度季度”声明成“类别”，
        # 然后AbsorbingLS(吸收固定效应最小二乘)：这些是固定效应分组，要把每个组自己的平均差异吸收掉
        reg_df["firm_yearquarter_fe"] = pd.Categorical(
        reg_df["StkcdRecall"].astype(str)
        + "_"
        + reg_df["yearQuarter"].astype(str)
    )

        y = reg_df[y_col]

        X = reg_df[["quarterLag1size90plus"] + control_cols]

        absorb = reg_df[["firm_yearquarter_fe"]]

        # 构造双向聚类标准误
        clusters = pd.DataFrame(
        {
            "firm_cluster": pd.factorize(reg_df["StkcdRecall"])[0],# 把公司代码转成整数编号
            "date_cluster": pd.factorize(reg_df["Trddt"])[0],
        },
        index=reg_df.index,# 让新建的 clusters（聚类变量表）的行索引，和 reg_df（当前回归样本表）的行索引完全一致。
    )
        # 搭建模型，明确被解释变量和解释变量
        model = AbsorbingLS(
            dependent=y,
            exog=X,
            absorb=absorb,# 吸收固定效应
        )
    
        fit = model.fit(
            cov_type="clustered",  # 使用聚类稳健协方差
            clusters=clusters,     # 按公司和日期双向聚类
            debiased=True,         # 开启有限样本修正
        )

        # 系数
        coef = fit.params["quarterLag1size90plus"]
        # 标准误差
        se = fit.std_errors["quarterLag1size90plus"]

        # 提取核心解释变量的95%置信区间
        ci = fit.conf_int(level=0.95).loc["quarterLag1size90plus"]

        # 从每一次回归结果里提取核心变量的系数、标准误、置信区间，并保存到结果表里。
        results.append({
            "window": i,
            "coef": coef,
            "se": se,
            "ci_low": float(ci.iloc[0]),   # 95%置信区间下界
            "ci_high": float(ci.iloc[1]),  # 95%置信区间上界
                   "nobs": fit.nobs,
        })

        # 每完成一个窗口，立即保存进度
        save_progress(results, fig2_progress_path, fig2_signature)

        # 释放本窗口回归临时对象和已经保存结果的收益列。
        del reg_df, y, X, absorb, clusters, model, fit, ci
        del df[y_col]
        gc.collect()

    coef_df = pd.DataFrame(results).sort_values("window").reset_index(drop=True)

    # 把小数收益率转换成百分比收益率
    coef_df[["coef", "se", "ci_low", "ci_high"]] = (
        coef_df[["coef", "se", "ci_low", "ci_high"]] * 100
    )

    # 保存图2回归结果，不额外保存行索引，使用带中文兼容标记的编码
    coef_df.to_csv(
        fig2_result_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"图2回归结果已保存到：{fig2_result_path}")

    print(coef_df.head())
    print(coef_df.tail())

    # ============================================================
    # 画 Figure 2：收益动态和长期反转
    # ============================================================
    draw_dynamic_figure(
        coef_df,
        "收益动态和长期反转（公司×年度季度固定效应）",
        fig2_out_path,
    )

    return [fig2_result_path, fig2_progress_path, fig2_out_path]


def run_figure3(df, control_cols, fig3_result_path, fig3_progress_path, fig3_out_path):
    """只估计图3的50个平均订单失衡窗口，缩尾仍在完整合并主样本上完成。"""
    oib_cols = [
        f"OIBvol_avg_t_tp{i}"
        for i in range(1, 51)
    ]

    pattern_cols = [
        "quarterLag1size90plus",
        "StkcdRecall",
        "Trddt",
        "yearQuarter",
    ] + control_cols

    pattern_df = df[pattern_cols]

    del df
    gc.collect()

    # 读取交易数据
    oib_df = pd.read_csv(
        oib_path,
        usecols=["Stkcd", "Trddt"] + oib_cols,
    )

    # 统一股票代码格式
    pattern_df["StkcdRecall_key"] = (
        pd.to_numeric(pattern_df["StkcdRecall"], errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.zfill(6)
    )

    oib_df["Stkcd_key"] = (
        pd.to_numeric(oib_df["Stkcd"], errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.zfill(6)
    )

    # 统一日期格式
    pattern_df["Trddt_key"] = pd.to_datetime(pattern_df["Trddt"])
    oib_df["Trddt_key"] = pd.to_datetime(oib_df["Trddt"])

    # 合并 OIB：被提示公司 × 日期
    fig3_df = pattern_df.merge(
        oib_df[["Stkcd_key", "Trddt_key"] + oib_cols],
        left_on=["StkcdRecall_key", "Trddt_key"],
        right_on=["Stkcd_key", "Trddt_key"],
        how="left",
        validate="m:1",
    )

    # 合并完成，释放两个中间表
    del pattern_df
    del oib_df
    gc.collect()

    # 对图 3 的订单不平衡变量进行全样本 1% 缩尾
    for col in oib_cols:
        # 将无穷值处理为缺失值
        fig3_df[col] = fig3_df[col].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        lower = fig3_df[col].quantile(0.01)
        upper = fig3_df[col].quantile(0.99)

        # 直接用缩尾值替换原值，避免同时保留两套50列数据。
        fig3_df[col] = fig3_df[col].clip(
            lower=lower,
            upper=upper,
        )

    # 保留原来回归使用的缩尾变量名。
    fig3_df.rename(
        columns={col: f"{col}_w" for col in oib_cols},
        inplace=True,
    )

    # 共同变量不随窗口变化，逐列检查一次缺失，避免每轮生成大块缺失标记。
    common_valid_rows = np.ones(len(fig3_df), dtype=bool)
    for col in pattern_cols:
        common_valid_rows &= fig3_df[col].notna().to_numpy()

    # 沿用原来的公司季度字符串标签，只生成一次。
    fig3_df["firm_yearquarter_fe"] = pd.Categorical(
        fig3_df["StkcdRecall"].astype(str)
        + "_"
        + fig3_df["yearQuarter"].astype(str)
    )

    # 缩尾完成后，再开始保存和运行回归结果
    fig3_results, fig3_signature = load_progress(
        fig3_progress_path,
        [pattern_sample_PATH, oib_path],
        "图3：订单不平衡，百分点",
    )

    done3 = {int(row["window"]) for row in fig3_results}
    todo3 = [i for i in range(1, 51) if i not in done3]

    print(f"图3已恢复 {len(done3)} 个窗口，待计算 {len(todo3)} 个。")

    for i in tqdm(
        todo3,
        total=len(todo3),
        desc="正在回归图3",
        unit="个窗口",
    ):
        y_col = f"OIBvol_avg_t_tp{i}_w"
        reg_cols = [
            y_col,
            "quarterLag1size90plus",
            "StkcdRecall",
            "Trddt",
            "yearQuarter",
        ] + control_cols

        # 与原来对reg_cols删除缺失行相同，保持样本和行顺序。
        valid_rows = common_valid_rows & fig3_df[y_col].notna().to_numpy()
        reg_df = fig3_df.loc[
            valid_rows,
            reg_cols + ["firm_yearquarter_fe"],
        ]
        del valid_rows

        # 只保留本窗口实际出现的类别，与原来逐窗口构造固定效应一致。
        reg_df["firm_yearquarter_fe"] = (
            reg_df["firm_yearquarter_fe"].cat.remove_unused_categories()
        )

        y = reg_df[y_col]
        X = reg_df[["quarterLag1size90plus"] + control_cols]
        absorb = reg_df[["firm_yearquarter_fe"]]

        clusters = pd.DataFrame(
            {
                "firm_cluster": pd.factorize(reg_df["StkcdRecall"])[0],# 允许同一家公司内部相关
                "date_cluster": pd.factorize(reg_df["Trddt"])[0], # 允许同一天不同公司之间相关
            },
            index=reg_df.index,
        )# 双向聚类标准误差，在同一个聚类内部允许序列相关，不同的聚类之间不允许。

        model = AbsorbingLS(
            dependent=y,
            exog=X,
            absorb=absorb,
        )

        fit = model.fit(
            cov_type="clustered",  # 使用聚类稳健协方差
            clusters=clusters,     # 按公司和日期双向聚类
            debiased=True,         # 开启有限样本修正
        )

        coef = fit.params["quarterLag1size90plus"]
        se = fit.std_errors["quarterLag1size90plus"]

        # 提取核心解释变量的95%置信区间
        ci = fit.conf_int(level=0.95).loc["quarterLag1size90plus"]

        fig3_results.append({
            "window": i,
            "coef": coef,
            "se": se,
            "ci_low": float(ci.iloc[0]),   # 95%置信区间下界
            "ci_high": float(ci.iloc[1]),  # 95%置信区间上界
            "nobs": fit.nobs,
        })

        # 每完成一个窗口，立即保存进度
        save_progress(fig3_results, fig3_progress_path, fig3_signature)

        # 结果保存后，释放临时对象和本窗口已经用完的缩尾列。
        del reg_df, y, X, absorb, clusters, model, fit, ci
        del fig3_df[y_col]
        gc.collect()

    fig3_coef_df = pd.DataFrame(fig3_results).sort_values("window").reset_index(drop=True)

    # 保存图3回归结果
    fig3_coef_df.to_csv(
        fig3_result_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"图3回归结果已保存到：{fig3_result_path}")

    draw_dynamic_figure(
        fig3_coef_df,
        "散户订单不平衡动态（公司×年度季度固定效应）",
        fig3_out_path,
    )

    # 不覆盖本图的交易日期固定效应版本由
    # “图3_散户订单不平衡动态_加入日期固定效应.py”单独估计并绘制。

    # 由于现有数据无法进一步区分 BJZZ（替代散户交易识别算法）口径下的全部订单与奇数手订单，
    # 本文不对原文附录图 A.1 和 A.2 进行严格复现。作为替代性稳健性检验，
    # 后续将订单不平衡的计算口径由按成交量计算的 `OIBvol` 替换为按成交笔数计算的 `OIBnum`，
    # 以检验基准结果是否受到少数较大金额交易的影响。该检验仍基于 CSMAR 小单买卖分类，
    # 因此主要考察结果对订单不平衡度量方式的敏感性，而不能用于检验不同散户交易识别算法或奇数手订单口径的影响。

    return [fig3_result_path, fig3_progress_path, fig3_out_path]


def main(figure_number: int, *, plot_only: bool | None = None) -> list[Path]:
    """显式运行选定图，返回本次产物路径；不会运行另一图。"""
    if figure_number not in {2, 3}:
        raise ValueError("图号只能是2或3。")
    task_name = "图2_收益动态" if figure_number == 2 else "图3_散户订单不平衡"
    output_dir = ROOT / "图表结果" / "01_主线图表" / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"图{figure_number}_公司季度固定效应_回归结果.csv"
    # 新闻控制来源已切换，使用新进度文件开始估计；后续同口径运行仍可校验并续算。
    progress_path = output_dir / f"图{figure_number}_公司季度固定效应_数行者分段新闻_2005至2025年_逐窗口进度.csv"
    if figure_number == 2:
        image_path = output_dir / "figure2_收益动态和长期反转.png"
        title = "收益动态和长期反转（公司×年度季度固定效应）"
    else:
        image_path = output_dir / "figure3_散户订单不平衡动态.png"
        title = "散户订单不平衡动态（公司×年度季度固定效应）"
    should_plot_only = PLOT_ONLY if plot_only is None else plot_only
    if should_plot_only:
        # 完整数值结果已是最终单位；重画时不再次乘100，也不读取研究面板。
        saved_results = pd.read_csv(result_path)
        draw_dynamic_figure(saved_results, title, image_path)
        return [image_path]
    if figure_number == 2:
        return run_figure2(
            result_path, progress_path, image_path,
        )
    return run_figure3(
        *load_pattern_panel(include_returns=False),
        result_path, progress_path, image_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=(2, 3), required=True, help="要运行的图号，必须显式指定。")
    parser.add_argument("--plot-only", action="store_true", default=None, help="只从已有完整数值结果重画当前图。")
    arguments = parser.parse_args()
    main(arguments.figure, plot_only=arguments.plot_only)
