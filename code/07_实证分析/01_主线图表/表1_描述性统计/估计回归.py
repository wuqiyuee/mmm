"""本脚本由统一任务入口调用，虽然文件名沿用“估计回归.py”，
实际负责描述性统计、订单失衡回归核对和表格绘图。

输入为模式公司最终回归面板，以及个股订单失衡源表；两表按被提示公司
与交易日一对一左合并，主面板一行代表一家被提示公司在一个交易日的观测。
研究期、字段类型、缺失处理、1%与99%分位缩尾均沿用来源脚本。
收益直接使用已缩尾字段，乘100后展示；提示次数与盈余惊喜仅在提示日统计。
订单失衡在合并后的主样本上缩尾，再按各自完整回归样本及单例剔除口径统计。

输出集中到项目根目录下“图表结果／01_主线图表／表1_描述性统计”，
保留原图片文件名；控制台继续打印订单失衡回归核对结果。
只在调用主函数或直接执行时读取面板、估计和绘图；导入本模块不读取面板或写文件。
"""

from __future__ import annotations

from pathlib import Path
import gc
import sys

import pandas as pd


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
SHARED_DIR = PROJECT_ROOT / "code" / "07_实证分析" / "90_公共模块"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

# 缩尾、统计与实际回归样本统一复用从原综合回归提取的订单失衡计算。
from 订单失衡估计 import winsorize_1pct, summary_statistics, run_oib_regression

PATTERN_PATH = PROJECT_ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"
OIB_PATH = PROJECT_ROOT / "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv"
OUTPUT_DIR = PROJECT_ROOT / "图表结果" / "01_主线图表" / "表1_描述性统计"
TABLE1_IMAGE_PATH = OUTPUT_DIR / "table1_描述性统计.png"
START_YEAR = 2005
END_YEAR = 2025


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


rhs_cols = (
    ["quarterLag1size90plus"]
    + own_ea_cols
    + disclosure_cols
    + media_cols
)


def export_table1_image(table, output_path):
    """把描述性统计结果输出为原文风格的三线表图片。"""
    # 绘图依赖与字体配置在调用时初始化，导入模块时不建立绘图缓存。
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"
    font_manager.fontManager.addfont(font_path)
    table_font = font_manager.FontProperties(fname=font_path)
    plt.rcParams["axes.unicode_minus"] = False

    rows = [
        ("chAdjreturn_t_tp1_Recall_w_百分比", "特征调整收益率（%），[t, t+1]"),
        ("chAdjreturn_t_tp10_Recall_w_百分比", "特征调整收益率（%），[t, t+10]"),
        ("OIBvol_avg_t_tp1_w", "小单成交量订单失衡（%），[t, t+1]"),
        ("OIBvol_avg_t_tp10_w", "小单成交量订单失衡（%），[t, t+10]"),
        ("quarterLag1size90plus", "线索事件（cue）"),
        ("nrCuesLag1S90p", "线索数量"),
        ("sue1CueLag1S90pEW_w", "sue1（等权）"),
        ("sue1CueLag1S90pVW_w", "sue1（市值加权）"),
        ("sue2CueLag1S90pEW_w", "sue2（等权）"),
        ("sue2CueLag1S90pVW_w", "sue2（市值加权）"),
        ("sue3CueLag1S90pEW_w", "sue3（等权）"),
        ("sue3CueLag1S90pVW_w", "sue3（市值加权）"),
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
    ax.text(
        0.015, 0.875,
        f"样本：模式公司，{START_YEAR}—{END_YEAR}年；本表为描述性统计，固定效应不适用。",
        fontsize=10.5, color="#444444", fontproperties=table_font,
    )

    image_table = ax.table(
        cellText=cell_text, colLabels=headers,
        cellLoc="right", colLoc="right",
        colWidths=[0.20, 0.078, 0.068, 0.068, 0.068,
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

    ax.text(
        0.015, 0.125,
        "注：线索数量和盈余惊喜仅在提示日统计；收益与线索事件按各自非缺失观测统计。",
        fontsize=10.0, color="#444444", fontproperties=table_font,
    )
    ax.text(
        0.015, 0.085,
        "订单失衡使用对应回归的有效样本，并剔除公司季度单例观测。",
        fontsize=10.0, color="#444444", fontproperties=table_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> list[Path]:
    """独立执行表1原有口径的统计与绘图，返回本次输出的图片路径。"""
    # 读取与左合并后仍是“被提示公司×交易日”；缺失订单失衡不会删去主面板行。
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

    # 表内每行是一项变量；订单失衡两行各自使用对应回归的有效样本。
    description_rows = {}

    # 当前收益率是小数，乘100后以百分比展示
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
        stats, regression_info, _ = run_oib_regression(main, outcome, rhs_cols)
        description_rows[outcome] = stats
        regression_rows.append(regression_info)

    description_table = pd.DataFrame.from_dict(
        description_rows,
        orient="index",
    )

    regression_check = pd.DataFrame(regression_rows)

    export_table1_image(description_table, TABLE1_IMAGE_PATH)
    print(f"表1图片已输出：{TABLE1_IMAGE_PATH}")


    print("\n订单失衡回归检查：")
    print(regression_check.to_string(index=False))

    return [TABLE1_IMAGE_PATH]


if __name__ == "__main__":
    main()
