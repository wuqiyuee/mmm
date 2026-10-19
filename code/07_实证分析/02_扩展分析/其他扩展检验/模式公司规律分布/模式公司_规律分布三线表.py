#!/usr/bin/env python3
"""统计模式公司规律分布，并绘制高清三线表。

本脚本：
1. 一行样本代表“某家公司的一期财报”，不是一家独立公司。
2. 正式标识表只保留是否为模式公司，没有保留具体命中了哪类规律。
   因此，要画十类规律分布，需要从原始公告数据重新提取主规律。
3. 本脚本调用现有识别脚本的函数，避免另写一套规律识别算法。
   只导入函数，不运行原脚本的主程序，也不改写正式标识表。
4. 重新识别的结果与正式标识表一致后，才绘图。
5. 图中的样本数、公司数、十类数量和占比都自动计算；没有写死旧图数字。

对应识别脚本：
code/02_构建变量/主回归/2识别pattern_firms/build_pattern_firm_qtr.py

输入：沿用上述识别脚本的原始公告、公司市场类型、交易日历文件；
      另读取中间结果中的正式模式公司季度标识表进行核对。
输出：图表结果/02_扩展分析/其他扩展检验/模式公司规律分布/
      pattern_firms_规律分布_纯学术三线表_高清.png

运行时会覆盖同名图片；不会输出额外的数据表，也不会修改原始数据。
需要的第三方库：pandas（数据处理库）、numpy（数值计算库，识别函数使用）、
matplotlib（绘图库）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

# 使用只保存图片的绘图方式，运行时不另外弹出绘图窗口。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import pandas as pd


# =====================================================================
# 一、平时需要调整的设置，都集中放在这里
# =====================================================================

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


PROJECT_ROOT = find_project_root()
SOURCE_SCRIPT = (
    PROJECT_ROOT
    / "code/02_构建变量/主回归/2识别pattern_firms/build_pattern_firm_qtr.py"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "图表结果/02_扩展分析/其他扩展检验/模式公司规律分布"
    / "pattern_firms_规律分布_纯学术三线表_高清.png"
)

# 下面的识别参数与当前上游脚本默认值一致。
# 如果以后修改上游识别口径，这里也要同步修改。
START_DATE = "1990-01-01"       # 公告日和财报截止日的起点。
ACCPER_END_DATE = "2025-12-31"  # 财报截止日上限。
ANN_END_DATE = "2026-06-30"     # 公告日上限，允许2025年年报在2026年披露。
MIN_CONSECUTIVE = 3            # 同类报告连续3期，即连续3年的同类财报。
SHARE_SCOPE = "all_a"          # 沪深A股，包括主板、创业板、科创板，不含北交所。

FIGURE_SIZE = (15, 7.8)        # 图片宽、高，单位为英寸；采用第4版较紧凑的比例。
DPI = 300                     # 每英寸像素数；可改为600，图片也会相应变大。
TITLE = "表 1  模式公司财报披露主模式分布"
SONG_FONT_PATH = Path.home() / "Library/Fonts/宋体.ttc"  # 本机已安装的常规宋体。


# =====================================================================
# 二、十类规律的中文名称和示意表达
# =====================================================================

# 左侧英文代码是上游已经使用的变量名，不能随意改动。
# 右侧依次是“中文名称、典型表达”。典型表达只是解释规律的示意，
# 这些是虚构示例，不代表真实样本，也不声称是样本中出现最多的具体日期。
# k（序数）、n（周序数）在不同公司、不同报告类型中可以取不同的值。
PATTERN_LABELS = {
    "same_calendar_month_and_date":
        ("固定公历月日", "10月31日、4月30日"),
    "same_kth_weekday_same_month":
        ("同月第 k 个星期几", "4月第4个周二"),
    "same_weekday_same_nth_week_since_accper":
        ("截止日后第 n 周的同一星期几", "截止日后第4周周六"),
    "same_kth_trading_day_same_month":
        ("同月第 k 个交易日", "10月第15个交易日"),
    "same_kth_trading_day_since_accper":
        ("截止日后第 k 个交易日", "截止日后第15个交易日"),
    "same_kth_nonweekend_day_same_month":
        ("同月第 k 个非周末日", "4月第20个非周末日"),
    "same_kth_weekday_since_accper":
        ("截止日后第 k 个相同星期几", "截止日后第9个周六"),
    "same_kth_nonweekend_day_since_accper":
        ("截止日后第 k 个非周末日", "截止日后第43个非周末日"),
    "same_weekday_same_nth_week_same_month":
        ("同月第 n 周的同一星期几", "10月第5周周二"),
    "same_kth_day_since_accper":
        ("截止日后第 k 个自然日", "截止日后第90天"),
}


def load_pattern_module():
    """导入现有识别脚本，取得其中的函数和默认输入路径。"""
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(f"找不到规律识别脚本：{SOURCE_SCRIPT}")

    # 路径中有中文且目录名以数字开头，所以按文件路径导入。
    # 导入后的模块名称不等于主程序名称，因此不会执行原脚本的主程序。
    spec = importlib.util.spec_from_file_location("pattern_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入规律识别脚本：{SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =====================================================================
# 三、读取原始数据，恢复“公司 × 财报期”的主规律明细
# =====================================================================

def rebuild_pattern_details(source) -> pd.DataFrame:
    """沿用上游的清洗和筛选口径，保留绘图需要的主规律类型。"""
    # source（已导入的识别模块）提供原始文件路径和规律识别函数。
    ann = pd.read_csv(source.DEFAULT_ANN_FILE, low_memory=False)
    cal = pd.read_csv(source.DEFAULT_CAL_FILE, low_memory=False)
    company = pd.read_csv(
        source.DEFAULT_COMPANY_FILE,
        encoding="utf-8-sig",
        usecols=["Stkcd", "Markettype"],
        low_memory=False,
    )

    # 1. 整理交易日历。State（开休市状态）为 O（开市）时才是交易日。
    if "State" not in cal.columns and "Statse" in cal.columns:
        cal = cal.rename(columns={"Statse": "State"})
    source.require_columns(cal, {"Markettype", "Clddt", "State"}, "交易日历")
    cal["Markettype"] = pd.to_numeric(cal["Markettype"], errors="coerce")
    cal["State"] = cal["State"].astype("string").str.strip().str.upper()

    # 2. 整理公告表。
    # Stkcd6（六位公司代码）；Accper（财报截止日）；Actudt（公告日期）。
    source.require_columns(
        ann, {"Stkcd", "Stknme", "Reptyp", "Sctcd", "Accper", "Annodt"}, "公告表"
    )
    ann["Stkcd6"] = ann["Stkcd"].map(source.normalize_stock_code)
    ann["Accper"] = pd.to_datetime(ann["Accper"], errors="coerce")
    ann["Actudt"] = pd.to_datetime(ann["Annodt"], errors="coerce")
    ann = ann.dropna(subset=["Accper", "Actudt", "Stkcd6", "Sctcd"]).copy()
    # 这里识别的是公告日期规律，与上游一样直接使用公告日期，不额外后移。
    # 是否能参与“第几个交易日”规律的判断，由上游的交易日历函数处理。

    # 3. 按公司市场类型筛选沪深A股，不依靠证券代码前缀猜测市场类型。
    company["Stkcd6"] = company["Stkcd"].map(source.normalize_stock_code)
    company["Markettype"] = pd.to_numeric(company["Markettype"], errors="coerce")
    company = company.dropna(subset=["Stkcd6", "Markettype"])
    if company["Stkcd6"].duplicated().any():
        raise ValueError("公司文件中的证券代码不唯一，不能安全合并市场类型。")
    ann = ann.merge(
        company[["Stkcd6", "Markettype"]],
        on="Stkcd6", how="left", validate="m:1",
    )
    if ann["Markettype"].isna().any():
        raise ValueError("部分公告未匹配到公司市场类型，请检查输入文件。")
    ann = ann.loc[
        ann["Markettype"].isin(source.MARKETTYPE_BY_SCOPE[SHARE_SCOPE])
    ].copy()
    ann = ann.drop(columns="Markettype")

    # 4. 识别报告类型，分别限制公告日和财报截止日。
    # report_type（报告类型）区分一季报、中报、三季报、年报。
    ann["report_type"] = source.infer_report_type(ann["Reptyp"])
    ann = ann.dropna(subset=["report_type"])
    ann = ann.loc[
        ann["Actudt"].between(pd.Timestamp(START_DATE), pd.Timestamp(ANN_END_DATE))
        & ann["Accper"].between(pd.Timestamp(START_DATE), pd.Timestamp(ACCPER_END_DATE))
    ].copy()

    # 5. 去掉“公告日等于财报截止日”的未披露占位记录。
    # 同一公司同一财报期有多条记录时，保留最早公告的一条。
    ann = ann.loc[ann["Actudt"].ne(ann["Accper"])].copy()
    ann = (
        ann.sort_values(["Stkcd6", "Accper", "Actudt"])
        .drop_duplicates(["Stkcd6", "Accper"], keep="first")
        .reset_index(drop=True)
    )

    # 6. 调用现有的四个函数，避免复制和改写十类规律的计算公式。
    calendar = source.build_a_share_calendar(cal)
    details = source.add_calendar_features(ann, calendar)
    details = source.compute_consecutive_flags(
        details, source.PATTERN_PRIORITY, MIN_CONSECUTIVE
    )
    details = source.select_primary_pattern(details, source.PATTERN_PRIORITY)

    # 连续3期指同类报告连续3年，第3期及以后满足规律才标记为1。
    # PatternFirmQtr（模式公司季度标识）：1=模式；0=历史足够但非模式；
    # 缺失=同类报告的连续历史不足3期，不能直接当作非模式。
    details["PatternFirmQtr"] = details["is_pattern_observation"].astype("Int8")
    details.loc[~details["history_eligible"], "PatternFirmQtr"] = pd.NA
    return details


# =====================================================================
# 四、核对正式标识表：一致才继续画图
# =====================================================================

def check_against_official_table(details: pd.DataFrame, source) -> None:
    """同时核对公司—财报期集合，以及每条观测的1、0或缺失标记。"""
    official_path = source.DEFAULT_OUT_DIR / source.DEFAULT_OUT_FILE
    official = pd.read_csv(
        official_path,
        encoding="utf-8-sig",
        usecols=["Stkcd", "Accper", "PatternFirmQtr"],
        dtype={"Stkcd": "string", "PatternFirmQtr": "Int8"},
    )
    official["Stkcd"] = official["Stkcd"].map(source.normalize_stock_code)
    official["Accper"] = pd.to_datetime(official["Accper"], errors="raise")
    valid_flags = official["PatternFirmQtr"].dropna().isin([0, 1]).all()
    # 代码标准化会把缺失变为空字符串，因此不能只检查是否为缺失值。
    valid_codes = official["Stkcd"].str.fullmatch(r"\d{6}", na=False).all()
    if not valid_flags or not valid_codes or official["Accper"].isna().any():
        raise ValueError("正式标识表中存在无效标记或缺失的公司—财报期键。")

    rebuilt = details[["Stkcd6", "Accper", "PatternFirmQtr"]].rename(
        columns={"Stkcd6": "Stkcd"}
    )
    if not rebuilt["Stkcd"].str.fullmatch(r"\d{6}", na=False).all():
        raise ValueError("重新识别的明细中存在无效公司代码，应为六位数字。")
    check = official.merge(
        rebuilt,
        on=["Stkcd", "Accper"], how="outer", validate="1:1",
        suffixes=("_正式", "_重算"), indicator=True,
    )
    # -1仅在核对时临时代表缺失，防止把“缺失”和“0”混为一谈。
    different = (
        check["_merge"].ne("both")
        | check["PatternFirmQtr_正式"].fillna(-1).ne(
            check["PatternFirmQtr_重算"].fillna(-1)
        )
    )
    if different.any():
        print("不一致记录示例（最多显示5条）：")
        print(check.loc[different].head(5).to_string(index=False))
        raise ValueError(
            f"重新识别与正式标识表有 {int(different.sum()):,} 条不一致记录。"
            "请先核对输入文件、日期及连续期数，确认正式标识表是否需要更新；"
            "本次停止绘图，不覆盖已有图片。"
        )
    print(f"核对通过：{len(official):,} 条公司—财报期观测的标记完全一致。")


# =====================================================================
# 五、统计十类主模式的数量和占比
# =====================================================================

def summarize_patterns(details: pd.DataFrame, source):
    """每条模式观测只归入一个主规律，十类数量之和等于模式观测总数。"""
    sample = details.loc[details["PatternFirmQtr"].eq(1).fillna(False)].copy()
    if sample.empty:
        raise ValueError("当前口径下没有模式公司观测，无法计算占比。")

    # selected_pattern_type（选中的主规律类型）来自现有识别函数。
    # 多个规律同时合格时，优先选整段长度较长者；并列按上游预设顺序。
    # 保持上游的“整段长度”口径，不改为“截至当期的连续长度”。
    counts = sample["selected_pattern_type"].value_counts()
    counts = counts.reindex(source.PATTERN_PRIORITY, fill_value=0)
    if int(counts.sum()) != len(sample):
        raise ValueError("部分模式观测没有归入已知的十类主规律，请检查识别结果。")

    summary = counts.rename_axis("规律代码").reset_index(name="观测数")
    summary["主模式类型"] = summary["规律代码"].map(
        lambda name: PATTERN_LABELS[name][0]
    )
    summary["典型表达"] = summary["规律代码"].map(
        lambda name: PATTERN_LABELS[name][1]
    )
    summary["占比"] = summary["观测数"] / len(sample) * 100
    # 从多到少排列；数量相同时，保持上游预设优先顺序。
    summary = summary.sort_values("观测数", ascending=False, kind="stable")
    return sample, summary.reset_index(drop=True)


# =====================================================================
# 六、设置字体，绘制三线表
# =====================================================================

def choose_font(candidates: list[str], purpose: str) -> str:
    """按顺序寻找本机已有字体，找不到时明确报错，避免输出方框字。"""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for index, name in enumerate(candidates):
        if name in installed:
            if index > 0:
                print(f"字体提示：{purpose}未找到首选字体，实际使用 {name}。")
            return name
    raise RuntimeError(f"没有找到可用的{purpose}字体，请在字体候选列表中补充本机字体。")


def draw_three_line_table(sample: pd.DataFrame, summary: pd.DataFrame) -> None:
    """采用第4版的紧凑布局和常规字重，白底，仅保留三条横线。"""
    # 本机的旧字体缓存没有列入后来安装的宋体，却记录了极粗的华文宋体。
    # 先注册现有的常规宋体文件，让绘图库能找到它；不另存或修改字体文件。
    if SONG_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(SONG_FONT_PATH))

    chinese_font = choose_font(
        ["SimSun", "Songti SC", "STSong", "Noto Serif CJK SC", "PingFang SC"],
        "中文（宋体优先）",
    )
    english_font = choose_font(
        ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "英文与数字（新罗马字体优先）",
    )
    # 字体列表中，新罗马字体负责数字和英文，中文字体负责汉字。
    body_font = FontProperties(family=[english_font, chinese_font], weight="normal")

    total = len(sample)
    firms = sample["Stkcd6"].nunique()
    first_date = sample["Accper"].min().strftime("%Y-%m-%d")
    last_date = sample["Accper"].max().strftime("%Y-%m-%d")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 所有位置都用0到1表示，方便直接对照图片理解。
    # x是左右位置，y是上下位置；左下角是(0, 0)，右上角是(1, 1)。
    # 标题、说明和图注共用左边界，参照第4版的排版方式。
    left_margin, right_margin = 0.04, 0.96
    ax.text(left_margin, 0.952, TITLE, ha="left", va="center",
            fontsize=21, fontproperties=body_font)
    ax.text(
        left_margin, 0.919,
        "统计口径：公司—财报期主模式频数与占比；固定效应不适用（非回归统计表）。",
        ha="left", va="center", fontsize=11.5, fontproperties=body_font,
    )
    ax.text(
        left_margin, 0.884,
        f"样本：模式公司季度标识 = 1 的公司—财报期观测；"
        f"观测数 = {total:,}；涉及公司数 = {firms:,}。",
        ha="left", va="center", fontsize=13, fontproperties=body_font,
    )
    ax.text(
        left_margin, 0.850,
        f"识别条件：同一公司、同一报告类型连续{MIN_CONSECUTIVE}期符合规律；"
        f"图中模式样本的财报截止日：{first_date} 至 {last_date}。",
        ha="left", va="center", fontsize=11.5, fontproperties=body_font,
    )

    # 先划分五列。两列文字使用统一的左起点，序号和数字使用列中心。
    # 六个边界依次围出：序号、主模式类型、典型表达、观测数、占比。
    column_edges = [left_margin, 0.100, 0.480, 0.755, 0.870, right_margin]
    column_centers = [
        (left + right) / 2
        for left, right in zip(column_edges[:-1], column_edges[1:])
    ]
    text_padding = 0.01  # 文字与本列左边界留一点间距。
    column_positions = [
        column_centers[0],
        column_edges[1] + text_padding,
        column_edges[2] + text_padding,
        column_centers[3],
        column_centers[4],
    ]
    # center（居中）、left（左对齐）：表头和正文共用这组位置与对齐方式。
    column_alignments = ["center", "left", "left", "center", "center"]

    # 三线表的三条线：表顶线、表头下方的线、表底线。
    # 降低画布高度后，各行更紧凑；不加灰底或行间分隔线。
    top_y, header_bottom_y, bottom_y = 0.820, 0.755, 0.125
    ax.hlines([top_y, header_bottom_y, bottom_y],
              xmin=column_edges[0], xmax=column_edges[-1],
              colors="black", linewidths=[1.3, 0.7, 1.3])

    header_y = (top_y + header_bottom_y) / 2
    headers = ["序号", "主模式类型", "典型表达（示意）", "观测数", "占比（%）"]
    for x, text, alignment in zip(column_positions, headers, column_alignments):
        ax.text(x, header_y, text, ha=alignment, va="center",
                fontsize=15, fontproperties=body_font)

    row_height = (header_bottom_y - bottom_y) / len(summary)
    for row_number, row in summary.iterrows():
        y = header_bottom_y - (row_number + 0.5) * row_height
        cells = [
            str(row_number + 1),
            row["主模式类型"],
            row["典型表达"],
            f"{int(row['观测数']):,}",
            f"{row['占比']:.2f}",
        ]
        for x, text, alignment in zip(column_positions, cells, column_alignments):
            ax.text(x, y, text, ha=alignment, va="center",
                    fontsize=15, fontproperties=body_font)

    notes = [
        "注：每条模式观测只归入一个主模式；多类规律同时合格时，先选连续段较长者，并列按识别脚本预设优先级选择。",
        f"占比以{total:,}条模式观测为分母，四舍五入后合计可能不等于100%；典型表达仅为示意。“非周末日”包括周一至周五的节假日。",
    ]
    for y, note in zip([0.075, 0.040], notes):
        ax.text(left_margin, y, note, ha="left", va="center",
                fontsize=10.5, fontproperties=body_font)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"图片已保存：{OUTPUT_PATH}")


# =====================================================================
# 七、主程序：按顺序完成上面的步骤
# =====================================================================

def main() -> None:
    source = load_pattern_module()
    print("第一步：读取原始数据，重新提取主规律。")
    details = rebuild_pattern_details(source)

    print("第二步：与正式模式公司季度标识表核对。")
    check_against_official_table(details, source)

    print("第三步：统计十类主模式的数量与占比。")
    sample, summary = summarize_patterns(details, source)
    print(f"模式观测数：{len(sample):,}；涉及公司数：{sample['Stkcd6'].nunique():,}")
    print(summary[["主模式类型", "观测数", "占比"]].to_string(
        index=False, formatters={"占比": lambda value: f"{value:.2f}%"}
    ))

    print("第四步：绘制并保存三线表。")
    draw_three_line_table(sample, summary)


if __name__ == "__main__":
    main()
