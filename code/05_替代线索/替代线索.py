# 本脚本基于模式公司的记忆矩阵，构造非盈余公告类线索：
# 十类专题其他事件；业绩预告、业绩快报；

from pathlib import Path
import numpy as np
import pandas as pd

# 根据脚本位置定位项目根目录，整个项目移动后无需修改路径。
ROOT = Path(__file__).resolve().parents[2]


#===============================================================================
# 路径设置
#===============================================================================

def build_paths(root_dir):
    """根据项目根目录，统一生成输入和输出路径。"""

    # 将传入的根目录转换为路径对象
    root = Path(root_dir)

    paths = {
        # 模式公司记忆矩阵
        "memory": (
            root
            / "中间结果/03_记忆矩阵"
            / "MemoryMatrixPattern(交易变量用公告后首个交易日).csv"
        ),

        # 业绩预告快报：公司 × 交易日控制表
        "forecast_events": (
            root
            / "中间结果/02_变量构造结果/5公司公告事件"
            / "业绩预告快报控制变量_公告日期口径.csv"
        ),

        # 专题其他事件：所有有效公告日期口径
        "other_events": (
            root
            / "中间结果/02_变量构造结果/5公司公告事件"
            / "专题研究事件控制变量_公告日期口径.csv"
        ),

        # 个股日度面板，用于提取公告前第3个交易日的市值
        "daily": (
            root
            / "中间结果/01_清洗后样本"
            / "DGTW构建用个股日度面板.csv"
        ),

        # 月度市值断点
        "size_bp": (
            root
            / "中间结果/02_变量构造结果/8月度市值分组"
            / "月度市值分组.csv"
        ),

        # 最终输出文件
        "output": (
            root
            / "中间结果/05_替代线索"
            / "替代线索_两类公告.csv"
        ),
    }

    return paths


#===============================================================================
# 函数工具
#===============================================================================
def normalize_stock_code(x):
    if pd.isna(x):
        return ""
    return str(x).strip().zfill(6)

def safe_qcut_2(s):
    """按中位数分成低、高两组，相同数值不拆分。"""

    # 1. 先将所有行的分组结果设为缺失
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    # 2. 只有非缺失的拥挤度参与计算
    valid = s.dropna()

    if valid.empty:
        return out

    # 3. 计算当前分组样本的中位数
    cutoff = valid.median()

    # 4. 小于等于中位数为低组1，大于中位数为高组2
    out.loc[valid.index] = np.where(
        valid.le(cutoff),
        1.0,
        2.0,
    )

    return out


# 读取记忆矩阵
def load_memory(memory_path):
    """读取记忆矩阵，规范证券代码，检查主键，并生成季度匹配键。"""

    # 1. 读取文件：证券代码保留为字符串，日期转换为日期类型
    memory = pd.read_csv(
        memory_path,
        dtype={
            "Stkcd": "string",
            "StkcdRecall": "string",
        },
        parse_dates=["Annodt", "event_Trddt"],
        encoding="utf-8-sig",
    )

    # 2. 检查输入是否为空
    if memory.empty:
        raise ValueError("记忆矩阵为空，请检查输入文件。")

    # 3. 规范线索公司、被提示公司的证券代码
    memory["Stkcd"] = memory["Stkcd"].apply(normalize_stock_code)
    memory["StkcdRecall"] = (
        memory["StkcdRecall"].apply(normalize_stock_code)
    )

    # 4. 检查关键字段是否缺失
    key_cols = ["Stkcd", "Annodt", "StkcdRecall"]
    required_cols = key_cols + ["event_Trddt"]

    if (
        memory[required_cols].isna().any().any()
        or memory["Stkcd"].eq("").any()
        or memory["StkcdRecall"].eq("").any()
    ):
        raise ValueError("记忆矩阵的公司代码或关键日期存在缺失。")

    # 5. 检查“线索公司 × 公告自然日 × 被提示公司”是否唯一
    duplicate_count = int(
        memory.duplicated(key_cols, keep=False).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"记忆矩阵存在主键重复，涉及 {duplicate_count} 行。"
        )

    # 6. 生成事件交易日所属的自然季度，供后续匹配替代公告
    memory["yearQuarter"] = (
        memory["event_Trddt"]
        .dt.to_period("Q")
        .astype(str)
    )

    return memory


# 读取两类替代公告事件
def load_alt_events(forecast_path, other_path):
    """读取预告快报、专题其他事件，合并到线索公司日。"""

    # 每项分别指定：输入路径、源标记字段、统一后的标记字段
    specs = [
        (
            forecast_path,
            "ownPerformanceForecastBrief",
            "altForecast",
        ),
        (
            other_path,
            "ownOtherEventER",
            "altOtherEventER",
        ),
    ]

    parts = []

    # 1. 分别读取并检查两张公司日事件表
    for path, source_col, target_col in specs:
        table = pd.read_csv(
            path,
            usecols=["StkcdRecall", "Trddt", source_col],
            dtype={"StkcdRecall": "string"},
            encoding="utf-8-sig",
        )

        # 原表中的公告公司，在本检验中充当线索公司
        table = table.rename(
            columns={
                "StkcdRecall": "Stkcd",
                "Trddt": "date",
                source_col: target_col,
            }
        )

        # 2. 检查代码后再标准化，防止空白字符串被补成六个零
        raw_codes = table["Stkcd"].astype("string").str.strip()

        if raw_codes.isna().any() or raw_codes.eq("").any():
            raise ValueError(f"事件表存在缺失或空白证券代码：{path}")

        table["Stkcd"] = raw_codes.apply(normalize_stock_code)

        # 上游已映射交易日，此处只转换数据类型，不移动日期
        table["date"] = pd.to_datetime(table["date"], errors="raise")

        if table["date"].isna().any():
            raise ValueError(f"事件表存在缺失的交易日期：{path}")

        # 3. 检查源标记：源表缺失不能直接解释为没有事件
        values = pd.to_numeric(table[target_col], errors="raise")

        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"事件标记存在缺失或非0、1取值：{path}")

        table[target_col] = values.astype("int8")

        # 4. 上游公司日表应当唯一，防止合并时膨胀行数
        if table.duplicated(["Stkcd", "date"]).any():
            raise ValueError(f"事件表在公司与交易日层面不唯一：{path}")

        # 每张表只保留确实发生该类事件的公司日
        table = table.loc[
            table[target_col].eq(1),
            ["Stkcd", "date", target_col],
        ].copy()

        parts.append(table)

    # 5. 外连接：保留任意一张表中出现的事件公司日
    event = parts[0].merge(
        parts[1],
        on=["Stkcd", "date"],
        how="outer",
        validate="1:1",
    )

    if event.empty:
        raise ValueError("两张事件表均没有有效替代事件。")

    # 仅将外连接产生的另一类事件缺失，填为0
    flag_cols = ["altForecast", "altOtherEventER"]
    event[flag_cols] = event[flag_cols].fillna(0).astype("int8")

    # 6. 生成与记忆矩阵一致的季度匹配键
    event["yearQuarter"] = (
        event["date"].dt.to_period("Q").astype(str)
    )

    return (
        event[
            [
                "Stkcd",
                "date",
                "yearQuarter",
                "altForecast",
                "altOtherEventER",
            ]
        ]
        .sort_values(["Stkcd", "date"])
        .reset_index(drop=True)
    )


# 匹配替代公告与记忆矩阵，并执行原文的逐行筛选
def match_alt_events_to_memory(event, memory):
    """按提示公司和自然季度匹配，再执行原文的两条删除规则。"""

    # 1. 匹配同一提示公司、同一自然季度的记忆记录
    alt_memory = event.merge(
        memory,
        on=["Stkcd", "yearQuarter"],
        how="inner",
        validate="m:m",
    )

    print("替代公告接入记忆矩阵后行数：", len(alt_memory))

    # 2. 删除替代公告日等于这一行参照财报日的配对
    same_reference_day = alt_memory["date"].eq(
        alt_memory["event_Trddt"]
    )

    print("与本行参照财报同日的配对数：",
          int(same_reference_day.sum()))

    alt_memory = alt_memory.loc[
        ~same_reference_day
    ].copy()

    # 3. 执行原文第二条规则：
    # 当期关联为1、滞后一期关联为0，
    # 且替代公告早于本行参照财报时，删除该配对
    before_reference = (
        alt_memory["quarterLag0Alt"].eq(1)
        & alt_memory["quarterLag1Alt"].eq(0)
        & alt_memory["date"].lt(alt_memory["event_Trddt"])
    ).fillna(False)

    print("满足原文第二条删除条件的配对数：",
          int(before_reference.sum()))

    alt_memory = alt_memory.loc[
        ~before_reference
    ].copy()

    print("两条筛选后剩余配对数：", len(alt_memory))

    return alt_memory.reset_index(drop=True)

# 合并提示公司前三交易日市值，并使用上月断点识别大市值公司
def add_cue_size(alt_memory, daily_path, size_bp_path):
    """合并提示公司历史市值和公告上月断点，生成大市值标记。"""

    # 1. 读取公司日度市值和市场交易日序号
    daily = pd.read_csv(
        daily_path,
        usecols=["Stkcd", "td", "Trddt", "size"],
        dtype={"Stkcd": "string"},
        parse_dates=["Trddt"],
        encoding="utf-8",
    )

    daily["Stkcd"] = daily["Stkcd"].apply(normalize_stock_code)
    daily["td"] = pd.to_numeric(daily["td"], errors="raise")
    daily["size"] = pd.to_numeric(daily["size"], errors="raise")

    # 检查公司日记录是否唯一，避免重复记录被随意取舍
    if daily.duplicated(["Stkcd", "td"]).any():
        raise ValueError("日度面板存在重复的公司与交易日记录。")

    # 2. 沿用现有方式，从日度面板提取日期与交易日序号的对应关系
    date_td = (
        daily[["Trddt", "td"]]
        .drop_duplicates()
        .rename(columns={"Trddt": "date", "td": "alt_td"})
    )

    result = alt_memory.merge(
        date_td,
        on="date",
        how="left",
        validate="m:1",
    )

    # 公告前第3个市场交易日
    result["td_tm3"] = result["alt_td"] - 3

    # 3. 按提示公司和历史交易日序号，合并当日总市值
    mktcap = daily[["Stkcd", "td", "size"]].rename(
        columns={
            "td": "td_tm3",
            "size": "mktcap_tm3",
        }
    )

    result = result.merge(
        mktcap,
        on=["Stkcd", "td_tm3"],
        how="left",
        validate="m:1",
    )

    # 4. 读取月度90%市值断点
    size_bp = pd.read_csv(
        size_bp_path,
        usecols=["Trdmnt", "marketCap_90th_percentile"],
        parse_dates=["Trdmnt"],
        encoding="utf-8-sig",
    )

    if size_bp["Trdmnt"].isna().any():
        raise ValueError("市值断点表存在缺失月份。")

    if size_bp["Trdmnt"].duplicated().any():
        raise ValueError("市值断点表同一个月份存在多条记录。")

    # 5. 使用替代公告上个月的断点
    result["Trdmnt_for_breakpoint"] = (
        result["date"]
        .dt.to_period("M")
        .sub(1)
        .dt.to_timestamp()
    )

    result = result.merge(
        size_bp.rename(
            columns={"Trdmnt": "Trdmnt_for_breakpoint"}
        ),
        on="Trdmnt_for_breakpoint",
        how="left",
        validate="m:1",
    )

    # 6. 沿用当前脚本：市值或断点缺失，删除对应配对
    missing_size = result["mktcap_tm3"].isna()
    missing_breakpoint = result["marketCap_90th_percentile"].isna()

    print("前三交易日市值缺失的配对数：", int(missing_size.sum()))
    print("上月市值断点缺失的配对数：",
          int(missing_breakpoint.sum()))

    result = result.dropna(
        subset=["mktcap_tm3", "marketCap_90th_percentile"]
    ).copy()

    # 7. 历史市值严格大于上月90%断点，标记为大市值公司
    result["size90plus"] = (
        result["mktcap_tm3"]
        > result["marketCap_90th_percentile"]
    ).astype("int8")

    return result

# 生成两类替代线索变量
event_specs = {
    "cueOtherEventER": "altOtherEventER",
    "cueForecast": "altForecast",
}

feature_map = {
    "lag1hiAEVs90p": "lag1hiAEV",
    "lag1hiavols90p": "lag1hiavol",
    "lag1hiavolSs90p": "lag1hiavolS",

    "lag1hi_sue1_abss90p": "lag1hi_sue1_abs",
    "lag1hi_sue2_abss90p": "lag1hi_sue2_abs",
    "lag1hi_sue3_abss90p": "lag1hi_sue3_abs",

    "lag1hi_sue1_poss90p": "lag1hi_sue1_pos",
    "lag1hi_sue1_negs90p": "lag1hi_sue1_neg",
    "lag1hi_sue2_poss90p": "lag1hi_sue2_pos",
    "lag1hi_sue2_negs90p": "lag1hi_sue2_neg",
    "lag1hi_sue3_poss90p": "lag1hi_sue3_pos",
    "lag1hi_sue3_negs90p": "lag1hi_sue3_neg",

    "lag1sue1poss90p": "lag1sue1pos",
    "lag1sue2poss90p": "lag1sue2pos",
    "lag1sue3poss90p": "lag1sue3pos",
    "lag1sue1negs90p": "lag1sue1neg",
    "lag1sue2negs90p": "lag1sue2neg",
    "lag1sue3negs90p": "lag1sue3neg",

    "lag1topMovers90p": "lag1topMover",
}


# 生成基础替代线索及历史记忆特征线索
def build_cue_variables(alt_memory, event_specs, feature_map):
    """按公告类别，生成基础线索和各类历史记忆特征线索。"""

    # 1. 检查参与计算的字段是否齐全
    required_cols = [
        "quarterLag1Alt",
        "size90plus",
        *event_specs.values(),
        *feature_map.values(),
    ]

    missing_cols = [
        col for col in required_cols
        if col not in alt_memory.columns
    ]

    if missing_cols:
        raise ValueError(f"生成替代线索缺少必要字段：{missing_cols}")

    # 2. 在副本上添加变量，避免直接改动传入的数据
    result = alt_memory.copy()

    # 3. 分别处理各类公告
    for prefix, event_col in event_specs.items():

        # 基础线索：发生该类公告、有滞后一期关联、提示公司为大市值
        result[f"{prefix}_lag1s90p"] = (
            result[event_col]
            * result["quarterLag1Alt"].fillna(0)
            * result["size90plus"]
        ).astype("int8")

        # 特征线索：发生该类公告、有对应历史特征、提示公司为大市值
        for suffix, source_col in feature_map.items():
            result[f"{prefix}_{suffix}"] = (
                result[event_col]
                * result[source_col].fillna(0)
                * result["size90plus"]
            ).astype("int8")

    return result

# 生成低、高干扰度替代线索
def add_interference_groups(alt_memory, event_specs):
    """按公告类别和替代公告年份，划分历史公告拥挤度。"""

    result = alt_memory.copy()

    # 1. 使用替代公告所在年份
    result["year"] = result["date"].dt.year

    # 2. 分别处理每类公告
    for prefix in event_specs:
        base = f"{prefix}_lag1s90p"
        score = f"{prefix}_nrFirmsAnnLag1s90p"
        group = f"{prefix}_interferenceGroupLag1s90p"

        # 3. 只有基础线索为1，才保留历史公告拥挤度
        # 其余行设为缺失，不参与年度中位数计算
        result[score] = result["nrFirmsAnnLag1Alt"].where(
            result[base].eq(1)
        )

        # 4. 在该类公告的配对行中，分别按年份划分低、高组
        result[group] = (
            result.groupby("year")[score]
            .transform(safe_qcut_2)
        )

        # 5. 生成低、高干扰度线索标记
        result[f"{prefix}_lowInterfLag1s90p"] = (
            result[group].eq(1)
        ).astype("int8")

        result[f"{prefix}_highInterfLag1s90p"] = (
            result[group].eq(2)
        ).astype("int8")

    return result


# 聚合到被提示公司与日期
def collapse_alt_cues(alt_memory, event_specs, feature_map):
    """明确选择线索标记，并聚合为唯一的被提示公司日记录。"""

    key_cols = ["StkcdRecall", "date"]

    # 1. 列出需要保留的指标后缀
    suffixes = [
        "lag1s90p",
        *feature_map.keys(),
        "lowInterfLag1s90p",
        "highInterfLag1s90p",
    ]

    # 2. 将公告类别前缀与指标后缀组合成完整列名
    cue_cols = [
        f"{prefix}_{suffix}"
        for prefix in event_specs
        for suffix in suffixes
    ]

    # 3. 检查必要字段
    missing_cols = [
        col for col in key_cols + cue_cols
        if col not in alt_memory.columns
    ]

    if missing_cols:
        raise ValueError(f"聚合替代线索缺少必要字段：{missing_cols}")

    if alt_memory[key_cols].isna().any().any():
        raise ValueError("聚合前存在缺失的被提示公司代码或日期。")

    # 4. 同一被提示公司当天，只要有一条配对满足，指标就记为1
    alt_cues = (
        alt_memory
        .groupby(key_cols, as_index=False)[cue_cols]
        .max()
        .sort_values(key_cols)
        .reset_index(drop=True)
    )

    return alt_cues, cue_cols

# 检查并保存最终替代线索表
def save_alt_cues(alt_cues, output_path):
    """检查公司日主键，将最终结果保存到指定路径。"""

    # 1. 将输出路径转换为路径对象
    output_path = Path(output_path)

    # 2. 检查主键字段是否齐全
    key_cols = ["StkcdRecall", "date"]

    missing_cols = [
        col for col in key_cols
        if col not in alt_cues.columns
    ]

    if missing_cols:
        raise ValueError(f"保存结果缺少主键字段：{missing_cols}")

    # 3. 检查主键是否缺失或重复
    if alt_cues[key_cols].isna().any().any():
        raise ValueError("最终结果存在缺失的被提示公司代码或日期。")

    if alt_cues.duplicated(key_cols).any():
        raise ValueError("最终结果在被提示公司与日期层面不唯一。")

    # 4. 在实际保存前创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. 保存结果，不输出行索引
    alt_cues.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print("替代线索表已保存：", output_path)
    print("输出行数：", len(alt_cues))
    print("输出列数：", alt_cues.shape[1])



# 主入口：按顺序组织整个构造流程
def main(root_dir):
    """读取数据，构造替代线索，聚合并保存最终结果。"""

    # 1. 根据项目根目录生成路径
    paths = build_paths(root_dir)

    # 2. 读取记忆矩阵和两类替代公告
    memory = load_memory(paths["memory"])

    event = load_alt_events(
        paths["forecast_events"],
        paths["other_events"],
    )

    # 3. 匹配记忆矩阵，并执行原文的两条逐行筛选
    alt_memory = match_alt_events_to_memory(
        event,
        memory,
    )

    # 4. 合并前三交易日市值，使用上月断点识别大市值公司
    alt_memory = add_cue_size(
        alt_memory,
        paths["daily"],
        paths["size_bp"],
    )

    # 5. 生成基础线索及历史记忆特征线索
    alt_memory = build_cue_variables(
        alt_memory,
        event_specs,
        feature_map,
    )

    # 6. 生成低、高干扰度线索
    alt_memory = add_interference_groups(
        alt_memory,
        event_specs,
    )

    # 7. 聚合到被提示公司与日期
    alt_cues, cue_cols = collapse_alt_cues(
        alt_memory,
        event_specs,
        feature_map,
    )

    print("聚合后行列数：", alt_cues.shape)
    print(
        alt_cues[cue_cols]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )

    # 8. 检查并保存结果
    save_alt_cues(alt_cues, paths["output"])

    return alt_cues


# 直接运行本脚本时，启动整个流程
if __name__ == "__main__":
    main(str(ROOT))