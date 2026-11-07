"""分别估计行业前4、前5提示的累计收益系数，并绘制独立图及对比图。

请自行运行本文件。本文件不重建行业排名或最终面板；导入时不读取研究数据。
每个累计收益窗口保留自身有效样本；小批窗口仅在有效样本完全相同时复用投影。
图中的系数不是将不同窗口的系数再次相加。
收益已在面板中缩尾，沿用原值；系数、标准误和置信区间均以百分点表示。
每完成一个窗口先保存核对摘要，再导出系数表，摘要用于中断后的恢复。
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
# 从入口旁的子目录直接导入，不依赖运行时修改模块搜索路径。
from 公共模块.行业排名回归公共模块 import (
    COMPUTATION_VERSION,
    ENGINE_VERSION,
    check_anchor,
    find_root,
    iter_cumulative_models,
    output_root,
    prepare_cache,
    runtime_guard,
    save_csv,
    save_json,
    verify_base_keys,
)

import pandas as pd


# 一、运行参数。TOP_NS（每行业名额）可改为(4,)或(5,)单独运行。
TOP_NS = (4, 5)
START_YEAR = 2005
END_YEAR = 2025
CHUNK_SIZE = 250_000  # 分块读取行数；只读取回归所需字段。
BATCH_SIZE = 5  # 窗口小批数，用于控制内存；不改变任何窗口的研究样本或估计设置。
HORIZONS = tuple(range(1, 51))  # 累计窗口终点；1对应[t,t+1]，含两个交易日。
SPEC_ID = 6  # 第6规格：完整控制、公司×年度季度及交易日期固定效应。
RESUME = True  # 续算：仅复用输入、数学口径及估计设置一致的已完成窗口。
DRAW_ONLY = False  # 只重画：不准备缓存、不读取大型输入面板、不重新估计。

RESULT_VERSION = 1
INTERVALS = (
    (99, "#bdbdbd", 0.8, 0.75),
    (95, "#78909c", 1.6, 0.90),
    (90, "#1f4e79", 2.8, 1.00),
)
NUMERIC_FIELDS = [
    "行业前几名", "窗口序号", "规格", "线索系数", "聚类标准误", "t值", "p值",
    "回归样本量", "有效提示公司日", "被提示公司聚类数", "日期聚类数",
    "删除缺失观测数", "删除单例组观测数", "固定效应秩", "残差自由度",
    "解释变量矩阵条件数", "R方", "组内R方",
    *[f"{level}%置信区间{side}" for level in (90, 95, 99) for side in ("下界", "上界")],
]
TEXT_FIELDS = [
    "提示变量", "收益窗口", "因变量", "规格说明", "日期固定效应",
    "被提示公司×年度季度固定效应", "求解方式", "结果单位", "输入签名", "模型状态",
]


def validate_settings():
    """先检查可编辑参数，避免选错提示列、窗口或固定效应。"""
    if not TOP_NS or any(type(n) is not int or n not in (4, 5) for n in TOP_NS):
        raise ValueError("每行业名额只支持4或5，且不能为空。")
    if len(set(TOP_NS)) != len(TOP_NS):
        raise ValueError("每行业名额不能重复。")
    if type(START_YEAR) is not int or type(END_YEAR) is not int or START_YEAR > END_YEAR:
        raise ValueError("起止年份必须为整数，且起始年不晚于结束年。")
    if type(CHUNK_SIZE) is not int or CHUNK_SIZE <= 0:
        raise ValueError("分块读取行数必须为正整数。")
    if type(BATCH_SIZE) is not int or BATCH_SIZE <= 0:
        raise ValueError("窗口小批数必须为正整数。")
    if not HORIZONS or any(type(h) is not int or not 1 <= h <= 50 for h in HORIZONS):
        raise ValueError("累计窗口终点必须为1至50的整数。")
    if tuple(sorted(set(HORIZONS))) != tuple(HORIZONS):
        raise ValueError("累计窗口必须递增排列，且不能重复。")
    if SPEC_ID != 6:
        raise ValueError("本累计图使用第6规格，请勿混入其他固定效应规格。")
    if type(RESUME) is not bool or type(DRAW_ONLY) is not bool:
        raise ValueError("续算和只重画开关必须为布尔值。")


def calculation_settings(top_n):
    """单套结果的估计设置；运行开关和读取块大小不改变模型。"""
    return {
        "结果版本": RESULT_VERSION, "行业前几名": top_n,
        "起始年份": START_YEAR, "结束年份": END_YEAR,
        "窗口终点": list(HORIZONS), "规格": SPEC_ID,
        "提示变量": f"quarterLag1IndTop{top_n}",
        "样本规则": "每个窗口单独检查自身收益及本规格所需变量",
        "置信区间": "双侧逐点区间，与回归显著性使用同一残差自由度的t分布",
    }


def entry_code_hash():
    """计算入口来源代码摘要；仅用于追溯，数学口径相同的优化允许续算。

    AST（抽象语法树）摘要保留实际代码，忽略注释与排版变化。
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    parameter_names = {
        "TOP_NS", "START_YEAR", "END_YEAR", "CHUNK_SIZE", "BATCH_SIZE", "HORIZONS", "SPEC_ID",
        "RESUME", "DRAW_ONLY",
    }
    tree.body = [
        node for node in tree.body
        if not (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in parameter_names
                    for target in node.targets)
        )
    ]
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()


def result_paths(root, top_n):
    folder = output_root(root) / "累计系数图"
    return {
        "结果表": folder / f"行业前{top_n}名_累计系数.csv",
        "摘要": folder / f"行业前{top_n}名_回归核对摘要.json",
        "图片": folder / f"行业前{top_n}名_累计系数图.png",
    }


def validate_rows(rows, top_n, input_id, *, complete=False):
    """检查完成标记、模型身份和必要统计量；不以存在文件代替估计完成。"""
    if not rows:
        raise ValueError(f"行业前{top_n}名没有已完成窗口。")
    frame = pd.DataFrame(rows)
    missing = [field for field in [*NUMERIC_FIELDS, *TEXT_FIELDS, "显著性", "常数项"]
               if field not in frame]
    if missing:
        raise ValueError(f"行业前{top_n}名结果缺少字段：{missing}")
    for field in NUMERIC_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
        if not frame[field].map(math.isfinite).all():
            raise ValueError(f"行业前{top_n}名结果中的{field}存在缺失或非有限数值。")
    for field in TEXT_FIELDS:
        if frame[field].isna().any() or frame[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"行业前{top_n}名结果中的{field}不完整。")
    expected = {
        "行业前几名": top_n, "提示变量": f"quarterLag1IndTop{top_n}",
        "规格": SPEC_ID, "输入签名": input_id, "模型状态": "完成",
        "结果单位": "百分点", "日期固定效应": "是",
        "被提示公司×年度季度固定效应": "是",
    }
    for field, value in expected.items():
        if not frame[field].eq(value).all():
            raise ValueError(f"行业前{top_n}名已有结果的{field}与本次口径不一致。")
    if frame["窗口序号"].duplicated().any() or not frame["窗口序号"].isin(HORIZONS).all():
        raise ValueError("已有结果包含重复窗口或本次未指定的窗口。")
    if (frame["聚类标准误"] <= 0).any() or (frame["残差自由度"] <= 0).any():
        raise ValueError("已有结果包含无效标准误或残差自由度。")
    if not frame["p值"].between(0, 1).all() or (frame["回归样本量"] <= 0).any():
        raise ValueError("已有结果的显著性概率或样本量无效。")
    if (frame["有效提示公司日"] <= 0).any() or (frame["有效提示公司日"] > frame["回归样本量"]).any():
        raise ValueError("已有结果的有效提示公司日数无效。")
    for _, row in frame.iterrows():
        h = int(row["窗口序号"])
        if row["因变量"] != f"chAdjreturn_t_tp{h}_Recall_w":
            raise ValueError(f"第{h}窗口的因变量与窗口终点不一致。")
        for level in (90, 95, 99):
            if not row[f"{level}%置信区间下界"] <= row["线索系数"] <= row[f"{level}%置信区间上界"]:
                raise ValueError(f"第{h}窗口的{level}%置信区间无效。")
    frame = frame.sort_values("窗口序号").reset_index(drop=True)
    if complete and frame["窗口序号"].tolist() != list(HORIZONS):
        raise ValueError("累计图所需窗口尚未全部完成，请先续算。")
    return frame


def snapshot_rows(rows):
    """保留原窗口字典，只将可选字段的缺失规范为空值。

    新旧结果可选字段不同，转为数据表时会自动补缺失；不能将这些补出的
    缺失数值写入严格摘要。必填数值和文字不在这里填补，继续由校验函数检查。
    """
    required = set(NUMERIC_FIELDS) | set(TEXT_FIELDS)
    result = []
    for original in rows:
        row = dict(original)
        for field, value in original.items():
            if field in required:
                continue
            if pd.isna(value) or (isinstance(value, str) and value == "" and field != "显著性"):
                row[field] = "" if field == "显著性" else None
        result.append(row)
    return result


def read_saved_result(paths, top_n, *, input_id=None):
    """摘要是权威进度；只重画时不核验或读取原始面板。"""
    if not paths["摘要"].is_file():
        raise FileNotFoundError(f"找不到回归核对摘要：{paths['摘要']}")
    metadata = json.loads(paths["摘要"].read_text(encoding="utf-8"))
    if metadata.get("设置") != calculation_settings(top_n):
        raise ValueError("已有结果设置与本次不同，不能续算或直接重画。")
    if metadata.get("输入信息", {}).get("version") != ENGINE_VERSION:
        raise ValueError("已有结果的数学口径版本与当前版本不同，不能续算或直接重画。")
    if input_id is not None and metadata.get("输入签名") != input_id:
        raise ValueError("输入数据或数学口径已经变化，不能复用旧窗口；请关闭续算后重新估计。")
    signature = metadata.get("输入签名")
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError("已有摘要缺少有效输入签名。")
    frame = validate_rows(metadata.get("模型结果", []), top_n, signature, complete=DRAW_ONLY)
    return metadata, frame


def table_anchors(root, top_n, input_id):
    """只读取同输入签名的表2第6规格，第1、第10窗口可直接复用。"""
    path = output_root(root) / "回归结果" / f"行业前{top_n}名" / "模型汇总.csv"
    if not path.is_file():
        return {}
    # 不把“不显著”的空星号读成缺失，保证复用后摘要仍能严格序列化。
    table = pd.read_csv(path, keep_default_na=False)
    required = {"输入签名", "规格", "窗口序号"}
    if not required.issubset(table.columns):
        raise ValueError(f"已有模型汇总缺少口径核对字段：{path}")
    selected = table.loc[
        table["输入签名"].eq(input_id)
        & pd.to_numeric(table["规格"], errors="raise").eq(SPEC_ID)
        & pd.to_numeric(table["窗口序号"], errors="raise").isin(set(HORIZONS) & {1, 10})
    ]
    if selected.empty:
        return {}
    selected = validate_rows(selected.to_dict("records"), top_n, input_id)
    records = snapshot_rows(selected.to_dict("records"))
    return {int(row["窗口序号"]): row for row in records}


def save_progress(paths, metadata, rows):
    """先保存完整窗口摘要，再导出表；中断后可从摘要恢复表。"""
    metadata = dict(metadata)
    metadata["模型结果"] = sorted(snapshot_rows(rows), key=lambda row: row["窗口序号"])
    metadata["已完成窗口"] = [row["窗口序号"] for row in metadata["模型结果"]]
    metadata["状态"] = "完成" if metadata["已完成窗口"] == list(HORIZONS) else "计算中"
    save_json(paths["摘要"], metadata)
    save_csv(paths["结果表"], metadata["模型结果"])


def common_estimation_settings(signature):
    """对比两种名额共同的估计口径，提示列本身不纳入共同控制列表。"""
    fields = ("version", "start_year", "end_year", "returns", "inference")
    if not isinstance(signature, dict) or any(field not in signature for field in (*fields, "x_columns")):
        raise ValueError("结果摘要缺少共同估计口径，不能绘制前4与前5对比图。")
    if not isinstance(signature["x_columns"], list) or not signature["x_columns"]:
        raise ValueError("结果摘要中的解释变量列表无效。")
    return {**{field: signature[field] for field in fields}, "x_columns": signature["x_columns"][1:]}


def verify_saved_comparison(metadata_by_top):
    """只凭小型摘要核对两套历史结果，不要求历史估计代码等于当前代码。"""
    if not {4, 5}.issubset(metadata_by_top):
        return
    required = {"quarter", "rows", "key_digest", "base_data_digest"}
    common_settings = []
    base_samples = []
    for top_n in (4, 5):
        metadata = metadata_by_top[top_n]
        common = common_estimation_settings(metadata.get("输入信息"))
        saved_common = metadata.get("公共估计设置")
        # 旧摘要额外保存的模块文件哈希是实现来源，不再作为数学口径相同性条件。
        if not isinstance(saved_common, dict) or {key: saved_common.get(key) for key in common} != common:
            raise ValueError(f"行业前{top_n}名摘要的公共估计设置缺失或与输入信息冲突。")
        base = metadata.get("基础样本摘要")
        if not isinstance(base, list) or not base or any(
            not isinstance(quarter, dict) or not required.issubset(quarter) for quarter in base
        ):
            raise ValueError(f"行业前{top_n}名缺少完整基础样本摘要，不能绘制两者对比图。")
        common_settings.append(common)
        base_samples.append(base)
    if common_settings[0] != common_settings[1]:
        raise ValueError("前4与前5历史结果的估计版本、控制变量或年份等口径不同，不能叠加比较。")
    if base_samples[0] != base_samples[1]:
        raise ValueError("前4与前5历史结果的基础公司日、控制或收益不同，不能叠加比较。")


def record_computation(metadata, code_hash):
    """保留旧窗口的来源代码，并追加本次计算实现，不覆盖历史来源记录。"""
    metadata = dict(metadata)
    history = list(metadata.get("计算实现记录", []))
    if not history and metadata.get("入口代码摘要"):
        history.append({
            "入口代码摘要": metadata["入口代码摘要"],
            "公共模块摘要": metadata.get("输入信息", {}).get("module_sha256"),
            "计算实现版本": metadata.get("计算实现版本", "旧版逐窗口估计"),
        })
    module_path = Path(__file__).resolve().parent / "公共模块/行业排名回归公共模块.py"
    current = {
        "入口代码摘要": code_hash,
        "公共模块摘要": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "计算实现版本": COMPUTATION_VERSION,
    }
    if current not in history:
        history.append(current)
    metadata["计算实现记录"] = history
    metadata["入口代码摘要"] = code_hash
    metadata["计算实现版本"] = COMPUTATION_VERSION
    return metadata


def plotting_tools():
    """自动选用本机中文字体，仅在绘图时加载绘图库。"""
    import matplotlib
    matplotlib.use("Agg")  # 无窗口绘图后端。
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti", "Arial Unicode MS", "Noto Sans CJK SC", "SimHei"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    else:
        raise RuntimeError("未找到可用中文字体，请先安装中文字体后只重画。")
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def style_axis(axis):
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("累计收益窗口终点（交易日）")
    axis.set_ylabel("提示变量系数（百分点）")
    ticks = [h for h in (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50) if h in HORIZONS]
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"[t,t+{h}]" for h in ticks])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")


def draw_individual(frame, top_n, path):
    plt = plotting_tools()
    figure, axis = plt.subplots(figsize=(10, 5))
    for level, color, width, alpha in INTERVALS:
        axis.vlines(frame["窗口序号"], frame[f"{level}%置信区间下界"], frame[f"{level}%置信区间上界"],
                    colors=color, linewidth=width, alpha=alpha, label=f"{level}%逐点置信区间")
    colors = frame["p值"].map(lambda value: "#d62728" if value < 0.10 else "#1a1a1a")
    axis.scatter(frame["窗口序号"], frame["线索系数"], color=colors, s=16, zorder=4)
    axis.scatter([], [], color="#d62728", s=16, label="红点：10%水平显著")
    axis.set_title(f"行业前{top_n}名提示：累计收益动态\n被提示公司×年度季度及交易日期固定效应")
    style_axis(axis)
    axis.legend(frameon=False, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def draw_comparison(frames, root):
    """叠加展示两种提示定义；阴影为各自95%区间，不作系数差异检验。"""
    if not {4, 5}.issubset(frames):
        return
    plt = plotting_tools()
    figure, axis = plt.subplots(figsize=(10, 5))
    for top_n, color in ((4, "#1f77b4"), (5, "#e67e22")):
        frame = frames[top_n]
        x = frame["窗口序号"].to_numpy(dtype=float)
        axis.plot(x, frame["线索系数"].to_numpy(dtype=float), color=color, linewidth=1.5, label=f"行业前{top_n}名")
        axis.fill_between(x, frame["95%置信区间下界"].to_numpy(dtype=float),
                          frame["95%置信区间上界"].to_numpy(dtype=float), color=color, alpha=0.15)
    axis.set_title("行业前4名与前5名提示：累计收益系数对比\n阴影为各自95%逐点置信区间")
    style_axis(axis)
    axis.legend(frameon=False, fontsize=10)
    path = output_root(root) / "累计系数图" / "行业前4名与前5名_累计系数对比.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"对比图已保存：{path}", flush=True)


def main():
    validate_settings()
    root = find_root()
    # 绘图库的字体缓存属于本任务临时文件，只放项目工作目录。
    font_cache = root / "work/行业排名回归/绘图字体缓存"
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(font_cache)
    frames = {}
    # 二、只重画直接读取小型结果摘要，完全跳过大面板及缓存准备。
    if DRAW_ONLY:
        metadata_by_top = {}
        for top_n in TOP_NS:
            paths = result_paths(root, top_n)
            metadata, frame = read_saved_result(paths, top_n)
            metadata_by_top[top_n] = metadata
            frames[top_n] = frame
            draw_individual(frame, top_n, paths["图片"])
            print(f"行业前{top_n}名已重画：{paths['图片']}", flush=True)
        verify_saved_comparison(metadata_by_top)
        draw_comparison(frames, root)
        return

    # 三、按名额准备紧凑缓存，核对前4与前5使用同一批被提示公司日。
    caches = [prepare_cache(root, n, START_YEAR, END_YEAR, CHUNK_SIZE) for n in TOP_NS]
    verify_base_keys(caches)
    code_hash = entry_code_hash()
    for top_n, cache in zip(TOP_NS, caches):
        paths = result_paths(root, top_n)
        input_id = cache["input_id"]
        anchors = table_anchors(root, top_n, input_id)
        rows = []
        metadata = {
            "设置": calculation_settings(top_n),
            "输入签名": input_id, "输入信息": cache["signature"],
            "公共估计设置": common_estimation_settings(cache["signature"]),
            "基础样本摘要": [
                {field: quarter[field] for field in ("quarter", "rows", "key_digest", "base_data_digest")}
                for quarter in cache["quarters"]
            ],
        }
        if RESUME and (paths["摘要"].exists() or paths["结果表"].exists()):
            metadata, _ = read_saved_result(paths, top_n, input_id=input_id)
            # 已在读取时校验；恢复原字典，避免新旧可选字段被数据表统一补缺失。
            rows = snapshot_rows(metadata["模型结果"])
        metadata = record_computation(metadata, code_hash)
        completed = {int(row["窗口序号"]): row for row in rows}
        # 四、先保存已有结果并优先复用表2锚点，避免把第1、第10窗口重复估计。
        if rows:
            save_progress(paths, metadata, rows)
        for horizon, anchor in sorted(anchors.items()):
            if horizon in completed:
                check_anchor(completed[horizon], anchor)
                continue
            rows.append(anchor)
            completed[horizon] = anchor
            save_progress(paths, metadata, rows)
            print(f"行业前{top_n}名：复用表2第6规格窗口[t,t+{horizon}]。", flush=True)
        # 五、小批估计缺失窗口；生成器仅对有效样本完全相同的窗口复用投影。
        pending = [h for h in HORIZONS if h not in completed]
        new_results = iter_cumulative_models(cache, pending, batch_size=BATCH_SIZE) if pending else ()
        for row in new_results:
            validate_rows([row], top_n, input_id)
            horizon = int(row["窗口序号"])
            if horizon not in pending or horizon in completed:
                raise ValueError("累计窗口生成器返回了非待估计窗口或重复窗口。")
            if horizon in anchors:
                check_anchor(row, anchors[horizon])
            rows.append(row)
            completed[horizon] = row
            save_progress(paths, metadata, rows)
            print(f"行业前{top_n}名：窗口[t,t+{horizon}]已完成并保存。", flush=True)
        frame = validate_rows(rows, top_n, input_id, complete=True)
        save_progress(paths, metadata, rows)
        frames[top_n] = frame
        # 六、完整结果通过核对后绘图，并保留小型结果供以后只重画。
        draw_individual(frame, top_n, paths["图片"])
        print(f"累计系数结果：{paths['结果表']}\n累计系数图：{paths['图片']}", flush=True)
    draw_comparison(frames, root)


if __name__ == "__main__":
    if DRAW_ONLY:
        main()
    else:
        with runtime_guard(find_root(), "累计图"):
            main()
