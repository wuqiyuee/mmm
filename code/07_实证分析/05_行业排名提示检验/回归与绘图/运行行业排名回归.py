"""分别估计行业前4、前5提示的两种收益窗口、六种回归规格。

输入为已经构造完成的行业提示面板，本文件不调用排名或面板构建入口。
两个窗口为[t,t+1]和[t,t+10]；每个门槛共12个模型。收益直接使用
已有缩尾列；标准误按被提示公司和事件起点日期双向聚类。

正式输出位于“图表结果/05_行业排名提示检验/回归结果/行业前N名”。
模型汇总的系数、标准误和置信区间均为百分点；系数明细同时保留小数和百分点列。
逐模型保存核对摘要及两张结果表，相同输入和估计口径可继续未完成模型。
每个门槛的12个模型完成后，按主回归样式生成紧凑版及两页完整系数表图片。
已有12个同口径模型时再次运行，会恢复结果并重新出图，不重新估计这些模型。
计算提速不改变估计口径时保留旧模型，并逐模型记录原计算实现的来源。
核对摘要是完整模型的权威快照，写表中断时可从摘要恢复结果表。
本文件只在直接执行时准备缓存、估计并保存；导入时不执行回归。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys


# 一、集中修改运行设置；两种提示分别估计，不同时放入一个模型。
# TOP_NS（每行业入选名额）；START_YEAR、END_YEAR（起止年份）。
TOP_NS = (4, 5)
START_YEAR = 2005
END_YEAR = 2025
# CHUNK_SIZE（每次读取行数）；只影响缓存准备时的内存占用。
CHUNK_SIZE = 250_000
HORIZONS = (1, 10)  # HORIZONS（累计收益窗口终点）。
SPEC_IDS = (1, 2, 3, 4, 5, 6)  # SPEC_IDS（六种递进回归规格）。

sys.dont_write_bytecode = True
SCRIPT_PATH = Path(__file__).resolve()
COMMON_DIR = SCRIPT_PATH.parent / "公共模块"
# 从入口旁的子目录直接导入，运行和编辑器均能看到明确的模块位置。
from 公共模块 import 行业排名回归公共模块 as common


def file_hash(path: Path) -> str:
    """记录实际使用的脚本内容，供追溯各模型的计算实现。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_source(settings: dict) -> dict:
    """保留每个模型原有的脚本校验值，提速续算时不覆盖旧来源。"""
    return {
        "入口脚本校验值": settings.get("入口脚本校验值", "未记录"),
        "公共模块校验值": settings.get("公共模块校验值", "未记录"),
        "估计口径版本": settings.get("估计口径版本", settings["缓存签名"].get("version")),
        "计算实现版本": settings.get("计算实现版本", "旧版未单独记录"),
    }


def validate_settings() -> None:
    """只接受本实验已准备的前4、前5，避免误读其他门槛文件。"""
    if not TOP_NS or any(type(n) is not int or n not in (4, 5) for n in TOP_NS):
        raise ValueError("每行业入选名额只能填写(4,)、(5,)或(4, 5)。")
    if len(set(TOP_NS)) != len(TOP_NS):
        raise ValueError("每行业入选名额不能重复。")
    if type(START_YEAR) is not int or type(END_YEAR) is not int or START_YEAR > END_YEAR:
        raise ValueError("起止年份必须为整数，且起始年份不能晚于结束年份。")
    if type(CHUNK_SIZE) is not int or CHUNK_SIZE < 1:
        raise ValueError("分块读取行数必须为正整数。")


def validate_model(model: dict, top_n: int, input_id: str) -> tuple[int, int]:
    """逐项检查可恢复的模型，缺少明细的模型不视为已经完成。"""
    summary = model.get("summary")
    coefficients = model.get("coefficients")
    if not isinstance(summary, dict) or not isinstance(coefficients, list) or not coefficients:
        raise ValueError("已保存模型缺少模型汇总或系数明细，不能继续计算。")
    required = {
        "行业前几名", "提示变量", "窗口序号", "规格", "线索系数", "聚类标准误",
        "p值", "回归样本量", "残差自由度", "结果单位", "输入签名", "模型状态",
    }
    if required - summary.keys():
        raise ValueError(f"模型汇总缺少字段：{sorted(required - summary.keys())}")
    horizon, spec_id = summary["窗口序号"], summary["规格"]
    if horizon not in HORIZONS or spec_id not in SPEC_IDS:
        raise ValueError("已保存模型包含本次任务以外的窗口或规格。")
    if (
        summary["行业前几名"] != top_n
        or summary["提示变量"] != f"quarterLag1IndTop{top_n}"
        or summary["输入签名"] != input_id
        or summary["结果单位"] != "百分点"
        or summary["模型状态"] != "完成"
    ):
        raise ValueError("已保存模型的行业门槛、提示变量、输入签名或完成状态不符。")
    for name in ("线索系数", "聚类标准误", "p值", "回归样本量", "残差自由度"):
        value = summary[name]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"模型汇总中的{name}不是有限数值。")
    if summary["聚类标准误"] <= 0 or summary["回归样本量"] <= 0 or summary["残差自由度"] <= 0:
        raise ValueError("模型标准误、样本量或残差自由度无效。")
    if not 0 <= summary["p值"] <= 1:
        raise ValueError("模型显著性概率不在零至一之间。")
    variables = []
    for row in coefficients:
        if not isinstance(row, dict) or not row.get("变量"):
            raise ValueError("系数明细缺少变量名称。")
        if any(row.get(key) != summary[key] for key in ("行业前几名", "窗口序号", "规格")):
            raise ValueError("系数明细的行业、窗口或规格与模型汇总不一致。")
        variables.append(row["变量"])
    if len(variables) != len(set(variables)) or variables.count(summary["提示变量"]) != 1:
        raise ValueError("系数明细存在重复变量，或缺少唯一的行业提示系数。")
    return int(horizon), int(spec_id)


def load_snapshot(path: Path, settings: dict, output_dir: Path) -> list[dict]:
    """允许数学口径不变的计算提速续算，输入或研究设置变化仍报错。"""
    if not path.is_file():
        if any((output_dir / name).exists() for name in ("模型汇总.csv", "系数明细.csv")):
            raise ValueError(f"已有结果表但缺少核对摘要，不能确认其口径，请先移走旧结果：{output_dir}")
        return []
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or snapshot.get("状态") not in {"完成", "计算中"}:
        raise ValueError(f"核对摘要的结构或状态无效：{path}")
    saved_settings = snapshot.get("设置")
    if not isinstance(saved_settings, dict):
        raise ValueError(f"核对摘要缺少原运行设置：{path}")
    research_fields = (
        "行业前几名", "起始年份", "结束年份", "窗口终点", "规格",
        "输入签名", "缓存签名", "结果单位",
    )
    if any(saved_settings.get(field) != settings[field] for field in research_fields):
        raise ValueError(f"输入或研究参数已变化，不能混用旧结果；请先移走原结果目录：{output_dir}")
    # 旧摘要没有单列估计版本，只在其缓存版本仍为当前数学口径时接受。
    saved_version = saved_settings.get(
        "估计口径版本", saved_settings["缓存签名"].get("version"),
    )
    if saved_version != settings["估计口径版本"]:
        raise ValueError(f"估计口径版本不同，不能续算旧模型：{path}")
    models = snapshot.get("模型结果")
    if not isinstance(models, list):
        raise ValueError(f"核对摘要缺少可恢复的模型记录：{path}")
    keys = [validate_model(model, settings["行业前几名"], settings["输入签名"]) for model in models]
    if len(keys) != len(set(keys)):
        raise ValueError(f"核对摘要中的窗口、规格键重复：{path}")
    if snapshot.get("已完成模型数") != len(models):
        raise ValueError(f"核对摘要中的完成数量与实际模型记录不符：{path}")
    expected = {(h, s) for h in HORIZONS for s in SPEC_IDS}
    if snapshot.get("状态") == "完成" and set(keys) != expected:
        raise ValueError(f"核对摘要标记完成，但没有完整的12个模型：{path}")
    for model in models:
        model.setdefault("实现来源", implementation_source(saved_settings))
    return models


def save_results(output_dir: Path, settings: dict, models: list[dict]) -> None:
    """先保存完整模型快照，再导出两张表；中断后可恢复缺少的导出。"""
    models.sort(key=lambda model: (model["summary"]["窗口序号"], model["summary"]["规格"]))
    summary_rows = [model["summary"] for model in models]
    coefficient_rows = [row for model in models for row in model["coefficients"]]
    common.save_json(output_dir / "回归核对摘要.json", {
        "设置": settings,
        "状态": "完成" if len(models) == len(HORIZONS) * len(SPEC_IDS) else "计算中",
        "已完成模型数": len(models),
        "应完成模型数": len(HORIZONS) * len(SPEC_IDS),
        "模型结果": models,
        "说明": "摘要保存完整模型作为续算依据；模型汇总及系数明细由同一摘要导出。",
    })
    common.save_csv(output_dir / "模型汇总.csv", summary_rows)
    common.save_csv(output_dir / "系数明细.csv", coefficient_rows)


def draw_result_images(root: Path, output_dir: Path, settings: dict, models: list[dict]) -> None:
    """使用已保存的完整模型绘制三张图片，不重新估计回归。"""
    top_n = settings["行业前几名"]
    expected_keys = [(h, s) for h in HORIZONS for s in SPEC_IDS]
    model_lookup = {
        validate_model(model, top_n, settings["输入签名"]): model
        for model in models
    }
    if len(models) != len(expected_keys) or set(model_lookup) != set(expected_keys):
        raise ValueError("图片式表格要求两个窗口、六种规格的12个模型全部完成。")
    ordered_models = [model_lookup[key] for key in expected_keys]
    summaries = [model["summary"] for model in ordered_models]

    # 一、数值结果保存且估计循环结束后才加载绘图模块，减少计算时的内存占用。
    import matplotlib
    matplotlib.use("Agg")
    public_dir = root / "code/07_实证分析/90_公共模块"
    if str(public_dir) not in sys.path:
        sys.path.insert(0, str(public_dir))
    from 回归表格绘制 import (
        A4_LANDSCAPE_HEIGHT_IN, A4_LANDSCAPE_WIDTH_IN,
        format_estimate, render_publication_table, variable_label,
    )

    cue = f"quarterLag1IndTop{top_n}"
    title = f"行业前{top_n}名公司提示与被提示公司收益"
    columns = [f"({s})" for s in SPEC_IDS] * len(HORIZONS)
    groups = [(0, 5, "收益窗口 [t, t+1]"), (6, 11, "收益窗口 [t, t+10]")]
    notes = [
        f"注：样本期为{settings['起始年份']}—{settings['结束年份']}年；提示变量为上一季度每行业市值前{top_n}名公司提示。",
        "第（2）列起含自身盈余公告前后各10日及当日的21项控制；第（3）列起另加6项控制。",
        "括号内为按被提示公司和事件起点日期双向聚类的标准误；系数与标准误单位均为百分点。",
        "决定系数为模型汇总中的总体决定系数；星号按原始双侧检验概率计算，未作多重检验调整。",
        "*、**、***分别表示10%、5%、1%显著水平；空白表示该变量未纳入或无法估计。",
    ]

    # 二、与主回归相同：两窗左右并排，各含六种规格，系数下方列示标准误。
    stat_rows = [
        {"kind": "stat", "label": "自身盈余公告控制", "line_before": True,
         "values": ["是" if s >= 2 else "否" for h, s in expected_keys]},
        {"kind": "stat", "label": "其他控制变量",
         "values": ["是" if s >= 3 else "否" for h, s in expected_keys]},
        {"kind": "stat", "label": "日期固定效应",
         "values": [row["日期固定效应"] for row in summaries]},
        {"kind": "stat", "label": "被提示公司×年度季度固定效应",
         "values": [row["被提示公司×年度季度固定效应"] for row in summaries]},
        {"kind": "stat", "label": "样本量",
         "values": [f"{int(row['回归样本量']):,}" for row in summaries]},
        {"kind": "stat", "label": r"$R^2$",
         "values": [f"{row['R方']:.4f}" for row in summaries]},
    ]
    estimates = [
        format_estimate(row["线索系数"], row["聚类标准误"], row["p值"])
        for row in summaries
    ]
    compact_rows = [
        {"kind": "panel", "label": f"每行业入选前{top_n}名；两窗各六种规格",
         "values": [], "bold": True},
        {"kind": "coefficient", "label": f"行业前{top_n}名提示",
         "values": [value[0] for value in estimates], "bold": True},
        {"kind": "standard_error", "label": "（聚类标准误）",
         "values": [value[1] for value in estimates], "color": "#444444"},
        *stat_rows,
    ]
    compact_path = output_dir / f"行业前{top_n}名_收益回归_紧凑版.png"
    render_publication_table(
        table_number="行业排名提示检验", title=title,
        subtitle="因变量：被提示公司的特征调整收益率；系数与标准误的单位均为百分点",
        group_headers=groups, column_labels=columns, rows=compact_rows,
        notes=notes, output_path=compact_path, width=23.0,
        label_right=0.285, body_fontsize=9.0,
    )
    print(f"已保存图片：{compact_path}", flush=True)

    # 三、完整系数表使用明细自带的百分点列，不能再乘100；沿用主回归分页位置。
    coefficient_lookup = {}
    variable_orders = {}
    for model in ordered_models:
        for row in model["coefficients"]:
            variable = row["变量"]
            coefficient_lookup[(row["窗口序号"], row["规格"], variable)] = row
            if variable in variable_orders and variable_orders[variable] != row["变量顺序"]:
                raise ValueError(f"各模型中的变量顺序不一致：{variable}")
            variable_orders[variable] = row["变量顺序"]
    variables = sorted(variable_orders, key=variable_orders.get)
    split = variables.index("ownEA_tp3") + 1
    labels = {
        cue: f"行业前{top_n}名提示",
        "logIncreaseAEV": "新闻量增加的对数",
        "logDecreaseAEV": "新闻量减少的对数",
    }
    for page, page_variables in enumerate((variables[:split], variables[split:]), start=1):
        rows = []
        for variable in page_variables:
            estimates = []
            for horizon, spec_id in expected_keys:
                detail = coefficient_lookup.get((horizon, spec_id, variable))
                if detail is None:
                    estimates.append(("", ""))
                else:
                    estimates.append(format_estimate(
                        detail["系数（百分点）"], detail["聚类标准误（百分点）"], detail["p值"],
                    ))
            rows.extend([
                {"kind": "coefficient", "label": labels.get(variable, variable_label(variable)),
                 "values": [value[0] for value in estimates], "bold": variable == cue,
                 "line_before": variable == "ownEA_tm10"},
                {"kind": "standard_error", "label": "",
                 "values": [value[1] for value in estimates], "color": "#444444"},
            ])
        if page == 2:
            rows.extend(stat_rows)
        suffix = "_续页" if page == 2 else ""
        path = output_dir / f"行业前{top_n}名_收益回归_完整系数版{suffix}.png"
        render_publication_table(
            table_number="行业排名提示检验" + ("（续）" if page == 2 else ""),
            title=title,
            subtitle=f"样本期：{settings['起始年份']}—{settings['结束年份']}年；每行业入选前{top_n}名",
            group_headers=groups, column_labels=columns, rows=rows,
            notes=notes + (["续见下一页。"] if page == 1 else []),
            output_path=path, width=A4_LANDSCAPE_WIDTH_IN,
            height=A4_LANDSCAPE_HEIGHT_IN, label_right=0.205,
            body_fontsize=8.5, compact_header=True,
            dependent_variable_header="因变量：被提示公司的特征调整收益率（百分点）",
        )
        print(f"已保存图片：{path}", flush=True)


def check_existing_plot(root: Path, top_n: int, summary: dict) -> None:
    """同口径累计图已存在时，核对第1、10窗与第6列的结果。"""
    if summary["规格"] != 6:
        return
    path = common.output_root(root) / "累计系数图" / f"行业前{top_n}名_累计系数.csv"
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matching = [row for row in rows if row.get("输入签名") == summary["输入签名"]
                and row.get("窗口序号") == str(summary["窗口序号"])
                and row.get("规格") == "6"]
    if len(matching) > 1:
        raise ValueError(f"累计系数结果存在重复的同口径窗口：{path}")
    if matching:
        other = matching[0]
        for field in ("线索系数", "聚类标准误", "回归样本量", "残差自由度"):
            other[field] = float(other[field])
        common.check_anchor(summary, other)


def main() -> None:
    """准备紧凑缓存，核对两套基础键，然后逐模型估计和保存。"""
    validate_settings()
    root = common.find_root()
    print(f"行业提示收益回归：{START_YEAR}—{END_YEAR}年，门槛{TOP_NS}。", flush=True)

    # 二、缓存包含50个累计窗口，供表格与累计图共同使用。
    # 列表只保留缓存元信息；真正估计时按门槛、模型依次读取，避免同时预载两套数据。
    caches = [common.prepare_cache(root, n, START_YEAR, END_YEAR, CHUNK_SIZE) for n in TOP_NS]
    common.verify_base_keys(caches)
    script_hash = file_hash(SCRIPT_PATH)
    helper_hash = file_hash(COMMON_DIR / "行业排名回归公共模块.py")

    # 三、同一套回归函数处理六种规格，参数和输入签名控制续算。
    for top_n, cache in zip(TOP_NS, caches):
        output_dir = common.output_root(root) / "回归结果" / f"行业前{top_n}名"
        settings = {
            "行业前几名": top_n, "起始年份": START_YEAR, "结束年份": END_YEAR,
            "窗口终点": list(HORIZONS), "规格": list(SPEC_IDS),
            "输入签名": cache["input_id"], "缓存签名": cache["signature"],
            "入口脚本校验值": script_hash, "公共模块校验值": helper_hash,
            "估计口径版本": common.ENGINE_VERSION,
            "计算实现版本": common.COMPUTATION_VERSION,
            "结果单位": "百分点",
        }
        models = load_snapshot(output_dir / "回归核对摘要.json", settings, output_dir)
        completed = {(m["summary"]["窗口序号"], m["summary"]["规格"]) for m in models}
        for model in models:
            check_existing_plot(root, top_n, model["summary"])
        if models:
            save_results(output_dir, settings, models)
        pending_models = [(h, s) for h in HORIZONS for s in SPEC_IDS if (h, s) not in completed]
        remaining = set(pending_models)
        print(f"行业前{top_n}名：已恢复{len(models)}/12个模型，待估计{len(pending_models)}个。", flush=True)
        # 四、同一规格的两个收益仅在有效样本完全一致时共享解释变量投影。
        # 公共模块可按规格批量计算，每返回一个完整模型仍立即核对并保存。
        for summary, coefficients in common.iter_table_models(cache, pending_models):
            model = {
                "summary": summary, "coefficients": coefficients,
                "实现来源": implementation_source(settings),
            }
            model_key = validate_model(model, top_n, cache["input_id"])
            if model_key not in remaining:
                raise ValueError("批量估计返回了未请求或重复的窗口、规格。")
            check_existing_plot(root, top_n, summary)
            models.append(model)
            save_results(output_dir, settings, models)
            remaining.remove(model_key)
            print(f"已完成{len(models)}/12个模型；提示系数={summary['线索系数']:.6f}个百分点。", flush=True)
        if remaining:
            raise ValueError(f"批量估计尚有模型未完成：{sorted(remaining)}")
        # 五、完整结果已保存后再绘图；恢复12/12个模型时也从这里补图。
        draw_result_images(root, output_dir, settings, models)
        print(f"行业前{top_n}名回归完成，结果目录：{output_dir}", flush=True)


if __name__ == "__main__":
    with common.runtime_guard(common.find_root(), "表格回归"):
        main()
