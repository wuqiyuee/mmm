"""独立准备行业前4、前5提示面板；请自行运行本文件。

本目录与“03_95分位检验”并列，不修改或覆盖原有脚本及原有面板。
默认先构造包含金融公司的日度行业排名，再分别生成前4和前5提示结果。
本入口只准备排名、提示事件与回归面板，不自动执行回归或绘图。

核心口径：
1. 排名时点为提示事件交易日前第3个市场交易日，使用当日总市值。
2. 排名参照池包含金融公司，其余市场、上市年限和交易状态规则沿用基础清洗口径。
3. 旧版细行业代码归并至前三字符，综合类M保留；粗代码无法细化时保留待核对。
4. 提示方允许金融公司，被提示方仍使用原项目的非金融主样本。
5. 使用归并后的事件日行业排除同业公司对，继续排除供应链、共同分析师。

运行时输出位置：
排名缓存与异常核对清单：项目根目录/work/行业排名提示检验。
独立提示事件及最终面板：项目根目录/中间结果/04_合并数据/行业排名提示检验。
相比原沪深300、市值90分位结果，本实验还统一了旧版同行业排除层级；
直接比较时不能把全部差异都归因于提示公司名单变化。

本文件以及公共模块在导入时均不准备数据或写出结果。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


# 一、只需在这里修改本次运行参数。
# RUN_STAGE（运行阶段）：all（全部）、rank（只准备排名）、panel（只准备面板）。
RUN_STAGE = "all"
START_YEAR = 2005
END_YEAR = 2025

# TOP_NS（每行业入选名额）：(4, 5)同时生成两套；只需要前5时改成(5,)。
TOP_NS = (4, 5)
CHUNK_SIZE = 400_000

# 旧版粗行业先用当时有效的同标准RESSET（锐思数据库）记录补充。
# 下列条目是逐项核对的例外，只在原代码、公司和日期区间同时匹配时使用。
# onlyIfIndustryCode（限定原代码）防止覆盖以后已补齐或已变更的有效行业。
# source（依据）和note（说明）会一并记录在排名缓存清单中，供复核。
INDUSTRY_OVERRIDES = [
    {
        "Stkcd": "000780",
        "start": "2002-03-06",
        "end": "2007-04-25",
        "industryGroup": "C01",
        "onlyIfIndustryCode": "C0",
        "source": "https://static.cninfo.com.cn/finalpage/2002-03-06/550159.html",
        "note": (
            "草原兴发2001年报在2002年3月6日公开，主营为肉羊、肉鸡产品生产加工销售；"
            "据主营及旧版分类定义人工推定为C01食品加工业，非报告直接列明的代码。"
            "仅细化原C0，截止2007年4月26日资产交割前一天，不延用至煤炭业务期间。"
            "交割日期核对：https://static.cninfo.com.cn/finalpage/2007-08-17/30838419.PDF"
        ),
    },
    {
        "Stkcd": "000809",
        "start": "2010-11-15",
        "end": "2010-11-15",
        "industryGroup": "C81",
        "onlyIfIndustryCode": "C8",
        "source": "https://epaper.cs.com.cn/zgzqb/images/2010-11/15/B004/ZQBXP0041115C.pdf",
        "note": (
            "中汇医药2010年11月15日公告明确当日启用医药、生物制品分类；"
            "锐思2001版行业记录将C81生效日记为次日。"
            "2009年报明确主营为中药材及中成药加工业，对应旧版C8110，归并为C81。"
            "年报：https://static.cninfo.com.cn/finalpage/2010-03-24/57721726.PDF"
            "仅修正当天C8记录，不提前其他公司或其他日期的锐思生效日。"
        ),
    },
]


def find_root() -> Path:
    """从脚本位置向上寻找项目根目录，项目移动后无需修改绝对路径。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "AGENTS.md").is_file() and (parent / "中间结果").is_dir():
            return parent
    raise FileNotFoundError("找不到包含AGENTS.md和中间结果的项目根目录。")


def load_module(path: Path, name: str):
    """只读加载模块，阻止导入既有脚本时在其目录产生字节码缓存。"""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> dict:
    """按指定阶段准备独立结果，不调用旧脚本的运行入口。"""
    if RUN_STAGE not in {"all", "rank", "panel"}:
        raise ValueError("运行阶段应为all（全部）、rank（排名）或panel（面板）。")
    if START_YEAR > END_YEAR:
        raise ValueError("起始年份不能晚于结束年份。")
    if not TOP_NS or any(
        isinstance(number, bool) or not isinstance(number, int) or number < 1
        for number in TOP_NS
    ):
        raise ValueError("行业名额必须为非空的正整数序列，例如(4, 5)。")
    if len(set(TOP_NS)) != len(TOP_NS):
        raise ValueError("行业名额不能重复。")
    if not isinstance(CHUNK_SIZE, int) or CHUNK_SIZE < 1:
        raise ValueError("分块读取行数必须为正整数。")

    root = find_root()
    module_dir = Path(__file__).resolve().parent / "公共模块"
    cache_dir = root / "work/行业排名提示检验"
    output_dir = root / "中间结果/04_合并数据/行业排名提示检验"
    ranking = load_module(module_dir / "行业排名.py", "industry_top_rank_helpers")

    print(f"本次事件年份：{START_YEAR}—{END_YEAR}。", flush=True)
    print(f"每行业入选名额：{TOP_NS}；提示方包含金融公司。", flush=True)
    print(f"排名及核对缓存：{cache_dir}", flush=True)
    print(f"独立面板输出：{output_dir}", flush=True)

    # 二、构造排名，或在仅准备面板时校验并读取已有完工缓存。
    if RUN_STAGE in {"all", "rank"}:
        rank_cache = ranking.build_rank_cache(
            root=root,
            cache_dir=cache_dir,
            start_year=START_YEAR,
            end_year=END_YEAR,
            chunk_size=CHUNK_SIZE,
            overrides=INDUSTRY_OVERRIDES,
        )
    else:
        rank_cache = ranking.load_rank_cache(
            root=root,
            cache_dir=cache_dir,
            start_year=START_YEAR,
            end_year=END_YEAR,
            overrides=INDUSTRY_OVERRIDES,
        )

    result = {"ranking": rank_cache}
    if RUN_STAGE == "rank":
        print("行业排名已准备完成；需要面板时把运行阶段改为panel（面板）。", flush=True)
        return result

    # 三、同一张排名表分别产生前4和前5面板，不按公司对重复计算名次。
    panels = load_module(module_dir / "构建行业提示面板.py", "industry_top_panel_helpers")
    result["panels"] = panels.build_panels(
        root=root,
        rank_cache=rank_cache,
        output_dir=output_dir,
        start_year=START_YEAR,
        end_year=END_YEAR,
        top_ns=TOP_NS,
        chunk_size=CHUNK_SIZE,
    )

    # 四、仅在用户实际运行后，在本实验目录记录输出位置。
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "本次运行结果.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=str)
    print(f"准备完成，结果索引：{summary_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
