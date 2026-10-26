#!/usr/bin/env python3
"""MMM 复现项目的统一任务调度入口。

这个脚本只负责组织现有脚本，不复制各脚本内部的计算逻辑。其主要功能是：

1. 按任务依赖关系确定执行顺序；
2. 默认只预览计划，显式加入 ``--run`` 后才实际执行；
3. 运行前检查脚本、外部输入和当前尚未接通的任务；
4. 每个任务使用独立 Python 进程，任务失败后立即停止；
5. 同时检查子进程退出状态和预期输出，避免误用旧文件；
6. 输出及其已登记来源均未更新时，默认复用现有结果；
7. 允许只运行一个或几个下游任务，以复用人工确认过的现有上游结果。

常用示例
--------

预览模式公司最终面板的完整依赖链：

    python code/00_项目总控.py --target pattern_panel

实际执行基础清洗链：

    python code/00_项目总控.py --target clean --run

实际执行变量构造链，并强制重算已经存在的输出：

    python code/00_项目总控.py --target variables --run --force

只运行已经具备输入的回归脚本，不自动展开上游依赖：

    python code/00_项目总控.py --only regression_main --run

查看全部任务及其接入状态：

    python code/00_项目总控.py --list

当前任务清单是项目接口的集中登记处。以后新增或完成一个单独脚本时，
只需在 ``build_tasks`` 中登记脚本、依赖、输入和输出，再把它加入适当目标。
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence


class TaskStatus(str, Enum):
    """任务当前是否允许由总控脚本启动。"""

    READY = "ready"
    BLOCKED = "blocked"
    PENDING = "pending"


STATUS_LABELS = {
    TaskStatus.READY: "可执行",
    TaskStatus.BLOCKED: "接口待对齐",
    TaskStatus.PENDING: "脚本待完成",
}


@dataclass(frozen=True)
class Task:
    """一个可独立启动的项目任务。路径均相对于项目根目录。"""

    key: str
    name: str
    stage: str
    script: str
    deps: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    status: TaskStatus = TaskStatus.READY
    note: str = ""


@dataclass(frozen=True)
class Target:
    """一个面向使用目的的任务集合。"""

    name: str
    tasks: tuple[str, ...]
    description: str


class PipelineError(RuntimeError):
    """总控脚本发现无法安全继续的情况。"""


def find_project_root() -> Path:
    """从当前总控脚本向上寻找项目根目录。"""

    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "code").is_dir() and (candidate / "input").is_dir():
            return candidate
    raise FileNotFoundError("无法找到同时包含 code 和 input 的项目根目录。")


def build_tasks() -> dict[str, Task]:
    """集中登记生产任务、依赖和当前接口状态。"""

    task_list = [
        # ------------------------------------------------------------------
        # 00 原始数据合并
        # ------------------------------------------------------------------
        Task(
            key="merge_daily",
            name="合并 CSMAR 个股日度数据",
            stage="00 原始数据合并",
            script="code/00_合并原始数据/合并日度（升级测试）数据.py",
            inputs=(
                "input/CSMAR/个股日度收益率（升级测试）1995-01-01至于1999-12-31",
            ),
            outputs=(
                "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv",
            ),
            note="源目录名称沿用当前单脚本配置；若下载目录调整，应同步修改该单脚本和此登记。",
        ),
        Task(
            key="merge_share_history",
            name="合并前复权股票历史日行情",
            stage="00 原始数据合并",
            script="code/00_合并原始数据/合并股票历史日行情信息表_前复权.py",
            inputs=(
                "input/CSMAR/股票历史日行情信息表(前复权)1996-01-01 至 2025-12-31",
            ),
            outputs=(
                "中间结果/00_合并原始数据/CSMAR_股票历史日行情信息表_前复权_1996_2025_合并结果.csv",
            ),
        ),
        Task(
            key="merge_forecasts",
            name="合并分析师预测指标文件",
            stage="00 原始数据合并",
            script="code/00_合并原始数据/合并分析师预测指标文件.py",
            inputs=(
                "input/CSMAR/分析师预测指标文件（2001-01-01至2025-12-31）",
            ),
            outputs=(
                "中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv",
            ),
        ),
        Task(
            key="merge_oib",
            name="合并个股买卖不平衡指标",
            stage="00 原始数据合并",
            script="code/00_合并原始数据/合并CSMAR个股买卖不平衡指标表.py",
            inputs=(
                "input/CSMAR/个股买卖不平衡指标表（日）2003-01-02至2026-08-31",
            ),
            outputs=(
                "中间结果/00_合并原始数据/CSMAR_个股买卖不平衡指标表_日_合并结果.csv",
            ),
        ),
        Task(
            key="merge_news",
            name="合并 CNRDS 报刊财经新闻统计",
            stage="00 原始数据合并",
            script="code/00_合并原始数据/合并CNRDS_报刊财经新闻量化统计数据.py",
            inputs=(
                "input/CNRDS/上市公司新闻舆情/财经新闻-CFND/报刊财经新闻/CFND_报刊财经新闻量化统计_ALL",
            ),
            outputs=(
                "中间结果/00_合并原始数据/报刊财经新闻量化统计_1994_2025.csv",
            ),
            note="原报刊来源保留给历史新闻特征；回归新闻控制使用数行者分段控制表。",
        ),
        Task(
            key="merge_announcements",
            name="合并公司公告事件表",
            stage="00 原始数据合并",
            script="code/00_合并原始数据/合并公司公告事件表.py",
            inputs=(
                "input/CSMAR/公司公告事件表（2002-03-07至2005-12-31）",
            ),
            outputs=(
                "中间结果/00_合并原始数据/CSMAR_公司公告事件表_2002_2025_合并结果.csv",
            ),
        ),
        Task(
            key="merge_top_mover_source",
            name="合并市场交易异常波动表",
            stage="00 原始数据合并",
            script="code/00_合并原始数据/合并CSMAR市场交易异常波动表_龙虎榜.py",
            inputs=(
                "input/CSMAR/市场交易异常波动表(龙虎榜)2002-08-02至2026-05-25",
            ),
            outputs=(
                "中间结果/00_合并原始数据/CSMAR_市场交易异常波动表_龙虎榜_2002-08-02_2026-05-25_合并结果.csv",
            ),
        ),
        # ------------------------------------------------------------------
        # 01 基础清洗
        # ------------------------------------------------------------------
        Task(
            key="clean_samples",
            name="构造基础样本池、窗口样本池和行业面板",
            stage="01 基础清洗",
            script="code/01_清洗数据/基础样本池与窗口样本池清洗.py",
            deps=("merge_daily",),
            inputs=(
                "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv",
                "input/CSMAR/上市公司行业分类表1990年12月10日—2026年4月20日/STK_INDUSTRYCLASS.csv",
                "input/RESSET/RESSET 股票_行业变更历史/RESSET_INDHIS_1.csv",
                "input/CSMAR/公司文件(1990-12-10至2026-08-21)/TRD_Co.csv",
                "input/CSMAR/日历文件(1990-12-19至2026-08-20)/CER_Cale.csv",
            ),
            outputs=(
                "中间结果/01_清洗后样本/市场交易日历_td.csv",
                "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv",
                "中间结果/01_清洗后样本/窗口变量目标日资格索引_基础清洗含停牌填充_td.csv",
                "中间结果/01_清洗后样本/完整个股交易日面板索引_td.csv",
                "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv",
                "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板.csv",
                "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv",
                "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺明细.csv",
            ),
        ),
        # ------------------------------------------------------------------
        # 02 变量构造
        # ------------------------------------------------------------------
        Task(
            key="dgtw",
            name="构造 DGTW 特征调整收益率",
            stage="02 变量构造",
            script="code/02_构建变量/主回归/1特征调整收益率计算/00_DGTW全流程.py",
            deps=("clean_samples",),
            inputs=(
                "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv",
                "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv",
                "input/CSMAR/资产负债表1990-12-31至2026-06-30/FS_Combas.csv",
            ),
            outputs=(
                "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW形成期组合标签.csv",
                "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW组合基准收益.csv",
                "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW特征调整收益率.csv",
            ),
        ),
        Task(
            key="oib_avol",
            name="构造订单失衡和异常成交量",
            stage="02 变量构造",
            script="code/02_构建变量/交易机制/2交易行为OIB、Avol/OIB(运行更快).py",
            deps=("merge_oib", "merge_daily", "clean_samples"),
            inputs=(
                "中间结果/00_合并原始数据/CSMAR_个股买卖不平衡指标表_日_合并结果.csv",
                "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv",
                "中间结果/01_清洗后样本/完整个股交易日面板索引_td.csv",
                "中间结果/01_清洗后样本/窗口变量目标日资格索引_基础清洗含停牌填充_td.csv",
                "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv",
            ),
            outputs=(
                "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv",
                "中间结果/02_变量构造结果/2交易行为OIB、Avol/Avol_CSMAR_prep.csv",
            ),
        ),
        Task(
            key="news_aes_aev",
            name="构造原报刊口径历史新闻指标",
            stage="02 变量构造",
            script="code/02_构建变量/主回归/4媒体AES、AEV/计算AES和AEV.py",
            deps=("merge_news", "merge_daily"),
            inputs=(
                "中间结果/00_合并原始数据/报刊财经新闻量化统计_1994_2025.csv",
                "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv",
            ),
            outputs=(
                "中间结果/02_变量构造结果/3报刊新闻_AES_AEV/报刊新闻_AES_AEV.csv",
                "中间结果/02_变量构造结果/3报刊新闻_AES_AEV/报刊新闻_AES_AEV_真实交易日控制变量.csv",
            ),
            note="自然日表继续供历史高新闻覆盖机制使用；正式回归改读数行者分段控制表。",
        ),
        Task(
            key="news_controls_datago",
            name="拼接数行者报刊与网络新闻控制变量",
            stage="02 变量构造",
            script="code/02_构建变量/主回归/4媒体AES、AEV/拼接DATAGO新闻控制变量.py",
            inputs=(
                "中间结果/02_变量构造结果/3报刊新闻_AES_AEV/DATAGO_报刊新闻_AES_AEV_真实交易日控制变量.csv",
                "中间结果/02_变量构造结果/3网络新闻_AES_AEV/DATAGO_网络新闻_AES_AEV_真实交易日控制变量.csv",
            ),
            outputs=(
                "中间结果/02_变量构造结果/3分段新闻_AES_AEV/DATAGO_分段新闻_真实交易日控制变量.csv",
            ),
            note="读取两份已用完整历史计算的数行者控制表；2010年前取报刊，此后取网络。更新原始新闻后应先更新两来源控制表。",
        ),
        Task(
            key="sue",
            name="构造三类标准化盈余惊喜",
            stage="02 变量构造",
            script="code/02_构建变量/心理机制/4PEAD/sue的计算.py",
            deps=("merge_forecasts", "merge_share_history", "merge_daily"),
            inputs=(
                "中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv",
                "中间结果/00_合并原始数据/CSMAR_股票历史日行情信息表_前复权_1996_2025_合并结果.csv",
                "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv",
                "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv",
                "input/CSMAR/日历文件(1990-12-19至2026-08-20)/CER_Cale.csv",
            ),
            outputs=(
                "中间结果/02_变量构造结果/4sue/sue.csv",
            ),
        ),
        Task(
            key="performance_forecast_controls",
            name="构造业绩预告与快报控制变量",
            stage="02 变量构造",
            script="code/02_构建变量/主回归/3公司公告事件/业绩预告快报控制变量_公告日期口径.py",
            deps=("clean_samples",),
            inputs=(
                "input/CSMAR/业绩预告&业绩快报",
                "中间结果/01_清洗后样本/市场交易日历_td.csv",
            ),
            outputs=(
                "中间结果/02_变量构造结果/5公司公告事件/业绩预告快报控制变量_公告日期口径.csv",
            ),
        ),
        Task(
            key="other_event_controls",
            name="构造专题研究事件控制变量",
            stage="02 变量构造",
            script="code/02_构建变量/主回归/3公司公告事件/专题研究事件控制变量_公告日期口径.py",
            deps=("clean_samples",),
            inputs=(
                "input/CSMAR/专题研究系列-事件研究",
                "中间结果/01_清洗后样本/市场交易日历_td.csv",
            ),
            outputs=(
                "中间结果/02_变量构造结果/5公司公告事件/专题研究事件控制变量_公告日期口径.csv",
                "中间结果/02_变量构造结果/5公司公告事件/专题研究事件控制变量_首次公告日期口径.csv",
            ),
        ),
        Task(
            key="pattern_firm_qtr",
            name="识别公司—财报季度模式",
            stage="02 变量构造",
            script="code/02_构建变量/主回归/2识别pattern_firms/build_pattern_firm_qtr.py",
            inputs=(
                "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv",
                "input/CSMAR/公司文件(1990-12-10至2026-08-21)/TRD_Co.csv",
                "input/CSMAR/日历文件(1990-12-19至2026-08-20)/CER_Cale.csv",
            ),
            outputs=(
                "中间结果/02_变量构造结果/6PatternFirmQtr/PatternFirmQtr.csv",
            ),
        ),
        Task(
            key="top_mover_csmar",
            name="根据龙虎榜识别极端涨跌股票",
            stage="02 变量构造",
            script="code/02_构建变量/心理机制/7TopMover/TopMover.py",
            deps=("merge_top_mover_source",),
            inputs=(
                "中间结果/00_合并原始数据/CSMAR_市场交易异常波动表_龙虎榜_2002-08-02_2026-05-25_合并结果.csv",
                "input/CSMAR/公司文件(1990-12-10至2026-08-21)/TRD_Co.csv",
            ),
            outputs=(
                "中间结果/02_变量构造结果/7TopMover/TopMover.csv",
            ),
        ),
        Task(
            key="top_mover_ranked",
            name="根据绝对收益排名构造两种 TopMover 口径",
            stage="02 变量构造",
            script="code/02_构建变量/心理机制/7TopMover/TopMover1.py",
            deps=("clean_samples",),
            inputs=(
                "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv",
            ),
            outputs=(
                "中间结果/02_变量构造结果/7TopMover/TopMover1_top100.csv",
                "中间结果/02_变量构造结果/7TopMover/TopMover1_top10pct.csv",
            ),
            note="同时输出前100名和前10%两种口径；当前模式公司记忆矩阵读取前100名口径。",
        ),
        Task(
            key="market_cap_groups",
            name="构造月度市值分组和断点",
            stage="02 变量构造",
            script="code/02_构建变量/主回归/月度市值分组.py",
            inputs=(
                "input/CSMAR/月个股回报率文件2000-01至2025-12/TRD_Mnth.csv",
            ),
            outputs=(
                "中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv",
            ),
        ),
        # ------------------------------------------------------------------
        # 03 记忆矩阵与共同分析师覆盖
        # ------------------------------------------------------------------
        Task(
            key="pattern_memory",
            name="构造模式公司记忆矩阵",
            stage="03 记忆矩阵",
            script="code/03_记忆矩阵&心理机制变量分组/pattern_firms记忆矩阵/只含pattern_firms（手动确认）.py",
            deps=(
                "clean_samples",
                "pattern_firm_qtr",
                "news_aes_aev",
                "oib_avol",
                "sue",
                "top_mover_ranked",
            ),
            inputs=(
                "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv",
                "中间结果/02_变量构造结果/6PatternFirmQtr/PatternFirmQtr.csv",
                "中间结果/02_变量构造结果/3报刊新闻_AES_AEV/报刊新闻_AES_AEV.csv",
                "中间结果/02_变量构造结果/2交易行为OIB、Avol/Avol_CSMAR_prep.csv",
                "中间结果/02_变量构造结果/4sue/sue.csv",
                "中间结果/02_变量构造结果/7TopMover/TopMover1_top100.csv",
                "中间结果/01_清洗后样本/市场交易日历_td.csv",
            ),
            outputs=(
                "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv",
            ),
            args=("--root", "{root}"),
            note="使用当前手动确认版，异常成交量读取独立表，极端涨跌股票沿用前100名口径。",
        ),
        Task(
            key="full_memory",
            name="构造全样本记忆矩阵",
            stage="03 记忆矩阵",
            script="code/03_记忆矩阵&心理机制变量分组/full_sample记忆矩阵/Full（封装优化版）.py",
            inputs=(
                "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv",
                "input/CSMAR/公司文件(1990-12-10至2026-08-21)/TRD_Co.csv",
            ),
            outputs=(
                "中间结果/03_记忆矩阵/MemoryMatrixFull(封装优化版).csv",
            ),
            args=("--root", "{root}"),
        ),
        Task(
            key="analyst_overlap_pattern",
            name="识别模式样本共同分析师覆盖",
            stage="03 记忆矩阵",
            script="code/03_记忆矩阵&心理机制变量分组/识别分析师共同覆盖/识别分析师共同覆盖（日度Annodt合并检验）.py",
            deps=("merge_forecasts", "pattern_memory"),
            inputs=(
                "中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv",
                "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv",
            ),
            outputs=(
                "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后/拆开AnanmID/Pattern样本_分析师共同覆盖_滚动365天.csv",
            ),
            args=(
                "--root",
                "{root}",
                "--samples",
                "pattern",
                "--lookback-days",
                "365",
                "--id-modes",
                "split",
            ),
        ),
        Task(
            key="analyst_overlap_full",
            name="识别全样本共同分析师覆盖",
            stage="03 记忆矩阵",
            script="code/03_记忆矩阵&心理机制变量分组/识别分析师共同覆盖/识别分析师共同覆盖（日度Annodt合并检验）.py",
            deps=("merge_forecasts", "full_memory"),
            inputs=(
                "中间结果/00_合并原始数据/CSMAR_分析师预测指标文件_合并结果.csv",
                "中间结果/03_记忆矩阵/MemoryMatrixFull(封装优化版).csv",
            ),
            outputs=(
                "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后/拆开AnanmID/Full样本_分析师共同覆盖_滚动365天.csv",
            ),
            args=(
                "--root",
                "{root}",
                "--samples",
                "full",
                "--lookback-days",
                "365",
                "--id-modes",
                "split",
            ),
        ),
        Task(
            key="memory_111_110",
            name="构造 111 与 110 两类记忆矩阵",
            stage="03 记忆矩阵",
            script="code/03_记忆矩阵&心理机制变量分组/full_sample记忆矩阵/构造111_110记忆矩阵.py",
            deps=("pattern_firm_qtr",),
            inputs=(
                "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv",
                "中间结果/02_变量构造结果/6PatternFirmQtr/PatternFirmQtr.csv",
            ),
            outputs=(
                "中间结果/03_记忆矩阵/MemoryMatrix_111.csv",
                "中间结果/03_记忆矩阵/MemoryMatrix_110.csv",
            ),
            args=("--root", "{root}"),
        ),
        # ------------------------------------------------------------------
        # 04 最终样本合并
        # ------------------------------------------------------------------
        Task(
            key="merge_pattern_panel",
            name="合并模式公司最终回归面板",
            stage="04 最终样本合并",
            script="code/04_合并数据/合并pattern_firms的数据.py",
            deps=(
                "pattern_memory",
                "analyst_overlap_pattern",
                "dgtw",
                "oib_avol",
                "news_controls_datago",
                "sue",
                "market_cap_groups",
                "performance_forecast_controls",
                "other_event_controls",
            ),
            inputs=(
                "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv",
                "中间结果/01_清洗后样本/市场交易日历_td.csv",
                "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv",
                "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW特征调整收益率.csv",
                "中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv",
                "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv",
                "中间结果/02_变量构造结果/4sue/sue.csv",
                "input/CSMAR/供应链网络关系指标表2001-12-31至2025-12-31/SC_NetworkRelationsIndex.csv",
                "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv",
                "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后/拆开AnanmID/Pattern样本_分析师共同覆盖_滚动365天.csv",
                "中间结果/02_变量构造结果/3分段新闻_AES_AEV/DATAGO_分段新闻_真实交易日控制变量.csv",
                "中间结果/02_变量构造结果/5公司公告事件/业绩预告快报控制变量_公告日期口径.csv",
                "中间结果/02_变量构造结果/5公司公告事件/专题研究事件控制变量_公告日期口径.csv",
            ),
            outputs=(
                "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
            ),
            note="新闻控制采用数行者分段表：2010年前报刊、2010年起网络；历史机制来源独立保留。",
        ),
        Task(
            key="merge_full_panel",
            name="分块合并全样本最终回归面板",
            stage="04 最终样本合并",
            script="code/04_合并数据/合并全样本数据(分块).py",
            deps=(
                "full_memory",
                "analyst_overlap_full",
                "dgtw",
                "oib_avol",
                "news_controls_datago",
                "market_cap_groups",
                "performance_forecast_controls",
                "other_event_controls",
            ),
            inputs=(
                "中间结果/03_记忆矩阵/MemoryMatrixFull(封装优化版).csv",
                "中间结果/01_清洗后样本/市场交易日历_td.csv",
                "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv",
                "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW特征调整收益率.csv",
                "中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv",
                "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv",
                "input/CSMAR/供应链网络关系指标表2001-12-31至2025-12-31/SC_NetworkRelationsIndex.csv",
                "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv",
                "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后/拆开AnanmID/Full样本_分析师共同覆盖_滚动365天.csv",
                "中间结果/02_变量构造结果/3分段新闻_AES_AEV/DATAGO_分段新闻_真实交易日控制变量.csv",
                "中间结果/02_变量构造结果/5公司公告事件/业绩预告快报控制变量_公告日期口径.csv",
                "中间结果/02_变量构造结果/5公司公告事件/专题研究事件控制变量_公告日期口径.csv",
            ),
            outputs=(
                "中间结果/04_合并数据/Full样本_最终回归面板.csv",
            ),
            note="新闻控制采用数行者分段表；全样本共同分析师覆盖仍由其前置任务检查。",
        ),
        Task(
            key="merge_pattern_daily_panel",
            name="合并模式公司单日收益基础面板",
            stage="04 最终样本合并",
            script="code/04_合并数据/合并pattern_firms的数据_单日特征调整收益.py",
            deps=(
                "pattern_memory",
                "analyst_overlap_pattern",
                "oib_avol",
                "news_controls_datago",
                "sue",
                "market_cap_groups",
                "performance_forecast_controls",
                "other_event_controls",
            ),
            inputs=(
                "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv",
                "中间结果/01_清洗后样本/市场交易日历_td.csv",
                "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv",
                "中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv",
                "中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv",
                "中间结果/02_变量构造结果/4sue/sue.csv",
                "input/CSMAR/供应链网络关系指标表2001-12-31至2025-12-31/SC_NetworkRelationsIndex.csv",
                "input/CSMAR/年、中、季报基本情况文件1990-12-31至2026-06-30/IAR_Rept.csv",
                "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后/拆开AnanmID/Pattern样本_分析师共同覆盖_滚动365天.csv",
                "中间结果/02_变量构造结果/3分段新闻_AES_AEV/DATAGO_分段新闻_真实交易日控制变量.csv",
                "中间结果/02_变量构造结果/5公司公告事件/业绩预告快报控制变量_公告日期口径.csv",
                "中间结果/02_变量构造结果/5公司公告事件/专题研究事件控制变量_公告日期口径.csv",
            ),
            outputs=(
                "中间结果/04_合并数据/Pattern样本_单日收益回归基础面板.csv",
            ),
            note="默认只生成单日基础面板，独立读取上游数据，不读取现有最终回归面板。",
        ),
        Task(
            key="cue_events_111_110",
            name="把 111 与 110 记忆矩阵转成提示事件",
            stage="04 最终样本合并",
            script="code/04_合并数据/构造111_110提示事件.py",
            deps=(
                "memory_111_110",
                "clean_samples",
                "market_cap_groups",
                "analyst_overlap_full",
            ),
            inputs=(
                "中间结果/03_记忆矩阵/MemoryMatrix_111.csv",
                "中间结果/03_记忆矩阵/MemoryMatrix_110.csv",
                "中间结果/01_清洗后样本/市场交易日历_td.csv",
                "中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv",
                "中间结果/03_记忆矩阵/分析师共同覆盖_参考文献后/拆开AnanmID/Full样本_分析师共同覆盖_滚动365天.csv",
            ),
            outputs=(
                "中间结果/04_合并数据/CueEvents_111.csv",
                "中间结果/04_合并数据/CueEvents_110.csv",
            ),
            args=("--root", "{root}"),
        ),
        # ------------------------------------------------------------------
        # 05、06 扩展分支
        # ------------------------------------------------------------------
        Task(
            key="alternative_cues",
            name="构造两类替代公告线索",
            stage="05 替代线索",
            script="code/05_替代线索/替代线索.py",
            deps=("pattern_memory", "merge_announcements", "market_cap_groups"),
            inputs=(
                "中间结果/03_记忆矩阵/MemoryMatrixPattern(交易变量用公告后首个交易日).csv",
                "中间结果/02_变量构造结果/公司公告事件/公司公告事件明细_下一交易日口径.csv",
                "中间结果/02_变量构造结果/8月度市值分组/月度市值分组.csv",
            ),
            outputs=(
                "中间结果/05_替代线索/替代线索_两类公告.csv",
            ),
            status=TaskStatus.BLOCKED,
            note="脚本仍读取旧公告明细和无编号月度市值目录，尚未接入本轮两张新控制表。",
        ),
        Task(
            key="daily_factors",
            name="计算交易策略所需日度因子",
            stage="06 交易策略",
            script="code/06_交易策略/计算日度因子.py",
            deps=("clean_samples", "dgtw"),
            inputs=(
                "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv",
                "中间结果/02_变量构造结果/1DGTW特征调整收益率/DGTW形成期组合标签.csv",
            ),
            outputs=(
                "中间结果/06_交易策略/自算日度因子.csv",
            ),
            status=TaskStatus.BLOCKED,
            note="脚本当前仍读取无编号 DGTW 目录。",
        ),
        Task(
            key="trading_strategy",
            name="准备并计算交易策略",
            stage="06 交易策略",
            script="code/06_交易策略/交易策略准备.py",
            deps=("daily_factors", "merge_pattern_panel"),
            status=TaskStatus.PENDING,
            note="交易策略准备脚本目前为空，尚未定义正式输入和输出。",
        ),
        # ------------------------------------------------------------------
        # 07、08 图表和回归
        # ------------------------------------------------------------------
        Task(
            key="cue_summary",
            name="生成提示变量描述统计",
            stage="08 回归与整理",
            script="code/02_构建变量/主回归/2识别pattern_firms/cue的两行描述统计.py",
            deps=("merge_pattern_panel",),
            inputs=(
                "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
            ),
            outputs=(
                "中间结果/04_合并数据/cue_summary_two_rows.csv",
            ),
            note="虽然位于变量构造目录，实际应在最终回归面板生成后运行。",
        ),
        Task(
            key="regression_main",
            name="运行描述统计和表 2 至表 4 基础回归",
            stage="07 实证分析",
            script="code/07_实证分析/运行主线回归.py",
            deps=("merge_pattern_panel", "merge_full_panel", "oib_avol"),
            inputs=(
                # 实际调用的源码参与结果复用判断，避免只修改逐表实现时仍复用旧结果。
                "code/07_实证分析/01_主线图表/表1_描述性统计/估计回归.py",
                "code/07_实证分析/01_主线图表/表2_收益回归/估计回归.py",
                "code/07_实证分析/01_主线图表/表3_散户订单失衡/估计回归.py",
                "code/07_实证分析/01_主线图表/表4_规模异质性/估计回归.py",
                "code/07_实证分析/01_主线图表/附表A2_买卖次数订单失衡/估计回归.py",
                "code/07_实证分析/90_公共模块/任务入口.py",
                "code/07_实证分析/90_公共模块/订单失衡估计.py",
                "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
                "中间结果/04_合并数据/Full样本_最终回归面板.csv",
                "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv",
            ),
            outputs=(
                "图表结果/01_主线图表/表1_描述性统计/table1_描述性统计.png",
                "图表结果/01_主线图表/表2_收益回归/表2_模型汇总.csv",
                "图表结果/01_主线图表/表2_收益回归/表2_系数明细.csv",
                "图表结果/01_主线图表/表3_散户订单失衡/表3_散户订单失衡结果.csv",
                "图表结果/01_主线图表/表3_散户订单失衡/表3_系数明细.csv",
                "图表结果/01_主线图表/附表A2_买卖次数订单失衡/附表A2_OIBnum_买卖次数订单失衡结果.csv",
                "图表结果/01_主线图表/附表A2_买卖次数订单失衡/附表A2_OIBnum_买卖次数订单失衡系数明细.csv",
                "图表结果/01_主线图表/表4_规模异质性/表4_规模异质性结果.csv",
                "图表结果/01_主线图表/表4_规模异质性/表4_规模异质性系数明细.csv",
            ),
            args=("--run",),
            note="依次调用拆分后的逐表估计；原综合脚本归档保留。",
        ),
        Task(
            key="regression_111_110",
            name="运行 111 与 110 提示事件比较回归",
            stage="07 实证分析",
            script="code/07_实证分析/02_扩展分析/其他扩展检验/提示事件111_110比较/表2_111_110比较.py",
            deps=("merge_full_panel", "cue_events_111_110"),
            inputs=(
                "中间结果/04_合并数据/Full样本_最终回归面板.csv",
                "中间结果/04_合并数据/CueEvents_111.csv",
                "中间结果/04_合并数据/CueEvents_110.csv",
            ),
            outputs=(
                "图表结果/02_扩展分析/其他扩展检验/提示事件111_110比较/表2_111_110模型汇总.csv",
                "图表结果/02_扩展分析/其他扩展检验/提示事件111_110比较/表2_111_110系数明细.csv",
            ),
            args=("--root", "{root}"),
        ),
        Task(
            key="table3_date_fe",
            name="运行加入日期固定效应的表 3",
            stage="07 实证分析",
            script="code/07_实证分析/01_主线图表/表3_散户订单失衡/估计回归_日期固定效应.py",
            deps=("merge_pattern_panel", "oib_avol"),
            inputs=(
                "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
                "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv",
            ),
            outputs=(
                "图表结果/01_主线图表/表3_散户订单失衡/表3_散户订单失衡_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/表3_散户订单失衡/表3_散户订单失衡_加入日期固定效应系数明细.csv",
            ),
            note="订单失衡数据读取当前编号目录。",
        ),
        Task(
            key="appendix_a2_a4_date_fe",
            name="运行加入日期固定效应的附表 A.2 至 A.4",
            stage="07 实证分析",
            script="code/07_实证分析/01_主线图表/附表A2-A4_日期固定效应/估计回归.py",
            deps=("merge_pattern_panel", "oib_avol"),
            inputs=(
                "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
                "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv",
            ),
            outputs=(
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/附表A2_OIBnum_买卖次数订单失衡_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/附表A2_OIBnum_买卖次数订单失衡_加入日期固定效应系数明细.csv",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/附表A3_OIBamount_成交金额订单失衡_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/附表A3_OIBamount_成交金额订单失衡_加入日期固定效应系数明细.csv",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/附表A4_OIBorder_成交笔数订单失衡_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/附表A4_OIBorder_成交笔数订单失衡_加入日期固定效应系数明细.csv",
            ),
            note="本项目三种订单失衡口径的日期固定效应扩展，与基础回归生成的附表 A.2 分开保存。独立脚本也会直接绘图，流程中将三张图片统一登记在绘表阶段。",
        ),
        Task(
            key="table4_date_fe",
            name="运行加入日期固定效应的表 4",
            stage="07 实证分析",
            script="code/07_实证分析/01_主线图表/表4_规模异质性/估计回归_日期固定效应.py",
            deps=("merge_pattern_panel",),
            inputs=(
                "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
            ),
            outputs=(
                "图表结果/01_主线图表/表4_规模异质性/表4_规模异质性_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/表4_规模异质性/表4_规模异质性_加入日期固定效应系数明细.csv",
            ),
        ),
        Task(
            key="table5",
            name="历史表5无日期固定效应规格（暂不接入）",
            stage="07 实证分析",
            script="code/07_实证分析/01_主线图表/表5_心理机制/估计回归与绘图.py",
            deps=("merge_pattern_panel",),
            inputs=(
                "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
            ),
            outputs=(
                "图表结果/01_主线图表/表5_心理机制/表5_SUE1_心理机制结果.csv",
                "图表结果/01_主线图表/表5_心理机制/表5_SUE2_心理机制结果.csv",
                "图表结果/01_主线图表/表5_心理机制/表5_SUE3_心理机制结果.csv",
            ),
            status=TaskStatus.BLOCKED,
            note="目前可执行的表5默认加入日期固定效应；历史基准结果保留，避免以日期规格冒充基准。",
        ),
        Task(
            key="table5_date_fe",
            name="运行加入日期固定效应的表 5",
            stage="07 实证分析",
            script="code/07_实证分析/01_主线图表/表5_心理机制/估计回归与绘图.py",
            deps=("merge_pattern_panel",),
            inputs=(
                "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
            ),
            outputs=(
                "图表结果/01_主线图表/表5_心理机制/表5_SUE1_心理机制_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/表5_心理机制/表5_SUE2_心理机制_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/表5_心理机制/表5_SUE3_心理机制_加入日期固定效应结果.csv",
            ),
        ),
        Task(
            key="figure2",
            name="生成收益动态和长期反转图",
            stage="07 实证分析",
            script="code/07_实证分析/01_主线图表/图2_收益动态/加入日期固定效应.py",
            deps=("merge_pattern_panel", "regression_main"),
            inputs=(
                # 实际调用的源码参与结果复用判断，避免只修改逐表实现时仍复用旧结果。
                "code/07_实证分析/90_公共模块/图2图3扩展_日期固定效应回归公共模块.py",
                "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
                "图表结果/01_主线图表/表2_收益回归/表2_模型汇总.csv",
            ),
            outputs=(
                "图表结果/01_主线图表/图2_收益动态/图2_收益动态和长期反转_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/图2_收益动态/figure2_收益动态和长期反转_加入日期固定效应.png",
            ),
        ),
        Task(
            key="figure3",
            name="生成散户订单失衡动态图",
            stage="07 实证分析",
            script="code/07_实证分析/01_主线图表/图3_散户订单不平衡/加入日期固定效应.py",
            deps=("merge_pattern_panel", "oib_avol", "table3_date_fe"),
            inputs=(
                # 实际调用的源码参与结果复用判断，避免只修改逐表实现时仍复用旧结果。
                "code/07_实证分析/90_公共模块/图2图3扩展_日期固定效应回归公共模块.py",
                "中间结果/04_合并数据/Pattern样本_最终回归面板.csv",
                "中间结果/02_变量构造结果/2交易行为OIB、Avol/OIB_CSMAR_prep.csv",
                "图表结果/01_主线图表/表3_散户订单失衡/表3_散户订单失衡_加入日期固定效应结果.csv",
            ),
            outputs=(
                "图表结果/01_主线图表/图3_散户订单不平衡/图3_散户订单不平衡动态_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/图3_散户订单不平衡/figure3_散户订单不平衡动态_加入日期固定效应.png",
            ),
            note="已接入编号后的订单失衡数据目录及逐表结果目录。",
        ),
        Task(
            key="draw_regression_tables",
            name="根据结构化回归结果绘制论文表格",
            stage="07 实证分析",
            script="code/07_实证分析/绘制主线回归表格.py",
            deps=(
                "regression_main",
                "table3_date_fe",
                "appendix_a2_a4_date_fe",
                "table4_date_fe",
            ),
            inputs=(
                # 实际调用的源码参与结果复用判断，避免只修改逐表实现时仍复用旧结果。
                "code/07_实证分析/01_主线图表/表2_收益回归/绘制表格.py",
                "code/07_实证分析/01_主线图表/表3_散户订单失衡/绘制表格.py",
                "code/07_实证分析/01_主线图表/表4_规模异质性/绘制表格.py",
                "code/07_实证分析/01_主线图表/附表A2_买卖次数订单失衡/绘制表格.py",
                "code/07_实证分析/01_主线图表/附表A2-A4_日期固定效应/绘制表格.py",
                "code/07_实证分析/90_公共模块/任务入口.py",
                "code/07_实证分析/90_公共模块/回归表格绘制.py",
                "图表结果/01_主线图表/表2_收益回归/表2_模型汇总.csv",
                "图表结果/01_主线图表/表2_收益回归/表2_系数明细.csv",
                "图表结果/01_主线图表/表3_散户订单失衡/表3_散户订单失衡结果.csv",
                "图表结果/01_主线图表/表3_散户订单失衡/表3_系数明细.csv",
                "图表结果/01_主线图表/表3_散户订单失衡/表3_散户订单失衡_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/表3_散户订单失衡/表3_散户订单失衡_加入日期固定效应系数明细.csv",
                "图表结果/01_主线图表/附表A2_买卖次数订单失衡/附表A2_OIBnum_买卖次数订单失衡结果.csv",
                "图表结果/01_主线图表/附表A2_买卖次数订单失衡/附表A2_OIBnum_买卖次数订单失衡系数明细.csv",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/附表A2_OIBnum_买卖次数订单失衡_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/附表A2_OIBnum_买卖次数订单失衡_加入日期固定效应系数明细.csv",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/附表A3_OIBamount_成交金额订单失衡_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/附表A3_OIBamount_成交金额订单失衡_加入日期固定效应系数明细.csv",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/附表A4_OIBorder_成交笔数订单失衡_加入日期固定效应结果.csv",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/附表A4_OIBorder_成交笔数订单失衡_加入日期固定效应系数明细.csv",
                "图表结果/01_主线图表/表4_规模异质性/表4_规模异质性结果.csv",
                "图表结果/01_主线图表/表4_规模异质性/表4_规模异质性_加入日期固定效应结果.csv",
            ),
            outputs=(
                "图表结果/01_主线图表/表2_收益回归/table2_收益回归_紧凑版.png",
                "图表结果/01_主线图表/表2_收益回归/table2_收益回归_完整系数版_模式公司主样本.png",
                "图表结果/01_主线图表/表2_收益回归/table2_收益回归_完整系数版_模式公司主样本_续页.png",
                "图表结果/01_主线图表/表2_收益回归/table2_收益回归_完整系数版_模式公司主样本.pdf",
                "图表结果/01_主线图表/表2_收益回归/table2_收益回归_完整系数版_完整样本.png",
                "图表结果/01_主线图表/表2_收益回归/table2_收益回归_完整系数版_完整样本_续页.png",
                "图表结果/01_主线图表/表3_散户订单失衡/table3_散户订单失衡_完整系数版.png",
                "图表结果/01_主线图表/表3_散户订单失衡/table3_散户订单失衡_完整系数版_加入日期固定效应.png",
                "图表结果/01_主线图表/附表A2_买卖次数订单失衡/appendix_A2_OIBnum_买卖次数订单失衡_完整系数版.png",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/appendix_A2_OIBnum_买卖次数订单失衡_完整系数版_加入日期固定效应.png",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/appendix_A3_OIBamount_成交金额订单失衡_完整系数版_加入日期固定效应.png",
                "图表结果/01_主线图表/附表A2-A4_日期固定效应/appendix_A4_OIBorder_成交笔数订单失衡_完整系数版_加入日期固定效应.png",
                "图表结果/01_主线图表/表4_规模异质性/table4_规模异质性_t_tp1扩展分析.png",
                "图表结果/01_主线图表/表4_规模异质性/table4_规模异质性_t_tp1扩展分析_加入日期固定效应.png",
            ),
        ),
    ]

    tasks = {task.key: task for task in task_list}
    if len(tasks) != len(task_list):
        raise PipelineError("任务清单中存在重复任务编号。")
    return tasks


def build_targets() -> dict[str, Target]:
    """登记面向研究目的的执行目标。"""

    target_list = [
        Target(
            name="raw_core",
            tasks=(
                "merge_daily",
                "merge_share_history",
                "merge_forecasts",
                "merge_oib",
                "merge_news",
                "merge_announcements",
                "merge_top_mover_source",
            ),
            description="合并主流程使用的原始数据",
        ),
        Target(
            name="clean",
            tasks=("clean_samples",),
            description="合并日度数据并生成基础清洗结果",
        ),
        Target(
            name="variables",
            tasks=(
                "dgtw",
                "oib_avol",
                "news_aes_aev",
                "news_controls_datago",
                "sue",
                "performance_forecast_controls",
                "other_event_controls",
                "pattern_firm_qtr",
                "top_mover_csmar",
                "top_mover_ranked",
                "market_cap_groups",
            ),
            description="生成当前已经完成的主要变量表",
        ),
        Target(
            name="pattern_matrix",
            tasks=("pattern_memory",),
            description="生成模式公司记忆矩阵",
        ),
        Target(
            name="full_matrix",
            tasks=("full_memory",),
            description="生成全样本记忆矩阵",
        ),
        Target(
            name="pattern_panel",
            tasks=("merge_pattern_panel",),
            description="从原始数据开始生成模式公司最终回归面板",
        ),
        Target(
            name="full_panel",
            tasks=("merge_full_panel",),
            description="从原始数据开始生成全样本最终回归面板",
        ),
        Target(
            name="main_panels",
            tasks=("merge_pattern_panel", "merge_full_panel"),
            description="生成模式公司和全样本两张最终回归面板",
        ),
        Target(
            name="comparison_111_110",
            tasks=("regression_111_110",),
            description="生成 111 与 110 提示事件并运行比较回归",
        ),
        Target(
            name="main_results",
            tasks=(
                "cue_summary",
                "regression_main",
                "table3_date_fe",
                "appendix_a2_a4_date_fe",
                "table4_date_fe",
                "table5_date_fe",
                "figure2",
                "figure3",
            ),
            description="运行主要回归、日期固定效应扩展附表、机制检验和动态图",
        ),
        Target(
            name="report_tables",
            tasks=("draw_regression_tables",),
            description="从结构化回归结果生成论文表格图片",
        ),
        Target(
            name="extensions",
            tasks=(
                "alternative_cues",
                "regression_111_110",
                "appendix_a2_a4_date_fe",
                "daily_factors",
                "trading_strategy",
            ),
            description="替代线索、111/110、三种订单失衡日期固定效应附表和交易策略扩展",
        ),
        Target(
            name="all",
            tasks=(
                "draw_regression_tables",
                "figure2",
                "figure3",
                "regression_111_110",
                "alternative_cues",
                "trading_strategy",
            ),
            description="当前登记的主流程与扩展流程",
        ),
    ]
    return {target.name: target for target in target_list}


def resolve_path(root: Path, relative_path: str) -> Path:
    """把任务清单中的相对路径解析到项目根目录。"""

    return root / Path(relative_path)


def format_args(args: Sequence[str], root: Path) -> list[str]:
    """替换任务参数中的项目根目录占位符。"""

    return [arg.format(root=str(root)) for arg in args]


def validate_registry(tasks: dict[str, Task], targets: dict[str, Target]) -> None:
    """在生成执行计划前检查任务清单内部的一致性。"""

    for task in tasks.values():
        unknown_deps = [dep for dep in task.deps if dep not in tasks]
        if unknown_deps:
            raise PipelineError(
                f"任务 {task.key} 引用了不存在的前置任务：{unknown_deps}"
            )

    for target in targets.values():
        unknown_tasks = [key for key in target.tasks if key not in tasks]
        if unknown_tasks:
            raise PipelineError(
                f"目标 {target.name} 引用了不存在的任务：{unknown_tasks}"
            )

    output_owner: dict[str, str] = {}
    for task in tasks.values():
        for output in task.outputs:
            previous = output_owner.get(output)
            if previous is not None:
                raise PipelineError(
                    f"输出 {output} 同时登记给了 {previous} 和 {task.key}。"
                )
            output_owner[output] = task.key

    # 先检查循环依赖，再递归计算每个任务的完整依赖集合。
    topological_order(tuple(tasks), tasks, include_dependencies=True)

    dependency_cache: dict[str, set[str]] = {}

    def all_dependencies(task_key: str) -> set[str]:
        cached = dependency_cache.get(task_key)
        if cached is not None:
            return cached
        result: set[str] = set()
        for dep in tasks[task_key].deps:
            result.add(dep)
            result.update(all_dependencies(dep))
        dependency_cache[task_key] = result
        return result

    for task in tasks.values():
        declared_dependencies = all_dependencies(task.key)
        for input_path in task.inputs:
            producer = output_owner.get(input_path)
            if producer is not None and producer not in declared_dependencies:
                raise PipelineError(
                    f"任务 {task.key} 读取 {producer} 的输出 {input_path}，"
                    "但没有声明相应依赖。"
                )


def topological_order(
    selected: Sequence[str],
    tasks: dict[str, Task],
    include_dependencies: bool,
) -> list[str]:
    """稳定拓扑排序；同一依赖层按任务登记顺序执行。"""

    selected_set = set(selected)
    included = set(selected_set)

    if include_dependencies:
        stack = list(selected)
        while stack:
            key = stack.pop()
            for dep in tasks[key].deps:
                if dep not in included:
                    included.add(dep)
                    stack.append(dep)

    registry_order = {key: index for index, key in enumerate(tasks)}
    indegree = {key: 0 for key in included}
    downstream: dict[str, list[str]] = {key: [] for key in included}

    for key in included:
        for dep in tasks[key].deps:
            if dep not in included:
                continue
            indegree[key] += 1
            downstream[dep].append(key)

    ready = sorted(
        (key for key, degree in indegree.items() if degree == 0),
        key=registry_order.__getitem__,
    )
    ordered: list[str] = []

    while ready:
        key = ready.pop(0)
        ordered.append(key)
        for child in sorted(
            downstream[key],
            key=registry_order.__getitem__,
        ):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort(key=registry_order.__getitem__)

    if len(ordered) != len(included):
        cycle_nodes = [key for key, degree in indegree.items() if degree > 0]
        cycle_nodes.sort(key=registry_order.__getitem__)
        raise PipelineError(
            "任务依赖存在循环，涉及：" + "、".join(cycle_nodes)
        )
    return ordered


def required_inputs(task: Task, tasks: dict[str, Task]) -> tuple[str, ...]:
    """返回任务明确声明会读取的输入，并保持登记顺序。"""

    # 前置任务可能生成多份互不相关的输出，不能把它们全部误判为当前任务输入。
    # 参数 tasks 保留在函数接口中，便于以后增加任务级输入解析规则。
    _ = tasks
    return tuple(dict.fromkeys(task.inputs))


def path_mtime_ns(path: Path) -> int:
    """返回文件时间；目录则递归取其中最新条目的修改时间。"""

    newest = path.stat().st_mtime_ns
    if not path.is_dir():
        return newest

    for child in path.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime_ns)
        except FileNotFoundError:
            # 预览期间若另一个进程恰好移动临时文件，不因此中断整个检查。
            continue
    return newest


def up_to_date(
    task: Task,
    tasks: dict[str, Task],
    root: Path,
) -> tuple[bool, str]:
    """根据已登记输入、前置输出和脚本时间判断能否复用结果。"""

    if not task.outputs:
        return False, "任务未登记正式输出"

    output_paths = [resolve_path(root, path) for path in task.outputs]
    missing_outputs = [path for path in output_paths if not path.exists()]
    if missing_outputs:
        return False, "存在尚未生成的输出"

    source_paths = [resolve_path(root, task.script)]
    source_paths.extend(
        resolve_path(root, path) for path in required_inputs(task, tasks)
    )
    if any(not path.exists() for path in source_paths):
        return False, "存在缺失来源，不能确认现有输出有效"

    newest_source = max(path_mtime_ns(path) for path in source_paths)
    oldest_output = min(path_mtime_ns(path) for path in output_paths)
    if oldest_output >= newest_source:
        return True, "全部输出存在，且不早于已登记来源"
    return False, "至少一个已登记来源比输出更新"


def selected_output_owners(
    ordered_keys: Sequence[str],
    tasks: dict[str, Task],
) -> dict[str, str]:
    """建立本次计划中输出路径到生产任务的映射。"""

    return {
        output: key
        for key in ordered_keys
        for output in tasks[key].outputs
    }


def preflight_issues(
    ordered_keys: Sequence[str],
    tasks: dict[str, Task],
    root: Path,
) -> list[str]:
    """列出在开始任何计算前即可发现的阻塞问题。"""

    issues: list[str] = []
    produced_in_plan = selected_output_owners(ordered_keys, tasks)

    for key in ordered_keys:
        task = tasks[key]
        script_path = resolve_path(root, task.script)

        if task.status is not TaskStatus.READY:
            issues.append(
                f"[{key}] {STATUS_LABELS[task.status]}：{task.note or '未提供说明'}"
            )

        if not script_path.is_file():
            issues.append(f"[{key}] 找不到脚本：{script_path}")

        for input_path_text in required_inputs(task, tasks):
            input_path = resolve_path(root, input_path_text)
            producer = produced_in_plan.get(input_path_text)
            if not input_path.exists() and producer is None:
                issues.append(f"[{key}] 缺少外部输入：{input_path}")

    return list(dict.fromkeys(issues))


def plan_decisions(
    ordered_keys: Sequence[str],
    tasks: dict[str, Task],
    root: Path,
    force: bool,
) -> dict[str, tuple[str, str]]:
    """生成完整计划，并把上游重算或阻塞状态向下游传播。"""

    decisions: dict[str, tuple[str, str]] = {}
    selected = set(ordered_keys)

    for key in ordered_keys:
        task = tasks[key]
        if task.status is TaskStatus.BLOCKED:
            decisions[key] = ("阻塞", task.note)
            continue
        if task.status is TaskStatus.PENDING:
            decisions[key] = ("待完成", task.note)
            continue

        unavailable_deps = [
            dep
            for dep in task.deps
            if dep in selected
            and decisions.get(dep, ("", ""))[0] in {"阻塞", "待完成"}
        ]
        if unavailable_deps:
            decisions[key] = (
                "阻塞",
                "前置任务尚不可执行：" + "、".join(unavailable_deps),
            )
            continue

        if force:
            decisions[key] = ("运行", "已要求强制重算")
            continue

        rerun_deps = [
            dep
            for dep in task.deps
            if dep in selected and decisions.get(dep, ("", ""))[0] == "运行"
        ]
        if rerun_deps:
            decisions[key] = (
                "运行",
                "本轮将更新前置任务：" + "、".join(rerun_deps),
            )
            continue

        reusable, reason = up_to_date(task, tasks, root)
        decisions[key] = ("复用" if reusable else "运行", reason)

    return decisions


def print_plan(
    ordered_keys: Sequence[str],
    tasks: dict[str, Task],
    root: Path,
    force: bool,
    include_dependencies: bool,
) -> None:
    """显示本次执行计划及各任务的判断结果。"""

    print(f"项目根目录：{root}")
    print(f"依赖展开：{'是' if include_dependencies else '否'}")
    print(f"已有结果处理：{'强制重算' if force else '可验证时复用'}")
    print("\n执行计划：")

    decisions = plan_decisions(ordered_keys, tasks, root, force)
    current_stage = None
    for index, key in enumerate(ordered_keys, start=1):
        task = tasks[key]
        if task.stage != current_stage:
            current_stage = task.stage
            print(f"\n{current_stage}")
        action, reason = decisions[key]
        print(f"  {index:02d}. [{action}] {key}：{task.name}")
        print(f"      原因：{reason}")
        if task.note and task.note != reason:
            print(f"      备注：{task.note}")


def output_snapshot(task: Task, root: Path) -> dict[Path, int | None]:
    """保存任务启动前各输出的修改时间。"""

    snapshot: dict[Path, int | None] = {}
    for output in task.outputs:
        path = resolve_path(root, output)
        snapshot[path] = path.stat().st_mtime_ns if path.exists() else None
    return snapshot


def validate_task_inputs(task: Task, tasks: dict[str, Task], root: Path) -> None:
    """在启动单个任务前再次检查其全部输入。"""

    missing = [
        resolve_path(root, path)
        for path in required_inputs(task, tasks)
        if not resolve_path(root, path).exists()
    ]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise PipelineError(f"任务 {task.key} 启动前仍缺少输入：\n{lines}")


def validate_task_outputs(
    task: Task,
    root: Path,
    before: dict[Path, int | None],
) -> None:
    """要求成功退出的任务真正生成并更新其正式输出。"""

    if not task.outputs:
        return

    missing: list[Path] = []
    empty: list[Path] = []
    unchanged: list[Path] = []

    for output in task.outputs:
        path = resolve_path(root, output)
        if not path.exists():
            missing.append(path)
            continue
        if path.is_file() and path.stat().st_size == 0:
            empty.append(path)
        old_mtime = before.get(path)
        if old_mtime is not None and path.stat().st_mtime_ns <= old_mtime:
            unchanged.append(path)

    details: list[str] = []
    if missing:
        details.append("未生成：\n" + "\n".join(f"  - {path}" for path in missing))
    if empty:
        details.append("空文件：\n" + "\n".join(f"  - {path}" for path in empty))
    if unchanged:
        details.append(
            "本次运行未更新：\n" + "\n".join(f"  - {path}" for path in unchanged)
        )
    if details:
        raise PipelineError(
            f"任务 {task.key} 虽然正常退出，但输出检查未通过：\n"
            + "\n".join(details)
        )


def run_task(
    task: Task,
    tasks: dict[str, Task],
    root: Path,
    python_executable: str,
) -> None:
    """在独立进程中执行一个任务，并核对其输出。"""

    validate_task_inputs(task, tasks, root)
    script_path = resolve_path(root, task.script)
    before = output_snapshot(task, root)

    for output in task.outputs:
        resolve_path(root, output).parent.mkdir(parents=True, exist_ok=True)

    command = [
        python_executable,
        str(script_path),
        *format_args(task.args, root),
    ]
    print(f"\n开始任务：{task.key}｜{task.name}", flush=True)
    print(f"命令：{shlex.join(command)}", flush=True)

    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise PipelineError(
            f"任务 {task.key} 执行失败，退出码为 {completed.returncode}。"
        )

    validate_task_outputs(task, root, before)
    print(f"任务完成并通过输出检查：{task.key}", flush=True)


def execute_plan(
    ordered_keys: Sequence[str],
    tasks: dict[str, Task],
    root: Path,
    python_executable: str,
    force: bool,
) -> None:
    """按计划顺序执行任务；任一任务失败后立即终止。"""

    completed_count = 0
    reused_count = 0
    ran_tasks: set[str] = set()

    for key in ordered_keys:
        task = tasks[key]
        upstream_ran = any(dep in ran_tasks for dep in task.deps)
        if not force and not upstream_ran:
            reusable, reason = up_to_date(task, tasks, root)
            if reusable:
                print(f"\n复用任务：{key}｜{reason}", flush=True)
                reused_count += 1
                continue

        run_task(task, tasks, root, python_executable)
        completed_count += 1
        ran_tasks.add(key)

    print("\n本次流程全部完成。")
    print(f"实际运行任务：{completed_count}")
    print(f"复用已有结果：{reused_count}")


def print_catalog(tasks: dict[str, Task], targets: dict[str, Target]) -> None:
    """打印全部执行目标和任务清单。"""

    print("可选目标：")
    for target in targets.values():
        print(f"  {target.name:<22} {target.description}")

    print("\n任务清单：")
    current_stage = None
    for task in tasks.values():
        if task.stage != current_stage:
            current_stage = task.stage
            print(f"\n{current_stage}")
        print(
            f"  {task.key:<28} [{STATUS_LABELS[task.status]}] {task.name}"
        )
        if task.note:
            print(f"      {task.note}")


def parse_args(
    tasks: dict[str, Task],
    targets: dict[str, Target],
) -> argparse.Namespace:
    """解析总控脚本命令行参数。"""

    parser = argparse.ArgumentParser(
        description="按依赖关系预览或执行 MMM 复现项目现有脚本。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--target",
        choices=tuple(targets),
        default=None,
        help="选择预定义执行目标；默认预览 pattern_panel。",
    )
    selection.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(tasks),
        help="选择一个或多个任务，并自动展开其前置依赖。",
    )
    selection.add_argument(
        "--only",
        nargs="+",
        choices=tuple(tasks),
        help="只运行列出的任务，不自动展开依赖；仍会检查现有输入。",
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="实际执行计划；不加此参数时只预览。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="不复用已存在且未过期的输出，强制重新运行全部选中任务。",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="启动子脚本所用的 Python 解释器，默认使用当前解释器。",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出全部执行目标和任务后退出。",
    )
    return parser.parse_args()


def selected_tasks(
    args: argparse.Namespace,
    targets: dict[str, Target],
) -> tuple[tuple[str, ...], bool, str]:
    """根据命令行参数返回任务、是否展开依赖和选择说明。"""

    if args.only:
        return tuple(args.only), False, "仅选任务"
    if args.tasks:
        return tuple(args.tasks), True, "自定义任务"

    target_name = args.target or "pattern_panel"
    target = targets[target_name]
    return target.tasks, True, f"目标 {target_name}：{target.description}"


def main() -> int:
    """总控入口。"""

    root = find_project_root()
    tasks = build_tasks()
    targets = build_targets()
    validate_registry(tasks, targets)
    args = parse_args(tasks, targets)

    if args.list:
        print_catalog(tasks, targets)
        return 0

    chosen, include_dependencies, selection_text = selected_tasks(args, targets)
    ordered = topological_order(chosen, tasks, include_dependencies)

    print(f"本次选择：{selection_text}")
    print_plan(
        ordered_keys=ordered,
        tasks=tasks,
        root=root,
        force=args.force,
        include_dependencies=include_dependencies,
    )

    issues = preflight_issues(ordered, tasks, root)
    if issues:
        print("\n运行前检查发现以下问题：")
        for issue in issues:
            print(f"  - {issue}")

    if not args.run:
        print("\n当前为预览模式；加入 --run 后才会实际执行。")
        return 0

    if issues:
        print("\n存在阻塞问题，本次未启动任何任务。")
        return 2

    try:
        execute_plan(
            ordered_keys=ordered,
            tasks=tasks,
            root=root,
            python_executable=args.python,
            force=args.force,
        )
    except KeyboardInterrupt:
        print("\n收到中断信号，流程已停止。", file=sys.stderr)
        return 130
    except PipelineError as exc:
        print(f"\n流程已停止：{exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
