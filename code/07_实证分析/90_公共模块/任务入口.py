"""拆分后的逐表入口与结果位置登记。

本模块仅登记任务和按需加载代码，导入时不读取研究数据、不创建目录。
两份原综合脚本保存在“99_历史与待核对/原综合脚本”；此处只使用新的逐表目录。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Sequence


TASKS = {
    "表1_描述性统计": {
        "说明": "从原综合回归提取的描述性统计；统计与图片同步生成。",
        "数值文件": (),
        "图片文件": ("table1_描述性统计.png",),
        "模式": "统计与绘图",
    },
    "表2_收益回归": {
        "说明": "模式公司与全样本、两种收益窗口、原有六种回归规格。",
        "数值文件": ("表2_模型汇总.csv", "表2_系数明细.csv"),
        "图片文件": (
            "table2_收益回归_紧凑版.png",
            "table2_收益回归_完整系数版_模式公司主样本.png",
            "table2_收益回归_完整系数版_模式公司主样本_续页.png",
            "table2_收益回归_完整系数版_模式公司主样本.pdf",
            "table2_收益回归_完整系数版_完整样本.png",
            "table2_收益回归_完整系数版_完整样本_续页.png",
        ),
        "模式": "估计与绘图",
    },
    "表3_散户订单失衡": {
        "说明": "成交量订单失衡的基准回归；另保留日期固定效应结果的独立绘表能力。",
        "数值文件": ("表3_散户订单失衡结果.csv", "表3_系数明细.csv"),
        "图片文件": ("table3_散户订单失衡_完整系数版.png",),
        "模式": "估计与绘图",
    },
    "表4_规模异质性": {
        "说明": "事件当日至下一交易日的两日收益窗口，属于规模异质性扩展分析。",
        "数值文件": ("表4_规模异质性结果.csv", "表4_规模异质性系数明细.csv"),
        "图片文件": ("table4_规模异质性_t_tp1扩展分析.png",),
        "模式": "估计与绘图",
    },
    "附表A2_买卖次数订单失衡": {
        "说明": "买卖次数订单失衡的基准回归；保持原综合回归的样本与估计口径。",
        "数值文件": (
            "附表A2_OIBnum_买卖次数订单失衡结果.csv",
            "附表A2_OIBnum_买卖次数订单失衡系数明细.csv",
        ),
        "图片文件": ("appendix_A2_OIBnum_买卖次数订单失衡_完整系数版.png",),
        "模式": "估计与绘图",
    },
    "附表A2-A4_日期固定效应": {
        "说明": "三种订单失衡日期固定效应的估计与绘图；数值和图片集中于本任务。",
        "数值文件": tuple(
            f"附表{number}_{metric}_{label}订单失衡_加入日期固定效应{kind}.csv"
            for number, metric, label in (
                ("A2", "OIBnum", "买卖次数"),
                ("A3", "OIBamount", "成交金额"),
                ("A4", "OIBorder", "成交笔数"),
            )
            for kind in ("结果", "系数明细")
        ),
        "图片文件": tuple(
            f"appendix_{number}_{metric}_{label}订单失衡_完整系数版_加入日期固定效应.png"
            for number, metric, label in (
                ("A2", "OIBnum", "买卖次数"),
                ("A3", "OIBamount", "成交金额"),
                ("A4", "OIBorder", "成交笔数"),
            )
        ),
        "模式": "估计与绘图",
    },
}


def find_project_root(source: str | Path) -> Path:
    """按项目标记寻根，调整任务目录层级时无需重算父目录层数。"""
    for parent in Path(source).resolve().parents:
        if (
            (parent / "AGENTS.md").is_file()
            and (parent / "code").is_dir()
            and (parent / "中间结果").is_dir()
        ):
            return parent
    raise FileNotFoundError("未找到包含项目说明、代码和中间结果的项目根目录。")


def task_paths(task_name: str, source: str | Path) -> tuple[Path, Path]:
    root = find_project_root(source)
    return (
        root / "code" / "07_实证分析" / "01_主线图表" / task_name,
        root / "图表结果" / "01_主线图表" / task_name,
    )


def show_task(task_name: str, source: str | Path, date_fixed_effects: bool = False) -> None:
    metadata = TASKS[task_name]
    task_dir, result_dir = task_paths(task_name, source)
    print(f"任务：{task_name}")
    print(metadata["说明"])
    print(f"运行入口：{task_dir / '运行本任务.py'}")
    input_dir = result_dir
    print(f"数值结果目录：{input_dir}")
    print(f"图片输出目录：{result_dir}")
    if metadata["模式"] != "仅绘图":
        estimate_name = "估计回归_日期固定效应.py" if date_fixed_effects else "估计回归.py"
        print(f"估计脚本：{task_dir / estimate_name}")
    if metadata["模式"] != "统计与绘图":
        print(f"绘图脚本：{task_dir / '绘制表格.py'}")
    for kind in ("数值文件", "图片文件"):
        for filename in metadata[kind]:
            if date_fixed_effects:
                if task_name == "表3_散户订单失衡":
                    filename = {
                        "表3_散户订单失衡结果.csv": "表3_散户订单失衡_加入日期固定效应结果.csv",
                        "表3_系数明细.csv": "表3_散户订单失衡_加入日期固定效应系数明细.csv",
                    }.get(filename, filename.replace(".png", "_加入日期固定效应.png"))
                elif task_name == "表4_规模异质性":
                    filename = filename.replace("结果.csv", "_加入日期固定效应结果.csv").replace("系数明细.csv", "_加入日期固定效应系数明细.csv").replace(".png", "_加入日期固定效应.png")
            directory = input_dir if kind == "数值文件" else result_dir
            print(f"{kind}：{directory / filename}")


def _load_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"任务模块不存在：{path}")
    module_name = f"_逐表任务_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载任务模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_task(source: str | Path, argv: Sequence[str] | None = None) -> list[Path]:
    """运行一项表格任务；不同任务不会互相触发估计或加载结果。"""
    task_dir = Path(source).resolve().parent
    task_name = task_dir.name
    if task_name not in TASKS:
        raise ValueError(f"未登记的拆分任务：{task_name}")
    metadata = TASKS[task_name]
    parser = argparse.ArgumentParser(description=f"{task_name}：{metadata['说明']}")
    parser.add_argument(
        "--stage",
        choices=("全部", "估计", "绘图", "查看"),
        default="绘图" if metadata["模式"] == "仅绘图" else "全部",
        help="阶段：默认执行本任务的完整流程；查看阶段只打印入口和结果位置。",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="仅读取此目录中的既有数值结果，限仅绘图阶段；图片仍写入新任务目录。",
    )
    parser.add_argument(
        "--date-fixed-effects",
        action="store_true",
        help="选择表3或表4的日期固定效应规格，可用于估计、绘图或完整流程。",
    )
    args = parser.parse_args(argv)
    if args.stage == "查看":
        if args.date_fixed_effects and task_name not in {"表3_散户订单失衡", "表4_规模异质性"}:
            parser.error("日期固定效应开关仅用于表3、表4。")
        show_task(task_name, source, date_fixed_effects=args.date_fixed_effects)
        return []
    if args.result_dir is not None and args.stage != "绘图":
        parser.error("--result-dir（数值结果目录）仅用于 --stage 绘图（仅绘图阶段）。")
    if args.result_dir is not None and not args.result_dir.is_dir():
        parser.error(f"指定的数值结果目录不存在：{args.result_dir}")
    if args.date_fixed_effects and (
        task_name not in {"表3_散户订单失衡", "表4_规模异质性"}
    ):
        parser.error(
            "日期固定效应开关仅用于表3、表4；表2原有六规格无需此开关。"
        )
    if metadata["模式"] == "统计与绘图" and args.stage == "绘图":
        parser.error("表1的统计与绘图同步完成，请选择全部或估计阶段。")
    if metadata["模式"] == "仅绘图" and args.stage in {"全部", "估计"}:
        parser.error("本任务只拆出了绘图功能，请先使用原独立估计脚本生成数值结果。")

    _, result_dir = task_paths(task_name, source)
    print(f"本次任务：{task_name}", flush=True)
    print(f"本次结果目录：{result_dir}", flush=True)
    written: list[Path] = []
    if args.stage in {"全部", "估计"}:
        estimate_name = "估计回归_日期固定效应.py" if args.date_fixed_effects else "估计回归.py"
        estimator = _load_module(task_dir / estimate_name)
        written.extend(estimator.main())
    if args.stage in {"全部", "绘图"} and metadata["模式"] != "统计与绘图":
        renderer = _load_module(task_dir / "绘制表格.py")
        if args.result_dir is not None:
            read_dir = args.result_dir.resolve()
        else:
            read_dir = result_dir
        print(f"绘图读取数值目录：{read_dir}", flush=True)
        kwargs = {"result_dir": read_dir}
        if task_name in {"表3_散户订单失衡", "表4_规模异质性"}:
            kwargs["date_fixed_effects"] = args.date_fixed_effects
        written.extend(renderer.main(**kwargs))
    print("本次生成文件：", flush=True)
    for path in written:
        print(path, flush=True)
    return written
