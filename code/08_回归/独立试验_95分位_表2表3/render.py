"""将独立95分位试验的真实估计结果绘制为两张论文式表格。

本模块只负责渲染，不读取原始样本、不重新估计，也不换算系数单位。
结果中的系数和标准误须已采用百分点。导入模块不会生成图片或缓存。
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
MAIN_FILENAMES = (
    "表2_95分位_收益回归.png",
    "表3_95分位_散户订单失衡.png",
)


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须为实际估计得到的有限数字。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label}必须为实际估计得到的有限数字。")
    return number


def _validate_model(row: dict[str, Any], table: int, spec: int, h: int) -> None:
    required = ("beta", "se", "p", "n", "r2", "controls", "date_fe", "firm_quarter_fe")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"表{table}规格{spec}窗口{h}缺少字段：{'、'.join(missing)}。")
    if "table" in row and int(row["table"]) != table:
        raise ValueError(f"表{table}结果混入了其他表的模型。")
    _finite(row["beta"], "系数")
    if _finite(row["se"], "标准误") < 0:
        raise ValueError("标准误不得为负数。")
    p = _finite(row["p"], "概率值")
    if not 0 <= p <= 1:
        raise ValueError("概率值必须介于0和1之间。")
    n = _finite(row["n"], "观测数")
    if n <= 0 or int(n) != n:
        raise ValueError("观测数必须为正整数。")
    _finite(row["r2"], "拟合优度")
    settings = {
        1: (0, False, False),
        2: (21, False, False),
        3: (27, False, False),
        4: (27, True, False),
        5: (27, False, True),
        6: (27, True, True),
    }
    expected = settings[spec]
    actual = (int(row["controls"]), bool(row["date_fe"]), bool(row["firm_quarter_fe"]))
    if actual != expected:
        raise ValueError(f"表{table}规格{spec}的控制变量或固定效应设置与预定规格不符。")


def _models(results: dict[str, Any], table: int) -> dict[tuple[int, int], dict[str, Any]]:
    rows = results.get(f"table{table}")
    if not isinstance(rows, list):
        raise ValueError(f"表{table}结果必须为模型列表。")
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "spec" not in row or "h" not in row:
            raise ValueError(f"表{table}模型缺少规格或窗口。")
        key = (int(row["spec"]), int(row["h"]))
        if key in lookup:
            raise ValueError(f"表{table}存在重复模型：规格{key[0]}、窗口{key[1]}。")
        lookup[key] = row
    expected = {(spec, h) for spec in (range(1, 7) if table == 2 else (6,)) for h in (1, 10)}
    if set(lookup) != expected:
        raise ValueError(f"表{table}模型不完整，或包含非预定规格／窗口。")
    for (spec, h), row in lookup.items():
        _validate_model(row, table, spec, h)
    return lookup


def _stars(p: Any) -> str:
    number = float(p)
    return "***" if number < 0.01 else "**" if number < 0.05 else "*" if number < 0.10 else ""


def _number(value: Any, digits: int = 4) -> str:
    number = float(value)
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:.{digits}f}"


def _probability(value: Any) -> str:
    return "<0.0001" if float(value) < 0.0001 else _number(value)


@contextmanager
def _setup_plotting():
    # 尊重运行入口指定的缓存；独立调用时，仅创建用后清理的项目内临时缓存。
    previous_cache = os.environ.get("MPLCONFIGDIR")
    temporary_cache = None
    if not previous_cache:
        work = ROOT / "work"
        work.mkdir(parents=True, exist_ok=True)
        temporary_cache = TemporaryDirectory(prefix="q95_matplotlib_", dir=work)
        os.environ["MPLCONFIGDIR"] = temporary_cache.name
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
        from matplotlib.font_manager import FontProperties

        if not FONT_PATH.is_file():
            raise FileNotFoundError(f"指定宋体字体不存在：{FONT_PATH}")
        font = FontProperties(fname=str(FONT_PATH))
        plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
        yield plt, font
    finally:
        if temporary_cache is not None:
            if previous_cache is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_cache
            temporary_cache.cleanup()


def _text(fig, font, x: float, y: float, text: str, *, size: float = 16,
          align: str = "left", weight: str = "normal", color: str = "#171717"):
    fig.text(x, y, text, fontproperties=font, fontsize=size, ha=align,
             va="center", weight=weight, color=color, linespacing=1.2)


def _rule(fig, y: float, *, width: float = 1.2, x0: float = 0.045, x1: float = 0.955):
    from matplotlib.lines import Line2D

    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure,
                          color="#222222", linewidth=width))


def _table(fig, font, title: str, headers: list[str], rows: list[tuple[str, list[str], str]],
           *, top: float, step: float, label_fraction: float = 0.33, size: float = 16) -> float:
    x0, x1 = 0.045, 0.955
    width = x1 - x0
    split = x0 + width * label_fraction
    centres = [split + (x1 - split) * (i + 0.5) / len(headers) for i in range(len(headers))]
    _text(fig, font, x0, top + step * 1.2, title, size=size + 1, weight="bold")
    _rule(fig, top + step * 0.58, width=1.4)
    _text(fig, font, x0 + 0.006, top, "变量与模型设定", size=size)
    for x, header in zip(centres, headers):
        _text(fig, font, x, top, header, size=size, align="center")
    _rule(fig, top - step * 0.55, width=0.9)
    for index, (label, values, kind) in enumerate(rows, start=1):
        y = top - step * index
        color = "#173C57" if kind == "coefficient" else "#171717"
        _text(fig, font, x0 + 0.006, y, label, size=size, color=color,
              weight="bold" if kind == "coefficient" else "normal")
        for x, value in zip(centres, values):
            _text(fig, font, x, y, value, size=size, align="center", color=color,
                  weight="bold" if kind == "coefficient" else "normal")
    bottom = top - step * (len(rows) + 0.6)
    _rule(fig, bottom, width=1.4)
    return bottom


def _result_rows(models: list[dict[str, Any]], *, table: int) -> list[tuple[str, list[str], str]]:
    rows = [
        ("95分位提示事件", [_number(m["beta"]) + _stars(m["p"]) for m in models], "coefficient"),
        ("（双向聚类标准误）", [f"({_number(m['se'])})" for m in models], "standard_error"),
        ("双侧检验概率值", [_probability(m["p"]) for m in models], "ordinary"),
    ]
    if table == 2:
        rows.extend([
            ("自身盈余公告控制（21项）", ["是" if int(m["controls"]) >= 21 else "否" for m in models], "ordinary"),
            ("其他事件及媒体控制（6项）", ["是" if int(m["controls"]) == 27 else "否" for m in models], "ordinary"),
        ])
    else:
        rows.append(("全部控制变量（27项）", ["是" for _ in models], "ordinary"))
    rows.extend([
        ("交易日期固定效应", ["是" if m["date_fe"] else "否" for m in models], "ordinary"),
        ("被提示公司×年度季度固定效应", ["是" if m["firm_quarter_fe"] else "否" for m in models], "ordinary"),
    ])
    if table == 2:
        rows.append(("常数项", ["是" for _ in models], "ordinary"))
    rows.extend([
        ("观测数", [f"{int(m['n']):,}" for m in models], "ordinary"),
        ("拟合优度", [_number(m["r2"]) for m in models], "ordinary"),
    ])
    return rows


def _notes(fig, font, *, start: float, step: float, size: float, table: int):
    lines = [
        "注：括号内为按被提示公司和交易日期双向聚类的标准误，非概率值。系数与标准误均以百分点报告。",
        "***、**、*分别表示在1%、5%、10%水平显著；概率值来自双侧检验，未做多重检验校正。",
        "95分位提示事件：提示公司事件前第3个交易日总市值，高于上月沪深主板参考样本的第95分位断点。",
        "该市值限制仅作用于提示公司；被提示公司样本不按市值截断。固定效应中的公司×年度季度为交互类别。",
        "因变量为公告日至所列窗口末日的累计特征调整收益。" if table == 2 else
        "因变量为每日小单买卖不平衡比例的窗口均值，单位为百分点；两列分别包含2个和11个交易日。",
    ]
    for index, line in enumerate(lines):
        _text(fig, font, 0.045, start - step * index, line, size=size, color="#363636")


def _render_table2(plt, font, lookup: dict, target: Path):
    fig = plt.figure(figsize=(18, 14), dpi=240)
    try:
        _text(fig, font, 0.5, 0.964, "表2　收益回归：提示公司市值高于第95分位", size=23, align="center", weight="bold")
        _text(fig, font, 0.5, 0.929, "2005—2025年规律公告公司主样本", size=17, align="center", color="#424242")
        _text(fig, font, 0.955, 0.899, "单位：百分点", size=14, align="right", color="#424242")
        for h, top, title in (
            (1, 0.850, "面板一　公告日至后1个交易日（共2个交易日）"),
            (10, 0.508, "面板二　公告日至后10个交易日（共11个交易日）"),
        ):
            models = [lookup[(spec, h)] for spec in range(1, 7)]
            _table(fig, font, title, [f"（{spec}）" for spec in range(1, 7)],
                   _result_rows(models, table=2), top=top, step=0.025, size=16)
        _notes(fig, font, start=0.193, step=0.030, size=12.5, table=2)
        with target.open("xb") as handle:
            fig.savefig(handle, format="png", dpi=240, facecolor="white")
    finally:
        plt.close(fig)


def _render_table3(plt, font, lookup: dict, target: Path):
    fig = plt.figure(figsize=(15, 10), dpi=240)
    try:
        _text(fig, font, 0.5, 0.954, "表3　散户订单失衡：提示公司市值高于第95分位", size=22, align="center", weight="bold")
        _text(fig, font, 0.5, 0.910, "2005—2025年规律公告公司主样本", size=17, align="center", color="#424242")
        _text(fig, font, 0.955, 0.865, "单位：百分点", size=14, align="right", color="#424242")
        models = [lookup[(6, h)] for h in (1, 10)]
        _table(fig, font, "因变量：窗口平均小单买卖不平衡比例",
               ["公告日至后1个交易日", "公告日至后10个交易日"],
               _result_rows(models, table=3), top=0.778, step=0.046,
               label_fraction=0.44, size=17)
        _notes(fig, font, start=0.280, step=0.044, size=11.5, table=3)
        with target.open("xb") as handle:
            fig.savefig(handle, format="png", dpi=240, facecolor="white")
    finally:
        plt.close(fig)


def render_results(results: dict, output: Path) -> list[Path]:
    """绘制真实结果，返回两张图片路径；遇到同名文件即停止，不覆盖既有交付。

    输入为估计模块产生的字典，包含表2的12个模型及表3的2个模型。
    每个模型的系数、标准误须已用百分点表示，本函数不乘100。
    输出参数为本次独立试验专用的图片目录。
    """
    table2 = _models(results, 2)
    table3 = _models(results, 3)
    output = Path(output)
    paths = [output / name for name in MAIN_FILENAMES]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"为保护既有图片，本次不覆盖：{'、'.join(str(path) for path in existing)}")
    output.mkdir(parents=True, exist_ok=True)
    with _setup_plotting() as (plt, font):
        _render_table2(plt, font, table2, paths[0])
        _render_table3(plt, font, table3, paths[1])
    return paths
