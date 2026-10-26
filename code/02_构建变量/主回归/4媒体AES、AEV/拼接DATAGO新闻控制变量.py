"""拼接 DATAGO（数行者数据库）的两套真实交易日新闻控制变量。

顺序：读取已计算完成的控制表 → 按日期选来源 → 检查键和值 → 写出统一表。
2010年1月1日前使用报刊，此后使用网络。保留回归期之前的报刊控制历史，
最终研究期仍由下游研究设计确定。两来源必须各自先用完整历史计算91日窗口
及交易日变化；此处不重新差分，也不跨来源填补缺失。

直接运行只生成统一新闻控制表，不重算原始新闻、记忆矩阵或回归。
临时文件放在项目 work（临时工作目录），全部完成后才替换正式结果。
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


# 一、路径和分段日期。
ROOT = Path(__file__).resolve().parents[4]
NEWSPAPER_PATH = ROOT / (
    "中间结果/02_变量构造结果/3报刊新闻_AES_AEV/"
    "DATAGO_报刊新闻_AES_AEV_真实交易日控制变量.csv"
)
NETWORK_PATH = ROOT / (
    "中间结果/02_变量构造结果/3网络新闻_AES_AEV/"
    "DATAGO_网络新闻_AES_AEV_真实交易日控制变量.csv"
)
OUTPUT_PATH = ROOT / (
    "中间结果/02_变量构造结果/3分段新闻_AES_AEV/"
    "DATAGO_分段新闻_真实交易日控制变量.csv"
)
SWITCH_DATE = pd.Timestamp("2010-01-01")
CHUNK_SIZE = 500_000
COLUMNS = [
    "Stkcd", "date", "AES", "AEV", "aesIncrease", "aesDecrease",
    "logIncreaseAEV", "logDecreaseAEV",
]


def main():
    """分块拼接，保留来源缺失值和公司—交易日唯一键。"""

    # 二、检查输入；单文件替换前不改动已有正式结果。
    for source in (NEWSPAPER_PATH, NETWORK_PATH):
        if not source.is_file():
            raise FileNotFoundError(f"请先生成完整历史新闻控制表：{source}")
        if OUTPUT_PATH.resolve() == source.resolve():
            raise ValueError("统一控制表不能覆盖任一来源表。")
    work_dir = ROOT / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if work_dir.stat().st_dev != OUTPUT_PATH.parent.stat().st_dev:
        raise ValueError("临时目录与统一控制表必须位于同一文件系统。")

    source_rows = {}
    first_chunk = True
    with TemporaryDirectory(prefix="datago_news_controls_", dir=work_dir) as temp_dir:
        pending = Path(temp_dir) / "分段新闻控制变量.csv"

        # 三、每个来源按公司和日期有序且唯一；两段日期互斥，拼接后不会重键。
        for label, source in (("报刊", NEWSPAPER_PATH), ("网络", NETWORK_PATH)):
            last_key = None
            selected_rows = 0
            reader = pd.read_csv(
                source, usecols=COLUMNS, dtype={"Stkcd": "string"},
                parse_dates=["date"], encoding="utf-8-sig", chunksize=CHUNK_SIZE,
            )
            for part in reader:
                part = part[COLUMNS].copy()
                part["Stkcd"] = part["Stkcd"].str.strip()
                part["date"] = pd.to_datetime(part["date"], errors="raise")
                if (
                    not part["Stkcd"].str.fullmatch(r"\d{6}", na=False).all()
                    or part["date"].isna().any()
                ):
                    raise ValueError(f"{label}控制表存在无效公司代码或缺失日期。")
                if part.empty:
                    continue
                keys = pd.MultiIndex.from_frame(part[["Stkcd", "date"]])
                if not keys.is_unique or not keys.is_monotonic_increasing:
                    raise ValueError(f"{label}控制表应按公司—日期排序且键唯一。")
                if last_key is not None and keys[0] <= last_key:
                    raise ValueError(f"{label}控制表在分块边界存在重复或乱序。")
                last_key = keys[-1]

                keep = part["date"].lt(SWITCH_DATE) if label == "报刊" else part["date"].ge(SWITCH_DATE)
                part = part.loc[keep].copy()
                if part.empty:
                    continue
                numeric = part[COLUMNS[2:]].apply(pd.to_numeric, errors="raise")
                if np.isinf(numeric.to_numpy(dtype=float)).any():
                    raise ValueError(f"{label}控制表存在无穷值。")
                if not numeric["AES"].dropna().between(0, 100).all():
                    raise ValueError(f"{label}新闻情绪应位于0—100。")
                if numeric[["AEV", "logIncreaseAEV", "logDecreaseAEV"]].lt(0).any().any():
                    raise ValueError(f"{label}新闻量及其变化对数不能为负数。")
                for column in ("aesIncrease", "aesDecrease"):
                    if not numeric[column].dropna().isin([0, 1]).all():
                        raise ValueError(f"{label}情绪变化标记应为0、1或缺失。")
                part[COLUMNS[2:]] = numeric

                # 四、只追加选定日期段，不改变上游已计算的指标或缺失状态。
                part.to_csv(
                    pending, mode="w" if first_chunk else "a", header=first_chunk,
                    index=False, encoding="utf-8-sig" if first_chunk else "utf-8",
                    date_format="%Y-%m-%d",
                )
                first_chunk = False
                selected_rows += len(part)
            if selected_rows == 0:
                raise ValueError(f"{label}在所需日期段没有可用的控制表记录。")
            source_rows[label] = selected_rows

        # 五、两段均完成后发布。文件按来源分段存放，下游按公司—交易日合并。
        pending.replace(OUTPUT_PATH)

    print("已保存统一新闻控制表：", OUTPUT_PATH)
    print("报刊日期：2010年1月1日前；网络日期：2010年1月1日起。")
    print("各来源行数：", source_rows, "；合计：", sum(source_rows.values()))


if __name__ == "__main__":
    main()
