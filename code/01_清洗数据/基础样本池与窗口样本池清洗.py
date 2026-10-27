"""
基础样本池、行业代码补缺和标签样本池的一体化清洗脚本。

本脚本完整实现以下三项功能：

1. 生成 A 股市场交易日历和基础清洗后的股票—交易日样本索引；
2. 使用 CSMAR 行业历史匹配行业代码，并使用 RESSET 行业历史补充缺失；
3. 构造完整股票—交易日面板，以及用于 DGTW 标签构造的收益窗口和滞后市值。

底层计算面板保留 1995—2026 年的前后缓冲期；金融业、上市年限、
异常交易状态和停牌填充观测的清洗统计仅报告 2000—2025 年。

统一处理顺序：

原始日度数据（只读取一次）
→ 匹配市场交易日序号
→ 构造完整股票—交易日面板
→ 计算成交天数（交易天数）
→ 匹配 CSMAR 行业
→ 使用 RESSET 补缺
→ 重新识别金融行业
→ 上市满一年筛选
→ ST/PT 状态筛选
→ 输出基础样本池、窗口变量目标日资格索引和 DGTW 构建面板。

主要输出：

- 中间结果/01_清洗后样本/市场交易日历_td.csv
- 中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv
- 中间结果/01_清洗后样本/窗口变量目标日资格索引_基础清洗含停牌填充_td.csv
- 中间结果/01_清洗后样本/完整个股交易日面板索引_td.csv
- 中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv
- 中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板.csv
- 中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺.csv
- 中间结果/02_变量构造结果/同行业识别/个股行业代码完整面板_RESSET补缺明细.csv
"""

from __future__ import annotations

import fcntl
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 一、项目路径和构造参数
# =============================================================================


def find_project_root() -> Path:
    """从当前脚本位置向上寻找同时包含 code 和 input 的项目根目录。"""

    script_dir = Path(__file__).resolve().parent

    for directory in (script_dir, *script_dir.parents):
        if (directory / "code").is_dir() and (directory / "input").is_dir():
            return directory

    raise FileNotFoundError("无法找到同时包含 code 和 input 文件夹的项目根目录。")


ROOT = find_project_root()

# 输入文件
DAILY_PATH = (
    ROOT
    / "中间结果/00_合并原始数据/CSMAR_日个股回报率(升级测试)_1995_2026.csv"
)
CSMAR_INDUSTRY_PATH = (
    ROOT / "input/CSMAR/上市公司行业分类表1990年12月10日—2026年4月20日/STK_INDUSTRYCLASS.csv"
)
RESSET_INDUSTRY_PATH = (
    ROOT / "input/RESSET/RESSET 股票_行业变更历史/RESSET_INDHIS_1.csv"
)
COMPANY_PATH = (
    ROOT / "input/CSMAR/公司文件(1990-12-10至2026-08-21)/TRD_Co.csv"
)
RAW_CALENDAR_PATH = (
    ROOT / "input/CSMAR/日历文件(1990-12-19至2026-08-20)/CER_Cale.csv"
)

# 输出文件
CLEAN_OUTPUT_DIR = ROOT / "中间结果/01_清洗后样本"
INDUSTRY_OUTPUT_DIR = ROOT / "中间结果/02_变量构造结果/同行业识别"
WINDOW_VARIABLE_TARGET_INDEX_OUTPUT_PATH = (
    CLEAN_OUTPUT_DIR
    / "窗口变量目标日资格索引_基础清洗含停牌填充_td.csv"
)


CALENDAR_OUTPUT_PATH = CLEAN_OUTPUT_DIR / "市场交易日历_td.csv"
BASE_SAMPLE_OUTPUT_PATH = CLEAN_OUTPUT_DIR / "基础清洗后个股日度样本索引_td.csv"
FULL_PANEL_OUTPUT_PATH = CLEAN_OUTPUT_DIR / "完整个股交易日面板索引_td.csv"
DGTW_OUTPUT_PATH = CLEAN_OUTPUT_DIR / "DGTW构建用个股日度面板.csv"

CSMAR_INDUSTRY_PANEL_OUTPUT_PATH = INDUSTRY_OUTPUT_DIR / "个股行业代码完整面板.csv"
RESSET_FILLED_PANEL_OUTPUT_PATH = (
    INDUSTRY_OUTPUT_DIR / "个股行业代码完整面板_RESSET补缺.csv"
)
RESSET_FILL_DETAIL_OUTPUT_PATH = (
    INDUSTRY_OUTPUT_DIR / "个股行业代码完整面板_RESSET补缺明细.csv"
)

FINAL_OUTPUT_PATHS = {
    "calendar": CALENDAR_OUTPUT_PATH,
    "base_sample": BASE_SAMPLE_OUTPUT_PATH,
    "full_panel": FULL_PANEL_OUTPUT_PATH,
    "dgtw": DGTW_OUTPUT_PATH,
    "csmar_industry": CSMAR_INDUSTRY_PANEL_OUTPUT_PATH,
    "resset_filled": RESSET_FILLED_PANEL_OUTPUT_PATH,
    "resset_detail": RESSET_FILL_DETAIL_OUTPUT_PATH,
    "window_variable_target_index": (
        WINDOW_VARIABLE_TARGET_INDEX_OUTPUT_PATH
    ),
}
OUTPUT_COMMIT_MANIFEST_PATH = CLEAN_OUTPUT_DIR / ".unified_cleaning_commit.json"
OUTPUT_PROCESS_LOCK_PATH = CLEAN_OUTPUT_DIR / ".unified_cleaning.lock"

# A 股市场类型和允许保留的交易状态
A_SHARE_MARKET_TYPES = (1, 4, 16, 32)
ALLOWED_TRADING_STATUSES = (1, 4, 7, 10, 13)

# 计算期保留前后缓冲数据，最终研究期的清洗报告另行限定。
BUFFER_START_DATE = pd.Timestamp("1995-01-01")
BUFFER_END_DATE = pd.Timestamp("2026-12-31")
REPORT_START_DATE = pd.Timestamp("2000-01-01")
REPORT_END_DATE = pd.Timestamp("2025-12-31")

# 保持 2000 年第一个市场交易日仍为 td=1，与现有下游文件兼容。
TD_ANCHOR_DATE = REPORT_START_DATE

# 日度数据出现交易日期无法匹配市场日历时，默认停止，避免静默截断样本。
ALLOW_UNMATCHED_DAILY_DATES = False

# DGTW 宽面板按股票分批计算，避免一次性在全样本上生成 50 个收益字段。
DGTW_STOCK_BATCH_SIZE = 100

# 收益窗口设置
LAG_RETURN_OFFSETS = (3, 2, 1)
FORWARD_RETURN_HORIZONS = tuple(range(1, 51))
FORWARD_RETURN_DIAGNOSTIC_HORIZONS = {1, 10, 50}


# =============================================================================
# 二、通用工具和运行前检查
# =============================================================================


def normalize_stock_code(value: object) -> object:
    """把股票代码统一为六位字符串；无法识别时返回缺失值。"""

    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if text == "":
        return pd.NA

    matched = re.search(r"\d{1,6}", text)
    if matched is None:
        return pd.NA

    return matched.group(0).zfill(6)


def require_columns(
    df: pd.DataFrame,
    required_columns: list[str] | tuple[str, ...],
    table_name: str,
) -> None:
    """检查数据表是否包含指定字段。"""

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"{table_name} 缺少必要字段：{missing_columns}")


def require_unique_key(
    df: pd.DataFrame,
    key_columns: list[str] | tuple[str, ...],
    table_name: str,
) -> None:
    """要求指定字段构成唯一键；发现重复时给出示例并停止。"""

    duplicate_mask = df.duplicated(list(key_columns), keep=False)
    if not duplicate_mask.any():
        return

    examples = (
        df.loc[duplicate_mask, list(key_columns)]
        .sort_values(list(key_columns))
        .head(20)
    )
    raise ValueError(
        f"{table_name} 的唯一键 {list(key_columns)} 存在重复。\n"
        f"重复示例：\n{examples.to_string(index=False)}"
    )


def validate_input_paths() -> None:
    """运行前一次性检查全部输入文件是否存在。"""

    input_paths = {
        "个股日度数据": DAILY_PATH,
        "CSMAR 行业历史": CSMAR_INDUSTRY_PATH,
        "RESSET 行业历史": RESSET_INDUSTRY_PATH,
        "公司上市日期": COMPANY_PATH,
        "原始市场交易日历": RAW_CALENDAR_PATH,
    }

    missing = [f"{name}：{path}" for name, path in input_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("以下输入文件不存在：\n" + "\n".join(missing))


def prepare_output_directories() -> None:
    """创建全部输出文件夹。"""

    CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INDUSTRY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def acquire_output_process_lock():
    """获取本流程的非阻塞独占锁，防止两个进程争用同一批输出。"""

    lock_file = OUTPUT_PROCESS_LOCK_PATH.open("a+", encoding="utf-8")

    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError(
            "检测到另一份一体化清洗脚本正在运行。为保护输出文件，本次运行已停止。"
        ) from exc

    return lock_file


def release_output_process_lock(lock_file) -> None:
    """释放本流程持有的进程独占锁。"""

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def build_staging_output_paths() -> dict[str, Path]:
    """为七个正式输出建立同目录隐藏暂存路径。"""

    return {
        name: final_path.with_name(f".{final_path.name}.unified.tmp")
        for name, final_path in FINAL_OUTPUT_PATHS.items()
    }


def cleanup_staging_outputs(staging_paths: dict[str, Path]) -> None:
    """删除本流程遗留的暂存文件，不触碰正式输出。"""

    for staging_path in staging_paths.values():
        staging_path.unlink(missing_ok=True)

        # DGTW 分批写出时还会使用一层内部临时文件。
        inner_temporary_path = staging_path.with_name(
            f".{staging_path.name}.tmp"
        )
        inner_temporary_path.unlink(missing_ok=True)


def build_backup_output_paths() -> dict[str, Path]:
    """为可能已经存在的正式输出生成同目录备份路径。"""

    return {
        name: final_path.with_name(f".{final_path.name}.unified.backup")
        for name, final_path in FINAL_OUTPUT_PATHS.items()
    }


def recover_interrupted_output_commit() -> None:
    """根据提交清单恢复被上次异常提交打断的旧输出集合。"""

    backup_paths = build_backup_output_paths()
    temporary_manifest_path = OUTPUT_COMMIT_MANIFEST_PATH.with_name(
        f".{OUTPUT_COMMIT_MANIFEST_PATH.name}.tmp"
    )

    if not OUTPUT_COMMIT_MANIFEST_PATH.exists():
        # 没有清单但仍有备份，表示上次提交已经完成、仅备份清理被打断。
        for backup_path in backup_paths.values():
            backup_path.unlink(missing_ok=True)
        temporary_manifest_path.unlink(missing_ok=True)
        return

    manifest = json.loads(
        OUTPUT_COMMIT_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    originally_existing = set(manifest["originally_existing"])

    for name, final_path in FINAL_OUTPUT_PATHS.items():
        backup_path = backup_paths[name]

        if name in originally_existing:
            if backup_path.exists():
                final_path.unlink(missing_ok=True)
                backup_path.replace(final_path)
            elif not final_path.exists():
                raise RuntimeError(
                    f"提交恢复失败：{name} 的正式文件和备份文件都不存在。"
                )
        else:
            # 上次运行前不存在的正式文件，如果已被部分提交，需要删除。
            final_path.unlink(missing_ok=True)

    OUTPUT_COMMIT_MANIFEST_PATH.unlink(missing_ok=True)
    temporary_manifest_path.unlink(missing_ok=True)
    for backup_path in backup_paths.values():
        backup_path.unlink(missing_ok=True)

    print("检测到上次未完成的输出提交，已恢复提交前的整套正式结果。")


def commit_staging_outputs(staging_paths: dict[str, Path]) -> None:
    """全部暂存文件齐备后，以可回滚事务替换七个正式输出。"""

    missing_staging = [
        name
        for name, path in staging_paths.items()
        if (not path.is_file()) or path.stat().st_size == 0
    ]
    if missing_staging:
        raise FileNotFoundError(
            f"以下暂存输出尚未生成，拒绝替换正式文件：{missing_staging}"
        )

    backup_paths = build_backup_output_paths()
    originally_existing = [
        name for name, path in FINAL_OUTPUT_PATHS.items() if path.exists()
    ]

    manifest = {"originally_existing": originally_existing}
    temporary_manifest_path = OUTPUT_COMMIT_MANIFEST_PATH.with_name(
        f".{OUTPUT_COMMIT_MANIFEST_PATH.name}.tmp"
    )
    temporary_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_manifest_path.replace(OUTPUT_COMMIT_MANIFEST_PATH)

    try:
        # 先把旧正式文件原子移动为同目录备份，不复制大文件内容。
        for name in originally_existing:
            backup_paths[name].unlink(missing_ok=True)
            FINAL_OUTPUT_PATHS[name].replace(backup_paths[name])

        # 旧集合全部备份完成后，再提交新集合。
        for name, final_path in FINAL_OUTPUT_PATHS.items():
            staging_paths[name].replace(final_path)

    except BaseException:
        # 删除已经提交的新文件，并恢复原先存在的旧文件。
        for name, final_path in FINAL_OUTPUT_PATHS.items():
            if name not in originally_existing:
                final_path.unlink(missing_ok=True)

        for name in reversed(originally_existing):
            backup_path = backup_paths[name]
            if backup_path.exists():
                FINAL_OUTPUT_PATHS[name].unlink(missing_ok=True)
                backup_path.replace(FINAL_OUTPUT_PATHS[name])

        OUTPUT_COMMIT_MANIFEST_PATH.unlink(missing_ok=True)
        raise

    # 先删除清单，表示新集合已经完整提交；再清理不再需要的旧文件备份。
    OUTPUT_COMMIT_MANIFEST_PATH.unlink(missing_ok=True)
    for backup_path in backup_paths.values():
        backup_path.unlink(missing_ok=True)

    for final_path in FINAL_OUTPUT_PATHS.values():
        print(f"正式输出已更新：{final_path}")


def print_stage(stage_number: int, title: str) -> None:
    """打印当前处理阶段。"""
    print("\n" + "=" * 80)
    print(f"阶段 {stage_number}：{title}")
    print("=" * 80)


def get_report_period_mask(panel: pd.DataFrame) -> pd.Series:
    """返回研究期清洗报告掩码，不改变缓冲期面板的筛选范围。"""

    require_columns(panel, ("Trddt",), "清洗报告面板")
    return panel["Trddt"].between(REPORT_START_DATE, REPORT_END_DATE)


def report_period_label() -> str:
    """返回控制台统计使用的研究期标签。"""

    return f"{REPORT_START_DATE.date()}至{REPORT_END_DATE.date()}"


# =============================================================================
# 三、交易日历
# =============================================================================


def build_a_share_trading_calendar(path: Path) -> pd.DataFrame:
    """读取原始日历，保留正常开市日，并生成统一市场交易日序号 td。"""

    calendar = pd.read_csv(
        path,
        encoding="utf-8-sig",
        parse_dates=["Clddt"],
    )
    calendar = calendar.rename(columns={"Clddt": "Trddt", "Statse": "State"})

    require_columns(
        calendar,
        ("Trddt", "Markettype", "Daywk", "State"),
        "原始市场交易日历",
    )

    calendar["Markettype"] = pd.to_numeric(
        calendar["Markettype"], errors="coerce"
    )
    calendar["State"] = calendar["State"].astype("string").str.strip()

    calendar = calendar.loc[
        calendar["Markettype"].eq(1)
        & calendar["State"].eq("O")
        & calendar["Trddt"].between(BUFFER_START_DATE, BUFFER_END_DATE)
    ].copy()

    duplicate_dates = calendar.duplicated("Trddt", keep=False)
    if duplicate_dates.any():
        print(
            "正常开市日历中发现重复日期，"
            f"将对 {duplicate_dates.sum():,} 条重复观测保留最后一条。"
        )
        calendar = calendar.drop_duplicates("Trddt", keep="last").copy()

    calendar = calendar.sort_values("Trddt").reset_index(drop=True)
    calendar["year"] = calendar["Trddt"].dt.year

    anchor_position = int(
        calendar["Trddt"].searchsorted(TD_ANCHOR_DATE, side="left")
    )
    if anchor_position >= len(calendar):
        raise ValueError(
            f"市场交易日历不包含 {TD_ANCHOR_DATE.date()} 之后的交易日，"
            "无法生成兼容的 td。"
        )

    calendar["td"] = (
        np.arange(len(calendar), dtype=np.int64) - anchor_position + 1
    )

    require_unique_key(calendar, ("Trddt",), "市场交易日历")
    require_unique_key(calendar, ("td",), "市场交易日历")

    return calendar


def save_trading_calendar(calendar: pd.DataFrame, path: Path) -> None:
    """保存带统一交易日序号的 A 股市场交易日历。"""

    output_columns = ["Trddt", "Markettype", "Daywk", "State", "year", "td"]
    output = calendar[output_columns]
    output.to_csv(path, index=False, encoding="utf-8")
    print(f"市场交易日历已输出：{path}，共 {len(output):,} 个交易日。")


# =============================================================================
# 四、个股日度数据和完整交易日面板
# =============================================================================


def load_and_prepare_daily_data(path: Path) -> pd.DataFrame:
    """一次性读取三个流程需要的全部日度字段，并保留 A 股观测。"""

    daily = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=[
            "Stkcd",
            "Trddt",
            "Markettype",
            "Filling",
            "Trdsta",
            "Dnshrtrd",
            "Dretwd",
            "Dretnd",
            "Dsmvosd",
            "Dsmvtll",
        ],
        parse_dates=["Trddt"],
    )

    daily["Stkcd"] = daily["Stkcd"].apply(normalize_stock_code)

    for col in (
        "Markettype",
        "Filling",
        "Trdsta",
        "Dnshrtrd",
        "Dretwd",
        "Dretnd",
        "Dsmvosd",
        "Dsmvtll",
    ):
        daily[col] = pd.to_numeric(daily[col], errors="coerce")

    invalid_key_mask = daily["Stkcd"].isna() | daily["Trddt"].isna()
    if invalid_key_mask.any():
        print(
            "股票代码或交易日期无法识别的观测："
            f"{invalid_key_mask.sum():,} 条，已删除。"
        )
        daily = daily.loc[~invalid_key_mask].copy()

    daily = daily.rename(columns={"Dretwd": "ret", "Dretnd": "retn"})
    daily["year"] = daily["Trddt"].dt.year
    daily["sizew"] = daily["Dsmvosd"] * 1000
    daily["size"] = daily["Dsmvtll"] * 1000
    daily = daily.drop(columns=["Dsmvosd", "Dsmvtll"])

    daily = daily.loc[
        daily["Markettype"].isin(A_SHARE_MARKET_TYPES)
        & daily["Filling"].isin((0, 1))
        & daily["Trddt"].between(BUFFER_START_DATE, BUFFER_END_DATE)
    ].copy()

    print(f"筛选 A 股市场且保留 Filling=0/1 后：{len(daily):,} 条观测。")
    return daily


def attach_trading_day_index(
    daily: pd.DataFrame,
    calendar: pd.DataFrame,
    allow_unmatched_dates: bool = ALLOW_UNMATCHED_DAILY_DATES,
) -> pd.DataFrame:
    """匹配市场交易日序号，并确定性处理股票—交易日重复观测。"""

    daily_with_td = pd.merge(
        daily,
        calendar[["Trddt", "td"]],
        how="left",
        on="Trddt",
        validate="many_to_one",
    )
    daily_with_td["_source_row_id"] = np.arange(
        len(daily_with_td), dtype=np.int64
    )

    unmatched_count = daily_with_td["td"].isna().sum()
    print(f"未匹配到市场交易日序号的观测：{unmatched_count:,} 条。")

    if unmatched_count:
        unmatched_dates = daily_with_td.loc[
            daily_with_td["td"].isna(), "Trddt"
        ]
        print(
            "未匹配日期范围："
            f"{unmatched_dates.min()} 至 {unmatched_dates.max()}"
        )
        print("未匹配日期示例：")
        print(unmatched_dates.drop_duplicates().sort_values().head(20).to_string(index=False))

        if not allow_unmatched_dates:
            raise ValueError(
                "存在无法匹配市场交易日历的 A 股日度观测。"
                "请先检查日历覆盖范围；若确认可以删除，再将 "
                "ALLOW_UNMATCHED_DAILY_DATES 改为 True。"
            )

    daily_with_td = daily_with_td.dropna(subset=["td"]).copy()
    daily_with_td["td"] = daily_with_td["td"].astype(np.int64)
    daily_with_td["_prefer_original_observation"] = (
        daily_with_td["Filling"].eq(0).astype("int8")
    )
    daily_with_td = daily_with_td.sort_values(
        [
            "Stkcd",
            "td",
            "_prefer_original_observation",
            "_source_row_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    duplicate_mask = daily_with_td.duplicated(["Stkcd", "td"], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    print(f"Stkcd-td 重复观测：{duplicate_count:,} 条。")

    if duplicate_count > 0:
        print("重复观测示例：")
        print(
            daily_with_td.loc[
                duplicate_mask,
                ["Stkcd", "Trddt", "td", "Filling", "Trdsta"],
            ].head(30)
        )
        # Filling=0 的真实观测优先于 Filling=1 的填充观测；
        # 同一优先级内保留原文件中最后出现的一行。
        daily_with_td = daily_with_td.drop_duplicates(
            ["Stkcd", "td"], keep="last"
        ).reset_index(drop=True)

    daily_with_td = daily_with_td.drop(
        columns=["_prefer_original_observation", "_source_row_id"]
    )

    require_unique_key(
        daily_with_td,
        ("Stkcd", "td"),
        "匹配交易日序号后的个股日度数据",
    )

    return daily_with_td


def build_complete_stock_td_panel(
    daily_with_td: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在每只股票自身的首末 td 之间构造连续完整面板。"""

    if daily_with_td.empty:
        raise ValueError("匹配交易日序号后的个股日度数据为空，无法构造完整面板。")

    original_columns = daily_with_td.columns.tolist()

    td_range = (
        daily_with_td.groupby("Stkcd", as_index=False)["td"]
        .agg(td_min="min", td_max="max")
        .sort_values("Stkcd")
        .reset_index(drop=True)
    )

    lengths = (td_range["td_max"] - td_range["td_min"] + 1).astype(int)

    full_panel = pd.DataFrame(
        {
            "Stkcd": np.repeat(td_range["Stkcd"].to_numpy(), lengths.to_numpy()),
            "td": np.concatenate(
                [
                    np.arange(low, high + 1, dtype=np.int64)
                    for low, high in zip(td_range["td_min"], td_range["td_max"])
                ]
            ),
        }
    )

    full_panel = pd.merge(
        full_panel,
        calendar[["td", "Trddt"]],
        how="left",
        on="td",
        validate="many_to_one",
    )
    full_panel = full_panel[["Stkcd", "td", "Trddt"]]

    complete_daily_panel = pd.merge(
        full_panel,
        daily_with_td.drop(columns=["Trddt"]),
        how="left",
        on=["Stkcd", "td"],
        validate="one_to_one",
    )
    complete_daily_panel = complete_daily_panel[original_columns]

    require_unique_key(full_panel, ("Stkcd", "td"), "完整股票交易日索引面板")

    expected_rows = int(lengths.sum())
    if len(full_panel) != expected_rows:
        raise AssertionError(
            f"完整面板应有 {expected_rows:,} 行，实际为 {len(full_panel):,} 行。"
        )

    return full_panel, complete_daily_panel


def save_full_panel(full_panel: pd.DataFrame, path: Path) -> None:
    """保存完整股票—交易日索引面板。"""

    full_panel.to_csv(path, index=False, encoding="utf-8")
    print(f"完整股票交易日索引面板已输出：{path}，共 {len(full_panel):,} 行。")


def report_panel_expansion(
    observed_daily: pd.DataFrame,
    complete_daily_panel: pd.DataFrame,
) -> None:
    """报告补齐交易日序列前后的样本量变化。"""

    before = len(observed_daily)
    after = len(complete_daily_panel)
    print(f"补齐前样本数：{before:,}")
    print(f"补齐后样本数：{after:,}")
    print(f"新增补齐观测数：{after - before:,}")


def report_td_continuity(panel: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """检查已按 Stkcd、td 排序的面板是否连续，并返回断点观测。"""

    td_gap = panel.groupby("Stkcd", sort=False)["td"].diff()
    gap_mask = td_gap.notna() & td_gap.ne(1)
    gaps = panel.loc[gap_mask, ["Stkcd", "Trddt", "td"]].copy()
    gaps["td_gap"] = td_gap.loc[gap_mask].to_numpy()

    print(f"{table_name}中 td 不连续的观测数：{len(gaps):,}")
    print(f"{table_name}中 td 不连续的股票数：{gaps['Stkcd'].nunique():,}")

    if not gaps.empty:
        print(gaps.head(30))

    return gaps


def add_trade_day_counts(
    panel: pd.DataFrame,
    month_window: str = "30D",
    month_min_periods: int = 15,
    year_window: str = "365D",
    year_min_periods: int = 120,
) -> pd.DataFrame:
    """在完整面板上计算近一个月和近一年有成交量的交易天数。"""

    require_columns(panel, ("Stkcd", "Trddt", "td", "Dnshrtrd"), "完整日度面板")

    # 完整面板在构造时已经按 Stkcd、td 排序；这里原地增加字段，
    # 避免为大面板再生成一份完整副本。
    out = panel
    out.reset_index(drop=True, inplace=True)
    out["has_volume"] = out["Dnshrtrd"].fillna(0).gt(0).astype("int8")
    out.drop(columns=["Dnshrtrd"], inplace=True)

    out["trade_days_1m"] = (
        out.groupby("Stkcd", sort=False)
        .rolling(
            window=month_window,
            on="Trddt",
            min_periods=month_min_periods,
            closed="right",
        )["has_volume"]
        .sum()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )

    out["trade_days_1y"] = (
        out.groupby("Stkcd", sort=False)
        .rolling(
            window=year_window,
            on="Trddt",
            min_periods=year_min_periods,
            closed="right",
        )["has_volume"]
        .sum()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )

    out["trade_days_1m"] = out["trade_days_1m"].astype("Int16")
    out["trade_days_1y"] = out["trade_days_1y"].astype("Int16")

    return out


# =============================================================================
# 五、CSMAR 行业历史匹配
# =============================================================================


def load_and_prepare_csmar_industry(path: Path) -> pd.DataFrame:
    """读取并整理 CSMAR 行业分类历史。"""

    industry = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=[
            "ImplementDate",
            "IndustryClassificationID",
            "Symbol",
            "IndustryCode",
        ],
        parse_dates=["ImplementDate"],
    )

    industry = industry.rename(
        columns={
            "Symbol": "Stkcd",
            "IndustryClassificationID": "std_id",
        }
    )
    industry["Stkcd"] = industry["Stkcd"].apply(normalize_stock_code)
    industry["IndustryCode"] = (
        industry["IndustryCode"].astype("string").str.strip().str.upper()
    )

    industry.loc[
        industry["IndustryCode"].isin(("", "NAN", "NONE", "<NA>")),
        "IndustryCode",
    ] = pd.NA

    industry = industry.loc[
        industry["std_id"].isin(("P0201", "P0207", "P0221"))
    ].copy()
    industry = industry.dropna(
        subset=["Stkcd", "std_id", "ImplementDate", "IndustryCode"]
    )

    duplicate_count = industry.duplicated(
        ["Stkcd", "std_id", "ImplementDate"], keep=False
    ).sum()
    if duplicate_count:
        print(
            "CSMAR 行业历史同一股票、标准和生效日存在重复记录："
            f"{duplicate_count:,} 条，保留最后一条。"
        )

    industry = (
        industry.sort_values(
            ["Stkcd", "std_id", "ImplementDate"], kind="mergesort"
        )
        .drop_duplicates(["Stkcd", "std_id", "ImplementDate"], keep="last")
        .reset_index(drop=True)
    )

    return industry


def assign_industry_standard_by_date(panel: pd.DataFrame) -> pd.DataFrame:
    """根据交易日期分配当期适用的行业分类标准。"""

    # 主流程后续不再需要未分配 std_id 的旧对象，直接原地增加字段以节省内存。
    out = panel
    out["std_id"] = np.select(
        [
            out["Trddt"] < pd.Timestamp("2012-10-26"),
            (
                out["Trddt"] >= pd.Timestamp("2012-10-26")
            ) & (
                out["Trddt"] < pd.Timestamp("2023-05-01")
            ),
            out["Trddt"] >= pd.Timestamp("2023-05-01"),
        ],
        ["P0201", "P0207", "P0221"],
        default="P0201",
    )
    return out


def match_csmar_industry_by_date(
    panel: pd.DataFrame,
    industry: pd.DataFrame,
) -> pd.DataFrame:
    """为每条日度观测向后匹配当时已经生效的 CSMAR 行业记录。"""

    if panel[["Trddt", "Stkcd", "std_id"]].isna().to_numpy().any():
        raise ValueError("主面板的 Stkcd、Trddt 或 std_id 存在缺失，不能匹配行业。")

    # 主流程不再需要匹配前的旧行序，原地排序可减少一份完整面板副本。
    panel.sort_values(
        ["Trddt", "Stkcd", "std_id"], kind="mergesort", inplace=True
    )
    panel.reset_index(drop=True, inplace=True)
    panel["_row_id"] = np.arange(len(panel), dtype=np.int64)

    # 近邻匹配只携带必要的键和内部行号，不复制收益、市值等宽表字段。
    left = panel[["_row_id", "Trddt", "Stkcd", "std_id"]].copy()
    right = (
        industry[["ImplementDate", "Stkcd", "std_id", "IndustryCode"]]
        .dropna(subset=["ImplementDate", "Stkcd", "std_id"])
        .sort_values(["ImplementDate", "Stkcd", "std_id"])
        .reset_index(drop=True)
    )

    matched_keys = pd.merge_asof(
        left,
        right,
        left_on="Trddt",
        right_on="ImplementDate",
        by=["Stkcd", "std_id"],
        direction="backward",
        allow_exact_matches=True,
    )

    if len(matched_keys) != len(left):
        raise AssertionError("CSMAR 行业匹配改变了主面板行数。")

    matched_keys.sort_values("_row_id", kind="mergesort", inplace=True)
    expected_row_ids = np.arange(len(panel), dtype=np.int64)
    if not np.array_equal(matched_keys["_row_id"].to_numpy(), expected_row_ids):
        raise AssertionError("CSMAR 行业匹配后的内部行号与主面板不一致。")

    panel["ImplementDate"] = matched_keys["ImplementDate"].to_numpy()
    panel["IndustryCode"] = matched_keys["IndustryCode"].to_numpy()
    panel.drop(columns=["_row_id"], inplace=True)

    return panel


def save_industry_panel(
    panel: pd.DataFrame,
    path: Path,
    label: str,
    encoding: str,
) -> None:
    """保存股票—交易日行业代码面板。"""

    output = panel[["Stkcd", "td", "Trddt", "IndustryCode"]]
    output.to_csv(path, index=False, encoding=encoding)
    print(f"{label}已输出：{path}，共 {len(output):,} 行。")


# =============================================================================
# 六、RESSET 行业历史整理和缺失补充
# =============================================================================


def prepare_resset_history(path: Path) -> pd.DataFrame:
    """读取并整理可用于行业代码补缺的 RESSET 行业历史。"""

    resset = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        usecols=[
            "StkCd",
            "InfoPubDt",
            "InfoSource",
            "IfPerformed",
            "EffDt",
            "CancelDt",
            "IndClsStd",
            "IndClsStdNm",
            "IndCd",
            "IndNm",
            "IndCd1",
            "IndNm1",
            "IndCd2",
            "IndNm2",
            "IndCd3",
            "IndNm3",
            "IndCd4",
            "IndNm4",
        ],
    )

    resset["Stkcd"] = resset["StkCd"].apply(normalize_stock_code)
    resset["EffDt"] = pd.to_datetime(resset["EffDt"], errors="coerce")
    resset["CancelDt"] = pd.to_datetime(resset["CancelDt"], errors="coerce")

    resset = resset.loc[resset["IndClsStd"].isin(("1", "22", "44"))].copy()
    resset["std_id"] = resset["IndClsStd"].map(
        {"1": "P0201", "22": "P0207", "44": "P0221"}
    )

    code_columns = ["IndCd", "IndCd1", "IndCd2", "IndCd3", "IndCd4"]
    for col in code_columns:
        resset[col] = resset[col].astype("string").str.strip().str.upper()
        resset.loc[
            resset[col].isin(("", "NAN", "NONE", "<NA>")), col
        ] = pd.NA

    resset["resset_code"] = resset["IndCd"]

    is_p0221 = resset["std_id"].eq("P0221")
    resset.loc[is_p0221, "resset_code"] = (
        resset.loc[
            is_p0221,
            ["IndCd4", "IndCd3", "IndCd2", "IndCd1", "IndCd"],
        ]
        .bfill(axis=1)
        .iloc[:, 0]
    )

    need_normalization = (
        resset["std_id"].eq("P0221")
        & resset["resset_code"].str.match(r"^[A-Z][A-Z][0-9]{2}$", na=False)
    )
    resset.loc[need_normalization, "resset_code"] = (
        resset.loc[need_normalization, "resset_code"].str[0]
        + resset.loc[need_normalization, "resset_code"].str[-2:]
    )

    resset = resset.dropna(
        subset=["Stkcd", "std_id", "EffDt", "resset_code"]
    ).copy()

    duplicate_count = resset.duplicated(
        ["Stkcd", "std_id", "EffDt"], keep=False
    ).sum()
    if duplicate_count:
        print(
            "RESSET 行业历史同一股票、标准和生效日存在重复记录："
            f"{duplicate_count:,} 条，保留最后一条。"
        )

    resset = (
        resset.sort_values(["Stkcd", "std_id", "EffDt"], kind="mergesort")
        .drop_duplicates(["Stkcd", "std_id", "EffDt"], keep="last")
        .sort_values(["EffDt", "Stkcd", "std_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    return resset


def empty_resset_detail() -> pd.DataFrame:
    """返回与正式补缺明细相同字段的空表。"""

    return pd.DataFrame(
        columns=[
            "index",
            "Stkcd",
            "Trddt",
            "std_id",
            "FilledIndustryCode",
            "IndNm",
            "IndNm1",
            "IndNm2",
            "IndNm3",
            "IndNm4",
            "RESSET_EffDt",
            "RESSET_CancelDt",
            "RESSET_InfoSource",
            "RESSET_IfPerformed",
        ]
    )


def fill_missing_industry(
    panel: pd.DataFrame,
    resset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """只使用 RESSET 补充原本缺失的行业代码，并生成补缺明细。"""

    original_row_count = len(panel)

    # 主流程后续只使用补缺结果，因此直接在传入面板上更新，降低峰值内存。
    panel_filled = panel
    panel_filled.reset_index(drop=True, inplace=True)
    panel_filled["_row_id"] = np.arange(len(panel_filled), dtype=np.int64)
    panel_filled["IndustryCode"] = (
        panel_filled["IndustryCode"].astype("string").str.strip().str.upper()
    )
    panel_filled.loc[
        panel_filled["IndustryCode"].isin(("", "NAN", "NONE", "<NA>")),
        "IndustryCode",
    ] = pd.NA

    missing = panel_filled.loc[panel_filled["IndustryCode"].isna()].copy()
    original_missing = len(missing)

    if missing.empty:
        panel_filled.drop(columns=["_row_id"], inplace=True)
        return panel_filled, empty_resset_detail(), 0, 0

    left = (
        missing.sort_values(["Trddt", "Stkcd", "std_id"])
        .reset_index(drop=True)
    )
    right = (
        resset[
            [
                "Stkcd",
                "std_id",
                "EffDt",
                "CancelDt",
                "resset_code",
                "IndNm",
                "IndNm1",
                "IndNm2",
                "IndNm3",
                "IndNm4",
                "InfoSource",
                "IfPerformed",
            ]
        ]
        .sort_values(["EffDt", "Stkcd", "std_id"])
        .reset_index(drop=True)
    )

    matched = pd.merge_asof(
        left,
        right,
        left_on="Trddt",
        right_on="EffDt",
        by=["Stkcd", "std_id"],
        direction="backward",
        allow_exact_matches=True,
    )

    valid_match = matched["EffDt"].notna() & (
        matched["CancelDt"].isna()
        | matched["Trddt"].le(matched["CancelDt"])
    )

    fill_index = matched.loc[valid_match, "_row_id"].astype(np.int64)
    fill_code = matched.loc[valid_match, "resset_code"].to_numpy()
    panel_filled.loc[fill_index, "IndustryCode"] = fill_code

    detail = matched.loc[
        valid_match,
        [
            "_row_id",
            "Stkcd",
            "Trddt",
            "std_id",
            "resset_code",
            "IndNm",
            "IndNm1",
            "IndNm2",
            "IndNm3",
            "IndNm4",
            "EffDt",
            "CancelDt",
            "InfoSource",
            "IfPerformed",
        ],
    ].copy()
    detail = detail.rename(
        columns={
            "_row_id": "index",
            "resset_code": "FilledIndustryCode",
            "EffDt": "RESSET_EffDt",
            "CancelDt": "RESSET_CancelDt",
            "InfoSource": "RESSET_InfoSource",
            "IfPerformed": "RESSET_IfPerformed",
        }
    )

    if detail["index"].duplicated().any():
        raise AssertionError("RESSET 补缺明细中的原面板行号不唯一。")
    if len(panel_filled) != original_row_count:
        raise AssertionError("RESSET 补缺改变了主面板行数。")

    filled_count = len(detail)
    remaining_missing = panel_filled["IndustryCode"].isna().sum()
    if original_missing - remaining_missing != filled_count:
        raise AssertionError("RESSET 补缺明细数与行业代码实际新增非缺失数不一致。")

    panel_filled.drop(columns=["_row_id"], inplace=True)
    return panel_filled, detail, original_missing, filled_count


def save_resset_outputs(
    panel_filled: pd.DataFrame,
    detail: pd.DataFrame,
    panel_path: Path,
    detail_path: Path,
    original_missing: int,
    filled_count: int,
) -> None:
    """保存 RESSET 补缺后的行业面板、补缺明细和质量统计。"""

    output_panel = panel_filled[["Stkcd", "td", "Trddt", "IndustryCode"]]
    output_panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    remaining_missing = output_panel["IndustryCode"].isna().sum()
    print("RESSET 行业代码补缺统计：")
    print(f"  原始缺失条数：{original_missing:,}")
    print(f"  成功补缺条数：{filled_count:,}")
    print(f"  补缺后缺失条数：{remaining_missing:,}")
    print(f"补缺后行业面板已输出：{panel_path}")
    print(f"补缺明细已输出：{detail_path}")


# =============================================================================
# 七、金融行业、上市年限和交易状态筛选
# =============================================================================


def calculate_financial_flag(panel: pd.DataFrame) -> pd.DataFrame:
    """根据补缺后的最终行业代码重新计算金融行业标记。"""

    # 补缺前面板在主流程中不再单独使用，原地计算可显著降低峰值内存。
    out = panel
    out["IndustryCode"] = (
        out["IndustryCode"].astype("string").str.strip().str.upper()
    )

    has_industry = out["IndustryCode"].notna()
    is_financial = (
        out["std_id"].eq("P0201")
        & out["IndustryCode"].str.startswith("I", na=False)
    ) | (
        out["std_id"].isin(("P0207", "P0221"))
        & out["IndustryCode"].str.startswith("J", na=False)
    )

    out["is_fin"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[has_industry, "is_fin"] = 0
    out.loc[is_financial, "is_fin"] = 1

    return out


def report_industry_match_quality(panel: pd.DataFrame) -> tuple[int, int]:
    """报告研究期内 CSMAR 加 RESSET 补缺后的行业匹配情况。"""

    report_mask = get_report_period_mask(panel)
    success_mask = report_mask & panel["is_fin"].notna()
    fail_mask = report_mask & panel["is_fin"].isna()
    total = int(report_mask.sum())
    success_count = int(success_mask.sum())
    fail_count = int(fail_mask.sum())
    success_rate = success_count / total if total else np.nan
    fail_rate = fail_count / total if total else np.nan

    print(f"行业匹配报告期：{report_period_label()}")
    print(f"行业匹配总观测数：{total:,}")
    print(f"行业匹配成功：{success_count:,}（占比 {success_rate:.2%}）")
    print(f"行业匹配失败：{fail_count:,}（占比 {fail_rate:.2%}）")

    if fail_count:
        failed_stocks = panel.loc[fail_mask, "Stkcd"]
        print(f"行业匹配失败股票数：{failed_stocks.nunique():,}")
        print("行业匹配失败次数最多的前 20 只股票：")
        print(failed_stocks.value_counts().head(20).to_string())

    return success_count, fail_count


def drop_financial_observations(panel: pd.DataFrame) -> pd.DataFrame:
    """删除明确识别为金融业的股票—日期观测；行业仍缺失的观测保留。"""

    financial_mask = panel["is_fin"].fillna(0).eq(1)
    report_mask = get_report_period_mask(panel)
    report_financial_mask = report_mask & financial_mask

    before = int(report_mask.sum())
    financial_observations = int(report_financial_mask.sum())
    financial_stocks = panel.loc[report_financial_mask, "Stkcd"].nunique()

    out = panel.loc[~financial_mask].copy()
    out = out.drop(
        columns=["std_id", "ImplementDate", "is_fin"],
        errors="ignore",
    )

    print(f"金融行业清洗报告期：{report_period_label()}")
    print(f"识别为金融行业的股票数：{financial_stocks:,}")
    print(f"删除金融行业股票—日期观测：{financial_observations:,} 条。")
    print(f"报告期样本数从 {before:,} 下降为 {before - financial_observations:,}。")

    return out


def load_company_listing_dates(path: Path) -> pd.DataFrame:
    """读取公司上市日期，并标准化股票代码。"""

    company = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=["Stkcd", "Listdt"],
        parse_dates=["Listdt"],
    )
    company["Stkcd"] = company["Stkcd"].apply(normalize_stock_code)
    return company


def report_listing_data_quality(company: pd.DataFrame) -> pd.DataFrame:
    """检查公司上市日期表中的重复股票代码。"""

    check = company[["Stkcd", "Listdt"]].copy()
    duplicate_mask = check.duplicated("Stkcd", keep=False)
    duplicates = check.loc[duplicate_mask].sort_values(["Stkcd", "Listdt"])

    print(f"公司上市日期表总行数：{len(check):,}")
    print(f"公司上市日期表股票数：{check['Stkcd'].nunique():,}")
    print(f"重复行数：{len(duplicates):,}")
    print(f"重复股票数：{duplicates['Stkcd'].nunique():,}")

    if not duplicates.empty:
        print("重复股票代码示例：")
        print(duplicates.head(50))

    return duplicates


def attach_company_listing_dates(
    panel: pd.DataFrame,
    company: pd.DataFrame,
) -> pd.DataFrame:
    """按照股票代码将上市日期匹配到日度面板。"""

    return pd.merge(
        panel,
        company,
        how="left",
        on="Stkcd",
        validate="many_to_one",
    )


def keep_listed_at_least_one_year(panel: pd.DataFrame) -> pd.DataFrame:
    """仅保留交易日时已经上市满一个日历年的观测。"""

    out = panel.copy()
    out["Listdt"] = pd.to_datetime(out["Listdt"], errors="coerce")
    report_mask = get_report_period_mask(out)
    before = int(report_mask.sum())

    unmatched_mask = report_mask & out["Listdt"].isna()
    unmatched_observations = int(unmatched_mask.sum())
    unmatched_stocks = out.loc[unmatched_mask, "Stkcd"].nunique()

    out["Listdt_1y"] = out["Listdt"] + pd.DateOffset(years=1)
    listed_at_least_one_year = out["Trddt"].ge(out["Listdt_1y"])
    listed_under_one_year_mask = (
        report_mask
        & out["Listdt"].notna()
        & ~listed_at_least_one_year
    )
    listed_under_one_year = int(listed_under_one_year_mask.sum())

    out = out.loc[listed_at_least_one_year].copy()
    out = out.drop(columns=["Listdt_1y"])

    total_removed = unmatched_observations + listed_under_one_year
    print(f"上市年限清洗报告期：{report_period_label()}")
    print(f"未匹配上市日期的观测：{unmatched_observations:,} 条。")
    print(f"未匹配上市日期的股票：{unmatched_stocks:,} 只。")
    print(f"删除上市不满一年观测：{listed_under_one_year:,} 条。")
    print(f"报告期样本数从 {before:,} 下降为 {before - total_removed:,}。")

    return out


def drop_st_pt_trading_status(
    panel: pd.DataFrame,
    allowed_statuses: tuple[int, ...] = ALLOWED_TRADING_STATUSES,
) -> pd.DataFrame:
    """仅保留允许的交易状态观测。"""

    allowed_mask = panel["Trdsta"].isin(allowed_statuses)
    report_mask = get_report_period_mask(panel)
    before = int(report_mask.sum())
    removed = int((report_mask & ~allowed_mask).sum())

    out = panel.loc[allowed_mask].copy()
    print(f"ST/PT 交易状态清洗报告期：{report_period_label()}")
    print(f"删除 ST/PT 等异常交易状态观测：{removed:,} 条。")
    print(f"报告期样本数从 {before:,} 下降为 {before - removed:,}。")
    return out


# =============================================================================
# 八、基础样本池分支
# =============================================================================


def keep_non_filled_observations(panel: pd.DataFrame) -> pd.DataFrame:
    """基础样本池分支仅保留 Filling 等于 0 的真实交易观测。"""

    non_filled_mask = panel["Filling"].eq(0)
    report_mask = get_report_period_mask(panel)
    before = int(report_mask.sum())
    removed = int((report_mask & ~non_filled_mask).sum())

    out = panel.loc[non_filled_mask].copy()
    print(f"停牌或暂停上市清洗报告期：{report_period_label()}")
    print(f"删除 Filling=1 的停牌或暂停上市观测：{removed:,} 条。")
    print(f"报告期样本数从 {before:,} 下降为 {before - removed:,}。")
    return out


def build_base_sample_index(eligible_panel: pd.DataFrame) -> pd.DataFrame:
    """生成只包含 Stkcd、Trddt 和 td 的基础样本索引。"""

    observed_panel = keep_non_filled_observations(eligible_panel)
    output = observed_panel[["Stkcd", "Trddt", "td"]].dropna(subset=["td"]).copy()
    output["td"] = output["td"].astype(np.int64)
    output = output.sort_values(["Stkcd", "td"]).reset_index(drop=True)

    require_unique_key(output, ("Stkcd", "td"), "基础清洗后样本索引")
    return output


def save_base_sample_index(panel: pd.DataFrame, path: Path) -> None:
    """保存基础清洗后的股票—交易日样本索引。"""

    panel.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"基础清洗后样本索引已输出：{path}，共 {len(panel):,} 行。")


# =============================================================================
# 九、窗口变量目标日资格索引分支
# =============================================================================

def build_window_variable_target_index(
    eligible_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    生成窗口变量目标日资格索引。

    eligible_panel 已完成：
    1. A股市场筛选；
    2. 明确金融行业观测剔除；
    3. 上市满一年筛选；
    4. ST/PT等异常交易状态筛选。

    本分支不执行 Filling=0 筛选，因此保留 Filling=0/1。
    """

    require_columns(
        eligible_panel,
        ("Stkcd", "Trddt", "td", "Filling"),
        "窗口变量目标日资格面板",
    )

    invalid_filling = ~eligible_panel["Filling"].isin((0, 1))
    if invalid_filling.any():
        raise ValueError(
            "窗口变量目标日资格面板存在 Filling 不属于 0/1 的观测。"
        )

    output = eligible_panel[
        ["Stkcd", "Trddt", "td"]
    ].dropna(
        subset=["Stkcd", "Trddt", "td"]
    ).copy()

    output["td"] = output["td"].astype(np.int64)
    output = output.sort_values(
        ["Stkcd", "td"]
    ).reset_index(drop=True)

    require_unique_key(
        output,
        ("Stkcd", "td"),
        "窗口变量目标日资格索引",
    )

    return output


def save_window_variable_target_index(
    panel: pd.DataFrame,
    path: Path,
) -> None:
    """保存窗口变量目标日资格索引。"""

    panel.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "窗口变量目标日资格索引已输出："
        f"{path}，共 {len(panel):,} 行。"
    )


# =============================================================================
# 十、DGTW 标签面板分支
# =============================================================================
def restore_cleaned_to_full_panel(
    full_panel: pd.DataFrame,
    cleaned_panel: pd.DataFrame,
) -> pd.DataFrame:
    """把合格观测恢复到完整面板，不合格日期仅保留三个索引字段。"""

    require_unique_key(cleaned_panel, ("Stkcd", "td"), "资格清洗后的日度面板")

    valid_stocks = cleaned_panel["Stkcd"].drop_duplicates()
    valid_full_panel = full_panel.loc[
        full_panel["Stkcd"].isin(valid_stocks)
    ].copy()

    restored = pd.merge(
        valid_full_panel,
        cleaned_panel.drop(columns=["Trddt"]),
        how="left",
        on=["Stkcd", "td"],
        validate="one_to_one",
    )

    return restored


def add_strict_td_window_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """在完整 td 面板上计算严格连续交易日的滞后和未来复合收益。"""

    # 每个批次规模有限，原地排序后直接增加收益字段，避免复制宽面板。
    out = panel
    out.sort_values(["Stkcd", "td"], kind="mergesort", inplace=True)
    out.reset_index(drop=True, inplace=True)
    out["raw_ret_t"] = out["ret"]
    out["_gross_ret"] = 1 + out["ret"]

    grouped = out.groupby("Stkcd", sort=False)

    for offset in LAG_RETURN_OFFSETS:
        lagged_return = grouped["ret"].shift(offset)
        lagged_td = grouped["td"].shift(offset)
        strict_lag = lagged_td.eq(out["td"] - offset)
        out[f"raw_ret_tm{offset}"] = lagged_return.where(strict_lag)

    current_product = pd.Series(1.0, index=out.index)
    current_nobs = pd.Series(0, index=out.index, dtype="Int16")

    for horizon in range(0, max(FORWARD_RETURN_HORIZONS) + 1):
        shifted_gross = grouped["_gross_ret"].shift(-horizon)
        shifted_td = grouped["td"].shift(-horizon)
        valid_horizon = shifted_gross.notna() & shifted_td.eq(out["td"] + horizon)

        current_product = current_product * shifted_gross.where(valid_horizon, 1.0)
        current_nobs = (
            current_nobs + valid_horizon.astype("int8")
        ).astype("Int16")

        if horizon not in FORWARD_RETURN_HORIZONS:
            continue

        output_column = f"raw_ret_t_t{horizon}"
        window_length = horizon + 1
        valid_full_window = current_nobs.eq(window_length)

        if horizon in FORWARD_RETURN_DIAGNOSTIC_HORIZONS:
            out[f"{output_column}_nobs"] = current_nobs
            out[f"{output_column}_valid_full"] = valid_full_window

        out[output_column] = np.where(
            valid_full_window,
            current_product - 1,
            np.nan,
        )

    out.drop(columns=["_gross_ret"], inplace=True)
    return out


def add_td_lagged_market_caps(panel: pd.DataFrame) -> pd.DataFrame:
    """提取严格对应当前 td 减 3 的总市值和流通市值。"""

    # 收益窗口函数已经把同一批次按 Stkcd、td 排好序，这里直接原地增加字段。
    out = panel
    lag = 3
    grouped = out.groupby("Stkcd", sort=False)
    lagged_td = grouped["td"].shift(lag)
    strict_lag = lagged_td.eq(out["td"] - lag)

    out["mktcap_tm3"] = grouped["size"].shift(lag).where(strict_lag)
    out["float_mktcap_tm3"] = grouped["sizew"].shift(lag).where(strict_lag)
    out["mktcap_tm3_valid"] = strict_lag & out["mktcap_tm3"].notna()
    out["float_mktcap_tm3_valid"] = (
        strict_lag & out["float_mktcap_tm3"].notna()
    )

    return out


def finalize_dgtw_output_columns(
    panel: pd.DataFrame,
    report_missing_returns: bool = True,
) -> pd.DataFrame:
    """删除仅用于清洗和诊断的临时字段。"""

    temporary_columns = [
        "Trdsta",
        "Filling",
        "has_volume",
        "Listdt",
        "std_id",
        "ImplementDate",
        "is_fin",
    ]
    panel.drop(columns=temporary_columns, errors="ignore", inplace=True)

    if report_missing_returns:
        print(f"最终 DGTW 面板收益率缺失观测：{panel['ret'].isna().sum():,} 条。")
    return panel


def build_and_save_dgtw_panel_in_batches(
    full_panel: pd.DataFrame,
    eligible_panel: pd.DataFrame,
    path: Path,
    stock_batch_size: int = DGTW_STOCK_BATCH_SIZE,
) -> None:
    """按股票分批构造并保存 DGTW 宽面板，降低峰值内存。"""

    if stock_batch_size <= 0:
        raise ValueError("stock_batch_size 必须是正整数。")

    valid_stocks = np.sort(
        eligible_panel["Stkcd"].dropna().astype(str).unique()
    )
    if len(valid_stocks) == 0:
        raise ValueError("资格清洗后没有任何股票，无法构造 DGTW 面板。")

    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.unlink(missing_ok=True)

    total_batches = (len(valid_stocks) + stock_batch_size - 1) // stock_batch_size
    total_rows = 0
    total_missing_returns = 0
    expected_output_columns: tuple[str, ...] | None = None

    # 两张输入表都按股票代码排好序，后续用二分边界切片，避免每批扫描全表。
    full_panel.sort_values(["Stkcd", "td"], kind="mergesort", inplace=True)
    full_panel.reset_index(drop=True, inplace=True)
    eligible_panel.sort_values(
        ["Stkcd", "td"], kind="mergesort", inplace=True
    )
    eligible_panel.reset_index(drop=True, inplace=True)

    full_stock_codes = full_panel["Stkcd"].to_numpy()
    eligible_stock_codes = eligible_panel["Stkcd"].to_numpy()
    full_rows_by_stock = full_panel.groupby("Stkcd", sort=False).size()
    valid_stock_row_counts = full_rows_by_stock.reindex(valid_stocks)
    if valid_stock_row_counts.isna().any():
        raise AssertionError("资格面板中的部分股票不存在于完整股票交易日面板。")
    expected_total_rows = int(valid_stock_row_counts.sum())

    try:
        for batch_number, start in enumerate(
            range(0, len(valid_stocks), stock_batch_size),
            start=1,
        ):
            batch_stocks = valid_stocks[start : start + stock_batch_size]
            first_stock = batch_stocks[0]
            last_stock = batch_stocks[-1]

            full_start = np.searchsorted(full_stock_codes, first_stock, side="left")
            full_end = np.searchsorted(full_stock_codes, last_stock, side="right")
            eligible_start = np.searchsorted(
                eligible_stock_codes, first_stock, side="left"
            )
            eligible_end = np.searchsorted(
                eligible_stock_codes, last_stock, side="right"
            )

            batch_full_panel = full_panel.iloc[full_start:full_end]
            batch_eligible_panel = eligible_panel.iloc[
                eligible_start:eligible_end
            ]

            batch_output = restore_cleaned_to_full_panel(
                batch_full_panel,
                batch_eligible_panel,
            )
            batch_output = add_strict_td_window_returns(batch_output)
            batch_output = add_td_lagged_market_caps(batch_output)
            batch_output = finalize_dgtw_output_columns(
                batch_output,
                report_missing_returns=False,
            )

            require_unique_key(
                batch_output,
                ("Stkcd", "td"),
                f"DGTW 第 {batch_number} 批面板",
            )

            output_stocks = np.sort(batch_output["Stkcd"].dropna().unique())
            if not np.array_equal(output_stocks, batch_stocks):
                raise AssertionError(
                    f"DGTW 第 {batch_number} 批输出股票集合与计划批次不一致。"
                )

            current_columns = tuple(batch_output.columns)
            if expected_output_columns is None:
                expected_output_columns = current_columns
            elif current_columns != expected_output_columns:
                raise AssertionError(
                    f"DGTW 第 {batch_number} 批字段或字段顺序与首批不一致。"
                )

            write_header = batch_number == 1
            batch_output.to_csv(
                temporary_path,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
                encoding="utf-8",
            )

            total_rows += len(batch_output)
            total_missing_returns += int(batch_output["ret"].isna().sum())

            print(
                f"DGTW 批次 {batch_number:,}/{total_batches:,} 已写出："
                f"{len(batch_output):,} 行，股票 {len(batch_stocks):,} 只。"
            )

            del batch_full_panel, batch_eligible_panel, batch_output

        if total_rows != expected_total_rows:
            raise AssertionError(
                "DGTW 分批输出总行数与有效股票在完整面板中的预期行数不一致："
                f"预期 {expected_total_rows:,}，实际 {total_rows:,}。"
            )

        temporary_path.replace(path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"最终 DGTW 面板收益率缺失观测：{total_missing_returns:,} 条。")
    print(f"DGTW 构建用日度面板已输出：{path}，共 {total_rows:,} 行。")


# =============================================================================
# 十一、统一入口
# =============================================================================


def run_unified_pipeline(staging_paths: dict[str, Path]) -> None:
    """把全部处理结果写入暂存路径，并在末尾统一提交。"""

    validate_input_paths()

    print_stage(1, "运行前检查公司上市日期表")
    company = load_company_listing_dates(COMPANY_PATH)
    duplicate_listing_dates = report_listing_data_quality(company)
    if not duplicate_listing_dates.empty:
        raise ValueError(
            "公司上市日期表存在重复股票代码，请先确认每只股票应保留的上市日期。"
        )

    print_stage(2, "生成 A 股市场交易日历暂存文件")
    trading_calendar = build_a_share_trading_calendar(RAW_CALENDAR_PATH)
    save_trading_calendar(trading_calendar, staging_paths["calendar"])

    print_stage(3, "一次性读取日度数据并构造完整股票—交易日面板")
    daily_data = load_and_prepare_daily_data(DAILY_PATH)
    daily_with_td = attach_trading_day_index(daily_data, trading_calendar)
    del daily_data

    full_panel, complete_daily_panel = build_complete_stock_td_panel(
        daily_with_td,
        trading_calendar,
    )
    save_full_panel(full_panel, staging_paths["full_panel"])
    report_panel_expansion(daily_with_td, complete_daily_panel)
    report_td_continuity(full_panel, "完整股票交易日索引面板")

    # 后续不再使用带 td 的原始日度表，释放引用以降低内存占用。
    del daily_with_td

    print_stage(4, "在资格筛选前计算近一个月和近一年成交天数")
    complete_daily_panel = add_trade_day_counts(complete_daily_panel)

    print_stage(5, "匹配 CSMAR 行业历史")
    csmar_industry = load_and_prepare_csmar_industry(CSMAR_INDUSTRY_PATH)
    complete_daily_panel = assign_industry_standard_by_date(complete_daily_panel)
    complete_daily_panel = match_csmar_industry_by_date(
        complete_daily_panel,
        csmar_industry,
    )
    save_industry_panel(
        complete_daily_panel,
        staging_paths["csmar_industry"],
        label="CSMAR 行业代码完整面板",
        encoding="utf-8",
    )
    del csmar_industry

    print_stage(6, "使用 RESSET 补充缺失行业代码")
    resset_industry = prepare_resset_history(RESSET_INDUSTRY_PATH)
    (
        complete_daily_panel,
        resset_detail,
        original_missing,
        filled_count,
    ) = fill_missing_industry(complete_daily_panel, resset_industry)

    save_resset_outputs(
        panel_filled=complete_daily_panel,
        detail=resset_detail,
        panel_path=staging_paths["resset_filled"],
        detail_path=staging_paths["resset_detail"],
        original_missing=original_missing,
        filled_count=filled_count,
    )
    del resset_industry, resset_detail

    print_stage(7, "依据补缺后的行业代码删除金融行业")
    complete_daily_panel = calculate_financial_flag(complete_daily_panel)
    report_industry_match_quality(complete_daily_panel)
    eligible_observed_panel = drop_financial_observations(complete_daily_panel)
    del complete_daily_panel

    print_stage(8, "筛选上市满一年且交易状态合格的观测")
    eligible_observed_panel = attach_company_listing_dates(
        eligible_observed_panel,
        company,
    )
    eligible_observed_panel = keep_listed_at_least_one_year(
        eligible_observed_panel
    )
    eligible_observed_panel = drop_st_pt_trading_status(
        eligible_observed_panel
    )
    del company

    print_stage(9, "生成并保存基础清洗后的样本索引")
    base_sample_index = build_base_sample_index(eligible_observed_panel)
    report_td_continuity(base_sample_index, "基础清洗后样本索引")
    save_base_sample_index(base_sample_index, staging_paths["base_sample"])
    del base_sample_index


    print_stage(10, "生成并保存窗口变量目标日资格索引")

    window_variable_target_index = build_window_variable_target_index(
        eligible_observed_panel
    )

    report_td_continuity(
        window_variable_target_index,
        "窗口变量目标日资格索引",
    )

    save_window_variable_target_index(
        window_variable_target_index,
        staging_paths["window_variable_target_index"],
    )

    del window_variable_target_index


    print_stage(11, "按股票分批构造并保存 DGTW 宽面板")

    build_and_save_dgtw_panel_in_batches(
        full_panel=full_panel,
        eligible_panel=eligible_observed_panel,
        path=staging_paths["dgtw"],
    )

    del full_panel, eligible_observed_panel


    print_stage(12, "统一提交八个正式输出")
    commit_staging_outputs(staging_paths)

    print("\n全部输出步骤已执行完毕。")


def main() -> None:
    """在独占锁保护下，一次运行完成三份原脚本的全部功能。"""

    prepare_output_directories()
    process_lock = acquire_output_process_lock()
    staging_paths = build_staging_output_paths()

    try:
        recover_interrupted_output_commit()
        cleanup_staging_outputs(staging_paths)
        run_unified_pipeline(staging_paths)
    finally:
        # 清理发生在释放锁之前，避免下一进程与本进程争用暂存文件。
        try:
            cleanup_staging_outputs(staging_paths)
        finally:
            release_output_process_lock(process_lock)


if __name__ == "__main__":
    main()
