import argparse
from pathlib import Path

import numpy as np
import pandas as pd


"""
计算 DGTW 特征调整收益率。

本脚本只负责最后一步：把个股原始收益减去同日同 DGTW 组合的基准收益。
DGTW 组合标签和组合基准收益已经由 01、02 两个脚本提前算好。
"""
ROOT = Path(__file__).resolve().parents[4]

base_sample_index_path = ROOT / "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv"
daily_variable_panel_path = ROOT / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv"
portfolio_assign_path = ROOT / "中间结果/02_变量构造结果/DGTW/DGTW形成期组合标签.csv"
benchmark_path = ROOT / "中间结果/02_变量构造结果/DGTW/DGTW组合基准收益.csv"
adjusted_return_out_path = ROOT / "中间结果/02_变量构造结果/DGTW/DGTW特征调整收益率.csv"

LAG_RETURN_OFFSETS = (3, 2, 1)
FORWARD_RETURN_HORIZONS = tuple(range(1, 51))
RETURN_WINDOW_SPECS = (
    tuple((f"raw_ret_tm{i}", f"bench_ret_tm{i}", f"chAdj_ret_tm{i}") for i in LAG_RETURN_OFFSETS)
    + (("raw_ret_t", "bench_ret_t", "chAdj_ret_t"),)
    + tuple(
        (f"raw_ret_t_t{i}", f"bench_ret_t_t{i}", f"chAdj_ret_t_t{i}")
        for i in FORWARD_RETURN_HORIZONS
    )
)
RAW_RETURN_COLUMNS = [raw_col for raw_col, _, _ in RETURN_WINDOW_SPECS]
BENCHMARK_COLUMNS = [bench_col for _, bench_col, _ in RETURN_WINDOW_SPECS]
ADJUSTED_RETURN_COLUMNS = [out_col for _, _, out_col in RETURN_WINDOW_SPECS]
OUTPUT_COLUMNS = [
    "Stkcd",
    "Trddt",
    "td",
    "formation_year",
    "portfolioNumber",
    *ADJUSTED_RETURN_COLUMNS,
]


def normalize_stock_code_series(series):
    """将股票代码统一成六位字符串，保留前导 0。"""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def require_columns(df, columns, name):
    """检查输入数据是否包含必要字段，避免后续 merge 或计算时静默失败。"""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def assert_unique_key(df, key_cols, name):
    """检查指定键是否唯一，保证后续 validate 参数和数据层级一致。"""
    duplicate_count = df.duplicated(key_cols).sum()
    if duplicate_count:
        raise ValueError(f"{name} has duplicate keys {key_cols}: {duplicate_count:,}")


def add_holding_formation_year(df):
    """
    生成交易日对应的 DGTW 形成年度。

    DGTW 组合每年 7 月开始生效，到下一年 6 月结束；
    因此 1-6 月交易日使用上一年 6 月形成的组合标签。
    """
    out = df.copy()
    trddt = pd.to_datetime(out["Trddt"], errors="coerce")
    out["formation_year"] = np.where(
        trddt.dt.month <= 6,
        trddt.dt.year - 1,
        trddt.dt.year,
    ).astype("int16")
    return out


def load_sample_keys():
    """
    读取基础清洗样本并匹配 DGTW 组合标签。

    基础清洗样本定义最终可计算的股票-交易日范围。这里先只保留键和日期，
    避免把大收益面板一次性读入内存。组合标签是一只股票在某个
    formation_year 的静态标签；合并后仍要检查交易日是否落在该标签
    的持有期内。
    """
    sample = pd.read_csv(
        base_sample_index_path,
        encoding="utf-8-sig",
        usecols=["Stkcd", "Trddt", "td"],
        dtype={"Stkcd": "string"},
        parse_dates=["Trddt"],
    )
    require_columns(sample, ["Stkcd", "Trddt", "td"], "base sample index")

    sample["Stkcd"] = normalize_stock_code_series(sample["Stkcd"])
    sample["td"] = pd.to_numeric(sample["td"], errors="coerce").astype("int64")
    sample = sample.dropna(subset=["Stkcd", "Trddt", "td"])
    assert_unique_key(sample, ["Stkcd", "td"], "base sample index")

    sample = add_holding_formation_year(sample)

    portfolio = pd.read_csv(
        portfolio_assign_path,
        encoding="utf-8-sig",
        usecols=["Stkcd", "formation_year", "dateBeg", "dateEnd", "portfolioNumber"],
        dtype={"Stkcd": "string"},
        parse_dates=["dateBeg", "dateEnd"],
    )
    require_columns(
        portfolio,
        ["Stkcd", "formation_year", "dateBeg", "dateEnd", "portfolioNumber"],
        "DGTW portfolio assignment",
    )

    portfolio["Stkcd"] = normalize_stock_code_series(portfolio["Stkcd"])
    portfolio["formation_year"] = pd.to_numeric(
        portfolio["formation_year"],
        errors="coerce",
    ).astype("int16")
    portfolio["portfolioNumber"] = pd.to_numeric(
        portfolio["portfolioNumber"],
        errors="coerce",
    ).astype("Int16")
    assert_unique_key(portfolio, ["Stkcd", "formation_year"], "DGTW portfolio assignment")

    sample = sample.merge(
        portfolio,
        on=["Stkcd", "formation_year"],
        how="left",
        validate="m:1",
    )

    valid_portfolio = (
        sample["portfolioNumber"].notna()
        & sample["dateBeg"].notna()
        & sample["dateEnd"].notna()
        & sample["Trddt"].between(sample["dateBeg"], sample["dateEnd"])
    )
    sample = sample.loc[
        valid_portfolio,
        ["Stkcd", "td", "Trddt", "formation_year", "portfolioNumber"],
    ].copy()
    sample["portfolioNumber"] = sample["portfolioNumber"].astype("int16")
    assert_unique_key(sample, ["Stkcd", "td"], "sample keys with DGTW portfolio")
    return sample


def load_benchmark_returns():
    """
    读取交易日 × DGTW 组合层面的市值加权基准收益。

    后续按 Trddt + portfolioNumber 合并到个股交易日，使每只股票使用
    自己所在 DGTW 组合在同一交易日的 benchmark。
    """
    benchmark = pd.read_csv(
        benchmark_path,
        encoding="utf-8-sig",
        usecols=["Trddt", "portfolioNumber", *BENCHMARK_COLUMNS],
        parse_dates=["Trddt"],
    )
    require_columns(
        benchmark,
        ["Trddt", "portfolioNumber", *BENCHMARK_COLUMNS],
        "DGTW benchmark returns",
    )
    benchmark["portfolioNumber"] = pd.to_numeric(
        benchmark["portfolioNumber"],
        errors="coerce",
    ).astype("int16")
    assert_unique_key(benchmark, ["Trddt", "portfolioNumber"], "DGTW benchmark returns")
    return benchmark


def calculate_adjusted_returns(df, specs=RETURN_WINDOW_SPECS):
    """
    按窗口逐列计算特征调整收益率。

    核心公式为 characteristic-adjusted return =
    raw stock return - portfolio benchmark return。t、t+1、t+10 等窗口
    都使用 RETURN_WINDOW_SPECS 中的映射规则逐列相减。
    """
    out = df.copy()
    require_columns(
        out,
        [column for raw_col, bench_col, _ in specs for column in (raw_col, bench_col)],
        "adjusted return input",
    )
    for raw_col, bench_col, adjusted_col in specs:
        out[adjusted_col] = out[raw_col] - out[bench_col]
    return out


def append_adjusted_return_chunks(sample_keys, benchmark, chunk_size):
    """
    按块读取大日度面板并流式写出特征调整收益率。

    DGTW 构建用日度面板很大，按块读取原始收益；sample_keys 和 benchmark
    相对稳定且更小，可以常驻内存用于每块匹配。第一块写表头，后续块追加，
    避免一次性构造完整大表。
    """
    adjusted_return_out_path.parent.mkdir(parents=True, exist_ok=True)
    if adjusted_return_out_path.exists():
        adjusted_return_out_path.unlink()

    total_input_rows = 0
    total_output_rows = 0
    missing_stats = {
        "chAdj_ret_t": 0,
        "chAdj_ret_t_t1": 0,
        "chAdj_ret_t_t10": 0,
    }
    first_chunk = True

    reader = pd.read_csv(
        daily_variable_panel_path,
        encoding="utf-8",
        usecols=["Stkcd", "td", *RAW_RETURN_COLUMNS],
        dtype={"Stkcd": "string"},
        chunksize=chunk_size,
    )

    for chunk_number, raw_chunk in enumerate(reader, start=1):
        total_input_rows += len(raw_chunk)
        raw_chunk["Stkcd"] = normalize_stock_code_series(raw_chunk["Stkcd"])
        raw_chunk["td"] = pd.to_numeric(raw_chunk["td"], errors="coerce").astype("int64")

        merged = raw_chunk.merge(
            sample_keys,
            on=["Stkcd", "td"],
            how="inner",
            validate="1:1",
        )
        if merged.empty:
            print(f"chunk {chunk_number}: input={len(raw_chunk):,}, matched=0", flush=True)
            continue

        merged = merged.merge(
            benchmark,
            on=["Trddt", "portfolioNumber"],
            how="left",
            validate="m:1",
        )
        merged = calculate_adjusted_returns(merged)
        out = merged[OUTPUT_COLUMNS].copy()

        for column in missing_stats:
            missing_stats[column] += int(out[column].isna().sum())

        out.to_csv(
            adjusted_return_out_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8-sig" if first_chunk else "utf-8",
        )
        first_chunk = False
        total_output_rows += len(out)

        print(
            f"chunk {chunk_number}: input={len(raw_chunk):,}, "
            f"matched={len(out):,}, total_out={total_output_rows:,}",
            flush=True,
        )

    return {
        "total_input_rows": total_input_rows,
        "total_output_rows": total_output_rows,
        "missing_stats": missing_stats,
    }


def parse_args():
    """解析命令行参数，主要用于控制大文件分块读取大小。"""
    parser = argparse.ArgumentParser(
        description="Calculate DGTW characteristic-adjusted returns.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Rows read per chunk from the daily variable panel.",
    )
    return parser.parse_args()


def main():
    """串联样本键、benchmark 和原始收益面板，生成最终 CSV。"""
    args = parse_args()

    print("Loading sample keys with DGTW portfolio assignment...", flush=True)
    sample_keys = load_sample_keys()
    print(f"sample keys with portfolio: {len(sample_keys):,}", flush=True)

    print("Loading DGTW benchmark returns...", flush=True)
    benchmark = load_benchmark_returns()
    print(f"benchmark rows: {len(benchmark):,}", flush=True)

    print("Processing raw return panel in chunks...", flush=True)
    summary = append_adjusted_return_chunks(
        sample_keys=sample_keys,
        benchmark=benchmark,
        chunk_size=args.chunk_size,
    )

    print("DGTW adjusted returns finished.", flush=True)
    print(f"daily variable input rows read: {summary['total_input_rows']:,}", flush=True)
    print(f"adjusted return rows written: {summary['total_output_rows']:,}", flush=True)
    for column, missing_count in summary["missing_stats"].items():
        ratio = missing_count / summary["total_output_rows"] if summary["total_output_rows"] else np.nan
        print(f"{column} missing: {missing_count:,} ({ratio:.4%})", flush=True)
    print(f"saved to: {adjusted_return_out_path}", flush=True)


if __name__ == "__main__":
    main()
