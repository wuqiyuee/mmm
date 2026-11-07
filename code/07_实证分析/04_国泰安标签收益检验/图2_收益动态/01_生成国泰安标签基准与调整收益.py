"""使用国泰安官方标签生成组合基准和沪深300口径的调整收益。

计算顺序：合并官方标签 → 按日期和组合计算加权基准 → 个股收益减基准
→ 各窗口自身全样本1%和99%缩尾 → 提取最终面板目标并保存。

基准成员使用全市场基础清洗样本，要求标签有效、t-3流通市值有效且为正。
50个窗口分别使用各自收益有效的成员加权，不要求50窗共同完整。
缩尾使用研究期全部筛样前公司日，不先筛控制变量完整值或固定效应单例，
不另加市值、股价或填充日筛选。目标公司自身不要求流通市值有效。

四个输出均保存在正式中间结果目录，不调用其他脚本、不读取工作临时目录：
1. 50窗组合基准：日期×官方组合，50个基准收益列。
2. 筛样前50窗未缩尾收益：公司×日期，供精确计算全样本分位点的列式中间表。
3. 50窗缩尾阈值：每窗有效样本数、第1和第99百分位。
4. 50窗调整收益：最终面板全部目标，包含50个未缩尾列和50个缩尾列。
缺失标签、收益或基准的观测保留为空；研究期外或未匹配的最终目标也保留空收益。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# 一、输入、输出和计算范围。
project_root = Path(__file__).resolve().parents[4]
target_panel_path = project_root / "中间结果/04_合并数据/Pattern样本_最终回归面板_沪深300.csv"
daily_panel_path = project_root / "中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv"
base_index_path = project_root / "中间结果/01_清洗后样本/基础清洗后个股日度样本索引_td.csv"
official_label_path = project_root / "input/CSMAR/DGTW（2000至2025）/DGTW股票分配表/STK_MKT_DGTWASSINGMENTS.csv"
output_dir = project_root / "中间结果/02_变量构造结果/1DGTW特征调整收益率"
benchmark_path = output_dir / "国泰安标签_沪深300_50窗组合基准.csv"
return_path = output_dir / "国泰安标签_沪深300_50窗调整收益.csv"
raw_return_path = output_dir / "国泰安标签_沪深300_筛样前50窗未缩尾收益.parquet"
threshold_path = output_dir / "国泰安标签_沪深300_50窗缩尾阈值.csv"

start_date = "2005-04-08"
end_date = "2025-12-31"
chunk_size = 100_000
stock_date_columns = ["Stkcd", "Trddt"]
target_columns = ["StkcdRecall", "Trddt"]
group_columns = ["Trddt", "portfolioNumber"]
raw_columns = [f"raw_ret_t_t{i}" for i in range(1, 51)]
benchmark_columns = [f"bench_ret_t_t{i}" for i in range(1, 51)]
adjusted_columns = [f"chAdjOfficial_t_tp{i}" for i in range(1, 51)]
winsor_columns = [f"{column}_w" for column in adjusted_columns]

for path in (target_panel_path, daily_panel_path, base_index_path, official_label_path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入文件：{path}")
for path in (benchmark_path, return_path, raw_return_path, threshold_path):
    if path.exists():
        raise FileExistsError(f"输出文件已存在，请先核对并移走旧文件，避免覆盖：{path}")


# 二、读取目标和基础资格表，以公司、日期两列作为表索引。
# seen（已读取标记）只记录该公司日是否已读取，用于检查跨数据块重复和补齐未匹配目标。
# 目标保留最终面板全部日期；基础资格只保留研究期，均不按填充标志删行。
targets = pd.read_csv(
    target_panel_path, usecols=target_columns, dtype={"StkcdRecall": "int32"}, parse_dates=["Trddt"],
)
if targets.empty:
    raise ValueError("沪深300最终回归面板为空。")
if targets.duplicated(target_columns).any():
    raise ValueError("沪深300最终回归面板存在重复的被提示公司和日期。")
targets["seen"] = False
targets = targets.set_index(target_columns)

base_sample = pd.read_csv(
    base_index_path, usecols=stock_date_columns, dtype={"Stkcd": "int32"}, parse_dates=["Trddt"],
)
base_sample = base_sample.loc[base_sample["Trddt"].between(start_date, end_date)].copy()
if base_sample.empty:
    raise ValueError("基础清洗样本索引在研究期内为空。")
if base_sample.duplicated(stock_date_columns).any():
    raise ValueError("基础清洗样本索引存在重复的公司和日期。")
base_sample["seen"] = False
base_sample = base_sample.set_index(stock_date_columns)
print(f"目标观测{len(targets):,}行；基础清洗资格{len(base_sample):,}行。", flush=True)


# 三、读取非北交所官方标签，直接将三个1—5分组编码为1—125，不重新分组。
labels = pd.read_csv(
    official_label_path,
    usecols=["Symbol", "TradingYear", "IsNotBSE", "MarketValue", "BooktoMarket", "Momentum"],
    dtype={"Symbol": "int32", "TradingYear": "int16"},
)
labels = labels.loc[labels["IsNotBSE"].eq(1)].copy()
if labels.duplicated(["Symbol", "TradingYear"]).any():
    raise ValueError("国泰安非北交所标签存在重复的公司和形成年度。")
for column in ("MarketValue", "BooktoMarket", "Momentum"):
    if not labels[column].isin([1, 2, 3, 4, 5]).all():
        raise ValueError(f"国泰安标签分组值不是1至5的整数：{column}")
labels["portfolioNumber"] = (
    (labels["MarketValue"] - 1) * 25 + (labels["BooktoMarket"] - 1) * 5 + labels["Momentum"]
).astype("int16")
labels = labels.rename(columns={"Symbol": "Stkcd", "TradingYear": "formation_year"})
labels = labels[["Stkcd", "formation_year", "portfolioNumber"]]


# 四、分块计算基准：先匹配基础资格和标签，再按日期、组合累计分子和分母。
# 各块的加权收益之和、有效权重之和分别相加，全部读完后才相除。
weight_sums = None
weighted_return_sums = None
true_flags = ["true", "1", "yes", "y"]
allowed_flags = true_flags + ["false", "0", "no", "n", ""]
daily_dtypes = {column: "float64" for column in raw_columns}
daily_dtypes.update({"Stkcd": "int32", "float_mktcap_tm3": "float64", "float_mktcap_tm3_valid": "string"})
for chunk_number, chunk in enumerate(pd.read_csv(
    daily_panel_path,
    usecols=[*stock_date_columns, "float_mktcap_tm3", "float_mktcap_tm3_valid", *raw_columns],
    dtype=daily_dtypes, parse_dates=["Trddt"], chunksize=chunk_size,
), 1):
    chunk = chunk.loc[chunk["Trddt"].between(start_date, end_date)].copy()
    if chunk.empty:
        continue

    # 先检查基础资格内的重复公司日，尚未按标签、市值或窗口收益筛选。
    members = chunk.join(base_sample, on=stock_date_columns, how="inner", validate="many_to_one", sort=False)
    if members.duplicated(stock_date_columns).any() or members["seen"].any():
        raise ValueError("日度面板在基础清洗资格池内存在重复公司日期，无法计算组合基准。")
    base_sample.loc[pd.MultiIndex.from_frame(members[stock_date_columns]), "seen"] = True

    # 对研究期内所有日度行检查标志取值，保持原有检查范围。
    flags = chunk["float_mktcap_tm3_valid"].str.strip().str.lower()
    if (flags.notna() & ~flags.isin(allowed_flags)).any():
        raise ValueError("流通市值有效标志包含无法识别的值，请检查日度面板。")

    # 持有期1—6月使用上年标签，7—12月使用当年标签。
    members["formation_year"] = members["Trddt"].dt.year - (members["Trddt"].dt.month <= 6)
    members = members.merge(labels, on=["Stkcd", "formation_year"], how="left", validate="many_to_one", sort=False)
    valid_weight = members["float_mktcap_tm3_valid"].str.strip().str.lower().isin(true_flags)
    members = members.loc[
        members["portfolioNumber"].notna() & valid_weight
        & np.isfinite(members["float_mktcap_tm3"]) & members["float_mktcap_tm3"].gt(0)
    ].copy()
    if not members.empty:
        members["portfolioNumber"] = members["portfolioNumber"].astype("int16")
        returns = members[raw_columns].where(np.isfinite(members[raw_columns]))
        weights = members["float_mktcap_tm3"]
        groups = [members[column] for column in group_columns]
        # 每个窗口的缺失收益既不计入分子，也不计入该窗口的分母。
        part_weights = returns.notna().mul(weights, axis=0).groupby(groups, sort=True).sum()
        part_weighted_returns = returns.fillna(0.0).mul(weights, axis=0).groupby(groups, sort=True).sum()
        if weight_sums is None:
            weight_sums = part_weights
            weighted_return_sums = part_weighted_returns
        else:
            weight_sums = weight_sums.add(part_weights, fill_value=0.0)
            weighted_return_sums = weighted_return_sums.add(part_weighted_returns, fill_value=0.0)
    if chunk_number % 20 == 0:
        print(f"基准计算已读取{chunk_number}块。", flush=True)

if weight_sums is None:
    raise ValueError("所有日期和官方组合的基准收益均为空，请核对输入。")
benchmark = weighted_return_sums / weight_sums.where(weight_sums > 0)
benchmark.columns = benchmark_columns
benchmark = benchmark.loc[np.isfinite(benchmark).any(axis=1)].sort_index()
if benchmark.empty:
    raise ValueError("所有日期和官方组合的基准收益均为空，请核对输入。")
del base_sample, weight_sums, weighted_return_sums, chunk, members
output_dir.mkdir(parents=True, exist_ok=True)
benchmark.reset_index().to_csv(
    benchmark_path, index=False, encoding="utf-8-sig", mode="x", date_format="%Y-%m-%d",
)
print(f"日期×组合基准已保存：{benchmark_path}；{len(benchmark):,}行。", flush=True)


# 五、再次分块读取完整日度面板，按公司、形成年度合并标签，再按日期、组合合并基准。
# 这里保留研究期全部公司日；不限制最终目标或控制变量完整值。
# 显式指定列式表的字段类型，保证整个窗口为空的数据块也能与其他块一起保存。
raw_schema = pa.schema([
    ("StkcdRecall", pa.int32()), ("Trddt", pa.timestamp("ns")),
    *[(column, pa.float64()) for column in adjusted_columns],
])
raw_dtypes = {column: "float64" for column in raw_columns}
raw_dtypes["Stkcd"] = "int32"
raw_rows = 0
with pq.ParquetWriter(raw_return_path, raw_schema, compression="snappy") as writer:
    for chunk_number, chunk in enumerate(pd.read_csv(
        daily_panel_path, usecols=[*stock_date_columns, *raw_columns],
        dtype=raw_dtypes, parse_dates=["Trddt"], chunksize=chunk_size,
    ), 1):
        chunk = chunk.loc[chunk["Trddt"].between(start_date, end_date)].copy()
        if chunk.empty:
            continue
        chunk["formation_year"] = chunk["Trddt"].dt.year - (chunk["Trddt"].dt.month <= 6)
        chunk = chunk.merge(labels, on=["Stkcd", "formation_year"], how="left", validate="many_to_one", sort=False)
        chunk = chunk.join(benchmark, on=group_columns, how="left", validate="many_to_one", sort=False)
        result = chunk[stock_date_columns].rename(columns={"Stkcd": "StkcdRecall"}).copy()
        for raw_column, benchmark_column, adjusted_column in zip(raw_columns, benchmark_columns, adjusted_columns):
            result[adjusted_column] = chunk[raw_column] - chunk[benchmark_column]
        result[adjusted_columns] = result[adjusted_columns].replace([np.inf, -np.inf], np.nan)
        writer.write_table(pa.Table.from_pandas(result, schema=raw_schema, preserve_index=False))
        raw_rows += len(result)
        if chunk_number % 20 == 0:
            print(f"未缩尾收益已读取{chunk_number}块，保存筛样前公司日{raw_rows:,}行。", flush=True)
if raw_rows == 0:
    raise ValueError("研究期内没有筛样前公司日，无法计算国泰安收益缩尾阈值。")
del benchmark, labels, chunk, result
print(f"筛样前未缩尾收益已保存：{raw_return_path}；{raw_rows:,}行。", flush=True)


# 六、逐窗口读取全部有效值，精确计算第1和第99百分位。
# 每次只读取一列，使用与主线相同的线性插值，不对分块分位数求平均。
raw_panel = pq.ParquetFile(raw_return_path)
lower_bounds = np.full(50, np.nan)
upper_bounds = np.full(50, np.nan)
threshold_records = []
for window, column in enumerate(adjusted_columns):
    arrow_column = raw_panel.read(columns=[column], use_threads=False).column(0)
    values = arrow_column.to_numpy(zero_copy_only=False)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError(f"第{window + 1}个窗口没有有效的国泰安调整收益，无法计算缩尾阈值。")
    lower_bounds[window], upper_bounds[window] = np.quantile(
        values, [0.01, 0.99], method="linear", overwrite_input=True,
    )
    threshold_records.append({
        "window": window + 1, "n_valid": len(values),
        "lower_p01": lower_bounds[window], "upper_p99": upper_bounds[window],
    })
    print(f"窗口{window + 1}：有效{len(values):,}行，缩尾边界"
          f"[{lower_bounds[window]:.8f}, {upper_bounds[window]:.8f}]。", flush=True)
    del arrow_column, values
pd.DataFrame(threshold_records).to_csv(threshold_path, index=False, encoding="utf-8-sig")
print(f"国泰安收益自身50窗缩尾阈值已保存：{threshold_path}", flush=True)


# 七、按公司、日期提取最终目标，并用各窗口同一套阈值缩尾。
# 低于下界替换为下界，高于上界替换为上界，不删行，缺失仍为缺失。
rows_written = 0
with return_path.open("x", encoding="utf-8-sig", newline="") as output_file:
    for batch in raw_panel.iter_batches(batch_size=chunk_size):
        result = batch.to_pandas().join(targets, on=target_columns, how="inner", validate="many_to_one", sort=False)
        if result.empty:
            continue
        if result.duplicated(target_columns).any() or result["seen"].any():
            raise ValueError("筛样前未缩尾收益表存在重复的目标公司和日期。")
        targets.loc[pd.MultiIndex.from_frame(result[target_columns]), "seen"] = True
        result = result.drop(columns="seen")
        for window, column in enumerate(adjusted_columns):
            result[winsor_columns[window]] = result[column].clip(lower_bounds[window], upper_bounds[window])
        result["StkcdRecall"] = result["StkcdRecall"].astype(str).str.zfill(6)
        result.to_csv(output_file, index=False, header=(rows_written == 0), date_format="%Y-%m-%d")
        rows_written += len(result)

    # 未匹配及研究期外目标按日期、股票排序，补入100列空收益。
    missing_targets = targets.loc[~targets["seen"]].reset_index().drop(columns="seen")
    missing_targets = missing_targets.sort_values(["Trddt", "StkcdRecall"])
    for start in range(0, len(missing_targets), 50_000):
        result = missing_targets.iloc[start:start + 50_000].copy()
        result = result.reindex(columns=target_columns + adjusted_columns + winsor_columns)
        result["StkcdRecall"] = result["StkcdRecall"].astype(str).str.zfill(6)
        result.to_csv(output_file, index=False, header=(rows_written == 0), date_format="%Y-%m-%d")
        rows_written += len(result)

if rows_written != len(targets):
    raise ValueError("输出行数与目标键数不一致，请核对收益文件。")
print(f"公司层面未缩尾及缩尾调整收益已保存：{return_path}", flush=True)
print(f"共{rows_written:,}行；日度面板未覆盖或研究期外的目标{len(missing_targets):,}行，收益保留为空。", flush=True)
