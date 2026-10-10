from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from linearmodels.iv import AbsorbingLS
from tqdm.auto import tqdm
import gc



# 解决 matplotlib 中文显示问题
font_path = "/System/Library/Fonts/Hiragino Sans GB.ttc"
font_manager.fontManager.addfont(font_path)
zh_font = font_manager.FontProperties(fname=font_path).get_name()

plt.rcParams["font.sans-serif"] = [zh_font]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path("/Users/qiyuewu/Desktop/mmm复现数据包")
pattern_sample_PATH = ROOT / "中间结果/04_合并数据/Pattern样本_最终回归面板.csv"
oib_path = ROOT / "中间结果/02_变量构造结果/交易行为OIB、Avol/OIB_CSMAR_prep_core.csv"

fig_dir = ROOT / "图表结果/论文复现/图形"
fig_dir.mkdir(parents=True, exist_ok=True)
fig2_out_path = fig_dir / "figure2_收益动态和长期反转.png"
fig3_out_path = fig_dir / "figure3_散户订单不平衡动态.png"

# ============================================================
# 读取 Figure 2 回归需要的变量
# ============================================================

all_cols = pd.read_csv(pattern_sample_PATH, nrows=0).columns.tolist()

# 被解释变量
y_cols = [
    f"chAdjreturn_t_tp{i}_Recall_w"
    for i in range(1, 51)
]

# 自己的财报公告日
own_e_cols = [
    col for col in all_cols
    if col.startswith("ownEA")
]

# 其他重大事件
disclosure_cols = [
    col for col in ["ownMajorDisclosure", "ownPerformanceForecast"]
    if col in all_cols
]

# 控制变量 = 自己的财报公告日+重大事件+新闻指标
control_cols = (
    own_e_cols
    + disclosure_cols
    + [
        "aesIncrease",
        "aesDecrease",
        "logIncreaseAEV",
        "logDecreaseAEV",
    ]
)

# 解释变量
base_cols = [
    "quarterLag1size90plus",
    "StkcdRecall",       
    "Trddt",            
    "yearQuarter",
]

# 只读取需要的变量
usecols = list(dict.fromkeys(y_cols + base_cols + control_cols))

missing = [col for col in usecols if col not in all_cols]
if missing:
    raise ValueError(f"CSV 中缺少变量: {missing}")

print("正在读取模式公司样本数据......")
df = pd.read_csv(pattern_sample_PATH, usecols=usecols)
print("模式公司数据读取成功！")

# ============================================================
# 循环回归：对应 Stata 的 forvalues i = 1/50
# ============================================================

results = []

# 进度条显示进度
pbar = tqdm(
    range(1, 51),
    total=50,
    desc="正在回归窗口",
    unit="个窗口",
    dynamic_ncols=True,
)

for i in pbar:
    pbar.set_postfix_str(f"当前窗口 {i}/50")
    y_col = f"chAdjreturn_t_tp{i}_Recall_w" #被解释变量

    # 用于回归的变量
    reg_cols = [y_col, "quarterLag1size90plus", "StkcdRecall", "Trddt", "yearQuarter"] + control_cols
    reg_df = df[reg_cols].dropna().copy()

    # 固定效应
    # “公司×年度季度”组
    # pd.Categorical(）将前面的“公司×年度季度”声明成“类别”，
    # 然后AbsorbingLS(吸收固定效应最小二乘)：这些是固定效应分组，要把每个组自己的平均差异吸收掉
    reg_df["firm_yearquarter_fe"] = pd.Categorical(
    reg_df["StkcdRecall"].astype(str)
    + "_"
    + reg_df["yearQuarter"].astype(str)
)

    y = reg_df[y_col]

    X = reg_df[["quarterLag1size90plus"] + control_cols]

    absorb = reg_df[["firm_yearquarter_fe"]]

    # 构造双向聚类标准误
    clusters = pd.DataFrame(
    {
        "firm_cluster": pd.factorize(reg_df["StkcdRecall"])[0],# 把公司代码转成整数编号
        "date_cluster": pd.factorize(reg_df["Trddt"])[0],
    },
    index=reg_df.index,# 让新建的 clusters（聚类变量表）的行索引，和 reg_df（当前回归样本表）的行索引完全一致。
)
    # 搭建模型，明确被解释变量和解释变量
    model = AbsorbingLS(
        dependent=y,
        exog=X,
        absorb=absorb,# 吸收固定效应
    )
    
    # 开始估计模型，使用聚类稳健标准误
    fit = model.fit(
        cov_type="clustered",
        clusters=clusters,
    )

    # 系数
    coef = fit.params["quarterLag1size90plus"]
    # 标准误差
    se = fit.std_errors["quarterLag1size90plus"]

    # 从每一次回归结果里提取核心变量的系数、标准误、置信区间，并保存到结果表里。
    results.append({
        "window": i,
        "coef": coef,
        "se": se,
        "ci_low": coef - 1.96 * se,# 置信区间下界
        "ci_high": coef + 1.96 * se,# 置信区间上界
        "nobs": fit.nobs,# 当前回归使用的观测数
    })

coef_df = pd.DataFrame(results) # 把前面存下来的列表转成一个表格

# 把小数收益率转换成百分比收益率
coef_df[["coef", "se", "ci_low", "ci_high"]] = (
    coef_df[["coef", "se", "ci_low", "ci_high"]] * 100
)

print(coef_df.head())
print(coef_df.tail())
# ============================================================
# 画 Figure 2：收益动态和长期反转
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.errorbar( # 画带误差棒的点图
    coef_df["window"],# 横轴
    coef_df["coef"],# 纵轴
    yerr=[# 每个点的上下置信区间长度
        coef_df["coef"] - coef_df["ci_low"],
        coef_df["ci_high"] - coef_df["coef"],
    ],
    fmt="o",# 用圆点表示系数
    color="blue",# 圆点的颜色是蓝色
    ecolor="blue",# 误差线的颜色是蓝色
    capsize=2,# 误差线两端的小横帽长度
    markersize=4,# 圆点大小
)

ax.axhline(0, color="black", linewidth=1) # 画一条横向的线
# 这条线在纵轴等于0的位置
# 颜色是黑色
# 线的宽度为1
# 系数高于 0 表示正效应，低于 0 表示负效应


ax.set_xlabel("事件窗口")# 设置横轴标题为“事件窗口”
ax.set_ylabel("收益(%)")# 设置纵轴标题为“百分比收益”
ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]) # 指定横轴上只显示这些刻度
ax.set_xticklabels([
    "[t,t+1]", "[t,t+5]", "[t,t+10]", "[t,t+15]", "[t,t+20]",
    "[t,t+25]", "[t,t+30]", "[t,t+35]", "[t,t+40]", "[t,t+45]",
    "[t,t+50]",
])# 把刚才的数字刻度换成更直观的事件窗口标签

plt.tight_layout()# 自动调整图形边距，避免标题、坐标轴标签、刻度标签被挤掉或截断。
plt.savefig(fig2_out_path, dpi=300, bbox_inches="tight")
print(f"图已保存到：{fig2_out_path}")
plt.show()

# ============================================================
# Figure 3：散户订单不平衡动态
# ============================================================

oib_cols = [
    f"OIBvol_avg_t_tp{i}"
    for i in range(1, 51)
]

pattern_cols = [
    "quarterLag1size90plus",
    "StkcdRecall",
    "Trddt",
    "yearQuarter",
] + control_cols

pattern_df = df[pattern_cols].copy()

del df
gc.collect()

# 读取交易数据
oib_df = pd.read_csv(
    oib_path,
    usecols=["Stkcd", "Trddt"] + oib_cols,
)

# 统一股票代码格式
pattern_df["StkcdRecall_key"] = (
    pd.to_numeric(pattern_df["StkcdRecall"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .str.zfill(6)
)

oib_df["Stkcd_key"] = (
    pd.to_numeric(oib_df["Stkcd"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .str.zfill(6)
)

# 统一日期格式
pattern_df["Trddt_key"] = pd.to_datetime(pattern_df["Trddt"])
oib_df["Trddt_key"] = pd.to_datetime(oib_df["Trddt"])

# 合并 OIB：被提示公司 × 日期
fig3_df = pattern_df.merge(
    oib_df[["Stkcd_key", "Trddt_key"] + oib_cols],
    left_on=["StkcdRecall_key", "Trddt_key"],
    right_on=["Stkcd_key", "Trddt_key"],
    how="left",
    validate="m:1",
)

# 合并完成，释放两个中间表
del pattern_df
del oib_df
gc.collect()

# 对图 3 的订单不平衡变量进行全样本 1% 缩尾
for col in oib_cols:
    # 将无穷值处理为缺失值
    fig3_df[col] = fig3_df[col].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    lower = fig3_df[col].quantile(0.01)
    upper = fig3_df[col].quantile(0.99)

    fig3_df[f"{col}_w"] = fig3_df[col].clip(
        lower=lower,
        upper=upper,
    )

# 缩尾完成后，再开始保存和运行回归结果
fig3_results = []

for i in tqdm(range(1, 51), total=50, desc="正在回归 Figure 3", unit="个窗口"):
    y_col = f"OIBvol_avg_t_tp{i}_w"

    reg_cols = [
        y_col,
        "quarterLag1size90plus",
        "StkcdRecall",
        "Trddt",
        "yearQuarter",
    ] + control_cols

    reg_df = fig3_df[reg_cols].dropna().copy()

    reg_df["firm_yearquarter_fe"] = pd.Categorical(
        reg_df["StkcdRecall"].astype(str)
        + "_"
        + reg_df["yearQuarter"].astype(str)
    )

    y = reg_df[y_col]
    X = reg_df[["quarterLag1size90plus"] + control_cols]
    absorb = reg_df[["firm_yearquarter_fe"]]

    clusters = pd.DataFrame(
        {
            "firm_cluster": pd.factorize(reg_df["StkcdRecall"])[0],# 允许同一家公司内部相关
            "date_cluster": pd.factorize(reg_df["Trddt"])[0], # 允许同一天不同公司之间相关
        },
        index=reg_df.index,
    )# 双向聚类标准误差，在同一个聚类内部允许序列相关，不同的聚类之间不允许。

    model = AbsorbingLS(
        dependent=y,
        exog=X,
        absorb=absorb,
    )

    fit = model.fit(
        cov_type="clustered",
        clusters=clusters,
    )

    coef = fit.params["quarterLag1size90plus"]
    se = fit.std_errors["quarterLag1size90plus"]

    fig3_results.append({
        "window": i,
        "coef": coef,
        "se": se,
        "ci_low": coef - 1.96 * se,
        "ci_high": coef + 1.96 * se,
        "nobs": fit.nobs,
    })

fig3_coef_df = pd.DataFrame(fig3_results)

fig, ax = plt.subplots(figsize=(10, 5))

# ax.errorbar() 是 matplotlib（Python 画图库）里用来画带误差棒的图的函数。
ax.errorbar(
    fig3_coef_df["window"],
    fig3_coef_df["coef"],
    yerr=[
        fig3_coef_df["coef"] - fig3_coef_df["ci_low"],
        fig3_coef_df["ci_high"] - fig3_coef_df["coef"],
    ],
    fmt="o",
    color="blue",
    ecolor="blue",
    capsize=2,
    markersize=4,
)

ax.axhline(0, color="black", linewidth=1)

ax.set_xlabel("事件窗口")
ax.set_ylabel("订单不平衡(%)")
ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
ax.set_xticklabels([
    "[t,t+1]", "[t,t+5]", "[t,t+10]", "[t,t+15]", "[t,t+20]",
    "[t,t+25]", "[t,t+30]", "[t,t+35]", "[t,t+40]", "[t,t+45]",
    "[t,t+50]",
])

plt.tight_layout()
fig.savefig(fig3_out_path, dpi=300, bbox_inches="tight")
print(f"图已保存到：{fig3_out_path}")
plt.show()

# 不覆盖本图的交易日期固定效应版本由
# “图3_散户订单不平衡动态_加入日期固定效应.py”单独估计并绘制。

# 由于现有数据无法进一步区分 BJZZ（替代散户交易识别算法）口径下的全部订单与奇数手订单，
# 本文不对原文附录图 A.1 和 A.2 进行严格复现。作为替代性稳健性检验，
# 后续将订单不平衡的计算口径由按成交量计算的 `OIBvol` 替换为按成交笔数计算的 `OIBnum`，
# 以检验基准结果是否受到少数较大金额交易的影响。该检验仍基于 CSMAR 小单买卖分类，
# 因此主要考察结果对订单不平衡度量方式的敏感性，而不能用于检验不同散户交易识别算法或奇数手订单口径的影响。
