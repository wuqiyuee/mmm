"""单日订单不平衡任务使用的公司季度固定效应分块估计器。

固定效应、双向聚类、有限样本修正、残差自由度和结果单位沿用
“图2图3扩展_公司季度固定效应回归公共模块.py”。本模块直接按输入
yearQuarter（年度季度）字段分块，不根据日期重新构造季度。

第一遍逐季度吸收公司固定效应，累加解释变量交叉积和因变量交叉积；
第二遍重新计算该季度的组内残差矩阵，累加目标系数的左右投影得分，
再按被提示公司和事件起点日期分别聚类。每次仅保留一个季度的组内
矩阵，不保存全样本残差矩阵或磁盘缓存。日期不参与固定效应吸收，
也不因日期为单例而删除观测。

构造时固定基础表的行顺序、右端变量和事件起点键；fit（估计方法）
只从同一行顺序的对齐表中读取本次因变量。输入基础表的公司、日期
和季度列不得在构造后原地修改。右端变量保留其输入数值精度：原表
为单精度浮点数时保留单精度，季度内计算统一提升为双精度浮点数。

为避免把数值不稳定的正规方程结果冒充公共核心的最小二乘分解结果，
原始交叉积矩阵条件数不小于1e12时明确报错，交由调用方检查数据或
改用公共核心。本模块不自行删除被吸收或共线的控制变量。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


FIRM_COL = "StkcdRecall"
DATE_COL = "Trddt"
QUARTER_COL = "yearQuarter"
KEY_COLS = [FIRM_COL, DATE_COL, QUARTER_COL]
DATE_FE_BASE_DEFINITION = "不适用（未加入日期固定效应）"


@dataclass
class _QuarterBlock:
    """一个季度删缺失、删公司季度单例并去均值后的临时结果。"""

    indices: np.ndarray
    firm_codes: np.ndarray
    date_codes: np.ndarray
    x: np.ndarray
    y: np.ndarray
    group_count: int
    largest_group: int
    original_x_squares: np.ndarray
    original_y_mean: float
    original_y_centered_squares: float


class QuarterFEEstimator:
    """复用静态基础表，为多个目标日估计同一公司季度规格。

    基础表缺失的公司、日期、季度或右端变量不进入任何相对日样本；
    每次估计再删除当期因变量缺失观测，随后只删公司季度单例。
    基础表静态有效样本必须满足公司和事件起点日期唯一。
    """

    def __init__(
        self,
        base: pd.DataFrame,
        rhs_cols: Sequence[str],
        cue_col: str,
    ) -> None:
        self.rhs_cols = list(rhs_cols)
        self.cue_col = cue_col
        if not self.rhs_cols or cue_col not in self.rhs_cols:
            raise ValueError("核心提示变量必须包含在非空解释变量列表中。")
        if len(self.rhs_cols) != len(set(self.rhs_cols)):
            raise ValueError("解释变量列表包含重复字段。")
        required = list(dict.fromkeys([*KEY_COLS, *self.rhs_cols]))
        missing = sorted(set(required) - set(base.columns))
        if missing:
            raise ValueError(f"公司季度分块估计基础表缺少变量：{missing}")
        if base.empty:
            raise ValueError("公司季度分块估计基础表为空。")

        self._row_count = len(base)
        # 只保留键列的底层数组，以便逐次检查目标表没有改变行的对应关系。
        self._key_arrays = {column: base[column].array for column in KEY_COLS}
        self._firm_codes, firms = pd.factorize(base[FIRM_COL], sort=False)
        self._date_codes, dates = pd.factorize(base[DATE_COL], sort=False)
        self._firm_codes = self._firm_codes.astype(np.int32, copy=False)
        self._date_codes = self._date_codes.astype(np.int32, copy=False)
        self._firm_count = len(firms)
        self._date_count = len(dates)
        self._cue_position = self.rhs_cols.index(cue_col)

        rhs_frame = base[self.rhs_cols]
        # 不把已有双精度或大整数输入强行压成单精度，避免改变其原值。
        single_precision = all(dtype == np.dtype("float32") for dtype in rhs_frame.dtypes)
        numeric_dtype = np.float32 if single_precision else np.float64
        self._rhs = rhs_frame.to_numpy(
            dtype=numeric_dtype,
            copy=False,
            na_value=np.nan,
        )
        self._static_valid = (
            base[KEY_COLS].notna().all(axis=1).to_numpy(dtype=bool)
            & ~np.isnan(self._rhs).any(axis=1)
        )
        if not self._static_valid.any():
            raise ValueError("基础表没有公司、日期、季度和右端变量均完整的观测。")
        if base.loc[self._static_valid, [FIRM_COL, DATE_COL]].duplicated().any():
            raise ValueError("基础表的被提示公司和事件起点日期不是唯一键。")

        quarter_codes = pd.factorize(base[QUARTER_COL], sort=False)[0]
        valid_indices = np.flatnonzero(self._static_valid)
        # 每个位置只归入原始季度字段所指定的一块，允许日期与季度字段不同。
        ordered = valid_indices[
            np.argsort(quarter_codes[valid_indices], kind="stable")
        ]
        boundaries = np.flatnonzero(np.diff(quarter_codes[ordered])) + 1
        self._quarter_indices = np.split(ordered, boundaries)

    def _validate_aligned(self, aligned: pd.DataFrame, outcome: str) -> None:
        """核对逐日对齐表仍与构造时的基础表逐行对应。"""
        if len(aligned) != self._row_count:
            raise ValueError("目标日对齐表与估计器基础表行数不一致。")
        missing = sorted(set([*KEY_COLS, outcome]) - set(aligned.columns))
        if missing:
            raise ValueError(f"目标日对齐表缺少变量：{missing}")
        for column in KEY_COLS:
            if not aligned[column].array.equals(self._key_arrays[column]):
                raise ValueError(
                    f"目标日对齐表的“{column}”或行顺序与基础表不一致。"
                )

    def _make_block(
        self,
        quarter_indices: np.ndarray,
        valid: np.ndarray,
        outcome_values: np.ndarray,
    ) -> tuple[_QuarterBlock | None, int]:
        """筛出本期有效行，只删本季度公司单例，并对公司分别去均值。"""
        indices = quarter_indices[valid[quarter_indices]]
        if len(indices) == 0:
            return None, 0

        firm_codes = self._firm_codes[indices]
        counts = np.bincount(firm_codes, minlength=self._firm_count)
        keep = counts[firm_codes] > 1
        singleton_removed = int((~keep).sum())
        if not keep.any():
            return None, singleton_removed
        indices = indices[keep]
        firm_codes = firm_codes[keep]
        # 单例组已删除；其它公司组的计数保持不变。全局编码的空组均为0。
        surviving_counts = np.where(counts > 1, counts, 0)
        group_count = int(np.count_nonzero(surviving_counts))

        x = self._rhs[indices].astype(np.float64, copy=False)
        y = outcome_values[indices].copy()
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("回归变量包含非有限数值。")
        original_x_squares = np.einsum("ij,ij->j", x, x)
        original_y_mean = float(y.mean())
        centered_y = y - original_y_mean
        original_y_centered_squares = float(centered_y @ centered_y)

        means = np.zeros(self._firm_count, dtype=np.float64)
        # 逐列吸收固定效应，避免额外生成季度宽表大小的组均值矩阵。
        for position in range(x.shape[1]):
            sums = np.bincount(
                firm_codes,
                weights=x[:, position],
                minlength=self._firm_count,
            )
            means.fill(0)
            np.divide(sums, surviving_counts, out=means, where=surviving_counts > 0)
            x[:, position] -= means[firm_codes]
        sums = np.bincount(firm_codes, weights=y, minlength=self._firm_count)
        means.fill(0)
        np.divide(sums, surviving_counts, out=means, where=surviving_counts > 0)
        y -= means[firm_codes]

        return _QuarterBlock(
            indices=indices,
            firm_codes=firm_codes,
            date_codes=self._date_codes[indices],
            x=x,
            y=y,
            group_count=group_count,
            largest_group=int(surviving_counts.max()),
            original_x_squares=original_x_squares,
            original_y_mean=original_y_mean,
            original_y_centered_squares=original_y_centered_squares,
        ), singleton_removed

    def fit(
        self,
        aligned: pd.DataFrame,
        *,
        horizon: int,
        outcome: str,
        outcome_label: str,
    ) -> dict[str, object]:
        """估计一个相对日，并返回与公共公司季度估计器同口径的结果。"""
        if not isinstance(horizon, (int, np.integer)) or horizon < 0:
            raise ValueError("相对交易日序号必须为非负整数。")
        self._validate_aligned(aligned, outcome)
        outcome_values = aligned[outcome].to_numpy(
            dtype=np.float64,
            copy=False,
            na_value=np.nan,
        )
        valid = self._static_valid & ~np.isnan(outcome_values)
        if not valid.any():
            raise ValueError(f"相对交易日 t+{horizon} 没有完整回归观测。")

        nvars = len(self.rhs_cols)
        xpx = np.zeros((nvars, nvars), dtype=np.float64)
        xpy = np.zeros(nvars, dtype=np.float64)
        original_x_squares = np.zeros(nvars, dtype=np.float64)
        nobs = group_count = singleton_removed = largest_group = 0
        original_y_mean = total_sum_squares = 0.0

        # 第一遍仅保存28维矩；合并各块的中心二阶矩，避免原始平方和相减。
        for indices in self._quarter_indices:
            block, removed = self._make_block(indices, valid, outcome_values)
            singleton_removed += removed
            if block is None:
                continue
            block_nobs = len(block.indices)
            combined_nobs = nobs + block_nobs
            mean_delta = block.original_y_mean - original_y_mean
            total_sum_squares += (
                block.original_y_centered_squares
                + mean_delta ** 2 * nobs * block_nobs / combined_nobs
            )
            original_y_mean += mean_delta * block_nobs / combined_nobs
            nobs = combined_nobs
            group_count += block.group_count
            largest_group = max(largest_group, block.largest_group)
            xpx += block.x.T @ block.x
            xpy += block.x.T @ block.y
            original_x_squares += block.original_x_squares
            del block

        if nobs == 0:
            raise ValueError("删除公司季度固定效应单例后没有剩余观测。")
        residual_degrees = nobs - group_count - nvars
        if residual_degrees <= 0:
            raise ValueError(f"相对交易日 t+{horizon} 的残差自由度不为正。")
        if not np.isfinite(xpx).all() or not np.isfinite(xpy).all():
            raise ValueError("解释变量交叉积包含非有限数值。")

        # 与公共核心一致，先检查组内列是否完全被吸收，再检查标准化矩阵秩。
        column_norms = np.sqrt(np.maximum(np.diag(xpx), 0.0))
        rounding_scale = max(100, largest_group) * np.finfo(float).eps
        absorbed = column_norms <= rounding_scale * np.maximum(
            np.sqrt(original_x_squares), 1.0,
        )
        absorbed_columns = [
            column for column, is_absorbed in zip(self.rhs_cols, absorbed)
            if is_absorbed
        ]
        if self.cue_col in absorbed_columns:
            raise ValueError(
                f"相对交易日 t+{horizon} 的线索变量被公司季度固定效应完全吸收。"
            )
        if absorbed_columns:
            raise ValueError(
                f"相对交易日 t+{horizon} 的控制变量被公司季度固定效应完全吸收："
                f"{absorbed_columns}。请检查控制变量设定。"
            )
        standardized_xpx = xpx / column_norms[:, None] / column_norms[None, :]
        if np.linalg.matrix_rank(standardized_xpx) < nvars:
            raise ValueError(
                f"相对交易日 t+{horizon} 的解释变量组内矩阵秩不足，"
                "无法分别识别全部系数。请检查共线控制变量。"
            )
        xpx_condition = float(np.linalg.cond(xpx))
        if not np.isfinite(xpx_condition) or xpx_condition >= 1e12:
            raise ValueError(
                f"相对交易日 t+{horizon} 的原始交叉积矩阵条件数为"
                f"{xpx_condition:.6g}；分块估计器不执行不稳定的正规方程，"
                "请检查变量量纲或改用公共核心的最小二乘分解。"
            )
        parameters = np.linalg.solve(xpx, xpy)
        xpx_inverse = np.linalg.inv(xpx)

        # 第二遍重新计算季度残差，按全期编码汇总得分，允许同一公司跨季度聚类。
        firm_left = np.zeros(self._firm_count, dtype=np.float64)
        firm_right = np.zeros(self._firm_count, dtype=np.float64)
        date_left = np.zeros(self._date_count, dtype=np.float64)
        date_right = np.zeros(self._date_count, dtype=np.float64)
        seen_firms = np.zeros(self._firm_count, dtype=bool)
        seen_dates = np.zeros(self._date_count, dtype=bool)
        intersection_product = residual_sum_squares = 0.0
        for indices in self._quarter_indices:
            block, _ = self._make_block(indices, valid, outcome_values)
            if block is None:
                continue
            residuals = block.y - block.x @ parameters
            residual_sum_squares += float(residuals @ residuals)
            # 保留逆矩阵的行/列双线性形式，不假定浮点逆矩阵严格对称。
            left_scores = (block.x @ xpx_inverse[self._cue_position, :]) * residuals
            right_scores = (block.x @ xpx_inverse[:, self._cue_position]) * residuals
            firm_left += np.bincount(
                block.firm_codes, weights=left_scores, minlength=self._firm_count,
            )
            firm_right += np.bincount(
                block.firm_codes, weights=right_scores, minlength=self._firm_count,
            )
            date_left += np.bincount(
                block.date_codes, weights=left_scores, minlength=self._date_count,
            )
            date_right += np.bincount(
                block.date_codes, weights=right_scores, minlength=self._date_count,
            )
            intersection_product += float(left_scores @ right_scores)
            seen_firms[block.firm_codes] = True
            seen_dates[block.date_codes] = True
            del block, residuals, left_scores, right_scores

        firm_count = int(seen_firms.sum())
        date_count = int(seen_dates.sum())
        if firm_count < 2 or date_count < 2:
            raise ValueError("被提示公司或事件起点日期聚类数少于2。")
        # 公司×事件起点日期已保证唯一，交集维度即单条观测。
        common_scale = (nobs - 1) / (nobs - nvars)
        cue_variance = float(common_scale * (
            firm_count / (firm_count - 1) * (firm_left @ firm_right)
            + date_count / (date_count - 1) * (date_left @ date_right)
            - nobs / (nobs - 1) * intersection_product
        ))
        if not np.isfinite(cue_variance) or cue_variance < -1e-12:
            raise ValueError(f"相对交易日 t+{horizon} 的聚类方差无效或明显为负。")
        coefficient_raw = float(parameters[self._cue_position])
        standard_error_raw = float(np.sqrt(max(cue_variance, 0.0)))
        if not np.isfinite(coefficient_raw) or not np.isfinite(standard_error_raw):
            raise ValueError(f"相对交易日 t+{horizon} 的核心系数或标准误非有限。")
        if standard_error_raw <= 0:
            raise ValueError(f"相对交易日 t+{horizon} 的核心变量标准误无效。")
        if not np.isfinite(total_sum_squares) or total_sum_squares <= 0:
            raise ValueError(f"相对交易日 t+{horizon} 的因变量没有有效变异。")

        t_value = coefficient_raw / standard_error_raw
        p_value = float(2 * stats.t.sf(abs(t_value), residual_degrees))
        stars = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
        coefficient = coefficient_raw * 100
        standard_error = standard_error_raw * 100
        return {
            "窗口序号": int(horizon),
            "相对交易日": f"t+{horizon}",
            "因变量": outcome_label,
            "结果单位": "百分点",
            "线索系数": coefficient,
            "聚类标准误": standard_error,
            "置信区间下界": coefficient - 1.96 * standard_error,
            "置信区间上界": coefficient + 1.96 * standard_error,
            "置信区间口径": "逐点95%置信区间",
            "t值": t_value,
            "p值": p_value,
            "显著性": stars,
            "控制变量": "是",
            "日期固定效应": "否",
            "日期固定效应基准": DATE_FE_BASE_DEFINITION,
            "被提示公司×年度季度固定效应": "是",
            "聚类方式": "被提示公司和事件起点交易日期双向聚类",
            "有限样本修正": "是",
            "回归样本量": nobs,
            "被提示公司聚类数": firm_count,
            "日期聚类数": date_count,
            "删除单例组观测数": singleton_removed,
            "解释变量矩阵条件数": xpx_condition,
            "系数求解方法": "正规方程（条件数已检查）",
            "R方": max(1 - residual_sum_squares / total_sum_squares, 0.0),
        }
