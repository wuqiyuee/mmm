"""表5心理机制脚本的规格映射和输入表头测试。"""

from __future__ import annotations

import csv
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import weakref

import numpy as np
import pandas as pd
from matplotlib import image as mpl_image

import 表5_心理机制 as table5


EXPECTED_OUTCOME_COLUMNS = [
    "chAdjreturn_t_tp1_Recall_w",
    "chAdjreturn_t_tp10_Recall_w",
]
EXPECTED_OWN_EA_COLS = (
    [f"ownEA_tm{i}" for i in range(10, 0, -1)]
    + ["ownEA"]
    + [f"ownEA_tp{i}" for i in range(1, 11)]
)
EXPECTED_DISCLOSURE_COLS = [
    "ownMajorDisclosure",
    "ownPerformanceForecast",
]
EXPECTED_MEDIA_CONTROL_COLS = [
    "aesIncrease",
    "aesDecrease",
    "logIncreaseAEV",
    "logDecreaseAEV",
]
EXPECTED_SHARED_FIELDS = {
    "high_media": (
        "lag1hiAEVsize90plus",
        "lag1hiAEVsize50_90",
        None,
        None,
    ),
    "high_total_volume": (
        "lag1hiavolsize90plus",
        "lag1hiavolsize50_90",
        None,
        None,
    ),
    "high_small_trade_volume": (
        "lag1hiavolSsize90plus",
        "lag1hiavolSsize50_90",
        None,
        None,
    ),
    "top_mover": (
        "lag1topMoversize90plus",
        "lag1topMoversize50_90",
        None,
        None,
    ),
    "interference": (
        "lowInterferenceEWLag1S90p",
        "lowInterferenceEWLag1S5090",
        "highInterferenceEWLag1S90p",
        "highInterferenceEWLag1S5090",
    ),
}
EXPECTED_SUE_FIELD_TEMPLATES = (
    ("lag1sue{}possize90plus", "lag1sue{}possize50_90"),
    ("lag1sue{}negsize90plus", "lag1sue{}negsize50_90"),
    ("lag1hi_sue{}_possize90plus", "lag1hi_sue{}_possize50_90"),
    ("lag1hi_sue{}_negsize90plus", "lag1hi_sue{}_negsize50_90"),
)
EXPECTED_MECHANISM_SEQUENCE_SUE1 = [
    (1, "high_total_volume", "异常总成交量", "lag1hiavolsize90plus", "lag1hiavolsize50_90", None, None),
    (2, "high_small_trade_volume", "异常散户成交量（A股小单替代）", "lag1hiavolSsize90plus", "lag1hiavolSsize50_90", None, None),
    (3, "high_media", "媒体覆盖", "lag1hiAEVsize90plus", "lag1hiAEVsize50_90", None, None),
    (4, "top_mover", "价格异动", "lag1topMoversize90plus", "lag1topMoversize50_90", None, None),
    (5, "positive_sue", "正向盈余惊喜", "lag1sue1possize90plus", "lag1sue1possize50_90", None, None),
    (6, "negative_sue", "负向盈余惊喜", "lag1sue1negsize90plus", "lag1sue1negsize50_90", None, None),
    (7, "extreme_positive_sue", "极端正向盈余惊喜", "lag1hi_sue1_possize90plus", "lag1hi_sue1_possize50_90", None, None),
    (8, "extreme_negative_sue", "极端负向盈余惊喜", "lag1hi_sue1_negsize90plus", "lag1hi_sue1_negsize50_90", None, None),
    (9, "interference", "干扰程度", "lowInterferenceEWLag1S90p", "lowInterferenceEWLag1S5090", "highInterferenceEWLag1S90p", "highInterferenceEWLag1S5090"),
]


class Table5MechanismTests(unittest.TestCase):
    """保护规格映射和小成本表头校验的外部行为。"""

    def test_outcomes_are_in_required_panel_order(self):
        """若收益窗口顺序或列名错误，表5面板会被错误标记。"""
        self.assertEqual(
            [outcome.column for outcome in table5.OUTCOME_SPECS],
            EXPECTED_OUTCOME_COLUMNS,
        )

    def test_build_mechanism_specs_maps_shared_sue_and_interference_fields(self):
        """若映射到错误的提示变量，机制比较会使用错误样本。"""
        specs_by_sue = {
            sue_id: table5.build_mechanism_specs(sue_id)
            for sue_id in (1, 2, 3)
        }

        for specs in specs_by_sue.values():
            self.assertEqual(len(specs), 9)
            self.assertEqual([spec.order for spec in specs], list(range(1, 10)))

        for key, expected_fields in EXPECTED_SHARED_FIELDS.items():
            expected = expected_fields
            for specs in specs_by_sue.values():
                spec = next(item for item in specs if item.key == key)
                self.assertEqual(
                    (
                        spec.first_90,
                        spec.first_5090,
                        spec.second_90,
                        spec.second_5090,
                    ),
                    expected,
                )

        for position, expected_fields in zip(
            range(4, 8),
            EXPECTED_SUE_FIELD_TEMPLATES,
            strict=True,
        ):
            for sue_id, specs in specs_by_sue.items():
                spec = specs[position]
                self.assertEqual(
                    (
                        spec.first_90,
                        spec.first_5090,
                        spec.second_90,
                        spec.second_5090,
                    ),
                    (
                        expected_fields[0].format(sue_id),
                        expected_fields[1].format(sue_id),
                        None,
                        None,
                    ),
                )

    def test_mechanism_sequence_locks_orders_labels_and_fields(self):
        """若机制排序或标签错配，后续按编号输出的列会与设计规格不一致。"""
        actual = [
            (
                spec.order,
                spec.key,
                spec.label,
                spec.first_90,
                spec.first_5090,
                spec.second_90,
                spec.second_5090,
            )
            for spec in table5.build_mechanism_specs(1)
        ]
        self.assertEqual(actual, EXPECTED_MECHANISM_SEQUENCE_SUE1)

    def test_specs_are_immutable_and_reject_unknown_sue_id(self):
        """若规格可变或接受错误 SUE，后续三份表会相互污染。"""
        outcome = table5.OUTCOME_SPECS[0]
        mechanism = table5.build_mechanism_specs(1)[0]

        with self.assertRaises(FrozenInstanceError):
            outcome.column = "wrong"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            mechanism.key = "wrong"  # type: ignore[misc]
        for invalid_sue_id in (0, 4, "1", None, True):
            with self.assertRaises(ValueError):
                table5.build_mechanism_specs(invalid_sue_id)

    def test_required_input_columns_cover_controls_keys_outcomes_and_all_sue_fields(self):
        """若读取字段遗漏，37GB（吉字节）面板会在长时读取后才失败。"""
        required = set(table5.required_input_columns())

        self.assertTrue(
            {
                "StkcdRecall",
                "Trddt",
                "year",
                "yearQuarter",
                *EXPECTED_OUTCOME_COLUMNS,
                *EXPECTED_OWN_EA_COLS,
                *EXPECTED_DISCLOSURE_COLS,
                *EXPECTED_MEDIA_CONTROL_COLS,
            }.issubset(required)
        )
        for sue_id in (1, 2, 3):
            for spec in table5.build_mechanism_specs(sue_id):
                self.assertIn(spec.first_90, required)
                self.assertIn(spec.first_5090, required)
                if spec.second_90 is not None:
                    self.assertIn(spec.second_90, required)
                if spec.second_5090 is not None:
                    self.assertIn(spec.second_5090, required)

    def test_read_and_validate_header_reports_all_missing_columns_without_reading_rows(self):
        """若表头不完整，应一次报告全部缺失字段而不是读取大文件正文。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "incomplete.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["StkcdRecall", "Trddt"])
                writer.writerow(["000001", "2020-01-01"])

            with self.assertRaises(ValueError) as context:
                table5.read_and_validate_header(path)

        message = str(context.exception)
        self.assertIn("缺少", message)
        missing = set(table5.required_input_columns()) - {"StkcdRecall", "Trddt"}
        for column in missing:
            self.assertIn(column, message)

    def test_rest_partition(self):
        """若其他关联提示未排除强关联或跨规模覆盖，两个比较组会被错误污染。"""
        spec = table5.build_mechanism_specs(1)[0]
        data = pd.DataFrame(
            {
                "quarterLag1size90plus": [1, 1, 0, pd.NA],
                "quarterLag1size50_90": [1, 0, 1, 1],
                "lag1hiavolsize90plus": [1, 0, 0, pd.NA],
                "lag1hiavolsize50_90": [0, 0, 1, 0],
            },
            index=[101, 103, 107, 109],
        )

        materialized, pairs = table5.materialize_comparisons(data, spec)

        self.assertEqual(materialized.index.tolist(), [101, 103, 107, 109])
        self.assertEqual(len(materialized), len(data))
        self.assertEqual(len(pairs), 2)
        self.assertEqual([pair.size_key for pair in pairs], ["90plus", "50_90"])
        self.assertEqual(
            materialized[pairs[0].first_col].tolist(),
            [1, 0, 0, pd.NA],
        )
        self.assertEqual(
            materialized[pairs[0].second_col].tolist(),
            [0, 1, 0, pd.NA],
        )
        self.assertEqual(
            materialized[pairs[1].first_col].tolist(),
            [0, 0, 1, 0],
        )
        self.assertEqual(
            materialized[pairs[1].second_col].tolist(),
            [1, 0, 0, 1],
        )
        for pair, baseline in zip(
            pairs,
            ["quarterLag1size90plus", "quarterLag1size50_90"],
            strict=True,
        ):
            self.assertIn(spec.key, pair.first_col)
            self.assertIn(pair.size_key, pair.first_col)
            self.assertIn(spec.key, pair.second_col)
            self.assertIn(pair.size_key, pair.second_col)
            observed = materialized[pair.first_col] + materialized[pair.second_col]
            pd.testing.assert_series_equal(
                observed,
                data[baseline].astype("Int64"),
                check_names=False,
            )

    def test_invalid_subset_raises(self):
        """若强关联提示不属于对应基准提示，回归比较的样本定义不成立。"""
        spec = table5.build_mechanism_specs(1)[0]
        data = pd.DataFrame(
            {
                "quarterLag1size90plus": [0],
                "quarterLag1size50_90": [1],
                "lag1hiavolsize90plus": [1],
                "lag1hiavolsize50_90": [0],
            }
        )

        with self.assertRaisesRegex(ValueError, "强关联提示.*基准提示"):
            table5.materialize_comparisons(data, spec)

    def test_invalid_binary_values_raise(self):
        """若核心字段接受非二元值，机制组别会静默偏离提示定义。"""
        spec = table5.build_mechanism_specs(1)[0]
        data = pd.DataFrame(
            {
                "quarterLag1size90plus": [1],
                "quarterLag1size50_90": [1],
                "lag1hiavolsize90plus": [2],
                "lag1hiavolsize50_90": [0],
            }
        )

        with self.assertRaisesRegex(ValueError, "只能取0、1或缺失"):
            table5.materialize_comparisons(data, spec)

    def test_interference_returns_low_high_pairs_without_rest(self):
        """干扰列仍应采用原文统一的强关联与其他关联行名。"""
        spec = table5.build_mechanism_specs(1)[-1]
        data = pd.DataFrame(
            {
                "lowInterferenceEWLag1S90p": [1, 0],
                "lowInterferenceEWLag1S5090": [0, 1],
                "highInterferenceEWLag1S90p": [0, 1],
                "highInterferenceEWLag1S5090": [1, 0],
            },
            index=[31, 37],
        )

        materialized, pairs = table5.materialize_comparisons(data, spec)

        self.assertEqual(len(materialized), len(data))
        self.assertEqual(materialized.index.tolist(), [31, 37])
        self.assertEqual([pair.first_label for pair in pairs], ["强关联提示", "强关联提示"])
        self.assertEqual([pair.second_label for pair in pairs], ["其他关联提示", "其他关联提示"])
        self.assertEqual(materialized[pairs[0].first_col].tolist(), [1, 0])
        self.assertEqual(materialized[pairs[0].second_col].tolist(), [0, 1])
        self.assertEqual(materialized[pairs[1].first_col].tolist(), [0, 1])
        self.assertEqual(materialized[pairs[1].second_col].tolist(), [1, 0])
        self.assertFalse(any("rest" in column for column in materialized.columns))


class Table5RegressionTests(unittest.TestCase):
    """保护表5核心估计器的固定效应和推断口径。"""

    @staticmethod
    def _build_panel() -> pd.DataFrame:
        """生成有充分组内变异、聚类数和固定效应组的可重复虚构面板。"""
        rng = np.random.default_rng(20260727)
        firms = [f"{firm:06d}" for firm in range(1, 13)]
        dates = pd.date_range("2020-01-02", periods=48, freq="7D")
        panel = pd.MultiIndex.from_product(
            [firms, dates],
            names=["StkcdRecall", "Trddt"],
        ).to_frame(index=False)
        row_count = len(panel)
        panel["yearQuarter"] = panel["Trddt"].dt.to_period("Q").astype(str)

        baseline_90 = rng.binomial(1, 0.62, row_count)
        baseline_5090 = rng.binomial(1, 0.58, row_count)
        strong_90 = baseline_90 * rng.binomial(1, 0.44, row_count)
        strong_5090 = baseline_5090 * rng.binomial(1, 0.39, row_count)
        panel["quarterLag1size90plus"] = baseline_90
        panel["quarterLag1size50_90"] = baseline_5090
        panel["lag1hiavolsize90plus"] = strong_90
        panel["lag1hiavolsize50_90"] = strong_5090
        panel["control_a"] = rng.normal(size=row_count)
        panel["control_b"] = rng.normal(size=row_count)

        rest_90 = ((baseline_90 == 1) & (strong_90 == 0)).astype(float)
        rest_5090 = (
            (baseline_5090 == 1) & (strong_5090 == 0)
        ).astype(float)
        fe_keys = (
            panel["StkcdRecall"].astype(str)
            + "_"
            + panel["yearQuarter"].astype(str)
        )
        fe_effects = {
            key: value
            for key, value in zip(
                fe_keys.unique(),
                rng.normal(scale=0.025, size=fe_keys.nunique()),
                strict=True,
            )
        }
        panel["chAdjreturn_t_tp1_Recall_w"] = (
            0.045 * strong_90
            + 0.010 * rest_90
            - 0.025 * strong_5090
            + 0.030 * rest_5090
            + 0.008 * panel["control_a"]
            - 0.005 * panel["control_b"]
            + fe_keys.map(fe_effects).astype(float)
            + rng.normal(scale=0.025, size=row_count)
        )
        return panel

    def test_drop_fe_singletons_iterates_until_no_new_singletons(self):
        """若只删除首轮单例，两组固定效应之间新产生的单例会残留。"""
        reg = pd.DataFrame(
            {
                "_fe_left": [0, 0, 1, 1, 2, 2],
                "_fe_right": [0, 1, 1, 2, 3, 3],
                "value": [10, 11, 12, 13, 20, 21],
            },
            index=[101, 103, 107, 109, 113, 127],
        )

        cleaned, removed = table5.drop_fe_singletons_iteratively(
            reg,
            ["_fe_left", "_fe_right"],
        )

        self.assertEqual(cleaned.index.tolist(), [113, 127])
        self.assertEqual(cleaned["value"].tolist(), [20, 21])
        self.assertEqual(removed, 4)

    def test_run_mechanism_regression_returns_two_wald_rows_and_four_coefficients(self):
        """若四个核心变量未同模估计或推断口径错，主结果与明细契约会失真。"""
        panel = self._build_panel()
        outcome = table5.OUTCOME_SPECS[0]
        spec = table5.build_mechanism_specs(1)[0]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            main_rows, coefficient_rows = table5.run_mechanism_regression(
                panel,
                outcome,
                spec,
                ["control_a", "control_b"],
            )

        self.assertEqual(len(main_rows), 2)
        self.assertEqual(len(coefficient_rows), 4)
        self.assertEqual(
            [row["规模组"] for row in main_rows],
            ["90plus", "50_90"],
        )
        self.assertEqual(
            {row["变量"] for row in coefficient_rows},
            {
                "table5_high_total_volume_90plus_strong",
                "table5_high_total_volume_90plus_rest",
                "table5_high_total_volume_50_90_strong",
                "table5_high_total_volume_50_90_rest",
            },
        )

        required_diagnostics = {
            "面板",
            "收益窗口",
            "因变量",
            "机制序号",
            "机制键",
            "机制",
            "控制变量",
            "显式常数项",
            "被提示公司×年度季度固定效应",
            "日期固定效应",
            "聚类方式",
            "有限样本自由度修正",
            "回归样本量",
            "被提示公司聚类数",
            "日期聚类数",
            "固定效应组数",
            "删除单例组观测数",
            "R方",
        }
        for row in main_rows + coefficient_rows:
            self.assertTrue(required_diagnostics.issubset(row))
            self.assertEqual(row["显式常数项"], "是")
            self.assertEqual(row["被提示公司×年度季度固定效应"], "是")
            self.assertEqual(row["日期固定效应"], "否")
            self.assertEqual(row["聚类方式"], "被提示公司和交易日期双向聚类")
            self.assertEqual(row["有限样本自由度修正"], "是")
            self.assertEqual(row["回归样本量"], len(panel))
            self.assertEqual(row["被提示公司聚类数"], 12)
            self.assertEqual(row["日期聚类数"], 48)
            self.assertGreater(row["固定效应组数"], 1)
            self.assertEqual(row["删除单例组观测数"], 0)
            self.assertTrue(np.isfinite(row["R方"]))

        coefficients = {
            row["变量"]: row["系数"]
            for row in coefficient_rows
        }
        coefficient_standard_errors = {
            row["变量"]: row["聚类标准误"]
            for row in coefficient_rows
        }
        coefficient_p_values = {
            row["变量"]: row["p值"]
            for row in coefficient_rows
        }
        for row in main_rows:
            self.assertTrue(
                {
                    "第一组变量",
                    "第二组变量",
                    "第一组系数",
                    "第二组系数",
                    "第一组聚类标准误",
                    "第二组聚类标准误",
                    "第一组聚类标准误（百分点）",
                    "第二组聚类标准误（百分点）",
                    "第一组p值",
                    "第二组p值",
                    "系数差",
                    "系数差（百分点）",
                    "Wald统计量",
                    "Wald自由度",
                    "Wald p值",
                }.issubset(row)
            )
            self.assertAlmostEqual(
                row["第一组系数"],
                coefficients[row["第一组变量"]],
            )
            self.assertAlmostEqual(
                row["第二组系数"],
                coefficients[row["第二组变量"]],
            )
            self.assertTrue(np.isfinite(row["第一组聚类标准误"]))
            self.assertTrue(np.isfinite(row["第二组聚类标准误"]))
            self.assertAlmostEqual(
                row["第一组聚类标准误"],
                coefficient_standard_errors[row["第一组变量"]],
            )
            self.assertAlmostEqual(
                row["第二组聚类标准误"],
                coefficient_standard_errors[row["第二组变量"]],
            )
            self.assertAlmostEqual(
                row["第一组p值"],
                coefficient_p_values[row["第一组变量"]],
            )
            self.assertAlmostEqual(
                row["第二组p值"],
                coefficient_p_values[row["第二组变量"]],
            )
            self.assertAlmostEqual(
                row["第一组聚类标准误（百分点）"],
                row["第一组聚类标准误"] * 100,
            )
            self.assertAlmostEqual(
                row["第二组聚类标准误（百分点）"],
                row["第二组聚类标准误"] * 100,
            )
            self.assertAlmostEqual(
                row["系数差"],
                row["第一组系数"] - row["第二组系数"],
            )
            self.assertAlmostEqual(
                row["系数差（百分点）"],
                row["系数差"] * 100,
            )
            self.assertTrue(np.isfinite(row["Wald统计量"]))
            self.assertEqual(row["Wald自由度"], 1)
            self.assertGreaterEqual(row["Wald p值"], 0.0)
            self.assertLessEqual(row["Wald p值"], 1.0)

    def test_date_fixed_effects_are_absorbed_and_reported_separately(self):
        """日期版必须真实吸收第二组固定效应，而不只是修改结果标签。"""
        panel = self._build_panel()
        main_rows, coefficient_rows = table5.run_mechanism_regression(
            panel,
            table5.OUTCOME_SPECS[0],
            table5.build_mechanism_specs(1)[0],
            ["control_a", "control_b"],
            date_fixed_effects=True,
        )

        for row in main_rows + coefficient_rows:
            self.assertEqual(row["日期固定效应"], "是")
            self.assertEqual(row["日期固定效应组数"], 48)
            self.assertEqual(row["回归样本量"], len(panel))
            self.assertTrue(np.isfinite(row["R方"]))

    def test_absorbed_core_variable_raises_chinese_error_with_model_context(self):
        """若核心变量只有固定效应组间变异，应中文指出机制和收益窗口后停止。"""
        panel = self._build_panel()
        fe_group = (
            panel.groupby(
                ["StkcdRecall", "yearQuarter"],
                observed=True,
                sort=False,
            )
            .ngroup()
            .mod(2)
        )
        panel["quarterLag1size90plus"] = 1
        panel["lag1hiavolsize90plus"] = fe_group

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaisesRegex(
                ValueError,
                (
                    r"异常总成交量机制.*\[t, t\+1\].*"
                    r"chAdjreturn_t_tp1_Recall_w.*固定效应.*吸收"
                ),
            ):
                table5.run_mechanism_regression(
                    panel,
                    table5.OUTCOME_SPECS[0],
                    table5.build_mechanism_specs(1)[0],
                    ["control_a", "control_b"],
                )

    def test_collinear_core_variables_raise_contextual_error_without_warning(self):
        """若核心变量有组内变异但完全共线，应明确报告共线性且不泄露警告。"""
        panel = self._build_panel()
        within_fe_binary = (
            panel.groupby(
                ["StkcdRecall", "yearQuarter"],
                observed=True,
                sort=False,
            )
            .cumcount()
            .mod(2)
        )
        panel["quarterLag1size90plus"] = 1
        panel["quarterLag1size50_90"] = 1
        panel["lag1hiavolsize90plus"] = within_fe_binary
        panel["lag1hiavolsize50_90"] = within_fe_binary

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(ValueError) as context:
                table5.run_mechanism_regression(
                    panel,
                    table5.OUTCOME_SPECS[0],
                    table5.build_mechanism_specs(1)[0],
                    ["control_a", "control_b"],
                )

        message = str(context.exception)
        self.assertIn("异常总成交量机制", message)
        self.assertIn("[t, t+1]", message)
        self.assertIn("chAdjreturn_t_tp1_Recall_w", message)
        self.assertRegex(message, r"完全共线|共线性")
        self.assertNotIn("被固定效应吸收", message)
        self.assertEqual(caught, [])

    def test_non_finite_core_inference_is_not_reported_as_estimated(self):
        """若核心变量聚类标准误非有限，模型必须标记为不可估计而非留下空白。"""
        panel = self._build_panel()
        outcome = table5.OUTCOME_SPECS[0]
        spec = table5.build_mechanism_specs(1)[0]
        original_fit = table5.AbsorbingLS.fit
        target = "table5_high_total_volume_90plus_strong"

        def fit_with_non_finite_standard_error(model, *args, **kwargs):
            fitted = original_fit(model, *args, **kwargs)
            standard_errors = fitted.std_errors.copy()
            standard_errors.loc[target] = np.nan

            class ResultProxy:
                def __getattr__(self, name):
                    if name == "std_errors":
                        return standard_errors
                    return getattr(fitted, name)

            return ResultProxy()

        with patch.object(
            table5.AbsorbingLS,
            "fit",
            new=fit_with_non_finite_standard_error,
        ):
            with self.assertRaisesRegex(
                table5.NonEstimableModelError,
                r"核心变量.*聚类标准误.*非有限",
            ):
                table5.run_mechanism_regression(
                    panel,
                    outcome,
                    spec,
                    ["control_a", "control_b"],
                )

    def test_run_regression_contextualizes_expected_input_errors(self):
        """若输入校验或数值转换失败，错误应保留原因并定位具体模型。"""
        cases = (
            ("非二元核心变量", "只能取0、1或缺失"),
            ("非法强关联子集", "强关联提示字段"),
            ("保留字控制变量", "不能命名为const"),
            ("控制变量数值转换", "数值转换失败"),
        )

        for case_name, expected_reason in cases:
            with self.subTest(case_name=case_name):
                panel = self._build_panel()
                controls = ["control_a", "control_b"]
                if case_name == "非二元核心变量":
                    panel.loc[0, "lag1hiavolsize90plus"] = 2
                elif case_name == "非法强关联子集":
                    panel.loc[0, "quarterLag1size90plus"] = 0
                    panel.loc[0, "lag1hiavolsize90plus"] = 1
                elif case_name == "保留字控制变量":
                    controls = ["const"]
                else:
                    panel["control_a"] = panel["control_a"].astype(object)
                    panel.loc[0, "control_a"] = "无法转换"

                with self.assertRaises(ValueError) as context:
                    table5.run_mechanism_regression(
                        panel,
                        table5.OUTCOME_SPECS[0],
                        table5.build_mechanism_specs(1)[0],
                        controls,
                    )

                message = str(context.exception)
                self.assertIn("异常总成交量机制", message)
                self.assertIn("[t, t+1]", message)
                self.assertIn("chAdjreturn_t_tp1_Recall_w", message)
                self.assertIn(expected_reason, message)

    def test_model_fit_wraps_only_expected_errors(self):
        """模型拟合可预期错误应加上下文，系统级和未知异常必须原样透传。"""
        panel = self._build_panel()
        outcome = table5.OUTCOME_SPECS[0]
        spec = table5.build_mechanism_specs(1)[0]

        expected_error = OverflowError("协方差计算数值上溢")
        with patch.object(
            table5.AbsorbingLS,
            "fit",
            side_effect=expected_error,
        ):
            try:
                table5.run_mechanism_regression(
                    panel,
                    outcome,
                    spec,
                    ["control_a", "control_b"],
                )
            except BaseException as exc:
                observed_error = exc
            else:
                self.fail("模型拟合错误未向调用方抛出。")

        self.assertIsInstance(observed_error, ValueError)
        expected_message = str(observed_error)
        self.assertIn("异常总成交量机制", expected_message)
        self.assertIn("[t, t+1]", expected_message)
        self.assertIn("chAdjreturn_t_tp1_Recall_w", expected_message)
        self.assertIn("模型拟合失败", expected_message)
        self.assertIn("协方差计算数值上溢", expected_message)
        self.assertIs(observed_error.__cause__, expected_error)

        passthrough_errors = (
            MemoryError("内存不足"),
            KeyboardInterrupt(),
            SystemExit(9),
            RuntimeError("未知运行时异常"),
        )
        for original_error in passthrough_errors:
            with self.subTest(error_type=type(original_error).__name__):
                with patch.object(
                    table5.AbsorbingLS,
                    "fit",
                    side_effect=original_error,
                ):
                    with self.assertRaises(type(original_error)) as context:
                        table5.run_mechanism_regression(
                            panel,
                            outcome,
                            spec,
                            ["control_a", "control_b"],
                        )
                self.assertIs(context.exception, original_error)


class Table5OrchestrationTests(unittest.TestCase):
    """保护34个模型的串行编排、跨SUE复用和统一结果契约。"""

    @staticmethod
    def _fake_runner(call_log):
        """返回记录调用键并生成完整轻量结果行的替身估计器。"""
        def runner(data, outcome, spec, control_cols):
            del data
            call_log.append(
                (
                    outcome.panel,
                    spec.order,
                    spec.first_90,
                    spec.first_5090,
                    tuple(control_cols),
                )
            )
            value = float(len(call_log))
            diagnostics = {
                "面板": outcome.panel,
                "收益窗口": outcome.label,
                "因变量": outcome.column,
                "机制序号": spec.order,
                "机制键": spec.key,
                "机制": spec.label,
                "控制变量": "是",
                "被提示公司×年度季度固定效应": "是",
                "聚类方式": "被提示公司和交易日期双向聚类",
                "回归样本量": 1000 + len(call_log),
                "被提示公司聚类数": 50,
                "日期聚类数": 80,
                "删除单例组观测数": 2,
                "R方": value / 100,
            }
            main_rows = []
            coefficient_rows = []
            roles = (
                ("first", "强关联提示", 1),
                ("second", "其他关联提示", 2),
            )
            for size_order, size_key in enumerate(("90plus", "50_90")):
                first_variable = f"{spec.key}_{size_key}_first"
                second_variable = f"{spec.key}_{size_key}_second"
                first_coefficient = value + size_order / 10
                second_coefficient = -value - size_order / 10
                main_rows.append(
                    {
                        **diagnostics,
                        "规模组": size_key,
                        "规模组标签": size_key,
                        "第一组变量": first_variable,
                        "第一组标签": "强关联提示",
                        "第二组变量": second_variable,
                        "第二组标签": "其他关联提示",
                        "第一组系数": first_coefficient,
                        "第二组系数": second_coefficient,
                        "第一组聚类标准误": 0.11,
                        "第二组聚类标准误": 0.22,
                        "第一组p值": 0.04,
                        "第二组p值": 0.20,
                        "系数差": first_coefficient - second_coefficient,
                        "Wald统计量": value,
                        "Wald自由度": 1,
                        "Wald p值": 0.05,
                    }
                )
                for role, label, within_pair_order in roles:
                    variable = (
                        first_variable if role == "first" else second_variable
                    )
                    coefficient_rows.append(
                        {
                            **diagnostics,
                            "规模组": size_key,
                            "规模组标签": size_key,
                            "变量": variable,
                            "变量标签": label,
                            "组内变量顺序": within_pair_order,
                            "系数": (
                                first_coefficient
                                if role == "first"
                                else second_coefficient
                            ),
                            "聚类标准误": (
                                0.11 if role == "first" else 0.22
                            ),
                        }
                    )
            return main_rows, coefficient_rows

        return runner

    def test_run_all_models_calls_34_models_with_shared_mechanisms_only_once(self):
        """若共享机制按SUE重复估计，调用数会从34膨胀到54。"""
        call_log = []
        output = StringIO()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with redirect_stdout(output):
                table5.run_all_models(
                    pd.DataFrame(),
                    regression_runner=self._fake_runner(call_log),
                )

        expected_controls = tuple(
            table5.OWN_EA_COLS
            + table5.DISCLOSURE_COLS
            + table5.MEDIA_CONTROL_COLS
        )
        expected_calls = []
        shared_specs_by_order = {
            spec.order: spec
            for spec in table5.build_mechanism_specs(1)
        }
        for outcome in table5.OUTCOME_SPECS:
            expected_calls.extend(
                (
                    outcome.panel,
                    order,
                    shared_specs_by_order[order].first_90,
                    shared_specs_by_order[order].first_5090,
                )
                for order in (1, 2, 3, 4, 9)
            )
        for sue_id in (1, 2, 3):
            sue_specs_by_order = {
                spec.order: spec
                for spec in table5.build_mechanism_specs(sue_id)
            }
            for outcome in table5.OUTCOME_SPECS:
                expected_calls.extend(
                    (
                        outcome.panel,
                        order,
                        sue_specs_by_order[order].first_90,
                        sue_specs_by_order[order].first_5090,
                    )
                    for order in (5, 6, 7, 8)
                )

        self.assertEqual(len(call_log), 34)
        self.assertEqual(
            [
                (panel, order, first_90, first_5090)
                for panel, order, first_90, first_5090, _ in call_log
            ],
            expected_calls,
        )
        self.assertTrue(
            all(controls == expected_controls for *_, controls in call_log)
        )
        shared_calls = [
            call
            for call in call_log
            if call[1] in (1, 2, 3, 4, 9)
        ]
        self.assertEqual(len(shared_calls), 10)
        self.assertEqual(
            len(
                {
                    (panel, order)
                    for panel, order, _, _, _ in shared_calls
                }
            ),
            10,
        )

        progress_lines = output.getvalue().strip().splitlines()
        self.assertEqual(len(progress_lines), 34)
        self.assertIn("1/34", progress_lines[0])
        self.assertIn("异常总成交量", progress_lines[0])
        self.assertIn("34/34", progress_lines[-1])
        self.assertIn("极端负向盈余惊喜", progress_lines[-1])

    def test_run_all_models_returns_sorted_isolated_tables_with_unified_metadata(self):
        """若结果未复制、未补元数据或未稳定排序，三份SUE表会错位或互相污染。"""
        call_log = []

        with redirect_stdout(StringIO()):
            main_results, coefficient_details = table5.run_all_models(
                pd.DataFrame(),
                regression_runner=self._fake_runner(call_log),
            )

        self.assertEqual(list(main_results), [1, 2, 3])
        self.assertEqual(list(coefficient_details), [1, 2, 3])
        self.assertIsNot(main_results, coefficient_details)
        required_main_columns = {
            "SUE口径",
            "SUE口径说明",
            "面板",
            "收益窗口",
            "因变量",
            "机制编号",
            "机制序号",
            "机制名称",
            "规模组",
            "强关联提示变量",
            "其他关联提示变量",
            "强关联提示系数",
            "强关联提示聚类标准误",
            "强关联提示p值",
            "其他关联提示系数",
            "其他关联提示聚类标准误",
            "其他关联提示p值",
            "系数差",
            "沃尔德统计量",
            "沃尔德自由度",
            "沃尔德p值",
            "回归样本量",
            "被提示公司聚类数",
            "日期聚类数",
            "删除单例组观测数",
            "R方",
            "控制变量",
            "固定效应",
            "聚类方式",
            "口径说明",
        }
        sue_notes = {
            1: "历史未调整每股收益",
            2: "历史调整每股收益",
            3: "分析师预测中位数",
        }
        expected_main_order = [
            (panel, mechanism_order, size_key)
            for panel in ("A", "B")
            for mechanism_order in range(1, 10)
            for size_key in ("90plus", "50_90")
        ]
        expected_coefficient_order = [
            (panel, mechanism_order, size_key, within_pair_order)
            for panel in ("A", "B")
            for mechanism_order in range(1, 10)
            for size_key in ("90plus", "50_90")
            for within_pair_order in (1, 2)
        ]

        for sue_id in (1, 2, 3):
            main = main_results[sue_id]
            details = coefficient_details[sue_id]
            self.assertEqual(main.shape[0], 36)
            self.assertEqual(details.shape[0], 72)
            self.assertTrue(required_main_columns.issubset(main.columns))
            self.assertTrue(
                {
                    "第一类变量",
                    "第一变量",
                    "比较变量",
                    "第二类变量",
                    "第二变量",
                    "第一类系数",
                    "比较类系数",
                    "第二类系数",
                    "第一类聚类标准误",
                    "比较类聚类标准误",
                    "第二类聚类标准误",
                    "第一类p值",
                    "比较类p值",
                    "第二类p值",
                }.isdisjoint(main.columns)
            )
            self.assertEqual(main["SUE口径"].unique().tolist(), [f"SUE{sue_id}"])
            self.assertEqual(
                main["SUE口径说明"].unique().tolist(),
                [sue_notes[sue_id]],
            )
            pd.testing.assert_series_equal(
                main["机制编号"],
                main["机制序号"],
                check_names=False,
            )
            pd.testing.assert_series_equal(
                details["机制编号"],
                details["机制序号"],
                check_names=False,
            )
            self.assertEqual(
                list(
                    main[["面板", "机制序号", "规模组"]]
                    .itertuples(index=False, name=None)
                ),
                expected_main_order,
            )
            self.assertEqual(
                list(
                    details[
                        ["面板", "机制序号", "规模组", "组内变量顺序"]
                    ].itertuples(index=False, name=None)
                ),
                expected_coefficient_order,
            )
            self.assertEqual(
                details["SUE口径"].unique().tolist(),
                [f"SUE{sue_id}"],
            )
            self.assertEqual(
                details["SUE口径说明"].unique().tolist(),
                [sue_notes[sue_id]],
            )

        for sue_id in (2, 3):
            shared_mask_1 = main_results[1]["机制序号"].isin((1, 2, 3, 4, 9))
            shared_mask_n = main_results[sue_id]["机制序号"].isin((1, 2, 3, 4, 9))
            pd.testing.assert_series_equal(
                main_results[1].loc[
                    shared_mask_1,
                    "强关联提示系数",
                ].reset_index(drop=True),
                main_results[sue_id].loc[
                    shared_mask_n,
                    "强关联提示系数",
                ].reset_index(drop=True),
                check_names=False,
            )
            self.assertIsNot(main_results[1], main_results[sue_id])
            self.assertIsNot(
                coefficient_details[1],
                coefficient_details[sue_id],
            )

        original_sue2_value = main_results[2].loc[0, "强关联提示系数"]
        main_results[1].loc[0, "强关联提示系数"] = -999.0
        self.assertEqual(
            main_results[2].loc[0, "强关联提示系数"],
            original_sue2_value,
        )

    def test_non_estimable_models_are_recorded_without_hiding_other_errors(self):
        attempts = []
        successful_call_log = []
        successful_runner = self._fake_runner(successful_call_log)

        def runner(data, outcome, spec, control_cols):
            attempts.append((outcome.panel, spec.order, spec.first_90))
            if spec.order == 7 and "sue3" in spec.first_90:
                raise table5.NonEstimableModelError(
                    "核心变量全样本没有有效变异"
                )
            return successful_runner(data, outcome, spec, control_cols)

        with redirect_stdout(StringIO()):
            main_results, coefficient_details = table5.run_all_models(
                pd.DataFrame(),
                regression_runner=runner,
            )

        self.assertEqual(len(attempts), 34)
        self.assertEqual(len(main_results[3]), 36)
        self.assertEqual(len(coefficient_details[3]), 72)
        unavailable_main = main_results[3].loc[
            main_results[3]["机制编号"].eq(7)
        ]
        unavailable_details = coefficient_details[3].loc[
            coefficient_details[3]["机制编号"].eq(7)
        ]
        self.assertEqual(len(unavailable_main), 4)
        self.assertEqual(len(unavailable_details), 8)
        self.assertEqual(
            unavailable_main["估计状态"].unique().tolist(),
            ["不可估计"],
        )
        self.assertTrue(unavailable_main["强关联提示系数"].isna().all())
        self.assertTrue(unavailable_main["其他关联提示系数"].isna().all())
        self.assertTrue(
            unavailable_main["不可估计原因"]
            .str.contains("没有有效变异")
            .all()
        )

        def invalid_runner(data, outcome, spec, control_cols):
            del data, outcome, spec, control_cols
            raise ValueError("核心字段不是二元变量")

        with self.assertRaisesRegex(ValueError, "不是二元变量"):
            table5.run_all_models(
                pd.DataFrame(),
                regression_runner=invalid_runner,
            )


class Table5OutputTests(unittest.TestCase):
    """保护三份结构化结果和六张单窗口论文式图片的输出契约。"""

    @staticmethod
    def _build_fake_results():
        call_log = []
        with redirect_stdout(StringIO()):
            return table5.run_all_models(
                pd.DataFrame(),
                regression_runner=Table5OrchestrationTests._fake_runner(
                    call_log
                ),
            )

    def test_format_helpers_use_stars_parenthesized_se_and_compact_p_values(self):
        self.assertEqual(table5.significance_stars(0.009), "***")
        self.assertEqual(table5.significance_stars(0.049), "**")
        self.assertEqual(table5.significance_stars(0.099), "*")
        self.assertEqual(table5.significance_stars(0.10), "")
        self.assertEqual(
            table5.format_coefficient(1.23456, 0.0456, 0.049),
            "1.235**\n(0.046)",
        )
        self.assertEqual(table5.format_p_value(0.0004), "<0.001")
        self.assertEqual(table5.format_p_value(0.04789), "0.048")

    def test_write_outputs_creates_three_csv_pairs_and_six_window_images(self):
        main_results, coefficient_details = self._build_fake_results()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_dir = root / "results"
            image_dir = root / "images"
            written = table5.write_outputs(
                main_results,
                coefficient_details,
                result_dir,
                image_dir,
            )

            self.assertEqual(len(written), 12)
            for sue_id in (1, 2, 3):
                summary_path = (
                    result_dir / f"表5_SUE{sue_id}_心理机制结果.csv"
                )
                detail_path = (
                    result_dir / f"表5_SUE{sue_id}_心理机制系数明细.csv"
                )
                main_image_path = (
                    image_dir / f"table5_SUE{sue_id}_心理机制.png"
                )
                extension_image_path = (
                    image_dir
                    / f"table5_SUE{sue_id}_心理机制_t_tp10扩展分析.png"
                )
                self.assertTrue(summary_path.is_file())
                self.assertTrue(detail_path.is_file())
                self.assertTrue(main_image_path.is_file())
                self.assertTrue(extension_image_path.is_file())
                self.assertEqual(
                    summary_path.read_bytes()[:3],
                    b"\xef\xbb\xbf",
                )
                self.assertEqual(
                    detail_path.read_bytes()[:3],
                    b"\xef\xbb\xbf",
                )

                summary = pd.read_csv(summary_path)
                detail = pd.read_csv(detail_path)
                self.assertEqual(len(summary), 36)
                self.assertEqual(len(detail), 72)
                self.assertEqual(set(summary["面板"]), {"A", "B"})
                self.assertEqual(summary["机制编号"].nunique(), 9)
                self.assertTrue(
                    {
                        "强关联提示系数",
                        "强关联提示聚类标准误",
                        "其他关联提示系数",
                        "其他关联提示聚类标准误",
                        "沃尔德p值",
                    }.issubset(summary.columns)
                )

                for image_path in (main_image_path, extension_image_path):
                    pixels = mpl_image.imread(image_path)
                    self.assertGreaterEqual(pixels.shape[0], 2_475)
                    self.assertLessEqual(pixels.shape[0], 2_485)
                    self.assertGreaterEqual(pixels.shape[1], 3_500)
                    self.assertLessEqual(pixels.shape[1], 3_510)
                    self.assertGreater(float(np.nanstd(pixels)), 0.01)

    def test_date_fe_outputs_use_separate_names_and_a4_images(self):
        main_results, coefficient_details = self._build_fake_results()
        for tables in (main_results, coefficient_details):
            for table in tables.values():
                table["日期固定效应"] = "是"
                table["固定效应"] = (
                    "被提示公司×年度季度固定效应、交易日期固定效应"
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            written = table5.write_outputs(
                main_results,
                coefficient_details,
                root / "results",
                root / "images",
                date_fixed_effects=True,
            )

            self.assertEqual(len(written), 12)
            for sue_id in (1, 2, 3):
                summary = (
                    root
                    / "results"
                    / f"表5_SUE{sue_id}_心理机制_加入日期固定效应结果.csv"
                )
                main_image = (
                    root
                    / "images"
                    / f"table5_SUE{sue_id}_心理机制_加入日期固定效应.png"
                )
                extension_image = (
                    root
                    / "images"
                    / (
                        f"table5_SUE{sue_id}_心理机制_t_tp10扩展分析"
                        "_加入日期固定效应.png"
                    )
                )
                self.assertTrue(summary.is_file())
                self.assertTrue(main_image.is_file())
                self.assertTrue(extension_image.is_file())
                self.assertFalse(
                    (
                        root
                        / "images"
                        / f"table5_SUE{sue_id}_心理机制.png"
                    ).exists()
                )
                pixels = mpl_image.imread(main_image)
                self.assertGreaterEqual(pixels.shape[0], 2_475)
                self.assertLessEqual(pixels.shape[0], 2_485)
                self.assertGreaterEqual(pixels.shape[1], 3_500)
                self.assertLessEqual(pixels.shape[1], 3_510)


class Table5CliTests(unittest.TestCase):
    """保护独立脚本的参数、紧凑读取和端到端入口。"""

    @staticmethod
    def _write_minimal_input(path: Path, *, duplicate_key: bool = False):
        row_count = 4
        values = {
            column: np.zeros(row_count, dtype="float32")
            for column in table5.required_input_columns()
            if column not in {"StkcdRecall", "Trddt", "yearQuarter"}
        }
        values.update(
            {
                "StkcdRecall": ["1", " 2 ", "000003", "000004"],
                "Trddt": [
                    "2019-12-31",
                    "2020-01-02",
                    "2020-04-01",
                    "2021-01-04",
                ],
                "yearQuarter": ["2019Q4", "2020Q1", "2020Q2", "2021Q1"],
                "year": [2019, 2020, 2020, 2021],
            }
        )
        frame = pd.DataFrame(values)
        if duplicate_key:
            frame.loc[1, ["StkcdRecall", "Trddt"]] = frame.loc[
                0,
                ["StkcdRecall", "Trddt"],
            ]
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    def test_parse_args_supports_defaults_and_all_runtime_options(self):
        defaults = table5.parse_args([])
        self.assertEqual(defaults.input, table5.DEFAULT_INPUT_PATH)
        self.assertEqual(defaults.result_dir, table5.DEFAULT_RESULT_DIR)
        self.assertEqual(defaults.image_dir, table5.DEFAULT_IMAGE_DIR)
        self.assertEqual(defaults.execution_mode, "auto")
        self.assertEqual(defaults.cache_path, table5.DEFAULT_CACHE_PATH)
        self.assertEqual(defaults.chunk_size, 250_000)
        self.assertFalse(defaults.rebuild_cache)
        self.assertFalse(defaults.keep_cache)
        self.assertFalse(defaults.date_fixed_effects)
        self.assertIsNone(defaults.start_year)
        self.assertIsNone(defaults.end_year)
        self.assertIsNone(defaults.nrows)
        self.assertFalse(defaults.check_only)

        custom = table5.parse_args(
            [
                "--input",
                "input.csv",
                "--result-dir",
                "results",
                "--image-dir",
                "images",
                "--execution-mode",
                "low-memory",
                "--cache-path",
                "work/cache.parquet",
                "--chunk-size",
                "100000",
                "--rebuild-cache",
                "--keep-cache",
                "--date-fixed-effects",
                "--start-year",
                "2010",
                "--end-year",
                "2020",
                "--nrows",
                "500",
                "--check-only",
            ]
        )
        self.assertEqual(custom.input, Path("input.csv"))
        self.assertEqual(custom.result_dir, Path("results"))
        self.assertEqual(custom.image_dir, Path("images"))
        self.assertEqual(custom.execution_mode, "low-memory")
        self.assertEqual(custom.cache_path, Path("work/cache.parquet"))
        self.assertEqual(custom.chunk_size, 100_000)
        self.assertTrue(custom.rebuild_cache)
        self.assertTrue(custom.keep_cache)
        self.assertTrue(custom.date_fixed_effects)
        self.assertEqual(custom.start_year, 2010)
        self.assertEqual(custom.end_year, 2020)
        self.assertEqual(custom.nrows, 500)
        self.assertTrue(custom.check_only)

    def test_resolve_execution_mode_uses_memory_for_nrows_and_cache_for_full(self):
        """若自动模式路由错误，小样本会变慢或全样本会再次耗尽内存。"""
        sampled = table5.parse_args(["--nrows", "100"])
        full = table5.parse_args([])
        explicit = table5.parse_args(
            ["--execution-mode", "in-memory"]
        )

        self.assertEqual(
            table5.resolve_execution_mode(sampled),
            "in-memory",
        )
        self.assertEqual(
            table5.resolve_execution_mode(full),
            "low-memory",
        )
        self.assertEqual(
            table5.resolve_execution_mode(explicit),
            "in-memory",
        )

    def test_load_input_data_uses_compact_types_filters_and_normalizes_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "panel.csv"
            self._write_minimal_input(path)
            loaded = table5.load_input_data(
                path,
                start_year=2020,
                end_year=2020,
            )

        self.assertEqual(len(loaded), 2)
        self.assertEqual(
            loaded["StkcdRecall"].astype("string").tolist(),
            ["000002", "000003"],
        )
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(loaded["Trddt"])
        )
        self.assertIsInstance(
            loaded["StkcdRecall"].dtype,
            pd.CategoricalDtype,
        )
        self.assertIsInstance(
            loaded["yearQuarter"].dtype,
            pd.CategoricalDtype,
        )
        self.assertEqual(str(loaded["year"].dtype), "float32")
        self.assertEqual(
            str(loaded["quarterLag1size90plus"].dtype),
            "float32",
        )
        self.assertFalse(
            loaded.duplicated(["StkcdRecall", "Trddt"]).any()
        )

    def test_load_input_data_rejects_duplicate_company_date_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicate.csv"
            self._write_minimal_input(path, duplicate_key=True)
            with self.assertRaisesRegex(
                ValueError,
                r"StkcdRecall.*Trddt.*唯一键",
            ):
                table5.load_input_data(path)

    def test_main_runs_header_load_models_and_twelve_output_writes(self):
        fake_data = pd.DataFrame(
            {
                "StkcdRecall": pd.Categorical(["000001"]),
                "marker": [1],
            }
        )
        fake_main = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
        fake_details = {
            1: pd.DataFrame(),
            2: pd.DataFrame(),
            3: pd.DataFrame(),
        }
        fake_paths = [Path(f"file_{index}") for index in range(12)]

        with (
            patch.object(table5, "read_and_validate_header") as header,
            patch.object(
                table5,
                "load_input_data",
                return_value=fake_data,
            ) as loader,
            patch.object(
                table5,
                "run_all_models",
                return_value=(fake_main, fake_details),
            ) as runner,
            patch.object(
                table5,
                "write_outputs",
                return_value=fake_paths,
            ) as writer,
            redirect_stdout(StringIO()) as output,
        ):
            table5.main(
                [
                    "--input",
                    "input.csv",
                    "--result-dir",
                    "results",
                    "--image-dir",
                    "images",
                    "--start-year",
                    "2010",
                    "--end-year",
                    "2020",
                    "--nrows",
                    "500",
                ]
            )

        header.assert_called_once_with(Path("input.csv"))
        loader.assert_called_once_with(
            Path("input.csv"),
            start_year=2010,
            end_year=2020,
            nrows=500,
        )
        runner.assert_called_once_with(fake_data)
        writer.assert_called_once_with(
            fake_main,
            fake_details,
            Path("results"),
            Path("images"),
        )
        self.assertIn("34个模型", output.getvalue())
        self.assertIn("12个输出文件", output.getvalue())

    def test_main_low_memory_reuses_cache_and_cleans_only_after_success(self):
        """若成功前删除缓存或命中缓存仍重建，长任务会无法安全重试。"""
        fake_main = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
        fake_details = {
            1: pd.DataFrame(),
            2: pd.DataFrame(),
            3: pd.DataFrame(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache = root / "panel.parquet"
            manifest = table5.cache_manifest_path(cache)
            cache.write_bytes(b"cache")
            manifest.write_text("{}", encoding="utf-8")

            with (
                patch.object(table5, "read_and_validate_header"),
                patch.object(
                    table5,
                    "cache_is_reusable",
                    return_value=True,
                ),
                patch.object(table5, "build_low_memory_cache") as builder,
                patch.object(
                    table5,
                    "run_all_models_low_memory",
                    side_effect=AssertionError("禁止逐模型整表估计"),
                ),
                patch.object(
                    table5,
                    "run_all_models_streaming",
                    return_value=(fake_main, fake_details),
                ) as runner,
                patch.object(
                    table5,
                    "write_outputs",
                    return_value=[Path(f"file_{index}") for index in range(12)],
                ),
                redirect_stdout(StringIO()),
            ):
                table5.main(
                    [
                        "--execution-mode",
                        "low-memory",
                        "--input",
                        "input.csv",
                        "--cache-path",
                        str(cache),
                    ]
                )

            builder.assert_not_called()
            runner.assert_called_once_with(
                cache,
                batch_size=table5.DEFAULT_CHUNK_SIZE,
            )
            self.assertFalse(cache.exists())
            self.assertFalse(manifest.exists())

    def test_main_low_memory_preserves_cache_when_regression_fails(self):
        """若模型失败时缓存被删除，修复后必须重新解析37GB源文件。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache = root / "panel.parquet"
            manifest = table5.cache_manifest_path(cache)
            cache.write_bytes(b"cache")
            manifest.write_text("{}", encoding="utf-8")

            with (
                patch.object(table5, "read_and_validate_header"),
                patch.object(
                    table5,
                    "cache_is_reusable",
                    return_value=True,
                ),
                patch.object(
                    table5,
                    "run_all_models_streaming",
                    side_effect=RuntimeError("回归失败"),
                ),
                patch.object(table5, "write_outputs") as writer,
                redirect_stdout(StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "回归失败"):
                    table5.main(
                        [
                            "--execution-mode",
                            "low-memory",
                            "--input",
                            "input.csv",
                            "--cache-path",
                            str(cache),
                        ]
                    )

            writer.assert_not_called()
            self.assertTrue(cache.exists())
            self.assertTrue(manifest.exists())

    def test_main_rebuilds_requested_cache_and_keeps_it_after_success(self):
        """若显式重建或保留参数失效，用户无法控制昂贵缓存的生命周期。"""
        fake_main = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
        fake_details = {
            1: pd.DataFrame(),
            2: pd.DataFrame(),
            3: pd.DataFrame(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache = root / "panel.parquet"
            manifest_path = table5.cache_manifest_path(cache)
            cache.write_bytes(b"old-cache")
            manifest_path.write_text("{}", encoding="utf-8")

            def rebuild(*args, **kwargs):
                del args, kwargs
                cache.write_bytes(b"new-cache")
                manifest_path.write_text('{"complete": true}', encoding="utf-8")
                return {"row_count": 4, "company_count": 4}

            with (
                patch.object(table5, "read_and_validate_header"),
                patch.object(
                    table5,
                    "cache_is_reusable",
                    return_value=True,
                ),
                patch.object(
                    table5,
                    "build_low_memory_cache",
                    side_effect=rebuild,
                ) as builder,
                patch.object(
                    table5,
                    "run_all_models_streaming",
                    return_value=(fake_main, fake_details),
                ) as runner,
                patch.object(
                    table5,
                    "write_outputs",
                    return_value=[Path(f"file_{index}") for index in range(12)],
                ),
                redirect_stdout(StringIO()),
            ):
                table5.main(
                    [
                        "--execution-mode",
                        "low-memory",
                        "--input",
                        "input.csv",
                        "--cache-path",
                        str(cache),
                        "--chunk-size",
                        "2",
                        "--rebuild-cache",
                        "--keep-cache",
                    ]
                )

            builder.assert_called_once_with(
                Path("input.csv"),
                cache,
                start_year=None,
                end_year=None,
                nrows=None,
                chunk_size=2,
            )
            runner.assert_called_once_with(cache, batch_size=2)
            self.assertEqual(cache.read_bytes(), b"new-cache")
            self.assertTrue(manifest_path.exists())


class Table5LowMemoryCacheTests(unittest.TestCase):
    """保护分块列式缓存、跨块唯一键和缓存复用契约。"""

    def test_build_low_memory_cache_preserves_rows_types_and_manifest(self):
        """若分块写入丢行或未规范键，低内存结果会偏离内存模式。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "panel.csv"
            cache = root / "panel.parquet"
            Table5CliTests._write_minimal_input(source)

            manifest = table5.build_low_memory_cache(
                source,
                cache,
                chunk_size=2,
            )
            cached = pd.read_parquet(cache)

            self.assertEqual(len(cached), 4)
            self.assertEqual(manifest["row_count"], 4)
            self.assertTrue(manifest["complete"])
            self.assertEqual(
                cached["StkcdRecall"].astype("string").tolist(),
                ["000001", "000002", "000003", "000004"],
            )
            self.assertTrue(
                pd.api.types.is_datetime64_any_dtype(cached["Trddt"])
            )
            self.assertEqual(str(cached["year"].dtype), "float32")
            self.assertTrue(table5.cache_manifest_path(cache).is_file())

    def test_build_low_memory_cache_rejects_duplicate_keys_across_chunks(self):
        """若只查块内重复，跨块重复的公司交易日会进入正式回归。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "duplicate.csv"
            cache = root / "duplicate.parquet"
            Table5CliTests._write_minimal_input(source)
            frame = pd.read_csv(source)
            frame.loc[3, ["StkcdRecall", "Trddt"]] = frame.loc[
                0,
                ["StkcdRecall", "Trddt"],
            ]
            frame.to_csv(source, index=False, encoding="utf-8-sig")

            with self.assertRaisesRegex(
                ValueError,
                r"StkcdRecall.*Trddt.*唯一键",
            ):
                table5.build_low_memory_cache(
                    source,
                    cache,
                    chunk_size=2,
                )

            self.assertFalse(cache.exists())
            self.assertFalse(table5.cache_manifest_path(cache).exists())

    def test_cache_reuse_requires_an_exact_source_and_filter_fingerprint(self):
        """若源文件或筛选参数变化仍复用缓存，正式结果会读取过期样本。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "panel.csv"
            cache = root / "panel.parquet"
            Table5CliTests._write_minimal_input(source)
            table5.build_low_memory_cache(
                source,
                cache,
                start_year=2020,
                end_year=2021,
                chunk_size=2,
            )

            self.assertTrue(
                table5.cache_is_reusable(
                    source,
                    cache,
                    start_year=2020,
                    end_year=2021,
                )
            )
            self.assertFalse(
                table5.cache_is_reusable(
                    source,
                    cache,
                    start_year=2019,
                    end_year=2021,
                )
            )

            source.write_bytes(source.read_bytes() + b"\n")
            self.assertFalse(
                table5.cache_is_reusable(
                    source,
                    cache,
                    start_year=2020,
                    end_year=2021,
                )
            )

    def test_load_cached_model_data_reads_only_one_models_columns(self):
        """若读取无关机制列，逐模型运行仍会浪费全样本内存。"""
        panel = Table5RegressionTests._build_panel()
        panel["unused_marker"] = 99
        outcome = table5.OUTCOME_SPECS[0]
        spec = table5.build_mechanism_specs(1)[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "model.parquet"
            panel.to_parquet(cache, index=False)
            loaded = table5.load_cached_model_data(
                cache,
                outcome,
                spec,
                ["control_a", "control_b"],
            )

        expected_columns = [
            "chAdjreturn_t_tp1_Recall_w",
            "StkcdRecall",
            "Trddt",
            "yearQuarter",
            "control_a",
            "control_b",
            "quarterLag1size90plus",
            "quarterLag1size50_90",
            "lag1hiavolsize90plus",
            "lag1hiavolsize50_90",
        ]
        self.assertEqual(list(loaded.columns), expected_columns)
        self.assertNotIn("unused_marker", loaded)
        self.assertIsInstance(
            loaded["StkcdRecall"].dtype,
            pd.CategoricalDtype,
        )
        self.assertIsInstance(
            loaded["yearQuarter"].dtype,
            pd.CategoricalDtype,
        )

    def test_run_all_models_low_memory_loads_and_releases_each_attempt(self):
        """若模型数据未逐次释放，34次估计会重新累积成整表内存压力。"""
        load_calls = []
        live_frames = []
        runner_calls = []
        runner = Table5OrchestrationTests._fake_runner(runner_calls)

        def loader(cache_path, outcome, spec, control_cols):
            self.assertEqual(Path(cache_path), Path("cache.parquet"))
            load_calls.append(
                (
                    outcome.panel,
                    spec.order,
                    spec.first_90,
                    tuple(control_cols),
                )
            )
            frame = pd.DataFrame({"marker": [len(load_calls)]})
            live_frames.append(weakref.ref(frame))
            return frame

        with redirect_stdout(StringIO()):
            main_results, coefficient_details = (
                table5.run_all_models_low_memory(
                    Path("cache.parquet"),
                    regression_runner=runner,
                    data_loader=loader,
                )
            )

        self.assertEqual(len(load_calls), 34)
        self.assertEqual(len(runner_calls), 34)
        self.assertEqual(
            [(panel, order) for panel, order, _, _ in load_calls[:10]],
            [
                ("A", 1),
                ("A", 2),
                ("A", 3),
                ("A", 4),
                ("A", 9),
                ("B", 1),
                ("B", 2),
                ("B", 3),
                ("B", 4),
                ("B", 9),
            ],
        )
        self.assertEqual(
            [(panel, order) for panel, order, _, _ in load_calls[-8:]],
            [
                ("A", 5),
                ("A", 6),
                ("A", 7),
                ("A", 8),
                ("B", 5),
                ("B", 6),
                ("B", 7),
                ("B", 8),
            ],
        )
        self.assertTrue(all(reference() is None for reference in live_frames))
        self.assertEqual(
            {sue_id: len(table) for sue_id, table in main_results.items()},
            {1: 36, 2: 36, 3: 36},
        )
        self.assertEqual(
            {
                sue_id: len(table)
                for sue_id, table in coefficient_details.items()
            },
            {1: 72, 2: 72, 3: 72},
        )

    def test_cached_model_regression_matches_in_memory_regression(self):
        """若列式往返改变类型或行序，回归推断会与原内存模式不一致。"""
        panel = Table5RegressionTests._build_panel()
        outcome = table5.OUTCOME_SPECS[0]
        spec = table5.build_mechanism_specs(1)[0]
        controls = ["control_a", "control_b"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            expected_main, expected_details = (
                table5.run_mechanism_regression(
                    panel,
                    outcome,
                    spec,
                    controls,
                )
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "model.parquet"
            panel.to_parquet(cache, index=False)
            cached_panel = table5.load_cached_model_data(
                cache,
                outcome,
                spec,
                controls,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                actual_main, actual_details = (
                    table5.run_mechanism_regression(
                        cached_panel,
                        outcome,
                        spec,
                        controls,
                    )
                )

        pd.testing.assert_frame_equal(
            pd.DataFrame(actual_main),
            pd.DataFrame(expected_main),
            check_dtype=False,
            rtol=1e-10,
            atol=1e-12,
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame(actual_details),
            pd.DataFrame(expected_details),
            check_dtype=False,
            rtol=1e-10,
            atol=1e-12,
        )


class Table5StreamingRegressionTests(unittest.TestCase):
    """保护正式全样本分块估计与现有黄金估计器的数值口径。"""

    def test_streaming_regression_matches_absorbingls_across_batches(self):
        """若跨块固定效应或双向聚类累计错误，完整推断会偏离黄金结果。"""
        panel = Table5RegressionTests._build_panel()
        outcome = table5.OUTCOME_SPECS[0]
        spec = table5.build_mechanism_specs(1)[0]
        controls = ["control_a", "control_b"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            expected_main, expected_details = (
                table5.run_mechanism_regression(
                    panel,
                    outcome,
                    spec,
                    controls,
                )
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "model.parquet"
            panel.to_parquet(
                cache,
                index=False,
                row_group_size=37,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                actual_main, actual_details = (
                    table5.run_mechanism_regression_streaming(
                        cache,
                        outcome,
                        spec,
                        controls,
                        batch_size=29,
                    )
                )

        pd.testing.assert_frame_equal(
            pd.DataFrame(actual_main),
            pd.DataFrame(expected_main),
            check_dtype=False,
            rtol=1e-9,
            atol=1e-11,
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame(actual_details),
            pd.DataFrame(expected_details),
            check_dtype=False,
            rtol=1e-9,
            atol=1e-11,
        )

    def test_streaming_date_fe_matches_two_way_absorbingls(self):
        """两维分块残差化必须与内存双固定效应黄金估计器一致。"""
        panel = Table5RegressionTests._build_panel()
        outcome = table5.OUTCOME_SPECS[0]
        spec = table5.build_mechanism_specs(1)[0]
        controls = ["control_a", "control_b"]
        expected_main, expected_details = table5.run_mechanism_regression(
            panel,
            outcome,
            spec,
            controls,
            date_fixed_effects=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "model.parquet"
            panel.to_parquet(
                cache,
                index=False,
                row_group_size=37,
            )
            actual_main, actual_details = (
                table5.run_mechanism_regression_streaming(
                    cache,
                    outcome,
                    spec,
                    controls,
                    batch_size=29,
                    date_fixed_effects=True,
                )
            )

        pd.testing.assert_frame_equal(
            pd.DataFrame(actual_main),
            pd.DataFrame(expected_main),
            check_dtype=False,
            rtol=1e-8,
            atol=1e-10,
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame(actual_details),
            pd.DataFrame(expected_details),
            check_dtype=False,
            rtol=1e-8,
            atol=1e-10,
        )
    def test_streaming_regression_never_calls_the_full_frame_loader(self):
        """若正式估计仍整表读取，1183万行模型会再次超过物理内存。"""
        panel = Table5RegressionTests._build_panel()
        outcome = table5.OUTCOME_SPECS[0]
        spec = table5.build_mechanism_specs(1)[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "model.parquet"
            panel.to_parquet(
                cache,
                index=False,
                row_group_size=41,
            )
            with patch.object(
                table5,
                "load_cached_model_data",
                side_effect=AssertionError("禁止整表读取"),
            ):
                main_rows, coefficient_rows = (
                    table5.run_mechanism_regression_streaming(
                        cache,
                        outcome,
                        spec,
                        ["control_a", "control_b"],
                        batch_size=31,
                    )
                )

        self.assertEqual(len(main_rows), 2)
        self.assertEqual(len(coefficient_rows), 4)

    def test_streaming_orchestration_routes_all_34_models_without_full_frames(self):
        """若正式编排仍走逐模型整表路径，单模型分块实现不会真正生效。"""
        runner_calls = []
        legacy_runner = Table5OrchestrationTests._fake_runner(runner_calls)

        def streaming_runner(
            cache_path,
            outcome,
            spec,
            control_cols,
            *,
            batch_size,
        ):
            self.assertEqual(Path(cache_path), Path("cache.parquet"))
            self.assertEqual(batch_size, 17)
            return legacy_runner(
                pd.DataFrame(),
                outcome,
                spec,
                control_cols,
            )

        with redirect_stdout(StringIO()):
            main_results, coefficient_details = (
                table5.run_all_models_streaming(
                    Path("cache.parquet"),
                    batch_size=17,
                    regression_runner=streaming_runner,
                )
            )

        self.assertEqual(len(runner_calls), 34)
        self.assertEqual(
            {key: len(value) for key, value in main_results.items()},
            {1: 36, 2: 36, 3: 36},
        )
        self.assertEqual(
            {key: len(value) for key, value in coefficient_details.items()},
            {1: 72, 2: 72, 3: 72},
        )


if __name__ == "__main__":
    unittest.main()
