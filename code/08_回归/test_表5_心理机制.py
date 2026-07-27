"""表5心理机制脚本的规格映射和输入表头测试。"""

from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

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
    "positiveNews",
    "negativeNews",
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
        """若其余提示未排除强联系或跨规模覆盖，两个比较组会被错误污染。"""
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
        """若强联系提示不属于对应基准提示，回归比较的样本定义不成立。"""
        spec = table5.build_mechanism_specs(1)[0]
        data = pd.DataFrame(
            {
                "quarterLag1size90plus": [0],
                "quarterLag1size50_90": [1],
                "lag1hiavolsize90plus": [1],
                "lag1hiavolsize50_90": [0],
            }
        )

        with self.assertRaisesRegex(ValueError, "强联系提示.*基准提示"):
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
        """若干扰机制生成其余提示，会把低高干扰比较误写成基准提示分割。"""
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
        self.assertEqual([pair.first_label for pair in pairs], ["低干扰提示", "低干扰提示"])
        self.assertEqual([pair.second_label for pair in pairs], ["高干扰提示", "高干扰提示"])
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
        for row in main_rows:
            self.assertTrue(
                {
                    "第一组变量",
                    "第二组变量",
                    "第一组系数",
                    "第二组系数",
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


if __name__ == "__main__":
    unittest.main()
