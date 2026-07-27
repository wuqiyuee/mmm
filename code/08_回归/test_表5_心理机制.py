"""表5心理机制脚本的规格映射和输入表头测试。"""

from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
