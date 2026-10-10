from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name(
    "图2扩展_等长两日非重叠窗口特征调整收益_加入日期固定效应.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("figure2_two_day_windows", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载待测试脚本：{MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestTwoDayWindowDefinitions(unittest.TestCase):
    def test_twenty_five_equal_windows_cover_t_through_t49(self):
        specs = MODULE.WINDOW_SPECS

        self.assertEqual(len(specs), 25)
        self.assertEqual(
            [(spec.start, spec.end) for spec in specs],
            [(start, start + 1) for start in range(0, 50, 2)],
        )
        self.assertTrue(all(spec.length == 2 for spec in specs))
        self.assertTrue(
            all(spec.source_column == "chAdj_ret_t_t1" for spec in specs)
        )

    def test_window_validation_accepts_current_definition(self):
        MODULE.validate_window_specs()


class TestTwoDayOutcomeAlignment(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame(
            {
                "StkcdRecall": ["000001", "000002"],
                "td": [10, 10],
                "Trddt": pd.to_datetime(["2020-01-02", "2020-01-02"]),
                "yearQuarter": ["2020Q1", "2020Q1"],
                "quarterLag1size90plus": [1.0, 0.0],
            }
        )
        self.raw_returns = pd.DataFrame(
            {
                "Stkcd": [
                    "000001",
                    "000001",
                    "000001",
                    "000002",
                    "000002",
                ],
                "td": [10, 12, 14, 10, 12],
                "Trddt": pd.to_datetime(
                    [
                        "2020-01-02",
                        "2020-01-06",
                        "2020-01-08",
                        "2020-01-02",
                        "2020-01-06",
                    ]
                ),
                "chAdj_ret_t_t1": [0.01, 0.02, 0.03, 0.04, np.nan],
            }
        )

    def test_window_uses_its_own_start_day_two_day_return(self):
        lookup, _ = MODULE.prepare_two_day_return_lookup(
            self.raw_returns,
            winsor_tail=0.0,
        )

        aligned = MODULE.attach_two_day_outcome(
            self.base,
            lookup,
            MODULE.WINDOW_SPECS[1],
        )

        self.assertEqual(aligned.loc[0, MODULE.SOURCE_TD_COL], 12)
        self.assertAlmostEqual(
            aligned.loc[0, MODULE.OUTCOME_VALUE_COL],
            0.02,
        )
        self.assertTrue(
            pd.isna(
                aligned.loc[
                    aligned["StkcdRecall"].eq("000002"),
                    MODULE.OUTCOME_VALUE_COL,
                ]
            ).all()
        )

    def test_common_sample_requires_all_requested_window_starts(self):
        lookup, _ = MODULE.prepare_two_day_return_lookup(
            self.raw_returns,
            winsor_tail=0.0,
        )

        mask = MODULE.build_common_sample_mask(
            self.base,
            lookup,
            specs=MODULE.WINDOW_SPECS[:3],
        )

        self.assertListEqual(mask.tolist(), [True, False])

    def test_duplicate_source_stock_day_is_rejected(self):
        duplicated = pd.concat(
            [self.raw_returns, self.raw_returns.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "股票和交易日序号不唯一"):
            MODULE.prepare_two_day_return_lookup(
                duplicated,
                winsor_tail=0.0,
            )


if __name__ == "__main__":
    unittest.main()
