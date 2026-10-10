from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name(
    "图2扩展_单日特征调整收益_加入日期固定效应.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("figure2_single_day", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载待测试脚本：{MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestSingleDayOutcomeAlignment(unittest.TestCase):
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

        self.raw_daily = pd.DataFrame(
            {
                "Stkcd": [
                    "000001",
                    "000001",
                    "000001",
                    "000002",
                    "000002",
                ],
                "td": [10, 11, 12, 10, 11],
                "Trddt": pd.to_datetime(
                    [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-02",
                        "2020-01-03",
                    ]
                ),
                "chAdj_ret_t": [0.01, 0.02, 0.03, 0.04, 0.05],
            }
        )

    def test_horizon_zero_reproduces_current_daily_adjusted_return(self):
        lookup, _, _ = MODULE.prepare_daily_return_lookup(
            self.raw_daily,
            winsor_tail=0.0,
        )

        aligned = MODULE.attach_daily_outcome(
            self.base,
            lookup,
            horizon=0,
        )

        np.testing.assert_allclose(
            aligned[MODULE.OUTCOME_VALUE_COL].to_numpy(),
            np.array([0.01, 0.04]),
        )
        self.assertTrue(
            aligned[MODULE.OUTCOME_TD_COL]
            .eq(aligned["td"])
            .all()
        )

    def test_horizon_uses_exact_market_trading_day_offset(self):
        lookup, _, _ = MODULE.prepare_daily_return_lookup(
            self.raw_daily,
            winsor_tail=0.0,
        )

        aligned = MODULE.attach_daily_outcome(
            self.base,
            lookup,
            horizon=2,
        )

        self.assertEqual(aligned.loc[0, MODULE.OUTCOME_TD_COL], 12)
        self.assertAlmostEqual(
            aligned.loc[0, MODULE.OUTCOME_VALUE_COL],
            0.03,
        )

    def test_missing_target_day_remains_missing_instead_of_zero(self):
        lookup, _, _ = MODULE.prepare_daily_return_lookup(
            self.raw_daily,
            winsor_tail=0.0,
        )

        aligned = MODULE.attach_daily_outcome(
            self.base,
            lookup,
            horizon=2,
        )

        self.assertTrue(
            pd.isna(
                aligned.loc[
                    aligned["StkcdRecall"].eq("000002"),
                    MODULE.OUTCOME_VALUE_COL,
                ]
            ).all()
        )

    def test_common_sample_requires_every_requested_horizon(self):
        lookup, _, _ = MODULE.prepare_daily_return_lookup(
            self.raw_daily,
            winsor_tail=0.0,
        )

        mask = MODULE.build_common_sample_mask(
            self.base,
            lookup,
            horizons=[1, 2],
        )

        self.assertListEqual(mask.tolist(), [True, False])

    def test_base_and_daily_source_must_share_the_same_td_calendar(self):
        lookup, _, _ = MODULE.prepare_daily_return_lookup(
            self.raw_daily,
            winsor_tail=0.0,
        )
        inconsistent = lookup.copy()
        inconsistent.loc[
            inconsistent[MODULE.OUTCOME_TD_COL].eq(10),
            MODULE.OUTCOME_DATE_COL,
        ] = pd.Timestamp("2020-01-01")

        with self.assertRaisesRegex(ValueError, "交易日序号口径不一致"):
            MODULE.validate_shared_td_calendar(
                self.base,
                inconsistent,
            )

    def test_duplicate_stock_day_in_daily_lookup_is_rejected(self):
        duplicated = pd.concat(
            [self.raw_daily, self.raw_daily.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "股票和交易日序号不唯一"):
            MODULE.prepare_daily_return_lookup(
                duplicated,
                winsor_tail=0.0,
            )

    def test_negative_horizon_is_rejected(self):
        lookup, _, _ = MODULE.prepare_daily_return_lookup(
            self.raw_daily,
            winsor_tail=0.0,
        )

        with self.assertRaisesRegex(ValueError, "不能为负数"):
            MODULE.attach_daily_outcome(
                self.base,
                lookup,
                horizon=-1,
            )


if __name__ == "__main__":
    unittest.main()
