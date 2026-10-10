"""DATAGO 报刊新闻聚合器的纯内存行为测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("聚合DATAGO报刊新闻.py")
SPEC = importlib.util.spec_from_file_location("datago_news_aggregator", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - 测试加载保护
    raise RuntimeError(f"无法加载模块: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DatagoNewsAggregationTests(unittest.TestCase):
    def test_score_classification_supports_explicit_neutral_band(self):
        scores = pd.Series(
            [0.20, -0.20, 0.04, -0.04, 0.0, np.nan, np.inf, -np.inf]
        )

        actual = MODULE.classify_sentiment(scores, neutral_band=0.05).tolist()

        self.assertEqual(
            actual,
            [
                "positive",
                "negative",
                "neutral",
                "neutral",
                "neutral",
                "unclassified",
                "unclassified",
                "unclassified",
            ],
        )

    def test_aggregation_counts_distinct_news_and_preserves_title_content_flags(self):
        relations = pd.DataFrame(
            {
                "news_id": ["n1", "n1", "n2", "n3", "n3", "n4"],
                "pub_date": [
                    "2020-01-01 08:00",
                    "2020-01-01 08:00",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                ],
                "stkcd": ["1", "1", "1", "1", "2", "1"],
                "is_ment_in_title": [1, 0, 0, 1, 1, 0],
                "is_ment_in_content": [1, 1, 1, 1, 1, 1],
            }
        )
        sentiment = pd.DataFrame(
            {
                "news_id": ["n1", "n2", "n3"],
                "pub_date": ["2020-01-01"] * 3,
                "overall_senti_score_adjust": [0.8, 0.0, -0.2],
            }
        )

        actual = MODULE.aggregate_daily_news(relations, sentiment)
        self.assertEqual(actual.attrs["sentiment_total_articles"], 4)
        self.assertEqual(actual.attrs["sentiment_unclassified_articles"], 1)
        actual = actual.set_index(["Scode", "Date"])

        stock_one = actual.loc[("000001", pd.Timestamp("2020-01-01"))]
        self.assertEqual(stock_one["Newsnum_Title"], 2)
        self.assertEqual(stock_one["Newsnum_Cont"], 4)
        self.assertEqual(stock_one["Allnews_All"], 4)
        self.assertEqual(stock_one["Posnews_All"], 1)
        self.assertEqual(stock_one["Neunews_All"], 1)
        self.assertEqual(stock_one["Negnews_All"], 1)
        self.assertEqual(stock_one["Unclassifiednews_All"], 1)

        stock_two = actual.loc[("000002", pd.Timestamp("2020-01-01"))]
        self.assertEqual(stock_two["Newsnum_Title"], 1)
        self.assertEqual(stock_two["Allnews_All"], 1)
        self.assertEqual(stock_two["Negnews_All"], 1)

    def test_conflicting_sentiment_rows_for_one_news_raise(self):
        sentiment = pd.DataFrame(
            {
                "news_id": ["n1", "n1"],
                "pub_date": ["2020-01-01", "2020-01-01"],
                "overall_senti_score_adjust": [0.1, 0.2],
            }
        )

        with self.assertRaises(ValueError):
            MODULE.prepare_sentiment(sentiment)


if __name__ == "__main__":
    unittest.main()
