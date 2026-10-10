from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib-cache"))

MODULE_PATH = Path(__file__).with_name("绘制回归表格.py")
SPEC = importlib.util.spec_from_file_location("regression_table_plotter", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载绘图脚本：{MODULE_PATH}")

PLOTTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTTER)


class Table2PaginationTests(unittest.TestCase):
    def test_split_after_own_ea_t_plus_3(self) -> None:
        variables = [
            "quarterLag1size90plus",
            "ownEA_tm1",
            "ownEA",
            "ownEA_tp1",
            "ownEA_tp2",
            "ownEA_tp3",
            "ownEA_tp4",
            "aesIncrease",
        ]

        first_page, second_page = PLOTTER.split_table2_variable_order(
            variables,
            split_after_variable="ownEA_tp3",
        )

        self.assertEqual(first_page[-1], "ownEA_tp3")
        self.assertEqual(second_page[0], "ownEA_tp4")
        self.assertEqual(first_page + second_page, variables)

    def test_continued_page_does_not_add_rule_before_second_variable(self) -> None:
        summary = PLOTTER.read_result(
            PLOTTER.TABLE2_SUMMARY_PATH,
            [
                "样本",
                "样本说明",
                "收益窗口",
                "规格",
                "线索系数（百分点）",
                "聚类标准误（百分点）",
                "p值",
                "回归样本量",
                "R方",
            ],
        )
        details = PLOTTER.read_result(
            PLOTTER.TABLE2_DETAIL_PATH,
            [
                "样本",
                "收益窗口",
                "规格",
                "变量",
                "变量顺序",
                "系数（百分点）",
                "聚类标准误（百分点）",
                "p值",
            ],
        )
        model_rows = PLOTTER.table2_model_rows(summary, "pattern")
        lookup = PLOTTER._table2_detail_lookup(details, "pattern")

        rows = PLOTTER._table2_full_coefficient_rows(
            model_rows,
            lookup,
            ["ownEA_tp4", "ownEA_tp5"],
        )

        second_coefficient_row = rows[2]
        self.assertFalse(second_coefficient_row.get("line_before", False))

    def test_r_squared_label_uses_math_text(self) -> None:
        summary = PLOTTER.read_result(
            PLOTTER.TABLE2_SUMMARY_PATH,
            [
                "样本",
                "样本说明",
                "收益窗口",
                "规格",
                "线索系数（百分点）",
                "聚类标准误（百分点）",
                "p值",
                "回归样本量",
                "R方",
            ],
        )
        model_rows = PLOTTER.table2_model_rows(summary, "pattern")

        stat_rows = PLOTTER._table2_full_stat_rows(model_rows)

        self.assertEqual(stat_rows[-1]["label"], r"$R^2$")

    def test_pattern_full_table_exports_two_images_and_two_page_pdf(self) -> None:
        self.assertAlmostEqual(
            PLOTTER.A4_LANDSCAPE_WIDTH_IN * 25.4,
            297.0,
            places=6,
        )
        self.assertAlmostEqual(
            PLOTTER.A4_LANDSCAPE_HEIGHT_IN * 25.4,
            210.0,
            places=6,
        )
        summary = PLOTTER.read_result(
            PLOTTER.TABLE2_SUMMARY_PATH,
            [
                "样本",
                "样本说明",
                "收益窗口",
                "规格",
                "线索系数（百分点）",
                "聚类标准误（百分点）",
                "p值",
                "回归样本量",
                "R方",
            ],
        )
        details = PLOTTER.read_result(
            PLOTTER.TABLE2_DETAIL_PATH,
            [
                "样本",
                "收益窗口",
                "规格",
                "变量",
                "变量顺序",
                "系数（百分点）",
                "聚类标准误（百分点）",
                "p值",
            ],
        )

        with tempfile.TemporaryDirectory(dir=ROOT / "work") as temp_dir:
            temp_path = Path(temp_dir)
            first_image = temp_path / "table2_page1.png"
            second_image = temp_path / "table2_page2.png"
            pdf_output = temp_path / "table2.pdf"

            PLOTTER.draw_table2_full_two_pages(
                summary,
                details,
                sample="pattern",
                first_page_output=first_image,
                second_page_output=second_image,
                pdf_output=pdf_output,
            )

            for output_path in [first_image, second_image, pdf_output]:
                self.assertTrue(output_path.exists(), output_path)
                self.assertGreater(output_path.stat().st_size, 1_000)

            expected_width = round(PLOTTER.A4_LANDSCAPE_WIDTH_IN * 300)
            expected_height = round(PLOTTER.A4_LANDSCAPE_HEIGHT_IN * 300)
            for image_path in [first_image, second_image]:
                image = PLOTTER.plt.imread(image_path)
                actual_height, actual_width = image.shape[:2]
                self.assertLessEqual(abs(actual_width - expected_width), 1)
                self.assertLessEqual(abs(actual_height - expected_height), 1)

            self.assertEqual(pdf_output.read_bytes()[:4], b"%PDF")

    def test_full_sample_exports_two_landscape_a4_images_without_pdf(self) -> None:
        summary = PLOTTER.read_result(
            PLOTTER.TABLE2_SUMMARY_PATH,
            [
                "样本",
                "样本说明",
                "收益窗口",
                "规格",
                "线索系数（百分点）",
                "聚类标准误（百分点）",
                "p值",
                "回归样本量",
                "R方",
            ],
        )
        details = PLOTTER.read_result(
            PLOTTER.TABLE2_DETAIL_PATH,
            [
                "样本",
                "收益窗口",
                "规格",
                "变量",
                "变量顺序",
                "系数（百分点）",
                "聚类标准误（百分点）",
                "p值",
            ],
        )

        with tempfile.TemporaryDirectory(dir=ROOT / "work") as temp_dir:
            temp_path = Path(temp_dir)
            first_image = temp_path / "table2_full_page1.png"
            second_image = temp_path / "table2_full_page2.png"

            PLOTTER.draw_table2_full_two_pages(
                summary,
                details,
                sample="full",
                first_page_output=first_image,
                second_page_output=second_image,
                pdf_output=None,
            )

            expected_width = round(PLOTTER.A4_LANDSCAPE_WIDTH_IN * 300)
            expected_height = round(PLOTTER.A4_LANDSCAPE_HEIGHT_IN * 300)
            for image_path in [first_image, second_image]:
                self.assertTrue(image_path.exists(), image_path)
                self.assertGreater(image_path.stat().st_size, 1_000)
                image = PLOTTER.plt.imread(image_path)
                actual_height, actual_width = image.shape[:2]
                self.assertLessEqual(abs(actual_width - expected_width), 1)
                self.assertLessEqual(abs(actual_height - expected_height), 1)

    def test_full_sample_continued_output_name(self) -> None:
        self.assertEqual(
            PLOTTER.TABLE2_ALL_FULL_CONTINUED_OUTPUT.name,
            "table2_收益回归_完整系数版_完整样本_续页.png",
        )


class Table4DateFixedEffectsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = PLOTTER.read_result(
            PLOTTER.TABLE4_RESULTS_PATH,
            [
                "规格",
                "样本",
                "因变量",
                "控制变量",
                "被提示公司×年度季度固定效应",
                "回归样本量",
                "R方",
                "变量",
                "变量顺序",
                "系数（百分点）",
                "聚类标准误（百分点）",
                "p值",
            ],
        )

    def test_date_fe_paths_do_not_overwrite_existing_outputs(self) -> None:
        self.assertNotEqual(
            PLOTTER.TABLE4_DATE_FE_RESULTS_PATH,
            PLOTTER.TABLE4_RESULTS_PATH,
        )
        self.assertNotEqual(
            PLOTTER.TABLE4_DATE_FE_EXTENSION_OUTPUT,
            PLOTTER.TABLE4_EXTENSION_OUTPUT,
        )

    def test_date_fe_row_is_only_added_when_requested(self) -> None:
        results = self.results.copy()
        results["日期固定效应"] = "是"

        original_rows = PLOTTER.build_table4_rows(results)
        date_fe_rows = PLOTTER.build_table4_rows(
            results,
            show_date_fixed_effects=True,
        )

        self.assertNotIn(
            "日期固定效应",
            [row["label"] for row in original_rows],
        )

        matched = [
            row
            for row in date_fe_rows
            if row["label"] == "日期固定效应"
        ]
        self.assertEqual(len(matched), 1)
        self.assertEqual(
            matched[0]["values"],
            ["是", "是", "是"],
        )

    def test_date_fe_plot_rejects_results_without_date_fe(self) -> None:
        results = self.results.copy()
        results["日期固定效应"] = "否"

        with tempfile.TemporaryDirectory(dir=ROOT / "work") as temp_dir:
            output = Path(temp_dir) / "invalid_table4.png"
            with self.assertRaisesRegex(
                ValueError,
                "日期固定效应",
            ):
                PLOTTER.draw_table4_extension(
                    results,
                    output,
                    show_date_fixed_effects=True,
                )


if __name__ == "__main__":
    unittest.main()
