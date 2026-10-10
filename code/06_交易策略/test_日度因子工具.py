from pathlib import Path
import importlib.util

import pandas as pd


def load_tools():
    module_path = Path(__file__).with_name("日度因子工具.py")
    spec = importlib.util.spec_from_file_location("daily_factor_tools", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weighted_mean_by_group_excludes_missing_returns():
    tools = load_tools()
    df = pd.DataFrame(
        {
            "Trddt": ["2020-01-02", "2020-01-02", "2020-01-02"],
            "bucket": ["H", "H", "H"],
            "ret": [0.10, 0.00, None],
            "weight": [2.0, 1.0, 100.0],
        }
    )

    out = tools.weighted_mean_by_group(
        df,
        group_cols=["Trddt", "bucket"],
        return_col="ret",
        weight_col="weight",
        value_name="leg_ret",
    )

    assert len(out) == 1
    assert abs(out.loc[0, "leg_ret"] - (0.10 * 2.0 + 0.00 * 1.0) / 3.0) < 1e-12
    assert out.loc[0, "n_stocks"] == 2


def test_make_high_low_factor_supports_both_directions():
    tools = load_tools()
    wide = pd.DataFrame(
        {
            "Trddt": ["2020-01-02"],
            "H": [0.08],
            "L": [0.02],
        }
    )

    mom = tools.make_high_low_factor(
        wide,
        high_col="H",
        low_col="L",
        output_col="mom",
        long_high=True,
    )
    st_rev = tools.make_high_low_factor(
        wide,
        high_col="H",
        low_col="L",
        output_col="st_rev",
        long_high=False,
    )

    assert mom.loc[0, "mom"] == 0.06
    assert st_rev.loc[0, "st_rev"] == -0.06


def test_make_ff_2x3_factors_uses_standard_smb_hml_formulas():
    tools = load_tools()
    legs = pd.DataFrame(
        {
            "Trddt": ["2020-01-02"] * 6,
            "size_port": ["S", "S", "S", "B", "B", "B"],
            "char_port": ["L", "M", "H", "L", "M", "H"],
            "leg_ret": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )

    out = tools.make_ff_2x3_factors(
        legs,
        date_col="Trddt",
        size_col="size_port",
        char_col="char_port",
        return_col="leg_ret",
        smb_col="smb",
        hml_col="hml",
    )

    assert abs(out.loc[0, "smb"] - ((0.01 + 0.02 + 0.03) / 3 - (0.04 + 0.05 + 0.06) / 3)) < 1e-12
    assert abs(out.loc[0, "hml"] - (0.5 * (0.03 + 0.06) - 0.5 * (0.01 + 0.04))) < 1e-12


if __name__ == "__main__":
    test_weighted_mean_by_group_excludes_missing_returns()
    test_make_high_low_factor_supports_both_directions()
    test_make_ff_2x3_factors_uses_standard_smb_hml_formulas()
    print("all tests passed")
