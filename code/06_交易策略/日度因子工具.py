from __future__ import annotations

from typing import Sequence

import pandas as pd


def weighted_mean_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    return_col: str,
    weight_col: str,
    value_name: str,
) -> pd.DataFrame:
    """Calculate value-weighted returns by group.

    Missing returns and non-positive weights are excluded from both numerator
    and denominator.
    """
    cols = list(group_cols) + [return_col, weight_col]
    tmp = df.loc[:, cols].copy()
    tmp[return_col] = pd.to_numeric(tmp[return_col], errors="coerce")
    tmp[weight_col] = pd.to_numeric(tmp[weight_col], errors="coerce")

    valid = tmp[return_col].notna() & tmp[weight_col].notna() & tmp[weight_col].gt(0)
    tmp = tmp.loc[valid].copy()

    if tmp.empty:
        return pd.DataFrame(columns=list(group_cols) + [value_name, "n_stocks", "sum_weight"])

    tmp["_weighted_ret"] = tmp[return_col] * tmp[weight_col]
    out = (
        tmp.groupby(list(group_cols), observed=True, dropna=False)
        .agg(
            sum_weighted_ret=("_weighted_ret", "sum"),
            sum_weight=(weight_col, "sum"),
            n_stocks=(return_col, "count"),
        )
        .reset_index()
    )
    out[value_name] = out["sum_weighted_ret"] / out["sum_weight"]
    return out.loc[:, list(group_cols) + [value_name, "n_stocks", "sum_weight"]]


def make_high_low_factor(
    wide: pd.DataFrame,
    high_col: str,
    low_col: str,
    output_col: str,
    long_high: bool,
) -> pd.DataFrame:
    """Build a long-short factor from precomputed high and low leg returns."""
    out = wide.copy()
    if long_high:
        out[output_col] = out[high_col] - out[low_col]
    else:
        out[output_col] = out[low_col] - out[high_col]
    return out


def make_ff_2x3_factors(
    legs: pd.DataFrame,
    date_col: str,
    size_col: str,
    char_col: str,
    return_col: str,
    smb_col: str,
    hml_col: str,
) -> pd.DataFrame:
    """Build Fama-French-style SMB and HML from six 2 x 3 portfolio returns."""
    wide = legs.pivot_table(
        index=date_col,
        columns=[size_col, char_col],
        values=return_col,
        aggfunc="first",
    )

    required = [("S", "L"), ("S", "M"), ("S", "H"), ("B", "L"), ("B", "M"), ("B", "H")]
    for col in required:
        if col not in wide.columns:
            wide[col] = pd.NA

    out = pd.DataFrame(index=wide.index)
    out[smb_col] = (
        (wide[("S", "L")] + wide[("S", "M")] + wide[("S", "H")]) / 3
        - (wide[("B", "L")] + wide[("B", "M")] + wide[("B", "H")]) / 3
    )
    out[hml_col] = (
        (wide[("S", "H")] + wide[("B", "H")]) / 2
        - (wide[("S", "L")] + wide[("B", "L")]) / 2
    )

    return out.reset_index()
