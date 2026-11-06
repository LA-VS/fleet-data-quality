"""Quality scorecard: the share of rows that pass every rule in each of five dimensions."""

from __future__ import annotations

import pandas as pd

DIMENSIONS = {
    "Completeness": ("blank",),
    "Validity": (
        "precision_loss",
        "hidden_whitespace",
        "mixed_date_format",
        "placeholder_date",
        "name_variant",
        "invalid_value",
    ),
    "Uniqueness": ("duplicate",),
    "Consistency": ("swapped_dates", "status_mismatch", "reg_mismatch", "conflicting_duplicate"),
    "Timeliness": ("active_on_expired",),
}


def scorecard(flags: pd.DataFrame, n_rows: int) -> pd.Series:
    out = {}
    for dimension, defects in DIMENSIONS.items():
        failing = flags.loc[flags["defect"].isin(defects), "row_id"].nunique()
        out[dimension] = 1 - failing / n_rows
    return pd.Series(out, name="share of rows passing")
