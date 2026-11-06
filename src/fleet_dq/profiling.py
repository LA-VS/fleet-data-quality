"""Profile each column of the extract: completeness, distinct values and the shapes values take."""

from __future__ import annotations

import re

import pandas as pd


def shape(value) -> str:
    """Letters become A, digits 9, spaces are shown, so '07/26' and '31/07/2026' look different."""
    if not isinstance(value, str):
        return "(blank)"
    s = re.sub(r"[A-Za-z]", "A", value)
    s = re.sub(r"\d", "9", s)
    s = re.sub(r"A+", "A", s)
    return s.replace(" ", "<nbsp>").replace(" ", "␣")


def profile(df: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        shapes = s.map(shape).value_counts()
        rows.append(
            {
                "column": col,
                "complete": s.notna().mean(),
                "distinct": s.nunique(),
                "shapes": len(shapes),
                "most common shapes": ", ".join(f"{k} ({v:,})" for k, v in shapes.head(top).items()),
            }
        )
    return pd.DataFrame(rows).set_index("column")
