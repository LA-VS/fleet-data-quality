"""Isolation Forest over the shape of each card's dates, for problems no rule is written for."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

from fleet_dq.clean import EXTRACT

FEATURES = ["validity_days", "lead_days", "age_days", "update_lag_days", "expiry_from_extract_days"]


def features(fixed: pd.DataFrame) -> pd.DataFrame:
    valid_from = fixed["valid_from"].fillna(fixed["created_date"])
    days = lambda delta: delta.dt.days  # noqa: E731
    return pd.DataFrame(
        {
            "validity_days": days(fixed["expiry_date"] - valid_from),
            "lead_days": days(valid_from - fixed["created_date"]),
            "age_days": days(EXTRACT - fixed["created_date"]),
            "update_lag_days": days(fixed["last_updated"].dt.normalize() - fixed["created_date"]),
            "expiry_from_extract_days": days(fixed["expiry_date"] - EXTRACT),
        },
        index=fixed.index,
    ).fillna(0)


def isolation_forest(fixed: pd.DataFrame, review_share: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Score every row and flag the most isolated `review_share` of rows for a person to look at."""
    x = features(fixed)
    model = IsolationForest(n_estimators=300, contamination=review_share, random_state=seed).fit(x)
    out = x.copy()
    out["score"] = -model.score_samples(x)
    out["anomalous"] = model.predict(x) == -1
    return out
