"""The whole pipeline: standardise, validate, fix, look for anomalies, resolve, then validate the output."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fleet_dq.anomaly import isolation_forest
from fleet_dq.clean import DEFAULT_THRESHOLD, fix, standardise
from fleet_dq.generate import DATE_COLUMNS, DATE_FORMAT, TIMESTAMP_FORMAT
from fleet_dq.resolve import resolve
from fleet_dq.rules import detect
from fleet_dq.scorecard import scorecard

OUTPUT_COLUMNS = [
    "card_number",
    "company",
    "vehicle_reg",
    "fuel_network",
    "card_type",
    "restriction_code",
    "status_code",
    "status_description",
    "created_date",
    "valid_from",
    "expiry_date",
    "last_updated",
]


@dataclass
class Result:
    std: pd.DataFrame
    fixed: pd.DataFrame
    flags: pd.DataFrame  # rules and entity resolution, against the extract's rows
    anomalies: pd.DataFrame
    golden: pd.DataFrame  # one row per resolved card
    keys: pd.DataFrame
    golden_flags: pd.DataFrame  # the same rules run on the cleaned register

    def scorecards(self, n_raw_rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {"Before": scorecard(self.flags, n_raw_rows), "After": scorecard(self.golden_flags, len(self.golden))}
        )


def as_extract(golden: pd.DataFrame) -> pd.DataFrame:
    """The cleaned register in the extract's text format, so the same rules can be run on it."""
    out = golden.reset_index(drop=True)[OUTPUT_COLUMNS].copy()
    for col in DATE_COLUMNS:
        out[col] = pd.to_datetime(out[col]).dt.strftime(DATE_FORMAT)
    out["last_updated"] = pd.to_datetime(out["last_updated"]).dt.strftime(TIMESTAMP_FORMAT)
    out = out.astype(object).where(out.notna(), None)
    out.index.name = "row_id"
    return out


def run(raw: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD, review_share: float = 0.01) -> Result:
    std = standardise(raw, threshold)
    rule_flags = detect(raw, std)
    fixed = fix(std)
    anomalies = isolation_forest(fixed, review_share)
    golden, keys, resolution_flags = resolve(fixed, anomalies["anomalous"])
    flags = pd.concat([rule_flags, resolution_flags], ignore_index=True)

    cleaned = as_extract(golden)
    golden_flags = detect(cleaned, standardise(cleaned, threshold))
    duplicated = cleaned["card_number"].notna() & cleaned["card_number"].duplicated(keep="first")
    golden_flags = pd.concat(
        [golden_flags, pd.DataFrame({"row_id": cleaned.index[duplicated], "column": "*", "defect": "duplicate"})],
        ignore_index=True,
    )
    return Result(std, fixed, flags, anomalies, golden, keys, golden_flags)
