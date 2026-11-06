"""Entity resolution: collapse duplicate rows into one record per card, with survivorship rules.

The card number is the natural key, but Excel rounding destroys its last three digits, and cards on the
same account share their first fifteen. A rounded number is matched back to a full one only when exactly
one full number has the same first fifteen digits and the same vehicle registration.
"""

from __future__ import annotations

import pandas as pd

from fleet_dq.cards import EXCEL_DIGITS, luhn_valid
from fleet_dq.clean import EXTRACT
from fleet_dq.generate import STATUS

# Survivorship rules.
# Status comes from the latest row, because status is the field that changes over a card's life.
# Every other field takes the most common value across the card's rows, so one bad row is outvoted;
# a tie goes to the latest row. Rows the Isolation Forest flags vote only when no other row can.
MAJORITY = (
    "company",
    "vehicle_reg",
    "fuel_network",
    "card_type",
    "restriction_code",
    "created_date",
    "valid_from",
    "expiry_date",
)


def _full(number) -> bool:
    return isinstance(number, str) and len(number) == 18 and luhn_valid(number) and not number.endswith("000")


def assign_keys(fixed: pd.DataFrame) -> pd.DataFrame:
    """A resolution key per row, and whether its card number was intact, recovered or unrecoverable."""
    full = fixed["card_number"].map(_full).astype(bool)
    prefix = fixed["card_number"].str[:EXCEL_DIGITS]
    reg = fixed["vehicle_reg"]
    candidates = (
        pd.DataFrame({"prefix": prefix[full], "reg": reg[full], "number": fixed.loc[full, "card_number"]})
        .drop_duplicates()
        .groupby(["prefix", "reg"])["number"]
        .agg(lambda s: s.iloc[0] if len(s) == 1 else None)
    )
    lookup = candidates.to_dict()

    keys = pd.DataFrame({"key": fixed["card_number"].where(full), "number_status": "intact"}, index=fixed.index)
    for i in fixed.index[~full]:
        match = lookup.get((prefix[i], reg[i]))
        if match:
            keys.loc[i, ["key", "number_status"]] = [match, "recovered"]
        else:
            keys.loc[i, ["key", "number_status"]] = [f"UNRESOLVED {prefix[i]} {reg[i]}", "unrecoverable"]
    return keys


def _majority(rows: pd.DataFrame, col: str) -> pd.Series:
    """Per key, the value with most votes; ties go to the value seen most recently."""
    votes = rows.loc[rows["voter"] & rows[col].notna(), ["key", col, "last_updated"]]
    tally = votes.groupby(["key", col], sort=False).agg(n=("last_updated", "size"), latest=("last_updated", "max"))
    tally = tally.reset_index().sort_values(["key", "n", "latest"], ascending=[True, False, False], kind="stable")
    return tally.drop_duplicates("key").set_index("key")[col]


def resolve(fixed: pd.DataFrame, anomalous: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the golden register (one row per key), each row's key, and the flags resolution raises."""
    keys = assign_keys(fixed)
    rows = fixed.join(keys)
    rows["anomalous"] = False if anomalous is None else anomalous.reindex(rows.index).fillna(False).astype(bool)
    has_normal = (~rows["anomalous"]).groupby(rows["key"]).transform("any")
    rows["voter"] = ~rows["anomalous"] | ~has_normal
    rows = rows.sort_values(["key", "last_updated"], ascending=[True, False], kind="stable")

    grouped = rows.groupby("key", sort=True)
    golden = pd.DataFrame(
        {
            "source_rows": grouped.size(),
            "status_code": grouped["status_code"].first(),
            "last_updated": grouped["last_updated"].max(),
        }
    )
    for col in MAJORITY:
        golden[col] = _majority(rows, col).reindex(golden.index)
    recovered_only = grouped["number_status"].agg(lambda s: (s == "recovered").all())
    golden["number_status"] = "intact"
    golden.loc[recovered_only, "number_status"] = "recovered"
    unresolved = golden.index.str.startswith("UNRESOLVED")
    golden.loc[unresolved, "number_status"] = "unrecoverable"
    golden["card_number"] = golden.index.where(~unresolved, None)

    golden.loc[(golden["status_code"] == "A") & (golden["expiry_date"] < EXTRACT), "status_code"] = "E"
    golden["status_description"] = golden["status_code"].map(STATUS)

    later = rows[rows.groupby("key").cumcount() > 0]
    duplicate = pd.DataFrame({"row_id": later.index, "column": "*", "defect": "duplicate"})
    stale = later[later["status_code"] != later["key"].map(golden["status_code"])]
    conflicting = pd.DataFrame({"row_id": stale.index, "column": "status_code", "defect": "conflicting_duplicate"})
    flags = pd.concat([duplicate, conflicting], ignore_index=True).assign(rule="entity_resolution")
    return golden, keys, flags
