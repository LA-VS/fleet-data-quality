"""Standardise the extract (formats and spellings), then fix the logical defects row by row."""

from __future__ import annotations

import pandas as pd

from fleet_dq.generate import EXTRACT_DATE, PLACEHOLDER_DATE, STATUS, TEXT_COLUMNS
from fleet_dq.names import clean_text, cluster_names

EXTRACT = pd.Timestamp(EXTRACT_DATE)
PLACEHOLDER = pd.Timestamp(PLACEHOLDER_DATE)
DEFAULT_THRESHOLD = 88.0  # chosen by evaluate.tune_threshold on a separate register (seed 101)


def parse_dates(values: pd.Series) -> pd.Series:
    """Read dd/mm/yyyy, ISO timestamps and MM/YY (taken as the last day of the month, when cards expire)."""
    s = values.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[us]")
    dmy = s.str.fullmatch(r"\d{2}/\d{2}/\d{4}").fillna(False).astype(bool)
    iso = s.str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}").fillna(False).astype(bool)
    mmyy = s.str.fullmatch(r"\d{2}/\d{2}").fillna(False).astype(bool)
    out[dmy] = pd.to_datetime(s[dmy], format="%d/%m/%Y", errors="coerce")  # 29/02 in a non leap year
    out[iso] = pd.to_datetime(s[iso], format="%Y-%m-%dT%H:%M:%S").dt.normalize()
    out[mmyy] = pd.to_datetime("01/" + s[mmyy], format="%d/%m/%y") + pd.offsets.MonthEnd(0)
    return out


def standardise(raw: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Same rows, cleaned formats. Keeps the original company spelling next to the canonical one."""
    std = pd.DataFrame(index=raw.index)
    std["card_number"] = raw["card_number"].map(clean_text)
    for col in TEXT_COLUMNS:
        if col in raw:
            std[col] = raw[col].map(clean_text)
    std["company_raw"] = std["company"]
    std["company"] = cluster_names(std["company_raw"], threshold)
    if "vehicle_reg" in std:
        std["vehicle_reg"] = std["vehicle_reg"].str.upper()
    if "embossed_reg" in std:
        std["embossed_reg"] = std["embossed_reg"].str.upper()
    for col in ("card_type", "restriction_code", "status_code"):
        std[col] = raw[col].map(clean_text)
    std["restriction_code"] = std["restriction_code"].str.upper()
    std["status_code"] = std["status_code"].str.upper()
    for col in ("created_date", "valid_from", "expiry_date"):
        std[col] = parse_dates(raw[col])
    std["last_updated"] = pd.to_datetime(raw["last_updated"].map(clean_text), format="%d/%m/%Y %H:%M")
    return std


def fix(std: pd.DataFrame) -> pd.DataFrame:
    """Repair what can be repaired within one row. Each rule says which value it trusts and why."""
    out = std.copy()

    # Expiry before creation is a swap: card lifetimes are two or three years, never negative.
    swapped = out["expiry_date"] < out["created_date"]
    out.loc[swapped, ["created_date", "expiry_date"]] = out.loc[swapped, ["expiry_date", "created_date"]].to_numpy()

    # 01/01/1990 is a system default, not a date. The real date is unknown, so it becomes blank.
    out.loc[out["valid_from"] == PLACEHOLDER, "valid_from"] = pd.NaT

    # A card still Active after its expiry date has expired.
    expired = (out["status_code"] == "A") & (out["expiry_date"] < EXTRACT)
    out.loc[expired, "status_code"] = "E"

    # The status code is what the card system acts on, so the description is rebuilt from it.
    out["status_description"] = out["status_code"].map(STATUS)

    # The embossed registration repeats the vehicle field and is the one that drifts, so it is dropped.
    return out.drop(columns=["embossed_reg"], errors="ignore")
