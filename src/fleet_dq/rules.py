"""Rule based validation with pandera, in two passes.

The raw schema checks formats on the extract exactly as delivered. The logical schema checks the
standardised rows, where dates are real dates and names are clustered. Each failed check becomes a flag:
a row, a column and the defect type the rule is written to catch.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from fleet_dq.cards import luhn_valid
from fleet_dq.clean import EXTRACT, PLACEHOLDER
from fleet_dq.generate import CARD_TYPES, FUEL_NETWORKS, RESTRICTIONS, STATUS

FLAG_COLUMNS = ["row_id", "column", "defect", "rule"]
HIDDEN_WHITESPACE = r" |^\s|\s$|\s{2,}"


def _check(fn, name: str) -> pa.Check:
    return pa.Check(fn, name=name)


NO_HIDDEN_WHITESPACE = _check(lambda s: ~s.str.contains(HIDDEN_WHITESPACE, regex=True), "no_hidden_whitespace")
DD_MM_YYYY = _check(lambda s: s.str.fullmatch(r"\d{2}/\d{2}/\d{4}"), "date_is_dd_mm_yyyy")
DATE_EXISTS = _check(
    lambda s: ~s.str.fullmatch(r"\d{2}/\d{2}/\d{4}") | pd.to_datetime(s, format="%d/%m/%Y", errors="coerce").notna(),
    "date_exists",
)


def raw_schema(has_embossed: bool = True) -> pa.DataFrameSchema:
    text = [NO_HIDDEN_WHITESPACE]
    columns = {
        "card_number": pa.Column(
            None,
            [
                _check(lambda s: s.str.fullmatch(r"\d{18}"), "card_number_has_18_digits"),
                _check(lambda s: s.map(luhn_valid), "card_number_passes_luhn"),
                _check(lambda s: ~s.str.endswith("000"), "card_number_not_excel_rounded"),
            ],
        ),
        "company": pa.Column(None, text),
        "vehicle_reg": pa.Column(None, text),
        "fuel_network": pa.Column(None, [*text, _check(lambda s: s.str.strip().isin(FUEL_NETWORKS), "known_network")]),
        "card_type": pa.Column(None, pa.Check.isin(CARD_TYPES)),
        "restriction_code": pa.Column(None, pa.Check.isin(RESTRICTIONS)),
        "status_code": pa.Column(None, pa.Check.isin(list(STATUS))),
        "status_description": pa.Column(None, text),
        "created_date": pa.Column(None, [DD_MM_YYYY, DATE_EXISTS]),
        "valid_from": pa.Column(None, [DD_MM_YYYY, DATE_EXISTS]),
        "expiry_date": pa.Column(None, [DD_MM_YYYY, DATE_EXISTS]),
        "last_updated": pa.Column(
            None, _check(lambda s: s.str.fullmatch(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}"), "timestamp")
        ),
    }
    if has_embossed:
        columns["embossed_reg"] = pa.Column(None, text)
    return pa.DataFrameSchema(columns)


def logical_schema(has_embossed: bool = True) -> pa.DataFrameSchema:
    checks = [
        _check(lambda d: ~(d["expiry_date"] < d["created_date"]), "expiry_not_before_created"),
        _check(lambda d: d["valid_from"] != PLACEHOLDER, "valid_from_not_placeholder"),
        # The later of the two dates is the real expiry, so a swapped pair does not also raise this flag.
        _check(
            lambda d: ~((d["status_code"] == "A") & (d[["created_date", "expiry_date"]].max(axis=1) < EXTRACT)),
            "active_card_not_expired",
        ),
        _check(
            lambda d: d["status_description"].isna() | (d["status_code"].map(STATUS) == d["status_description"]),
            "status_code_matches_description",
        ),
        _check(lambda d: d["company"] == d["company_raw"], "company_is_canonical_spelling"),
    ]
    if has_embossed:
        checks.append(_check(lambda d: d["vehicle_reg"] == d["embossed_reg"], "registrations_agree"))
    return pa.DataFrameSchema(checks=checks)


# Which defect each rule is written to catch, and which column a row level rule reports against.
RULE_DEFECT = {
    "not_nullable": "blank",
    "card_number_has_18_digits": "precision_loss",
    "card_number_passes_luhn": "precision_loss",
    "card_number_not_excel_rounded": "precision_loss",
    "no_hidden_whitespace": "hidden_whitespace",
    "date_is_dd_mm_yyyy": "mixed_date_format",
    "date_exists": "date_typo",  # a typo that lands on a day the month does not have, such as 29/02/2082
    "expiry_not_before_created": "swapped_dates",
    "valid_from_not_placeholder": "placeholder_date",
    "active_card_not_expired": "active_on_expired",
    "status_code_matches_description": "status_mismatch",
    "company_is_canonical_spelling": "name_variant",
    "registrations_agree": "reg_mismatch",
}
RULE_COLUMN = {
    "expiry_not_before_created": "expiry_date",
    "valid_from_not_placeholder": "valid_from",
    "active_card_not_expired": "status_code",
    "status_code_matches_description": "status_description",
    "company_is_canonical_spelling": "company",
    "registrations_agree": "embossed_reg",
}


def _failures(schema: pa.DataFrameSchema, df: pd.DataFrame) -> pd.DataFrame:
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as err:
        return err.failure_cases
    return pd.DataFrame(columns=["column", "check", "index"])


def _flags(failures: pd.DataFrame) -> pd.DataFrame:
    f = failures.dropna(subset=["index"]).copy()
    f["rule"] = f["check"].astype(str).str.replace(r"^isin\(.*", "allowed_values", regex=True)
    f["column"] = [RULE_COLUMN.get(r, c) for r, c in zip(f["rule"], f["column"], strict=True)]
    f["defect"] = f["rule"].map(RULE_DEFECT).fillna("invalid_value")
    f["row_id"] = f["index"].astype(int)
    return f[FLAG_COLUMNS].drop_duplicates(["row_id", "column", "defect"])


def detect(raw: pd.DataFrame, std: pd.DataFrame) -> pd.DataFrame:
    """Every rule failure as a flag (row_id, column, defect, rule)."""
    has_embossed = "embossed_reg" in raw
    flags = pd.concat(
        [_flags(_failures(raw_schema(has_embossed), raw)), _flags(_failures(logical_schema(has_embossed), std))]
    )
    return flags.drop_duplicates(["row_id", "column", "defect"]).sort_values(["row_id", "column"], ignore_index=True)
