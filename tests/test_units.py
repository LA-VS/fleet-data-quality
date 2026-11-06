"""Unit tests for card numbers, names, date parsing, fixes, rules and survivorship."""

import numpy as np
import pandas as pd
import pytest

from fleet_dq import cards, names
from fleet_dq.clean import fix, parse_dates, standardise
from fleet_dq.evaluate import detection, merged_operators, pairwise
from fleet_dq.generate import COLUMNS, STATUS
from fleet_dq.resolve import resolve
from fleet_dq.rules import detect


def test_luhn_known_number():
    assert cards.luhn_valid("79927398713")
    assert not cards.luhn_valid("79927398710")


def test_card_number_is_18_digits_and_valid():
    number = cards.card_number(123456, 42)
    assert len(number) == 18 and number.startswith(cards.ISSUER) and cards.luhn_valid(number)


def test_excel_keeps_15_significant_digits():
    assert cards.excel_rounded("707781123456000429") == "707781123456000000"


def test_clean_text_removes_hidden_whitespace():
    assert names.clean_text(" Ambervale  Logistics Ltd ") == "Ambervale Logistics Ltd"
    assert names.clean_text("   ") is None


@pytest.mark.parametrize(
    "variant",
    ["AMBERVALE LOGISTICS LIMITED", "Ambervale LOGS LTD.", "Ambervale Logistics", "Ambervale Logistics L"],
)
def test_normalise_reaches_the_same_key(variant):
    assert names.normalise(variant) == names.normalise("Ambervale Logistics Ltd") == "AMBERVALE LOGISTICS"


def test_variants_differ_and_carry_no_hidden_whitespace():
    rng = np.random.default_rng(0)
    for op in names.OPERATORS:
        for v in names.variant_pool(op, rng, 10):
            assert v != op and v == names.clean_text(v)


def test_clustering_joins_typos_but_keeps_similar_companies_apart():
    values = pd.Series(
        ["Silvermead Logistics Ltd"] * 5
        + ["Silvermead Logistics North Ltd"] * 3
        + ["Silvermead Logisitcs Ltd", "Hartwyn Haulage Ltd", "Hartwyn Logistics Ltd"]
    )
    out = names.cluster_names(values, 88)
    assert out.iloc[8] == "Silvermead Logistics Ltd"
    assert out.iloc[5] == "Silvermead Logistics North Ltd"
    assert out.iloc[9] != out.iloc[10]


def test_parse_dates_reads_all_three_formats():
    out = parse_dates(pd.Series(["31/07/2026", "2021-03-04T00:00:00", "02/27", "29/02/2082", None]))
    assert list(out[:3]) == [pd.Timestamp("2026-07-31"), pd.Timestamp("2021-03-04"), pd.Timestamp("2027-02-28")]
    assert out[3:].isna().all()


def raw_row(**changes) -> dict:
    row = {
        "card_number": cards.card_number(111111, 1),
        "company": "Ambervale Logistics Ltd",
        "vehicle_reg": "AB12 CDE",
        "embossed_reg": "AB12 CDE",
        "fuel_network": "Fuelnet",
        "card_type": "Fuel only",
        "restriction_code": "R01",
        "status_code": "A",
        "status_description": "Active",
        "created_date": "01/03/2025",
        "valid_from": "05/03/2025",
        "expiry_date": "31/03/2028",
        "last_updated": "01/04/2026 10:00",
    }
    return {**row, **changes}


def frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    df.index.name = "row_id"
    return df


def test_rules_flag_each_defect_in_the_right_place():
    raw = frame(
        [
            raw_row(),
            raw_row(card_number=cards.excel_rounded(cards.card_number(111111, 1))),
            raw_row(company="Ambervale Logistics Ltd "),
            raw_row(restriction_code=None),
            raw_row(expiry_date="03/28"),
            raw_row(valid_from="01/01/1990"),
            raw_row(created_date="31/03/2028", expiry_date="01/03/2025"),
            raw_row(created_date="01/03/2021", valid_from="05/03/2021", expiry_date="31/03/2024"),
            raw_row(status_description="Stopped"),
            raw_row(embossed_reg="AB12 CDF"),
        ]
    )
    flags = detect(raw, standardise(raw))
    found = set(zip(flags["row_id"], flags["column"], flags["defect"], strict=True))
    assert found == {
        (1, "card_number", "precision_loss"),
        (2, "company", "hidden_whitespace"),
        (3, "restriction_code", "blank"),
        (4, "expiry_date", "mixed_date_format"),
        (5, "valid_from", "placeholder_date"),
        (6, "expiry_date", "swapped_dates"),
        (7, "status_code", "active_on_expired"),
        (8, "status_description", "status_mismatch"),
        (9, "embossed_reg", "reg_mismatch"),
    }


def test_fix_repairs_logic_and_blanks_placeholders():
    raw = frame(
        [
            raw_row(created_date="31/03/2028", expiry_date="01/03/2025"),
            raw_row(valid_from="01/01/1990"),
            raw_row(created_date="01/03/2021", valid_from="05/03/2021", expiry_date="31/03/2024"),
            raw_row(status_description="Stopped"),
        ]
    )
    fixed = fix(standardise(raw))
    assert fixed.at[0, "created_date"] == pd.Timestamp("2025-03-01")
    assert pd.isna(fixed.at[1, "valid_from"])
    assert fixed.at[2, "status_code"] == "E"
    assert fixed.at[3, "status_description"] == STATUS["A"]
    assert "embossed_reg" not in fixed


def test_survivorship_and_number_recovery():
    full = cards.card_number(111111, 1)
    neighbour = cards.card_number(111111, 2)  # same account, same first 15 digits
    raw = frame(
        [
            raw_row(status_code="A", status_description="Active", last_updated="01/01/2026 09:00"),
            raw_row(status_code="S", status_description="Stopped", last_updated="01/05/2026 09:00"),
            raw_row(card_number=cards.excel_rounded(full), fuel_network="Pumpway", last_updated="01/02/2026 09:00"),
            raw_row(card_number=neighbour, vehicle_reg="XY70 ZZZ", embossed_reg="XY70 ZZZ"),
            raw_row(card_number=cards.excel_rounded(full), vehicle_reg="NO11 NEW", embossed_reg="NO11 NEW"),
        ]
    )
    golden, keys, flags = resolve(fix(standardise(raw)))
    assert keys.loc[2, ["key", "number_status"]].tolist() == [full, "recovered"]
    assert keys.loc[4, "number_status"] == "unrecoverable"
    card = golden.loc[full]
    assert card["source_rows"] == 3
    assert card["status_code"] == "S"  # the latest row decides status
    assert card["fuel_network"] == "Fuelnet"  # two votes against one
    assert set(flags.loc[flags["defect"] == "duplicate", "row_id"]) == {0, 2}
    assert set(flags.loc[flags["defect"] == "conflicting_duplicate", "row_id"]) == {0, 2}


def test_pairwise_metrics():
    truth = pd.Series(["a", "a", "b", "b"])
    assert pairwise(truth, pd.Series([1, 1, 2, 2]))["f1"] == 1
    merged = pairwise(truth, pd.Series([1, 1, 1, 1]))
    assert merged["recall"] == 1 and merged["precision"] == pytest.approx(2 / 6)
    assert merged_operators(truth, pd.Series([1, 1, 1, 1])) == ["a", "b"]


def test_detection_needs_row_column_and_type_to_match():
    defects = pd.DataFrame({"row_id": [1, 2], "column": ["company", "company"], "defect": ["blank", "blank"]})
    flags = pd.DataFrame({"row_id": [1, 2], "column": ["company", "vehicle_reg"], "defect": ["blank", "blank"]})
    out = detection(flags, defects).loc["blank"]
    assert (out["caught"], out["precision"], out["recall"]) == (1, 0.5, 0.5)
