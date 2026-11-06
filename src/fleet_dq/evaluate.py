"""Score the pipeline against the ground truth the generator logged."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fleet_dq.generate import Register, as_text
from fleet_dq.pipeline import as_extract

KEY = ["row_id", "column", "defect"]
FIELDS = [
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
]


def detection(flags: pd.DataFrame, defects: pd.DataFrame) -> pd.DataFrame:
    """Per defect type: how many were injected, how many flags were raised, and how many flags were right.

    A flag counts only if it names the right row, the right column and the right defect type.
    """
    truth = defects[KEY].drop_duplicates()
    raised = flags[KEY].drop_duplicates()
    hit = raised.merge(truth, on=KEY, how="inner")
    out = (
        pd.DataFrame(
            {
                "injected": truth["defect"].value_counts(),
                "flagged": raised["defect"].value_counts(),
                "caught": hit["defect"].value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    out.loc["All defects"] = out.sum()
    out["precision"] = out["caught"] / out["flagged"].replace(0, np.nan)
    out["recall"] = out["caught"] / out["injected"].replace(0, np.nan)
    return out


def pairwise(true_labels: pd.Series, predicted: pd.Series) -> dict:
    """Pairwise precision and recall of a clustering: of the pairs of rows put together, how many belong
    together, and of the pairs that belong together, how many were put together."""
    table = pd.crosstab(true_labels, predicted).to_numpy()

    def pairs(x):
        return (x * (x - 1) / 2).sum()

    together = pairs(table)
    precision = together / pairs(table.sum(axis=0))
    recall = together / pairs(table.sum(axis=1))
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall)}


def true_operator(register: Register) -> pd.Series:
    company = register.truth.set_index("card_id")["company"]
    return register.row_cards.set_index("row_id")["card_id"].map(company).rename("company")


def name_matching(register: Register, canonical: pd.Series) -> dict:
    truth = true_operator(register).reindex(canonical.index)
    return {**pairwise(truth, canonical), "rows_exactly_right": (canonical == truth).mean()}


def resolution(register: Register, keys: pd.DataFrame) -> dict:
    card = register.row_cards.set_index("row_id")["card_id"].reindex(keys.index)
    return {
        **pairwise(card, keys["key"]),
        "true_cards": card.nunique(),
        "resolved_cards": keys["key"].nunique(),
        "unrecoverable_numbers": (keys.groupby("key")["number_status"].first() == "unrecoverable").sum(),
    }


def _truth_text(register: Register) -> pd.DataFrame:
    return pd.DataFrame([as_text(c) for c in register.truth.to_dict("records")], index=register.truth["card_id"])


def accuracy_before(register: Register) -> pd.Series:
    """Share of the extract's cells that hold the correct value for their card."""
    truth = _truth_text(register)
    card = register.row_cards.set_index("row_id")["card_id"].reindex(register.raw.index)
    expected = truth.loc[card.to_numpy(), FIELDS].set_axis(register.raw.index)
    return (register.raw[FIELDS] == expected).mean()


def accuracy_after(register: Register, golden: pd.DataFrame, keys: pd.DataFrame) -> pd.Series:
    """Share of true cards whose cleaned record holds the correct value, field by field.

    Each cleaned record is matched to the card most of its rows came from. A card split over several
    records is judged on its largest; a card with no record of its own counts as wrong in every field.
    """
    truth = _truth_text(register)
    card = register.row_cards.set_index("row_id")["card_id"].reindex(keys.index)
    owner = pd.DataFrame({"key": keys["key"], "card": card}).value_counts().reset_index(name="rows")
    owner = owner.sort_values("rows", ascending=False).drop_duplicates("key").drop_duplicates("card")
    cleaned = as_extract(golden).set_axis(golden.index)
    matched = cleaned.loc[owner["key"], FIELDS].set_axis(owner["card"].to_numpy())
    correct = (matched == truth.loc[matched.index, FIELDS]).sum()
    return correct / len(truth)


def merged_operators(true_labels: pd.Series, predicted: pd.Series) -> list[str]:
    """Operators that share their main cluster with another operator: two real customers made one."""
    main = pd.crosstab(true_labels, predicted).idxmax(axis=1)
    return sorted(main[main.duplicated(keep=False)].index)


def tune_threshold(register: Register, thresholds=range(70, 101, 2)) -> pd.DataFrame:
    """Name matching quality at each threshold, on a register kept apart from the one being reported.

    The chosen threshold has the best pairwise F1 among those that merge no two operators. A merge
    hides a customer inside another; a spelling left unmatched is still visible in the output.
    """
    from fleet_dq.names import cluster_names

    truth = true_operator(register)
    rows = []
    for t in thresholds:
        canonical = cluster_names(register.raw["company"], t).reindex(truth.index)
        rows.append({"threshold": t, **pairwise(truth, canonical), "merged": len(merged_operators(truth, canonical))})
    table = pd.DataFrame(rows).set_index("threshold")
    table["chosen"] = False
    table.loc[table[table["merged"] == 0]["f1"].idxmax(), "chosen"] = True
    return table
