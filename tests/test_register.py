"""Checks on the generator, the committed data and the whole pipeline."""

import pandas as pd
import pytest

from fleet_dq import evaluate as ev
from fleet_dq.data import load
from fleet_dq.generate import generate
from fleet_dq.pipeline import run


@pytest.fixture(scope="module")
def committed():
    return load()


@pytest.fixture(scope="module")
def result(committed):
    return run(committed.raw)


def test_generator_is_deterministic():
    a, b = generate(n_cards=200, seed=3), generate(n_cards=200, seed=3)
    pd.testing.assert_frame_equal(a.raw, b.raw)
    pd.testing.assert_frame_equal(a.defects, b.defects)


def test_committed_data_matches_the_generator(committed):
    fresh = generate()
    pd.testing.assert_frame_equal(committed.raw, fresh.raw, check_dtype=False)
    pd.testing.assert_frame_equal(
        committed.defects.fillna(""),
        fresh.defects.fillna("").astype({"row_id": int, "card_id": int}),
        check_dtype=False,
    )


def test_every_logged_change_is_a_real_change(committed):
    changed = committed.defects[~committed.defects["defect"].isin(["duplicate"])]
    assert (changed["clean_value"].fillna("") != changed["dirty_value"].fillna("")).all()


def test_one_latest_row_per_card(committed):
    latest = committed.row_cards[committed.row_cards["is_latest"]]
    assert latest["card_id"].is_unique and len(latest) == len(committed.truth)


def test_names_are_fictional_operators_only(committed):
    from fleet_dq.names import OPERATORS

    assert set(committed.truth["company"]) <= set(OPERATORS)


def test_the_tuned_threshold_is_the_one_in_use():
    from fleet_dq.clean import DEFAULT_THRESHOLD

    table = ev.tune_threshold(generate(seed=101))
    assert table.index[table["chosen"]].tolist() == [DEFAULT_THRESHOLD]


def test_pipeline_headline_numbers(committed, result):
    detected = ev.detection(result.flags, committed.defects).loc["All defects"]
    assert detected["recall"] > 0.95 and detected["precision"] > 0.99
    assert ev.resolution(committed, result.keys)["f1"] > 0.99
    assert result.scorecards(len(committed.raw))["After"].min() > 0.98
