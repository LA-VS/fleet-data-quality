"""Write and read the generated register and its ground truth."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fleet_dq.generate import DATE_COLUMNS, Register
from fleet_dq.paths import RAW_DIR


def save(register: Register, raw_dir: Path = RAW_DIR) -> None:
    truth_dir = raw_dir / "ground_truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    register.raw.to_csv(raw_dir / "register.csv", encoding="utf-8", lineterminator="\n")
    register.truth.to_csv(truth_dir / "cards.csv", index=False, lineterminator="\n")
    register.row_cards.to_csv(truth_dir / "row_cards.csv", index=False, lineterminator="\n")
    register.defects.to_csv(truth_dir / "defects.csv", index=False, encoding="utf-8", lineterminator="\n")


def read_raw(path: Path) -> pd.DataFrame:
    """The extract as text, exactly as written: padding kept, empty cells blank."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], index_col="row_id").rename(index=int)


def load(raw_dir: Path = RAW_DIR) -> Register:
    truth_dir = raw_dir / "ground_truth"
    truth = pd.read_csv(truth_dir / "cards.csv", dtype={"card_number": str})
    for col in DATE_COLUMNS:
        truth[col] = pd.to_datetime(truth[col]).dt.date
    truth["last_updated"] = pd.to_datetime(truth["last_updated"]).dt.to_pydatetime()
    defects = pd.read_csv(truth_dir / "defects.csv", dtype=str, keep_default_na=False, na_values=[""])
    defects[["row_id", "card_id"]] = defects[["row_id", "card_id"]].astype(int)
    return Register(
        truth=truth,
        raw=read_raw(raw_dir / "register.csv"),
        row_cards=pd.read_csv(truth_dir / "row_cards.csv"),
        defects=defects,
    )
