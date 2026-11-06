"""Generate a clean fuel card register, then a dirty extract of it with every defect logged.

The clean register is the ground truth: one row per card. The extract repeats each card several times,
as registers merged from several system exports do, and then damages it in the ways listed in
`DefectRates`. Every change is written to a defect log, so a cleaning pipeline can be scored.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from fleet_dq import cards, names

EXTRACT_DATE = date(2026, 6, 30)
PLACEHOLDER_DATE = date(1990, 1, 1)
STATUS = {"A": "Active", "S": "Stopped", "C": "Cancelled", "E": "Expired"}
FUEL_NETWORKS = ("Fuelnet", "Roadline", "Pumpway", "Tankstop")
CARD_TYPES = ("Fuel only", "Fuel and oil", "Fuel, oil and lubricants", "All purchases")
RESTRICTIONS = ("R01", "R02", "R03", "R04", "R05")
REG_LETTERS = list("ABCDEFGHJKLMNOPRSTUVWXY")

COLUMNS = (
    "card_number",
    "company",
    "vehicle_reg",
    "embossed_reg",
    "fuel_network",
    "card_type",
    "restriction_code",
    "status_code",
    "status_description",
    "created_date",
    "valid_from",
    "expiry_date",
    "last_updated",
)
TEXT_COLUMNS = ("company", "vehicle_reg", "embossed_reg", "fuel_network", "status_description")
DATE_COLUMNS = ("created_date", "valid_from", "expiry_date")
DATE_FORMAT = "%d/%m/%Y"
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M"


@dataclass(frozen=True)
class DefectRates:
    """How often each defect is injected. Rates are per row unless the comment says otherwise."""

    mean_rows_per_card: float = 4.6
    conflicting_duplicate: float = 0.3  # older copies of stopped or cancelled cards that still say Active
    name_variant: float = 0.35
    hidden_whitespace: float = 0.03  # per text cell
    blank_status_description: float = 0.04
    blank_restriction_code: float = 0.06
    placeholder_date: float = 0.08
    mixed_date_format: float = 0.10  # per date cell
    precision_loss: float = 0.15
    swapped_dates: float = 0.02
    active_on_expired: float = 0.10  # share of rows for expired cards
    status_mismatch: float = 0.03
    reg_mismatch: float = 0.04
    date_typo: float = 0.01


@dataclass
class Register:
    truth: pd.DataFrame  # one row per card, typed
    raw: pd.DataFrame  # the dirty extract, all text, indexed by row_id
    row_cards: pd.DataFrame  # row_id, card_id, is_latest
    defects: pd.DataFrame  # row_id, card_id, column, defect, clean_value, dirty_value


def month_end(d: date) -> date:
    return d.replace(day=calendar.monthrange(d.year, d.month)[1])


def add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    year, month = d.year + m // 12, m % 12 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def random_reg(rng: np.random.Generator) -> str:
    """A UK style registration such as AB12 CDE. Invented, not checked against real vehicles."""
    letters = rng.choice(REG_LETTERS, size=5)
    age = int(rng.choice([*range(16, 27), *range(66, 76)]))
    return f"{letters[0]}{letters[1]}{age:02d} {letters[2]}{letters[3]}{letters[4]}"


def clean_register(n_cards: int = 3500, seed: int = 7) -> pd.DataFrame:
    """The ground truth: one correct row per card."""
    rng = np.random.default_rng(seed)
    n_ops = len(names.OPERATORS)
    share = rng.dirichlet(np.full(n_ops, 2.0))
    op_of_card = np.sort(rng.choice(n_ops, size=n_cards, p=share))
    accounts = rng.choice(np.arange(100_000, 1_000_000), size=n_ops, replace=False)
    home_network = rng.integers(len(FUEL_NETWORKS), size=n_ops)
    next_seq = rng.integers(1, 40, size=n_ops)

    regs: set[str] = set()
    start = date(2016, 1, 1)
    span = (EXTRACT_DATE - timedelta(days=14) - start).days
    rows = []
    for card_id, op in enumerate(op_of_card):
        seq = int(next_seq[op])
        next_seq[op] += 1 + int(rng.random() < 0.1)  # occasional gaps where cards were never issued
        reg = random_reg(rng)
        while reg in regs:
            reg = random_reg(rng)
        regs.add(reg)

        created = start + timedelta(days=int(rng.integers(span)))
        valid_from = created + timedelta(days=int(rng.integers(0, 15)))
        expiry = month_end(add_months(valid_from, int(rng.choice([24, 36]))))
        if expiry < EXTRACT_DATE:
            status = "E" if rng.random() < 0.9 else "C"
        else:
            status = str(rng.choice(["A", "S", "C"], p=[0.85, 0.08, 0.07]))

        created_at = datetime.combine(created, datetime.min.time()) + timedelta(hours=9)
        extract_at = datetime.combine(EXTRACT_DATE, datetime.min.time()) + timedelta(hours=6)
        earliest = datetime.combine(expiry, datetime.min.time()) if status == "E" else created_at + timedelta(days=1)
        earliest = min(earliest, extract_at - timedelta(days=1))
        last_updated = earliest + timedelta(minutes=int(rng.integers((extract_at - earliest).total_seconds() // 60)))

        network = int(home_network[op]) if rng.random() < 0.8 else int(rng.integers(len(FUEL_NETWORKS)))
        rows.append(
            {
                "card_id": card_id,
                "card_number": cards.card_number(int(accounts[op]), seq),
                "company": names.OPERATORS[op],
                "vehicle_reg": reg,
                "embossed_reg": reg,
                "fuel_network": FUEL_NETWORKS[network],
                "card_type": str(rng.choice(CARD_TYPES, p=[0.45, 0.3, 0.15, 0.1])),
                "restriction_code": str(rng.choice(RESTRICTIONS)),
                "status_code": status,
                "status_description": STATUS[status],
                "created_date": created,
                "valid_from": valid_from,
                "expiry_date": expiry,
                "last_updated": last_updated.replace(second=0),
            }
        )
    return pd.DataFrame(rows)


def as_text(record: dict) -> dict:
    """A register row as the extract stores it: every field text, dates as dd/mm/yyyy."""
    out = {}
    for col in COLUMNS:
        v = record[col]
        if col == "last_updated":
            out[col] = v.strftime(TIMESTAMP_FORMAT)
        elif col in DATE_COLUMNS:
            out[col] = v.strftime(DATE_FORMAT)
        else:
            out[col] = v
    return out


def _typo_year(text: str) -> str | None:
    """Swap the last two digits of the year (2027 becomes 2072) when that moves it a decade or more."""
    year = text[-4:]
    swapped = year[:2] + year[3] + year[2]
    return text[:-4] + swapped if int(swapped) - int(year) >= 10 else None


def _with_typo(reg: str, rng: np.random.Generator) -> str:
    positions = [i for i, c in enumerate(reg) if c.isalpha()]
    i = int(rng.choice(positions))
    choices = [c for c in REG_LETTERS if c != reg[i]]
    return reg[:i] + str(rng.choice(choices)) + reg[i + 1 :]


def dirty_extract(
    truth: pd.DataFrame, rates: DefectRates, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pools = {op: names.variant_pool(op, rng, size=int(rng.integers(5, 13))) for op in names.OPERATORS}

    # Copies of each card. The latest copy is the truth; older copies carry older timestamps.
    rows, card_of_row, latest = [], [], []
    for card in truth.to_dict("records"):
        n = 1 + int(rng.poisson(rates.mean_rows_per_card - 1))
        base = as_text(card)
        offsets = np.sort(rng.choice(np.arange(1, 600), size=n - 1, replace=False))
        created_at = datetime.combine(card["created_date"], datetime.min.time()) + timedelta(hours=9)
        expired = card["expiry_date"] < EXTRACT_DATE
        for j in range(n):
            row = dict(base)
            if j > 0:
                stamp = card["last_updated"] - timedelta(days=int(offsets[j - 1]), minutes=int(rng.integers(1440)))
                if stamp <= created_at:
                    stamp = created_at + timedelta(minutes=j)
                row["last_updated"] = stamp.strftime(TIMESTAMP_FORMAT)
                if card["status_code"] in "SC" and not expired and rng.random() < rates.conflicting_duplicate:
                    row["status_code"], row["status_description"] = "A", STATUS["A"]
            rows.append(row)
            card_of_row.append(card["card_id"])
            latest.append(j == 0)

    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    card_of_row = np.array(card_of_row)[order]
    latest = np.array(latest)[order]
    truth_text = {c["card_id"]: as_text(c) for c in truth.to_dict("records")}
    truth_by_id = truth.set_index("card_id")
    n = len(rows)
    log: list[tuple] = []

    def record(i: int, column: str, defect: str, dirty) -> None:
        clean = truth_text[card_of_row[i]].get(column, "")
        log.append((i, int(card_of_row[i]), column, defect, clean, dirty))

    def pick(eligible: np.ndarray, rate: float) -> np.ndarray:
        return np.flatnonzero(eligible & (rng.random(n) < rate))

    for i in np.flatnonzero(~latest):
        record(i, "*", "duplicate", "")
        if rows[i]["status_code"] != truth_text[card_of_row[i]]["status_code"]:
            record(i, "status_code", "conflicting_duplicate", rows[i]["status_code"])

    # Logical defects go on separate rows, so each one has a single cause.
    used = np.zeros(n, dtype=bool)
    typo_year = [_typo_year(r["expiry_date"]) for r in rows]
    typo_rows = pick(np.array([t is not None for t in typo_year]), rates.date_typo)
    for i in typo_rows:
        rows[i]["expiry_date"] = typo_year[i]
        record(i, "expiry_date", "date_typo", typo_year[i])
        used[i] = True
    swapped = pick(~used, rates.swapped_dates)
    for i in swapped:
        r = rows[i]
        r["created_date"], r["expiry_date"] = r["expiry_date"], r["created_date"]
        record(i, "expiry_date", "swapped_dates", r["expiry_date"])
        used[i] = True
    expired_card = np.array([truth_by_id.at[c, "status_code"] == "E" for c in card_of_row])
    for i in pick(expired_card & ~used, rates.active_on_expired):
        rows[i]["status_code"], rows[i]["status_description"] = "A", STATUS["A"]
        record(i, "status_code", "active_on_expired", "A")
        used[i] = True
    mismatched = pick(~used, rates.status_mismatch)
    for i in mismatched:
        true_description = truth_text[card_of_row[i]]["status_description"]
        other = [d for code, d in STATUS.items() if code != rows[i]["status_code"] and d != true_description]
        rows[i]["status_description"] = str(rng.choice(other))
        record(i, "status_description", "status_mismatch", rows[i]["status_description"])
        used[i] = True

    for i in pick(np.ones(n, dtype=bool), rates.placeholder_date):
        rows[i]["valid_from"] = PLACEHOLDER_DATE.strftime(DATE_FORMAT)
        record(i, "valid_from", "placeholder_date", rows[i]["valid_from"])
    not_mismatched = np.ones(n, dtype=bool)
    not_mismatched[mismatched] = False
    for i in pick(not_mismatched, rates.blank_status_description):
        rows[i]["status_description"] = None
        record(i, "status_description", "blank", None)
    for i in pick(np.ones(n, dtype=bool), rates.blank_restriction_code):
        rows[i]["restriction_code"] = None
        record(i, "restriction_code", "blank", None)

    for i in pick(np.ones(n, dtype=bool), rates.reg_mismatch):
        reg = rows[i]["vehicle_reg"]
        rows[i]["embossed_reg"] = _with_typo(reg, rng) if rng.random() < 0.5 else random_reg(rng)
        record(i, "embossed_reg", "reg_mismatch", rows[i]["embossed_reg"])

    for i in pick(np.ones(n, dtype=bool), rates.name_variant):
        pool = pools[rows[i]["company"]]
        weights = 1 / np.arange(1, len(pool) + 1)
        rows[i]["company"] = str(rng.choice(pool, p=weights / weights.sum()))
        record(i, "company", "name_variant", rows[i]["company"])

    for i in pick(np.array([not r["card_number"].endswith("000") for r in rows]), rates.precision_loss):
        rows[i]["card_number"] = cards.excel_rounded(rows[i]["card_number"])
        record(i, "card_number", "precision_loss", rows[i]["card_number"])

    # MM/YY drops the day, so it is not applied where the expiry is a typo or holds a swapped creation date.
    no_mmyy = np.zeros(n, dtype=bool)
    no_mmyy[typo_rows] = True
    no_mmyy[swapped] = True
    for col in DATE_COLUMNS:
        eligible = ~no_mmyy if col == "expiry_date" else np.ones(n, dtype=bool)
        for i in pick(eligible, rates.mixed_date_format):
            d = datetime.strptime(rows[i][col], DATE_FORMAT)
            rows[i][col] = d.strftime("%m/%y") if col == "expiry_date" else d.strftime("%Y-%m-%dT%H:%M:%S")
            record(i, col, "mixed_date_format", rows[i][col])

    for col in TEXT_COLUMNS:
        for i in pick(np.array([isinstance(r[col], str) for r in rows]), rates.hidden_whitespace):
            v = rows[i][col]
            kind = int(rng.integers(3))
            if kind == 0 and " " in v:
                k = v.index(" ")
                v = v[:k] + " " + v[k + 1 :]
            elif kind == 1:
                v = v + "  "
            else:
                v = " " + v
            rows[i][col] = v
            record(i, col, "hidden_whitespace", v)

    raw = pd.DataFrame(rows, columns=list(COLUMNS))
    raw.index.name = "row_id"
    row_cards = pd.DataFrame({"row_id": np.arange(n), "card_id": card_of_row, "is_latest": latest})
    defects = pd.DataFrame(log, columns=["row_id", "card_id", "column", "defect", "clean_value", "dirty_value"])
    return raw, row_cards, defects.sort_values(["row_id", "column", "defect"], ignore_index=True)


def generate(n_cards: int = 3500, rates: DefectRates | None = None, seed: int = 7) -> Register:
    truth = clean_register(n_cards, seed)
    raw, row_cards, defects = dirty_extract(truth, rates or DefectRates(), seed + 1)
    return Register(truth, raw, row_cards, defects)
