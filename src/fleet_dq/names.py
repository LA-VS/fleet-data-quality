"""Fictional fleet operators, the spelling variants a register collects, and clustering them back.

Every operator name is invented. Any resemblance to a real company is coincidental.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

OPERATORS = (
    "Ambervale Logistics Ltd",
    "Brackenmoor Haulage Ltd",
    "Copperleigh Fleet Services Ltd",
    "Dunmarsh Transport Ltd",
    "Elderwick Couriers Ltd",
    "Fernhollow Distribution Ltd",
    "Gildersby Freight Ltd",
    "Hartwyn Haulage Ltd",
    "Hartwyn Logistics Ltd",
    "Ivelford Vehicle Services Ltd",
    "Juniper Cross Transport Ltd",
    "Kestrelmoor Deliveries Ltd",
    "Larchmere Fleet Services Ltd",
    "Marlowgate Logistics Ltd",
    "Nethercombe Haulage Ltd",
    "Osterfield Couriers Ltd",
    "Penrowan Freight Services Ltd",
    "Quillbridge Distribution Ltd",
    "Ravensholt Transport Ltd",
    "Silvermead Logistics Ltd",
    "Silvermead Logistics North Ltd",
    "Thornwick Vehicle Services Ltd",
    "Ullerton Haulage Ltd",
    "Vantmoor Deliveries Ltd",
    "Wexcombe Fleet Services Ltd",
    "Yarrowdale Couriers Ltd",
)

# Abbreviations a legacy system with short fields produces. The generator uses them to make variants;
# the cleaner expands them back, as an analyst would after profiling the distinct values.
ABBREVIATIONS = {
    "Services": ("SERVS", "SVCS"),
    "Logistics": ("LOGS",),
    "Distribution": ("DIST",),
    "Transport": ("TRANS", "TPT"),
    "Deliveries": ("DELIVS",),
    "Couriers": ("COURIER",),
    "Haulage": ("HLG",),
    "Freight": ("FRT",),
    "Vehicle": ("VEH",),
    "Fleet": ("FLT",),
}
EXPANSIONS = {abbr: word.upper() for word, abbrs in ABBREVIATIONS.items() for abbr in abbrs}
LEGAL_SUFFIXES = {"LTD", "LIMITED", "PLC"}
FIELD_LIMIT = 20


# Making variants


def _suffix(s: str, rng: np.random.Generator) -> str:
    return s.removesuffix(" Ltd") + str(rng.choice([" LTD", " Limited", " LTD.", " Ltd.", ""]))


def _abbreviate(s: str, rng: np.random.Generator) -> str:
    words = [w for w in ABBREVIATIONS if w in s.split()]
    if not words:
        return s
    word = str(rng.choice(words))
    return s.replace(word, str(rng.choice(ABBREVIATIONS[word])), 1)


def _typo(s: str, rng: np.random.Generator) -> str:
    letters = [i for i, c in enumerate(s) if c.isalpha()][1:]
    i = int(rng.choice(letters))
    kind = int(rng.integers(4))
    if kind == 0 and i + 1 < len(s) and s[i + 1].isalpha():
        return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    if kind == 1:
        return s[:i] + s[i + 1 :]
    if kind == 2:
        return s[:i] + s[i] + s[i:]
    new = str(rng.choice(list("aeiourstnl")))
    return s[:i] + (new.upper() if s[i].isupper() else new) + s[i + 1 :]


def _upper(s: str, rng: np.random.Generator) -> str:
    return s.upper()


def _truncate(s: str, rng: np.random.Generator) -> str:
    return s[:FIELD_LIMIT].rstrip()


# Applied in this order: truncation comes last, as it does when a value is cut to fit a field.
TRANSFORMS = (_suffix, _abbreviate, _typo, _upper, _truncate)


def variant_pool(name: str, rng: np.random.Generator, size: int) -> list[str]:
    """Distinct misspellings of `name`, each made by one or two transformations."""
    pool: list[str] = []
    for _ in range(50 * size):
        if len(pool) == size:
            break
        chosen = sorted(rng.choice(len(TRANSFORMS), size=int(rng.integers(1, 3)), replace=False))
        s = name
        for j in chosen:
            s = TRANSFORMS[j](s, rng)
        if s != name and s not in pool:
            pool.append(s)
    return pool


# Cleaning and clustering


def clean_text(value):
    """Replace non breaking and other odd spaces, collapse runs of spaces, trim. Blanks stay blank."""
    if not isinstance(value, str):
        return value
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def normalise(name: str) -> str:
    """A matching key: upper case, no punctuation, abbreviations expanded, legal suffix dropped."""
    tokens = re.sub(r"[.,]", " ", clean_text(name).upper()).split()
    tokens = [EXPANSIONS.get(t, t) for t in tokens]
    # A legal suffix, or what is left of one after truncation ("LIM", "L").
    while len(tokens) > 2 and (tokens[-1] in LEGAL_SUFFIXES or "LIMITED".startswith(tokens[-1])):
        tokens.pop()
    return " ".join(tokens)


def cluster_names(values: pd.Series, threshold: float) -> pd.Series:
    """Map every name to a canonical name for its cluster.

    Distinct matching keys are taken in order of how many rows use them. Each joins the most similar
    existing cluster if the similarity is at least `threshold` (0 to 100), and otherwise starts a new
    cluster. A cluster's canonical name is its most common spelling.
    """
    cleaned = values.map(clean_text)
    counts = cleaned.value_counts()
    counts = counts.iloc[np.lexsort((counts.index.to_numpy(), -counts.to_numpy()))]
    key_of = {v: normalise(v) for v in counts.index}
    key_counts = counts.groupby(key_of).sum()
    key_counts = key_counts.iloc[np.lexsort((key_counts.index.to_numpy(), -key_counts.to_numpy()))]

    reps: list[str] = []
    cluster_of: dict[str, int] = {}
    for key in key_counts.index:
        match = process.extractOne(key, reps, scorer=fuzz.ratio, score_cutoff=threshold) if reps else None
        if match is None:
            cluster_of[key] = len(reps)
            reps.append(key)
        else:
            cluster_of[key] = match[2]

    clusters = pd.Series({v: cluster_of[key_of[v]] for v in counts.index})
    canonical = {c: group.index[0] for c, group in counts.groupby(clusters, sort=False)}
    return cleaned.map(lambda v: canonical[clusters[v]] if isinstance(v, str) else v)
