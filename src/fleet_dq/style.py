"""One chart style for every figure in the repo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from fleet_dq.paths import REPO_ROOT

FIGURES_DIR = REPO_ROOT / "reports" / "figures"
BEFORE = "#9aa3ad"
AFTER = "#3b6ea8"
ACCENT = "#3b6ea8"
MUTED = "#9aa3ad"
HIGHLIGHT = "#c0392b"


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path
