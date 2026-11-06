"""Generate the register and its ground truth into data/raw/. Fixed seed, so the output never changes."""

from fleet_dq.data import save
from fleet_dq.generate import generate
from fleet_dq.paths import RAW_DIR

if __name__ == "__main__":
    register = generate()
    save(register)
    print(f"{len(register.truth):,} cards, {len(register.raw):,} rows, {len(register.defects):,} logged defects")
    print(f"Written to {RAW_DIR}")
