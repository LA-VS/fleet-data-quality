from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
TRUTH_DIR = RAW_DIR / "ground_truth"
REGISTER_CSV = RAW_DIR / "register.csv"
