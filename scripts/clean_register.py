"""Run the pipeline on data/raw/register.csv and write the cleaned register and every flag it raised."""

from fleet_dq.data import read_raw
from fleet_dq.paths import REGISTER_CSV, REPO_ROOT
from fleet_dq.pipeline import as_extract, run

if __name__ == "__main__":
    raw = read_raw(REGISTER_CSV)
    result = run(raw)
    out = REPO_ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    cleaned = as_extract(result.golden)
    cleaned["number_status"] = result.golden["number_status"].to_numpy()
    cleaned.to_csv(out / "register_clean.csv", lineterminator="\n")
    result.flags.to_csv(out / "flags.csv", index=False, lineterminator="\n")
    print(f"{len(raw):,} rows in, {len(cleaned):,} cards out, {len(result.flags):,} flags")
    print(result.scorecards(len(raw)).round(3).to_string())
