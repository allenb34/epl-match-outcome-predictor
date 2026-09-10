"""Diagnostics for data/processed/matches_dataset.csv - run after
build_features.py, before trusting the dataset for modeling.

Checks (each prints a clear pass/fail line, exits 1 on any failure):
    1. Row count matches data/raw/historical_matches.csv exactly (build_features.py
       must never drop rows, only leave cells as NaN).
    2. Result label distribution matches the raw distribution (features
       shouldn't have altered labels).
    3. No row has a fabricated result (H/D/A only, no other values).
    4. Season coverage: all three expected seasons present with 380 rows each.
    5. Missing-value report per feature column, so cold-start sparsity is
       visible rather than silently averaged away. Cold start isn't
       confined to 2023-24 (the first season in our data): a newly
       promoted team's first match in ANY season is also a legitimate
       cold start, since we have no Championship history for them. The
       real invariant checked here isn't "no NaN after season 1" but
       "every NaN row's matches_played count is actually 0" - i.e. NaN
       only ever means genuinely no history, never a computation bug.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED_SEASONS = {"2023-24": 380, "2024-25": 380, "2025-26": 380}


def main():
    raw_path = RAW_DIR / "historical_matches.csv"
    processed_path = PROCESSED_DIR / "matches_dataset.csv"
    if not raw_path.exists() or not processed_path.exists():
        print("FATAL: run fetch_historical_results.py then build_features.py first.", file=sys.stderr)
        sys.exit(1)

    raw = pd.read_csv(raw_path)
    dataset = pd.read_csv(processed_path)

    failures = []

    if len(dataset) != len(raw):
        failures.append(f"row count mismatch: raw={len(raw)}, processed={len(dataset)} (rows were dropped or duplicated)")
    else:
        print(f"OK  row count preserved: {len(dataset)} rows")

    raw_dist = raw["result"].value_counts().sort_index()
    processed_dist = dataset["result"].value_counts().sort_index()
    if not raw_dist.equals(processed_dist):
        failures.append(f"result distribution changed: raw={raw_dist.to_dict()} processed={processed_dist.to_dict()}")
    else:
        print(f"OK  result distribution unchanged: {raw_dist.to_dict()}")

    bad_results = set(dataset["result"].unique()) - {"H", "D", "A"}
    if bad_results:
        failures.append(f"unexpected result values: {bad_results}")
    else:
        print("OK  result column only contains H/D/A")

    season_counts = dataset["season"].value_counts().to_dict()
    for season, expected in EXPECTED_SEASONS.items():
        actual = season_counts.get(season, 0)
        if actual != expected:
            failures.append(f"season {season}: expected {expected} matches, found {actual}")
    if not any(f.startswith("season ") for f in failures):
        print(f"OK  all 3 expected seasons present with 380 matches each: {season_counts}")

    print("\nMissing values per feature column (expected: cold-start rows only - see next check):")
    feature_cols = [c for c in dataset.columns if c not in raw.columns]
    na_counts = dataset[feature_cols].isna().sum()
    na_counts = na_counts[na_counts > 0]
    if na_counts.empty:
        print("  none")
    else:
        print(na_counts.to_string())
    bad_home_na = dataset[dataset["home_form_pts_last5"].isna() & (dataset["home_matches_played_last5"] != 0)]
    bad_away_na = dataset[dataset["away_form_pts_last5"].isna() & (dataset["away_matches_played_last5"] != 0)]
    if not bad_home_na.empty or not bad_away_na.empty:
        failures.append(
            f"found {len(bad_home_na) + len(bad_away_na)} rows where a rolling-form NaN doesn't correspond to "
            "an actual 0 matches_played count - this would mean a real computation bug, not genuine cold start"
        )
    else:
        n_cold_start = dataset["home_matches_played_last5"].eq(0).sum() + dataset["away_matches_played_last5"].eq(0).sum()
        print(f"OK  every rolling-form NaN corresponds to a genuine 0-matches-played cold start ({n_cold_start} team-sides)")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
