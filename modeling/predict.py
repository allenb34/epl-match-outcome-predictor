"""Score upcoming fixtures with the trained Random Forest model.

Reuses build_features.py's leak-free replay functions directly (imported,
not reimplemented) so upcoming-fixture features are computed by the exact
same logic that produced the training data:
    1. Replay historical_matches.csv + current_season_played.csv together,
       in chronological order, to reach the true end-state of every team's
       history (rolling form, home/away win %, H2H) as of right now.
    2. For each upcoming fixture, call compute_match_features() read-only
       against that end-state - no mutation, since we're predicting, not
       simulating results.

This intentionally ignores intra-matchday ordering (e.g. a Saturday 12:30
kickoff's result technically happens before that same day's 5:30 kickoff,
but both draw on the same "end-state" here) - a standard simplifying
assumption for this kind of walk-forward feature engineering, worth noting
in the README rather than solving with a more complex per-kickoff replay.

Output: data/processed/predictions.csv with columns
    match_id, utc_date, matchday, home_team, away_team,
    home_win_prob, draw_prob, away_win_prob, predicted_result, confidence
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data_collection"))
from build_features import build_team_histories, compute_match_features, update_histories  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.pkl"
MODEL_PATH = ARTIFACTS_DIR / "random_forest_final.pkl"
IMPUTER_PATH = ARTIFACTS_DIR / "imputer.pkl"

LABEL_TO_RESULT = {0: "H", 1: "D", 2: "A"}
RESULT_TO_DISPLAY = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def load_known_results() -> pd.DataFrame:
    """historical_matches.csv + current_season_played.csv, chronologically
    sorted - this is every real result we know about, used to build the
    end-state team history that upcoming fixtures are featurized against.
    """
    historical = pd.read_csv(RAW_DIR / "historical_matches.csv", parse_dates=["utc_date"])
    current = pd.read_csv(RAW_DIR / "current_season_played.csv", parse_dates=["utc_date"])
    combined = pd.concat([historical, current], ignore_index=True)
    return combined.sort_values("utc_date").reset_index(drop=True)


def build_end_state_histories():
    known_results = load_known_results()
    team_history, team_home_history, team_away_history, h2h_history = build_team_histories()
    for _, match in known_results.iterrows():
        update_histories(match, team_history, team_home_history, team_away_history, h2h_history)
    return team_history, team_home_history, team_away_history, h2h_history, known_results


def main():
    for path in (FEATURE_COLUMNS_PATH, MODEL_PATH, IMPUTER_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run train_final_model.py first.", file=sys.stderr)
            sys.exit(1)

    fixtures_path = RAW_DIR / "upcoming_fixtures.csv"
    if not fixtures_path.exists():
        print(f"ERROR: {fixtures_path} not found. Run fetch_upcoming_fixtures.py first.", file=sys.stderr)
        sys.exit(1)

    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    team_history, team_home_history, team_away_history, h2h_history, known_results = build_end_state_histories()
    print(f"Replayed {len(known_results)} known results to build end-state team histories.")

    fixtures = pd.read_csv(fixtures_path, parse_dates=["utc_date"])
    print(f"Scoring {len(fixtures)} upcoming fixtures...")

    feature_rows = []
    for _, fx in fixtures.iterrows():
        feats = compute_match_features(
            fx["home_team"], fx["away_team"], team_history, team_home_history, team_away_history, h2h_history
        )
        feature_rows.append(feats)

    X = pd.DataFrame(feature_rows)[feature_columns]
    X_imp = imputer.transform(X)
    proba = model.predict_proba(X_imp)
    predicted_label = proba.argmax(axis=1)

    predictions = fixtures.copy()
    predictions["home_win_prob"] = proba[:, 0]
    predictions["draw_prob"] = proba[:, 1]
    predictions["away_win_prob"] = proba[:, 2]
    predictions["predicted_result"] = [LABEL_TO_RESULT[p] for p in predicted_label]
    predictions["predicted_result_display"] = [RESULT_TO_DISPLAY[LABEL_TO_RESULT[p]] for p in predicted_label]
    predictions["confidence"] = proba.max(axis=1)

    cold_start_count = X["home_matches_played_last5"].eq(0).sum() + X["away_matches_played_last5"].eq(0).sum()
    if cold_start_count:
        print(f"  Note: {cold_start_count} team-sides among these fixtures have zero prior matches "
              "in our data (newly promoted or very early in the current season) - their features are "
              "median-imputed, treat those specific predictions with extra skepticism.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"\nSaved {len(predictions)} predictions to {out_path}")
    print(f"Predicted result breakdown: {predictions['predicted_result_display'].value_counts().to_dict()}")

    print("\nNext 5 fixtures:")
    for _, row in predictions.head(5).iterrows():
        print(f"  {row['utc_date'].date()}  {row['home_team']:<18} vs {row['away_team']:<18} "
              f"-> {row['predicted_result_display']:<10} ({row['confidence']:.0%} confidence)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
