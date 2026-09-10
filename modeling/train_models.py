"""Compare Random Forest vs XGBoost on matches_dataset.csv using time-based
cross-validation, and report which one wins.

Why CV instead of a single train/test split: with only 1,140 rows across 3
seasons (see STATUS.md / plan doc "Redirect" section for why - football-data.org's
free tier caps historical access at 3 seasons), one holdout split would leave a
validation set too small and too date-dependent to trust. Instead this uses
sklearn's TimeSeriesSplit (5 expanding-window folds over the chronologically
sorted matches - fold k trains on everything before it and validates on the
next chronological chunk, never on the future) and pools every fold's held-out
predictions into one combined accuracy/log-loss/confusion-matrix report. The
very first ~1/6 of matches are never scored (they're only ever in a training
fold) - that's inherent to expanding-window CV, not a bug.

Target: result mapped to Home Win=0 / Draw=1 / Away Win=2.
Features: the 16 engineered columns from build_features.py (see FEATURE_COLUMNS
below) - no shots/SOT/corners, per the data-source limitation; no team-name
one-hot encoding either, since with only 1,140 rows and 20+ teams that would
mostly just let the model memorize teams rather than generalize from form/H2H.

Missing values (genuine cold-start NaN, see sanity_check.py) are median-imputed,
with the imputer refit on each fold's training portion only - never on data the
fold hasn't "seen" yet, to avoid leaking future-season medians backward.

This script only compares models and reports metrics; it does not persist a
deployable model (there isn't one single model from CV - each fold fits its
own). train_final_model.py does that, using whichever model type wins here.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "matches_dataset.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

RANDOM_STATE = 42
N_SPLITS = 5

FEATURE_COLUMNS = [
    "home_form_pts_last5", "home_goals_scored_last5", "home_goals_conceded_last5", "home_matches_played_last5",
    "away_form_pts_last5", "away_goals_scored_last5", "away_goals_conceded_last5", "away_matches_played_last5",
    "home_team_home_win_pct", "home_team_home_matches_played",
    "away_team_away_win_pct", "away_team_away_matches_played",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_meetings",
]
RESULT_TO_LABEL = {"H": 0, "D": 1, "A": 2}
LABEL_NAMES = ["Home Win", "Draw", "Away Win"]


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found. Run build_features.py first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(DATASET_PATH, parse_dates=["utc_date"])
    return df.sort_values("utc_date").reset_index(drop=True)


def make_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE, eval_metric="mlogloss"
        ),
    }


def run_cv(model_name: str, model_factory, X: pd.DataFrame, y: np.ndarray) -> dict:
    """Runs expanding-window CV, returns pooled out-of-fold predictions plus
    per-fold metrics. model_factory is a zero-arg callable returning a fresh,
    unfit model (so each fold gets its own clean instance)."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_true, oof_pred, oof_proba, oof_idx = [], [], [], []
    fold_metrics = []

    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        model = model_factory()
        model.fit(X_train_imp, y_train)
        pred = model.predict(X_val_imp)
        proba = model.predict_proba(X_val_imp)

        fold_acc = accuracy_score(y_val, pred)
        fold_ll = log_loss(y_val, proba, labels=[0, 1, 2])
        fold_metrics.append({"fold": fold_i, "train_n": len(train_idx), "val_n": len(val_idx),
                              "accuracy": fold_acc, "log_loss": fold_ll})
        print(f"  [{model_name}] fold {fold_i}: train={len(train_idx)} val={len(val_idx)} "
              f"acc={fold_acc:.3f} log_loss={fold_ll:.3f}")

        oof_true.append(y_val)
        oof_pred.append(pred)
        oof_proba.append(proba)
        oof_idx.append(val_idx)

    oof_true = np.concatenate(oof_true)
    oof_pred = np.concatenate(oof_pred)
    oof_proba = np.concatenate(oof_proba)
    oof_idx = np.concatenate(oof_idx)

    return {
        "fold_metrics": fold_metrics,
        "oof_true": oof_true,
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "oof_idx": oof_idx,
        "overall_accuracy": accuracy_score(oof_true, oof_pred),
        "overall_log_loss": log_loss(oof_true, oof_proba, labels=[0, 1, 2]),
        "confusion_matrix": confusion_matrix(oof_true, oof_pred, labels=[0, 1, 2]),
    }


def print_confusion_matrix(cm: np.ndarray, title: str) -> None:
    print(f"\n{title}")
    header = " " * 14 + "".join(f"{name:>12}" for name in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  actual {LABEL_NAMES[i]:<7}" + "".join(f"{v:>12}" for v in row))


def main():
    df = load_dataset()
    print(f"Loaded {len(df)} matches from {DATASET_PATH}\n")

    X = df[FEATURE_COLUMNS]
    y = df["result"].map(RESULT_TO_LABEL).to_numpy()

    models = make_models()
    results = {}
    for name in models:
        print(f"Running {N_SPLITS}-fold expanding-window CV for {name}...")
        # fresh unfit instance per fold, same hyperparameters each time
        factory = (lambda n=name: make_models()[n])
        results[name] = run_cv(name, factory, X, y)
        print()

    n_scored = len(results["Random Forest"]["oof_true"])
    baseline_pred = np.zeros(n_scored, dtype=int)  # always "Home Win" = 0
    baseline_acc = accuracy_score(results["Random Forest"]["oof_true"], baseline_pred)

    print("=" * 78)
    print("MODEL COMPARISON (pooled out-of-fold predictions, all CV folds combined)")
    print("=" * 78)
    print(f"{'Metric':<32}{'Baseline (always Home Win)':>26}{'Random Forest':>14}{'XGBoost':>14}")
    print("-" * 78)
    print(f"{'Accuracy':<32}{baseline_acc:>26.3f}{results['Random Forest']['overall_accuracy']:>14.3f}"
          f"{results['XGBoost']['overall_accuracy']:>14.3f}")
    print(f"{'Log-loss (baseline: n/a)':<32}{'--':>26}{results['Random Forest']['overall_log_loss']:>14.3f}"
          f"{results['XGBoost']['overall_log_loss']:>14.3f}")
    print(f"\n({n_scored} of {len(df)} matches were scored - the earliest matches are only ever "
          "in a training fold under expanding-window CV, never validated on.)")

    for name in ["Random Forest", "XGBoost"]:
        print_confusion_matrix(results[name]["confusion_matrix"], f"{name} confusion matrix (rows=actual, cols=predicted)")

    winner = "Random Forest" if results["Random Forest"]["overall_accuracy"] >= results["XGBoost"]["overall_accuracy"] else "XGBoost"
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"Winner by pooled CV accuracy: {winner}")
    print(
        f"  Random Forest: acc={results['Random Forest']['overall_accuracy']:.3f}, "
        f"log_loss={results['Random Forest']['overall_log_loss']:.3f}\n"
        f"  XGBoost:       acc={results['XGBoost']['overall_accuracy']:.3f}, "
        f"log_loss={results['XGBoost']['overall_log_loss']:.3f}\n"
        f"  Baseline:      acc={baseline_acc:.3f} (always predict Home Win)"
    )
    print(
        "\nCaveats:\n"
        f"  - Only {n_scored} matches are ever scored under 5-fold expanding-window CV on a "
        f"{len(df)}-row dataset - each fold's validation slice is small (~{n_scored // N_SPLITS} matches), "
        "so a few-point accuracy gap between the two models is within noise, not a reliable winner.\n"
        "  - Every EPL match-outcome model struggles with draws specifically - check the confusion matrices "
        "above for whether either model predicts Draw at all, or collapses to Home/Away only.\n"
        "  - This comparison uses fixed hyperparameters (no tuning) to keep the comparison honest and "
        "avoid overfitting hyperparameter choices to this small dataset."
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(FEATURE_COLUMNS, ARTIFACTS_DIR / "feature_columns.pkl")
    summary = {
        "n_total_rows": len(df),
        "n_scored_rows": int(n_scored),
        "n_splits": N_SPLITS,
        "baseline_accuracy": float(baseline_acc),
        "winner": winner,
        "models": {
            name: {
                "overall_accuracy": float(results[name]["overall_accuracy"]),
                "overall_log_loss": float(results[name]["overall_log_loss"]),
                "confusion_matrix": results[name]["confusion_matrix"].tolist(),
                "fold_metrics": results[name]["fold_metrics"],
            }
            for name in results
        },
    }
    with open(ARTIFACTS_DIR / "cv_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved CV comparison summary to {ARTIFACTS_DIR / 'cv_comparison.json'}")
    print(f"Saved feature column list to {ARTIFACTS_DIR / 'feature_columns.pkl'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
