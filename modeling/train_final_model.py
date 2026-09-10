"""Train the final Random Forest model on ALL 1,140 rows for production predictions.

Model choice (Random Forest over XGBoost) was finalized after the honest
CV comparison in train_models.py - see that script's output / cv_comparison.json
for the accuracy/log-loss numbers this decision was based on: Random Forest
beat both the baseline (46.4% vs 42.5%) and XGBoost (46.4% vs 41.7%, where
XGBoost actually underperformed the baseline with these fixed hyperparameters).
XGBoost's code is left in train_models.py for the comparison record but is not
used from here on.

Same hyperparameters as the winning CV config (n_estimators=200, max_depth=6,
random_state=42) and the same FEATURE_COLUMNS - imported directly from
train_models.py rather than redefined, so this script can never silently drift
from what was actually evaluated.

Missing values (genuine cold-start NaN - see sanity_check.py) are median-imputed
using an imputer fit on the FULL dataset here (unlike train_models.py, which
refit per CV fold to avoid leakage across folds - there's no "future" to leak
from once we're training on everything for production use).
"""
import sys
from pathlib import Path

import joblib

from train_models import DATASET_PATH, FEATURE_COLUMNS, RESULT_TO_LABEL, load_dataset, make_models
from sklearn.impute import SimpleImputer

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "random_forest_final.pkl"
IMPUTER_PATH = ARTIFACTS_DIR / "imputer.pkl"


def main():
    df = load_dataset()
    print(f"Loaded {len(df)} rows from {DATASET_PATH}")

    X = df[FEATURE_COLUMNS]
    y = df["result"].map(RESULT_TO_LABEL).to_numpy()

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    model = make_models()["Random Forest"]
    model.fit(X_imp, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(FEATURE_COLUMNS, ARTIFACTS_DIR / "feature_columns.pkl")

    print(f"Trained Random Forest on all {len(df)} rows.")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved imputer to {IMPUTER_PATH}")

    importances = sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda t: -t[1])
    print("\nFeature importances (full-data fit):")
    for feat, imp in importances:
        print(f"  {feat:<32} {imp:.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
