"""Evaluate the chosen model (Random Forest) honestly: accuracy vs baseline,
confusion matrix, WHY draw recall is weak (not just that it is), a calibration
check, and a brief note on why Random Forest beat XGBoost.

Re-runs the same 5-fold expanding-window CV as train_models.py (importing
run_cv/make_models/load_dataset/FEATURE_COLUMNS/RESULT_TO_LABEL/LABEL_NAMES
directly, rather than reimplementing it) specifically to get the pooled
out-of-fold TRUE labels and PREDICTED PROBABILITIES in memory - train_models.py
only persisted the aggregated confusion matrix and accuracy/log-loss to
cv_comparison.json, not the raw per-row arrays, and the draw-recall root-cause
analysis below needs the actual per-row P(draw) values. Re-running CV costs a
few seconds and guarantees this evaluation is scored on the exact same folds
train_models.py reported, not a fresh split that could disagree with it.

Headline framing (per Allen's instruction, 2026-09-10): lead with draw recall,
not topline accuracy. 46.4% vs a 42.5% baseline is a real but thin lift - the
more important finding is how badly the model misses draws specifically, and
WHY (see "DRAW RECALL ROOT CAUSE" below for the class-imbalance-vs-signal-gap
diagnostic this script runs to answer that, not just assert it).
"""
import json
import sys
from pathlib import Path

import numpy as np

from train_models import FEATURE_COLUMNS, LABEL_NAMES, RESULT_TO_LABEL, load_dataset, make_models, run_cv

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DRAW_LABEL = RESULT_TO_LABEL["D"]

# Confidence bins for the calibration check (max predicted-class probability -> actual accuracy in that bin)
CONFIDENCE_BINS = [(0.0, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 1.01)]


def print_confusion_matrix(cm: np.ndarray, title: str) -> None:
    print(f"\n{title}")
    header = " " * 14 + "".join(f"{name:>12}" for name in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  actual {LABEL_NAMES[i]:<7}" + "".join(f"{v:>12}" for v in row))


def draw_recall_root_cause(oof_true: np.ndarray, oof_proba: np.ndarray) -> dict:
    """Distinguishes "model rarely predicts draw even when it should"
    (threshold/argmax issue - some signal exists but never wins 3-way argmax)
    from "the features genuinely don't separate draws" (signal gap).
    """
    is_actual_draw = oof_true == DRAW_LABEL
    p_draw = oof_proba[:, DRAW_LABEL]

    mean_p_draw_when_actual_draw = p_draw[is_actual_draw].mean()
    mean_p_draw_when_not_draw = p_draw[~is_actual_draw].mean()

    predicted_labels = oof_proba.argmax(axis=1)
    predicted_draw_rate = (predicted_labels == DRAW_LABEL).mean()
    actual_draw_rate = is_actual_draw.mean()

    # Precision@k: take the k=n_actual_draws matches the model rated MOST
    # likely to be a draw (by P(draw), regardless of whether draw actually
    # won the 3-way argmax) - if that top-k set is enriched for real draws
    # well above the base rate, there IS real ranking signal that argmax is
    # just failing to surface. If it's close to the base rate, there isn't.
    k = int(is_actual_draw.sum())
    top_k_idx = np.argsort(-p_draw)[:k]
    precision_at_k = is_actual_draw[top_k_idx].mean()

    return {
        "mean_p_draw_when_actual_draw": float(mean_p_draw_when_actual_draw),
        "mean_p_draw_when_not_draw": float(mean_p_draw_when_not_draw),
        "predicted_draw_rate": float(predicted_draw_rate),
        "actual_draw_rate": float(actual_draw_rate),
        "precision_at_k": float(precision_at_k),
        "k": k,
        "baseline_rate_for_precision_at_k": float(actual_draw_rate),
    }


def calibration_check(oof_true: np.ndarray, oof_proba: np.ndarray) -> list[dict]:
    """For each confidence bin (max predicted-class probability), reports
    how often the model was actually right within that bin. Well-calibrated
    means "60% confident" predictions are right ~60% of the time.
    """
    predicted_labels = oof_proba.argmax(axis=1)
    confidence = oof_proba.max(axis=1)
    is_correct = predicted_labels == oof_true

    rows = []
    for low, high in CONFIDENCE_BINS:
        mask = (confidence >= low) & (confidence < high)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": f"[{low:.2f}, {high:.2f})", "n": 0, "mean_confidence": None, "actual_accuracy": None})
            continue
        rows.append({
            "bin": f"[{low:.2f}, {high:.2f})",
            "n": n,
            "mean_confidence": float(confidence[mask].mean()),
            "actual_accuracy": float(is_correct[mask].mean()),
        })
    return rows


def main():
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df["result"].map(RESULT_TO_LABEL).to_numpy()

    print("Re-running 5-fold expanding-window CV for Random Forest (same as train_models.py)...")
    rf_factory = lambda: make_models()["Random Forest"]
    result = run_cv("Random Forest", rf_factory, X, y)
    oof_true, oof_proba = result["oof_true"], result["oof_proba"]
    n_scored = len(oof_true)

    baseline_acc = float((oof_true == RESULT_TO_LABEL["H"]).mean())

    print("\n" + "=" * 78)
    print("HEADLINE: DRAW RECALL, NOT TOPLINE ACCURACY")
    print("=" * 78)
    cm = result["confusion_matrix"]
    draw_recall = cm[DRAW_LABEL, DRAW_LABEL] / cm[DRAW_LABEL].sum()
    print(
        f"Random Forest: {result['overall_accuracy']:.1%} accuracy vs a {baseline_acc:.1%} baseline "
        "(always predict Home Win) - a real but thin lift, not a strong result on its own.\n"
        f"The more important finding: the model catches only {cm[DRAW_LABEL, DRAW_LABEL]} of "
        f"{cm[DRAW_LABEL].sum()} actual draws ({draw_recall:.1%} recall). Most true draws get "
        f"misclassified as Home Win ({cm[DRAW_LABEL, RESULT_TO_LABEL['H']]} of them) - the model is "
        "effectively a Home/Away classifier that occasionally guesses Draw, not a genuine 3-way classifier."
    )
    print_confusion_matrix(cm, "Confusion matrix (rows=actual, cols=predicted)")

    print("\n" + "=" * 78)
    print("DRAW RECALL ROOT CAUSE: threshold/argmax issue, or genuine signal gap?")
    print("=" * 78)
    root_cause = draw_recall_root_cause(oof_true, oof_proba)
    print(
        f"Mean P(Draw) when the match WAS actually a draw:     {root_cause['mean_p_draw_when_actual_draw']:.3f}\n"
        f"Mean P(Draw) when the match was NOT a draw:           {root_cause['mean_p_draw_when_not_draw']:.3f}\n"
        f"  -> gap: {root_cause['mean_p_draw_when_actual_draw'] - root_cause['mean_p_draw_when_not_draw']:+.3f}\n"
        f"Model's predicted draw rate (how often Draw wins the 3-way argmax): {root_cause['predicted_draw_rate']:.1%}\n"
        f"Actual draw rate in the scored data:                                {root_cause['actual_draw_rate']:.1%}\n"
        f"Precision@k (top {root_cause['k']} matches by P(Draw), regardless of argmax): "
        f"{root_cause['precision_at_k']:.1%} were actually draws (base rate: {root_cause['baseline_rate_for_precision_at_k']:.1%})"
    )
    gap = root_cause["mean_p_draw_when_actual_draw"] - root_cause["mean_p_draw_when_not_draw"]
    precision_lift = root_cause["precision_at_k"] - root_cause["baseline_rate_for_precision_at_k"]
    if gap > 0.03 and precision_lift > 0.03:
        verdict = (
            "THRESHOLD/ARGMAX ISSUE, not a pure signal gap: P(Draw) IS meaningfully higher for actual "
            f"draws ({root_cause['mean_p_draw_when_actual_draw']:.3f} vs {root_cause['mean_p_draw_when_not_draw']:.3f}), "
            f"and ranking by P(Draw) beats the base rate by {precision_lift:+.1%} at precision@k. The features "
            "carry real (if weak) draw signal - it just rarely wins the 3-way argmax because Home/Away "
            "probabilities are usually higher. Future work: probability-threshold tuning or class "
            "rebalancing (e.g. class_weight, or a lower decision threshold specifically for Draw) is likely "
            "to help more than new features."
        )
    else:
        verdict = (
            "GENUINE SIGNAL GAP: P(Draw) for actual draws is barely distinguishable from non-draws "
            f"({root_cause['mean_p_draw_when_actual_draw']:.3f} vs {root_cause['mean_p_draw_when_not_draw']:.3f}), "
            f"and ranking by P(Draw) only beats the base rate by {precision_lift:+.1%}. The current feature set "
            "(goals/form/H2H, no shots data) doesn't actually separate draws from decisive results. Future "
            "work: threshold tuning alone won't fix this - it needs features that distinguish 'evenly matched' "
            "from 'one side is just better' (e.g. league-position gap, expected-goals proxies) rather than "
            "reweighting what's already here."
        )
    print(f"\nVERDICT: {verdict}")

    print("\n" + "=" * 78)
    print("CALIBRATION CHECK: does 'X% confident' mean ~X% right?")
    print("=" * 78)
    calib_rows = calibration_check(oof_true, oof_proba)
    print(f"{'Confidence bin':<18}{'n':>6}{'mean confidence':>18}{'actual accuracy':>18}")
    for row in calib_rows:
        if row["n"] == 0:
            print(f"{row['bin']:<18}{0:>6}{'--':>18}{'--':>18}")
        else:
            print(f"{row['bin']:<18}{row['n']:>6}{row['mean_confidence']:>18.3f}{row['actual_accuracy']:>18.3f}")
    print(
        "\nWell-calibrated means each row's 'actual accuracy' column roughly matches its 'mean confidence' "
        "column. Systematic over- or under-confidence (accuracy consistently below/above confidence) means "
        "predicted probabilities shouldn't be read as literal win percentages without adjustment."
    )

    print("\n" + "=" * 78)
    print("WHY RANDOM FOREST BEAT XGBOOST")
    print("=" * 78)
    with open(ARTIFACTS_DIR / "cv_comparison.json", encoding="utf-8") as f:
        cv_summary = json.load(f)
    rf_acc = cv_summary["models"]["Random Forest"]["overall_accuracy"]
    xgb_acc = cv_summary["models"]["XGBoost"]["overall_accuracy"]
    print(
        f"Random Forest {rf_acc:.1%} vs XGBoost {xgb_acc:.1%} (XGBoost underperformed even the baseline). "
        f"With only {n_scored} scored rows and 3 classes, this is exactly the regime where XGBoost's more "
        "flexible boosting tends to overfit relative to Random Forest's more conservative, bagging-averaged "
        "bias - not evidence that XGBoost is a worse algorithm in general, just a poor fit for this little data."
    )

    print("\n" + "=" * 78)
    print("LIMITATIONS (for README)")
    print("=" * 78)
    limitations = [
        f"Small sample: only {n_scored} matches are ever scored under 5-fold expanding-window CV on a "
        f"{len(df)}-row dataset (free-tier API access caps historical data at 3 seasons) - a few points of "
        "accuracy either direction is within noise.",
        f"Draws are hard to predict: {draw_recall:.1%} recall, and the root-cause analysis above found this "
        f"is a {'threshold/argmax' if (gap > 0.03 and precision_lift > 0.03) else 'genuine feature signal'} "
        "issue, not just 'draws are inherently unpredictable' - see the VERDICT above.",
        "No shots/SOT/corners/cards data: football-data.org's free tier doesn't expose these fields at all "
        "(confirmed from the raw API schema), so features are limited to goals/form/H2H.",
        "No player-availability/injury data: two teams with identical recent form can have very different "
        "real availability (injuries, suspensions) that this model has no way to see.",
        "Early-season and newly-promoted-team cold start: rolling features are NaN (median-imputed) for a "
        "team's first few matches in the dataset - see sanity_check.py's cold-start integrity check.",
    ]
    for l in limitations:
        print(f"  - {l}")

    # --- Save a summary for docs/export_for_web.py to reuse ---
    summary = {
        "n_total_rows": len(df),
        "n_scored_rows": n_scored,
        "baseline_accuracy": baseline_acc,
        "model_accuracy": result["overall_accuracy"],
        "model_log_loss": result["overall_log_loss"],
        "confusion_matrix": cm.tolist(),
        "label_names": LABEL_NAMES,
        "draw_recall": float(draw_recall),
        "draw_root_cause": root_cause,
        "draw_root_cause_verdict": "threshold_argmax" if (gap > 0.03 and precision_lift > 0.03) else "signal_gap",
        "calibration": calib_rows,
        "limitations": limitations,
    }
    with open(ARTIFACTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved evaluation summary to {ARTIFACTS_DIR / 'evaluation_summary.json'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
