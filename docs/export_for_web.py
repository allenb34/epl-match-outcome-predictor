"""One-off export: turns predictions.csv + evaluation_summary.json into the
static docs/index.html (fixtures dashboard) and docs/performance.html
(model performance page), with zero Python needed at runtime.

Standalone - reads existing pipeline outputs (data/processed/predictions.csv,
modeling/artifacts/evaluation_summary.json) but doesn't modify or retrain
anything. Re-run this whenever predictions or evaluation change and the site
needs updating (the weekly GitHub Actions refresh does this automatically).

Framing (per Allen's explicit instruction, 2026-09-10): the confusion matrix
and the precision@k diagnostic must be visible ON the performance page itself,
not just in STATUS.md - precision@k is "the most defensible part of this
project." The headline insight - beats baseline on Home/Away, no signal on
match evenness, verified via precision@k not just eyeballed from the
confusion matrix - leads both pages, not the 46.4% accuracy number alone.
Future Work names concrete candidates (league-position gap, xG proxies), not
"more features needed."

All statistics rendered on both pages are computed in Python directly from
evaluation_summary.json / predictions.csv and injected as plain HTML strings
(not recomputed in JS) - the numbers should never be able to drift from what
evaluate.py actually measured. PREDICTIONS_JSON is still embedded for a
lightweight client-side team-name search/filter over the fixtures table
(the only actual interactivity on either page); PERFORMANCE_JSON is embedded
purely for inspectability, matching the sibling project's "plain JSON export
alongside the embedded HTML" convention.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "modeling" / "artifacts"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "predictions.csv"
EVALUATION_PATH = ARTIFACTS_DIR / "evaluation_summary.json"
DOCS_DIR = Path(__file__).resolve().parent

INDEX_TEMPLATE_PATH = DOCS_DIR / "index_template.html"
INDEX_PATH = DOCS_DIR / "index.html"
PERFORMANCE_TEMPLATE_PATH = DOCS_DIR / "performance_template.html"
PERFORMANCE_PATH = DOCS_DIR / "performance.html"

FUTURE_WORK = [
    (
        "League-position gap",
        "The current feature set has no notion of \"how good is this team relative to its opponent right "
        "now\" beyond short-term form - a team's league-table position (or points-per-game normalized "
        "across the season) relative to its opponent is a direct, cheap-to-compute proxy for exactly the "
        "\"evenly matched vs. one side is just better\" signal the precision@k diagnostic found missing.",
    ),
    (
        "Expected-goals (xG) proxies",
        "Real shot-quality data (xG) isn't available from football-data.org's free tier, but a rough proxy "
        "- e.g. goals scored/conceded relative to shots-on-target ratios from a supplementary source, or a "
        "simple Poisson-based expected-goals estimate from historical scoring rates - would capture "
        "\"who's been getting the run of play\" in a way raw goals-only rolling form can't.",
    ),
]


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        print(f"ERROR: {PREDICTIONS_PATH} not found. Run modeling/predict.py first.", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(PREDICTIONS_PATH, parse_dates=["utc_date"])


def load_evaluation() -> dict:
    if not EVALUATION_PATH.exists():
        print(f"ERROR: {EVALUATION_PATH} not found. Run modeling/evaluate.py first.", file=sys.stderr)
        sys.exit(1)
    with open(EVALUATION_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------- HTML fragment builders (all computed from real data, never hardcoded numbers) ----------

def fixtures_table_html(predictions: pd.DataFrame) -> str:
    rows = []
    for _, fx in predictions.iterrows():
        date_label = fx["utc_date"].strftime("%a %d %b, %H:%M UTC")
        result_class = {"Home Win": "tag-home", "Draw": "tag-draw", "Away Win": "tag-away"}[fx["predicted_result_display"]]
        rows.append(
            f'<tr data-home="{fx["home_team"]}" data-away="{fx["away_team"]}">'
            f'<td>{date_label}</td>'
            f'<td>{fx["home_team"]}</td><td>{fx["away_team"]}</td>'
            f'<td><span class="tag {result_class}">{fx["predicted_result_display"]}</span></td>'
            f'<td>{fx["confidence"]:.0%}</td>'
            f'<td class="probs">H {fx["home_win_prob"]:.0%} &middot; D {fx["draw_prob"]:.0%} &middot; A {fx["away_win_prob"]:.0%}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def confusion_matrix_html(cm: list, label_names: list) -> str:
    header = "".join(f"<th>{name}</th>" for name in label_names)
    body_rows = []
    for i, row in enumerate(cm):
        row_total = sum(row)
        recall = row[i] / row_total if row_total else 0
        cells = "".join(
            f'<td class="{"diag" if i == j else ""}">{v}</td>' for j, v in enumerate(row)
        )
        body_rows.append(f'<tr><th>{label_names[i]}</th>{cells}<td class="recall">{recall:.1%}</td></tr>')
    return f"""
    <table class="matrix">
      <thead>
        <tr><th rowspan="2"></th><th colspan="{len(label_names)}">Predicted</th><th rowspan="2">Recall</th></tr>
        <tr>{header}</tr>
      </thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
    <p class="matrix-caption">Rows = actual result, columns = predicted result. Recall = how often that
    actual outcome was correctly identified.</p>
    """


def precision_at_k_html(evaluation: dict) -> str:
    rc = evaluation["draw_root_cause"]
    gap = rc["mean_p_draw_when_actual_draw"] - rc["mean_p_draw_when_not_draw"]
    lift = rc["precision_at_k"] - rc["baseline_rate_for_precision_at_k"]
    verdict_label = "Genuine signal gap" if evaluation["draw_root_cause_verdict"] == "signal_gap" else "Threshold / argmax issue"
    return f"""
    <div class="diagnostic">
      <h3>Precision@k diagnostic: does the model have ANY draw signal?</h3>
      <p>The confusion matrix above shows draws are rarely <em>predicted</em> - but that alone can't tell
      you whether the model has weak draw signal that argmax discards, or no draw signal at all. This
      diagnostic answers that directly: take the top {rc['k']} fixtures the model rated <strong>most
      likely to be a draw</strong> by raw P(Draw), regardless of what actually won the 3-way argmax, and
      check how many really were draws.</p>
      <div class="stat-grid">
        <div class="stat"><div class="stat-label">Mean P(Draw), actual draws</div><div class="stat-value">{rc['mean_p_draw_when_actual_draw']:.3f}</div></div>
        <div class="stat"><div class="stat-label">Mean P(Draw), actual non-draws</div><div class="stat-value">{rc['mean_p_draw_when_not_draw']:.3f}</div></div>
        <div class="stat"><div class="stat-label">Gap</div><div class="stat-value {'stat-good' if gap > 0.03 else 'stat-bad'}">{gap:+.3f}</div></div>
        <div class="stat"><div class="stat-label">Precision@{rc['k']}</div><div class="stat-value">{rc['precision_at_k']:.1%}</div></div>
        <div class="stat"><div class="stat-label">Base rate (random)</div><div class="stat-value">{rc['baseline_rate_for_precision_at_k']:.1%}</div></div>
        <div class="stat"><div class="stat-label">Lift over random</div><div class="stat-value {'stat-good' if lift > 0.03 else 'stat-bad'}">{lift:+.1%}</div></div>
      </div>
      <p class="verdict"><strong>Verdict: {verdict_label}.</strong>
      P(Draw) for actual draws ({rc['mean_p_draw_when_actual_draw']:.3f}) is essentially identical to
      P(Draw) for non-draws ({rc['mean_p_draw_when_not_draw']:.3f}), and ranking fixtures by P(Draw) only
      beats random guessing by {lift:+.1%}. If this had been a threshold problem, that ranking would show
      real separation even though argmax still picks Home/Away - it doesn't. The current feature set
      (goals/form/H2H, no shots data) genuinely doesn't distinguish an evenly-matched game from a lopsided
      one. See Future Work below for what would actually address this.</p>
    </div>
    """


def calibration_table_html(evaluation: dict) -> str:
    rows = []
    for row in evaluation["calibration"]:
        if row["n"] == 0:
            rows.append(f'<tr><td>{row["bin"]}</td><td>0</td><td>--</td><td>--</td></tr>')
            continue
        gap = row["actual_accuracy"] - row["mean_confidence"]
        gap_class = "stat-bad" if abs(gap) > 0.08 else ""
        rows.append(
            f'<tr><td>{row["bin"]}</td><td>{row["n"]}</td>'
            f'<td>{row["mean_confidence"]:.1%}</td><td class="{gap_class}">{row["actual_accuracy"]:.1%}</td></tr>'
        )
    return f"""
    <table class="calibration">
      <thead><tr><th>Confidence bin</th><th>n</th><th>Mean confidence</th><th>Actual accuracy</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p class="matrix-caption">Well-calibrated means each row's "actual accuracy" roughly matches its "mean
    confidence." The highest-confidence bin here is mildly overconfident - don't read a 60%+ predicted
    probability as literally that reliable.</p>
    """


def limitations_html(limitations: list) -> str:
    return "\n".join(f"<li>{l}</li>" for l in limitations)


def future_work_html() -> str:
    items = "\n".join(f"<li><strong>{title}.</strong> {desc}</li>" for title, desc in FUTURE_WORK)
    return f"<ul class='future-work'>{items}</ul>"


def build_index_html(predictions: pd.DataFrame, evaluation: dict):
    if not INDEX_TEMPLATE_PATH.exists():
        print(f"ERROR: {INDEX_TEMPLATE_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    draw_recall = evaluation["draw_recall"]
    predicted_draw_count = int((predictions["predicted_result_display"] == "Draw").sum())
    headline_html = f"""
    <div class="headline-callout">
      <strong>Headline finding:</strong> this model beats the baseline on distinguishing Home Win from
      Away Win, but has <strong>no real signal on match evenness</strong> - it catches only
      {draw_recall:.1%} of actual draws, and only predicts {predicted_draw_count} draws across all
      {len(predictions)} upcoming fixtures below. Verified with a precision@k diagnostic, not just eyeballed
      from the confusion matrix - see the <a href="performance.html">model performance page</a> for the
      full breakdown.
    </div>
    """

    template = INDEX_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__HEADLINE_CALLOUT_HTML__", headline_html)
    html = html.replace("__FIXTURES_TABLE_HTML__", fixtures_table_html(predictions))
    html = html.replace("__TOTAL_FIXTURES__", str(len(predictions)))
    html = html.replace(
        "__LAST_UPDATED__", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )
    predictions_json = predictions.assign(
        utc_date=predictions["utc_date"].astype(str)
    ).to_dict(orient="records")
    html = html.replace("__PREDICTIONS_JSON__", json.dumps(predictions_json))

    if "__" in html and any(ph in html for ph in ["__HEADLINE_CALLOUT_HTML__", "__FIXTURES_TABLE_HTML__", "__PREDICTIONS_JSON__"]):
        print("ERROR: index template placeholder(s) were not fully substituted.", file=sys.stderr)
        sys.exit(1)

    INDEX_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {INDEX_PATH}")


def build_performance_html(evaluation: dict):
    if not PERFORMANCE_TEMPLATE_PATH.exists():
        print(f"ERROR: {PERFORMANCE_TEMPLATE_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    template = PERFORMANCE_TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "__MODEL_ACCURACY__": f"{evaluation['model_accuracy']:.1%}",
        "__BASELINE_ACCURACY__": f"{evaluation['baseline_accuracy']:.1%}",
        "__MODEL_LOG_LOSS__": f"{evaluation['model_log_loss']:.3f}",
        "__N_SCORED_ROWS__": str(evaluation["n_scored_rows"]),
        "__N_TOTAL_ROWS__": str(evaluation["n_total_rows"]),
        "__DRAW_RECALL__": f"{evaluation['draw_recall']:.1%}",
        "__CONFUSION_MATRIX_HTML__": confusion_matrix_html(evaluation["confusion_matrix"], evaluation["label_names"]),
        "__PRECISION_AT_K_HTML__": precision_at_k_html(evaluation),
        "__CALIBRATION_TABLE_HTML__": calibration_table_html(evaluation),
        "__LIMITATIONS_HTML__": limitations_html(evaluation["limitations"]),
        "__FUTURE_WORK_HTML__": future_work_html(),
        "__PERFORMANCE_JSON__": json.dumps(evaluation),
    }
    html = template
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    remaining = [p for p in replacements if p in html]
    if remaining:
        print(f"ERROR: performance template placeholder(s) not substituted: {remaining}", file=sys.stderr)
        sys.exit(1)

    PERFORMANCE_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {PERFORMANCE_PATH}")


def main():
    predictions = load_predictions()
    evaluation = load_evaluation()

    # Plain JSON exports, for inspectability - not what the pages actually load (see module docstring)
    with open(DOCS_DIR / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions.assign(utc_date=predictions["utc_date"].astype(str)).to_dict(orient="records"), f, indent=2)
    with open(DOCS_DIR / "performance.json", "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    print(f"Wrote {DOCS_DIR / 'predictions.json'} and {DOCS_DIR / 'performance.json'}")

    build_index_html(predictions, evaluation)
    build_performance_html(evaluation)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
