# EPL Match Outcome Predictor — Status

Last updated: 2026-09-10 (Stage 5 complete and verified, Stage 6 next)

Full build plan: `C:\Users\allen\.claude\plans\spec-epl-match-outcome-lovely-sunbeam.md`
(includes a "Redirect (2026-09-09)" section — read that before anything else, it
supersedes parts of the original plan below it).

## Key context a fresh session needs

- **Historical data source changed mid-build.** The original plan used
  football-data.co.uk (free CSVs). That site went down site-wide (503) for 10+
  hours and Allen redirected to **football-data.org** instead (same API/key
  already used for fixtures). This is final, not a fallback.
- **Free tier only gives 3 complete historical seasons**: 2023-24, 2024-25,
  2025-26 (`season=2023/2024/2025`). `season=2022` and earlier returns 403
  (hard wall, not throttling). Do not attempt to pull more historical seasons
  from this API — it will fail.
- **No shots/SOT/corners/cards data exists anywhere in football-data.org's
  schema** (confirmed from a raw match object) — only date, teams, full-time
  score, half-time score, matchday, referee. The feature set was trimmed to
  match: rolling form (points/goals only), home-advantage indicator, H2H.
- **Time-based cross-validation, not a single holdout split** — confirmed by
  Allen given the small sample (1,140 rows). Use `sklearn.TimeSeriesSplit` /
  expanding-window folds over the chronologically-sorted dataset.
- **Current season (2026-27) matches are NOT training data.** Its 30
  already-played matches (`data/raw/current_season_played.csv`) exist only to
  seed rolling-form features for predicting upcoming fixtures — never fed into
  training/CV as labeled rows.
- **.env already has a working `FOOTBALL_DATA_API_KEY`** (copied from the
  sibling `pl-transfer-value-predictor` project, same key, same person's
  account). Don't regenerate or ask for a new one.
- **Windows quirks** (see memory `windows-python-env`): `truststore.inject_into_ssl()`
  is required before any `requests` HTTPS call (already wired into
  `data_collection/http_utils.py`). Console scripts aren't on PATH — invoke
  Python directly (`python foo.py`), not via a bare command.
- Project has its own (not yet created) git repo — nothing has been pushed to
  GitHub yet. Repo creation and first push need Allen's explicit go-ahead
  before they happen (one-way public action).

## Done

### Stage 1 — Data collection (verified against live APIs)
- `data_collection/http_utils.py` — shared rate limiter, retry/backoff, error logging.
- `data_collection/fetch_historical_results.py` — pulls 2023-24/2024-25/2025-26
  finished matches from football-data.org. Output: `data/raw/historical_matches.csv`
  (1,140 rows). Real result split: 492 home wins (43.2%), 369 away wins, 279 draws
  — **43.2% is the actual baseline to beat**, not the spec's ~45-46% estimate.
- `data_collection/fetch_upcoming_fixtures.py` — pulls (a) scheduled fixtures →
  `data/raw/upcoming_fixtures.csv` (350 rows) and (b) current season's played
  matches → `data/raw/current_season_played.csv` (30 rows, context only).
- All three scripts run clean end-to-end against real data, outputs spot-checked.

### Stage 2 — Feature engineering (verified against live data)
- `data_collection/build_features.py` — leak-free chronological replay
  (`build_team_histories` / `compute_match_features` / `update_histories` /
  `replay_and_featurize`, all reusable and meant to be imported by
  `modeling/predict.py` later rather than reimplemented). Output:
  `data/processed/matches_dataset.csv` (1,140 rows, 16 engineered feature columns
  + original columns, zero rows dropped).
- `data_collection/sanity_check.py` — row count, result-distribution, season
  coverage, and cold-start-integrity checks. Currently **all pass**.
  - Note: first version of this check wrongly assumed cold-start (NaN rolling
    features) only happens in the first season (2023-24). It correctly caught
    25 "unexpected" NaN cases in later seasons — turned out to be newly
    promoted teams' (Ipswich Town, Southampton, Leicester City, Sunderland,
    Leeds United) first-ever match in our 3-season window, which is genuine
    cold start. Fixed the check to validate the real invariant instead: every
    NaN must correspond to an actual `matches_played == 0`. If you touch this
    file again, don't reintroduce the season-1-only assumption.

## Done (continued)

### Stage 3a — Model comparison (`modeling/train_models.py`, verified against real data)
- Added `xgboost>=2.0` to `requirements.txt` and installed it (`pip install xgboost` -
  wasn't present initially, `ModuleNotFoundError` on first run, fixed).
- 5-fold expanding-window `TimeSeriesSplit` CV (pooled out-of-fold predictions,
  950 of 1,140 rows ever scored — the earliest ~1/6 is only ever training data
  under expanding-window CV, that's inherent to the method not a bug).
- Median imputation for cold-start NaNs, refit per-fold on train portion only.
- **Real results**: baseline (always Home Win) = 42.5% accuracy. Random Forest =
  46.4% accuracy, log-loss 1.061 — modest lift over baseline. **XGBoost
  underperformed the baseline at 41.7% accuracy, log-loss 1.293** — worse than
  guessing home win every time, with these fixed (untuned) hyperparameters.
  **Winner: Random Forest.** Both models are weak on draws specifically — RF's
  confusion matrix catches only 19 of 243 actual draws (mostly misclassifies
  them as Home Win); XGBoost does somewhat better on draws (51/243) but at the
  cost of overall accuracy. This is consistent with the well-known "draws are
  hard to predict" limitation the spec anticipated — now backed by real numbers
  to cite in the README, not just the general expectation.
- Saved `modeling/artifacts/cv_comparison.json` (full metrics) and
  `modeling/artifacts/feature_columns.pkl` (the 16-column feature list, for
  `train_final_model.py` / `predict.py` to reuse exactly).

### Stage 3b — Final model (`modeling/train_final_model.py`, verified against real data)
- Retrained **Random Forest** (the CV winner) on all 1,140 rows, same
  hyperparameters as the winning CV config (`n_estimators=200, max_depth=6,
  random_state=42`). Imports `FEATURE_COLUMNS` / `RESULT_TO_LABEL` /
  `load_dataset` / `make_models` directly from `train_models.py` rather than
  redefining them, so it can't silently drift from what CV actually evaluated.
- Median imputer fit on the full dataset (unlike `train_models.py`, which
  refits per-fold — no "future" to leak from once training on everything).
- Saved `modeling/artifacts/random_forest_final.pkl`, `imputer.pkl`,
  `feature_columns.pkl`. Verified: reloaded all three artifacts fresh and ran
  a sample `predict_proba` — works, `n_features_in_` = 16 as expected.
- Feature importances (full-data fit) make intuitive sense — dominated by
  `home_team_home_win_pct` (0.196) and `away_team_away_win_pct` (0.111), i.e.
  the home-advantage indicator carries the most signal of anything in the set.
  H2H features are weakest (0.02–0.04 each) — worth noting in the README as a
  candidate to simplify/drop in a v2 if it doesn't earn its complexity.

Stage 3 (modeling) is now fully done and verified.

### Stage 4 — Evaluation (`modeling/evaluate.py`, verified against real data)
Re-runs the same CV as `train_models.py` (imports `run_cv`/`make_models` rather
than reimplementing) specifically to get pooled out-of-fold true labels +
predicted probabilities in memory, since `cv_comparison.json` only persisted
the aggregated confusion matrix, not per-row arrays.

**Real findings (per Allen's framing instruction — draw recall first, not
topline accuracy):**
- 46.4% accuracy vs 42.5% baseline — "a real but thin lift, not a strong
  result on its own" (this is the framing used in the script's own output,
  not just this summary).
- **Draw recall: 7.8%** (19 of 243 actual draws caught). 124 of those 243 get
  misclassified as Home Win. The model is "effectively a Home/Away classifier
  that occasionally guesses Draw."
- **Root-cause investigation (this is the part Allen specifically asked for
  — not just asserting draws are hard, but checking WHY):**
  - Mean P(Draw) for actual-draw rows: 0.246. Mean P(Draw) for non-draw rows:
    0.247. **Essentially identical — no gap.**
  - Precision@k (top 243 matches ranked by P(Draw)): 25.9% actually draws,
    vs a 25.6% base rate — **no better than random ranking.**
  - **Verdict: GENUINE SIGNAL GAP, not a threshold/argmax issue.** If it had
    been a threshold issue, P(Draw) would be meaningfully elevated for true
    draws even when Home/Away still wins the argmax — it isn't. The current
    feature set (goals/form/H2H, no shots data) doesn't separate "evenly
    matched" from "one side is just better" at all.
  - **This determines the Future Work recommendation**: threshold tuning or
    class rebalancing won't fix this on their own — it needs features that
    capture match evenness (e.g. league-position gap, expected-goals proxies),
    not reweighting of what's already there.
- **Calibration**: roughly reasonable trend (confidence bin 0.40–0.50 →
  actual accuracy 0.422; 0.50–0.60 → 0.551), but the highest-confidence bin
  (0.60–1.01, mean confidence 0.662) is mildly overconfident — actual accuracy
  only 0.598 there. Worth a one-line README caveat: don't read "60%+ confident"
  predictions as literally that reliable.
- **Why Random Forest beat XGBoost** (one-sentence ask, not deep investigation):
  46.4% vs 41.7%, with XGBoost actually underperforming baseline — with only
  950 scored rows and 3 classes, this is exactly the regime where XGBoost's
  more flexible boosting tends to overfit relative to Random Forest's more
  conservative bagging-averaged bias. Not evidence XGBoost is worse in
  general, just a poor fit for this little data.
- Saved `modeling/artifacts/evaluation_summary.json` for `docs/export_for_web.py`
  to reuse in Stage 5 (baseline/model accuracy, confusion matrix, draw
  root-cause verdict, calibration table, limitations list — all as structured
  data, not just printed text).

Allen reviewed these findings and approved proceeding to Stage 5, with explicit
framing instructions (see below) — no longer blocked.

### Stage 5 — Static site (`modeling/predict.py` done; `docs/` in progress)

**Allen's framing instructions for this stage** (2026-09-10, follow these when
writing the dashboard, not just the accuracy number):
- Headline insight: "beats baseline on Home/Away distinction, but has no
  signal on match evenness — verified via precision@k, not just observed from
  the confusion matrix." Lead with this, not the 46.4% accuracy figure alone.
- The confusion matrix AND the precision@k diagnostic must both be visible on
  the dashboard's performance page itself — not just in STATUS.md/evaluate.py
  output. Allen called precision@k "the most defensible part of this project."
- Future Work must name concrete next feature candidates — **league-position
  gap** and **xG proxies** specifically — not a vague "more features needed."

**`modeling/predict.py`** (done, verified against real data):
- Reuses `build_team_histories`/`compute_match_features`/`update_histories`
  directly from `data_collection/build_features.py` (imported via sys.path
  insert, not reimplemented) — replays `historical_matches.csv` +
  `current_season_played.csv` together (1,170 known results) to reach the
  true end-state of every team's history, then scores each of the 350
  upcoming fixtures read-only against that end-state.
- Output: `data/processed/predictions.csv` (350 rows: probabilities +
  predicted result + confidence per fixture).
- Real result: predicted breakdown is 258 Home Win / 77 Away Win / **only 15
  Draw** (4.3%) — even more skewed than the 9.5% CV predicted-draw-rate,
  consistent with the Stage 4 finding. Worth surfacing on the dashboard itself
  as a live example of the draw-blindness finding, not just an abstract stat.
- Noted: intra-matchday ordering is ignored (all of a matchday's fixtures are
  featurized against the same end-state) — a standard simplifying assumption,
  documented in the script's docstring and worth one README line.

**`docs/export_for_web.py` + `docs/index_template.html` + `docs/performance_template.html`
(done, verified in the Browser pane via `file://`, both pages, no server):**
- All statistics on both pages are rendered as plain HTML strings computed in
  Python directly from `evaluation_summary.json` / `predictions.csv` — never
  recomputed in JS, so they can't drift from what `evaluate.py` measured.
- `index.html`: headline callout (draw-recall + precision@k framing, links to
  performance page) above the fixtures table; fixtures table embeds
  `PREDICTIONS_JSON` for a working client-side team-name filter (verified:
  typing "Arsenal" correctly narrowed 350 rows to 35). No console errors.
- `performance.html`: leads with "Beats baseline on Home/Away distinction, but
  has no signal on match evenness" (Allen's exact framing) before any
  topline-accuracy number. Confusion matrix AND the full precision@k
  diagnostic (mean P(Draw) split by actual class, precision@243, lift over
  random, verdict) are both rendered directly on the page, not just in
  STATUS.md/JSON. Calibration table, why-RF-beat-XGBoost paragraph,
  limitations list, and a Future Work section naming **league-position gap**
  and **expected-goals (xG) proxies** specifically (concrete, not vague).
  Verified via `get_page_text` — every number matches `evaluate.py`'s console
  output exactly (46.4%/42.5%/7.8%/0.246 vs 0.247/25.9% vs 25.6%/etc).
- Also writes plain `docs/predictions.json` and `docs/performance.json`
  exports (inspectable, matching the sibling project's convention) — not what
  the pages actually load from (avoids `file://` CORS issues, same reason as
  the transfer predictor's `index.html`).

Stage 5 is now fully done and verified.

## Next steps (in order)

1. **Stage 6**: `.github/workflows/refresh_data.yml` (mirror
   `pl-transfer-value-predictor`'s weekly-refresh pattern: fetch → build
   features → sanity-check → retrain final model → predict → export → commit
   docs/data if changed), `README.md` (architecture tree, key decisions +
   why, known limitations, how to run, automatic updates), then confirm with
   Allen before creating the GitHub repo and pushing (repo creation + first
   push is a one-way public action, needs explicit go-ahead per the plan's
   verification section).

## File map (what exists so far)

```
epl-match-outcome-predictor/
├── STATUS.md                          <- this file
├── .env / .env.example / .gitignore / requirements.txt
├── data/
│   ├── raw/
│   │   ├── historical_matches.csv     (1,140 rows, 2023-24 to 2025-26)
│   │   ├── current_season_played.csv  (30 rows, 2026-27 so far — context only)
│   │   ├── upcoming_fixtures.csv      (350 rows, scheduled)
│   │   ├── *_raw.json                 (cached raw API responses)
│   │   └── collection_errors.log
│   └── processed/
│       └── matches_dataset.csv        (1,140 rows, 16 engineered features, sanity-checked)
├── data_collection/
│   ├── http_utils.py
│   ├── fetch_historical_results.py
│   ├── fetch_upcoming_fixtures.py
│   ├── build_features.py
│   └── sanity_check.py
├── modeling/
│   ├── train_models.py                (RF vs XGBoost CV comparison, done)
│   ├── train_final_model.py           (RF retrained on all data, done)
│   ├── evaluate.py                    (Stage 4, done — see findings above)
│   ├── predict.py                     (Stage 5, done — scores upcoming fixtures)
│   └── artifacts/
│       ├── cv_comparison.json         (full CV metrics both models)
│       ├── feature_columns.pkl        (16-column feature list)
│       ├── random_forest_final.pkl    (production model)
│       ├── imputer.pkl                (median imputer, fit on full data)
│       └── evaluation_summary.json    (Stage 4 findings, structured for Stage 5 reuse)
├── data/processed/predictions.csv     (350 upcoming fixtures scored, Stage 5)
├── docs/                              (Stage 5, done — GitHub Pages root)
│   ├── export_for_web.py
│   ├── index_template.html / index.html
│   ├── performance_template.html / performance.html
│   └── predictions.json / performance.json   (plain exports, not what the pages load from)
└── .github/workflows/                 <- empty, Stage 6
```
