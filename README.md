# EPL Match Outcome Predictor

Predicts Premier League match outcomes (Home Win / Draw / Away Win) from historical results,
using a Random Forest trained on football-data.org data. Built as a small end-to-end data
science project: data collection, leak-free feature engineering, model comparison, and an
honest evaluation - including a diagnostic that pins down *why* the model struggles with
draws, not just that it does.

**The headline finding, up front: this model beats the baseline on distinguishing Home Win
from Away Win, but has no real signal on match evenness.** It reaches 46.4% accuracy against
a 42.5% "always predict Home Win" baseline - a real but thin lift - and catches only 7.8% of
actual draws. A precision@k diagnostic (rank fixtures by predicted P(Draw), check how many of
the most-confident-Draw predictions were actually draws) confirms this isn't a fixable
threshold quirk: P(Draw) for actual draws (0.246) is statistically indistinguishable from
P(Draw) for non-draws (0.247), and ranking by it beats random guessing by only +0.3%. The
current feature set (goals/form/head-to-head - no shots data, see below) genuinely doesn't
separate an evenly-matched game from a lopsided one. Full breakdown, including the confusion
matrix and the diagnostic itself, is on the live [model performance page](https://allenb34.github.io/epl-match-outcome-predictor/performance.html).

This README documents what was actually built and why, including a mid-build data-source
outage that forced a real architecture change. It is not a sales pitch for the model's
accuracy.

**Live site**: https://allenb34.github.io/epl-match-outcome-predictor/

**Tech stack**: `requests` (API calls), `pandas` (data wrangling), `scikit-learn`
(RandomForestClassifier, cross-validation, imputation), `xgboost` (comparison model),
`python-dotenv` (API key loading), `joblib` (model persistence), `truststore` (see Setup).
Dashboard is static HTML/CSS/vanilla JS, no framework.

## Architecture

```
data_collection/
  http_utils.py               -> shared rate limiter, retry/backoff, error logging
  fetch_historical_results.py -> football-data.org: 3 complete seasons of finished matches
  fetch_upcoming_fixtures.py  -> football-data.org: scheduled fixtures + current season's played matches
  build_features.py           -> leak-free chronological replay -> engineered feature dataset
  sanity_check.py             -> row count, result distribution, cold-start integrity checks

modeling/
  train_models.py             -> Random Forest vs XGBoost, time-based (expanding-window) CV comparison
  train_final_model.py        -> retrains the winner (Random Forest) on all data
  evaluate.py                 -> accuracy vs baseline, confusion matrix, draw-recall root-cause diagnostic, calibration
  predict.py                  -> scores upcoming fixtures using the final model
  artifacts/                  -> trained model, imputer, CV/evaluation summaries (JSON, inspectable)

data/
  raw/                        -> raw API responses + per-source CSVs, plus collection_errors.log
  processed/                  -> matches_dataset.csv (training data), predictions.csv (scored fixtures)

docs/                          -> GitHub Pages root (served from /docs on main)
  export_for_web.py           -> renders index.html + performance.html from real data, zero JS recomputation
  index.html                  -> upcoming fixtures + predictions, with a team-name filter
  performance.html            -> confusion matrix, precision@k diagnostic, calibration, limitations, future work
```

Pipeline: **fetch historical results + upcoming fixtures** (football-data.org) → **engineer
features** (leak-free chronological replay) → **compare models** (time-based CV) → **retrain
the winner on everything** → **score upcoming fixtures** → **export the static site**.

## Key decisions and why

**The historical data source changed mid-build, and the new source's limits reshaped the
whole feature set.** The original plan used football-data.co.uk's free CSVs. That site went
down site-wide (confirmed via both direct requests and a real browser - the same "temporarily
unavailable" nginx page either way) and stayed down for 10+ hours of periodic checks, long
enough to redirect to football-data.org instead - the same API already used for fixtures.
That source turned out to have real limits, discovered live rather than assumed:
- **Free tier caps historical access at exactly 3 complete seasons** (2023-24, 2024-25,
  2025-26 - `season=2022` and earlier returns `403 Forbidden`, a hard wall, not throttling).
  1,140 total historical matches.
- **Shots, shots on target, corners, and cards don't exist anywhere in this API's schema** -
  not a paid-tier gate, genuinely absent from a raw match object. Feature engineering was
  scoped down to what's actually available: rolling form (points/goals, last 5 matches),
  each team's own home/away win-rate history, and head-to-head record.

**Time-based cross-validation, not a single holdout split.** With only 1,140 rows across 3
seasons, one train/test split would leave a validation set too small and too date-dependent
to trust. `train_models.py` uses 5-fold expanding-window `TimeSeriesSplit` instead - each fold
trains on everything chronologically before it and validates on the next chunk, pooling all
folds' held-out predictions into one combined report. (The earliest ~1/6 of matches are only
ever training data under this scheme, never scored - that's inherent to expanding-window CV.)

**Random Forest beat XGBoost, and not narrowly.** 46.4% accuracy / 1.061 log-loss vs
XGBoost's 41.7% / 1.293 - XGBoost actually underperformed the baseline. With only ~190 rows
per CV fold and three classes, this is exactly the regime where XGBoost's more flexible
boosting tends to overfit relative to Random Forest's more conservative, bagging-averaged
bias. Not evidence XGBoost is a worse algorithm generally - a poor fit for this little data.

**The draw-recall investigation was a specific diagnostic, not just an observation.** Seeing
7.8% draw recall in a confusion matrix only tells you the model rarely *predicts* draw - it
doesn't say whether there's weak signal that argmax is discarding (fixable with threshold
tuning / rebalancing) or no signal at all (needs new features). The precision@k diagnostic
settles it: rank all fixtures by raw P(Draw) regardless of what won the 3-way argmax, take the
top-k (k = number of actual draws), and check the hit rate. Here it's 25.9%, against a 25.6%
base rate - no better than random. That result is what points Future Work at new features
(below) rather than at threshold tuning.

**Current-season matches are context, not training data.** The 2026-27 season's
already-played matches feed rolling-form features for predicting *its own remaining*
fixtures, but are never used as labeled CV/training rows - scoring a model on matches from
the same season it's mid-way through predicting would be a subtle leak.

## Known limitations

State plainly, not undersold:

- **Small sample.** Only 950 of 1,140 matches are ever scored under 5-fold expanding-window
  CV - a few points of accuracy either direction is within noise. This is a consequence of
  football-data.org's free-tier 3-season cap, not a choice.
- **Draws are hard to predict, and it's a real feature-signal gap, not a tunable threshold
  issue** - see the precision@k diagnostic above and on the [performance page](https://allenb34.github.io/epl-match-outcome-predictor/performance.html).
- **No shots/SOT/corners/cards data** - football-data.org's free tier doesn't expose these
  fields at all, so features are limited to goals/form/head-to-head.
- **No player-availability/injury data** - two teams with identical recent form can have very
  different real-world availability (injuries, suspensions, rotation) invisible to this model.
- **Early-season and newly-promoted-team cold start.** A team's first few matches in the
  dataset (2023-24's opening weeks, or a newly promoted team's debut in any season) have
  NaN rolling features, median-imputed - `sanity_check.py` verifies every NaN corresponds to
  a genuine zero-prior-matches case, not a computation bug.
- **Mild overconfidence at the high-confidence end.** The 60%+ predicted-probability bin
  delivers only ~60% actual accuracy in calibration checks - close, but don't read a 60%+
  prediction as more reliable than that.
- **Predictions ignore intra-matchday ordering** - all of a matchday's fixtures are featurized
  against the same end-of-history state, even though a Saturday 12:30 kickoff's result
  technically precedes that same day's 5:30 kickoff. A standard simplifying assumption for
  this kind of walk-forward feature engineering.

## How to run it

**Setup:**
```bash
cd epl-match-outcome-predictor
pip install -r requirements.txt
```
Register a free API key at https://www.football-data.org/client/register, then put it in
`.env` (copy `.env.example` if starting fresh):
```
FOOTBALL_DATA_API_KEY=your_key_here
```
Note: if you hit `CERTIFICATE_VERIFY_FAILED` errors on Windows behind a TLS-inspecting proxy,
that's what `truststore` in `requirements.txt` is for - it's already wired in via
`data_collection/http_utils.py`.

**Run in order:**
```bash
cd data_collection
python fetch_historical_results.py   # -> data/raw/historical_matches.csv (3 seasons, needs API key)
python fetch_upcoming_fixtures.py    # -> data/raw/upcoming_fixtures.csv + current_season_played.csv
python build_features.py             # -> data/processed/matches_dataset.csv
python sanity_check.py               # verifies row counts, distributions, cold-start integrity

cd ../modeling
python train_models.py               # Random Forest vs XGBoost, time-based CV comparison
python evaluate.py                   # accuracy vs baseline, confusion matrix, draw-recall diagnostic
python train_final_model.py          # retrains Random Forest on all data -> random_forest_final.pkl
python predict.py                    # scores upcoming fixtures -> data/processed/predictions.csv

cd ../docs
python export_for_web.py             # -> index.html + performance.html
```
Each `fetch_*` script is rate-limited to football-data.org's free tier (10 requests/minute).

## Automatic updates

The data, model, and static site (`docs/`) refresh automatically every **Friday at 06:00
UTC** via GitHub Actions (`.github/workflows/refresh_data.yml`), ahead of the weekend's match
round. It runs the full pipeline above end to end, then commits and pushes whatever changed in
`data/`, `modeling/artifacts/`, and `docs/` using the repo's built-in `GITHUB_TOKEN`. If the
data is identical to last week's, it skips the commit rather than creating an empty one.

**Safety**: if any pipeline step fails (API down, rate-limited, bad data) or the resulting
dataset has an implausibly low row count, the job fails loudly and stops *before* the commit
step - the live site is never overwritten with broken or partial data. This matters in
practice, not just in theory: football-data.co.uk (the originally planned data source) went
down for 10+ hours during development, which is exactly the class of failure this guard
exists for.

**Triggering it manually**: go to the repo's **Actions** tab → **Weekly data refresh** →
**Run workflow**.

**Adding/rotating the API key secret**: the workflow needs `FOOTBALL_DATA_API_KEY` available
as a GitHub Actions secret (never committed to the repo). To add or update it:
**Settings → Secrets and variables → Actions → New repository secret**, name it exactly
`FOOTBALL_DATA_API_KEY`, and paste in a key from https://www.football-data.org/client/register.

## Future work

Concrete next feature candidates, prioritized by what the precision@k diagnostic actually
found missing - not a generic "collect more features" note:

- **League-position gap.** The current feature set has no notion of "how good is this team
  relative to its opponent right now" beyond short-term (last-5) form. A team's league-table
  position, or points-per-game normalized across the season, relative to its opponent, is a
  direct, cheap-to-compute proxy for exactly the "evenly matched vs. one side is just better"
  signal the diagnostic found absent.
- **Expected-goals (xG) proxies.** Real shot-quality data isn't available from
  football-data.org's free tier, but a rough proxy - goals scored/conceded relative to a
  supplementary shots-on-target source, or a simple Poisson-based expected-goals estimate from
  historical scoring rates - would capture "who's been getting the run of play" in a way raw
  goals-only rolling form can't.
- Re-run the season-range decision once football-data.org's 4th complete season (2026-27)
  is available, and consider whether `HISTORICAL_SEASONS` in `fetch_historical_results.py`
  should become a rolling window rather than a hardcoded 3-season list.
