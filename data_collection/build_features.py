"""Turn historical_matches.csv into a modeling-ready dataset of engineered,
leak-free features (one row per match).

Feature set (per the plan doc's 2026-09-09 Redirect section - shots/SOT
were dropped entirely since football-data.org doesn't expose them at any
tier; only date/teams/goals/result are available, so every feature below
is derived from those alone):

    - Rolling last-5 form: points, goals scored, goals conceded (any venue,
      per team)
    - Home-advantage indicator: each team's own cumulative win rate in ALL
      of its prior HOME matches (home_team) / AWAY matches (away_team) in
      the dataset - a per-team empirical home-field-strength signal,
      distinct from short-term form
    - Head-to-head: win/draw/loss tally between these exact two teams from
      every prior meeting in the dataset (either venue)

Leak-free by construction: for match N, every feature is computed from the
team-history state as it existed strictly BEFORE match N, then the
histories are updated with match N's actual result. This is a single
chronological pass (sorted by utc_date), not a groupby+shift, specifically
so the exact same replay logic can be reused by modeling/predict.py to
featurize upcoming fixtures from the end-state of history (see
build_team_histories / compute_match_features below - both are imported
directly by predict.py rather than duplicated).

Early-season sparsity: 2023-24 is the first season in our data, so any
team's first few matches that season have fewer than 5 prior matches (or
zero) to roll over - those cells are NaN, not zero, so they're
distinguishable from "genuinely poor recent form." A `*_matches_played`
count column is included for every rolling stat so this is inspectable
rather than silently averaged away. This resolves itself from 2024-25
onward since history carries across season boundaries. A related edge
case: a team relegated and later promoted back has a real gap in top-flight
history (we don't have Championship data) - their rolling stats would look
further back in time than 5 real matches ago. Both are documented as known
limitations in the README, not silently patched over.

Output: data/processed/matches_dataset.csv - the historical_matches.csv
columns plus every engineered feature and the result label. Row count is
guaranteed to equal historical_matches.csv's row count (checked by
sanity_check.py) - no rows are ever dropped for missing history, only
left as NaN.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ROLLING_WINDOW = 5


def build_team_histories():
    """Fresh, empty history state. Returns the four defaultdicts that
    compute_match_features reads and this module's replay loop mutates.
    """
    team_history = defaultdict(list)  # team -> [{"points", "gf", "ga"}, ...] any venue, chronological
    team_home_history = defaultdict(list)  # team -> [1 if won at home else 0, ...] home matches only
    team_away_history = defaultdict(list)  # team -> [1 if won away else 0, ...] away matches only
    h2h_history = defaultdict(list)  # frozenset({team_a, team_b}) -> [(home_team_at_time, result), ...]
    return team_history, team_home_history, team_away_history, h2h_history


def _rolling_form(history: list[dict], window: int = ROLLING_WINDOW) -> dict:
    recent = history[-window:]
    n = len(recent)
    if n == 0:
        return {"form_pts": np.nan, "goals_scored": np.nan, "goals_conceded": np.nan, "matches_played": 0}
    return {
        "form_pts": sum(h["points"] for h in recent),
        "goals_scored": sum(h["gf"] for h in recent),
        "goals_conceded": sum(h["ga"] for h in recent),
        "matches_played": n,
    }


def _win_pct(results: list[int]) -> float:
    if not results:
        return np.nan
    return sum(results) / len(results)


def _h2h_summary(meetings: list[tuple[str, str]], home_team: str, away_team: str) -> dict:
    home_wins = draws = away_wins = 0
    for home_at_time, result in meetings:
        if result == "D":
            draws += 1
            continue
        winner = home_at_time if result == "H" else (away_team if home_at_time == home_team else home_team)
        if winner == home_team:
            home_wins += 1
        else:
            away_wins += 1
    return {
        "h2h_home_wins": home_wins,
        "h2h_draws": draws,
        "h2h_away_wins": away_wins,
        "h2h_meetings": len(meetings),
    }


def compute_match_features(
    home_team: str,
    away_team: str,
    team_history: dict,
    team_home_history: dict,
    team_away_history: dict,
    h2h_history: dict,
) -> dict:
    """Read-only: computes features from the CURRENT state of the history
    dicts, without mutating them. Safe to call for a real historical match
    (before updating with its result) or for an upcoming fixture (after
    replaying all known results).
    """
    home_form = _rolling_form(team_history[home_team])
    away_form = _rolling_form(team_history[away_team])
    h2h_key = frozenset({home_team, away_team})
    h2h = _h2h_summary(h2h_history[h2h_key], home_team, away_team)

    return {
        "home_form_pts_last5": home_form["form_pts"],
        "home_goals_scored_last5": home_form["goals_scored"],
        "home_goals_conceded_last5": home_form["goals_conceded"],
        "home_matches_played_last5": home_form["matches_played"],
        "away_form_pts_last5": away_form["form_pts"],
        "away_goals_scored_last5": away_form["goals_scored"],
        "away_goals_conceded_last5": away_form["goals_conceded"],
        "away_matches_played_last5": away_form["matches_played"],
        "home_team_home_win_pct": _win_pct(team_home_history[home_team]),
        "home_team_home_matches_played": len(team_home_history[home_team]),
        "away_team_away_win_pct": _win_pct(team_away_history[away_team]),
        "away_team_away_matches_played": len(team_away_history[away_team]),
        "h2h_home_wins": h2h["h2h_home_wins"],
        "h2h_draws": h2h["h2h_draws"],
        "h2h_away_wins": h2h["h2h_away_wins"],
        "h2h_meetings": h2h["h2h_meetings"],
    }


def update_histories(
    match: pd.Series,
    team_history: dict,
    team_home_history: dict,
    team_away_history: dict,
    h2h_history: dict,
) -> None:
    """Mutates the history dicts with this match's actual result. Call
    AFTER computing that match's features, never before.
    """
    home_team, away_team, result = match["home_team"], match["away_team"], match["result"]
    home_goals, away_goals = match["home_goals"], match["away_goals"]

    points_home = 3 if result == "H" else (1 if result == "D" else 0)
    points_away = 3 if result == "A" else (1 if result == "D" else 0)
    team_history[home_team].append({"points": points_home, "gf": home_goals, "ga": away_goals})
    team_history[away_team].append({"points": points_away, "gf": away_goals, "ga": home_goals})

    team_home_history[home_team].append(1 if result == "H" else 0)
    team_away_history[away_team].append(1 if result == "A" else 0)

    h2h_key = frozenset({home_team, away_team})
    h2h_history[h2h_key].append((home_team, result))


def replay_and_featurize(matches_df: pd.DataFrame) -> pd.DataFrame:
    """The leak-free chronological pass. matches_df must already be sorted
    by utc_date. Returns matches_df with engineered feature columns added.
    """
    team_history, team_home_history, team_away_history, h2h_history = build_team_histories()

    feature_rows = []
    for _, match in matches_df.iterrows():
        feats = compute_match_features(
            match["home_team"], match["away_team"], team_history, team_home_history, team_away_history, h2h_history
        )
        feature_rows.append(feats)
        update_histories(match, team_history, team_home_history, team_away_history, h2h_history)

    features_df = pd.DataFrame(feature_rows, index=matches_df.index)
    return pd.concat([matches_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)


def main():
    historical_path = RAW_DIR / "historical_matches.csv"
    if not historical_path.exists():
        print(f"FATAL: {historical_path} not found. Run fetch_historical_results.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(historical_path, parse_dates=["utc_date"])
    df = df.sort_values("utc_date").reset_index(drop=True)
    print(f"Loaded {len(df)} historical matches.")

    dataset = replay_and_featurize(df)

    cold_start = dataset["home_matches_played_last5"].eq(0) | dataset["away_matches_played_last5"].eq(0)
    print(f"{cold_start.sum()} rows have at least one side with zero prior matches (early-season cold start).")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "matches_dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"\nSaved {len(dataset)} feature rows to {out_path}")
    print(f"Feature columns: {[c for c in dataset.columns if c not in df.columns]}")


if __name__ == "__main__":
    main()
