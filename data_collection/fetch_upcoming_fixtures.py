"""Collect the current EPL season's fixtures from football-data.org (v4):
both the scheduled (unplayed) matches to predict, and the already-played
matches from the same in-progress season.

Same API/key/rate-limit pattern as fetch_historical_results.py and the
transfer value predictor's fetch_player_stats.py (free tier: 10
requests/minute, X-Auth-Token header).

Since both this script and fetch_historical_results.py pull from
football-data.org, team names are consistent between "historical" and
"upcoming" data (both use the API's shortName) - no cross-source name
reconciliation is needed in build_features.py.

The current season's played matches are deliberately kept separate from
historical_matches.csv: per Allen's instruction, they're used only to seed
rolling-form features going into upcoming-fixture predictions, never as
labeled training/CV rows (the model shouldn't be evaluated on matches from
the same season it's trying to predict the rest of).

Output:
    data/raw/upcoming_fixtures.csv    columns: match_id, utc_date, matchday, home_team, away_team
    data/raw/current_season_played.csv  same shape as historical_matches.csv (see fetch_historical_results.py)
Raw API responses are cached under data/raw/ for reproducibility/debugging.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from http_utils import RateLimiter, get_json, log_error

BASE_URL = "https://api.football-data.org/v4"
COMPETITION_CODE = "PL"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RATE_LIMITER = RateLimiter(min_interval=6.5)


def load_api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print(
            "ERROR: FOOTBALL_DATA_API_KEY is not set.\n"
            "Get a free key at https://www.football-data.org/client/register "
            "and add it to .env as FOOTBALL_DATA_API_KEY=<your key>.",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def current_season_start_year() -> int:
    """EPL seasons run Aug-May. If we're in Jan-Jun, the season started last calendar year."""
    from datetime import date

    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


def result_letter(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def fetch_matches(session, headers, status: str, context: str) -> list[dict]:
    data = get_json(
        session,
        f"{BASE_URL}/competitions/{COMPETITION_CODE}/matches",
        RATE_LIMITER,
        context=context,
        headers=headers,
        params={"status": status},
    )
    if data is None:
        print(f"FATAL: could not fetch matches ({status}). See collection_errors.log.")
        sys.exit(1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / f"{status.lower()}_matches_raw.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    matches = data.get("matches", [])
    if not matches:
        log_error(context, f"response contained no {status} matches")
    return matches


def scheduled_matches_to_rows(matches: list[dict]) -> list[dict]:
    rows = []
    for m in matches:
        home = m.get("homeTeam", {})
        away = m.get("awayTeam", {})
        if not home.get("name") or not away.get("name"):
            log_error("scheduled_matches_to_rows", f"match missing team name: {m.get('id')}")
            continue
        rows.append(
            {
                "match_id": m.get("id"),
                "utc_date": m.get("utcDate"),
                "matchday": m.get("matchday"),
                "home_team": home.get("shortName") or home.get("name"),
                "away_team": away.get("shortName") or away.get("name"),
            }
        )
    return rows


def finished_matches_to_rows(matches: list[dict], season_start_year: int) -> list[dict]:
    rows = []
    for m in matches:
        home = m.get("homeTeam", {})
        away = m.get("awayTeam", {})
        full_time = m.get("score", {}).get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")
        if not home.get("name") or not away.get("name") or home_goals is None or away_goals is None:
            log_error("finished_matches_to_rows", f"match missing required fields: {m.get('id')}")
            continue
        referees = m.get("referees") or []
        rows.append(
            {
                "match_id": m.get("id"),
                "utc_date": m.get("utcDate"),
                "season": f"{season_start_year}-{(season_start_year + 1) % 100:02d}",
                "matchday": m.get("matchday"),
                "home_team": home.get("shortName") or home.get("name"),
                "away_team": away.get("shortName") or away.get("name"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result_letter(home_goals, away_goals),
                "referee": referees[0]["name"] if referees else None,
            }
        )
    return rows


def main():
    api_key = load_api_key()
    headers = {"X-Auth-Token": api_key}
    session = requests.Session()
    season_start_year = current_season_start_year()

    print("Fetching scheduled Premier League fixtures...")
    scheduled = fetch_matches(session, headers, "SCHEDULED", "competitions/PL/matches?status=SCHEDULED")
    print(f"  {len(scheduled)} scheduled fixtures found.")

    upcoming_df = pd.DataFrame(
        scheduled_matches_to_rows(scheduled),
        columns=["match_id", "utc_date", "matchday", "home_team", "away_team"],
    ).sort_values("utc_date").reset_index(drop=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    upcoming_path = RAW_DIR / "upcoming_fixtures.csv"
    upcoming_df.to_csv(upcoming_path, index=False)
    print(f"  Saved {len(upcoming_df)} upcoming fixtures to {upcoming_path}")
    if not upcoming_df.empty:
        row = upcoming_df.iloc[0]
        print(f"  Next fixture: {row['home_team']} vs {row['away_team']} on {row['utc_date']}")

    print(f"\nFetching played matches so far this season ({season_start_year}-{(season_start_year + 1) % 100:02d})...")
    finished = fetch_matches(session, headers, "FINISHED", "competitions/PL/matches?status=FINISHED (current season)")
    print(f"  {len(finished)} played matches found.")

    played_df = pd.DataFrame(
        finished_matches_to_rows(finished, season_start_year),
        columns=[
            "match_id", "utc_date", "season", "matchday", "home_team", "away_team",
            "home_goals", "away_goals", "result", "referee",
        ],
    ).sort_values("utc_date").reset_index(drop=True)
    played_path = RAW_DIR / "current_season_played.csv"
    played_df.to_csv(played_path, index=False)
    print(f"  Saved {len(played_df)} played matches to {played_path}")
    print("  (context for rolling-form features only - not used as training/CV rows)")


if __name__ == "__main__":
    main()
