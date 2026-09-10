"""Collect finished EPL match results from football-data.org (v4).

Originally planned against football-data.co.uk's free CSVs, but that site
went down site-wide (503, confirmed via both direct requests and a real
browser) and stayed down for 10+ hours of periodic checks - long enough
that Allen redirected this to football-data.org instead (2026-09-09),
reusing the same API/key already proven working for upcoming fixtures.

Confirmed live against the free tier (see plan doc for the full report):
    - Exactly 3 complete seasons are accessible: 2023-24, 2024-25, 2025-26
      (season params 2023/2024/2025, 380 finished matches each). Anything
      season=2022 or earlier returns 403 - a hard free-tier wall, not
      throttling.
    - Per-match fields are limited to date, teams, full-time score,
      half-time score, matchday, and referee. Shots, shots on target,
      corners, and cards do not exist anywhere in this API's schema (not a
      v1 scoping choice - genuinely absent), so build_features.py can only
      derive form/goals-based features, not shot-based ones.

The current in-progress season (2026-27) is deliberately NOT fetched here -
its already-played matches are pulled by fetch_upcoming_fixtures.py instead
and used only as recent-form context for predictions, never as a labeled
training row, per Allen's instruction.

Output: data/raw/historical_matches.csv with columns
    match_id, utc_date, season, matchday, home_team, away_team,
    home_goals, away_goals, result, referee
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

# The only seasons confirmed accessible on the free tier as of 2026-09-09.
# season=2022 and earlier return 403 (subscription-restricted), not a
# transient limit - re-check this list if the API's policy ever changes.
HISTORICAL_SEASONS = [2023, 2024, 2025]

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


def result_letter(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def fetch_season(session, headers, season_start_year: int) -> list[dict]:
    data = get_json(
        session,
        f"{BASE_URL}/competitions/{COMPETITION_CODE}/matches",
        RATE_LIMITER,
        context=f"competitions/PL/matches?season={season_start_year}&status=FINISHED",
        headers=headers,
        params={"season": season_start_year, "status": "FINISHED"},
    )
    if data is None:
        print(f"FATAL: could not fetch season {season_start_year}. See collection_errors.log.")
        sys.exit(1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / f"historical_matches_{season_start_year}_raw.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    matches = data.get("matches", [])
    if not matches:
        log_error(f"season {season_start_year}", "response contained no finished matches")
    return matches


def matches_to_rows(matches: list[dict], season_start_year: int) -> list[dict]:
    rows = []
    for m in matches:
        home = m.get("homeTeam", {})
        away = m.get("awayTeam", {})
        full_time = m.get("score", {}).get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")
        if not home.get("name") or not away.get("name") or home_goals is None or away_goals is None:
            log_error(f"season {season_start_year}", f"match missing required fields: {m.get('id')}")
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

    all_rows = []
    for season_start_year in HISTORICAL_SEASONS:
        print(f"Fetching finished matches for {season_start_year}-{(season_start_year + 1) % 100:02d}...")
        matches = fetch_season(session, headers, season_start_year)
        rows = matches_to_rows(matches, season_start_year)
        print(f"  {len(rows)} finished matches collected.")
        all_rows.extend(rows)

    df = pd.DataFrame(
        all_rows,
        columns=[
            "match_id", "utc_date", "season", "matchday", "home_team", "away_team",
            "home_goals", "away_goals", "result", "referee",
        ],
    )
    df = df.sort_values("utc_date").reset_index(drop=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "historical_matches.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} historical matches across {len(HISTORICAL_SEASONS)} seasons to {out_path}")
    print("Result breakdown:")
    print(df["result"].value_counts().rename({"H": "Home win", "D": "Draw", "A": "Away win"}).to_string())


if __name__ == "__main__":
    main()
