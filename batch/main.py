"""
batch/main.py — run the model and push results

Usage:
    python -m batch.main                   # current year, use cache
    python -m batch.main --year 2025       # specific year
    python -m batch.main --force-refresh   # bypass cache, re-fetch from API
"""
from __future__ import annotations

import argparse
import datetime
import os

from dotenv import load_dotenv

from worster_underwood_cfb import get_college_football_games
from batch.model import build_all_outputs
from batch.sheets import write_to_sheets

from batch.database import write_to_database


def run(year: int | None = None, force_refresh: bool = False) -> None:
    load_dotenv()

    if year is None:
        # The season is named for the year it starts in, but bowls and the CFP
        # run into January — so before June, "current season" is last year's.
        now = datetime.datetime.now()
        year = now.year if now.month >= 6 else now.year - 1

    print(f"=== Worster-Underwood CFB | {year} season ===")

    print("Fetching game data...")
    df, ly_df = get_college_football_games(year, force_refresh=force_refresh)
    print(f"  {year}: {len(df)} games | {year - 1}: {len(ly_df)} games")

    print("Running models...")
    underwood, worster, combined, upcoming = build_all_outputs(df, ly_df)
    print(f"  Underwood: {len(underwood)} FBS teams ranked")
    print(f"  Worster:   {len(worster)} FBS teams ranked")
    print(f"  Combined:  {len(combined)} FBS teams ranked")
    print(f"  Upcoming:  {len(upcoming)} unplayed games")

    print("Writing to Google Sheets...")
    write_to_sheets(underwood, worster, combined, upcoming)

    if os.getenv("DATABASE_URL"):
        print("Writing to database...")
        write_to_database(underwood, worster, combined, upcoming, season=year)
    else:
        print("DATABASE_URL not set — skipping database write.")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Worster-Underwood CFB model and push results."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Season year (default: current calendar year)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass the data cache and re-fetch from the CFBD API",
    )
    args = parser.parse_args()
    run(year=args.year, force_refresh=args.force_refresh)
