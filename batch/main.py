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

from dotenv import load_dotenv

from worster_underwood_cfb import get_college_football_games
from batch.model import build_all_outputs
from batch.sheets import write_to_sheets

# Uncomment when database.py is implemented:
# from batch.database import write_to_database


def run(year: int | None = None, force_refresh: bool = False) -> None:
    load_dotenv()

    if year is None:
        year = datetime.datetime.now().year

    print(f"=== Worster-Underwood CFB | {year} season ===")

    print("Fetching game data...")
    df, ly_df = get_college_football_games(year, force_refresh=force_refresh)
    print(f"  {year}: {len(df)} games | {year - 1}: {len(ly_df)} games")

    print("Running models...")
    underwood, worster, upcoming = build_all_outputs(df, ly_df)
    print(f"  Underwood: {len(underwood)} FBS teams ranked")
    print(f"  Worster:   {len(worster)} FBS teams ranked")
    print(f"  Upcoming:  {len(upcoming)} unplayed games")

    print("Writing to Google Sheets...")
    write_to_sheets(underwood, worster, upcoming)

    # Uncomment when database.py is implemented:
    # print("Writing to database...")
    # write_to_database(underwood, worster, upcoming)

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
