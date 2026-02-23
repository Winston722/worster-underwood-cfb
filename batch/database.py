"""
batch/database.py — database writer (not yet implemented)

Intended as the long-term replacement / complement to sheets.py once a
production database is chosen (e.g. Supabase/Postgres).

The interface intentionally mirrors write_to_sheets() so that main.py
can call both with the same arguments.

Design notes for when this gets built:
    - Store the connection string in DATABASE_URL env var
    - Use psycopg2 or SQLAlchemy for Postgres / Supabase
    - Prefer upsert over replace so historical rows are preserved
    - Consider separate tables: underwood_rankings, worster_rankings, upcoming_games
    - Add a `run_date` column so the website can show when rankings were last updated
"""
from __future__ import annotations

import pandas as pd


def write_to_database(
    underwood: pd.DataFrame,
    worster: pd.DataFrame,
    upcoming: pd.DataFrame,
) -> None:
    """
    Write rankings and upcoming games to the production database.

    Args:
        underwood: Formatted Underwood power ratings.
        worster:   Formatted Worster résumé rankings.
        upcoming:  Upcoming unplayed games.
    """
    raise NotImplementedError(
        "Database writer not yet implemented. See batch/database.py for design notes."
    )
