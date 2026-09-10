"""
batch/database.py — Postgres writer

Writes the four pipeline outputs to the Postgres database the website reads
(combined_rankings, underwood_rankings, worster_rankings, upcoming_games).
Each row is stamped with (season, run_date); rewriting the same
(season, run_date) replaces that run's rows, so the daily job is idempotent
while history is preserved.

combined_rankings is what the site keys off: getSeasons() reads its distinct
seasons, so a season absent here is invisible to every page.

Configuration:
    DATABASE_URL — Postgres connection string (e.g. a Neon URL).
"""
from __future__ import annotations

import datetime
import os

import pandas as pd

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS underwood_rankings (
        season          integer          NOT NULL,
        run_date        date             NOT NULL,
        rank            integer          NOT NULL,
        team            text             NOT NULL,
        adjusted_rating double precision,
        rating          double precision NOT NULL,
        std_dev         double precision NOT NULL,
        PRIMARY KEY (season, run_date, team)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS worster_rankings (
        season          integer NOT NULL,
        run_date        date    NOT NULL,
        rank            integer NOT NULL,
        team            text    NOT NULL,
        adjusted_rating double precision,
        wins            integer NOT NULL,
        losses          integer NOT NULL,
        PRIMARY KEY (season, run_date, team)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS combined_rankings (
        season           integer NOT NULL,
        run_date         date    NOT NULL,
        rank             integer NOT NULL,
        team             text    NOT NULL,
        wu_rating        double precision,
        underwood_rating double precision,
        worster_rating   double precision,
        disagreement     double precision,
        PRIMARY KEY (season, run_date, team)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS upcoming_games (
        season       integer     NOT NULL,
        run_date     date        NOT NULL,
        game_id      bigint      NOT NULL,
        week         integer     NOT NULL,
        home_team    text        NOT NULL,
        away_team    text        NOT NULL,
        start_date   timestamptz,
        neutral_site boolean     NOT NULL,
        talent       double precision,
        competitive  double precision,
        PRIMARY KEY (season, run_date, game_id)
    )
    """,
]

_TABLES = (
    "underwood_rankings",
    "worster_rankings",
    "combined_rankings",
    "upcoming_games",
)


def _num(value):
    """Float, or None for missing values — Postgres wants NULL, not NaN."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _to_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _to_utc_timestamp(value):
    """Parse a start date; naive values are CFBD UTC wall times, so localize."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


def _underwood_rows(df: pd.DataFrame, season: int, run_date: datetime.date) -> list[tuple]:
    return [
        (season, run_date, int(r["Rank"]), str(r["Team"]),
         _num(r["Adjusted Rating"]), float(r["Rating"]), float(r["Std Dev"]))
        for _, r in df.iterrows()
    ]


def _worster_rows(df: pd.DataFrame, season: int, run_date: datetime.date) -> list[tuple]:
    return [
        (season, run_date, int(r["Rank"]), str(r["team"]),
         _num(r["Adjusted Rating"]),
         int(r["wins"]), int(r["losses"]))
        for _, r in df.iterrows()
    ]


def _combined_rows(df: pd.DataFrame, season: int, run_date: datetime.date) -> list[tuple]:
    return [
        (season, run_date, int(r["Rank"]), str(r["Team"]),
         _num(r["WU Rating"]), _num(r["Underwood Rating"]),
         _num(r["Worster Rating"]), _num(r["Disagreement"]))
        for _, r in df.iterrows()
    ]


def _upcoming_rows(df: pd.DataFrame, season: int, run_date: datetime.date) -> list[tuple]:
    return [
        (season, run_date, int(r["id"]), int(r["week"]),
         str(r["homeTeam"]), str(r["awayTeam"]),
         _to_utc_timestamp(r["startDate"]), _to_bool(r["neutralSite"]),
         _num(r["Talent"]), _num(r["Competitive"]))
        for _, r in df.iterrows()
    ]


def write_to_database(
    underwood: pd.DataFrame,
    worster: pd.DataFrame,
    combined: pd.DataFrame,
    upcoming: pd.DataFrame,
    season: int,
    run_date: datetime.date | None = None,
    database_url: str | None = None,
) -> None:
    """
    Write rankings and upcoming games to the production database.

    Args:
        underwood: Formatted Underwood power ratings (Rank, Team, Rating, Std Dev).
        worster:   Formatted Worster résumé rankings (Rank, team, wins, losses, ...).
        combined:  WU ensemble ratings (Rank, Team, WU Rating, ..., Disagreement).
        upcoming:  Upcoming unplayed games (id, season, week, startDate, ...).
        season:    Season year the rows belong to.
        run_date:  Defaults to today; rows for the same (season, run_date) are replaced.
        database_url: Defaults to the DATABASE_URL environment variable.
    """
    import psycopg  # imported here so the sheets-only path needs no DB driver

    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set in the environment")
    if run_date is None:
        run_date = datetime.date.today()

    rows = {
        "underwood_rankings": _underwood_rows(underwood, season, run_date),
        "worster_rankings": _worster_rows(worster, season, run_date),
        "combined_rankings": _combined_rows(combined, season, run_date),
        "upcoming_games": _upcoming_rows(upcoming, season, run_date),
    }
    inserts = {
        "underwood_rankings": """
            INSERT INTO underwood_rankings
                (season, run_date, rank, team, adjusted_rating, rating, std_dev)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        "worster_rankings": """
            INSERT INTO worster_rankings
                (season, run_date, rank, team, adjusted_rating, wins, losses)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        "combined_rankings": """
            INSERT INTO combined_rankings
                (season, run_date, rank, team, wu_rating,
                 underwood_rating, worster_rating, disagreement)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        "upcoming_games": """
            INSERT INTO upcoming_games
                (season, run_date, game_id, week, home_team, away_team,
                 start_date, neutral_site, talent, competitive)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
    }

    # One transaction: either the whole run lands or none of it does.
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            for ddl in _DDL:
                cur.execute(ddl)
            for table in _TABLES:
                cur.execute(
                    f"DELETE FROM {table} WHERE season = %s AND run_date = %s",
                    (season, run_date),
                )
                if rows[table]:
                    cur.executemany(inserts[table], rows[table])
