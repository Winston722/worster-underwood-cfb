"""
batch/database.py — Postgres writer

Writes the three pipeline outputs to the Postgres database the website reads
(underwood_rankings, worster_rankings, upcoming_games). Each row is stamped
with (season, run_date); rewriting the same (season, run_date) replaces that
run's rows, so the daily job is idempotent while history is preserved.

Configuration:
    DATABASE_URL — Postgres connection string (e.g. a Neon URL).

Not yet written here (still produced by the spreadsheet layer):
    - worster adjusted_rating (depth-decay gap) — written as NULL
    - combined_rankings (wu_rating / disagreement)
    - upcoming talent / competitive indices — written as NULL
The website renders NULL indices as "—", so these are safe to leave empty
until that logic is ported.
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

_TABLES = ("underwood_rankings", "worster_rankings", "upcoming_games")


def _adjusted_rating(rating: pd.Series) -> pd.Series:
    """Scale raw ratings linearly onto [-30, +30] across the ranked teams."""
    r = pd.to_numeric(rating)
    span = r.max() - r.min()
    if span == 0:
        return pd.Series(0.0, index=r.index)
    return -30 + 60 * (r - r.min()) / span


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
    adjusted = _adjusted_rating(df["Rating"])
    return [
        (season, run_date, int(r["Rank"]), str(r["Team"]),
         round(float(adjusted[i]), 2), float(r["Rating"]), float(r["Std Dev"]))
        for i, r in df.iterrows()
    ]


def _worster_rows(df: pd.DataFrame, season: int, run_date: datetime.date) -> list[tuple]:
    return [
        (season, run_date, int(r["Rank"]), str(r["team"]),
         None,  # adjusted_rating: depth-decay logic not ported yet
         int(r["wins"]), int(r["losses"]))
        for _, r in df.iterrows()
    ]


def _upcoming_rows(df: pd.DataFrame, season: int, run_date: datetime.date) -> list[tuple]:
    return [
        (season, run_date, int(r["id"]), int(r["week"]),
         str(r["homeTeam"]), str(r["awayTeam"]),
         _to_utc_timestamp(r["startDate"]), _to_bool(r["neutralSite"]),
         None, None)  # talent / competitive: computed downstream today
        for _, r in df.iterrows()
    ]


def write_to_database(
    underwood: pd.DataFrame,
    worster: pd.DataFrame,
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
