"""
batch/database.py — Postgres database writer (Vercel Postgres / Neon)

Authentication:
    Set DATABASE_URL in your .env or environment. The value comes from the
    Vercel dashboard under Storage → your database → .env.local tab
    (use the POSTGRES_URL value).

Schema is created automatically on first run via create_tables(). Each daily
run upserts into the four tables, so historical rows are preserved and the
latest run for a given (season, run_date) always reflects the most recent data.

Tables:
    underwood_rankings  — Underwood power ratings
    worster_rankings    — Worster résumé rankings (top-level columns only)
    combined_rankings   — WU ensemble ratings + disagreement scores
    upcoming_games      — Unplayed games with Talent and Competitive indices
"""
from __future__ import annotations

import datetime
import os

import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _get_connection():
    """Open and return a psycopg2 connection using DATABASE_URL."""
    import psycopg2  # type: ignore

    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env or environment. "
            "Find it in the Vercel dashboard under Storage → your database → .env.local "
            "(use the POSTGRES_URL value)."
        )
    return psycopg2.connect(url)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS underwood_rankings (
    id              SERIAL PRIMARY KEY,
    season          SMALLINT    NOT NULL,
    run_date        DATE        NOT NULL,
    rank            SMALLINT    NOT NULL,
    team            TEXT        NOT NULL,
    adjusted_rating NUMERIC(6,2),
    rating          NUMERIC(8,4),
    std_dev         NUMERIC(6,2),
    UNIQUE (season, run_date, team)
);

CREATE TABLE IF NOT EXISTS worster_rankings (
    id              SERIAL PRIMARY KEY,
    season          SMALLINT    NOT NULL,
    run_date        DATE        NOT NULL,
    rank            SMALLINT    NOT NULL,
    team            TEXT        NOT NULL,
    adjusted_rating NUMERIC(6,2),
    wins            SMALLINT,
    losses          SMALLINT,
    UNIQUE (season, run_date, team)
);

CREATE TABLE IF NOT EXISTS combined_rankings (
    id               SERIAL PRIMARY KEY,
    season           SMALLINT    NOT NULL,
    run_date         DATE        NOT NULL,
    rank             SMALLINT    NOT NULL,
    team             TEXT        NOT NULL,
    wu_rating        NUMERIC(6,2),
    underwood_rating NUMERIC(6,2),
    worster_rating   NUMERIC(6,2),
    disagreement     NUMERIC(6,2),
    UNIQUE (season, run_date, team)
);

CREATE TABLE IF NOT EXISTS upcoming_games (
    id           SERIAL PRIMARY KEY,
    season       SMALLINT    NOT NULL,
    run_date     DATE        NOT NULL,
    game_id      BIGINT      NOT NULL,
    week         SMALLINT,
    home_team    TEXT,
    away_team    TEXT,
    start_date   TEXT,
    neutral_site BOOLEAN,
    talent       NUMERIC(5,3),
    competitive  NUMERIC(5,3),
    UNIQUE (season, run_date, game_id)
);
"""


def create_tables() -> None:
    """
    Create all tables if they don't already exist. Safe to call on every run.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLES_SQL)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def _upsert(conn, table: str, rows: list[tuple], columns: list[str], conflict_cols: list[str]) -> int:
    """
    Bulk-upsert rows into a table. Returns the number of rows affected.
    Uses INSERT ... ON CONFLICT (...) DO UPDATE so existing rows are refreshed.
    """
    if not rows:
        return 0

    import psycopg2.extras  # type: ignore

    col_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    conflict = ", ".join(conflict_cols)
    update_set = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in columns
        if c not in conflict_cols
    )

    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {update_set}"
    )

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_to_database(
    underwood: pd.DataFrame,
    worster: pd.DataFrame,
    combined: pd.DataFrame,
    upcoming: pd.DataFrame,
    season: int | None = None,
    run_date: datetime.date | None = None,
) -> None:
    """
    Upsert all four output DataFrames into the production database.

    Args:
        underwood: Formatted Underwood power ratings.
        worster:   Formatted Worster résumé rankings.
        combined:  WU ensemble ratings and disagreement scores.
        upcoming:  Upcoming unplayed games with Talent/Competitive indices.
        season:    Season year (defaults to current calendar year).
        run_date:  Date of this run (defaults to today).
    """
    if season is None:
        season = datetime.datetime.now().year
    if run_date is None:
        run_date = datetime.date.today()

    create_tables()
    conn = _get_connection()
    try:
        # --- underwood_rankings ---
        u_rows = [
            (season, run_date, int(r["Rank"]), r["Team"],
             float(r["Adjusted Rating"]), float(r["Rating"]), float(r["Std Dev"]))
            for _, r in underwood.iterrows()
        ]
        n_u = _upsert(conn, "underwood_rankings", u_rows,
                      ["season", "run_date", "rank", "team", "adjusted_rating", "rating", "std_dev"],
                      ["season", "run_date", "team"])

        # --- worster_rankings ---
        w_rows = [
            (season, run_date, int(r["Rank"]), r["team"],
             float(r["Adjusted Rating"]), int(r["wins"]), int(r["losses"]))
            for _, r in worster.iterrows()
        ]
        n_w = _upsert(conn, "worster_rankings", w_rows,
                      ["season", "run_date", "rank", "team", "adjusted_rating", "wins", "losses"],
                      ["season", "run_date", "team"])

        # --- combined_rankings ---
        c_rows = [
            (season, run_date, int(r["Rank"]), r["Team"],
             float(r["WU Rating"]), float(r["Underwood Rating"]),
             float(r["Worster Rating"]), float(r["Disagreement"]))
            for _, r in combined.iterrows()
        ]
        n_c = _upsert(conn, "combined_rankings", c_rows,
                      ["season", "run_date", "rank", "team", "wu_rating",
                       "underwood_rating", "worster_rating", "disagreement"],
                      ["season", "run_date", "team"])

        # --- upcoming_games ---
        g_rows = [
            (season, run_date, int(r["id"]), r.get("week"), r["homeTeam"], r["awayTeam"],
             r.get("startDate"), bool(r.get("neutralSite", False)),
             None if pd.isna(r["Talent"]) else float(r["Talent"]),
             None if pd.isna(r["Competitive"]) else float(r["Competitive"]))
            for _, r in upcoming.iterrows()
        ]
        n_g = _upsert(conn, "upcoming_games", g_rows,
                      ["season", "run_date", "game_id", "week", "home_team", "away_team",
                       "start_date", "neutral_site", "talent", "competitive"],
                      ["season", "run_date", "game_id"])

        conn.commit()
        print(f"  underwood_rankings: {n_u} rows upserted")
        print(f"  worster_rankings:   {n_w} rows upserted")
        print(f"  combined_rankings:  {n_c} rows upserted")
        print(f"  upcoming_games:     {n_g} rows upserted")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
