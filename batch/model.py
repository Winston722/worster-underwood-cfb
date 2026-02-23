"""
batch/model.py — output formatting

Pure data transformation: runs the models and shapes the results into
clean DataFrames ready to be written anywhere (Google Sheets, a database, etc.).
No I/O happens here.
"""
from __future__ import annotations

import pandas as pd

from worster_underwood_cfb import (
    prepare_schedule,
    add_weight,
    get_ratings,
    get_error,
    combined,
    get_worster,
)


def get_fbs_teams(df: pd.DataFrame) -> set[str]:
    """Return the set of FBS team names from the raw games DataFrame."""
    fbs_home = df[df["homeClassification"] == "fbs"]["homeTeam"]
    fbs_away = df[df["awayClassification"] == "fbs"]["awayTeam"]
    return set(fbs_home).union(set(fbs_away))


def build_underwood_output(df: pd.DataFrame, ly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the Underwood power-rating model and return a ranked, FBS-only DataFrame.

    Columns: Rank, Team, Rating, Std Dev
    """
    fbs_teams = get_fbs_teams(df)

    sched = add_weight(prepare_schedule(df))
    ratings = get_ratings(sched)
    result = combined(ratings, get_error(sched, ratings))

    result = result[result["team"].isin(fbs_teams)].reset_index(drop=True)
    result["rating"] = result["rating"].round(2)
    result["pseudo_sd"] = result["pseudo_sd"].round(2)
    result.insert(0, "Rank", range(1, len(result) + 1))
    result.columns = ["Rank", "Team", "Rating", "Std Dev"]
    return result


def build_worster_output(df: pd.DataFrame, ly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the Worster résumé-ranking model and return a ranked, FBS-only DataFrame.

    Columns: Rank, team, wins, losses, wins_from_1best, wins_from_1worst, ...
    """
    fbs_teams = get_fbs_teams(df)

    result = get_worster(df, ly_df)
    result = result[result["team"].isin(fbs_teams)].reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result


def build_upcoming_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return upcoming (unplayed) games involving at least one FBS team.

    Columns: id, season, week, startDate, neutralSite, homeId, awayId, homeTeam, awayTeam
    Datetime columns are converted to strings for spreadsheet compatibility.
    """
    fbs_teams = get_fbs_teams(df)

    upcoming = df[df["awayPoints"].isna()].copy()
    upcoming = upcoming[
        upcoming["homeTeam"].isin(fbs_teams) | upcoming["awayTeam"].isin(fbs_teams)
    ]
    upcoming = upcoming[
        ["id", "season", "week", "startDate", "neutralSite", "homeId", "awayId", "homeTeam", "awayTeam"]
    ]

    for col in upcoming.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        upcoming[col] = upcoming[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    return upcoming.reset_index(drop=True)


def build_all_outputs(
    df: pd.DataFrame, ly_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run both models and build the upcoming games table.

    Returns:
        (underwood, worster, upcoming)
    """
    underwood = build_underwood_output(df, ly_df)
    worster = build_worster_output(df, ly_df)
    upcoming = build_upcoming_output(df)
    return underwood, worster, upcoming
