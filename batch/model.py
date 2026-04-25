"""
batch/model.py — output formatting

Pure data transformation: runs the models and shapes the results into
clean DataFrames ready to be written anywhere (Google Sheets, a database, etc.).
No I/O happens here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from worster_underwood_cfb import (
    prepare_schedule,
    add_weight,
    get_ratings,
    get_error,
    combined,
    get_adjusted_rating,
    get_worster,
    get_worster_rating,
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
    # Compute adjusted rating before rounding raw rating, then reorder columns
    # explicitly so the rename below is unambiguous.
    adjusted = get_adjusted_rating(result["rating"]).round(2)
    result = pd.DataFrame({
        "team": result["team"],
        "adjusted_rating": adjusted,
        "rating": result["rating"].round(2),
        "pseudo_sd": result["pseudo_sd"].round(2),
    })
    result.insert(0, "Rank", range(1, len(result) + 1))
    result.columns = ["Rank", "Team", "Adjusted Rating", "Rating", "Std Dev"]
    return result


def build_worster_output(df: pd.DataFrame, ly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the Worster résumé-ranking model and return a ranked, FBS-only DataFrame.

    Columns: Rank, team, wins, losses, wins_from_1best, wins_from_1worst, ...
    """
    fbs_teams = get_fbs_teams(df)

    result = get_worster(df, ly_df)
    result = result[result["team"].isin(fbs_teams)].reset_index(drop=True)
    result.insert(1, "adjusted_rating", get_worster_rating(result).round(2))
    result.insert(0, "Rank", range(1, len(result) + 1))
    result.rename(columns={"adjusted_rating": "Adjusted Rating"}, inplace=True)
    return result


def build_upcoming_output(df: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    """
    Return upcoming (unplayed) games involving at least one FBS team, with
    game-quality indices derived from WU ratings.

    Talent index:     How good are the two teams on average?
                      1.0 = #1 vs #2 (highest avg WU rating possible)
                      0.0 = #(N-1) vs #N (lowest avg WU rating possible)

    Competitive index: How close are the two teams?
                      1.0 = zero gap (perfectly matched)
                      0.0 = #1 vs #N (maximum possible gap)

    Games where either team has no WU rating (FCS opponents, no games played
    yet) get NaN for both indices rather than being dropped.

    Columns: id, season, week, startDate, neutralSite, homeId, awayId,
             homeTeam, awayTeam, Talent, Competitive
    """
    fbs_teams = get_fbs_teams(df)

    upcoming = df[df["awayPoints"].isna()].copy()
    upcoming = upcoming[
        upcoming["homeTeam"].isin(fbs_teams) | upcoming["awayTeam"].isin(fbs_teams)
    ]
    upcoming = upcoming[
        ["id", "season", "week", "startDate", "neutralSite", "homeId", "awayId", "homeTeam", "awayTeam"]
    ].reset_index(drop=True)

    for col in upcoming.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        upcoming[col] = upcoming[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Compute index bounds from the full set of WU ratings
    wu = combined.set_index("Team")["WU Rating"].sort_values(ascending=False)
    sorted_wu = wu.values
    talent_max = (sorted_wu[0] + sorted_wu[1]) / 2
    talent_min = (sorted_wu[-2] + sorted_wu[-1]) / 2
    gap_max = sorted_wu[0] - sorted_wu[-1]

    home_wu = upcoming["homeTeam"].map(wu)
    away_wu = upcoming["awayTeam"].map(wu)

    avg_wu = (home_wu + away_wu) / 2
    gap_wu = (home_wu - away_wu).abs()

    talent_denom = talent_max - talent_min
    upcoming["Talent"] = (
        ((avg_wu - talent_min) / talent_denom).clip(0, 1).round(3)
        if talent_denom != 0 else np.nan
    )
    upcoming["Competitive"] = (
        (1 - gap_wu / gap_max).clip(0, 1).round(3)
        if gap_max != 0 else np.nan
    )

    return upcoming


def build_combined_output(underwood: pd.DataFrame, worster: pd.DataFrame) -> pd.DataFrame:
    """
    Join the two model outputs and compute ensemble metrics.

    WU Rating:     z-normalize both adjusted ratings, average, rescale to [-30, +30].
    Disagreement:  how much the models disagree, scaled by the team's rank so
                   that disagreement about top teams carries more weight.
                   Formula: |underwood_adj - worster_adj| / log10(wu_rank + 1)

    Columns: Rank, Team, WU Rating, Underwood Rating, Worster Rating, Disagreement
    """
    u = underwood[["Team", "Adjusted Rating"]].rename(
        columns={"Team": "team", "Adjusted Rating": "Underwood Rating"}
    )
    w = worster[["team", "Adjusted Rating"]].rename(
        columns={"Adjusted Rating": "Worster Rating"}
    )

    result = u.merge(w, on="team", how="inner")

    # Z-normalize each model before averaging so differences in distributional
    # shape don't bias the combined rating, then rescale back to [-30, +30].
    u_z = (result["Underwood Rating"] - result["Underwood Rating"].mean()) / result["Underwood Rating"].std()
    w_z = (result["Worster Rating"] - result["Worster Rating"].mean()) / result["Worster Rating"].std()
    result["WU Rating"] = get_adjusted_rating((u_z + w_z) / 2).round(2)

    result = result.sort_values("WU Rating", ascending=False).reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))

    result["Disagreement"] = (
        (result["Underwood Rating"] - result["Worster Rating"]).abs()
        / np.log10(result["Rank"] + 1)
    ).round(2)

    result.rename(columns={"team": "Team"}, inplace=True)
    return result[["Rank", "Team", "WU Rating", "Underwood Rating", "Worster Rating", "Disagreement"]]


def build_all_outputs(
    df: pd.DataFrame, ly_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run both models and build the upcoming games table.

    Returns:
        (underwood, worster, combined, upcoming)
    """
    underwood = build_underwood_output(df, ly_df)
    worster = build_worster_output(df, ly_df)
    combined = build_combined_output(underwood, worster)
    upcoming = build_upcoming_output(df, combined)
    return underwood, worster, combined, upcoming
