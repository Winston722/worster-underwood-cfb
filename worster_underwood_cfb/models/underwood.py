from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

__all__ = ["get_initial", "get_rating", "get_ratings", "get_error", "combined"]


def get_initial(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Run a single weighted least-squares regression across all games simultaneously.

    Each game is a row in a sparse design matrix: winner column = +1, loser = -1.
    The target is the HFA-adjusted margin. The solution vector is an initial rating
    for every team (minimum-norm solution via scipy lsqr).

    Returns the schedule DataFrame with 'coefs_winner' and 'coefs_loser' columns
    appended so downstream per-team refinement can look up opponent ratings cheaply.
    """
    schedule = schedule.copy()  # don't mutate the caller's DataFrame

    extras = schedule[["hfa_margin", "weight"]]
    transform = schedule.drop(["hfa_margin", "weight"], axis=1)

    teams = sorted(set(transform["winner"].unique()).union(transform["loser"].unique()))
    n = len(transform)
    m = len(teams)
    x = lil_matrix((n, m), dtype=int)

    team_indices = {team: idx for idx, team in enumerate(teams)}
    winners = transform["winner"].map(team_indices).values
    losers = transform["loser"].map(team_indices).values

    x[np.arange(n), winners] = 1
    x[np.arange(n), losers] = -1

    y = extras["hfa_margin"].to_numpy()
    w = extras["weight"].to_numpy()

    xw = x.multiply(np.sqrt(w[:, np.newaxis]))
    yw = y * np.sqrt(w)

    result, istop, itn, *_ = lsqr(xw, yw)
    assert istop in (1, 2, 3, 4), f"lsqr did not converge cleanly (istop={istop})"

    r1_ratings = pd.DataFrame({"teams": teams, "coefs": result})

    schedule = schedule.set_index("winner", drop=False)
    r1_ratings = r1_ratings.set_index("teams", drop=False)
    with_winner = schedule.join(r1_ratings, how="left").set_index("loser", drop=False)

    with_ratings = (
        with_winner
        .join(r1_ratings, how="left", lsuffix="_winner", rsuffix="_loser")
        .drop(["teams_winner", "teams_loser"], axis=1)
        .reset_index(drop=True)
    )
    return with_ratings


def get_rating(subject: str, initial: pd.DataFrame) -> float:
    """
    Refine a single team's rating using the initial global ratings as fixed opponent baselines.

    For each game involving `subject`, the target is:
        y = hfa_margin + opponent_initial_rating
    This is a single-variable WLS fit (x = 1 everywhere), equivalent to a weighted mean of y.
    """
    with_ratings = initial[["winner", "loser", "hfa_margin", "weight", "coefs_winner", "coefs_loser"]]
    subject_mask = (with_ratings["winner"] == subject) | (with_ratings["loser"] == subject)
    subject_data = with_ratings[subject_mask].copy()

    # Flip margin sign so it's always from subject's perspective
    subject_data["hfa_margin"] *= np.where(subject_data["winner"] == subject, 1, -1)
    subject_data.columns = ["team1", "team2", "hfa_margin", "weight", "rating_team1", "rating_team2"]

    subject_data["y"] = subject_data["hfa_margin"] + subject_data["rating_team2"]
    y = subject_data["y"].to_numpy()
    w = subject_data["weight"].to_numpy()

    # Single-variable WLS: x is all 1s, so solution = weighted mean of y
    xw = np.sqrt(w)
    yw = y * np.sqrt(w)
    result, _, _, _ = np.linalg.lstsq(xw[:, np.newaxis], yw, rcond=0.1)
    return float(result[0])


def get_ratings(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Compute refined ratings for all teams in the schedule.

    Returns a DataFrame with columns ['teams', 'ratings'], sorted descending by rating.
    """
    initial = get_initial(schedule)
    teams = sorted(set(schedule["winner"].unique()).union(schedule["loser"].unique()))
    ratings_list = [get_rating(t, initial) for t in teams]
    return (
        pd.DataFrame({"teams": teams, "ratings": ratings_list})
        .sort_values("ratings", ascending=False)
        .reset_index(drop=True)
    )


def get_error(schedule: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team prediction error with Bayesian shrinkage.

    For each team, calculates weighted RMSE of (predicted_margin - actual_margin),
    then applies shrinkage toward a prior of ~6pt RMSE:

        pseudo_sd = (rmse * games + 6 * 22) / (games + 22)

    The pseudo-count of 22 ≈ a full FBS regular season. Teams with fewer games
    are pulled toward the prior; full-schedule teams are barely affected.

    Returns a DataFrame with columns ['team', 'pseudo_sd'].
    """
    ratings = ratings.copy()  # don't mutate the caller's ratings DataFrame
    error_schedule = schedule.drop(["week"], axis=1)

    error_schedule = error_schedule.set_index("winner", drop=False)
    ratings = ratings.set_index("teams", drop=False).sort_values(by=["ratings"], ascending=False)
    with_winner = error_schedule.join(ratings, how="left").set_index("loser", drop=False)

    with_ratings = (
        with_winner
        .join(ratings, how="left", lsuffix="_winner", rsuffix="_loser")
        .drop(["teams_winner", "teams_loser"], axis=1)
        .reset_index(drop=True)
    )
    with_ratings["error"] = (
        with_ratings["hfa_margin"]
        - (with_ratings["ratings_winner"] - with_ratings["ratings_loser"])
    ) ** 2
    with_ratings.drop(["hfa_margin", "ratings_winner", "ratings_loser"], axis=1, inplace=True)

    with_ratings2 = with_ratings.copy()
    with_ratings.columns = ["team1", "team2", "weight", "error"]
    with_ratings2.columns = ["team2", "team1", "weight", "error"]

    error_set = pd.concat([with_ratings, with_ratings2], ignore_index=True).drop(["team2"], axis=1)

    error_sum = pd.DataFrame(
        error_set.groupby("team1").apply(
            lambda x: (x["weight"] * x["error"]).sum(), include_groups=False
        )
    )
    error_count = error_set.drop(columns="weight").groupby("team1").count()

    error_total = error_sum.join(error_count, lsuffix="r", rsuffix="l").reset_index()
    error_total.columns = ["team", "error", "games"]
    error_total["rmse"] = (error_total["error"] / error_total["games"]) ** 0.5

    # Bayesian shrinkage: prior of ~6pt RMSE with pseudo-count of 22 games (≈ full FBS season).
    # Teams with fewer games get pulled toward the prior; full-schedule teams are barely affected.
    error_total["pseudo_sd"] = (
        (error_total["rmse"] * error_total["games"]) + 6 * 22
    ) / (error_total["games"] + 22)

    return error_total.drop(["error", "games", "rmse"], axis=1)


def combined(ratings: pd.DataFrame, error: pd.DataFrame) -> pd.DataFrame:
    """
    Join ratings and error into a single output DataFrame.

    Returns columns ['team', 'rating', 'pseudo_sd'], sorted descending by rating.
    """
    result = (
        ratings
        .set_index("teams")
        .join(error.set_index("team"), how="left")
        .reset_index()
    )
    result.columns = ["team", "rating", "pseudo_sd"]
    return result


# ---------------------------------------------------------------------------
# Parameter-optimization helpers (not yet fully implemented)
# ---------------------------------------------------------------------------

def error_hfa(hfa: float, schedule: pd.DataFrame) -> float:
    """
    Return total pseudo_sd for a given HFA value.
    Intended for use with scipy.optimize to find the optimal HFA constant.
    TODO: wire up to prepare_schedule so hfa is applied before fitting.
    """
    raise NotImplementedError("error_hfa optimization not yet implemented")


def error_decay(decay: float, hfa: float, schedule: pd.DataFrame) -> float:
    """
    Return total pseudo_sd for a given decay value.
    Intended for use with scipy.optimize to find the optimal time-decay exponent.
    TODO: wire up to add_weight so decay is applied before fitting.
    """
    raise NotImplementedError("error_decay optimization not yet implemented")
