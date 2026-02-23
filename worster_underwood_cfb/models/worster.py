from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["get_worster"]


def _prepare_worster_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ['winner', 'loser'] from a raw games DataFrame.
    Worster only needs win/loss outcomes — no margins required.
    """
    rename_map = {
        "home_team": "homeTeam",
        "away_team": "awayTeam",
        "home_points": "homePoints",
        "away_points": "awayPoints",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    raw = (
        df.reindex(columns=["homeTeam", "awayTeam", "homePoints", "awayPoints"])
        .dropna(subset=["homePoints", "awayPoints"])
        .reset_index(drop=True)
    )
    if raw.empty:
        return pd.DataFrame(columns=["winner", "loser"])

    assert not raw[["homeTeam", "awayTeam"]].isna().any().any(), "Null team name(s)."
    hp = pd.to_numeric(raw["homePoints"], errors="raise").to_numpy()
    ap = pd.to_numeric(raw["awayPoints"], errors="raise").to_numpy()
    margin = hp - ap
    # Drop ties: Worster is a pure win/loss system; a tie produces neither a win
    # nor a loss and can't be represented cleanly without fractional records.
    raw = raw[hp != ap].reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame(columns=["winner", "loser"])
    hp = pd.to_numeric(raw["homePoints"], errors="raise").to_numpy()
    ap = pd.to_numeric(raw["awayPoints"], errors="raise").to_numpy()
    margin = hp - ap

    home_win = margin > 0
    winners = np.where(home_win, raw["homeTeam"].to_numpy(), raw["awayTeam"].to_numpy())
    losers = np.where(home_win, raw["awayTeam"].to_numpy(), raw["homeTeam"].to_numpy())
    return pd.DataFrame({"winner": winners, "loser": losers})


def create_team_metrics(schedule: pd.DataFrame, K: int = 15, prefix: str = "") -> pd.DataFrame:
    """
    Build résumé features for every team in the schedule.

    Columns produced (per team):
        {prefix}wins, {prefix}losses,
        {prefix}wins_from_1best  ... {prefix}wins_from_{K}best   (win totals of beaten opponents, best-to-worst)
        {prefix}wins_from_1worst ... {prefix}wins_from_{K}worst  (win totals of opponents who beat you, worst-to-best)

    Args:
        schedule: DataFrame with ['winner', 'loser'] columns.
        K:        Depth of résumé ladder (default 15).
        prefix:   Column prefix, e.g. "ly_" for last-year features.
    """
    assert {"winner", "loser"}.issubset(schedule.columns), "schedule must have ['winner','loser']"
    n_games = len(schedule)
    if n_games == 0:
        cols = ["team", f"{prefix}wins", f"{prefix}losses"]
        for i in range(1, K + 1):
            cols += [f"{prefix}wins_from_{i}best", f"{prefix}wins_from_{i}worst"]
        return pd.DataFrame(columns=cols)

    both = pd.concat([schedule["winner"], schedule["loser"]], ignore_index=True)
    codes, teams = pd.factorize(both, sort=True)
    T = len(teams)
    win_codes = codes[:n_games]
    lose_codes = codes[n_games:]

    wins = np.bincount(win_codes, minlength=T).astype(np.int16)
    losses = np.bincount(lose_codes, minlength=T).astype(np.int16)

    # For each team: which opponents did they beat?
    order_w = np.argsort(win_codes, kind="mergesort")
    losers_sorted = lose_codes[order_w]
    ends_w = np.cumsum(wins)
    starts_w = np.concatenate(([0], ends_w[:-1]))
    beaten_lists = [losers_sorted[starts_w[i]:ends_w[i]] for i in range(T)]

    # For each team: which opponents beat them?
    order_l = np.argsort(lose_codes, kind="mergesort")
    winners_sorted = win_codes[order_l]
    ends_l = np.cumsum(losses)
    starts_l = np.concatenate(([0], ends_l[:-1]))
    lostto_lists = [winners_sorted[starts_l[i]:ends_l[i]] for i in range(T)]

    best = np.zeros((T, K), dtype=np.int16)
    worst = np.zeros((T, K), dtype=np.int16)
    for i in range(T):
        if beaten_lists[i].size:
            b = np.sort(wins[beaten_lists[i]])[::-1]  # descending: best beaten opponents first
            best[i, : min(K, b.size)] = b[:K]
        if lostto_lists[i].size:
            w = np.sort(wins[lostto_lists[i]])  # ascending: weakest teams that beat you first
            worst[i, : min(K, w.size)] = w[:K]

    data: dict = {
        "team": teams.to_numpy(),
        f"{prefix}wins": wins,
        f"{prefix}losses": losses,
    }
    for i in range(1, K + 1):
        data[f"{prefix}wins_from_{i}best"] = best[:, i - 1]
        data[f"{prefix}wins_from_{i}worst"] = worst[:, i - 1]
    return pd.DataFrame(data)


def get_worster(
    df: pd.DataFrame,
    ly_df: pd.DataFrame,
    K: int = 15,
) -> pd.DataFrame:
    """
    Rank all teams by résumé, using last year's résumé as a tiebreaker.

    Sort priority: wins → quality of beaten/lost-to opponents at each depth
    (1 through K) → last year's equivalent metrics.

    Args:
        df:    Current-season games DataFrame (from get_college_football_games).
        ly_df: Prior-season games DataFrame.
        K:     Résumé ladder depth (default 15).

    Returns:
        DataFrame with columns ['team', 'wins', 'losses',
        'wins_from_1best', 'wins_from_1worst', ..., 'ly_wins', 'ly_losses',
        'ly_wins_from_1best', ...], sorted by the cascading résumé key.
    """
    cur_sched = _prepare_worster_schedule(df)
    ly_sched = _prepare_worster_schedule(ly_df)

    cur_metrics = create_team_metrics(cur_sched, K=K, prefix="")
    ly_metrics = create_team_metrics(ly_sched, K=K, prefix="ly_")

    joined = (
        cur_metrics.set_index("team")
        .join(ly_metrics.set_index("team"), how="left", validate="one_to_one")
        .reset_index()
    )

    # Fill NaNs for teams that didn't appear last year
    ly_cols = ["ly_wins", "ly_losses"] + \
              [f"ly_wins_from_{i}best" for i in range(1, K + 1)] + \
              [f"ly_wins_from_{i}worst" for i in range(1, K + 1)]
    for c in ly_cols:
        if c in joined.columns:
            joined[c] = joined[c].fillna(0).astype(np.int16)

    # Enforce column order
    ordered_cols = ["team", "wins", "losses"]
    for i in range(1, K + 1):
        ordered_cols += [f"wins_from_{i}best", f"wins_from_{i}worst"]
    ordered_cols += ["ly_wins", "ly_losses"]
    for i in range(1, K + 1):
        ordered_cols += [f"ly_wins_from_{i}best", f"ly_wins_from_{i}worst"]
    joined = joined.reindex(columns=ordered_cols)

    # Cascading sort
    sort_cols = ["wins"]
    asc_flags = [False]
    for i in range(1, K + 1):
        sort_cols += [f"wins_from_{i}best", f"wins_from_{i}worst"]
        asc_flags += [False, False]
    sort_cols += ["ly_wins"]
    asc_flags += [False]
    for i in range(1, K + 1):
        sort_cols += [f"ly_wins_from_{i}best", f"ly_wins_from_{i}worst"]
        asc_flags += [False, False]

    return joined.sort_values(by=sort_cols, ascending=asc_flags, kind="mergesort").reset_index(drop=True)
