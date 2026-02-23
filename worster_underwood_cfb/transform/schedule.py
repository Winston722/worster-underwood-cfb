from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["prepare_schedule", "add_weight"]


def prepare_schedule(
    df: pd.DataFrame,
    hfa: int = 3,
    decay: float = 1 / 3,  # kept for compat; unused here
) -> pd.DataFrame:
    """
    Transform a raw games DataFrame into ['week','winner','loser','hfa_margin']
    ready for add_weight().

    Accepts either camelCase (CFBD API) or snake_case column names.
    Drops canceled/incomplete games (missing scores).
    Postseason games are mapped to week 18.
    HFA (home field advantage) is subtracted from the home team's margin so
    that the resulting margin reflects true team quality, not venue.
    """
    if df.empty:
        return pd.DataFrame(columns=["week", "winner", "loser", "hfa_margin"])

    # Normalize snake_case -> camelCase so the rest of the logic is column-stable.
    rename_map = {
        "season_type": "seasonType",
        "neutral_site": "neutralSite",
        "home_team": "homeTeam",
        "away_team": "awayTeam",
        "home_points": "homePoints",
        "away_points": "awayPoints",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["seasonType", "week", "neutralSite", "homeTeam", "awayTeam", "homePoints", "awayPoints"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"prepare_schedule missing required columns: {missing}")

    # 1) Drop canceled / incomplete
    df = df.dropna(subset=["homePoints", "awayPoints"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["week", "winner", "loser", "hfa_margin"])

    # 2) Fail-fast invariants
    assert not df[["seasonType", "week", "homeTeam", "awayTeam"]].isna().any().any(), \
        "Nulls in required non-score fields."
    assert not df["neutralSite"].isna().any(), "neutralSite should be non-null after drop."

    # 3) Types + postseason mapping (robust to Enums / weird casings)
    df["seasonType"] = df["seasonType"].astype(str).str.split(".").str[-1].str.lower()
    df["week"] = pd.to_numeric(df["week"], errors="raise", downcast="integer")
    df.loc[df["seasonType"].eq("postseason"), "week"] = 18
    df["week"] = df["week"].astype("int16")
    assert (df["week"] >= 1).all(), "week must be >= 1"

    # Pull arrays once (normalize neutral booleans safely)
    ns = (
        df["neutralSite"]
        .replace({"True": True, "False": False, "true": True, "false": False, 1: True, 0: False})
        .astype(bool)
        .to_numpy()
    )
    hp = pd.to_numeric(df["homePoints"], errors="raise").to_numpy()
    ap = pd.to_numeric(df["awayPoints"], errors="raise").to_numpy()
    wk = df["week"].to_numpy()
    home = df["homeTeam"].to_numpy(object)
    away = df["awayTeam"].to_numpy(object)

    # 4) Margins & outcomes
    margin = hp - ap  # home-perspective true margin
    # Ties are kept: a neutral-site tie contributes adj_margin=0 (teams are equal);
    # a home-team tie contributes adj_margin=+HFA for the away team (they matched
    # the home team despite the venue disadvantage). Both are meaningful regression signal.
    # For ties, we arbitrarily assign the away team as "winner" so the data structure
    # is consistent — the regression only uses the numeric margin, not the label.
    home_field = np.where(ns, 0, hfa)  # 0 if neutral site, else HFA
    adj_home = margin - home_field     # remove HFA from home side

    home_win = margin > 0
    winners = np.where(home_win, home, away)
    losers = np.where(home_win, away, home)
    hfa_margin = np.where(home_win, adj_home, -adj_home)

    return pd.DataFrame({"week": wk, "winner": winners, "loser": losers, "hfa_margin": hfa_margin})


def add_weight(df: pd.DataFrame, decay: float = 1 / 3) -> pd.DataFrame:
    """
    Assign a weight to each game based on recency and how many FBS games each
    participating team has played.

    Weight formula (per game):
        sqrt((total_games / max_total_games) / (weeks_ago ** decay))
    Weights are then normalized to sum to 100.

    The game-count factor implicitly down-weights FCS opponents because they
    appear in far fewer FBS games than a full-schedule FBS team.

    Args:
        df:    DataFrame with columns ['week','winner','loser','hfa_margin'].
        decay: Time-decay exponent (default 1/3). Higher = steeper recency bias.

    Returns:
        DataFrame with an added 'weight' column.
    """
    if df.empty:
        return df.assign(weight=pd.Series(dtype="float64"))[
            ["week", "winner", "loser", "hfa_margin", "weight"]
        ]

    assert decay > 0, "decay must be > 0"
    assert not df[["winner", "loser"]].isna().any().any(), "winner/loser must be non-null"
    assert (df["week"] >= 1).all(), "week must be >= 1"

    winner_vals = df["winner"].values
    loser_vals = df["loser"].values
    week_vals = df["week"].values

    # Efficient team encoding + game-count via bincount
    both_teams = np.concatenate([winner_vals, loser_vals])
    codes, _ = pd.factorize(both_teams, sort=False)
    n = len(df)
    counts = np.bincount(codes)
    total_games = counts[codes[:n]] + counts[codes[n:]]

    weeks_ago = (week_vals.max() + 1) - week_vals
    max_games = total_games.max()

    if max_games > 0:
        weights = np.sqrt((total_games / max_games) / (weeks_ago ** decay))
        weights *= 100.0 / weights.sum()  # normalize to sum to 100
    else:
        weights = np.zeros(n, dtype=np.float64)

    result = df[["week", "winner", "loser", "hfa_margin"]].copy()
    result["weight"] = weights
    return result
