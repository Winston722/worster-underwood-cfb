import numpy as np
import pandas as pd

from worster_underwood_cfb.models.underwood import get_adjusted_rating
from worster_underwood_cfb.models.worster import get_worster_rating
from batch.model import build_combined_output, build_upcoming_output


def test_basic_math():
    assert 2 + 2 == 4


# ---------------------------------------------------------------------------
# get_adjusted_rating
# ---------------------------------------------------------------------------

def test_adjusted_rating_matches_excel_formula():
    """Matches the Excel formula: ((raw - min) / (max - min)) * 60 - 30."""
    raw = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result = get_adjusted_rating(raw)
    expected = ((raw - raw.min()) / (raw.max() - raw.min())) * 60 - 30
    pd.testing.assert_series_equal(result, expected)


def test_adjusted_rating_bounds():
    """Top team is +30, bottom team is -30."""
    raw = pd.Series([5.1, 3.2, 8.7, 1.0, 6.4])
    result = get_adjusted_rating(raw)
    assert np.isclose(result.max(), 30.0)
    assert np.isclose(result.min(), -30.0)


def test_adjusted_rating_constant_input():
    """All-equal ratings return all zeros (no division by zero)."""
    raw = pd.Series([7.0, 7.0, 7.0])
    result = get_adjusted_rating(raw)
    assert (result == 0.0).all()


# ---------------------------------------------------------------------------
# get_worster_rating
# ---------------------------------------------------------------------------

def _make_worster_df(wins_list: list[int]) -> pd.DataFrame:
    """
    Build a minimal Worster DataFrame where each team has a distinct win total
    and all other columns are zero. All ties are broken at depth 0 (wins).
    """
    n = len(wins_list)
    K = 15
    data: dict = {
        "team": [f"Team{i}" for i in range(n)],
        "wins": wins_list,
        "losses": [0] * n,
    }
    for i in range(1, K + 1):
        data[f"wins_from_{i}best"] = [0] * n
        data[f"wins_from_{i}worst"] = [0] * n
    data["ly_wins"] = [0] * n
    data["ly_losses"] = [0] * n
    for i in range(1, K + 1):
        data[f"ly_wins_from_{i}best"] = [0] * n
        data[f"ly_wins_from_{i}worst"] = [0] * n
    return pd.DataFrame(data)


def test_worster_rating_r1_uniform():
    """
    r=1 produces uniform spacing between every adjacent pair.
    All ties here are broken at depth 0 (wins), so r^0=1 for every pair
    regardless of r — verifying the linear baseline.
    """
    df = _make_worster_df([5, 4, 3, 2, 1])
    ratings = get_worster_rating(df, r=1.0)

    assert np.isclose(ratings.iloc[0], 30.0), "Top team should be +30"
    assert np.isclose(ratings.iloc[-1], -30.0), "Bottom team should be -30"

    diffs = np.diff(ratings.values)
    assert np.allclose(diffs, diffs[0]), "r=1 should produce equal-sized gaps"


def test_worster_rating_r1_matches_linear_scale():
    """
    r=1 gives the same result as evenly spacing n teams across [-30, +30].
    This is the formal equivalence between r=1 and the old smooth rank scale.
    """
    n = 10
    df = _make_worster_df(list(range(n, 0, -1)))
    ratings = get_worster_rating(df, r=1.0).values

    expected = np.linspace(30.0, -30.0, n)
    assert np.allclose(ratings, expected)


def test_worster_rating_decay_smaller_gap_for_deeper_tie():
    """
    With r < 1, a pair separated at a deeper sort column gets a smaller gap
    than a pair separated at wins.
    """
    K = 15
    # Team A vs B: differ at wins (depth 0) → gap = r^0 = 1
    # Team B vs C: identical wins, differ at wins_from_1best (depth 1) → gap = r^1
    data: dict = {
        "team": ["A", "B", "C"],
        "wins": [3, 2, 2],       # A vs B breaks at wins; B vs C tied on wins
        "losses": [0, 0, 0],
        "wins_from_1best": [0, 5, 3],  # B vs C breaks here
    }
    for i in range(2, K + 1):
        data[f"wins_from_{i}best"] = [0, 0, 0]
    for i in range(1, K + 1):
        data[f"wins_from_{i}worst"] = [0, 0, 0]
    data["ly_wins"] = [0, 0, 0]
    data["ly_losses"] = [0, 0, 0]
    for i in range(1, K + 1):
        data[f"ly_wins_from_{i}best"] = [0, 0, 0]
        data[f"ly_wins_from_{i}worst"] = [0, 0, 0]
    df = pd.DataFrame(data)

    r = 0.85
    ratings = get_worster_rating(df, r=r)

    gap_AB = ratings.iloc[0] - ratings.iloc[1]  # broken at depth 0 → larger
    gap_BC = ratings.iloc[1] - ratings.iloc[2]  # broken at depth 1 → smaller

    assert gap_AB > gap_BC, "Gap broken at wins should be larger than gap broken deeper"


def test_worster_rating_bounds():
    """Top team is always +30, bottom team is always -30."""
    df = _make_worster_df([8, 7, 5, 3, 1])
    for r in [0.5, 0.85, 1.0]:
        ratings = get_worster_rating(df, r=r)
        assert np.isclose(ratings.iloc[0], 30.0)
        assert np.isclose(ratings.iloc[-1], -30.0)


# ---------------------------------------------------------------------------
# build_combined_output
# ---------------------------------------------------------------------------

def _make_underwood_df(teams, adj_ratings):
    return pd.DataFrame({
        "Rank": range(1, len(teams) + 1),
        "Team": teams,
        "Adjusted Rating": adj_ratings,
        "Rating": adj_ratings,
        "Std Dev": [5.0] * len(teams),
    })


def _make_worster_df_combined(teams, adj_ratings):
    return pd.DataFrame({
        "Rank": range(1, len(teams) + 1),
        "Adjusted Rating": adj_ratings,
        "team": teams,
        "wins": [5] * len(teams),
        "losses": [0] * len(teams),
    })


def test_combined_wu_rating_normalized():
    """WU Rating z-normalizes both models before averaging, then rescales to [-30, +30].
    Top team is always +30, bottom team is always -30."""
    teams = ["A", "B", "C"]
    u_adj = [20.0, 0.0, -20.0]
    w_adj = [10.0, 10.0, -10.0]

    underwood = _make_underwood_df(teams, u_adj)
    worster = _make_worster_df_combined(teams, w_adj)
    result = build_combined_output(underwood, worster).set_index("Team")

    assert np.isclose(result.loc["A", "WU Rating"], 30.0)
    assert np.isclose(result.loc["B", "WU Rating"], 13.92, atol=0.01)
    assert np.isclose(result.loc["C", "WU Rating"], -30.0)


def test_combined_sorted_by_wu_rating():
    """Output is sorted descending by WU Rating."""
    teams = ["A", "B", "C"]
    underwood = _make_underwood_df(teams, [10.0, 30.0, -5.0])
    worster = _make_worster_df_combined(teams, [20.0, 10.0, 0.0])
    result = build_combined_output(underwood, worster)

    assert result["WU Rating"].is_monotonic_decreasing


def test_combined_disagreement_formula():
    """Disagreement matches the Excel formula: |u - w| / log10(rank + 1)."""
    teams = ["A", "B"]
    underwood = _make_underwood_df(teams, [30.0, -30.0])
    worster = _make_worster_df_combined(teams, [10.0, -10.0])
    result = build_combined_output(underwood, worster)

    for _, row in result.iterrows():
        expected = abs(row["Underwood Rating"] - row["Worster Rating"]) / np.log10(row["Rank"] + 1)
        assert np.isclose(row["Disagreement"], round(expected, 2))


def test_combined_higher_rank_amplifies_disagreement():
    """For equal raw disagreement, rank-1 team has a higher disagreement score than a lower-ranked team."""
    teams = ["A", "B", "C", "D"]
    underwood = _make_underwood_df(teams, [25.0, 20.0, 5.0, 0.0])
    worster = _make_worster_df_combined(teams, [15.0, 10.0, -5.0, -10.0])
    result = build_combined_output(underwood, worster).set_index("Team")

    # All teams have |u - w| = 10; rank 1 should have the largest disagreement score
    assert result.loc["A", "Disagreement"] > result.loc["B", "Disagreement"]
    assert result.loc["B", "Disagreement"] > result.loc["C", "Disagreement"]


# ---------------------------------------------------------------------------
# build_upcoming_output — Talent and Competitive indices
# ---------------------------------------------------------------------------

def _make_combined_df(teams, wu_ratings):
    """Minimal combined DataFrame with Rank, Team, WU Rating."""
    return pd.DataFrame({
        "Rank": range(1, len(teams) + 1),
        "Team": teams,
        "WU Rating": wu_ratings,
        "Underwood Rating": wu_ratings,
        "Worster Rating": wu_ratings,
        "Disagreement": [0.0] * len(teams),
    })


def _make_raw_games_df(matchups):
    """
    Build a minimal raw games DataFrame with unplayed games.
    matchups: list of (homeTeam, awayTeam) tuples.
    """
    rows = []
    for home, away in matchups:
        rows.append({
            "homeTeam": home, "awayTeam": away,
            "homePoints": None, "awayPoints": None,
            "homeClassification": "fbs", "awayClassification": "fbs",
            "id": len(rows), "season": 2025, "week": 1,
            "startDate": "2025-09-06 12:00:00", "neutralSite": False,
            "homeId": len(rows) * 2, "awayId": len(rows) * 2 + 1,
        })
    return pd.DataFrame(rows)


def test_upcoming_top_two_talent_is_one():
    """The game between #1 and #2 should have Talent index = 1.0."""
    teams = ["A", "B", "C", "D"]
    wu =    [30.0, 10.0, -10.0, -30.0]
    combined = _make_combined_df(teams, wu)
    games = _make_raw_games_df([("A", "B")])
    result = build_upcoming_output(games, combined)
    assert np.isclose(result.loc[0, "Talent"], 1.0)


def test_upcoming_bottom_two_talent_is_zero():
    """The game between the last two teams should have Talent index = 0.0."""
    teams = ["A", "B", "C", "D"]
    wu =    [30.0, 10.0, -10.0, -30.0]
    combined = _make_combined_df(teams, wu)
    games = _make_raw_games_df([("C", "D")])
    result = build_upcoming_output(games, combined)
    assert np.isclose(result.loc[0, "Talent"], 0.0)


def test_upcoming_top_vs_bottom_competitive_is_zero():
    """#1 vs #N should have Competitive index = 0.0."""
    teams = ["A", "B", "C", "D"]
    wu =    [30.0, 10.0, -10.0, -30.0]
    combined = _make_combined_df(teams, wu)
    games = _make_raw_games_df([("A", "D")])
    result = build_upcoming_output(games, combined)
    assert np.isclose(result.loc[0, "Competitive"], 0.0)


def test_upcoming_equal_teams_competitive_is_one():
    """Two teams with identical WU ratings should have Competitive index = 1.0."""
    teams = ["A", "B", "C", "D"]
    wu =    [30.0, 0.0, 0.0, -30.0]
    combined = _make_combined_df(teams, wu)
    games = _make_raw_games_df([("B", "C")])
    result = build_upcoming_output(games, combined)
    assert np.isclose(result.loc[0, "Competitive"], 1.0)


def test_upcoming_missing_rating_gives_nan():
    """A game involving a team with no WU rating should yield NaN indices."""
    teams = ["A", "B"]
    wu =    [30.0, -30.0]
    combined = _make_combined_df(teams, wu)
    games = _make_raw_games_df([("A", "FCS_Team")])  # FCS_Team not in combined
    result = build_upcoming_output(games, combined)
    assert pd.isna(result.loc[0, "Talent"])
    assert pd.isna(result.loc[0, "Competitive"])