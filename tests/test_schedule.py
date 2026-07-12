import pandas as pd
import pytest

from worster_underwood_cfb.transform.schedule import prepare_schedule, add_weight


def make_games(rows):
    cols = ["seasonType", "week", "neutralSite", "homeTeam", "awayTeam", "homePoints", "awayPoints"]
    return pd.DataFrame(rows, columns=cols)


def test_home_win_hfa_subtracted():
    df = make_games([["regular", 1, False, "A", "B", 24, 14]])
    out = prepare_schedule(df, hfa=3)
    assert out.loc[0, "winner"] == "A"
    assert out.loc[0, "loser"] == "B"
    assert out.loc[0, "hfa_margin"] == 7  # 10-point win minus 3 HFA


def test_away_win_hfa_credited():
    df = make_games([["regular", 1, False, "A", "B", 14, 24]])
    out = prepare_schedule(df, hfa=3)
    assert out.loc[0, "winner"] == "B"
    # Away team won by 10 at a hostile venue: margin from their side is 10 + 3
    assert out.loc[0, "hfa_margin"] == 13


def test_neutral_site_no_hfa():
    df = make_games([["regular", 1, True, "A", "B", 24, 14]])
    out = prepare_schedule(df, hfa=3)
    assert out.loc[0, "hfa_margin"] == 10


def test_postseason_maps_to_week_18():
    df = make_games([["postseason", 1, True, "A", "B", 24, 14]])
    out = prepare_schedule(df)
    assert out.loc[0, "week"] == 18


def test_unplayed_games_dropped():
    df = make_games(
        [
            ["regular", 1, False, "A", "B", 24, 14],
            ["regular", 2, False, "C", "D", None, None],
        ]
    )
    out = prepare_schedule(df)
    assert len(out) == 1


def test_snake_case_columns_accepted():
    df = pd.DataFrame(
        [
            {
                "season_type": "regular",
                "week": 1,
                "neutral_site": False,
                "home_team": "A",
                "away_team": "B",
                "home_points": 24,
                "away_points": 14,
            }
        ]
    )
    out = prepare_schedule(df)
    assert out.loc[0, "winner"] == "A"


def test_missing_columns_raise():
    with pytest.raises(KeyError):
        prepare_schedule(pd.DataFrame([{"homeTeam": "A"}]))


def _weighted_schedule(rows):
    return add_weight(pd.DataFrame(rows, columns=["week", "winner", "loser", "hfa_margin"]))


def test_weights_sum_to_100():
    out = _weighted_schedule(
        [[1, "A", "B", 10], [2, "C", "D", 3], [3, "A", "C", 7], [3, "B", "D", 1]]
    )
    assert out["weight"].sum() == pytest.approx(100)


def test_recent_games_weigh_more():
    # Same four teams throughout, so game counts are equal; only recency differs
    out = _weighted_schedule(
        [[1, "A", "B", 10], [1, "C", "D", 3], [5, "A", "C", 7], [5, "B", "D", 1]]
    )
    early = out[out["week"] == 1]["weight"].iloc[0]
    late = out[out["week"] == 5]["weight"].iloc[0]
    assert late > early


def test_low_game_count_teams_weigh_less():
    # A and B play 3 games each; E and F appear once (like an FCS opponent)
    out = _weighted_schedule(
        [
            [1, "A", "B", 10],
            [1, "E", "F", 3],
            [2, "A", "B", 7],
            [2, "A", "B", 1],
        ]
    )
    fcs_game = out[out["winner"] == "E"]["weight"].iloc[0]
    fbs_game = out[(out["winner"] == "A") & (out["week"] == 1)]["weight"].iloc[0]
    assert fcs_game < fbs_game


def test_empty_input_round_trips():
    empty = pd.DataFrame(columns=["seasonType", "week", "neutralSite", "homeTeam", "awayTeam", "homePoints", "awayPoints"])
    out = add_weight(prepare_schedule(empty))
    assert list(out.columns) == ["week", "winner", "loser", "hfa_margin", "weight"]
    assert out.empty
