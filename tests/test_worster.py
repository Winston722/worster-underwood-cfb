import pandas as pd

from worster_underwood_cfb.models.worster import NO_LOSS, create_team_metrics, get_worster


def sched(pairs):
    """pairs: list of (winner, loser) tuples."""
    return pd.DataFrame(pairs, columns=["winner", "loser"])


def games(pairs):
    """Raw games where the first team always wins 28-7 at home."""
    return pd.DataFrame(
        [{"home_team": w, "away_team": l, "home_points": 28, "away_points": 7} for w, l in pairs]
    )


EMPTY_LY = pd.DataFrame(columns=["home_team", "away_team", "home_points", "away_points"])


def test_win_loss_counts():
    m = create_team_metrics(sched([("A", "B"), ("A", "C"), ("B", "C")]), K=3)
    m = m.set_index("team")
    assert m.loc["A", "wins"] == 2 and m.loc["A", "losses"] == 0
    assert m.loc["B", "wins"] == 1 and m.loc["B", "losses"] == 1
    assert m.loc["C", "wins"] == 0 and m.loc["C", "losses"] == 2


def test_best_ladder_sorted_descending():
    # A beat B (2 wins) and C (1 win): ladder should read [2, 1, 0]
    m = create_team_metrics(
        sched([("A", "B"), ("A", "C"), ("B", "X"), ("B", "Y"), ("C", "Z")]), K=3
    ).set_index("team")
    assert m.loc["A", "wins_from_1best"] == 2
    assert m.loc["A", "wins_from_2best"] == 1
    assert m.loc["A", "wins_from_3best"] == 0


def test_worst_ladder_weakest_first_and_no_loss_sentinel():
    # X (3 wins) and Y (2 wins) both beat A: weakest conqueror first -> [2, 3, NO_LOSS]
    m = create_team_metrics(
        sched([("X", "A"), ("Y", "A"), ("X", "B"), ("X", "C"), ("Y", "D")]), K=3
    ).set_index("team")
    assert m.loc["A", "wins_from_1worst"] == 2
    assert m.loc["A", "wins_from_2worst"] == 3
    assert m.loc["A", "wins_from_3worst"] == NO_LOSS
    # X never lost: every slot is the sentinel
    assert m.loc["X", "wins_from_1worst"] == NO_LOSS


def test_more_wins_ranks_first():
    out = get_worster(games([("A", "B"), ("A", "C"), ("B", "C")]), EMPTY_LY)
    assert out.loc[0, "team"] == "A"


def test_last_year_breaks_ties():
    # A and B have identical current seasons; only A has last-year wins
    current = games([("A", "S1"), ("B", "S2")])
    ly = games([("A", "S1")])
    out = get_worster(current, ly)
    teams = list(out["team"])
    assert teams.index("A") < teams.index("B")


def test_ties_are_dropped():
    df = pd.DataFrame(
        [{"home_team": "A", "away_team": "B", "home_points": 21, "away_points": 21}]
    )
    out = get_worster(df, EMPTY_LY)
    assert out.empty
