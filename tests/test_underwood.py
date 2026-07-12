import numpy as np
import pandas as pd
import pytest

from worster_underwood_cfb.models.underwood import combined, get_error, get_ratings


def make_schedule(games, weight=1.0):
    """games: list of (week, winner, loser, margin)."""
    df = pd.DataFrame(games, columns=["week", "winner", "loser", "hfa_margin"])
    df["weight"] = weight
    return df


# A noise-free round robin where true ratings are A=10, B=0, C=-10:
# every margin equals the true rating gap exactly.
EXACT = [
    (1, "A", "B", 10.0),
    (2, "A", "C", 20.0),
    (3, "B", "C", 10.0),
] * 2


def test_recovers_true_rating_gaps_exactly():
    ratings = get_ratings(make_schedule(EXACT)).set_index("teams")["ratings"]
    assert ratings["A"] - ratings["B"] == pytest.approx(10.0, abs=1e-6)
    assert ratings["B"] - ratings["C"] == pytest.approx(10.0, abs=1e-6)


def test_ratings_sorted_descending():
    ratings = get_ratings(make_schedule(EXACT))
    assert list(ratings["teams"]) == ["A", "B", "C"]
    assert ratings["ratings"].is_monotonic_decreasing


def test_ratings_invariant_to_weight_scale():
    r1 = get_ratings(make_schedule(EXACT, weight=1.0)).set_index("teams")["ratings"]
    r2 = get_ratings(make_schedule(EXACT, weight=0.01)).set_index("teams")["ratings"]
    assert np.allclose(r1.to_numpy(), r2.to_numpy())


def test_pseudo_sd_invariant_to_weight_scale():
    # The published Std Dev must not depend on the arbitrary normalization
    # constant that add_weight applies to the weights.
    for w in (1.0, 0.01):
        sched = make_schedule(EXACT, weight=w)
        err = get_error(sched, get_ratings(sched)).set_index("team")["pseudo_sd"]
        if w == 1.0:
            baseline = err
    assert np.allclose(baseline.to_numpy(), err.to_numpy())


def test_perfect_predictions_yield_pure_prior():
    # Residuals are all zero, so shrinkage should return exactly the prior:
    # (0 * games + 6 * 22) / (games + 22)
    sched = make_schedule(EXACT)
    err = get_error(sched, get_ratings(sched)).set_index("team")["pseudo_sd"]
    games = 4  # each team plays 4 games
    expected = (0 * games + 6 * 22) / (games + 22)
    assert np.allclose(err.to_numpy(), expected)


def test_combined_output_shape():
    sched = make_schedule(EXACT)
    ratings = get_ratings(sched)
    out = combined(ratings, get_error(sched, ratings))
    assert list(out.columns) == ["team", "rating", "pseudo_sd"]
    assert len(out) == 3
    assert out["pseudo_sd"].notna().all()
