"""Scoring and the Poisson layer, pinned to values computable by hand."""

from __future__ import annotations

import math

import pytest

from fpl.scoring import Scoring, clean_sheet_probability, threshold_probability


def test_reads_coefficients_by_position(bootstrap):
    s = Scoring.from_bootstrap(bootstrap)
    scoring = bootstrap["game_config"]["scoring"]
    assert s.goal(1) == scoring["goals_scored"]["GKP"]
    assert s.goal(4) == scoring["goals_scored"]["FWD"]
    assert s.clean_sheet(2) == scoring["clean_sheets"]["DEF"]
    assert s.defensive_contribution(1) == 0


def test_goal_value_decreases_up_the_pitch(bootstrap):
    """A structural property of FPL scoring: the further forward, the cheaper a
    goal. If this inverts, the game has changed fundamentally."""
    s = Scoring.from_bootstrap(bootstrap)
    assert s.goal(1) >= s.goal(2) > s.goal(3) > s.goal(4)


@pytest.mark.parametrize(
    ("opponent_xg", "expected"),
    [(0.8, 0.449), (1.0, 0.368), (1.2, 0.301), (1.5, 0.223), (2.0, 0.135)],
)
def test_clean_sheet_probability(opponent_xg: float, expected: float):
    """P(0) = e^-lambda. Halving roughly every 0.7 goals of opponent threat,
    which is why fixture-adjusting defenders matters more than forwards."""
    assert clean_sheet_probability(opponent_xg) == pytest.approx(expected, abs=1e-3)


def test_clean_sheet_probability_is_monotonic():
    values = [clean_sheet_probability(x / 10) for x in range(1, 40)]
    assert values == sorted(values, reverse=True)


def test_threshold_probability_matches_direct_summation():
    """Cross-check the hand-rolled Poisson tail against an independent sum."""
    rate, threshold = 9.07, 10
    direct = 1 - sum(
        math.exp(-rate) * rate**k / math.factorial(k) for k in range(threshold)
    )
    assert threshold_probability(rate, threshold) == pytest.approx(direct, abs=1e-9)


def test_defender_near_threshold_is_a_coin_flip():
    """A defender averaging ~9 defensive actions against a threshold of 10 hits
    it a little under half the time. That is the point: it recurs every match,
    so it is estimable from a small sample -- unlike a defender's goals."""
    p = threshold_probability(9.07, 10)
    assert 0.35 < p < 0.5


def test_threshold_probability_edges():
    assert threshold_probability(0.0, 10) == 0.0
    assert threshold_probability(-1.0, 10) == 0.0
    assert threshold_probability(100.0, 10) == pytest.approx(1.0, abs=1e-6)


def test_threshold_probability_rises_with_rate():
    values = [threshold_probability(r, 12) for r in range(1, 25)]
    assert values == sorted(values)
