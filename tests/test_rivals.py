"""League-relative strategy.

The result these tests exist to protect: ownership cannot change the
mean-optimal squad. Your score relative to the field is `yours - theirs`, and by
linearity of expectation its mean is `E[yours] - E[theirs]` -- the second term
independent of everything you choose. Every "fade the template for more expected
points" argument is therefore wrong.

Ownership governs variance instead, and variance is what actually decides
whether you finish above seven specific people.
"""

from __future__ import annotations

import pytest

from fpl.optimise import Rules, solve
from fpl.rivals import (
    ENDGAME_GAMEWEEKS,
    MAX_OVERLAP_WEIGHT,
    Standing,
    describe,
    effective_ownership,
    overlap_weight,
    ownership,
    standing_from_history,
)

from tests.test_optimise import HORIZON, make_pool


@pytest.fixture
def rules(bootstrap) -> Rules:
    return Rules.from_bootstrap(bootstrap)


def standing(points: int, rivals: list[int], remaining: int) -> Standing:
    return Standing(entry_id=1, points=points, rival_points=rivals, gameweeks_remaining=remaining)


# -- ownership ------------------------------------------------------------


def test_ownership_is_exact_not_estimated():
    """The advantage of a small league: you fetch every rival's squad rather
    than estimating from global percentages."""
    picks = {1: [10, 11, 12], 2: [10, 11, 20], 3: [10, 30, 31]}
    share = ownership(picks, [10, 11, 20, 30, 99])
    assert share[10] == pytest.approx(1.0)
    assert share[11] == pytest.approx(2 / 3)
    assert share[20] == pytest.approx(1 / 3)
    assert share[99] == 0.0


def test_ownership_of_an_empty_league_is_empty():
    assert ownership({}, [1, 2, 3]) == {}


def test_effective_ownership_counts_the_captain_twice():
    """A player everyone owns and half the league captains returns more than
    100% of his score to the field."""
    picks = {1: [10, 11], 2: [10, 11]}
    eo = effective_ownership(picks, {1: 10}, [10, 11])
    assert eo[10] == pytest.approx(1.5)
    assert eo[11] == pytest.approx(1.0)


# -- the variance dial ----------------------------------------------------


def test_no_tilt_early_in_the_season():
    """Almost any gap is recoverable by playing well when most of the season
    remains, and distorting the squad for it costs points now for nothing."""
    assert overlap_weight(standing(100, [400], remaining=30)) == 0.0


def test_no_tilt_when_the_gap_is_trivial():
    assert overlap_weight(standing(500, [502], remaining=2)) == pytest.approx(0.0, abs=0.05)


def test_leading_late_favours_the_template():
    """Positive weight means prefer players rivals own -- shared players cancel
    out of the difference between your scores, so overlap removes swing."""
    w = overlap_weight(standing(600, [520, 500], remaining=2))
    assert w > 0


def test_trailing_late_favours_differentials():
    w = overlap_weight(standing(500, [600, 480], remaining=2))
    assert w < 0


def test_tilt_grows_as_the_season_runs_out():
    early = overlap_weight(standing(600, [520], remaining=ENDGAME_GAMEWEEKS - 1))
    late = overlap_weight(standing(600, [520], remaining=1))
    assert late > early > 0


def test_tilt_grows_with_the_gap():
    small = overlap_weight(standing(500, [515], remaining=2))
    large = overlap_weight(standing(500, [600], remaining=2))
    assert large < small < 0


def test_tilt_is_bounded():
    """A tilt on top of a points objective, not a replacement for it. A squad
    chasing variance while ignoring points loses on both counts."""
    extreme = overlap_weight(standing(0, [9999], remaining=0))
    assert abs(extreme) <= MAX_OVERLAP_WEIGHT


def test_no_rivals_means_no_tilt():
    assert overlap_weight(standing(500, [], remaining=1)) == 0.0


def test_rank_and_gap():
    s = standing(500, [600, 550, 400], remaining=5)
    assert s.rank == 3
    assert s.leader_gap == 100
    assert not s.is_leading


def test_leader_gap_is_negative_when_ahead():
    s = standing(700, [600, 550], remaining=5)
    assert s.is_leading and s.leader_gap == -100 and s.rank == 1


# -- effect on the actual solve -------------------------------------------


def test_overlap_weight_of_zero_changes_nothing(rules: Rules):
    """The default must be exactly the points-maximising squad. This is the
    linearity-of-expectation result in executable form: ownership is inert on
    the mean."""
    pool = make_pool(80)
    share = {p.element_id: (p.element_id % 3) / 2 for p in pool}
    plain = solve(pool, rules, horizon=HORIZON)
    tilted = solve(pool, rules, horizon=HORIZON, overlap=share, overlap_weight=0.0)
    assert {p.element_id for p in plain.squad} == {p.element_id for p in tilted.squad}


def test_positive_weight_pulls_the_xi_toward_owned_players(rules: Rules):
    pool = make_pool(81)
    share = {p.element_id: 1.0 if p.element_id % 2 == 0 else 0.0 for p in pool}

    plain = solve(pool, rules, horizon=HORIZON)
    template = solve(pool, rules, horizon=HORIZON, overlap=share, overlap_weight=1.5)

    before = sum(share[p.element_id] for p in plain.starting)
    after = sum(share[p.element_id] for p in template.starting)
    assert after > before


def test_negative_weight_pushes_the_xi_toward_differentials(rules: Rules):
    pool = make_pool(82)
    share = {p.element_id: 1.0 if p.element_id % 2 == 0 else 0.0 for p in pool}

    plain = solve(pool, rules, horizon=HORIZON)
    differential = solve(pool, rules, horizon=HORIZON, overlap=share, overlap_weight=-1.5)

    before = sum(share[p.element_id] for p in plain.starting)
    after = sum(share[p.element_id] for p in differential.starting)
    assert after < before


def test_tilting_costs_expected_points(rules: Rules):
    """It has to, and saying so plainly matters. The tilt buys a variance
    profile by giving up mean, which is only worth it when the mean alone
    cannot get you where you need to be."""
    pool = make_pool(83)
    share = {p.element_id: 1.0 if p.element_id % 2 == 0 else 0.0 for p in pool}

    plain = solve(pool, rules, horizon=HORIZON)
    tilted = solve(pool, rules, horizon=HORIZON, overlap=share, overlap_weight=-1.5)

    plain_ep = sum(p.ep_horizon for p in plain.starting)
    tilted_ep = sum(p.ep_horizon for p in tilted.starting)
    assert tilted_ep <= plain_ep + 1e-9


def test_tilted_squad_is_still_legal(rules: Rules):
    from tests.test_optimise import assert_legal

    pool = make_pool(84)
    share = {p.element_id: (p.element_id % 4) / 3 for p in pool}
    assert_legal(solve(pool, rules, horizon=HORIZON, overlap=share, overlap_weight=-2.0), rules)


# -- standings and explanation --------------------------------------------


def test_standing_read_from_histories():
    histories = {
        1: {"current": [{"total_points": 50}, {"total_points": 110}]},
        2: {"current": [{"total_points": 60}, {"total_points": 130}]},
        3: {"current": [{"total_points": 40}, {"total_points": 90}]},
    }
    s = standing_from_history(1, histories, gameweeks_remaining=30)
    assert s.points == 110
    assert sorted(s.rival_points) == [90, 130]
    assert s.rank == 2


def test_standing_is_none_before_any_gameweek():
    assert standing_from_history(1, {1: {"current": []}}, gameweeks_remaining=38) is None


def test_describe_explains_the_tilt():
    """A squad that suddenly prefers template players has to say why, or it
    reads as the model changing its mind."""
    ahead = standing(600, [520], remaining=2)
    behind = standing(500, [600], remaining=2)
    level = standing(500, [502], remaining=30)

    assert "reduce swing" in describe(ahead, overlap_weight(ahead))
    assert "differentials" in describe(behind, overlap_weight(behind))
    assert "ignoring ownership" in describe(level, overlap_weight(level))
