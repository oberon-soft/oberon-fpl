"""Rates and the two-part shrinkage.

The split matters. A single constant doing both jobs compressed elite players'
edge over the pack by around 40%, which changed whether a premium striker was
worth buying -- the model's most consequential decision.
"""

from __future__ import annotations

import pytest

from fpl.config import Shrinkage
from fpl.rates import extract, positional_means, price_prior, shrink

SEASON = {
    "season_name": "2025/26",
    "minutes": 2750,
    "starts": 30,
    "total_points": 209,
    "expected_goals": "2.94",
    "expected_assists": "1.75",
    "defensive_contribution": 277,
    "bonus": 30,
    "saves": 0,
    "yellow_cards": 4,
}


def test_extract_converts_to_per_90():
    r = extract(226597, 2, SEASON)
    assert r is not None
    assert r.per90["xg"] == pytest.approx(2.94 * 90 / 2750, abs=1e-6)
    assert r.per90["dc"] == pytest.approx(277 * 90 / 2750, abs=1e-6)
    assert r.points_per_game == pytest.approx(209 / 38, abs=1e-6)


def test_extract_rejects_a_player_who_never_played():
    assert extract(1, 2, {**SEASON, "minutes": 0}) is None


def test_expected_minutes_and_start_probability():
    r = extract(226597, 2, SEASON)
    assert r.expected_minutes == pytest.approx(2750 / 38, abs=1e-6)
    assert r.p_start == pytest.approx(30 / 38, abs=1e-6)
    assert r.p_sixty < r.p_start  # starters do not always see the hour out


def _population():
    """A spread of defenders: one elite, several ordinary, one tiny sample."""
    rows = []
    for i, (minutes, xg_total) in enumerate(
        [(2750, 2.94), (2600, 1.2), (2400, 0.9), (2200, 1.1), (900, 0.4), (120, 0.9)]
    ):
        rows.append(
            (55 + i, extract(1000 + i, 2, {**SEASON, "minutes": minutes, "expected_goals": str(xg_total)}))
        )
    return rows


def test_full_season_barely_shrinks_at_the_default():
    """The whole point of splitting the constants. A player with 2750 minutes has
    a well-measured rate; sampling noise should move it hardly at all."""
    observed = _population()
    means = positional_means(r for _, r in observed)
    elite = observed[0][1]

    noise_only = Shrinkage(noise_k=150.0, regression=1.0)
    shrunk = shrink(elite, means, noise_only)
    assert shrunk["xg"] == pytest.approx(elite.per90["xg"], rel=0.06)


def test_small_sample_is_pulled_hard_toward_the_mean():
    observed = _population()
    means = positional_means(r for _, r in observed)
    tiny = observed[-1][1]  # 120 minutes
    assert tiny.minutes == 120

    shrunk = shrink(tiny, means, Shrinkage(noise_k=150.0, regression=1.0))
    distance_before = abs(tiny.per90["xg"] - means[2]["xg"])
    distance_after = abs(shrunk["xg"] - means[2]["xg"])
    assert distance_after < distance_before * 0.6


def test_regression_scales_everyone_equally():
    """Unlike noise shrinkage, it does not depend on sample size."""
    observed = _population()
    means = positional_means(r for _, r in observed)
    flat = Shrinkage(noise_k=150.0, regression=0.9)
    unit = Shrinkage(noise_k=150.0, regression=1.0)

    for _, rates in observed:
        a = shrink(rates, means, flat)
        b = shrink(rates, means, unit)
        for key in a:
            assert a[key] == pytest.approx(b[key] * 0.9, abs=1e-9)


def test_aggressive_noise_k_compresses_the_elite_edge():
    """Reproduces the bug the split fixes.

    With a large noise constant, a full-season elite player's advantage over the
    positional mean shrinks substantially -- and captaincy then amplifies a
    number that has already been flattened, so the premium never looks worth it.
    """
    observed = _population()
    means = positional_means(r for _, r in observed)
    elite = observed[0][1]
    mean_xg = means[2]["xg"]

    tight = shrink(elite, means, Shrinkage(noise_k=150.0, regression=1.0))
    loose = shrink(elite, means, Shrinkage(noise_k=900.0, regression=1.0))

    edge_tight = tight["xg"] - mean_xg
    edge_loose = loose["xg"] - mean_xg
    assert edge_loose < edge_tight * 0.85


def test_positional_means_ignore_bit_part_players():
    """Otherwise the prior is dragged toward zero by academy names who never
    play, and every shrunk rate goes with it."""
    observed = _population()
    with_cutoff = positional_means((r for _, r in observed), min_minutes=600)
    without = positional_means((r for _, r in observed), min_minutes=0)
    assert with_cutoff[2]["xg"] != without[2]["xg"]


def test_price_prior_fills_the_gap_for_an_unknown_player():
    observed = _population()
    prior = price_prior(2, 55, observed)
    assert prior is not None and prior.imputed
    assert prior.minutes > 0


def test_price_prior_never_outranks_a_known_quantity():
    """It returns the average of comparable players, so the model cannot fall in
    love with a stranger it has no evidence about."""
    observed = _population()
    prior = price_prior(2, 55, observed)
    best_observed = max(r.per90["xg"] for _, r in observed if r.minutes >= 900)
    assert prior.per90["xg"] < best_observed


def test_price_prior_returns_none_without_comparables():
    assert price_prior(4, 55, _population()) is None


def test_imputed_players_excluded_from_the_priors_they_feed():
    """Otherwise imputed rates would reinforce themselves each run."""
    observed = _population()
    prior = price_prior(2, 55, observed)
    with_imputed = observed + [(55, prior)]
    assert positional_means(r for _, r in observed) == positional_means(
        r for _, r in with_imputed
    )
