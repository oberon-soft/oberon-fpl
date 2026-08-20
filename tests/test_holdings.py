"""Selling prices, and the checksum that keeps them honest.

The sell-on fee is the one place where getting the arithmetic wrong does not
produce a loud failure. Mis-state the budget and the solver does not refuse --
there is always some cheaper legal squad -- it quietly sells players to fit, and
reports those sales as recommendations.
"""

from __future__ import annotations

import pytest

from fpl.holdings import (
    Reconciliation,
    ValueSemantics,
    arrivals,
    purchase_price_candidates,
    reconcile,
    sell_prices,
    selling_price,
)
from fpl.optimise import Rules, solve

from tests.test_optimise import HORIZON, make_pool


@pytest.mark.parametrize(
    ("purchase", "now", "expected"),
    [
        (60, 60, 60),   # unchanged
        (60, 61, 60),   # +0.1 returns nothing: half of one unit rounds down to zero
        (60, 62, 61),   # +0.2 returns 0.1
        (60, 63, 61),   # +0.3 still returns 0.1
        (60, 64, 62),   # +0.4 returns 0.2
        (60, 65, 62),   # +0.5 returns 0.2
        (60, 70, 65),   # +1.0 returns 0.5
    ],
)
def test_you_keep_half_the_profit_rounded_down(purchase: int, now: int, expected: int):
    assert selling_price(purchase, now) == expected


@pytest.mark.parametrize(("purchase", "now"), [(60, 59), (60, 55), (120, 100)])
def test_falls_have_no_protection(purchase: int, now: int):
    """You sell at market and take the whole loss -- the fee is one-sided."""
    assert selling_price(purchase, now) == now


def test_selling_price_never_exceeds_market():
    for purchase in range(40, 140):
        for now in range(40, 140):
            assert selling_price(purchase, now) <= now


def test_selling_price_never_below_purchase_on_a_rise():
    for purchase in range(40, 140):
        for now in range(purchase, 140):
            assert selling_price(purchase, now) >= purchase


# -- the reconciliation that settles what `value` means -------------------


def test_no_movement_is_undecidable():
    """Gameweek one. Every player is held at what they cost, so market and
    selling totals agree and `value` cannot distinguish between them."""
    r = reconcile(reported_value=1000, market_total=1000, selling_total=1000)
    assert r.semantics is ValueSemantics.UNKNOWN
    assert "nothing to distinguish" in r.explain()


def test_value_matching_market_means_no_fee_on_team_value():
    """If FPL reports the market total once prices have moved, the sell-on fee
    does not apply to team value and this whole module can be retired."""
    r = reconcile(reported_value=1014, market_total=1014, selling_total=1007)
    assert r.semantics is ValueSemantics.MARKET
    assert "not applying a sell-on fee" in r.explain()


def test_value_below_market_means_the_fee_is_real():
    r = reconcile(reported_value=1007, market_total=1014, selling_total=1007)
    assert r.semantics is ValueSemantics.NET_OF_FEE
    assert r.agrees
    assert "purchase prices are correct" in r.explain()


def test_a_wrong_purchase_price_shows_up_as_a_discrepancy():
    """The checksum's real job: catching our own error the week it appears
    rather than letting it drift."""
    r = reconcile(reported_value=1007, market_total=1014, selling_total=1005)
    assert r.semantics is ValueSemantics.NET_OF_FEE
    assert not r.agrees
    assert r.discrepancy == 2
    assert "a tracked purchase price is wrong" in r.explain()


def test_purchase_candidates_are_cheapest_first():
    """A lower purchase price implies a higher selling price, so searching from
    the cheapest finds the assignment reproducing `value` soonest."""
    assert purchase_price_candidates([62, 60, 61, 60], fallback=63) == [60, 61, 62]


def test_purchase_candidates_fall_back_when_no_snapshots_exist():
    assert purchase_price_candidates([], fallback=63) == [63]


def test_arrivals_are_the_players_who_appeared():
    assert arrivals({1, 2, 3}, {2, 3, 4}) == {4}


def test_sell_prices_omit_unknown_players_rather_than_guessing():
    """Better to fall back to market for an untracked player -- generous by at
    most half a rise -- than to invent a purchase price."""
    prices = sell_prices({10: 60, 11: 50}, {10: 64, 11: 50, 12: 70})
    assert prices == {10: 62, 11: 50}
    assert 12 not in prices


# -- the failure this prevents --------------------------------------------


@pytest.fixture
def rules(bootstrap) -> Rules:
    return Rules.from_bootstrap(bootstrap)


def test_budgeting_from_value_forces_spurious_selling(rules: Rules):
    """The bug, reproduced end to end -- and it is quieter than infeasibility.

    Budget from FPL's `value` (net of the sell-on fee) while costing candidates
    at market price and the squad you already own no longer fits. The solver does
    not fail; there is always some cheaper legal squad. It simply sells players
    to satisfy a budget that was mis-stated, and reports those sales as
    recommendations. A loud error would have been kinder.
    """
    pool = make_pool(40)
    base = solve(pool, rules, horizon=HORIZON)
    held = [p.element_id for p in base.squad]

    market_total = sum(p.now_cost for p in base.squad)
    # Every player has risen 0.4 since purchase, so FPL credits 0.2 each.
    value_net_of_fee = market_total - 2 * len(held)

    forced = solve(
        pool, rules, horizon=HORIZON, current=held,
        free_transfers=1, budget=value_net_of_fee,
    )
    assert forced.transfers_in, "expected the mis-stated budget to force sales"
    # Cheaper than the squad it started from, which is the tell: the sales were
    # not an upgrade, they were a way to fit an arithmetic error.
    assert forced.spend < market_total


def test_costing_held_players_at_selling_price_restores_the_hold(rules: Rules):
    """The fix. Budget and costs in the same units, so the squad you own fits
    and no transfer is invented to satisfy arithmetic."""
    pool = make_pool(40)
    base = solve(pool, rules, horizon=HORIZON)
    held = [p.element_id for p in base.squad]

    market_total = sum(p.now_cost for p in base.squad)
    value_net_of_fee = market_total - 2 * len(held)
    costs = {p.element_id: p.now_cost - 2 for p in base.squad}

    sol = solve(
        pool, rules, horizon=HORIZON, current=held,
        free_transfers=1, budget=value_net_of_fee, costs=costs,
    )
    assert len(sol.squad) == rules.squad_size
    assert not sol.transfers_in, "holding should be affordable again"
    assert sol.hits == 0


def test_cost_overrides_only_touch_the_players_named(rules: Rules):
    """An unheld player must still be costed at market -- you pay the asking
    price for someone you do not own."""
    pool = make_pool(41)
    base = solve(pool, rules, horizon=HORIZON)
    overridden = base.squad[0]
    costs = {overridden.element_id: overridden.now_cost - 5}

    sol = solve(pool, rules, horizon=HORIZON, costs=costs)
    for p in sol.squad:
        if p.element_id != overridden.element_id:
            assert costs.get(p.element_id, p.now_cost) == p.now_cost
