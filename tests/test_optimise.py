"""Optimiser tests.

Mostly properties rather than expected squads. A solver bug does not throw --
it returns a confident, plausible, illegal answer, and the only defence is
asserting the constraints hold on whatever comes back.
"""

from __future__ import annotations

import random

import pytest

from fpl.optimise import (
    TRANSFER_COST,
    Chip,
    InfeasibleError,
    Rules,
    Solution,
    chip_gain,
    solve,
)
from fpl.project import Projection

HORIZON = [1, 2, 3]


@pytest.fixture
def rules(bootstrap) -> Rules:
    return Rules.from_bootstrap(bootstrap)


def make_pool(seed: int = 0, per_position: int = 14, teams: int = 20) -> list[Projection]:
    """A synthetic player pool wide enough to be solvable but not trivially so."""
    rng = random.Random(seed)
    pool: list[Projection] = []
    element_id = 0
    for element_type in (1, 2, 3, 4):
        for k in range(per_position):
            element_id += 1
            cost = rng.randrange(40, 130, 5)
            base = 1.0 + (cost - 40) / 90 * 4.0 + rng.uniform(-0.6, 0.6)
            pool.append(
                Projection(
                    code=element_id,
                    element_id=element_id,
                    element_type=element_type,
                    web_name=f"P{element_id}",
                    team_id=element_id % teams + 1,
                    now_cost=cost,
                    by_gameweek={gw: max(0.0, base + rng.uniform(-0.4, 0.4)) for gw in HORIZON},
                )
            )
    return pool


def assert_legal(sol: Solution, rules: Rules) -> None:
    """Every structural rule, checked on the returned answer."""
    assert len(sol.squad) == rules.squad_size
    assert len(sol.starting) == rules.starting_size
    assert len(sol.bench) == rules.squad_size - rules.starting_size
    assert sol.spend <= rules.budget

    for element_type, quota in rules.squad_select.items():
        assert sum(1 for p in sol.squad if p.element_type == element_type) == quota
        playing = sum(1 for p in sol.starting if p.element_type == element_type)
        assert rules.play_min[element_type] <= playing <= rules.play_max[element_type]

    for club in {p.team_id for p in sol.squad}:
        assert sum(1 for p in sol.squad if p.team_id == club) <= rules.team_limit

    squad_ids = {p.element_id for p in sol.squad}
    assert {p.element_id for p in sol.starting} <= squad_ids
    assert not ({p.element_id for p in sol.bench} & {p.element_id for p in sol.starting})


@pytest.mark.parametrize("seed", range(6))
def test_solution_always_satisfies_every_constraint(seed: int, rules: Rules):
    """The single highest-value test here. Run over several random pools,
    because a formulation can be right for one and wrong for another."""
    assert_legal(solve(make_pool(seed), rules, horizon=HORIZON), rules)


def test_captain_chosen_per_gameweek_and_from_the_xi(rules: Rules):
    """The armband is free to change weekly, so it is a decision per gameweek
    rather than one fixed player -- which is worth more than a single doubling
    when two players are captain-grade on different fixtures."""
    sol = solve(make_pool(3), rules, horizon=HORIZON)
    assert set(sol.captains) == set(HORIZON)
    xi_ids = {p.element_id for p in sol.starting}
    for gw, captain in sol.captains.items():
        assert captain.element_id in xi_ids


def test_captain_is_the_best_starter_that_week(rules: Rules):
    sol = solve(make_pool(4), rules, horizon=HORIZON)
    for gw, captain in sol.captains.items():
        best = max(p.by_gameweek.get(gw, 0.0) for p in sol.starting)
        assert captain.by_gameweek.get(gw, 0.0) == pytest.approx(best)


def test_budget_is_binding(rules: Rules):
    """A pool with expensive players should push spend near the cap. If it does
    not, money is being stranded and the objective is wrong."""
    sol = solve(make_pool(5), rules, horizon=HORIZON)
    assert sol.spend > rules.budget * 0.9


def test_richer_pool_scores_at_least_as_well(rules: Rules):
    """Adding candidates can never make the optimum worse -- it only widens the
    feasible set. A regression here means the solve is not finding the optimum."""
    pool = make_pool(6)
    extra = [
        Projection(
            code=900 + i, element_id=900 + i, element_type=et, web_name=f"X{i}",
            team_id=i % 20 + 1, now_cost=45,
            by_gameweek={gw: 9.0 for gw in HORIZON},
        )
        for i, et in enumerate((1, 2, 3, 4, 2, 3))
    ]
    base = solve(pool, rules, horizon=HORIZON)
    richer = solve(pool + extra, rules, horizon=HORIZON)
    assert richer.objective >= base.objective - 1e-6


def test_infeasible_pool_raises_rather_than_returning_nonsense(rules: Rules):
    with pytest.raises(InfeasibleError):
        solve(make_pool(7, per_position=2), rules, horizon=HORIZON)


def test_empty_candidates_raises(rules: Rules):
    with pytest.raises(InfeasibleError):
        solve([], rules, horizon=HORIZON)


# -- transfers ------------------------------------------------------------


def test_no_transfers_taken_when_the_current_squad_is_already_optimal(rules: Rules):
    pool = make_pool(8)
    fresh = solve(pool, rules, horizon=HORIZON)
    current = [p.element_id for p in fresh.squad]

    held = solve(pool, rules, horizon=HORIZON, current=current, free_transfers=1)
    assert held.transfers_in == []
    assert held.hits == 0


def test_free_transfer_is_used_when_it_helps(rules: Rules):
    """A clearly better player at the same price and club should be brought in
    for free -- and exactly one, since only one transfer is free."""
    pool = make_pool(9)
    fresh = solve(pool, rules, horizon=HORIZON)
    current = [p.element_id for p in fresh.squad]

    weakest = min(fresh.starting, key=lambda p: p.ep_horizon)
    upgrade = Projection(
        code=999, element_id=999, element_type=weakest.element_type, web_name="Upgrade",
        team_id=weakest.team_id, now_cost=weakest.now_cost,
        by_gameweek={gw: weakest.by_gameweek[gw] + 3.0 for gw in HORIZON},
    )
    sol = solve(pool + [upgrade], rules, horizon=HORIZON, current=current, free_transfers=1)
    assert 999 in {p.element_id for p in sol.transfers_in}
    assert sol.hits == 0


def test_marginal_upgrade_does_not_justify_a_hit(rules: Rules):
    """A hit costs four points. A transfer worth a fraction of a point per week
    must not be taken -- this is the trap of acting on differences smaller than
    the model's own error."""
    pool = make_pool(10)
    fresh = solve(pool, rules, horizon=HORIZON)
    current = [p.element_id for p in fresh.squad]

    weakest = min(fresh.starting, key=lambda p: p.ep_horizon)
    marginal = [
        Projection(
            code=1000 + i, element_id=1000 + i, element_type=weakest.element_type,
            web_name=f"Marginal{i}", team_id=weakest.team_id, now_cost=weakest.now_cost,
            by_gameweek={gw: weakest.by_gameweek[gw] + 0.05 for gw in HORIZON},
        )
        for i in range(3)
    ]
    sol = solve(pool + marginal, rules, horizon=HORIZON, current=current, free_transfers=1)
    assert sol.hits == 0


def test_large_gain_does_justify_a_hit(rules: Rules):
    pool = make_pool(11)
    fresh = solve(pool, rules, horizon=HORIZON)
    current = [p.element_id for p in fresh.squad]

    targets = sorted(fresh.starting, key=lambda p: p.ep_horizon)[:2]
    huge = [
        Projection(
            code=1100 + i, element_id=1100 + i, element_type=t.element_type,
            web_name=f"Star{i}", team_id=t.team_id, now_cost=t.now_cost,
            by_gameweek={gw: t.by_gameweek[gw] + 8.0 for gw in HORIZON},
        )
        for i, t in enumerate(targets)
    ]
    sol = solve(pool + huge, rules, horizon=HORIZON, current=current, free_transfers=1)
    assert sol.hits >= 1
    assert len(sol.transfers_in) == len(sol.transfers_out)


def test_transfers_in_and_out_always_balance(rules: Rules):
    """Squad size is fixed, so every player in implies one out."""
    pool = make_pool(12)
    fresh = solve(pool, rules, horizon=HORIZON)
    current = [p.element_id for p in fresh.squad]
    sol = solve(pool, rules, horizon=HORIZON, current=current, free_transfers=2)
    assert len(sol.transfers_in) == len(sol.transfers_out)


def test_banked_transfers_allow_more_moves_without_hits(rules: Rules):
    pool = make_pool(13)
    fresh = solve(pool, rules, horizon=HORIZON)
    current = [p.element_id for p in fresh.squad]

    targets = sorted(fresh.starting, key=lambda p: p.ep_horizon)[:3]
    upgrades = [
        Projection(
            code=1200 + i, element_id=1200 + i, element_type=t.element_type,
            web_name=f"Up{i}", team_id=t.team_id, now_cost=t.now_cost,
            by_gameweek={gw: t.by_gameweek[gw] + 2.0 for gw in HORIZON},
        )
        for i, t in enumerate(targets)
    ]
    one = solve(pool + upgrades, rules, horizon=HORIZON, current=current, free_transfers=1)
    five = solve(pool + upgrades, rules, horizon=HORIZON, current=current, free_transfers=5)
    assert five.hits <= one.hits


# -- chips ----------------------------------------------------------------


def test_wildcard_ignores_the_transfer_penalty(rules: Rules):
    """Unlimited transfers, so a wildcard can never score worse than holding."""
    pool = make_pool(14)
    fresh = solve(pool, rules, horizon=HORIZON)
    current = [p.element_id for p in fresh.squad]

    held = solve(pool, rules, horizon=HORIZON, current=current, free_transfers=1)
    wild = solve(pool, rules, horizon=HORIZON, current=current, chip=Chip.WILDCARD)
    assert wild.objective >= held.objective - 1e-6
    assert wild.hits == 0


def test_free_hit_solves_a_single_gameweek(rules: Rules):
    """It reverts afterwards, so nothing beyond the first gameweek is kept and
    optimising for it would be wrong."""
    sol = solve(make_pool(15), rules, horizon=HORIZON, chip=Chip.FREE_HIT)
    assert set(sol.captains) == {HORIZON[0]}
    assert_legal(sol, rules)


def test_bench_boost_values_the_bench_fully(rules: Rules):
    """With all 15 scoring, the solver should stop buying cheap fodder."""
    pool = make_pool(16)
    normal = solve(pool, rules, horizon=HORIZON)
    boosted = solve(pool, rules, horizon=HORIZON, chip=Chip.BENCH_BOOST)
    normal_bench = sum(p.ep_horizon for p in normal.bench)
    boosted_bench = sum(p.ep_horizon for p in boosted.bench)
    assert boosted_bench > normal_bench


def test_triple_captain_beats_the_ordinary_armband(rules: Rules):
    pool = make_pool(17)
    normal = solve(pool, rules, horizon=HORIZON)
    tripled = solve(pool, rules, horizon=HORIZON, chip=Chip.TRIPLE_CAPTAIN)
    assert tripled.objective > normal.objective


@pytest.mark.parametrize("chip", [Chip.WILDCARD, Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN])
def test_chip_gain_is_never_negative(chip: Chip, rules: Rules):
    """Playing a chip cannot make this gameweek worse -- only the opportunity
    cost of not saving it, which is the stopping layer's problem, not this one."""
    assert chip_gain(make_pool(18), rules, chip, horizon=HORIZON) >= -1e-6


@pytest.mark.parametrize("chip", list(Chip))
def test_every_chip_mode_returns_a_legal_squad(chip: Chip, rules: Rules):
    assert_legal(solve(make_pool(19), rules, horizon=HORIZON, chip=chip), rules)


def test_force_include_is_honoured(rules: Rules):
    """Used to answer "what does insisting on this player cost me?" -- the
    question that showed captaincy flips the premium decision."""
    pool = make_pool(20)
    baseline = solve(pool, rules, horizon=HORIZON)
    outsider = min(
        (p for p in pool if p.element_id not in {q.element_id for q in baseline.squad}),
        key=lambda p: p.ep_horizon,
    )
    forced = solve(pool, rules, horizon=HORIZON, force_include=[outsider.element_id])
    assert outsider.element_id in {p.element_id for p in forced.starting}
    assert forced.objective <= baseline.objective + 1e-6


def test_bench_boost_gain_is_inflated_without_a_held_squad(rules: Rules):
    """A free solve buys itself a better bench and credits the chip for it.

    Measured against a squad you actually hold, the gain is what your real bench
    would score -- a much smaller and more honest number. This is the difference
    between "what is bench boost worth" and "what is a wildcard plus a bench
    boost worth", and only the first is a decision you get to make.
    """
    pool = make_pool(21)
    held = solve(pool, rules, horizon=HORIZON)
    current = [p.element_id for p in held.squad]

    free_gain = chip_gain(pool, rules, Chip.BENCH_BOOST, horizon=HORIZON)
    held_gain = chip_gain(
        pool, rules, Chip.BENCH_BOOST, horizon=HORIZON, current=current, free_transfers=1
    )
    assert free_gain > held_gain


def test_wildcard_is_worthless_when_the_squad_is_already_optimal(rules: Rules):
    """Wildcard value is distance-from-optimal, so it is zero at the optimum and
    grows as the squad drifts. That falls out of the formulation rather than
    needing to be modelled."""
    pool = make_pool(22)
    best = solve(pool, rules, horizon=HORIZON)
    current = [p.element_id for p in best.squad]
    gain = chip_gain(
        pool, rules, Chip.WILDCARD, horizon=HORIZON, current=current, free_transfers=1
    )
    assert gain == pytest.approx(0.0, abs=1e-6)


def test_wildcard_gains_when_the_squad_has_drifted(rules: Rules):
    """Force a deliberately poor starting squad; a wildcard should now be worth
    something."""
    pool = make_pool(23)
    cheap = sorted(pool, key=lambda p: p.ep_horizon)
    current = []
    for element_type, quota in rules.squad_select.items():
        current += [
            p.element_id
            for p in [q for q in cheap if q.element_type == element_type][:quota]
        ]
    gain = chip_gain(
        pool, rules, Chip.WILDCARD, horizon=HORIZON, current=current, free_transfers=1
    )
    assert gain > 0
