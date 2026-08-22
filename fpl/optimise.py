"""Squad selection as an integer linear program.

The objective is linear because expectation is linear -- the same property that
lets `project.py` decompose points into one term per category makes the total a
plain weighted sum here, which is what the whole LP relaxation and branch-and-
bound machinery needs. Optimise a variance-penalised or quantile objective
instead and none of this applies; you would be solving many linear problems and
voting rather than one non-linear one.

Greedy selection fails on this problem because the constraints interact. Sorting
by points per million strands budget you cannot spend, or fills the three-per-
club limit before reaching the players you actually want. Only a global solve
sees that.

Every structural rule -- squad size, positional quotas, budget, club limit --
is read from the API rather than hardcoded, so a rule change arrives with the
data instead of requiring a release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from fpl.project import Projection

#: Points deducted per transfer beyond the free allowance.
TRANSFER_COST = 4


class Chip(StrEnum):
    NONE = "none"
    WILDCARD = "wildcard"
    """Unlimited transfers, squad changes persist. Modelled by dropping the
    transfer penalty entirely."""
    FREE_HIT = "freehit"
    """Unlimited transfers for one gameweek, then reverts. Same as a wildcard but
    solved over a single-gameweek horizon, because nothing after it is kept."""
    BENCH_BOOST = "bboost"
    """All 15 score. Modelled by weighting the bench equally with the XI."""
    TRIPLE_CAPTAIN = "3xc"
    """Captain scores treble rather than double."""


@dataclass(frozen=True)
class Rules:
    """Structural constraints, read from bootstrap-static."""

    squad_size: int
    starting_size: int
    budget: int
    team_limit: int
    squad_select: dict[int, int]
    play_min: dict[int, int]
    play_max: dict[int, int]

    @classmethod
    def from_bootstrap(cls, boot: dict[str, Any]) -> Rules:
        settings = boot["game_settings"]
        types = boot["element_types"]
        return cls(
            squad_size=settings["squad_squadsize"],
            starting_size=settings["squad_squadplay"],
            budget=settings["squad_total_spend"],
            team_limit=settings["squad_team_limit"],
            squad_select={t["id"]: t["squad_select"] for t in types},
            play_min={t["id"]: t["squad_min_play"] for t in types},
            play_max={t["id"]: t["squad_max_play"] for t in types},
        )


@dataclass(frozen=True)
class Solution:
    squad: list[Projection]
    starting: list[Projection]
    bench: list[Projection]
    #: Gameweek -> the player captained that week. The armband is re-chosen each
    #: week at no cost, so it is a per-gameweek decision, not a fixed one.
    captains: dict[int, Projection]
    transfers_in: list[Projection]
    transfers_out: list[Projection]
    hits: int
    objective: float
    spend: int
    chip: Chip = Chip.NONE
    notes: list[str] = field(default_factory=list)

    @property
    def hit_cost(self) -> int:
        return self.hits * TRANSFER_COST


class InfeasibleError(RuntimeError):
    """No legal squad exists under these constraints.

    Usually means the candidate pool is too thin -- not enough projected players
    in a position, or nothing affordable after the club limit bites.
    """


def _captain_multiplier(chip: Chip) -> int:
    return 3 if chip is Chip.TRIPLE_CAPTAIN else 2


def solve(
    candidates: Sequence[Projection],
    rules: Rules,
    *,
    horizon: Sequence[int],
    bench_weight: float = 0.12,
    current: Iterable[int] = (),
    free_transfers: int = 1,
    budget: int | None = None,
    chip: Chip = Chip.NONE,
    force_include: Iterable[int] = (),
    costs: dict[int, int] | None = None,
    overlap: dict[int, float] | None = None,
    overlap_weight: float = 0.0,
) -> Solution:
    """Best squad under the rules.

    With `current` empty this is a from-scratch selection: the opening squad, or
    what a wildcard would buy. With `current` populated it is a transfer
    decision, and moves beyond `free_transfers` are charged against the objective
    so the solver decides for itself whether a hit pays for itself.

    `bench_weight` values bench players below starters without ignoring them --
    they matter for rotation and injuries but rarely score. A bench boost sets it
    to 1.0, which is exactly what that chip means.
    """
    if not candidates:
        raise InfeasibleError("no candidates")

    gameweeks = list(horizon)
    if chip is Chip.FREE_HIT:
        gameweeks = gameweeks[:1]
    if not gameweeks:
        raise InfeasibleError("empty horizon")

    if chip is Chip.BENCH_BOOST:
        bench_weight = 1.0

    n = len(candidates)
    g = len(gameweeks)
    current_ids = set(current)
    unlimited = chip in (Chip.WILDCARD, Chip.FREE_HIT) or not current_ids

    # Held players are costed at what selling them actually returns, which is
    # below market once they have risen. Costing them at market while
    # budgeting from FPL's `value` -- which is net of the fee -- makes the
    # constraint tight enough to reject the squad you already own.
    costs = costs or {}
    cost = np.array(
        [costs.get(p.element_id, p.now_cost) for p in candidates], dtype=float
    )
    pos = np.array([p.element_type for p in candidates])
    team = np.array([p.team_id for p in candidates])
    weekly_ep = np.array([
        [p.by_gameweek.get(gw, 0.0) for gw in gameweeks] for p in candidates
    ])

    total_budget = rules.budget if budget is None else budget

    # Variables: x (in squad) | y (starting, per gameweek) | z (captain, per
    # gameweek) | h (hits).
    #
    # The XI is indexed by gameweek because you re-pick it every week at no
    # cost, so choosing one XI for a whole horizon optimises the wrong decision
    # -- it will start a player whose team has no fixture this week on the
    # strength of good fixtures later. It also lets the squad be valued the way
    # it is actually used: a squad that rotates well is worth more than the sum
    # of one fixed eleven.
    x0, y0, z0 = 0, n, n + n * g
    h0 = z0 + n * g
    n_vars = h0 + (0 if unlimited else 1)

    objective = np.zeros(n_vars)
    # Starters score in full; a bench place is worth something but rarely much.
    objective[y0:y0 + n * g] = ((1 - bench_weight) * weekly_ep / g).flatten()
    objective[x0:x0 + n] = bench_weight * weekly_ep.mean(axis=1)
    # Captaincy adds the player's points a second (or third) time.
    bonus = _captain_multiplier(chip) - 1
    objective[z0:z0 + n * g] = (weekly_ep * bonus / g).flatten()

    # Overlap with rivals is a variance dial, not a points dial. By linearity
    # of expectation the field's expected score does not depend on your
    # choices, so ownership cannot move the mean-optimal squad -- but players
    # you share with a rival cancel out of the difference between your scores,
    # so overlap is exactly what governs how far apart you can drift. Positive
    # weight protects a lead, negative manufactures swing to chase one.
    if overlap and overlap_weight:
        share = np.array([overlap.get(p.element_id, 0.0) for p in candidates])
        objective[x0:x0 + n] += overlap_weight * share

    if not unlimited:
        objective[h0] = -TRANSFER_COST

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    r = 0

    def constrain(entries: list[tuple[int, float]], lo: float, hi: float) -> None:
        nonlocal r
        for col, val in entries:
            rows.append(r)
            cols.append(col)
            vals.append(val)
        lower.append(lo)
        upper.append(hi)
        r += 1

    def yi(i: int, k: int) -> int:
        return y0 + i * g + k

    def zi(i: int, k: int) -> int:
        return z0 + i * g + k

    constrain([(x0 + i, 1.0) for i in range(n)], rules.squad_size, rules.squad_size)
    constrain([(x0 + i, cost[i]) for i in range(n)], 0, total_budget)

    for element_type, quota in rules.squad_select.items():
        members = [i for i in range(n) if pos[i] == element_type]
        constrain([(x0 + i, 1.0) for i in members], quota, quota)

    for club in np.unique(team):
        members = [i for i in range(n) if team[i] == club]
        constrain([(x0 + i, 1.0) for i in members], 0, rules.team_limit)

    for k in range(g):
        # A legal eleven every gameweek, not once across the horizon.
        constrain([(yi(i, k), 1.0) for i in range(n)], rules.starting_size, rules.starting_size)
        for element_type in rules.squad_select:
            members = [i for i in range(n) if pos[i] == element_type]
            constrain(
                [(yi(i, k), 1.0) for i in members],
                rules.play_min[element_type],
                rules.play_max[element_type],
            )
        # Exactly one captain per gameweek.
        constrain([(zi(i, k), 1.0) for i in range(n)], 1, 1)

    for i in range(n):
        for k in range(g):
            constrain([(yi(i, k), 1.0), (x0 + i, -1.0)], -np.inf, 0)   # start => own
            constrain([(zi(i, k), 1.0), (yi(i, k), -1.0)], -np.inf, 0)  # captain => start

    if not unlimited:
        # hits >= (players bought) - free_transfers, and hits >= 0. Linearises
        # max(0, n_transfers - free) without a max().
        bought = [i for i in range(n) if candidates[i].element_id not in current_ids]
        constrain(
            [(x0 + i, 1.0) for i in bought] + [(h0, -1.0)], -np.inf, float(free_transfers)
        )

    for element_id in force_include:
        matches = [i for i in range(n) if candidates[i].element_id == element_id]
        if matches:
            constrain([(x0 + matches[0], 1.0)], 1, 1)

    matrix = coo_matrix((vals, (rows, cols)), shape=(r, n_vars)).tocsr()

    integrality = np.ones(n_vars)
    ub = np.ones(n_vars)
    if not unlimited:
        ub[h0] = rules.squad_size  # at most a full teardown

    result = milp(
        c=-objective,
        constraints=LinearConstraint(matrix, lower, upper),
        integrality=integrality,
        bounds=Bounds(np.zeros(n_vars), ub),
    )
    if not result.success:
        raise InfeasibleError(result.message)

    picked = result.x[x0:x0 + n] > 0.5
    y = result.x[y0:y0 + n * g].reshape(n, g)
    z = result.x[z0:z0 + n * g].reshape(n, g)
    # The XI reported is the one for the imminent gameweek -- the only one you
    # are actually about to submit. Later gameweeks' elevens are solved for, so
    # squad depth is valued correctly, but they are not decisions yet.
    starting = y[:, 0] > 0.5

    squad = [candidates[i] for i in range(n) if picked[i]]
    xi = [candidates[i] for i in range(n) if starting[i]]
    bench = [candidates[i] for i in range(n) if picked[i] and not starting[i]]
    captains = {gameweeks[k]: candidates[int(np.argmax(z[:, k]))] for k in range(g)}

    squad_ids = {p.element_id for p in squad}
    transfers_in = [p for p in squad if p.element_id not in current_ids] if current_ids else []
    transfers_out_ids = current_ids - squad_ids if current_ids else set()
    hits = 0 if unlimited else max(0, len(transfers_in) - free_transfers)

    notes: list[str] = []
    if chip is not Chip.NONE:
        notes.append(f"solved with {chip} active")
    if hits:
        notes.append(f"{hits} hit(s) taken, costing {hits * TRANSFER_COST} points")

    return Solution(
        squad=sorted(squad, key=lambda p: (p.element_type, -p.ep_horizon)),
        starting=sorted(xi, key=lambda p: (p.element_type, -p.ep_horizon)),
        bench=sorted(bench, key=lambda p: -p.ep_horizon),
        captains=captains,
        transfers_in=transfers_in,
        transfers_out=[p for p in candidates if p.element_id in transfers_out_ids],
        hits=hits,
        objective=float(result.fun * -1),
        spend=int(sum(p.now_cost for p in squad)),
        chip=chip,
        notes=notes,
    )


def chip_gain(
    candidates: Sequence[Projection],
    rules: Rules,
    chip: Chip,
    **kwargs: Any,
) -> float:
    """How many points playing `chip` this gameweek would add.

    Two solves and a subtraction. This scalar is the entire interface to the
    stopping policy -- 599 players, Poisson, shrinkage and the solver all
    collapse to one number per chip per gameweek, which is why the timing
    decision is separable and testable on synthetic inputs.

    **Pass `current` if you own a squad.** Without it both solves are free to
    buy any fifteen players, so the answer is the value of "wildcard, then play
    the chip" rather than the value of playing the chip. Bench boost flatters
    itself worst under that mistake: a free solve simply buys a better bench and
    credits the chip with the difference. Measured against a held squad it
    reports what your actual bench would score, which is the decision you face.

    Deciding *when* to spend it is not this function's job. That is an optimal
    stopping problem, and it needs the distribution of future gains rather than
    today's value.
    """
    baseline = solve(candidates, rules, chip=Chip.NONE, **kwargs)
    with_chip = solve(candidates, rules, chip=chip, **kwargs)
    return with_chip.objective - baseline.objective
