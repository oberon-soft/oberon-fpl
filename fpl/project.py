"""Expected points per player per gameweek.

The whole calculation rests on linearity of expectation. Enumerating joint
outcomes -- played 90 *and* scored one *and* kept a clean sheet *and* reached the
defensive threshold -- is a combinatorial explosion. But E[A + B] = E[A] + E[B]
holds whether or not A and B are independent, so expected points decomposes into
one term per scoring category and the joint distribution is never needed.

That is worth stating explicitly because it is why the mean is cheap and
everything else is not. Clean sheets and goals conceded are the same event viewed
twice, and it does not matter here at all. Ask for the *spread* of points instead
and those correlations become load-bearing, no closed form survives, and you are
simulating.

Two terms have no closed form even for the mean. Bonus is a rank statistic across
all 22 players on the pitch, so it is approximated from the player's own prior
rate rather than derived. Saves are scaled by how much shooting the opponent is
expected to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fpl.config import Config
from fpl.rates import PlayerRates
from fpl.scoring import (
    CONCEDED_PER_POINT,
    SAVES_PER_POINT,
    Scoring,
    clean_sheet_probability,
    threshold_probability,
)

GOALKEEPER = 1
DEFENDER = 2


@dataclass(frozen=True)
class Projection:
    code: int
    element_id: int
    element_type: int
    web_name: str
    team_id: int
    now_cost: int
    #: Expected points per gameweek across the horizon, in gameweek order.
    by_gameweek: dict[int, float]
    imputed: bool = False
    availability: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def next_event(self) -> int | None:
        return min(self.by_gameweek) if self.by_gameweek else None

    @property
    def ep_next(self) -> float:
        gw = self.next_event
        return self.by_gameweek[gw] if gw is not None else 0.0

    @property
    def ep_horizon(self) -> float:
        if not self.by_gameweek:
            return 0.0
        return sum(self.by_gameweek.values()) / len(self.by_gameweek)


def expected_points(
    rates: PlayerRates,
    shrunk: dict[str, float],
    element_type: int,
    difficulty: int,
    scoring: Scoring,
    config: Config,
) -> float:
    """Expected points for one player in one fixture.

    Each term is a rate times a coefficient. Nothing here inspects a joint
    outcome, because linearity of expectation means nothing has to.
    """
    attack = config.fixture.attack[difficulty]
    opponent_xg = config.fixture.opponent_xg[difficulty]
    scale = rates.minutes_scale
    p60 = rates.p_sixty

    # Appearance. Two points for an hour, one for anything less. The sub term is
    # a small allowance for cameos beyond the starts we can observe.
    points = p60 * scoring.long_play
    points += max(0.0, rates.p_start - p60 + 0.08) * scoring.short_play

    # Attacking returns, fixture-adjusted and scaled to expected minutes.
    points += shrunk["xg"] * attack * scale * scoring.goal(element_type)
    points += shrunk["xa"] * attack * scale * scoring.assist(element_type)

    # Clean sheet: P(opponent scores zero), conditional on playing the hour that
    # qualifies for it.
    points += clean_sheet_probability(opponent_xg) * p60 * scoring.clean_sheet(element_type)

    if element_type in (GOALKEEPER, DEFENDER):
        conceded = scoring.conceded(element_type) / CONCEDED_PER_POINT
        points += opponent_xg * scale * conceded

    # Defensive contribution: all-or-nothing at a threshold, so this is
    # P(Poisson count reaches it) rather than a rate times a coefficient.
    threshold = config.squad.dc_threshold.get(element_type)
    if threshold is not None:
        hit = threshold_probability(shrunk["dc"] * scale, threshold)
        points += hit * scoring.defensive_contribution(element_type) * rates.p_start

    if element_type == GOALKEEPER:
        # More shots faced means more saves. Scaled against an average fixture.
        shot_volume = opponent_xg / config.fixture.opponent_xg[3]
        points += (shrunk["saves"] * shot_volume * scale) * scoring.save / SAVES_PER_POINT

    # Bonus has no closed form -- it is a rank across all 22 players on the pitch
    # -- so the player's own prior rate stands in for it.
    points += shrunk["bonus"] * scale
    points += shrunk["yellow"] * scale * scoring.yellow_card

    return max(0.0, points)


def project_player(
    *,
    element: dict,
    rates: PlayerRates,
    shrunk: dict[str, float],
    team_fixtures: list[tuple[int, int]],
    scoring: Scoring,
    config: Config,
    availability: float,
    missed_events: set[int] = frozenset(),
    ep_multiplier: float | None = None,
) -> Projection:
    """Expected points across the horizon for one player.

    `missed_events` and `ep_multiplier` come from the overrides table. They are
    applied here, at the end, rather than folded into the rates -- a suspension
    is a statement about specific gameweeks, not about how good a player is.
    """
    element_type = element["element_type"]
    by_gameweek: dict[int, float] = {}
    notes: list[str] = []

    for gameweek, difficulty in team_fixtures:
        if gameweek in missed_events:
            by_gameweek[gameweek] = 0.0
            continue
        points = expected_points(rates, shrunk, element_type, difficulty, scoring, config)
        points *= availability
        if ep_multiplier is not None:
            points *= ep_multiplier
        # Doubles accumulate: a team playing twice in one gameweek scores twice.
        by_gameweek[gameweek] = by_gameweek.get(gameweek, 0.0) + points

    if missed_events:
        notes.append(f"unavailable for GW{sorted(missed_events)}")
    if rates.imputed:
        notes.append("no prior-season data; rates inferred from price")
    if availability < 1.0:
        notes.append(f"availability {availability:.0%}")

    return Projection(
        code=element["code"],
        element_id=element["id"],
        element_type=element_type,
        web_name=element["web_name"],
        team_id=element["team"],
        now_cost=element["now_cost"],
        by_gameweek=by_gameweek,
        imputed=rates.imputed,
        availability=availability,
        notes=notes,
    )
