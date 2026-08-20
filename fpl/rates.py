"""Per-90 player rates from prior-season totals.

Shrinkage here does two jobs that a single constant used to conflate, and
conflating them measurably distorted the model: elite players' edge over the
pack was compressed by roughly 40%, which changed whether a premium striker was
worth buying at all.

`noise_k` corrects sampling error. A rate measured over few minutes is
unreliable and belongs closer to the positional mean; a rate measured over a full
season is well determined and should barely move. Weight is
minutes / (minutes + noise_k), which approaches 1 for a regular starter.

`regression` is a flat multiplicative pull applied to everyone regardless of
sample size. It captures the genuine tendency of last season's outliers to
decline, which has nothing to do with how precisely they were measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from fpl.config import Shrinkage

#: Rates modelled per 90 minutes. `dc` is the raw defensive-action count, which
#: for defenders is clearances+blocks+interceptions+tackles and for midfielders
#: and forwards additionally includes recoveries -- verified against the API,
#: where the totals reconcile exactly.
RATE_KEYS = ("xg", "xa", "dc", "bonus", "saves", "yellow")

_SOURCE_FIELDS = {
    "xg": "expected_goals",
    "xa": "expected_assists",
    "dc": "defensive_contribution",
    "bonus": "bonus",
    "saves": "saves",
    "yellow": "yellow_cards",
}

#: A full season is 38 matches. Used to turn totals into per-match expectations.
MATCHES_PER_SEASON = 38
#: Starters rarely play the full 90; this converts P(start) into P(60+ minutes),
#: which is what the appearance and clean-sheet points actually hinge on.
START_TO_SIXTY = 0.92


@dataclass(frozen=True)
class PlayerRates:
    code: int
    element_type: int
    minutes: int
    starts: int
    total_points: int
    per90: dict[str, float]
    #: True when these came from a price-based prior rather than observed play.
    imputed: bool = False

    @property
    def expected_minutes(self) -> float:
        """Minutes per match, averaged over a whole season including absences."""
        return self.minutes / MATCHES_PER_SEASON

    @property
    def minutes_scale(self) -> float:
        """Per-90 rates -> per-match expectations."""
        return self.expected_minutes / 90.0

    @property
    def p_start(self) -> float:
        return min(1.0, self.starts / MATCHES_PER_SEASON)

    @property
    def p_sixty(self) -> float:
        return min(1.0, self.p_start * START_TO_SIXTY)

    @property
    def points_per_game(self) -> float:
        """Prior-season PPG -- one of the baselines every projection is scored
        against, and the dumbest model worth beating."""
        return self.total_points / MATCHES_PER_SEASON


def extract(code: int, element_type: int, season: dict[str, Any]) -> PlayerRates | None:
    """Turn one `history_past` row into per-90 rates."""
    minutes = season.get("minutes") or 0
    if minutes < 1:
        return None
    per90 = {
        key: float(season.get(field) or 0) * 90.0 / minutes
        for key, field in _SOURCE_FIELDS.items()
    }
    return PlayerRates(
        code=code,
        element_type=element_type,
        minutes=minutes,
        starts=season.get("starts") or 0,
        total_points=season.get("total_points") or 0,
        per90=per90,
    )


def positional_means(rates: Iterable[PlayerRates], min_minutes: int = 600) -> dict[int, dict[str, float]]:
    """Minutes-weighted mean rate per position, the target of shrinkage.

    Restricted to players with real minutes so the mean describes footballers who
    actually play, not the long tail of substitutes and academy names that would
    otherwise drag every prior toward zero.
    """
    buckets: dict[int, list[PlayerRates]] = {}
    for r in rates:
        if r.minutes >= min_minutes and not r.imputed:
            buckets.setdefault(r.element_type, []).append(r)

    means: dict[int, dict[str, float]] = {}
    for element_type, group in buckets.items():
        total = sum(r.minutes for r in group)
        means[element_type] = {
            key: sum(r.per90[key] * r.minutes for r in group) / total for key in RATE_KEYS
        }
    return means


def shrink(
    rates: PlayerRates,
    means: dict[int, dict[str, float]],
    shrinkage: Shrinkage,
) -> dict[str, float]:
    """Apply both corrections and return usable per-90 rates."""
    mean = means.get(rates.element_type)
    if mean is None:
        return dict(rates.per90)

    weight = rates.minutes / (rates.minutes + shrinkage.noise_k)
    return {
        key: (weight * rates.per90[key] + (1 - weight) * mean[key]) * shrinkage.regression
        for key in RATE_KEYS
    }


def price_prior(
    element_type: int,
    now_cost: int,
    observed: list[tuple[int, PlayerRates]],
) -> PlayerRates | None:
    """Rates for a player with no prior-season data, inferred from price.

    Roughly a quarter of the player pool has never appeared in the league --
    promoted-club squads, signings from abroad, academy graduates. Excluding them
    outright leaves a hole the optimiser cannot see into, and the hole grows as
    January signings arrive.

    FPL prices new players according to its own expectation of them, which makes
    price a serviceable proxy. This averages the observed rates of same-position
    players within a narrow price band, widening the band until enough comparable
    players are found.

    Deliberately conservative: it returns the *average* of comparable players, so
    an unknown never outranks a known quantity at the same price. That is the
    right bias -- it puts promoted-team players back in contention at modest
    projections rather than letting the model fall in love with a stranger.
    """
    same_position = [r for cost, r in observed if r.element_type == element_type and not r.imputed]
    if not same_position:
        return None

    for band in (5, 10, 15, 25):
        peers = [
            r
            for cost, r in observed
            if r.element_type == element_type
            and not r.imputed
            and abs(cost - now_cost) <= band
            and r.minutes >= 900
        ]
        if len(peers) >= 5:
            break
    else:
        peers = [r for r in same_position if r.minutes >= 900]
        if not peers:
            return None

    n = len(peers)
    return PlayerRates(
        code=-1,
        element_type=element_type,
        minutes=int(sum(r.minutes for r in peers) / n),
        starts=int(sum(r.starts for r in peers) / n),
        total_points=int(sum(r.total_points for r in peers) / n),
        per90={key: sum(r.per90[key] for r in peers) / n for key in RATE_KEYS},
        imputed=True,
    )
