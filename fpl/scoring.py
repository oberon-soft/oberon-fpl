"""Points, read from the game rather than hardcoded.

`game_config.scoring` is a dict keyed by position, so the points function is a
dot product against it. If FPL retunes a coefficient mid-season the model follows
automatically -- and the scheduled contract test catches any change to the
*shape* that this code would not survive.

Two ratios are not exposed anywhere in the API and have to be asserted: saves
score one point per three, and goals conceded cost one point per two. They are
constants here rather than magic numbers inline, so there is one place to change
them when the contract test eventually fires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: Saves per point, for goalkeepers.
SAVES_PER_POINT = 3
#: Goals conceded per point deducted, for goalkeepers and defenders.
CONCEDED_PER_POINT = 2


@dataclass(frozen=True)
class Scoring:
    """The scoring table for one season, as published by the API."""

    raw: dict[str, Any]
    position_name: dict[int, str]

    @classmethod
    def from_bootstrap(cls, boot: dict[str, Any]) -> Scoring:
        return cls(
            raw=boot["game_config"]["scoring"],
            position_name={
                t["id"]: t["singular_name_short"] for t in boot["element_types"]
            },
        )

    def _by_position(self, key: str, element_type: int) -> float:
        value = self.raw[key]
        if isinstance(value, dict):
            return float(value[self.position_name[element_type]])
        return float(value)

    def goal(self, element_type: int) -> float:
        return self._by_position("goals_scored", element_type)

    def assist(self, element_type: int) -> float:
        return self._by_position("assists", element_type)

    def clean_sheet(self, element_type: int) -> float:
        return self._by_position("clean_sheets", element_type)

    def defensive_contribution(self, element_type: int) -> float:
        return self._by_position("defensive_contribution", element_type)

    def conceded(self, element_type: int) -> float:
        return self._by_position("goals_conceded", element_type)

    @property
    def long_play(self) -> float:
        return float(self.raw["long_play"])

    @property
    def short_play(self) -> float:
        return float(self.raw["short_play"])

    @property
    def save(self) -> float:
        return float(self.raw["saves"])

    @property
    def yellow_card(self) -> float:
        return float(self.raw["yellow_cards"])


def clean_sheet_probability(opponent_xg: float) -> float:
    """P(opponent scores zero), from a Poisson with mean `opponent_xg`.

    Goals arrive as rare, roughly independent events at a roughly steady rate,
    which is what a Poisson describes. The k=0 case collapses the sum to a single
    term, so this needs no series and no library.

    Real football violates independence slightly at low scores -- 0-0 and 1-1
    happen a little more often than this predicts, because a team two goals up
    stops pushing. Dixon-Coles corrects exactly that. The correction is small
    against the error in our estimate of `opponent_xg`, so it is deliberately
    not applied.
    """
    return math.exp(-opponent_xg)


def threshold_probability(rate: float, threshold: int) -> float:
    """P(a Poisson count with mean `rate` reaches `threshold`).

    Used for defensive contribution, where the points are all-or-nothing: a
    defender scores two for reaching ten defensive actions and zero for nine.

    This is why DC is the highest-value thing a simple model can exploit. Goals
    are rare, so a defender's xG is nearly pure noise over one season. Defensive
    actions fire most matches, so the rate is well measured from a small sample,
    and the payoff is near-deterministic for the right player.
    """
    if rate <= 0:
        return 0.0
    # P(X >= k) = 1 - P(X <= k-1), summed directly. `threshold` is ~10-12, so
    # this is a dozen terms and needs no special function.
    cumulative = 0.0
    term = math.exp(-rate)
    for k in range(threshold):
        if k:
            term *= rate / k
        cumulative += term
    return max(0.0, 1.0 - cumulative)
