"""Your league, and what it changes about the problem.

The result that shapes this whole module: **effective ownership does not change
the expected-value optimum at all.**

Your score relative to the field is `your points - field points`, and by
linearity of expectation its mean is `E[your points] - E[field points]`. The
second term does not depend on anything you choose. So maximising expected
relative score and maximising expected absolute score give the identical squad,
and every "fade the template" argument phrased in terms of expected points is
simply wrong.

What ownership actually governs is **variance**. A player you and a rival both
own contributes nothing to the difference between your scores -- whatever they
return, you both get it. Only the players you hold that they do not, and they
hold that you do not, move you apart. So

    Var(your score - theirs) ~ sum of variance over the symmetric difference

and overlap is a variance dial, not a points dial. That makes the strategy
legible: ahead, you want low variance, so you maximise overlap and copy the
template; behind, you need variance, so you minimise overlap and take
differentials. Level, you ignore it entirely and maximise points.

That dial is linear in the selection variables, so it drops straight into the
existing ILP as a bonus term with a signed weight -- no simulation, and the
objective stays linear.

Eight managers is also what makes this worth doing. Against seven million you
would be estimating ownership; against seven you fetch it exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import structlog

log = structlog.get_logger()

#: Gameweeks remaining below which the standing is treated as close to final,
#: so the variance dial is turned up. Early on almost any gap is recoverable
#: through ordinary play and skewing the squad for it is pure cost.
ENDGAME_GAMEWEEKS = 12

#: Largest overlap weight applied, in points per gameweek. Deliberately modest:
#: this is a tilt on top of a points-maximising objective, not a replacement for
#: it, and a squad that chases variance while ignoring points loses on both.
MAX_OVERLAP_WEIGHT = 0.6


@dataclass(frozen=True)
class Standing:
    """Where you sit in the league, and how settled that is."""

    entry_id: int
    points: int
    #: Points of every rival, best first.
    rival_points: list[int]
    gameweeks_remaining: int

    @property
    def rank(self) -> int:
        return 1 + sum(1 for p in self.rival_points if p > self.points)

    @property
    def leader_gap(self) -> int:
        """Positive when you are behind the leader, negative when you lead."""
        best = max(self.rival_points, default=self.points)
        return best - self.points

    @property
    def is_leading(self) -> bool:
        return self.leader_gap < 0


def overlap_weight(standing: Standing) -> float:
    """Signed weight on squad overlap with rivals, in points per gameweek.

    Positive means "prefer players your rivals own" -- protect a lead by
    removing variance. Negative means "prefer players they do not" -- manufacture
    variance because the mean is not enough to catch up.

    Scaled by two things. The gap, because a two-point lead is not worth
    distorting a squad for. And the time remaining, because variance is only
    valuable when there is not enough football left to close a gap by playing
    well -- early in a season the honest answer is almost always to maximise
    points and let the table sort itself out.
    """
    if not standing.rival_points:
        return 0.0

    urgency = max(0.0, 1.0 - standing.gameweeks_remaining / ENDGAME_GAMEWEEKS)
    if urgency <= 0:
        return 0.0

    # A gap only matters relative to what a gameweek can swing; roughly 60 points
    # separates a good gameweek from a poor one across a squad.
    significance = min(1.0, abs(standing.leader_gap) / 60.0)
    magnitude = MAX_OVERLAP_WEIGHT * urgency * significance

    return magnitude if standing.is_leading else -magnitude


def ownership(
    rival_picks: dict[int, Sequence[int]], element_ids: Sequence[int]
) -> dict[int, float]:
    """Fraction of rivals holding each player.

    Exact, not estimated. This is the whole advantage of a small league.
    """
    if not rival_picks:
        return {}
    counts: dict[int, int] = {}
    for picks in rival_picks.values():
        for element_id in set(picks):
            counts[element_id] = counts.get(element_id, 0) + 1
    n = len(rival_picks)
    return {e: counts.get(e, 0) / n for e in element_ids}


def effective_ownership(
    rival_picks: dict[int, Sequence[int]],
    captains: dict[int, int],
    element_ids: Sequence[int],
) -> dict[int, float]:
    """Ownership weighted by multiplier, so a captained player counts twice.

    Effective ownership is what determines how much of a haul the field actually
    banks -- a 100%-owned player captained by half the league returns 150% of his
    score to the field, not 100%.

    Reported for context and used nowhere in the objective, because as the module
    docstring explains it cannot change the mean-optimal squad. Keeping the
    distinction visible stops it quietly creeping back in.
    """
    if not rival_picks:
        return {}
    weight: dict[int, float] = {}
    for entry_id, picks in rival_picks.items():
        captain = captains.get(entry_id)
        for element_id in set(picks):
            weight[element_id] = weight.get(element_id, 0.0) + (
                2.0 if element_id == captain else 1.0
            )
    n = len(rival_picks)
    return {e: weight.get(e, 0.0) / n for e in element_ids}


def describe(standing: Standing, weight: float) -> str:
    """One line explaining why the objective was tilted, for the recommendation.

    A squad that suddenly prefers template players needs to say so, or it reads
    as the model changing its mind.
    """
    if weight == 0:
        return (
            f"Rank {standing.rank} of {len(standing.rival_points) + 1}; "
            "maximising points, ignoring ownership"
        )
    if weight > 0:
        return (
            f"Leading by {abs(standing.leader_gap)} with "
            f"{standing.gameweeks_remaining} gameweeks left; favouring players "
            "your rivals own to reduce swing"
        )
    return (
        f"Behind by {standing.leader_gap} with {standing.gameweeks_remaining} "
        "gameweeks left; favouring differentials to create swing"
    )


def standing_from_history(
    entry_id: int,
    histories: dict[int, dict[str, Any]],
    gameweeks_remaining: int,
) -> Standing | None:
    """Build a Standing from each entry's history payload."""
    def total(history: dict[str, Any]) -> int | None:
        rows = history.get("current") or []
        return rows[-1].get("total_points") if rows else None

    mine = total(histories.get(entry_id, {}))
    if mine is None:
        return None

    rivals = [
        t
        for other, history in histories.items()
        if other != entry_id and (t := total(history)) is not None
    ]
    return Standing(
        entry_id=entry_id,
        points=mine,
        rival_points=rivals,
        gameweeks_remaining=gameweeks_remaining,
    )
