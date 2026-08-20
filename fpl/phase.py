"""Gameweek phase, derived rather than stored.

Every transition here is already published by the API: `finished`,
`data_checked`, `is_current`, `is_next` and `deadline_time_epoch`. Deriving the
phase from a snapshot plus the clock means it cannot drift after a missed job or
a postponement, which a stored phase would.

This is also what lets the cron schedule stay dumb. FPL deadlines move constantly
with TV scheduling, so the decide job fires hourly and gates itself on the phase
rather than trying to encode a moving target in a cron expression.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

#: Hours before a deadline at which team news starts arriving in volume
#: (press conferences) and the availability poller becomes worth running.
NEWS_WINDOW_HOURS = 48
#: Hours before a deadline at which the decide job produces the final answer.
PRE_DEADLINE_HOURS = 3


class Phase(StrEnum):
    PRESEASON = "PRESEASON"
    """No gameweek has finished. Prior-season priors are all there is."""

    SETTLING = "SETTLING"
    """Matches played, stats not yet final. Do not refit on partial data."""

    PLANNING = "PLANNING"
    """Data settled, deadline distant. Refit and draft a plan."""

    NEWS_WINDOW = "NEWS_WINDOW"
    """Within 48h. Availability polling matters; the plan may still move."""

    PRE_DEADLINE = "PRE_DEADLINE"
    """Within 3h. Produce the answer that gets acted on."""

    LOCKED = "LOCKED"
    """Deadline passed, gameweek in progress. Nothing to decide."""


@dataclass(frozen=True)
class PhaseState:
    phase: Phase
    next_gameweek: int | None
    next_deadline: datetime | None
    current_gameweek: int | None
    #: Highest gameweek whose stats FPL has finalised. Refits must not use data
    #: newer than this, and must not re-run if it has not advanced.
    last_settled_gameweek: int | None

    @property
    def hours_to_deadline(self) -> float | None:
        if self.next_deadline is None:
            return None
        return (self.next_deadline - datetime.now(UTC)).total_seconds() / 3600


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def derive_phase(events: list[dict[str, Any]], now: datetime | None = None) -> PhaseState:
    """Compute the phase from a bootstrap `events` list.

    `now` is injectable so the whole state machine is testable without patching
    the clock, and so a backtest can ask what the phase was at any past instant.
    """
    now = now or datetime.now(UTC)

    settled = [e["id"] for e in events if e.get("finished") and e.get("data_checked")]
    last_settled = max(settled) if settled else None

    current = next((e for e in events if e.get("is_current")), None)
    upcoming = sorted(
        (e for e in events if _parse(e["deadline_time"]) > now),
        key=lambda e: e["deadline_time"],
    )
    nxt = upcoming[0] if upcoming else None

    next_gw = nxt["id"] if nxt else None
    next_deadline = _parse(nxt["deadline_time"]) if nxt else None
    current_gw = current["id"] if current else None

    def state(phase: Phase) -> PhaseState:
        return PhaseState(
            phase=phase,
            next_gameweek=next_gw,
            next_deadline=next_deadline,
            current_gameweek=current_gw,
            last_settled_gameweek=last_settled,
        )

    # A finished gameweek whose stats are not final yet. Bonus points and the
    # defensive-contribution tallies are still moving; refitting now would train
    # on numbers that are about to change.
    if current is not None and current.get("finished") and not current.get("data_checked"):
        return state(Phase.SETTLING)

    if next_deadline is None:
        return state(Phase.LOCKED)

    hours = (next_deadline - now).total_seconds() / 3600

    if hours <= 0:
        return state(Phase.LOCKED)
    if hours <= PRE_DEADLINE_HOURS:
        return state(Phase.PRE_DEADLINE)
    if hours <= NEWS_WINDOW_HOURS:
        return state(Phase.NEWS_WINDOW)
    if last_settled is None:
        return state(Phase.PRESEASON)
    return state(Phase.PLANNING)
