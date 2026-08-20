"""Source freshness and the gate that decides whether to recommend anything.

A source is never stale in the abstract -- only relative to what you are about to
do. Availability flags are irrelevant while planning and critical three hours
before a deadline. So tolerance is a function of phase, and most cells in the
table below are generous; only the PRE_DEADLINE column is tight.

The invariant that matters: BLOCKED must never silently degrade into a
recommendation. A system that quietly falls back to last week's numbers at 2am
before a deadline is worse than one that says it is broken.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from fpl.phase import Phase


class Source(StrEnum):
    BOOTSTRAP = "bootstrap"
    FIXTURES = "fixtures"
    PLAYER_HISTORY = "player_history"
    OWN_SQUAD = "own_squad"
    RATES = "rates"
    """A derived artefact, but an input all the same -- rates fitted before the
    last settled gameweek are as stale as an unfetched endpoint."""


class Status(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    INVALID = "INVALID"
    """Fetched, but malformed or violating an invariant."""
    UNINFORMATIVE = "UNINFORMATIVE"
    """Well-formed and carrying no signal. Preseason zeroes look FRESH to any
    check based on status code or schema; this is the category that catches
    them before the model confidently produces nonsense."""
    UNKNOWN = "UNKNOWN"


class Verdict(StrEnum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"


#: Maximum acceptable age in hours, by phase and source. `None` means the source
#: is not required in that phase and its age is ignored entirely.
TOLERANCE: dict[Phase, dict[Source, float | None]] = {
    Phase.PRESEASON: {
        Source.BOOTSTRAP: 48, Source.FIXTURES: 168,
        Source.PLAYER_HISTORY: None, Source.OWN_SQUAD: None, Source.RATES: None,
    },
    Phase.SETTLING: {
        Source.BOOTSTRAP: 48, Source.FIXTURES: 168,
        Source.PLAYER_HISTORY: None, Source.OWN_SQUAD: None, Source.RATES: None,
    },
    Phase.PLANNING: {
        Source.BOOTSTRAP: 36, Source.FIXTURES: 168,
        Source.PLAYER_HISTORY: 168, Source.OWN_SQUAD: 336, Source.RATES: 168,
    },
    Phase.NEWS_WINDOW: {
        Source.BOOTSTRAP: 12, Source.FIXTURES: 168,
        Source.PLAYER_HISTORY: 168, Source.OWN_SQUAD: 336, Source.RATES: 168,
    },
    Phase.PRE_DEADLINE: {
        Source.BOOTSTRAP: 3, Source.FIXTURES: 72,
        Source.PLAYER_HISTORY: 168, Source.OWN_SQUAD: 336, Source.RATES: 168,
    },
    Phase.LOCKED: {
        Source.BOOTSTRAP: 72, Source.FIXTURES: 168,
        Source.PLAYER_HISTORY: None, Source.OWN_SQUAD: None, Source.RATES: None,
    },
}

#: Sources whose staleness degrades a recommendation but does not block it.
#: Everything else, when required by the phase, blocks.
DEGRADE_ONLY: frozenset[Source] = frozenset({Source.OWN_SQUAD, Source.PLAYER_HISTORY})


@dataclass(frozen=True)
class SourceState:
    source: Source
    status: Status
    #: When we fetched it.
    fetched_at: datetime | None
    #: When the data itself was current. Differs from fetched_at for anything
    #: with its own timestamp -- odds pulled ten minutes ago may reflect a line
    #: from two days back, which is stale data freshly fetched.
    as_of: datetime | None = None

    def age_hours(self, now: datetime) -> float | None:
        ts = self.as_of or self.fetched_at
        if ts is None:
            return None
        return (now - ts).total_seconds() / 3600


@dataclass(frozen=True)
class Readiness:
    verdict: Verdict
    phase: Phase
    #: Sources that failed their tolerance, with a human-readable reason.
    problems: dict[Source, str]

    def explain(self) -> str:
        if not self.problems:
            return f"{self.verdict} in {self.phase}"
        detail = "; ".join(f"{s}: {r}" for s, r in sorted(self.problems.items()))
        return f"{self.verdict} in {self.phase} -- {detail}"


def evaluate(
    phase: Phase,
    states: dict[Source, SourceState],
    now: datetime | None = None,
) -> Readiness:
    """Decide whether the phase's required sources are good enough to act on."""
    now = now or datetime.now(UTC)
    tolerances = TOLERANCE[phase]
    problems: dict[Source, str] = {}
    blocking = False

    for source, limit in tolerances.items():
        if limit is None:
            continue

        state = states.get(source)
        if state is None:
            problems[source] = "never fetched"
            blocking = blocking or source not in DEGRADE_ONLY
            continue

        if state.status in (Status.INVALID, Status.UNINFORMATIVE):
            problems[source] = state.status.lower()
            blocking = blocking or source not in DEGRADE_ONLY
            continue

        age = state.age_hours(now)
        if age is None or age > limit:
            shown = "unknown age" if age is None else f"{age:.1f}h old (limit {limit:.0f}h)"
            problems[source] = shown
            blocking = blocking or source not in DEGRADE_ONLY

    if blocking:
        return Readiness(Verdict.BLOCKED, phase, problems)
    if problems:
        return Readiness(Verdict.DEGRADED, phase, problems)
    return Readiness(Verdict.READY, phase, problems)


def next_due(state: SourceState, limit_hours: float) -> datetime | None:
    """When this source next needs refreshing. Used to keep jobs cheap no-ops."""
    ts = state.as_of or state.fetched_at
    return None if ts is None else ts + timedelta(hours=limit_hours)
