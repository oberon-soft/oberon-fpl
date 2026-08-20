"""The gate that stands between stale data and a recommendation you act on."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fpl.freshness import (
    TOLERANCE,
    Readiness,
    Source,
    SourceState,
    Status,
    Verdict,
    evaluate,
)
from fpl.phase import Phase

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def state(source: Source, *, hours_old: float, status: Status = Status.FRESH) -> SourceState:
    return SourceState(source, status, fetched_at=NOW - timedelta(hours=hours_old))


def all_fresh(hours_old: float = 1.0) -> dict[Source, SourceState]:
    return {s: state(s, hours_old=hours_old) for s in Source}


def test_ready_when_everything_is_current():
    assert evaluate(Phase.PRE_DEADLINE, all_fresh(), NOW).verdict is Verdict.READY


def test_stale_bootstrap_blocks_before_a_deadline():
    """Three hours out, a twelve-hour-old snapshot predates the team news that
    is the entire reason for running then."""
    s = all_fresh()
    s[Source.BOOTSTRAP] = state(Source.BOOTSTRAP, hours_old=12)
    result = evaluate(Phase.PRE_DEADLINE, s, NOW)
    assert result.verdict is Verdict.BLOCKED
    assert Source.BOOTSTRAP in result.problems


def test_same_staleness_is_fine_while_planning():
    """Tolerance is relative to what you are about to do, not absolute."""
    s = all_fresh()
    s[Source.BOOTSTRAP] = state(Source.BOOTSTRAP, hours_old=12)
    assert evaluate(Phase.PLANNING, s, NOW).verdict is Verdict.READY


def test_own_squad_degrades_rather_than_blocks():
    s = all_fresh()
    s[Source.OWN_SQUAD] = state(Source.OWN_SQUAD, hours_old=400)
    result = evaluate(Phase.PLANNING, s, NOW)
    assert result.verdict is Verdict.DEGRADED
    assert Source.OWN_SQUAD in result.problems


def test_uninformative_blocks_even_though_it_is_fresh():
    """The failure mode that returns 200 and tells you nothing. Preseason zeroes
    are well-formed, recently fetched, and completely unmodellable."""
    s = all_fresh()
    s[Source.BOOTSTRAP] = state(Source.BOOTSTRAP, hours_old=0.1, status=Status.UNINFORMATIVE)
    result = evaluate(Phase.PRE_DEADLINE, s, NOW)
    assert result.verdict is Verdict.BLOCKED
    assert "uninformative" in result.problems[Source.BOOTSTRAP]


def test_never_fetched_blocks():
    s = all_fresh()
    del s[Source.BOOTSTRAP]
    assert evaluate(Phase.PRE_DEADLINE, s, NOW).verdict is Verdict.BLOCKED


def test_as_of_beats_fetched_at():
    """Data pulled ten minutes ago that reflects a line from two days back is
    stale data, freshly fetched."""
    s = all_fresh()
    s[Source.BOOTSTRAP] = SourceState(
        Source.BOOTSTRAP, Status.FRESH,
        fetched_at=NOW - timedelta(minutes=10),
        as_of=NOW - timedelta(hours=30),
    )
    assert evaluate(Phase.PRE_DEADLINE, s, NOW).verdict is Verdict.BLOCKED


@pytest.mark.parametrize("phase", list(Phase))
def test_every_phase_has_a_complete_tolerance_row(phase: Phase):
    """A phase missing from the table would silently skip every check."""
    assert phase in TOLERANCE
    assert set(TOLERANCE[phase]) == set(Source)


@pytest.mark.parametrize("phase", list(Phase))
def test_blocked_never_degrades_into_a_recommendation(phase: Phase):
    """The core invariant. Whatever the phase, if a required source is missing
    the verdict is BLOCKED -- never DEGRADED, never READY."""
    required = [s for s, limit in TOLERANCE[phase].items() if limit is not None]
    for source in required:
        if source in {Source.OWN_SQUAD, Source.PLAYER_HISTORY}:
            continue
        s = all_fresh()
        del s[source]
        assert evaluate(phase, s, NOW).verdict is Verdict.BLOCKED


def test_explain_names_the_offending_source():
    s = all_fresh()
    s[Source.FIXTURES] = state(Source.FIXTURES, hours_old=999)
    text = evaluate(Phase.PRE_DEADLINE, s, NOW).explain()
    assert "fixtures" in text and "BLOCKED" in text
