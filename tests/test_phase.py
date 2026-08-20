"""The phase machine is pure and clock-injectable, so it tests without a
database, without the network, and without patching time."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fpl.phase import Phase, derive_phase


def event(
    eid: int,
    deadline: datetime,
    *,
    finished: bool = False,
    data_checked: bool = False,
    is_current: bool = False,
    is_next: bool = False,
) -> dict:
    return {
        "id": eid,
        "deadline_time": deadline.isoformat().replace("+00:00", "Z"),
        "finished": finished,
        "data_checked": data_checked,
        "is_current": is_current,
        "is_next": is_next,
    }


NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def test_preseason_before_any_gameweek_settles():
    events = [event(1, NOW + timedelta(days=5)), event(2, NOW + timedelta(days=12))]
    state = derive_phase(events, now=NOW)
    assert state.phase is Phase.PRESEASON
    assert state.next_gameweek == 1
    assert state.last_settled_gameweek is None


@pytest.mark.parametrize(
    ("hours", "expected"),
    [
        (72, Phase.PLANNING),
        (47, Phase.NEWS_WINDOW),
        (4, Phase.NEWS_WINDOW),
        (2, Phase.PRE_DEADLINE),
        (0.5, Phase.PRE_DEADLINE),
    ],
)
def test_phase_tracks_distance_to_deadline(hours: float, expected: Phase):
    events = [
        event(1, NOW - timedelta(days=7), finished=True, data_checked=True),
        event(2, NOW + timedelta(hours=hours), is_next=True),
    ]
    assert derive_phase(events, now=NOW).phase is expected


def test_settling_blocks_a_refit_on_partial_data():
    """Bonus and defensive-contribution tallies are still moving while
    data_checked is false. Refitting here trains on numbers about to change."""
    events = [
        event(1, NOW - timedelta(hours=6), finished=True, data_checked=False, is_current=True),
        event(2, NOW + timedelta(days=6)),
    ]
    state = derive_phase(events, now=NOW)
    assert state.phase is Phase.SETTLING
    assert state.last_settled_gameweek is None


def test_locked_once_the_deadline_passes():
    events = [event(1, NOW - timedelta(hours=2), is_current=True)]
    assert derive_phase(events, now=NOW).phase is Phase.LOCKED


def test_last_settled_ignores_finished_but_unchecked_gameweeks():
    events = [
        event(1, NOW - timedelta(days=14), finished=True, data_checked=True),
        event(2, NOW - timedelta(days=7), finished=True, data_checked=True),
        event(3, NOW - timedelta(days=1), finished=True, data_checked=False, is_current=True),
        event(4, NOW + timedelta(days=6)),
    ]
    assert derive_phase(events, now=NOW).last_settled_gameweek == 2


def test_next_deadline_is_the_soonest_future_one_not_the_flagged_one():
    """is_next lags reality around postponements; the clock does not."""
    events = [
        event(1, NOW - timedelta(days=1), finished=True, data_checked=True),
        event(3, NOW + timedelta(days=9), is_next=True),
        event(2, NOW + timedelta(days=3)),
    ]
    assert derive_phase(events, now=NOW).next_gameweek == 2


def test_real_payload_parses(bootstrap):
    state = derive_phase(bootstrap["events"])
    assert state.phase in set(Phase)
    assert state.next_gameweek is None or 1 <= state.next_gameweek <= 38
