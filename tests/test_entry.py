"""Reading your own squad back, and reconstructing the transfer balance.

FPL publishes transfers *made* per gameweek but never the balance remaining, so
the balance has to be rebuilt from history. Getting it wrong in either direction
is costly: too low and the model refuses a transfer you could make for free, too
high and it recommends one that silently costs four points.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from fpl.client import FPLClient
from fpl.config import FPL_API
from fpl.entry import Squad, free_transfers_from_history, load_squad

CAP = 5


def history(transfers: list[int], chips: list[tuple[int, str]] | None = None) -> dict:
    return {
        "current": [
            {"event": i + 1, "event_transfers": n} for i, n in enumerate(transfers)
        ],
        "chips": [{"event": e, "name": n} for e, n in (chips or [])],
    }


def test_starts_with_one():
    assert free_transfers_from_history(history([]), cap=CAP, current_event=1) == 1


def test_unused_transfers_bank():
    """Three quiet gameweeks should leave four in hand, not one."""
    assert free_transfers_from_history(history([0, 0, 0]), cap=CAP, current_event=4) == 4


def test_banking_stops_at_the_cap():
    """Read from game settings, not assumed -- FPL changed this rule recently."""
    assert free_transfers_from_history(history([0] * 12), cap=CAP, current_event=13) == CAP


def test_using_a_transfer_spends_it():
    assert free_transfers_from_history(history([0, 0, 1]), cap=CAP, current_event=4) == 3


def test_spending_everything_leaves_one_next_week():
    assert free_transfers_from_history(history([0, 0, 3]), cap=CAP, current_event=4) == 1


def test_hits_do_not_push_the_balance_negative():
    """Taking a -8 uses more transfers than you had; next week is still one."""
    assert free_transfers_from_history(history([3]), cap=CAP, current_event=2) == 1


def test_wildcard_transfers_do_not_consume_the_bank():
    """A wildcard's transfers are free, so a fifteen-player teardown must not
    wipe out a balance that was accumulating."""
    banked = free_transfers_from_history(
        history([0, 0, 15], chips=[(3, "wildcard")]), cap=CAP, current_event=4
    )
    assert banked == 4


def test_free_hit_also_exempt():
    banked = free_transfers_from_history(
        history([0, 0, 11], chips=[(3, "freehit")]), cap=CAP, current_event=4
    )
    assert banked == 4


def test_bench_boost_is_not_exempt():
    """It is a team chip, not a transfer chip -- transfers made alongside it are
    ordinary transfers and do spend the bank."""
    banked = free_transfers_from_history(
        history([0, 0, 2], chips=[(3, "bboost")]), cap=CAP, current_event=4
    )
    assert banked == 2


def test_future_gameweeks_are_ignored():
    """Only completed gameweeks inform the balance for the one coming."""
    assert free_transfers_from_history(
        history([0, 0, 0, 3, 3]), cap=CAP, current_event=3
    ) == 3


PICKS = {
    "entry_history": {"bank": 5, "value": 1003, "event_transfers": 1},
    "picks": [
        {"element": 100 + i, "position": i + 1, "multiplier": 1 if i < 11 else 0,
         "is_captain": i == 0, "is_vice_captain": i == 1}
        for i in range(15)
    ],
}


@respx.mock
def test_load_squad_reads_public_picks():
    respx.get(f"{FPL_API}/entry/7/event/3/picks/").mock(
        return_value=httpx.Response(200, json=PICKS)
    )
    respx.get(f"{FPL_API}/entry/7/history/").mock(
        return_value=httpx.Response(200, json=history([0, 0, 1]))
    )
    with FPLClient() as client:
        squad = load_squad(client, 7, 3, transfer_cap=CAP)

    assert squad is not None
    assert len(squad.element_ids) == 15
    assert squad.captain == 100 and squad.vice_captain == 101
    assert squad.budget == 1003 + 5


@respx.mock
def test_missing_picks_is_not_an_error():
    """404 before a deadline is the normal state, including for your own squad.
    Callers fall back to a from-scratch solve, which is right for GW1."""
    respx.get(f"{FPL_API}/entry/7/event/1/picks/").mock(
        return_value=httpx.Response(404, json={"detail": "Not found."})
    )
    with FPLClient() as client:
        assert load_squad(client, 7, 1, transfer_cap=CAP) is None


@respx.mock
def test_history_failure_still_yields_a_squad():
    """Losing the transfer balance should degrade to a conservative one, not
    lose the squad entirely."""
    respx.get(f"{FPL_API}/entry/7/event/3/picks/").mock(
        return_value=httpx.Response(200, json=PICKS)
    )
    respx.get(f"{FPL_API}/entry/7/history/").mock(return_value=httpx.Response(503))
    with FPLClient() as client:
        squad = load_squad(client, 7, 3, transfer_cap=CAP)
    assert squad is not None and squad.free_transfers == 1


def test_budget_combines_value_and_bank():
    squad = Squad(
        entry_id=1, event=3, element_ids=[], captain=None, vice_captain=None,
        bank=12, value=1004, free_transfers=1,
    )
    assert squad.budget == 1016


def test_no_log_call_shadows_structlogs_reserved_key():
    """`event` is structlog's key for the message itself, so passing it as a
    field raises TypeError at call time -- invisible until that branch runs,
    which for the 404 path meant the first Friday before a deadline.
    """
    import re
    from pathlib import Path

    offenders = []
    for path in Path("fpl").glob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"log\.(info|warning|error|debug)\(.*[,(]\s*event=", line):
                offenders.append(f"{path}:{n}")
    assert not offenders, f"structlog reserved-key collision: {offenders}"
