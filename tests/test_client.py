"""Client tests run against recorded payloads. CI never touches the live API."""

from __future__ import annotations

import copy

import httpx
import pytest
import respx

from fpl.client import (
    FPLClient,
    FPLError,
    assert_bootstrap_sane,
    bootstrap_is_informative,
)
from fpl.config import FPL_API


def test_sane_payload_passes(bootstrap):
    assert_bootstrap_sane(bootstrap)


@pytest.mark.parametrize("missing", ["elements", "teams", "events", "game_config"])
def test_missing_top_level_key_is_invalid(bootstrap, missing):
    broken = copy.copy(bootstrap)
    del broken[missing]
    with pytest.raises(FPLError, match=missing):
        assert_bootstrap_sane(broken)


def test_wrong_team_count_is_invalid(bootstrap):
    broken = copy.copy(bootstrap)
    broken["teams"] = bootstrap["teams"][:19]
    with pytest.raises(FPLError, match="20 teams"):
        assert_bootstrap_sane(broken)


def test_preseason_payload_is_sane_but_uninformative(bootstrap):
    """Recorded the day before the season opened: structurally perfect, and
    describing a season that has already finished. Exactly the case a
    status-code check waves through."""
    assert_bootstrap_sane(bootstrap)
    assert bootstrap_is_informative(bootstrap) is False


def test_carried_over_minutes_do_not_count_as_informative(bootstrap):
    """The trap. Preseason, `minutes` and `total_points` still hold last
    season's totals -- they only reset at the opening kickoff. A check based on
    "do players have minutes" passes while the current season has not started."""
    played = [e for e in bootstrap["elements"] if e.get("minutes", 0) > 0]
    assert played, "fixture should contain last season's carried-over minutes"
    assert bootstrap_is_informative(bootstrap) is False


def test_informative_once_a_gameweek_has_settled(bootstrap):
    live = copy.deepcopy(bootstrap)
    live["events"][0]["finished"] = True
    live["events"][0]["data_checked"] = True
    for i, t in enumerate(live["teams"]):
        t["strength_attack_home"] = 1000 + i
    assert bootstrap_is_informative(live) is True


def test_settled_gameweek_alone_is_not_enough(bootstrap):
    """Team strengths stay at zero for a while into the season; the fixture
    layer is unusable until they populate."""
    live = copy.deepcopy(bootstrap)
    live["events"][0]["finished"] = True
    live["events"][0]["data_checked"] = True
    assert bootstrap_is_informative(live) is False


@respx.mock
def test_bootstrap_round_trip(bootstrap):
    respx.get(f"{FPL_API}/bootstrap-static/").mock(
        return_value=httpx.Response(200, json=bootstrap)
    )
    with FPLClient() as client:
        assert client.bootstrap()["teams"] == bootstrap["teams"]


@respx.mock
def test_non_200_raises():
    respx.get(f"{FPL_API}/bootstrap-static/").mock(return_value=httpx.Response(503))
    with FPLClient() as client, pytest.raises(FPLError, match="503"):
        client.bootstrap()


@respx.mock
def test_picks_404_before_a_deadline():
    """Picks are private until the deadline passes -- including your own. The
    caller has to treat this as "not yet", not as an error worth alerting on."""
    respx.get(f"{FPL_API}/entry/12345/event/1/picks/").mock(
        return_value=httpx.Response(404, json={"detail": "Not found."})
    )
    with FPLClient() as client, pytest.raises(FPLError, match="404"):
        client.entry_picks(12345, 1)


@respx.mock
def test_league_members_reads_new_entries_preseason(league):
    """Before any gameweek completes, `standings.results` is empty and every
    member sits in `new_entries`. Reading only standings returns nobody."""
    assert league["standings"]["results"] == []
    respx.get(f"{FPL_API}/leagues-classic/999999/standings/?page_standings=1").mock(
        return_value=httpx.Response(200, json=league)
    )
    with FPLClient() as client:
        members = client.league_members(999999)
    assert members == {
        1000001: "Synthetic FC",
        1000002: "Placeholder Rovers",
        1000003: "Fixture United",
    }


@respx.mock
def test_league_members_merges_both_sources(league):
    """Mid-season: most members have migrated to standings, a recent joiner has
    not. Both have to be read or the joiner is silently dropped."""
    payload = copy.deepcopy(league)
    payload["standings"]["results"] = [
        {"entry": 1000001, "entry_name": "Synthetic FC", "rank": 1, "total": 120},
        {"entry": 1000002, "entry_name": "Placeholder Rovers", "rank": 2, "total": 98},
    ]
    payload["new_entries"]["results"] = [
        {"entry": 1000004, "entry_name": "Late Arrival", "joined_time": "2026-10-01T00:00:00Z",
         "player_first_name": "Di", "player_last_name": "Example"},
    ]
    respx.get(f"{FPL_API}/leagues-classic/999999/standings/?page_standings=1").mock(
        return_value=httpx.Response(200, json=payload)
    )
    with FPLClient() as client:
        members = client.league_members(999999)
    assert set(members) == {1000001, 1000002, 1000004}


@respx.mock
def test_league_members_discards_manager_names(league):
    """The API returns player_first_name / player_last_name. Nothing downstream
    needs them, so they are dropped at the boundary rather than stored and
    protected."""
    respx.get(f"{FPL_API}/leagues-classic/999999/standings/?page_standings=1").mock(
        return_value=httpx.Response(200, json=league)
    )
    with FPLClient() as client:
        members = client.league_members(999999)
    assert all(isinstance(v, str) for v in members.values())
    surnames = {
        r["player_last_name"] for r in league["new_entries"]["results"]
    }
    assert not surnames & set(members.values())
