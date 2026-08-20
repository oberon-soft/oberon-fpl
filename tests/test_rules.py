"""Assumptions this model bakes in, asserted against a recorded payload.

These are not really tests of our code -- they are tests of our *understanding*
of the game. FPL changes rules between seasons (defensive contribution and the
whole price_change_* block are both recent), and a silent rule change produces a
model that is confidently wrong rather than obviously broken.

The scheduled contract workflow runs the same assertions against the live API.
"""

from __future__ import annotations

from fpl.config import CONFIG


def test_scoring_table_is_read_not_hardcoded(bootstrap):
    """The points function is a dot product against this dict. If it moves, the
    model follows automatically -- but only if the shape stays the same."""
    scoring = bootstrap["game_config"]["scoring"]
    for key in ("goals_scored", "assists", "clean_sheets", "defensive_contribution"):
        assert key in scoring
    for pos in ("GKP", "DEF", "MID", "FWD"):
        assert pos in scoring["goals_scored"]
        assert pos in scoring["clean_sheets"]


def test_defensive_contribution_is_scored(bootstrap):
    """The recent rule change that most affects defender and holding-midfield
    valuation, and the one a naive xG-only model cannot see at all."""
    dc = bootstrap["game_config"]["scoring"]["defensive_contribution"]
    assert dc["DEF"] > 0 and dc["MID"] > 0 and dc["FWD"] > 0
    assert dc["GKP"] == 0


def test_squad_structure_comes_from_the_api(bootstrap):
    """ILP constraints are read from element_types rather than hardcoded."""
    select = {t["id"]: t["squad_select"] for t in bootstrap["element_types"]}
    assert sum(select.values()) == bootstrap["game_settings"]["squad_squadsize"]
    assert bootstrap["game_settings"]["squad_squadplay"] == 11
    assert bootstrap["game_settings"]["squad_team_limit"] == 3


def test_dc_thresholds_cover_every_scoring_position(bootstrap):
    """Thresholds are not exposed by the API, so they are asserted in config and
    have to be kept in step with the positions that actually score DC points."""
    dc = bootstrap["game_config"]["scoring"]["defensive_contribution"]
    short = {t["id"]: t["singular_name_short"] for t in bootstrap["element_types"]}
    scoring_positions = {pid for pid, name in short.items() if dc[name] > 0}
    assert set(CONFIG.squad.dc_threshold) == scoring_positions


def test_transfer_rules_match_expectations(bootstrap):
    gs = bootstrap["game_settings"]
    assert gs["max_extra_free_transfers"] == 4      # bank up to 5 total
    assert gs["transfers_sell_on_fee"] == 0.5
    assert gs["squad_total_spend"] == 1000


def test_chips_are_split_into_halves(bootstrap):
    """Eight chips across two windows. This forces first-half chips to be spent
    on ordinary gameweeks rather than saved for the doubles that cluster later,
    which is what the stopping policy's declining threshold has to reflect."""
    chips = bootstrap["chips"]
    windows = {(c["name"], c["start_event"], c["stop_event"]) for c in chips}
    names = {c["name"] for c in chips}
    assert names == {"wildcard", "freehit", "bboost", "3xc"}
    assert len(windows) == 8
    assert any(stop <= 19 for _, _, stop in windows)
    assert any(start >= 20 for _, start, _ in windows)


def test_opta_code_derives_from_code(bootstrap):
    """`code` is the stable Opta player id and survives across seasons, unlike
    `element_id`. Any cross-season join must use it."""
    for e in bootstrap["elements"]:
        assert e["opta_code"] == f"p{e['code']}"


def test_fixture_difficulty_is_within_range(fixtures_payload):
    for f in fixtures_payload:
        for key in ("team_h_difficulty", "team_a_difficulty"):
            if f.get(key) is not None:
                assert 1 <= f[key] <= 5
