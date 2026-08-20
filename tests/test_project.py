"""Expected points assembly.

These are mostly structural properties rather than golden numbers. The absolute
values depend on parameters that will be retuned; the relationships between them
are what must not silently invert.
"""

from __future__ import annotations

import pytest

from fpl.config import CONFIG
from fpl.project import expected_points, project_player
from fpl.rates import extract
from fpl.scoring import Scoring

SEASON = {
    "season_name": "2025/26", "minutes": 2750, "starts": 30, "total_points": 209,
    "expected_goals": "2.94", "expected_assists": "1.75",
    "defensive_contribution": 277, "bonus": 30, "saves": 0, "yellow_cards": 4,
}


@pytest.fixture
def scoring(bootstrap) -> Scoring:
    return Scoring.from_bootstrap(bootstrap)


def rates_for(element_type: int, **overrides):
    return extract(1, element_type, {**SEASON, **overrides})


def element(element_type: int = 2, **kw):
    return {
        "id": 1, "code": 1, "web_name": "Test", "team": 1,
        "element_type": element_type, "now_cost": 60, **kw,
    }


def ep(element_type: int, difficulty: int, scoring: Scoring, **season) -> float:
    r = rates_for(element_type, **season)
    shrunk = dict(r.per90)
    return expected_points(r, shrunk, element_type, difficulty, scoring, CONFIG)


def test_easier_fixtures_score_more(scoring):
    values = [ep(2, d, scoring) for d in (1, 2, 3, 4, 5)]
    assert values == sorted(values, reverse=True)


def test_clean_sheet_matters_far_more_to_defenders(scoring):
    """A defender's fixture swing is dominated by clean-sheet probability, which
    falls exponentially in opponent threat. A forward barely notices."""
    defender = ep(2, 2, scoring) - ep(2, 5, scoring)
    forward = ep(4, 2, scoring) - ep(4, 5, scoring)
    assert defender > forward


def test_defensive_contribution_is_worth_real_points(scoring):
    """The recent rule change a naive xG-only model cannot see. A busy defender
    should gain materially from defensive actions alone."""
    busy = ep(2, 3, scoring, defensive_contribution=400)
    quiet = ep(2, 3, scoring, defensive_contribution=50)
    assert busy - quiet > 0.5


def test_goalkeepers_score_no_defensive_contribution(scoring):
    busy = ep(1, 3, scoring, defensive_contribution=400)
    quiet = ep(1, 3, scoring, defensive_contribution=50)
    assert busy == pytest.approx(quiet, abs=1e-9)


def test_more_minutes_means_more_points(scoring):
    assert ep(3, 3, scoring, minutes=3200, starts=36) > ep(3, 3, scoring, minutes=900, starts=10)


def test_expected_points_never_negative(scoring):
    """A card-prone defender conceding heavily must still floor at zero -- a
    negative projection would let the optimiser 'gain' points by benching."""
    awful = ep(2, 5, scoring, minutes=200, starts=2, yellow_cards=20,
               expected_goals="0", expected_assists="0", defensive_contribution=0, bonus=0)
    assert awful >= 0.0


def _project(scoring, fixtures, **kw):
    r = rates_for(2)
    return project_player(
        element=element(), rates=r, shrunk=dict(r.per90),
        team_fixtures=fixtures, scoring=scoring, config=CONFIG, **kw
    )


def test_horizon_is_projected_per_gameweek(scoring):
    p = _project(scoring, [(1, 3), (2, 2), (3, 4)], availability=1.0)
    assert sorted(p.by_gameweek) == [1, 2, 3]
    assert p.next_event == 1
    assert p.ep_next == p.by_gameweek[1]
    assert p.ep_horizon == pytest.approx(sum(p.by_gameweek.values()) / 3)


def test_double_gameweeks_accumulate(scoring):
    """Two fixtures in one gameweek means two lots of points, not an average."""
    single = _project(scoring, [(1, 3)], availability=1.0)
    double = _project(scoring, [(1, 3), (1, 3)], availability=1.0)
    assert double.by_gameweek[1] == pytest.approx(single.by_gameweek[1] * 2)


def test_override_zeroes_the_missed_gameweeks(scoring):
    """The Anderson case: FPL reported him available with no news while he was
    suspended. Overrides are the only path for knowledge the API lacks."""
    p = _project(scoring, [(1, 3), (2, 3), (3, 3)], availability=1.0, missed_events={1, 2})
    assert p.by_gameweek[1] == 0.0
    assert p.by_gameweek[2] == 0.0
    assert p.by_gameweek[3] > 0.0
    assert any("unavailable" in n for n in p.notes)


def test_availability_scales_every_gameweek(scoring):
    full = _project(scoring, [(1, 3), (2, 3)], availability=1.0)
    doubtful = _project(scoring, [(1, 3), (2, 3)], availability=0.75)
    for gw in full.by_gameweek:
        assert doubtful.by_gameweek[gw] == pytest.approx(full.by_gameweek[gw] * 0.75)


def test_ep_multiplier_applies(scoring):
    full = _project(scoring, [(1, 3)], availability=1.0)
    halved = _project(scoring, [(1, 3)], availability=1.0, ep_multiplier=0.5)
    assert halved.by_gameweek[1] == pytest.approx(full.by_gameweek[1] * 0.5)


def test_player_with_no_fixtures_projects_nothing(scoring):
    """A blank gameweek, or a team whose fixture was postponed."""
    p = _project(scoring, [], availability=1.0)
    assert p.by_gameweek == {}
    assert p.ep_next == 0.0 and p.ep_horizon == 0.0
