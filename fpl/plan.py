"""Refit rates and project expected points across the horizon.

Writes every projection to the database, stamped with MODEL_VERSION and
alongside three baselines. That logging is not incidental -- it is the entire
validation strategy. FPL publishes its own `ep_next`, and if this model cannot
beat it there is no edge worth deploying, which is a thing worth learning in a
fortnight rather than at the end of a season.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from fpl import db
from fpl.client import FPLClient, FPLError
from fpl.config import CONFIG, MODEL_VERSION
from fpl.freshness import Source, Status
from fpl.phase import Phase, derive_phase
from fpl.project import Projection, project_player
from fpl.rates import PlayerRates, extract, positional_means, price_prior, shrink
from fpl.scoring import Scoring

log = structlog.get_logger()

#: Statuses that take a player out of contention entirely. `d` (doubtful) is
#: handled by scaling rather than exclusion.
UNAVAILABLE = frozenset({"i", "u", "s"})
#: Below this, a doubt is treated as an absence.
MIN_AVAILABILITY = 0.5


def ensure_player_seasons(conn, client: FPLClient, elements: list[dict[str, Any]]) -> int:
    """Fetch prior-season totals for any player we have not seen before.

    Roughly 600 requests the first time and near zero afterwards. Players with
    genuinely no Premier League history get a marker row, so a debutant is not
    re-fetched on every run for the rest of the season.
    """
    codes = [e["code"] for e in elements]
    missing = db.codes_missing_history(conn, codes)
    if not missing:
        return 0

    by_code = {e["code"]: e for e in elements}
    log.info("fetching_player_history", players=len(missing))

    rows: list[tuple[int, dict[str, Any]]] = []
    no_history: list[int] = []
    failed = 0

    for code in missing:
        element = by_code[code]
        try:
            summary = client.player(element["id"])
        except (FPLError, httpx.HTTPError):
            failed += 1
            continue
        past = summary.get("history_past") or []
        if past:
            rows.extend((code, season) for season in past)
        else:
            no_history.append(code)

    if rows:
        db.write_player_seasons(conn, rows)
    if no_history:
        db.mark_history_fetched(conn, no_history)
    if failed:
        log.warning("player_history_incomplete", failed=failed)

    log.info("player_history_stored", seasons=len(rows), without_history=len(no_history))
    return len(rows)


def build_projections(
    elements: list[dict[str, Any]],
    seasons: dict[int, dict[str, Any]],
    fixtures: dict[int, list[tuple[int, int]]],
    scoring: Scoring,
    overrides: dict[str, dict[str, Any]],
) -> list[Projection]:
    """Project every selectable player across the horizon."""
    observed: list[tuple[int, PlayerRates]] = []
    for element in elements:
        season = seasons.get(element["code"])
        if not season:
            continue
        rates = extract(element["code"], element["element_type"], season)
        if rates is not None:
            observed.append((element["now_cost"], rates))

    means = positional_means(r for _, r in observed)
    observed_by_code = {r.code: r for _, r in observed}

    projections: list[Projection] = []
    for element in elements:
        if element["status"] in UNAVAILABLE:
            continue

        availability = (
            1.0
            if element["status"] == "a"
            else (element.get("chance_of_playing_next_round") or 0) / 100.0
        )
        if availability < MIN_AVAILABILITY:
            continue

        team_fixtures = fixtures.get(element["team"], [])
        if not team_fixtures:
            continue

        rates = observed_by_code.get(element["code"])
        if rates is None:
            rates = price_prior(element["element_type"], element["now_cost"], observed)
            if rates is None:
                continue

        override = overrides.get(element["web_name"]) or overrides.get(str(element["code"]))
        missed = override["miss_events"] if override else frozenset()
        multiplier = override["ep_multiplier"] if override else None

        projections.append(
            project_player(
                element=element,
                rates=rates,
                shrunk=shrink(rates, means, CONFIG.shrinkage),
                team_fixtures=team_fixtures,
                scoring=scoring,
                config=CONFIG,
                availability=availability,
                missed_events=missed,
                ep_multiplier=multiplier,
            )
        )

    return projections


def run() -> int:
    with db.connect() as conn, FPLClient() as client:
        db.migrate(conn)

        try:
            boot = client.bootstrap()
        except FPLError as exc:
            log.error("bootstrap_failed", error=str(exc))
            return 1

        state = derive_phase(boot["events"])
        if state.phase is Phase.SETTLING:
            # Bonus and defensive-contribution tallies are still moving. Refitting
            # now trains on numbers that are about to change.
            log.info("skipping_refit", reason="gameweek stats not final", phase=str(state.phase))
            return 0
        if state.next_gameweek is None:
            log.info("skipping_refit", reason="no upcoming gameweek")
            return 0

        horizon = list(
            range(state.next_gameweek, min(state.next_gameweek + CONFIG.squad.horizon, 39))
        )

        ensure_player_seasons(conn, client, boot["elements"])
        db.record_freshness(conn, Source.PLAYER_HISTORY, Status.FRESH)

        prior = _previous_season(conn)
        seasons = db.load_player_seasons(conn, prior)
        fixtures = db.fixture_difficulties(conn, horizon)
        overrides = db.load_overrides(conn, state.next_gameweek)
        scoring = Scoring.from_bootstrap(boot)

        projections = build_projections(
            boot["elements"], seasons, fixtures, scoring, overrides
        )

        snapshot = {e["id"]: e for e in boot["elements"]}
        rows = []
        for p in projections:
            element = snapshot[p.element_id]
            season = seasons.get(p.code) or {}
            rows.append(
                {
                    "event": p.next_event,
                    "code": p.code,
                    "element_id": p.element_id,
                    "ep": round(p.ep_next, 3),
                    "ep_horizon": round(p.ep_horizon, 3),
                    "now_cost": p.now_cost,
                    "baseline_ep_next": element.get("ep_next"),
                    "baseline_ppg": round((season.get("total_points") or 0) / 38.0, 3),
                    "imputed": p.imputed,
                }
            )

        written = db.write_projections(conn, MODEL_VERSION, rows)
        db.record_freshness(conn, Source.RATES, Status.FRESH, row_count=len(projections))
        db.log_event(
            conn,
            "plan",
            {
                "model_version": MODEL_VERSION,
                "phase": str(state.phase),
                "horizon": horizon,
                "projected": len(projections),
                "imputed": sum(1 for p in projections if p.imputed),
                "overrides": len(overrides),
            },
        )

        log.info(
            "plan_complete",
            model_version=MODEL_VERSION,
            projected=len(projections),
            imputed=sum(1 for p in projections if p.imputed),
            written=written,
            horizon=f"GW{horizon[0]}-{horizon[-1]}",
            overrides=len(overrides),
        )
    return 0


def _previous_season(conn) -> str:
    """Which season's totals to fit on.

    Falls back to whatever the most recent stored season is, so this keeps
    working across season boundaries without a hardcoded string.
    """
    row = conn.execute(
        """
        SELECT season_name FROM player_seasons
        WHERE season_name <> '__none__'
        ORDER BY season_name DESC LIMIT 1
        """
    ).fetchone()
    return row["season_name"] if row else "__none__"
