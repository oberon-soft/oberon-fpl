"""Daily ingest.

Idempotent: re-running on the same day updates rather than duplicating, so a
retry after a partial failure is safe and a double-scheduled job is harmless.

Records a freshness row for every source whether it succeeded or not. A failure
that leaves no evidence is indistinguishable from a job that never ran, and the
gate downstream needs to tell those apart.
"""

from __future__ import annotations

import structlog

from fpl import db, holdings
from fpl.client import FPLClient, FPLError, assert_bootstrap_sane, bootstrap_is_informative
from fpl.config import CONFIG
from fpl.entry import load_squad
from fpl.freshness import Source, Status
from fpl.phase import derive_phase

log = structlog.get_logger()


def run() -> int:
    with db.connect() as conn, FPLClient() as client:
        db.migrate(conn)

        try:
            boot = client.bootstrap()
            assert_bootstrap_sane(boot)
        except FPLError as exc:
            log.error("bootstrap_failed", error=str(exc))
            db.record_freshness(conn, Source.BOOTSTRAP, Status.INVALID, error=str(exc))
            return 1

        # A payload can be perfectly well-formed and still carry nothing to model
        # on. Preseason every stat is zero and every team strength is unset, which
        # passes any schema check. Flagging it here is what stops the model
        # confidently projecting from noise.
        informative = bootstrap_is_informative(boot)
        status = Status.FRESH if informative else Status.UNINFORMATIVE

        n_players = db.write_player_snapshot(conn, boot["elements"])
        db.write_event_snapshot(conn, boot["events"])
        db.record_freshness(conn, Source.BOOTSTRAP, status, row_count=n_players)

        try:
            fixtures = client.fixtures()
            n_fixtures = db.write_fixtures(conn, fixtures)
            db.record_freshness(conn, Source.FIXTURES, Status.FRESH, row_count=n_fixtures)
        except FPLError as exc:
            log.error("fixtures_failed", error=str(exc))
            db.record_freshness(conn, Source.FIXTURES, Status.INVALID, error=str(exc))
            n_fixtures = 0

        state = derive_phase(boot["events"])

        # Own-squad state is ingestion, not decision. Keeping it here means
        # seeding happens on the daily job the moment picks become public --
        # a few hours after a deadline -- rather than waiting for a phase the
        # decide job happens to consider actionable.
        synced = _sync_own_squad(conn, client, boot, state)
        rivals = _sync_rivals(conn, client, boot, state)

        db.log_event(
            conn,
            "ingest",
            {
                "players": n_players,
                "fixtures": n_fixtures,
                "phase": str(state.phase),
                "next_gameweek": state.next_gameweek,
                "informative": informative,
                "holdings_synced": synced,
                "rival_squads": rivals,
            },
        )

        log.info(
            "ingest_complete",
            players=n_players,
            fixtures=n_fixtures,
            phase=str(state.phase),
            next_gameweek=state.next_gameweek,
            informative=informative,
            holdings_synced=synced,
            rival_squads=rivals,
        )
    return 0


def _sync_own_squad(conn, client, boot, state) -> bool:
    """Record the last confirmed squad and price any arrivals.

    Picks become public at a deadline, so the newest readable squad is the most
    recent gameweek whose deadline has passed. Returns whether anything was
    synced -- false before the first deadline, which is not a failure.
    """
    if not CONFIG.entry_id:
        return False

    settled = [
        e for e in boot["events"]
        if state.next_gameweek is None or e["id"] < state.next_gameweek
    ]
    if not settled:
        return False
    latest = max(e["id"] for e in settled)

    transfer_cap = 1 + boot["game_settings"]["max_extra_free_transfers"]
    try:
        squad = load_squad(client, CONFIG.entry_id, latest, transfer_cap=transfer_cap)
    except FPLError as exc:
        log.warning("own_squad_fetch_failed", error=str(exc))
        return False
    if squad is None:
        return False

    deadline = next(
        (e["deadline_time"] for e in boot["events"] if e["id"] == latest), None
    )
    from datetime import datetime

    holdings.sync(
        conn,
        entry_id=CONFIG.entry_id,
        event=latest,
        picks=squad.picks_raw,
        deadline_date=datetime.fromisoformat(deadline.replace("Z", "+00:00")).date(),
        now_costs={e["id"]: e["now_cost"] for e in boot["elements"]},
        reported_value=squad.value,
    )
    db.record_freshness(conn, Source.OWN_SQUAD, Status.FRESH)
    return True


def _sync_rivals(conn, client, boot, state) -> int:
    """Record every league member's confirmed squad.

    Cheap -- one call per member per gameweek -- and unlike prices this data does
    not evaporate: past picks stay retrievable, so a missed run backfills. It is
    fetched daily anyway because the alternative is remembering to.

    Only ids and team names are stored. The endpoint also returns each manager's
    real name and there is no reason for this system to hold it.
    """
    if not (CONFIG.league_id and state.next_gameweek):
        return 0
    latest = state.next_gameweek - 1
    if latest < 1:
        return 0

    try:
        members = client.league_members(CONFIG.league_id)
    except FPLError as exc:
        log.warning("league_fetch_failed", error=str(exc))
        return 0
    if not members:
        return 0

    db.upsert_entries(conn, members, self_id=CONFIG.entry_id)

    stored = 0
    for entry_id in members:
        if entry_id == CONFIG.entry_id:
            continue  # already recorded, with purchase prices, by _sync_own_squad
        try:
            picks = client.entry_picks(entry_id, latest)
        except FPLError:
            continue
        db.write_entry_picks(conn, entry_id, latest, picks["picks"])
        stored += 1

    log.info("rivals_synced", members=len(members), squads=stored, gameweek=latest)
    return stored
