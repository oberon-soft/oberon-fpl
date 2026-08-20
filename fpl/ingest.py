"""Daily ingest.

Idempotent: re-running on the same day updates rather than duplicating, so a
retry after a partial failure is safe and a double-scheduled job is harmless.

Records a freshness row for every source whether it succeeded or not. A failure
that leaves no evidence is indistinguishable from a job that never ran, and the
gate downstream needs to tell those apart.
"""

from __future__ import annotations

import structlog

from fpl import db
from fpl.client import FPLClient, FPLError, assert_bootstrap_sane, bootstrap_is_informative
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
        db.log_event(
            conn,
            "ingest",
            {
                "players": n_players,
                "fixtures": n_fixtures,
                "phase": str(state.phase),
                "next_gameweek": state.next_gameweek,
                "informative": informative,
            },
        )

        log.info(
            "ingest_complete",
            players=n_players,
            fixtures=n_fixtures,
            phase=str(state.phase),
            next_gameweek=state.next_gameweek,
            informative=informative,
        )
    return 0
