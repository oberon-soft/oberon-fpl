"""Produce the recommendation you act on.

Runs hourly and gates itself. FPL deadlines move constantly with TV scheduling,
so encoding the timing in cron guarantees eventually firing at the wrong hour;
encoding it in the derived phase means a missed run self-heals on the next one.
The cost of an hourly no-op is one API call.

The invariant this module exists to hold: BLOCKED never degrades into a
recommendation. Silence plus a firing alert is unambiguous. A recommendation
quietly built on last week's numbers, caveated in a footer nobody reads at 22:50
on a Friday, is worse than no recommendation at all.
"""

from __future__ import annotations

from dataclasses import replace

import structlog

from fpl import db, holdings, notify
from fpl.client import FPLClient, FPLError
from fpl.config import CONFIG, MODEL_VERSION
from fpl.entry import load_squad
from fpl.freshness import Source, Status, Verdict, evaluate
from fpl.optimise import Rules
from fpl.phase import Phase, derive_phase
from fpl.plan import build_projections
from fpl.recommend import build, render
from fpl.scoring import Scoring

log = structlog.get_logger()

#: Phases in which a recommendation is worth producing at all. Outside these the
#: job exits immediately -- there is nothing to decide while a gameweek is in
#: progress or its stats are still settling.
ACTIONABLE = frozenset({Phase.PRE_DEADLINE, Phase.NEWS_WINDOW, Phase.PRESEASON})


def run(force: bool = False) -> int:
    with db.connect() as conn, FPLClient() as client:
        db.migrate(conn)

        try:
            boot = client.bootstrap()
        except FPLError as exc:
            log.error("bootstrap_failed", error=str(exc))
            return 1

        state = derive_phase(boot["events"])
        if not force and state.phase not in ACTIONABLE:
            log.info("nothing_to_decide", phase=str(state.phase))
            return 0
        if state.next_gameweek is None:
            log.info("nothing_to_decide", reason="no upcoming gameweek")
            return 0

        readiness = evaluate(state.phase, db.load_freshness(conn))
        if readiness.verdict is Verdict.BLOCKED and not force:
            # Deliberately silent on the recommendation channel. The BLOCKED row
            # is what Alertmanager alerts on; sending a caveated recommendation
            # here would train you to ignore the caveat.
            log.warning("blocked", detail=readiness.explain())
            db.log_event(conn, "decide_blocked", {
                "phase": str(state.phase),
                "event": state.next_gameweek,
                "problems": {str(k): v for k, v in readiness.problems.items()},
            })
            return 0

        event = state.next_gameweek
        horizon = list(range(event, min(event + CONFIG.squad.horizon, 39)))
        rules = Rules.from_bootstrap(boot)

        candidates = build_projections(
            boot["elements"],
            db.load_player_seasons(conn, _latest_season(conn)),
            db.fixture_difficulties(conn, horizon),
            Scoring.from_bootstrap(boot),
            db.load_overrides(conn, event),
        )
        if not candidates:
            log.error("no_candidates", gameweek=event)
            return 1

        current = None
        costs: dict[int, int] = {}
        # The last confirmed squad is the previous gameweek's, since picks stay
        # private until a deadline passes -- your own included. Before GW1 there
        # is no previous gameweek to ask about, and event 0 is not a thing.
        if CONFIG.entry_id and event > 1:
            transfer_cap = 1 + boot["game_settings"]["max_extra_free_transfers"]
            current = load_squad(
                client, CONFIG.entry_id, event - 1, transfer_cap=transfer_cap
            )
            if current:
                # Restate the budget in the same units the optimiser costs in.
                # See Squad.budget: FPL's `value` is net of the sell-on fee, and
                # mixing it with now_cost coefficients can make holding your own
                # squad infeasible.
                held_now = sum(
                    e["now_cost"] for e in boot["elements"]
                    if e["id"] in set(current.element_ids)
                )
                current = replace(current, holdings_at_current_price=held_now)
                db.record_freshness(conn, Source.OWN_SQUAD, Status.FRESH)

                # Price any arrivals from the snapshot taken at their deadline,
                # then reconcile against FPL's published value. Seeding is
                # time-sensitive: a purchase price is knowable exactly only at
                # the moment of purchase, and inference afterwards.
                sell_costs, reconciliation = holdings.sync(
                    conn,
                    entry_id=CONFIG.entry_id,
                    event=event - 1,
                    picks=current.picks_raw,
                    deadline_date=_deadline_date(boot["events"], event - 1),
                    now_costs={e["id"]: e["now_cost"] for e in boot["elements"]},
                    reported_value=current.value,
                )
                if reconciliation.semantics is holdings.ValueSemantics.NET_OF_FEE:
                    # The fee is real, so cost held players at what selling them
                    # returns and budget from FPL's own figure.
                    costs = sell_costs
                    current = replace(current, holdings_at_current_price=None)

        recommendation = build(
            candidates,
            rules,
            event=event,
            horizon=horizon,
            kind="final" if state.phase is Phase.PRE_DEADLINE else "plan",
            readiness=readiness,
            positions={t["id"]: t["singular_name_short"] for t in boot["element_types"]},
            teams={t["id"]: t["short_name"] for t in boot["teams"]},
            current=current,
            costs=costs,
        )

        text = render(recommendation)
        row_id = db.write_recommendation(conn, MODEL_VERSION, recommendation.to_payload())

        delivered = notify.send(
            text,
            title=f"FPL GW{event}: {'hold' if recommendation.is_hold else 'transfer'}",
        )
        if delivered:
            db.mark_notified(conn, row_id)

        log.info(
            "decide_complete",
            gameweek=event,
            phase=str(state.phase),
            kind=recommendation.kind,
            verdict=str(recommendation.verdict),
            hold=recommendation.is_hold,
            hits=recommendation.solution.hits,
            notified=delivered,
        )
        print(text)
    return 0


def _latest_season(conn) -> str:
    row = conn.execute(
        """
        SELECT season_name FROM player_seasons
        WHERE season_name <> '__none__' ORDER BY season_name DESC LIMIT 1
        """
    ).fetchone()
    return row["season_name"] if row else "__none__"


def _deadline_date(events: list[dict], event: int):
    """The date of a gameweek's deadline, for looking up the snapshot price."""
    from datetime import datetime

    for e in events:
        if e["id"] == event:
            return datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00")).date()
    return None
