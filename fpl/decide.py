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

from fpl import db, holdings, notify, rivals
from fpl.client import FPLClient, FPLError
from fpl.config import CONFIG, MODEL_VERSION
from fpl.entry import load_squad
from fpl.freshness import Source, Status, Verdict, evaluate
from fpl.optimise import Rules
from fpl.phase import Phase, derive_phase
from fpl.plan import build_projections
from fpl.recommend import build, render, render_html
from fpl.scoring import Scoring

log = structlog.get_logger()

#: Phases in which a recommendation is worth producing at all. Outside these the
#: job exits immediately -- there is nothing to decide while a gameweek is in
#: progress or its stats are still settling.
ACTIONABLE = frozenset(
    {Phase.PRE_DEADLINE, Phase.NEWS_WINDOW, Phase.PLANNING, Phase.PRESEASON}
)


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

                # Selling prices come from the holdings the ingest job
                # maintains. Where a purchase price is known, cost the player at
                # what selling them returns; the rest fall back to market, which
                # is generous by at most half a rise.
                held = db.load_holdings(conn, CONFIG.entry_id)
                if held:
                    costs = holdings.sell_prices(
                        held, {e["id"]: e["now_cost"] for e in boot["elements"]}
                    )

        # League-relative tilt. Ownership cannot change the mean-optimal squad
        # -- the field's expected score does not depend on your choices -- so
        # this is purely a variance instrument, and it stays inert until the
        # standing and the calendar both say variance is worth buying.
        overlap: dict[int, float] = {}
        overlap_weight = 0.0
        standing_note = None
        if CONFIG.league_id and CONFIG.entry_id and event > 1:
            picks = db.rival_picks(conn, event - 1, exclude=CONFIG.entry_id)
            if picks:
                overlap = rivals.ownership(
                    picks, [p.element_id for p in candidates]
                )
                standing = _standing(client, conn, event, boot)
                if standing:
                    overlap_weight = rivals.overlap_weight(standing)
                    standing_note = rivals.describe(standing, overlap_weight)

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
            overlap=overlap,
            overlap_weight=overlap_weight,
        )
        if standing_note:
            recommendation.notes.append(standing_note)

        text = render(recommendation)
        payload = recommendation.to_payload()
        row_id = db.write_recommendation(conn, MODEL_VERSION, payload)

        # Run hourly, send rarely. The hourly schedule exists so the job can
        # react to team news without a cron expression that would eventually
        # fire at the wrong hour -- not so that it can mail you every hour.
        should_send, reason = _worth_sending(
            db.last_notified(conn, event), recommendation.kind, recommendation.digest()
        )
        if not should_send:
            log.info("send_suppressed", gameweek=event, reason=reason)
            print(text)
            return 0

        delivered = notify.send(
            text,
            html=render_html(recommendation),
            title=f"FPL GW{event}: {'hold' if recommendation.is_hold else 'transfer'}",
        )
        if delivered:
            db.mark_notified(conn, row_id)

        log.info(
            "decide_complete",
            gameweek=event,
            phase=str(state.phase),
            kind=recommendation.kind,
            send_reason=reason,
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


def _standing(client, conn, event: int, boot) -> rivals.Standing | None:
    """Everyone's points total, for the league-relative tilt.

    One call per member. Failures degrade to no tilt rather than no
    recommendation -- the points-maximising squad is the correct default, so
    losing this input costs nothing but the variance adjustment.
    """
    members = conn.execute("SELECT entry_id FROM entries").fetchall()
    if not members:
        return None

    histories: dict[int, dict] = {}
    for row in members:
        try:
            histories[row["entry_id"]] = client.entry_history(row["entry_id"])
        except FPLError:
            continue

    remaining = sum(1 for e in boot["events"] if e["id"] >= event)
    return rivals.standing_from_history(CONFIG.entry_id, histories, remaining)


def _worth_sending(
    previous: dict | None, kind: str, digest: dict
) -> tuple[bool, str]:
    """Whether this recommendation says anything the last one did not.

    Three things earn a message. The first plan once the data settles. The final
    confirmation before the deadline, which is sent even when identical, because
    "still this" the evening before is worth knowing. And any change of advice in
    between -- which is exactly the injury-news case the hourly schedule exists
    to catch.

    Everything else is silence. A recommendation that arrives unchanged every
    hour is one you learn to skim, and then the one that matters gets skimmed
    too.
    """
    if previous is None:
        return True, "first recommendation for this gameweek"
    if previous["kind"] != kind:
        return True, f"escalated from {previous['kind']} to {kind}"
    if previous["payload"].get("digest") != digest:
        return True, "advice changed since the last message"
    return False, "unchanged since the last message"
