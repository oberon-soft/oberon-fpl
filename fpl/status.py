"""Print the derived phase and the freshness verdict.

Exists so that "why didn't it recommend anything?" always has a concrete answer
you can get in one command, rather than something you reverse-engineer from logs
at 22:50 on a Friday.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fpl import db
from fpl.freshness import evaluate
from fpl.phase import derive_phase


def run() -> int:
    with db.connect() as conn:
        db.migrate(conn)
        rows = conn.execute(
            """
            SELECT DISTINCT ON (event_id) event_id, deadline_time, finished,
                   data_checked, is_current, is_next
            FROM event_snapshots ORDER BY event_id, captured_on DESC
            """
        ).fetchall()
        states = db.load_freshness(conn)

    if not rows:
        print("no event snapshots yet -- run `fpl ingest` first")
        return 1

    events = [
        {
            "id": r["event_id"],
            "deadline_time": r["deadline_time"].isoformat().replace("+00:00", "Z"),
            "finished": r["finished"],
            "data_checked": r["data_checked"],
            "is_current": r["is_current"],
            "is_next": r["is_next"],
        }
        for r in rows
    ]

    state = derive_phase(events)
    readiness = evaluate(state.phase, states)
    now = datetime.now(UTC)

    print(f"phase            {state.phase}")
    print(f"next gameweek    {state.next_gameweek}")
    if state.next_deadline:
        print(f"deadline         {state.next_deadline:%Y-%m-%d %H:%M UTC}"
              f"  ({state.hours_to_deadline:.1f}h)")
    print(f"last settled GW  {state.last_settled_gameweek}")
    print(f"verdict          {readiness.explain()}")
    print()
    print(f"{'source':<18}{'status':<16}{'age':>10}")
    for source, s in sorted(states.items()):
        age = s.age_hours(now)
        print(f"{source:<18}{s.status:<16}{f'{age:.1f}h' if age is not None else '-':>10}")

    return 0
