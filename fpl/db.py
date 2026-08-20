"""Postgres access.

Sync rather than async on purpose: these are batch cron jobs with no concurrency
to exploit, and psycopg's sync API keeps the ingest path readable.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import resources
from typing import Any, Iterator

import psycopg
import structlog
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from fpl.freshness import Source, SourceState, Status

log = structlog.get_logger()


def dsn() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


@contextmanager
def connect(url: str | None = None) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(url or dsn(), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def migrate(conn: psycopg.Connection) -> None:
    """Apply schema.sql. Every statement is IF NOT EXISTS, so this is idempotent
    and safe to run at the start of any job."""
    sql = resources.files("fpl").joinpath("schema.sql").read_text()
    conn.execute(sql)


# -- freshness ------------------------------------------------------------


def record_freshness(
    conn: psycopg.Connection,
    source: Source,
    status: Status,
    *,
    as_of: datetime | None = None,
    row_count: int | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO source_freshness (source, status, fetched_at, as_of, row_count, error)
        VALUES (%s, %s, now(), %s, %s, %s)
        ON CONFLICT (source) DO UPDATE SET
            status = EXCLUDED.status,
            fetched_at = EXCLUDED.fetched_at,
            as_of = EXCLUDED.as_of,
            row_count = EXCLUDED.row_count,
            error = EXCLUDED.error,
            updated_at = now()
        """,
        (source.value, status.value, as_of, row_count, error),
    )


def load_freshness(conn: psycopg.Connection) -> dict[Source, SourceState]:
    rows = conn.execute("SELECT * FROM source_freshness").fetchall()
    out: dict[Source, SourceState] = {}
    for r in rows:
        try:
            src = Source(r["source"])
        except ValueError:
            continue  # a source this version no longer knows about
        out[src] = SourceState(
            source=src,
            status=Status(r["status"]),
            fetched_at=r["fetched_at"],
            as_of=r["as_of"],
        )
    return out


# -- snapshots ------------------------------------------------------------


def write_player_snapshot(
    conn: psycopg.Connection, elements: list[dict[str, Any]], captured_on: datetime | None = None
) -> int:
    """Append today's player state. Re-running the same day overwrites rather
    than duplicating, so the job stays idempotent."""
    day = (captured_on or datetime.now(UTC)).date()
    rows = [
        (
            day, e["id"], e["code"], e["web_name"], e["team"], e["element_type"],
            e["now_cost"], e.get("selected_by_percent"), e["status"],
            e.get("chance_of_playing_next_round"), e.get("news") or None,
            e.get("ep_next"), e.get("minutes"), e.get("total_points"), Jsonb(e),
        )
        for e in elements
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO player_snapshots (
                captured_on, element_id, code, web_name, team_id, element_type,
                now_cost, selected_by, status, chance_next, news, ep_next,
                minutes, total_points, payload)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (captured_on, element_id) DO UPDATE SET
                now_cost = EXCLUDED.now_cost,
                selected_by = EXCLUDED.selected_by,
                status = EXCLUDED.status,
                chance_next = EXCLUDED.chance_next,
                news = EXCLUDED.news,
                ep_next = EXCLUDED.ep_next,
                minutes = EXCLUDED.minutes,
                total_points = EXCLUDED.total_points,
                payload = EXCLUDED.payload,
                captured_at = now()
            """,
            rows,
        )
    return len(rows)


def write_event_snapshot(
    conn: psycopg.Connection, events: list[dict[str, Any]], captured_on: datetime | None = None
) -> int:
    day = (captured_on or datetime.now(UTC)).date()
    rows = [
        (
            day, e["id"], e["deadline_time"], e["finished"], e["data_checked"],
            e["is_current"], e["is_next"], e.get("average_entry_score"),
        )
        for e in events
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO event_snapshots (captured_on, event_id, deadline_time,
                finished, data_checked, is_current, is_next, average_score)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (captured_on, event_id) DO UPDATE SET
                finished = EXCLUDED.finished,
                data_checked = EXCLUDED.data_checked,
                is_current = EXCLUDED.is_current,
                is_next = EXCLUDED.is_next,
                average_score = EXCLUDED.average_score
            """,
            rows,
        )
    return len(rows)


def write_fixtures(conn: psycopg.Connection, fixtures: list[dict[str, Any]]) -> int:
    rows = [
        (
            f["id"], f.get("event"), f.get("kickoff_time"), f["team_h"], f["team_a"],
            f.get("team_h_difficulty"), f.get("team_a_difficulty"),
            f.get("finished", False), Jsonb(f),
        )
        for f in fixtures
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO fixtures (fixture_id, event, kickoff_time, team_h, team_a,
                team_h_difficulty, team_a_difficulty, finished, payload)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (fixture_id) DO UPDATE SET
                event = EXCLUDED.event,
                kickoff_time = EXCLUDED.kickoff_time,
                team_h_difficulty = EXCLUDED.team_h_difficulty,
                team_a_difficulty = EXCLUDED.team_a_difficulty,
                finished = EXCLUDED.finished,
                payload = EXCLUDED.payload,
                updated_at = now()
            """,
            rows,
        )
    return len(rows)


def log_event(conn: psycopg.Connection, event_type: str, payload: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO events (event_type, payload) VALUES (%s, %s)",
        (event_type, Jsonb(payload)),
    )


# -- prior-season totals --------------------------------------------------


def write_player_seasons(conn: psycopg.Connection, rows: list[tuple[int, dict[str, Any]]]) -> int:
    """Store `history_past` entries. Keyed on code, so re-fetching is a no-op."""
    payload = [
        (
            code, s["season_name"], s.get("minutes") or 0, s.get("starts") or 0,
            s.get("total_points") or 0, Jsonb(s),
        )
        for code, s in rows
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO player_seasons (code, season_name, minutes, starts, total_points, payload)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (code, season_name) DO UPDATE SET
                minutes = EXCLUDED.minutes,
                starts = EXCLUDED.starts,
                total_points = EXCLUDED.total_points,
                payload = EXCLUDED.payload,
                fetched_at = now()
            """,
            payload,
        )
    return len(payload)


def load_player_seasons(conn: psycopg.Connection, season: str) -> dict[int, dict[str, Any]]:
    rows = conn.execute(
        "SELECT code, payload FROM player_seasons WHERE season_name = %s", (season,)
    ).fetchall()
    return {r["code"]: r["payload"] for r in rows}


def codes_missing_history(conn: psycopg.Connection, codes: list[int]) -> list[int]:
    """Which players we have never fetched history for.

    Absence of a row is ambiguous -- a genuine Premier League debutant has no
    history to store -- so this is paired with a marker row rather than retried
    forever. See `mark_history_fetched`.
    """
    rows = conn.execute(
        "SELECT DISTINCT code FROM player_seasons WHERE code = ANY(%s)", (codes,)
    ).fetchall()
    seen = {r["code"] for r in rows}
    return [c for c in codes if c not in seen]


def mark_history_fetched(conn: psycopg.Connection, codes: list[int]) -> None:
    """Record that a player was checked and genuinely has no prior season.

    Without this, every debutant is re-fetched on every run forever -- 138 wasted
    requests a day at the start of this season.
    """
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO player_seasons (code, season_name, minutes, starts, total_points, payload)
            VALUES (%s, '__none__', 0, 0, 0, '{}'::jsonb)
            ON CONFLICT (code, season_name) DO NOTHING
            """,
            [(c,) for c in codes],
        )


# -- overrides ------------------------------------------------------------


def load_overrides(conn: psycopg.Connection, event: int) -> dict[str, dict[str, Any]]:
    """Active overrides, keyed by web_name.

    Matched on name rather than code because that is what you have to hand when
    entering one under time pressure. `code` is stored when known and wins if
    both are present.
    """
    rows = conn.execute(
        """
        SELECT code, web_name, miss_events, ep_multiplier, reason
        FROM overrides
        WHERE expires_after IS NULL OR expires_after >= %s
        """,
        (event,),
    ).fetchall()
    return {
        (r["web_name"] or str(r["code"])): {
            "code": r["code"],
            "miss_events": set(r["miss_events"] or []),
            "ep_multiplier": float(r["ep_multiplier"]) if r["ep_multiplier"] is not None else None,
            "reason": r["reason"],
        }
        for r in rows
    }


# -- projections ----------------------------------------------------------


def write_projections(
    conn: psycopg.Connection,
    model_version: str,
    rows: list[dict[str, Any]],
) -> int:
    """Append this run's projections.

    Never updated. Each run is a separate record of what the model believed at a
    point in time, which is the entire validation harness: actuals arrive days
    later and the comparison against ep_next, price and prior-season PPG says
    whether there is any edge. Because these were written before the outcome
    existed, lookahead is structurally impossible.
    """
    payload = [
        (
            model_version, r["event"], r["code"], r["element_id"], r["ep"],
            r["ep_horizon"], r["now_cost"], r["baseline_ep_next"], r["baseline_ppg"],
            r["imputed"],
        )
        for r in rows
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO projections (model_version, event, code, element_id, ep,
                ep_horizon, now_cost, baseline_ep_next, baseline_ppg, imputed)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
            """,
            payload,
        )
    return len(payload)


def latest_snapshot(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """Most recent player snapshot, one row per player."""
    return conn.execute(
        """
        SELECT DISTINCT ON (element_id) element_id, code, web_name, team_id,
               element_type, now_cost, status, chance_next, ep_next, payload
        FROM player_snapshots
        ORDER BY element_id, captured_on DESC
        """
    ).fetchall()


def fixture_difficulties(conn: psycopg.Connection, events: list[int]) -> dict[int, list[tuple[int, int]]]:
    """Team id -> [(gameweek, difficulty)] over the given gameweeks."""
    rows = conn.execute(
        """
        SELECT event, team_h, team_a, team_h_difficulty, team_a_difficulty
        FROM fixtures WHERE event = ANY(%s)
        """,
        (events,),
    ).fetchall()
    out: dict[int, list[tuple[int, int]]] = {}
    for r in rows:
        if r["team_h_difficulty"] is not None:
            out.setdefault(r["team_h"], []).append((r["event"], r["team_h_difficulty"]))
        if r["team_a_difficulty"] is not None:
            out.setdefault(r["team_a"], []).append((r["event"], r["team_a_difficulty"]))
    for team in out:
        out[team].sort()
    return out


# -- recommendations ------------------------------------------------------


def write_recommendation(
    conn: psycopg.Connection, model_version: str, payload: dict[str, Any]
) -> int:
    """Persist before notifying.

    Order matters: the database write is the durable record, the message is a
    convenience. A webhook that has rotated must not lose the recommendation.
    """
    row = conn.execute(
        """
        INSERT INTO recommendations (model_version, event, verdict, kind, payload)
        VALUES (%s, %s, %s, %s, %s) RETURNING id
        """,
        (model_version, payload["event"], payload["verdict"], payload["kind"], Jsonb(payload)),
    ).fetchone()
    return row["id"]


def mark_notified(conn: psycopg.Connection, recommendation_id: int) -> None:
    conn.execute(
        "UPDATE recommendations SET notified_at = now() WHERE id = %s", (recommendation_id,)
    )
