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
