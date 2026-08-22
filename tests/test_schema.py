"""Schema drift detection.

The bug this exists to prevent ran in production for two days. `CREATE TABLE IF
NOT EXISTS` does nothing when the table is already there, so a column added to a
definition reaches new databases and silently skips existing ones. Local
development kept dropping tables, so the schema was always current there, and
the cluster quietly ran against a `projections` table with no `imputed` column
until every plan job died on it.
"""

from __future__ import annotations

from importlib import resources

from fpl.db import expected_columns, schema_drift

SCHEMA = resources.files("fpl").joinpath("schema.sql").read_text()


def test_schema_parses_into_tables_and_columns():
    parsed = expected_columns(SCHEMA)
    assert len(parsed) >= 10
    assert "code" in parsed["player_snapshots"]
    assert "purchase_price" in parsed["squad_holdings"]


def test_constraint_clauses_are_not_mistaken_for_columns():
    parsed = expected_columns(SCHEMA)
    for columns in parsed.values():
        assert not {"PRIMARY", "CONSTRAINT", "UNIQUE", "CHECK"} & columns


def test_no_drift_against_itself():
    parsed = expected_columns(SCHEMA)
    assert schema_drift(parsed, parsed) == {}


def test_missing_column_is_detected():
    live = {"projections": {"event", "code"}}
    expected = {"projections": {"event", "code", "imputed"}}
    assert schema_drift(expected, live) == {"projections": {"imputed"}}


def test_missing_table_is_detected():
    assert schema_drift({"projections": {"event"}}, {}) == {"projections": {"event"}}


def test_extra_live_columns_are_not_drift():
    """Leftovers from a removed feature are harmless; missing columns break
    writes. The check is deliberately one-directional."""
    live = {"projections": {"event", "code", "retired_field"}}
    expected = {"projections": {"event", "code"}}
    assert schema_drift(expected, live) == {}


def test_every_column_added_after_release_has_an_alter():
    """Any column added to an existing table must also appear as an idempotent
    ALTER, or existing databases never receive it.

    Checked structurally rather than by convention: a CREATE TABLE alone is
    indistinguishable from a correct migration when read, and only differs
    against a database that already exists.
    """
    alters = {
        line.split("ADD COLUMN IF NOT EXISTS")[1].split()[0]
        for line in SCHEMA.splitlines()
        if "ADD COLUMN IF NOT EXISTS" in line
    }
    # `imputed` was the column that caused the outage; it must stay covered.
    assert "imputed" in alters


def test_alters_are_idempotent():
    """Migrate runs on every job start, so a non-idempotent ALTER would fail the
    second time and take the whole pipeline with it."""
    for line in SCHEMA.splitlines():
        if line.strip().startswith("ALTER TABLE"):
            assert "IF NOT EXISTS" in line, f"not idempotent: {line.strip()}"
