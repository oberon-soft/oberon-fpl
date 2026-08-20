from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def bootstrap() -> dict[str, Any]:
    """A real bootstrap-static payload, trimmed to 40 players.

    Recorded rather than fetched: CI must never depend on the live FPL API.
    Drift against the real thing is caught by the scheduled contract workflow,
    which is allowed to hit the network and is allowed to fail loudly.
    """
    return json.loads((FIXTURES / "bootstrap.json").read_text())


@pytest.fixture(scope="session")
def fixtures_payload() -> list[dict[str, Any]]:
    return json.loads((FIXTURES / "fixtures.json").read_text())


@pytest.fixture(scope="session")
def league() -> dict[str, Any]:
    """A classic league payload in its preseason shape: members present in
    `new_entries`, `standings` still empty.

    Synthetic rather than recorded. The real payload carries other people's
    names, and this repo is public.
    """
    return json.loads((FIXTURES / "league.json").read_text())
