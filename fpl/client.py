"""Read-only client for the FPL API.

Every endpoint here is public and unauthenticated. That is a deliberate
constraint, not an accident: this system recommends and never acts, so it needs
no session, no cookie and no write path. Your own squad is read back through the
same public entry endpoint used for rivals.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from fpl.config import FPL_API

log = structlog.get_logger()

USER_AGENT = "oberon-fpl/0.1 (+https://github.com/oberon-soft/oberon-fpl)"


class FPLError(RuntimeError):
    pass


class FPLClient:
    def __init__(self, base_url: str = FPL_API, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )

    def __enter__(self) -> FPLClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str) -> Any:
        r = self._client.get(path)
        if r.status_code != 200:
            raise FPLError(f"GET {path} -> {r.status_code}")
        return r.json()

    # -- global state ----------------------------------------------------

    def bootstrap(self) -> dict[str, Any]:
        """Players, teams, gameweeks, scoring rules. The one large payload."""
        return self._get("/bootstrap-static/")

    def fixtures(self) -> list[dict[str, Any]]:
        """All fixtures. Carries difficulty ratings and, once played, stats."""
        return self._get("/fixtures/")

    def player(self, element_id: int) -> dict[str, Any]:
        """Per-gameweek history, prior-season totals, upcoming fixtures."""
        return self._get(f"/element-summary/{element_id}/")

    # -- entries (yours and rivals' -- same endpoints, no auth) -----------

    def entry(self, entry_id: int) -> dict[str, Any]:
        return self._get(f"/entry/{entry_id}/")

    def entry_history(self, entry_id: int) -> dict[str, Any]:
        return self._get(f"/entry/{entry_id}/history/")

    def entry_picks(self, entry_id: int, gameweek: int) -> dict[str, Any]:
        """A squad as confirmed at that gameweek's deadline.

        Returns 404 until the deadline passes -- picks are private beforehand,
        including your own. That is why planning always works from the last
        confirmed squad rather than live state.
        """
        return self._get(f"/entry/{entry_id}/event/{gameweek}/picks/")

    def league_standings(self, league_id: int, page: int = 1) -> dict[str, Any]:
        return self._get(
            f"/leagues-classic/{league_id}/standings/?page_standings={page}"
        )

    def league_members(self, league_id: int) -> dict[int, str]:
        """Every entry in a league, keyed by entry id.

        Members appear in one of two places and you need both. Before any
        gameweek completes, `standings.results` is empty and everyone sits in
        `new_entries`. They migrate to `standings` once there is something to
        rank. Anyone joining mid-season reappears in `new_entries` until the next
        update, so reading only `standings` silently drops recent joiners.
        """
        payload = self.league_standings(league_id)
        members: dict[int, str] = {}
        for row in payload.get("new_entries", {}).get("results", []):
            members[row["entry"]] = row["entry_name"]
        for row in payload.get("standings", {}).get("results", []):
            members[row["entry"]] = row["entry_name"]
        return members


def assert_bootstrap_sane(boot: dict[str, Any]) -> None:
    """Guard against the failure mode that returns 200 and tells you nothing.

    A well-formed payload full of zeroes -- which is exactly what the API serves
    in preseason -- passes any check based on status code or schema. These are
    content assertions: they distinguish "fetched successfully" from "fetched
    something worth acting on".

    Raises FPLError on structural problems. Emptiness is reported by
    `bootstrap_is_informative` instead, since preseason emptiness is legitimate.
    """
    for key in ("elements", "teams", "events", "element_types", "game_config"):
        if key not in boot:
            raise FPLError(f"bootstrap missing {key!r}")
    if len(boot["teams"]) != 20:
        raise FPLError(f"expected 20 teams, got {len(boot['teams'])}")
    if len(boot["events"]) != 38:
        raise FPLError(f"expected 38 gameweeks, got {len(boot['events'])}")
    if not boot["elements"]:
        raise FPLError("bootstrap has no players")
    if "scoring" not in boot["game_config"]:
        raise FPLError("game_config missing scoring table")


def bootstrap_is_informative(boot: dict[str, Any]) -> bool:
    """True once the payload carries *current-season* signal.

    The obvious check -- do players have minutes -- does not work. Preseason,
    `minutes` and `total_points` on an element still hold last season's totals
    and only reset at the opening kickoff, so they look entirely healthy while
    describing a season that has finished.

    `strength_attack_home` is a better tell: FPL leaves it at zero until it has
    calibrated team ratings, and it is genuinely unusable until then. The
    authoritative signal, though, is whether any gameweek has actually completed.

    Callers use this to distinguish STALE from VALID-BUT-UNINFORMATIVE rather
    than silently modelling noise.
    """
    any_finished = any(e.get("finished") and e.get("data_checked") for e in boot["events"])
    attack = {t.get("strength_attack_home", 0) for t in boot["teams"]}
    return any_finished and attack != {0}
