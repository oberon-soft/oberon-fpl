"""Your squad, read back from the public entry endpoint.

There is no authentication anywhere in this system, so the squad it plans around
is the one confirmed at the last deadline -- picks are private until then, your
own included. That is not the limitation it sounds like: planning wants your last
*confirmed* squad, and you only act once a week anyway.

Free transfers are derived rather than read. FPL exposes transfers made per
gameweek but not the balance remaining, so the balance is reconstructed from the
history: one per gameweek, banking up to the configured cap, spent by transfers
made. Gameweeks where a wildcard or free hit was active are skipped, since those
transfers are free and do not consume the allowance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from fpl.client import FPLClient, FPLError

log = structlog.get_logger()

#: Chips that make a gameweek's transfers free, so they do not spend the bank.
UNLIMITED_TRANSFER_CHIPS = frozenset({"wildcard", "freehit"})


@dataclass(frozen=True)
class Squad:
    entry_id: int
    event: int
    element_ids: list[int]
    captain: int | None
    vice_captain: int | None
    bank: int
    value: int
    free_transfers: int
    chips_used: list[str] = field(default_factory=list)

    #: Sum of `now_cost` for the players currently held, supplied by the caller
    #: because this module does not see the price table. Falls back to `value`
    #: when unavailable, which is the conservative choice.
    holdings_at_current_price: int | None = None

    @property
    def budget(self) -> int:
        """What a replacement squad may cost, valued at current prices.

        The subtlety that makes this not simply `value + bank`: FPL's `value` is
        the sum of *selling* prices, reduced by the 50% sell-on fee, while the
        optimiser costs every candidate at `now_cost`. Mixing the two makes the
        constraint too tight -- for a squad of risers, `sum(now_cost)` exceeds
        `value`, and the solver would declare your own current squad infeasible
        rather than offering to hold it.

        So the budget is stated in the same units as the costs: current prices
        of what you hold, plus the bank.

        This is exact while you keep a player and slightly generous when you sell
        one who has risen -- you are credited `now_cost` where FPL would pay the
        selling price, an error of at most half the rise, rounded up. Generous
        is the right direction to be wrong in here, because the alternative is a
        solver that refuses to return any squad at all.

        Exact selling prices need purchase prices, which only the authenticated
        `my-team` endpoint publishes. They can be reconstructed instead: record
        each player's price when they first appear in a picks diff, then use
        `value` as a per-gameweek checksum, since a correct set of purchase
        prices must reproduce it exactly.
        """
        if self.holdings_at_current_price is not None:
            return self.holdings_at_current_price + self.bank
        return self.value + self.bank


def free_transfers_from_history(
    history: dict[str, Any], *, cap: int, current_event: int
) -> int:
    """Reconstruct the free transfer balance.

    `cap` is 1 + max_extra_free_transfers, read from game settings rather than
    assumed -- FPL changed this rule recently and will again.
    """
    chip_events = {
        c["event"] for c in history.get("chips", []) if c.get("name") in UNLIMITED_TRANSFER_CHIPS
    }
    available = 1
    for row in history.get("current", []):
        event = row["event"]
        if event >= current_event:
            break
        used = 0 if event in chip_events else (row.get("event_transfers") or 0)
        available = min(cap, max(0, available - used) + 1)
    return max(1, available)


def load_squad(
    client: FPLClient, entry_id: int, event: int, *, transfer_cap: int
) -> Squad | None:
    """The squad confirmed at `event`'s deadline, or None if it has not passed.

    A 404 here is the normal state before a deadline, not a fault. Callers treat
    it as "no confirmed squad yet" and fall back to a from-scratch solve, which
    is exactly right for GW1.
    """
    try:
        picks = client.entry_picks(entry_id, event)
    except FPLError as exc:
        if "404" in str(exc):
            log.info("no_confirmed_squad", entry_id=entry_id, gameweek=event)
            return None
        raise

    try:
        history = client.entry_history(entry_id)
    except FPLError:
        history = {}

    entry = picks.get("entry_history") or {}
    captain = next((p["element"] for p in picks["picks"] if p.get("is_captain")), None)
    vice = next((p["element"] for p in picks["picks"] if p.get("is_vice_captain")), None)

    return Squad(
        entry_id=entry_id,
        event=event,
        element_ids=[p["element"] for p in picks["picks"]],
        captain=captain,
        vice_captain=vice,
        bank=entry.get("bank", 0),
        value=entry.get("value", 0),
        free_transfers=free_transfers_from_history(
            history, cap=transfer_cap, current_event=event + 1
        ),
        chips_used=[c["name"] for c in history.get("chips", [])],
    )
