"""Purchase prices, and the selling prices derived from them.

This is the one piece of state FPL will not hand you. The authenticated
`my-team` endpoint publishes selling prices directly; the public API does not,
so they have to be reconstructed -- and getting them wrong is not a rounding
error. Cost a held player at their market price while budgeting from FPL's
`value`, which is net of the sell-on fee, and the squad you already own no
longer fits its own budget. The solver does not object: it quietly sells
players to close the gap and presents those sales as recommendations.

Reconstruction has two halves. Purchase prices come from a picks diff plus the
daily price snapshots: a player first appearing in gameweek N was bought in the
window between deadlines, and the snapshots bound what they could have cost.

The second half is what makes it trustworthy. FPL publishes `value` per
gameweek, and a correct set of purchase prices must reproduce it exactly. That
is a checksum on every assumption in this module, evaluated weekly, and it
settles by observation a question that cannot currently be settled by reading:
whether `value` is stated net of the fee at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import structlog

log = structlog.get_logger()


def selling_price(purchase: int, now_cost: int) -> int:
    """What FPL pays you for a player, in 0.1m units.

    You keep half the profit, rounded down. A 0.1 rise returns nothing, 0.2
    returns 0.1, 0.3 still returns 0.1, 0.4 returns 0.2. Falls have no
    protection at all -- you sell at the market price and take the full loss.
    """
    if now_cost <= purchase:
        return now_cost
    return purchase + (now_cost - purchase) // 2


class ValueSemantics(StrEnum):
    """What FPL's `value` field turns out to mean.

    Undecidable before a deadline passes -- `last_deadline_value` is null and no
    entry has picks -- and it changes whether the sell-on fee needs modelling at
    all. So it is measured rather than assumed.
    """

    NET_OF_FEE = "net_of_fee"
    """`value` is the sum of selling prices. The fee is real and this module
    earns its place."""
    MARKET = "market"
    """`value` is the sum of current prices. No fee applies to team value and
    holdings tracking is unnecessary."""
    UNKNOWN = "unknown"
    """Not yet distinguishable -- typically because no price has moved, which is
    the state for the whole of gameweek one."""


@dataclass(frozen=True)
class Reconciliation:
    semantics: ValueSemantics
    reported_value: int
    market_total: int
    selling_total: int

    @property
    def agrees(self) -> bool:
        """Whether our tracked purchase prices reproduce FPL's figure."""
        return self.selling_total == self.reported_value

    @property
    def discrepancy(self) -> int:
        return self.reported_value - self.selling_total

    def explain(self) -> str:
        if self.semantics is ValueSemantics.UNKNOWN:
            return (
                f"value={self.reported_value} matches both market and selling "
                "totals -- no price has moved yet, nothing to distinguish"
            )
        if self.semantics is ValueSemantics.MARKET:
            return (
                f"value={self.reported_value} equals the market total; FPL is "
                "not applying a sell-on fee to team value"
            )
        if self.agrees:
            return (
                f"value={self.reported_value} matches our selling total; "
                "purchase prices are correct"
            )
        return (
            f"value={self.reported_value} but our selling total is "
            f"{self.selling_total} (off by {self.discrepancy}); a tracked "
            "purchase price is wrong"
        )


def reconcile(
    reported_value: int, market_total: int, selling_total: int
) -> Reconciliation:
    """Compare our reconstruction against FPL's published figure.

    Three outcomes worth distinguishing. If the market and selling totals are
    equal, nothing has moved and the check tells us nothing. If `value` matches
    the market total while they differ, there is no fee on team value and this
    module can be retired. Otherwise `value` is net of the fee, and whether it
    matches our selling total says if our purchase prices are right.
    """
    if market_total == selling_total:
        semantics = ValueSemantics.UNKNOWN
    elif reported_value == market_total:
        semantics = ValueSemantics.MARKET
    else:
        semantics = ValueSemantics.NET_OF_FEE
    return Reconciliation(semantics, reported_value, market_total, selling_total)


def purchase_price_candidates(
    prices_in_window: list[int], fallback: int
) -> list[int]:
    """Prices a player could have been bought at, cheapest first.

    A transfer happens between two deadlines, and prices move once a day, so the
    daily snapshots over that window enumerate every possibility. Cheapest first
    because a lower purchase price implies a higher selling price, so trying
    those first finds the assignment that reproduces `value` soonest when a
    discrepancy needs resolving.
    """
    return sorted(set(prices_in_window)) or [fallback]


def sell_prices(
    holdings: dict[int, int], now_costs: dict[int, int]
) -> dict[int, int]:
    """Selling price per element, for every player whose purchase price is known.

    Players missing from `holdings` are omitted rather than guessed. The caller
    falls back to market price for those, which is generous by at most half a
    rise -- the right direction to be wrong, since being too conservative makes
    holding your own squad infeasible.
    """
    return {
        element_id: selling_price(purchase, now_costs[element_id])
        for element_id, purchase in holdings.items()
        if element_id in now_costs
    }


def arrivals(previous: set[int], current: set[int]) -> set[int]:
    """Players who joined the squad between two gameweeks.

    Their purchase price is whatever they cost during that window, which the
    daily snapshots record. This is the only moment a purchase price is knowable
    rather than inferred, which is why seeding must not be missed.
    """
    return current - previous


def sync(
    conn,
    *,
    entry_id: int,
    event: int,
    picks: list[dict[str, Any]],
    deadline_date: Any,
    now_costs: dict[int, int],
    reported_value: int,
) -> tuple[dict[int, int], Reconciliation]:
    """Record this gameweek's squad, price any arrivals, and reconcile.

    Returns the selling prices to cost held players at, and the reconciliation
    that says whether to trust them.

    Purchase prices are taken from the snapshot at the gameweek's deadline. A
    player first appearing in gameweek N was bought between the previous deadline
    and this one, and most transfers are made close to the deadline -- but the
    real reason this works is that the snapshot exists at all. The API will never
    tell you what a player cost last Tuesday.
    """
    from fpl import db

    squad = {p["element"] for p in picks}
    previous = db.previous_picks(conn, entry_id, event)
    new = arrivals(previous, squad)
    departed = previous - squad

    db.write_entry_picks(conn, entry_id, event, picks)

    priced: list[tuple[int, int, int]] = []
    unpriced: list[int] = []
    for element_id in sorted(new):
        price = db.price_on_or_before(conn, element_id, deadline_date)
        if price is None:
            # No snapshot covers the window -- ingest was not running yet. Fall
            # back to the current price, which understates any rise since
            # purchase and so understates the selling price. Conservative, and
            # the reconciliation will flag it.
            price = now_costs.get(element_id)
            if price is None:
                unpriced.append(element_id)
                continue
            log.warning("purchase_price_inferred", element_id=element_id, gameweek=event)
        priced.append((element_id, price, event))

    if priced:
        db.record_holdings(conn, entry_id, priced)
    if departed:
        db.mark_sold(conn, entry_id, departed, event)
    if unpriced:
        log.warning("purchase_price_unknown", elements=unpriced, gameweek=event)

    held = db.load_holdings(conn, entry_id)
    prices = sell_prices(held, now_costs)

    market_total = sum(now_costs[e] for e in squad if e in now_costs)
    # Players without a tracked purchase price contribute their market price,
    # which is the same fallback the optimiser uses for them.
    selling_total = sum(prices.get(e, now_costs.get(e, 0)) for e in squad)

    rec = reconcile(reported_value, market_total, selling_total)
    db.write_reconciliation(conn, entry_id, event, rec)
    log.info(
        "holdings_synced",
        gameweek=event,
        arrivals=len(priced),
        departures=len(departed),
        semantics=str(rec.semantics),
        agrees=rec.agrees,
        detail=rec.explain(),
    )
    return prices, rec
