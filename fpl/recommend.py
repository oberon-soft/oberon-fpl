"""Turn projections into something you can act on.

The rendered output names its own uncertainty. A recommendation that says only
"transfer A for B" invites more confidence than the model has earned, so the
margin over the next-best alternative is always shown -- when it is small, the
right reading is that the model is indifferent and you should decide on grounds
it cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from fpl.entry import Squad
from fpl.freshness import Readiness, Verdict
from fpl.optimise import TRANSFER_COST, Chip, Rules, Solution, solve
from fpl.project import Projection

#: A transfer whose horizon gain is below this is inside the model's own error.
#: Acting on it is churn: the argmax flips on noise, and you burn the free
#: transfer you would rather have banked.
MARGIN_OF_INDIFFERENCE = 0.15


@dataclass(frozen=True)
class Recommendation:
    event: int
    kind: str
    verdict: Verdict
    solution: Solution
    positions: dict[int, str]
    teams: dict[int, str]
    current: Squad | None = None
    readiness: Readiness | None = None
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_hold(self) -> bool:
        return not self.solution.transfers_in

    def digest(self) -> dict[str, Any]:
        """The decision-relevant subset, for deciding whether to send again.

        Deliberately narrow: who to buy, who to sell, who to captain, which chip.
        Projections drift a little every hour and the squad ordering shifts with
        them, but none of that changes what you would do -- and a message that
        arrives when nothing has changed teaches you to stop reading it.
        """
        captain = self.solution.captains.get(self.event)
        return {
            "in": sorted(p.element_id for p in self.solution.transfers_in),
            "out": sorted(p.element_id for p in self.solution.transfers_out),
            "captain": captain.element_id if captain else None,
            "chip": str(self.solution.chip),
            "hits": self.solution.hits,
        }

    def to_payload(self) -> dict[str, Any]:
        """Structured form for the database. Kept separate from the rendered text
        so the record stays queryable rather than being a blob of prose."""
        return {
            "event": self.event,
            "kind": self.kind,
            "verdict": str(self.verdict),
            "chip": str(self.solution.chip),
            "hold": self.is_hold,
            "hits": self.solution.hits,
            "spend": self.solution.spend,
            "objective": round(self.solution.objective, 3),
            "transfers_in": [
                {"element_id": p.element_id, "name": p.web_name, "cost": p.now_cost}
                for p in self.solution.transfers_in
            ],
            "transfers_out": [
                {"element_id": p.element_id, "name": p.web_name, "cost": p.now_cost}
                for p in self.solution.transfers_out
            ],
            "squad": [
                {
                    "element_id": p.element_id,
                    "name": p.web_name,
                    "position": self.positions[p.element_type],
                    "team": self.teams[p.team_id],
                    "cost": p.now_cost,
                    "ep": round(p.ep_next, 3),
                    "ep_horizon": round(p.ep_horizon, 3),
                    "starting": p in self.solution.starting,
                }
                for p in self.solution.squad
            ],
            "captains": {
                str(gw): p.web_name for gw, p in sorted(self.solution.captains.items())
            },
            "digest": self.digest(),
            "alternatives": [{"name": n, "delta": round(d, 3)} for n, d in self.alternatives],
            "notes": self.notes,
        }


def _alternatives(
    candidates: Sequence[Projection],
    rules: Rules,
    solution: Solution,
    *,
    horizon: Sequence[int],
    current: Sequence[int],
    free_transfers: int,
    budget: int,
    costs: dict[int, int] | None = None,
    limit: int = 3,
) -> list[tuple[str, float]]:
    """What the next-best moves would have been, and by how much they lose.

    Re-solves with each recommended signing banned. If the best alternative is
    barely behind, the model has no real preference and the output should say so
    rather than presenting one option as the answer.
    """
    if not solution.transfers_in:
        return []

    banned = {p.element_id for p in solution.transfers_in}
    pool = [p for p in candidates if p.element_id not in banned]
    try:
        rival = solve(
            pool, rules, horizon=horizon, current=current,
            free_transfers=free_transfers, budget=budget, costs=costs,
        )
    except Exception:
        return []

    if not rival.transfers_in:
        return [("hold instead", rival.objective - solution.objective)]
    return [
        (p.web_name, rival.objective - solution.objective) for p in rival.transfers_in
    ][:limit]


def build(
    candidates: Sequence[Projection],
    rules: Rules,
    *,
    event: int,
    horizon: Sequence[int],
    kind: str,
    readiness: Readiness,
    positions: dict[int, str],
    teams: dict[int, str],
    current: Squad | None = None,
    chip: Chip = Chip.NONE,
    costs: dict[int, int] | None = None,
    overlap: dict[int, float] | None = None,
    overlap_weight: float = 0.0,
) -> Recommendation:
    current_ids = current.element_ids if current else []
    free_transfers = current.free_transfers if current else 1
    budget = current.budget if current else rules.budget

    solution = solve(
        candidates, rules, horizon=horizon, current=current_ids,
        free_transfers=free_transfers, budget=budget, chip=chip, costs=costs,
        overlap=overlap, overlap_weight=overlap_weight,
    )

    alternatives = _alternatives(
        candidates, rules, solution, horizon=horizon, current=current_ids,
        free_transfers=free_transfers, budget=budget, costs=costs,
    )

    notes = list(solution.notes)
    if current is None:
        notes.append("no confirmed squad yet -- solved from scratch")
    if any(p.imputed for p in solution.squad):
        imputed = [p.web_name for p in solution.squad if p.imputed]
        notes.append(f"rates inferred from price for {', '.join(imputed)}")
    if alternatives and abs(alternatives[0][1]) < MARGIN_OF_INDIFFERENCE:
        notes.append(
            f"margin over the next-best option is {abs(alternatives[0][1]):.2f} pts/gw "
            "-- inside model error, so treat this as a coin flip"
        )

    return Recommendation(
        event=event, kind=kind, verdict=readiness.verdict, solution=solution,
        positions=positions, teams=teams, current=current, readiness=readiness,
        alternatives=alternatives, notes=notes,
    )


def render(rec: Recommendation) -> str:
    """Plain text, readable in a terminal or a chat client."""
    sol = rec.solution
    lines: list[str] = []

    header = f"GW{rec.event} — {rec.kind}"
    if sol.chip is not Chip.NONE:
        header += f" (chip: {sol.chip})"
    lines.append(header)
    lines.append("=" * len(header))

    if rec.verdict is not Verdict.READY:
        lines.append(f"[{rec.verdict}] {rec.readiness.explain() if rec.readiness else ''}")
        lines.append("")

    if rec.current is None:
        lines.append("No confirmed squad — this is a full squad selection.")
    elif rec.is_hold:
        lines.append(f"HOLD. No transfer worth making; bank the free transfer.")
        lines.append(f"Free transfers available: {rec.current.free_transfers}")
    else:
        for out_p, in_p in zip(sol.transfers_out, sol.transfers_in):
            lines.append(
                f"TRANSFER  {out_p.web_name} ({out_p.now_cost / 10:.1f}) "
                f"-> {in_p.web_name} ({in_p.now_cost / 10:.1f})"
            )
        if sol.hits:
            lines.append(f"Cost: {sol.hit_cost} pts ({sol.hits} hit)")
        else:
            lines.append("Cost: free")

    lines.append("")
    next_gw = rec.event
    captain = sol.captains.get(next_gw)
    if captain:
        lines.append(f"CAPTAIN   {captain.web_name}  ({captain.ep_next:.2f} pts projected)")

    lines.append("")
    lines.append(f"{'':<4}{'pos':<5}{'player':<18}{'team':<6}{'£':>5}  {'ep':>5}")
    for p in sol.starting:
        mark = "(C)" if captain and p.element_id == captain.element_id else ""
        lines.append(
            f"XI  {rec.positions[p.element_type]:<5}{p.web_name[:17]:<18}"
            f"{rec.teams[p.team_id]:<6}{p.now_cost / 10:>5.1f}  {p.ep_next:>5.2f} {mark}"
        )
    for p in sol.bench:
        lines.append(
            f"sub {rec.positions[p.element_type]:<5}{p.web_name[:17]:<18}"
            f"{rec.teams[p.team_id]:<6}{p.now_cost / 10:>5.1f}  {p.ep_next:>5.2f}"
        )

    lines.append("")
    lines.append(f"Spend {sol.spend / 10:.1f}  ·  projected {sol.objective:.1f} pts/gw over the horizon")

    if len(sol.captains) > 1:
        rota = ", ".join(f"GW{gw}={p.web_name}" for gw, p in sorted(sol.captains.items()))
        lines.append(f"Captain plan: {rota}")

    if rec.alternatives:
        lines.append("")
        lines.append("Next best:")
        for name, delta in rec.alternatives:
            lines.append(f"  {name:<20}{delta:+.2f} pts/gw")

    if rec.notes:
        lines.append("")
        for note in rec.notes:
            lines.append(f"note: {note}")

    return "\n".join(lines)


# -- HTML rendering -------------------------------------------------------
#
# Not a monospace <pre>. Two rounds of font fixes failed to align the columns,
# because mail clients sanitise CSS unpredictably and a lost font-family is
# invisible until an accented name shifts a row. Tables are the one layout
# primitive email clients have handled reliably for twenty years, so the columns
# are real cells and alignment stops depending on the renderer's font choice.
#
# Everything is inline-styled with no external stylesheet, no flexbox and no
# grid, for the same reason.

_CELL = "padding:3px 10px 3px 0;font-size:14px;border-bottom:1px solid #eee"
_HEAD = "padding:0 10px 4px 0;font-size:11px;text-transform:uppercase;letter-spacing:.05em;opacity:.6;text-align:left"
_NUM = "text-align:right;font-variant-numeric:tabular-nums"


def _esc(value: object) -> str:
    return (
        str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _squad_table(rec: Recommendation) -> str:
    sol = rec.solution
    captain = sol.captains.get(rec.event)
    rows: list[str] = [
        "<tr>"
        f'<th style="{_HEAD}"></th>'
        f'<th style="{_HEAD}">Pos</th>'
        f'<th style="{_HEAD}">Player</th>'
        f'<th style="{_HEAD}">Team</th>'
        f'<th style="{_HEAD};{_NUM}">£m</th>'
        f'<th style="{_HEAD};{_NUM}">EP</th>'
        "</tr>"
    ]
    for group, players in (("XI", sol.starting), ("Bench", sol.bench)):
        for i, p in enumerate(players):
            is_captain = captain is not None and p.element_id == captain.element_id
            label = group if i == 0 else ""
            name = _esc(p.web_name) + (
                ' <strong style="color:#1a7f37">(C)</strong>' if is_captain else ""
            )
            dim = "" if group == "XI" else "opacity:.55;"
            rows.append(
                "<tr>"
                f'<td style="{_CELL};{dim}font-size:11px;opacity:.5">{label}</td>'
                f'<td style="{_CELL};{dim}">{_esc(rec.positions[p.element_type])}</td>'
                f'<td style="{_CELL};{dim}">{name}</td>'
                f'<td style="{_CELL};{dim}">{_esc(rec.teams[p.team_id])}</td>'
                f'<td style="{_CELL};{_NUM};{dim}">{p.now_cost / 10:.1f}</td>'
                f'<td style="{_CELL};{_NUM};{dim}">{p.ep_next:.2f}</td>'
                "</tr>"
            )
    return (
        '<table cellpadding="0" cellspacing="0" border="0" '
        'style="border-collapse:collapse;width:100%;max-width:520px">'
        + "".join(rows)
        + "</table>"
    )


def render_html(rec: Recommendation) -> str:
    """The recommendation as an email body.

    Deliberately colour-light: mail clients invert dark mode unpredictably, so
    the design leans on weight and spacing rather than background fills, and
    specifies no page background at all.
    """
    sol = rec.solution
    out: list[str] = [
        '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,'
        'Helvetica,Arial,sans-serif;max-width:560px">'
    ]

    heading = f"GW{rec.event} &middot; {_esc(rec.kind)}"
    if sol.chip is not Chip.NONE:
        heading += f" &middot; {_esc(sol.chip)}"
    out.append(f'<h2 style="margin:0 0 2px;font-size:19px">{heading}</h2>')

    if rec.verdict is not Verdict.READY:
        detail = _esc(rec.readiness.explain()) if rec.readiness else str(rec.verdict)
        out.append(
            '<p style="margin:8px 0;padding:8px 10px;border-left:3px solid #b45309;'
            f'font-size:13px">{detail}</p>'
        )

    # The action, stated before anything else. This is the whole message.
    if rec.current is None:
        action = "Full squad selection — no confirmed squad yet."
    elif rec.is_hold:
        action = (
            "<strong>Hold.</strong> No transfer worth making; bank the free "
            f"transfer ({rec.current.free_transfers} available)."
        )
    else:
        moves = " &nbsp;·&nbsp; ".join(
            f"{_esc(o.web_name)} &rarr; <strong>{_esc(i.web_name)}</strong>"
            for o, i in zip(sol.transfers_out, sol.transfers_in)
        )
        cost = f"−{sol.hit_cost} pts ({sol.hits} hit)" if sol.hits else "free"
        action = f"{moves}<br><span style='opacity:.6'>Cost: {cost}</span>"
    out.append(f'<p style="margin:10px 0 14px;font-size:15px">{action}</p>')

    captain = sol.captains.get(rec.event)
    if captain:
        out.append(
            '<p style="margin:0 0 14px;font-size:15px">Captain: '
            f"<strong>{_esc(captain.web_name)}</strong> "
            f'<span style="opacity:.6">({captain.ep_next:.2f} projected)</span></p>'
        )

    out.append(_squad_table(rec))
    out.append(
        '<p style="margin:12px 0 0;font-size:13px;opacity:.7">'
        f"Spend {sol.spend / 10:.1f} &middot; {sol.objective:.1f} pts/gw projected "
        "over the horizon</p>"
    )

    if len(sol.captains) > 1:
        rota = ", ".join(
            f"GW{gw} {_esc(p.web_name)}" for gw, p in sorted(sol.captains.items())
        )
        out.append(
            f'<p style="margin:4px 0 0;font-size:13px;opacity:.7">Captain plan: {rota}</p>'
        )

    if rec.alternatives:
        items = "".join(
            f"<li>{_esc(name)} <span style='opacity:.6'>{delta:+.2f} pts/gw</span></li>"
            for name, delta in rec.alternatives
        )
        out.append(
            '<p style="margin:14px 0 4px;font-size:13px;font-weight:600">Next best</p>'
            f'<ul style="margin:0;padding-left:18px;font-size:13px">{items}</ul>'
        )

    if rec.notes:
        notes = "".join(f"<li>{_esc(n)}</li>" for n in rec.notes)
        out.append(
            '<ul style="margin:14px 0 0;padding-left:18px;font-size:12px;opacity:.65">'
            f"{notes}</ul>"
        )

    out.append("</div>")
    return "".join(out)
