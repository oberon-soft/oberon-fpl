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
) -> Recommendation:
    current_ids = current.element_ids if current else []
    free_transfers = current.free_transfers if current else 1
    budget = current.budget if current else rules.budget

    solution = solve(
        candidates, rules, horizon=horizon, current=current_ids,
        free_transfers=free_transfers, budget=budget, chip=chip, costs=costs,
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
