# oberon-fpl

Fantasy Premier League projection and squad optimisation.

**Recommends, never acts.** Every endpoint it touches is public and
unauthenticated. There is no session, no cookie, and no write path — it tells you
what to do and you click it.

## Why it is shaped this way

**Three layers, one direction of flow.**

```
projections    expected points per player per gameweek
     ↓
optimiser      MILP over the squad constraints → recommendation
     ↓         (and, with a mode flag, chip_gain scalars)
stopping       spend a one-shot chip now, or hold
```

The stopping layer only ever sees one scalar per chip per gameweek, so
everything upstream — 599 players, Poisson, shrinkage, the solver — collapses at
that interface. It is testable in isolation with synthetic inputs, which matters
for something making irreversible decisions.

**Transfer banking lives inside the optimiser; chips do not.** Transfers
regenerate weekly and a six-gameweek deterministic lookahead captures most of
their value. Chips are one-shot across nineteen gameweeks and their value is
driven by tail events, which a point-estimate horizon cannot represent. Short
horizon and renewable → fold in. Long horizon, one-shot, tail-driven → break out.

**Forward-tested, not backtested.** Every projection is written to `projections`
when it is made, stamped with `MODEL_VERSION`, alongside three baselines: FPL's
own `ep_next`, price, and prior-season points per game. Actuals arrive days later.
Because the projections were recorded before the outcome existed, lookahead is
structurally impossible — which is not true of any backtest built on a post-hoc
archive. If it cannot beat `ep_next`, there is no edge, and you learn that in a
fortnight rather than a season.

## Layout

| Path | What it does |
|---|---|
| `fpl/client.py` | Read-only FPL API client, plus content assertions |
| `fpl/phase.py` | Gameweek phase, derived from the API rather than stored |
| `fpl/freshness.py` | Per-source staleness and the gate on acting |
| `fpl/config.py` | Model parameters, versioned with the code |
| `fpl/schema.sql` | Postgres tables |
| `fpl/ingest.py` | Daily snapshot |
| `tests/contract.py` | Rule assertions run against the live API on a schedule |

## Two ideas worth knowing before reading the code

**Phase is derived, never stored.** `finished`, `data_checked`, `is_current` and
`deadline_time_epoch` already describe the state. A stored copy drifts after a
missed job or a postponement; a derived one cannot. This is also why the cron
schedules are dumb — deadlines move constantly with TV scheduling, so `decide`
runs hourly and gates itself on the phase.

**Staleness is relative to what you are about to do.** Availability flags are
irrelevant while planning and critical three hours before a deadline, so
`TOLERANCE` is indexed by phase. Most cells are generous; only `PRE_DEADLINE` is
tight.

The invariant the tests pin down: **BLOCKED never degrades into a recommendation.**
Silence plus a firing alert beats a recommendation quietly built on last week's
numbers.

## Three failure categories, not two

`STALE` — fetch failed or never ran.

`INVALID` — fetched, but malformed or violating an invariant.

`UNINFORMATIVE` — well-formed, recently fetched, and carrying no signal. This is
the dangerous one. Preseason, `minutes` and `total_points` still hold *last
season's* totals and only reset at the opening kickoff, so a check based on
"do players have minutes" waves it straight through. `bootstrap_is_informative`
keys on whether a gameweek has actually settled instead.

## Overrides

FPL's availability flags were empty for all fifteen players of the opening squad
while one of them was suspended. Overrides therefore live in Postgres, not the
repo — they must be addable at 16:00 on a Friday without a CI run.

Model *parameters* go the other way, into `config.py`, because a recommendation
is only reproducible if the parameters moved in lockstep with the code.

## Commands

```bash
fpl migrate    # apply schema, exit
fpl ingest     # snapshot bootstrap-static and fixtures
fpl status     # derived phase + freshness verdict — "why no recommendation?"
fpl plan       # refit and draft            (Phase 2/3)
fpl decide     # the answer you act on      (Phase 4)
```

One image, three commands, differing only in CronJob `args` — so ingest and model
can never be different commits.

## Development

```bash
uv sync
uv run pytest
```

CI runs against recorded payloads and never touches the live API. Drift is caught
by the scheduled `contract` workflow, which is allowed to hit the network and to
fail loudly without blocking a deploy. It will fire eventually — defensive
contribution and the `price_change_*` block are both recent additions.

## Configuration

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection string |
| `FPL_ENTRY_ID` | Your entry id — reads your confirmed squad after each deadline |
| `FPL_LEAGUE_ID` | Classic league id for rival ingestion (Phase 6) |

All three come from the `fpl-credentials` Secret via `envFrom`, applied by hand
and never committed:

```bash
kubectl create secret generic fpl-credentials -n fpl \
  --from-literal=DATABASE_URL=postgresql://... \
  --from-literal=FPL_ENTRY_ID=... \
  --from-literal=FPL_LEAGUE_ID=...
```

Your own picks come from the same public endpoint as everyone else's, one
gameweek behind. That is not a limitation: planning wants your last *confirmed*
squad, and picks are private until the deadline passes anyway.

## On other people's data

The league endpoint returns `player_first_name` and `player_last_name` for every
member. They are dropped in `league_members` and never stored.

This is not access control, it is scope. The model needs entry ids and picks; a
name only ever reaches a line of text you read. Collecting less is a stronger
guarantee than protecting more — a dump of this database identifies nobody, and
there is no secret to rotate or leak.

The same rule governs the test fixtures. `tests/fixtures/league.json` is
synthetic. Recorded payloads in this repo must never contain real entry ids,
team names or manager names, because the repo is public and rival picks are other
people's data even though the API serves them openly.

## Status

Phase 1 complete: client, phase machine, freshness gate, schema, ingest.
Phase 2 (projections) and Phase 3 (optimiser) next.
