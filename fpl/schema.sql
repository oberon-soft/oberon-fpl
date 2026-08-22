-- oberon-fpl schema.
--
-- Two things worth knowing about the shape here.
--
-- First, `element_id` is reassigned every season by FPL; `code` is the stable
-- Opta player id and survives across seasons (`opta_code` is literally 'p' ||
-- code). Anything that needs to join across seasons must key on `code`.
--
-- Second, snapshots are append-only and never updated. Price and ownership are
-- not recoverable from the API after the fact -- it only ever returns today --
-- so a day not captured is a day gone. Everything else here can be rebuilt.

CREATE TABLE IF NOT EXISTS player_snapshots (
    captured_on     DATE        NOT NULL,
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    element_id      INTEGER     NOT NULL,
    code            INTEGER     NOT NULL,
    web_name        TEXT        NOT NULL,
    team_id         INTEGER     NOT NULL,
    element_type    INTEGER     NOT NULL,
    now_cost        INTEGER     NOT NULL,
    selected_by     NUMERIC(6,3),
    status          TEXT        NOT NULL,
    chance_next     INTEGER,
    news            TEXT,
    ep_next         NUMERIC(6,2),   -- FPL's own projection; our benchmark to beat
    minutes         INTEGER,
    total_points    INTEGER,
    payload         JSONB       NOT NULL,
    PRIMARY KEY (captured_on, element_id)
);

CREATE INDEX IF NOT EXISTS player_snapshots_code_idx
    ON player_snapshots (code, captured_on DESC);
CREATE INDEX IF NOT EXISTS player_snapshots_captured_idx
    ON player_snapshots (captured_on DESC);

-- Gameweek metadata as captured. Kept because `data_checked` flipping is the
-- trigger for a refit, and we want the history of when that happened.
CREATE TABLE IF NOT EXISTS event_snapshots (
    captured_on     DATE        NOT NULL,
    event_id        INTEGER     NOT NULL,
    deadline_time   TIMESTAMPTZ NOT NULL,
    finished        BOOLEAN     NOT NULL,
    data_checked    BOOLEAN     NOT NULL,
    is_current      BOOLEAN     NOT NULL,
    is_next         BOOLEAN     NOT NULL,
    average_score   INTEGER,
    PRIMARY KEY (captured_on, event_id)
);

CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id      INTEGER     PRIMARY KEY,
    event           INTEGER,
    kickoff_time    TIMESTAMPTZ,
    team_h          INTEGER     NOT NULL,
    team_a          INTEGER     NOT NULL,
    team_h_difficulty INTEGER,
    team_a_difficulty INTEGER,
    finished        BOOLEAN     NOT NULL DEFAULT false,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload         JSONB       NOT NULL
);

CREATE INDEX IF NOT EXISTS fixtures_event_idx ON fixtures (event);

-- Per-gameweek actuals. Backfills from element-summary, so unlike snapshots
-- this is reconstructable.
CREATE TABLE IF NOT EXISTS player_gameweeks (
    season          TEXT        NOT NULL,
    event           INTEGER     NOT NULL,
    code            INTEGER     NOT NULL,
    element_id      INTEGER     NOT NULL,
    minutes         INTEGER     NOT NULL,
    total_points    INTEGER     NOT NULL,
    payload         JSONB       NOT NULL,
    PRIMARY KEY (season, event, code)
);

-- Prior-season totals from element-summary's `history_past`.
--
-- This is the durable source for player rates. `bootstrap-static` carries the
-- same per-90 figures right now, but only because it has not reset yet -- those
-- fields hold last season's values until the opening kickoff and then zero out.
-- Keyed on `code` so it survives element_id reassignment between seasons.
CREATE TABLE IF NOT EXISTS player_seasons (
    code            INTEGER     NOT NULL,
    season_name     TEXT        NOT NULL,
    minutes         INTEGER     NOT NULL,
    starts          INTEGER     NOT NULL,
    total_points    INTEGER     NOT NULL,
    payload         JSONB       NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (code, season_name)
);

CREATE INDEX IF NOT EXISTS player_seasons_season_idx ON player_seasons (season_name);

-- Every projection we ever made, stamped with the model version that made it.
-- This is the whole validation harness: actuals arrive a few days later and the
-- comparison against `ep_next`, price and prior-season PPG tells us whether the
-- model has any edge. Forward-tested rather than backtested, so no lookahead is
-- structurally possible.
CREATE TABLE IF NOT EXISTS projections (
    made_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version   TEXT        NOT NULL,
    event           INTEGER     NOT NULL,
    code            INTEGER     NOT NULL,
    element_id      INTEGER     NOT NULL,
    ep              NUMERIC(6,3) NOT NULL,
    ep_horizon      NUMERIC(6,3),
    now_cost        INTEGER     NOT NULL,
    baseline_ep_next NUMERIC(6,2),
    baseline_ppg    NUMERIC(6,3),
    -- True when rates came from the price prior rather than observed play.
    -- Scored separately: imputed projections are a different claim about the
    -- world and deserve their own accuracy number.
    imputed         BOOLEAN     NOT NULL DEFAULT false,
    PRIMARY KEY (model_version, event, code, made_at)
);

CREATE INDEX IF NOT EXISTS projections_event_idx ON projections (event, model_version);

CREATE TABLE IF NOT EXISTS recommendations (
    id              BIGSERIAL   PRIMARY KEY,
    made_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version   TEXT        NOT NULL,
    event           INTEGER     NOT NULL,
    verdict         TEXT        NOT NULL,     -- READY | DEGRADED | BLOCKED
    kind            TEXT        NOT NULL,     -- plan | final | plan_changed
    payload         JSONB       NOT NULL,     -- squad, transfers, captain, reasoning
    notified_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS recommendations_event_idx
    ON recommendations (event, made_at DESC);

-- League members, for labelling output.
--
-- `player_first_name` and `player_last_name` are returned by the API and are
-- deliberately not stored. The model never reads a name -- it needs entry ids
-- and picks -- so the only place a name would appear is a line of text you
-- read. Collecting less is a better guarantee than protecting more, and it means
-- a database dump leaked from this cluster identifies nobody.
--
-- `entry_name` is the team name: a handle the manager chose, not their identity.
CREATE TABLE IF NOT EXISTS entries (
    entry_id        INTEGER     PRIMARY KEY,
    entry_name      TEXT        NOT NULL,
    is_self         BOOLEAN     NOT NULL DEFAULT false,
    first_seen      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Squads as confirmed at a deadline. Ours and rivals' use the same table
-- because they come from the same public endpoint.
CREATE TABLE IF NOT EXISTS entry_picks (
    entry_id        INTEGER     NOT NULL,
    event           INTEGER     NOT NULL,
    element_id      INTEGER     NOT NULL,
    position        INTEGER     NOT NULL,
    multiplier      INTEGER     NOT NULL,     -- 0 bench, 1 playing, 2 captain, 3 TC
    is_captain      BOOLEAN     NOT NULL,
    is_vice_captain BOOLEAN     NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entry_id, event, element_id)
);

-- Operational facts the API does not know. These live in the database rather
-- than the repo precisely because they must be addable at 16:00 on a Friday
-- without a CI run. FPL's availability flags were empty for all fifteen of the
-- opening squad while one of them was actually suspended, so this path is
-- load-bearing, not a convenience.
CREATE TABLE IF NOT EXISTS overrides (
    id              BIGSERIAL   PRIMARY KEY,
    code            INTEGER,
    web_name        TEXT,
    miss_events     INTEGER[]   NOT NULL DEFAULT '{}',
    ep_multiplier   NUMERIC(4,2),
    reason          TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_after   INTEGER,     -- gameweek after which this no longer applies
    CONSTRAINT overrides_identifies_player CHECK (code IS NOT NULL OR web_name IS NOT NULL)
);

-- Purchase prices, and therefore selling prices.
--
-- The one piece of state the public API will not give you: `my-team` publishes
-- selling prices but needs authentication. Reconstructed from a picks diff plus
-- the daily price snapshots, and checked every gameweek against FPL's published
-- `value`, which a correct set of purchase prices must reproduce exactly.
--
-- `confirmed` marks a purchase price the reconciliation has verified. Until
-- then it is the best inference from the transfer window, and callers should
-- prefer being generous over being tight -- a budget that is too conservative
-- makes holding your own squad infeasible.
CREATE TABLE IF NOT EXISTS squad_holdings (
    entry_id        INTEGER     NOT NULL,
    element_id      INTEGER     NOT NULL,
    purchase_price  INTEGER     NOT NULL,
    acquired_event  INTEGER     NOT NULL,
    confirmed       BOOLEAN     NOT NULL DEFAULT false,
    sold_event      INTEGER,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entry_id, element_id, acquired_event)
);

CREATE INDEX IF NOT EXISTS squad_holdings_active_idx
    ON squad_holdings (entry_id) WHERE sold_event IS NULL;

-- Weekly reconciliation of our reconstruction against FPL's published value.
-- Also settles, by observation, whether `value` is stated net of the sell-on
-- fee at all -- a question that cannot be answered before a deadline passes.
CREATE TABLE IF NOT EXISTS value_reconciliation (
    entry_id        INTEGER     NOT NULL,
    event           INTEGER     NOT NULL,
    reported_value  INTEGER     NOT NULL,
    market_total    INTEGER     NOT NULL,
    selling_total   INTEGER     NOT NULL,
    semantics       TEXT        NOT NULL,
    agrees          BOOLEAN     NOT NULL,
    checked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entry_id, event)
);

-- One row per source per run. The decide job reads the latest per source,
-- derives the phase, and evaluates the tolerance table against these.
CREATE TABLE IF NOT EXISTS source_freshness (
    source          TEXT        PRIMARY KEY,
    status          TEXT        NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL,
    as_of           TIMESTAMPTZ,
    row_count       INTEGER,
    error           TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- General audit log, following the puck pattern.
CREATE TABLE IF NOT EXISTS events (
    id              BIGSERIAL   PRIMARY KEY,
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_type      TEXT        NOT NULL,
    payload         JSONB       NOT NULL
);

CREATE INDEX IF NOT EXISTS events_type_occurred_idx
    ON events (event_type, occurred_at DESC);


-- Additive migrations.
--
-- `CREATE TABLE IF NOT EXISTS` is not a migration system. It does nothing at
-- all when the table already exists, so a column added to a definition above
-- reaches a fresh database and never reaches an existing one. That failed
-- silently: local development kept dropping and recreating tables, so the
-- schema was always current there, while the cluster ran for two days against a
-- `projections` table with no `imputed` column and every plan job died on it.
--
-- Any column added after a table's first release belongs here as well as in the
-- definition above. `ADD COLUMN IF NOT EXISTS` is idempotent, so both paths
-- converge: new databases get it from the CREATE, existing ones from the ALTER.
ALTER TABLE projections ADD COLUMN IF NOT EXISTS imputed BOOLEAN NOT NULL DEFAULT false;
