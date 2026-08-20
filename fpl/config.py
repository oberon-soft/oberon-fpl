"""Model parameters, versioned with the code.

These live here rather than in the database on purpose: a recommendation is only
reproducible after the fact if the parameters that produced it moved in lockstep
with the model. Operational facts that change between deploys -- "this player is
suspended for GW1-2" -- are overrides and live in Postgres instead.

Bump MODEL_VERSION whenever a change here would alter a projection. It is stamped
onto every row written to `projections` and `recommendations`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

MODEL_VERSION = "0.1.0"

FPL_API = "https://fantasy.premierleague.com/api"


@dataclass(frozen=True)
class Shrinkage:
    """Two distinct corrections that a single constant used to conflate.

    `noise_k` handles sampling error: a rate measured over few minutes is
    unreliable and should be pulled toward the positional mean. The weight is
    minutes / (minutes + noise_k), so it correctly approaches zero shrinkage for
    a full-season player.

    `regression` is a separate, flat multiplicative pull toward the mean applied
    to everyone regardless of sample size. It captures the real tendency of last
    season's outliers to decline, which has nothing to do with measurement error.
    """

    noise_k: float = 150.0
    regression: float = 0.90


@dataclass(frozen=True)
class Fixture:
    """Difficulty -> team strength multipliers.

    v1 uses FPL's own difficulty ratings rather than bookmaker odds. Odds are
    better, but they add a credential, an external dependency and a failure mode.
    Phase 2.5 A/Bs them against this and keeps whichever actually improves
    projection correlation.
    """

    attack: dict[int, float] = field(
        default_factory=lambda: {1: 1.45, 2: 1.28, 3: 1.00, 4: 0.80, 5: 0.62}
    )
    opponent_xg: dict[int, float] = field(
        default_factory=lambda: {1: 0.85, 2: 1.05, 3: 1.35, 4: 1.70, 5: 2.10}
    )


@dataclass(frozen=True)
class Squad:
    horizon: int = 6
    bench_weight: float = 0.12
    #: Defensive-contribution thresholds by element_type. Not exposed by the API,
    #: so they are asserted here and checked by the contract test.
    dc_threshold: dict[int, int] = field(default_factory=lambda: {2: 10, 3: 12, 4: 12})


@dataclass(frozen=True)
class Config:
    shrinkage: Shrinkage = field(default_factory=Shrinkage)
    fixture: Fixture = field(default_factory=Fixture)
    squad: Squad = field(default_factory=Squad)

    #: Your FPL entry id. Used to read your own confirmed squad after each
    #: deadline via the public entry endpoint -- no authentication involved.
    entry_id: int | None = None
    #: Classic league id for rival ingestion (Phase 6). Backfills, so not urgent.
    league_id: int | None = None

    @classmethod
    def from_env(cls) -> Config:
        def _int(name: str) -> int | None:
            v = os.environ.get(name)
            return int(v) if v else None

        return cls(entry_id=_int("FPL_ENTRY_ID"), league_id=_int("FPL_LEAGUE_ID"))


CONFIG = Config.from_env()
