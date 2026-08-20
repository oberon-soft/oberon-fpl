"""Run the rule assertions against live payloads.

Invoked only by the scheduled contract workflow. Shares its assertions with the
recorded-fixture tests so there is exactly one description of what we believe
about the game, rather than two that can drift apart.

    python -m tests.contract <bootstrap.json> <fixtures.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fpl.client import assert_bootstrap_sane, bootstrap_is_informative

from tests import test_rules


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    bootstrap = json.loads(Path(argv[1]).read_text())
    fixtures = json.loads(Path(argv[2]).read_text())

    checks = [
        ("structurally sane", lambda: assert_bootstrap_sane(bootstrap)),
        ("scoring table shape", lambda: test_rules.test_scoring_table_is_read_not_hardcoded(bootstrap)),
        ("defensive contribution scored", lambda: test_rules.test_defensive_contribution_is_scored(bootstrap)),
        ("squad structure", lambda: test_rules.test_squad_structure_comes_from_the_api(bootstrap)),
        ("dc thresholds cover scoring positions", lambda: test_rules.test_dc_thresholds_cover_every_scoring_position(bootstrap)),
        ("transfer rules", lambda: test_rules.test_transfer_rules_match_expectations(bootstrap)),
        ("chips split into halves", lambda: test_rules.test_chips_are_split_into_halves(bootstrap)),
        ("opta_code derives from code", lambda: test_rules.test_opta_code_derives_from_code(bootstrap)),
        ("fixture difficulty range", lambda: test_rules.test_fixture_difficulty_is_within_range(fixtures)),
    ]

    failures = 0
    for name, check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 -- report every failure, not the first
            print(f"FAIL  {name}: {exc}")
            failures += 1
        else:
            print(f"ok    {name}")

    # Informative-ness is reported, never failed on: preseason emptiness is
    # legitimate and would otherwise page every morning in July.
    print(f"note  informative: {bootstrap_is_informative(bootstrap)}")
    print(f"note  players: {len(bootstrap['elements'])}, fixtures: {len(fixtures)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
