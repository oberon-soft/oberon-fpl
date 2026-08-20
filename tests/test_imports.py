"""Every module imports cleanly.

Trivial, and it earns its place: `fpl/status.py` shipped importing
`load_freshness` from `fpl.freshness` when it lives in `fpl.db`. Nothing in the
suite touched that module, so the failure only appeared the first time the
command was run against a real database -- which on the cluster would have meant
a CrashLoopBackOff at 21:00 on a Friday rather than a red CI run.

Import-time errors in a CLI whose subcommands import lazily are invisible until
executed. This makes them visible.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import fpl

MODULES = [name for _, name, _ in pkgutil.walk_packages(fpl.__path__, "fpl.")]


def test_package_has_modules():
    assert MODULES, "no submodules discovered -- the walk is broken, not the code"


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str):
    importlib.import_module(module)


def test_cli_subcommands_resolve():
    """argparse accepts every advertised subcommand. Catches a command wired
    into the parser but never given a branch in main()."""
    from fpl.__main__ import main

    for command in ("ingest", "plan", "decide", "migrate", "status"):
        with pytest.raises(SystemExit) as exc:
            main([command, "--help"])
        assert exc.value.code == 0
