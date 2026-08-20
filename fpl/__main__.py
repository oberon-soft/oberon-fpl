"""Entrypoint dispatch.

One image, three commands, differing only in the CronJob `args`. A single image
means the ingest and the model can never be different commits, which is the
version-skew bug that would otherwise only show up in production after a deploy.
"""

from __future__ import annotations

import argparse
import logging
import sys

import structlog


def _configure_logging(verbose: bool) -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if verbose else logging.INFO
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fpl")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="snapshot bootstrap-static and fixtures")
    sub.add_parser("plan", help="refit rates and project expected points")
    decide = sub.add_parser("decide", help="produce the recommendation to act on")
    decide.add_argument(
        "--force", action="store_true",
        help="ignore the phase and freshness gates; for inspecting what the "
             "model currently thinks, never for acting on",
    )
    sub.add_parser("migrate", help="apply the schema and exit")
    sub.add_parser("status", help="print the derived phase and freshness verdict")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "ingest":
        from fpl.ingest import run

        return run()

    if args.command == "migrate":
        from fpl import db

        with db.connect() as conn:
            db.migrate(conn)
        return 0

    if args.command == "decide":
        from fpl.decide import run

        return run(force=args.force)

    if args.command == "plan":
        from fpl.plan import run

        return run()

    if args.command == "status":
        from fpl.status import run

        return run()

    print(f"{args.command!r} is not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
