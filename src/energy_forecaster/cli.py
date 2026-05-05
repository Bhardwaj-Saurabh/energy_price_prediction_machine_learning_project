"""Energy Forecaster command-line interface.

Single entrypoint exposed by the ``[project.scripts]`` table in
``pyproject.toml``. Subcommands map 1-to-1 onto use cases — the CLI is a
*framework* in the clean-architecture sense, just like the future FastAPI
app: it parses arguments, calls the composition root to build a wired use
case, executes it, and renders the result. No business logic lives here.

Adding a subcommand: register it in :func:`_build_parser`, and add a
corresponding ``_run_<command>`` handler that takes the parsed
``argparse.Namespace`` and returns an exit code.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from energy_forecaster.application.errors import ApplicationError
from energy_forecaster.application.use_cases.ingest_entsoe_load import (
    IngestEntsoeLoadResult,
)
from energy_forecaster.composition import build_ingest_entsoe_load
from energy_forecaster.config.settings import get_settings
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for ``energy-forecaster ...``. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return _run_ingest(args)

    # argparse's `required=True` on the subparsers above exits with code 2
    # before reaching this point. The guard catches the case where a future
    # subcommand is registered in the parser but not handled here.
    raise AssertionError(f"Unknown command: {args.command}")  # pragma: no cover


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="energy-forecaster",
        description="Energy demand & price forecaster — local CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest",
        help="Fetch load observations for one or more zones over a window.",
        description=(
            "Run the IngestEntsoeLoad use case end-to-end against the "
            "configured adapters. Outputs a summary of how many "
            "observations were fetched and persisted."
        ),
    )
    ingest.add_argument(
        "--zone",
        action="append",
        required=True,
        choices=[z.value for z in BiddingZone],
        help="Bidding zone (repeatable). Example: --zone DE_LU --zone FR",
    )
    ingest.add_argument(
        "--start",
        required=True,
        type=_parse_timestamp,
        help=(
            "Start of the ingest window. Either a bare date "
            "('2026-05-04', interpreted as midnight UTC) or a full ISO "
            "timestamp with an offset ('2026-05-04T12:00:00+00:00'). "
            "Naked datetimes without a timezone are rejected — they are "
            "ambiguous."
        ),
    )
    ingest.add_argument(
        "--end",
        required=True,
        type=_parse_timestamp,
        help="End of the ingest window (exclusive). Same format rules as --start.",
    )

    return parser


def _parse_timestamp(raw: str) -> datetime:
    """Argparse type converter. Accepts ISO date or full UTC timestamp."""
    if "T" not in raw:
        # Bare date: interpret as midnight UTC. ``date.fromisoformat`` is
        # stricter than ``datetime.fromisoformat`` and only accepts the
        # YYYY-MM-DD form, which is exactly what we want for this branch.
        try:
            d = datetime.fromisoformat(raw).date()
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid date {raw!r}: {exc}") from exc
        return datetime(d.year, d.month, d.day, tzinfo=UTC)

    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid timestamp {raw!r}: {exc}") from exc

    if dt.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"timestamp {raw!r} has a time component but no timezone — "
            f"add a UTC offset (e.g. +00:00) or use a bare date"
        )
    return dt


def _run_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    use_case = build_ingest_entsoe_load(settings)
    zones = [BiddingZone(z) for z in args.zone]

    try:
        result = use_case.execute(zones=zones, start=args.start, end=args.end)
    except ApplicationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_result(result)
    return 0


def _print_result(result: IngestEntsoeLoadResult) -> None:
    print("Ingest complete:")
    print(f"  Zones processed:       {result.zones_processed}")
    print(f"  Observations fetched:  {result.observations_fetched}")
    print(f"  Observations inserted: {result.observations_inserted}")
    print(f"  Started at:            {result.started_at.isoformat()}")
    print(f"  Finished at:           {result.finished_at.isoformat()}")
    print(f"  Duration:              {result.duration_seconds:.3f} s")
