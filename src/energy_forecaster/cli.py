"""Energy Forecaster command-line interface.

Single entrypoint exposed by the ``[project.scripts]`` table in
``pyproject.toml``. Subcommands map 1-to-1 onto use cases — the CLI is a
*framework* in the clean-architecture sense, just like the future FastAPI
app: it parses arguments, configures structured logging, builds a
correlation-id-bound logger, calls the composition root to assemble a
wired use case, executes it, and renders the result. No business logic
lives here.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from energy_forecaster.adapters.logger.structlog_logger import (
    StructlogLogger,
    configure_structlog,
)
from energy_forecaster.application.errors import ApplicationError
from energy_forecaster.application.ports.logger import Logger
from energy_forecaster.application.use_cases.ingest_entsoe_load import (
    IngestEntsoeLoadResult,
)
from energy_forecaster.application.use_cases.ingest_weather import (
    IngestWeatherResult,
)
from energy_forecaster.composition import (
    build_ingest_entsoe_load,
    build_ingest_weather,
    build_run_feature_engineering,
    build_run_inference,
    build_run_training,
)
from energy_forecaster.config.settings import Settings, get_settings
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.inference.runner import InferenceResult
from energy_forecaster.pipelines.training.runner import TrainingResult


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for ``energy-forecaster ...``. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_structlog(log_level=settings.log_level, environment=settings.environment)

    # One correlation_id per CLI invocation. Every log line emitted while
    # this process runs carries it, which is the cheapest first step
    # toward distributed tracing — once we add HTTP serving, the same
    # field becomes the request ID.
    logger = StructlogLogger().bind(correlation_id=str(uuid.uuid4()))

    if args.command == "ingest":
        return _run_ingest(args, settings=settings, logger=logger)
    if args.command == "weather":
        return _run_weather(args, settings=settings, logger=logger)
    if args.command == "features":
        return _run_features(args, settings=settings, logger=logger)
    if args.command == "train":
        return _run_train(args, settings=settings, logger=logger)
    if args.command == "predict":
        return _run_predict(args, settings=settings, logger=logger)

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
    _add_zone_window_args(ingest)

    weather = subparsers.add_parser(
        "weather",
        help="Fetch hourly weather readings for one or more zones over a window.",
        description=(
            "Run the IngestWeather use case against the configured weather "
            "adapter (synthetic by default; set EF_WEATHER_SOURCE=open_meteo "
            "to hit the real Open-Meteo API)."
        ),
    )
    _add_zone_window_args(weather)

    features = subparsers.add_parser(
        "features",
        help="Build the feature matrix from previously ingested JSONL.",
        description=(
            "Run the feature engineering Kedro pipeline. Reads "
            "load_observations/ and weather_readings/ from the configured "
            "data root, joins on (zone, timestamp), adds calendar + lag "
            "features, validates against FeatureMatrixSchema, and writes "
            "Parquet."
        ),
    )
    features.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Destination Parquet file. Defaults to <EF_LOCAL_DATA_ROOT>/features.parquet."),
    )

    train = subparsers.add_parser(
        "train",
        help="Train the demand-forecasting model on the feature matrix.",
        description=(
            "Run the training Kedro pipeline against the configured feature "
            "matrix, then register the resulting model with the configured "
            "ModelRegistry (MLflow). Outputs a summary of the model version "
            "and test-set MAPE."
        ),
    )
    train.add_argument(
        "--features",
        type=Path,
        default=None,
        help=(
            "Path to the feature matrix Parquet. Defaults to <EF_LOCAL_DATA_ROOT>/features.parquet."
        ),
    )

    predict = subparsers.add_parser(
        "predict",
        help="Run inference: load a registered model and emit LoadForecasts.",
        description=(
            "Backtest-mode inference. Loads the specified model version "
            "from the configured ModelRegistry (MLflow), reads the most "
            "recent N hours from the feature matrix, predicts loads, and "
            "persists LoadForecast entities to JSONL."
        ),
    )
    predict.add_argument(
        "--model",
        type=str,
        default="demand_forecaster@champion",
        help=(
            "Model version to load. Accepts both run-id form "
            "('demand_forecaster@<run_id>') and alias form "
            "('demand_forecaster@champion'). Defaults to "
            "'demand_forecaster@champion' — the alias the training "
            "runner sets when promoting a winning challenger."
        ),
    )
    predict.add_argument(
        "--features",
        type=Path,
        default=None,
        help=(
            "Path to the feature matrix Parquet. Defaults to <EF_LOCAL_DATA_ROOT>/features.parquet."
        ),
    )
    predict.add_argument(
        "--hours",
        type=int,
        default=24,
        help="How many of the most recent hours to predict per zone (default 24).",
    )

    return parser


def _add_zone_window_args(sub: argparse.ArgumentParser) -> None:
    """Shared --zone/--start/--end flags for every ingest-style command."""
    sub.add_argument(
        "--zone",
        action="append",
        required=True,
        choices=[z.value for z in BiddingZone],
        help="Bidding zone (repeatable). Example: --zone DE_LU --zone FR",
    )
    sub.add_argument(
        "--start",
        required=True,
        type=_parse_timestamp,
        help=(
            "Start of the window. Either a bare date "
            "('2026-05-04', interpreted as midnight UTC) or a full ISO "
            "timestamp with an offset ('2026-05-04T12:00:00+00:00'). "
            "Naked datetimes without a timezone are rejected — they are "
            "ambiguous."
        ),
    )
    sub.add_argument(
        "--end",
        required=True,
        type=_parse_timestamp,
        help="End of the window (exclusive). Same format rules as --start.",
    )


def _parse_timestamp(raw: str) -> datetime:
    """Argparse type converter. Accepts ISO date or full UTC timestamp."""
    if "T" not in raw:
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


def _run_ingest(args: argparse.Namespace, *, settings: Settings, logger: Logger) -> int:
    use_case = build_ingest_entsoe_load(settings, logger=logger)
    zones = [BiddingZone(z) for z in args.zone]

    try:
        result = use_case.execute(zones=zones, start=args.start, end=args.end)
    except ApplicationError as exc:
        logger.error("ingest.failed", error=str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_load_result(result)
    return 0


def _run_weather(args: argparse.Namespace, *, settings: Settings, logger: Logger) -> int:
    use_case = build_ingest_weather(settings, logger=logger)
    zones = [BiddingZone(z) for z in args.zone]

    try:
        result = use_case.execute(zones=zones, start=args.start, end=args.end)
    except ApplicationError as exc:
        logger.error("weather.failed", error=str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_weather_result(result)
    return 0


def _run_features(args: argparse.Namespace, *, settings: Settings, logger: Logger) -> int:
    runner = build_run_feature_engineering(settings)
    log = logger.bind(operation="feature_engineering")
    log.info("features.start", output=str(args.output) if args.output else "default")

    started = time.monotonic()
    try:
        output_path = runner(args.output)
    except Exception as exc:
        # Deliberate framework-layer boundary catch: the feature pipeline
        # may raise pandera, kedro, pyarrow, or filesystem errors. They
        # are logged with their type and surfaced as exit code 1; we do
        # not let them crash the CLI with a stack trace.
        log.error("features.failed", error=str(exc), error_type=type(exc).__name__)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    duration = time.monotonic() - started

    log.info(
        "features.done",
        output_path=str(output_path),
        duration_seconds=round(duration, 3),
    )
    _print_features_result(output_path, duration)
    return 0


def _run_train(args: argparse.Namespace, *, settings: Settings, logger: Logger) -> int:
    runner = build_run_training(settings)
    log = logger.bind(operation="training")
    log.info(
        "training.start",
        features=str(args.features) if args.features else "default",
    )

    try:
        result = runner(args.features)
    except Exception as exc:
        # Same boundary-catch policy as the features handler: any
        # MLflow / LightGBM / pandera / filesystem error becomes a
        # clean exit-code-1 outcome with a logged error type.
        log.error("training.failed", error=str(exc), error_type=type(exc).__name__)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    log.info(
        "training.done",
        model_version=result.model_version.value,
        test_mape=result.test_mape,
        train_size=result.train_size,
        test_size=result.test_size,
        promoted=result.promoted,
        previous_champion=(
            result.previous_champion.value if result.previous_champion is not None else None
        ),
        duration_seconds=round(result.duration_seconds, 3),
    )
    _print_training_result(result)
    return 0


def _print_load_result(result: IngestEntsoeLoadResult) -> None:
    print("Ingest complete:")
    print(f"  Zones processed:       {result.zones_processed}")
    print(f"  Observations fetched:  {result.observations_fetched}")
    print(f"  Observations inserted: {result.observations_inserted}")
    print(f"  Started at:            {result.started_at.isoformat()}")
    print(f"  Finished at:           {result.finished_at.isoformat()}")
    print(f"  Duration:              {result.duration_seconds:.3f} s")


def _print_weather_result(result: IngestWeatherResult) -> None:
    print("Weather ingest complete:")
    print(f"  Zones processed:    {result.zones_processed}")
    print(f"  Readings fetched:   {result.readings_fetched}")
    print(f"  Readings inserted:  {result.readings_inserted}")
    print(f"  Started at:         {result.started_at.isoformat()}")
    print(f"  Finished at:        {result.finished_at.isoformat()}")
    print(f"  Duration:           {result.duration_seconds:.3f} s")


def _print_features_result(output_path: Path, duration_seconds: float) -> None:
    # Read the freshly written Parquet for a row + column count. pandas is
    # already a project dep; the cost is negligible for sane sizes.
    import pandas as pd

    df = pd.read_parquet(output_path)
    print("Feature engineering complete:")
    print(f"  Output:    {output_path}")
    print(f"  Rows:      {len(df)}")
    print(f"  Columns:   {len(df.columns)}")
    print(f"  Duration:  {duration_seconds:.3f} s")


def _print_training_result(result: TrainingResult) -> None:
    print("Training complete:")
    print(f"  Model version: {result.model_version.value}")
    print(f"  Train rows:    {result.train_size}")
    print(f"  Test rows:     {result.test_size}")
    print(f"  Test MAPE:     {result.test_mape:.4f}")
    if result.promoted:
        if result.previous_champion is None:
            print("  Promotion:     promoted to @champion (inaugural)")
        else:
            print(f"  Promotion:     promoted to @champion (was {result.previous_champion.value})")
    else:
        prev = result.previous_champion.value if result.previous_champion is not None else "<none>"
        print(f"  Promotion:     no — incumbent {prev} kept @champion")
    print(f"  Started at:    {result.started_at.isoformat()}")
    print(f"  Finished at:   {result.finished_at.isoformat()}")
    print(f"  Duration:      {result.duration_seconds:.3f} s")


def _run_predict(args: argparse.Namespace, *, settings: Settings, logger: Logger) -> int:
    runner = build_run_inference(settings)
    log = logger.bind(operation="inference")
    model_version = ModelVersion(args.model)
    log.info(
        "predict.start",
        model_version=model_version.value,
        hours=args.hours,
        features=str(args.features) if args.features else "default",
    )

    try:
        result = runner(model_version, args.features, args.hours)
    except Exception as exc:
        log.error("predict.failed", error=str(exc), error_type=type(exc).__name__)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    log.info(
        "predict.done",
        model_version=result.model_version.value,
        forecasts_produced=result.forecasts_produced,
        forecasts_inserted=result.forecasts_inserted,
        duration_seconds=round(result.duration_seconds, 3),
    )
    _print_inference_result(result)
    return 0


def _print_inference_result(result: InferenceResult) -> None:
    print("Inference complete:")
    print(f"  Model version:        {result.model_version.value}")
    print(f"  Forecasts produced:   {result.forecasts_produced}")
    print(f"  Forecasts inserted:   {result.forecasts_inserted}")
    print(f"  Started at:           {result.started_at.isoformat()}")
    print(f"  Finished at:          {result.finished_at.isoformat()}")
    print(f"  Duration:             {result.duration_seconds:.3f} s")
