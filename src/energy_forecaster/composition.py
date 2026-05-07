"""Composition root — the *only* place that wires concrete adapters to ports.

Every other module sees only Protocol-typed interfaces and receives its
dependencies through its constructor. This module reads :class:`Settings`,
chooses concrete adapters per environment, and returns ready-to-call use
case instances to the framework layer (CLI, FastAPI app, Prefect flow,
…). Anything in the codebase that imports a concrete adapter *and* a use
case must live here — and only here.

Branching policy:
  * **ENTSO-E.** ``entsoe_api_key`` unset → InMemoryEntsoeClient
    (synthetic). Set → EntsoePyClient (real, HTTP). The key's presence
    is the discriminator because no key means no real call is possible.
  * **Weather.** ``weather_source == "synthetic"`` → InMemoryWeatherClient.
    ``weather_source == "open_meteo"`` → OpenMeteoClient. Open-Meteo is
    keyless so the discriminator is an explicit setting rather than
    credential presence.

The build functions take an injected ``logger`` so the framework layer
(CLI, FastAPI) can establish a request-scoped bound logger — typically
with a ``correlation_id`` — and have it flow into every use case.
"""

from collections.abc import Callable
from pathlib import Path

from energy_forecaster.adapters.clock.system_clock import SystemClock
from energy_forecaster.adapters.entsoe_client.entsoe_py import EntsoePyClient
from energy_forecaster.adapters.entsoe_client.in_memory import InMemoryEntsoeClient
from energy_forecaster.adapters.load_forecast_repo.local_fs import (
    LocalFsLoadForecastRepository,
)
from energy_forecaster.adapters.load_observation_repo.local_fs import (
    LocalFsLoadObservationRepository,
)
from energy_forecaster.adapters.model_registry.mlflow_registry import (
    MLflowModelRegistry,
)
from energy_forecaster.adapters.weather_client.in_memory import InMemoryWeatherClient
from energy_forecaster.adapters.weather_client.open_meteo import OpenMeteoClient
from energy_forecaster.adapters.weather_reading_repo.local_fs import (
    LocalFsWeatherReadingRepository,
)
from energy_forecaster.application.ports.entsoe_client import EntsoeClient
from energy_forecaster.application.ports.logger import Logger
from energy_forecaster.application.ports.weather_client import WeatherClient
from energy_forecaster.application.use_cases.ingest_entsoe_load import (
    IngestEntsoeLoad,
)
from energy_forecaster.application.use_cases.ingest_weather import IngestWeather
from energy_forecaster.config.settings import Settings
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.feature_engineering.runner import (
    run_feature_engineering,
)
from energy_forecaster.pipelines.inference.runner import (
    InferenceResult,
    run_inference,
)
from energy_forecaster.pipelines.training.runner import (
    TrainingResult,
    run_training,
)


def build_ingest_entsoe_load(settings: Settings, *, logger: Logger) -> IngestEntsoeLoad:
    """Wire :class:`IngestEntsoeLoad` for the given environment."""
    entsoe: EntsoeClient
    if settings.entsoe_api_key is None:
        entsoe = InMemoryEntsoeClient()
    else:
        entsoe = EntsoePyClient(api_key=settings.entsoe_api_key.get_secret_value())

    return IngestEntsoeLoad(
        entsoe=entsoe,
        repo=LocalFsLoadObservationRepository(root=settings.local_data_root),
        clock=SystemClock(),
        logger=logger,
    )


def build_ingest_weather(settings: Settings, *, logger: Logger) -> IngestWeather:
    """Wire :class:`IngestWeather` for the given environment."""
    weather: WeatherClient
    if settings.weather_source == "synthetic":
        weather = InMemoryWeatherClient()
    else:
        weather = OpenMeteoClient()

    return IngestWeather(
        weather=weather,
        repo=LocalFsWeatherReadingRepository(root=settings.local_data_root),
        clock=SystemClock(),
        logger=logger,
    )


def build_run_feature_engineering(
    settings: Settings,
) -> Callable[[Path | None], Path]:
    """Return a partially-applied feature engineering runner.

    The returned closure captures the on-disk paths derived from
    ``settings.local_data_root`` (load JSONL dir, weather JSONL dir, and
    a default Parquet output path). Callers can override the output path
    per invocation; everything else is fixed at composition time.

    The feature engineering pipeline has no ports/adapters to wire — it
    is a self-contained Kedro DAG — but going through composition keeps
    every CLI command's path uniform: settings → wired callable.
    """
    load_directory = settings.local_data_root / "load_observations"
    weather_directory = settings.local_data_root / "weather_readings"
    default_output = settings.local_data_root / "features.parquet"

    def _run(output_path: Path | None = None) -> Path:
        return run_feature_engineering(
            load_directory=load_directory,
            weather_directory=weather_directory,
            output_path=output_path or default_output,
        )

    return _run


def build_run_training(settings: Settings) -> Callable[[Path | None], TrainingResult]:
    """Return a partially-applied training runner.

    The closure captures the default features-input path (under
    ``local_data_root``) and constructs an :class:`MLflowModelRegistry`
    pointed at ``settings.mlflow_tracking_uri``. The MLflow adapter is
    inert until ``register`` is called, so this is safe at composition.
    """
    default_features = settings.local_data_root / "features.parquet"
    registry = MLflowModelRegistry(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name="energy_forecaster",
    )

    def _run(features_path: Path | None = None) -> TrainingResult:
        return run_training(
            features_path=features_path or default_features,
            registry=registry,
        )

    return _run


def build_run_inference(
    settings: Settings,
) -> Callable[[ModelVersion, Path | None, int], InferenceResult]:
    """Return a partially-applied inference runner.

    Captures the registry, the LocalFs forecast repo, the system clock,
    and the default features path from settings. The caller picks the
    model version per invocation (and optionally the features path and
    number of hours to predict).
    """
    default_features = settings.local_data_root / "features.parquet"
    registry = MLflowModelRegistry(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name="energy_forecaster",
    )
    repo = LocalFsLoadForecastRepository(root=settings.local_data_root)
    clock = SystemClock()

    def _run(
        model_version: ModelVersion,
        features_path: Path | None = None,
        hours: int = 24,
    ) -> InferenceResult:
        return run_inference(
            features_path=features_path or default_features,
            registry=registry,
            repo=repo,
            clock=clock,
            model_version=model_version,
            hours=hours,
        )

    return _run
