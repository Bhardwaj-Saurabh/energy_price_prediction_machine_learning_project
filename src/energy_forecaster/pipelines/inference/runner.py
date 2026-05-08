"""Programmatic runner for the inference pipeline.

Two flavours, sharing the registry/repo wiring:

  * :func:`run_inference` — *backtest* mode. Predicts on the most-recent
    rows of the existing feature matrix. ``as_of_time`` is set to
    ``delivery_time - 24h`` per forecast (day-ahead pseudo-history).
  * :func:`run_forward_inference` — *forward* mode. Predicts the next N
    hours from "now". Builds feature rows on the fly from observations
    + a weather forecast; ``as_of_time`` is the same fixed clock-now
    for every forecast. Uses recursive ``load_lag_1h`` filling — see
    :mod:`energy_forecaster.pipelines.inference.forward`.

Both never touch infrastructure inside their pure helpers; only this
runner does port work — same architectural rule training follows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import ParquetDataset

from energy_forecaster.application.ports.clock import Clock
from energy_forecaster.application.ports.load_forecast_repository import (
    LoadForecastRepository,
)
from energy_forecaster.application.ports.load_observation_repository import (
    LoadObservationRepository,
)
from energy_forecaster.application.ports.model_registry import ModelRegistry
from energy_forecaster.application.ports.weather_client import WeatherClient
from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.inference.forward import (
    build_partial_features,
    predict_recursively,
)
from energy_forecaster.pipelines.inference.pipeline import (
    create_inference_pipeline,
)

# Same feature column tuple as backtest mode and the training pipeline.
# Kept local to mirror the existing convention (each pipeline owns its
# own feature list so the three can diverge intentionally).
_FORWARD_FEATURE_COLUMNS: tuple[str, ...] = (
    "temp_c",
    "wind_10m_ms",
    "wind_100m_ms",
    "ghi_wm2",
    "cloud_cover_pct",
    "precip_mm",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "load_lag_1h",
    "load_lag_24h",
    "load_lag_168h",
    "zone_cat",
)

_KEDRO_LOGGER_NAMES: tuple[str, ...] = (
    "kedro",
    "kedro.framework",
    "kedro.io",
    "kedro.pipeline",
    "kedro.runner",
)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Summary returned by :func:`run_inference`."""

    model_version: ModelVersion
    forecasts_produced: int
    forecasts_inserted: int
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


def _silence_kedro_loggers() -> None:
    for name in _KEDRO_LOGGER_NAMES:
        logging.getLogger(name).setLevel(logging.WARNING)


def run_inference(
    *,
    features_path: Path,
    registry: ModelRegistry,
    repo: LoadForecastRepository,
    clock: Clock,
    model_version: ModelVersion,
    hours: int = 24,
) -> InferenceResult:
    """Run the inference pipeline and persist the resulting forecasts.

    Backtest mode: predicts on the most recent ``hours`` rows per zone
    in the feature matrix. Forward inference (real day-ahead) lands in
    a follow-up that adds a forecast-weather adapter; this version
    stays useful for backtest analysis and pipeline validation.
    """
    _silence_kedro_loggers()
    started_at = clock.now()

    model = registry.load(model_version)

    catalog = DataCatalog(
        {
            "features": ParquetDataset(filepath=str(features_path)),
            "model": MemoryDataset(model),
            "model_version": MemoryDataset(model_version),
            "hours": MemoryDataset(hours),
            "prediction_inputs": MemoryDataset(),
            "prediction_data": MemoryDataset(),
            "forecasts": MemoryDataset(),
        }
    )
    SequentialRunner().run(create_inference_pipeline(), catalog)

    forecasts: list[LoadForecast] = catalog.load("forecasts")
    inserted = repo.add_many(forecasts)
    finished_at = clock.now()

    return InferenceResult(
        model_version=model_version,
        forecasts_produced=len(forecasts),
        forecasts_inserted=inserted,
        started_at=started_at,
        finished_at=finished_at,
    )


def _floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def run_forward_inference(
    *,
    registry: ModelRegistry,
    forecast_repo: LoadForecastRepository,
    observation_repo: LoadObservationRepository,
    weather: WeatherClient,
    clock: Clock,
    model_version: ModelVersion,
    zones: Iterable[BiddingZone] = tuple(BiddingZone),
    hours: int = 24,
) -> InferenceResult:
    """Run forward inference and persist the resulting day-ahead forecasts.

    Picks the next ``hours`` hourly delivery slots after ``clock.now()``
    (floored to the hour) and produces one forecast per slot per zone.
    All forecasts share the same ``as_of_time = clock.now()`` — the
    moment the run was kicked off — so a downstream consumer can group
    them as "the 12:00 forecast for tomorrow's day-ahead market".

    Per-zone flow (the heart of the function):
      1. Read observations covering the lag window (``[delivery[0] - 168h,
         delivery[0])``) so ``load_lag_24h`` and ``load_lag_168h`` can be
         filled deterministically.
      2. Fetch the weather forecast covering the delivery window.
      3. Build the partial feature DataFrame.
      4. Look up the observed load at ``delivery[0] - 1h`` as the seed
         for recursive ``load_lag_1h`` filling.
      5. Predict iteratively, feeding each prediction back as the next
         row's ``load_lag_1h``.
      6. Convert predictions to :class:`LoadForecast` entities.

    Step (5) is where prediction errors compound through the horizon —
    the recursive prediction pattern. Document, accept, monitor.
    """
    started_at = clock.now()
    as_of_time = _floor_to_hour(started_at)

    # Delivery hours: hour after now, hour after that, ..., for ``hours`` slots.
    delivery_times = [as_of_time + timedelta(hours=h) for h in range(1, hours + 1)]
    lookback_start = delivery_times[0] - timedelta(hours=168)

    model = registry.load(model_version)

    forecasts: list[LoadForecast] = []
    for zone in zones:
        observations = observation_repo.find_by_zone(
            zone, since=lookback_start, until=delivery_times[0]
        )
        # The weather window is closed on the right of the last delivery
        # time, so we add an extra hour to the upper bound — the port's
        # half-open contract excludes ``end``.
        weather_forecast = list(
            weather.fetch_forecast(
                zone=zone,
                start=delivery_times[0],
                end=delivery_times[-1] + timedelta(hours=1),
            )
        )

        partial = build_partial_features(
            zone=zone,
            delivery_times=delivery_times,
            observations=observations,
            weather_forecast=weather_forecast,
        )

        # Seed for recursive lag_1h: observed load at hour-before-first-delivery.
        seed_time = delivery_times[0] - timedelta(hours=1)
        seed_load = next(
            (o.load.value for o in observations if o.timestamp_utc == seed_time),
            None,
        )
        if seed_load is None:
            raise ValueError(
                f"Missing observation for {zone.value} at {seed_time} "
                "(needed to seed recursive load_lag_1h)"
            )

        predictions = predict_recursively(
            model=model,
            partial_features=partial,
            initial_lag_1h=seed_load,
            feature_columns=list(_FORWARD_FEATURE_COLUMNS),
        )

        for delivery_time, predicted in zip(delivery_times, predictions, strict=True):
            forecasts.append(
                LoadForecast(
                    zone=zone,
                    as_of_time=as_of_time,
                    delivery_time=delivery_time,
                    predicted_load=EnergyMW(predicted),
                    model_version=model_version,
                )
            )

    inserted = forecast_repo.add_many(forecasts)
    finished_at = clock.now()

    return InferenceResult(
        model_version=model_version,
        forecasts_produced=len(forecasts),
        forecasts_inserted=inserted,
        started_at=started_at,
        finished_at=finished_at,
    )
