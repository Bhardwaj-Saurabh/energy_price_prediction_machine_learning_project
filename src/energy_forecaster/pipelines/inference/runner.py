"""Programmatic runner for the inference pipeline.

Bridges the pure Kedro pipeline to two ports: :class:`ModelRegistry`
(read side, ``load(version)``) and :class:`LoadForecastRepository`
(write side, ``add_many(forecasts)``). The pipeline itself never
touches infrastructure — same architectural rule the training runner
follows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import ParquetDataset

from energy_forecaster.application.ports.clock import Clock
from energy_forecaster.application.ports.load_forecast_repository import (
    LoadForecastRepository,
)
from energy_forecaster.application.ports.model_registry import ModelRegistry
from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.inference.pipeline import (
    create_inference_pipeline,
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
