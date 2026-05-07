"""Programmatic runner for the training pipeline.

Bridges the pure Kedro pipeline (no port interaction) to the
:class:`ModelRegistry` port: the pipeline produces a
``training_artifacts`` dict; the runner pulls that out of the catalog
and calls ``registry.register(...)`` to persist + version the model.
The runner returns a :class:`TrainingResult` for the CLI to render.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import ParquetDataset

from energy_forecaster.application.ports.model_registry import ModelRegistry
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.training.pipeline import create_training_pipeline

_DEFAULT_REGISTERED_NAME: str = "demand_forecaster"

_KEDRO_LOGGER_NAMES: tuple[str, ...] = (
    "kedro",
    "kedro.framework",
    "kedro.io",
    "kedro.pipeline",
    "kedro.runner",
)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Summary returned by :func:`run_training`.

    The model object itself does not appear here — it is now in the
    registry, and the version string is the canonical handle. Callers
    that want the artifact load it through the registry by version.
    """

    model_version: ModelVersion
    train_size: int
    test_size: int
    test_mape: float
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


def _silence_kedro_loggers() -> None:
    for name in _KEDRO_LOGGER_NAMES:
        logging.getLogger(name).setLevel(logging.WARNING)


def run_training(
    *,
    features_path: Path,
    registry: ModelRegistry,
    registered_name: str = _DEFAULT_REGISTERED_NAME,
) -> TrainingResult:
    """Run the training pipeline and register the resulting model.

    The Kedro pipeline is pure (reads features, fits, evaluates,
    bundles artifacts). The registry interaction happens here, after
    the pipeline completes — that keeps every node reorderable,
    cacheable, and parallelisable without depending on infrastructure.
    """
    _silence_kedro_loggers()
    started_at = datetime.now(UTC)

    catalog = DataCatalog(
        {
            "features": ParquetDataset(filepath=str(features_path)),
            "training_data": MemoryDataset(),
            "trained_model": MemoryDataset(),
            "metrics": MemoryDataset(),
            "training_artifacts": MemoryDataset(),
        }
    )
    SequentialRunner().run(create_training_pipeline(), catalog)

    artifacts: dict[str, Any] = catalog.load("training_artifacts")
    metrics: dict[str, float] = artifacts["metrics"]

    version = registry.register(
        model=artifacts["model"],
        registered_name=registered_name,
        params=artifacts["params"],
        metrics=metrics,
    )

    finished_at = datetime.now(UTC)
    started_monotonic = time.monotonic()
    _ = started_monotonic  # silence unused linter; we read wall-clock above

    return TrainingResult(
        model_version=version,
        train_size=int(metrics["train_size"]),
        test_size=int(metrics["test_size"]),
        test_mape=float(metrics["mape"]),
        started_at=started_at,
        finished_at=finished_at,
    )
