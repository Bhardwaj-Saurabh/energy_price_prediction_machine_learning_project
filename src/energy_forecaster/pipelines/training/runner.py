"""Programmatic runner for the training pipeline.

Bridges the pure Kedro pipeline (no port interaction) to the
:class:`ModelRegistry` port: the pipeline produces a
``training_artifacts`` dict; the runner pulls that out of the catalog,
calls ``registry.register(...)`` to persist + version the model, then
runs the champion/challenger promotion rule.

The promotion rule itself lives in :mod:`energy_forecaster.domain.rules.promotion`
(``should_promote``). The runner orchestrates: query the registry for
the current ``@champion``, fetch its MAPE, call the rule, and update
the alias if the challenger wins. The decision is policy (domain code);
the comparison machinery is mechanism (infrastructure).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import ParquetDataset

from energy_forecaster.application.ports.model_registry import ModelRegistry
from energy_forecaster.domain.rules.promotion import should_promote
from energy_forecaster.domain.value_objects.mape import MAPE
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.training.pipeline import create_training_pipeline

_DEFAULT_REGISTERED_NAME: str = "demand_forecaster"
_CHAMPION_ALIAS: str = "champion"
_MAPE_METRIC_KEY: str = "mape"

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

    ``promoted`` and ``previous_champion`` document the champion/challenger
    decision: True if the new model became ``@champion``, False if the
    incumbent kept the alias. ``previous_champion`` is the version that
    held ``@champion`` *before* this run (None if no champion existed).
    """

    model_version: ModelVersion
    train_size: int
    test_size: int
    test_mape: float
    promoted: bool
    previous_champion: ModelVersion | None
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


def _silence_kedro_loggers() -> None:
    for name in _KEDRO_LOGGER_NAMES:
        logging.getLogger(name).setLevel(logging.WARNING)


def _evaluate_promotion(
    *,
    registry: ModelRegistry,
    registered_name: str,
    new_version: ModelVersion,
    new_mape: MAPE,
) -> tuple[bool, ModelVersion | None]:
    """Run the champion/challenger comparison and update the alias.

    Returns ``(promoted, previous_champion)``. ``previous_champion`` is
    the version that held ``@champion`` before this run (or None if no
    champion existed yet). The rule itself lives in domain code; this
    function is just the registry plumbing around it.
    """
    incumbent = registry.get_alias(registered_name, _CHAMPION_ALIAS)

    if incumbent is None:
        # First run for this registered model — install the new version
        # as the inaugural champion. No comparison needed.
        registry.set_alias(registered_name, _CHAMPION_ALIAS, new_version)
        return True, None

    incumbent_mape_value = registry.get_metric(incumbent, _MAPE_METRIC_KEY)
    if incumbent_mape_value is None:
        # Defensive: incumbent has no MAPE recorded (legacy model
        # registered before this metric was tracked). Treat the
        # challenger as automatically better — a comparable metric is
        # required to honour the should_promote contract.
        registry.set_alias(registered_name, _CHAMPION_ALIAS, new_version)
        return True, incumbent

    if should_promote(challenger=new_mape, champion=MAPE(incumbent_mape_value)):
        registry.set_alias(registered_name, _CHAMPION_ALIAS, new_version)
        return True, incumbent
    return False, incumbent


def run_training(
    *,
    features_path: Path,
    registry: ModelRegistry,
    registered_name: str = _DEFAULT_REGISTERED_NAME,
) -> TrainingResult:
    """Run the training pipeline, register the model, and consider promotion.

    The Kedro pipeline is pure (reads features, fits, evaluates,
    bundles artifacts). Both registry interactions — ``register()`` and
    the promotion comparison — happen here, after the pipeline
    completes, so every node stays reorderable and cacheable.
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

    promoted, previous_champion = _evaluate_promotion(
        registry=registry,
        registered_name=registered_name,
        new_version=version,
        new_mape=MAPE(float(metrics[_MAPE_METRIC_KEY])),
    )

    finished_at = datetime.now(UTC)

    return TrainingResult(
        model_version=version,
        train_size=int(metrics["train_size"]),
        test_size=int(metrics["test_size"]),
        test_mape=float(metrics[_MAPE_METRIC_KEY]),
        promoted=promoted,
        previous_champion=previous_champion,
        started_at=started_at,
        finished_at=finished_at,
    )
