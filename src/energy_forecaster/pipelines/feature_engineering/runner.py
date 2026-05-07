"""Programmatic runner for the feature engineering pipeline.

Wraps Kedro's :class:`SequentialRunner` with a function that takes
filesystem paths, builds a :class:`DataCatalog`, runs the pipeline, and
returns the path the feature matrix was written to.

Logging: Kedro 1.x configures rich console logging on import, which
floods stdout in tests and CLI runs. We tame it to WARNING in this
module so structured-logging output stays the source of truth. When we
later integrate Kedro's logging into structlog, this is the seam.
"""

from __future__ import annotations

import logging
from pathlib import Path

from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import ParquetDataset

from energy_forecaster.pipelines.feature_engineering.pipeline import (
    create_feature_engineering_pipeline,
)

_KEDRO_LOGGER_NAMES: tuple[str, ...] = (
    "kedro",
    "kedro.framework",
    "kedro.io",
    "kedro.pipeline",
    "kedro.runner",
)


def _silence_kedro_loggers() -> None:
    """Drop Kedro's chatty INFO logs to WARNING for the rest of the process."""
    for name in _KEDRO_LOGGER_NAMES:
        logging.getLogger(name).setLevel(logging.WARNING)


def run_feature_engineering(
    *,
    load_directory: Path,
    weather_directory: Path,
    output_path: Path,
) -> Path:
    """Run the pipeline end-to-end.

    Reads JSONL from ``load_directory`` and ``weather_directory``, builds
    the feature matrix, writes it to ``output_path`` as Parquet, and
    returns ``output_path`` for caller convenience.
    """
    _silence_kedro_loggers()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = DataCatalog(
        {
            "load_directory": MemoryDataset(load_directory),
            "weather_directory": MemoryDataset(weather_directory),
            "feature_matrix": ParquetDataset(filepath=str(output_path)),
        }
    )
    SequentialRunner().run(create_feature_engineering_pipeline(), catalog)
    return output_path
