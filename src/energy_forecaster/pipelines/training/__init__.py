"""Training pipeline — Kedro DAG that fits + evaluates + registers a model.

Public surface:
  * :func:`create_training_pipeline` — the Kedro Pipeline factory.
  * :func:`run_training` — runner that wires the catalog, runs the DAG,
    and registers the resulting model via :class:`ModelRegistry`.
  * :class:`TrainingResult` — summary returned by ``run_training``.
"""

from energy_forecaster.pipelines.training.pipeline import create_training_pipeline
from energy_forecaster.pipelines.training.runner import (
    TrainingResult,
    run_training,
)

__all__ = ["TrainingResult", "create_training_pipeline", "run_training"]
