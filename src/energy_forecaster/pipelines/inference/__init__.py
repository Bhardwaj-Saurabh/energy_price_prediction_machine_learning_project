"""Inference pipeline — Kedro DAG that turns features + model into forecasts.

Public surface:
  * :func:`create_inference_pipeline` — Pipeline factory.
  * :func:`run_inference` — runner that loads the model, runs the DAG,
    and persists forecasts via :class:`LoadForecastRepository`.
  * :class:`InferenceResult` — summary returned by ``run_inference``.
"""

from energy_forecaster.pipelines.inference.pipeline import (
    create_inference_pipeline,
)
from energy_forecaster.pipelines.inference.runner import (
    InferenceResult,
    run_inference,
)

__all__ = ["InferenceResult", "create_inference_pipeline", "run_inference"]
