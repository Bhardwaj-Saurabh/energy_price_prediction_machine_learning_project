"""MLflowModelRegistry — production adapter implementing the ModelRegistry port.

MLflow handles model serialisation, versioning, and the run-level audit
trail (params, metrics, artifacts, environment). The application layer
talks to the :class:`ModelRegistry` Protocol; ``mlflow`` itself only
appears in this file and in tests that exercise the adapter directly.

Tracking URI handling:
  * ``file:./mlruns`` (the default) writes runs to a local directory.
  * ``http://...`` points at a running MLflow Tracking Server (the prod
    target — Azure-hosted, Postgres-backed, Blob artifact store).
The adapter is agnostic; MLflow's tracking client picks the backend.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.lightgbm

from energy_forecaster.domain.value_objects.model_version import ModelVersion


class MLflowModelRegistry:
    """Logs each training run and registers the model under a versioned name.

    The constructor stores configuration only; no network call runs
    until :meth:`register` is invoked. That makes instantiation safe in
    the composition root regardless of whether a tracking server is
    reachable.
    """

    def __init__(self, *, tracking_uri: str, experiment_name: str) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name

    def register(
        self,
        *,
        model: Any,
        registered_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
    ) -> ModelVersion:
        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            mlflow.lightgbm.log_model(
                lgb_model=model,
                name="model",
                registered_model_name=registered_name,
            )
            run_id = run.info.run_id
        # The opaque identifier is sufficient for the domain. The adapter
        # could later resolve it to MLflow's incrementing version number
        # via the Model Registry API if a consumer needs it.
        return ModelVersion(f"{registered_name}@{run_id}")
