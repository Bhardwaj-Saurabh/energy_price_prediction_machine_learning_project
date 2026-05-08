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
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from energy_forecaster.domain.value_objects.model_version import ModelVersion

# MLflow run IDs are 32-character lowercase hex strings. We use this to
# disambiguate run-id-form ModelVersions from alias-form ones (where the
# part after ``@`` is a short human-readable string like ``champion``).
_RUN_ID_HEX_LEN: int = 32


def _looks_like_run_id(suffix: str) -> bool:
    return len(suffix) == _RUN_ID_HEX_LEN and all(c in "0123456789abcdef" for c in suffix)


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

    def _client(self) -> MlflowClient:
        return MlflowClient(tracking_uri=self._tracking_uri)

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

    def load(self, version: ModelVersion) -> Any:
        """Load a previously registered LightGBM model from MLflow.

        Accepts both run-id-form (``<name>@<run_id>``) and alias-form
        (``<name>@<alias>``, e.g. ``demand_forecaster@champion``)
        ModelVersions. The form is detected by length + character set
        of the part after ``@``.
        """
        mlflow.set_tracking_uri(self._tracking_uri)
        try:
            name, suffix = version.value.split("@", 1)
        except ValueError as exc:
            raise ValueError(
                f"ModelVersion {version.value!r} is not in "
                f"'<registered_name>@<run_id_or_alias>' form expected by "
                f"MLflowModelRegistry"
            ) from exc
        if _looks_like_run_id(suffix):
            return mlflow.lightgbm.load_model(f"runs:/{suffix}/model")
        return mlflow.lightgbm.load_model(f"models:/{name}@{suffix}")

    def get_alias(self, registered_name: str, alias: str) -> ModelVersion | None:
        """Return the version currently behind ``alias``, or None if unset.

        MLflow raises ``MlflowException`` when the registered model
        doesn't exist OR the alias isn't set. We treat both as "no
        such alias" — the training runner uses this to detect first-run
        promotion (no champion yet).
        """
        try:
            mv = self._client().get_model_version_by_alias(name=registered_name, alias=alias)
        except MlflowException:
            return None
        return ModelVersion(f"{registered_name}@{mv.run_id}")

    def get_metric(self, version: ModelVersion, metric_key: str) -> float | None:
        """Look up ``metric_key`` on the run associated with ``version``.

        Both run-id-form and alias-form versions are accepted; alias is
        resolved to a run via the MLflow Model Registry first.
        """
        name, suffix = version.value.split("@", 1)
        if _looks_like_run_id(suffix):
            run_id: str = suffix
        else:
            mv = self._client().get_model_version_by_alias(name=name, alias=suffix)
            # MLflow's ModelVersion.run_id is typed Optional in the
            # stubs; in practice every registered version we create is
            # backed by a run. The guard exists to satisfy the type
            # checker and surface the impossible case clearly.
            if mv.run_id is None:  # pragma: no cover
                raise ValueError(  # pragma: no cover
                    f"Alias {suffix!r} on {name!r} resolved to a model version with no run_id"
                )
            run_id = mv.run_id
        run = self._client().get_run(run_id)
        value = run.data.metrics.get(metric_key)
        return float(value) if value is not None else None

    def set_alias(self, registered_name: str, alias: str, version: ModelVersion) -> None:
        """Point ``alias`` at ``version`` (must be run-id form).

        MLflow's alias API takes the registered-model version *number*
        (an integer), not the run_id. We look it up by searching the
        registry for the matching run_id; the lookup is cheap.
        """
        name, suffix = version.value.split("@", 1)
        if not _looks_like_run_id(suffix):
            raise ValueError(
                f"set_alias requires a run-id-form ModelVersion, got alias-form {version.value!r}"
            )
        if name != registered_name:
            raise ValueError(
                f"ModelVersion is registered under {name!r}, but set_alias was "
                f"called for {registered_name!r}"
            )
        client = self._client()
        # Each registered version has its own integer ``version`` field
        # in MLflow; aliases attach to those numbers, not to run_ids.
        # Find the version whose run_id matches.
        candidates = client.search_model_versions(f"name='{registered_name}'")
        matching = [v for v in candidates if v.run_id == suffix]
        if not matching:
            raise ValueError(
                f"No registered version found for run_id {suffix!r} under name {registered_name!r}"
            )
        client.set_registered_model_alias(
            name=registered_name,
            alias=alias,
            version=matching[0].version,
        )
