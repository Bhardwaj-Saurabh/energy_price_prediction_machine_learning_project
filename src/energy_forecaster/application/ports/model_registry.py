"""ModelRegistry port — persistence + versioning for trained model artifacts."""

from typing import Any, Protocol

from energy_forecaster.domain.value_objects.model_version import ModelVersion


class ModelRegistry(Protocol):
    """Logs a training run and registers the trained model.

    The application layer never picks pickle files or filesystem paths
    out of the air — every model is reached through this port. The
    concrete adapter (MLflow today) handles serialisation, versioning,
    aliasing (``@champion``), and the audit trail of params + metrics.

    The ``model`` argument is intentionally typed as ``Any`` here: the
    domain has no opinion on the model object's type (LightGBM Booster,
    sklearn estimator, …). The concrete adapter inspects the type and
    routes to the appropriate MLflow flavour.
    """

    def register(
        self,
        *,
        model: Any,
        registered_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
    ) -> ModelVersion:
        """Log the training run and register the model.

        Returns a :class:`ModelVersion` with an opaque identifier the
        registry uses to retrieve the artifact later. The format is the
        adapter's choice; downstream code treats the value as opaque.
        """
        ...
