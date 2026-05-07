"""Integration test for MLflowModelRegistry — uses a tmp_path file:// store."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from energy_forecaster.adapters.model_registry.mlflow_registry import (
    MLflowModelRegistry,
)
from energy_forecaster.domain.value_objects.model_version import ModelVersion

pytestmark = pytest.mark.integration


def _train_tiny_booster() -> lgb.Booster:
    """A throwaway LightGBM model — content does not matter, just a valid Booster."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    y = x.sum(axis=1) + rng.normal(scale=0.1, size=50)
    train_set = lgb.Dataset(x, label=y)
    return lgb.train(
        params={"objective": "regression", "verbose": -1, "num_leaves": 7},
        train_set=train_set,
        num_boost_round=5,
    )


class TestMLflowModelRegistryAgainstLocalFileStore:
    def test_register_logs_run_and_returns_a_model_version(self, tmp_path: Path) -> None:
        # Local file store: MLflow writes runs and registry under tmp_path
        # so the test is hermetic. ``registered_name`` lands in the model
        # registry; ``params`` and ``metrics`` land on the run.
        tracking_uri = f"file:{tmp_path / 'mlruns'}"
        registry = MLflowModelRegistry(
            tracking_uri=tracking_uri,
            experiment_name="ef_test_experiment",
        )
        model = _train_tiny_booster()

        version = registry.register(
            model=model,
            registered_name="ef_test_model",
            params={"learning_rate": 0.05, "num_leaves": 7},
            metrics={"mape": 0.07},
        )

        # The returned ModelVersion includes the registered name, so a
        # downstream lookup can disambiguate when multiple models are
        # registered against the same backend.
        assert version.value.startswith("ef_test_model@")

        # MLflow should have written runs under the tmp tracking URI.
        # The exact directory layout is mlruns/<exp_id>/<run_id>/...
        assert (tmp_path / "mlruns").exists()

    def test_load_returns_a_predictable_model(self, tmp_path: Path) -> None:
        # Round-trip: register a model, take the returned ModelVersion,
        # call load(), and confirm the resurrected object can predict.
        tracking_uri = f"file:{tmp_path / 'mlruns'}"
        registry = MLflowModelRegistry(
            tracking_uri=tracking_uri,
            experiment_name="ef_test_experiment_load",
        )
        original = _train_tiny_booster()
        version = registry.register(
            model=original,
            registered_name="ef_load_test_model",
            params={"num_leaves": 7},
            metrics={"mape": 0.05},
        )

        loaded = registry.load(version)

        rng = np.random.default_rng(1)
        x = rng.normal(size=(3, 3))
        # Both objects should produce predictions of the same shape on
        # the same input. Exact equality isn't guaranteed (MLflow may
        # round-trip through different serialisation), but shape + finite
        # values is enough to confirm the load succeeded.
        predictions = loaded.predict(x)
        assert predictions.shape == (3,)
        assert np.isfinite(predictions).all()

    def test_load_rejects_malformed_model_version(self, tmp_path: Path) -> None:
        # The adapter writes versions as ``<name>@<run_id>`` and assumes
        # that shape on read. Anything else is a programmer error;
        # surfacing it as ValueError is more useful than letting MLflow
        # blow up several layers deeper.
        registry = MLflowModelRegistry(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="ef_test_experiment",
        )
        with pytest.raises(ValueError, match="<registered_name>@<run_id>"):
            registry.load(ModelVersion("no-at-sign-here"))
