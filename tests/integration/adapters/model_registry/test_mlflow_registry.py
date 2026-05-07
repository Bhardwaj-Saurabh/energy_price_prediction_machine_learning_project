"""Integration test for MLflowModelRegistry — uses a tmp_path file:// store."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from energy_forecaster.adapters.model_registry.mlflow_registry import (
    MLflowModelRegistry,
)

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
