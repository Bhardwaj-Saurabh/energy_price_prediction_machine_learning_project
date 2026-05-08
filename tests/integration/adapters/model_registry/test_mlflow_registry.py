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
        with pytest.raises(ValueError, match=r"run_id_or_alias"):
            registry.load(ModelVersion("no-at-sign-here"))


class TestAliasOperations:
    """Champion/challenger plumbing against the real MLflow Model Registry."""

    def test_get_alias_returns_none_when_unset(self, tmp_path: Path) -> None:
        # No model has been registered yet — every alias query should
        # return None rather than raising. The training runner relies on
        # this to detect first-run promotion.
        registry = MLflowModelRegistry(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="ef_alias_test",
        )
        assert registry.get_alias("does_not_exist", "champion") is None

    def test_register_then_set_and_get_alias(self, tmp_path: Path) -> None:
        registry = MLflowModelRegistry(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="ef_alias_test",
        )
        version = registry.register(
            model=_train_tiny_booster(),
            registered_name="ef_alias_model",
            params={"num_leaves": 7},
            metrics={"mape": 0.05},
        )

        # Initially no champion.
        assert registry.get_alias("ef_alias_model", "champion") is None

        # Set + get round trip.
        registry.set_alias("ef_alias_model", "champion", version)
        retrieved = registry.get_alias("ef_alias_model", "champion")
        assert retrieved == version

    def test_get_metric_returns_logged_value(self, tmp_path: Path) -> None:
        registry = MLflowModelRegistry(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="ef_metric_test",
        )
        version = registry.register(
            model=_train_tiny_booster(),
            registered_name="ef_metric_model",
            params={"num_leaves": 7},
            metrics={"mape": 0.07, "test_size": 100.0},
        )

        assert registry.get_metric(version, "mape") == 0.07
        assert registry.get_metric(version, "test_size") == 100.0
        # Unknown metric returns None, not an error.
        assert registry.get_metric(version, "never_logged") is None

    def test_load_via_alias_uri(self, tmp_path: Path) -> None:
        # Register, set @champion, then load via the alias-form
        # ModelVersion. Confirms the run-id vs alias dispatcher in load()
        # picks the right MLflow URI.
        registry = MLflowModelRegistry(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="ef_load_alias_test",
        )
        version = registry.register(
            model=_train_tiny_booster(),
            registered_name="ef_load_alias_model",
            params={"num_leaves": 7},
            metrics={"mape": 0.05},
        )
        registry.set_alias("ef_load_alias_model", "champion", version)

        loaded = registry.load(ModelVersion("ef_load_alias_model@champion"))

        rng = np.random.default_rng(2)
        x = rng.normal(size=(3, 3))
        predictions = loaded.predict(x)
        assert predictions.shape == (3,)

    def test_get_metric_via_alias(self, tmp_path: Path) -> None:
        # Alias-form ModelVersions should also resolve through to the
        # underlying run's metrics.
        registry = MLflowModelRegistry(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="ef_metric_alias_test",
        )
        version = registry.register(
            model=_train_tiny_booster(),
            registered_name="ef_metric_alias_model",
            params={"num_leaves": 7},
            metrics={"mape": 0.04},
        )
        registry.set_alias("ef_metric_alias_model", "champion", version)

        via_alias = registry.get_metric(ModelVersion("ef_metric_alias_model@champion"), "mape")
        assert via_alias == 0.04

    def test_set_alias_rejects_alias_form_input(self, tmp_path: Path) -> None:
        # ``set_alias`` requires a run-id-form ModelVersion because the
        # MLflow API needs the integer registry version. Passing
        # ``...@champion`` is a programming error and surfaces clearly.
        registry = MLflowModelRegistry(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="ef_set_alias_test",
        )
        with pytest.raises(ValueError, match="run-id-form"):
            registry.set_alias(
                "any_model",
                "challenger",
                ModelVersion("any_model@champion"),
            )

    def test_set_alias_rejects_mismatched_name(self, tmp_path: Path) -> None:
        # The ModelVersion's name part must match the ``registered_name``
        # argument — calling set_alias with a different name is a bug.
        registry = MLflowModelRegistry(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="ef_set_alias_test",
        )
        version = registry.register(
            model=_train_tiny_booster(),
            registered_name="model_a",
            params={},
            metrics={"mape": 0.05},
        )
        with pytest.raises(ValueError, match="registered under"):
            registry.set_alias("model_b", "champion", version)

    def test_set_alias_rejects_unknown_run_id(self, tmp_path: Path) -> None:
        # If the run_id doesn't correspond to any registered version,
        # set_alias surfaces a clear error rather than silently creating
        # a dangling alias.
        registry = MLflowModelRegistry(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="ef_unknown_run_test",
        )
        # Register at least one model so the registered model exists.
        registry.register(
            model=_train_tiny_booster(),
            registered_name="model_x",
            params={},
            metrics={"mape": 0.05},
        )
        bogus_run_id = "00000000000000000000000000000000"
        with pytest.raises(ValueError, match="No registered version found"):
            registry.set_alias("model_x", "champion", ModelVersion(f"model_x@{bogus_run_id}"))
