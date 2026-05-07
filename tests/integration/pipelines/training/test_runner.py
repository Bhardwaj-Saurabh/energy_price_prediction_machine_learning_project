"""Integration test for run_training — full DAG + registry handoff."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecaster.pipelines.training.runner import run_training
from tests.unit.application.fakes import FakeModelRegistry

pytestmark = pytest.mark.integration


def _write_features_parquet(path: Path, hours: int = 400) -> None:
    """Synthesise a feature matrix and write Parquet at ``path``."""
    rng = np.random.default_rng(seed=7)
    rows = []
    for h in range(hours):
        ts = datetime(2026, 5, 4, tzinfo=UTC) + timedelta(hours=h)
        rows.append(
            {
                "timestamp_utc": ts,
                "zone": "DE_LU",
                "load_mw": 50_000.0 + 100.0 * h + rng.normal(0, 100),
                "temp_c": 15.0 + rng.normal(0, 2),
                "wind_10m_ms": 4.0,
                "wind_100m_ms": 8.0,
                "ghi_wm2": 300.0,
                "cloud_cover_pct": 50.0,
                "precip_mm": 0.0,
                "hour_of_day": ts.hour,
                "day_of_week": ts.weekday(),
                "is_weekend": ts.weekday() >= 5,
                "load_lag_1h": 50_000.0 + 100.0 * (h - 1) if h >= 1 else None,
                "load_lag_24h": 50_000.0 + 100.0 * (h - 24) if h >= 24 else None,
                "load_lag_168h": (50_000.0 + 100.0 * (h - 168) if h >= 168 else None),
            }
        )
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


class TestRunTraining:
    def test_full_pipeline_registers_a_model_and_returns_metrics(self, tmp_path: Path) -> None:
        features_path = tmp_path / "features.parquet"
        _write_features_parquet(features_path, hours=400)
        registry = FakeModelRegistry(next_version="demand_forecaster@abc-123")

        result = run_training(features_path=features_path, registry=registry)

        # Result has the expected shape.
        assert result.model_version.value == "demand_forecaster@abc-123"
        assert result.train_size + result.test_size == 400 - 168
        assert result.test_mape >= 0.0

        # Registry got exactly one call with the right registered_name
        # and the LightGBM hyperparams.
        assert len(registry.calls) == 1
        call = registry.calls[0]
        assert call.registered_name == "demand_forecaster"
        assert call.params["objective"] == "regression"
        assert "mape" in call.metrics
