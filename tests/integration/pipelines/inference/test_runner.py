"""Integration test for run_inference — full DAG against tmp_path + fakes."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.inference.runner import run_inference
from tests.unit.application.fakes import (
    FakeClock,
    FakeLoadForecastRepository,
    FakeModelRegistry,
)

pytestmark = pytest.mark.integration


def _write_features(path: Path, hours: int = 300) -> None:
    rows = []
    base = 50_000.0
    for h in range(hours):
        ts = datetime(2026, 5, 4, tzinfo=UTC) + timedelta(hours=h)
        rows.append(
            {
                "timestamp_utc": ts,
                "zone": "DE_LU",
                "load_mw": base + 100.0 * h,
                "temp_c": 15.0,
                "wind_10m_ms": 4.0,
                "wind_100m_ms": 8.0,
                "ghi_wm2": 300.0,
                "cloud_cover_pct": 50.0,
                "precip_mm": 0.0,
                "hour_of_day": ts.hour,
                "day_of_week": ts.weekday(),
                "is_weekend": ts.weekday() >= 5,
                "load_lag_1h": base + 100.0 * (h - 1) if h >= 1 else None,
                "load_lag_24h": base + 100.0 * (h - 24) if h >= 24 else None,
                "load_lag_168h": base + 100.0 * (h - 168) if h >= 168 else None,
            }
        )
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


class _StubModel:
    """Trivial naive model for the runner integration test — predicts
    yesterday's value at the same hour. Returns a numpy array, same as a
    real lightgbm.Booster's predict()."""

    def predict(self, X: pd.DataFrame) -> Any:
        return X["load_lag_24h"].to_numpy()


class TestRunInference:
    def test_full_pipeline_loads_model_predicts_and_persists(self, tmp_path: Path) -> None:
        features_path = tmp_path / "features.parquet"
        _write_features(features_path, hours=300)

        registry = FakeModelRegistry()
        version = ModelVersion("demand_forecaster@abc-123")
        registry.preload(version, _StubModel())

        repo = FakeLoadForecastRepository()
        clock = FakeClock(now=datetime(2026, 5, 14, 12, tzinfo=UTC))

        result = run_inference(
            features_path=features_path,
            registry=registry,
            repo=repo,
            clock=clock,
            model_version=version,
            hours=24,
        )

        assert result.forecasts_produced == 24
        assert result.forecasts_inserted == 24
        assert result.model_version == version
        # duration_seconds is the started→finished delta; assert it is
        # reachable so the property stays exercised.
        assert result.duration_seconds == 0.0  # FakeClock did not advance
        # Repo got exactly the 24 forecasts the pipeline produced.
        assert len(repo.all()) == 24
        # Each forecast's model_version matches what was loaded.
        assert all(f.model_version == version for f in repo.all())

    def test_re_running_inserts_zero_new_forecasts(self, tmp_path: Path) -> None:
        features_path = tmp_path / "features.parquet"
        _write_features(features_path, hours=300)

        registry = FakeModelRegistry()
        version = ModelVersion("demand_forecaster@abc-123")
        registry.preload(version, _StubModel())
        repo = FakeLoadForecastRepository()
        clock = FakeClock(now=datetime(2026, 5, 14, 12, tzinfo=UTC))

        first = run_inference(
            features_path=features_path,
            registry=registry,
            repo=repo,
            clock=clock,
            model_version=version,
            hours=24,
        )
        second = run_inference(
            features_path=features_path,
            registry=registry,
            repo=repo,
            clock=clock,
            model_version=version,
            hours=24,
        )

        assert first.forecasts_inserted == 24
        assert second.forecasts_produced == 24
        assert second.forecasts_inserted == 0
