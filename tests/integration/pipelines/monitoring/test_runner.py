"""Integration test for run_monitoring — exercises the full runner against
fakes and a real on-disk feature matrix."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.monitoring.runner import (
    MonitoringResult,
    run_monitoring,
)
from tests.unit.application.fakes import (
    FakeClock,
    FakeLoadForecastRepository,
    FakeLoadObservationRepository,
)

pytestmark = pytest.mark.integration


_NOW = datetime(2026, 5, 8, 12, tzinfo=UTC)
_VERSION = ModelVersion("demand_forecaster@v1")


def _seed_forecasts_and_observations(
    forecast_repo: FakeLoadForecastRepository,
    observation_repo: FakeLoadObservationRepository,
    *,
    zone: BiddingZone,
    start: datetime,
    hours: int,
    predicted_mw: float,
    actual_mw: float,
) -> None:
    """Seed paired forecasts and observations at hourly cadence.

    A constant prediction and constant actual makes MAPE deterministic:
    ``abs(actual - predicted) / actual``.
    """
    for h in range(hours):
        delivery = start + timedelta(hours=h)
        forecast_repo.add_many(
            [
                LoadForecast(
                    zone=zone,
                    as_of_time=delivery - timedelta(hours=24),
                    delivery_time=delivery,
                    predicted_load=EnergyMW(predicted_mw),
                    model_version=_VERSION,
                )
            ]
        )
        observation_repo.add_many(
            [
                LoadObservation(
                    zone=zone,
                    timestamp_utc=delivery,
                    load=EnergyMW(actual_mw),
                )
            ]
        )


def _write_features(
    path: Path,
    *,
    baseline_hours: int,
    recent_hours: int,
    drift: bool,
) -> None:
    """Write a feature matrix split into a baseline window and a recent
    window separated by one hour. Set ``drift=True`` to shift the
    recent slice's distributions far enough that PSI > 0.20.
    """
    rng = np.random.default_rng(seed=42)
    baseline_end = _NOW - timedelta(hours=recent_hours + 1)
    baseline_start = baseline_end - timedelta(hours=baseline_hours)
    recent_start = _NOW - timedelta(hours=recent_hours)

    rows: list[dict[str, object]] = []
    for h in range(baseline_hours):
        ts = baseline_start + timedelta(hours=h)
        rows.append(
            {
                "timestamp_utc": ts,
                "zone": "DE_LU",
                "load_mw": 50_000.0,
                "temp_c": float(rng.normal(loc=15.0, scale=3.0)),
                "wind_10m_ms": 4.0,
                "wind_100m_ms": 8.0,
                "ghi_wm2": 300.0,
                "cloud_cover_pct": 50.0,
                "precip_mm": 0.0,
                "hour_of_day": ts.hour,
                "day_of_week": ts.weekday(),
                "is_weekend": ts.weekday() >= 5,
                "load_lag_1h": 50_000.0,
                "load_lag_24h": 50_000.0,
                "load_lag_168h": 50_000.0,
            }
        )
    for h in range(recent_hours):
        ts = recent_start + timedelta(hours=h)
        # Shift temp_c by 8°C in the recent slice when drift=True;
        # well past the 0.20 PSI threshold for a Gaussian feature.
        temp_loc = 23.0 if drift else 15.0
        rows.append(
            {
                "timestamp_utc": ts,
                "zone": "DE_LU",
                "load_mw": 50_000.0,
                "temp_c": float(rng.normal(loc=temp_loc, scale=3.0)),
                "wind_10m_ms": 4.0,
                "wind_100m_ms": 8.0,
                "ghi_wm2": 300.0,
                "cloud_cover_pct": 50.0,
                "precip_mm": 0.0,
                "hour_of_day": ts.hour,
                "day_of_week": ts.weekday(),
                "is_weekend": ts.weekday() >= 5,
                "load_lag_1h": 50_000.0,
                "load_lag_24h": 50_000.0,
                "load_lag_168h": 50_000.0,
            }
        )

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


class TestRunMonitoringHappyPath:
    def test_calm_inputs_produce_no_retrain_recommendation(self, tmp_path: Path) -> None:
        # Stable distributions + accurate forecasts → both signals well
        # under their thresholds → retrain_recommended is False.
        features_path = tmp_path / "features.parquet"
        _write_features(features_path, baseline_hours=300, recent_hours=168, drift=False)

        forecast_repo = FakeLoadForecastRepository()
        observation_repo = FakeLoadObservationRepository()
        _seed_forecasts_and_observations(
            forecast_repo,
            observation_repo,
            zone=BiddingZone.DE_LU,
            start=_NOW - timedelta(hours=24),
            hours=24,
            predicted_mw=49_500.0,  # 1% under truth
            actual_mw=50_000.0,
        )

        result = run_monitoring(
            features_path=features_path,
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            clock=FakeClock(now=_NOW),
        )

        assert isinstance(result, MonitoringResult)
        assert result.rolling_mape_by_zone["DE_LU"] == pytest.approx(0.01, abs=1e-9)
        assert result.max_psi < 0.20
        assert result.retrain_recommended is False
        assert result.window_end == _NOW
        assert result.window_start == _NOW - timedelta(hours=168)
        # FakeClock does not auto-advance; duration is exactly 0.
        assert result.duration_seconds == 0.0

    def test_drifted_features_recommend_retrain(self, tmp_path: Path) -> None:
        # Same accurate forecasts, but a temperature distribution shift
        # in the recent slice — PSI breaches 0.20 → retrain.
        features_path = tmp_path / "features.parquet"
        _write_features(features_path, baseline_hours=2_000, recent_hours=500, drift=True)

        forecast_repo = FakeLoadForecastRepository()
        observation_repo = FakeLoadObservationRepository()
        _seed_forecasts_and_observations(
            forecast_repo,
            observation_repo,
            zone=BiddingZone.DE_LU,
            start=_NOW - timedelta(hours=24),
            hours=24,
            predicted_mw=49_500.0,
            actual_mw=50_000.0,
        )

        result = run_monitoring(
            features_path=features_path,
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            clock=FakeClock(now=_NOW),
            recent_hours=500,
        )

        assert result.psi_by_feature["temp_c"] > 0.20
        assert result.retrain_recommended is True

    def test_high_rolling_mape_recommends_retrain(self, tmp_path: Path) -> None:
        # Calm features, but the forecasts are 10% off → retrain on the
        # MAPE leg of the OR rule.
        features_path = tmp_path / "features.parquet"
        _write_features(features_path, baseline_hours=300, recent_hours=168, drift=False)

        forecast_repo = FakeLoadForecastRepository()
        observation_repo = FakeLoadObservationRepository()
        _seed_forecasts_and_observations(
            forecast_repo,
            observation_repo,
            zone=BiddingZone.DE_LU,
            start=_NOW - timedelta(hours=24),
            hours=24,
            predicted_mw=45_000.0,  # 10% under
            actual_mw=50_000.0,
        )

        result = run_monitoring(
            features_path=features_path,
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            clock=FakeClock(now=_NOW),
        )

        assert result.rolling_mape_by_zone["DE_LU"] == pytest.approx(0.10, abs=1e-9)
        assert result.retrain_recommended is True


class TestRunMonitoringMissingData:
    def test_no_observations_returns_empty_mape_and_no_retrain(self, tmp_path: Path) -> None:
        # Forecasts exist but no observations — rolling MAPE cannot be
        # computed. Combined with calm features, no retrain.
        features_path = tmp_path / "features.parquet"
        _write_features(features_path, baseline_hours=300, recent_hours=168, drift=False)

        forecast_repo = FakeLoadForecastRepository()
        forecast_repo.add_many(
            [
                LoadForecast(
                    zone=BiddingZone.DE_LU,
                    as_of_time=_NOW - timedelta(hours=48),
                    delivery_time=_NOW - timedelta(hours=24),
                    predicted_load=EnergyMW(50_000.0),
                    model_version=_VERSION,
                )
            ]
        )

        result = run_monitoring(
            features_path=features_path,
            forecast_repo=forecast_repo,
            observation_repo=FakeLoadObservationRepository(),
            clock=FakeClock(now=_NOW),
        )

        assert result.rolling_mape_by_zone == {}
        assert result.max_rolling_mape.value == 0.0
        assert result.retrain_recommended is False

    def test_features_entirely_inside_window_skips_psi(self, tmp_path: Path) -> None:
        # All feature rows are inside the recent window — baseline is
        # empty so PSI cannot be computed. Runner falls back to MAPE-
        # only and does not crash.
        features_path = tmp_path / "features.parquet"
        # 50 hours of features, all within the last 168 → baseline empty
        _write_features(features_path, baseline_hours=0, recent_hours=50, drift=False)

        forecast_repo = FakeLoadForecastRepository()
        observation_repo = FakeLoadObservationRepository()
        _seed_forecasts_and_observations(
            forecast_repo,
            observation_repo,
            zone=BiddingZone.DE_LU,
            start=_NOW - timedelta(hours=24),
            hours=24,
            predicted_mw=49_500.0,
            actual_mw=50_000.0,
        )

        result = run_monitoring(
            features_path=features_path,
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            clock=FakeClock(now=_NOW),
        )

        assert result.psi_by_feature == {}
        assert result.max_psi == 0.0
        assert result.rolling_mape_by_zone["DE_LU"] == pytest.approx(0.01, abs=1e-9)
        assert result.retrain_recommended is False
