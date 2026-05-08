"""Integration test for run_inference — full DAG against tmp_path + fakes."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.entities.weather_reading import WeatherReading
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.inference.runner import (
    run_forward_inference,
    run_inference,
)
from tests.unit.application.fakes import (
    FakeClock,
    FakeLoadForecastRepository,
    FakeLoadObservationRepository,
    FakeModelRegistry,
    FakeWeatherClient,
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


_FORWARD_NOW = datetime(2026, 5, 8, 12, 30, tzinfo=UTC)
_FORWARD_AS_OF = datetime(2026, 5, 8, 12, tzinfo=UTC)  # floored from _FORWARD_NOW


def _seed_observations(zone: BiddingZone, hours: int = 200) -> list[LoadObservation]:
    """Hourly observations ending an hour before the first delivery."""
    return [
        LoadObservation(
            zone=zone,
            timestamp_utc=_FORWARD_AS_OF - timedelta(hours=h),
            load=EnergyMW(50_000.0 + 100.0 * h),
        )
        for h in range(hours)
    ]


def _seed_forecast_weather(zone: BiddingZone, hours: int = 24) -> list[WeatherReading]:
    """Synthetic weather forecast covering the next ``hours`` delivery slots."""
    return [
        WeatherReading(
            zone=zone,
            timestamp_utc=_FORWARD_AS_OF + timedelta(hours=h + 1),
            temp_c=15.0,
            wind_10m_ms=4.0,
            wind_100m_ms=8.0,
            ghi_wm2=300.0,
            cloud_cover_pct=50.0,
            precip_mm=0.0,
        )
        for h in range(hours)
    ]


class TestRunForwardInference:
    def test_persists_forecasts_with_shared_as_of_time(self) -> None:
        zone = BiddingZone.DE_LU
        registry = FakeModelRegistry()
        version = ModelVersion("demand_forecaster@v1")
        registry.preload(version, _StubModel())

        observation_repo = FakeLoadObservationRepository()
        observation_repo.add_many(_seed_observations(zone))

        forecast_repo = FakeLoadForecastRepository()
        weather = FakeWeatherClient()
        weather.seed_forecast(zone, _seed_forecast_weather(zone))
        clock = FakeClock(now=_FORWARD_NOW)

        result = run_forward_inference(
            registry=registry,
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            weather=weather,
            clock=clock,
            model_version=version,
            zones=[zone],
            hours=24,
        )

        assert result.forecasts_produced == 24
        assert result.forecasts_inserted == 24

        # All forecasts share the same as_of_time = floored clock.now().
        produced = forecast_repo.all()
        assert len({f.as_of_time for f in produced}) == 1
        assert produced[0].as_of_time == _FORWARD_AS_OF

        # Delivery times are the next 24 hours from the floored now.
        expected = [_FORWARD_AS_OF + timedelta(hours=h) for h in range(1, 25)]
        assert sorted(f.delivery_time for f in produced) == expected

    def test_re_running_inserts_zero_new_forecasts(self) -> None:
        # Forecast identity is (zone, delivery_time, model_version);
        # re-running the same window dedupes.
        zone = BiddingZone.DE_LU
        registry = FakeModelRegistry()
        version = ModelVersion("demand_forecaster@v1")
        registry.preload(version, _StubModel())

        observation_repo = FakeLoadObservationRepository()
        observation_repo.add_many(_seed_observations(zone))

        forecast_repo = FakeLoadForecastRepository()
        weather = FakeWeatherClient()
        weather.seed_forecast(zone, _seed_forecast_weather(zone))

        first = run_forward_inference(
            registry=registry,
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            weather=weather,
            clock=FakeClock(now=_FORWARD_NOW),
            model_version=version,
            zones=[zone],
            hours=24,
        )
        second = run_forward_inference(
            registry=registry,
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            weather=weather,
            clock=FakeClock(now=_FORWARD_NOW),
            model_version=version,
            zones=[zone],
            hours=24,
        )

        assert first.forecasts_inserted == 24
        assert second.forecasts_inserted == 0

    def test_missing_seed_observation_raises(self) -> None:
        # No observation at delivery[0] - 1h means recursive lag_1h
        # has no starting value. Must raise rather than guess.
        #
        # ``hours=20`` makes delivery[19] = as_of + 20h; lag_24h for that
        # row points at as_of - 4h, not as_of itself — so we can prune
        # the observation at exactly ``as_of`` (the seed time) without
        # also breaking any lag_24h lookup.
        zone = BiddingZone.DE_LU
        registry = FakeModelRegistry()
        version = ModelVersion("demand_forecaster@v1")
        registry.preload(version, _StubModel())

        observation_repo = FakeLoadObservationRepository()
        observations = [o for o in _seed_observations(zone) if o.timestamp_utc != _FORWARD_AS_OF]
        observation_repo.add_many(observations)

        forecast_repo = FakeLoadForecastRepository()
        weather = FakeWeatherClient()
        weather.seed_forecast(zone, _seed_forecast_weather(zone, hours=20))

        with pytest.raises(ValueError, match="seed recursive load_lag_1h"):
            run_forward_inference(
                registry=registry,
                forecast_repo=forecast_repo,
                observation_repo=observation_repo,
                weather=weather,
                clock=FakeClock(now=_FORWARD_NOW),
                model_version=version,
                zones=[zone],
                hours=20,
            )
