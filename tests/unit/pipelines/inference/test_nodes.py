"""Unit tests for the inference pipeline's pure-function nodes."""

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.inference.nodes import (
    build_forecasts,
    predict_loads,
    slice_recent_features,
)


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _features_frame(hours: int, zone: str = "DE_LU") -> pd.DataFrame:
    """Build a feature matrix with all schema columns populated, lags
    valid from hour 168 onward."""
    rows = []
    base = 50_000.0
    for h in range(hours):
        ts = _utc(2026, 5, 4) + timedelta(hours=h)
        rows.append(
            {
                "timestamp_utc": ts,
                "zone": zone,
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
    return df


class TestSliceRecentFeatures:
    def test_returns_last_n_rows_per_zone(self) -> None:
        df = _features_frame(hours=300)
        out = slice_recent_features(df, hours=24)
        assert len(out["X"]) == 24
        assert len(out["zones"]) == 24

    def test_drops_rows_with_missing_lags(self) -> None:
        # First 168 rows have at least one NaN lag — they cannot be
        # predicted on. With hours=400 input + hours=24 slice, all 24
        # output rows must be drawn from the lag-complete tail.
        df = _features_frame(hours=400)
        out = slice_recent_features(df, hours=24)
        # All 24 rows have non-NaN inputs (the X matrix won't contain NaN).
        assert not out["X"].isna().any().any()

    def test_per_zone_tail_does_not_bleed_across_zones(self) -> None:
        de = _features_frame(hours=300, zone="DE_LU")
        fr = _features_frame(hours=300, zone="FR")
        df = pd.concat([de, fr]).reset_index(drop=True)

        out = slice_recent_features(df, hours=10)
        # 10 rows per zone, 2 zones = 20 total.
        assert len(out["X"]) == 20
        assert out["zones"].count("DE_LU") == 10
        assert out["zones"].count("FR") == 10


class TestPredictLoads:
    def test_passes_X_to_model_and_returns_predictions(self) -> None:
        class _StubModel:
            def predict(self, X: pd.DataFrame) -> Any:
                return X["load_lag_24h"].to_numpy()

        df = _features_frame(hours=300)
        inputs = slice_recent_features(df, hours=5)
        out = predict_loads(_StubModel(), inputs)
        assert "predictions" in out
        assert len(out["predictions"]) == 5


class TestBuildForecasts:
    def _seed(self) -> dict[str, Any]:
        df = _features_frame(hours=200)
        inputs = slice_recent_features(df, hours=3)

        class _Stub:
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.array([45_000.0, 46_000.0, 47_000.0])

        return predict_loads(_Stub(), inputs)

    def test_constructs_one_forecast_per_prediction(self) -> None:
        data = self._seed()
        forecasts = build_forecasts(data, model_version=ModelVersion("demand_forecaster@v1"))
        assert len(forecasts) == 3
        assert all(isinstance(f, LoadForecast) for f in forecasts)

    def test_as_of_time_is_24h_before_delivery(self) -> None:
        data = self._seed()
        forecasts = build_forecasts(data, model_version=ModelVersion("demand_forecaster@v1"))
        for f in forecasts:
            assert f.delivery_time - f.as_of_time == timedelta(hours=24)

    def test_negative_predictions_are_clipped_to_zero(self) -> None:
        # LightGBM occasionally predicts slightly below zero on low-load
        # periods. EnergyMW would reject negatives; the node clips so
        # the entity construction succeeds.
        df = _features_frame(hours=200)
        inputs = slice_recent_features(df, hours=2)

        class _NegStub:
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.array([-5.0, 100.0])

        data = predict_loads(_NegStub(), inputs)
        forecasts = build_forecasts(data, model_version=ModelVersion("demand_forecaster@v1"))
        assert forecasts[0].predicted_load.value == 0.0
        assert forecasts[1].predicted_load.value == 100.0

    def test_zone_string_round_trips_to_bidding_zone_enum(self) -> None:
        df = _features_frame(hours=200, zone="FR")
        inputs = slice_recent_features(df, hours=2)

        class _Stub:
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.array([100.0, 200.0])

        data = predict_loads(_Stub(), inputs)
        forecasts = build_forecasts(data, model_version=ModelVersion("demand_forecaster@v1"))
        assert all(f.zone is BiddingZone.FR for f in forecasts)


def test_unsupported_zone_string_propagates_as_value_error() -> None:
    # The build_forecasts node passes the zone string straight to
    # BiddingZone(); an unknown zone raises ValueError. We do not catch
    # it because that would mask a real upstream bug — it is data
    # corruption, not an expected model output.
    df = _features_frame(hours=200, zone="DE_LU")
    inputs = slice_recent_features(df, hours=1)

    class _Stub:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([100.0])

    data = predict_loads(_Stub(), inputs)
    data["zones"] = ["XX_NOT_A_ZONE"]
    with pytest.raises(ValueError):
        build_forecasts(data, model_version=ModelVersion("v"))
