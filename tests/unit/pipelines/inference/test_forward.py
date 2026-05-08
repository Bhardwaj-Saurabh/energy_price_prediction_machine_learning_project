"""Unit tests for the forward-inference helpers."""

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.entities.weather_reading import WeatherReading
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.pipelines.inference.forward import (
    build_partial_features,
    predict_recursively,
)


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _observation(zone: BiddingZone, ts: datetime, mw: float) -> LoadObservation:
    return LoadObservation(zone=zone, timestamp_utc=ts, load=EnergyMW(mw))


def _reading(zone: BiddingZone, ts: datetime, **fields: float) -> WeatherReading:
    defaults = {
        "temp_c": 15.0,
        "wind_10m_ms": 4.0,
        "wind_100m_ms": 8.0,
        "ghi_wm2": 200.0,
        "cloud_cover_pct": 50.0,
        "precip_mm": 0.0,
    }
    defaults.update(fields)
    return WeatherReading(zone=zone, timestamp_utc=ts, **defaults)


def _seed_history(zone: BiddingZone, end: datetime, hours: int) -> list[LoadObservation]:
    """Hourly observations ending at ``end`` (exclusive). Linear ramp so
    test assertions on lag_24h / lag_168h are easy to compute."""
    return [
        _observation(zone, end - timedelta(hours=h), 50_000.0 + 100.0 * (hours - h))
        for h in range(1, hours + 1)
    ]


class TestBuildPartialFeatures:
    def test_one_row_per_delivery_time_in_chronological_order(self) -> None:
        zone = BiddingZone.DE_LU
        delivery_times = [_utc(2026, 5, 9, h) for h in range(3)]
        observations = _seed_history(zone, end=_utc(2026, 5, 9), hours=200)
        weather = [_reading(zone, t) for t in delivery_times]

        df = build_partial_features(
            zone=zone,
            delivery_times=delivery_times,
            observations=observations,
            weather_forecast=weather,
        )

        assert len(df) == 3
        assert list(df["timestamp_utc"]) == delivery_times

    def test_calendar_features_match_delivery_time(self) -> None:
        # 2026-05-09 is a Saturday — day_of_week=5, is_weekend=True.
        zone = BiddingZone.DE_LU
        delivery_times = [_utc(2026, 5, 9, 14)]
        observations = _seed_history(zone, end=_utc(2026, 5, 9, 14), hours=200)
        weather = [_reading(zone, _utc(2026, 5, 9, 14))]

        df = build_partial_features(
            zone=zone,
            delivery_times=delivery_times,
            observations=observations,
            weather_forecast=weather,
        )

        assert df.at[0, "hour_of_day"] == 14
        assert df.at[0, "day_of_week"] == 5
        assert df.at[0, "is_weekend"] is True or df.at[0, "is_weekend"] == True  # noqa: E712

    def test_lag_features_pulled_from_observations(self) -> None:
        zone = BiddingZone.DE_LU
        delivery = _utc(2026, 5, 9, 12)
        # Observation at delivery - 24h = 2026-05-08T12:00 with load=99_999.
        # Observation at delivery - 168h = 2026-05-02T12:00 with load=12_345.
        observations = [
            _observation(zone, delivery - timedelta(hours=24), 99_999.0),
            _observation(zone, delivery - timedelta(hours=168), 12_345.0),
        ]
        weather = [_reading(zone, delivery)]

        df = build_partial_features(
            zone=zone,
            delivery_times=[delivery],
            observations=observations,
            weather_forecast=weather,
        )

        assert df.at[0, "load_lag_24h"] == 99_999.0
        assert df.at[0, "load_lag_168h"] == 12_345.0

    def test_load_lag_1h_is_nan_until_recursive_fill(self) -> None:
        zone = BiddingZone.DE_LU
        delivery_times = [_utc(2026, 5, 9, h) for h in range(2)]
        observations = _seed_history(zone, end=_utc(2026, 5, 9), hours=200)
        weather = [_reading(zone, t) for t in delivery_times]

        df = build_partial_features(
            zone=zone,
            delivery_times=delivery_times,
            observations=observations,
            weather_forecast=weather,
        )

        assert df["load_lag_1h"].isna().all()

    def test_weather_fields_propagate_into_feature_row(self) -> None:
        zone = BiddingZone.DE_LU
        delivery = _utc(2026, 5, 9, 8)
        observations = _seed_history(zone, end=_utc(2026, 5, 9, 8), hours=200)
        weather = [
            _reading(
                zone,
                delivery,
                temp_c=21.5,
                wind_10m_ms=6.0,
                wind_100m_ms=11.0,
                ghi_wm2=420.0,
                cloud_cover_pct=30.0,
                precip_mm=0.2,
            )
        ]

        df = build_partial_features(
            zone=zone,
            delivery_times=[delivery],
            observations=observations,
            weather_forecast=weather,
        )

        assert df.at[0, "temp_c"] == 21.5
        assert df.at[0, "ghi_wm2"] == 420.0
        assert df.at[0, "precip_mm"] == 0.2

    def test_missing_lag_24h_observation_raises(self) -> None:
        zone = BiddingZone.DE_LU
        delivery = _utc(2026, 5, 9, 12)
        # Only the lag_168h observation is provided; lag_24h is missing.
        observations = [_observation(zone, delivery - timedelta(hours=168), 50_000.0)]
        weather = [_reading(zone, delivery)]

        with pytest.raises(ValueError, match="load_lag_24h"):
            build_partial_features(
                zone=zone,
                delivery_times=[delivery],
                observations=observations,
                weather_forecast=weather,
            )

    def test_missing_lag_168h_observation_raises(self) -> None:
        # The lag_24h observation is present but lag_168h is not. The
        # error message must name the right column so the operator
        # knows how far back the missing data is.
        zone = BiddingZone.DE_LU
        delivery = _utc(2026, 5, 9, 12)
        observations = [_observation(zone, delivery - timedelta(hours=24), 50_000.0)]
        weather = [_reading(zone, delivery)]

        with pytest.raises(ValueError, match="load_lag_168h"):
            build_partial_features(
                zone=zone,
                delivery_times=[delivery],
                observations=observations,
                weather_forecast=weather,
            )

    def test_missing_weather_forecast_raises(self) -> None:
        zone = BiddingZone.DE_LU
        delivery = _utc(2026, 5, 9, 12)
        observations = _seed_history(zone, end=delivery, hours=200)

        with pytest.raises(ValueError, match="weather forecast"):
            build_partial_features(
                zone=zone,
                delivery_times=[delivery],
                observations=observations,
                weather_forecast=[],
            )

    def test_observations_for_other_zones_are_ignored(self) -> None:
        # An observation for FR with the right timestamp must not be
        # used as DE_LU's lag input. The function must filter by zone.
        zone = BiddingZone.DE_LU
        delivery = _utc(2026, 5, 9, 12)
        observations = [
            # Wrong zone — should be ignored
            _observation(BiddingZone.FR, delivery - timedelta(hours=24), 1.0),
            _observation(BiddingZone.FR, delivery - timedelta(hours=168), 2.0),
        ]
        weather = [_reading(zone, delivery)]

        with pytest.raises(ValueError, match="load_lag_24h"):
            build_partial_features(
                zone=zone,
                delivery_times=[delivery],
                observations=observations,
                weather_forecast=weather,
            )

    def test_unsorted_delivery_times_are_sorted_in_output(self) -> None:
        # Caller passes delivery times out of order; the function must
        # sort so the recursive predictor iterates chronologically.
        zone = BiddingZone.DE_LU
        delivery_times = [_utc(2026, 5, 9, 5), _utc(2026, 5, 9, 1), _utc(2026, 5, 9, 3)]
        observations = _seed_history(zone, end=_utc(2026, 5, 9, 6), hours=200)
        weather = [_reading(zone, t) for t in delivery_times]

        df = build_partial_features(
            zone=zone,
            delivery_times=delivery_times,
            observations=observations,
            weather_forecast=weather,
        )

        assert list(df["timestamp_utc"]) == [
            _utc(2026, 5, 9, 1),
            _utc(2026, 5, 9, 3),
            _utc(2026, 5, 9, 5),
        ]


class _IdentityModel:
    """Predicts ``load_lag_1h`` directly. Lets the recursive test assert
    that prediction k feeds into row k+1's lag input."""

    def predict(self, X: pd.DataFrame) -> Any:
        return X["load_lag_1h"].to_numpy()


class _ConstantModel:
    """Predicts a fixed value regardless of input. Useful for confirming
    the loop produces the right number of outputs."""

    def __init__(self, value: float) -> None:
        self._value = value

    def predict(self, X: pd.DataFrame) -> Any:
        return np.full(len(X), self._value)


class _NegativeModel:
    """Always predicts -1. The recursive helper must clip to zero."""

    def predict(self, X: pd.DataFrame) -> Any:
        return np.full(len(X), -1.0)


_FEATURE_COLUMNS: list[str] = [
    "temp_c",
    "wind_10m_ms",
    "wind_100m_ms",
    "ghi_wm2",
    "cloud_cover_pct",
    "precip_mm",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "load_lag_1h",
    "load_lag_24h",
    "load_lag_168h",
    "zone_cat",
]


def _build_three_row_partial(zone: BiddingZone = BiddingZone.DE_LU) -> pd.DataFrame:
    delivery_times = [_utc(2026, 5, 9, h) for h in range(3)]
    observations = _seed_history(zone, end=_utc(2026, 5, 9), hours=200)
    weather = [_reading(zone, t) for t in delivery_times]
    return build_partial_features(
        zone=zone,
        delivery_times=delivery_times,
        observations=observations,
        weather_forecast=weather,
    )


class TestPredictRecursively:
    def test_returns_one_prediction_per_row(self) -> None:
        partial = _build_three_row_partial()
        predictions = predict_recursively(
            model=_ConstantModel(50_000.0),
            partial_features=partial,
            initial_lag_1h=49_000.0,
            feature_columns=_FEATURE_COLUMNS,
        )
        assert len(predictions) == 3

    def test_each_prediction_feeds_next_rows_lag_1h(self) -> None:
        # Identity model returns the lag_1h it was fed. With initial=42,
        # row 0 predicts 42 → row 1's lag_1h becomes 42 → row 1 predicts
        # 42 → row 2 predicts 42. The chain must hold.
        partial = _build_three_row_partial()
        predictions = predict_recursively(
            model=_IdentityModel(),
            partial_features=partial,
            initial_lag_1h=42_000.0,
            feature_columns=_FEATURE_COLUMNS,
        )
        assert predictions == [42_000.0, 42_000.0, 42_000.0]

    def test_caller_dataframe_is_not_mutated(self) -> None:
        # The function copies internally; load_lag_1h on the caller's
        # frame stays NaN even though the predictor filled it in its
        # working copy.
        partial = _build_three_row_partial()
        predict_recursively(
            model=_ConstantModel(50_000.0),
            partial_features=partial,
            initial_lag_1h=49_000.0,
            feature_columns=_FEATURE_COLUMNS,
        )
        assert partial["load_lag_1h"].isna().all()

    def test_negative_predictions_are_clipped_to_zero(self) -> None:
        # A pathological model that returns -1 on every call. The
        # clipping is the same defensive policy backtest mode applies.
        partial = _build_three_row_partial()
        predictions = predict_recursively(
            model=_NegativeModel(),
            partial_features=partial,
            initial_lag_1h=50_000.0,
            feature_columns=_FEATURE_COLUMNS,
        )
        assert predictions == [0.0, 0.0, 0.0]
