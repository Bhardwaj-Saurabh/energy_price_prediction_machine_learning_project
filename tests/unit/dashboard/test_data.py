"""Unit tests for the dashboard's data-loading helpers."""

from datetime import UTC, datetime, timedelta

import pandas as pd

from energy_forecaster.dashboard.data import load_actual_vs_predicted
from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from tests.unit.application.fakes import (
    FakeLoadForecastRepository,
    FakeLoadObservationRepository,
)


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


_VERSION = ModelVersion("demand_forecaster@v1")


def _forecast(zone: BiddingZone, delivery: datetime, mw: float) -> LoadForecast:
    return LoadForecast(
        zone=zone,
        as_of_time=delivery - timedelta(hours=24),
        delivery_time=delivery,
        predicted_load=EnergyMW(mw),
        model_version=_VERSION,
    )


def _observation(zone: BiddingZone, ts: datetime, mw: float) -> LoadObservation:
    return LoadObservation(zone=zone, timestamp_utc=ts, load=EnergyMW(mw))


class TestLoadActualVsPredicted:
    def test_pairs_forecast_with_matching_observation(self) -> None:
        forecast_repo = FakeLoadForecastRepository()
        observation_repo = FakeLoadObservationRepository()
        delivery = _utc(2026, 5, 7, 12)
        forecast_repo.add_many([_forecast(BiddingZone.DE_LU, delivery, 50_000.0)])
        observation_repo.add_many([_observation(BiddingZone.DE_LU, delivery, 50_500.0)])

        df = load_actual_vs_predicted(
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            zone=BiddingZone.DE_LU,
            since=_utc(2026, 5, 7),
            until=_utc(2026, 5, 8),
        )

        assert list(df.columns) == ["delivery_time", "predicted_load_mw", "actual_load_mw"]
        assert len(df) == 1
        assert df.iloc[0]["predicted_load_mw"] == 50_000.0
        assert df.iloc[0]["actual_load_mw"] == 50_500.0

    def test_unmatched_forecast_gets_nan_actual(self) -> None:
        # A forecast for an hour we have not observed yet must render
        # — its actual is NaN, the chart will draw it as a gap. The
        # alternative (drop unmatched) would silently truncate the
        # forecast line at the last observation, which is misleading.
        forecast_repo = FakeLoadForecastRepository()
        observation_repo = FakeLoadObservationRepository()
        delivery = _utc(2026, 5, 7, 12)
        forecast_repo.add_many([_forecast(BiddingZone.DE_LU, delivery, 50_000.0)])

        df = load_actual_vs_predicted(
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            zone=BiddingZone.DE_LU,
            since=_utc(2026, 5, 7),
            until=_utc(2026, 5, 8),
        )

        assert len(df) == 1
        assert df.iloc[0]["predicted_load_mw"] == 50_000.0
        assert pd.isna(df.iloc[0]["actual_load_mw"])

    def test_results_are_sorted_by_delivery_time(self) -> None:
        forecast_repo = FakeLoadForecastRepository()
        observation_repo = FakeLoadObservationRepository()
        # Insert forecasts in reverse chronological order. The output
        # must come back sorted ascending so the line chart renders
        # left-to-right.
        for h in (5, 1, 3, 0, 4, 2):
            delivery = _utc(2026, 5, 7) + timedelta(hours=h)
            forecast_repo.add_many([_forecast(BiddingZone.DE_LU, delivery, 50_000.0)])

        df = load_actual_vs_predicted(
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            zone=BiddingZone.DE_LU,
            since=_utc(2026, 5, 7),
            until=_utc(2026, 5, 8),
        )

        assert list(df["delivery_time"]) == sorted(df["delivery_time"])

    def test_filters_by_zone(self) -> None:
        # Forecasts for FR must not appear when querying DE_LU.
        forecast_repo = FakeLoadForecastRepository()
        observation_repo = FakeLoadObservationRepository()
        delivery = _utc(2026, 5, 7, 12)
        forecast_repo.add_many(
            [
                _forecast(BiddingZone.DE_LU, delivery, 50_000.0),
                _forecast(BiddingZone.FR, delivery, 30_000.0),
            ]
        )

        df = load_actual_vs_predicted(
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            zone=BiddingZone.DE_LU,
            since=_utc(2026, 5, 7),
            until=_utc(2026, 5, 8),
        )

        assert len(df) == 1
        assert df.iloc[0]["predicted_load_mw"] == 50_000.0

    def test_filters_by_window(self) -> None:
        forecast_repo = FakeLoadForecastRepository()
        observation_repo = FakeLoadObservationRepository()
        for h in range(5):
            forecast_repo.add_many(
                [_forecast(BiddingZone.DE_LU, _utc(2026, 5, 7) + timedelta(hours=h), 50_000.0)]
            )

        df = load_actual_vs_predicted(
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            zone=BiddingZone.DE_LU,
            since=_utc(2026, 5, 7, 1),
            until=_utc(2026, 5, 7, 4),
        )

        assert len(df) == 3  # hours 1, 2, 3 (until is exclusive)

    def test_empty_window_returns_empty_dataframe_with_columns(self) -> None:
        # An empty repo must still return a frame with the expected
        # columns so the chart helper's ``df.empty`` branch fires
        # cleanly rather than KeyError-ing on missing columns.
        df = load_actual_vs_predicted(
            forecast_repo=FakeLoadForecastRepository(),
            observation_repo=FakeLoadObservationRepository(),
            zone=BiddingZone.DE_LU,
            since=_utc(2026, 5, 7),
            until=_utc(2026, 5, 8),
        )

        assert df.empty
        assert list(df.columns) == ["delivery_time", "predicted_load_mw", "actual_load_mw"]
