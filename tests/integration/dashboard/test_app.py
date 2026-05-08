"""Integration tests for the Dash app — exercises layout + callbacks
without spinning up a real server.

The Dash app's underlying Flask server can be reached via ``app.server``,
but for callback-shape assertions we test the registered callback
functions directly. Dash's callback decorator is a thin wrapper — it
returns the original function — so unit-testable as plain Python.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import pytest
from dash import Dash

from energy_forecaster.config.settings import Environment, Settings
from energy_forecaster.dashboard.app import (
    create_app,
    make_actual_vs_predicted_callback,
    make_drift_callback,
)
from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.mape import MAPE
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.monitoring.runner import MonitoringResult
from tests.unit.application.fakes import (
    FakeClock,
    FakeLoadForecastRepository,
    FakeLoadObservationRepository,
    FakeLogger,
)

pytestmark = pytest.mark.integration


_NOW = datetime(2026, 5, 8, 12, tzinfo=UTC)
_VERSION = ModelVersion("demand_forecaster@v1")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment=Environment.LOCAL,
        local_data_root=tmp_path,
    )


def _seed_pair(
    forecast_repo: FakeLoadForecastRepository,
    observation_repo: FakeLoadObservationRepository,
    *,
    zone: BiddingZone,
    delivery: datetime,
    predicted: float,
    actual: float,
) -> None:
    forecast_repo.add_many(
        [
            LoadForecast(
                zone=zone,
                as_of_time=delivery - timedelta(hours=24),
                delivery_time=delivery,
                predicted_load=EnergyMW(predicted),
                model_version=_VERSION,
            )
        ]
    )
    observation_repo.add_many(
        [LoadObservation(zone=zone, timestamp_utc=delivery, load=EnergyMW(actual))]
    )


def _stub_monitoring_result(*, retrain: bool) -> MonitoringResult:
    return MonitoringResult(
        rolling_mape_by_zone={"DE_LU": 0.02},
        psi_by_feature={"temp_c": 0.05, "wind_10m_ms": 0.30},
        max_rolling_mape=MAPE(0.02),
        max_psi=0.30,
        retrain_recommended=retrain,
        window_start=_NOW - timedelta(days=7),
        window_end=_NOW,
        started_at=_NOW,
        finished_at=_NOW,
    )


class TestCreateApp:
    def test_returns_a_dash_app(self, tmp_path: Path) -> None:
        app = create_app(
            _settings(tmp_path),
            logger=FakeLogger(),
            forecast_repo=FakeLoadForecastRepository(),
            observation_repo=FakeLoadObservationRepository(),
            monitoring_runner=lambda: _stub_monitoring_result(retrain=False),
            clock=FakeClock(now=_NOW),
        )
        assert isinstance(app, Dash)

    def test_layout_contains_expected_components(self, tmp_path: Path) -> None:
        # The layout is the contract a future test could lock down with
        # selenium. For now we assert the IDs the callbacks bind to —
        # if any drifts, the callbacks would silently fail at runtime.
        app = create_app(
            _settings(tmp_path),
            logger=FakeLogger(),
            forecast_repo=FakeLoadForecastRepository(),
            observation_repo=FakeLoadObservationRepository(),
            monitoring_runner=lambda: _stub_monitoring_result(retrain=False),
            clock=FakeClock(now=_NOW),
        )

        # Walk the component tree gathering ids.
        def _ids(component: Any) -> list[str]:
            collected: list[str] = []
            this_id = getattr(component, "id", None)
            if isinstance(this_id, str):
                collected.append(this_id)
            children = getattr(component, "children", None)
            if isinstance(children, list):
                for child in children:
                    collected.extend(_ids(child))
            elif children is not None and not isinstance(children, str):
                collected.extend(_ids(children))
            return collected

        ids = _ids(app.layout)
        assert "zone-dropdown" in ids
        assert "days-back" in ids
        assert "refresh-drift" in ids
        assert "avp-chart" in ids
        assert "drift-summary" in ids
        assert "psi-chart" in ids


class TestActualVsPredictedCallback:
    def test_returns_a_figure_for_a_seeded_zone(self) -> None:
        forecast_repo = FakeLoadForecastRepository()
        observation_repo = FakeLoadObservationRepository()
        for h in range(3):
            _seed_pair(
                forecast_repo,
                observation_repo,
                zone=BiddingZone.DE_LU,
                delivery=_NOW - timedelta(hours=h + 1),
                predicted=50_000.0,
                actual=50_500.0,
            )

        callback = make_actual_vs_predicted_callback(
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            clock=FakeClock(now=_NOW),
        )
        figure = callback("DE_LU", 7)

        assert isinstance(figure, go.Figure)
        # Two traces — predicted + actual.
        assert {trace.name for trace in figure.data} == {"Predicted", "Actual"}

    def test_window_is_clock_now_minus_days(self) -> None:
        # A forecast 8 days old should not appear when days_back=7.
        forecast_repo = FakeLoadForecastRepository()
        observation_repo = FakeLoadObservationRepository()
        _seed_pair(
            forecast_repo,
            observation_repo,
            zone=BiddingZone.DE_LU,
            delivery=_NOW - timedelta(days=8),
            predicted=50_000.0,
            actual=50_500.0,
        )

        callback = make_actual_vs_predicted_callback(
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            clock=FakeClock(now=_NOW),
        )
        figure = callback("DE_LU", 7)

        # The empty-data branch fires; the figure has zero data traces
        # and the placeholder annotation.
        assert len(figure.data) == 0
        assert figure.layout.annotations


class TestDriftCallback:
    def test_no_action_verdict_is_rendered(self) -> None:
        callback = make_drift_callback(lambda: _stub_monitoring_result(retrain=False))
        summary, figure = callback(0)
        assert "no action" in summary
        assert isinstance(figure, go.Figure)

    def test_retrain_verdict_is_rendered(self) -> None:
        callback = make_drift_callback(lambda: _stub_monitoring_result(retrain=True))
        summary, _ = callback(1)
        assert "RETRAIN" in summary

    def test_window_dates_are_in_summary(self) -> None:
        # Operators read the window line to know what timeframe the
        # PSI / MAPE numbers came from. Lock that detail in.
        callback = make_drift_callback(lambda: _stub_monitoring_result(retrain=False))
        summary, _ = callback(0)
        assert "2026-05-01" in summary  # window_start = _NOW - 7 days
        assert "2026-05-08" in summary  # window_end = _NOW

    def test_psi_figure_has_data(self) -> None:
        callback = make_drift_callback(lambda: _stub_monitoring_result(retrain=False))
        _, figure = callback(0)
        # The stub seeds two features, both go into the bar trace.
        assert len(figure.data) == 1
        assert len(figure.data[0].y) == 2
