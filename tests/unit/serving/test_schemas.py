"""Unit tests for the serving Pydantic schemas (transport DTOs)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from energy_forecaster.serving.schemas import (
    ForecastResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)


class TestPredictRequest:
    def test_defaults_are_set(self) -> None:
        req = PredictRequest()
        assert req.model == "demand_forecaster@champion"
        assert req.hours == 24

    def test_explicit_values_are_kept(self) -> None:
        req = PredictRequest(model="other@v1", hours=48)
        assert req.model == "other@v1"
        assert req.hours == 48

    def test_hours_below_minimum_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(hours=0)

    def test_hours_above_maximum_is_rejected(self) -> None:
        # 168 hours = 1 week; anything past that suggests a bug or a
        # week-ahead use case the pipeline doesn't yet support.
        with pytest.raises(ValidationError):
            PredictRequest(hours=200)


class TestForecastResponse:
    def test_round_trips_through_json(self) -> None:
        original = ForecastResponse(
            zone="DE_LU",
            as_of_time=datetime(2026, 5, 4, 0, tzinfo=UTC),
            delivery_time=datetime(2026, 5, 5, 0, tzinfo=UTC),
            predicted_load_mw=58_400.0,
            model_version="demand_forecaster@v1",
        )
        as_json = original.model_dump_json()
        restored = ForecastResponse.model_validate_json(as_json)
        assert restored == original


class TestHealthResponse:
    def test_constructs_with_status_and_environment(self) -> None:
        body = HealthResponse(status="ok", environment="local")
        assert body.status == "ok"
        assert body.environment == "local"


class TestPredictResponse:
    def test_constructs_and_serialises(self) -> None:
        body = PredictResponse(
            model_version="demand_forecaster@v1",
            forecasts_produced=24,
            forecasts_inserted=24,
            started_at=datetime(2026, 5, 4, 0, tzinfo=UTC),
            finished_at=datetime(2026, 5, 4, 0, 0, 1, tzinfo=UTC),
            duration_seconds=1.0,
        )
        as_json = body.model_dump_json()
        restored = PredictResponse.model_validate_json(as_json)
        assert restored == body
