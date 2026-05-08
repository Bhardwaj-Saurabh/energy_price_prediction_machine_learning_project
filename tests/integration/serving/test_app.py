"""Integration tests for the FastAPI app (real routes, real LocalFs repo)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from energy_forecaster.adapters.load_forecast_repo.local_fs import (
    LocalFsLoadForecastRepository,
)
from energy_forecaster.application.errors import DataSourceUnavailableError
from energy_forecaster.config.settings import Environment, Settings
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.inference.runner import InferenceResult
from energy_forecaster.serving.app import InferenceRunner, create_app
from tests.unit.application.fakes import FakeLogger

pytestmark = pytest.mark.integration


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment=Environment.LOCAL,
        local_data_root=tmp_path,
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )


def _seed_forecast_jsonl(tmp_path: Path, zone: str, count: int = 3) -> None:
    directory = tmp_path / "load_forecasts"
    directory.mkdir(parents=True, exist_ok=True)
    base_delivery = datetime(2026, 5, 5, tzinfo=UTC)
    with (directory / f"{zone}.jsonl").open("w", encoding="utf-8") as f:
        for h in range(count):
            delivery = base_delivery + timedelta(hours=h)
            f.write(
                json.dumps(
                    {
                        "zone": zone,
                        "as_of_time": (delivery - timedelta(hours=24)).isoformat(),
                        "delivery_time": delivery.isoformat(),
                        "predicted_load": 50_000.0 + 100.0 * h,
                        "model_version": "demand_forecaster@test_run_id",
                    }
                )
                + "\n"
            )


def _unused_runner(
    mv: ModelVersion, features: Path | None = None, hours: int = 24
) -> InferenceResult:
    """Default runner for tests that exercise non-/predict endpoints.

    Raises if accidentally called — surfaces a "test wired wrong" bug
    rather than silently returning fake data."""
    raise AssertionError(  # pragma: no cover
        "inference runner should not be called for this test"
    )


def _build_app(
    tmp_path: Path,
    *,
    inference_runner: InferenceRunner = _unused_runner,
    logger: FakeLogger | None = None,
) -> tuple[Any, FakeLogger]:
    settings = _settings(tmp_path)
    repo = LocalFsLoadForecastRepository(root=tmp_path)
    log = logger or FakeLogger()
    app = create_app(
        settings,
        logger=log,
        forecast_repo=repo,
        inference_runner=inference_runner,
    )
    return app, log


class TestHealthEndpoint:
    def test_returns_ok_with_environment(self, tmp_path: Path) -> None:
        app, _ = _build_app(tmp_path)
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "environment": "local"}


class TestGetForecastEndpoint:
    def test_returns_persisted_forecasts_for_known_zone(self, tmp_path: Path) -> None:
        _seed_forecast_jsonl(tmp_path, "DE_LU", count=5)
        app, _ = _build_app(tmp_path)

        with TestClient(app) as client:
            response = client.get("/forecast/DE_LU")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 5
        assert all(row["zone"] == "DE_LU" for row in body)
        delivery_times = [row["delivery_time"] for row in body]
        assert delivery_times == sorted(delivery_times)

    def test_filters_by_since_and_until(self, tmp_path: Path) -> None:
        _seed_forecast_jsonl(tmp_path, "DE_LU", count=5)
        app, _ = _build_app(tmp_path)

        # Only deliveries at 02:00 and 03:00 fall in [02:00, 04:00).
        # ``params=`` lets httpx URL-encode the ``+`` in the offset; if
        # we passed the timestamps in the path string, the ``+`` would
        # round-trip as a literal space and Pydantic would reject it.
        since = "2026-05-05T02:00:00+00:00"
        until = "2026-05-05T04:00:00+00:00"
        with TestClient(app) as client:
            response = client.get("/forecast/DE_LU", params={"since": since, "until": until})

        assert response.status_code == 200
        assert len(response.json()) == 2

    def test_returns_empty_list_when_no_forecasts_exist(self, tmp_path: Path) -> None:
        # No JSONL written for FR; the repo returns empty rather than
        # 404, since "no forecasts yet" is a valid state for a new zone.
        _seed_forecast_jsonl(tmp_path, "DE_LU", count=3)
        app, _ = _build_app(tmp_path)

        with TestClient(app) as client:
            response = client.get("/forecast/FR")

        assert response.status_code == 200
        assert response.json() == []

    def test_unknown_zone_returns_400(self, tmp_path: Path) -> None:
        app, _ = _build_app(tmp_path)
        with TestClient(app) as client:
            response = client.get("/forecast/ES")

        assert response.status_code == 400
        assert "Unknown zone" in response.json()["detail"]

    def test_request_id_header_is_used_in_logs(self, tmp_path: Path) -> None:
        _seed_forecast_jsonl(tmp_path, "DE_LU", count=1)
        app, logger = _build_app(tmp_path)

        with TestClient(app) as client:
            response = client.get("/forecast/DE_LU", headers={"X-Request-Id": "trace-abc-123"})

        assert response.status_code == 200
        served_events = [c for c in logger.calls if c.event == "forecast.served"]
        assert len(served_events) == 1
        assert served_events[0].context["request_id"] == "trace-abc-123"


def _runner_returning(result: InferenceResult) -> InferenceRunner:
    def _runner(mv: ModelVersion, features: Path | None = None, hours: int = 24) -> InferenceResult:
        return result

    return _runner


def _capturing_runner(captured: dict[str, Any], result: InferenceResult) -> InferenceRunner:
    def _runner(mv: ModelVersion, features: Path | None = None, hours: int = 24) -> InferenceResult:
        captured["model_version"] = mv
        captured["hours"] = hours
        return result

    return _runner


def _runner_raising(exc: Exception) -> InferenceRunner:
    def _runner(mv: ModelVersion, features: Path | None = None, hours: int = 24) -> InferenceResult:
        raise exc

    return _runner


class TestPredictEndpoint:
    def test_runs_inference_and_returns_result(self, tmp_path: Path) -> None:
        result = InferenceResult(
            model_version=ModelVersion("demand_forecaster@v1"),
            forecasts_produced=24,
            forecasts_inserted=24,
            started_at=datetime(2026, 5, 7, 12, tzinfo=UTC),
            finished_at=datetime(2026, 5, 7, 12, 0, 1, tzinfo=UTC),
        )
        app, _ = _build_app(tmp_path, inference_runner=_runner_returning(result))

        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={"model": "demand_forecaster@champion", "hours": 24},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["model_version"] == "demand_forecaster@v1"
        assert body["forecasts_produced"] == 24
        assert body["forecasts_inserted"] == 24

    def test_default_body_uses_champion_alias(self, tmp_path: Path) -> None:
        # Sending an empty body should fill in the @champion default.
        # The capturing runner records what the route forwarded.
        captured: dict[str, Any] = {}
        now = datetime(2026, 5, 7, 12, tzinfo=UTC)
        result = InferenceResult(
            model_version=ModelVersion("demand_forecaster@v1"),
            forecasts_produced=24,
            forecasts_inserted=24,
            started_at=now,
            finished_at=now,
        )
        app, _ = _build_app(tmp_path, inference_runner=_capturing_runner(captured, result))

        with TestClient(app) as client:
            response = client.post("/predict", json={})

        assert response.status_code == 200
        captured_mv = captured["model_version"]
        assert isinstance(captured_mv, ModelVersion)
        assert captured_mv.value == "demand_forecaster@champion"
        assert captured["hours"] == 24

    def test_application_error_returns_502(self, tmp_path: Path) -> None:
        # ApplicationError → 502 (upstream / infrastructure problem).
        app, _ = _build_app(
            tmp_path,
            inference_runner=_runner_raising(
                DataSourceUnavailableError("simulated upstream outage")
            ),
        )

        with TestClient(app) as client:
            response = client.post("/predict", json={})

        assert response.status_code == 502
        assert "simulated upstream outage" in response.json()["detail"]

    def test_unhandled_error_returns_500(self, tmp_path: Path) -> None:
        # Any non-ApplicationError exception → 500 (server-side bug).
        app, _ = _build_app(
            tmp_path,
            inference_runner=_runner_raising(RuntimeError("simulated bug")),
        )

        with TestClient(app) as client:
            response = client.post("/predict", json={})

        assert response.status_code == 500
        assert "simulated bug" in response.json()["detail"]

    def test_invalid_hours_is_rejected_by_pydantic(self, tmp_path: Path) -> None:
        # PredictRequest enforces 1 <= hours <= 168. Out-of-range values
        # never reach the runner.
        app, _ = _build_app(tmp_path)
        with TestClient(app) as client:
            response = client.post("/predict", json={"hours": 0})

        assert response.status_code == 422  # FastAPI validation error


class TestComposedAppViaBuildApp:
    def test_build_app_returns_a_working_fastapi_app(self, tmp_path: Path) -> None:
        # End-to-end check that composition.build_app wires everything
        # correctly. Health is a safe endpoint to hit because it does
        # not touch the inference runner.
        from fastapi import FastAPI

        from energy_forecaster.composition import build_app

        settings = _settings(tmp_path)
        app = build_app(settings, logger=FakeLogger())
        # build_app advertises ``object`` (to keep FastAPI out of the
        # composition module's public type surface). Tests narrow it
        # back here so TestClient is happy.
        assert isinstance(app, FastAPI)

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
