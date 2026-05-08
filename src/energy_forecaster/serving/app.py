"""FastAPI application factory.

Same architectural slot as :mod:`energy_forecaster.cli` — a *framework*
that parses inbound requests, binds a request-scoped logger, calls into
its injected dependencies, and renders responses. **No business logic
in routes.** The body of each route is at most a few lines: pull data
via a port, transform, return.

Composition (rather than import) is how this module gets its hands on
the forecast repo and the inference runner. The composition root passes
them in; the serving module stays unaware of MLflow, Kedro, or the
filesystem layout — same dependency-rule the use cases follow.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi import Path as FastApiPath

from energy_forecaster.application.errors import ApplicationError
from energy_forecaster.application.ports.load_forecast_repository import (
    LoadForecastRepository,
)
from energy_forecaster.application.ports.logger import Logger
from energy_forecaster.config.settings import Settings
from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.inference.runner import InferenceResult
from energy_forecaster.serving.schemas import (
    ForecastResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

InferenceRunner = Callable[[ModelVersion, Path | None, int], InferenceResult]


def _to_response(forecast: LoadForecast) -> ForecastResponse:
    """Convert a domain entity into the wire-shape DTO."""
    return ForecastResponse(
        zone=forecast.zone.value,
        as_of_time=forecast.as_of_time,
        delivery_time=forecast.delivery_time,
        predicted_load_mw=forecast.predicted_load.value,
        model_version=forecast.model_version.value,
    )


def create_app(
    settings: Settings,
    *,
    logger: Logger,
    forecast_repo: LoadForecastRepository,
    inference_runner: InferenceRunner,
) -> FastAPI:
    """Build a FastAPI app around the injected dependencies.

    Each request gets its own bound logger with a ``request_id`` field —
    sourced from the inbound ``X-Request-Id`` header if present, or
    generated on the fly. This mirrors the CLI's per-invocation
    correlation_id so HTTP logs and pipeline logs share a tracing key.
    """
    app = FastAPI(
        title="Energy Forecaster",
        version="0.1.0",
        description=(
            "HTTP serving layer for the energy demand forecaster. "
            "Reads persisted forecasts and triggers fresh inference runs."
        ),
    )

    def _request_logger(request_id: str | None) -> Logger:
        return logger.bind(request_id=request_id or str(uuid.uuid4()))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", environment=settings.environment.value)

    @app.get("/forecast/{zone}", response_model=list[ForecastResponse])
    def get_forecast(
        zone: Annotated[str, FastApiPath(description="Bidding zone (DE_LU, FR, GB).")],
        since: Annotated[
            datetime | None,
            Query(description="Lower bound on delivery_time (inclusive)."),
        ] = None,
        until: Annotated[
            datetime | None,
            Query(description="Upper bound on delivery_time (exclusive)."),
        ] = None,
        x_request_id: Annotated[str | None, Header()] = None,
    ) -> list[ForecastResponse]:
        log = _request_logger(x_request_id).bind(operation="get_forecast", zone=zone)
        try:
            zone_enum = BiddingZone(zone)
        except ValueError as exc:
            log.warning("forecast.unknown_zone")
            raise HTTPException(
                status_code=400,
                detail=(f"Unknown zone {zone!r}. Supported: {[z.value for z in BiddingZone]}."),
            ) from exc

        forecasts = list(forecast_repo.find_by_zone(zone_enum, since=since, until=until))
        log.info("forecast.served", count=len(forecasts))
        return [_to_response(f) for f in forecasts]

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        request: PredictRequest,
        x_request_id: Annotated[str | None, Header()] = None,
    ) -> PredictResponse:
        log = _request_logger(x_request_id).bind(
            operation="predict", model=request.model, hours=request.hours
        )
        log.info("predict.start")

        try:
            result = inference_runner(ModelVersion(request.model), None, request.hours)
        except ApplicationError as exc:
            log.error("predict.application_error", error=str(exc))
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            # Boundary catch — same policy as the CLI predict handler.
            # MLflow / pyarrow / filesystem errors become 500s with a
            # logged error type rather than crashing the request.
            log.error(
                "predict.unhandled_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        log.info(
            "predict.done",
            forecasts_produced=result.forecasts_produced,
            forecasts_inserted=result.forecasts_inserted,
            duration_seconds=round(result.duration_seconds, 3),
        )
        return PredictResponse(
            model_version=result.model_version.value,
            forecasts_produced=result.forecasts_produced,
            forecasts_inserted=result.forecasts_inserted,
            started_at=result.started_at,
            finished_at=result.finished_at,
            duration_seconds=result.duration_seconds,
        )

    return app
