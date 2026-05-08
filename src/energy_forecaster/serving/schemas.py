"""Pydantic request / response models for the FastAPI serving layer.

These are *transport* schemas — what the HTTP layer accepts and emits —
distinct from the Pandera *data* schemas in ``contracts/`` (which
validate DataFrame shape) and the domain *entity* dataclasses (which
enforce business invariants). Names use snake_case to match the JSON
the API speaks; conversions to/from domain entities live in
:mod:`serving.app`.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Minimal liveness check response."""

    status: str
    environment: str


class ForecastResponse(BaseModel):
    """Wire-shape for one persisted :class:`LoadForecast`."""

    zone: str
    as_of_time: datetime
    delivery_time: datetime
    predicted_load_mw: float
    model_version: str


class PredictRequest(BaseModel):
    """Body for ``POST /predict``.

    Both fields default — ``model`` to the @champion alias (the same
    default the CLI uses) and ``hours`` to a day-ahead window. The
    server runs the inference pipeline synchronously; for production
    workloads this should move to a background task that returns 202.
    """

    model: str = Field(default="demand_forecaster@champion")
    hours: int = Field(default=24, ge=1, le=168)


class PredictResponse(BaseModel):
    """Body returned by ``POST /predict``."""

    model_version: str
    forecasts_produced: int
    forecasts_inserted: int
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
