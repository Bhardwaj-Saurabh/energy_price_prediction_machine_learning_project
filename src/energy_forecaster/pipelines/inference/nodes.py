"""Pure-function nodes for the inference pipeline.

Same shape rules as the training pipeline's nodes: pure, no I/O, no
port interaction. The runner loads the model from the registry and
hands it in via the catalog; these nodes never call MLflow themselves.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from pandera.typing import DataFrame

from energy_forecaster.contracts.feature_matrix_schema import FeatureMatrixSchema
from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.model_version import ModelVersion

# Same feature columns the training pipeline uses — keep these two
# tuples in sync. We deliberately don't share them via import so the
# training and inference pipelines can diverge if needed (e.g. inference
# adds a confidence-interval feature) without one breaking the other.
_FEATURE_COLUMNS: tuple[str, ...] = (
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
)

# How far back ``as_of_time`` is from ``delivery_time``. Day-ahead
# forecasts are typically issued ~24h before delivery; that matches the
# domain entity's invariant (delivery_time > as_of_time).
_BACKTEST_LEAD_HOURS: int = 24


def slice_recent_features(features: DataFrame[FeatureMatrixSchema], hours: int) -> dict[str, Any]:
    """Take the last ``hours`` rows per zone as prediction inputs.

    Drops rows with NaN lags (same as training — the model can't predict
    on missing lag features). Returns the X feature matrix plus the
    metadata needed downstream to build :class:`LoadForecast` entities.
    """
    df = features.dropna(subset=["load_lag_1h", "load_lag_24h", "load_lag_168h"]).copy()
    df = df.sort_values(["zone", "timestamp_utc"])
    df["zone_cat"] = df["zone"].astype("category").cat.codes

    recent = df.groupby("zone", sort=False).tail(hours).reset_index(drop=True)
    return {
        "X": recent[list(_FEATURE_COLUMNS)],
        "zones": recent["zone"].tolist(),
        "delivery_times": recent["timestamp_utc"].tolist(),
    }


def predict_loads(model: Any, prediction_inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the model on the prediction inputs and return raw predictions."""
    predictions = model.predict(prediction_inputs["X"])
    return {**prediction_inputs, "predictions": list(predictions)}


def build_forecasts(
    prediction_data: dict[str, Any], model_version: ModelVersion
) -> list[LoadForecast]:
    """Construct :class:`LoadForecast` entities from raw predictions.

    Backtest semantics: each delivery hour's ``as_of_time`` is fixed at
    24 hours before the delivery hour. That mirrors the day-ahead market
    cadence the model is trained for and satisfies the entity's
    ``delivery_time > as_of_time`` invariant.

    Tiny negative predictions are clipped to zero. LightGBM occasionally
    overshoots on low-load periods and produces values just below zero;
    :class:`EnergyMW` would reject those, but the appropriate response
    is to clip rather than crash — predictions are estimates, not
    measurements.
    """
    forecasts: list[LoadForecast] = []
    for zone_str, delivery_ts, prediction in zip(
        prediction_data["zones"],
        prediction_data["delivery_times"],
        prediction_data["predictions"],
        strict=True,
    ):
        delivery_dt = pd.Timestamp(delivery_ts).to_pydatetime()
        as_of_dt = delivery_dt - timedelta(hours=_BACKTEST_LEAD_HOURS)
        clipped_load = max(0.0, float(prediction))
        forecasts.append(
            LoadForecast(
                zone=BiddingZone(zone_str),
                as_of_time=_floor_to_hour(as_of_dt),
                delivery_time=_floor_to_hour(delivery_dt),
                predicted_load=EnergyMW(clipped_load),
                model_version=model_version,
            )
        )
    return forecasts


def _floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)
