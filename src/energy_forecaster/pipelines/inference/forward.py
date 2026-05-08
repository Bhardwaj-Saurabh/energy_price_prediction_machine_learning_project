"""Pure helpers for forward (day-ahead) inference.

Forward inference differs from backtest inference in two ways:

  1. **Time direction.** Predictions cover *future* delivery hours; the
     model has never seen ground truth for them and the feature row is
     synthesised from observations + a weather forecast.
  2. **Recursive lag_1h.** ``load_lag_1h`` for a future hour is, by
     definition, the load *at the previous future hour* — which is
     itself a prediction. We walk delivery times in order, feed each
     prediction back as the next row's lag_1h, and accept that
     prediction errors compound forward through the horizon. For the
     24-hour day-ahead window this is acceptable; for week-long
     horizons it gets noisy fast.

This module owns the two pieces that are unique to forward mode:
:func:`build_partial_features` (deterministic feature assembly) and
:func:`predict_recursively` (the recursive prediction loop). Backtest
mode keeps using :mod:`energy_forecaster.pipelines.inference.nodes`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.entities.weather_reading import WeatherReading
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


def build_partial_features(
    *,
    zone: BiddingZone,
    delivery_times: list[datetime],
    observations: list[LoadObservation],
    weather_forecast: list[WeatherReading],
) -> pd.DataFrame:
    """Assemble feature rows for ``delivery_times``, all columns except ``load_lag_1h``.

    Every delivery time needs:
      * a weather reading at exactly ``t`` (from the forecast adapter),
      * an observation at exactly ``t - 24h`` (for ``load_lag_24h``),
      * an observation at exactly ``t - 168h`` (for ``load_lag_168h``).

    Missing data raises :class:`ValueError` — the runner cannot recover
    and silently producing NaN-laden rows would fail downstream model
    inference with an opaque error.

    The returned DataFrame is sorted by ``timestamp_utc`` ascending so
    the recursive predictor can iterate in chronological order without
    a re-sort.
    """
    obs_by_time: dict[datetime, float] = {
        o.timestamp_utc: o.load.value for o in observations if o.zone == zone
    }
    weather_by_time: dict[datetime, WeatherReading] = {
        w.timestamp_utc: w for w in weather_forecast if w.zone == zone
    }

    rows: list[dict[str, object]] = []
    for delivery_time in sorted(delivery_times):
        try:
            lag_24h_value = obs_by_time[delivery_time - timedelta(hours=24)]
        except KeyError as exc:
            raise ValueError(
                f"Missing observation for {zone.value} at {delivery_time - timedelta(hours=24)} "
                f"(needed for load_lag_24h at delivery {delivery_time})"
            ) from exc
        try:
            lag_168h_value = obs_by_time[delivery_time - timedelta(hours=168)]
        except KeyError as exc:
            raise ValueError(
                f"Missing observation for {zone.value} at {delivery_time - timedelta(hours=168)} "
                f"(needed for load_lag_168h at delivery {delivery_time})"
            ) from exc
        try:
            weather = weather_by_time[delivery_time]
        except KeyError as exc:
            raise ValueError(
                f"Missing weather forecast for {zone.value} at {delivery_time}"
            ) from exc

        rows.append(
            {
                "timestamp_utc": delivery_time,
                "zone": zone.value,
                "temp_c": weather.temp_c,
                "wind_10m_ms": weather.wind_10m_ms,
                "wind_100m_ms": weather.wind_100m_ms,
                "ghi_wm2": weather.ghi_wm2,
                "cloud_cover_pct": weather.cloud_cover_pct,
                "precip_mm": weather.precip_mm,
                "hour_of_day": delivery_time.hour,
                "day_of_week": delivery_time.weekday(),
                "is_weekend": delivery_time.weekday() >= 5,
                "load_lag_24h": lag_24h_value,
                "load_lag_168h": lag_168h_value,
                "load_lag_1h": float("nan"),
            }
        )

    df = pd.DataFrame(rows).reset_index(drop=True)
    # Match the training pipeline's zone encoding. Single-zone forward
    # runs always produce code 0; for multi-zone runs the order tracks
    # the alphabetical category order, same as training.
    df["zone_cat"] = df["zone"].astype("category").cat.codes
    return df


def predict_recursively(
    *,
    model: Any,
    partial_features: pd.DataFrame,
    initial_lag_1h: float,
    feature_columns: list[str],
) -> list[float]:
    """Iterate ``partial_features`` chronologically and predict each row.

    For row 0, ``load_lag_1h`` is set to ``initial_lag_1h`` (the
    observed load at ``delivery_times[0] - 1h``, looked up by the
    runner). For each subsequent row, ``load_lag_1h`` is the
    just-produced prediction. Returns predictions in DataFrame order.

    Negative predictions are clipped to zero — same boundary safeguard
    the backtest runner applies. ``EnergyMW`` would reject a negative
    value at construction; clipping here keeps the prediction story
    "estimates that round down to zero on edge cases" rather than
    crashing the run.
    """
    df = partial_features.copy()
    predictions: list[float] = []
    lag_1h = initial_lag_1h
    for i in range(len(df)):
        df.at[i, "load_lag_1h"] = lag_1h
        x_row = df.iloc[[i]][feature_columns]
        raw = float(model.predict(x_row)[0])
        clipped = max(0.0, raw)
        predictions.append(clipped)
        lag_1h = clipped
    return predictions
