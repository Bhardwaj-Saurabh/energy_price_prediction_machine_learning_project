"""Pure-function nodes for the monitoring pipeline.

Same shape rules as the inference and training pipelines: each function
is pure, takes inputs from the catalog, returns named outputs. No I/O,
no port interaction — the runner does the port work and hands data in.

The two nodes here compute the *inputs* that
:func:`energy_forecaster.domain.rules.retrain.should_retrain` consumes:
rolling MAPE per zone (performance drift) and PSI per feature (data
drift). The verdict itself is computed in the runner — keeping it out
of the DAG keeps the rule a single call site rather than a node that
hides the policy.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import pandas as pd

from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.pipelines.monitoring.metrics import (
    mape,
    population_stability_index,
)


def compute_rolling_mape_per_zone(
    forecasts: Iterable[LoadForecast],
    observations: Iterable[LoadObservation],
) -> dict[str, float]:
    """Per-zone MAPE on the forecasts that have matching observations.

    Aligns on ``(zone, delivery_time == timestamp_utc)``. A forecast for
    a delivery hour we do not yet have an observation for is silently
    dropped — that hour has not happened yet (or has not been ingested).
    Zones with zero matched pairs are omitted from the result rather
    than reported as ``nan``; the runner treats "no data" and "low
    score" as different signals.

    If a delivery hour has multiple forecasts (different model
    versions), the *last one in input order* wins. Callers that want a
    specific version should filter upstream.
    """
    forecasts_by_zone: dict[str, dict[datetime, float]] = {}
    for f in forecasts:
        forecasts_by_zone.setdefault(f.zone.value, {})[f.delivery_time] = f.predicted_load.value

    observations_by_zone: dict[str, dict[datetime, float]] = {}
    for o in observations:
        observations_by_zone.setdefault(o.zone.value, {})[o.timestamp_utc] = o.load.value

    result: dict[str, float] = {}
    for zone, predicted in forecasts_by_zone.items():
        truth = observations_by_zone.get(zone, {})
        common_times = sorted(predicted.keys() & truth.keys())
        if not common_times:
            continue
        actuals = [truth[t] for t in common_times]
        predictions = [predicted[t] for t in common_times]
        result[zone] = mape(actuals, predictions)
    return result


def compute_psi_per_feature(
    baseline_features: pd.DataFrame,
    recent_features: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    """PSI for each feature column comparing baseline vs recent windows.

    Both inputs are expected to share the listed columns. Columns that
    are constant in baseline and constant in recent at the same value
    return PSI=0 by construction (handled in the metric helper); columns
    that are constant only in baseline return PSI=inf, which the rule
    treats as a strong drift signal.

    Returns a ``{feature_name: psi}`` mapping. Empty input frames raise
    via the helper so the bug does not silently produce a verdict.
    """
    return {
        column: population_stability_index(
            expected=baseline_features[column].to_numpy(dtype=float),
            observed=recent_features[column].to_numpy(dtype=float),
        )
        for column in feature_columns
    }
