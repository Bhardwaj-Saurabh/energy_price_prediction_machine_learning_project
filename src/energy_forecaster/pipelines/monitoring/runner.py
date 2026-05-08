"""Programmatic runner for the monitoring pipeline.

Bridges two pure node functions to two read ports:
:class:`LoadForecastRepository` (predictions) and
:class:`LoadObservationRepository` (truth). The nodes themselves are
in :mod:`energy_forecaster.pipelines.monitoring.nodes` and never touch
infrastructure — same architectural rule the inference and training
runners follow.

Unlike inference/training, this runner does *not* go through a Kedro
``Pipeline``. Monitoring has only two independent compute steps over
in-memory data; the catalog/DAG machinery would add more boilerplate
than it removes. Inference and training keep Kedro because they have
Parquet inputs and longer DAGs that benefit from the visualisation.

Window semantics:
  * ``window_end   = clock.now()``
  * ``window_start = window_end - recent_hours``
  * Rolling MAPE is computed on forecasts whose ``delivery_time`` falls
    inside ``[window_start, window_end)`` matched against observations
    in the same window.
  * PSI compares the *baseline* slice of the feature matrix (rows older
    than ``window_start``) against the *recent* slice (rows from
    ``window_start`` onwards).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from energy_forecaster.application.ports.clock import Clock
from energy_forecaster.application.ports.load_forecast_repository import (
    LoadForecastRepository,
)
from energy_forecaster.application.ports.load_observation_repository import (
    LoadObservationRepository,
)
from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.rules.retrain import should_retrain
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.mape import MAPE
from energy_forecaster.pipelines.monitoring.nodes import (
    compute_psi_per_feature,
    compute_rolling_mape_per_zone,
)

# Which feature columns get a PSI score. Same set the training pipeline
# uses (kept as a local tuple rather than imported from training/nodes
# so the two pipelines can diverge — same convention as the inference
# pipeline). If you add a feature that the model uses, add it here too,
# otherwise drift on it goes undetected.
_MONITORED_FEATURES: tuple[str, ...] = (
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
)

# Default recent window: 168 hours = 7 days. Long enough to give the
# rolling MAPE statistical heft, short enough to surface a recent
# regime change. Configurable per call.
_DEFAULT_RECENT_HOURS: int = 168


@dataclass(frozen=True, slots=True)
class MonitoringResult:
    """Summary returned by :func:`run_monitoring`.

    ``rolling_mape_by_zone`` is empty when no zone has matched truth
    pairs in the window. ``psi_by_feature`` is empty when the feature
    matrix is entirely on one side of ``window_start`` (and PSI cannot
    be computed). In either case ``max_rolling_mape`` / ``max_psi``
    fall back to ``0.0`` so the rule does not fire on a missing-data
    signal alone.
    """

    rolling_mape_by_zone: dict[str, float]
    psi_by_feature: dict[str, float]
    max_rolling_mape: MAPE
    max_psi: float
    retrain_recommended: bool
    window_start: datetime
    window_end: datetime
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


def run_monitoring(
    *,
    features_path: Path,
    forecast_repo: LoadForecastRepository,
    observation_repo: LoadObservationRepository,
    clock: Clock,
    zones: Iterable[BiddingZone] = tuple(BiddingZone),
    recent_hours: int = _DEFAULT_RECENT_HOURS,
) -> MonitoringResult:
    """Compute drift signals and return a retrain verdict.

    Reads forecasts + observations for each requested zone in the
    recent window, splits the feature matrix into baseline / recent
    slices, computes rolling MAPE and PSI, and applies the retrain
    rule. Does *not* trigger retraining — the orchestrator consumes
    ``retrain_recommended`` and decides what to do.
    """
    started_at = clock.now()
    window_end = started_at
    window_start = window_end - timedelta(hours=recent_hours)

    forecasts: list[LoadForecast] = []
    observations: list[LoadObservation] = []
    for zone in zones:
        forecasts.extend(forecast_repo.find_by_zone(zone, since=window_start, until=window_end))
        observations.extend(
            observation_repo.find_by_zone(zone, since=window_start, until=window_end)
        )

    rolling_mape_by_zone = compute_rolling_mape_per_zone(forecasts, observations)

    features = pd.read_parquet(features_path)
    baseline_features = features[features["timestamp_utc"] < window_start]
    recent_features = features[features["timestamp_utc"] >= window_start]

    psi_by_feature: dict[str, float] = {}
    if not baseline_features.empty and not recent_features.empty:
        psi_by_feature = compute_psi_per_feature(
            baseline_features, recent_features, list(_MONITORED_FEATURES)
        )

    max_mape_value = max(rolling_mape_by_zone.values(), default=0.0)
    max_psi_value = max(psi_by_feature.values(), default=0.0)
    verdict = should_retrain(rolling_mape=MAPE(max_mape_value), max_psi=max_psi_value)

    finished_at = clock.now()
    return MonitoringResult(
        rolling_mape_by_zone=rolling_mape_by_zone,
        psi_by_feature=psi_by_feature,
        max_rolling_mape=MAPE(max_mape_value),
        max_psi=max_psi_value,
        retrain_recommended=verdict,
        window_start=window_start,
        window_end=window_end,
        started_at=started_at,
        finished_at=finished_at,
    )
