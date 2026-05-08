"""Data-loading helpers that bridge ports to dashboard-shaped DataFrames.

Reads from :class:`LoadForecastRepository` and
:class:`LoadObservationRepository` via the application ports and
returns DataFrames in the exact shape the chart helpers expect. Keeps
the chart layer pure (no port awareness) and the app layer simple (it
just wires inputs to outputs).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from energy_forecaster.application.ports.load_forecast_repository import (
    LoadForecastRepository,
)
from energy_forecaster.application.ports.load_observation_repository import (
    LoadObservationRepository,
)
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


def load_actual_vs_predicted(
    *,
    forecast_repo: LoadForecastRepository,
    observation_repo: LoadObservationRepository,
    zone: BiddingZone,
    since: datetime,
    until: datetime,
) -> pd.DataFrame:
    """Build a DataFrame with one row per forecast in the window.

    Columns: ``delivery_time``, ``predicted_load_mw``, ``actual_load_mw``.
    Outer-joined on the predicted side: every forecast renders even if
    its delivery hour has not been observed yet (the actual column is
    NaN, the chart will show a gap). Forecasts and observations are
    matched on ``delivery_time == timestamp_utc``. If two forecasts
    share a delivery time (different model versions), the *latest by
    iteration order* wins — same convention as the monitoring node.

    The DataFrame is sorted by ``delivery_time`` ascending so the line
    chart renders left-to-right without a re-sort downstream.
    """
    forecasts = forecast_repo.find_by_zone(zone, since=since, until=until)
    observations = observation_repo.find_by_zone(zone, since=since, until=until)

    actuals_by_time = {o.timestamp_utc: o.load.value for o in observations}

    rows = [
        {
            "delivery_time": f.delivery_time,
            "predicted_load_mw": f.predicted_load.value,
            "actual_load_mw": actuals_by_time.get(f.delivery_time, float("nan")),
        }
        for f in forecasts
    ]

    df = pd.DataFrame(rows, columns=["delivery_time", "predicted_load_mw", "actual_load_mw"])
    if df.empty:
        return df
    return df.sort_values("delivery_time").reset_index(drop=True)
