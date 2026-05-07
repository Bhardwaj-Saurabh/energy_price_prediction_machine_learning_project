"""Pandera schema for the feature-engineering pipeline output.

The feature matrix is the wide DataFrame consumed by training and
inference. Columns are: the two raw observation columns (load + weather),
calendar features (hour-of-day, day-of-week, weekend), and load lag
features (1h, 24h, 168h). Lag columns are nullable because the very
first hours in a window have no past value to refer to.
"""

from typing import Annotated, ClassVar

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from energy_forecaster.domain.entities.weather_reading import (
    MAX_CLOUD_COVER_PCT,
    MAX_GHI_WM2,
    MAX_PRECIP_MM,
    MAX_TEMP_C,
    MAX_WIND_MS,
    MIN_CLOUD_COVER_PCT,
    MIN_GHI_WM2,
    MIN_PRECIP_MM,
    MIN_TEMP_C,
    MIN_WIND_MS,
)
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import MAX_PLAUSIBLE_LOAD_MW

_VALID_ZONES: tuple[str, ...] = tuple(z.value for z in BiddingZone)


class FeatureMatrixSchema(pa.DataFrameModel):
    """One row per (zone, hour). Columns: target + features (load+weather+calendar+lags)."""

    timestamp_utc: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    zone: Series[str] = pa.Field(isin=_VALID_ZONES)

    # Target column.
    load_mw: Series[float] = pa.Field(ge=0.0, le=MAX_PLAUSIBLE_LOAD_MW)

    # Weather features (joined from WeatherReading).
    temp_c: Series[float] = pa.Field(ge=MIN_TEMP_C, le=MAX_TEMP_C)
    wind_10m_ms: Series[float] = pa.Field(ge=MIN_WIND_MS, le=MAX_WIND_MS)
    wind_100m_ms: Series[float] = pa.Field(ge=MIN_WIND_MS, le=MAX_WIND_MS)
    ghi_wm2: Series[float] = pa.Field(ge=MIN_GHI_WM2, le=MAX_GHI_WM2)
    cloud_cover_pct: Series[float] = pa.Field(ge=MIN_CLOUD_COVER_PCT, le=MAX_CLOUD_COVER_PCT)
    precip_mm: Series[float] = pa.Field(ge=MIN_PRECIP_MM, le=MAX_PRECIP_MM)

    # Calendar features.
    hour_of_day: Series[int] = pa.Field(ge=0, le=23)
    day_of_week: Series[int] = pa.Field(ge=0, le=6)
    is_weekend: Series[bool]

    # Lag features (nullable because the very first hours have no past value).
    load_lag_1h: Series[float] = pa.Field(ge=0.0, le=MAX_PLAUSIBLE_LOAD_MW, nullable=True)
    load_lag_24h: Series[float] = pa.Field(ge=0.0, le=MAX_PLAUSIBLE_LOAD_MW, nullable=True)
    load_lag_168h: Series[float] = pa.Field(ge=0.0, le=MAX_PLAUSIBLE_LOAD_MW, nullable=True)

    class Config:
        strict = True
        coerce = False
        unique: ClassVar[list[str]] = ["zone", "timestamp_utc"]
