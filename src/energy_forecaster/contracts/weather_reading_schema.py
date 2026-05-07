"""Pandera schema and converter for weather-reading DataFrames.

Mirrors :mod:`load_observation_schema` for weather. Six measurement
columns instead of one, but the pattern is identical: composite
uniqueness on ``(zone, timestamp_utc)``, strict on extras, coerced
datetimes, ranges shared with the domain entity's plausibility checks.
"""

from collections.abc import Iterable
from typing import Annotated, ClassVar

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

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
    WeatherReading,
)
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone

_VALID_ZONES: tuple[str, ...] = tuple(z.value for z in BiddingZone)


class WeatherReadingSchema(pa.DataFrameModel):
    """Wide-format hourly weather readings (one row per zone x hour)."""

    timestamp_utc: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    zone: Series[str] = pa.Field(isin=_VALID_ZONES)
    temp_c: Series[float] = pa.Field(ge=MIN_TEMP_C, le=MAX_TEMP_C)
    wind_10m_ms: Series[float] = pa.Field(ge=MIN_WIND_MS, le=MAX_WIND_MS)
    wind_100m_ms: Series[float] = pa.Field(ge=MIN_WIND_MS, le=MAX_WIND_MS)
    ghi_wm2: Series[float] = pa.Field(ge=MIN_GHI_WM2, le=MAX_GHI_WM2)
    cloud_cover_pct: Series[float] = pa.Field(ge=MIN_CLOUD_COVER_PCT, le=MAX_CLOUD_COVER_PCT)
    precip_mm: Series[float] = pa.Field(ge=MIN_PRECIP_MM, le=MAX_PRECIP_MM)

    class Config:
        strict = True
        # See LoadObservationSchema for the coerce=False rationale —
        # rejecting naive timestamps at the boundary is the whole point.
        coerce = False
        unique: ClassVar[list[str]] = ["zone", "timestamp_utc"]


_FLOAT_COLUMNS: tuple[str, ...] = (
    "temp_c",
    "wind_10m_ms",
    "wind_100m_ms",
    "ghi_wm2",
    "cloud_cover_pct",
    "precip_mm",
)


@pa.check_types
def to_weather_dataframe(
    readings: Iterable[WeatherReading],
) -> DataFrame[WeatherReadingSchema]:
    """Convert validated domain entities into a validated DataFrame."""
    rows = [
        {
            "timestamp_utc": r.timestamp_utc,
            "zone": r.zone.value,
            "temp_c": r.temp_c,
            "wind_10m_ms": r.wind_10m_ms,
            "wind_100m_ms": r.wind_100m_ms,
            "ghi_wm2": r.ghi_wm2,
            "cloud_cover_pct": r.cloud_cover_pct,
            "precip_mm": r.precip_mm,
        }
        for r in readings
    ]
    if not rows:
        empty = {
            "timestamp_utc": pd.Series([], dtype="datetime64[ns, UTC]"),
            "zone": pd.Series([], dtype=str),
        }
        for column in _FLOAT_COLUMNS:
            empty[column] = pd.Series([], dtype=float)
        return pd.DataFrame(empty)  # type: ignore[no-any-return]
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    return df  # type: ignore[no-any-return]
