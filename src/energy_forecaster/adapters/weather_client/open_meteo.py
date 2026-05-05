"""OpenMeteoClient — production weather adapter backed by Open-Meteo's free API.

Open-Meteo serves historical and forecast weather without an API key.
We hit the archive endpoint for closed past windows; future support for
live forecast windows can land in a sibling adapter when the inference
pipeline needs it.

Boundary responsibilities (same template as :class:`EntsoePyClient`):
  * Map :class:`BiddingZone` to a representative lat/lon. One point per
    zone is a deliberate simplification — production-grade would weight
    multiple stations across the area; we promote that model when we
    have a forecast quality reason to.
  * Translate any HTTP / network / parse failure into
    :class:`DataSourceUnavailableError`. The use case must never see
    ``requests`` exceptions.
  * Skip rows where any measurement is missing — Open-Meteo represents
    those as JSON ``null``. Including them would crash :class:`WeatherReading`
    construction downstream (correctly), but skipping at the boundary is
    quieter and matches the load adapter's NaN-skip behaviour.
"""

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import requests

from energy_forecaster.application.errors import DataSourceUnavailableError
from energy_forecaster.domain.entities.weather_reading import WeatherReading
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone

# Representative lat/lon per zone. Choices are major cities at the
# population/load centre — a single-point approximation suitable for a
# portfolio piece. Real systems weight multiple stations.
_ZONE_LAT_LON: dict[BiddingZone, tuple[float, float]] = {
    BiddingZone.DE_LU: (50.11, 8.68),  # Frankfurt am Main
    BiddingZone.FR: (48.85, 2.35),  # Paris
    BiddingZone.GB: (51.50, -0.13),  # London
}

_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
_REQUEST_TIMEOUT_SECONDS: float = 30.0

# Open-Meteo variable names → our domain field names. Pinning the names
# here makes any API change show up in this one dict, not strewn across
# the parsing code.
_HOURLY_VARS: dict[str, str] = {
    "temperature_2m": "temp_c",
    "wind_speed_10m": "wind_10m_ms",
    "wind_speed_100m": "wind_100m_ms",
    "shortwave_radiation": "ghi_wm2",
    "cloud_cover": "cloud_cover_pct",
    "precipitation": "precip_mm",
}


class OpenMeteoClient:
    """HTTP-backed :class:`WeatherClient` implementation."""

    def fetch_weather(
        self,
        *,
        zone: BiddingZone,
        start: datetime,
        end: datetime,
    ) -> Iterable[WeatherReading]:
        lat, lon = _ZONE_LAT_LON[zone]
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "hourly": ",".join(_HOURLY_VARS),
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }

        try:
            response = requests.get(_ARCHIVE_URL, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise DataSourceUnavailableError(
                f"Open-Meteo query failed for {zone.value}: {exc}"
            ) from exc

        return list(_to_readings(payload, zone=zone, start=start, end=end))


def _to_readings(
    payload: dict[str, Any], *, zone: BiddingZone, start: datetime, end: datetime
) -> Iterable[WeatherReading]:
    hourly = payload.get("hourly")
    if not hourly:
        return

    times = hourly["time"]
    columns = {ours: hourly[theirs] for theirs, ours in _HOURLY_VARS.items()}

    for i, ts_str in enumerate(times):
        # Open-Meteo returns naive ISO timestamps when timezone=UTC; we
        # add tzinfo here so downstream entity construction is given the
        # UTC-aware datetime it requires.
        ts = datetime.fromisoformat(ts_str).replace(tzinfo=UTC)
        if not (start <= ts < end):
            continue

        row = {field: columns[field][i] for field in columns}
        if any(value is None for value in row.values()):
            continue

        yield WeatherReading(zone=zone, timestamp_utc=ts, **row)
