"""InMemoryWeatherClient — synthetic-data weather adapter for local demos.

Like :class:`InMemoryEntsoeClient`, this is a real adapter that satisfies
the :class:`WeatherClient` Protocol — not a test double. It generates
deterministic, plausible-looking hourly weather so the local CLI can run
end-to-end without network. Tests of the use case use a separate
``FakeWeatherClient`` (with seed/fail control) in ``tests/`` because
those test-control hooks have no place on a production adapter.
"""

from collections.abc import Iterable
from datetime import datetime, timedelta
from math import cos, pi, sin

from energy_forecaster.domain.entities.weather_reading import WeatherReading
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone

# Per-zone baselines — rough mid-spring averages for the representative
# city of each zone (Frankfurt, Paris, London). Adding a new zone without
# an entry fails loudly with KeyError, which is the desired behaviour.
_ZONE_TEMP_BASELINE_C: dict[BiddingZone, float] = {
    BiddingZone.DE_LU: 14.0,
    BiddingZone.FR: 16.0,
    BiddingZone.GB: 12.0,
}
_DIURNAL_TEMP_AMPLITUDE_C: float = 6.0
_BASE_WIND_10M_MS: float = 4.0
_BASE_WIND_100M_MS: float = 8.0
_GHI_PEAK_WM2: float = 600.0
_BASE_CLOUD_PCT: float = 50.0


class InMemoryWeatherClient:
    """Generates synthetic hourly :class:`WeatherReading` aggregates."""

    def fetch_weather(
        self,
        *,
        zone: BiddingZone,
        start: datetime,
        end: datetime,
    ) -> Iterable[WeatherReading]:
        temp_baseline = _ZONE_TEMP_BASELINE_C[zone]
        readings: list[WeatherReading] = []
        cursor = _floor_to_hour(start)
        if cursor < start:
            cursor += timedelta(hours=1)
        while cursor < end:
            hour = cursor.hour
            # Daily temperature: warmest mid-afternoon, coolest before dawn.
            # The phase shift puts the maximum at hour 15 instead of 6.
            temp = temp_baseline + _DIURNAL_TEMP_AMPLITUDE_C * sin(2.0 * pi * (hour - 9) / 24.0)
            # Solar irradiance — clipped to zero between dusk and dawn.
            solar = _GHI_PEAK_WM2 * max(0.0, sin(pi * hour / 24.0))
            # A gentle daily wave on cloud cover, never out of [0, 100].
            cloud = max(0.0, min(100.0, _BASE_CLOUD_PCT + 20.0 * cos(2.0 * pi * hour / 24.0)))
            readings.append(
                WeatherReading(
                    zone=zone,
                    timestamp_utc=cursor,
                    temp_c=round(temp, 2),
                    wind_10m_ms=_BASE_WIND_10M_MS,
                    wind_100m_ms=_BASE_WIND_100M_MS,
                    ghi_wm2=round(solar, 2),
                    cloud_cover_pct=round(cloud, 2),
                    precip_mm=0.0,
                )
            )
            cursor += timedelta(hours=1)
        return readings


def _floor_to_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)
