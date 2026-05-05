"""Use case: ingest hourly weather observations for one or more bidding zones."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from energy_forecaster.application.ports.clock import Clock
from energy_forecaster.application.ports.weather_client import WeatherClient
from energy_forecaster.application.ports.weather_reading_repository import (
    WeatherReadingRepository,
)
from energy_forecaster.domain import require_utc
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


@dataclass(frozen=True, slots=True)
class IngestWeatherResult:
    """Summary returned by :meth:`IngestWeather.execute`."""

    zones_processed: int
    readings_fetched: int
    readings_inserted: int
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


class IngestWeather:
    """Fetch hourly weather observations and persist them, deduplicated.

    Mirrors :class:`IngestEntsoeLoad` in shape: same ports pattern (read,
    write, clock), same fail-fast semantics on the first adapter error,
    same UTC-only window invariants. Differences are isolated to the
    entity types and the names — a deliberate consequence of treating the
    architecture as a template rather than a one-off.
    """

    def __init__(
        self,
        *,
        weather: WeatherClient,
        repo: WeatherReadingRepository,
        clock: Clock,
    ) -> None:
        self._weather = weather
        self._repo = repo
        self._clock = clock

    def execute(
        self,
        *,
        zones: Sequence[BiddingZone],
        start: datetime,
        end: datetime,
    ) -> IngestWeatherResult:
        if not zones:
            raise ValueError("zones must be non-empty")
        require_utc("IngestWeather.start", start)
        require_utc("IngestWeather.end", end)
        if start >= end:
            raise ValueError(
                f"start {start.isoformat()} must be strictly before end {end.isoformat()}"
            )

        started_at = self._clock.now()
        total_fetched = 0
        total_inserted = 0

        for zone in zones:
            readings = list(self._weather.fetch_weather(zone=zone, start=start, end=end))
            total_fetched += len(readings)
            total_inserted += self._repo.add_many(readings)

        finished_at = self._clock.now()

        return IngestWeatherResult(
            zones_processed=len(zones),
            readings_fetched=total_fetched,
            readings_inserted=total_inserted,
            started_at=started_at,
            finished_at=finished_at,
        )
