"""Use case: ingest hourly weather observations for one or more bidding zones."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from energy_forecaster.application.ports.clock import Clock
from energy_forecaster.application.ports.logger import Logger
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
    write, clock, logger), same fail-fast semantics on the first adapter
    error, same UTC-only window invariants.
    """

    def __init__(
        self,
        *,
        weather: WeatherClient,
        repo: WeatherReadingRepository,
        clock: Clock,
        logger: Logger,
    ) -> None:
        self._weather = weather
        self._repo = repo
        self._clock = clock
        self._logger = logger

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

        log = self._logger.bind(operation="ingest_weather")
        log.info(
            "weather.start",
            zones=[z.value for z in zones],
            start=start.isoformat(),
            end=end.isoformat(),
        )

        started_at = self._clock.now()
        total_fetched = 0
        total_inserted = 0

        for zone in zones:
            zone_log = log.bind(zone=zone.value)
            readings = list(self._weather.fetch_weather(zone=zone, start=start, end=end))
            inserted = self._repo.add_many(readings)
            total_fetched += len(readings)
            total_inserted += inserted
            zone_log.debug(
                "weather.zone.done",
                fetched=len(readings),
                inserted=inserted,
            )

        finished_at = self._clock.now()

        log.info(
            "weather.done",
            zones_processed=len(zones),
            readings_fetched=total_fetched,
            readings_inserted=total_inserted,
            duration_seconds=round((finished_at - started_at).total_seconds(), 3),
        )

        return IngestWeatherResult(
            zones_processed=len(zones),
            readings_fetched=total_fetched,
            readings_inserted=total_inserted,
            started_at=started_at,
            finished_at=finished_at,
        )
