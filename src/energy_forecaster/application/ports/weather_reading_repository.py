"""WeatherReadingRepository port — persistence for WeatherReading aggregates."""

from collections.abc import Iterable
from typing import Protocol

from energy_forecaster.domain.entities.weather_reading import WeatherReading


class WeatherReadingRepository(Protocol):
    """Persists and retrieves :class:`WeatherReading` aggregates.

    Identity is the (zone, timestamp_utc) pair, mirroring the load
    repository. Implementations MUST enforce uniqueness on this composite
    key so re-ingesting a window produces zero new rows, not duplicates.
    """

    def add_many(self, readings: Iterable[WeatherReading]) -> int:
        """Insert readings, deduplicated by (zone, timestamp_utc).

        Returns the number of newly inserted rows. Idempotent.
        """
        ...
