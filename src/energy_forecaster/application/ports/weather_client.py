"""WeatherClient port — read-side interface to a weather data source."""

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

from energy_forecaster.domain.entities.weather_reading import WeatherReading
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


class WeatherClient(Protocol):
    """Fetches validated :class:`WeatherReading` aggregates for a zone+window.

    By the time data crosses this boundary it is already in the form of
    domain entities — the adapter resolves any zone-to-coordinates mapping,
    handles unit conversions, retries, and translates transport / parse
    failures into :class:`DataSourceUnavailableError`.

    Window is half-open ``[start, end)`` to match every other ingest port.
    """

    def fetch_weather(
        self,
        *,
        zone: BiddingZone,
        start: datetime,
        end: datetime,
    ) -> Iterable[WeatherReading]:
        """Return readings for ``zone`` in the half-open window ``[start, end)``.
        Both timestamps must be timezone-aware UTC.
        """
        ...
