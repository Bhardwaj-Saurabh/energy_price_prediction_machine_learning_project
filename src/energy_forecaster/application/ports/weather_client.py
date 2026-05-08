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

    Two methods, one entity. ``fetch_weather`` returns *observed*
    readings (a measurement happened); ``fetch_forecast`` returns
    *predicted* readings (a forecast model says it will). The data
    shape is the same — same fields, same units — because the consumer
    treats them the same. The distinction is the source, kept explicit
    at the API surface so call sites declare which one they wanted.

    Window is half-open ``[start, end)`` for both methods.
    """

    def fetch_weather(
        self,
        *,
        zone: BiddingZone,
        start: datetime,
        end: datetime,
    ) -> Iterable[WeatherReading]:
        """Return *observed* readings for ``zone`` in ``[start, end)``.

        Both timestamps must be timezone-aware UTC and in the past
        relative to the adapter's data source.
        """
        ...

    def fetch_forecast(
        self,
        *,
        zone: BiddingZone,
        start: datetime,
        end: datetime,
    ) -> Iterable[WeatherReading]:
        """Return *forecasted* readings for ``zone`` in ``[start, end)``.

        Both timestamps must be timezone-aware UTC and in the future
        relative to the adapter's data source. The horizon limit is
        adapter-specific (Open-Meteo: 16 days).
        """
        ...
