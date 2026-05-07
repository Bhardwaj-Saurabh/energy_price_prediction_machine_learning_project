"""Composition root — the *only* place that wires concrete adapters to ports.

Every other module sees only Protocol-typed interfaces and receives its
dependencies through its constructor. This module reads :class:`Settings`,
chooses concrete adapters per environment, and returns ready-to-call use
case instances to the framework layer (CLI, FastAPI app, Prefect flow,
…). Anything in the codebase that imports a concrete adapter *and* a use
case must live here — and only here.

Branching policy:
  * **ENTSO-E.** ``entsoe_api_key`` unset → InMemoryEntsoeClient
    (synthetic). Set → EntsoePyClient (real, HTTP). The key's presence
    is the discriminator because no key means no real call is possible.
  * **Weather.** ``weather_source == "synthetic"`` → InMemoryWeatherClient.
    ``weather_source == "open_meteo"`` → OpenMeteoClient. Open-Meteo is
    keyless so the discriminator is an explicit setting rather than
    credential presence.

The build functions take an injected ``logger`` so the framework layer
(CLI, FastAPI) can establish a request-scoped bound logger — typically
with a ``correlation_id`` — and have it flow into every use case.
"""

from energy_forecaster.adapters.clock.system_clock import SystemClock
from energy_forecaster.adapters.entsoe_client.entsoe_py import EntsoePyClient
from energy_forecaster.adapters.entsoe_client.in_memory import InMemoryEntsoeClient
from energy_forecaster.adapters.load_observation_repo.local_fs import (
    LocalFsLoadObservationRepository,
)
from energy_forecaster.adapters.weather_client.in_memory import InMemoryWeatherClient
from energy_forecaster.adapters.weather_client.open_meteo import OpenMeteoClient
from energy_forecaster.adapters.weather_reading_repo.local_fs import (
    LocalFsWeatherReadingRepository,
)
from energy_forecaster.application.ports.entsoe_client import EntsoeClient
from energy_forecaster.application.ports.logger import Logger
from energy_forecaster.application.ports.weather_client import WeatherClient
from energy_forecaster.application.use_cases.ingest_entsoe_load import (
    IngestEntsoeLoad,
)
from energy_forecaster.application.use_cases.ingest_weather import IngestWeather
from energy_forecaster.config.settings import Settings


def build_ingest_entsoe_load(settings: Settings, *, logger: Logger) -> IngestEntsoeLoad:
    """Wire :class:`IngestEntsoeLoad` for the given environment."""
    entsoe: EntsoeClient
    if settings.entsoe_api_key is None:
        entsoe = InMemoryEntsoeClient()
    else:
        entsoe = EntsoePyClient(api_key=settings.entsoe_api_key.get_secret_value())

    return IngestEntsoeLoad(
        entsoe=entsoe,
        repo=LocalFsLoadObservationRepository(root=settings.local_data_root),
        clock=SystemClock(),
        logger=logger,
    )


def build_ingest_weather(settings: Settings, *, logger: Logger) -> IngestWeather:
    """Wire :class:`IngestWeather` for the given environment."""
    weather: WeatherClient
    if settings.weather_source == "synthetic":
        weather = InMemoryWeatherClient()
    else:
        weather = OpenMeteoClient()

    return IngestWeather(
        weather=weather,
        repo=LocalFsWeatherReadingRepository(root=settings.local_data_root),
        clock=SystemClock(),
        logger=logger,
    )
