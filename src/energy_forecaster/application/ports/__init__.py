"""Application ports — Protocol interfaces for every external dependency.

Anything the application layer reaches outside itself for (the wall clock,
external APIs, persistence, logging) is reached through a port defined
here. The production adapter and the test fake both implement the same
Protocol; the use case sees only the Protocol and never the concrete type.
"""

from energy_forecaster.application.ports.clock import Clock
from energy_forecaster.application.ports.entsoe_client import EntsoeClient
from energy_forecaster.application.ports.load_forecast_repository import (
    LoadForecastRepository,
)
from energy_forecaster.application.ports.load_observation_repository import (
    LoadObservationRepository,
)
from energy_forecaster.application.ports.logger import Logger
from energy_forecaster.application.ports.model_registry import ModelRegistry
from energy_forecaster.application.ports.weather_client import WeatherClient
from energy_forecaster.application.ports.weather_reading_repository import (
    WeatherReadingRepository,
)

__all__ = [
    "Clock",
    "EntsoeClient",
    "LoadForecastRepository",
    "LoadObservationRepository",
    "Logger",
    "ModelRegistry",
    "WeatherClient",
    "WeatherReadingRepository",
]
