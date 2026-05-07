"""Pandera schemas + entity-to-DataFrame converters.

Anything in the codebase that hands a DataFrame to a downstream consumer
goes through this package — schema validation runs at the boundary so
silent dtype drift or duplicate keys are caught at the producer, not at
the consumer.
"""

from energy_forecaster.contracts.load_observation_schema import (
    LoadObservationSchema,
    to_load_dataframe,
)
from energy_forecaster.contracts.weather_reading_schema import (
    WeatherReadingSchema,
    to_weather_dataframe,
)

__all__ = [
    "LoadObservationSchema",
    "WeatherReadingSchema",
    "to_load_dataframe",
    "to_weather_dataframe",
]
