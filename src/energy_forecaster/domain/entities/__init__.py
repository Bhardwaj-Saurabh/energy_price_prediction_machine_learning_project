"""Domain entities — things that exist with identity, time, and state."""

from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.entities.weather_reading import WeatherReading

__all__ = ["LoadObservation", "WeatherReading"]
