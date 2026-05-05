"""Domain entities — things that exist with identity, time, and state."""

from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.entities.price_forecast import PriceForecast
from energy_forecaster.domain.entities.weather_reading import WeatherReading

__all__ = ["LoadForecast", "LoadObservation", "PriceForecast", "WeatherReading"]
