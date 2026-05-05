"""Domain layer — pure business types, zero infrastructure dependencies.

The dependency rule says nothing in this package may import from
`adapters/`, `pipelines/`, `serving/`, or any third-party framework
(Kedro, MLflow, Feast, FastAPI, Azure SDKs, pandas, numpy, requests).
Domain code expresses *what* the business is; everything else is *how*.
"""

from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.entities.weather_reading import WeatherReading
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.horizon import HorizonHours
from energy_forecaster.domain.value_objects.mape import MAPE
from energy_forecaster.domain.value_objects.price import PriceEUR

__all__ = [
    "MAPE",
    "BiddingZone",
    "EnergyMW",
    "HorizonHours",
    "LoadObservation",
    "PriceEUR",
    "WeatherReading",
]
