"""Value objects — immutable, validated wrappers around primitive measurements."""

from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.horizon import HorizonHours
from energy_forecaster.domain.value_objects.mape import MAPE
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.domain.value_objects.price import PriceEUR

__all__ = [
    "MAPE",
    "BiddingZone",
    "EnergyMW",
    "HorizonHours",
    "ModelVersion",
    "PriceEUR",
]
