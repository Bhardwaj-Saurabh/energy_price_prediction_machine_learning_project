"""A predicted wholesale electricity price for a future hourly delivery slot."""

from dataclasses import dataclass
from datetime import datetime

from energy_forecaster.domain._validation import require_utc
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.domain.value_objects.price import PriceEUR


@dataclass(frozen=True, slots=True)
class PriceForecast:
    """A point forecast of wholesale electricity price for a single delivery hour.

    The shape mirrors :class:`LoadForecast` deliberately — both are produced
    by the same training and serving infrastructure, only the target
    differs. See ``LoadForecast`` for the rationale behind storing both
    ``as_of_time`` and ``delivery_time`` and tagging each forecast with a
    ``model_version``.
    """

    zone: BiddingZone
    as_of_time: datetime
    delivery_time: datetime
    predicted_price: PriceEUR
    model_version: ModelVersion

    def __post_init__(self) -> None:
        require_utc("PriceForecast.as_of_time", self.as_of_time)
        require_utc("PriceForecast.delivery_time", self.delivery_time)
        if (
            self.delivery_time.minute != 0
            or self.delivery_time.second != 0
            or self.delivery_time.microsecond != 0
        ):
            raise ValueError(
                f"PriceForecast.delivery_time must be aligned to the hour, "
                f"got {self.delivery_time.isoformat()}"
            )
        if self.delivery_time <= self.as_of_time:
            raise ValueError(
                f"PriceForecast.delivery_time {self.delivery_time.isoformat()} "
                f"must be after as_of_time {self.as_of_time.isoformat()}"
            )
