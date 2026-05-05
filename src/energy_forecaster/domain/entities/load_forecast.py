"""A predicted electrical load for a future hourly delivery slot."""

from dataclasses import dataclass
from datetime import datetime

from energy_forecaster.domain._validation import require_utc
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.model_version import ModelVersion


@dataclass(frozen=True, slots=True)
class LoadForecast:
    """A point forecast of electrical load for a single delivery hour.

    Two timestamps are mandatory:
      - ``as_of_time``    — when the prediction was made; the cutoff before
                            which all input features must already be known.
                            Storing this is what lets us audit time-leakage
                            and replay historical predictions.
      - ``delivery_time`` — the hour the prediction is *for*. Day-ahead
                            markets clear hourly, so we enforce alignment
                            on the hour at this boundary; non-aligned
                            timestamps almost certainly indicate a bug.

    ``model_version`` is carried on every forecast so monitoring, rollback,
    and lineage tracing can answer 'which model produced this?' from the
    forecast row itself, with no out-of-band lookup.
    """

    zone: BiddingZone
    as_of_time: datetime
    delivery_time: datetime
    predicted_load: EnergyMW
    model_version: ModelVersion

    def __post_init__(self) -> None:
        require_utc("LoadForecast.as_of_time", self.as_of_time)
        require_utc("LoadForecast.delivery_time", self.delivery_time)
        if (
            self.delivery_time.minute != 0
            or self.delivery_time.second != 0
            or self.delivery_time.microsecond != 0
        ):
            raise ValueError(
                f"LoadForecast.delivery_time must be aligned to the hour, "
                f"got {self.delivery_time.isoformat()}"
            )
        if self.delivery_time <= self.as_of_time:
            raise ValueError(
                f"LoadForecast.delivery_time {self.delivery_time.isoformat()} "
                f"must be after as_of_time {self.as_of_time.isoformat()}"
            )
