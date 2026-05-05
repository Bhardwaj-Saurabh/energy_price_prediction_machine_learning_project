"""A measured electrical load at a point in time, for a bidding zone."""

from dataclasses import dataclass
from datetime import datetime

from energy_forecaster.domain._validation import require_utc
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW


@dataclass(frozen=True, slots=True)
class LoadObservation:
    """A single observed electrical load.

    Identity is the (zone, timestamp_utc) pair: two observations with the
    same identity are considered the same observation regardless of which
    upstream source provided them. This is how downstream pipelines
    deduplicate readings from primary and replay feeds.
    """

    zone: BiddingZone
    timestamp_utc: datetime
    load: EnergyMW

    def __post_init__(self) -> None:
        require_utc("LoadObservation.timestamp_utc", self.timestamp_utc)
