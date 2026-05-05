"""A measured electrical load at a point in time, for a bidding zone."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW


@dataclass(frozen=True, slots=True)
class LoadObservation:
    """A single observed electrical load.

    Identity is the (zone, timestamp_utc) pair: two observations with the
    same identity are considered the same observation regardless of which
    upstream source provided them. This is how downstream pipelines
    deduplicate readings from primary and replay feeds.

    All timestamps must be timezone-aware and expressed in UTC. Naive
    datetimes are the single largest source of bugs in time-series ML, so
    they are rejected at this boundary instead of being silently coerced.
    """

    zone: BiddingZone
    timestamp_utc: datetime
    load: EnergyMW

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError(
                "LoadObservation.timestamp_utc must be timezone-aware, got a naive datetime"
            )
        if self.timestamp_utc.utcoffset() != timedelta(0):
            raise ValueError(
                f"LoadObservation.timestamp_utc must be UTC (offset 0), "
                f"got offset {self.timestamp_utc.utcoffset()}"
            )
