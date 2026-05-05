"""EntsoeClient port — read-side interface to the ENTSO-E Transparency Platform."""

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


class EntsoeClient(Protocol):
    """Fetches validated domain entities from ENTSO-E.

    The concrete adapter (using ``entsoe-py``) handles EIC code mapping,
    XML parsing, retries, rate-limit pacing, and translating any HTTP /
    transport failure into ``DataSourceUnavailable``. By the time data
    crosses this port boundary it is already in the form of validated
    :class:`LoadObservation` aggregates, so the use case can stay
    framework-agnostic.

    Returning ``Iterable`` (rather than ``list``) lets a future streaming
    adapter avoid materialising large windows in memory. The use case
    iterates exactly once.
    """

    def fetch_load(
        self,
        *,
        zone: BiddingZone,
        start: datetime,
        end: datetime,
    ) -> Iterable[LoadObservation]:
        """Return load observations for ``zone`` in the half-open window
        ``[start, end)``. Both timestamps must be timezone-aware UTC.
        """
        ...
