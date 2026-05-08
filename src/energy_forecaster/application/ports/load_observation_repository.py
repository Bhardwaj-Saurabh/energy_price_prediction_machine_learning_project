"""LoadObservationRepository port — persistence for LoadObservation aggregates."""

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


class LoadObservationRepository(Protocol):
    """Persists and retrieves :class:`LoadObservation` aggregates.

    Identity is the (zone, timestamp_utc) pair. Implementations MUST
    enforce uniqueness on this composite key — re-ingesting the same
    window must produce zero new rows, not duplicates. The Postgres
    adapter does this with ``ON CONFLICT DO NOTHING``; the in-memory test
    double does it with a dict keyed by (zone, ts).
    """

    def add_many(self, observations: Iterable[LoadObservation]) -> int:
        """Insert observations, deduplicated by (zone, timestamp_utc).

        Returns the number of rows newly inserted (i.e. excluding those
        skipped by the dedup constraint). Idempotent — calling twice with
        the same input is safe and the second call returns 0.
        """
        ...

    def find_by_zone(
        self,
        zone: BiddingZone,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[LoadObservation]:
        """Return observations for ``zone`` ordered by ``timestamp_utc`` ascending.

        ``since`` is inclusive on the lower bound, ``until`` is exclusive
        on the upper — same convention as the LoadForecastRepository.
        Returns ``[]`` when the zone has no recorded observations; not
        an error condition.
        """
        ...
