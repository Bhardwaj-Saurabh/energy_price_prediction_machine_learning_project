"""LoadObservationRepository port — persistence for LoadObservation aggregates."""

from collections.abc import Iterable
from typing import Protocol

from energy_forecaster.domain.entities.load_observation import LoadObservation


class LoadObservationRepository(Protocol):
    """Persists and retrieves :class:`LoadObservation` aggregates.

    Identity is the (zone, timestamp_utc) pair. Implementations MUST
    enforce uniqueness on this composite key — re-ingesting the same
    window must produce zero new rows, not duplicates. The Postgres
    adapter does this with ``ON CONFLICT DO NOTHING``; the in-memory test
    double does it with a dict keyed by (zone, ts).

    For now the port exposes only a write path; query methods are added
    when a use case needs them, not preemptively.
    """

    def add_many(self, observations: Iterable[LoadObservation]) -> int:
        """Insert observations, deduplicated by (zone, timestamp_utc).

        Returns the number of rows newly inserted (i.e. excluding those
        skipped by the dedup constraint). Idempotent — calling twice with
        the same input is safe and the second call returns 0.
        """
        ...
