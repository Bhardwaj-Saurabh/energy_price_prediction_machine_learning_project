"""Use case: ingest day-ahead load observations from ENTSO-E."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from energy_forecaster.application.ports.clock import Clock
from energy_forecaster.application.ports.entsoe_client import EntsoeClient
from energy_forecaster.application.ports.load_observation_repository import (
    LoadObservationRepository,
)
from energy_forecaster.domain import require_utc
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


@dataclass(frozen=True, slots=True)
class IngestEntsoeLoadResult:
    """Summary returned by :meth:`IngestEntsoeLoad.execute`.

    The counts are the visible signal of what the run did; the timestamps
    are the audit trail of when it happened. Both come from the injected
    Clock so the result is fully reproducible in tests.
    """

    zones_processed: int
    observations_fetched: int
    observations_inserted: int
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


class IngestEntsoeLoad:
    """Fetch load observations from ENTSO-E for one or more bidding zones
    and persist them, deduplicated by (zone, timestamp_utc).

    Dependencies (all ports — concrete types are wired in by the
    composition root, never imported from here):
      * ``entsoe`` — read side; returns validated LoadObservations.
      * ``repo``   — write side; idempotent add_many.
      * ``clock``  — wall clock for the started_at / finished_at audit
                     fields. Injected so tests are deterministic.

    Failure mode: fail-fast. The first ``DataSourceUnavailable`` raised
    by the ENTSO-E adapter aborts the run; later zones are not attempted.
    A "best-effort, partial result" mode is a deliberate later
    decision — adding it without explicit demand is premature.
    """

    def __init__(
        self,
        *,
        entsoe: EntsoeClient,
        repo: LoadObservationRepository,
        clock: Clock,
    ) -> None:
        self._entsoe = entsoe
        self._repo = repo
        self._clock = clock

    def execute(
        self,
        *,
        zones: Sequence[BiddingZone],
        start: datetime,
        end: datetime,
    ) -> IngestEntsoeLoadResult:
        if not zones:
            raise ValueError("zones must be non-empty")
        require_utc("IngestEntsoeLoad.start", start)
        require_utc("IngestEntsoeLoad.end", end)
        if start >= end:
            raise ValueError(
                f"start {start.isoformat()} must be strictly before end {end.isoformat()}"
            )

        started_at = self._clock.now()
        total_fetched = 0
        total_inserted = 0

        for zone in zones:
            observations = list(self._entsoe.fetch_load(zone=zone, start=start, end=end))
            total_fetched += len(observations)
            total_inserted += self._repo.add_many(observations)

        finished_at = self._clock.now()

        return IngestEntsoeLoadResult(
            zones_processed=len(zones),
            observations_fetched=total_fetched,
            observations_inserted=total_inserted,
            started_at=started_at,
            finished_at=finished_at,
        )
