"""In-memory fakes for the application ports.

These are not mocks — they are real implementations of the same Protocol
that production adapters implement, just backed by in-process state. The
behavioural contract enforced here (UTC ordering, deduplication on
identity, error propagation) matches what the Postgres / ENTSO-E adapters
must enforce in production. Testing the use case against these fakes is
testing it against the *contract*, not against a recorded sequence of
calls.
"""

from collections.abc import Iterable
from datetime import datetime, timedelta

from energy_forecaster.application.errors import DataSourceUnavailableError
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


class FakeClock:
    """Controllable clock for deterministic tests.

    Use ``advance(delta)`` between operations when a test needs the
    started_at and finished_at fields of a result to differ — calling
    ``now()`` does not auto-advance.
    """

    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


class FakeEntsoeClient:
    """Predetermined-data ENTSO-E stand-in.

    ``seed`` loads observations into the fake before a test runs; the
    fake then returns the subset whose timestamp falls in the requested
    window. ``fail_on`` flips a single zone into raising
    ``DataSourceUnavailableError`` so the use case's failure path can be
    exercised without resorting to mock side-effects.
    """

    def __init__(self) -> None:
        self._data: dict[BiddingZone, list[LoadObservation]] = {}
        self._fail_on_zone: BiddingZone | None = None

    def seed(self, zone: BiddingZone, observations: Iterable[LoadObservation]) -> None:
        self._data[zone] = list(observations)

    def fail_on(self, zone: BiddingZone) -> None:
        self._fail_on_zone = zone

    def fetch_load(
        self,
        *,
        zone: BiddingZone,
        start: datetime,
        end: datetime,
    ) -> Iterable[LoadObservation]:
        if zone == self._fail_on_zone:
            raise DataSourceUnavailableError(f"ENTSO-E unavailable for {zone}")
        return [obs for obs in self._data.get(zone, []) if start <= obs.timestamp_utc < end]


class FakeLoadObservationRepository:
    """In-memory repo with the same dedup contract as the real Postgres adapter.

    Stores observations keyed by (zone, timestamp_utc) — the composite
    primary key the production schema uses. ``add_many`` returns the
    number of *new* rows so the use case's "observations_inserted" count
    matches what Postgres' ``ON CONFLICT DO NOTHING`` would return.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[BiddingZone, datetime], LoadObservation] = {}

    def add_many(self, observations: Iterable[LoadObservation]) -> int:
        new_count = 0
        for obs in observations:
            key = (obs.zone, obs.timestamp_utc)
            if key not in self._store:
                self._store[key] = obs
                new_count += 1
        return new_count

    def all(self) -> list[LoadObservation]:
        """Test-only helper: dump every stored observation for assertions."""
        return list(self._store.values())
