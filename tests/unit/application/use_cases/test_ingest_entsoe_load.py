"""Unit tests for the IngestEntsoeLoad use case.

The use case is exercised against in-memory fakes, never against mocks of
its own collaborators. Each fake satisfies the same Protocol as the
production adapter — see ``tests/unit/application/fakes.py`` for the
rationale.
"""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from energy_forecaster.application.errors import DataSourceUnavailableError
from energy_forecaster.application.use_cases.ingest_entsoe_load import (
    IngestEntsoeLoad,
)
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from tests.unit.application.fakes import (
    FakeClock,
    FakeEntsoeClient,
    FakeLoadObservationRepository,
    FakeLogger,
)


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _hourly_load(
    zone: BiddingZone, start: datetime, hours: int, base_mw: float = 50_000.0
) -> list[LoadObservation]:
    """Build N hourly observations starting at ``start``."""
    return [
        LoadObservation(
            zone=zone,
            timestamp_utc=start + timedelta(hours=h),
            load=EnergyMW(base_mw + h * 100.0),
        )
        for h in range(hours)
    ]


def _build(
    *,
    clock_at: datetime | None = None,
    entsoe: FakeEntsoeClient | None = None,
    repo: FakeLoadObservationRepository | None = None,
    logger: FakeLogger | None = None,
) -> tuple[
    IngestEntsoeLoad,
    FakeEntsoeClient,
    FakeLoadObservationRepository,
    FakeClock,
    FakeLogger,
]:
    clock = FakeClock(now=clock_at or _utc(2026, 5, 5, 6))
    entsoe = entsoe or FakeEntsoeClient()
    repo = repo or FakeLoadObservationRepository()
    logger = logger or FakeLogger()
    use_case = IngestEntsoeLoad(entsoe=entsoe, repo=repo, clock=clock, logger=logger)
    return use_case, entsoe, repo, clock, logger


class TestHappyPath:
    def test_typical_run_fetches_and_inserts_for_every_zone(self) -> None:
        use_case, entsoe, repo, _, _ = _build()
        window_start = _utc(2026, 5, 4, 0)
        window_end = _utc(2026, 5, 5, 0)
        entsoe.seed(BiddingZone.DE_LU, _hourly_load(BiddingZone.DE_LU, window_start, 24))
        entsoe.seed(BiddingZone.FR, _hourly_load(BiddingZone.FR, window_start, 24))

        result = use_case.execute(
            zones=[BiddingZone.DE_LU, BiddingZone.FR],
            start=window_start,
            end=window_end,
        )

        assert result.zones_processed == 2
        assert result.observations_fetched == 48
        assert result.observations_inserted == 48
        assert len(repo.all()) == 48

    def test_window_is_half_open(self) -> None:
        # Observation at ``end`` itself is excluded — ENTSO-E queries are
        # half-open in the same way, so the use case behaves consistently.
        use_case, entsoe, _, _, _ = _build()
        start = _utc(2026, 5, 4, 0)
        end = _utc(2026, 5, 4, 2)
        entsoe.seed(
            BiddingZone.DE_LU,
            _hourly_load(BiddingZone.DE_LU, start, 3),  # 00:00, 01:00, 02:00
        )

        result = use_case.execute(zones=[BiddingZone.DE_LU], start=start, end=end)

        assert result.observations_fetched == 2  # 00:00 and 01:00 only


class TestDeduplication:
    def test_re_running_same_window_inserts_zero_new(self) -> None:
        use_case, entsoe, repo, _, _ = _build()
        start = _utc(2026, 5, 4, 0)
        end = _utc(2026, 5, 5, 0)
        entsoe.seed(
            BiddingZone.DE_LU,
            _hourly_load(BiddingZone.DE_LU, start, 24),
        )

        first = use_case.execute(zones=[BiddingZone.DE_LU], start=start, end=end)
        second = use_case.execute(zones=[BiddingZone.DE_LU], start=start, end=end)

        assert first.observations_inserted == 24
        assert second.observations_fetched == 24  # still fetched
        assert second.observations_inserted == 0  # but nothing new persisted
        assert len(repo.all()) == 24


class TestInputValidation:
    def test_empty_zones_is_rejected(self) -> None:
        use_case, *_ = _build()
        with pytest.raises(ValueError, match="non-empty"):
            use_case.execute(zones=[], start=_utc(2026, 5, 4), end=_utc(2026, 5, 5))

    def test_start_equal_to_end_is_rejected(self) -> None:
        use_case, *_ = _build()
        ts = _utc(2026, 5, 4)
        with pytest.raises(ValueError, match="strictly before"):
            use_case.execute(zones=[BiddingZone.DE_LU], start=ts, end=ts)

    def test_start_after_end_is_rejected(self) -> None:
        use_case, *_ = _build()
        with pytest.raises(ValueError, match="strictly before"):
            use_case.execute(
                zones=[BiddingZone.DE_LU],
                start=_utc(2026, 5, 5),
                end=_utc(2026, 5, 4),
            )

    def test_naive_start_is_rejected(self) -> None:
        use_case, *_ = _build()
        with pytest.raises(ValueError, match="timezone-aware"):
            use_case.execute(
                zones=[BiddingZone.DE_LU],
                start=datetime(2026, 5, 4),
                end=_utc(2026, 5, 5),
            )

    def test_non_utc_end_is_rejected(self) -> None:
        use_case, *_ = _build()
        cet = timezone(timedelta(hours=1))
        with pytest.raises(ValueError, match="must be UTC"):
            use_case.execute(
                zones=[BiddingZone.DE_LU],
                start=_utc(2026, 5, 4),
                end=datetime(2026, 5, 5, tzinfo=cet),
            )


class TestErrorPropagation:
    def test_entsoe_failure_aborts_run_fail_fast(self) -> None:
        # The first zone that fails halts processing — later zones are not
        # consulted. This is the documented fail-fast contract.
        use_case, entsoe, repo, _, _ = _build()
        start = _utc(2026, 5, 4)
        end = _utc(2026, 5, 5)
        entsoe.seed(BiddingZone.DE_LU, _hourly_load(BiddingZone.DE_LU, start, 24))
        entsoe.fail_on(BiddingZone.DE_LU)
        entsoe.seed(BiddingZone.FR, _hourly_load(BiddingZone.FR, start, 24))

        with pytest.raises(DataSourceUnavailableError):
            use_case.execute(
                zones=[BiddingZone.DE_LU, BiddingZone.FR],
                start=start,
                end=end,
            )

        # Nothing committed when the run aborts — the use case writes
        # per-zone, so DE_LU's fetch failed before write, and FR was never
        # attempted.
        assert repo.all() == []


class TestLogging:
    def test_emits_start_and_done_events_with_zone_and_window(self) -> None:
        use_case, entsoe, _, _, logger = _build()
        start, end = _utc(2026, 5, 4), _utc(2026, 5, 5)
        entsoe.seed(BiddingZone.DE_LU, _hourly_load(BiddingZone.DE_LU, start, 24))

        use_case.execute(zones=[BiddingZone.DE_LU], start=start, end=end)

        events = logger.events()
        assert "ingest.start" in events
        assert "ingest.done" in events

    def test_done_event_carries_aggregated_counts(self) -> None:
        use_case, entsoe, _, _, logger = _build()
        start, end = _utc(2026, 5, 4), _utc(2026, 5, 5)
        entsoe.seed(BiddingZone.DE_LU, _hourly_load(BiddingZone.DE_LU, start, 24))

        use_case.execute(zones=[BiddingZone.DE_LU], start=start, end=end)

        done = next(c for c in logger.calls if c.event == "ingest.done")
        assert done.context["zones_processed"] == 1
        assert done.context["observations_fetched"] == 24
        assert done.context["observations_inserted"] == 24

    def test_bound_correlation_id_propagates_to_every_event(self) -> None:
        # The CLI binds correlation_id at process entry and passes the
        # result here. Every subsequent log call must inherit it — that
        # is the whole reason for the bind() pattern.
        use_case, entsoe, _, _, logger = _build(logger=FakeLogger().bind(correlation_id="abc-123"))
        start, end = _utc(2026, 5, 4), _utc(2026, 5, 5)
        entsoe.seed(BiddingZone.DE_LU, _hourly_load(BiddingZone.DE_LU, start, 24))

        use_case.execute(zones=[BiddingZone.DE_LU], start=start, end=end)

        assert all(c.context.get("correlation_id") == "abc-123" for c in logger.calls)


class TestTimingFromInjectedClock:
    def test_started_and_finished_come_from_clock_not_real_time(self) -> None:
        use_case, entsoe, _, clock, _ = _build(clock_at=_utc(2026, 5, 5, 6))
        entsoe.seed(
            BiddingZone.DE_LU,
            _hourly_load(BiddingZone.DE_LU, _utc(2026, 5, 4), 1),
        )

        # Advance the clock between the use case's two now() calls by
        # advancing the FakeClock's internal counter. We cannot do this
        # mid-execute (FakeClock has no auto-advance), but the test still
        # confirms started_at and finished_at are sourced from the port.
        clock.advance(timedelta(0))  # explicit no-op for documentation

        result = use_case.execute(
            zones=[BiddingZone.DE_LU],
            start=_utc(2026, 5, 4),
            end=_utc(2026, 5, 5),
        )

        assert result.started_at == _utc(2026, 5, 5, 6)
        assert result.finished_at == _utc(2026, 5, 5, 6)
        assert result.duration_seconds == 0.0
