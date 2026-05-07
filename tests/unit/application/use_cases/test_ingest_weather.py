"""Unit tests for the IngestWeather use case.

Mirrors the IngestEntsoeLoad test suite — same fakes, same patterns,
same assertions adapted for weather entities. The repetition is the
point: identical pipelines should look identical in their tests.
"""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from energy_forecaster.application.errors import DataSourceUnavailableError
from energy_forecaster.application.use_cases.ingest_weather import IngestWeather
from energy_forecaster.domain.entities.weather_reading import WeatherReading
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from tests.unit.application.fakes import (
    FakeClock,
    FakeLogger,
    FakeWeatherClient,
    FakeWeatherReadingRepository,
)


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _hourly_weather(zone: BiddingZone, start: datetime, hours: int) -> list[WeatherReading]:
    return [
        WeatherReading(
            zone=zone,
            timestamp_utc=start + timedelta(hours=h),
            temp_c=15.0,
            wind_10m_ms=4.0,
            wind_100m_ms=8.0,
            ghi_wm2=300.0,
            cloud_cover_pct=50.0,
            precip_mm=0.0,
        )
        for h in range(hours)
    ]


def _build(
    *,
    clock_at: datetime | None = None,
    logger: FakeLogger | None = None,
) -> tuple[
    IngestWeather,
    FakeWeatherClient,
    FakeWeatherReadingRepository,
    FakeClock,
    FakeLogger,
]:
    clock = FakeClock(now=clock_at or _utc(2026, 5, 5, 6))
    weather = FakeWeatherClient()
    repo = FakeWeatherReadingRepository()
    logger = logger or FakeLogger()
    use_case = IngestWeather(weather=weather, repo=repo, clock=clock, logger=logger)
    return use_case, weather, repo, clock, logger


class TestHappyPath:
    def test_typical_run_fetches_and_inserts_for_every_zone(self) -> None:
        use_case, weather, repo, _, _ = _build()
        window_start = _utc(2026, 5, 4, 0)
        window_end = _utc(2026, 5, 5, 0)
        weather.seed(BiddingZone.DE_LU, _hourly_weather(BiddingZone.DE_LU, window_start, 24))
        weather.seed(BiddingZone.FR, _hourly_weather(BiddingZone.FR, window_start, 24))

        result = use_case.execute(
            zones=[BiddingZone.DE_LU, BiddingZone.FR],
            start=window_start,
            end=window_end,
        )

        assert result.zones_processed == 2
        assert result.readings_fetched == 48
        assert result.readings_inserted == 48
        assert len(repo.all()) == 48


class TestDeduplication:
    def test_re_running_same_window_inserts_zero_new(self) -> None:
        use_case, weather, repo, _, _ = _build()
        start, end = _utc(2026, 5, 4), _utc(2026, 5, 5)
        weather.seed(BiddingZone.DE_LU, _hourly_weather(BiddingZone.DE_LU, start, 24))

        first = use_case.execute(zones=[BiddingZone.DE_LU], start=start, end=end)
        second = use_case.execute(zones=[BiddingZone.DE_LU], start=start, end=end)

        assert first.readings_inserted == 24
        assert second.readings_fetched == 24
        assert second.readings_inserted == 0
        assert len(repo.all()) == 24


class TestInputValidation:
    def test_empty_zones_rejected(self) -> None:
        use_case, *_ = _build()
        with pytest.raises(ValueError, match="non-empty"):
            use_case.execute(zones=[], start=_utc(2026, 5, 4), end=_utc(2026, 5, 5))

    def test_start_not_before_end_rejected(self) -> None:
        use_case, *_ = _build()
        ts = _utc(2026, 5, 4)
        with pytest.raises(ValueError, match="strictly before"):
            use_case.execute(zones=[BiddingZone.DE_LU], start=ts, end=ts)

    def test_naive_start_rejected(self) -> None:
        use_case, *_ = _build()
        with pytest.raises(ValueError, match="timezone-aware"):
            use_case.execute(
                zones=[BiddingZone.DE_LU],
                start=datetime(2026, 5, 4),
                end=_utc(2026, 5, 5),
            )

    def test_non_utc_end_rejected(self) -> None:
        use_case, *_ = _build()
        cet = timezone(timedelta(hours=1))
        with pytest.raises(ValueError, match="must be UTC"):
            use_case.execute(
                zones=[BiddingZone.DE_LU],
                start=_utc(2026, 5, 4),
                end=datetime(2026, 5, 5, tzinfo=cet),
            )


class TestErrorPropagation:
    def test_weather_failure_aborts_run_fail_fast(self) -> None:
        use_case, weather, repo, _, _ = _build()
        start, end = _utc(2026, 5, 4), _utc(2026, 5, 5)
        weather.seed(BiddingZone.DE_LU, _hourly_weather(BiddingZone.DE_LU, start, 24))
        weather.fail_on(BiddingZone.DE_LU)
        weather.seed(BiddingZone.FR, _hourly_weather(BiddingZone.FR, start, 24))

        with pytest.raises(DataSourceUnavailableError):
            use_case.execute(zones=[BiddingZone.DE_LU, BiddingZone.FR], start=start, end=end)
        assert repo.all() == []


class TestLogging:
    def test_emits_start_and_done_events(self) -> None:
        use_case, weather, _, _, logger = _build()
        start, end = _utc(2026, 5, 4), _utc(2026, 5, 5)
        weather.seed(BiddingZone.DE_LU, _hourly_weather(BiddingZone.DE_LU, start, 24))

        use_case.execute(zones=[BiddingZone.DE_LU], start=start, end=end)

        assert {"weather.start", "weather.done"}.issubset(set(logger.events()))

    def test_correlation_id_is_inherited_from_bound_logger(self) -> None:
        use_case, weather, _, _, logger = _build(logger=FakeLogger().bind(correlation_id="xyz-789"))
        start, end = _utc(2026, 5, 4), _utc(2026, 5, 5)
        weather.seed(BiddingZone.DE_LU, _hourly_weather(BiddingZone.DE_LU, start, 24))

        use_case.execute(zones=[BiddingZone.DE_LU], start=start, end=end)

        assert all(c.context.get("correlation_id") == "xyz-789" for c in logger.calls)
