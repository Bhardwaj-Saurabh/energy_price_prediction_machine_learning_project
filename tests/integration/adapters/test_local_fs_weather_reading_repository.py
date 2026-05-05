"""Integration tests for LocalFsWeatherReadingRepository."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from energy_forecaster.adapters.weather_reading_repo.local_fs import (
    LocalFsWeatherReadingRepository,
    _deserialise,
)
from energy_forecaster.application.ports.weather_reading_repository import (
    WeatherReadingRepository,
)
from energy_forecaster.domain.entities.weather_reading import WeatherReading
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone

pytestmark = pytest.mark.integration


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _hourly_weather(zone: BiddingZone, start: datetime, hours: int) -> list[WeatherReading]:
    return [
        WeatherReading(
            zone=zone,
            timestamp_utc=start + timedelta(hours=h),
            temp_c=15.0 + h,
            wind_10m_ms=4.0,
            wind_100m_ms=8.0,
            ghi_wm2=300.0,
            cloud_cover_pct=50.0,
            precip_mm=0.0,
        )
        for h in range(hours)
    ]


def _read_file(path: Path) -> list[WeatherReading]:
    return [_deserialise(line) for line in path.read_text("utf-8").splitlines() if line]


class TestConstructorBehaviour:
    def test_subdir_is_created_eagerly(self, tmp_path: Path) -> None:
        LocalFsWeatherReadingRepository(root=tmp_path)
        assert (tmp_path / "weather_readings").is_dir()

    def test_satisfies_repository_protocol_structurally(self, tmp_path: Path) -> None:
        repo: WeatherReadingRepository = LocalFsWeatherReadingRepository(root=tmp_path)
        assert repo.add_many([]) == 0


class TestWriteAndRoundtrip:
    def test_writes_one_line_per_reading(self, tmp_path: Path) -> None:
        repo = LocalFsWeatherReadingRepository(root=tmp_path)
        readings = _hourly_weather(BiddingZone.DE_LU, _utc(2026, 5, 4), 3)

        inserted = repo.add_many(readings)

        assert inserted == 3
        path = tmp_path / "weather_readings" / "DE_LU.jsonl"
        assert _read_file(path) == readings

    def test_each_zone_gets_its_own_file(self, tmp_path: Path) -> None:
        repo = LocalFsWeatherReadingRepository(root=tmp_path)
        de = _hourly_weather(BiddingZone.DE_LU, _utc(2026, 5, 4), 2)
        fr = _hourly_weather(BiddingZone.FR, _utc(2026, 5, 4), 3)
        repo.add_many([*de, *fr])

        assert {p.name for p in (tmp_path / "weather_readings").iterdir()} == {
            "DE_LU.jsonl",
            "FR.jsonl",
        }


class TestDeduplication:
    def test_re_writing_same_readings_inserts_zero(self, tmp_path: Path) -> None:
        repo = LocalFsWeatherReadingRepository(root=tmp_path)
        readings = _hourly_weather(BiddingZone.DE_LU, _utc(2026, 5, 4), 3)

        first = repo.add_many(readings)
        second = repo.add_many(readings)

        assert first == 3
        assert second == 0
        assert len(_read_file(tmp_path / "weather_readings" / "DE_LU.jsonl")) == 3

    def test_overlap_inserts_only_new_rows(self, tmp_path: Path) -> None:
        repo = LocalFsWeatherReadingRepository(root=tmp_path)
        first = _hourly_weather(BiddingZone.DE_LU, _utc(2026, 5, 4), 3)
        repo.add_many(first)

        # Hours [2, 3, 4) — overlaps the first by hour 2 only.
        overlap_plus_new = _hourly_weather(BiddingZone.DE_LU, _utc(2026, 5, 4, 2), 3)
        inserted = repo.add_many(overlap_plus_new)

        assert inserted == 2
