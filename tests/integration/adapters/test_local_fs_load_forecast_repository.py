"""Integration tests for LocalFsLoadForecastRepository."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from energy_forecaster.adapters.load_forecast_repo.local_fs import (
    LocalFsLoadForecastRepository,
    deserialise,
)
from energy_forecaster.application.ports.load_forecast_repository import (
    LoadForecastRepository,
)
from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.model_version import ModelVersion

pytestmark = pytest.mark.integration


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _make_forecast(
    *,
    zone: BiddingZone = BiddingZone.DE_LU,
    delivery_hour: int = 12,
    model_version: str = "demand_forecaster@v1",
    predicted: float = 55_000.0,
) -> LoadForecast:
    delivery = _utc(2026, 5, 5, delivery_hour)
    return LoadForecast(
        zone=zone,
        as_of_time=delivery - timedelta(hours=24),
        delivery_time=delivery,
        predicted_load=EnergyMW(predicted),
        model_version=ModelVersion(model_version),
    )


def _read_file(path: Path) -> list[LoadForecast]:
    return [deserialise(line) for line in path.read_text("utf-8").splitlines() if line]


class TestConstructorAndProtocol:
    def test_subdir_is_created_eagerly(self, tmp_path: Path) -> None:
        LocalFsLoadForecastRepository(root=tmp_path)
        assert (tmp_path / "load_forecasts").is_dir()

    def test_satisfies_repository_protocol_structurally(self, tmp_path: Path) -> None:
        repo: LoadForecastRepository = LocalFsLoadForecastRepository(root=tmp_path)
        assert repo.add_many([]) == 0


class TestWriteAndRoundtrip:
    def test_writes_one_line_per_forecast(self, tmp_path: Path) -> None:
        repo = LocalFsLoadForecastRepository(root=tmp_path)
        forecasts = [_make_forecast(delivery_hour=h) for h in range(10, 13)]

        inserted = repo.add_many(forecasts)

        assert inserted == 3
        path = tmp_path / "load_forecasts" / "DE_LU.jsonl"
        assert _read_file(path) == forecasts


class TestDeduplication:
    def test_same_zone_delivery_model_is_deduped(self, tmp_path: Path) -> None:
        repo = LocalFsLoadForecastRepository(root=tmp_path)
        forecasts = [_make_forecast() for _ in range(2)]

        first = repo.add_many(forecasts[:1])
        second = repo.add_many(forecasts)  # one new (none) + one dup

        assert first == 1
        assert second == 0

    def test_different_model_versions_for_same_delivery_coexist(self, tmp_path: Path) -> None:
        repo = LocalFsLoadForecastRepository(root=tmp_path)
        f1 = _make_forecast(model_version="demand_forecaster@v1")
        f2 = _make_forecast(model_version="demand_forecaster@v2")

        # Same zone + delivery_time, different model_version → two rows.
        inserted = repo.add_many([f1, f2])

        assert inserted == 2
        path = tmp_path / "load_forecasts" / "DE_LU.jsonl"
        assert len(_read_file(path)) == 2
