"""Integration tests for LocalFsLoadObservationRepository — touches real filesystem."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from energy_forecaster.adapters.load_observation_repo.local_fs import (
    LocalFsLoadObservationRepository,
    _deserialise,
)
from energy_forecaster.application.ports.load_observation_repository import (
    LoadObservationRepository,
)
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW

pytestmark = pytest.mark.integration


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _hourly_load(
    zone: BiddingZone, start: datetime, hours: int, base_mw: float = 50_000.0
) -> list[LoadObservation]:
    return [
        LoadObservation(
            zone=zone,
            timestamp_utc=start + timedelta(hours=h),
            load=EnergyMW(base_mw + h * 100.0),
        )
        for h in range(hours)
    ]


def _read_file(path: Path) -> list[LoadObservation]:
    return [_deserialise(line) for line in path.read_text("utf-8").splitlines() if line]


class TestConstructorBehaviour:
    def test_subdir_is_created_eagerly(self, tmp_path: Path) -> None:
        # The composition root should surface permission / path errors at
        # startup, not at the first ingest. Verifying eager mkdir is what
        # locks that behaviour in.
        LocalFsLoadObservationRepository(root=tmp_path)
        assert (tmp_path / "load_observations").is_dir()

    def test_existing_subdir_is_reused(self, tmp_path: Path) -> None:
        (tmp_path / "load_observations").mkdir()
        LocalFsLoadObservationRepository(root=tmp_path)  # must not raise

    def test_satisfies_the_repository_protocol_structurally(self, tmp_path: Path) -> None:
        repo: LoadObservationRepository = LocalFsLoadObservationRepository(root=tmp_path)
        assert repo.add_many([]) == 0


class TestFirstWrite:
    def test_empty_input_creates_no_file(self, tmp_path: Path) -> None:
        repo = LocalFsLoadObservationRepository(root=tmp_path)
        result = repo.add_many([])
        assert result == 0
        # Subdir exists (from constructor) but is empty.
        assert list((tmp_path / "load_observations").iterdir()) == []

    def test_writes_one_line_per_observation_to_zone_file(self, tmp_path: Path) -> None:
        repo = LocalFsLoadObservationRepository(root=tmp_path)
        observations = _hourly_load(BiddingZone.DE_LU, _utc(2026, 5, 4), 3)

        inserted = repo.add_many(observations)

        assert inserted == 3
        path = tmp_path / "load_observations" / "DE_LU.jsonl"
        assert path.exists()
        round_tripped = _read_file(path)
        assert round_tripped == observations

    def test_serialised_line_has_expected_shape(self, tmp_path: Path) -> None:
        # Format is part of the contract — downstream tools and humans
        # both rely on it. Locking the schema in here makes any future
        # accidental change visible in a diff.
        repo = LocalFsLoadObservationRepository(root=tmp_path)
        repo.add_many(
            [
                LoadObservation(
                    zone=BiddingZone.DE_LU,
                    timestamp_utc=_utc(2026, 5, 4, 12),
                    load=EnergyMW(58_400.0),
                )
            ]
        )

        line = (tmp_path / "load_observations" / "DE_LU.jsonl").read_text("utf-8").strip()
        record = json.loads(line)
        assert record == {
            "zone": "DE_LU",
            "timestamp_utc": "2026-05-04T12:00:00+00:00",
            "load": 58_400.0,
        }


class TestMultipleZones:
    def test_each_zone_gets_its_own_file(self, tmp_path: Path) -> None:
        repo = LocalFsLoadObservationRepository(root=tmp_path)
        de = _hourly_load(BiddingZone.DE_LU, _utc(2026, 5, 4), 2)
        fr = _hourly_load(BiddingZone.FR, _utc(2026, 5, 4), 3)
        gb = _hourly_load(BiddingZone.GB, _utc(2026, 5, 4), 4)

        inserted = repo.add_many([*de, *fr, *gb])

        assert inserted == 9
        load_dir = tmp_path / "load_observations"
        assert {p.name for p in load_dir.iterdir()} == {
            "DE_LU.jsonl",
            "FR.jsonl",
            "GB.jsonl",
        }
        assert len(_read_file(load_dir / "DE_LU.jsonl")) == 2
        assert len(_read_file(load_dir / "FR.jsonl")) == 3
        assert len(_read_file(load_dir / "GB.jsonl")) == 4


class TestDeduplication:
    def test_re_writing_same_observations_inserts_zero(self, tmp_path: Path) -> None:
        repo = LocalFsLoadObservationRepository(root=tmp_path)
        observations = _hourly_load(BiddingZone.DE_LU, _utc(2026, 5, 4), 3)

        first = repo.add_many(observations)
        second = repo.add_many(observations)

        assert first == 3
        assert second == 0
        # File still has only the original 3 lines — nothing appended.
        assert len(_read_file(tmp_path / "load_observations" / "DE_LU.jsonl")) == 3

    def test_mixed_new_and_existing_inserts_only_new(self, tmp_path: Path) -> None:
        repo = LocalFsLoadObservationRepository(root=tmp_path)
        first_batch = _hourly_load(BiddingZone.DE_LU, _utc(2026, 5, 4), 3)
        repo.add_many(first_batch)

        # Second batch overlaps the first by 1 hour and adds 2 new hours.
        overlap_plus_new = _hourly_load(BiddingZone.DE_LU, _utc(2026, 5, 4, 2), 3)

        inserted = repo.add_many(overlap_plus_new)

        assert inserted == 2
        assert len(_read_file(tmp_path / "load_observations" / "DE_LU.jsonl")) == 5
