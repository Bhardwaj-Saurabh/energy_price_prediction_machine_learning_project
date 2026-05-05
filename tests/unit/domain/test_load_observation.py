"""Unit tests for the LoadObservation entity."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


class TestLoadObservationConstruction:
    def test_typical_observation_constructs(self) -> None:
        obs = LoadObservation(
            zone=BiddingZone.DE_LU,
            timestamp_utc=_utc(2026, 5, 5, 12),
            load=EnergyMW(58_400.0),
        )
        assert obs.zone is BiddingZone.DE_LU
        assert obs.timestamp_utc == _utc(2026, 5, 5, 12)
        assert obs.load == EnergyMW(58_400.0)

    def test_naive_timestamp_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            LoadObservation(
                zone=BiddingZone.FR,
                timestamp_utc=datetime(2026, 5, 5, 12),
                load=EnergyMW(50_000.0),
            )

    def test_non_utc_timestamp_is_rejected(self) -> None:
        cet = timezone(timedelta(hours=1))
        with pytest.raises(ValueError, match="must be UTC"):
            LoadObservation(
                zone=BiddingZone.FR,
                timestamp_utc=datetime(2026, 5, 5, 12, tzinfo=cet),
                load=EnergyMW(50_000.0),
            )


class TestLoadObservationIdentity:
    def test_observations_with_same_identity_are_equal(self) -> None:
        a = LoadObservation(
            zone=BiddingZone.GB,
            timestamp_utc=_utc(2026, 5, 5),
            load=EnergyMW(30_000.0),
        )
        b = LoadObservation(
            zone=BiddingZone.GB,
            timestamp_utc=_utc(2026, 5, 5),
            load=EnergyMW(30_000.0),
        )
        assert a == b
        assert hash(a) == hash(b)

    def test_observations_with_different_load_are_not_equal(self) -> None:
        # The dataclass equality compares ALL fields, so a different load value
        # produces a different observation. Identity-based dedup (zone + ts only)
        # is the responsibility of the ingestion use case, not the entity.
        a = LoadObservation(
            zone=BiddingZone.GB,
            timestamp_utc=_utc(2026, 5, 5),
            load=EnergyMW(30_000.0),
        )
        b = LoadObservation(
            zone=BiddingZone.GB,
            timestamp_utc=_utc(2026, 5, 5),
            load=EnergyMW(30_001.0),
        )
        assert a != b

    def test_is_immutable(self) -> None:
        obs = LoadObservation(
            zone=BiddingZone.GB,
            timestamp_utc=_utc(2026, 5, 5),
            load=EnergyMW(30_000.0),
        )
        with pytest.raises(AttributeError):
            obs.zone = BiddingZone.FR  # type: ignore[misc]
