"""Unit tests for the LoadForecast entity."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.model_version import ModelVersion


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _make(
    *,
    zone: BiddingZone = BiddingZone.DE_LU,
    as_of_time: datetime | None = None,
    delivery_time: datetime | None = None,
    predicted_load: EnergyMW | None = None,
    model_version: ModelVersion | None = None,
) -> LoadForecast:
    return LoadForecast(
        zone=zone,
        as_of_time=as_of_time or _utc(2026, 5, 5, 6),
        delivery_time=delivery_time or _utc(2026, 5, 6, 12),
        predicted_load=predicted_load or EnergyMW(48_500.0),
        model_version=model_version or ModelVersion("demand_de_lu/12"),
    )


class TestLoadForecastHappyPath:
    def test_typical_forecast_constructs(self) -> None:
        f = _make()
        assert f.zone is BiddingZone.DE_LU
        assert f.delivery_time > f.as_of_time
        assert f.predicted_load == EnergyMW(48_500.0)
        assert f.model_version == ModelVersion("demand_de_lu/12")


class TestLoadForecastTimestamps:
    def test_naive_as_of_time_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            _make(as_of_time=datetime(2026, 5, 5, 6))

    def test_non_utc_as_of_time_is_rejected(self) -> None:
        cet = timezone(timedelta(hours=1))
        with pytest.raises(ValueError, match="must be UTC"):
            _make(as_of_time=datetime(2026, 5, 5, 6, tzinfo=cet))

    def test_naive_delivery_time_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            _make(delivery_time=datetime(2026, 5, 6, 12))

    def test_non_utc_delivery_time_is_rejected(self) -> None:
        cet = timezone(timedelta(hours=1))
        with pytest.raises(ValueError, match="must be UTC"):
            _make(delivery_time=datetime(2026, 5, 6, 12, tzinfo=cet))


class TestLoadForecastDeliveryAlignment:
    @pytest.mark.parametrize(
        "delivery",
        [
            _utc(2026, 5, 6, 12, 30),  # half-past
            datetime(2026, 5, 6, 12, 0, 1, tzinfo=UTC),  # one second past
            datetime(2026, 5, 6, 12, 0, 0, 1, tzinfo=UTC),  # one microsecond past
        ],
    )
    def test_non_hour_aligned_delivery_is_rejected(self, delivery: datetime) -> None:
        with pytest.raises(ValueError, match="aligned to the hour"):
            _make(delivery_time=delivery)


class TestLoadForecastTimeOrdering:
    def test_delivery_at_same_instant_as_as_of_is_rejected(self) -> None:
        ts = _utc(2026, 5, 5, 12)
        with pytest.raises(ValueError, match="must be after as_of_time"):
            _make(as_of_time=ts, delivery_time=ts)

    def test_delivery_before_as_of_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be after as_of_time"):
            _make(
                as_of_time=_utc(2026, 5, 5, 12),
                delivery_time=_utc(2026, 5, 5, 11),
            )


class TestLoadForecastIdentity:
    def test_equal_when_all_fields_match(self) -> None:
        a = _make()
        b = _make()
        assert a == b
        assert hash(a) == hash(b)

    def test_is_immutable(self) -> None:
        f = _make()
        with pytest.raises(AttributeError):
            f.predicted_load = EnergyMW(0.0)  # type: ignore[misc]
