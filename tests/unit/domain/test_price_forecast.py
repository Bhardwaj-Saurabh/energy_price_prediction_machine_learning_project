"""Unit tests for the PriceForecast entity.

PriceForecast mirrors LoadForecast in shape and validation, so these tests
focus on the differences (the value field is PriceEUR, including negative
prices) and confirm the shared validation paths via a single happy + a
single failure case each — the comprehensive validation matrix lives in
test_load_forecast.py and applies identically here.
"""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from energy_forecaster.domain.entities.price_forecast import PriceForecast
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.domain.value_objects.price import PriceEUR


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _make(
    *,
    as_of_time: datetime | None = None,
    delivery_time: datetime | None = None,
    predicted_price: PriceEUR | None = None,
) -> PriceForecast:
    return PriceForecast(
        zone=BiddingZone.FR,
        as_of_time=as_of_time or _utc(2026, 5, 5, 14),
        delivery_time=delivery_time or _utc(2026, 5, 6, 13),
        predicted_price=predicted_price or PriceEUR(72.5),
        model_version=ModelVersion("price_fr/3"),
    )


class TestPriceForecastHappyPath:
    def test_positive_price_constructs(self) -> None:
        f = _make()
        assert f.predicted_price == PriceEUR(72.5)

    def test_negative_price_constructs(self) -> None:
        # Negative wholesale prices are valid market events, not errors —
        # the forecast must support them so we can predict curtailment.
        f = _make(predicted_price=PriceEUR(-15.0))
        assert f.predicted_price == PriceEUR(-15.0)


class TestPriceForecastValidation:
    def test_naive_delivery_time_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            _make(delivery_time=datetime(2026, 5, 6, 13))

    def test_non_utc_as_of_time_is_rejected(self) -> None:
        cet = timezone(timedelta(hours=1))
        with pytest.raises(ValueError, match="must be UTC"):
            _make(as_of_time=datetime(2026, 5, 5, 14, tzinfo=cet))

    def test_non_hour_aligned_delivery_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="aligned to the hour"):
            _make(delivery_time=_utc(2026, 5, 6, 13, 30))

    def test_delivery_not_after_as_of_is_rejected(self) -> None:
        ts = _utc(2026, 5, 5, 12)
        with pytest.raises(ValueError, match="must be after as_of_time"):
            _make(as_of_time=ts, delivery_time=ts)


class TestPriceForecastImmutability:
    def test_is_immutable(self) -> None:
        f = _make()
        with pytest.raises(AttributeError):
            f.predicted_price = PriceEUR(0.0)  # type: ignore[misc]
