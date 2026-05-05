"""Unit tests for the WeatherReading entity."""

import math
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from energy_forecaster.domain.entities.weather_reading import (
    MAX_CLOUD_COVER_PCT,
    MAX_GHI_WM2,
    MAX_PRECIP_MM,
    MAX_TEMP_C,
    MAX_WIND_MS,
    MIN_CLOUD_COVER_PCT,
    MIN_GHI_WM2,
    MIN_PRECIP_MM,
    MIN_TEMP_C,
    MIN_WIND_MS,
    WeatherReading,
)
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


def _make(**overrides: Any) -> WeatherReading:
    """Build a valid WeatherReading, applying any per-test overrides.

    The kwargs accept Any because tests deliberately push invalid values
    through this helper to confirm validation rejects them at runtime.
    """
    base: dict[str, Any] = {
        "zone": BiddingZone.DE_LU,
        "timestamp_utc": datetime(2026, 5, 5, 12, tzinfo=UTC),
        "temp_c": 15.0,
        "wind_10m_ms": 5.0,
        "wind_100m_ms": 10.0,
        "ghi_wm2": 400.0,
        "cloud_cover_pct": 50.0,
        "precip_mm": 0.0,
    }
    base.update(overrides)
    return WeatherReading(**base)


class TestWeatherReadingHappyPath:
    def test_typical_reading_constructs(self) -> None:
        wr = _make()
        assert wr.zone is BiddingZone.DE_LU
        assert wr.temp_c == 15.0
        assert wr.cloud_cover_pct == 50.0


class TestWeatherReadingTimestamp:
    def test_naive_timestamp_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            _make(timestamp_utc=datetime(2026, 5, 5, 12))

    def test_non_utc_timestamp_is_rejected(self) -> None:
        cet = timezone(timedelta(hours=1))
        with pytest.raises(ValueError, match="must be UTC"):
            _make(timestamp_utc=datetime(2026, 5, 5, 12, tzinfo=cet))


@pytest.mark.parametrize(
    "field,below_min,above_max",
    [
        ("temp_c", MIN_TEMP_C - 1.0, MAX_TEMP_C + 1.0),
        ("wind_10m_ms", MIN_WIND_MS - 1.0, MAX_WIND_MS + 1.0),
        ("wind_100m_ms", MIN_WIND_MS - 1.0, MAX_WIND_MS + 1.0),
        ("ghi_wm2", MIN_GHI_WM2 - 1.0, MAX_GHI_WM2 + 1.0),
        ("cloud_cover_pct", MIN_CLOUD_COVER_PCT - 1.0, MAX_CLOUD_COVER_PCT + 1.0),
        ("precip_mm", MIN_PRECIP_MM - 1.0, MAX_PRECIP_MM + 1.0),
    ],
)
class TestWeatherReadingFieldRanges:
    def test_below_minimum_is_rejected(
        self, field: str, below_min: float, above_max: float
    ) -> None:
        with pytest.raises(ValueError, match="outside plausible range"):
            _make(**{field: below_min})

    def test_above_maximum_is_rejected(
        self, field: str, below_min: float, above_max: float
    ) -> None:
        with pytest.raises(ValueError, match="outside plausible range"):
            _make(**{field: above_max})


@pytest.mark.parametrize(
    "field",
    ["temp_c", "wind_10m_ms", "wind_100m_ms", "ghi_wm2", "cloud_cover_pct", "precip_mm"],
)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
class TestWeatherReadingFiniteness:
    def test_non_finite_field_is_rejected(self, field: str, bad: float) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            _make(**{field: bad})


class TestWeatherReadingIdentity:
    def test_equal_when_all_fields_match(self) -> None:
        a = _make()
        b = _make()
        assert a == b
        assert hash(a) == hash(b)

    def test_is_immutable(self) -> None:
        wr = _make()
        with pytest.raises(AttributeError):
            wr.temp_c = 20.0  # type: ignore[misc]
