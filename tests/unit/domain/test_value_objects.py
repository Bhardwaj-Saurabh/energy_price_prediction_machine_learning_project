"""Unit tests for the domain value objects.

These tests are pure: no I/O, no fixtures beyond literal values, no shared
state. They exist to nail down the construction-time invariants that every
downstream layer is allowed to assume.
"""

import math

import pytest

from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import (
    MAX_PLAUSIBLE_LOAD_MW,
    EnergyMW,
)
from energy_forecaster.domain.value_objects.horizon import (
    MAX_HORIZON_HOURS,
    MIN_HORIZON_HOURS,
    HorizonHours,
)
from energy_forecaster.domain.value_objects.price import (
    MAX_PLAUSIBLE_PRICE_EUR,
    MIN_PLAUSIBLE_PRICE_EUR,
    PriceEUR,
)

# ---------------------------------------------------------------------------
# BiddingZone
# ---------------------------------------------------------------------------


class TestBiddingZone:
    def test_string_value_round_trips(self) -> None:
        assert BiddingZone.DE_LU == "DE_LU"
        assert BiddingZone("FR") is BiddingZone.FR

    def test_unknown_zone_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            BiddingZone("ES")

    def test_zones_are_hashable_and_distinct(self) -> None:
        assert {BiddingZone.DE_LU, BiddingZone.FR, BiddingZone.GB} == set(BiddingZone)


# ---------------------------------------------------------------------------
# EnergyMW
# ---------------------------------------------------------------------------


class TestEnergyMW:
    def test_typical_value_constructs(self) -> None:
        assert EnergyMW(45_000.0).value == 45_000.0

    def test_zero_is_valid(self) -> None:
        assert EnergyMW(0.0).value == 0.0

    def test_upper_bound_is_inclusive(self) -> None:
        assert EnergyMW(MAX_PLAUSIBLE_LOAD_MW).value == MAX_PLAUSIBLE_LOAD_MW

    def test_negative_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            EnergyMW(-1.0)

    def test_above_plausible_upper_bound_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="plausible upper bound"):
            EnergyMW(MAX_PLAUSIBLE_LOAD_MW + 1.0)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            EnergyMW(bad)

    def test_is_immutable(self) -> None:
        x = EnergyMW(100.0)
        with pytest.raises(AttributeError):
            x.value = 200.0  # type: ignore[misc]

    def test_equality_is_by_value(self) -> None:
        assert EnergyMW(50.0) == EnergyMW(50.0)
        assert EnergyMW(50.0) != EnergyMW(51.0)

    def test_is_hashable(self) -> None:
        assert {EnergyMW(1.0), EnergyMW(1.0), EnergyMW(2.0)} == {
            EnergyMW(1.0),
            EnergyMW(2.0),
        }


# ---------------------------------------------------------------------------
# PriceEUR
# ---------------------------------------------------------------------------


class TestPriceEUR:
    def test_typical_positive_constructs(self) -> None:
        assert PriceEUR(85.5).value == 85.5

    def test_negative_within_bounds_is_valid(self) -> None:
        # Negative wholesale prices are real — must not be rejected.
        assert PriceEUR(-50.0).value == -50.0

    def test_upper_bound_inclusive(self) -> None:
        assert PriceEUR(MAX_PLAUSIBLE_PRICE_EUR).value == MAX_PLAUSIBLE_PRICE_EUR

    def test_lower_bound_inclusive(self) -> None:
        assert PriceEUR(MIN_PLAUSIBLE_PRICE_EUR).value == MIN_PLAUSIBLE_PRICE_EUR

    def test_above_upper_bound_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            PriceEUR(MAX_PLAUSIBLE_PRICE_EUR + 1.0)

    def test_below_lower_bound_rejected(self) -> None:
        with pytest.raises(ValueError, match="below"):
            PriceEUR(MIN_PLAUSIBLE_PRICE_EUR - 1.0)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            PriceEUR(bad)


# ---------------------------------------------------------------------------
# HorizonHours
# ---------------------------------------------------------------------------


class TestHorizonHours:
    def test_lower_bound_constructs(self) -> None:
        assert HorizonHours(MIN_HORIZON_HOURS).value == MIN_HORIZON_HOURS

    def test_upper_bound_constructs(self) -> None:
        assert HorizonHours(MAX_HORIZON_HOURS).value == MAX_HORIZON_HOURS

    @pytest.mark.parametrize("bad", [MIN_HORIZON_HOURS - 1, 0, MAX_HORIZON_HOURS + 1, -5, 1000])
    def test_out_of_range_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="between"):
            HorizonHours(bad)

    def test_float_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be int"):
            HorizonHours(24.0)  # type: ignore[arg-type]

    def test_bool_is_rejected_even_though_python_says_it_is_int(self) -> None:
        # `isinstance(True, int)` is True in Python — explicit check needed.
        # mypy considers bool a valid int here, so no type-ignore is required.
        with pytest.raises(TypeError, match="must be int"):
            HorizonHours(True)
