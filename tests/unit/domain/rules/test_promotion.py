"""Unit tests for the champion/challenger promotion rule."""

import pytest

from energy_forecaster.domain.rules.promotion import (
    PROMOTION_MAPE_DELTA,
    should_promote,
)
from energy_forecaster.domain.value_objects.mape import MAPE


class TestShouldPromoteDefaultDelta:
    def test_challenger_clearly_better_is_promoted(self) -> None:
        assert should_promote(challenger=MAPE(0.04), champion=MAPE(0.05))

    def test_challenger_exactly_at_threshold_is_promoted(self) -> None:
        # The threshold is inclusive — a challenger that exactly meets the
        # required margin promotes. Documenting this here so a future change
        # to strict-less-than is a deliberate decision, not an accident.
        assert should_promote(
            challenger=MAPE(0.05 - PROMOTION_MAPE_DELTA),
            champion=MAPE(0.05),
        )

    def test_challenger_just_short_of_threshold_is_not_promoted(self) -> None:
        # 0.4pp better is not enough — the gate is 0.5pp.
        assert not should_promote(
            challenger=MAPE(0.05 - 0.004),
            champion=MAPE(0.05),
        )

    def test_challenger_equal_to_champion_is_not_promoted(self) -> None:
        assert not should_promote(challenger=MAPE(0.05), champion=MAPE(0.05))

    def test_challenger_worse_than_champion_is_not_promoted(self) -> None:
        assert not should_promote(challenger=MAPE(0.06), champion=MAPE(0.05))


class TestShouldPromoteCustomDelta:
    def test_zero_delta_promotes_any_at_least_as_good_challenger(self) -> None:
        # delta=0 means "promote if not strictly worse" — consistent with the
        # inclusive threshold documented in TestShouldPromoteDefaultDelta.
        assert should_promote(MAPE(0.04999), MAPE(0.05), delta=0.0)
        assert should_promote(MAPE(0.05), MAPE(0.05), delta=0.0)
        assert not should_promote(MAPE(0.05001), MAPE(0.05), delta=0.0)

    def test_larger_delta_demands_larger_improvement(self) -> None:
        # With a 1pp gate, a 0.6pp improvement is no longer enough.
        assert not should_promote(
            challenger=MAPE(0.044),
            champion=MAPE(0.05),
            delta=0.01,
        )
        assert should_promote(
            challenger=MAPE(0.04),
            champion=MAPE(0.05),
            delta=0.01,
        )

    def test_negative_delta_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            should_promote(MAPE(0.04), MAPE(0.05), delta=-0.001)
