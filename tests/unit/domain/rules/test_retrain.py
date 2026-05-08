"""Unit tests for the retraining trigger rule."""

import pytest

from energy_forecaster.domain.rules.retrain import (
    RETRAIN_MAPE_THRESHOLD,
    RETRAIN_PSI_THRESHOLD,
    should_retrain,
)
from energy_forecaster.domain.value_objects.mape import MAPE


class TestShouldRetrainDefaultThresholds:
    def test_both_below_threshold_does_not_retrain(self) -> None:
        assert not should_retrain(rolling_mape=MAPE(0.03), max_psi=0.05)

    def test_mape_at_threshold_triggers_retrain(self) -> None:
        # The threshold is inclusive — exactly hitting it triggers,
        # mirroring the promotion rule's convention.
        assert should_retrain(rolling_mape=MAPE(RETRAIN_MAPE_THRESHOLD), max_psi=0.05)

    def test_mape_above_threshold_triggers_retrain(self) -> None:
        assert should_retrain(rolling_mape=MAPE(0.10), max_psi=0.05)

    def test_psi_at_threshold_triggers_retrain(self) -> None:
        assert should_retrain(rolling_mape=MAPE(0.01), max_psi=RETRAIN_PSI_THRESHOLD)

    def test_psi_above_threshold_triggers_retrain(self) -> None:
        assert should_retrain(rolling_mape=MAPE(0.01), max_psi=0.40)

    def test_both_breached_still_triggers(self) -> None:
        # OR rule: redundant breach is still a breach.
        assert should_retrain(rolling_mape=MAPE(0.10), max_psi=0.40)

    def test_mape_just_below_threshold_does_not_retrain(self) -> None:
        # 4.99% MAPE with calm features stays in production.
        assert not should_retrain(rolling_mape=MAPE(0.0499), max_psi=0.05)

    def test_psi_just_below_threshold_does_not_retrain(self) -> None:
        assert not should_retrain(rolling_mape=MAPE(0.01), max_psi=0.1999)


class TestShouldRetrainCustomThresholds:
    def test_stricter_mape_threshold_triggers_earlier(self) -> None:
        # A team that contracted to a 3% MAPE retrains sooner.
        assert should_retrain(
            rolling_mape=MAPE(0.04),
            max_psi=0.05,
            mape_threshold=0.03,
        )
        assert not should_retrain(
            rolling_mape=MAPE(0.04),
            max_psi=0.05,
            mape_threshold=0.05,
        )

    def test_looser_psi_threshold_tolerates_more_drift(self) -> None:
        # PSI=0.25 is a retrain at the default 0.20 gate but not at 0.30.
        assert should_retrain(
            rolling_mape=MAPE(0.01),
            max_psi=0.25,
        )
        assert not should_retrain(
            rolling_mape=MAPE(0.01),
            max_psi=0.25,
            psi_threshold=0.30,
        )

    def test_negative_mape_threshold_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="MAPE threshold"):
            should_retrain(
                rolling_mape=MAPE(0.01),
                max_psi=0.05,
                mape_threshold=-0.01,
            )

    def test_negative_psi_threshold_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="PSI threshold"):
            should_retrain(
                rolling_mape=MAPE(0.01),
                max_psi=0.05,
                psi_threshold=-0.01,
            )
