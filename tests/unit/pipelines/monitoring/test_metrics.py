"""Unit tests for the monitoring pipeline's numerical helpers."""

import numpy as np
import pytest

from energy_forecaster.pipelines.monitoring.metrics import (
    mape,
    population_stability_index,
)


class TestMape:
    def test_identical_arrays_return_zero(self) -> None:
        assert mape([100.0, 200.0, 300.0], [100.0, 200.0, 300.0]) == 0.0

    def test_uniform_ten_percent_error_returns_point_one(self) -> None:
        # Predictions 10% below truth on every row → MAPE = 0.10.
        actuals = [100.0, 200.0, 300.0]
        predictions = [90.0, 180.0, 270.0]
        assert mape(actuals, predictions) == pytest.approx(0.10)

    def test_accepts_numpy_arrays(self) -> None:
        # The use case will pass numpy arrays after extracting columns
        # from a DataFrame; both sequence types must work.
        a = np.array([100.0, 200.0])
        p = np.array([110.0, 220.0])
        assert mape(a, p) == pytest.approx(0.10)

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            mape([], [])

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="align"):
            mape([1.0, 2.0, 3.0], [1.0, 2.0])


class TestPopulationStabilityIndex:
    def test_identical_distributions_score_near_zero(self) -> None:
        # Same RNG-seeded distribution on both sides → PSI ≈ 0.
        rng = np.random.default_rng(seed=0)
        sample = rng.normal(loc=0.0, scale=1.0, size=10_000)
        # Use disjoint halves so the bin edges aren't identically
        # supported by the same points (which would be circular).
        psi = population_stability_index(sample[:5_000], sample[5_000:])
        assert psi < 0.05

    def test_shifted_distribution_scores_above_significant_threshold(self) -> None:
        # A 1-sigma mean shift is well past "significant drift" (0.20).
        rng = np.random.default_rng(seed=1)
        baseline = rng.normal(loc=0.0, scale=1.0, size=10_000)
        shifted = rng.normal(loc=1.0, scale=1.0, size=10_000)
        psi = population_stability_index(baseline, shifted)
        assert psi > 0.20

    def test_concentrated_distribution_scores_above_zero(self) -> None:
        # Same mean, half the variance — the observed mass is tighter
        # than baseline. PSI should pick this up as drift even without a
        # mean shift.
        rng = np.random.default_rng(seed=2)
        baseline = rng.normal(loc=0.0, scale=1.0, size=10_000)
        narrow = rng.normal(loc=0.0, scale=0.5, size=10_000)
        psi = population_stability_index(baseline, narrow)
        assert psi > 0.05

    def test_handles_binary_feature_without_crashing(self) -> None:
        # ``is_weekend`` has only two unique values. Quantile edges
        # collapse; the helper must dedupe and compute a finite score.
        baseline = np.array([0.0] * 700 + [1.0] * 300)
        observed = np.array([0.0] * 600 + [1.0] * 400)
        psi = population_stability_index(baseline, observed, bins=10)
        # Mass shifted from bin 0 → bin 1; PSI should be finite and > 0.
        assert np.isfinite(psi)
        assert psi > 0.0

    def test_drops_nans_before_binning(self) -> None:
        # NaN-poisoned input must not propagate into the score.
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.nan, np.nan])
        psi = population_stability_index(baseline, observed)
        assert np.isfinite(psi)
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_constant_baseline_with_matching_observed_returns_zero(self) -> None:
        # Degenerate case: all baseline values identical and observed
        # matches — no drift.
        psi = population_stability_index(
            expected=[5.0, 5.0, 5.0, 5.0],
            observed=[5.0, 5.0, 5.0, 5.0],
        )
        assert psi == 0.0

    def test_constant_baseline_with_mismatched_observed_returns_inf(self) -> None:
        # Degenerate case: baseline is a delta at 5; observed is not.
        # Defined as ``inf`` so the retrain rule fires unambiguously.
        psi = population_stability_index(
            expected=[5.0, 5.0, 5.0, 5.0],
            observed=[1.0, 2.0, 3.0, 4.0],
        )
        assert psi == float("inf")

    def test_too_few_bins_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2 bins"):
            population_stability_index([1.0, 2.0], [1.0, 2.0], bins=1)

    def test_empty_after_nan_drop_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            population_stability_index(
                expected=[float("nan"), float("nan")],
                observed=[1.0, 2.0],
            )

    def test_extreme_observed_value_is_counted_in_top_bin(self) -> None:
        # The right edge gets nudged so values exactly at or above the
        # baseline max are still binned. A baseline of 1..10 and an
        # observed value of 10 (exactly at the upper edge) must not be
        # silently dropped — that would give a falsely small PSI.
        baseline = np.arange(1.0, 11.0)
        observed = np.array([10.0] * 100)
        psi = population_stability_index(baseline, observed)
        assert np.isfinite(psi)
        assert psi > 0.20  # heavy concentration in one bin = strong drift
