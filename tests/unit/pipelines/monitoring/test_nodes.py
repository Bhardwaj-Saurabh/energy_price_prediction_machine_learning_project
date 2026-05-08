"""Unit tests for the monitoring pipeline's pure nodes."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.model_version import ModelVersion
from energy_forecaster.pipelines.monitoring.nodes import (
    compute_psi_per_feature,
    compute_rolling_mape_per_zone,
)


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _forecast(zone: BiddingZone, delivery: datetime, predicted_mw: float) -> LoadForecast:
    return LoadForecast(
        zone=zone,
        as_of_time=delivery - timedelta(hours=24),
        delivery_time=delivery,
        predicted_load=EnergyMW(predicted_mw),
        model_version=ModelVersion("demand_forecaster@v1"),
    )


def _observation(zone: BiddingZone, ts: datetime, load_mw: float) -> LoadObservation:
    return LoadObservation(zone=zone, timestamp_utc=ts, load=EnergyMW(load_mw))


class TestComputeRollingMapePerZone:
    def test_perfect_predictions_yield_zero_mape(self) -> None:
        forecasts = [_forecast(BiddingZone.DE_LU, _utc(2026, 5, 7, h), 50_000.0) for h in range(3)]
        observations = [
            _observation(BiddingZone.DE_LU, _utc(2026, 5, 7, h), 50_000.0) for h in range(3)
        ]
        assert compute_rolling_mape_per_zone(forecasts, observations) == {"DE_LU": 0.0}

    def test_uniform_ten_percent_underprediction(self) -> None:
        # Predicting 90% of truth on every hour → MAPE = 0.10.
        forecasts = [_forecast(BiddingZone.DE_LU, _utc(2026, 5, 7, h), 90_000.0) for h in range(3)]
        observations = [
            _observation(BiddingZone.DE_LU, _utc(2026, 5, 7, h), 100_000.0) for h in range(3)
        ]
        result = compute_rolling_mape_per_zone(forecasts, observations)
        assert result["DE_LU"] == pytest.approx(0.10)

    def test_unmatched_forecasts_are_skipped(self) -> None:
        # 3 forecasts, only 2 have observations. MAPE is computed on the
        # matched 2; the unmatched one is silently dropped.
        forecasts = [
            _forecast(BiddingZone.DE_LU, _utc(2026, 5, 7, 0), 100_000.0),
            _forecast(BiddingZone.DE_LU, _utc(2026, 5, 7, 1), 100_000.0),
            _forecast(BiddingZone.DE_LU, _utc(2026, 5, 7, 2), 100_000.0),  # no obs
        ]
        observations = [
            _observation(BiddingZone.DE_LU, _utc(2026, 5, 7, 0), 100_000.0),
            _observation(BiddingZone.DE_LU, _utc(2026, 5, 7, 1), 100_000.0),
        ]
        assert compute_rolling_mape_per_zone(forecasts, observations) == {"DE_LU": 0.0}

    def test_zone_with_no_matched_pairs_is_omitted(self) -> None:
        # FR has forecasts but the observations are for a different
        # delivery hour; the zone drops out rather than reporting nan.
        forecasts = [
            _forecast(BiddingZone.DE_LU, _utc(2026, 5, 7, 0), 100_000.0),
            _forecast(BiddingZone.FR, _utc(2026, 5, 7, 0), 50_000.0),
        ]
        observations = [
            _observation(BiddingZone.DE_LU, _utc(2026, 5, 7, 0), 100_000.0),
            _observation(BiddingZone.FR, _utc(2026, 5, 8, 0), 50_000.0),
        ]
        result = compute_rolling_mape_per_zone(forecasts, observations)
        assert "DE_LU" in result
        assert "FR" not in result

    def test_per_zone_mapes_are_independent(self) -> None:
        forecasts = [
            _forecast(BiddingZone.DE_LU, _utc(2026, 5, 7, 0), 90_000.0),  # 10% under
            _forecast(BiddingZone.FR, _utc(2026, 5, 7, 0), 60_000.0),  # 20% over
        ]
        observations = [
            _observation(BiddingZone.DE_LU, _utc(2026, 5, 7, 0), 100_000.0),
            _observation(BiddingZone.FR, _utc(2026, 5, 7, 0), 50_000.0),
        ]
        result = compute_rolling_mape_per_zone(forecasts, observations)
        assert result["DE_LU"] == pytest.approx(0.10)
        assert result["FR"] == pytest.approx(0.20)

    def test_empty_inputs_return_empty_dict(self) -> None:
        assert compute_rolling_mape_per_zone([], []) == {}

    def test_duplicate_forecast_for_delivery_hour_uses_last(self) -> None:
        # If two model versions wrote forecasts for the same delivery
        # hour, the iteration order decides which is matched. Document
        # the contract: last one in wins.
        forecasts = [
            _forecast(BiddingZone.DE_LU, _utc(2026, 5, 7, 0), 80_000.0),  # would give MAPE=0.20
            _forecast(BiddingZone.DE_LU, _utc(2026, 5, 7, 0), 100_000.0),  # this one wins → 0
        ]
        observations = [
            _observation(BiddingZone.DE_LU, _utc(2026, 5, 7, 0), 100_000.0),
        ]
        assert compute_rolling_mape_per_zone(forecasts, observations) == {"DE_LU": 0.0}


class TestComputePsiPerFeature:
    def test_identical_distributions_return_low_psi(self) -> None:
        rng = np.random.default_rng(seed=0)
        sample = rng.normal(size=2_000)
        baseline = pd.DataFrame({"x": sample[:1_000], "y": sample[:1_000]})
        recent = pd.DataFrame({"x": sample[1_000:], "y": sample[1_000:]})
        result = compute_psi_per_feature(baseline, recent, ["x", "y"])
        assert result["x"] < 0.1
        assert result["y"] < 0.1

    def test_shifted_distribution_flagged(self) -> None:
        # One feature drifts; the other does not. PSI must flag only
        # the drifted feature.
        rng = np.random.default_rng(seed=1)
        baseline = pd.DataFrame(
            {
                "stable": rng.normal(loc=0.0, scale=1.0, size=2_000),
                "drifted": rng.normal(loc=0.0, scale=1.0, size=2_000),
            }
        )
        recent = pd.DataFrame(
            {
                "stable": rng.normal(loc=0.0, scale=1.0, size=2_000),
                "drifted": rng.normal(loc=1.5, scale=1.0, size=2_000),
            }
        )
        result = compute_psi_per_feature(baseline, recent, ["stable", "drifted"])
        assert result["stable"] < 0.1
        assert result["drifted"] > 0.20

    def test_only_named_columns_are_scored(self) -> None:
        # Extra columns in the frames must not appear in the output.
        # Important because the feature matrix carries identity columns
        # (zone, timestamp_utc) we never want to score.
        rng = np.random.default_rng(seed=2)
        baseline = pd.DataFrame(
            {
                "feature_a": rng.normal(size=500),
                "ignored": rng.normal(size=500),
            }
        )
        recent = pd.DataFrame(
            {
                "feature_a": rng.normal(size=500),
                "ignored": rng.normal(size=500),
            }
        )
        result = compute_psi_per_feature(baseline, recent, ["feature_a"])
        assert set(result.keys()) == {"feature_a"}

    def test_handles_low_cardinality_feature(self) -> None:
        # A binary feature like ``is_weekend`` would crash a naive PSI;
        # this confirms the helper's quantile-dedup path flows through.
        baseline = pd.DataFrame({"is_weekend": [0.0] * 700 + [1.0] * 300})
        recent = pd.DataFrame({"is_weekend": [0.0] * 600 + [1.0] * 400})
        result = compute_psi_per_feature(baseline, recent, ["is_weekend"])
        assert np.isfinite(result["is_weekend"])
        assert result["is_weekend"] > 0.0
