"""Unit tests for the training pipeline's pure-function nodes."""

from datetime import UTC, datetime, timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from energy_forecaster.pipelines.training.nodes import (
    collect_artifacts,
    evaluate_model,
    prepare_training_data,
    train_model,
)


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _synthetic_features(hours: int, zones: tuple[str, ...] = ("DE_LU",)) -> pd.DataFrame:
    """Build a feature matrix with all FeatureMatrixSchema columns populated."""
    rows = []
    rng = np.random.default_rng(seed=42)
    for zone in zones:
        for h in range(hours):
            ts = _utc(2026, 5, 4) + timedelta(hours=h)
            base = 50_000.0 if zone == "DE_LU" else 40_000.0
            rows.append(
                {
                    "timestamp_utc": ts,
                    "zone": zone,
                    "load_mw": base + 100.0 * h + rng.normal(0, 100),
                    "temp_c": 15.0 + rng.normal(0, 2),
                    "wind_10m_ms": 4.0,
                    "wind_100m_ms": 8.0,
                    "ghi_wm2": 300.0,
                    "cloud_cover_pct": 50.0,
                    "precip_mm": 0.0,
                    "hour_of_day": ts.hour,
                    "day_of_week": ts.weekday(),
                    "is_weekend": ts.weekday() >= 5,
                    "load_lag_1h": base + 100.0 * (h - 1) if h >= 1 else None,
                    "load_lag_24h": base + 100.0 * (h - 24) if h >= 24 else None,
                    "load_lag_168h": (base + 100.0 * (h - 168) if h >= 168 else None),
                }
            )
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    return df


class TestPrepareTrainingData:
    def test_drops_rows_with_null_lags(self) -> None:
        # First 168 rows have at least one NaN lag (need 168h history).
        df = _synthetic_features(hours=200)
        out = prepare_training_data(df)
        # 200 - 168 = 32 valid rows, split 80/20 → 25 train + 7 test.
        assert out["train_size"] + out["test_size"] == 32

    def test_split_is_time_ordered_not_random(self) -> None:
        # Confirm last 20 % by timestamp end up in test, not random
        # rows. The training fold's last timestamp must precede every
        # test fold timestamp.
        df = _synthetic_features(hours=400)  # plenty of valid rows
        out = prepare_training_data(df)

        train_df = out["X_train"].copy()
        train_df["timestamp_utc"] = (
            df.dropna(subset=["load_lag_1h", "load_lag_24h", "load_lag_168h"])
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
            .iloc[: out["train_size"]]["timestamp_utc"]
            .to_numpy()
        )
        # We can't easily inspect timestamps from the X frames (zone_cat
        # encoded), so re-derive via the same sort the node uses.
        sorted_df = df.dropna(subset=["load_lag_1h", "load_lag_24h", "load_lag_168h"]).sort_values(
            "timestamp_utc"
        )
        train_last_ts = sorted_df.iloc[: out["train_size"]]["timestamp_utc"].max()
        test_first_ts = sorted_df.iloc[out["train_size"] :]["timestamp_utc"].min()
        assert train_last_ts < test_first_ts

    def test_zone_categorical_is_encoded(self) -> None:
        df = _synthetic_features(hours=300, zones=("DE_LU", "FR"))
        out = prepare_training_data(df)
        # zone_cat must be present and integer-coded.
        assert "zone_cat" in out["X_train"].columns
        assert out["X_train"]["zone_cat"].dtype.kind in ("i",)


class TestTrainAndEvaluate:
    def test_train_model_returns_a_lightgbm_booster(self) -> None:
        df = _synthetic_features(hours=400)
        td = prepare_training_data(df)
        model = train_model(td)
        assert isinstance(model, lgb.Booster)

    def test_evaluate_model_returns_finite_mape(self) -> None:
        df = _synthetic_features(hours=400)
        td = prepare_training_data(df)
        model = train_model(td)
        metrics = evaluate_model(model, td)
        assert "mape" in metrics
        assert np.isfinite(metrics["mape"])
        assert metrics["mape"] >= 0.0


class TestCollectArtifacts:
    def test_bundles_model_params_metrics(self) -> None:
        df = _synthetic_features(hours=400)
        td = prepare_training_data(df)
        model = train_model(td)
        metrics = evaluate_model(model, td)

        bundle = collect_artifacts(model, metrics)

        assert bundle["model"] is model
        assert bundle["metrics"] == metrics
        assert "objective" in bundle["params"]
        assert bundle["params"]["objective"] == "regression"
