"""Integration test for run_feature_engineering — full DAG against tmp_path."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from energy_forecaster.contracts.feature_matrix_schema import FeatureMatrixSchema
from energy_forecaster.pipelines.feature_engineering.runner import (
    run_feature_engineering,
)

pytestmark = pytest.mark.integration


def _write_load_jsonl(directory: Path, zone: str, start: datetime, hours: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{zone}.jsonl").open("w", encoding="utf-8") as f:
        for h in range(hours):
            ts = (start + timedelta(hours=h)).isoformat()
            f.write(
                json.dumps({"zone": zone, "timestamp_utc": ts, "load": 50_000.0 + 100.0 * h}) + "\n"
            )


def _write_weather_jsonl(directory: Path, zone: str, start: datetime, hours: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{zone}.jsonl").open("w", encoding="utf-8") as f:
        for h in range(hours):
            ts = (start + timedelta(hours=h)).isoformat()
            f.write(
                json.dumps(
                    {
                        "zone": zone,
                        "timestamp_utc": ts,
                        "temp_c": 15.0,
                        "wind_10m_ms": 4.0,
                        "wind_100m_ms": 8.0,
                        "ghi_wm2": 300.0,
                        "cloud_cover_pct": 50.0,
                        "precip_mm": 0.0,
                    }
                )
                + "\n"
            )


class TestRunFeatureEngineering:
    def test_full_pipeline_writes_validated_parquet(self, tmp_path: Path) -> None:
        load_dir = tmp_path / "load_observations"
        weather_dir = tmp_path / "weather_readings"
        out_path = tmp_path / "features.parquet"

        start = datetime(2026, 5, 4, tzinfo=UTC)
        _write_load_jsonl(load_dir, "DE_LU", start, 200)
        _write_weather_jsonl(weather_dir, "DE_LU", start, 200)

        run_feature_engineering(
            load_directory=load_dir,
            weather_directory=weather_dir,
            output_path=out_path,
        )

        assert out_path.exists()
        df = pd.read_parquet(out_path)
        # Schema is the contract; round-trip through Parquet must preserve it.
        FeatureMatrixSchema.validate(df)
        # 200 input hours → 200 output rows (inner join, all hours overlap).
        assert len(df) == 200

    def test_inner_join_drops_unmatched_hours(self, tmp_path: Path) -> None:
        load_dir = tmp_path / "load_observations"
        weather_dir = tmp_path / "weather_readings"
        out_path = tmp_path / "features.parquet"

        start = datetime(2026, 5, 4, tzinfo=UTC)
        # Load has 200h; weather only has 100h. Inner join keeps the 100
        # overlapping hours.
        _write_load_jsonl(load_dir, "DE_LU", start, 200)
        _write_weather_jsonl(weather_dir, "DE_LU", start, 100)

        run_feature_engineering(
            load_directory=load_dir,
            weather_directory=weather_dir,
            output_path=out_path,
        )

        df = pd.read_parquet(out_path)
        assert len(df) == 100

    def test_multiple_zones_yield_per_zone_lags(self, tmp_path: Path) -> None:
        load_dir = tmp_path / "load_observations"
        weather_dir = tmp_path / "weather_readings"
        out_path = tmp_path / "features.parquet"

        start = datetime(2026, 5, 4, tzinfo=UTC)
        for zone in ("DE_LU", "FR"):
            _write_load_jsonl(load_dir, zone, start, 200)
            _write_weather_jsonl(weather_dir, zone, start, 200)

        run_feature_engineering(
            load_directory=load_dir,
            weather_directory=weather_dir,
            output_path=out_path,
        )

        df = pd.read_parquet(out_path)
        assert set(df["zone"].unique()) == {"DE_LU", "FR"}
        # Each zone independently gets 200 rows; first 24 of each have
        # null 24h lag, not 24 of the first zone only.
        first_24_per_zone = df.sort_values(["zone", "timestamp_utc"]).groupby("zone").head(24)
        assert first_24_per_zone["load_lag_24h"].isna().all()
