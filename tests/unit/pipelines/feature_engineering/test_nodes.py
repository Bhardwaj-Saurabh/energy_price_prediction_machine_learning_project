"""Unit tests for the feature-engineering nodes (pure DataFrame functions)."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pandera.pandas as pa
import pytest

from energy_forecaster.contracts.feature_matrix_schema import FeatureMatrixSchema
from energy_forecaster.pipelines.feature_engineering.nodes import (
    add_lag_features,
    add_time_features,
    join_load_and_weather,
)


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _hourly_load_frame(zone: str, start: datetime, hours: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [start + timedelta(hours=h) for h in range(hours)], utc=True
            ).astype("datetime64[ns, UTC]"),
            "zone": [zone] * hours,
            "load_mw": [50_000.0 + 100.0 * h for h in range(hours)],
        }
    )


def _hourly_weather_frame(zone: str, start: datetime, hours: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [start + timedelta(hours=h) for h in range(hours)], utc=True
            ).astype("datetime64[ns, UTC]"),
            "zone": [zone] * hours,
            "temp_c": [15.0] * hours,
            "wind_10m_ms": [4.0] * hours,
            "wind_100m_ms": [8.0] * hours,
            "ghi_wm2": [300.0] * hours,
            "cloud_cover_pct": [50.0] * hours,
            "precip_mm": [0.0] * hours,
        }
    )


class TestJoin:
    def test_inner_join_drops_rows_with_missing_weather(self) -> None:
        load = _hourly_load_frame("DE_LU", _utc(2026, 5, 4), 5)
        weather = _hourly_weather_frame("DE_LU", _utc(2026, 5, 4), 3)

        joined = join_load_and_weather(load, weather)
        assert len(joined) == 3

    def test_join_keeps_zone_and_timestamp_columns(self) -> None:
        load = _hourly_load_frame("DE_LU", _utc(2026, 5, 4), 2)
        weather = _hourly_weather_frame("DE_LU", _utc(2026, 5, 4), 2)

        joined = join_load_and_weather(load, weather)
        assert {"zone", "timestamp_utc", "load_mw", "temp_c"}.issubset(joined.columns)

    def test_no_overlap_yields_empty_result(self) -> None:
        load = _hourly_load_frame("DE_LU", _utc(2026, 5, 4), 2)
        weather = _hourly_weather_frame("FR", _utc(2026, 5, 4), 2)  # different zone

        joined = join_load_and_weather(load, weather)
        assert joined.empty


class TestTimeFeatures:
    def test_hour_of_day_extracted_correctly(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(
                    ["2026-05-04 00:00", "2026-05-04 12:00", "2026-05-04 23:00"],
                    utc=True,
                ),
                "zone": ["DE_LU"] * 3,
            }
        )
        out = add_time_features(df)
        assert out["hour_of_day"].tolist() == [0, 12, 23]

    def test_day_of_week_uses_monday_zero_convention(self) -> None:
        # 2026-05-04 is a Monday → 0; 2026-05-10 is a Sunday → 6.
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(["2026-05-04 00:00", "2026-05-10 00:00"], utc=True),
                "zone": ["DE_LU"] * 2,
            }
        )
        out = add_time_features(df)
        assert out["day_of_week"].tolist() == [0, 6]

    def test_is_weekend_set_for_saturday_and_sunday_only(self) -> None:
        # Friday (4), Saturday (5), Sunday (6), Monday (0).
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(
                    [
                        "2026-05-08 00:00",  # Fri
                        "2026-05-09 00:00",  # Sat
                        "2026-05-10 00:00",  # Sun
                        "2026-05-11 00:00",  # Mon
                    ],
                    utc=True,
                ),
                "zone": ["DE_LU"] * 4,
            }
        )
        out = add_time_features(df)
        assert out["is_weekend"].tolist() == [False, True, True, False]


class TestLagFeatures:
    def _seed_for_lags(self, hours: int) -> pd.DataFrame:
        df = _hourly_load_frame("DE_LU", _utc(2026, 5, 4), hours)
        df_w = _hourly_weather_frame("DE_LU", _utc(2026, 5, 4), hours)
        joined = join_load_and_weather(df, df_w)
        return add_time_features(joined)

    def test_first_24_hours_have_null_24h_lag(self) -> None:
        df = self._seed_for_lags(48)
        out = add_lag_features(df)
        # First 24 rows of the (only) zone have no 24h-prior value.
        assert out["load_lag_24h"].iloc[:24].isna().all()
        # From row 24 onward, the 24h lag matches the value 24 hours earlier.
        assert (
            out["load_lag_24h"].iloc[24:].to_numpy() == out["load_mw"].iloc[:24].to_numpy()
        ).all()

    def test_lags_do_not_bleed_across_zones(self) -> None:
        # If zone DE_LU has 5 rows and FR has 5 rows, FR's first row's 1h
        # lag must be NaN — it must not pick up DE_LU's last value.
        de_load = _hourly_load_frame("DE_LU", _utc(2026, 5, 4), 5)
        fr_load = _hourly_load_frame("FR", _utc(2026, 5, 4), 5)
        de_w = _hourly_weather_frame("DE_LU", _utc(2026, 5, 4), 5)
        fr_w = _hourly_weather_frame("FR", _utc(2026, 5, 4), 5)
        joined = pd.concat(
            [
                join_load_and_weather(de_load, de_w),
                join_load_and_weather(fr_load, fr_w),
            ]
        ).reset_index(drop=True)
        with_time = add_time_features(joined)

        out = add_lag_features(with_time)

        # First row per zone has no 1h lag.
        first_per_zone = out.sort_values(["zone", "timestamp_utc"]).groupby("zone").head(1)
        assert first_per_zone["load_lag_1h"].isna().all()

    def test_output_validates_against_feature_matrix_schema(self) -> None:
        # The @pa.check_types decorator on add_lag_features validates the
        # output. Calling it without raising is the assertion.
        df = self._seed_for_lags(200)  # enough rows for all three lag windows
        out = add_lag_features(df)
        FeatureMatrixSchema.validate(out)

    def test_invalid_input_fails_validation(self) -> None:
        # An out-of-range temperature must fail at the schema boundary,
        # not silently propagate downstream.
        df = self._seed_for_lags(48)
        df.loc[0, "temp_c"] = 200.0  # above MAX_TEMP_C
        # Pandera's @pa.check_types runs on the *return* value; since this
        # node does not modify temp_c, the bad value is preserved and the
        # decorator raises.
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            add_lag_features(df)
