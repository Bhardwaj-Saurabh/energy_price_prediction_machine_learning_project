"""Tests for WeatherReadingSchema and to_weather_dataframe."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pandera.pandas as pa
import pytest

from energy_forecaster.contracts.weather_reading_schema import (
    WeatherReadingSchema,
    to_weather_dataframe,
)
from energy_forecaster.domain.entities.weather_reading import WeatherReading
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _hourly_weather(zone: BiddingZone, start: datetime, hours: int) -> list[WeatherReading]:
    return [
        WeatherReading(
            zone=zone,
            timestamp_utc=start + timedelta(hours=h),
            temp_c=15.0,
            wind_10m_ms=4.0,
            wind_100m_ms=8.0,
            ghi_wm2=300.0,
            cloud_cover_pct=50.0,
            precip_mm=0.0,
        )
        for h in range(hours)
    ]


class TestConverterHappyPath:
    def test_typical_readings_round_trip(self) -> None:
        readings = _hourly_weather(BiddingZone.DE_LU, _utc(2026, 5, 4), 2)

        df = to_weather_dataframe(readings)

        expected_columns = {
            "timestamp_utc",
            "zone",
            "temp_c",
            "wind_10m_ms",
            "wind_100m_ms",
            "ghi_wm2",
            "cloud_cover_pct",
            "precip_mm",
        }
        assert set(df.columns) == expected_columns
        assert len(df) == 2

    def test_empty_input_yields_empty_validated_frame(self) -> None:
        df = to_weather_dataframe([])
        assert df.empty
        WeatherReadingSchema.validate(df)


class TestSchemaRejection:
    def _base_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(["2026-05-04 00:00", "2026-05-04 01:00"], utc=True),
                "zone": ["DE_LU", "DE_LU"],
                "temp_c": [15.0, 16.0],
                "wind_10m_ms": [4.0, 4.5],
                "wind_100m_ms": [8.0, 8.5],
                "ghi_wm2": [300.0, 350.0],
                "cloud_cover_pct": [50.0, 55.0],
                "precip_mm": [0.0, 0.5],
            }
        )

    @pytest.mark.parametrize(
        ("column", "bad_value"),
        [
            ("temp_c", -100.0),  # below MIN_TEMP_C
            ("temp_c", 100.0),  # above MAX_TEMP_C
            ("wind_10m_ms", -1.0),
            ("wind_10m_ms", 200.0),
            ("wind_100m_ms", 200.0),
            ("ghi_wm2", -1.0),
            ("ghi_wm2", 2000.0),
            ("cloud_cover_pct", -1.0),
            ("cloud_cover_pct", 101.0),
            ("precip_mm", -1.0),
            ("precip_mm", 600.0),
        ],
    )
    def test_out_of_range_value_is_rejected(self, column: str, bad_value: float) -> None:
        df = self._base_frame()
        df.loc[1, column] = bad_value
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            WeatherReadingSchema.validate(df)

    def test_unknown_zone_is_rejected(self) -> None:
        df = self._base_frame()
        df.loc[1, "zone"] = "ES"
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            WeatherReadingSchema.validate(df)

    def test_naive_timestamp_column_is_rejected(self) -> None:
        df = self._base_frame()
        df["timestamp_utc"] = pd.to_datetime(["2026-05-04 00:00", "2026-05-04 01:00"])
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            WeatherReadingSchema.validate(df)

    def test_extra_column_is_rejected(self) -> None:
        df = self._base_frame()
        df["extra"] = [0.0, 0.0]
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            WeatherReadingSchema.validate(df)

    def test_duplicate_zone_timestamp_pair_is_rejected(self) -> None:
        df = self._base_frame()
        df.loc[1, "timestamp_utc"] = df.loc[0, "timestamp_utc"]
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            WeatherReadingSchema.validate(df)
