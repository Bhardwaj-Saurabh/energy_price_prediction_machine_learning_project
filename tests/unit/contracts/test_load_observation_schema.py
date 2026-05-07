"""Tests for LoadObservationSchema and to_load_dataframe."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pandera.pandas as pa
import pytest

from energy_forecaster.contracts.load_observation_schema import (
    LoadObservationSchema,
    to_load_dataframe,
)
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _hourly_load(zone: BiddingZone, start: datetime, hours: int) -> list[LoadObservation]:
    return [
        LoadObservation(
            zone=zone,
            timestamp_utc=start + timedelta(hours=h),
            load=EnergyMW(50_000.0 + h * 100.0),
        )
        for h in range(hours)
    ]


class TestConverterHappyPath:
    def test_typical_observations_round_trip(self) -> None:
        observations = _hourly_load(BiddingZone.DE_LU, _utc(2026, 5, 4), 3)

        df = to_load_dataframe(observations)

        assert list(df.columns) == ["timestamp_utc", "zone", "load_mw"]
        assert len(df) == 3
        assert df["zone"].tolist() == ["DE_LU"] * 3
        assert df["load_mw"].tolist() == [50_000.0, 50_100.0, 50_200.0]

    def test_empty_input_yields_empty_validated_frame(self) -> None:
        df = to_load_dataframe([])
        assert df.empty
        assert list(df.columns) == ["timestamp_utc", "zone", "load_mw"]
        # Schema validation passes on the empty frame too.
        LoadObservationSchema.validate(df)

    def test_multiple_zones_in_one_frame(self) -> None:
        observations = [
            *_hourly_load(BiddingZone.DE_LU, _utc(2026, 5, 4), 2),
            *_hourly_load(BiddingZone.FR, _utc(2026, 5, 4), 3),
        ]
        df = to_load_dataframe(observations)
        assert df["zone"].tolist() == ["DE_LU", "DE_LU", "FR", "FR", "FR"]


class TestSchemaTypeCoercion:
    def test_pandas_default_microsecond_unit_is_coerced_to_nanoseconds(self) -> None:
        # pandas 3.x defaults to ``datetime64[us, UTC]``. The schema asks
        # for nanoseconds; ``coerce=True`` performs the cast so producers
        # don't have to think about it.
        observations = _hourly_load(BiddingZone.DE_LU, _utc(2026, 5, 4), 1)
        df = to_load_dataframe(observations)
        assert str(df["timestamp_utc"].dtype) == "datetime64[ns, UTC]"


class TestSchemaRejection:
    def _base_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(["2026-05-04 00:00", "2026-05-04 01:00"], utc=True),
                "zone": ["DE_LU", "DE_LU"],
                "load_mw": [50_000.0, 51_000.0],
            }
        )

    def test_unknown_zone_is_rejected(self) -> None:
        df = self._base_frame()
        df.loc[1, "zone"] = "ES"  # not a supported BiddingZone
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            LoadObservationSchema.validate(df)

    def test_negative_load_is_rejected(self) -> None:
        df = self._base_frame()
        df.loc[1, "load_mw"] = -1.0
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            LoadObservationSchema.validate(df)

    def test_load_above_plausibility_cap_is_rejected(self) -> None:
        df = self._base_frame()
        df.loc[1, "load_mw"] = 250_000.0  # above MAX_PLAUSIBLE_LOAD_MW
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            LoadObservationSchema.validate(df)

    def test_naive_timestamp_column_is_rejected(self) -> None:
        df = self._base_frame()
        df["timestamp_utc"] = pd.to_datetime(["2026-05-04 00:00", "2026-05-04 01:00"])
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            LoadObservationSchema.validate(df)

    def test_extra_column_is_rejected(self) -> None:
        df = self._base_frame()
        df["extra"] = [0, 0]
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            LoadObservationSchema.validate(df)

    def test_duplicate_zone_timestamp_pair_is_rejected(self) -> None:
        df = self._base_frame()
        # Make both rows the same identity — composite uniqueness must reject.
        df.loc[1, "timestamp_utc"] = df.loc[0, "timestamp_utc"]
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            LoadObservationSchema.validate(df)
