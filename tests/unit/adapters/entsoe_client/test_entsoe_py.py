"""Unit tests for EntsoePyClient.

We do not call real ENTSO-E here. The library boundary is the adapter's
private ``_client`` attribute; we replace it with a stand-in whose
``query_load`` returns a synthetic pandas DataFrame in the same shape
the real library produces. Live exercises against the actual API live
in ``tests/live/`` and run only on demand.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from entsoe.exceptions import NoMatchingDataError

from energy_forecaster.adapters.entsoe_client.entsoe_py import EntsoePyClient
from energy_forecaster.application.errors import DataSourceUnavailableError
from energy_forecaster.application.ports.entsoe_client import EntsoeClient
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW


def _hourly_index_in_local_time(start_utc: datetime, hours: int, tz: str) -> pd.DatetimeIndex:
    """Build a DatetimeIndex whose underlying UTC instants are
    ``start_utc, start_utc+1h, …`` but whose label timezone is ``tz`` —
    this is what the real entsoe-py returns (country-local labels)."""
    timestamps = [start_utc + timedelta(hours=h) for h in range(hours)]
    return pd.DatetimeIndex([pd.Timestamp(ts) for ts in timestamps]).tz_convert(tz)


def _fake_load_dataframe(
    start_utc: datetime,
    hours: int,
    *,
    values: list[float] | None = None,
    tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    if values is None:
        values = [50_000.0 + 100.0 * h for h in range(hours)]
    return pd.DataFrame(
        {"Actual Load": values},
        index=_hourly_index_in_local_time(start_utc, hours, tz),
    )


class _RecordingFakeClient:
    """Stand-in for EntsoePandasClient used in tests.

    Records calls so we can assert the country code mapping, and either
    returns a preset DataFrame or raises a preset exception. Not a mock
    framework — just a small object that satisfies the same signature.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._return: pd.DataFrame | None = None
        self._raise: BaseException | None = None

    def returns(self, df: pd.DataFrame) -> None:
        self._return = df
        self._raise = None

    def raises(self, exc: BaseException) -> None:
        self._return = None
        self._raise = exc

    def query_load(
        self, *, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        self.calls.append({"country_code": country_code, "start": start, "end": end})
        if self._raise is not None:
            raise self._raise
        assert self._return is not None
        return self._return


@pytest.fixture
def adapter_with_fake() -> tuple[EntsoePyClient, _RecordingFakeClient]:
    adapter = EntsoePyClient(api_key="test-key")
    fake = _RecordingFakeClient()
    # Boundary is the library; substitute the wrapped object so tests do not
    # touch the real EntsoePandasClient.
    adapter._client = fake  # type: ignore[assignment]
    return adapter, fake


class TestProtocolConformance:
    def test_satisfies_the_entsoe_client_protocol_structurally(self) -> None:
        adapter: EntsoeClient = EntsoePyClient(api_key="test-key")
        assert hasattr(adapter, "fetch_load")


class TestZoneCodeMapping:
    @pytest.mark.parametrize(
        ("zone", "expected_code"),
        [
            (BiddingZone.DE_LU, "DE_LU"),
            (BiddingZone.FR, "FR"),
            (BiddingZone.GB, "GB"),
        ],
    )
    def test_each_bidding_zone_maps_to_its_entsoe_country_code(
        self,
        adapter_with_fake: tuple[EntsoePyClient, _RecordingFakeClient],
        zone: BiddingZone,
        expected_code: str,
    ) -> None:
        adapter, fake = adapter_with_fake
        fake.returns(_fake_load_dataframe(datetime(2026, 5, 4, tzinfo=UTC), 1))

        list(
            adapter.fetch_load(
                zone=zone,
                start=datetime(2026, 5, 4, tzinfo=UTC),
                end=datetime(2026, 5, 4, 1, tzinfo=UTC),
            )
        )

        assert fake.calls[0]["country_code"] == expected_code


class TestSuccessfulQuery:
    def test_dataframe_rows_become_load_observations(
        self, adapter_with_fake: tuple[EntsoePyClient, _RecordingFakeClient]
    ) -> None:
        adapter, fake = adapter_with_fake
        fake.returns(
            _fake_load_dataframe(
                datetime(2026, 5, 4, tzinfo=UTC), 3, values=[50_000.0, 51_000.0, 52_000.0]
            )
        )

        observations = list(
            adapter.fetch_load(
                zone=BiddingZone.DE_LU,
                start=datetime(2026, 5, 4, tzinfo=UTC),
                end=datetime(2026, 5, 4, 3, tzinfo=UTC),
            )
        )

        assert [o.load for o in observations] == [
            EnergyMW(50_000.0),
            EnergyMW(51_000.0),
            EnergyMW(52_000.0),
        ]
        assert all(o.zone is BiddingZone.DE_LU for o in observations)

    def test_local_time_indices_are_converted_to_utc(
        self, adapter_with_fake: tuple[EntsoePyClient, _RecordingFakeClient]
    ) -> None:
        # entsoe-py returns Europe/Berlin-localised labels for DE_LU.
        # The underlying instants are 00:00, 01:00 UTC; the labels in
        # Berlin during May (CEST, UTC+2) are 02:00, 03:00. The adapter
        # must surface them back as UTC instants on the LoadObservation.
        adapter, fake = adapter_with_fake
        fake.returns(_fake_load_dataframe(datetime(2026, 5, 4, tzinfo=UTC), 2, tz="Europe/Berlin"))

        observations = list(
            adapter.fetch_load(
                zone=BiddingZone.DE_LU,
                start=datetime(2026, 5, 4, tzinfo=UTC),
                end=datetime(2026, 5, 4, 2, tzinfo=UTC),
            )
        )

        assert [o.timestamp_utc for o in observations] == [
            datetime(2026, 5, 4, 0, tzinfo=UTC),
            datetime(2026, 5, 4, 1, tzinfo=UTC),
        ]


class TestNaNFiltering:
    def test_rows_with_nan_load_are_skipped(
        self, adapter_with_fake: tuple[EntsoePyClient, _RecordingFakeClient]
    ) -> None:
        # ENTSO-E occasionally publishes incomplete rows. Filtering at
        # the adapter keeps the use case from seeing partial data and
        # avoids EnergyMW raising on NaN downstream.
        adapter, fake = adapter_with_fake
        fake.returns(
            _fake_load_dataframe(
                datetime(2026, 5, 4, tzinfo=UTC),
                3,
                values=[50_000.0, float("nan"), 52_000.0],
            )
        )

        observations = list(
            adapter.fetch_load(
                zone=BiddingZone.DE_LU,
                start=datetime(2026, 5, 4, tzinfo=UTC),
                end=datetime(2026, 5, 4, 3, tzinfo=UTC),
            )
        )

        assert len(observations) == 2
        assert [o.load.value for o in observations] == [50_000.0, 52_000.0]


class TestEmptyResponse:
    def test_no_matching_data_yields_zero_observations_not_an_error(
        self, adapter_with_fake: tuple[EntsoePyClient, _RecordingFakeClient]
    ) -> None:
        # ENTSO-E returns an explicit "no data" response for windows it
        # has no data for (often: very recent slots, or very old ones).
        # The use case's contract treats empty windows as a normal
        # outcome — so the adapter must NOT propagate it as a failure.
        adapter, fake = adapter_with_fake
        fake.raises(NoMatchingDataError("No data for window"))

        observations = list(
            adapter.fetch_load(
                zone=BiddingZone.DE_LU,
                start=datetime(2026, 5, 4, tzinfo=UTC),
                end=datetime(2026, 5, 5, tzinfo=UTC),
            )
        )

        assert observations == []


class TestErrorTranslation:
    def test_unexpected_exception_becomes_data_source_unavailable_error(
        self, adapter_with_fake: tuple[EntsoePyClient, _RecordingFakeClient]
    ) -> None:
        # HTTP errors, network failures, parse errors — anything we did
        # not specifically expect — are turned into a layer-neutral
        # ApplicationError so the use case can react without depending on
        # the specifics of any third-party library.
        adapter, fake = adapter_with_fake
        fake.raises(RuntimeError("connection reset"))

        with pytest.raises(DataSourceUnavailableError, match="connection reset"):
            list(
                adapter.fetch_load(
                    zone=BiddingZone.DE_LU,
                    start=datetime(2026, 5, 4, tzinfo=UTC),
                    end=datetime(2026, 5, 5, tzinfo=UTC),
                )
            )

    def test_wrapped_error_includes_zone_in_message(
        self, adapter_with_fake: tuple[EntsoePyClient, _RecordingFakeClient]
    ) -> None:
        adapter, fake = adapter_with_fake
        fake.raises(RuntimeError("boom"))

        with pytest.raises(DataSourceUnavailableError, match="GB"):
            list(
                adapter.fetch_load(
                    zone=BiddingZone.GB,
                    start=datetime(2026, 5, 4, tzinfo=UTC),
                    end=datetime(2026, 5, 5, tzinfo=UTC),
                )
            )
