"""Unit tests for OpenMeteoClient — mocked HTTP, no network.

The library boundary is ``requests.get``. We monkeypatch it to a stand-in
that returns a synthetic JSON payload in the same shape Open-Meteo
emits. Live exercises against the real API live in ``tests/live/``.
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from energy_forecaster.adapters.weather_client.open_meteo import OpenMeteoClient
from energy_forecaster.application.errors import DataSourceUnavailableError
from energy_forecaster.application.ports.weather_client import WeatherClient
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _open_meteo_payload(times: list[str], **columns: list[Any]) -> dict[str, Any]:
    """Build a synthetic Open-Meteo response. Pass per-variable column lists."""
    hourly: dict[str, Any] = {"time": times}
    hourly.update(columns)
    return {"latitude": 50.1, "longitude": 8.7, "hourly": hourly}


class _FakeResponse:
    def __init__(self, payload: dict[str, Any] | None = None, status: int = 200) -> None:
        self._payload = payload or {}
        self._status = status

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")

    def json(self) -> dict[str, Any]:
        return self._payload


class _RecordingHTTP:
    """Captures every requests.get call and returns a preset response."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response: _FakeResponse = _FakeResponse({})
        self.raise_exception: Exception | None = None

    def __call__(self, url: str, *, params: dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if self.raise_exception is not None:
            raise self.raise_exception
        return self.response


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> _RecordingHTTP:
    # Patch via dotted-path string so mypy strict's no_implicit_reexport
    # never sees an attribute access on the adapter's `requests` import.
    fake = _RecordingHTTP()
    monkeypatch.setattr("energy_forecaster.adapters.weather_client.open_meteo.requests.get", fake)
    return fake


def _full_payload(times: list[str]) -> dict[str, Any]:
    """Synthesise a payload with all six required hourly variables present."""
    n = len(times)
    return _open_meteo_payload(
        times,
        temperature_2m=[15.0] * n,
        wind_speed_10m=[4.0] * n,
        wind_speed_100m=[8.0] * n,
        shortwave_radiation=[300.0] * n,
        cloud_cover=[50.0] * n,
        precipitation=[0.0] * n,
    )


class TestProtocolConformance:
    def test_satisfies_weather_client_protocol_structurally(self) -> None:
        client: WeatherClient = OpenMeteoClient()
        assert hasattr(client, "fetch_weather")


class TestRequestShape:
    def test_calls_archive_url_with_zone_specific_lat_lon(self, http: _RecordingHTTP) -> None:
        http.response = _FakeResponse(_full_payload(["2026-05-04T00:00"]))
        client = OpenMeteoClient()

        list(
            client.fetch_weather(
                zone=BiddingZone.DE_LU,
                start=_utc(2026, 5, 4),
                end=_utc(2026, 5, 4, 1),
            )
        )

        assert len(http.calls) == 1
        call = http.calls[0]
        assert call["url"].endswith("/v1/archive")
        # Frankfurt — the documented DE_LU representative point.
        assert call["params"]["latitude"] == 50.11
        assert call["params"]["longitude"] == 8.68
        assert call["params"]["start_date"] == "2026-05-04"
        assert call["params"]["end_date"] == "2026-05-04"
        assert call["params"]["wind_speed_unit"] == "ms"
        assert call["params"]["timezone"] == "UTC"

    def test_each_zone_maps_to_its_documented_lat_lon(self, http: _RecordingHTTP) -> None:
        http.response = _FakeResponse(_full_payload(["2026-05-04T00:00"]))
        client = OpenMeteoClient()
        expected_coords = {
            BiddingZone.DE_LU: (50.11, 8.68),
            BiddingZone.FR: (48.85, 2.35),
            BiddingZone.GB: (51.50, -0.13),
        }
        for zone, (lat, lon) in expected_coords.items():
            http.calls.clear()
            list(
                client.fetch_weather(
                    zone=zone,
                    start=_utc(2026, 5, 4),
                    end=_utc(2026, 5, 4, 1),
                )
            )
            assert http.calls[0]["params"]["latitude"] == lat
            assert http.calls[0]["params"]["longitude"] == lon


class TestResponseParsing:
    def test_payload_rows_become_weather_readings(self, http: _RecordingHTTP) -> None:
        http.response = _FakeResponse(
            _full_payload(["2026-05-04T00:00", "2026-05-04T01:00", "2026-05-04T02:00"])
        )
        client = OpenMeteoClient()

        readings = list(
            client.fetch_weather(
                zone=BiddingZone.DE_LU,
                start=_utc(2026, 5, 4),
                end=_utc(2026, 5, 4, 3),
            )
        )

        assert len(readings) == 3
        assert all(r.zone is BiddingZone.DE_LU for r in readings)
        assert [r.timestamp_utc for r in readings] == [
            _utc(2026, 5, 4, 0),
            _utc(2026, 5, 4, 1),
            _utc(2026, 5, 4, 2),
        ]

    def test_rows_outside_window_are_filtered(self, http: _RecordingHTTP) -> None:
        # Open-Meteo returns whole-day data for any (start_date, end_date),
        # so we must trim to the requested half-open window. Here we ask
        # for hours 1 and 2; the payload also includes 0 and 3 (outside).
        http.response = _FakeResponse(
            _full_payload(
                [
                    "2026-05-04T00:00",
                    "2026-05-04T01:00",
                    "2026-05-04T02:00",
                    "2026-05-04T03:00",
                ]
            )
        )
        client = OpenMeteoClient()

        readings = list(
            client.fetch_weather(
                zone=BiddingZone.DE_LU,
                start=_utc(2026, 5, 4, 1),
                end=_utc(2026, 5, 4, 3),
            )
        )

        assert [r.timestamp_utc for r in readings] == [
            _utc(2026, 5, 4, 1),
            _utc(2026, 5, 4, 2),
        ]

    def test_rows_with_any_null_field_are_skipped(self, http: _RecordingHTTP) -> None:
        # Open-Meteo encodes missing observations as JSON null. Letting
        # them through would make WeatherReading raise on a None float,
        # which is correct but noisy — quieter to skip at the boundary.
        http.response = _FakeResponse(
            _open_meteo_payload(
                ["2026-05-04T00:00", "2026-05-04T01:00", "2026-05-04T02:00"],
                temperature_2m=[15.0, None, 17.0],
                wind_speed_10m=[4.0, 4.0, 4.0],
                wind_speed_100m=[8.0, 8.0, 8.0],
                shortwave_radiation=[100.0, 200.0, 300.0],
                cloud_cover=[50.0, 50.0, 50.0],
                precipitation=[0.0, 0.0, 0.0],
            )
        )
        client = OpenMeteoClient()

        readings = list(
            client.fetch_weather(
                zone=BiddingZone.DE_LU,
                start=_utc(2026, 5, 4),
                end=_utc(2026, 5, 4, 3),
            )
        )

        assert len(readings) == 2
        assert {r.temp_c for r in readings} == {15.0, 17.0}

    def test_empty_or_missing_hourly_block_yields_no_readings(self, http: _RecordingHTTP) -> None:
        # Some Open-Meteo error responses still come back with HTTP 200
        # but no ``hourly`` block. We treat that as zero observations,
        # not as a failure — same contract as the load adapter's
        # NoMatchingDataError handling.
        http.response = _FakeResponse({"latitude": 50.1, "longitude": 8.7})
        client = OpenMeteoClient()

        readings = list(
            client.fetch_weather(
                zone=BiddingZone.DE_LU,
                start=_utc(2026, 5, 4),
                end=_utc(2026, 5, 4, 1),
            )
        )

        assert readings == []


class TestErrorTranslation:
    def test_request_exception_becomes_data_source_unavailable(self, http: _RecordingHTTP) -> None:
        http.raise_exception = RuntimeError("connection reset")
        client = OpenMeteoClient()

        with pytest.raises(DataSourceUnavailableError, match="connection reset"):
            list(
                client.fetch_weather(
                    zone=BiddingZone.DE_LU,
                    start=_utc(2026, 5, 4),
                    end=_utc(2026, 5, 5),
                )
            )

    def test_http_error_becomes_data_source_unavailable(self, http: _RecordingHTTP) -> None:
        http.response = _FakeResponse({}, status=503)
        client = OpenMeteoClient()

        with pytest.raises(DataSourceUnavailableError, match="HTTP 503"):
            list(
                client.fetch_weather(
                    zone=BiddingZone.GB,
                    start=_utc(2026, 5, 4),
                    end=_utc(2026, 5, 5),
                )
            )

    def test_wrapped_error_includes_zone(self, http: _RecordingHTTP) -> None:
        http.raise_exception = RuntimeError("boom")
        client = OpenMeteoClient()

        with pytest.raises(DataSourceUnavailableError, match="GB"):
            list(
                client.fetch_weather(
                    zone=BiddingZone.GB,
                    start=_utc(2026, 5, 4),
                    end=_utc(2026, 5, 5),
                )
            )
