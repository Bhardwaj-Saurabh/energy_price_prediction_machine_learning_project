"""Unit tests for InMemoryWeatherClient — the synthetic demo adapter."""

from datetime import UTC, datetime, timedelta

from energy_forecaster.adapters.weather_client.in_memory import InMemoryWeatherClient
from energy_forecaster.application.ports.weather_client import WeatherClient
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


class TestProtocolConformance:
    def test_satisfies_weather_client_protocol_structurally(self) -> None:
        client: WeatherClient = InMemoryWeatherClient()
        assert hasattr(client, "fetch_weather")


class TestWindowContract:
    def test_returns_one_reading_per_hour_in_window(self) -> None:
        client = InMemoryWeatherClient()
        readings = list(
            client.fetch_weather(
                zone=BiddingZone.DE_LU,
                start=_utc(2026, 5, 4),
                end=_utc(2026, 5, 5),
            )
        )
        assert len(readings) == 24

    def test_window_is_half_open(self) -> None:
        client = InMemoryWeatherClient()
        readings = list(
            client.fetch_weather(
                zone=BiddingZone.DE_LU,
                start=_utc(2026, 5, 4, 10),
                end=_utc(2026, 5, 4, 12),
            )
        )
        assert [r.timestamp_utc for r in readings] == [
            _utc(2026, 5, 4, 10),
            _utc(2026, 5, 4, 11),
        ]

    def test_subhour_start_floors_to_next_hour(self) -> None:
        client = InMemoryWeatherClient()
        readings = list(
            client.fetch_weather(
                zone=BiddingZone.DE_LU,
                start=_utc(2026, 5, 4, 10) + timedelta(minutes=15),
                end=_utc(2026, 5, 4, 13),
            )
        )
        assert [r.timestamp_utc for r in readings] == [
            _utc(2026, 5, 4, 11),
            _utc(2026, 5, 4, 12),
        ]


class TestSyntheticPattern:
    def test_each_zone_has_distinct_temperature_baseline(self) -> None:
        # Three zones, three baselines — easy to differentiate at any
        # given hour.
        client = InMemoryWeatherClient()
        temps_at_06 = {
            zone: next(
                iter(
                    client.fetch_weather(
                        zone=zone,
                        start=_utc(2026, 5, 4, 6),
                        end=_utc(2026, 5, 4, 7),
                    )
                )
            ).temp_c
            for zone in (BiddingZone.DE_LU, BiddingZone.FR, BiddingZone.GB)
        }
        assert len(set(temps_at_06.values())) == 3

    def test_solar_irradiance_is_zero_at_midnight(self) -> None:
        # Confirm the documented "ghi=0 at night" property — sin(0) is 0.
        client = InMemoryWeatherClient()
        midnight = next(
            iter(
                client.fetch_weather(
                    zone=BiddingZone.DE_LU,
                    start=_utc(2026, 5, 4, 0),
                    end=_utc(2026, 5, 4, 1),
                )
            )
        )
        assert midnight.ghi_wm2 == 0.0

    def test_all_fields_are_within_domain_bounds(self) -> None:
        client = InMemoryWeatherClient()
        for zone in (BiddingZone.DE_LU, BiddingZone.FR, BiddingZone.GB):
            for r in client.fetch_weather(zone=zone, start=_utc(2026, 5, 4), end=_utc(2026, 5, 5)):
                # WeatherReading would have rejected at construction; this
                # is a belt-and-braces check that the synthetic curves
                # never push any field out of the plausible range.
                assert -60 <= r.temp_c <= 60
                assert 0 <= r.wind_10m_ms <= 100
                assert 0 <= r.wind_100m_ms <= 100
                assert 0 <= r.ghi_wm2 <= 1500
                assert 0 <= r.cloud_cover_pct <= 100
                assert 0 <= r.precip_mm <= 500
