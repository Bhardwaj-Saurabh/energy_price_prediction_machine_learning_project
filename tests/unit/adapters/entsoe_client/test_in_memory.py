"""Unit tests for InMemoryEntsoeClient.

The synthetic generator is pure — given the same zone and window it
returns the same observations. These tests lock in the half-open window
contract, the hourly cadence, and the diurnal-pattern shape so the
demo data does not silently change.
"""

from datetime import UTC, datetime, timedelta
from math import pi, sin

from energy_forecaster.adapters.entsoe_client.in_memory import InMemoryEntsoeClient
from energy_forecaster.application.ports.entsoe_client import EntsoeClient
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def test_satisfies_the_entsoe_client_protocol_structurally() -> None:
    client: EntsoeClient = InMemoryEntsoeClient()
    observations = list(
        client.fetch_load(
            zone=BiddingZone.DE_LU,
            start=_utc(2026, 5, 4),
            end=_utc(2026, 5, 4, 1),
        )
    )
    assert len(observations) == 1


class TestWindowContract:
    def test_returns_one_observation_per_hour_in_window(self) -> None:
        client = InMemoryEntsoeClient()
        observations = list(
            client.fetch_load(
                zone=BiddingZone.DE_LU,
                start=_utc(2026, 5, 4),
                end=_utc(2026, 5, 5),
            )
        )
        assert len(observations) == 24

    def test_window_is_half_open(self) -> None:
        # The end timestamp itself must be excluded — same convention as the
        # real ENTSO-E API and the use case's test fixtures.
        client = InMemoryEntsoeClient()
        observations = list(
            client.fetch_load(
                zone=BiddingZone.DE_LU,
                start=_utc(2026, 5, 4, 10),
                end=_utc(2026, 5, 4, 12),
            )
        )
        assert [o.timestamp_utc for o in observations] == [
            _utc(2026, 5, 4, 10),
            _utc(2026, 5, 4, 11),
        ]

    def test_subhour_start_is_floored_to_next_hour_boundary(self) -> None:
        # We forecast hourly slots; readings off the hour are nonsensical.
        # A start of 10:30 should produce the 11:00 reading next.
        client = InMemoryEntsoeClient()
        observations = list(
            client.fetch_load(
                zone=BiddingZone.DE_LU,
                start=_utc(2026, 5, 4, 10) + timedelta(minutes=30),
                end=_utc(2026, 5, 4, 13),
            )
        )
        assert [o.timestamp_utc for o in observations] == [
            _utc(2026, 5, 4, 11),
            _utc(2026, 5, 4, 12),
        ]

    def test_empty_window_yields_no_observations(self) -> None:
        client = InMemoryEntsoeClient()
        observations = list(
            client.fetch_load(
                zone=BiddingZone.DE_LU,
                start=_utc(2026, 5, 4, 10),
                end=_utc(2026, 5, 4, 10),
            )
        )
        assert observations == []


class TestSyntheticPattern:
    def test_value_at_each_hour_matches_the_documented_formula(self) -> None:
        # The CLI demo, dashboards, and humans inspecting the JSONL all
        # rely on this exact shape. Pinning it makes any drift visible.
        client = InMemoryEntsoeClient()
        baseline = 60_000.0  # DE_LU baseline in the module
        amplitude = 15_000.0
        for hour in (0, 6, 12, 18):
            obs = next(
                iter(
                    client.fetch_load(
                        zone=BiddingZone.DE_LU,
                        start=_utc(2026, 5, 4, hour),
                        end=_utc(2026, 5, 4, hour + 1),
                    )
                )
            )
            expected = round(baseline + amplitude * sin(2 * pi * hour / 24.0), 2)
            assert obs.load == EnergyMW(expected)

    def test_each_zone_has_distinct_baseline(self) -> None:
        client = InMemoryEntsoeClient()
        # At 06:00 the diurnal modulation is sin(pi/2) == 1, so the value
        # is baseline + amplitude — easiest hour to differentiate zones.
        loads_at_06 = {
            zone: next(
                iter(
                    client.fetch_load(
                        zone=zone,
                        start=_utc(2026, 5, 4, 6),
                        end=_utc(2026, 5, 4, 7),
                    )
                )
            ).load
            for zone in (BiddingZone.DE_LU, BiddingZone.FR, BiddingZone.GB)
        }
        # All distinct (different baselines).
        assert len({load.value for load in loads_at_06.values()}) == 3
        # Ordering matches the baselines (DE > FR > GB).
        assert (
            loads_at_06[BiddingZone.DE_LU].value
            > loads_at_06[BiddingZone.FR].value
            > loads_at_06[BiddingZone.GB].value
        )

    def test_all_loads_are_within_the_domain_bounds(self) -> None:
        # EnergyMW would have rejected at construction — this test is a
        # belt-and-braces guard that the synthetic amplitude never pushes
        # the smallest baseline (GB) below zero or any baseline above the
        # plausibility cap.
        client = InMemoryEntsoeClient()
        for zone in (BiddingZone.DE_LU, BiddingZone.FR, BiddingZone.GB):
            for obs in client.fetch_load(
                zone=zone,
                start=_utc(2026, 5, 4),
                end=_utc(2026, 5, 5),
            ):
                assert 0 <= obs.load.value <= 200_000
