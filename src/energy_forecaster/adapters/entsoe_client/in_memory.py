"""InMemoryEntsoeClient — synthetic-data ENTSO-E adapter for local demos.

This is a *real* adapter that satisfies :class:`EntsoeClient` — not a test
double. It exists so the local CLI is exercisable end-to-end before the
HTTP-backed :class:`EntsoePyClient` (chunk 5c) is wired up. The data it
returns is deterministic and follows a plausible-looking sinusoidal
daily pattern around a zone-specific baseline; it should NOT be used to
train or evaluate models.

When the real adapter lands, the composition root will pick this one
only when ``settings.entsoe_api_key`` is unset, so demos still work
without credentials. Tests of the use case continue to use
``FakeEntsoeClient`` from ``tests/unit/application/fakes.py`` because it
exposes test-control methods (``seed``, ``fail_on``) that have no place
on a production adapter.
"""

from collections.abc import Iterable
from datetime import datetime, timedelta
from math import pi, sin

from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW

# Plausible bidding-zone baselines (rough peak-demand numbers, not precise).
# Adding a new zone without an entry here will fail loudly with KeyError —
# that is the desired behaviour. Defensive fallbacks would mask the bug.
_ZONE_BASELINES_MW: dict[BiddingZone, float] = {
    BiddingZone.DE_LU: 60_000.0,
    BiddingZone.FR: 50_000.0,
    BiddingZone.GB: 35_000.0,
}

# Amplitude of the synthetic diurnal cycle. The minimum baseline (35_000
# MW, GB) minus this amplitude must stay positive — EnergyMW rejects
# negative loads. 15_000 keeps GB in [20_000, 50_000] which satisfies the
# domain constraint.
_DIURNAL_AMPLITUDE_MW: float = 15_000.0


class InMemoryEntsoeClient:
    """Generates synthetic hourly load observations on demand, in-process."""

    def fetch_load(
        self,
        *,
        zone: BiddingZone,
        start: datetime,
        end: datetime,
    ) -> Iterable[LoadObservation]:
        baseline = _ZONE_BASELINES_MW[zone]
        observations: list[LoadObservation] = []
        cursor = _floor_to_hour(start)
        # Drop observations strictly before the requested start to honour the
        # half-open ``[start, end)`` contract documented on the port.
        if cursor < start:
            cursor += timedelta(hours=1)
        while cursor < end:
            modulation = sin(2.0 * pi * cursor.hour / 24.0)
            value = baseline + _DIURNAL_AMPLITUDE_MW * modulation
            observations.append(
                LoadObservation(
                    zone=zone,
                    timestamp_utc=cursor,
                    load=EnergyMW(round(value, 2)),
                )
            )
            cursor += timedelta(hours=1)
        return observations


def _floor_to_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)
