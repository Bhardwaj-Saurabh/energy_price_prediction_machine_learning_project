"""Live tests against the real Open-Meteo archive endpoint.

Open-Meteo is keyless and free, so these tests need no credentials —
just network connectivity. They are still gated by ``pytest.mark.live``
because we don't want CI hammering Open-Meteo on every commit; the
unit tests above prove the parsing, mapping, and error translation.
"""

from datetime import UTC, datetime, timedelta

import pytest

from energy_forecaster.adapters.weather_client.open_meteo import OpenMeteoClient
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone

pytestmark = pytest.mark.live


def _recent_window() -> tuple[datetime, datetime]:
    # Use a week-old window so the archive endpoint is guaranteed to
    # have published data for it. The archive lags real time by ~5 days.
    end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=7)
    start = end - timedelta(hours=24)
    return start, end


@pytest.mark.parametrize("zone", [BiddingZone.DE_LU, BiddingZone.FR, BiddingZone.GB])
def test_real_open_meteo_returns_some_readings(zone: BiddingZone) -> None:
    start, end = _recent_window()
    readings = list(OpenMeteoClient().fetch_weather(zone=zone, start=start, end=end))

    assert len(readings) > 0
    assert all(r.zone is zone for r in readings)
    assert all(start <= r.timestamp_utc < end for r in readings)
