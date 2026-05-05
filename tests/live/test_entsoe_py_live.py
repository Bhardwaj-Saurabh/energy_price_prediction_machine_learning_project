"""Live tests against the real ENTSO-E Transparency Platform.

These run only when explicitly invoked (`make test-live`) and require a
valid ``EF_ENTSOE_API_KEY``. They are excluded from the default test run
because they hit the real network — slow, flakey, and would tie CI to
ENTSO-E's uptime.

The point of these tests is sanity, not coverage. They confirm that the
adapter's request shape and response parsing actually work against the
production API; the unit tests confirm everything else.
"""

from datetime import UTC, datetime, timedelta

import pytest

from energy_forecaster.adapters.entsoe_client.entsoe_py import EntsoePyClient
from energy_forecaster.config.settings import get_settings
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone

pytestmark = pytest.mark.live


@pytest.fixture
def adapter() -> EntsoePyClient:
    settings = get_settings()
    if settings.entsoe_api_key is None:
        pytest.skip("EF_ENTSOE_API_KEY is not set — skipping live test")
    return EntsoePyClient(api_key=settings.entsoe_api_key.get_secret_value())


def _recent_24h_window() -> tuple[datetime, datetime]:
    # ENTSO-E publishes data with a delay of a few hours. Using a window
    # that ends 24h ago avoids querying for slots that may not yet exist.
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(days=1)
    start = end - timedelta(hours=24)
    return start, end


@pytest.mark.parametrize("zone", [BiddingZone.DE_LU, BiddingZone.FR, BiddingZone.GB])
def test_real_entsoe_returns_some_load_observations(
    adapter: EntsoePyClient, zone: BiddingZone
) -> None:
    start, end = _recent_24h_window()
    observations = list(adapter.fetch_load(zone=zone, start=start, end=end))

    assert len(observations) > 0
    assert all(o.zone is zone for o in observations)
    assert all(start <= o.timestamp_utc <= end for o in observations)
