"""Integration test for SystemClock — touches real wall-clock time."""

from datetime import UTC, datetime, timedelta

import pytest

from energy_forecaster.adapters.clock.system_clock import SystemClock
from energy_forecaster.application.ports.clock import Clock

pytestmark = pytest.mark.integration


def test_now_returns_a_recent_utc_datetime() -> None:
    # Bracket the call by real time queries from outside the adapter to
    # confirm SystemClock isn't lagging or jumping ahead. Using a 1-second
    # window is generous — the actual gap is sub-millisecond.
    before = datetime.now(UTC)
    result = SystemClock().now()
    after = datetime.now(UTC)

    assert result.tzinfo is not None
    assert result.utcoffset() == timedelta(0)
    assert before <= result <= after


def test_satisfies_the_clock_protocol_structurally() -> None:
    # The annotation `clock: Clock = SystemClock()` is a structural
    # assignment — mypy verifies SystemClock implements every method on
    # the Protocol, but runtime would happily accept a wrong type. The
    # explicit `now()` call below is what locks in the actual contract.
    clock: Clock = SystemClock()
    result = clock.now()
    assert isinstance(result, datetime)
