"""Clock port — the *only* sanctioned source of 'now' in the application layer."""

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    """An injectable clock returning the current instant in UTC.

    Direct calls to ``datetime.now()`` or ``time.time()`` are forbidden
    outside the SystemClock adapter. Tests pass a FakeClock with a fixed
    or programmatically-advanced ``now`` so time-windowed features and
    freshness checks become deterministic — that is the entire reason
    this port exists.
    """

    def now(self) -> datetime:
        """Return the current UTC datetime (timezone-aware)."""
        ...
