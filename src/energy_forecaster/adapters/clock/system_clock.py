"""SystemClock — production adapter that reads real wall-clock time."""

from datetime import UTC, datetime


class SystemClock:
    """Reads the real wall clock and returns the current UTC datetime.

    This is the *only* place outside of test fakes where ``datetime.now()``
    may be called. Every other module accesses time through the
    :class:`Clock` port, which lets us inject a deterministic FakeClock in
    tests without touching production code.

    The class has no state and no constructor parameters — instances are
    interchangeable. We still make it a class (rather than a free
    function) so it conforms structurally to the Clock Protocol and is
    instantiable from the composition root with the same syntax as every
    other adapter.
    """

    def now(self) -> datetime:
        return datetime.now(UTC)
