"""Forecast horizon expressed as an integer number of hours."""

from dataclasses import dataclass

# Day-ahead forecasting is the primary product (24h). We allow up to one week
# (168h) to leave room for week-ahead variants. Any horizon outside this range
# is almost certainly a bug — extending the range is a deliberate decision,
# not a quiet runtime widening.
MIN_HORIZON_HOURS: int = 1
MAX_HORIZON_HOURS: int = 168


@dataclass(frozen=True, slots=True)
class HorizonHours:
    """A forecast horizon in whole hours, between 1 and 168 inclusive.

    The runtime ``isinstance`` checks defend against payloads that bypass
    static typing (JSON inputs, YAML config). bool is rejected explicitly
    because ``True``/``False`` are instances of int in Python and we never
    want them silently coerced into a horizon.
    """

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError(f"HorizonHours must be int, got {type(self.value).__name__}")
        if not (MIN_HORIZON_HOURS <= self.value <= MAX_HORIZON_HOURS):
            raise ValueError(
                f"HorizonHours must be between {MIN_HORIZON_HOURS} and "
                f"{MAX_HORIZON_HOURS}, got {self.value}"
            )
