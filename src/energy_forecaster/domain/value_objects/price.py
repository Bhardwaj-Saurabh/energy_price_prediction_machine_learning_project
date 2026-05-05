"""Wholesale electricity price in EUR per MWh."""

from dataclasses import dataclass
from math import isfinite

# Wholesale electricity prices CAN go negative — when generation (often
# renewables) exceeds demand and curtailment is impossible, operators pay
# consumers to take the energy. Historic ENTSO-E records show prices roughly
# in the range -€500 to +€4,000/MWh; the bounds below give comfortable
# headroom for future market behaviour while still rejecting obvious
# data corruption.
MAX_PLAUSIBLE_PRICE_EUR: float = 10_000.0
MIN_PLAUSIBLE_PRICE_EUR: float = -1_000.0


@dataclass(frozen=True, slots=True)
class PriceEUR:
    """A wholesale electricity price in EUR per MWh.

    Negative prices are valid — see module docstring. The value must be
    finite and within plausible market bounds.
    """

    value: float

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise ValueError(f"PriceEUR must be finite, got {self.value!r}")
        if self.value > MAX_PLAUSIBLE_PRICE_EUR:
            raise ValueError(
                f"PriceEUR {self.value} exceeds {MAX_PLAUSIBLE_PRICE_EUR} EUR/MWh "
                f"— likely a data error"
            )
        if self.value < MIN_PLAUSIBLE_PRICE_EUR:
            raise ValueError(
                f"PriceEUR {self.value} below {MIN_PLAUSIBLE_PRICE_EUR} EUR/MWh "
                f"— likely a data error"
            )
