"""Electrical power magnitude in megawatts."""

from dataclasses import dataclass
from math import isfinite

# Plausible upper bound for an instantaneous bidding-zone load. Germany's
# all-time peak is ~80 GW; we accept up to 200 GW so future market expansion
# does not produce false rejects, but reject values that almost certainly
# indicate a unit error (e.g. a kW value mis-labelled as MW).
MAX_PLAUSIBLE_LOAD_MW: float = 200_000.0


@dataclass(frozen=True, slots=True)
class EnergyMW:
    """An instantaneous power measurement in megawatts.

    Construction validates that the value is finite, non-negative, and within
    the plausible operational range. Once constructed, the instance is
    immutable and hashable — safe to use as a dict key or set member.
    """

    value: float

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise ValueError(f"EnergyMW must be finite, got {self.value!r}")
        if self.value < 0:
            raise ValueError(f"EnergyMW must be non-negative, got {self.value}")
        if self.value > MAX_PLAUSIBLE_LOAD_MW:
            raise ValueError(
                f"EnergyMW {self.value} exceeds plausible upper bound "
                f"{MAX_PLAUSIBLE_LOAD_MW} — likely a unit error"
            )
