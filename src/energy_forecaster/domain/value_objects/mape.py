"""Mean Absolute Percentage Error — model accuracy metric."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True, order=True)
class MAPE:
    """Mean Absolute Percentage Error, expressed as a fraction in [0, +inf).

    A MAPE of 0.05 means 5% mean error. The value is intentionally unbounded
    above — a broken model can score MAPE > 1.0, and capping it would mask
    that signal. Negative or non-finite values are always errors and are
    rejected at construction.

    ``order=True`` lets the promotion rule and monitoring code compare two
    MAPEs directly (``challenger < champion``) without unwrapping the float.
    """

    value: float

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise ValueError(f"MAPE must be finite, got {self.value!r}")
        if self.value < 0:
            raise ValueError(f"MAPE must be non-negative, got {self.value}")
