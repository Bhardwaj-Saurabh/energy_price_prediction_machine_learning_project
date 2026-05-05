"""Champion / challenger promotion policy."""

from energy_forecaster.domain.value_objects.mape import MAPE

# Default promotion margin: a challenger must beat the champion by at least
# 0.5 percentage points of MAPE on the test window before it gets promoted.
# Expressed as a fraction (MAPE is a fraction): 0.005 == 0.5pp. This default
# is sourced from the PRD; the production wiring will read the value from
# config in a later chunk and pass it as the ``delta`` keyword.
PROMOTION_MAPE_DELTA: float = 0.005


def should_promote(
    challenger: MAPE,
    champion: MAPE,
    *,
    delta: float = PROMOTION_MAPE_DELTA,
) -> bool:
    """Decide whether the challenger replaces the champion.

    Returns True iff the challenger's MAPE is at least ``delta`` better
    (lower) than the champion's. The margin is one-sided and inclusive at
    the threshold — exactly meeting the margin promotes.

    The function is pure: same inputs yield the same answer with no
    side-effects. That makes it safe to call from anywhere — training,
    monitoring, CI gates — and trivial to unit-test.
    """
    if delta < 0:
        raise ValueError(f"Promotion delta must be non-negative, got {delta}")
    return challenger.value <= champion.value - delta
