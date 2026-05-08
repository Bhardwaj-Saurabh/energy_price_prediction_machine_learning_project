"""Domain rules — pure business policy expressed as functions.

A rule answers a *should-we* question from the business: should this
challenger replace the champion, should this run trigger retraining, is
this forecast inside the supported horizon. Rules are pure functions, so
they are trivially unit-testable and reusable from training, monitoring,
serving, and CI gates alike.
"""

from energy_forecaster.domain.rules.promotion import (
    PROMOTION_MAPE_DELTA,
    should_promote,
)
from energy_forecaster.domain.rules.retrain import (
    RETRAIN_MAPE_THRESHOLD,
    RETRAIN_PSI_THRESHOLD,
    should_retrain,
)

__all__ = [
    "PROMOTION_MAPE_DELTA",
    "RETRAIN_MAPE_THRESHOLD",
    "RETRAIN_PSI_THRESHOLD",
    "should_promote",
    "should_retrain",
]
