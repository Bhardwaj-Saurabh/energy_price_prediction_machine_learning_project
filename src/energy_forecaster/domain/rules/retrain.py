"""Retraining trigger policy — performance drift OR data drift.

Sibling of :mod:`energy_forecaster.domain.rules.promotion`. Where promotion
asks "should this challenger replace the champion", retraining asks "is
the world different enough that we need a new challenger at all".

The rule is intentionally an OR: production-ML wisdom is that *either*
signal degrading is enough to act on. Waiting for both to breach
simultaneously is how you ship a stale model into a regime change.
"""

from energy_forecaster.domain.value_objects.mape import MAPE

# Default rolling-MAPE threshold: 5% — the PRD's headline accuracy target.
# Once observed accuracy drifts past this, we no longer believe the model
# is meeting its contract. Same scale as ``MAPE.value`` (a fraction).
RETRAIN_MAPE_THRESHOLD: float = 0.05

# Default PSI threshold: 0.20. Industry convention is PSI < 0.10 = stable,
# 0.10-0.20 = moderate drift (worth watching), >= 0.20 = significant drift
# (worth acting on). We act at the high band; the moderate band is for the
# dashboard, not the trigger.
RETRAIN_PSI_THRESHOLD: float = 0.20


def should_retrain(
    *,
    rolling_mape: MAPE,
    max_psi: float,
    mape_threshold: float = RETRAIN_MAPE_THRESHOLD,
    psi_threshold: float = RETRAIN_PSI_THRESHOLD,
) -> bool:
    """Decide whether to trigger a retrain.

    Returns True iff *either* the rolling MAPE has reached the accuracy
    threshold (performance drift) *or* the worst per-feature PSI has
    reached the drift threshold (data drift). Thresholds are inclusive —
    exactly hitting either threshold triggers, mirroring the promotion
    rule's convention.

    The function is pure: same inputs yield the same answer with no
    side-effects. The monitoring pipeline computes the inputs and calls
    this rule; the orchestrator (cron / Prefect later) consumes the
    decision and acts on it.
    """
    if mape_threshold < 0:
        raise ValueError(f"MAPE threshold must be non-negative, got {mape_threshold}")
    if psi_threshold < 0:
        raise ValueError(f"PSI threshold must be non-negative, got {psi_threshold}")
    return rolling_mape.value >= mape_threshold or max_psi >= psi_threshold
