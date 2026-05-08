"""Pure-numerical helpers used by the monitoring pipeline.

Numpy / sklearn live here, not in domain or application — same rule as
``pipelines/training/nodes.py``. The functions are pure (no I/O, no
hidden state); the monitoring use case calls them with arrays it has
already aligned and filtered.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_percentage_error

# PSI epsilon: a tiny floor for per-bin proportions so that a single
# empty bucket does not send the running sum to ±inf via ``ln(0)``. The
# value 1e-4 is the convention in production drift-monitoring code; it
# is small enough to not distort genuine signal and large enough to keep
# the math finite. Documented so a future change is intentional.
_PSI_EPSILON: float = 1e-4

# Default bin count for PSI. The 0.20 retrain threshold in
# ``domain.rules.retrain`` is calibrated against deciles (10 bins);
# changing the bin count without re-calibrating the threshold changes
# the meaning of the gate. Keep these two constants in lockstep.
_DEFAULT_PSI_BINS: int = 10


def mape(
    actuals: Sequence[float] | NDArray[np.float64],
    predictions: Sequence[float] | NDArray[np.float64],
) -> float:
    """Mean Absolute Percentage Error.

    Thin wrapper over scikit-learn's implementation that asserts the
    inputs are non-empty and equal-length. Empty inputs would silently
    return ``nan`` from sklearn — surfacing the bug at the boundary is
    cheaper than chasing a NaN through the verdict downstream.
    """
    a = np.asarray(actuals, dtype=float)
    p = np.asarray(predictions, dtype=float)
    if a.size == 0:
        raise ValueError("MAPE undefined for empty input")
    if a.shape != p.shape:
        raise ValueError(f"actuals and predictions must align: {a.shape} vs {p.shape}")
    return float(mean_absolute_percentage_error(a, p))


def population_stability_index(
    expected: Sequence[float] | NDArray[np.float64],
    observed: Sequence[float] | NDArray[np.float64],
    *,
    bins: int = _DEFAULT_PSI_BINS,
) -> float:
    """Population Stability Index — distribution-shift score.

    Computes ``Σ (o[i] - e[i]) * ln(o[i] / e[i])`` after binning both
    inputs by the *expected* distribution's quantile edges. Quantile
    bins make ``e[i] ≈ 1/bins`` by construction, which keeps the score
    sensitive to shifts in mass rather than to feature outliers.

    Industry interpretation (calibrated for ``bins=10``):
      * < 0.10: stable
      * 0.10-0.20: moderate drift (watch)
      * >= 0.20: significant drift (act)

    NaNs in either input are dropped before binning. If ``expected``
    has fewer unique values than ``bins`` (e.g. a binary feature), the
    quantile edges deduplicate and the score is computed over the
    actual number of distinct buckets — a clean degenerate case rather
    than a crash.
    """
    if bins < 2:
        raise ValueError(f"PSI requires at least 2 bins, got {bins}")

    e = np.asarray(expected, dtype=float)
    o = np.asarray(observed, dtype=float)
    e = e[~np.isnan(e)]
    o = o[~np.isnan(o)]

    if e.size == 0 or o.size == 0:
        raise ValueError("PSI undefined when either side is empty after dropping NaN")

    # Quantile-based edges from the baseline. ``np.unique`` collapses
    # duplicates that arise on low-cardinality features so we never
    # call ``np.histogram`` with non-monotonic edges.
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(e, quantiles))
    if edges.size < 2:
        # All baseline values identical — distribution is a delta. PSI
        # is 0 if observed matches, undefined otherwise. Return 0 when
        # observed is also constant at the same value, else a large
        # finite signal so the rule fires.
        return 0.0 if np.all(o == edges[0]) else float("inf")

    # Nudge the upper edge by 1 ULP. ``np.histogram``'s last bin is
    # already closed on the right so values equal to ``edges[-1]`` are
    # counted, but observed values that are theoretically equal but
    # arrive 1 ULP above due to floating-point rounding would be
    # dropped. This is the cheapest fix.
    edges = edges.copy()
    edges[-1] = np.nextafter(edges[-1], np.inf)

    e_counts, _ = np.histogram(e, bins=edges)
    o_counts, _ = np.histogram(o, bins=edges)

    e_props = np.maximum(e_counts / e.size, _PSI_EPSILON)
    o_props = np.maximum(o_counts / o.size, _PSI_EPSILON)

    return float(np.sum((o_props - e_props) * np.log(o_props / e_props)))
