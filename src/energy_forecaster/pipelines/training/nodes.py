"""Pure-function nodes for the training pipeline.

Each function is a Kedro ``node`` candidate: takes inputs from the
catalog, returns outputs as named datasets. No I/O, no port interaction,
no side effects — port-touching (model registration) happens at the
edge in :mod:`runner`.

Defaults are intentionally hard-coded:
  * ``_FEATURE_COLUMNS`` is the model's input feature set
  * ``_LIGHTGBM_PARAMS`` are sensible starting hyperparameters
  * ``_TRAIN_FRACTION`` is the time-ordered split point

These move to YAML / Pydantic Settings when a second consumer starts to
read them. Hard-coding now is the rulebook's "don't add config knobs
for stages we have not reached" rule applied.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
from pandera.typing import DataFrame
from sklearn.metrics import mean_absolute_percentage_error

from energy_forecaster.contracts.feature_matrix_schema import FeatureMatrixSchema

# Columns the model is trained on. Excludes target (``load_mw``),
# identity columns (``timestamp_utc``), and the raw zone string (we use
# ``zone_cat`` as a categorical-encoded version).
_FEATURE_COLUMNS: tuple[str, ...] = (
    "temp_c",
    "wind_10m_ms",
    "wind_100m_ms",
    "ghi_wm2",
    "cloud_cover_pct",
    "precip_mm",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "load_lag_1h",
    "load_lag_24h",
    "load_lag_168h",
    "zone_cat",
)
_TARGET_COLUMN: str = "load_mw"

# Time-ordered split. The last 20 % of timestamps becomes the test
# window — random splitting on time-series data leaks future into past
# and gives optimistic accuracy figures.
_TRAIN_FRACTION: float = 0.8

# LightGBM defaults. Conservative learning rate, default leaf shape,
# MAPE as the eval metric so train-time logs match what we evaluate on.
_LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "mape",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "verbose": -1,
}
_NUM_BOOST_ROUND: int = 100


def prepare_training_data(
    features: DataFrame[FeatureMatrixSchema],
) -> dict[str, Any]:
    """Drop rows with NaN lags, encode the zone categorical, time-order split.

    Returns a dict carrying ``X_train``, ``y_train``, ``X_test``,
    ``y_test`` plus the row counts. Returning a dict (rather than four
    separate datasets) keeps the Kedro DAG less wide while still letting
    downstream nodes destructure cleanly.
    """
    df = features.dropna(subset=["load_lag_1h", "load_lag_24h", "load_lag_168h"]).copy()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df["zone_cat"] = df["zone"].astype("category").cat.codes

    n = len(df)
    split = int(n * _TRAIN_FRACTION)

    return {
        "X_train": df.iloc[:split][list(_FEATURE_COLUMNS)],
        "y_train": df.iloc[:split][_TARGET_COLUMN],
        "X_test": df.iloc[split:][list(_FEATURE_COLUMNS)],
        "y_test": df.iloc[split:][_TARGET_COLUMN],
        "train_size": split,
        "test_size": n - split,
    }


def train_model(training_data: dict[str, Any]) -> lgb.Booster:
    """Fit a LightGBM regressor on the training split."""
    train_set = lgb.Dataset(training_data["X_train"], label=training_data["y_train"])
    booster = lgb.train(
        params=_LIGHTGBM_PARAMS,
        train_set=train_set,
        num_boost_round=_NUM_BOOST_ROUND,
    )
    return booster


def evaluate_model(model: lgb.Booster, training_data: dict[str, Any]) -> dict[str, float]:
    """Compute test-set MAPE (and a few diagnostics) for the trained model."""
    predictions = model.predict(training_data["X_test"])
    mape = float(mean_absolute_percentage_error(training_data["y_test"], predictions))
    return {
        "mape": mape,
        "train_size": float(training_data["train_size"]),
        "test_size": float(training_data["test_size"]),
    }


def collect_artifacts(model: lgb.Booster, metrics: dict[str, float]) -> dict[str, Any]:
    """Bundle ``(model, params, metrics)`` so the runner can hand them to
    :class:`ModelRegistry` after the pipeline completes."""
    return {
        "model": model,
        "params": dict(_LIGHTGBM_PARAMS),
        "metrics": metrics,
    }
