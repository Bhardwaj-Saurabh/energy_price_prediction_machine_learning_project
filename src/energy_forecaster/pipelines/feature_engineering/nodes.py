"""Pure-function nodes for the feature engineering pipeline.

Each function is a Kedro ``node`` candidate: takes one or more
DataFrames, returns a new DataFrame, no side effects, no I/O. This is
what lets Kedro reorder, parallelise, or cache them based on the catalog.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from energy_forecaster.contracts.feature_matrix_schema import FeatureMatrixSchema
from energy_forecaster.contracts.load_observation_schema import LoadObservationSchema
from energy_forecaster.contracts.weather_reading_schema import WeatherReadingSchema

# Lag offsets in hours. 1h = "value an hour ago", 24h = "same hour
# yesterday" (daily seasonality), 168h = "same hour last week" (weekly).
# The list is hard-coded for now; promote to YAML config when a second
# consumer reads it.
_LAG_HOURS: tuple[int, ...] = (1, 24, 168)


def join_load_and_weather(
    load_df: DataFrame[LoadObservationSchema],
    weather_df: DataFrame[WeatherReadingSchema],
) -> pd.DataFrame:
    """Inner-join the two sources on (zone, timestamp_utc).

    Inner is deliberate: rows without both a load reading AND a weather
    reading are dropped. Training on partial rows would silently impute
    NaN-handling decisions; better to make the producer responsible for
    completeness.
    """
    return load_df.merge(weather_df, on=["zone", "timestamp_utc"], how="inner")


def add_time_features(joined: pd.DataFrame) -> pd.DataFrame:
    """Add ``hour_of_day``, ``day_of_week``, and ``is_weekend`` columns.

    All three are pure functions of ``timestamp_utc`` — they exist as
    columns rather than being recomputed at training time so the model's
    feature set is fully observable in the persisted feature matrix.
    """
    df = joined.copy()
    df["hour_of_day"] = df["timestamp_utc"].dt.hour.astype(int)
    df["day_of_week"] = df["timestamp_utc"].dt.dayofweek.astype(int)
    df["is_weekend"] = df["day_of_week"] >= 5
    return df


@pa.check_types
def add_lag_features(df: pd.DataFrame) -> DataFrame[FeatureMatrixSchema]:
    """Add ``load_lag_{1,24,168}h`` columns, computed per zone in time order.

    Sorted by (zone, timestamp_utc) before lagging so that
    ``groupby('zone').shift(N)`` returns the load value N hours earlier
    *within the same zone*. The first N rows of each zone get NaN.
    """
    df = df.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    grouped = df.groupby("zone", sort=False)["load_mw"]
    for hours in _LAG_HOURS:
        df[f"load_lag_{hours}h"] = grouped.shift(hours)
    return df  # type: ignore[no-any-return]
