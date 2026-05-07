"""Kedro DAG for feature engineering.

The pipeline is data, not code: each node names its inputs and outputs
as catalog dataset names. The catalog (configured at runtime in
:mod:`runner`) decides where each dataset physically lives — memory for
intermediates, JSONL directories for inputs, Parquet for the output
feature matrix.

The DAG, in roughly the topology Kedro will execute::

    load_directory    -> read_load    -> load_df
    weather_directory -> read_weather -> weather_df
    (load_df, weather_df) -> join_load_and_weather -> joined
    joined            -> add_time_features -> with_time_features
    with_time_features -> add_lag_features -> feature_matrix
"""

from kedro.pipeline import Pipeline, node

from energy_forecaster.pipelines.feature_engineering.io import (
    read_load_observations,
    read_weather_readings,
)
from energy_forecaster.pipelines.feature_engineering.nodes import (
    add_lag_features,
    add_time_features,
    join_load_and_weather,
)


def create_feature_engineering_pipeline() -> Pipeline:
    """Build the feature engineering DAG.

    Inputs (must exist in the catalog when ``run`` is called):
      * ``load_directory``    — Path to ``load_observations/``
      * ``weather_directory`` — Path to ``weather_readings/``

    Output:
      * ``feature_matrix`` — wide DataFrame validated by FeatureMatrixSchema
    """
    return Pipeline(
        [
            node(
                func=read_load_observations,
                inputs="load_directory",
                outputs="load_df",
                name="read_load",
            ),
            node(
                func=read_weather_readings,
                inputs="weather_directory",
                outputs="weather_df",
                name="read_weather",
            ),
            node(
                func=join_load_and_weather,
                inputs=["load_df", "weather_df"],
                outputs="joined",
                name="join_load_and_weather",
            ),
            node(
                func=add_time_features,
                inputs="joined",
                outputs="with_time_features",
                name="add_time_features",
            ),
            node(
                func=add_lag_features,
                inputs="with_time_features",
                outputs="feature_matrix",
                name="add_lag_features",
            ),
        ]
    )
