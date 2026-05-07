"""Kedro DAG for inference.

Inputs (must exist in the catalog when the runner invokes this):
  * ``features``      — Parquet of FeatureMatrixSchema-validated rows.
  * ``model``         — A pre-loaded model object (the runner calls
                        ``registry.load(version)`` and puts the result
                        here as a MemoryDataset).
  * ``model_version`` — The :class:`ModelVersion` to stamp on every
                        produced forecast.
  * ``hours``         — How many recent hours (per zone) to predict.

Output:
  * ``forecasts`` — list[LoadForecast]; the runner takes this and hands
    it to :class:`LoadForecastRepository` for persistence.

The DAG, in execution order::

    (features, hours) -> slice_recent_features -> prediction_inputs
    (model, inputs)   -> predict_loads         -> prediction_data
    (data, version)   -> build_forecasts       -> forecasts
"""

from kedro.pipeline import Pipeline, node

from energy_forecaster.pipelines.inference.nodes import (
    build_forecasts,
    predict_loads,
    slice_recent_features,
)


def create_inference_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=slice_recent_features,
                inputs=["features", "hours"],
                outputs="prediction_inputs",
                name="slice_recent_features",
            ),
            node(
                func=predict_loads,
                inputs=["model", "prediction_inputs"],
                outputs="prediction_data",
                name="predict_loads",
            ),
            node(
                func=build_forecasts,
                inputs=["prediction_data", "model_version"],
                outputs="forecasts",
                name="build_forecasts",
            ),
        ]
    )
