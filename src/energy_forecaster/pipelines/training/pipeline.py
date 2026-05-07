"""Kedro DAG for training the demand model.

Inputs (must exist in the catalog when the runner invokes this):
  * ``features`` — Parquet of :class:`FeatureMatrixSchema`-validated rows.

Outputs (terminal):
  * ``training_artifacts`` — dict with ``model``, ``params``, ``metrics``
    that the runner hands to :class:`ModelRegistry` for registration.
    Keeping registration *outside* the pipeline keeps every node pure.

The DAG, in execution order::

    features        -> prepare_training_data -> training_data
    training_data   -> train_model           -> trained_model
    (model, td)     -> evaluate_model        -> metrics
    (model, metrics)-> collect_artifacts     -> training_artifacts
"""

from kedro.pipeline import Pipeline, node

from energy_forecaster.pipelines.training.nodes import (
    collect_artifacts,
    evaluate_model,
    prepare_training_data,
    train_model,
)


def create_training_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_training_data,
                inputs="features",
                outputs="training_data",
                name="prepare_training_data",
            ),
            node(
                func=train_model,
                inputs="training_data",
                outputs="trained_model",
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "training_data"],
                outputs="metrics",
                name="evaluate_model",
            ),
            node(
                func=collect_artifacts,
                inputs=["trained_model", "metrics"],
                outputs="training_artifacts",
                name="collect_artifacts",
            ),
        ]
    )
