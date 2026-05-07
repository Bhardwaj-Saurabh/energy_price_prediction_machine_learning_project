"""Feature engineering pipeline — Kedro DAG that builds the feature matrix.

Public surface:
  * :func:`create_feature_engineering_pipeline` — the Kedro Pipeline factory.
  * :func:`run_feature_engineering` — programmatic runner that wires a
    DataCatalog from filesystem paths and executes the pipeline.

Adapters from earlier chunks (LocalFs repos) write the JSONL inputs;
this pipeline reads them, joins, enriches, validates, and writes a
Parquet feature matrix.
"""

from energy_forecaster.pipelines.feature_engineering.pipeline import (
    create_feature_engineering_pipeline,
)
from energy_forecaster.pipelines.feature_engineering.runner import (
    run_feature_engineering,
)

__all__ = ["create_feature_engineering_pipeline", "run_feature_engineering"]
