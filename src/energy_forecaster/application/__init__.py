"""Application layer — use cases + Protocol-defined ports + layer-neutral errors.

The dependency rule: this layer imports from ``domain`` only. It must NOT
import from ``adapters/``, framework packages (Kedro, MLflow, FastAPI,
Azure SDKs, pandas, requests), or the composition root. Concrete adapters
satisfy the ports defined here at runtime, but never appear in this
layer's source.
"""

from energy_forecaster.application.errors import (
    ApplicationError,
    DataSourceUnavailableError,
)
from energy_forecaster.application.use_cases.ingest_entsoe_load import (
    IngestEntsoeLoad,
    IngestEntsoeLoadResult,
)

__all__ = [
    "ApplicationError",
    "DataSourceUnavailableError",
    "IngestEntsoeLoad",
    "IngestEntsoeLoadResult",
]
