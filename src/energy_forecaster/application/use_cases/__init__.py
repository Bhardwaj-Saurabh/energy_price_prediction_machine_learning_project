"""Application use cases — orchestrate domain types and ports."""

from energy_forecaster.application.use_cases.ingest_entsoe_load import (
    IngestEntsoeLoad,
    IngestEntsoeLoadResult,
)

__all__ = ["IngestEntsoeLoad", "IngestEntsoeLoadResult"]
