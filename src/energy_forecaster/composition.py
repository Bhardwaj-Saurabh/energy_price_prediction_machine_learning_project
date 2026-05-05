"""Composition root — the *only* place that wires concrete adapters to ports.

Every other module sees only Protocol-typed interfaces and receives its
dependencies through its constructor. This module reads :class:`Settings`,
chooses concrete adapters per environment, and returns ready-to-call use
case instances to the framework layer (CLI, FastAPI app, Prefect flow,
…). Anything in the codebase that imports a concrete adapter *and* a use
case must live here — and only here.

Branching policy:
  Currently, every environment uses :class:`InMemoryEntsoeClient` because
  the real HTTP adapter is not yet implemented. When chunk 5c lands and
  introduces ``EntsoePyClient``, the branch will become roughly::

      if settings.entsoe_api_key is None:
          entsoe = InMemoryEntsoeClient()
      else:
          entsoe = EntsoePyClient(api_key=settings.entsoe_api_key.get_secret_value())

  Local-mode demos without a key continue to work; production mode picks
  the real adapter automatically.
"""

from energy_forecaster.adapters.clock.system_clock import SystemClock
from energy_forecaster.adapters.entsoe_client.in_memory import InMemoryEntsoeClient
from energy_forecaster.adapters.load_observation_repo.local_fs import (
    LocalFsLoadObservationRepository,
)
from energy_forecaster.application.use_cases.ingest_entsoe_load import (
    IngestEntsoeLoad,
)
from energy_forecaster.config.settings import Settings


def build_ingest_entsoe_load(settings: Settings) -> IngestEntsoeLoad:
    """Wire :class:`IngestEntsoeLoad` for the given environment."""
    return IngestEntsoeLoad(
        entsoe=InMemoryEntsoeClient(),
        repo=LocalFsLoadObservationRepository(root=settings.local_data_root),
        clock=SystemClock(),
    )
