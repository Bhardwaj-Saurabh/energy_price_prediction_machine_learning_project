"""Composition root — the *only* place that wires concrete adapters to ports.

Every other module sees only Protocol-typed interfaces and receives its
dependencies through its constructor. This module reads :class:`Settings`,
chooses concrete adapters per environment, and returns ready-to-call use
case instances to the framework layer (CLI, FastAPI app, Prefect flow,
…). Anything in the codebase that imports a concrete adapter *and* a use
case must live here — and only here.

Branching policy:
  * ``entsoe_api_key`` unset → :class:`InMemoryEntsoeClient` (synthetic
    demo data; no network). This is what runs in `make test`, in CI, and
    on a developer's laptop without credentials.
  * ``entsoe_api_key`` set → :class:`EntsoePyClient` (real ENTSO-E API).
    This is what runs in production and in opt-in `make test-live` runs.

The key never reaches business code — only its presence/absence shapes
the choice of adapter, and the adapter takes the unwrapped string at
its constructor only.
"""

from energy_forecaster.adapters.clock.system_clock import SystemClock
from energy_forecaster.adapters.entsoe_client.entsoe_py import EntsoePyClient
from energy_forecaster.adapters.entsoe_client.in_memory import InMemoryEntsoeClient
from energy_forecaster.adapters.load_observation_repo.local_fs import (
    LocalFsLoadObservationRepository,
)
from energy_forecaster.application.ports.entsoe_client import EntsoeClient
from energy_forecaster.application.use_cases.ingest_entsoe_load import (
    IngestEntsoeLoad,
)
from energy_forecaster.config.settings import Settings


def build_ingest_entsoe_load(settings: Settings) -> IngestEntsoeLoad:
    """Wire :class:`IngestEntsoeLoad` for the given environment."""
    entsoe: EntsoeClient
    if settings.entsoe_api_key is None:
        entsoe = InMemoryEntsoeClient()
    else:
        entsoe = EntsoePyClient(api_key=settings.entsoe_api_key.get_secret_value())

    return IngestEntsoeLoad(
        entsoe=entsoe,
        repo=LocalFsLoadObservationRepository(root=settings.local_data_root),
        clock=SystemClock(),
    )
