"""Unit tests for the composition root.

These are smoke tests, not behaviour tests: they confirm that the
composition root assembles a use case with the right concrete adapter
types, given a Settings instance. Behavioural assertions belong in the
use-case and adapter test files where the concrete types are tested
directly.
"""

from pathlib import Path

from pydantic import SecretStr

from energy_forecaster.adapters.clock.system_clock import SystemClock
from energy_forecaster.adapters.entsoe_client.entsoe_py import EntsoePyClient
from energy_forecaster.adapters.entsoe_client.in_memory import InMemoryEntsoeClient
from energy_forecaster.adapters.load_observation_repo.local_fs import (
    LocalFsLoadObservationRepository,
)
from energy_forecaster.application.use_cases.ingest_entsoe_load import (
    IngestEntsoeLoad,
)
from energy_forecaster.composition import build_ingest_entsoe_load
from energy_forecaster.config.settings import Environment, Settings


def test_build_ingest_entsoe_load_returns_a_use_case(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment=Environment.LOCAL,
        local_data_root=tmp_path,
    )
    use_case = build_ingest_entsoe_load(settings)
    assert isinstance(use_case, IngestEntsoeLoad)


def test_no_api_key_picks_in_memory_entsoe(tmp_path: Path) -> None:
    # Reach into the use case's private fields to confirm wiring. This is
    # the one place where peeking at internals is acceptable — the whole
    # job of this test is to verify the wiring contract that no other test
    # can observe.
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment=Environment.LOCAL,
        local_data_root=tmp_path,
        entsoe_api_key=None,
    )
    use_case = build_ingest_entsoe_load(settings)
    assert isinstance(use_case._entsoe, InMemoryEntsoeClient)
    assert isinstance(use_case._repo, LocalFsLoadObservationRepository)
    assert isinstance(use_case._clock, SystemClock)


def test_api_key_present_picks_real_entsoe_py_client(tmp_path: Path) -> None:
    # The real adapter's constructor stores the key only — no network
    # call — so it is safe to instantiate with a placeholder.
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment=Environment.LOCAL,
        local_data_root=tmp_path,
        entsoe_api_key=SecretStr("placeholder-not-real"),
    )
    use_case = build_ingest_entsoe_load(settings)
    assert isinstance(use_case._entsoe, EntsoePyClient)
