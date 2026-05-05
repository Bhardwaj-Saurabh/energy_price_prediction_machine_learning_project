"""Concrete implementations of the LoadObservationRepository port."""

from energy_forecaster.adapters.load_observation_repo.local_fs import (
    LocalFsLoadObservationRepository,
)

__all__ = ["LocalFsLoadObservationRepository"]
