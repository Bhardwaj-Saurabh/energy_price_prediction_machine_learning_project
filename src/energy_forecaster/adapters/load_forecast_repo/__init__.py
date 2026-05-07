"""Concrete implementations of the LoadForecastRepository port."""

from energy_forecaster.adapters.load_forecast_repo.local_fs import (
    LocalFsLoadForecastRepository,
)

__all__ = ["LocalFsLoadForecastRepository"]
