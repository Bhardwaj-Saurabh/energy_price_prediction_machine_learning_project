"""LoadForecastRepository port — persistence for LoadForecast aggregates."""

from collections.abc import Iterable
from typing import Protocol

from energy_forecaster.domain.entities.load_forecast import LoadForecast


class LoadForecastRepository(Protocol):
    """Persists and retrieves :class:`LoadForecast` aggregates.

    Identity is the (zone, delivery_time, model_version) triple — the
    same delivery hour can be predicted by several model versions, and
    each prediction is its own row. Re-running inference with the same
    model version is idempotent (no duplicate rows); a different model
    version produces additional rows alongside the originals.
    """

    def add_many(self, forecasts: Iterable[LoadForecast]) -> int:
        """Insert forecasts, deduplicated by
        (zone, delivery_time, model_version). Returns the number of new
        rows. Idempotent on (key) — calling twice with the same input
        is safe and the second call returns 0.
        """
        ...
