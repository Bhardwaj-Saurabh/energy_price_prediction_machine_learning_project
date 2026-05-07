"""Local-filesystem implementation of LoadForecastRepository.

Storage layout::

    <root>/load_forecasts/<zone>.jsonl

One file per bidding zone, one JSON record per line. Identity is the
``(zone, delivery_time, model_version)`` triple — the same delivery
hour can have multiple forecasts coming from different model versions,
and each is preserved.
"""

import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW
from energy_forecaster.domain.value_objects.model_version import ModelVersion

_SUBDIR = "load_forecasts"


class LocalFsLoadForecastRepository:
    """Persist :class:`LoadForecast` aggregates to JSON-Lines files."""

    def __init__(self, root: Path) -> None:
        self._dir = root / _SUBDIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def add_many(self, forecasts: Iterable[LoadForecast]) -> int:
        by_zone: dict[BiddingZone, list[LoadForecast]] = defaultdict(list)
        for f in forecasts:
            by_zone[f.zone].append(f)

        total_new = 0
        for zone, batch in by_zone.items():
            existing = self._existing_keys(zone)
            new = [f for f in batch if (f.delivery_time, f.model_version.value) not in existing]
            if not new:
                continue
            self._append(zone, new)
            total_new += len(new)
        return total_new

    def _file_for(self, zone: BiddingZone) -> Path:
        return self._dir / f"{zone.value}.jsonl"

    def _existing_keys(self, zone: BiddingZone) -> set[tuple[datetime, str]]:
        path = self._file_for(zone)
        if not path.exists():
            return set()
        keys: set[tuple[datetime, str]] = set()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                keys.add(
                    (
                        datetime.fromisoformat(record["delivery_time"]),
                        record["model_version"],
                    )
                )
        return keys

    def _append(self, zone: BiddingZone, batch: list[LoadForecast]) -> None:
        path = self._file_for(zone)
        with path.open("a", encoding="utf-8") as f:
            for forecast in batch:
                f.write(_serialise(forecast) + "\n")


def _serialise(f: LoadForecast) -> str:
    return json.dumps(
        {
            "zone": f.zone.value,
            "as_of_time": f.as_of_time.isoformat(),
            "delivery_time": f.delivery_time.isoformat(),
            "predicted_load": f.predicted_load.value,
            "model_version": f.model_version.value,
        }
    )


def deserialise(line: str) -> LoadForecast:
    """Inverse of :func:`_serialise`. Public for downstream readers."""
    record = json.loads(line)
    return LoadForecast(
        zone=BiddingZone(record["zone"]),
        as_of_time=datetime.fromisoformat(record["as_of_time"]),
        delivery_time=datetime.fromisoformat(record["delivery_time"]),
        predicted_load=EnergyMW(record["predicted_load"]),
        model_version=ModelVersion(record["model_version"]),
    )
