"""Local-filesystem implementation of LoadObservationRepository.

Storage layout::

    <root>/load_observations/<zone>.jsonl

One file per bidding zone. Each line is one JSON object with the keys
``zone``, ``timestamp_utc`` (ISO-8601 with ``+00:00`` offset), and
``load`` (float MW). The format is deliberately human-readable and
``cat``-friendly so local development is inspectable without extra
tooling. The Azure Blob adapter that arrives later will use Parquet at
the same logical boundary; the application layer will not notice the
difference.

Dedup is enforced by reading the existing keys for the target zone
before each write — the same contract Postgres' ``ON CONFLICT DO
NOTHING`` will provide in the production adapter. The implementation
assumes a single writer; concurrent writers are explicitly out of scope
for the local-fs path (use the Postgres adapter when concurrency
matters).
"""

import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW

_SUBDIR = "load_observations"


class LocalFsLoadObservationRepository:
    """Persist :class:`LoadObservation` aggregates to JSON-Lines files on local disk.

    The constructor materialises the storage subdirectory eagerly so the
    composition root surfaces any permission or path errors at startup
    rather than at the first ingest call. Callers pass in the
    ``local_data_root`` from Settings; the adapter owns the
    ``load_observations/`` segment under it.
    """

    def __init__(self, root: Path) -> None:
        self._dir = root / _SUBDIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def add_many(self, observations: Iterable[LoadObservation]) -> int:
        # Group by zone so we open each zone file at most once per call —
        # important when the use case ingests several zones in a single run.
        by_zone: dict[BiddingZone, list[LoadObservation]] = defaultdict(list)
        for obs in observations:
            by_zone[obs.zone].append(obs)

        total_new = 0
        for zone, obs_list in by_zone.items():
            existing_keys = self._existing_timestamps(zone)
            new_obs = [o for o in obs_list if o.timestamp_utc not in existing_keys]
            if not new_obs:
                continue
            self._append(zone, new_obs)
            total_new += len(new_obs)

        return total_new

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _file_for(self, zone: BiddingZone) -> Path:
        return self._dir / f"{zone.value}.jsonl"

    def _existing_timestamps(self, zone: BiddingZone) -> set[datetime]:
        path = self._file_for(zone)
        if not path.exists():
            return set()
        keys: set[datetime] = set()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                keys.add(datetime.fromisoformat(record["timestamp_utc"]))
        return keys

    def _append(self, zone: BiddingZone, observations: list[LoadObservation]) -> None:
        path = self._file_for(zone)
        with path.open("a", encoding="utf-8") as f:
            for obs in observations:
                f.write(_serialise(obs) + "\n")


def _serialise(obs: LoadObservation) -> str:
    return json.dumps(
        {
            "zone": obs.zone.value,
            "timestamp_utc": obs.timestamp_utc.isoformat(),
            "load": obs.load.value,
        }
    )


def deserialise(line: str) -> LoadObservation:
    """Parse one JSONL line into a :class:`LoadObservation`.

    Public so the feature-engineering pipeline (and tests) can read what
    this adapter wrote. The reverse direction of :func:`_serialise` —
    keep them in sync if either side ever changes.
    """
    record = json.loads(line)
    return LoadObservation(
        zone=BiddingZone(record["zone"]),
        timestamp_utc=datetime.fromisoformat(record["timestamp_utc"]),
        load=EnergyMW(record["load"]),
    )
