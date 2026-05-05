"""Local-filesystem implementation of WeatherReadingRepository.

Storage layout::

    <root>/weather_readings/<zone>.jsonl

Mirrors :class:`LocalFsLoadObservationRepository` exactly. The Azure
Blob adapter that arrives later uses Parquet at the same logical
boundary; the application layer does not notice the format change.
"""

import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from energy_forecaster.domain.entities.weather_reading import WeatherReading
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone

_SUBDIR = "weather_readings"


class LocalFsWeatherReadingRepository:
    """Persist :class:`WeatherReading` aggregates to JSON-Lines files."""

    def __init__(self, root: Path) -> None:
        self._dir = root / _SUBDIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def add_many(self, readings: Iterable[WeatherReading]) -> int:
        by_zone: dict[BiddingZone, list[WeatherReading]] = defaultdict(list)
        for r in readings:
            by_zone[r.zone].append(r)

        total_new = 0
        for zone, batch in by_zone.items():
            existing = self._existing_timestamps(zone)
            new = [r for r in batch if r.timestamp_utc not in existing]
            if not new:
                continue
            self._append(zone, new)
            total_new += len(new)
        return total_new

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

    def _append(self, zone: BiddingZone, batch: list[WeatherReading]) -> None:
        path = self._file_for(zone)
        with path.open("a", encoding="utf-8") as f:
            for r in batch:
                f.write(_serialise(r) + "\n")


def _serialise(r: WeatherReading) -> str:
    return json.dumps(
        {
            "zone": r.zone.value,
            "timestamp_utc": r.timestamp_utc.isoformat(),
            "temp_c": r.temp_c,
            "wind_10m_ms": r.wind_10m_ms,
            "wind_100m_ms": r.wind_100m_ms,
            "ghi_wm2": r.ghi_wm2,
            "cloud_cover_pct": r.cloud_cover_pct,
            "precip_mm": r.precip_mm,
        }
    )


def _deserialise(line: str) -> WeatherReading:
    """Inverse of :func:`_serialise`. Used by tests; not part of the port API."""
    record = json.loads(line)
    return WeatherReading(
        zone=BiddingZone(record["zone"]),
        timestamp_utc=datetime.fromisoformat(record["timestamp_utc"]),
        temp_c=record["temp_c"],
        wind_10m_ms=record["wind_10m_ms"],
        wind_100m_ms=record["wind_100m_ms"],
        ghi_wm2=record["ghi_wm2"],
        cloud_cover_pct=record["cloud_cover_pct"],
        precip_mm=record["precip_mm"],
    )
