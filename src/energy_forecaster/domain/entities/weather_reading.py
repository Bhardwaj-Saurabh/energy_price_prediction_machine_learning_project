"""An hourly weather measurement at a representative point in a bidding zone."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

from energy_forecaster.domain._validation import require_utc
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone

# Physical plausibility bounds. These exist to reject obvious data corruption
# (units mix-ups, sensor faults), not to constrain valid science. The ranges
# are intentionally generous — extending them is a deliberate decision, not
# something that should happen quietly to "fix" rejected rows.
MIN_TEMP_C: float = -60.0
MAX_TEMP_C: float = 60.0
MIN_WIND_MS: float = 0.0
MAX_WIND_MS: float = 100.0  # Cat-5 hurricane ~70 m/s; 100 leaves headroom.
MIN_GHI_WM2: float = 0.0
MAX_GHI_WM2: float = 1500.0  # Solar constant ≈ 1361 W/m²; surface max varies.
MIN_CLOUD_COVER_PCT: float = 0.0
MAX_CLOUD_COVER_PCT: float = 100.0
MIN_PRECIP_MM: float = 0.0
MAX_PRECIP_MM: float = 500.0  # Extreme hourly precipitation cap.


@dataclass(frozen=True, slots=True)
class WeatherReading:
    """An hourly weather observation tied to a bidding zone and UTC timestamp.

    Identity is the (zone, timestamp_utc) pair, mirroring LoadObservation:
    two readings with the same identity are the same observation regardless
    of source.

    The six measurement fields are validated inline rather than as
    individual value objects because they always travel together. If, in
    future, one of them gets used independently (e.g. temperature inside a
    heating-degree-day calculation outside this reading), promote it to a
    value object at that point.
    """

    zone: BiddingZone
    timestamp_utc: datetime
    temp_c: float
    wind_10m_ms: float
    wind_100m_ms: float
    ghi_wm2: float
    cloud_cover_pct: float
    precip_mm: float

    def __post_init__(self) -> None:
        require_utc("WeatherReading.timestamp_utc", self.timestamp_utc)

        self._check("temp_c", self.temp_c, MIN_TEMP_C, MAX_TEMP_C)
        self._check("wind_10m_ms", self.wind_10m_ms, MIN_WIND_MS, MAX_WIND_MS)
        self._check("wind_100m_ms", self.wind_100m_ms, MIN_WIND_MS, MAX_WIND_MS)
        self._check("ghi_wm2", self.ghi_wm2, MIN_GHI_WM2, MAX_GHI_WM2)
        self._check(
            "cloud_cover_pct",
            self.cloud_cover_pct,
            MIN_CLOUD_COVER_PCT,
            MAX_CLOUD_COVER_PCT,
        )
        self._check("precip_mm", self.precip_mm, MIN_PRECIP_MM, MAX_PRECIP_MM)

    @staticmethod
    def _check(name: str, value: float, lo: float, hi: float) -> None:
        if not isfinite(value):
            raise ValueError(f"WeatherReading.{name} must be finite, got {value!r}")
        if not (lo <= value <= hi):
            raise ValueError(f"WeatherReading.{name}={value} outside plausible range [{lo}, {hi}]")
