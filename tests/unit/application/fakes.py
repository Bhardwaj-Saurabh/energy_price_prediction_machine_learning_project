"""In-memory fakes for the application ports.

These are not mocks — they are real implementations of the same Protocol
that production adapters implement, just backed by in-process state. The
behavioural contract enforced here (UTC ordering, deduplication on
identity, error propagation) matches what the Postgres / ENTSO-E adapters
must enforce in production. Testing the use case against these fakes is
testing it against the *contract*, not against a recorded sequence of
calls.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from energy_forecaster.application.errors import DataSourceUnavailableError
from energy_forecaster.domain.entities.load_forecast import LoadForecast
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.entities.weather_reading import WeatherReading
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.model_version import ModelVersion


class FakeClock:
    """Controllable clock for deterministic tests.

    Use ``advance(delta)`` between operations when a test needs the
    started_at and finished_at fields of a result to differ — calling
    ``now()`` does not auto-advance.
    """

    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


class FakeEntsoeClient:
    """Predetermined-data ENTSO-E stand-in.

    ``seed`` loads observations into the fake before a test runs; the
    fake then returns the subset whose timestamp falls in the requested
    window. ``fail_on`` flips a single zone into raising
    ``DataSourceUnavailableError`` so the use case's failure path can be
    exercised without resorting to mock side-effects.
    """

    def __init__(self) -> None:
        self._data: dict[BiddingZone, list[LoadObservation]] = {}
        self._fail_on_zone: BiddingZone | None = None

    def seed(self, zone: BiddingZone, observations: Iterable[LoadObservation]) -> None:
        self._data[zone] = list(observations)

    def fail_on(self, zone: BiddingZone) -> None:
        self._fail_on_zone = zone

    def fetch_load(
        self,
        *,
        zone: BiddingZone,
        start: datetime,
        end: datetime,
    ) -> Iterable[LoadObservation]:
        if zone == self._fail_on_zone:
            raise DataSourceUnavailableError(f"ENTSO-E unavailable for {zone}")
        return [obs for obs in self._data.get(zone, []) if start <= obs.timestamp_utc < end]


class FakeLoadObservationRepository:
    """In-memory repo with the same dedup contract as the real Postgres adapter.

    Stores observations keyed by (zone, timestamp_utc) — the composite
    primary key the production schema uses. ``add_many`` returns the
    number of *new* rows so the use case's "observations_inserted" count
    matches what Postgres' ``ON CONFLICT DO NOTHING`` would return.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[BiddingZone, datetime], LoadObservation] = {}

    def add_many(self, observations: Iterable[LoadObservation]) -> int:
        new_count = 0
        for obs in observations:
            key = (obs.zone, obs.timestamp_utc)
            if key not in self._store:
                self._store[key] = obs
                new_count += 1
        return new_count

    def all(self) -> list[LoadObservation]:
        """Test-only helper: dump every stored observation for assertions."""
        return list(self._store.values())

    def find_by_zone(
        self,
        zone: BiddingZone,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[LoadObservation]:
        matching = [o for o in self._store.values() if o.zone == zone]
        if since is not None:
            matching = [o for o in matching if o.timestamp_utc >= since]
        if until is not None:
            matching = [o for o in matching if o.timestamp_utc < until]
        matching.sort(key=lambda o: o.timestamp_utc)
        return matching


class FakeWeatherClient:
    """Predetermined-data weather stand-in.

    Same shape as :class:`FakeEntsoeClient` — ``seed`` loads observed
    readings, ``seed_forecast`` loads forecasted readings, ``fail_on``
    flips a zone into raising :class:`DataSourceUnavailableError`. The
    two fetch methods read from independent stores so a test can prime
    one side without polluting the other.
    """

    def __init__(self) -> None:
        self._data: dict[BiddingZone, list[WeatherReading]] = {}
        self._forecast: dict[BiddingZone, list[WeatherReading]] = {}
        self._fail_on_zone: BiddingZone | None = None

    def seed(self, zone: BiddingZone, readings: Iterable[WeatherReading]) -> None:
        self._data[zone] = list(readings)

    def seed_forecast(self, zone: BiddingZone, readings: Iterable[WeatherReading]) -> None:
        self._forecast[zone] = list(readings)

    def fail_on(self, zone: BiddingZone) -> None:
        self._fail_on_zone = zone

    def fetch_weather(
        self,
        *,
        zone: BiddingZone,
        start: datetime,
        end: datetime,
    ) -> Iterable[WeatherReading]:
        if zone == self._fail_on_zone:
            raise DataSourceUnavailableError(f"Open-Meteo unavailable for {zone}")
        return [r for r in self._data.get(zone, []) if start <= r.timestamp_utc < end]

    def fetch_forecast(
        self,
        *,
        zone: BiddingZone,
        start: datetime,
        end: datetime,
    ) -> Iterable[WeatherReading]:
        if zone == self._fail_on_zone:
            raise DataSourceUnavailableError(f"Open-Meteo unavailable for {zone}")
        return [r for r in self._forecast.get(zone, []) if start <= r.timestamp_utc < end]


class FakeWeatherReadingRepository:
    """In-memory weather repo with the same dedup contract as production."""

    def __init__(self) -> None:
        self._store: dict[tuple[BiddingZone, datetime], WeatherReading] = {}

    def add_many(self, readings: Iterable[WeatherReading]) -> int:
        new_count = 0
        for r in readings:
            key = (r.zone, r.timestamp_utc)
            if key not in self._store:
                self._store[key] = r
                new_count += 1
        return new_count

    def all(self) -> list[WeatherReading]:
        return list(self._store.values())


@dataclass(frozen=True, slots=True)
class LogCall:
    """One captured log invocation. ``context`` includes any bound fields
    inherited from ``bind()`` plus per-call keyword arguments."""

    level: str
    event: str
    context: dict[str, Any]


@dataclass
class FakeLogger:
    """Records every log call into a shared list for assertion in tests.

    ``bind()`` returns a *new* FakeLogger that shares the recording list
    with its parent and merges in the additional context. This mirrors
    structlog's BoundLogger semantics — once bound, every subsequent
    call carries the bound fields automatically.
    """

    calls: list[LogCall] = field(default_factory=list)
    _bound: dict[str, Any] = field(default_factory=dict)

    def bind(self, **context: Any) -> "FakeLogger":
        return FakeLogger(calls=self.calls, _bound={**self._bound, **context})

    def _record(self, level: str, event: str, **context: Any) -> None:
        self.calls.append(LogCall(level=level, event=event, context={**self._bound, **context}))

    def debug(self, event: str, **context: Any) -> None:
        self._record("debug", event, **context)

    def info(self, event: str, **context: Any) -> None:
        self._record("info", event, **context)

    def warning(self, event: str, **context: Any) -> None:
        self._record("warning", event, **context)

    def error(self, event: str, **context: Any) -> None:
        self._record("error", event, **context)

    def events(self) -> list[str]:
        """Test convenience: list of recorded event names in order."""
        return [c.event for c in self.calls]


@dataclass(frozen=True, slots=True)
class _RegistryCall:
    registered_name: str
    params: dict[str, Any]
    metrics: dict[str, float]


@dataclass
class FakeModelRegistry:
    """Records every register() call and returns a deterministic ModelVersion.

    Same shape as :class:`MLflowModelRegistry` (in adapters/) but with no
    serialisation, no run-id generation, no network. The fake associates
    each registered model with the returned ModelVersion so ``load()``
    is symmetric with ``register()``.
    """

    calls: list[_RegistryCall] = field(default_factory=list)
    next_version: str = "fake_model@v1"
    _models: dict[str, Any] = field(default_factory=dict)
    _metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    _aliases: dict[tuple[str, str], ModelVersion] = field(default_factory=dict)

    def register(
        self,
        *,
        model: Any,
        registered_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
    ) -> ModelVersion:
        self.calls.append(
            _RegistryCall(
                registered_name=registered_name,
                params=dict(params),
                metrics=dict(metrics),
            )
        )
        version = ModelVersion(self.next_version)
        self._models[version.value] = model
        self._metrics[version.value] = dict(metrics)
        return version

    def load(self, version: ModelVersion) -> Any:
        # If the version is alias-form (no model stored under that
        # exact key), resolve via the alias map first.
        if version.value not in self._models:
            try:
                name, suffix = version.value.split("@", 1)
            except ValueError as exc:
                raise KeyError(f"FakeModelRegistry has no model for {version.value!r}") from exc
            resolved = self._aliases.get((name, suffix))
            if resolved is None:
                raise KeyError(f"FakeModelRegistry has no model for {version.value!r}")
            return self._models[resolved.value]
        return self._models[version.value]

    def get_alias(self, registered_name: str, alias: str) -> ModelVersion | None:
        return self._aliases.get((registered_name, alias))

    def get_metric(self, version: ModelVersion, metric_key: str) -> float | None:
        # Resolve alias to the underlying version first.
        resolved_key = version.value
        if resolved_key not in self._metrics:
            try:
                name, suffix = version.value.split("@", 1)
                resolved = self._aliases.get((name, suffix))
                if resolved is not None:
                    resolved_key = resolved.value
            except ValueError:
                pass
        metrics = self._metrics.get(resolved_key)
        if metrics is None:
            return None
        return metrics.get(metric_key)

    def set_alias(self, registered_name: str, alias: str, version: ModelVersion) -> None:
        self._aliases[(registered_name, alias)] = version

    def preload(self, version: ModelVersion, model: Any) -> None:
        """Test helper: pre-register a (version, model) pair without
        going through register(). Useful when the test only needs to
        exercise the load path."""
        self._models[version.value] = model

    def preload_metric(self, version: ModelVersion, key: str, value: float) -> None:
        """Test helper: attach a metric to a version that was
        ``preload()``ed rather than registered. Useful for setting up
        a champion's MAPE without running training."""
        self._metrics.setdefault(version.value, {})[key] = value


@dataclass
class FakeLoadForecastRepository:
    """In-memory forecast repo with the same dedup contract as production.

    Identity is the (zone, delivery_time, model_version) triple — same as
    the LocalFs adapter. Tests assert on :meth:`all` to inspect what the
    runner forwarded.
    """

    _store: dict[tuple[BiddingZone, datetime, str], LoadForecast] = field(default_factory=dict)

    def add_many(self, forecasts: Iterable[LoadForecast]) -> int:
        new_count = 0
        for f in forecasts:
            key = (f.zone, f.delivery_time, f.model_version.value)
            if key not in self._store:
                self._store[key] = f
                new_count += 1
        return new_count

    def all(self) -> list[LoadForecast]:
        return list(self._store.values())

    def find_by_zone(
        self,
        zone: BiddingZone,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[LoadForecast]:
        matching = [f for f in self._store.values() if f.zone == zone]
        if since is not None:
            matching = [f for f in matching if f.delivery_time >= since]
        if until is not None:
            matching = [f for f in matching if f.delivery_time < until]
        matching.sort(key=lambda f: f.delivery_time)
        return matching
