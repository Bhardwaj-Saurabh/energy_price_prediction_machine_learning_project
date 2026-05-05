# Energy Forecaster — Engineering Rulebook

This file is the contract for how this project is built. Claude Code loads it into
every conversation; the human reads it to learn *why* a production ML system looks
the way it does. Rules below are non-negotiable unless the human explicitly
overrides one in conversation. If a rule conflicts with a request, surface the
conflict before acting.

---

## 0. Mission and working agreement

**Mission.** Build a production-grade Real-Time Energy Demand & Price Forecaster
(day-ahead MW + EUR/MWh for DE-LU, FR, GB) as a portfolio piece. Full PRD lives in
`docs/PRD.docx`. The architectural targets (Azure + OSS hybrid, Kedro / Prefect /
Feast / MLflow / FastAPI) are summarised in Claude's project memory.

**Working agreement.**
- Build in **small reviewable chunks**. One concern per chunk. Propose before
  implementing. Wait for "go" on anything beyond a trivial edit.
- **Teach as we go.** Every new tool or pattern gets a 2-3 sentence "what is this,
  why this one over alternatives" before it lands.
- **Local first, Azure later.** Phases 1-3 of build run entirely on the laptop
  with local equivalents (local Postgres, filesystem, `.env`). Azure adapters
  (Blob, Key Vault, Container Apps, App Insights, Event Grid) are added only
  when the human says we're lifting to cloud. The OSS layer (Kedro, Prefect,
  Feast, MLflow, FastAPI, DVC, Pandera) is identical to the cloud target — that
  is the part being learned and it must lift unchanged.
- **No premature scaffolding.** Do not create empty folders or stub modules for
  stages we have not reached. The repo grows with the work, not ahead of it.

---

## 1. Architecture — Clean Architecture, ML edition

The codebase is organised into **four layers**. Dependencies point inward only.
An inner layer never imports an outer one. This is what makes the local→Azure
migration a swap of adapters rather than a rewrite, and what makes the domain
testable without spinning up infrastructure.

```
┌────────────────────────────────────────────────────────────┐
│  Frameworks & Drivers                                      │  outer
│    Kedro pipelines, Prefect flows, FastAPI app, Dash app,  │
│    CLI entrypoints, Docker, Bicep                          │
├────────────────────────────────────────────────────────────┤
│  Adapters (Infrastructure)                                 │
│    EntsoePyClient, OpenMeteoHttpClient,                    │
│    FeastFeatureStore, MLflowModelRegistry,                 │
│    LocalFsObjectStore / BlobObjectStore,                   │
│    PostgresForecastRepo, StructlogLogger                   │
├────────────────────────────────────────────────────────────┤
│  Application (Use Cases)                                   │
│    IngestEntsoeLoad, BuildFeatureMatrix,                   │
│    TrainChampionModel, ScoreNextHorizon,                   │
│    ServePriceForecast, EvaluateChallenger                  │
├────────────────────────────────────────────────────────────┤
│  Domain                                                    │  inner
│    Entities: BiddingZone, LoadObservation,                 │
│      WeatherReading, PriceForecast, ModelVersion           │
│    Value objects: HorizonHours, MAPE, EnergyMW, PriceEUR   │
│    Pure rules: forecast horizons, holiday adjustments,     │
│      challenger-promotion policy                           │
└────────────────────────────────────────────────────────────┘
```

### The dependency rule (read this twice)

- **Domain** imports nothing from the project except other domain. No pandas, no
  numpy in entity definitions. No `requests`, `azure.*`, `mlflow.*`, `feast.*`,
  `kedro.*`, `fastapi.*`. Pure Python + dataclasses + typing.
- **Application** imports domain and **ports** (Protocol interfaces). Never
  imports a concrete adapter. Never imports a framework.
- **Adapters** import application + ports + their specific third-party library.
  This is the only place `mlflow`, `feast`, `entsoe-py`, `azure-*`, `psycopg`
  may be imported.
- **Frameworks** wire everything together. Kedro nodes call use cases. FastAPI
  routes call use cases. They never contain business logic themselves.

### Ports and adapters (the swap mechanism)

Every external dependency is reached through a **Protocol** defined in the
application layer. Concrete implementations live in adapters. Swapping local
filesystem for Azure Blob is changing one line in the composition root.

```python
# src/energy_forecaster/application/ports/object_store.py
from typing import Protocol
from pathlib import PurePosixPath

class ObjectStore(Protocol):
    def put(self, key: PurePosixPath, data: bytes) -> None: ...
    def get(self, key: PurePosixPath) -> bytes: ...
    def exists(self, key: PurePosixPath) -> bool: ...

# src/energy_forecaster/adapters/object_store/local_fs.py  (Phase 1-3)
class LocalFsObjectStore:  # implements ObjectStore structurally
    ...

# src/energy_forecaster/adapters/object_store/azure_blob.py  (Phase 4+)
class BlobObjectStore:  # implements ObjectStore structurally
    ...
```

The application code only ever sees `ObjectStore`. It does not know or care
which one it got. **This is the entire payoff of the architecture.**

### Module map (gets created chunk-by-chunk, not up-front)

```
src/energy_forecaster/
  domain/              # pure, no I/O, no frameworks
    entities/
    value_objects/
    rules/
  application/         # use cases + port definitions
    use_cases/
    ports/
    errors.py
  adapters/            # one folder per port; concrete impls
    object_store/
    feature_store/
    model_registry/
    forecast_repo/
    weather_client/
    entsoe_client/
    clock/
    logger/
  pipelines/           # Kedro pipelines (frameworks)
  flows/               # Prefect flows (frameworks)
  serving/             # FastAPI app (frameworks)
  dashboard/           # Dash app (frameworks)
  contracts/           # Pandera schemas — boundary contracts
  config/              # Pydantic Settings, env loading
  composition.py       # the ONLY place adapters are wired to ports
```

### The composition root

There is exactly **one** module that knows about both ports and concrete
adapters: `composition.py`. It reads config, instantiates the right adapter for
the current environment, and hands it to the use case. Every other module
receives its dependencies through constructor injection.

```python
# composition.py — illustrative
def build_ingest_use_case(settings: Settings) -> IngestEntsoeLoad:
    object_store: ObjectStore = (
        LocalFsObjectStore(settings.local_data_root)
        if settings.environment == "local"
        else BlobObjectStore(settings.blob_container_url)
    )
    return IngestEntsoeLoad(
        entsoe=EntsoePyClient(settings.entsoe_api_key),
        store=object_store,
        clock=SystemClock(),
        logger=structlog.get_logger(),
    )
```

If the human ever sees `if env == "prod"` inside a use case or domain module,
that is a bug. Branch on environment in composition and config, never in
business code.

---

## 2. Configuration — config-based, not code-based

A production ML system has dozens of knobs (zones, lookback windows, model
hyperparameters, thresholds, retraining cadence). They all live in config, not
in code. Code reads config. Code never embeds tunables.

### Two kinds of config

1. **Hyperparameters and business knobs** → YAML in `conf/` (Kedro convention).
   Versioned in git. Diffable. Reviewable.
2. **Secrets and environment-specific URLs** → environment variables, loaded
   into a Pydantic `Settings` class. `.env` for local dev (gitignored). Key
   Vault for Azure later. Never both. Never inline.

### Layout

```
conf/
  base/                # defaults — committed
    catalog.yml        # Kedro DataCatalog: where every dataset lives
    parameters.yml     # business knobs (zones, horizons, thresholds)
    parameters_train.yml
    parameters_serve.yml
    logging.yml
  local/               # local-dev overrides — committed (no secrets)
    catalog.yml        # points to data/01_raw, data/02_intermediate, etc.
  prod/                # cloud overrides — committed (no secrets)
    catalog.yml        # points to azure://... URIs

.env                   # GITIGNORED — secrets only (ENTSOE_API_KEY, DB_URL)
.env.example           # COMMITTED — same keys, dummy values
```

### Rules

- **No magic numbers in code.** Lookback hours, horizon, train/test split, model
  hyperparameters, optuna trial count, MAPE thresholds, PSI thresholds — all in
  YAML. If you find yourself typing a number into a function, stop and ask
  whether it belongs in `parameters.yml`.
- **No `os.environ.get()` outside `config/`.** All env access goes through one
  Pydantic `Settings` class. Other modules import `settings: Settings`, never
  `os.environ`.
- **No string-typed config.** Pydantic gives you typed, validated config. Use
  it. Bidding zones are an `Enum`. URLs are `AnyUrl`. Paths are `Path`.
- **Same code, different config.** The Kedro pipeline that runs locally is the
  same pipeline that runs in Azure. Only the catalog changes.

---

## 3. Separation of concerns — what goes where

Each module has one reason to change. If a module would change for two
unrelated reasons (a new data source AND a new model), it is two modules.

### Pipelines never mix concerns

- **Ingestion pipeline** fetches raw data and writes it. It never transforms
  features, never trains, never scores.
- **Feature pipeline** reads raw data, builds features, writes feature tables.
  It does not fetch and does not train.
- **Training pipeline** reads features, trains, registers model. It does not
  fetch, does not transform, does not serve.
- **Inference pipeline** loads a model, scores features, writes forecasts. It
  does not train.
- **Monitoring pipeline** reads forecasts and actuals, computes drift/MAPE,
  emits metrics. It does not train (it triggers training).

These boundaries map onto Kedro pipeline modules. They map onto separate
container images. They map onto separate Prefect flows. The boundary is
deliberate.

### Logging, errors, and time

- **Logging.** Structured (`structlog`). Adapters log the call. Use cases log
  the decision. Domain does not log. Every log line in a request flow carries
  the same `correlation_id` (set at the entrypoint, propagated via context).
- **Errors.** Domain raises domain exceptions (`BiddingZoneNotSupported`,
  `ForecastHorizonInvalid`). Adapters translate infra errors into application
  errors (`HTTPError → DataSourceUnavailable`). Use cases catch application
  errors and decide. Frameworks return appropriate HTTP/exit codes.
- **Time.** Never call `datetime.now()` directly outside `adapters/clock/`.
  Inject a `Clock` port. Tests pass a `FrozenClock`. This is the single
  highest-leverage testability rule in the system — without it, time-windowed
  features and freshness checks become untestable.

---

## 4. Data contracts — Pandera at every boundary

DataFrames cross module boundaries. Without contracts, a schema change
silently breaks a downstream stage and you find out in production.

- Every dataset in the Kedro catalog has a corresponding Pandera schema in
  `src/energy_forecaster/contracts/`.
- Every Kedro node that produces or consumes a DataFrame is decorated with
  `@pa.check_types`. Mypy enforces the type. Pandera enforces the schema at
  runtime.
- Schema changes are PR-reviewable, breakage-visible.

```python
# contracts/load_observation.py
import pandera as pa
from pandera.typing import Series, DataFrame

class LoadObservationSchema(pa.DataFrameModel):
    timestamp_utc: Series[pa.DateTime] = pa.Field(unique=True)
    bidding_zone: Series[str] = pa.Field(isin=["DE_LU", "FR", "GB"])
    load_mw: Series[float] = pa.Field(ge=0, le=200_000)
```

---

## 5. Testing — the pyramid is real

- **Unit tests** on domain and use cases. No I/O. Sub-millisecond. Many.
  Live in `tests/unit/`. Run on every save.
- **Integration tests** on adapters. Use `testcontainers` for Postgres,
  recorded HTTP for ENTSO-E and Open-Meteo (`vcrpy` or fixtures). Live in
  `tests/integration/`.
- **Contract tests** on the FastAPI app. Schemathesis generates from the
  OpenAPI schema. Live in `tests/api/`.
- **End-to-end tests** on the whole pipeline. One happy-path, runs in CI on a
  schedule. Live in `tests/e2e/`.

**Coverage target: >80% on application + adapters.** Domain should approach
100% — there is no excuse, it is pure code.

**Forbidden in tests:**
- Mocking the database. Use a real Postgres via testcontainers. (Mocked DB
  tests pass when migrations are broken — this is a real failure mode that has
  shipped to prod in the wild.)
- Mocking your own code. If you are mocking a class you wrote, the design is
  wrong — inject a port and pass a fake.
- Sleeping for time. Use the `Clock` port and advance it.

---

## 6. Code rules

### Always

- Type-annotate everything. `mypy --strict` is the contract.
- Inject dependencies through constructors. No module-level singletons except
  in `composition.py`.
- Use `pathlib.Path`, never `os.path`.
- Use `structlog` for diagnostics, never `print()`.
- Use named constants or config for any number that has meaning.
- Write a one-line module docstring stating the module's single responsibility.

### Never

- `from x import *`.
- Mutable default arguments (`def f(xs=[])`).
- Catching `Exception` without re-raising or deliberately recovering. Every
  except block must log AND either re-raise or take a documented action.
- `pickle` for model artifacts (MLflow handles serialisation; pickle is a
  security and forward-compatibility hazard).
- Notebooks committed to `src/` or `tests/`. If exploration is needed, use
  `notebooks/` (gitignored or `nbstripout`-cleaned).
- A `utils.py` dumping ground. Name modules by what they do.
- Direct `os.environ` access outside `config/`.
- Direct `datetime.now()` / `time.time()` outside `adapters/clock/`.
- Branching on `environment == "prod"` inside business code. That branch lives
  in `composition.py`.
- Importing `azure.*`, `mlflow.*`, `feast.*`, `kedro.*`, `fastapi.*`,
  `entsoe.*`, `prefect.*` from `domain/` or `application/`. Adapter-only.

### Style

- `ruff` for lint and format. `ruff check --fix` and `ruff format` are the only
  formatters. Configured in `pyproject.toml`.
- Line length 100. Imports sorted by ruff (`I` rule). Typed dicts over plain
  dicts when the shape is known.
- Public functions get docstrings. Private functions earn comments only when
  the *why* is non-obvious.

---

## 7. ML-specific rules

These are easy to get wrong and expensive to fix.

- **Training-serving skew is the enemy.** The same feature transformation must
  run at training time and serving time. Feast is how we enforce this — both
  paths read from a feature definition, not from ad-hoc pandas. If you find
  yourself writing a transformation in two places, stop.
- **Time leakage is the silent killer.** Splits are time-ordered, never random.
  Feature lookbacks reference only data available at `as_of_time`. Tests must
  prove this with a `FrozenClock`.
- **Every model artifact is registered.** Models go through the
  `ModelRegistry` port (MLflow today). No `joblib.dump()` to a folder. No
  loading by file path. The only way to load a model in serving code is by
  alias (`@champion`).
- **Every model gets a model card.** Generated as part of the training
  pipeline. Lives next to the artifact. Includes: dataset window, features
  used, hyperparameters, eval metrics, intended use, known limitations.
- **Champion/challenger promotion is policy, not vibes.** Promotion rule lives
  in `domain/rules/promotion.py` (e.g. challenger MAPE must beat champion by
  ≥ 0.5pp on the test window). Code calls the rule; humans don't override it.
- **Retraining triggers are config-driven.** PSI > 0.2, KS p < 0.05, rolling
  MAPE thresholds, monthly forced — all values live in `parameters.yml`, the
  monitoring pipeline reads them, and the same pipeline emits the trigger
  signal.

---

## 8. Tooling — what Claude reaches for

When working on this project, prefer these in order:

- **`Plan` agent** before any non-trivial chunk (anything touching > 2 files or
  introducing a new layer). The plan is reviewed *with the human* before
  implementation begins.
- **`Explore` agent** for codebase searches once we have more than ~20 files.
- **`simplify` skill** before declaring a chunk done. Reviews the change for
  reuse opportunities and avoidable complexity.
- **`review` skill** before opening a PR.
- **`security-review` skill** before any push that touches auth, secrets, or
  cloud surface area.
- **`TodoWrite`** to track multi-step chunks visibly. One todo per concrete
  step. Mark done immediately on completion, never batch.

If unsure whether a chunk is "non-trivial enough" to warrant `Plan`, err on the
side of planning. The point of small chunks is reviewability; a plan is the
cheapest review.

---

## 9. Definition of done — for any chunk

A chunk is done when **all** of the following are true. If any is false, the
chunk is not done; do not declare success.

1. **Tests added or updated.** New behaviour has at least one unit test;
   adapters have at least one integration test.
2. **`make test` passes.** Locally, with no skipped tests.
3. **`make lint` passes.** Ruff clean. Mypy strict clean.
4. **Pandera schemas updated** if a DataFrame boundary changed.
5. **Config changes documented.** New keys appear in `parameters.yml` /
   `.env.example` with comments explaining what they do.
6. **No TODOs or commented-out code** left behind. If something is deferred,
   it is a real issue/note, not a comment.
7. **Human-facing summary** stated: what changed, what they can now run, what
   the next chunk is.

---

## 10. Glossary (so the human is never lost)

- **Bidding zone.** A geographic area with a single wholesale electricity
  price. We forecast for DE-LU, FR, GB.
- **Day-ahead.** A market that clears today for delivery hours tomorrow. Our
  primary forecast horizon.
- **MAPE.** Mean Absolute Percentage Error. Our headline accuracy metric.
- **PSI.** Population Stability Index. Detects feature distribution drift.
- **Champion / challenger.** The currently-deployed model vs. a candidate
  trying to replace it.
- **Feature store.** A system (Feast) that defines features once and serves
  them consistently to training (offline) and inference (online).
- **Online store.** Low-latency feature lookup for serving. Local: SQLite.
- **Offline store.** Historical feature retrieval for training. Local:
  Postgres.
- **Composition root.** The single module that wires concrete adapters to
  ports and hands them to use cases. Every DI framework's job, but we do it by
  hand because it is one file.
- **Port.** A `Protocol` interface owned by the application layer.
- **Adapter.** A concrete implementation of a port, owned by the
  infrastructure layer.

---

## 11. When in doubt

- Check `docs/PRD.docx` for product intent.
- Check Claude's project memory for architectural decisions.
- Check this file for engineering rules.
- If still ambiguous, **ask** before implementing. A question is cheaper than a
  rewrite.
