# Energy Forecaster

Real-Time Energy Demand & Price Forecaster for European bidding zones (DE-LU, FR, GB).
Production-grade MLOps project: Kedro pipelines, MLflow registry, Feast feature
store, Prefect orchestration, FastAPI serving — running locally first, with Azure
adapters added in later phases.

> **Status:** Phase 1, day 0 — project skeleton only. No data, no models yet.
> Full architectural rules live in [.claude/CLAUDE.md](.claude/CLAUDE.md).
> Product spec lives in [docs/PRD.docx](docs/PRD.docx).

## Prerequisites

- macOS or Linux
- [`uv`](https://docs.astral.sh/uv/) `>= 0.5` — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Python 3.12 (uv will fetch it if missing — version pinned in `.python-version`)
- `make` (preinstalled on macOS / most Linux)

## Quickstart

```bash
make install     # create locked .venv with dev tools
make test        # run pytest with >=80% coverage gate
make lint        # ruff lint + format check
make typecheck   # mypy --strict over src/ and tests/
make check       # everything CI runs (lint + typecheck + test)
make help        # list all targets
```

A green `make check` is the bar for "ready to merge".

## Project layout

```
.
├── .claude/CLAUDE.md     # engineering rulebook — clean architecture, config, ML rules
├── docs/PRD.docx         # product requirements document
├── src/energy_forecaster # the package (currently just version metadata)
├── tests/                # pytest tests (currently one smoke test)
├── Makefile              # developer entrypoints
├── pyproject.toml        # project + tooling config (ruff, mypy, pytest)
└── .python-version       # 3.12
```

The full module layout (domain / application / adapters / pipelines / serving)
is described in `.claude/CLAUDE.md` §1 and is built up chunk-by-chunk as we go —
empty folders are not pre-created.

## Roadmap

| Phase | Scope                                                            |
|-------|------------------------------------------------------------------|
| 1     | **Skeleton + ingestion** (ENTSO-E, Open-Meteo) — local Postgres, local FS |
| 2     | Feature engineering, Feast, training pipeline, MLflow            |
| 3     | FastAPI serving, batch inference flow, Dash dashboard            |
| 4     | Evidently monitoring, retraining triggers, structured observability |
| 5     | Lift to Azure (Container Apps, Blob, PostgreSQL Flexible, Key Vault) |

Currently inside Phase 1. The OSS layer is identical between local and Azure;
only adapters change.
