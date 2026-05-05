# Energy Forecaster — developer entrypoints.
# Every command runs through `uv run` so the project's locked virtualenv is used
# without anyone having to remember to activate it.

.PHONY: help install sync lock lint format typecheck test test-live check clean

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Create/refresh the locked virtualenv (incl. dev tools).
	uv sync

sync: install  ## Alias for install.

lock:  ## Regenerate uv.lock without installing.
	uv lock

lint:  ## Ruff lint + format check (no writes).
	uv run ruff check .
	uv run ruff format --check .

format:  ## Apply ruff auto-fixes and formatter.
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## Mypy strict over src/ and tests/.
	uv run mypy

test:  ## Run pytest with coverage gate (>=80%). Excludes `live` tests.
	uv run pytest

test-live:  ## Run live tests against real APIs (needs EF_ENTSOE_API_KEY).
	uv run pytest -m live --cov-fail-under=0 --no-cov

check: lint typecheck test  ## Lint + typecheck + test — what CI runs.

clean:  ## Remove caches and build artefacts.
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
