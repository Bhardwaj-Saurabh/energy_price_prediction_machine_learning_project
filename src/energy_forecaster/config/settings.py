"""Application configuration loaded from environment variables and .env files.

This module is the *only* place in the codebase that reads ``os.environ``
or otherwise interacts with the runtime environment to resolve config
values. All other layers receive a :class:`Settings` instance through the
composition root and treat it as an immutable, validated data object.

Two kinds of values live here:
  - Deployment-shaped values (which environment we're in, where local data
    lives, log level) — drive composition decisions.
  - Secrets (API keys, future DB passwords) — wrapped in :class:`SecretStr`
    so they cannot be accidentally logged via ``repr()`` or ``print()``.

Hyperparameters and business knobs (forecast horizons, model thresholds,
training hyperparameters) do NOT belong here — they go in Kedro's
``conf/`` YAML when that pipeline arrives. Settings is for *how* the app
is wired; YAML is for *what* the app does.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment — drives composition-root branching only.

    The rulebook forbids ``if env == "prod"`` checks inside business code.
    The composition module reads this once and selects concrete adapters
    accordingly; everywhere else, the chosen adapters are injected through
    constructors and the environment is invisible.
    """

    LOCAL = "local"
    PROD = "prod"


# Standard logging levels accepted by the stdlib logging module and structlog.
# Defined as a Literal type so Pydantic enforces it at construction time and
# typos like "INF0" (zero instead of O) fail fast instead of being silently
# treated as the WARNING fallback.
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Where weather data comes from. ``synthetic`` runs against the
# in-memory adapter (deterministic, no network — useful for demos, CI,
# and offline development). ``open_meteo`` hits the real, keyless
# Open-Meteo API. ENTSO-E uses ``entsoe_api_key`` presence as its
# discriminator because no key means no real call is possible; weather
# has no such constraint, so the choice is exposed explicitly here.
WeatherSource = Literal["synthetic", "open_meteo"]


class Settings(BaseSettings):
    """Typed, validated runtime configuration for the Energy Forecaster.

    Values are read from environment variables prefixed with ``EF_`` and,
    optionally, from a ``.env`` file in the working directory. Environment
    variables always win over the ``.env`` file. Unknown keys inside the
    ``.env`` file raise an error (``extra="forbid"``) — this catches typos
    like ``EF_ENVIROMENT`` instead of letting them fall back to defaults.

    Caveat: ``extra="forbid"`` does NOT catch unknown bare env vars, because
    pydantic-settings only consults env vars matching declared field names.
    Typo detection therefore holds for the ``.env`` file path but not for
    direct env-var sets — keep that in mind when debugging "my override is
    being ignored" symptoms in shells.

    Tests construct ``Settings(...)`` directly with keyword arguments to
    bypass the env layer entirely; production code should call
    :func:`get_settings` to receive a process-wide cached instance.
    """

    environment: Environment = Environment.LOCAL
    log_level: LogLevel = "INFO"

    # Where local-mode adapters write data on disk. The Path type tells
    # Pydantic to coerce strings from env into pathlib.Path; we do not
    # check for existence here because the directory may not exist yet at
    # config-load time (it gets created by the local-fs adapter).
    local_data_root: Path = Path("./data")

    # Optional during early phases — ingestion is not wired up yet. Once
    # the ENTSO-E adapter lands, the use case will assert this is set when
    # running in environments that need it; the absence is a configuration
    # error, not a domain error.
    entsoe_api_key: SecretStr | None = None

    weather_source: WeatherSource = "synthetic"

    @field_validator("entsoe_api_key", mode="before")
    @classmethod
    def _blank_string_is_unset(cls, value: object) -> object:
        """Treat empty / whitespace-only strings as 'not configured'.

        A user who copies ``.env.example`` will get placeholder lines like
        ``EF_ENTSOE_API_KEY=`` (no value). The intent is clearly "I have
        not set this", not "the key is the empty string". Without this
        coercion the composition root would pick the real ENTSO-E adapter
        and fire a request with an empty token, which fails confusingly.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    model_config = SettingsConfigDict(
        env_prefix="EF_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` instance.

    Cached so the .env file and environment are read exactly once per
    process. Tests that need a fresh instance can call
    ``get_settings.cache_clear()`` between runs, or construct
    ``Settings(...)`` directly with keyword overrides instead of going
    through this function.
    """
    return Settings()
