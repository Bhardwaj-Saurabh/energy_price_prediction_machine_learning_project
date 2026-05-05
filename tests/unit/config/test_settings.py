"""Unit tests for the Settings loader.

Strategy: every test that exercises env-var behaviour uses ``monkeypatch``
to clear the entire ``EF_`` namespace first, then sets exactly what it
wants to test. ``_env_file=None`` is passed to Settings to ensure no
ambient ``.env`` file in the working directory leaks into a test run.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from energy_forecaster.config.settings import (
    Environment,
    Settings,
    get_settings,
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every EF_-prefixed env var so tests are deterministic."""
    import os

    for key in list(os.environ):
        if key.upper().startswith("EF_"):
            monkeypatch.delenv(key, raising=False)
    # Each test gets a fresh cache for get_settings().
    get_settings.cache_clear()


def _settings_no_env_file(**kwargs: object) -> Settings:
    """Construct Settings without consulting any .env file in CWD."""
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg, arg-type]


class TestDefaults:
    def test_defaults_apply_when_nothing_is_set(self) -> None:
        s = _settings_no_env_file()
        assert s.environment is Environment.LOCAL
        assert s.log_level == "INFO"
        assert s.local_data_root == Path("./data")
        assert s.entsoe_api_key is None


class TestEnvironmentVariableLoading:
    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EF_ENVIRONMENT", "prod")
        monkeypatch.setenv("EF_LOG_LEVEL", "DEBUG")
        s = _settings_no_env_file()
        assert s.environment is Environment.PROD
        assert s.log_level == "DEBUG"

    def test_env_vars_are_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ef_environment", "prod")
        s = _settings_no_env_file()
        assert s.environment is Environment.PROD

    def test_path_env_var_is_coerced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EF_LOCAL_DATA_ROOT", "/tmp/forecaster")
        s = _settings_no_env_file()
        assert s.local_data_root == Path("/tmp/forecaster")
        assert isinstance(s.local_data_root, Path)


class TestValidation:
    def test_invalid_environment_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EF_ENVIRONMENT", "staging")
        with pytest.raises(ValidationError):
            _settings_no_env_file()

    def test_invalid_log_level_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Catches typos like INF0 (zero) instead of silently falling back.
        monkeypatch.setenv("EF_LOG_LEVEL", "INF0")
        with pytest.raises(ValidationError):
            _settings_no_env_file()

    def test_unknown_key_in_env_file_is_rejected(self, tmp_path: Path) -> None:
        # `extra="forbid"` catches typos in `.env` files (e.g.
        # EF_ENVIROMENT instead of EF_ENVIRONMENT). Note: pydantic-settings
        # does NOT raise for unknown bare env vars — env-var loading only
        # consults names matching declared fields, so an unknown one is
        # simply not picked up. The .env file path, where every line is
        # parsed, is what this assertion guards.
        env_file = tmp_path / ".env"
        env_file.write_text("EF_ENVIROMENT=prod\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            Settings(_env_file=env_file)  # type: ignore[call-arg]


class TestSecretMasking:
    def test_entsoe_api_key_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EF_ENTSOE_API_KEY", "very-secret-token")
        s = _settings_no_env_file()
        assert s.entsoe_api_key is not None
        assert s.entsoe_api_key.get_secret_value() == "very-secret-token"

    def test_empty_string_resolves_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Common ``.env.example`` pattern: ``EF_ENTSOE_API_KEY=`` (no
        # value, signalling "I have not set this"). The Settings layer
        # coerces this to None so the composition root never sees an
        # empty SecretStr it would treat as "configured".
        monkeypatch.setenv("EF_ENTSOE_API_KEY", "")
        assert _settings_no_env_file().entsoe_api_key is None

    def test_whitespace_only_string_resolves_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EF_ENTSOE_API_KEY", "   ")
        assert _settings_no_env_file().entsoe_api_key is None

    def test_secret_value_is_masked_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # SecretStr is the only thing that prevents accidental logging of
        # credentials — confirm that converting Settings to its string
        # form does not leak the underlying value.
        monkeypatch.setenv("EF_ENTSOE_API_KEY", "very-secret-token")
        s = _settings_no_env_file()
        assert "very-secret-token" not in repr(s)
        assert "very-secret-token" not in str(s)


class TestEnvFileLoading:
    def test_env_file_values_are_loaded(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "EF_ENVIRONMENT=prod\nEF_LOG_LEVEL=ERROR\n",
            encoding="utf-8",
        )
        s = Settings(_env_file=env_file)  # type: ignore[call-arg]
        assert s.environment is Environment.PROD
        assert s.log_level == "ERROR"

    def test_env_var_beats_env_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("EF_LOG_LEVEL=DEBUG\n", encoding="utf-8")
        monkeypatch.setenv("EF_LOG_LEVEL", "ERROR")
        s = Settings(_env_file=env_file)  # type: ignore[call-arg]
        assert s.log_level == "ERROR"


class TestKeywordOverrideForTests:
    def test_kwargs_construction_bypasses_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Tests should be able to pin every value explicitly — confirms the
        # pattern documented in the Settings docstring.
        monkeypatch.setenv("EF_ENVIRONMENT", "prod")
        s = _settings_no_env_file(environment=Environment.LOCAL)
        assert s.environment is Environment.LOCAL


class TestGetSettingsCaching:
    def test_get_settings_returns_same_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EF_ENVIRONMENT", "prod")
        a = get_settings()
        b = get_settings()
        assert a is b

    def test_cache_clear_picks_up_new_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EF_ENVIRONMENT", "prod")
        first = get_settings()
        assert first.environment is Environment.PROD

        monkeypatch.setenv("EF_ENVIRONMENT", "local")
        # Without clearing, the cached instance still wins.
        assert get_settings().environment is Environment.PROD

        get_settings.cache_clear()
        assert get_settings().environment is Environment.LOCAL
