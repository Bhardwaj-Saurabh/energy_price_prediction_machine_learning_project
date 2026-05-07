"""Unit tests for the CLI argument parser and command handlers.

The CLI invokes the real composition root, which builds the real
LocalFs adapter and writes to disk — so these tests run with
``EF_LOCAL_DATA_ROOT`` pointed at ``tmp_path`` to keep the test
filesystem isolated. We do not mock out the use case; we let it run end
to end against the synthetic InMemoryEntsoeClient.
"""

import json
from pathlib import Path

import pytest

from energy_forecaster.cli import main
from energy_forecaster.config.settings import get_settings


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin the data root and neutralise the developer's local ``.env``.

    We can't easily disable .env loading from the CLI path (Settings
    reads the file in its config), so we override every key the file
    might define. ``EF_ENTSOE_API_KEY=""`` is critical: without it, a
    developer's real API key from .env would leak into tests and the
    composition root would pick the live adapter.
    """
    import os

    for key in list(os.environ):
        if key.upper().startswith("EF_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EF_LOCAL_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("EF_ENTSOE_API_KEY", "")
    # Default weather source for CLI tests is the deterministic synthetic
    # adapter — no network surprises in CI.
    monkeypatch.setenv("EF_WEATHER_SOURCE", "synthetic")
    # Silence structlog output so CLI-stdout assertions are not polluted
    # by log lines. The logging logic itself is exercised in the use case
    # and adapter test suites.
    monkeypatch.setenv("EF_LOG_LEVEL", "CRITICAL")
    get_settings.cache_clear()


class TestArgumentParsing:
    def test_help_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        # argparse exits with code 0 on --help. Catching SystemExit confirms
        # the parser is wired without invoking any business logic.
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "energy-forecaster" in captured.out

    def test_missing_command_exits_two(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2

    def test_unknown_zone_exits_two(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "ingest",
                    "--zone",
                    "ES",  # not a supported BiddingZone value
                    "--start",
                    "2026-05-04",
                    "--end",
                    "2026-05-05",
                ]
            )
        assert exc.value.code == 2

    def test_naive_timestamp_with_time_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "ingest",
                    "--zone",
                    "DE_LU",
                    "--start",
                    "2026-05-04T12:00:00",
                    "--end",
                    "2026-05-05",
                ]
            )
        assert exc.value.code == 2

    def test_invalid_bare_date_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "ingest",
                    "--zone",
                    "DE_LU",
                    "--start",
                    "not-a-date",
                    "--end",
                    "2026-05-05",
                ]
            )
        assert exc.value.code == 2

    def test_invalid_iso_timestamp_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "ingest",
                    "--zone",
                    "DE_LU",
                    "--start",
                    "2026-05-04Tbroken",
                    "--end",
                    "2026-05-05",
                ]
            )
        assert exc.value.code == 2


class TestApplicationErrorHandling:
    def test_use_case_raising_application_error_returns_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Simulate an ENTSO-E outage by injecting a use-case stand-in that
        # raises DataSourceUnavailableError. We patch the import site in
        # cli.py — that is the seam between the framework layer and the
        # composition root.
        from energy_forecaster.application.errors import DataSourceUnavailableError

        class _FailingUseCase:
            def execute(self, **kwargs: object) -> None:
                raise DataSourceUnavailableError("simulated upstream outage")

        monkeypatch.setattr(
            "energy_forecaster.cli.build_ingest_entsoe_load",
            lambda settings, *, logger: _FailingUseCase(),
        )

        exit_code = main(
            [
                "ingest",
                "--zone",
                "DE_LU",
                "--start",
                "2026-05-04",
                "--end",
                "2026-05-05",
            ]
        )

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "simulated upstream outage" in captured.err


class TestIngestEndToEnd:
    def test_single_zone_one_day_writes_24_observations(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        exit_code = main(
            [
                "ingest",
                "--zone",
                "DE_LU",
                "--start",
                "2026-05-04",
                "--end",
                "2026-05-05",
            ]
        )
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Observations fetched:  24" in captured.out
        assert "Observations inserted: 24" in captured.out

        # And the JSONL file actually exists with 24 records.
        jsonl = tmp_path / "load_observations" / "DE_LU.jsonl"
        assert jsonl.exists()
        records = [json.loads(line) for line in jsonl.read_text().splitlines()]
        assert len(records) == 24
        assert all(r["zone"] == "DE_LU" for r in records)

    def test_multiple_zones(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        exit_code = main(
            [
                "ingest",
                "--zone",
                "DE_LU",
                "--zone",
                "FR",
                "--start",
                "2026-05-04",
                "--end",
                "2026-05-05",
            ]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Zones processed:       2" in captured.out
        assert "Observations fetched:  48" in captured.out

        load_dir = tmp_path / "load_observations"
        assert {p.name for p in load_dir.iterdir()} == {
            "DE_LU.jsonl",
            "FR.jsonl",
        }

    def test_rerunning_same_window_inserts_zero(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        for _ in range(2):
            exit_code = main(
                [
                    "ingest",
                    "--zone",
                    "DE_LU",
                    "--start",
                    "2026-05-04",
                    "--end",
                    "2026-05-05",
                ]
            )
            assert exit_code == 0

        captured = capsys.readouterr()
        # Last printed output is the second run; assert the zero-insert.
        assert "Observations inserted: 0" in captured.out


class TestWeatherSubcommand:
    def test_weather_command_runs_end_to_end(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        exit_code = main(
            [
                "weather",
                "--zone",
                "DE_LU",
                "--start",
                "2026-05-04",
                "--end",
                "2026-05-05",
            ]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Weather ingest complete" in captured.out
        assert "Readings fetched:   24" in captured.out
        assert "Readings inserted:  24" in captured.out

        jsonl = tmp_path / "weather_readings" / "DE_LU.jsonl"
        assert jsonl.exists()
        assert len(jsonl.read_text().splitlines()) == 24

    def test_weather_command_dedups_on_rerun(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        for _ in range(2):
            assert (
                main(
                    [
                        "weather",
                        "--zone",
                        "DE_LU",
                        "--start",
                        "2026-05-04",
                        "--end",
                        "2026-05-05",
                    ]
                )
                == 0
            )
        captured = capsys.readouterr()
        assert "Readings inserted:  0" in captured.out

    def test_weather_command_propagates_application_error_as_exit_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from energy_forecaster.application.errors import DataSourceUnavailableError

        class _FailingUseCase:
            def execute(self, **kwargs: object) -> None:
                raise DataSourceUnavailableError("simulated weather outage")

        monkeypatch.setattr(
            "energy_forecaster.cli.build_ingest_weather",
            lambda settings, *, logger: _FailingUseCase(),
        )

        exit_code = main(
            [
                "weather",
                "--zone",
                "DE_LU",
                "--start",
                "2026-05-04",
                "--end",
                "2026-05-05",
            ]
        )
        assert exit_code == 1
        assert "simulated weather outage" in capsys.readouterr().err


class TestFeaturesSubcommand:
    """End-to-end checks for ``energy-forecaster features``.

    Each test stages JSONL inputs under the configured data root, runs
    the CLI, and asserts on both the printed result and the Parquet
    output. The fixture's ``EF_LOCAL_DATA_ROOT=tmp_path`` is what makes
    these runs hermetic.
    """

    @staticmethod
    def _seed_jsonl(tmp_path: Path, hours: int = 200) -> None:
        from datetime import UTC, datetime, timedelta

        load_dir = tmp_path / "load_observations"
        weather_dir = tmp_path / "weather_readings"
        load_dir.mkdir(parents=True, exist_ok=True)
        weather_dir.mkdir(parents=True, exist_ok=True)
        start = datetime(2026, 5, 4, tzinfo=UTC)

        with (load_dir / "DE_LU.jsonl").open("w", encoding="utf-8") as f:
            for h in range(hours):
                ts = (start + timedelta(hours=h)).isoformat()
                f.write(
                    json.dumps(
                        {
                            "zone": "DE_LU",
                            "timestamp_utc": ts,
                            "load": 50_000.0 + 100.0 * h,
                        }
                    )
                    + "\n"
                )
        with (weather_dir / "DE_LU.jsonl").open("w", encoding="utf-8") as f:
            for h in range(hours):
                ts = (start + timedelta(hours=h)).isoformat()
                f.write(
                    json.dumps(
                        {
                            "zone": "DE_LU",
                            "timestamp_utc": ts,
                            "temp_c": 15.0,
                            "wind_10m_ms": 4.0,
                            "wind_100m_ms": 8.0,
                            "ghi_wm2": 300.0,
                            "cloud_cover_pct": 50.0,
                            "precip_mm": 0.0,
                        }
                    )
                    + "\n"
                )

    def test_default_output_path_under_data_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._seed_jsonl(tmp_path)

        exit_code = main(["features"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Feature engineering complete" in captured.out
        assert "Rows:      200" in captured.out
        assert (tmp_path / "features.parquet").exists()

    def test_explicit_output_path_is_respected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._seed_jsonl(tmp_path)
        custom = tmp_path / "subdir" / "my_features.parquet"

        exit_code = main(["features", "--output", str(custom)])

        assert exit_code == 0
        assert custom.exists()
        # Default location should NOT exist when --output is set.
        assert not (tmp_path / "features.parquet").exists()

    def test_pipeline_failure_returns_exit_one(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Inject a runner that raises — simulates a Pandera / Kedro /
        # filesystem error path. The CLI must catch, log, and return 1
        # rather than letting the stack trace escape to the user.
        from collections.abc import Callable

        def _failing(output_path: Path | None = None) -> Path:
            raise RuntimeError("simulated pipeline failure")

        def _build_failing_runner(
            settings: object,
        ) -> Callable[[Path | None], Path]:
            return _failing

        monkeypatch.setattr(
            "energy_forecaster.cli.build_run_feature_engineering",
            _build_failing_runner,
        )

        exit_code = main(["features"])

        assert exit_code == 1
        assert "simulated pipeline failure" in capsys.readouterr().err


class TestTrainSubcommand:
    """End-to-end checks for ``energy-forecaster train``.

    The runner builds an MLflowModelRegistry under the hood, which is
    safe to instantiate without a tracking server (no I/O until
    ``register`` is called). We monkeypatch the runner to a fake to
    keep these tests hermetic and fast — the real-MLflow path is
    covered by the integration test in
    ``tests/integration/adapters/model_registry/``.
    """

    def test_runs_via_fake_runner_and_returns_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from collections.abc import Callable
        from datetime import UTC, datetime

        from energy_forecaster.domain.value_objects.model_version import (
            ModelVersion,
        )
        from energy_forecaster.pipelines.training.runner import TrainingResult

        def _runner(features_path: Path | None = None) -> TrainingResult:
            now = datetime(2026, 5, 7, 12, tzinfo=UTC)
            return TrainingResult(
                model_version=ModelVersion("test_model@v1"),
                train_size=80,
                test_size=20,
                test_mape=0.07,
                started_at=now,
                finished_at=now,
            )

        def _build(settings: object) -> Callable[[Path | None], TrainingResult]:
            return _runner

        monkeypatch.setattr("energy_forecaster.cli.build_run_training", _build)

        exit_code = main(["train"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Training complete" in captured.out
        assert "test_model@v1" in captured.out
        assert "Test MAPE:     0.0700" in captured.out

    def test_runner_failure_returns_exit_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from collections.abc import Callable

        from energy_forecaster.pipelines.training.runner import TrainingResult

        def _runner(features_path: Path | None = None) -> TrainingResult:
            raise RuntimeError("simulated training failure")

        def _build(settings: object) -> Callable[[Path | None], TrainingResult]:
            return _runner

        monkeypatch.setattr("energy_forecaster.cli.build_run_training", _build)

        exit_code = main(["train"])

        assert exit_code == 1
        assert "simulated training failure" in capsys.readouterr().err


class TestTimestampParsing:
    def test_full_iso_with_offset_is_accepted(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(
            [
                "ingest",
                "--zone",
                "DE_LU",
                "--start",
                "2026-05-04T00:00:00+00:00",
                "--end",
                "2026-05-04T03:00:00+00:00",
            ]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Observations fetched:  3" in captured.out
