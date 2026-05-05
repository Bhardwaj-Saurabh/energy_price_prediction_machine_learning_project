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
            lambda settings: _FailingUseCase(),
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
            lambda settings: _FailingUseCase(),
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
