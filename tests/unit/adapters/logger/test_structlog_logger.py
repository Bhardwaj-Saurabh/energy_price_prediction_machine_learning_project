"""Unit tests for StructlogLogger and configure_structlog.

We use structlog's ``capture_logs`` testing helper rather than parsing
stdout — it captures the structured event dicts before any renderer
runs, which is what we actually care about.
"""

import structlog

from energy_forecaster.adapters.logger.structlog_logger import (
    StructlogLogger,
    configure_structlog,
)
from energy_forecaster.application.ports.logger import Logger
from energy_forecaster.config.settings import Environment


class TestProtocolConformance:
    def test_satisfies_logger_protocol_structurally(self) -> None:
        logger: Logger = StructlogLogger()
        assert hasattr(logger, "bind")


class TestEventEmission:
    def test_info_emits_event_with_context(self) -> None:
        with structlog.testing.capture_logs() as captured:
            StructlogLogger().info("ingest.start", zone="DE_LU")

        event = captured[-1]
        assert event["event"] == "ingest.start"
        assert event["log_level"] == "info"
        assert event["zone"] == "DE_LU"

    def test_each_level_method_routes_correctly(self) -> None:
        with structlog.testing.capture_logs() as captured:
            logger = StructlogLogger()
            logger.debug("d")
            logger.info("i")
            logger.warning("w")
            logger.error("e")

        levels = [c["log_level"] for c in captured]
        events = [c["event"] for c in captured]
        assert levels == ["debug", "info", "warning", "error"]
        assert events == ["d", "i", "w", "e"]


class TestBind:
    def test_bind_returns_a_new_logger_with_context_attached(self) -> None:
        with structlog.testing.capture_logs() as captured:
            child = StructlogLogger().bind(correlation_id="abc-123")
            child.info("ingest.start")

        assert captured[-1]["correlation_id"] == "abc-123"

    def test_bind_is_chainable(self) -> None:
        # Each bind() should accumulate context, not replace it. This is
        # what lets the use case bind ``zone`` on top of a CLI-bound
        # ``correlation_id`` and have both fields appear together.
        with structlog.testing.capture_logs() as captured:
            base = StructlogLogger().bind(correlation_id="abc-123")
            zoned = base.bind(zone="DE_LU")
            zoned.info("ingest.zone.done")

        event = captured[-1]
        assert event["correlation_id"] == "abc-123"
        assert event["zone"] == "DE_LU"

    def test_bind_does_not_mutate_the_parent_logger(self) -> None:
        # Each bind returns a new BoundLogger; the parent must keep its
        # original context, so multiple zones logged from the same parent
        # never leak each other's bound fields.
        with structlog.testing.capture_logs() as captured:
            base = StructlogLogger().bind(correlation_id="root")
            base.bind(zone="DE_LU").info("first")
            base.bind(zone="FR").info("second")

        first, second = captured[-2], captured[-1]
        assert first["zone"] == "DE_LU"
        assert second["zone"] == "FR"
        assert first["correlation_id"] == second["correlation_id"] == "root"


class TestConfigureStructlog:
    def test_configures_without_raising_for_local_environment(self) -> None:
        # Smoke test: configuration is global state, but the call itself
        # must complete without error in either environment. Output
        # rendering is structlog's responsibility; we verify only that
        # we wired its inputs correctly.
        configure_structlog(log_level="INFO", environment=Environment.LOCAL)

    def test_configures_without_raising_for_prod_environment(self) -> None:
        configure_structlog(log_level="WARNING", environment=Environment.PROD)

    def test_calling_after_configure_does_not_raise(self) -> None:
        # Smoke test: after configuration the level methods must still be
        # callable. Output is to PrintLoggerFactory's stderr; we don't
        # assert on the rendered text — that's structlog's domain.
        configure_structlog(log_level="INFO", environment=Environment.LOCAL)
        StructlogLogger().info("after.configure")
