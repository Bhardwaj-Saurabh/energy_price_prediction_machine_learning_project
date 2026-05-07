"""StructlogLogger — production adapter implementing the Logger port via structlog.

The application layer talks to the :class:`Logger` Protocol; the only
import of ``structlog`` in the codebase is here. The factory function
:func:`configure_structlog` wires the global processor chain — call it
once at process entry (the CLI does this in ``main()``).

Output format is environment-driven:
  * ``LOCAL`` — coloured, human-readable console output (``ConsoleRenderer``).
  * ``PROD``  — newline-delimited JSON (``JSONRenderer``), the format
    Application Insights and Grafana ingest natively.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

from energy_forecaster.config.settings import Environment, LogLevel


class StructlogLogger:
    """Implements :class:`Logger` by delegating to a structlog ``BoundLogger``."""

    def __init__(self, logger: Any | None = None) -> None:
        self._logger = logger if logger is not None else structlog.get_logger()

    def bind(self, **context: Any) -> StructlogLogger:
        return StructlogLogger(logger=self._logger.bind(**context))

    def debug(self, event: str, **context: Any) -> None:
        self._logger.debug(event, **context)

    def info(self, event: str, **context: Any) -> None:
        self._logger.info(event, **context)

    def warning(self, event: str, **context: Any) -> None:
        self._logger.warning(event, **context)

    def error(self, event: str, **context: Any) -> None:
        self._logger.error(event, **context)


def configure_structlog(*, log_level: LogLevel, environment: Environment) -> None:
    """Configure structlog's global processor chain.

    Idempotent — calling twice replaces the previous configuration. The
    CLI invokes this once at startup; tests can call it freely.

    The processor chain is intentionally short:
      * ``add_log_level`` adds ``level`` field.
      * ``TimeStamper(iso)`` adds ``timestamp`` field.
      * Renderer turns the event dict into output (console or JSON).

    Bound context (e.g. ``correlation_id``) flows through the chain
    automatically because structlog stores it on the BoundLogger; no
    extra processor is needed.
    """
    renderer: Any
    if environment is Environment.LOCAL:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[log_level]
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
