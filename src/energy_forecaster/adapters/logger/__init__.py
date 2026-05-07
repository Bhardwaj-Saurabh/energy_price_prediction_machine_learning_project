"""Concrete implementations of the Logger port."""

from energy_forecaster.adapters.logger.structlog_logger import (
    StructlogLogger,
    configure_structlog,
)

__all__ = ["StructlogLogger", "configure_structlog"]
