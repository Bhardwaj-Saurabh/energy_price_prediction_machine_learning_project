"""Logger port — structured logging interface used by application code."""

from typing import Any, Protocol


class Logger(Protocol):
    """Structured logger.

    The two key calls beyond the level methods are:

      * ``bind(**context)`` — return a *new* logger with the given key/value
        pairs permanently attached to every subsequent log call. The
        original logger is untouched. This is how request-scoped fields
        like ``correlation_id`` propagate without being threaded through
        every function signature.
      * level methods (``info``, ``warning``, …) — accept a short event
        name as the positional argument and arbitrary keyword context.
        Renderers turn this into either a human-readable line (local) or
        a JSON object (production).

    Adapters satisfy this Protocol structurally — see
    :class:`StructlogLogger` for the production implementation and
    :class:`FakeLogger` (in ``tests/``) for the test double.
    """

    def bind(self, **context: Any) -> "Logger":
        """Return a child logger with ``context`` attached to every call."""
        ...

    def debug(self, event: str, **context: Any) -> None:
        """Log at DEBUG level."""
        ...

    def info(self, event: str, **context: Any) -> None:
        """Log at INFO level."""
        ...

    def warning(self, event: str, **context: Any) -> None:
        """Log at WARNING level."""
        ...

    def error(self, event: str, **context: Any) -> None:
        """Log at ERROR level."""
        ...
