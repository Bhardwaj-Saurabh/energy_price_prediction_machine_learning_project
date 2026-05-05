"""Cross-cutting validators used by multiple domain entities.

This module is private to the domain layer (leading underscore). External
layers should not import from here — they get validation transitively
through the entities themselves. The helpers here exist purely to remove
duplication between entities that share invariants.
"""

from datetime import datetime, timedelta


def require_utc(field_name: str, ts: datetime) -> None:
    """Raise ValueError unless ``ts`` is a timezone-aware datetime with offset 0.

    Naive datetimes (``tzinfo is None``) are the single largest source of
    bugs in time-series ML — they get silently coerced into local time on
    serialisation boundaries. We reject them here so no downstream code
    has to defend against them.

    The offset check accepts any tzinfo that represents UTC, including
    ``datetime.UTC``, ``timezone.utc``, and ``zoneinfo.ZoneInfo("UTC")``.
    Equality between tzinfo *instances* is not required.
    """
    if ts.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware, got a naive datetime")
    if ts.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be UTC (offset 0), got offset {ts.utcoffset()}")
