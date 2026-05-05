"""Application-layer error types.

Adapters translate library-specific failures (HTTPError, psycopg.Error,
botocore exceptions, …) into these layer-neutral types at their boundary.
Use cases catch and decide how to respond; frameworks map them onto
appropriate exit codes or HTTP statuses.

Domain errors (e.g. ``ValueError`` from a value-object constructor) are a
distinct category and are NOT wrapped — they signal a programmer or
upstream-data bug, not an environment failure.
"""


class ApplicationError(Exception):
    """Base class for all errors raised by the application layer."""


class DataSourceUnavailableError(ApplicationError):
    """An external data source could not be reached or returned a fatal error.

    Adapters raise this in place of HTTP errors, network timeouts, or any
    upstream-API failure that the use case cannot recover from in the
    current invocation. The retry / backoff policy lives in the adapter,
    not in the use case.
    """
