"""European bidding zones supported by this forecaster."""

from enum import StrEnum


class BiddingZone(StrEnum):
    """A geographic area with a single wholesale electricity price.

    The string values are stable internal identifiers used throughout the
    domain and serialisation layers. They are NOT the ENTSO-E EIC codes —
    mapping to and from EIC codes is the responsibility of the ENTSO-E
    adapter, so the domain stays decoupled from any external coding scheme.
    """

    DE_LU = "DE_LU"
    FR = "FR"
    GB = "GB"
