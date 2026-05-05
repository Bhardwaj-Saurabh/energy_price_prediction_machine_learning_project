"""EntsoePyClient — production adapter backed by the ``entsoe-py`` library.

The boundary between this file and the rest of the codebase is the
:class:`EntsoeClient` Protocol. By the time data crosses that boundary it
is already in the form of :class:`LoadObservation` aggregates: validated
domain entities with UTC timestamps. The application layer never sees
``pandas`` types, ``entsoe`` exceptions, or EIC area codes.

What this adapter is responsible for:
  * Mapping our :class:`BiddingZone` enum to entsoe-py's country code keys.
    The mapping is explicit and one-way; if entsoe-py renames its codes,
    we update this dict and nothing else moves.
  * Translating library and HTTP failures into
    :class:`DataSourceUnavailableError` so the use case sees a
    layer-neutral error type.
  * Filtering out NaN observations. ENTSO-E occasionally publishes
    incomplete rows; they would otherwise fail :class:`EnergyMW`
    validation downstream, which is correct but noisy.
  * Treating an empty response (``NoMatchingDataError``) as "zero
    observations", not as a failure — the use case's contract allows
    empty windows and inserts nothing.
"""

from collections.abc import Iterable
from datetime import datetime

import pandas as pd

# Import from the submodule directly: ``entsoe.__init__`` does
# ``from .entsoe import EntsoePandasClient`` without an ``__all__``, which
# mypy strict's ``no_implicit_reexport`` rejects. The submodule path is
# stable across recent versions of entsoe-py.
from entsoe.entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

from energy_forecaster.application.errors import DataSourceUnavailableError
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import EnergyMW

# Domain → entsoe-py country code. We keep the mapping explicit and
# one-directional even where the strings happen to match (e.g. "DE_LU"),
# so a rename in either layer is a single-file change here. Adding a new
# zone without a mapping fails loudly with KeyError at call time.
_BIDDING_ZONE_TO_ENTSOE_CODE: dict[BiddingZone, str] = {
    BiddingZone.DE_LU: "DE_LU",
    BiddingZone.FR: "FR",
    BiddingZone.GB: "GB",
}

# entsoe-py's parse_loads emits a DataFrame with a single column for
# realised demand. We pin the name here so a library change is a
# single-line fix instead of a debugging hunt.
_LOAD_COLUMN: str = "Actual Load"


class EntsoePyClient:
    """HTTP-backed :class:`EntsoeClient` implementation.

    The constructor stores credentials only; no network call happens
    until :meth:`fetch_load` is invoked. That makes it safe to instantiate
    in the composition root at process start without paying a startup
    latency tax.
    """

    def __init__(self, api_key: str) -> None:
        self._client = EntsoePandasClient(api_key=api_key)

    def fetch_load(
        self,
        *,
        zone: BiddingZone,
        start: datetime,
        end: datetime,
    ) -> Iterable[LoadObservation]:
        country_code = _BIDDING_ZONE_TO_ENTSOE_CODE[zone]

        try:
            df = self._client.query_load(
                country_code=country_code,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
            )
        except NoMatchingDataError:
            # Empty window is a valid outcome, not a failure.
            return []
        except Exception as exc:
            # Any other failure (HTTP, network, parse) is an environment
            # error from the application's perspective. Wrapping makes the
            # use case independent of which library backs this adapter.
            raise DataSourceUnavailableError(
                f"ENTSO-E load query failed for {zone.value}: {exc}"
            ) from exc

        return list(_to_observations(df, zone))


def _to_observations(df: pd.DataFrame, zone: BiddingZone) -> Iterable[LoadObservation]:
    series = df[_LOAD_COLUMN].tz_convert("UTC")
    for ts, value in series.items():
        if pd.isna(value):
            continue
        yield LoadObservation(
            zone=zone,
            timestamp_utc=ts.to_pydatetime(),
            load=EnergyMW(float(value)),
        )
