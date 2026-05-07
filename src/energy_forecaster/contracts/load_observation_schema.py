"""Pandera schema and converter for load-observation DataFrames.

Where this matters: when feature engineering pipelines (Kedro nodes,
Prefect flows) start consuming load observations as wide-format
DataFrames, this module is the *only* legitimate way to construct one.
``@pa.check_types`` validates the output frame against
:class:`LoadObservationSchema`, so every downstream node receives a
DataFrame whose columns, dtypes, and value ranges have been confirmed.

Constants are imported from the domain so the schema's bounds and the
:class:`EnergyMW` constructor's bounds cannot drift apart.
"""

from collections.abc import Iterable
from typing import Annotated, ClassVar

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.domain.value_objects.energy import MAX_PLAUSIBLE_LOAD_MW

_VALID_ZONES: tuple[str, ...] = tuple(z.value for z in BiddingZone)


class LoadObservationSchema(pa.DataFrameModel):
    """Wide-format hourly load observations (one row per zone x hour).

    The composite ``(zone, timestamp_utc)`` key is the same identity
    the domain entity enforces. Any DataFrame crossing into feature
    engineering must pass this schema, which is why every adapter that
    materialises a DataFrame from observations must do so through
    :func:`to_load_dataframe`.
    """

    timestamp_utc: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    zone: Series[str] = pa.Field(isin=_VALID_ZONES)
    load_mw: Series[float] = pa.Field(ge=0.0, le=MAX_PLAUSIBLE_LOAD_MW)

    class Config:
        # Reject extra columns — extension is a deliberate schema change,
        # not a quiet drift.
        strict = True
        # ``coerce=False`` so that a naive timestamp column (dtype
        # ``datetime64[ns]`` with no tz) is *rejected* rather than
        # silently localised to UTC. The producer must construct a
        # tz-aware column explicitly; the converter below does so by
        # casting through ``datetime64[ns, UTC]``.
        coerce = False
        # Composite uniqueness — the same (zone, ts) pair may not appear
        # twice. Postgres' ``ON CONFLICT DO NOTHING`` enforces the same
        # constraint at the storage tier. ``ClassVar`` is required by ruff
        # (RUF012) to distinguish Pandera's class-level configuration from
        # accidentally-shared mutable instance state.
        unique: ClassVar[list[str]] = ["zone", "timestamp_utc"]


@pa.check_types
def to_load_dataframe(
    observations: Iterable[LoadObservation],
) -> DataFrame[LoadObservationSchema]:
    """Convert validated domain entities into a validated DataFrame.

    The decorator runs :class:`LoadObservationSchema` against the return
    value, so any silent dtype drift or duplicate row would fail loudly
    at this boundary instead of leaking into downstream nodes.
    """
    rows = [
        {
            "timestamp_utc": obs.timestamp_utc,
            "zone": obs.zone.value,
            "load_mw": obs.load.value,
        }
        for obs in observations
    ]
    if not rows:
        # Pandera struggles to validate column dtypes on truly empty
        # frames unless they were created with the right dtype upfront.
        return pd.DataFrame(  # type: ignore[no-any-return]
            {
                "timestamp_utc": pd.Series([], dtype="datetime64[ns, UTC]"),
                "zone": pd.Series([], dtype=str),
                "load_mw": pd.Series([], dtype=float),
            }
        )
    df = pd.DataFrame(rows)
    # Pandas 3.x defaults to ``datetime64[us, UTC]``; the schema demands
    # ``[ns, UTC]``. Cast explicitly so the schema receives the dtype it
    # asks for without relying on Pandera's coercion (which we disabled).
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    return df  # type: ignore[no-any-return]
