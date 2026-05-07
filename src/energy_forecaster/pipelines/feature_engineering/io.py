"""Read JSONL directories into validated DataFrames.

The LocalFs repos write JSONL one file per zone; these readers walk
those files and produce :class:`LoadObservationSchema` /
:class:`WeatherReadingSchema` DataFrames via the entity converters.

Why these are functions, not Kedro DataSets: a custom DataSet would be
more idiomatic Kedro but adds boilerplate this project does not yet
need. We can promote them to a ``JSONLDirectoryDataset`` later when a
second pipeline reads the same shape.
"""

from pathlib import Path

from pandera.typing import DataFrame

from energy_forecaster.adapters.load_observation_repo.local_fs import (
    deserialise as _deserialise_load,
)
from energy_forecaster.adapters.weather_reading_repo.local_fs import (
    deserialise as _deserialise_weather,
)
from energy_forecaster.contracts.load_observation_schema import (
    LoadObservationSchema,
    to_load_dataframe,
)
from energy_forecaster.contracts.weather_reading_schema import (
    WeatherReadingSchema,
    to_weather_dataframe,
)
from energy_forecaster.domain.entities.load_observation import LoadObservation
from energy_forecaster.domain.entities.weather_reading import WeatherReading


def read_load_observations(
    directory: Path,
) -> DataFrame[LoadObservationSchema]:
    """Read every ``*.jsonl`` file in ``directory`` and return a validated frame.

    Files are read in sorted name order so the result is deterministic
    regardless of filesystem ordering. The contracts converter validates
    the output schema before it is returned.
    """
    observations: list[LoadObservation] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            observations.extend(_deserialise_load(line) for line in f)
    return to_load_dataframe(observations)


def read_weather_readings(
    directory: Path,
) -> DataFrame[WeatherReadingSchema]:
    """Read every ``*.jsonl`` file in ``directory`` and return a validated frame."""
    readings: list[WeatherReading] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            readings.extend(_deserialise_weather(line) for line in f)
    return to_weather_dataframe(readings)
