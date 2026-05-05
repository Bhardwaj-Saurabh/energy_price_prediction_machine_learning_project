"""Smoke test: the package imports and exposes a version string.

If this fails, the project is not installed correctly — fix the install
before debugging anything else.
"""

import energy_forecaster


def test_package_version_is_a_string() -> None:
    assert isinstance(energy_forecaster.__version__, str)
    assert energy_forecaster.__version__ != ""
