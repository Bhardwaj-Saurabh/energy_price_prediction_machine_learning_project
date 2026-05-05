"""Allow ``python -m energy_forecaster ...`` to invoke the CLI.

The same entrypoint is also exposed as the ``energy-forecaster`` console
script via ``[project.scripts]`` in ``pyproject.toml``; both routes hit
:func:`energy_forecaster.cli.main`.
"""

from energy_forecaster.cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())  # pragma: no cover
