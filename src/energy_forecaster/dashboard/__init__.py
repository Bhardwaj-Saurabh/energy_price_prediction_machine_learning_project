"""Dash dashboard — analytical web app for inspecting forecasts and drift.

Same architectural slot as :mod:`energy_forecaster.serving` — a
*framework* that depends on the application's ports (forecast +
observation repos, monitoring runner) via the composition root. The
dashboard module is split into three pieces with a strict dependency
direction:

  * :mod:`charts` — pure functions that turn data structures into
    plotly Figures. No I/O, no port access, no Dash.
  * :mod:`data`   — pure-ish functions that read ports and return
    data structures the chart helpers consume. Dash-free.
  * :mod:`app`    — the Dash app factory: layout, callbacks, glue.

Tests of charts.py do not need any I/O. Tests of data.py use the
existing fakes. The Dash app itself is smoke-tested through
``app.server.test_client()`` (Flask under the hood) for fast,
network-free assertions.
"""
