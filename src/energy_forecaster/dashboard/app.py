"""Dash application factory for the dashboard.

Same architectural slot as :mod:`energy_forecaster.serving.app` — a
*framework* that takes its dependencies via the composition root and
wires layout + callbacks. No business logic in callbacks; each
callback resolves its inputs via :mod:`data` and renders via
:mod:`charts`.

Caching is intentionally absent. Every callback invocation re-reads
the JSONL repos and (for the drift card) re-runs the monitoring
pipeline. For a local-first portfolio piece this is fine — the
feature matrix is small and reads are millisecond-fast. Adding
``flask_caching`` is the obvious next step when the data grows.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from energy_forecaster.application.ports.clock import Clock
from energy_forecaster.application.ports.load_forecast_repository import (
    LoadForecastRepository,
)
from energy_forecaster.application.ports.load_observation_repository import (
    LoadObservationRepository,
)
from energy_forecaster.application.ports.logger import Logger
from energy_forecaster.config.settings import Settings
from energy_forecaster.dashboard.charts import actual_vs_predicted, psi_by_feature
from energy_forecaster.dashboard.data import load_actual_vs_predicted
from energy_forecaster.domain.value_objects.bidding_zone import BiddingZone
from energy_forecaster.pipelines.monitoring.runner import MonitoringResult

# Days-back options shown in the dropdown. Wider windows tax the
# JSONL reader, but we'll cross that bridge when the local data root
# pushes past a few weeks.
_DAYS_BACK_OPTIONS: tuple[int, ...] = (1, 3, 7, 14, 30)
_DEFAULT_DAYS_BACK: int = 7
_DEFAULT_ZONE: str = BiddingZone.DE_LU.value


def make_actual_vs_predicted_callback(
    forecast_repo: LoadForecastRepository,
    observation_repo: LoadObservationRepository,
    clock: Clock,
) -> Callable[[str, int], go.Figure]:
    """Return the callback that refreshes the actual-vs-predicted chart.

    Defined outside :func:`create_app` so unit tests can call it as a
    regular function — Dash's callback registration is a thin wrapper
    over the function, not a transformation, so the inner logic is
    fully testable without spinning up a server.
    """

    def _callback(zone_value: str, days_back: int) -> go.Figure:
        zone = BiddingZone(zone_value)
        until = clock.now()
        since = until - timedelta(days=days_back)
        df = load_actual_vs_predicted(
            forecast_repo=forecast_repo,
            observation_repo=observation_repo,
            zone=zone,
            since=since,
            until=until,
        )
        return actual_vs_predicted(df, zone=zone_value)

    return _callback


def make_drift_callback(
    monitoring_runner: Callable[[], MonitoringResult],
) -> Callable[[int], tuple[str, go.Figure]]:
    """Return the callback that runs monitoring and renders the verdict + PSI chart.

    Takes a no-arg ``monitoring_runner`` closure (the composition root
    captures features path, repos, clock). The callback's input is a
    button click count — Dash needs *some* trigger or the callback
    never fires. We use the click count as a trigger but ignore its
    value otherwise.
    """

    def _callback(_n_clicks: int) -> tuple[str, go.Figure]:
        result = monitoring_runner()
        verdict = "RETRAIN" if result.retrain_recommended else "no action"
        summary = (
            f"Verdict: {verdict}  |  "
            f"Max rolling MAPE: {result.max_rolling_mape.value:.4f}  |  "
            f"Max PSI: {result.max_psi:.4f}  |  "
            f"Window: {result.window_start.date().isoformat()} → "
            f"{result.window_end.date().isoformat()}"
        )
        return summary, psi_by_feature(result.psi_by_feature)

    return _callback


def _build_layout() -> html.Div:
    """Construct the static layout. Inputs above, charts below."""
    return html.Div(
        style={"fontFamily": "sans-serif", "maxWidth": "1100px", "margin": "0 auto"},
        children=[
            html.H1("Energy Forecaster — Dashboard"),
            html.Div(
                style={"display": "flex", "gap": "24px", "alignItems": "flex-end"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Zone"),
                            dcc.Dropdown(
                                id="zone-dropdown",
                                options=[{"label": z.value, "value": z.value} for z in BiddingZone],
                                value=_DEFAULT_ZONE,
                                clearable=False,
                                style={"width": "200px"},
                            ),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Label("Days back"),
                            dcc.Dropdown(
                                id="days-back",
                                options=[{"label": str(d), "value": d} for d in _DAYS_BACK_OPTIONS],
                                value=_DEFAULT_DAYS_BACK,
                                clearable=False,
                                style={"width": "120px"},
                            ),
                        ],
                    ),
                    html.Button(
                        "Refresh drift",
                        id="refresh-drift",
                        n_clicks=0,
                        style={"height": "36px"},
                    ),
                ],
            ),
            dcc.Graph(id="avp-chart"),
            html.Div(
                id="drift-summary",
                style={
                    "padding": "12px",
                    "border": "1px solid #ddd",
                    "borderRadius": "4px",
                    "marginTop": "12px",
                    "fontFamily": "monospace",
                },
            ),
            dcc.Graph(id="psi-chart"),
        ],
    )


def create_app(
    settings: Settings,
    *,
    logger: Logger,
    forecast_repo: LoadForecastRepository,
    observation_repo: LoadObservationRepository,
    monitoring_runner: Callable[[], MonitoringResult],
    clock: Clock,
) -> Dash:
    """Build a Dash app around the injected dependencies.

    Every callback is registered with no implicit globals — the data
    plumbing lives in closures returned by ``make_*_callback``. The
    ``logger`` parameter is accepted for symmetry with the other
    framework factories; future per-request bound logging will hang
    off it.
    """
    del logger  # reserved for callback-level logging once it lands
    del settings  # only the environment label is read once on bootstrap

    app = Dash(__name__, title="Energy Forecaster Dashboard")
    app.layout = _build_layout()

    avp_cb = make_actual_vs_predicted_callback(forecast_repo, observation_repo, clock)
    drift_cb = make_drift_callback(monitoring_runner)

    app.callback(
        Output("avp-chart", "figure"),
        Input("zone-dropdown", "value"),
        Input("days-back", "value"),
    )(avp_cb)

    app.callback(
        Output("drift-summary", "children"),
        Output("psi-chart", "figure"),
        Input("refresh-drift", "n_clicks"),
    )(drift_cb)

    return app
