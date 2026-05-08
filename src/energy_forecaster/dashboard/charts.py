"""Pure plotly figure builders for the dashboard.

Each function takes plain data structures (DataFrame, dict) and returns
a :class:`plotly.graph_objects.Figure`. No I/O, no port access, no Dash.
That separation keeps the figure logic unit-testable without a server,
without fakes, and without selecting features the app layer happens to
care about — the chart contract is "given this shape, draw this."
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def actual_vs_predicted(df: pd.DataFrame, *, zone: str) -> go.Figure:
    """Line chart of predicted vs actual load over time.

    The DataFrame must carry ``delivery_time`` (datetime),
    ``predicted_load_mw`` (float), and ``actual_load_mw`` (float). Rows
    with NaN actuals are kept — they show as gaps in the actual line —
    while predicted always renders. Empty input returns a placeholder
    figure with an explanatory annotation rather than a blank one, so
    the dashboard is never confusingly silent.
    """
    figure = go.Figure()
    if df.empty:
        figure.add_annotation(
            text=f"No forecasts found for {zone} in the selected window.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        figure.update_layout(
            title=f"Predicted vs Actual load — {zone}",
            xaxis_title="Delivery time (UTC)",
            yaxis_title="Load (MW)",
            template="plotly_white",
        )
        return figure

    figure.add_trace(
        go.Scatter(
            x=df["delivery_time"],
            y=df["predicted_load_mw"],
            mode="lines",
            name="Predicted",
            line={"color": "#1f77b4"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=df["delivery_time"],
            y=df["actual_load_mw"],
            mode="lines+markers",
            name="Actual",
            line={"color": "#d62728"},
            connectgaps=False,
        )
    )
    figure.update_layout(
        title=f"Predicted vs Actual load — {zone}",
        xaxis_title="Delivery time (UTC)",
        yaxis_title="Load (MW)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return figure


def psi_by_feature(psi: dict[str, float], *, top_n: int = 5) -> go.Figure:
    """Horizontal bar chart of the top-N features by PSI.

    Sorted descending so the worst-drift feature is at the top. The
    band thresholds (stable / moderate / significant) are drawn as
    background reference rectangles to make a 0.86 PSI visually
    obviously past the 0.20 gate. Empty input gets a friendly message
    rather than a blank figure.
    """
    figure = go.Figure()
    if not psi:
        figure.add_annotation(
            text="No PSI data — feature matrix is too short to split.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        figure.update_layout(
            title="PSI by feature",
            template="plotly_white",
        )
        return figure

    ranked = sorted(psi.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    features = [name for name, _ in reversed(ranked)]  # reversed so highest is at top
    values = [value for _, value in reversed(ranked)]

    figure.add_trace(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker={"color": "#1f77b4"},
            name="PSI",
        )
    )
    # Reference lines at the moderate (0.10) and significant (0.20) gates.
    figure.add_vline(x=0.10, line_dash="dot", line_color="#999")
    figure.add_vline(x=0.20, line_dash="dot", line_color="#d62728")
    figure.update_layout(
        title=f"Top {len(ranked)} features by PSI",
        xaxis_title="PSI",
        yaxis_title="",
        template="plotly_white",
        showlegend=False,
        margin={"l": 120, "r": 20, "t": 60, "b": 40},
    )
    return figure
