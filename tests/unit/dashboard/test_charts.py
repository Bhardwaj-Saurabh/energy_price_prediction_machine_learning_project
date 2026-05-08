"""Unit tests for the dashboard's pure chart helpers."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go

from energy_forecaster.dashboard.charts import actual_vs_predicted, psi_by_feature


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _avp_df(rows: int = 3) -> pd.DataFrame:
    """Tiny actual-vs-predicted DataFrame for chart-shape assertions."""
    return pd.DataFrame(
        [
            {
                "delivery_time": _utc(2026, 5, 7) + timedelta(hours=h),
                "predicted_load_mw": 50_000.0 + 100.0 * h,
                "actual_load_mw": 50_500.0 + 100.0 * h,
            }
            for h in range(rows)
        ]
    )


class TestActualVsPredicted:
    def test_returns_a_plotly_figure(self) -> None:
        fig = actual_vs_predicted(_avp_df(), zone="DE_LU")
        assert isinstance(fig, go.Figure)

    def test_has_two_traces_one_predicted_one_actual(self) -> None:
        # The names are part of the contract — they appear in the
        # legend and are how a reader of the dashboard distinguishes
        # the two lines. Locking them in here.
        fig = actual_vs_predicted(_avp_df(), zone="DE_LU")
        names = sorted(trace.name for trace in fig.data)
        assert names == ["Actual", "Predicted"]

    def test_zone_appears_in_title(self) -> None:
        fig = actual_vs_predicted(_avp_df(), zone="FR")
        assert "FR" in fig.layout.title.text

    def test_x_axis_uses_delivery_time(self) -> None:
        # Plotly converts tz-aware Timestamps to numpy datetime64
        # (without tz) when assigning to a trace's x. Compare via
        # pandas to_datetime to bridge the representation gap rather
        # than asserting on raw types.
        df = _avp_df(rows=3)
        fig = actual_vs_predicted(df, zone="DE_LU")
        predicted_trace = next(t for t in fig.data if t.name == "Predicted")
        x_values = pd.to_datetime(list(predicted_trace.x), utc=True)
        assert list(x_values) == list(df["delivery_time"])

    def test_empty_dataframe_returns_annotated_figure(self) -> None:
        # Empty input gets a friendly placeholder — the dashboard must
        # never render a confusingly silent blank chart.
        fig = actual_vs_predicted(pd.DataFrame(), zone="GB")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert fig.layout.annotations  # the explanatory text is there
        assert "GB" in fig.layout.annotations[0].text


class TestPsiByFeature:
    def test_returns_a_plotly_figure(self) -> None:
        fig = psi_by_feature({"temp_c": 0.05, "wind_10m_ms": 0.12})
        assert isinstance(fig, go.Figure)

    def test_orders_by_descending_psi(self) -> None:
        # Highest PSI must render at the top of the horizontal bar
        # chart — that's the value of a sorted view. Plotly renders
        # the y-axis with the *last* bar at the top, so the input
        # order is reversed-sorted before constructing the trace.
        fig = psi_by_feature({"a": 0.05, "b": 0.30, "c": 0.10})
        bar_trace = fig.data[0]
        # y is feature names, last element is the top of the chart.
        assert bar_trace.y[-1] == "b"  # highest PSI on top
        assert bar_trace.y[0] == "a"  # lowest at bottom

    def test_top_n_caps_the_number_of_bars(self) -> None:
        fig = psi_by_feature({f"feat_{i}": float(i) / 10 for i in range(20)}, top_n=3)
        assert len(fig.data[0].y) == 3

    def test_empty_dict_returns_annotated_figure(self) -> None:
        fig = psi_by_feature({})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert fig.layout.annotations
        assert "PSI" in fig.layout.annotations[0].text
