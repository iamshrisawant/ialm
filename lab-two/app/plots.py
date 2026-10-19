"""Neutral Plotly visualizations for regression diagnostics.

All figures pass through :func:`_apply` so fonts, margins, gridlines, and the
accent colour stay consistent across the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Design tokens (kept in sync with ui.py / config.toml theme).
ACCENT = "#2563eb"
INK = "#1f2937"
MUTED = "#6b7280"
BORDER = "#e5e7eb"
POINT = "#4f7fe0"
REF = "#9ca3af"
FILL = "#eef4ff"

_FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
_GRID = "#eef0f3"


def _apply(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply the shared look to a figure."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=_FONT, color=INK, size=13),
        title=dict(text=title, font=dict(size=15, color=INK)) if title else None,
        margin=dict(l=60, r=20, t=52 if title else 18, b=56),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=[ACCENT, MUTED, REF],
        hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=BORDER, borderwidth=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False, linecolor=BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False, linecolor=BORDER)
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, xanchor="center", yanchor="middle",
        font=dict(color=MUTED),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def histogram(values: np.ndarray, label: str = "Value") -> go.Figure:
    """Histogram for exploring a single numeric column."""
    values = np.asarray(values)
    if values.size == 0:
        return _empty_figure("No values to plot.")
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values, marker=dict(color=POINT, line=dict(color="white", width=0.6)),
            opacity=0.9,
        )
    )
    fig.update_layout(xaxis_title=label, yaxis_title="Count", showlegend=False)
    return _apply(fig, f"Distribution of {label}")


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Diverging heatmap of a correlation matrix (legible at a glance)."""
    if corr.empty:
        return _empty_figure("No correlations to plot.")
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=[str(c) for c in corr.columns],
            y=[str(c) for c in corr.index],
            zmin=-1, zmax=1, zmid=0,
            colorscale=[
                (0.0, "#1e3a8a"), (0.25, "#93b4f5"),
                (0.5, "#f8fafc"),
                (0.75, "#f0a6a6"), (1.0, "#b91c1c"),
            ],
            colorbar=dict(title="r", thickness=12),
        )
    )
    fig.update_layout(margin=dict(l=120, r=20, t=52, b=80))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return _apply(fig, "Numeric correlations")


def actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, label: str = "Data") -> go.Figure:
    """Scatter of actual vs predicted with a 45 degrees reference line."""
    if len(y_true) == 0:
        return _empty_figure("No predictions to plot.")
    fig = px.scatter(x=y_true, y=y_pred, labels={"x": "Actual", "y": "Predicted"})
    fig.update_traces(
        marker=dict(color=POINT, size=6, opacity=0.65, line=dict(width=0)),
        selector=dict(mode="markers"),
    )
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color=REF, dash="dash"), name="Perfect prediction",
            showlegend=True,
        )
    )
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=[lo, hi])
    return _apply(fig, f"Actual vs. Predicted - {label}")


def residual_distribution(residuals: np.ndarray, label: str = "Data") -> go.Figure:
    """Histogram of residuals (actual - predicted)."""
    if len(residuals) == 0:
        return _empty_figure("No residuals to plot.")
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=residuals, name="Residuals",
            marker=dict(color=POINT, line=dict(color="white", width=0.6)),
            opacity=0.9,
        )
    )
    fig.add_vline(x=0, line=dict(color=REF, dash="dash"))
    fig.update_layout(
        xaxis_title="Residual (actual - predicted)", yaxis_title="Count",
        showlegend=False,
    )
    return _apply(fig, f"Residual Distribution - {label}")


def residual_vs_predicted(y_pred: np.ndarray, residuals: np.ndarray, label: str = "Data") -> go.Figure:
    """Residuals vs predicted values to spot heteroscedasticity / bias."""
    if len(y_pred) == 0:
        return _empty_figure("No predictions to plot.")
    fig = px.scatter(x=y_pred, y=residuals, labels={"x": "Predicted", "y": "Residual"})
    fig.update_traces(
        marker=dict(color=POINT, size=6, opacity=0.65, line=dict(width=0)),
        selector=dict(mode="markers"),
    )
    fig.add_hline(y=0, line=dict(color=REF, dash="dash"))
    return _apply(fig, f"Residuals vs. Predicted - {label}")
