"""Permutation feature importance with interpretation and causation caveat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from . import plots as _plots
except ImportError:  # run as a top-level module
    import plots as _plots

INK = "#1f2937"
LINE = "#4b5563"
POINT = _plots.POINT
FILL = "#e5e7eb"


@dataclass
class ImportanceResult:
    names: list[str]
    values: np.ndarray
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.names) > 0

    def sorted_frame(self) -> pd.DataFrame:
        order = np.argsort(self.values)[::-1]
        return pd.DataFrame({
            "Feature": [self.names[i] for i in order],
            "Importance": [float(self.values[i]) for i in order],
        })


def bar_chart(result: ImportanceResult, top_n: int = 15) -> go.Figure:
    if not result.ok:
        fig = go.Figure()
        fig.add_annotation(
            text=result.error or "Importance not available.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color=LINE),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=40, r=40, t=40, b=40))
        return fig

    df = result.sorted_frame().head(top_n).iloc[::-1]  # highest on top
    fig = go.Figure(
        data=[go.Bar(
            x=df["Importance"], y=df["Feature"], orientation="h",
            marker=dict(color=POINT, line=dict(color="white", width=0.5)),
        )]
    )
    fig.update_layout(
        xaxis_title="Decrease in test score when feature is shuffled",
        yaxis_title="",
        height=min(620, 40 + 28 * len(df)),
    )
    return _plots._apply(fig, "Permutation feature importance (test set)")


CAUSATION_NOTE = (
    "Importance does not establish causation. A feature can appear important "
    "because it is correlated with the true driver, because of preprocessing "
    "choices, or because of target leakage. Use it to generate hypotheses, not "
    "to infer causal effect."
)
