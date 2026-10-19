"""Shared UI design system for the app.

One place for the visual language so every step looks the same: a single accent
colour, consistent headings, one vocabulary for advisories, carded sections,
consistent tables/plots, and a single way to explain elements to learners.
Streamlit + stdlib only; no internal ``data-testid`` selectors are relied upon.
"""

from __future__ import annotations

import html
from contextlib import contextmanager
from typing import Optional, Sequence

import pandas as pd
import streamlit as st

# -- design tokens ---------------------------------------------------------- #
ACCENT = "#2563eb"
ACCENT_SOFT = "#eff4ff"
INK = "#1f2937"
MUTED = "#6b7280"
BORDER = "#e5e7eb"
SURFACE = "#f8fafc"

HELP_STATE_KEY = "show_help"


def esc(text: object) -> str:
    return html.escape(str(text))


def inject_css() -> None:
    """Inject the app's stylesheet (our own classes only)."""
    st.markdown(
        f"""
        <style>
        .re-badge {{
            display: inline-block; padding: 0.12rem 0.6rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
            text-transform: uppercase; color: {ACCENT};
            background: {ACCENT_SOFT}; border: 1px solid #dbe4ff;
        }}
        .re-title {{
            font-weight: 650; letter-spacing: -0.015em; margin: 0.35rem 0 0.1rem 0;
        }}
        .re-section {{
            font-weight: 600; font-size: 1.05rem; margin: 0.2rem 0 0.4rem 0;
            color: {INK};
        }}
        .re-muted {{ color: {MUTED}; font-size: 0.86rem; line-height: 1.5; }}
        .re-help {{
            color: {MUTED}; font-size: 0.82rem; line-height: 1.45;
            margin: -0.15rem 0 0.5rem 0;
        }}
        .re-help b {{ color: {INK}; font-weight: 600; }}
        .re-good {{ color: #15803d; font-weight: 600; }}
        .re-bad {{ color: #b91c1c; font-weight: 600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_help() -> bool:
    """Whether inline explanations are currently enabled."""
    return bool(st.session_state.get(HELP_STATE_KEY, True))


def step_header(n: int, title: str, guide=None) -> None:
    """Step badge + title + a collapsed 'how this step works' guide."""
    st.markdown(f'<span class="re-badge">Step {n} of 6</span>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="re-title">{esc(title)}</h1>', unsafe_allow_html=True)
    if guide is not None:
        with st.expander("How this step works", expanded=False):
            st.markdown(f"**What.** {guide.what}")
            st.markdown(f"**Why.** {guide.why}")
            st.markdown(f"**Observe.** {guide.observe}")
            st.markdown(f"**Pitfalls.** {guide.pitfalls}")


def section(title: str, help: Optional[str] = None) -> None:
    """Consistent in-page section heading with an optional 'how to read' note."""
    st.markdown(f'<div class="re-section">{esc(title)}</div>', unsafe_allow_html=True)
    if help:
        note(help)


def note(text: str) -> None:
    """A muted explanatory line (hidden when explanations are toggled off)."""
    if not text or not show_help():
        return
    st.markdown(f'<div class="re-help">{esc(text)}</div>', unsafe_allow_html=True)


def raw_note(text: str) -> None:
    """A muted note that is always shown regardless of the help toggle."""
    if text:
        st.markdown(f'<div class="re-help">{esc(text)}</div>', unsafe_allow_html=True)


def examples(good: str, bad: str) -> None:
    """A compact good-vs-bad reading hint for a plot."""
    if not show_help():
        return
    st.markdown(
        f'<div class="re-help"><span class="re-good">Good:</span> {esc(good)} '
        f'&nbsp;·&nbsp; <span class="re-bad">Red flags:</span> {esc(bad)}</div>',
        unsafe_allow_html=True,
    )


_ADVISORIES = {
    "info": st.info,
    "warning": st.warning,
    "error": st.error,
    "success": st.success,
}


def advisory(level: str, text: str) -> None:
    """Single vocabulary for messages: info | warning | error | success."""
    _ADVISORIES.get(level, st.info)(text)


def _help_for(items: Sequence[str], helps) -> list[Optional[str]]:
    """Normalise a help argument to a list aligned with ``items``."""
    if helps is None:
        return [None] * len(items)
    if isinstance(helps, dict):
        return [helps.get(label) for label in items]
    return list(helps)


def metric_row(items: Sequence[tuple[str, object]], helps=None) -> None:
    """A row of compact metric cards, optionally with per-card help tooltips."""
    if not items:
        return
    cols = st.columns(len(items))
    help_values = _help_for([str(label) for label, _ in items], helps)
    for col, (label, value), help_text in zip(cols, items, help_values):
        col.metric(str(label), str(value), help=help_text)


def stat_row(items: Sequence[tuple[str, object]], helps=None) -> None:
    """Backwards-compatible alias for :func:`metric_row`."""
    metric_row(items, helps=helps)


@contextmanager
def card(label: Optional[str] = None, help: Optional[str] = None):
    """A bordered container, optionally with a label and a help note."""
    with st.container(border=True):
        if label:
            st.markdown(
                f'<div class="re-section">{esc(label)}</div>', unsafe_allow_html=True
            )
        if help:
            note(help)
        yield


def data_table(df: pd.DataFrame, **kwargs) -> None:
    """Consistent dataframe rendering."""
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)


def show_plot(fig) -> None:
    """Consistent plot rendering."""
    st.plotly_chart(fig, width="stretch")
