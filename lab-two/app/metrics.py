"""Evaluation metrics and plain-language interpretations.

Reuses metric computation from ``models``; adds formatting, result tables, and
the R² interpretation (including negative R²) for students.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    from .models import r2_interpretation
except ImportError:  # run as a top-level module
    import models  # noqa: F401

    r2_interpretation = models.r2_interpretation


def format_metric(name: str, value: Optional[float]) -> str:
    if value is None:
        return "—"
    if name == "R2":
        return f"{value:.4f}"
    return f"{value:,.4f}"


def metrics_table(train_metrics: dict, test_metrics: dict) -> list[dict]:
    """Rows for a train-vs-test metrics comparison table."""
    rows = []
    for key in ("MAE", "MSE", "RMSE", "R2"):
        rows.append({
            "Metric": key,
            "Train": format_metric(key, train_metrics.get(key)),
            "Test": format_metric(key, test_metrics.get(key)),
        })
    return rows


_NUMERIC_COLUMNS = ("MAE", "MSE", "RMSE", "Test R2", "Train R2", "Gap")


def comparison_frame(results: list) -> pd.DataFrame:
    """A formatted one-row-per-model table of test and train scores.

    Failed models are kept as rows (with a status) so the comparison shows what
    was attempted; numeric columns are rounded for readability.
    """
    rows = []
    for r in results:
        if getattr(r, "success", False):
            test_r2 = r.test_metrics.get("R2")
            train_r2 = r.train_metrics.get("R2")
            gap = (
                None if (test_r2 is None or train_r2 is None)
                else train_r2 - test_r2
            )
            rows.append({
                "Model": r.label,
                "MAE": r.test_metrics.get("MAE"),
                "MSE": r.test_metrics.get("MSE"),
                "RMSE": r.test_metrics.get("RMSE"),
                "Test R2": test_r2,
                "Train R2": train_r2,
                "Gap": gap,
                "Status": "OK",
            })
        else:
            rows.append({
                "Model": r.label,
                "Status": f"Failed: {r.error or 'unknown error'}",
            })
    df = pd.DataFrame(rows)
    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    return df


def comparison_summary(results: list) -> str:
    """A one-line plain-language verdict over successfully trained models."""
    ok = [r for r in results if getattr(r, "success", False)]
    if not ok:
        return ""
    best = max(ok, key=lambda r: r.test_metrics.get("R2", float("-inf")))
    best_r2 = best.test_metrics.get("R2")
    parts = []
    if best_r2 is not None:
        parts.append(f"Best test R²: {best.label} ({best_r2:.3f}).")

    worst_gap, worst_label = None, None
    for r in ok:
        gap = r.train_metrics.get("R2", 0.0) - r.test_metrics.get("R2", 0.0)
        if worst_gap is None or gap > worst_gap:
            worst_gap, worst_label = gap, r.label

    if worst_gap is not None and worst_gap > 0.15:
        parts.append(
            f"Largest train–test gap: {worst_label} (+{worst_gap:.3f}) — a sign of "
            "overfitting; consider regularization, lower tree depth, or fewer "
            "polynomial terms."
        )
    elif worst_gap is not None:
        parts.append(
            "Train and test scores are close across models, so performance "
            "appears to generalise."
        )
    return " ".join(parts)


def explain_gap(train_r2: float, test_r2: float) -> str:
    diff = train_r2 - test_r2
    if diff > 0.15:
        return (
            f"Training R² ({train_r2:.3f}) is much higher than test R² "
            f"({test_r2:.3f}, gap {diff:.3f}) — a classic sign of overfitting: "
            "the model memorized training patterns that do not generalize."
        )
    if diff < -0.05:
        return (
            f"Test R² ({test_r2:.3f}) exceeds training R² ({train_r2:.3f}) — "
            "can happen by chance with small test sets; not necessarily overfitting."
        )
    return (
        f"Training R² ({train_r2:.3f}) and test R² ({test_r2:.3f}) are close — "
        "performance generalizes reasonably, assuming the split is representative."
    )


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Basic residual statistics to help judge bias and spread."""
    r = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if r.size == 0:
        return {"mean": 0.0, "std": 0.0, "max_abs": 0.0}
    return {
        "mean": float(np.mean(r)),
        "std": float(np.std(r)),
        "max_abs": float(np.max(np.abs(r))),
    }


def residual_hint(summary: dict, target_std: Optional[float] = None) -> str:
    """A short plain-language note about the residual pattern."""
    mean = summary.get("mean", 0.0)
    std = summary.get("std", 0.0)
    notes = []
    if abs(mean) > 0.1 * (std if std else 1.0) and std > 0:
        notes.append(
            "Residual mean is not near zero, suggesting a systematic bias the "
            "model is not capturing."
        )
    else:
        notes.append("Residual mean is near zero (little systematic bias).")
    if target_std and std > 0.5 * target_std:
        notes.append(
            "Residual spread is large relative to the target — predictions are "
            "noisy; consider more features or a different model."
        )
    return " ".join(notes)
