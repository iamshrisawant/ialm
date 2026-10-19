"""Column-type detection and leakage-aware problem definition.

Classifies columns as numeric, categorical, date-like, or identifier-like, and
flags potential target leakage (e.g. ``CO2(tCO2)`` when predicting
``Usage_kWh``) without silently dropping user-selected features.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

_ID_TOKENS = {"id", "index", "key", "uuid", "guid", "identifier"}
_DATE_TOKENS = {"date", "datetime", "timestamp"}


@dataclass
class ColumnProfile:
    name: str
    kind: str  # "numeric" | "categorical" | "date" | "identifier"
    n_missing: int
    n_unique: int
    constant: bool
    pct_missing: float


@dataclass
class ProblemDefinition:
    target: Optional[str]
    features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    date_features: list[str]
    identifier_features: list[str]
    profiles: dict[str, ColumnProfile] = field(default_factory=dict)
    constant_target: bool = False
    target_missing: int = 0
    target_outlier_count: int = 0
    target_is_numeric: bool = True
    leakage_flags: list[str] = field(default_factory=list)
    leakage_notes: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t]


def _name_looks_identifier(name: str) -> bool:
    toks = _tokens(name)
    if any(t in _ID_TOKENS for t in toks):
        return True
    lower = str(name).strip().lower()
    return lower.endswith("_id") or lower.endswith("_key") or lower.endswith("_index")


def _looks_identifier(name: str, series: pd.Series, n_rows: int) -> bool:
    """Identifier-like: name tokens (id/index/key/uuid) or near-unique text data."""
    if _name_looks_identifier(name):
        return True
    # High-cardinality text column with (almost) all-unique values. Guard with
    # an absolute threshold so small categorical columns are not misclassified.
    if not (pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)):
        n_unique = series.nunique(dropna=True)
        if n_rows >= 20 and n_unique >= 20 and n_unique / n_rows >= 0.95:
            return True
    return False


def _looks_date(name: str, series: pd.Series) -> bool:
    """Date-like: strong name token, datetime dtype, or values that parse."""
    if set(_tokens(name)) & _DATE_TOKENS:
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # Only trust parseability for non-numeric columns (numbers stay numeric).
    if not pd.api.types.is_numeric_dtype(series):
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.8:
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def profile_columns(df: pd.DataFrame) -> dict[str, ColumnProfile]:
    n_rows = len(df)
    profiles: dict[str, ColumnProfile] = {}
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        constant = n_unique <= 1
        pct_missing = (n_missing / n_rows * 100) if n_rows else 0.0

        # Identifier name tokens take precedence so numeric-looking IDs are caught.
        if _name_looks_identifier(col):
            kind = "identifier"
        elif pd.api.types.is_numeric_dtype(series):
            kind = "numeric"
        elif _looks_date(col, series):
            kind = "date"
        elif _looks_identifier(col, series, n_rows):
            kind = "identifier"
        else:
            kind = "categorical"

        profiles[col] = ColumnProfile(
            name=col,
            kind=kind,
            n_missing=n_missing,
            n_unique=n_unique,
            constant=constant,
            pct_missing=pct_missing,
        )
    return profiles


def date_columns(profiles: dict[str, ColumnProfile]) -> list[str]:
    """Names of all date-like columns (for chronological splitting)."""
    return [name for name, p in profiles.items() if p.kind == "date"]


def _derivable_pair(a: str, b: str) -> bool:
    """Heuristic: one column may be derived from / correlated with the other.

    Uses token/substring matching on normalized names, plus the explicit
    steel-industry case (Usage_kWh vs CO2(tCO2)).
    """
    na = re.sub(r"[^a-z0-9]", "", str(a).lower())
    nb = re.sub(r"[^a-z0-9]", "", str(b).lower())
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    if {na, nb} == {"usagekwh", "co2tco2"}:
        return True
    # Token-subset match: {"usage"} ⊆ {"usage", "rate"}.
    ta, tb = set(_tokens(a)), set(_tokens(b))
    smaller = ta if len(ta) <= len(tb) else tb
    larger = tb if len(ta) <= len(tb) else ta
    if smaller and smaller <= larger and len("".join(smaller)) >= 4:
        return True
    return False



def define_problem(
    df: pd.DataFrame,
    target: Optional[str],
    features: list[str],
    profiles: dict[str, ColumnProfile],
) -> ProblemDefinition:
    """Build a leakage-aware problem definition.

    Never silently removes user-selected features; instead it flags concerns
    and offers recommendations.
    """
    if target is None or target not in df.columns:
        return ProblemDefinition(
            target=None, features=[], numeric_features=[], categorical_features=[],
            date_features=[], identifier_features=[],
        )

    profiles = profiles or profile_columns(df)
    target_series = df[target]
    target_numeric = pd.api.types.is_numeric_dtype(df[target])

    const_target = bool(target_numeric and target_series.nunique(dropna=True) <= 1)
    target_missing = int(target_series.isna().sum())

    # Outlier count on the numeric target (IQR rule).
    target_outlier_count = 0
    if target_numeric and target_missing < len(target_series):
        clean = target_series.dropna()
        if len(clean) > 0:
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                target_outlier_count = int(
                    ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
                )

    selected = [f for f in features if f in df.columns and f != target]
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    date_features: list[str] = []
    identifier_features: list[str] = []

    for f in selected:
        kind = profiles.get(f, ColumnProfile(f, "categorical", 0, 0, False, 0.0)).kind
        if kind == "numeric":
            numeric_features.append(f)
        elif kind == "categorical":
            categorical_features.append(f)
        elif kind == "date":
            date_features.append(f)
        else:
            identifier_features.append(f)

    leakage_flags: list[str] = []
    leakage_notes: list[str] = []
    recommendations: list[str] = []

    if not target_numeric:
        leakage_flags.append(
            f"Target '{target}' is not numeric; regression requires a numeric target."
        )
        leakage_notes.append(
            "Regression predicts continuous numeric values. A categorical or "
            "text target is better handled by classification."
        )
        recommendations.append(
            f"Choose a numeric target instead of '{target}'."
        )

    # Identifier features selected as predictors -> potential leakage / no signal.
    if identifier_features:
        leakage_flags.append(
            "Identifier-like feature(s) selected as predictor(s): "
            + ", ".join(identifier_features)
        )
        leakage_notes.append(
            "Identifier columns (e.g. IDs, indices) usually carry no predictive "
            "signal and can leak information if they encode row order or target "
            "metadata. Consider removing them unless you have a specific reason."
        )
        recommendations.append(
            "Remove identifier-like features from the predictor list unless "
            "intentionally used."
        )

    # Derived-outcome leakage: target vs a candidate derived predictor.
    for f in selected:
        if f == target:
            continue
        if _derivable_pair(target, f):
            leakage_flags.append(
                f"Potential leakage: '{f}' may be derived from the target '{target}'."
            )
            leakage_notes.append(
                f"'{f}' looks like it could be derived from '{target}' (similar "
                f"name). Using it as a predictor may inflate performance "
                f"artificially (target leakage). For example, predicting "
                f"``Usage_kWh`` while using ``CO2(tCO2)`` can leak information "
                f"about electricity consumption."
            )
            recommendations.append(
                f"Consider excluding '{f}' from predictors, or keep it only to "
                f"demonstrate leakage — but interpret results with caution."
            )

    # Missingness in target / features.
    if target_missing:
        recommendations.append(
            f"Target '{target}' has {target_missing} missing value(s); rows "
            "with missing targets cannot be used for training."
        )
    high_missing_features = [
        f for f in selected if profiles.get(f, ColumnProfile(f, "", 0, 0, False, 0)).pct_missing > 50
    ]
    if high_missing_features:
        recommendations.append(
            "Feature(s) with >50% missing values: "
            + ", ".join(high_missing_features)
            + ". Consider imputation or exclusion."
        )

    # Constant features among predictors.
    const_features = [
        f for f in selected if profiles.get(f, ColumnProfile(f, "", 0, 0, False, 0)).constant
    ]
    if const_features:
        recommendations.append(
            "Constant predictor(s) (no variation): "
            + ", ".join(const_features)
            + ". They provide no signal; consider excluding."
        )

    return ProblemDefinition(
        target=target,
        features=selected,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        date_features=date_features,
        identifier_features=identifier_features,
        profiles=profiles,
        constant_target=const_target,
        target_missing=target_missing,
        target_outlier_count=target_outlier_count,
        target_is_numeric=bool(target_numeric),
        leakage_flags=leakage_flags,
        leakage_notes=leakage_notes,
        recommendations=recommendations,
    )
