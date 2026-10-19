"""Model registry, presets, and failure-isolated training for regression.

Supports Simple/Multiple Linear, Polynomial, Ridge, Lasso, Decision Tree,
Random Forest, and K-Nearest Neighbors regression. Each model is trained
independently so one failure never aborts a comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, MSE, RMSE, R² for a set of predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def r2_interpretation(r2: float) -> str:
    """Plain-language explanation of R², including negative values."""
    if r2 < 0:
        return (
            "Negative R²: the model predicts worse than a horizontal line at the "
            "mean of the target. This usually signals underfitting, leakage, or "
            "a target that is not linearly related to the features."
        )
    if r2 < 0.25:
        return "Weak explanatory power (R² close to 0)."
    if r2 < 0.5:
        return "Moderate explanatory power."
    if r2 < 0.75:
        return "Strong explanatory power."
    return "Very strong explanatory power."


# --------------------------------------------------------------------------- #
# Model specifications
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    key: str
    label: str
    description: str
    build: Callable[[dict], RegressorMixin]
    param_schema: dict  # widget config for the UI
    presets: dict[str, dict]  # preset name -> param values
    concept: str  # short teaching note


def _linear(params: dict) -> RegressorMixin:
    return LinearRegression()


def _first_column(X):
    X = np.asarray(X)
    return X[:, :1]


def _simple_linear(params: dict) -> RegressorMixin:
    """Linear regression on a single (the first selected) feature."""
    return Pipeline(
        steps=[
            ("first", FunctionTransformer(_first_column, validate=False)),
            ("linear", LinearRegression()),
        ]
    )


def _polynomial(params: dict) -> RegressorMixin:
    degree = int(params.get("degree", 2))
    include_bias = bool(params.get("include_bias", True))
    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=include_bias)),
            ("linear", LinearRegression()),
        ]
    )


def _ridge(params: dict) -> RegressorMixin:
    return Ridge(alpha=float(params.get("alpha", 1.0)), random_state=42)


def _lasso(params: dict) -> RegressorMixin:
    return Lasso(alpha=float(params.get("alpha", 0.001)), random_state=42, max_iter=100000)


def _tree(params: dict) -> RegressorMixin:
    return DecisionTreeRegressor(
        max_depth=params.get("max_depth"),
        min_samples_split=int(params.get("min_samples_split", 2)),
        random_state=42,
    )


def _forest(params: dict) -> RegressorMixin:
    return RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=params.get("max_depth"),
        min_samples_split=int(params.get("min_samples_split", 2)),
        random_state=42,
        n_jobs=1,
    )


def _knn(params: dict) -> RegressorMixin:
    return KNeighborsRegressor(
        n_neighbors=int(params.get("n_neighbors", 5)),
        weights=params.get("weights", "uniform"),
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "simple_linear": ModelSpec(
        key="simple_linear",
        label="Simple Linear Regression",
        description="Linear regression using a single predictor (the first "
        "selected feature): y = b₀ + b₁x. Useful to visualise a straight-line "
        "relationship before adding more features.",
        build=_simple_linear,
        param_schema={},
        presets={"baseline": {}},
        concept=(
            "The slope b₁ is the change in the target per one-unit change in the "
            "feature; b₀ is the intercept. Only one feature is used, so select a "
            "single feature to control which one."
        ),
    ),
    "linear": ModelSpec(
        key="linear",
        label="Multiple Linear Regression",
        description="Fits a straight line/plane: y = b₀ + b₁x₁ + … + bₚxₚ. With "
        "one feature it reduces to simple linear regression.",
        build=_linear,
        param_schema={},
        presets={
            "baseline": {},
        },
        concept=(
            "Coefficients show how much the target changes per one-unit change in "
            "each feature (holding others fixed). The intercept is the predicted "
            "value when all features are zero."
        ),
    ),
    "polynomial": ModelSpec(
        key="polynomial",
        label="Polynomial Regression",
        description="Adds polynomial and interaction terms (degree ≥ 2) before "
        "fitting linear regression, capturing curved relationships.",
        build=_polynomial,
        param_schema={
            "degree": {"widget": "slider", "min": 2, "max": 5, "value": 2, "step": 1,
                       "help": "Higher degree = more flexible curve, more overfitting risk."},
            "include_bias": {"widget": "checkbox", "value": True,
                             "help": "Include intercept (column of ones)."},
        },
        presets={
            "baseline": {"degree": 2, "include_bias": True},
            "complex": {"degree": 4, "include_bias": True},
        },
        concept=(
            "Degree controls curve flexibility. Low degree may underfit (miss "
            "curves); high degree may overfit (memorize noise)."
        ),
    ),
    "ridge": ModelSpec(
        key="ridge",
        label="Ridge Regression",
        description="Linear regression with L2 regularization: penalizes large "
        "coefficients, shrinking them toward zero but rarely exactly to zero.",
        build=_ridge,
        param_schema={
            "alpha": {"widget": "slider", "min": 0.0, "max": 10.0, "value": 1.0,
                      "step": 0.1,
                      "help": "Higher alpha = stronger regularization, more shrinkage."},
        },
        presets={
            "baseline": {"alpha": 1.0},
            "regularized": {"alpha": 10.0},
        },
        concept=(
            "Regularization trades a little bias for lower variance, helping when "
            "features are correlated or the sample is small."
        ),
    ),
    "lasso": ModelSpec(
        key="lasso",
        label="Lasso Regression",
        description="Linear regression with L1 regularization: can shrink some "
        "coefficients exactly to zero, performing automatic feature selection.",
        build=_lasso,
        param_schema={
            "alpha": {"widget": "slider", "min": 0.0, "max": 1.0, "value": 0.001,
                      "step": 0.001, "format": "%.4f",
                      "help": "Higher alpha = stronger shrinkage, more zero coefficients."},
        },
        presets={
            "baseline": {"alpha": 0.001},
            "regularized": {"alpha": 0.01},
        },
        concept=(
            "L1 penalty drives unimportant feature coefficients to zero, which can "
            "make the model easier to interpret."
        ),
    ),
    "tree": ModelSpec(
        key="tree",
        label="Decision Tree Regression",
        description="Splits the feature space into regions and predicts the mean "
        "target within each region. Captures non-linear patterns and interactions.",
        build=_tree,
        param_schema={
            "max_depth": {"widget": "selectbox", "options": [None] + list(range(1, 21)),
                          "value": 5, "label": "Max depth",
                          "help": "None = grow until leaves are pure (overfits more)."},
            "min_samples_split": {"widget": "slider", "min": 2, "max": 20, "value": 2,
                                  "step": 1,
                                  "help": "Minimum samples required to split a node."},
        },
        presets={
            "baseline": {"max_depth": 5, "min_samples_split": 2},
            "complex": {"max_depth": None, "min_samples_split": 2},
        },
        concept=(
            "Trees model non-linear relationships without assuming linearity, but "
            "a single tree easily overfits the training data."
        ),
    ),
    "forest": ModelSpec(
        key="forest",
        label="Random Forest Regression",
        description="Ensemble of many decision trees trained on bootstrapped "
        "samples and feature subsets; averages their predictions to reduce variance.",
        build=_forest,
        param_schema={
            "n_estimators": {"widget": "slider", "min": 10, "max": 300, "value": 100,
                             "step": 10, "help": "More trees = more stable, slower."},
            "max_depth": {"widget": "selectbox", "options": [None] + list(range(1, 21)),
                          "value": 5, "label": "Max depth per tree",
                          "help": "None = unlimited depth (overfits more)."},
            "min_samples_split": {"widget": "slider", "min": 2, "max": 20, "value": 2,
                                  "step": 1},
        },
        presets={
            "baseline": {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
            "complex": {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
        },
        concept=(
            "Bagging many decorrelated trees reduces variance compared to a single "
            "tree — the core idea of ensemble learning."
        ),
    ),
    "knn": ModelSpec(
        key="knn",
        label="K-Nearest Neighbors Regression",
        description="Predicts the average target of the k nearest training points "
        "in feature space. Sensitive to feature scaling (hence preprocessing).",
        build=_knn,
        param_schema={
            "n_neighbors": {"widget": "slider", "min": 1, "max": 50, "value": 5,
                            "step": 1,
                            "help": "Fewer neighbors = more flexible, noisier."},
            "weights": {"widget": "selectbox", "options": ["uniform", "distance"],
                        "value": "uniform",
                        "help": "'distance' weights closer neighbors more heavily."},
        },
        presets={
            "baseline": {"n_neighbors": 5, "weights": "uniform"},
        },
        concept=(
            "KNN makes no parametric assumptions but relies on a meaningful "
            "distance metric — always scale features first."
        ),
    ),
}

PRESET_ORDER = ["baseline", "regularized", "complex"]


def spec(key: str) -> Optional[ModelSpec]:
    return MODEL_SPECS.get(key)


# --------------------------------------------------------------------------- #
# Training with failure isolation
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    key: str
    label: str
    success: bool
    model: Optional[RegressorMixin] = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    y_train_pred: Optional[np.ndarray] = None
    y_test_pred: Optional[np.ndarray] = None
    importance: Optional[np.ndarray] = None
    importance_names: list[str] = field(default_factory=list)
    error: Optional[str] = None


def train_model(
    key: str,
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[list[str]] = None,
    compute_importance: bool = True,
) -> ModelResult:
    """Train a single model, isolating failures so one never aborts others."""
    s = spec(key)
    if s is None:
        return ModelResult(key=key, label=key, success=False,
                           error=f"Unknown model: {key}")

    label = s.label
    try:
        if X_train.shape[0] < 1 or X_train.shape[1] < 1:
            return ModelResult(
                key=key, label=label, success=False,
                error="No usable training data (empty feature matrix).",
            )
        model = clone(s.build(params))
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        importance = None
        importance_names = list(feature_names or [])
        if compute_importance and X_test.shape[0] > 0 and X_test.shape[1] > 0:
            try:
                imp = permutation_importance(
                    model, X_test, y_test, n_repeats=10, random_state=42,
                    n_jobs=1,
                )
                importance = imp.importances_mean
                if importance_names is None or len(importance_names) != len(importance):
                    importance_names = [f"feature_{i}" for i in range(len(importance))]
            except Exception:  # noqa: BLE001
                importance = None

        return ModelResult(
            key=key, label=label, success=True, model=model,
            train_metrics=train_metrics, test_metrics=test_metrics,
            y_train_pred=y_train_pred, y_test_pred=y_test_pred,
            importance=importance, importance_names=importance_names,
        )
    except Exception as exc:  # noqa: BLE001 - isolate failures
        return ModelResult(key=key, label=label, success=False,
                           error=f"{type(exc).__name__}: {exc}")


def train_all(
    keys: list[str],
    params_map: dict[str, dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[list[str]] = None,
    compute_importance: bool = True,
) -> list[ModelResult]:
    """Train several models; failures are captured per-model."""
    return [
        train_model(
            key, params_map.get(key, {}), X_train, y_train, X_test, y_test,
            feature_names=feature_names, compute_importance=compute_importance,
        )
        for key in keys
    ]


def _linear_solver(model):
    """Return (coef, intercept, expanded_names) for linear-family models."""
    if isinstance(model, Pipeline):
        if "poly" in model.named_steps and "linear" in model.named_steps:
            return "poly", model.named_steps["poly"], model.named_steps["linear"]
        if "first" in model.named_steps and "linear" in model.named_steps:
            return "first", None, model.named_steps["linear"]
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        return "direct", None, model
    return None


def coefficients(result: ModelResult, feature_names: list[str]) -> Optional[dict]:
    """Extract intercept and coefficients for linear-family models.

    Returns ``{"intercept": float, "terms": [(name, coef), ...]}`` or None for
    models without interpretable coefficients (trees, forests, KNN).
    """
    if not result.success or result.model is None:
        return None
    kind = _linear_solver(result.model)
    if kind is None:
        return None
    _, poly, lin = kind
    coef = np.ravel(lin.coef_)
    intercept = float(np.ravel(lin.intercept_)[0])

    if kind == "poly" and poly is not None:
        try:
            names = list(poly.get_feature_names_out(feature_names))
        except Exception:  # noqa: BLE001
            names = [f"term_{i}" for i in range(len(coef))]
    elif kind == "first":
        names = [feature_names[0] if feature_names else "feature_0"]
    else:
        names = list(feature_names) if len(coef) == len(feature_names) else \
            [f"feature_{i}" for i in range(len(coef))]

    terms = [(str(n), float(c)) for n, c in zip(names, coef)]
    return {"intercept": intercept, "terms": terms}
