"""Leakage-safe preprocessing with scikit-learn Pipelines and ColumnTransformers.

Imputers, scalers, and encoders are fitted on training data only. Supports
random train/test splitting and optional chronological splitting when a date
column is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    method: str  # "random" | "chronological"
    test_size: float
    random_state: Optional[int]
    n_train: int
    n_test: int


@dataclass
class Preprocessed:
    X_train_arr: np.ndarray
    X_test_arr: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names_out: list[str]
    split: SplitResult


def _to_epoch_seconds(series: pd.Series) -> np.ndarray:
    """Convert a date-like column to seconds since epoch (NaT -> NaN)."""
    parsed = pd.to_datetime(series, errors="coerce")
    ns = parsed.astype(np.int64).to_numpy(dtype=np.float64)
    ns[ns == np.iinfo(np.int64).min] = np.nan  # iNaT -> NaN
    return ns / 1_000_000_000.0


def _impute_date(X: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """Convert date columns to numeric (seconds since epoch)."""
    X = X.copy()
    for col in date_cols:
        if col not in X.columns:
            continue
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = _to_epoch_seconds(X[col])
    return X


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    date_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Construct a ColumnTransformer fit only on training data.

    Numeric: median impute, optional standard scaling.
    Categorical: most-frequent impute, one-hot encode (handle unknown).
    Date: converted to epoch seconds then median-imputed + scaled.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    date_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))
    if date_features:
        transformers.append(("date", date_pipe, date_features))

    if not transformers:
        raise ValueError("No features to preprocess.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_data(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    test_size: float,
    random_state: Optional[int],
    chronological: bool = False,
    date_column: Optional[str] = None,
) -> SplitResult:
    """Split into train/test.

    When ``chronological`` and a date column exist, sort the full dataframe by
    date (the date column need not be a predictor) and take the last
    ``test_size`` fraction as test. Otherwise use a random split with the seed.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not in the dataset.")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError("The target must be a numeric column for regression.")
    features = [f for f in features if f in df.columns and f != target]
    if not features:
        raise ValueError("Select at least one predictor feature.")

    data = df.loc[df[target].notna()].copy()
    n = len(data)
    if n < 2:
        raise ValueError("Insufficient samples with non-missing target to split.")

    use_chrono = bool(chronological and date_column and date_column in data.columns)
    if use_chrono:
        order = pd.to_datetime(data[date_column], errors="coerce")
        order = order.fillna(pd.Timestamp.max)  # missing dates go last
        data = data.loc[order.sort_values().index]

    X = data[features]
    y = data[target]

    n_test = max(1, int(round(n * test_size)))
    n_test = min(n_test, n - 1)

    if use_chrono:
        X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
        y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
        method = "chronological"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=n_test / n, random_state=random_state, shuffle=True
        )
        method = "random"

    return SplitResult(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        method=method,
        test_size=test_size,
        random_state=random_state,
        n_train=len(X_train),
        n_test=len(X_test),
    )


def display_feature_names(names: list[str]) -> list[str]:
    """Turn transformer-prefixed names into readable labels.

    ``ColumnTransformer.get_feature_names_out`` returns names such as
    ``num__age`` or ``cat__sex_male``. Students should see ``age`` and
    ``sex_male``; duplicate labels are disambiguated with a numeric suffix.
    """
    out: list[str] = []
    seen: dict[str, int] = {}
    for raw in names:
        name = str(raw)
        if "__" in name:
            _, _, rest = name.partition("__")
            if rest:
                name = rest
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 0
        out.append(name)
    return out


def preprocess(
    split: SplitResult,
    numeric_features: list[str],
    categorical_features: list[str],
    date_features: list[str],
    scale_numeric: bool = True,
) -> Preprocessed:
    """Fit preprocessor on training data only; transform train and test."""
    X_train = split.X_train.copy()
    X_test = split.X_test.copy()

    # Convert date columns to numeric before sklearn pipeline.
    if date_features:
        X_train = _impute_date(X_train, date_features)
        X_test = _impute_date(X_test, date_features)
        # Treat converted date columns as numeric for scaling.
        numeric_features = list(numeric_features) + list(date_features)
        date_features = []

    preprocessor = build_preprocessor(
        numeric_features, categorical_features, date_features, scale_numeric
    )

    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    X_train_arr = np.asarray(X_train_arr)
    X_test_arr = np.asarray(X_test_arr)
    if X_train_arr.shape[1] == 0:
        raise ValueError(
            "All selected features are empty after preprocessing "
            "(entirely missing or unusable). Choose different features."
        )

    try:
        feature_names_out = list(preprocessor.get_feature_names_out())
    except Exception:  # noqa: BLE001
        feature_names_out = [f"feature_{i}" for i in range(X_train_arr.shape[1])]

    return Preprocessed(
        X_train_arr=X_train_arr,
        X_test_arr=X_test_arr,
        y_train=split.y_train.to_numpy(dtype=float),
        y_test=split.y_test.to_numpy(dtype=float),
        preprocessor=preprocessor,
        feature_names_out=feature_names_out,
        split=split,
    )
