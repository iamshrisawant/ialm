"""Automated tests for Regression Explorer using small synthetic datasets.

No private datasets, no network access. Streamlit UI functions are not tested
here (they require a running script context); pure logic is covered.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "lab-two" / "app"


def _load_app_package():
    """Load lab-two/app as a top-level ``app`` package so relative imports work."""
    if "app" in sys.modules:
        return sys.modules["app"]
    pkg = types.ModuleType("app")
    pkg.__path__ = [str(APP_DIR)]
    pkg.__file__ = str(APP_DIR / "__init__.py")
    sys.modules["app"] = pkg
    for mod_name in [
        "data_io", "column_types", "preprocess", "models",
        "metrics", "plots", "importance", "state", "explain",
    ]:
        spec = importlib.util.spec_from_file_location(
            f"app.{mod_name}", APP_DIR / f"{mod_name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "app"
        sys.modules[f"app.{mod_name}"] = mod
        spec.loader.exec_module(mod)
    return pkg


_load_app_package()

from app import column_types, data_io, explain, models, metrics, preprocess, state  # noqa: E402
from app.column_types import date_columns, define_problem, profile_columns  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame({
        "usage": 2.5 * rng.normal(size=n) + 10.0,
        "temp": rng.normal(20, 5, size=n),
        "rooms": rng.integers(1, 6, size=n),
        "zone": rng.choice(["A", "B", "C"], size=n),
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "meter_id": [f"M{i:04d}" for i in range(n)],
        "derived": 0.4 * (2.5 * rng.normal(size=n) + 10.0) + rng.normal(0, 0.5, size=n),
    })
    return df


@pytest.fixture
def processed(synthetic_df):
    split = preprocess.split_data(
        synthetic_df, "usage", ["temp", "rooms", "zone", "date"],
        test_size=0.25, random_state=3,
    )
    return preprocess.preprocess(
        split,
        numeric_features=["temp", "rooms"],
        categorical_features=["zone"],
        date_features=["date"],
        scale_numeric=True,
    )


# --------------------------------------------------------------------------- #
# data_io
# --------------------------------------------------------------------------- #
def test_load_valid_csv(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    assert data_io.load_csv_from_disk(p).df.shape == (2, 2)


def test_load_empty_csv(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    res = data_io.load_csv_from_disk(p)
    assert not res.ok and res.error


def test_load_bytes():
    res = data_io.load_csv_from_bytes(b"x,y\n1,2\n3,4\n", "t.csv")
    assert res.ok and res.df.shape == (2, 2)


def test_no_numeric_columns():
    res = data_io.load_csv_from_bytes(b"name,city\nAna,Madrid\nBob,Madrid\n", "t.csv")
    assert not res.ok and "numeric" in res.error.lower()


def test_single_column_rejected():
    res = data_io.load_csv_from_bytes(b"a\n1\n2\n", "one.csv")
    assert not res.ok


def test_bom_and_whitespace_columns():
    raw = "\ufeffdate, value \n2023-01-01,1\n2023-01-02,2\n".encode("utf-8")
    res = data_io.load_csv_from_bytes(raw, "bom.csv")
    assert res.ok
    assert "date" in res.df.columns and "value" in res.df.columns


def test_semicolon_delimiter_sniffed():
    raw = b"a;b\n1;2\n3;4\n5;6\n"
    res = data_io.load_csv_from_bytes(raw, "semi.csv")
    assert res.ok
    assert list(res.df.columns) == ["a", "b"]
    assert res.df.shape == (3, 2)


def test_tab_delimiter_sniffed():
    raw = b"a\tb\n1\t2\n3\t4\n5\t6\n"
    res = data_io.load_csv_from_bytes(raw, "tab.tsv")
    assert res.ok and list(res.df.columns) == ["a", "b"]


def test_numbers_extracted_from_descriptive_text():
    raw = (
        "bp,sg,label\n"
        "< 112,1.019 - 1.021,x\n"
        "< 100,1.010 - 1.012,y\n"
        "< 120,1.020 - 1.022,z\n"
    ).encode("utf-8")
    res = data_io.load_csv_from_bytes(raw, "ranges.csv")
    assert res.ok
    assert res.df["bp"].tolist() == [112, 100, 120]
    assert res.df["sg"].tolist() == [1.019, 1.010, 1.020]
    assert any("Extracted numbers" in w for w in res.warnings)


def test_dates_are_not_extracted_as_numbers():
    raw = (b"when,val\n"
           b"01/01/2018 00:00,5\n02/01/2018 00:00,6\n"
           b"03/01/2018 00:00,7\n04/01/2018 00:00,8\n")
    res = data_io.load_csv_from_bytes(raw, "d.csv")
    assert res.ok
    assert "when" not in res.df.select_dtypes(include="number").columns
    kinds = {p.name: p.kind for p in column_types.profile_columns(res.df).values()}
    assert kinds["when"] == "date"


def test_free_text_is_not_extracted_as_numbers():
    raw = b"kind,val\ntype 1,5\ntype 2,6\ntype 3,7\ntype 4,8\n"
    res = data_io.load_csv_from_bytes(raw, "t.csv")
    assert res.ok
    assert "kind" not in res.df.select_dtypes(include="number").columns




def test_upload_size_limit(monkeypatch):
    monkeypatch.setattr(data_io, "MAX_UPLOAD_MB", 0)
    res = data_io.load_csv_from_bytes(b"a,b\n1,2\n", "big.csv")
    assert not res.ok and "limit" in res.error.lower()


def test_data_signature_changes():
    a = pd.DataFrame({"x": [1, 2, 3]})
    b = pd.DataFrame({"x": [1, 2, 4]})
    assert data_io.data_signature(a) != data_io.data_signature(b)
    assert data_io.data_signature(a) == data_io.data_signature(a.copy())


# --------------------------------------------------------------------------- #
# column_types
# --------------------------------------------------------------------------- #
def test_profile_columns_kinds(synthetic_df):
    kinds = {p.name: p.kind for p in profile_columns(synthetic_df).values()}
    assert kinds["temp"] == "numeric"
    assert kinds["zone"] == "categorical"
    assert kinds["meter_id"] == "identifier"
    assert kinds["date"] == "date"


def test_identifier_false_positives_avoided():
    df = pd.DataFrame({
        "humidity": [0.1, 0.2, 0.3],
        "valid": [1, 0, 1],
        "grid": [1, 2, 3],
        "reading_id": ["a", "b", "c"],
    })
    kinds = {p.name: p.kind for p in profile_columns(df).values()}
    assert kinds["humidity"] == "numeric"
    assert kinds["valid"] == "numeric"
    assert kinds["grid"] == "numeric"
    assert kinds["reading_id"] == "identifier"


def test_day_of_week_not_date():
    df = pd.DataFrame({"Day_of_week": ["Mon", "Tue", "Wed"], "x": [1, 2, 3]})
    kinds = {p.name: p.kind for p in profile_columns(df).values()}
    assert kinds["Day_of_week"] == "categorical"


def test_date_columns_helper(synthetic_df):
    profiles = profile_columns(synthetic_df)
    assert "date" in date_columns(profiles)


def test_define_problem_keeps_features(synthetic_df):
    profiles = profile_columns(synthetic_df)
    problem = define_problem(
        synthetic_df, target="usage",
        features=["temp", "rooms", "zone", "meter_id"], profiles=profiles,
    )
    assert set(problem.features) == {"temp", "rooms", "zone", "meter_id"}
    assert any("meter_id" in f for f in problem.leakage_flags)
    assert len(problem.recommendations) >= 1
    assert problem.target_is_numeric


def test_define_problem_derived_outcome_leakage():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Usage_kWh": rng.normal(100, 20, 50),
        "CO2(tCO2)": rng.normal(2, 0.5, 50),
        "temp": rng.normal(20, 5, 50),
        "meter_id": [f"M{i:03d}" for i in range(50)],
    })
    profiles = profile_columns(df)
    problem = define_problem(
        df, target="Usage_kWh",
        features=["temp", "CO2(tCO2)", "meter_id"], profiles=profiles,
    )
    assert set(problem.features) == {"temp", "CO2(tCO2)", "meter_id"}
    assert any("CO2(tCO2)" in f for f in problem.leakage_flags)
    assert any("meter_id" in f for f in problem.leakage_flags)
    assert any("CO2(tCO2)" in r for r in problem.recommendations)
    assert len(problem.recommendations) >= 2


def test_define_problem_constant_target(synthetic_df):
    profiles = profile_columns(synthetic_df)
    dfc = synthetic_df.copy()
    dfc["usage"] = 5.0
    assert define_problem(dfc, "usage", ["temp"], profiles).constant_target is True


def test_define_problem_non_numeric_target(synthetic_df):
    profiles = profile_columns(synthetic_df)
    problem = define_problem(synthetic_df, "zone", ["temp"], profiles)
    assert problem.target_is_numeric is False
    assert any("not numeric" in f for f in problem.leakage_flags)


# --------------------------------------------------------------------------- #
# preprocess
# --------------------------------------------------------------------------- #
def test_split_shapes(synthetic_df):
    s = preprocess.split_data(
        synthetic_df, "usage", ["temp", "rooms"], test_size=0.25, random_state=3,
    )
    assert s.n_train + s.n_test == 200
    assert s.method == "random"


def test_chronological_split_sorts_by_date():
    df = pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6, 7, 8],
        "x": [10, 20, 30, 40, 50, 60, 70, 80],
        "when": pd.to_datetime([
            "2023-01-05", "2023-01-01", "2023-01-08", "2023-01-03",
            "2023-01-07", "2023-01-02", "2023-01-06", "2023-01-04",
        ]),
    })
    # date column is NOT a predictor feature.
    split = preprocess.split_data(
        df, "y", ["x"], test_size=0.25, random_state=None,
        chronological=True, date_column="when",
    )
    assert split.method == "chronological"
    # n_test = round(8 * 0.25) = 2; the two latest dates are Jan 7 (y=5) and
    # Jan 8 (y=3).
    assert split.n_test == 2
    assert set(split.y_test.tolist()) == {3, 5}


def test_split_rejects_non_numeric_target(synthetic_df):
    with pytest.raises(ValueError):
        preprocess.split_data(synthetic_df, "zone", ["temp"], 0.2, 1)


def test_split_rejects_no_features(synthetic_df):
    with pytest.raises(ValueError):
        preprocess.split_data(synthetic_df, "usage", [], 0.2, 1)


def test_preprocess_fit_train_only(synthetic_df):
    split = preprocess.split_data(
        synthetic_df, "usage", ["temp", "rooms", "zone", "date"], 0.25, 3,
    )
    p = preprocess.preprocess(
        split, ["temp", "rooms"], ["zone"], ["date"], scale_numeric=True,
    )
    assert p.X_train_arr.shape[1] == p.X_test_arr.shape[1]
    assert p.y_train.shape[0] == split.n_train
    assert not np.isnan(p.X_train_arr).any() and not np.isnan(p.X_test_arr).any()
    assert len(p.feature_names_out) == p.X_train_arr.shape[1]


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def test_train_all_metrics(processed):
    keys = ["simple_linear", "linear", "ridge", "lasso", "tree", "forest", "knn", "polynomial"]
    results = models.train_all(
        keys, {k: {} for k in keys},
        processed.X_train_arr, processed.y_train,
        processed.X_test_arr, processed.y_test,
        feature_names=processed.feature_names_out, compute_importance=False,
    )
    assert len(results) == len(keys)
    for r in results:
        assert r.success, f"{r.label} failed: {r.error}"
        assert set(r.test_metrics) == {"MAE", "MSE", "RMSE", "R2"}


def test_tree_and_forest_allow_none_depth(processed):
    for key in ("tree", "forest"):
        spec = models.spec(key)
        for preset, vals in spec.presets.items():
            params = dict(vals)
            r = models.train_model(
                key, params, processed.X_train_arr, processed.y_train,
                processed.X_test_arr, processed.y_test,
                feature_names=processed.feature_names_out, compute_importance=False,
            )
            assert r.success, f"{key}/{preset} failed: {r.error}"


def test_coefficients_simple_linear(processed):
    r = models.train_model(
        "simple_linear", {}, processed.X_train_arr, processed.y_train,
        processed.X_test_arr, processed.y_test,
        feature_names=processed.feature_names_out, compute_importance=False,
    )
    coef = models.coefficients(r, processed.feature_names_out)
    assert coef is not None and len(coef["terms"]) == 1
    assert isinstance(coef["intercept"], float)


def test_coefficients_polynomial(processed):
    r = models.train_model(
        "polynomial", {"degree": 2, "include_bias": True},
        processed.X_train_arr, processed.y_train,
        processed.X_test_arr, processed.y_test,
        feature_names=processed.feature_names_out, compute_importance=False,
    )
    coef = models.coefficients(r, processed.feature_names_out)
    assert coef is not None and len(coef["terms"]) > 1


def test_coefficients_none_for_tree(processed):
    r = models.train_model(
        "tree", {}, processed.X_train_arr, processed.y_train,
        processed.X_test_arr, processed.y_test,
        feature_names=processed.feature_names_out, compute_importance=False,
    )
    assert models.coefficients(r, processed.feature_names_out) is None


def test_permutation_importance_shape(processed):
    r = models.train_model(
        "ridge", {}, processed.X_train_arr, processed.y_train,
        processed.X_test_arr, processed.y_test,
        feature_names=processed.feature_names_out, compute_importance=True,
    )
    assert r.importance is not None
    assert len(r.importance) == processed.X_train_arr.shape[1]
    assert len(r.importance_names) == len(r.importance)


def test_compute_metrics_simple():
    y = np.array([1.0, 2.0, 3.0])
    m = models.compute_metrics(y, y)
    assert abs(m["MAE"]) < 1e-9 and abs(m["R2"] - 1.0) < 1e-9


def test_r2_interpretation_negative():
    assert "Negative" in metrics.r2_interpretation(-0.3)


def test_unknown_model_fails_gracefully(processed):
    r = models.train_model(
        "does_not_exist", {}, processed.X_train_arr, processed.y_train,
        processed.X_test_arr, processed.y_test,
    )
    assert not r.success and r.error


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_residual_summary():
    s = metrics.residual_summary(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert s["mean"] == 0.0 and s["max_abs"] == 0.0


def test_metrics_table_rows():
    rows = metrics.metrics_table({"MAE": 1.0, "MSE": 2.0, "RMSE": 1.4, "R2": 0.9},
                                 {"MAE": 1.2, "MSE": 2.2, "RMSE": 1.5, "R2": 0.8})
    assert [r["Metric"] for r in rows] == ["MAE", "MSE", "RMSE", "R2"]


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
def test_state_fingerprint_changes():
    s1 = state.AppState(target="y", features=["a"], test_size=0.2)
    s1.refresh_fingerprint()
    s2 = state.AppState(target="y", features=["a", "b"], test_size=0.2)
    s2.refresh_fingerprint()
    assert s1.fingerprint != s2.fingerprint
    assert s1.results_match_current() is False


def test_state_data_vs_model_fingerprint():
    s = state.AppState(target="y", features=["a"], data_signature="sig")
    s.refresh_fingerprint()
    data_fp = s.data_fingerprint
    s.model_keys = ["ridge"]
    s.model_params = {"ridge": {"alpha": 5.0}}
    s.refresh_fingerprint()
    # data fingerprint is unchanged by model configuration
    assert s.data_fingerprint == data_fp


# --------------------------------------------------------------------------- #
# explanation layer & display formatting
# --------------------------------------------------------------------------- #
def test_display_feature_names_strips_prefixes():
    names = preprocess.display_feature_names(
        ["num__age", "cat__sex_male", "date__when"]
    )
    assert names == ["age", "sex_male", "when"]


def test_display_feature_names_disambiguates_duplicates():
    names = preprocess.display_feature_names(["num__x", "cat__x"])
    assert names == ["x", "x (1)"]


def test_explanation_registry_populated():
    assert explain.help_for("train.comparison")
    assert explain.help_for("dataset.stats")
    assert explain.column_kind("numeric")
    glossary_terms = [term for term, _ in explain.glossary()]
    assert "R²" in glossary_terms and "Residual" in glossary_terms
    assert explain.reading_results()


def test_comparison_frame_and_summary(processed):
    results = models.train_all(
        ["linear", "tree"], {"linear": {}, "tree": {}},
        processed.X_train_arr, processed.y_train,
        processed.X_test_arr, processed.y_test,
        feature_names=processed.feature_names_out, compute_importance=False,
    )
    df = metrics.comparison_frame(results)
    assert {"Model", "MAE", "Test R2", "Train R2", "Gap"}.issubset(df.columns)
    assert len(df) == len(results)
    summary = metrics.comparison_summary(results)
    assert "Best test" in summary


def test_comparison_frame_keeps_failures():
    class _R:
        label = "Broken"
        success = False
        error = "boom"
        test_metrics = {}
        train_metrics = {}

    df = metrics.comparison_frame([_R()])
    assert df.iloc[0]["Status"].startswith("Failed")

