"""Regression Explorer — Streamlit app.

Guided, student-friendly workflow for interactive regression experimentation.
Six steps mirror the learning workflow: Explore → Define → Configure → Select
→ Train/Evaluate → Interpret/Export. Navigation is a persistent sidebar; one
step is rendered at a time, and every output carries a short "how to read this"
note (togglable from the sidebar).

Run from the repository root:
    streamlit run lab-two/app/main.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the app directory is importable when streamlit runs this file directly
# (no parent package) as well as when imported as part of the ``app`` package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

try:
    from . import column_types, data_io, explain, importance, metrics, models, plots, preprocess, state, ui
    from .column_types import date_columns, define_problem, profile_columns
except ImportError:  # run as a top-level script (streamlit run app/main.py)
    import column_types  # noqa: F401
    import data_io  # noqa: F401
    import explain  # noqa: F401
    import importance  # noqa: F401
    import metrics  # noqa: F401
    import models  # noqa: F401
    import plots  # noqa: F401
    import preprocess  # noqa: F401
    import state  # noqa: F401
    import ui  # noqa: F401
    from column_types import date_columns, define_problem, profile_columns

st.set_page_config(page_title="Regression Explorer", layout="wide")
ui.inject_css()

S = state.get_state()

METRIC_ORDER = ["MAE", "MSE", "RMSE", "R2"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ensure_state(key: str, default) -> None:
    if key not in st.session_state:
        st.session_state[key] = default


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include="number").columns)


def _get_df() -> "pd.DataFrame | None":
    return st.session_state.get("_loaded_df")


def _get_profiles(df: pd.DataFrame):
    return profile_columns(df)


def _header(n: int, title: str, step: str) -> None:
    """Step badge + title + collapsible guide."""
    ui.step_header(n, title, explain.guide(step))


def _flag(text: str) -> None:
    """An advisory (kept as a thin wrapper over the shared vocabulary)."""
    ui.advisory("warning", text)


STEP_NAMES = [
    "Explore Data",
    "Define Problem",
    "Configure Experiment",
    "Select Models",
    "Train & Evaluate",
    "Interpret & Export",
]


def _nav_labels() -> list[str]:
    return [f"{i + 1}. {name}" for i, name in enumerate(STEP_NAMES)]


def _set_nav(label: str) -> None:
    """Jump to a workflow step (used as a button callback).

    Setting the radio's session key inside a callback is safe because callbacks
    run before the script body on the next rerun, i.e. before the nav widget is
    re-instantiated.
    """
    st.session_state["nav_step"] = label


def _goto_button(label: str, target_index: int, key: str) -> None:
    st.button(label, key=key, on_click=_set_nav, args=(_nav_labels()[target_index],))


def _require_dataset() -> bool:
    """Guard for steps that need a loaded dataset; offers a jump back to Step 1."""
    if _get_df() is None:
        ui.advisory("info", "Load a dataset in Step 1 first.")
        _goto_button("Go to Step 1 — Explore Data", 0, key="goto_step1")
        return False
    return True


def _download_button(label: str, data: bytes, file_name: str, mime: str, key: str) -> None:
    st.download_button(label=label, data=data, file_name=file_name, mime=mime, key=key)


def _clear_derived(clear_split: bool = True) -> None:
    """Drop results and (optionally) split/preprocessing state."""
    if clear_split:
        for k in ("_split", "_split_key"):
            st.session_state.pop(k, None)
    for k in ("_preprocessed", "_processed_key", "_results_for_export",
              "_processed_for_export"):
        st.session_state.pop(k, None)
    state.reset_results(S)


def _reset_for_dataset(df: pd.DataFrame, name: str, source: str) -> None:
    """Reset derived state when a different dataset is loaded."""
    S.data_signature = data_io.data_signature(df)
    S.dataset_name = name
    S.dataset_source = source
    S.df_shape = df.shape
    S.target = None
    S.features = []
    S.problem = None
    st.session_state["_loaded_df"] = df
    for k in ("_problem", "_profiles"):
        st.session_state.pop(k, None)
    _clear_derived(clear_split=True)
    # Clear widget state so stale selections don't carry over.
    for k in list(st.session_state.keys()):
        if k.startswith(("define_", "cfg_", "select_", "param_", "explore_", "_preset")):
            st.session_state.pop(k, None)


# --------------------------------------------------------------------------- #
# Step 1 — Explore Data
# --------------------------------------------------------------------------- #
def step_explore() -> None:
    _header(1, "Explore Data", "explore")

    with ui.card("Choose a dataset"):
        result = data_io.render_dataset_selector()
    if not result.ok:
        return

    df = result.df
    changed = (
        data_io.data_signature(df) != S.data_signature
        or result.name != S.dataset_name
        or result.source != S.dataset_source
    )
    if changed:
        _reset_for_dataset(df, result.name, result.source)
    else:
        st.session_state["_loaded_df"] = df

    st.session_state["_profiles"] = profile_columns(df)

    profiles = st.session_state["_profiles"]

    ui.stat_row([
        ("Rows", f"{df.shape[0]:,}"),
        ("Columns", f"{df.shape[1]:,}"),
        ("Missing cells", f"{int(df.isna().sum().sum()):,}"),
        ("Duplicate rows", f"{int(df.duplicated().sum()):,}"),
    ], helps=[
        "Number of observations (data points) in the dataset.",
        "Number of variables available as target or features.",
        "Total empty cells across all rows and columns.",
        "Rows that are exact copies of an earlier row.",
    ])
    ui.note(explain.help_for("dataset.stats"))

    num_cols = _numeric_cols(df)

    tab_preview, tab_types, tab_stats, tab_charts = st.tabs(
        ["Preview", "Types & missing", "Statistics", "Distributions & correlations"]
    )

    with tab_preview:
        ui.section("First rows", help=explain.help_for("dataset.preview"))
        ui.data_table(df.head(10))

    with tab_types:
        ui.section("Column types & missing values",
                   help=explain.help_for("dataset.types"))
        type_df = pd.DataFrame([
            {
                "Column": p.name,
                "Kind": p.kind,
                "Stored as": str(df[p.name].dtype),
                "Missing": p.n_missing,
                "Missing %": round(p.pct_missing, 2),
                "Unique": p.n_unique,
                "Constant": "yes" if p.constant else "",
            }
            for p in profiles.values()
        ])
        ui.data_table(type_df)
        kinds_present = [k for k in ("numeric", "categorical", "date", "identifier")
                         if any(p.kind == k for p in profiles.values())]
        if kinds_present:
            ui.note(explain.help_for("column.kinds"))
            for kind in kinds_present:
                st.markdown(f"- **{kind}** — {explain.column_kind(kind)}")

    with tab_stats:
        ui.section("Descriptive statistics (numeric columns)",
                   help=explain.help_for("dataset.stats_table"))
        if num_cols:
            ui.data_table(df[num_cols].describe().T)
        else:
            ui.advisory("info", "No numeric columns detected.")

    with tab_charts:
        ui.section("Distribution", help=explain.help_for("dataset.distribution"))
        if num_cols:
            dist_col = st.selectbox("Column", options=num_cols, key="explore_dist_col")
            ui.show_plot(
                plots.histogram(df[dist_col].dropna().to_numpy(), label=dist_col)
            )
        else:
            ui.advisory("info", "No numeric columns to plot.")

        ui.section("Numeric correlations",
                   help=explain.help_for("dataset.correlation"))
        if len(num_cols) >= 2:
            corr = df[num_cols].corr(numeric_only=True)
            ui.show_plot(plots.correlation_heatmap(corr))
        else:
            ui.advisory("info", "Need at least two numeric columns for correlations.")


# --------------------------------------------------------------------------- #
# Step 2 — Define the Regression Problem
# --------------------------------------------------------------------------- #
def step_define() -> None:
    _header(2, "Define the Regression Problem", "define")

    if not _require_dataset():
        return

    df = _get_df()
    if df is None:
        ui.advisory("info", "Dataset not in memory. Reload it in Step 1.")
        _goto_button("Go to Step 1 — Explore Data", 0, key="goto_step1b")
        return

    profiles = st.session_state.get("_profiles") or _get_profiles(df)
    numeric_cols = _numeric_cols(df)

    left, right = st.columns([1, 2], gap="large")

    with left:
        with ui.card("Target — what to predict", help=explain.help_for("define.target")):
            _ensure_state("define_target", S.target if S.target in numeric_cols else None)
            if st.session_state["define_target"] not in numeric_cols:
                st.session_state["define_target"] = None
            target = st.selectbox(
                "Target column",
                options=numeric_cols,
                index=None,
                placeholder="Select a numeric target…",
                key="define_target",
                help="Must be numeric and continuous. The model predicts this value.",
            )
            S.target = target

            if target is not None:
                target_series = df[target]
                ui.stat_row([
                    ("Missing in target", int(target_series.isna().sum())),
                    ("Unique values", int(target_series.nunique(dropna=True))),
                ], helps=[
                    "Target rows without a value cannot be used for training or "
                    "evaluation.",
                    "Distinct target values. One unique value means a constant "
                    "target the model cannot learn from.",
                ])
                if target_series.nunique(dropna=True) <= 1:
                    ui.advisory("error", "Target is constant (no variation) — a "
                                "model cannot learn from it.")

    with right:
        with ui.card("Predictor features — the inputs",
                     help=explain.help_for("define.features")):
            options = [c for c in df.columns if c != target]
            prev = st.session_state.get("define_features")
            if prev is None:
                prev = list(options)
            prev = [c for c in prev if c in options]
            st.session_state["define_features"] = prev
            features = st.multiselect(
                "Feature columns",
                options=options,
                key="define_features",
                help="Inputs used to predict the target. Identifier-like columns "
                     "are flagged below; nothing is removed automatically.",
            )
            S.features = features

    if target is None:
        ui.advisory("info", "Choose a numeric target to analyse the problem.")
        return
    if not features:
        ui.advisory("info", "Select at least one predictor feature.")
        return

    problem = define_problem(df, target, features, profiles)
    S.problem = problem
    st.session_state["_problem"] = problem

    if not problem.target_is_numeric:
        _flag(f"Target '{target}' is not numeric; regression needs a numeric target.")

    if problem.constant_target:
        _flag("Constant target. The target does not vary, so no meaningful "
              "regression is possible.")
    if problem.target_missing:
        ui.advisory("warning",
                    f"Target has {problem.target_missing} missing value(s); those "
                    "rows cannot be used for training or evaluation.")
    if problem.target_outlier_count:
        ui.advisory("info",
                    f"Target has {problem.target_outlier_count} potential "
                    "outlier(s) (IQR rule). Outliers can strongly influence linear "
                    "models and tree splits.")

    if problem.leakage_flags or problem.recommendations:
        with ui.card("Checks & recommendations",
                     help=explain.help_for("define.flags")):
            for flag in problem.leakage_flags:
                _flag(flag)
            for note in problem.leakage_notes:
                st.markdown(note)
            if problem.recommendations:
                for r in problem.recommendations:
                    st.markdown(f"- {r}")
            if problem.leakage_flags:
                st.caption(
                    "These are recommendations, not enforced exclusions. You keep "
                    "full control over which features to use."
                )

    ui.section("Selected features by type", help=explain.help_for("define.type_counts"))
    ui.stat_row([
        ("Numeric", len(problem.numeric_features)),
        ("Categorical", len(problem.categorical_features)),
        ("Date", len(problem.date_features)),
        ("Identifier-like", len(problem.identifier_features)),
    ], helps=[
        explain.column_kind("numeric"),
        explain.column_kind("categorical"),
        explain.column_kind("date"),
        explain.column_kind("identifier"),
    ])
    if problem.identifier_features:
        st.caption(
            "Identifier-like features usually carry no predictive signal and can "
            "leak row-order or metadata information."
        )


# --------------------------------------------------------------------------- #
# Step 3 — Configure the Experiment
# --------------------------------------------------------------------------- #
def step_configure() -> None:
    _header(3, "Configure the Experiment", "configure")

    if S.target is None or not S.features:
        ui.advisory("info", "Define a target and features in Step 2 first.")
        _goto_button("Go to Step 2 — Define Problem", 1, key="goto_step2")
        return

    with ui.card("Split settings", help=explain.help_for("config.split")):
        c1, c2 = st.columns(2)
        _ensure_state("cfg_test_size", S.test_size)
        test_size = c1.slider(
            "Test proportion",
            min_value=0.1, max_value=0.5, step=0.05,
            key="cfg_test_size",
            help="Fraction of data held out as unseen test data.",
        )
        _ensure_state("cfg_seed", int(S.random_state if S.random_state is not None else 42))
        random_state = c2.number_input(
            "Random seed", min_value=0, step=1, key="cfg_seed",
            help="Makes the random split and model randomness reproducible.",
        )
        S.test_size = float(test_size)
        S.random_state = int(random_state)

        df = _get_df()
        profiles = st.session_state.get("_profiles") or _get_profiles(df)
        date_cols = date_columns(profiles)
        date_options = [None] + date_cols
        _ensure_state("cfg_date_col", S.date_column if S.date_column in date_options else None)
        if st.session_state["cfg_date_col"] not in date_options:
            st.session_state["cfg_date_col"] = None
        date_col = st.selectbox(
            "Date column for chronological split (optional)",
            options=date_options,
            key="cfg_date_col",
            format_func=lambda v: "None (random split)" if v is None else str(v),
            help="When set, data is sorted by date and the last fraction becomes "
            "the test set, respecting time order. The date column can be used for "
            "ordering even if it is not a predictor.",
        )
        S.date_column = date_col

        if S.date_column is None and date_cols:
            st.caption(
                "Date-like columns detected: " + ", ".join(str(c) for c in date_cols)
                + ". Select one above to split chronologically."
            )

        _ensure_state("cfg_scale", S.scale_numeric)
        S.scale_numeric = st.checkbox(
            "Scale numeric features",
            key="cfg_scale",
            help="Standardize numeric features (mean 0, std 1) so distance-based "
            "models (KNN, regularized linear) treat all features comparably.",
        )

    ui.raw_note(f"Current split method: {S.split_method}.")

    S.refresh_fingerprint()
    # Invalidate split/results whenever the data configuration changes.
    data_fp = S.data_fingerprint
    if st.session_state.get("_split_key") not in (None, data_fp):
        _clear_derived(clear_split=True)

    if st.button("Prepare train/test split", type="primary", key="cfg_prepare"):
        try:
            split = preprocess.split_data(
                df, S.target, S.features, S.test_size, S.random_state,
                chronological=(S.date_column is not None),
                date_column=S.date_column,
            )
            st.session_state.pop("_preprocessed", None)
            st.session_state.pop("_processed_key", None)
            state.reset_results(S)
            st.session_state["_split"] = split
            st.session_state["_split_key"] = data_fp
            ui.advisory("success",
                        f"Split prepared: {split.n_train} train / {split.n_test} "
                        f"test ({split.method}). Preprocessing is fit on training "
                        "data only.")
        except Exception as exc:  # noqa: BLE001
            ui.advisory("error", f"Could not prepare split: {exc}")

    split = st.session_state.get("_split")
    if split and st.session_state.get("_split_key") == data_fp:
        ui.section("Split summary", help=explain.help_for("config.summary"))
        ui.stat_row([
            ("Train rows", f"{split.n_train:,}"),
            ("Test rows", f"{split.n_test:,}"),
            ("Method", split.method),
        ], helps=[
            "Rows used to fit the models and preprocessing.",
            "Rows held out to estimate performance on unseen data.",
            "'random' shuffles rows; 'chronological' trains on the past and "
            "tests on the future.",
        ])


# --------------------------------------------------------------------------- #
# Step 4 — Select and Configure Models
# --------------------------------------------------------------------------- #
def _sync_presets(preset_choice: str) -> None:
    """Apply a preset by writing widget session keys *before* they render."""
    applied = st.session_state.setdefault("_preset_applied", {})
    for key in S.model_keys:
        spec = models.spec(key)
        if spec is None:
            continue
        desired = preset_choice if preset_choice in spec.presets else None
        if desired and applied.get(key) != desired:
            for pname, value in spec.presets[desired].items():
                st.session_state[f"param_{key}_{pname}"] = value
            applied[key] = desired
        elif not desired:
            applied.pop(key, None)


def _param_widget(spec, prefix: str) -> dict:
    """Render parameter widgets for a model spec and return current values."""
    params: dict = {}
    if not spec.param_schema:
        return params
    with st.expander(f"Configure {spec.label}", expanded=False):
        st.caption(spec.description)
        st.caption(spec.concept)
        for pname, pconf in spec.param_schema.items():
            key = f"{prefix}_{pname}"
            label = pconf.get("label", pname)
            widget = pconf["widget"]
            default = pconf["value"]
            _ensure_state(key, default)

            if widget == "slider":
                if not (pconf["min"] <= st.session_state[key] <= pconf["max"]):
                    st.session_state[key] = default
                kwargs = dict(
                    min_value=pconf["min"], max_value=pconf["max"],
                    step=pconf.get("step", 1), key=key, help=pconf.get("help", ""),
                )
                if pconf.get("format"):
                    kwargs["format"] = pconf["format"]
                params[pname] = st.slider(label, **kwargs)
            elif widget == "checkbox":
                params[pname] = st.checkbox(label, key=key, help=pconf.get("help", ""))
            elif widget == "selectbox":
                opts = pconf["options"]
                if st.session_state[key] not in opts:
                    st.session_state[key] = default
                params[pname] = st.selectbox(
                    label, options=opts, key=key, help=pconf.get("help", ""),
                    format_func=lambda v: "Unlimited" if v is None else str(v),
                )
    return params


def step_select() -> None:
    _header(4, "Select and Configure Models", "select")

    _ensure_state("select_models", S.model_keys or ["linear"])
    st.session_state["select_models"] = [
        k for k in st.session_state["select_models"] if models.spec(k) is not None
    ]
    model_keys = st.multiselect(
        "Models to compare",
        options=list(models.MODEL_SPECS.keys()),
        format_func=lambda k: models.MODEL_SPECS[k].label,
        key="select_models",
        help="All models train on the same data, features, and test split. "
        "Failures in one model do not abort the others.",
    )
    S.model_keys = model_keys
    ui.note(explain.help_for("select.models"))
    if not model_keys:
        ui.advisory("warning", "Select at least one model.")

    with ui.card("Presets"):
        st.caption(
            "A preset sets sensible starting parameters for every selected model. "
            "Choose one, then fine-tune individual parameters below."
        )
        _ensure_state("select_preset", "None (keep current)")
        preset_choice = st.selectbox(
            "Apply preset to all selected models",
            options=["None (keep current)"] + models.PRESET_ORDER,
            key="select_preset",
        )
        if preset_choice != "None (keep current)":
            st.caption(explain.preset_help(preset_choice))

    _sync_presets(preset_choice)

    params_map: dict[str, dict] = {}
    for key in S.model_keys:
        spec = models.spec(key)
        if spec is None:
            continue
        params_map[key] = _param_widget(spec, f"param_{key}")
    S.model_params = params_map

    _ensure_state("select_importance", S.compute_importance)
    S.compute_importance = st.checkbox(
        "Compute permutation feature importance",
        key="select_importance",
        help="Shuffles each feature on the test set to estimate how much the "
        "model relies on it. Adds computation time.",
    )

    S.refresh_fingerprint()


# --------------------------------------------------------------------------- #
# Step 5 — Train and Evaluate
# --------------------------------------------------------------------------- #
def _ensure_experiment() -> "tuple | None":
    """Ensure split and preprocessing match the current data configuration.

    Returns (split, processed, problem) or None (with a message shown).
    """
    df = _get_df()
    if df is None:
        ui.advisory("info", "Load a dataset in Step 1 first.")
        _goto_button("Go to Step 1 — Explore Data", 0, key="goto_step1c")
        return None
    if not S.target or not S.features:
        ui.advisory("info", "Define a target and features in Step 2 first.")
        _goto_button("Go to Step 2 — Define Problem", 1, key="goto_step2c")
        return None
    if not S.model_keys:
        ui.advisory("info", "Select at least one model in Step 4 first.")
        _goto_button("Go to Step 4 — Select Models", 3, key="goto_step4c")
        return None

    profiles = st.session_state.get("_profiles") or _get_profiles(df)
    problem = define_problem(df, S.target, S.features, profiles)
    S.problem = problem
    if not problem.target_is_numeric:
        ui.advisory("error", "The target must be numeric for regression.")
        return None

    S.refresh_fingerprint()
    data_fp = S.data_fingerprint

    if st.session_state.get("_split_key") != data_fp or "_split" not in st.session_state:
        try:
            split = preprocess.split_data(
                df, S.target, S.features, S.test_size, S.random_state,
                chronological=(S.date_column is not None),
                date_column=S.date_column,
            )
        except Exception as exc:  # noqa: BLE001
            ui.advisory("error", f"Could not prepare split: {exc}")
            return None
        # Drop stale preprocessing/results *before* storing the new split.
        _clear_derived(clear_split=False)
        st.session_state["_split"] = split
        st.session_state["_split_key"] = data_fp

    split = st.session_state["_split"]

    if st.session_state.get("_processed_key") != data_fp or "_preprocessed" not in st.session_state:
        try:
            processed = preprocess.preprocess(
                split, problem.numeric_features, problem.categorical_features,
                problem.date_features, scale_numeric=S.scale_numeric,
            )
        except Exception as exc:  # noqa: BLE001
            ui.advisory("error", f"Preprocessing failed: {exc}")
            return None
        # Drop any stale results *before* storing the freshly fitted objects.
        state.reset_results(S)
        st.session_state["_preprocessed"] = processed
        st.session_state["_processed_key"] = data_fp

    processed = st.session_state["_preprocessed"]
    return split, processed, problem


def step_train() -> None:
    _header(5, "Train and Evaluate", "train")

    ensured = _ensure_experiment()
    if ensured is None:
        return
    split, processed, problem = ensured

    ui.stat_row([
        ("Train rows", f"{split.n_train:,}"),
        ("Test rows", f"{split.n_test:,}"),
        ("Split", split.method),
        ("Scaling", "on" if S.scale_numeric else "off"),
    ], helps=[
        "Rows used to fit the models and preprocessing.",
        "Rows held out to estimate performance on unseen data.",
        "'random' shuffles rows; 'chronological' trains on the past and tests "
        "on the future.",
        "Whether numeric features were standardized before training.",
    ])
    ui.note(explain.help_for("config.summary"))

    if st.button("Train models", type="primary", key="train_run"):
        with st.spinner("Training models..."):
            results = models.train_all(
                S.model_keys, S.model_params,
                processed.X_train_arr, processed.y_train,
                processed.X_test_arr, processed.y_test,
                feature_names=processed.feature_names_out,
                compute_importance=S.compute_importance,
            )
        S.results = results
        S.results_fingerprint = S.fingerprint
        S.trained = True
        st.session_state["_results_for_export"] = results
        st.session_state["_processed_for_export"] = processed

    if not S.results:
        ui.advisory("info", "Click **Train models** to fit the selected models on "
                    "this configuration.")
        return

    if not S.results_match_current():
        ui.advisory("warning",
                    "Configuration changed since the last training run. Click "
                    "**Train models** to refresh results (stale results are not "
                    "shown).")
        return

    results = S.results
    display_names = preprocess.display_feature_names(processed.feature_names_out)
    target_std = float(processed.y_test.std()) if processed.y_test.size else 0.0

    ui.section("Model comparison (test set)",
               help=explain.help_for("train.comparison"))
    comp_df = metrics.comparison_frame(results)
    if not comp_df.empty:
        ui.data_table(comp_df)
        verdict = metrics.comparison_summary(results)
        if verdict:
            ui.note(verdict)
        _download_button(
            "Download model comparison (CSV)",
            comp_df.to_csv(index=False).encode("utf-8"),
            "model_comparison.csv", "text/csv", key="dl_comparison",
        )

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if failed:
        with ui.card("Models that failed"):
            for r in failed:
                ui.advisory("error", f"{r.label}: {r.error or 'Unknown error.'}")

    if not successful:
        ui.advisory("error", "No model trained successfully. Check the errors above.")
        return

    ui.section("Results by model")
    tabs = st.tabs([r.label for r in successful])
    for tab, r in zip(tabs, successful):
        with tab:
            ui.stat_row(
                [(f"Test {mname}", metrics.format_metric(mname, r.test_metrics.get(mname)))
                 for mname in METRIC_ORDER],
                helps=[explain.metric_help(mname) for mname in METRIC_ORDER],
            )

            with st.expander("Train vs. test metrics", expanded=True):
                ui.note(explain.help_for("train.metrics_table"))
                ui.data_table(
                    pd.DataFrame(metrics.metrics_table(r.train_metrics, r.test_metrics))
                )
                st.caption(metrics.explain_gap(
                    r.train_metrics.get("R2", 0.0), r.test_metrics.get("R2", 0.0)
                ))
                st.caption(metrics.r2_interpretation(r.test_metrics.get("R2", 0.0)))

            residuals = processed.y_test - r.y_test_pred
            summary = metrics.residual_summary(processed.y_test, r.y_test_pred)
            ui.note(explain.help_for("train.residual_summary"))
            st.caption(
                f"Residuals — mean {summary['mean']:.4f}, std {summary['std']:.4f}, "
                f"max |error| {summary['max_abs']:.4f}. "
                + metrics.residual_hint(summary, target_std)
            )

            colA, colB, colC = st.columns(3)
            with colA:
                ui.show_plot(plots.actual_vs_predicted(
                    processed.y_test, r.y_test_pred, label=f"Test — {r.label}"))
                ui.examples(
                    "points hug the dashed 45° line",
                    "a systematic curve or fan shape",
                )
            with colB:
                ui.show_plot(plots.residual_distribution(
                    residuals, label=f"Test — {r.label}"))
                ui.examples(
                    "centred on zero and roughly symmetric",
                    "a long tail, a second peak, or an offset centre",
                )
            with colC:
                ui.show_plot(plots.residual_vs_predicted(
                    r.y_test_pred, residuals, label=f"Test — {r.label}"))
                ui.examples(
                    "a shapeless band around the zero line",
                    "a funnel (spread grows with prediction) or a curve",
                )

            st.caption("Test predictions (first rows)")
            ui.note(explain.help_for("train.predictions"))
            n = min(25, len(processed.y_test))
            ui.data_table(pd.DataFrame({
                "actual": processed.y_test[:n],
                "predicted": r.y_test_pred[:n],
                "residual": (processed.y_test - r.y_test_pred)[:n],
            }))

            coef = models.coefficients(r, display_names)
            if coef is not None:
                with st.expander("Coefficients (linear model)"):
                    ui.note(explain.help_for("train.coefficients"))
                    st.caption(explain.concept("intercept"))
                    terms = coef["terms"]
                    coef_df = pd.DataFrame(terms, columns=["Term", "Coefficient"])
                    ui.data_table(coef_df)
                    st.write(f"Intercept: **{coef['intercept']:.4f}**")
                    st.caption(
                        "Coefficients are on the preprocessed (often scaled) "
                        "features, so compare their magnitudes, not raw units."
                        if S.scale_numeric else
                        "Coefficients are on the original feature units."
                    )


# --------------------------------------------------------------------------- #
# Step 6 — Interpret and Export
# --------------------------------------------------------------------------- #
def step_interpret() -> None:
    _header(6, "Interpret and Export", "interpret")

    results = st.session_state.get("_results_for_export")
    processed = st.session_state.get("_processed_for_export")
    if not results or processed is None:
        ui.advisory("info", "Train models in Step 5 first.")
        _goto_button("Go to Step 5 — Train & Evaluate", 4, key="goto_step5")
        return
    if not S.results_match_current():
        ui.advisory("warning",
                    "Configuration changed since the last training run. Retrain "
                    "in Step 5 to export results for the current configuration.")
        _goto_button("Go to Step 5 — Train & Evaluate", 4, key="goto_step5b")
        return

    split = st.session_state.get("_split")
    successful = [r for r in results if r.success]

    tab_cfg, tab_metrics, tab_pred, tab_imp = st.tabs(
        ["Configuration", "Metrics", "Predictions", "Importance"]
    )

    with tab_cfg:
        ui.section("Experiment configuration (for reproducibility)",
                   help=explain.help_for("interpret.config"))
        cfg = {
            "dataset": S.dataset_name,
            "source": S.dataset_source,
            "data_signature": S.data_signature,
            "rows": S.df_shape[0],
            "columns": S.df_shape[1],
            "target": S.target,
            "features": S.features,
            "test_size": S.test_size,
            "random_state": S.random_state,
            "split_method": split.method if split else None,
            "date_column": S.date_column,
            "scale_numeric": S.scale_numeric,
            "models": S.model_keys,
            "model_params": S.model_params,
            "compute_importance": S.compute_importance,
            "fingerprint": S.fingerprint,
        }
        with st.expander("Show raw configuration", expanded=False):
            st.json(cfg, expanded=False)
        _download_button(
            "Download configuration (JSON)",
            json.dumps(cfg, indent=2, default=str).encode("utf-8"),
            "config.json", "application/json", key="dl_config",
        )

        ui.section("Reading the results")
        st.markdown(explain.reading_results())

        ui.section("Concepts & glossary")
        for term, definition in explain.glossary():
            st.markdown(f"- **{term}** — {definition}")

    with tab_metrics:
        ui.section("Metrics & comparison (test set)",
                   help=explain.help_for("interpret.metrics"))
        mdf = metrics.comparison_frame(results)
        if not mdf.empty:
            ui.data_table(mdf)
            _download_button(
                "Download metrics (CSV)", mdf.to_csv(index=False).encode("utf-8"),
                "metrics.csv", "text/csv", key="dl_metrics",
            )
        else:
            ui.advisory("info", "No successful models to summarise.")

    with tab_pred:
        ui.section("Test predictions", help=explain.help_for("interpret.predictions"))
        if successful:
            sel = st.selectbox(
                "Model for predictions",
                options=[r.key for r in successful],
                format_func=lambda k: next(r.label for r in successful if r.key == k),
                key="interp_model",
            )
            chosen = next(r for r in successful if r.key == sel)
            pred_df = pd.DataFrame({
                "actual": processed.y_test, "predicted": chosen.y_test_pred,
                "residual": processed.y_test - chosen.y_test_pred,
            })
            ui.data_table(pred_df.head(50))
            _download_button(
                "Download predictions (CSV)",
                pred_df.to_csv(index=False).encode("utf-8"),
                f"predictions_{sel}.csv", "text/csv", key="dl_preds",
            )
        else:
            ui.advisory("info", "No successful models to export predictions for.")

    with tab_imp:
        ui.section("Feature importance (permutation, test set)",
                   help=explain.help_for("interpret.importance"))
        st.caption(explain.concept("permutation_importance"))
        display_names = preprocess.display_feature_names(processed.feature_names_out)
        shown = False
        for r in results:
            if r.success and r.importance is not None and len(r.importance):
                shown = True
                st.markdown(f"**{r.label}**")
                imp = importance.ImportanceResult(
                    names=list(display_names), values=r.importance)
                ui.show_plot(importance.bar_chart(imp))
                st.caption(importance.CAUSATION_NOTE)
        if not shown:
            ui.advisory("info", "No permutation importance available (disabled or "
                        "unsupported).")


# --------------------------------------------------------------------------- #
# App entry: persistent sidebar navigation (one step rendered at a time)
# --------------------------------------------------------------------------- #
def _progress_state() -> int:
    """Number of leading steps that are ready (0-6)."""
    checks = [
        bool(S.dataset_name),
        bool(S.target and S.features),
        st.session_state.get("_split") is not None,
        bool(S.model_keys),
        S.results_match_current(),
        S.results_match_current(),
    ]
    done = 0
    for ok in checks:
        if ok:
            done += 1
        else:
            break
    return done


def _sidebar_status() -> None:
    st.sidebar.markdown("#### Regression Explorer")
    st.sidebar.caption("Interactive regression workflow")
    st.sidebar.checkbox(
        "Explain outputs",
        key=ui.HELP_STATE_KEY,
        value=True,
        help="Show short 'how to read this' notes under each table, chart, and "
             "metric group.",
    )
    st.sidebar.divider()
    done = _progress_state()
    st.sidebar.progress(done / 6.0, text=f"{done} of 6 steps ready")
    st.sidebar.caption("Current setup")
    st.sidebar.write(f"**Dataset:** {S.dataset_name or '— none —'}")
    st.sidebar.write(f"**Target:** {S.target or '—'}")
    st.sidebar.write(f"**Features:** {len(S.features)} selected")
    models_txt = ", ".join(models.MODEL_SPECS[k].label for k in S.model_keys
                           if models.spec(k)) or "—"
    st.sidebar.write(f"**Models:** {models_txt}")
    st.sidebar.divider()


def _sidebar_nav() -> list[str]:
    labels = _nav_labels()
    st.sidebar.radio(
        "Workflow steps", labels,
        key="nav_step", label_visibility="visible",
    )
    st.sidebar.divider()
    if st.sidebar.button("Reset session", key="nav_reset", type="secondary"):
        st.session_state.clear()
        st.rerun()
    return labels


def _step_nav_buttons(labels: list[str], idx: int) -> None:
    st.divider()
    left, right, spacer = st.columns([1, 1, 4])
    if idx > 0:
        left.button("← Back", key="nav_back",
                    on_click=_set_nav, args=(labels[idx - 1],))
    if idx < len(labels) - 1:
        right.button("Next →", key="nav_next",
                     on_click=_set_nav, args=(labels[idx + 1],))


def main() -> None:
    steps = [
        step_explore, step_define, step_configure, step_select, step_train, step_interpret,
    ]
    _sidebar_status()
    labels = _sidebar_nav()

    current = st.session_state.get("nav_step", labels[0])
    if current not in labels:
        current = labels[0]
    idx = labels.index(current)

    steps[idx]()
    _step_nav_buttons(labels, idx)


if __name__ == "__main__":
    main()
