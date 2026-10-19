"""End-to-end smoke tests that execute the real Streamlit script via AppTest.

These run the actual ``lab-two/app/main.py`` and fail on any uncaught exception,
covering the sidebar navigation and the wiring that unit tests miss.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "lab-two" / "app" / "main.py"

warnings.filterwarnings("ignore")

STEPS = [
    "1. Explore Data",
    "2. Define Problem",
    "3. Configure Experiment",
    "4. Select Models",
    "5. Train & Evaluate",
    "6. Interpret & Export",
]


def _assert_clean(at: AppTest, stage: str) -> None:
    errors = [e.value for e in at.exception]
    assert not errors, f"{stage}: {errors}"


def _run() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=300).run()


def _go(at: AppTest, step_index: int) -> None:
    at.radio(key="nav_step").set_value(STEPS[step_index])
    at.run()


def test_initial_render_has_navigation():
    at = _run()
    _assert_clean(at, "initial render")
    assert list(at.radio(key="nav_step").options) == STEPS
    # Top-level navigation is the sidebar; any tabs are within-step detail only.
    tab_labels = [t.label for t in at.tabs]
    assert not any(s in tab_labels for s in STEPS)


def test_next_button_advances_step():
    at = _run()
    at.button(key="nav_next").click().run()
    _assert_clean(at, "next")
    assert at.radio(key="nav_step").value == STEPS[1]


def test_step_guide_is_a_collapsed_expander():
    at = _run()
    labels = [e.label for e in at.expander]
    assert "How this step works" in labels


def test_explain_toggle_controls_inline_notes():
    at = _run()
    assert at.checkbox(key="show_help").value is True
    # Dataset shape explanation is visible by default.
    notes = "\n".join(m.value for m in at.markdown)
    assert "rows are observations" in notes

    at.checkbox(key="show_help").set_value(False).run()
    _assert_clean(at, "help off")
    notes = "\n".join(m.value for m in at.markdown)
    assert "rows are observations" not in notes


def test_full_workflow_insurance():
    at = _run()
    _go(at, 0)
    at.selectbox(key="ds_disk_choice").select("insurance.csv")
    at.run(); _assert_clean(at, "load")

    _go(at, 1)
    at.selectbox(key="define_target").select("charges")
    feats = [c for c in ("age", "sex", "bmi", "children", "smoker", "region")
             if c in at.multiselect(key="define_features").options]
    at.multiselect(key="define_features").set_value(feats)
    at.run(); _assert_clean(at, "define")

    _go(at, 2)
    at.button(key="cfg_prepare").click()
    at.run(); _assert_clean(at, "prepare")

    _go(at, 3)
    at.multiselect(key="select_models").set_value(["linear", "ridge", "tree", "forest"])
    at.run(); _assert_clean(at, "select")

    _go(at, 4)
    at.button(key="train_run").click()
    at.run(); _assert_clean(at, "train")

    assert at.dataframe, "expected result tables after training"
    # The comparison table is formatted and exposes the train-test gap column.
    assert any("Gap" in getattr(df.value, "columns", []) for df in at.dataframe)

    _go(at, 5)
    _assert_clean(at, "interpret")
    assert at.dataframe


def test_steel_leakage_flag_and_coefficients():
    at = _run()
    _go(at, 0)
    at.selectbox(key="ds_disk_choice").select("steel-industry.csv")
    at.run(); _assert_clean(at, "load steel")

    _go(at, 1)
    at.selectbox(key="define_target").select("Usage_kWh")
    at.multiselect(key="define_features").set_value(
        ["NSM", "Lagging_Current_Power_Factor", "CO2(tCO2)"]
    )
    at.run(); _assert_clean(at, "define steel")

    md = "\n".join(m.value for m in at.markdown)
    assert "CO2(tCO2)" in md  # BOM decoded + leakage advisory rendered

    _go(at, 2)
    at.button(key="cfg_prepare").click(); at.run(); _assert_clean(at, "prepare")
    _go(at, 3)
    at.multiselect(key="select_models").set_value(
        ["simple_linear", "polynomial", "ridge"]
    )
    at.checkbox(key="select_importance").set_value(True)
    at.run(); _assert_clean(at, "select")
    _go(at, 4)
    at.button(key="train_run").click(); at.run(); _assert_clean(at, "train")
    assert at.dataframe

    # Displayed feature names are cleaned of transformer prefixes.
    joined = "\n".join(str(df.value) for df in at.dataframe)
    assert "num__" not in joined and "cat__" not in joined


def test_switching_dataset_resets_target():
    at = _run()
    _go(at, 0)
    at.selectbox(key="ds_disk_choice").select("insurance.csv"); at.run()
    _go(at, 1)
    at.selectbox(key="define_target").select("charges")
    at.multiselect(key="define_features").set_value(["age", "bmi"]); at.run()
    _assert_clean(at, "define insurance")

    _go(at, 0)
    at.selectbox(key="ds_disk_choice").select("steel-industry.csv")
    at.run(); _assert_clean(at, "switch dataset")

    _go(at, 1)
    assert at.selectbox(key="define_target").value is None


def test_chronological_split_ui_path():
    at = _run()
    _go(at, 0)
    at.selectbox(key="ds_disk_choice").select("steel-industry.csv"); at.run()
    _go(at, 1)
    at.selectbox(key="define_target").select("Usage_kWh")
    at.multiselect(key="define_features").set_value(
        ["NSM", "Lagging_Current_Power_Factor"]
    ); at.run()
    _go(at, 2)
    date_opts = at.selectbox(key="cfg_date_col").options
    assert date_opts  # a date column option exists
    at.selectbox(key="cfg_date_col").select("date"); at.run()
    at.button(key="cfg_prepare").click(); at.run()
    _go(at, 3)
    at.multiselect(key="select_models").set_value(["knn", "tree"]); at.run()
    _go(at, 4)
    at.button(key="train_run").click(); at.run()
    _assert_clean(at, "chronological flow")


def test_ckd_metadata_dataset_loads_after_coercion():
    at = _run()
    _go(at, 0)
    at.selectbox(key="ds_disk_choice").select("ckd.csv")
    at.run()
    _assert_clean(at, "load ckd")
    # Metadata/range values are extracted, so a numeric target becomes available.
    _go(at, 1)
    options = at.selectbox(key="define_target").options
    assert options, "expected at least one numeric target after coercion"
