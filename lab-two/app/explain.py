"""Guided learning text for the whole app.

Single source of learner-facing copy: per-step guides, metric/concept
definitions, per-element "how to read this" notes, a glossary, and presets.
Kept concise so it helps without overwhelming.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepGuide:
    title: str
    what: str
    why: str
    observe: str
    pitfalls: str


STEP_GUIDES: dict[str, StepGuide] = {
    "explore": StepGuide(
        title="Explore Data",
        what=(
            "Pick a dataset and inspect it: dimensions, preview, data types, "
            "missing values, duplicates, descriptive statistics, distributions, "
            "and numeric correlations. Columns are classified as numeric, "
            "categorical, date-like, or identifier-like."
        ),
        why=(
            "You cannot model what you do not understand. Missing values, "
            "duplicates, constant columns, and identifier-like columns all "
            "affect which target and features are sensible."
        ),
        observe=(
            "Look at missing-value counts, highly skewed distributions, near-"
            "zero variance columns, and strong correlations between candidates "
            "for target and features (a possible leakage hint)."
        ),
        pitfalls=(
            "Treating an identifier (e.g. a row ID) as a feature can leak "
            "information; ignoring missingness can silently drop most of your "
            "data later."
        ),
    ),
    "define": StepGuide(
        title="Define the Regression Problem",
        what=(
            "Choose a numeric target (what you want to predict) and predictor "
            "features (what you use to predict it). The target must be a "
            "continuous quantity; features are the inputs."
        ),
        why=(
            "Regression learns a mapping from features to target. Picking a "
            "categorical target or a feature derived from the target produces "
            "meaningless or misleading results."
        ),
        observe=(
            "Constant targets, targets with many missing values, outliers, "
            "identifier-like features, and features that look derived from the "
            "target (for example ``CO2(tCO2)`` when predicting ``Usage_kWh``)."
        ),
        pitfalls=(
            "Target leakage: using a predictor that already contains information "
            "about the target inflates performance. We flag it and recommend "
            "exclusion, but never silently drop features you selected — you "
            "decide, and interpret with caution."
        ),
    ),
    "configure": StepGuide(
        title="Configure the Experiment",
        what=(
            "Choose train/test proportion, random seed, and whether to split "
            "randomly or chronologically (when a date column exists). The "
            "split simulates how the model will perform on unseen data."
        ),
        why=(
            "Training and evaluating on the same data overstates performance. "
            "Fitting preprocessors on the whole dataset before splitting leaks "
            "test information into training (preprocessing leakage)."
        ),
        observe=(
            "Chronological splitting respects time order (important for series "
            "like the steel industry data). The random seed makes results "
            "reproducible."
        ),
        pitfalls=(
            "Random splitting on time-ordered data can let the model 'see the "
            "future. Always fit imputers, scalers, and encoders on training "
            "data only."
        ),
    ),
    "select": StepGuide(
        title="Select and Configure Models",
        what=(
            "Pick one or more regression algorithms and adjust their parameters "
            "(polynomial degree, regularization strength, tree depth, number of "
            "neighbors, …). Presets offer starting points: baseline, "
            "regularized, complex."
        ),
        why=(
            "Different algorithms make different assumptions (linearity, "
            "smoothness, locality). Comparing them on identical data and split "
            "shows which assumptions fit your problem."
        ),
        observe=(
            "Preset purpose and the concept notes: degree trades flexibility "
            "against overfitting; alpha controls shrinkage; depth controls tree "
            "complexity; more neighbors smooth KNN predictions."
        ),
        pitfalls=(
            "Defaults are not magic. A complex model can score well in-sample "
            "but fail out-of-sample; always judge by test performance."
        ),
    ),
    "train": StepGuide(
        title="Train and Evaluate",
        what=(
            "Fit models on training data only and evaluate on the held-out test "
            "set. Metrics: MAE, MSE, RMSE, R² — for both training and test. "
            "Plots: actual vs. predicted, residual distribution, residuals vs. "
            "predicted, and a test-prediction table."
        ),
        why=(
            "Train/test separation tells you whether learning generalizes. "
            "Residual plots reveal patterns a single metric hides (bias, "
            "heteroscedasticity, outliers)."
        ),
        observe=(
            "Negative R² means the model predicts worse than the mean. A large "
            "train–test R² gap suggests overfitting. Points far from the 45° "
            "line are poorly predicted cases."
        ),
        pitfalls=(
            "Comparing models trained on different data or splits is unfair. "
            "Individual model failures (e.g. Lasso non-convergence) are caught "
            "per-model so the rest of the comparison still runs."
        ),
    ),
    "interpret": StepGuide(
        title="Interpret and Export",
        what=(
            "Read the results: coefficients/intercept for linear models, "
            "permutation importance for others, residual patterns, overfitting "
            "or underfitting signals. Download metrics, predictions, "
            "comparisons, and the full configuration for reproducibility."
        ),
        why=(
            "Regression is a tool for understanding, not just prediction. "
            "Recording the configuration lets you (or others) reproduce the "
            "experiment exactly."
        ),
        observe=(
            "Large coefficients need scaling context; permutation importance "
            "ranks predictive contribution but does not prove causation. "
            "Systematic residual structure (e.g. a curve) suggests the model "
            "misses a non-linear pattern."
        ),
        pitfalls=(
            "Do not confuse correlation with causation, and do not treat a "
            "single high test R² as proof — check residual plots and the "
            "train/test gap too."
        ),
    ),
}

METRIC_GUIDES = {
    "MAE": (
        "Mean Absolute Error: average absolute difference between predicted and "
        "actual. In the same units as the target; easy to interpret."
    ),
    "MSE": (
        "Mean Squared Error: average squared difference. Penalizes large errors "
        "more heavily than MAE."
    ),
    "RMSE": (
        "Root Mean Squared Error: MSE in the original units. Commonly reported "
        "because it matches the target's units."
    ),
    "R2": (
        "R² (coefficient of determination): fraction of target variance "
        "explained by the model. 1 = perfect, 0 = no better than predicting the "
        "mean. Negative means worse than the mean."
    ),
}

CONCEPT_NOTES = {
    "coefficient": (
        "In linear regression, a coefficient is the expected change in the "
        "target for a one-unit change in that feature, holding all other "
        "features fixed."
    ),
    "intercept": (
        "The predicted target when all features equal zero. Often not "
        "meaningful on its own if zero is outside the data range."
    ),
    "residual": (
        "Residual = actual − predicted. The part of the target the model did "
        "not capture. Patterns in residuals suggest model misspecification."
    ),
    "train_test_split": (
        "Train data is used to fit the model; held-out test data simulates "
        "unseen observations. Performance on test data estimates real-world "
        "accuracy."
    ),
    "regularization": (
        "Ridge (L2) and Lasso (L1) add a penalty for large coefficients, "
        "trading some bias for lower variance and better generalization."
    ),
    "overfitting_underfitting": (
        "Overfitting: learns training noise, high training performance, poor "
        "test performance. Underfitting: misses real patterns, poor on both."
    ),
    "permutation_importance": (
        "Shuffling one feature on the test set and measuring the drop in "
        "performance estimates how much the model relies on that feature. Not "
        "causal evidence."
    ),
}

# Plain-language definitions for the detected column kinds (Step 1 legend).
COLUMN_KIND_NOTES = {
    "numeric": "A number. Used directly, optionally standardized (scaled).",
    "categorical": "A label or category. One-hot encoded into 0/1 indicator columns.",
    "date": "A date or timestamp. Converted to elapsed time (seconds since 1970).",
    "identifier": (
        "Looks like an ID or index — usually carries no predictive signal and "
        "is a leakage risk."
    ),
}

# One short "what it is / how to read it" note per UI element, keyed by a stable
# id. The app looks these up with :func:`help_for` so all learner-facing text
# lives in one place.
ELEMENT_HELP = {
    # Step 1 — Explore
    "dataset.stats": (
        "Quick shape of the data: rows are observations, columns are variables. "
        "Duplicate rows and a high share of missing cells are worth addressing "
        "before modelling."
    ),
    "dataset.preview": (
        "The first 10 rows exactly as loaded. Each row is one observation and "
        "each header is a variable. Confirm that numbers were not read as text."
    ),
    "dataset.types": (
        "'Type' is how a column is stored (number, text, or date). 'Missing' "
        "counts empty cells and 'Unique' counts distinct values. A column with a "
        "single unique value is constant and carries no signal."
    ),
    "dataset.stats_table": (
        "Summary statistics for numeric columns: count, mean, standard deviation "
        "(std), min, the 25/50/75th percentiles, and max. A large gap between "
        "mean and median (50%) suggests a skewed distribution."
    ),
    "dataset.distribution": (
        "How often each value of the selected column occurs. Look at the shape: "
        "symmetric, skewed (long tail), or multi-peaked. Isolated bars far from "
        "the bulk are outliers."
    ),
    "dataset.correlation": (
        "Pairwise correlation r between numeric columns, from -1 (strong inverse) "
        "to +1 (strong direct); near 0 means little linear relationship. A "
        "feature strongly correlated with your target can signal leakage."
    ),
    "column.kinds": "How each column is classified and preprocessed.",
    # Step 2 — Define
    "define.target": (
        "The continuous number you want to predict (e.g. charges, price, energy "
        "use). It must be numeric — categorical targets belong to classification."
    ),
    "define.features": (
        "The inputs the model may use. More is not always better: irrelevant or "
        "leaking features can hurt. Nothing is removed automatically."
    ),
    "define.type_counts": (
        "How the selected features will be handled: numeric values are used "
        "directly, categorical values are one-hot encoded, dates become elapsed "
        "time, and identifier-like columns are flagged as likely uninformative."
    ),
    "define.flags": (
        "Automatic checks on your target and features. They are advisory — the "
        "app never drops a feature you selected. Read each and decide whether to "
        "act."
    ),
    # Step 3 — Configure
    "config.split": (
        "The test proportion is the share of rows held out for evaluation. The "
        "seed makes the random split reproducible. Chronological splitting trains "
        "on the past and tests on the future. Scaling standardizes numeric "
        "features so distance-based and regularized models treat them comparably."
    ),
    "config.summary": (
        "Your prepared split: training vs testing rows and whether the split was "
        "random or chronological. Preprocessing is fit on training rows only, so "
        "the test rows stay unseen."
    ),
    # Step 4 — Select
    "select.models": (
        "Pick one or more algorithms to compare on identical data, features, and "
        "split. Each makes different assumptions (linearity, smoothness, "
        "locality); judging them on the same test set shows which fits your data. "
        "A failure in one model never stops the others."
    ),
    # Step 5 — Train & evaluate
    "train.comparison": (
        "One row per model, scored on the held-out test set. Lower MAE/RMSE is "
        "better; higher test R² is better. The 'Gap' column (train R² − test R²) "
        "flags overfitting when large and positive."
    ),
    "train.metrics_table": (
        "Your model's scores on training and test data side by side. Ideally the "
        "two are close; a much higher training score means the model memorized "
        "rather than learned."
    ),
    "train.residual_summary": (
        "Residual = actual − predicted. A mean near zero means little systematic "
        "bias; a spread comparable to the target means noisy predictions."
    ),
    "train.plot.actual": (
        "Each point is one test observation: x is the true value, y is the "
        "prediction. Points on the dashed line are perfect. A symmetric cloud is "
        "expected; systematic curvature means the model misses structure."
    ),
    "train.plot.resid_dist": (
        "Histogram of residuals. Ideally centred on zero and roughly symmetric. A "
        "long tail or a second peak can indicate outliers or a missing pattern."
    ),
    "train.plot.resid_pred": (
        "Residuals against predicted values. Ideally a shapeless band around the "
        "dashed zero line. A funnel (spread growing with the prediction) means the "
        "model is less reliable for larger values."
    ),
    "train.predictions": (
        "The first test predictions with each residual. Large residuals are the "
        "cases the model gets most wrong — inspect them for data issues."
    ),
    "train.coefficients": (
        "For linear models, each coefficient is the change in the target for a "
        "one-unit change in that feature, holding the others fixed. With scaling "
        "on, compare magnitudes rather than raw units."
    ),
    # Step 6 — Interpret & export
    "interpret.config": (
        "The exact settings that produced these results. Save it to reproduce the "
        "experiment later."
    ),
    "interpret.metrics": "Test-set metrics for every trained model.",
    "interpret.predictions": (
        "Actual vs predicted values on the test set for the selected model."
    ),
    "interpret.importance": (
        "How much test performance drops when each feature is shuffled. Longer "
        "bars mean the model relies on that feature more — a useful hint, not "
        "proof of causation."
    ),
}

READING_RESULTS = (
    "- **Negative R²**: the model predicts worse than the mean of the target; "
    "consider simpler features, more data, or a different algorithm.\n"
    "- **Large train–test R² gap**: likely overfitting; try regularization "
    "(Ridge/Lasso), lower tree depth, fewer polynomial degrees, or more data.\n"
    "- **Close train and test R² with low absolute values**: likely underfitting; "
    "try a more flexible model or additional features.\n"
    "- **Residuals with a clear pattern** (curve, funnel): the model is missing "
    "structure; consider polynomial features or transforming inputs."
)

# (term, definition) pairs for the Concepts & Glossary section.
GLOSSARY = (
    ("Target", "The continuous quantity being predicted."),
    ("Feature", "An input variable used to predict the target."),
    ("Train / test split",
     "Training data fits the model; held-out test data estimates performance on "
     "unseen observations."),
    ("MAE", METRIC_GUIDES["MAE"]),
    ("MSE", METRIC_GUIDES["MSE"]),
    ("RMSE", METRIC_GUIDES["RMSE"]),
    ("R²", METRIC_GUIDES["R2"]),
    ("Residual", CONCEPT_NOTES["residual"]),
    ("Coefficient", CONCEPT_NOTES["coefficient"]),
    ("Intercept", CONCEPT_NOTES["intercept"]),
    ("Scaling",
     "Transforming numeric features to a common scale (mean 0, std 1) so "
     "distance-based and regularized models are not dominated by large units."),
    ("One-hot encoding",
     "Turning a categorical column into several 0/1 columns, one per category."),
    ("Regularization", CONCEPT_NOTES["regularization"]),
    ("Overfitting / underfitting", CONCEPT_NOTES["overfitting_underfitting"]),
    ("Leakage",
     "When a feature (or the way data was prepared) contains information about "
     "the target that would not be available when predicting new data, inflating "
     "scores."),
    ("Permutation importance", CONCEPT_NOTES["permutation_importance"]),
    ("Skew",
     "Asymmetry of a distribution. A long right tail (mean > median) is positive "
     "skew; the opposite is negative skew."),
)


def guide(step: str) -> StepGuide:
    return STEP_GUIDES[step]


def metric_help(key: str) -> str:
    return METRIC_GUIDES.get(key, "")


def concept(key: str) -> str:
    return CONCEPT_NOTES.get(key, "")


PRESET_NOTES = {
    "baseline": (
        "Sensible defaults that work reasonably across many problems. A good "
        "first run to establish a reference score."
    ),
    "regularized": (
        "Stronger regularization (larger penalty). Useful when features are "
        "many or correlated, or when the baseline overfits."
    ),
    "complex": (
        "A more flexible model (higher polynomial degree, deeper trees, more "
        "trees). Can capture structure but overfits more easily."
    ),
}


def preset_help(name: str) -> str:
    return PRESET_NOTES.get(name, "")


def help_for(key: str) -> str:
    """Short 'how to read this' note for a UI element id."""
    return ELEMENT_HELP.get(key, "")


def column_kind(kind: str) -> str:
    """Plain-language definition of a detected column kind."""
    return COLUMN_KIND_NOTES.get(kind, "")


def glossary() -> list[tuple[str, str]]:
    """Ordered (term, definition) pairs for the Concepts & Glossary section."""
    return list(GLOSSARY)


def reading_results() -> str:
    """Markdown bullets on how to read metric and residual results."""
    return READING_RESULTS

