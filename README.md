# Regression Explorer

An interactive lab for **learning how regression works, starting from a raw
dataset**. You pick a CSV, inspect it, choose what to predict, train several
regression models, and read the diagnostics — all in one guided flow.

No coding is required to use it. It is built for students who want to *see* how
choices about data, features, and algorithms change results.

---

## Quick start

```bash
# From this folder (where ialm/ lives)
source ialm/bin/activate
python -m pip install -r requirements.txt
streamlit run lab-two/app/main.py
```

Open `http://localhost:8501`. Datasets are discovered automatically from
`lab-two/data/` — drop in any CSV and it appears. You can also upload a file.

---

## What is regression? (30-second version)

Regression learns a relationship between **inputs (features)** and a
**continuous output (the target)** so it can predict the target for new rows.
For example: predict a person's medical `charges` from `age`, `bmi`, `smoker`.

Good regression is not just "get a high score". It is about:

- choosing a sensible target and features,
- splitting data so evaluation is honest,
- checking whether the model generalises (vs. memorises),
- and looking at *where* and *why* it is wrong.

This tool walks you through exactly those steps.

---

## How to use it (a guided walkthrough)

Navigation is the **sidebar on the left**: pick a step, or use **Next →** at the
bottom of each page. Only one step shows at a time, and your choices are kept as
you move around. The sidebar also shows your current setup and a readiness bar.

Each step has a **"How this step works"** panel you can open if you want the
concept explained. Every table, chart, and metric group also carries a short
**"how to read this"** note, and metric cards have hover tooltips. If you prefer
a cleaner screen, untick **"Explain outputs"** in the sidebar. A **Concepts &
glossary** section in the final step defines every term used (R², residual,
leakage, one-hot encoding, and more).

### 1. Explore Data
Choose a dataset. Look at:
- **Rows / columns / missing cells / duplicates** (top metrics),
- the **Preview**, **Types & missing**, and **Statistics** tabs,
- **Distributions & correlations** — is the target skewed? Are two variables
  suspiciously correlated?

*Watch out for:* identifier columns (like row IDs), heavy missingness, and
variables that are almost duplicates of each other.

### 2. Define Problem
Pick a **numeric target** and the **features** you will predict it from.
The tool flags things worth a second look:
- constant targets (nothing to learn),
- missing values in the target,
- outliers,
- **identifier-like** features,
- **potential leakage** — a feature that may already contain the answer.

Nothing is removed automatically; the flags are advice so you can decide.

### 3. Configure Experiment
Set the **test proportion** and a **random seed** (so results are reproducible).
If a date column exists, you can split **chronologically** (train on the past,
test on the future) instead of randomly.

*Why it matters:* hitting a high score on the same data you trained on proves
nothing. The test set simulates unseen data.

### 4. Select Models
Choose one or more algorithms and compare them on the same data and split:
Simple/Multiple Linear, Polynomial, Ridge, Lasso, Decision Tree, Random Forest,
and K-Nearest Neighbors. **Presets** give sensible starting parameters
(baseline / regularized / complex); you can then fine-tune individual settings.

### 5. Train & Evaluate
Click **Train models**. You get:
- a **comparison table** of all models (MAE, MSE, RMSE, R²),
- one **tab per model** with train-vs-test metrics and three diagnostic plots:
  - **Actual vs. Predicted** — points on the dashed line are well predicted,
  - **Residual distribution** — should be centred on zero,
  - **Residuals vs. Predicted** — look for funnels or curves,
- coefficients for linear models.

One model failing (e.g. a Lasso convergence issue) never stops the others.

### 6. Interpret & Export
Read the findings and, if you like, download everything for reproducibility:
configuration (JSON), metrics and predictions (CSV), and feature importance.
**Permutation importance** ranks how much the model relies on each feature — it
is a hint, not proof of causation.

---

## Two suggested exercises

1. **Baseline to better** — load `insurance.csv`, predict `charges` from
   `age, sex, bmi, children, smoker, region`. Compare Linear, Polynomial, and
   Decision Tree. Which overfits? Does scaling change KNN?
2. **Leakage** — load `steel-industry.csv`, predict `Usage_kWh`, and include
   `CO2(tCO2)` as a feature. The tool flags it as possible leakage. What happens
   to the score when you remove it? (This mirrors the notebook in `lab-two/`.)

---

## Reading the results — a cheat sheet

- **Negative R²** — worse than always predicting the mean.
- **Large train–test gap** — overfitting (too flexible); regularize, simplify.
- **Low scores on both** — underfitting (too simple); add features or flexibility.
- **Residuals form a curve or funnel** — the model is missing structure;
  consider polynomial features or transforming inputs.

---

## Datasets and data handling

Place CSVs in `lab-two/data/`; they appear in the selector. Existing course
datasets (`insurance.csv`, `concrete-data.csv`, `steel-industry.csv`, …) work
as-is. Loading is tolerant: BOMs are stripped, common delimiters
(`, ; \t |`) are detected, and measurement-style text such as `"< 112"` or
`"1.019 - 1.021"` is converted to numbers with a visible warning.

Preprocessors (imputers, scalers, encoders) are fitted on **training data only**,
so the test set stays unseen. Uploaded files are treated as untrusted input:
never executed and never transmitted anywhere.

---

## Repository layout

```
.streamlit/config.toml    # light theme + upload limit
ialm/                     # existing Python virtual environment (not committed)
lab-two/
├── data/                 # CSV datasets (discovered dynamically)
└── app/                  # application code
    ├── main.py           # Streamlit app (6 steps, sidebar navigation)
    ├── ui.py             # shared design system + interpretation helpers
    ├── data_io.py        # dataset discovery, loading, validation
    ├── column_types.py   # column typing + leakage-aware problem definition
    ├── preprocess.py     # train-only pipelines, split logic, name formatting
    ├── models.py         # model registry, presets, failure-isolated training
    ├── metrics.py        # MAE/MSE/RMSE/R², comparison tables, explanations
    ├── plots.py          # diagnostics charts (shared styling)
    ├── importance.py     # permutation feature importance
    ├── state.py          # session state + configuration fingerprinting
    └── explain.py        # all learner-facing text (guides, help, glossary)
tests/                    # synthetic-data unit + Streamlit smoke tests
requirements.txt
```

`lab-one-eda/` and `lab-two/lab-two.ipynb` are preserved untouched.

## Tests

Tests use small synthetic datasets only — no private data, no network:

```bash
source ialm/bin/activate
python -m pytest tests/
```

## Deploy (Streamlit Community Cloud)

1. Push the repository to GitHub (excluding `ialm/`).
2. Create a new app at streamlit.io/cloud from the repository.
3. Set the main file path to `lab-two/app/main.py`. No secrets are required.
4. Dependencies install from `requirements.txt`; the free tier is sufficient.
