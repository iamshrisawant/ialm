"""Dataset discovery, loading, and validation.

Datasets are discovered dynamically from ``lab-two/data/`` (no hard-coded
filenames). Uploaded CSVs are supported and treated as untrusted input: they
are never executed and never transmitted anywhere.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from . import ui
except ImportError:  # run as a top-level module
    import ui

# Project root is two levels above this file: <root>/lab-two/app/data_io.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "lab-two" / "data"

MAX_UPLOAD_MB = 50
_CACHE_TTL_SECONDS = 30

# Matches measurement descriptors whose whole value is a number, optionally
# prefixed by a comparison sign and/or suffixed by a range, e.g. "< 112",
# "≥ 227.944", "1.019 - 1.021", "138 - 143". The capture group is required by
# pandas' ``Series.str.extract``. Anchors prevent hijacking free text or dates
# such as "3 bedrooms" or "01/01/2018 00:00".
_MEASURE_RE = re.compile(
    r"^\s*(?:[<>]=?|≤|≥|±)?\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+))"
    r"(?:\s*[-–]\s*[-+]?(?:\d+\.?\d*|\.\d+))?"
    r"\s*$"
)



@dataclass
class LoadResult:
    """Outcome of loading a dataset."""

    df: Optional[pd.DataFrame]
    source: str  # "disk" | "upload" | "none"
    name: str
    error: Optional[str] = None
    warnings: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.df is not None and self.error is None


def discover_csv_files() -> list[Path]:
    """Return sorted CSV paths under the data directory, or [] if missing."""
    if not DATA_DIR.is_dir():
        return []
    return sorted(p for p in DATA_DIR.glob("*.csv") if p.is_file())


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace/BOM from column names and de-duplicate them."""
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for col in df.columns:
        name = str(col).replace("\ufeff", "").strip()
        if not name:
            name = "column"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    df.columns = new_cols
    return df


def _read_csv(source) -> pd.DataFrame:
    """Read a CSV robustly: tolerate BOMs and sniff common delimiters.

    ``source`` may be a filesystem path or a file-like object. UTF-8 (with or
    without BOM) is assumed; the delimiter is detected among ``, ; \\t |`` so
    semicolon- and tab-separated exports load correctly.
    """
    if hasattr(source, "read"):
        raw = source.read()
    else:
        raw = Path(source).read_bytes()

    text = raw.decode("utf-8-sig", errors="replace")
    sample = "\n".join(text.splitlines()[:20])
    sep = ","
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:  # noqa: BLE001 - not enough/regular data to sniff
        sep = ","

    return pd.read_csv(io.StringIO(text), sep=sep, skipinitialspace=True)


def _coerce_numeric(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str], list[str]]:
    """Attempt to coerce text columns to numeric where sensible.

    Strategy per text column:
    1. parse values directly (``"3.5"`` → 3.5);
    2. otherwise extract a measurement value from descriptors such as
       ``"< 112"`` or ``"1.019 - 1.021"`` (common in exported course datasets).

    A column is only converted when at least 80% of its non-null values parse.
    Unconvertible values become NaN so malformed data is surfaced, not hidden.

    Returns (frame, directly_coerced, extracted_coerced).
    """
    direct: list[str] = []
    extracted: list[str] = []
    for col in df.columns:
        # pandas 3.0 uses a dedicated string dtype (not ``object``), so decide
        # by "is it already numeric/datetime?" rather than an object check.
        if (pd.api.types.is_numeric_dtype(df[col])
                or pd.api.types.is_datetime64_any_dtype(df[col])):
            continue
        non_null_original = int(df[col].notna().sum())
        if non_null_original == 0:
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() / non_null_original >= 0.8:
            df[col] = converted
            direct.append(col)
            continue

        pulled = df[col].astype("string").str.extract(_MEASURE_RE)[0]
        converted = pd.to_numeric(pulled, errors="coerce")
        if converted.notna().sum() / non_null_original >= 0.8:
            df[col] = converted
            extracted.append(col)
    return df, direct, extracted


def load_csv_from_bytes(raw: bytes, name: str) -> LoadResult:
    """Load a CSV from raw bytes (uploaded file). Treated as untrusted input."""
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        return LoadResult(
            None, "upload", name,
            error=f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.",
        )
    try:
        df = _read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - surface any parse failure gracefully
        return LoadResult(None, "upload", name, error=f"Could not parse CSV: {exc}")
    return _validate(df, "upload", name)


def load_csv_from_disk(path: Path) -> LoadResult:
    """Load a CSV from disk."""
    try:
        df = _read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return LoadResult(None, "disk", path.name, error=f"Could not read '{path.name}': {exc}")
    return _validate(df, "disk", path.name)


def _validate(df: pd.DataFrame, source: str, name: str) -> LoadResult:
    warnings: list[str] = []

    if df is None or df.empty:
        return LoadResult(None, source, name, error="The dataset is empty.")

    df = _clean_columns(df)
    n_rows, n_cols = df.shape
    if n_cols == 0:
        return LoadResult(None, source, name, error="The dataset has no columns.")
    if n_rows == 0:
        return LoadResult(None, source, name, error="The dataset has no rows.")
    if n_cols < 2:
        return LoadResult(
            None, source, name,
            error="Need at least two columns (a target and a predictor feature).",
        )

    # Malformed-data handling: try coercing object columns to numeric where
    # most values parse, otherwise leave as-is so the student can inspect.
    _, direct, extracted = _coerce_numeric(df)
    if direct:
        warnings.append(
            "Coerced columns to numeric: " + ", ".join(direct) +
            ". Unparseable values became missing values."
        )
    if extracted:
        warnings.append(
            "Extracted numbers from descriptive text in: " + ", ".join(extracted) +
            " (e.g. '< 112' → 112, '1.019 - 1.021' → 1.019). Verify this is "
            "appropriate for your analysis."
        )

    missing_total = int(df.isna().sum().sum())
    if missing_total:
        warnings.append(
            f"Contains {missing_total} missing value(s) across "
            f"{int((df.isna().any(axis=1)).sum())} row(s)."
        )

    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        warnings.append(f"Contains {dup_rows} duplicate row(s).")

    # Suitability: at least one numeric column is needed for regression.
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return LoadResult(
            None, source, name,
            error=(
                "No numeric columns found, so regression is not possible. The "
                "file may use only text/descriptive values, a metadata layout, "
                "or be separated differently. Try a dataset with numeric "
                "columns, or upload a clean CSV."
            ),
        )

    return LoadResult(df, source, name, warnings=tuple(warnings))


def data_signature(df: pd.DataFrame) -> str:
    """A stable signature of the data values (shape + content hash).

    Used to detect when a different dataset/upload is loaded even if it shares
    a filename, so stale results are never shown as current.
    """
    try:
        h = int(pd.util.hash_pandas_object(df, index=False).sum())
        return f"{df.shape[0]}x{df.shape[1]}:{h & 0xFFFFFFFFFFFFFFFF:016x}"
    except Exception:  # noqa: BLE001
        return f"{df.shape[0]}x{df.shape[1]}"


@st.cache_data(show_spinner=False, ttl=_CACHE_TTL_SECONDS)
def cached_disk_names() -> Tuple[str, ...]:
    """Cached list of on-disk CSV names (refreshed periodically)."""
    return tuple(p.name for p in discover_csv_files())


@st.cache_data(show_spinner=False)
def _cached_load_disk(path_str: str, mtime: float) -> LoadResult:
    return load_csv_from_disk(Path(path_str))


@st.cache_data(show_spinner=False)
def _cached_load_bytes(raw: bytes, name: str) -> LoadResult:
    return load_csv_from_bytes(raw, name)


def load_disk_cached(path: Path) -> LoadResult:
    """Load a disk CSV, cached by path and modification time."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return _cached_load_disk(str(path), mtime)


def load_bytes_cached(raw: bytes, name: str) -> LoadResult:
    """Load uploaded bytes, cached by content."""
    return _cached_load_bytes(raw, name)


def render_dataset_selector(upload_key: str = "uploader") -> LoadResult:
    """Render the dataset source selector and return a LoadResult."""
    ui.raw_note(
        "Select a CSV from lab-two/data/ or upload your own. Datasets are "
        "discovered dynamically; no filenames are hard-coded."
    )

    disk_names = cached_disk_names()
    source = st.radio(
        "Source",
        options=("From data/ folder", "Upload a CSV"),
        horizontal=True,
        key="ds_source",
    )

    result: Optional[LoadResult] = None

    if source == "From data/ folder":
        if not disk_names:
            ui.advisory("info", "No CSV files found in lab-two/data/.")
            return LoadResult(None, "disk", "", error="No datasets available.")
        choice = st.selectbox("Dataset", options=disk_names, key="ds_disk_choice")
        if choice:
            result = load_disk_cached(DATA_DIR / choice)
    else:
        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key=upload_key,
            help=f"Treated as untrusted input; never executed or transmitted "
                 f"(limit {MAX_UPLOAD_MB} MB).",
        )
        if uploaded is not None:
            result = load_bytes_cached(uploaded.getvalue(), uploaded.name)

    if result is None:
        return LoadResult(None, "none", "", error="No dataset selected.")

    if not result.ok:
        ui.advisory("error", result.error)
        return result

    for w in result.warnings:
        ui.advisory("warning", w)

    ui.advisory(
        "success",
        f"Loaded {result.name} ({result.df.shape[0]} rows × "
        f"{result.df.shape[1]} columns) from {result.source}.",
    )
    return result
