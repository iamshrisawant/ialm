"""Session-state helpers with configuration fingerprinting.

Separate fingerprints track the *data* configuration (dataset signature,
target, features, split settings) and the *model* configuration (selected
models and their parameters). Results are tagged with the combined fingerprint
that produced them, so stale results are never presented as belonging to the
current configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import streamlit as st


def _fingerprint(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class AppState:
    dataset_name: str = ""
    dataset_source: str = ""
    data_signature: str = ""
    df_shape: tuple[int, int] = (0, 0)
    target: Optional[str] = None
    features: list[str] = field(default_factory=list)
    problem: Optional[Any] = None  # ProblemDefinition
    test_size: float = 0.2
    random_state: Optional[int] = 42
    date_column: Optional[str] = None
    scale_numeric: bool = True
    model_keys: list[str] = field(default_factory=list)
    model_params: dict[str, dict] = field(default_factory=dict)
    compute_importance: bool = True
    fingerprint: str = ""
    data_fingerprint: str = ""
    results: list[Any] = field(default_factory=list)  # list[ModelResult]
    results_fingerprint: str = ""
    trained: bool = False

    # -- fingerprints ------------------------------------------------------- #
    @property
    def split_method(self) -> str:
        return "chronological" if self.date_column else "random"

    def data_config(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "data_signature": self.data_signature,
            "target": self.target,
            "features": sorted(self.features),
            "test_size": self.test_size,
            "random_state": self.random_state,
            "split_method": self.split_method,
            "date_column": self.date_column,
            "scale_numeric": self.scale_numeric,
        }

    def model_config(self) -> dict:
        return {
            "model_keys": sorted(self.model_keys),
            "model_params": self.model_params,
            "compute_importance": self.compute_importance,
        }

    def config_dict(self) -> dict:
        cfg = self.data_config()
        cfg.update(self.model_config())
        return cfg

    def refresh_fingerprint(self) -> None:
        self.data_fingerprint = _fingerprint(self.data_config())
        self.fingerprint = _fingerprint(self.config_dict())

    def results_match_current(self) -> bool:
        return self.trained and self.results_fingerprint == self.fingerprint


def get_state() -> AppState:
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()
    return st.session_state["app_state"]


def reset_results(state: AppState) -> None:
    """Clear results when configuration changes (avoid stale results)."""
    state.results = []
    state.results_fingerprint = ""
    state.trained = False
