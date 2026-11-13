"""Session state helpers (the only module besides components/pages that imports Streamlit).

st.session_state keeps data alive between clicks:
  datasets        name -> original DataFrame (never modified)
  cleaned         name -> cleaned copy (created only after the user confirms)
  results         saved analysis results used by the Report Generator
  activity_log    newest-first list of events
  settings        researcher name, alpha, confidence level, decimals
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

import config


def _load_settings() -> dict:
    s = dict(config.DEFAULT_SETTINGS)
    try:
        s.update(json.loads(config.SETTINGS_FILE.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        pass
    return s


def save_settings():
    try:
        config.SETTINGS_FILE.write_text(json.dumps(st.session_state.settings, indent=2), encoding="utf-8")
    except OSError:
        st.warning("Could not save settings to disk (they remain active for this session).")


def init_state():
    defaults = {"datasets": {}, "cleaned": {}, "active_dataset": None, "results": [],
                "activity_log": [], "result_counter": 0, "valid_ranges": {}}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    if "settings" not in st.session_state:
        st.session_state.settings = _load_settings()


def setting(key: str):
    return st.session_state.settings.get(key, config.DEFAULT_SETTINGS[key])


def log_activity(action: str, dataset: str = "", detail: str = ""):
    st.session_state.activity_log.insert(
        0, {"time": datetime.now(), "action": action, "dataset": dataset, "detail": detail})
    del st.session_state.activity_log[200:]


def add_dataset(name: str, df: pd.DataFrame, detail: str = "Dataset loaded"):
    """Store a dataset (suffix added if the name already exists) and make it active."""
    base, i = name, 2
    while name in st.session_state.datasets:
        name, i = f"{base} ({i})", i + 1
    st.session_state.datasets[name] = df
    st.session_state.active_dataset = name
    log_activity("Dataset uploaded", name, f"{detail}: {len(df)} rows × {df.shape[1]} columns")
    return name


def remove_dataset(name: str):
    st.session_state.datasets.pop(name, None)
    st.session_state.cleaned.pop(name, None)
    st.session_state.results = [r for r in st.session_state.results if r["dataset"] != name and
                                r["dataset"] != f"{name} (cleaned)"]
    st.session_state.active_dataset = next(iter(st.session_state.datasets), None)


def load_sample():
    from modules.sample_data import generate_sample_data
    return add_dataset(config.SAMPLE_DATASET_NAME, generate_sample_data(), "Simulated sample loaded")


def save_result(kind: str, title: str, dataset: str, **payload) -> int:
    """Store an analysis result for the report and log it."""
    st.session_state.result_counter += 1
    rid = st.session_state.result_counter
    st.session_state.results.append({"id": rid, "kind": kind, "title": title, "dataset": dataset,
                                     "time": datetime.now(), **payload})
    labels = {"descriptives": "Descriptive statistics", "reliability": "Reliability analysis",
              "correlation": "Correlation analysis", "test": "Hypothesis test", "chart": "Chart"}
    log_activity(f"{labels.get(kind, kind)} saved", dataset, title)
    return rid
