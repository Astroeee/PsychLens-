"""Session state helpers (Streamlit-aware). Also bridges the session to account storage.

st.session_state keeps data alive between clicks:
  user            signed-in account (or guest) dict, None when signed out
  datasets        name -> original DataFrame (never modified)
  cleaned         name -> cleaned copy (created only after the user confirms)
  results         saved analysis results used by the Report Generator
  activity_log    newest-first list of events
  settings        researcher name, institution, alpha, confidence level, decimals
Datasets, cleaned copies and settings are saved to the account; guests are not saved.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

import config
from modules import auth, storage


def init_state():
    defaults = {"user": None, "datasets": {}, "cleaned": {}, "active_dataset": None, "results": [],
                "activity_log": [], "result_counter": 0, "valid_ranges": {}}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("settings", dict(config.DEFAULT_SETTINGS))


def setting(key: str):
    return st.session_state.settings.get(key, config.DEFAULT_SETTINGS[key])


# ------------------------------------------------------------ account ------
def is_authenticated() -> bool:
    return st.session_state.get("user") is not None


def is_guest() -> bool:
    u = st.session_state.get("user")
    return bool(u and u.get("guest"))


def _uid():
    u = st.session_state.get("user")
    return None if (not u or u.get("guest")) else u["id"]


def _safe(fn, *args):
    """Run a storage action; a disk problem must never crash the app."""
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not save to your account storage: {exc}")


def login(user: dict):
    ss = st.session_state
    ss.user = user
    ss.settings = {**config.DEFAULT_SETTINGS, **auth.get_settings(user["id"])}
    try:
        ss.datasets, ss.cleaned = storage.load_all(user["id"])
    except Exception:  # noqa: BLE001
        ss.datasets, ss.cleaned = {}, {}
    ss.active_dataset = next(iter(ss.datasets), None)
    log_activity("Signed in", detail=f"{len(ss.datasets)} saved dataset(s) restored")


def guest_login():
    st.session_state.user = {"id": None, "username": "guest", "display_name": "Guest", "email": "", "guest": True}
    st.session_state.settings = {**config.DEFAULT_SETTINGS, "researcher_name": "Guest"}
    log_activity("Guest session started")


def logout():
    st.session_state.clear()


def save_settings():
    uid = _uid()
    if uid is not None:
        _safe(auth.save_settings, uid, st.session_state.settings)


# ---------------------------------------------------------- activity -------
def log_activity(action: str, dataset: str = "", detail: str = ""):
    st.session_state.activity_log.insert(
        0, {"time": datetime.now(), "action": action, "dataset": dataset, "detail": detail})
    del st.session_state.activity_log[200:]


# ---------------------------------------------------------- datasets -------
def add_dataset(name: str, df: pd.DataFrame, detail: str = "Dataset loaded"):
    """Store a dataset (suffix added if the name already exists) and make it active."""
    base, i = name, 2
    while name in st.session_state.datasets:
        name, i = f"{base} ({i})", i + 1
    st.session_state.datasets[name] = df
    st.session_state.active_dataset = name
    if _uid() is not None:
        _safe(storage.save_dataset, _uid(), name, df)
    log_activity("Dataset uploaded", name, f"{detail}: {len(df)} rows × {df.shape[1]} columns")
    return name


def remove_dataset(name: str):
    ss = st.session_state
    ss.datasets.pop(name, None)
    ss.cleaned.pop(name, None)
    ss.results = [r for r in ss.results if r["dataset"].replace(" (cleaned)", "") != name]
    ss.active_dataset = next(iter(ss.datasets), None)
    if _uid() is not None:
        _safe(storage.delete_dataset, _uid(), name)


def set_cleaned(name: str, df: pd.DataFrame):
    st.session_state.cleaned[name] = df
    if _uid() is not None:
        _safe(storage.save_cleaned, _uid(), name, df)


def discard_cleaned(name: str):
    st.session_state.cleaned.pop(name, None)
    if _uid() is not None:
        _safe(storage.delete_cleaned, _uid(), name)


def replace_original(name: str):
    ss = st.session_state
    ss.datasets[name] = ss.cleaned.pop(name)
    if _uid() is not None:
        _safe(storage.delete_cleaned, _uid(), name)
        _safe(storage.save_dataset, _uid(), name, ss.datasets[name])


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
              "correlation": "Correlation analysis", "test": "Hypothesis test", "chart": "Chart",
              "regression": "Regression", "factor": "Factor analysis", "power": "Power analysis"}
    log_activity(f"{labels.get(kind, kind)} saved", dataset, title)
    return rid
