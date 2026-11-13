"""Dataset selection shared by all analysis pages."""
import streamlit as st

import config
from components.cards import callout
from modules.state import load_sample


def select_dataset(key: str):
    """Show the dataset picker. Returns (label, DataFrame). Stops the page if no data."""
    ss = st.session_state
    names = list(ss.datasets)
    if not names:
        callout("<b>No dataset loaded yet.</b> Load the simulated sample dataset or upload your own file in Data Management.", "info")
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("Load sample dataset", type="primary", key=f"{key}_sample"):
            load_sample()
            st.rerun()
        c2.page_link("pages/2_Data_Management.py", label="Go to Data Management", icon=":material/upload:")
        st.stop()

    idx = names.index(ss.active_dataset) if ss.active_dataset in names else 0
    c1, c2 = st.columns([3, 2])
    name = c1.selectbox("Dataset", names, index=idx, key=f"{key}_ds")
    ss.active_dataset = name
    use_clean = False
    if name in ss.cleaned:
        use_clean = c2.toggle("Use cleaned version", value=True, key=f"{key}_clean",
                              help="A cleaned copy exists. The original is never modified.")
    df = ss.cleaned[name] if use_clean else ss.datasets[name]
    label = f"{name} (cleaned)" if use_clean else name
    return label, df


def is_simulated(label: str) -> bool:
    return label.startswith("Sample") or "simulated" in label.lower()
