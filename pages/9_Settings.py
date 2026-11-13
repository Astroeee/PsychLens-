"""Settings: researcher profile, statistical defaults, session data."""
import importlib.metadata as md

import streamlit as st

import config
from components.cards import callout, card_title
from components.layout import page_header
from components.tables import show_df
from modules.state import load_sample, log_activity, save_settings

import pandas as pd

page_header("Settings", "Configure the application")
ss = st.session_state
s = ss.settings

with st.container(border=True):
    card_title("Researcher profile", "Used in report headers")
    a, b = st.columns(2)
    name = a.text_input("Name", value=s["researcher_name"], key="set_name")
    inst = b.text_input("Institution", value=s["institution"], key="set_inst")

with st.container(border=True):
    card_title("Statistical defaults", "Applied across all analysis pages")
    a, b, c = st.columns(3)
    alpha = a.selectbox("Significance level (α)", [0.10, 0.05, 0.01, 0.001], index=[0.10, 0.05, 0.01, 0.001].index(s["alpha"]) if s["alpha"] in [0.10, 0.05, 0.01, 0.001] else 1, key="set_alpha")
    conf = b.selectbox("Confidence level", [0.90, 0.95, 0.99], index=[0.90, 0.95, 0.99].index(s["confidence"]) if s["confidence"] in [0.90, 0.95, 0.99] else 1,
                       format_func=lambda x: f"{x * 100:.0f}%", key="set_conf")
    dec = c.selectbox("Decimal places in tables", [1, 2, 3, 4], index=[1, 2, 3, 4].index(s["decimals"]) if s["decimals"] in [1, 2, 3, 4] else 1, key="set_dec")
    if st.button("Save settings", type="primary", key="set_save"):
        ss.settings = {"researcher_name": name.strip() or "Researcher", "institution": inst.strip(),
                       "alpha": alpha, "confidence": conf, "decimals": dec}
        save_settings()
        st.success("Settings saved.")
        st.rerun()

with st.container(border=True):
    card_title("Session data", "Data lives in memory for this browser session only")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Load sample dataset", key="set_sample"):
        load_sample(); st.success("Sample dataset loaded.")
    if c2.button("Clear saved results", key="set_clr_res"):
        ss.results = []; st.success("Saved results cleared.")
    if c3.button("Clear activity log", key="set_clr_log"):
        ss.activity_log = []; st.success("Activity log cleared.")
    sure = c4.checkbox("Confirm", key="set_confirm")
    if c4.button("Remove all datasets", disabled=not sure, key="set_clr_data"):
        ss.datasets, ss.cleaned, ss.results, ss.active_dataset = {}, {}, [], None
        log_activity("All datasets removed"); st.success("All datasets removed.")
    if ss.activity_log:
        log = pd.DataFrame(ss.activity_log)
        st.download_button("Export activity log (CSV)", log.to_csv(index=False).encode("utf-8-sig"), "activity_log.csv", "text/csv", key="set_dl_log")

with st.container(border=True):
    card_title("About")
    st.markdown(f"**{config.APP_NAME}** · {config.APP_TAGLINE}. Uploaded data is processed locally and is never sent anywhere by this app.")
    vers = []
    for lib in ("streamlit", "pandas", "numpy", "scipy", "statsmodels", "pingouin", "plotly", "matplotlib", "openpyxl", "reportlab"):
        try: vers.append((lib, md.version(lib)))
        except Exception: vers.append((lib, "not installed"))  # noqa: BLE001
    show_df(pd.DataFrame(vers, columns=["Library", "Version"]))
