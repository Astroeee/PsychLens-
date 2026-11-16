"""Dashboard: summary cards, data-quality charts, recent datasets and activity."""
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from components.cards import callout, card_title, empty_state, kpi_card
from components.layout import page_header
from components.tables import show_df, show_plot
from modules import data_processing as dp
from modules.state import load_sample
from modules.visualization import _theme

page_header("Dashboard", "Overview of your research data")
ss = st.session_state
datasets = ss.datasets

if not datasets:
    with st.container(border=True):
        empty_state("Welcome to PsychLens", "Load the simulated sample dataset to explore every feature, or upload your own CSV/Excel file.")
        c1, c2, c3, c4 = st.columns([2, 1.3, 1.3, 2])
        if c2.button("Load sample dataset", type="primary", key="dash_sample"):
            load_sample()
            st.rerun()
        c3.page_link("pages/2_Data_Management.py", label="Upload data", icon=":material/upload:")
    st.stop()

names = list(datasets)
active = ss.active_dataset if ss.active_dataset in names else names[0]
active = st.selectbox("Active dataset", names, index=names.index(active), key="dash_ds")
ss.active_dataset = active
df = ss.cleaned.get(active, datasets[active])
parts = dp.split_columns(df)
miss = dp.missing_percentage(df)

c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("Datasets", len(datasets), f"{len(ss.cleaned)} with cleaned copy", "database")
with c2: kpi_card("Respondents", f"{len(df):,}", "rows in active dataset", "users", "green")
with c3: kpi_card("Variables", df.shape[1], f"{len(parts['numeric'])} numeric · {len(parts['categorical'])} categorical", "columns")
with c4: kpi_card("Missing data", f"{miss:.1f}%", "of all cells", "alert", "amber")
st.write("")

left, right = st.columns([2, 1])
with left, st.container(border=True):
    card_title("Distribution preview", "Pick any numeric variable")
    nums = parts["numeric"]
    if nums:
        var = st.selectbox("Variable", nums, key="dash_var", label_visibility="collapsed")
        v = pd.to_numeric(df[var], errors="coerce").dropna()
        fig = go.Figure(go.Histogram(x=v, nbinsx=20, marker_color=config.COLORS["primary"], opacity=0.85))
        show_plot(_theme(fig, "", var, "Frequency", legend=False), key="dash_hist")
    else:
        st.caption("No numeric variables found.")
with right, st.container(border=True):
    card_title("Variable types", "Composition of the dataset")
    counts = {k.capitalize(): len(v) for k, v in parts.items() if v}
    fig = go.Figure(go.Pie(labels=list(counts), values=list(counts.values()), hole=0.62, sort=False,
                           marker=dict(colors=config.PALETTE), textinfo="value"))
    fig.add_annotation(text=f"<b>{df.shape[1]}</b><br>variables", showarrow=False, font=dict(size=14))
    show_plot(_theme(fig, "", "", "", legend=True), key="dash_donut")

left, right = st.columns([2, 1])
with left, st.container(border=True):
    card_title("Missing data by variable", "Number of missing values per column")
    ms = dp.missing_summary(df)
    ms = ms[ms["Missing"] > 0]
    if ms.empty:
        callout("No missing values detected in this dataset.", "ok")
    else:
        fig = go.Figure(go.Bar(x=ms["Column"], y=ms["Missing"], marker_color=config.COLORS["warning"],
                               text=ms["Missing %"].map("{:.1f}%".format), textposition="outside"))
        show_plot(_theme(fig, "", "", "Missing values", legend=False), key="dash_missing")
with right, st.container(border=True):
    card_title("Data quality", "Automatic checks")
    inv = dp.find_invalid_values(df, ss.valid_ranges.get(active))
    complete = df.dropna().shape[0]
    st.markdown(f"**{dp.duplicate_count(df)}** duplicate rows  \n**{len(inv)}** potentially invalid values  \n"
                f"**{complete / max(len(df), 1) * 100:.0f}%** complete cases ({complete} of {len(df)})")
    st.page_link("pages/2_Data_Management.py", label="Review & clean data", icon=":material/arrow_forward:")


def ago(t):
    s = (datetime.now() - t).total_seconds()
    return "just now" if s < 60 else f"{int(s // 60)} min ago" if s < 3600 else f"{int(s // 3600)} h ago" if s < 86400 else f"{int(s // 86400)} d ago"


left, right = st.columns(2)
with left, st.container(border=True):
    card_title("Recently analyzed datasets", "Datasets with saved analyses")
    log = ss.activity_log
    rows = []
    for name in names:
        saved = [e for e in log if e["dataset"].replace(" (cleaned)", "") == name and "saved" in e["action"]]
        if saved:
            d = ss.cleaned.get(name, datasets[name])
            rows.append({"Dataset": name, "Rows": len(d), "Columns": d.shape[1], "Analyses saved": len(saved),
                         "Last analysis": ago(saved[0]["time"])})
    if rows:
        show_df(pd.DataFrame(rows))
    else:
        st.caption("No analyses saved yet. Use “Save to report” on any analysis page.")
with right, st.container(border=True):
    card_title("Recent analysis activity", "Latest events")
    if not ss.activity_log:
        st.caption("No activity yet.")
    for e in ss.activity_log[:6]:
        st.markdown(f'<div class="act"><div class="dot">●</div><div><div class="t">{e["action"]}</div>'
                    f'<div class="d">{e["dataset"]} · {e["detail"]}</div></div><div class="ago">{ago(e["time"])}</div></div>',
                    unsafe_allow_html=True)
