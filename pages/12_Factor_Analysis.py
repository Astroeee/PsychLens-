"""Exploratory factor analysis: suitability checks, factor retention, loadings."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config
from components.cards import apa_box, callout, card_title, kpi_card
from components.layout import page_header
from components.selectors import select_dataset
from components.tables import fmt, show_df, show_plot
from modules import data_processing as dp
from modules import factor_analysis as fa
from modules.state import save_result
from modules.visualization import _theme

page_header("Factor Analysis", "Explore the underlying structure of questionnaire items")
label, df = select_dataset("fa")
nums = dp.numeric_columns(df)
likert = list(dp.detect_likert_columns(df))

with st.container(border=True):
    card_title("Settings", "Select the items, then choose extraction and rotation")
    items = st.multiselect("Items (3 or more)", nums, default=likert, key="fa_items")
    a, b, c, d = st.columns(4)
    method = a.selectbox("Extraction", ["paf", "pca"], key="fa_method",
                         format_func={"paf": "Principal axis factoring", "pca": "Principal components"}.get)
    rotation = b.selectbox("Rotation", ["varimax", "promax", "none"], key="fa_rot",
                           format_func={"varimax": "Varimax (uncorrelated factors)", "promax": "Promax (correlated factors)", "none": "None"}.get)
    auto = c.radio("Number of factors", ["Automatic (parallel analysis)", "Choose myself"], key="fa_auto")
    n_f = d.number_input("Factors", 1, max(len(items) - 1, 1), 2, key="fa_nf", disabled=auto.startswith("Auto")) if len(items) > 2 else 1
    cutoff = st.slider("Display loadings with |value| ≥", 0.10, 0.60, 0.30, 0.05, key="fa_cut")

if len(items) < 3:
    callout("Select at least 3 items.", "info")
    st.stop()
try:
    res = fa.analyze(df, items, None if auto.startswith("Auto") else int(n_f), method, rotation, cutoff)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("KMO", f"{res.kmo_overall:.2f}", fa.kmo_label(res.kmo_overall), "check")
with c2: kpi_card("Bartlett's test", "p < .001" if res.bartlett[2] < .001 else f"p = {res.bartlett[2]:.3f}", f"χ²({res.bartlett[1]}) = {res.bartlett[0]:.1f}", "alert", "amber")
with c3: kpi_card("Factors suggested", f"{res.kaiser_k} · {res.parallel_k}", "Kaiser · parallel analysis", "columns", "green")
with c4: kpi_card("Respondents used", res.n, f"{res.n_total - res.n} excluded", "users")
st.write("")
apa_box(fa.apa_summary(res))
for w in res.warnings:
    callout(w, "warn")

tab_s, tab_l, tab_f, tab_i = st.tabs(["Scree & factor retention", "Loadings", "Fit & communalities", "Interpretation"])
with tab_s:
    with st.container(border=True):
        card_title("Scree plot", "Keep factors whose eigenvalue is above the parallel-analysis line (random-data benchmark)")
        x = list(range(1, len(res.eigenvalues) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=res.eigenvalues, mode="lines+markers", name="Eigenvalue (your data)", line=dict(color=config.COLORS["primary"], width=2)))
        fig.add_trace(go.Scatter(x=x, y=res.parallel, mode="lines", name="Parallel analysis (95th percentile)", line=dict(color=config.COLORS["danger"], dash="dash")))
        fig.add_hline(y=1, line_dash="dot", line_color="#9CA3AF", annotation_text="Kaiser (eigenvalue = 1)")
        show_plot(_theme(fig, "", "Component / factor number", "Eigenvalue"), key="fa_scree")
        st.caption("The Kaiser criterion tends to over-extract; parallel analysis is generally preferred. Combine them with theory and interpretability.")
with tab_l:
    with st.container(border=True):
        title = "Pattern matrix (rotated loadings)" if rotation != "none" else "Unrotated loadings"
        card_title(title, f"Loadings below {cutoff:.2f} are blanked in the table")
        L = res.loadings
        shown = L.copy().round(2).astype(object)
        shown[L.abs() < cutoff] = ""
        shown.insert(0, "Item", L.index)
        show_df(shown)
        z = L.to_numpy()
        fig = go.Figure(go.Heatmap(z=z, x=list(L.columns), y=list(L.index), zmin=-1, zmax=1,
                                   colorscale=[[0, config.COLORS["danger"]], [0.5, "#FFFFFF"], [1, config.COLORS["primary"]]],
                                   text=np.round(z, 2).astype(str), texttemplate="%{text}"))
        fig.update_yaxes(autorange="reversed")
        show_plot(_theme(fig, "", "", "", legend=False), key="fa_heat")
        if res.structure is not None:
            with st.expander("Structure matrix and factor correlations"):
                show_df(fmt(res.structure.reset_index(names="Item"), 2))
                show_df(fmt(res.factor_corr.reset_index(names="Factor"), 2))
    with st.container(border=True):
        card_title("Item assignment", "Primary factor of each item, plus cross-loading flags")
        show_df(fmt(res.assignment, 2))
        groups = {f: res.assignment.loc[res.assignment["Primary factor"] == f, "Item"].tolist() for f in res.loadings.columns}
        st.markdown("**Items per factor:** " + " · ".join(f"**{f}**: {', '.join(v) or '—'}" for f, v in groups.items()))
        st.caption("To check each subscale's reliability, select that factor's items in Reliability Analysis.")
with tab_f:
    l, r = st.columns(2)
    with l, st.container(border=True):
        card_title("Variance explained")
        show_df(fmt(res.variance, 2))
        st.caption(f"Residual correlations: RMSR = {res.rmsr:.3f}; {res.pct_large_resid:.0f}% exceed |.05| (RMSR < .06–.08 suggests an acceptable fit).")
    with r, st.container(border=True):
        card_title("Communalities and sampling adequacy")
        show_df(fmt(res.communalities, 3))
with tab_i:
    with st.container(border=True):
        st.write(fa.interpret(res))
        callout("<b>Notes.</b> Pearson correlations are used; short Likert items (≤ 5 points) with skewed responses may be better analysed with polychoric "
                "correlations. Factor solutions are exploratory and sample-dependent; replicate them in a new sample before relying on them.", "info")

if st.button("Save to report", type="primary", key="fa_save"):
    tables = {"Eigenvalues and parallel analysis": __import__("pandas").DataFrame(
        {"Factor": range(1, len(res.eigenvalues) + 1), "Eigenvalue": res.eigenvalues, "Parallel analysis (95th pct.)": res.parallel}),
        "Variance explained": res.variance, "Communalities": res.communalities, "Item assignment": res.assignment}
    save_result("factor", f"{len(items)} items, {res.n_factors} factors", label, apa=fa.apa_summary(res),
                loadings=res.loadings.round(3).reset_index(names="Item"), tables=tables,
                interpretation=fa.interpret(res), warnings=res.warnings)
    st.success("Saved. Find it in the Report Generator.")
