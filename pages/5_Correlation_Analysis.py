"""Pearson/Spearman correlations with p-values, CIs, heatmap and scatterplot."""
import streamlit as st

from components.cards import callout, card_title
from components.layout import page_header
from components.selectors import select_dataset
from components.tables import fmt, show_df, show_plot
from modules import correlation as cor
from modules import data_processing as dp
from modules.state import save_result, setting
from modules.visualization import build_figure

page_header("Correlation Analysis", "Explore relationships between variables")
label, df = select_dataset("cor")
dec, alpha, conf = setting("decimals"), setting("alpha"), setting("confidence")
nums = dp.numeric_columns(df)

with st.container(border=True):
    card_title("Analysis settings")
    sel = st.multiselect("Variables (2 or more)", nums, default=[c for c in nums if not c.startswith("Q")][:4], key="cor_vars")
    a, b = st.columns(2)
    method = a.radio("Method", ["pearson", "spearman"], horizontal=True, key="cor_method",
                     format_func=lambda m: "Pearson (linear, continuous)" if m == "pearson" else "Spearman (rank-based, monotonic)")
    adjust = b.selectbox("Multiple-comparison adjustment", ["none", "holm", "bonferroni", "fdr_bh"], key="cor_adj",
                         format_func={"none": "None", "holm": "Holm", "bonferroni": "Bonferroni", "fdr_bh": "Benjamini–Hochberg (FDR)"}.get)
if len(sel) < 2:
    callout("Select at least two variables.", "info")
    st.stop()

res = cor.correlation_matrix(df, sel, method, conf, adjust, alpha)
sym = res.symbol
callout(f"Pairwise-complete observations are used for each pair (N shown per pair). "
        f"{'Pearson assumes a linear relationship and roughly continuous, bivariate-normal data; check the scatterplot for outliers.' if method == 'pearson' else 'Spearman uses ranks and is robust to outliers and non-normality, but captures monotonic relationships only.'} "
        f"Significance is evaluated at α = {alpha}" + (f" using {adjust} adjusted p-values." if adjust != "none" else "."), "info")

tab_m, tab_p, tab_h, tab_s = st.tabs(["Correlation matrix", "Pairs, p-values & CIs", "Heatmap", "Scatterplot"])
with tab_m:
    with st.container(border=True):
        card_title(f"{'Pearson r' if method == 'pearson' else 'Spearman ρ'} matrix", "Lower triangle shown with significance stars (* p < .05, ** p < .01, *** p < .001)")
        apa = cor.apa_matrix(res)
        show_df(apa)
        with st.expander("Full coefficient, p-value and N matrices"):
            st.markdown(f"**{sym}**"); show_df(fmt(res.r.reset_index(names="Variable"), 3))
            st.markdown("**p (adjusted)**"); show_df(fmt(res.p_adj.reset_index(names="Variable"), 4))
            st.markdown("**N**"); show_df(res.n.reset_index(names="Variable"))
with tab_p:
    with st.container(border=True):
        card_title("All variable pairs", f"{conf * 100:.0f}% confidence intervals use the Fisher z transformation" + (" (Bonett–Wright SE for Spearman)" if method == "spearman" else ""))
        show_df(fmt(res.pairs, 3))
        st.download_button("Download CSV", res.pairs.to_csv(index=False).encode("utf-8-sig"), "correlations.csv", "text/csv", key="cor_dl")
with tab_h:
    with st.container(border=True):
        show_plot(build_figure("Correlation heatmap", df, cols=sel, method=method, title=f"{'Pearson' if method == 'pearson' else 'Spearman'} correlation heatmap"), key="cor_heat")
with tab_s:
    with st.container(border=True):
        a, b, c = st.columns(3)
        x = a.selectbox("X variable", sel, key="cor_x")
        y = b.selectbox("Y variable", sel, index=1, key="cor_y")
        g = c.selectbox("Color by", ["(none)"] + dp.grouping_candidates(df), key="cor_g")
        if x == y:
            st.info("Choose two different variables.")
        else:
            pr = cor.correlate_pair(df[x], df[y], method, conf)
            st.markdown(f"**{sym} = {pr['r']:.2f}**, {conf * 100:.0f}% CI [{pr['ci_low']:.2f}, {pr['ci_high']:.2f}], N = {pr['n']}")
            show_plot(build_figure("Scatterplot", df, x=x, y=y, by=None if g == "(none)" else g, trendline=True, title=f"{y} vs {x}"), key="cor_sc")

note = "* p < .05. ** p < .01. *** p < .001." + (f" p-values adjusted ({adjust})." if adjust != "none" else "")
if st.button("Save to report", type="primary", key="cor_save"):
    save_result("correlation", f"{len(sel)} variables", label, method_name="Pearson" if method == "pearson" else "Spearman",
                matrix_apa=cor.apa_matrix(res), pairs=res.pairs.drop(columns=["Significant"]), note=note)
    st.success("Saved. Find it in the Report Generator.")
