"""Descriptive statistics with tables and distribution charts."""
import pandas as pd
import streamlit as st

from components.cards import card_title, callout
from components.layout import page_header
from components.selectors import select_dataset
from components.tables import fmt, show_df, show_plot
from modules import data_processing as dp
from modules import descriptive_statistics as ds
from modules.state import save_result, setting
from modules.visualization import build_figure

page_header("Descriptive Statistics", "Summarize and visualize your variables")
label, df = select_dataset("desc")
dec = setting("decimals")
parts = dp.split_columns(df)
nums = parts["numeric"]
if not nums:
    callout("No numeric variables were found in this dataset.", "warn")
    st.stop()

tab_tbl, tab_cat, tab_chart = st.tabs(["Numeric summary", "Categorical frequencies", "Charts"])

with tab_tbl:
    with st.container(border=True):
        card_title("Choose variables")
        default = [c for c in nums if not c.startswith("Q")][:5] or nums[:5]
        sel = st.multiselect("Numeric variables", nums, default=default, key="desc_vars")
        grp_opts = ["(none)"] + dp.grouping_candidates(df)
        by = st.selectbox("Split by group (optional)", grp_opts, key="desc_by")
    if not sel:
        callout("Select at least one variable.", "info")
    else:
        res = ds.describe_variables(df, sel)
        main_cols = ["Variable", "N", "Missing", "Mean", "Median", "Mode", "SD", "Variance", "Min", "Max", "Range",
                     "Skewness", "Kurtosis", "Q1", "Q3", "IQR"]
        with st.container(border=True):
            card_title("Descriptive statistics", "Skewness and kurtosis use bias-corrected estimators; kurtosis is excess kurtosis")
            show_df(fmt(res[main_cols], dec))
            multi = res.loc[res["Modes (count)"] > 1, "Variable"].tolist()
            if multi:
                st.caption(f"Variables with more than one mode (lowest shown): {', '.join(multi)}.")
            with st.expander("Extra: standard error and Shapiro–Wilk normality test"):
                show_df(fmt(res[["Variable", "SE", "Shapiro-Wilk p"]], 4))
        c1, c2, _ = st.columns([1, 1, 4])
        c1.download_button("Download CSV", res.to_csv(index=False).encode("utf-8-sig"), "descriptive_statistics.csv", "text/csv", key="desc_dl")
        if c2.button("Save to report", type="primary", key="desc_save"):
            save_result("descriptives", f"{len(sel)} variables", label, table=res[main_cols])
            st.success("Saved. Find it in the Report Generator.")
        if by != "(none)":
            with st.container(border=True):
                card_title(f"Statistics by {by}")
                for v in sel:
                    st.markdown(f"**{v}**")
                    show_df(fmt(ds.describe_by_group(df, v, by), dec))

with tab_cat:
    cats = dp.grouping_candidates(df)
    if not cats:
        callout("No categorical variables found.", "info")
    else:
        c = st.selectbox("Variable", cats, key="desc_cat")
        with st.container(border=True):
            card_title(f"Frequency table: {c}")
            ft = ds.frequency_table(df[c])
            show_df(fmt(ft, dec))
            st.download_button("Download CSV", ft.to_csv(index=False).encode("utf-8-sig"), f"frequencies_{c}.csv", "text/csv", key="desc_ft_dl")
        with st.container(border=True):
            show_plot(build_figure("Frequency distribution", df, x=c, title=f"Frequency distribution of {c}"), key="desc_freq_chart")

with tab_chart:
    kind = st.radio("Chart", ["Histogram", "Boxplot", "Distribution (density)", "Bar chart"], horizontal=True, key="desc_kind")
    with st.container(border=True):
        if kind in ("Histogram", "Distribution (density)"):
            v = st.selectbox("Variable", nums, key="desc_hv")
            bins = st.slider("Bins", 5, 60, 20, key="desc_bins")
            fig = build_figure("Histogram", df, x=v, bins=bins, kde=(kind != "Histogram"), title=f"Distribution of {v}")
        elif kind == "Boxplot":
            v = st.selectbox("Variable", nums, key="desc_bv")
            g = st.selectbox("Group by (optional)", ["(none)"] + dp.grouping_candidates(df), key="desc_bg")
            fig = build_figure("Boxplot", df, y=v, by=None if g == "(none)" else g, title=f"{v}" + ("" if g == "(none)" else f" by {g}"))
        else:
            cats = dp.grouping_candidates(df)
            if not cats:
                st.stop()
            c = st.selectbox("Categorical variable", cats, key="desc_barc")
            y = st.selectbox("Mean of (optional)", ["(counts)"] + nums, key="desc_bary")
            fig = build_figure("Bar chart", df, x=c, y=None if y == "(counts)" else y, agg="mean",
                               title=f"{c}" if y == "(counts)" else f"Mean {y} by {c}")
        show_plot(fig, key="desc_chart")
