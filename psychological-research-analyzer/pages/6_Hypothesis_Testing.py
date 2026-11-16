"""Hypothesis testing: choose a test from the research design, check assumptions, report results."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from components.cards import apa_box, callout, card_title
from components.layout import page_header
from components.selectors import select_dataset
from components.tables import assumptions_table, fmt, show_df, show_plot
from modules import data_processing as dp
from modules import hypothesis_testing as ht
from modules.state import save_result, setting
from modules.visualization import _theme, build_figure

page_header("Hypothesis Testing", "Compare groups and test hypotheses with assumption checks")
label, df = select_dataset("ht")
alpha, conf, dec = setting("alpha"), setting("confidence"), setting("decimals")
nums = dp.numeric_columns(df)
groupers = dp.grouping_candidates(df)

TESTS = {
    "Independent samples t-test": "Compare the means of two separate groups on a numeric outcome (different respondents in each group).",
    "Paired samples t-test": "Compare two numeric measures from the same respondents (e.g., before/after).",
    "One-sample t-test": "Compare one numeric variable's mean with a known or theoretical value.",
    "One-way ANOVA": "Compare the means of two or more separate groups on a numeric outcome.",
    "Chi-square test of independence": "Test association between two categorical variables.",
    "Mann-Whitney U test": "Non-parametric alternative to the independent t-test (ordinal data or non-normal outcomes).",
    "Wilcoxon signed-rank test": "Non-parametric alternative to the paired t-test.",
}

with st.expander("Which test fits my research design?", expanded=False):
    show_df(pd.DataFrame([
        ("Two independent groups, numeric outcome", "Independent t-test (Welch by default) · Mann-Whitney U if ordinal/non-normal"),
        ("Same respondents measured twice", "Paired t-test · Wilcoxon signed-rank if ordinal/non-normal"),
        ("One sample vs. a known value", "One-sample t-test"),
        ("Three or more independent groups", "One-way ANOVA (Welch if variances differ)"),
        ("Two categorical variables", "Chi-square test of independence (Fisher's exact for sparse 2×2 tables)"),
    ], columns=["Research design", "Typical test"]))
    st.caption("Assumption checks are shown with every result but never choose the test for you: the choice rests on your design and variable types.")

with st.container(border=True):
    card_title("1 · Choose the test")
    test = st.selectbox("Test", list(TESTS), key="ht_test", label_visibility="collapsed")
    st.caption(TESTS[test])

res = None
with st.container(border=True):
    card_title("2 · Select variables")
    try:
        if test in ("Independent samples t-test", "Mann-Whitney U test"):
            if not nums or not groupers:
                callout("This test needs a numeric variable and a grouping variable.", "warn"); st.stop()
            a, b = st.columns(2)
            dv = a.selectbox("Outcome (numeric)", nums, key="ht_dv")
            g = b.selectbox("Grouping variable", groupers, key="ht_g")
            levels = sorted(df[g].dropna().unique(), key=str)
            if len(levels) < 2:
                callout("The grouping variable has fewer than 2 levels.", "warn"); st.stop()
            c, d = st.columns(2)
            l1 = c.selectbox("Group 1", levels, key=f"ht_l1_{test}_{g}")
            l2 = d.selectbox("Group 2", [l for l in levels if l != l1], key=f"ht_l2_{test}_{g}")
            opt = None
            if test.startswith("Independent"):
                opt = st.radio("Variance assumption", ["Welch (does not assume equal variances) – recommended", "Student (assumes equal variances)"], key="ht_var")
            ok = st.checkbox("I confirm each respondent belongs to only one group and observations are independent.", key="ht_ind")
            if not ok:
                callout("Confirm the research design above to run the test.", "info"); st.stop()
            x1, x2 = df.loc[df[g] == l1, dv], df.loc[df[g] == l2, dv]
            if test.startswith("Independent"):
                res = ht.independent_t_test(x1, x2, str(l1), str(l2), dv, opt.startswith("Student"), alpha, conf)
            else:
                res = ht.mann_whitney_u(x1, x2, str(l1), str(l2), dv, alpha)
            plot = lambda: build_figure("Boxplot", df[df[g].isin([l1, l2])], y=dv, by=g, title=f"{dv} by {g}")

        elif test in ("Paired samples t-test", "Wilcoxon signed-rank test"):
            if len(nums) < 2:
                callout("This test needs two numeric variables.", "warn"); st.stop()
            a, b = st.columns(2)
            v1 = a.selectbox("Measure 1 (e.g., before)", nums, key="ht_v1")
            v2 = b.selectbox("Measure 2 (e.g., after)", [c for c in nums if c != v1], key="ht_v2")
            ok = st.checkbox("I confirm both variables were measured on the same respondents (matched pairs).", key="ht_pair")
            if not ok:
                callout("Confirm the research design above to run the test.", "info"); st.stop()
            if test.startswith("Paired"):
                res = ht.paired_t_test(df[v1], df[v2], v1, v2, alpha, conf)
            else:
                res = ht.wilcoxon_signed_rank(df[v1], df[v2], v1, v2, alpha)
            long = pd.DataFrame({"Measure": [v1] * len(df) + [v2] * len(df), "Value": pd.concat([df[v1], df[v2]], ignore_index=True)})
            plot = lambda: build_figure("Boxplot", long, y="Value", by="Measure", title=f"{v1} vs {v2}")

        elif test == "One-sample t-test":
            if not nums:
                callout("This test needs a numeric variable.", "warn"); st.stop()
            a, b = st.columns(2)
            v = a.selectbox("Variable", nums, key="ht_ov")
            mu = b.number_input("Test value (μ₀)", value=0.0, key="ht_mu")
            res = ht.one_sample_t_test(df[v], mu, v, alpha, conf)
            plot = lambda: build_figure("Histogram", df, x=v, title=f"Distribution of {v}", kde=True)

        elif test == "One-way ANOVA":
            if not nums or not groupers:
                callout("This test needs a numeric variable and a grouping factor.", "warn"); st.stop()
            a, b = st.columns(2)
            dv = a.selectbox("Outcome (numeric)", nums, key="ht_adv")
            g = b.selectbox("Factor (groups)", groupers, key="ht_ag")
            levels = sorted(df[g].dropna().unique(), key=str)
            chosen = st.multiselect("Groups to include", levels, default=levels, key=f"ht_alevels_{g}")
            opt = st.radio("Variance assumption", ["Fisher (assumes equal variances; Tukey HSD post-hoc)",
                                                    "Welch (robust to unequal variances; Holm-adjusted post-hoc)"], key="ht_avar")
            ok = st.checkbox("I confirm groups consist of different respondents and observations are independent.", key="ht_aind")
            if not ok:
                callout("Confirm the research design above to run the test.", "info"); st.stop()
            res = ht.one_way_anova(df, dv, g, chosen, opt.startswith("Welch"), alpha, conf)
            plot = lambda: build_figure("Boxplot", df[df[g].isin(chosen)], y=dv, by=g, title=f"{dv} by {g}")

        else:  # chi-square
            if len(groupers) < 2:
                callout("This test needs two categorical variables.", "warn"); st.stop()
            a, b = st.columns(2)
            v1 = a.selectbox("Variable 1", groupers, key="ht_c1")
            v2 = b.selectbox("Variable 2", [c for c in groupers if c != v1], key="ht_c2")
            corr = st.checkbox("Apply Yates continuity correction (2×2 tables only)", value=True, key="ht_yates")
            ok = st.checkbox("I confirm each respondent appears in only one cell (independent observations).", key="ht_cind")
            if not ok:
                callout("Confirm the research design above to run the test.", "info"); st.stop()
            res = ht.chi_square_independence(df, v1, v2, corr, alpha)
            plot = lambda: build_figure("Bar chart", df.dropna(subset=[v1, v2]), x=v1, by=v2, title=f"{v1} by {v2}")
    except ValueError as exc:
        st.error(str(exc)); st.stop()

st.write("")
with st.container(border=True):
    card_title(res.name, res.variables)
    st.markdown(f"*{res.hypotheses}*")
    apa_box(res.apa)
    st.markdown(f"<div class='callout {'ok' if res.significant else 'plain'}'>{'Statistically significant' if res.significant else 'Not statistically significant'} at α = {alpha}</div>", unsafe_allow_html=True)
    for w in res.warnings:
        callout(w, "warn")

l, r = st.columns(2)
with l, st.container(border=True):
    card_title("Test results")
    rows = [("Test statistic", f"{res.statistic_name} = {res.statistic:.3f}"), ("Degrees of freedom", res.df_text or "n/a"),
            ("p-value", f"{res.p:.4f}" if res.p == res.p else "n/a"), ("Sample size", res.n)]
    if res.effect_name:
        rows.append((f"Effect size ({res.effect_name})", f"{res.effect:.3f} – {res.effect_label}"))
    if res.ci:
        rows.append((res.ci_label, f"[{res.ci[0]:.3f}, {res.ci[1]:.3f}]"))
    show_df(pd.DataFrame(rows, columns=["Statistic", "Value"]).astype(str))
with r, st.container(border=True):
    card_title("Visualization")
    show_plot(plot(), key="ht_plot")

with st.container(border=True):
    card_title("Assumption checks", "Informational: used to judge whether the chosen test is appropriate")
    show_df(fmt(assumptions_table(res.assumptions), 3))
    if res.descriptives is not None:
        card_title("Descriptive statistics")
        show_df(fmt(res.descriptives, dec))

for name, t in res.extra_tables.items():
    with st.expander(name, expanded=name.startswith("Post-hoc")):
        show_df(fmt(t, 4))

with st.container(border=True):
    card_title("Interpretation", "Generated from the statistics only; it does not replace your own judgment")
    st.write(res.interpretation)

if st.button("Save to report", type="primary", key="ht_save"):
    save_result("test", f"{res.name}: {res.variables}", label, result=res)
    st.success("Saved. Find it in the Report Generator.")
