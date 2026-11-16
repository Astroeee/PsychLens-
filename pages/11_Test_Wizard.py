"""Test Wizard: answer a few questions about your research design and get a recommended test."""
import streamlit as st

from components.cards import callout, card_title, pill
from components.layout import page_header
from components.tables import fmt, show_df
from modules import data_processing as dp
from modules import test_wizard as tw

page_header("Test Wizard", "Not sure which statistical test to use? Answer a few questions")
ss = st.session_state

callout("The wizard suggests a test from your research question and design. It explains why, but <b>you</b> remain responsible for the choice, "
        "and the final decision should consider your hypotheses, measurement level and field conventions.", "info")

with st.container(border=True):
    card_title("1 · What do you want to do?")
    goal = st.radio("Goal", tw.GOALS, key="wz_goal", label_visibility="collapsed")

outcome, groups, design, normal, small_n, both_num, binary, scale_task = "numeric", 2, "independent", "yes", False, True, True, "reliability"
snap_used = False

with st.container(border=True):
    card_title("2 · Describe your design and data")
    g = tw.GOALS
    if goal == g[0]:
        outcome = st.radio("What type of variable is your outcome (the thing you measure)?", list(tw.OUTCOMES), format_func=tw.OUTCOMES.get, key="wz_outcome")
        design = st.radio("How were the groups or conditions formed?", list(tw.DESIGNS), format_func=tw.DESIGNS.get, key="wz_design")
        groups = 2 if st.radio("How many groups or conditions are you comparing?", ["2", "3 or more"], horizontal=True, key="wz_groups") == "2" else 3
    elif goal == g[1]:
        outcome = st.radio("What type of variable is it?", ["numeric", "ordinal", "categorical"], format_func=tw.OUTCOMES.get, key="wz_outcome1")
    elif goal == g[2]:
        kind = st.radio("What kinds of variables?", ["Both numeric", "One or both ordinal", "Both categorical"], key="wz_rel_kind")
        outcome = {"Both numeric": "numeric", "One or both ordinal": "ordinal", "Both categorical": "categorical"}[kind]
    elif goal == g[3]:
        outcome = {"Numeric (e.g., a score)": "numeric", "Two categories (yes/no)": "categorical",
                   "Ordered or 3+ categories": "ordinal"}[st.radio("What are you predicting?", ["Numeric (e.g., a score)", "Two categories (yes/no)", "Ordered or 3+ categories"], key="wz_pred")]
    elif goal == g[4]:
        scale_task = "reliability" if st.radio("What is your question?", ["Do my items hang together (reliability)?", "How many dimensions do my items measure (structure)?"], key="wz_scale").startswith("Do") else "structure"

    needs_dist = (goal in (g[0], g[1], g[2])) and outcome in ("numeric", "ordinal")
    if needs_dist and outcome == "numeric":
        st.markdown("**Distribution and sample size**")
        if ss.datasets:
            with st.expander("Check my data (optional)"):
                name = st.selectbox("Dataset", list(ss.datasets), key="wz_ds")
                d = ss.cleaned.get(name, ss.datasets[name])
                nums = dp.numeric_columns(d)
                if nums:
                    ycol = st.selectbox("Outcome variable", nums, key="wz_ycol")
                    gcol = st.selectbox("Grouping variable (optional)", ["(none)"] + dp.grouping_candidates(d), key="wz_gcol")
                    snap = tw.data_snapshot(d, ycol, None if gcol == "(none)" else gcol)
                    show_df(fmt(snap["table"], 3))
                    lev = f", Levene p = {snap['levene_p']:.3f}" if snap["levene_p"] == snap["levene_p"] else ""
                    st.caption(f"Smallest group n = {snap['min_n']}; normality looks {'acceptable' if snap['normal'] == 'yes' else 'questionable' if snap['normal'] == 'no' else 'unclear'}{lev}.")
                    snap_used = st.checkbox("Use these results to answer the two questions below", key="wz_use")
        if snap_used:
            normal, small_n = snap["normal"], snap["small_n"]
            st.info(f"Using your data: normality = {normal}, small sample = {'yes' if small_n else 'no'}.")
        else:
            normal = {"Yes / roughly": "yes", "No, clearly skewed": "no", "Not sure": "unsure"}[
                st.radio("Is the outcome roughly normally distributed?", ["Yes / roughly", "No, clearly skewed", "Not sure"], horizontal=True, key="wz_norm")]
            small_n = st.radio("How many cases do you have per group (or in total for a single group)?", ["30 or more", "Fewer than 30"], horizontal=True, key="wz_n") == "Fewer than 30"
    elif goal == g[5] or goal == g[4]:
        st.caption("No further details needed.")

rec = tw.recommend(goal, outcome, groups, design, normal, small_n, outcome == "numeric", outcome == "categorical", scale_task)

st.write("")
with st.container(border=True):
    st.markdown(f"##### Recommended: {rec.title} &nbsp; " + (pill("available in this app", "ok") if rec.available else pill("not in this app yet", "na")),
                unsafe_allow_html=True)
    st.write(rec.why)
    c1, c2 = st.columns(2)
    with c1:
        if rec.assumptions:
            st.markdown("**Assumptions to check**\n" + "\n".join(f"- {a}" for a in rec.assumptions))
        if rec.effect_size:
            st.markdown(f"**Effect size to report:** {rec.effect_size}")
    with c2:
        if rec.alternatives:
            st.markdown("**Alternatives**\n" + "\n".join(f"- {a}" for a in rec.alternatives))
        if rec.report_as:
            st.markdown(f"**APA format:** `{rec.report_as}`")
    for n in rec.notes:
        callout(n, "warn")
    if rec.available and rec.page:
        if st.button(f"Open {rec.title.split('(')[0].strip()}", type="primary", key="wz_open"):
            if rec.test_key:
                ss["ht_test"] = rec.test_key
            st.switch_page(rec.page)

with st.expander("Quick reference: common tests"):
    import pandas as pd
    show_df(pd.DataFrame([
        ("Two independent groups, numeric outcome", "Independent t-test (Welch)", "Mann–Whitney U"),
        ("Same people twice, numeric outcome", "Paired t-test", "Wilcoxon signed-rank"),
        ("3+ independent groups, numeric outcome", "One-way ANOVA", "Kruskal–Wallis*"),
        ("3+ repeated measures", "Repeated-measures ANOVA*", "Friedman*"),
        ("Two categorical variables", "Chi-square", "Fisher's exact"),
        ("Two numeric variables", "Pearson r", "Spearman ρ"),
        ("Predict numeric outcome", "Linear regression", "—"),
        ("Predict yes/no outcome", "Logistic regression", "—"),
    ], columns=["Situation", "Parametric / standard", "Non-parametric / alternative"]))
    st.caption("* Not yet available in this app.")
