"""Power analysis: sample size, power and detectable effect size."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from components.cards import callout, card_title, kpi_card
from components.layout import page_header
from components.tables import show_df, show_plot
from modules import power_analysis as pa
from modules.state import save_result, setting
from modules.visualization import _theme

page_header("Power Analysis", "Plan your sample size before collecting data")
callout("Use this <b>before</b> collecting data. Choose an effect size you can justify from earlier research or the smallest effect that would matter in practice. "
        "Computing power from the effect observed in your own data (“post-hoc power”) is not informative.", "info")

test = st.selectbox("Statistical test", list(pa.TESTS), key="pw_test")
meta = pa.TESTS[test]
sm, md, lg = pa.BENCHMARKS[meta["effect"]]

with st.container(border=True):
    card_title("Inputs")
    solve = st.radio("What do you want to find?", ["Sample size", "Power", "Detectable effect size"], horizontal=True, key="pw_solve")
    a, b, c = st.columns(3)
    effect = None
    if solve != "Detectable effect size":
        bench = a.selectbox("Effect size benchmark", [f"Small ({sm})", f"Medium ({md})", f"Large ({lg})", "Custom"], index=1, key=f"pw_bench_{test}")
        defaults = {"Small": sm, "Medium": md, "Large": lg}
        base = defaults.get(bench.split(" ")[0], md)
        effect = b.number_input(meta["label"], min_value=0.001, max_value=float(pa.EFFECT_MAX[meta["effect"]]), value=float(base), step=0.01,
                                format="%.3f", key=f"pw_eff_{test}_{bench}")
    alpha = c.selectbox("α (significance level)", [0.05, 0.01, 0.001, 0.10], index=[0.05, 0.01, 0.001, 0.10].index(setting("alpha")) if setting("alpha") in [0.05, 0.01, 0.001, 0.10] else 0, key="pw_alpha")
    kw, alt = {}, "two-sided"
    d1, d2, d3 = st.columns(3)
    if solve != "Power":
        power = d1.slider("Target power (1 − β)", 0.50, 0.99, 0.80, 0.01, key="pw_power")
    else:
        power = 0.80
    n_in = None
    if solve != "Sample size":
        unit = "per group" if test.startswith("Independent") else "total participants"
        n_in = d2.number_input(f"Sample size ({unit})", min_value=4, max_value=100000, value=50, key=f"pw_n_{test}")
    if test in ("Independent-samples t-test", "Paired-samples t-test", "One-sample t-test", "Correlation (Pearson)"):
        alt = d3.radio("Tails", ["two-sided", "greater"], horizontal=True, key="pw_alt", format_func=lambda x: "Two-tailed" if x == "two-sided" else "One-tailed")
    e1, e2 = st.columns(2)
    if test == "Independent-samples t-test":
        kw["ratio"] = e1.number_input("Allocation ratio (group 2 ÷ group 1)", 0.2, 5.0, 1.0, 0.1, key="pw_ratio")
    if test == "One-way ANOVA":
        kw["groups"] = int(e1.number_input("Number of groups", 2, 20, 3, key="pw_groups"))
    if test == "Chi-square test":
        kw["df"] = int(e1.number_input("Degrees of freedom", 1, 50, 2, key="pw_df", help="(rows − 1) × (columns − 1) for independence tests"))
    if test == "Multiple regression (R² test)":
        kw["predictors"] = int(e1.number_input("Number of predictors", 1, 50, 3, key="pw_k"))
    dropout = e2.slider("Expected attrition / dropout", 0, 50, 0, 5, format="%d%%", key="pw_drop") / 100 if solve == "Sample size" else 0.0

try:
    res = pa.analyze(test, solve, effect, power, n_in, alpha, alt, dropout, **kw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("Sample size", f"{res.n:,}", res.n_unit, "users")
with c2: kpi_card("Total N", f"{res.n_total:,}", "all participants", "database", "green")
with c3: kpi_card("Power", f"{res.power:.2f}", "probability of detecting the effect", "check")
with c4: kpi_card(meta["label"], f"{res.effect:.3f}", "effect size", "columns", "amber")
st.write("")
with st.container(border=True):
    card_title("Result and wording for your proposal", "Copy and adapt (cite where your effect size comes from)")
    st.text_area("Sentence", res.sentence, height=110, key=f"pw_sent_{solve}_{test}", label_visibility="collapsed")
    for w in res.warnings:
        callout(w, "warn")
    show_df(res.table())

l, r = st.columns([3, 2])
with l, st.container(border=True):
    card_title("Power curve", "How power grows with sample size for this effect")
    n_top = max(res.n * 2, 20)
    ns = np.unique(np.linspace(max(pa.N_MIN[test] + (kw.get("groups", 0) if test == "One-way ANOVA" else 0) + (kw.get("predictors", 0) + 1 if test.startswith("Multiple") else 0), 4), n_top, 60).astype(int))
    curve = pa.power_curve(test, res.effect, ns, alpha, alt, **kw)
    fig = go.Figure(go.Scatter(x=curve["n"], y=curve["Power"], mode="lines", line=dict(color=config.COLORS["primary"], width=3), name="Power"))
    fig.add_hline(y=0.8, line_dash="dot", line_color="#9CA3AF", annotation_text="0.80")
    fig.add_trace(go.Scatter(x=[res.n], y=[res.power], mode="markers", marker=dict(size=12, color=config.COLORS["danger"]), name="Your design"))
    fig.update_yaxes(range=[0, 1.02])
    show_plot(_theme(fig, "", f"Sample size ({res.n_unit})", "Power"), key="pw_curve")
with r, st.container(border=True):
    card_title("Sample size for typical effects", f"{int(round(0.8 * 100))}% power, α = {alpha:g}")
    rows = []
    for name, e in zip(["Small", "Medium", "Large"], pa.BENCHMARKS[meta["effect"]]):
        try:
            n = pa.required_n(test, e, 0.80, alpha, alt, **kw)
            rows.append((f"{name} ({meta['label']} = {e})", n, pa._n_total(test, n, **kw)))
        except ValueError:
            rows.append((f"{name} ({e})", "—", "—"))
    show_df(pd.DataFrame(rows, columns=["Effect", "n", "Total N"]).astype(str))
    st.caption("Benchmarks are Cohen's conventions and are only a fallback; field-specific effects are usually smaller.")

if st.button("Save to report", type="primary", key="pw_save"):
    save_result("power", f"{test} ({solve.lower()})", "Study planning", table=res.table(), sentence=res.sentence, warnings=res.warnings)
    st.success("Saved. Find it in the Report Generator.")
