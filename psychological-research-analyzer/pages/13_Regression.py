"""Linear, hierarchical and logistic regression with diagnostics."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import config
from components.cards import apa_box, callout, card_title, kpi_card
from components.layout import page_header
from components.selectors import select_dataset
from components.tables import assumptions_table, fmt, show_df, show_plot
from modules import data_processing as dp
from modules import regression as rg
from modules.state import save_result, setting
from modules.visualization import _theme

page_header("Regression", "Predict outcomes and estimate the unique contribution of each predictor")
label, df = select_dataset("rg")
alpha, conf, dec = setting("alpha"), setting("confidence"), setting("decimals")
parts = dp.split_columns(df)
nums = parts["numeric"]
pred_pool = nums + parts["categorical"]
C = config.COLORS

model = st.radio("Model type", ["Linear regression", "Hierarchical regression", "Logistic regression (binary outcome)"],
                 horizontal=True, key="rg_model")


def ref_levels(preds, key):
    refs = {}
    cats = [p for p in preds if rg.is_categorical(df[p])]
    if cats:
        with st.expander("Reference categories for categorical predictors"):
            for p in cats:
                lv = sorted(df[p].dropna().astype(str).unique())
                refs[p] = st.selectbox(f"{p}: reference level", lv, key=f"{key}_ref_{p}")
    return refs


def coef_plot(coef, kind="B"):
    c = coef[coef["Term"] != "Intercept"]
    if kind == "B":
        x, lo, hi, xl = c["B"], c["CI lower"], c["CI upper"], "Coefficient (B) with CI"
    else:
        x, lo, hi, xl = c["Odds ratio"], c["OR CI lower"], c["OR CI upper"], "Odds ratio with CI (log scale)"
    fig = go.Figure(go.Scatter(x=x, y=c["Term"], mode="markers", marker=dict(color=C["primary"], size=9),
                               error_x=dict(type="data", symmetric=False, array=hi - x, arrayminus=x - lo, color="#9CA3AF")))
    fig.add_vline(x=0 if kind == "B" else 1, line_dash="dash", line_color=C["danger"])
    fig.update_yaxes(autorange="reversed")
    if kind != "B":
        fig.update_xaxes(type="log")
    return _theme(fig, "", xl, "", legend=False)


# =========================================================== linear / hier ===
if not model.startswith("Logistic"):
    with st.container(border=True):
        card_title("Model specification")
        dv = st.selectbox("Outcome (numeric)", nums, key="rg_dv")
        pool = [c for c in pred_pool if c != dv]
        if model == "Linear regression":
            preds = st.multiselect("Predictors", pool, key=f"rg_preds_{dv}")
            blocks = [preds]
        else:
            st.caption("Enter predictors in steps (blocks). Each step adds to the previous ones; ΔR² shows what each block adds.")
            b1 = st.multiselect("Block 1", pool, key=f"rg_b1_{dv}")
            b2 = st.multiselect("Block 2 (adds to block 1)", [c for c in pool if c not in b1], key=f"rg_b2_{dv}")
            b3 = st.multiselect("Block 3 (optional)", [c for c in pool if c not in b1 + b2], key=f"rg_b3_{dv}")
            blocks = [b1, b2, b3]
            preds = b1 + b2 + b3
        refs = ref_levels(preds, "rg")
    if not preds or (model != "Linear regression" and not blocks[0]):
        callout("Select at least one predictor" + (" in Block 1." if model != "Linear regression" else "."), "info")
        st.stop()
    try:
        res = rg.fit_ols(df, dv, preds, refs, conf, alpha)
        hier = rg.hierarchical_ols(df, dv, blocks, refs) if model != "Linear regression" else None
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("R²", f"{res.r2:.{dec}f}", f"adjusted {res.adj_r2:.{dec}f}", "check")
    with c2: kpi_card("F test", f"{res.f:.2f}", f"df = {res.f_df[0]}, {res.f_df[1]}", "columns", "green")
    with c3: kpi_card("p (model)", "< .001" if res.f_p < .001 else f"{res.f_p:.3f}", "overall model", "alert", "amber")
    with c4: kpi_card("Cases used", res.n, f"{res.n_excluded} excluded", "users")
    st.write("")
    apa_box(rg.apa_ols(res))
    for w in res.warnings:
        callout(w, "warn")

    tab_c, tab_d, tab_p = st.tabs(["Coefficients" if hier is None else "Steps & coefficients", "Assumption checks", "Diagnostic plots"])
    with tab_c:
        if hier is not None:
            with st.container(border=True):
                card_title("Hierarchical regression: model comparison")
                show_df(fmt(hier, 4))
        with st.container(border=True):
            card_title("Coefficients" + (" (final model)" if hier is not None else ""), f"B = unstandardized; β = standardized (numeric predictors only); {conf * 100:.0f}% CIs")
            show_df(fmt(res.coef, 4))
            show_plot(coef_plot(res.coef), key="rg_forest")
    with tab_d:
        with st.container(border=True):
            card_title("Assumption and diagnostic checks", "Informational: look at the plots before deciding")
            show_df(fmt(assumptions_table([{"Check": r["Check"], "Test": r["Test / statistic"], "Statistic": r["Statistic"], "p": r["p"],
                                            "Result": r["Result"], "Note": ""} for _, r in res.diagnostics.iterrows()]).drop(columns=["Note"]), 3))
            st.caption("Durbin–Watson is only meaningful if rows are in a meaningful order (e.g., time).")
    with tab_p:
        d = res.data
        l, r = st.columns(2)
        with l, st.container(border=True):
            card_title("Residuals vs fitted", "Look for a random cloud (no curve or funnel)")
            fig = go.Figure(go.Scatter(x=d["Fitted"], y=d["Std. residual"], mode="markers", marker=dict(color=C["primary"], opacity=.7)))
            fig.add_hline(y=0, line_dash="dash", line_color=C["danger"])
            show_plot(_theme(fig, "", "Fitted values", "Standardized residual", legend=False), key="rg_resfit")
        with r, st.container(border=True):
            card_title("Normal Q–Q plot of residuals", "Points should follow the line")
            (osm, osr), (sl, ic, _) = stats.probplot(d["Std. residual"], dist="norm")
            fig = go.Figure(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color=C["primary"], opacity=.7)))
            fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[ic + sl * osm.min(), ic + sl * osm.max()], mode="lines", line=dict(color=C["danger"])))
            show_plot(_theme(fig, "", "Theoretical quantiles", "Sample quantiles", legend=False), key="rg_qq")
        l, r = st.columns(2)
        with l, st.container(border=True):
            card_title("Cook's distance", "Cases above 4/n deserve a look")
            fig = go.Figure(go.Bar(x=list(range(1, len(d) + 1)), y=d["Cook's D"], marker_color=C["primary"]))
            fig.add_hline(y=4 / res.n, line_dash="dash", line_color=C["danger"])
            show_plot(_theme(fig, "", "Case", "Cook's D", legend=False), key="rg_cook")
        with r, st.container(border=True):
            if len(preds) == 1 and not rg.is_categorical(df[preds[0]]):
                card_title("Scatterplot with regression line")
                xs = pd.to_numeric(df.loc[d.index, preds[0]])
                fig = go.Figure(go.Scatter(x=xs, y=df.loc[d.index, dv], mode="markers", marker=dict(color=C["primary"], opacity=.7), name="Cases"))
                o = np.argsort(xs.to_numpy())
                fig.add_trace(go.Scatter(x=xs.to_numpy()[o], y=d["Fitted"].to_numpy()[o], mode="lines", line=dict(color=C["danger"]), name="Fitted"))
                show_plot(_theme(fig, "", preds[0], dv), key="rg_scatter")
            else:
                card_title("Observed vs fitted")
                fig = go.Figure(go.Scatter(x=d["Fitted"], y=df.loc[d.index, dv], mode="markers", marker=dict(color=C["primary"], opacity=.7)))
                show_plot(_theme(fig, "", "Fitted", dv, legend=False), key="rg_obsfit")
        flag = d[(d["Cook's D"] > 4 / res.n) | (d["Std. residual"].abs() > 3)]
        if len(flag):
            with st.expander(f"Potentially influential cases ({len(flag)})"):
                show_df(fmt(flag.reset_index(names="Row"), 3))
    with st.container(border=True):
        card_title("Interpretation", "Generated from the statistics only")
        st.write(rg.interpret_ols(res))
    if st.button("Save to report", type="primary", key="rg_save"):
        extra = {"Model fit diagnostics": res.diagnostics}
        if hier is not None:
            extra = {"Hierarchical model comparison": hier, **extra}
        save_result("regression", f"{'Hierarchical' if hier is not None else 'Linear'} regression: {dv}", label, model_type="linear",
                    apa=rg.apa_ols(res), coef=res.coef, extra_tables=extra, interpretation=rg.interpret_ols(res), warnings=res.warnings)
        st.success("Saved. Find it in the Report Generator.")

# ================================================================ logistic ===
else:
    binary = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    if not binary:
        callout("No variable with exactly two categories was found for the outcome.", "warn")
        st.stop()
    with st.container(border=True):
        card_title("Model specification")
        a, b = st.columns(2)
        dv = a.selectbox("Outcome (two categories)", binary, key="rg_ldv")
        levels = sorted(df[dv].dropna().astype(str).unique())
        event = b.selectbox("Event level (predict the odds of…)", levels, index=len(levels) - 1, key=f"rg_ev_{dv}")
        pool = [c for c in pred_pool if c != dv]
        preds = st.multiselect("Predictors", pool, key=f"rg_lpreds_{dv}")
        thr = st.slider("Classification cut-off", 0.1, 0.9, 0.5, 0.05, key="rg_thr")
        refs = ref_levels(preds, "rgl")
    if not preds:
        callout("Select at least one predictor.", "info")
        st.stop()
    try:
        res = rg.fit_logistic(df, dv, preds, event, refs, thr, conf, alpha)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    fit = res.fit.set_index("Statistic")["Value"]
    k = len(res.coef) - 1
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Nagelkerke R²", f"{fit['Nagelkerke R²']:.{dec}f}", f"McFadden {fit['McFadden pseudo-R²']:.2f}", "check")
    with c2: kpi_card("Model χ²", f"{fit[f'Likelihood-ratio χ²({k})']:.2f}", f"df = {k}", "columns", "green")
    with c3: kpi_card("AUC", f"{res.auc:.2f}", "discrimination", "alert", "amber")
    with c4: kpi_card("Cases used", res.n, f"{res.n_events} events · {res.n_excluded} excluded", "users")
    st.write("")
    apa_box(rg.apa_logit(res))
    for w in res.warnings:
        callout(w, "warn")
    tab_c, tab_f, tab_r = st.tabs(["Coefficients & odds ratios", "Model fit & classification", "ROC curve"])
    with tab_c:
        with st.container(border=True):
            card_title("Coefficients", f"Odds of {dv} = '{res.event}' vs '{res.reference}'; {conf * 100:.0f}% Wald CIs")
            show_df(fmt(res.coef, 4))
            show_plot(coef_plot(res.coef, "OR"), key="rg_lforest")
    with tab_f:
        l, r = st.columns(2)
        with l, st.container(border=True):
            card_title("Model fit")
            show_df(fmt(res.fit, 3))
            hl = res.hosmer_lemeshow
            if np.isfinite(hl[0]):
                st.caption(f"Hosmer–Lemeshow: χ²({hl[1]}) = {hl[0]:.2f}, p = {hl[2]:.3f} (p > .05 suggests no evidence of poor calibration; low power with small samples).")
        with r, st.container(border=True):
            card_title("Classification", "Based on the chosen cut-off")
            tbl = res.classification.copy()
            tbl.iloc[:, 1] = tbl.iloc[:, 1].map(lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else v)
            show_df(tbl)
    with tab_r:
        with st.container(border=True):
            roc = rg.roc_curve(res.data["Observed"].to_numpy(), res.data["Predicted probability"].to_numpy())
            fig = go.Figure(go.Scatter(x=roc["FPR"], y=roc["TPR"], mode="lines", name=f"AUC = {res.auc:.2f}", line=dict(color=C["primary"], width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#9CA3AF", dash="dash")))
            show_plot(_theme(fig, "", "False positive rate (1 − specificity)", "True positive rate (sensitivity)"), key="rg_roc")
    with st.container(border=True):
        card_title("Interpretation", "Generated from the statistics only")
        st.write(rg.interpret_logit(res))
    if st.button("Save to report", type="primary", key="rg_lsave"):
        save_result("regression", f"Logistic regression: {dv}", label, model_type="logistic", apa=rg.apa_logit(res), coef=res.coef,
                    extra_tables={"Model fit": res.fit, "Classification": res.classification}, interpretation=rg.interpret_logit(res), warnings=res.warnings)
        st.success("Saved. Find it in the Report Generator.")
