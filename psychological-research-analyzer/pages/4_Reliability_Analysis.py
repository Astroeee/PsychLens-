"""Reliability analysis: Cronbach's alpha and item statistics."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config
from components.cards import apa_box, callout, card_title, kpi_card
from components.layout import page_header
from components.selectors import select_dataset
from components.tables import fmt, show_df, show_plot
from modules import data_processing as dp
from modules import reliability as rel
from modules.state import save_result, setting
from modules.visualization import _theme

page_header("Reliability Analysis", "Evaluate the internal consistency of questionnaire scales")
label, df = select_dataset("rel")
dec, conf = setting("decimals"), setting("confidence")
nums = dp.numeric_columns(df)
likert = list(dp.detect_likert_columns(df))

with st.container(border=True):
    card_title("Scale definition", "Select the items that make up one scale")
    sel = st.multiselect("Questionnaire items", nums, default=likert, key="rel_items")
    name = st.text_input("Scale name", value="Scale", key="rel_name")
    with st.expander("Reverse-score items (optional)"):
        st.caption("Reverse-worded items must be recoded before computing alpha. This is applied to a temporary copy; your dataset is not changed.")
        rev = st.multiselect("Items to reverse", sel, key="rel_rev")
        a, b = st.columns(2)
        smin = a.number_input("Scale minimum", value=1.0, key="rel_min")
        smax = b.number_input("Scale maximum", value=5.0, key="rel_max")

if len(sel) < 2:
    callout("Select at least two items to compute reliability.", "info")
    st.stop()

work = rel.reverse_score(df, rev, smin, smax) if rev else df
try:
    res = rel.reliability_analysis(work, sel, conf)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("Cronbach's α", f"{res.alpha:.{dec + 1}f}", f"{conf * 100:.0f}% CI [{res.ci[0]:.2f}, {res.ci[1]:.2f}]", "check")
with c2: kpi_card("Number of items", res.n_items, "in the scale", "columns", "green")
with c3: kpi_card("Valid respondents", res.n_valid, f"{res.n_excluded} excluded (listwise)", "users")
with c4: kpi_card("Mean inter-item r", f"{res.mean_inter_item_r:.2f}", f"standardized α = {res.alpha_standardized:.2f}", "copy", "amber")
st.write("")
apa_box(rel.apa_summary(res))
for w in res.warnings:
    callout(w, "warn")

l, r = st.columns([3, 2])
with l, st.container(border=True):
    card_title("Item statistics", "Corrected item-total correlation and alpha if the item were deleted")
    show_df(fmt(res.item_stats, 3))
with r, st.container(border=True):
    card_title("Alpha if item deleted", f"Dashed line = alpha of the full scale ({res.alpha:.2f})")
    s = res.item_stats
    fig = go.Figure(go.Bar(x=s["Item"], y=s["Alpha if item deleted"], marker_color=config.COLORS["primary"]))
    fig.add_hline(y=res.alpha, line_dash="dash", line_color=config.COLORS["danger"])
    lo = float(np.nanmin(s["Alpha if item deleted"])) if s["Alpha if item deleted"].notna().any() else 0
    fig.update_yaxes(range=[max(0, min(lo, res.alpha) - 0.05), min(1, max(float(np.nanmax(s["Alpha if item deleted"])) if s["Alpha if item deleted"].notna().any() else 1, res.alpha) + 0.03)])
    show_plot(_theme(fig, "", "Item", "Alpha", legend=False), key="rel_ald")

with st.container(border=True):
    card_title("Inter-item correlations")
    z = res.inter_item_corr
    fig = go.Figure(go.Heatmap(z=z.to_numpy(), x=list(z.columns), y=list(z.index), zmin=-1, zmax=1,
                               colorscale=[[0, config.COLORS["danger"]], [0.5, "#FFFFFF"], [1, config.COLORS["primary"]]],
                               text=z.round(2).astype(str).to_numpy(), texttemplate="%{text}"))
    fig.update_yaxes(autorange="reversed")
    show_plot(_theme(fig, "", "", "", legend=False), key="rel_heat")

with st.container(border=True):
    card_title("How to interpret these results")
    callout(rel.interpret_alpha(res.alpha), "info")
    st.markdown(
        "- **Alpha is not validity.** It describes internal consistency only.\n"
        "- **Number of items matters:** alpha rises as items are added, even if they are weakly related.\n"
        "- **Very high alpha (> .90)** can signal redundant items rather than a better scale.\n"
        "- **Corrected item-total r** below about .30 suggests an item relates weakly to the rest of the scale.\n"
        "- **Alpha if item deleted:** a higher value suggests the item lowers consistency, but decisions about removing "
        "items should rest on content and theory, not on alpha alone.\n"
        "- **Assumptions:** items should measure one construct (consider factor analysis) and be scored in the same direction.")
    show_df(rel.GUIDELINES)
    try:
        import pingouin as pg
        a, ci = pg.cronbach_alpha(data=work[sel].dropna())
        st.caption(f"Cross-check with Pingouin: α = {a:.4f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}] (this app: {res.alpha:.4f}).")
    except Exception:  # noqa: BLE001
        pass

if st.button("Save to report", type="primary", key="rel_save"):
    save_result("reliability", name, label, apa=rel.apa_summary(res), item_table=res.item_stats, n_excluded=res.n_excluded,
                interpretation=rel.interpret_alpha(res.alpha), warnings=res.warnings)
    st.success("Saved. Find it in the Report Generator.")
