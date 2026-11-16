"""Data Visualization: build, customize and download charts."""
import streamlit as st

from components.cards import callout, card_title
from components.layout import page_header
from components.selectors import select_dataset
from components.tables import show_image, show_plot
from modules import data_processing as dp
from modules import visualization as viz
from modules.state import save_result

page_header("Data Visualization", "Build, customize and download charts")
label, df = select_dataset("viz")
nums = dp.numeric_columns(df)
groupers = dp.grouping_candidates(df)
anycol = list(df.columns)

left, right = st.columns([1, 2.4])
params: dict = {}
with left, st.container(border=True):
    card_title("Chart settings")
    kind = st.selectbox("Chart type", viz.CHART_KINDS, key="viz_kind")
    ok = True
    if kind == "Histogram":
        if not nums: ok = False
        else:
            params["x"] = st.selectbox("Variable", nums, key=f"viz_x_{kind}")
            params["by"] = st.selectbox("Group by (optional)", ["(none)"] + groupers, key=f"viz_by_{kind}")
            params["bins"] = st.slider("Bins", 5, 80, 20, key=f"viz_bins_{kind}")
            params["kde"] = st.checkbox("Overlay density curve", key=f"viz_kde_{kind}")
    elif kind == "Boxplot":
        if not nums: ok = False
        else:
            params["y"] = st.selectbox("Numeric variable", nums, key=f"viz_y_{kind}")
            params["by"] = st.selectbox("Group by (optional)", ["(none)"] + groupers, key=f"viz_by_{kind}")
    elif kind == "Bar chart":
        if not groupers: ok = False
        else:
            params["x"] = st.selectbox("Category (X)", groupers, key=f"viz_x_{kind}")
            yy = st.selectbox("Value (Y, optional)", ["(counts)"] + nums, key=f"viz_y_{kind}")
            params["y"] = None if yy == "(counts)" else yy
            if params["y"]:
                params["agg"] = st.selectbox("Aggregate", viz.AGGREGATES[:3], key=f"viz_agg_{kind}")
            params["by"] = st.selectbox("Split by (optional)", ["(none)"] + [g for g in groupers if g != params["x"]], key=f"viz_by_{kind}")
    elif kind == "Scatterplot":
        if len(nums) < 2: ok = False
        else:
            params["x"] = st.selectbox("X variable", nums, key=f"viz_x_{kind}")
            params["y"] = st.selectbox("Y variable", [c for c in nums if c != params["x"]], key=f"viz_y_{kind}")
            params["by"] = st.selectbox("Color by (optional)", ["(none)"] + groupers, key=f"viz_by_{kind}")
            params["trendline"] = st.checkbox("Add linear trendline", value=True, key=f"viz_trend_{kind}")
    elif kind == "Line graph":
        if not nums: ok = False
        else:
            params["x"] = st.selectbox("X (ordered variable)", anycol, key=f"viz_x_{kind}")
            params["y"] = st.selectbox("Y variable", [c for c in nums if c != params["x"]] or nums, key=f"viz_y_{kind}")
            params["agg"] = st.selectbox("Aggregate Y by X", viz.AGGREGATES[:3], key=f"viz_agg_{kind}")
            params["by"] = st.selectbox("Separate lines by (optional)", ["(none)"] + [g for g in groupers if g != params["x"]], key=f"viz_by_{kind}")
    elif kind == "Correlation heatmap":
        if len(nums) < 2: ok = False
        else:
            params["cols"] = st.multiselect("Variables", nums, default=[c for c in nums if not c.startswith("Q")][:6], key=f"viz_cols_{kind}")
            params["method"] = st.radio("Method", ["pearson", "spearman"], horizontal=True, key=f"viz_method_{kind}")
    else:  # Frequency distribution
        if not groupers: ok = False
        else:
            params["x"] = st.selectbox("Variable", groupers, key=f"viz_x_{kind}")
    st.markdown("---")
    params["title"] = st.text_input("Chart title", value=kind, key=f"viz_title_{kind}")
    if kind != "Correlation heatmap":
        params["xlabel"] = st.text_input("X-axis label (blank = default)", key=f"viz_xl_{kind}")
        params["ylabel"] = st.text_input("Y-axis label (blank = default)", key=f"viz_yl_{kind}")
    if params.get("by") == "(none)":
        params["by"] = None

with right, st.container(border=True):
    if not ok:
        callout("This chart type needs variables that are not available in this dataset.", "warn")
        st.stop()
    if kind == "Correlation heatmap" and len(params.get("cols", [])) < 2:
        callout("Select at least two variables.", "info")
        st.stop()
    try:
        fig = viz.build_figure(kind, df, **params)
        show_plot(fig, key="viz_fig")
        png = viz.render_png(kind, df, **params)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not draw this chart: {exc}")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    plotly_png = viz.figure_to_png(fig)
    c1.download_button("Download PNG", plotly_png or png, "chart.png", "image/png", key="viz_dl_png")
    c2.download_button("Download PNG (print)", png, "chart_print.png", "image/png", key="viz_dl_mpl",
                       help="Static Matplotlib version, ideal for documents.")
    c3.download_button("Interactive HTML", fig.to_html(include_plotlyjs="cdn"), "chart.html", "text/html", key="viz_dl_html")
    if c4.button("Add to report", type="primary", key="viz_save"):
        save_result("chart", params.get("title") or kind, label, png=png)
        st.success("Added to the report.")
    with st.expander("Preview of the print version"):
        show_image(png)
