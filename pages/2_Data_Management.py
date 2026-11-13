"""Data Management: upload, preview, profile, quality checks, cleaning and export."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from components.cards import callout, card_title, kpi_card
from components.layout import page_header
from components.tables import show_df, show_plot
from modules import data_processing as dp
from modules.sample_data import SAMPLE_DICTIONARY
from modules.state import add_dataset, load_sample, log_activity, remove_dataset
from modules.visualization import _theme

page_header("Data Management", "Upload, inspect, and clean your datasets")
ss = st.session_state

tab_up, tab_prev, tab_prof, tab_q, tab_clean = st.tabs(
    ["Upload", "Preview", "Column profile", "Quality checks", "Clean & export"])

# ------------------------------------------------------------------ upload --
with tab_up:
    c1, c2 = st.columns([3, 2])
    with c1, st.container(border=True):
        card_title("Upload a dataset", "CSV or Excel (.xlsx). Rows = respondents, columns = variables.")
        file = st.file_uploader("Choose a file", type=["csv", "xlsx"], key="dm_file", label_visibility="collapsed")
        if file is not None:
            raw = file.getvalue()
            if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"File is larger than {config.MAX_UPLOAD_MB} MB.")
            else:
                sheet = None
                try:
                    if file.name.lower().endswith(".xlsx"):
                        sheets = dp.excel_sheet_names(raw)
                        sheet = st.selectbox("Sheet", sheets, key="dm_sheet") if len(sheets) > 1 else sheets[0]
                    df_new = dp.load_dataset(raw, file.name, sheet)
                    st.success(f"Read {len(df_new):,} rows × {df_new.shape[1]} columns.")
                    for lvl, msg in dp.validate_dataframe(df_new):
                        {"warning": st.warning, "info": st.info, "error": st.error}[lvl](msg)
                    name = st.text_input("Dataset name", value=file.name.rsplit(".", 1)[0], key="dm_name")
                    if st.button("Add dataset", type="primary", key="dm_add"):
                        add_dataset(name.strip() or file.name, df_new)
                        st.rerun()
                except dp.DataLoadError as exc:
                    st.error(str(exc))
    with c2, st.container(border=True):
        card_title("Try the sample dataset", "Simulated data · ~100 respondents")
        st.caption("Includes demographics, sleep, stress, motivation, productivity and eight Likert items. "
                   "It contains a few deliberate data-quality problems for you to find and clean.")
        if st.button("Load sample dataset", key="dm_sample"):
            load_sample()
            st.rerun()

    if ss.datasets:
        with st.container(border=True):
            card_title("Loaded datasets")
            rows = [{"Dataset": n, "Rows": len(d), "Columns": d.shape[1],
                     "Missing %": round(dp.missing_percentage(d), 1), "Cleaned copy": "Yes" if n in ss.cleaned else "No"}
                    for n, d in ss.datasets.items()]
            show_df(pd.DataFrame(rows))
    if any(n.startswith("Sample") for n in ss.datasets):
        with st.expander("Sample dataset data dictionary (simulated data)"):
            show_df(SAMPLE_DICTIONARY)

if not ss.datasets:
    st.stop()

# ------------------------------------------------------- dataset selection --
names = list(ss.datasets)
idx = names.index(ss.active_dataset) if ss.active_dataset in names else 0
c1, c2, c3 = st.columns([3, 2, 1])
active = c1.selectbox("Active dataset", names, index=idx, key="dm_active")
ss.active_dataset = active
orig = ss.datasets[active]
has_clean = active in ss.cleaned
view_clean = c2.radio("Version", ["Original", "Cleaned"], horizontal=True, key="dm_version",
                      disabled=not has_clean) == "Cleaned" if has_clean else False
df = ss.cleaned[active] if view_clean else orig
if c3.button("Remove dataset", key="dm_remove"):
    remove_dataset(active)
    st.rerun()

# ----------------------------------------------------------------- preview --
with tab_prev:
    parts = dp.split_columns(df)
    kind = {c: k for k, cols in parts.items() for c in cols}
    sym = {"numeric": "#", "identifier": "ID", "categorical": "Aa", "text": "Aa", "datetime": "⏱"}
    c1, c2, c3, c4 = st.columns([2, 1.5, 1, 1])
    q = c1.text_input("Search", placeholder="Filter rows containing…", key="dm_search")
    sort_col = c2.selectbox("Sort by", ["(none)"] + list(df.columns), key="dm_sort")
    asc = c3.toggle("Ascending", value=True, key="dm_asc")
    per = c4.selectbox("Rows per page", [25, 50, 100, 250], key="dm_per")
    view = df
    if q:
        mask = df.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False, regex=False)).any(axis=1)
        view = df[mask]
    if sort_col != "(none)":
        view = view.sort_values(sort_col, ascending=asc, na_position="last")
    pages_n = max((len(view) - 1) // per + 1, 1)
    page = st.number_input("Page", 1, pages_n, 1, key="dm_page")
    shown = view.iloc[(page - 1) * per: page * per].copy()
    shown.columns = [f"{sym[kind[c]]}  {c}" for c in shown.columns]
    show_df(shown, height=460)
    st.caption(f"Page {page} of {pages_n} · Result set: {len(view):,} rows × {df.shape[1]} columns"
               + (" · cleaned version" if view_clean else ""))

# ----------------------------------------------------------------- profile --
with tab_prof:
    parts = dp.split_columns(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Rows", f"{len(df):,}", "respondents", "users")
    with c2: kpi_card("Columns", df.shape[1], "variables", "columns", "green")
    with c3: kpi_card("Numeric", len(parts["numeric"]), "variables", "database")
    with c4: kpi_card("Categorical", len(parts["categorical"]), "variables", "copy", "amber")
    st.write("")
    with st.container(border=True):
        card_title("Column profile", "Data types, variable types and missing values")
        show_df(dp.profile_columns(df))

# ----------------------------------------------------------------- quality --
with tab_q:
    nums = dp.numeric_columns(df)
    with st.expander("Define valid ranges (optional)"):
        st.caption("Values outside these limits are flagged as potentially invalid. Nothing is changed automatically.")
        ranges = ss.valid_ranges.setdefault(active, {})
        pick = st.multiselect("Variables with a known valid range", nums, default=list(ranges), key="dm_rng_cols")
        new_ranges = {}
        for c in pick:
            a, b = st.columns(2)
            lo = a.number_input(f"{c} – minimum", value=float(ranges.get(c, (df[c].min(), None))[0] if c in ranges else df[c].min()), key=f"dm_lo_{c}")
            hi = b.number_input(f"{c} – maximum", value=float(ranges.get(c, (None, df[c].max()))[1] if c in ranges else df[c].max()), key=f"dm_hi_{c}")
            new_ranges[c] = (lo, hi)
        ss.valid_ranges[active] = new_ranges
    inv = dp.find_invalid_values(df, ss.valid_ranges.get(active))
    dups = dp.duplicate_count(df)
    ms = dp.missing_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Missing cells", f"{dp.missing_percentage(df):.1f}%", f"{int(ms['Missing'].sum())} cells", "alert", "amber")
    with c2: kpi_card("Columns with gaps", int((ms["Missing"] > 0).sum()), "of " + str(df.shape[1]), "columns")
    with c3: kpi_card("Duplicate rows", dups, "exact duplicates", "copy")
    with c4: kpi_card("Invalid flags", len(inv), "potential problems", "alert", "amber")
    st.write("")
    l, r = st.columns(2)
    with l, st.container(border=True):
        card_title("Missing values per column")
        m = ms[ms["Missing"] > 0]
        if m.empty:
            callout("No missing values.", "ok")
        else:
            fig = go.Figure(go.Bar(x=m["Column"], y=m["Missing"], marker_color=config.COLORS["warning"]))
            show_plot(_theme(fig, "", "", "Missing", legend=False), key="dm_missing_chart")
            show_df(m)
    with r, st.container(border=True):
        card_title("Potentially invalid values", "Flags only — review before cleaning")
        if inv.empty:
            callout("No invalid values detected.", "ok")
        else:
            show_df(inv)
    with st.container(border=True):
        card_title("Duplicate records", "Every row involved in a duplicate group")
        if dups == 0:
            callout("No duplicate rows.", "ok")
        else:
            show_df(dp.duplicate_rows(df))

# ------------------------------------------------------------------- clean --
with tab_clean:
    callout("<b>Your original data is never modified.</b> Cleaning creates a separate cleaned copy, and only after you confirm.", "info")
    parts = dp.split_columns(orig)
    with st.container(border=True):
        card_title("1 · Duplicates and invalid values")
        a, b = st.columns(2)
        drop_dups = a.checkbox(f"Remove duplicate rows ({dp.duplicate_count(orig)} found)", key="cl_dups")
        inv_o = dp.find_invalid_values(orig, ss.valid_ranges.get(active))
        inv_nan = b.checkbox(f"Set flagged invalid values to missing ({len(inv_o)} found)", key="cl_inv")
    with st.container(border=True):
        card_title("2 · Missing values", "Mean/median imputation reduces variability; consider the impact on your analysis.")
        a, b = st.columns(2)
        num_m = a.selectbox("Numeric variables", ["keep", "drop_rows", "mean", "median", "mode"], key="cl_num",
                            format_func=lambda x: {"keep": "Leave as missing", "drop_rows": "Remove rows with missing",
                                                   "mean": "Fill with mean", "median": "Fill with median", "mode": "Fill with mode"}[x])
        cat_m = b.selectbox("Categorical variables", ["keep", "drop_rows", "mode", "label"], key="cl_cat",
                            format_func=lambda x: {"keep": "Leave as missing", "drop_rows": "Remove rows with missing",
                                                   "mode": "Fill with most frequent", "label": "Label as “Missing”"}[x])
        with_gaps = [c for c in orig.columns if orig[c].isna().any() and c not in parts["identifier"]]
        targets = st.multiselect("Apply to columns", with_gaps, default=with_gaps, key="cl_cols")
    with st.container(border=True):
        card_title("3 · Formatting")
        a, b, c = st.columns(3)
        strip = a.checkbox("Trim whitespace in text", key="cl_strip")
        names_std = b.checkbox("Standardize column names (snake_case)", key="cl_names")
        case = c.selectbox("Text case", ["keep", "lower", "upper", "title"], key="cl_case")
        text_cols = [c for c in orig.columns if c in parts["categorical"] + parts["text"]]
        to_num = st.multiselect("Convert text columns to numeric", text_cols, key="cl_tonum")

    opts = dp.CleaningOptions(drop_duplicates=drop_dups, invalid_to_nan=inv_nan, ranges=ss.valid_ranges.get(active, {}),
                              numeric_missing=num_m, categorical_missing=cat_m, missing_columns=targets,
                              strip_text=strip, text_case=case, standardize_names=names_std, to_numeric=to_num)
    preview, log = dp.apply_cleaning(orig, opts)
    with st.container(border=True):
        card_title("Preview of changes", "Nothing is saved until you confirm below")
        for line in log:
            st.markdown(f"- {line}")
        confirm = st.checkbox("I confirm I want to create a cleaned copy (the original stays unchanged)", key="cl_confirm")
        if st.button("Create cleaned copy", type="primary", disabled=not confirm, key="cl_apply"):
            ss.cleaned[active] = preview
            log_activity("Data cleaned", active, "; ".join(log[:-1])[:140] or "No changes")
            st.success("Cleaned copy created. Analysis pages now use it by default (toggle “Use cleaned version”).")
            st.rerun()
    if has_clean:
        with st.container(border=True):
            card_title("Export and manage")
            d = ss.cleaned[active]
            a, b, c, e = st.columns(4)
            a.download_button("Download cleaned CSV", dp.to_csv_bytes(d), f"{active}_cleaned.csv", "text/csv", key="dl_csv")
            b.download_button("Download cleaned Excel", dp.to_excel_bytes(d), f"{active}_cleaned.xlsx",
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
            if c.button("Discard cleaned copy", key="cl_discard"):
                ss.cleaned.pop(active, None)
                st.rerun()
            replace = e.checkbox("Replace original", key="cl_replace_chk", help="Overwrites the stored original with the cleaned copy.")
            if replace and e.button("Confirm replace", key="cl_replace"):
                ss.datasets[active] = ss.cleaned.pop(active)
                log_activity("Original replaced", active, "User replaced the original with the cleaned copy")
                st.rerun()
