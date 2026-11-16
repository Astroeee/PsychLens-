"""Report Generator: assemble saved analyses into an APA-style report (PDF + CSV)."""
from datetime import datetime

import streamlit as st

from components.cards import callout, card_title
from components.layout import page_header
from components.selectors import is_simulated, select_dataset
from components.tables import show_df
from modules import data_processing as dp
from modules import report_generator as rg
from modules.state import log_activity, setting

page_header("Report Generator", "Compile your saved analyses into an APA-style report")
ss = st.session_state
label, df = select_dataset("rep")

KIND_NAMES = {"descriptives": "Descriptive statistics", "reliability": "Reliability", "correlation": "Correlation",
              "test": "Hypothesis test", "chart": "Chart",
              "regression": "Regression", "factor": "Factor analysis", "power": "Power analysis"}
saved = ss.results
base = label.replace(" (cleaned)", "")

with st.container(border=True):
    card_title("1 · Report details")
    a, b = st.columns(2)
    title = a.text_input("Report title", value="Statistical Analysis Report", key="rep_title")
    author = b.text_input("Author", value=setting("researcher_name"), key="rep_author")
    inst = st.text_input("Institution", value=setting("institution"), key="rep_inst")

with st.container(border=True):
    card_title("2 · Choose what to include", "Add analyses with “Save to report” on any analysis page")
    if not saved:
        callout("No saved analyses yet. The report will contain the dataset overview only. Run an analysis and press <b>Save to report</b>.", "info")
        include = []
    else:
        show_only = st.checkbox("Show only analyses from the selected dataset", value=True, key="rep_filter")
        pool = [r for r in saved if not show_only or r["kind"] == "power" or r["dataset"].replace(" (cleaned)", "") == base]
        include = []
        for r in pool:
            c1, c2, c3 = st.columns([0.6, 5, 1])
            keep = c1.checkbox("", value=True, key=f"rep_inc_{r['id']}", label_visibility="collapsed")
            c2.markdown(f"**{KIND_NAMES[r['kind']]}** · {r['title']}  \n<span style='color:#6B7280;font-size:.8rem'>{r['dataset']} · {r['time']:%H:%M}</span>",
                        unsafe_allow_html=True)
            if c3.button("Remove", key=f"rep_rm_{r['id']}"):
                ss.results = [x for x in ss.results if x["id"] != r["id"]]
                st.rerun()
            if keep:
                include.append(r)
        if not pool:
            st.caption("No saved analyses for this dataset.")

with st.container(border=True):
    card_title("3 · Researcher notes (optional)", "Included verbatim in your own words. The report never writes conclusions for you.")
    notes = st.text_area("Notes", height=140, key="rep_notes", label_visibility="collapsed",
                         placeholder="e.g., Participants were recruited from … Separate paragraphs with a blank line.")

parts = dp.split_columns(df)
overview = {"name": base, "rows": len(df), "cols": df.shape[1], "numeric": len(parts["numeric"]),
            "categorical": len(parts["categorical"]), "missing_pct": dp.missing_percentage(df),
            "duplicates": dp.duplicate_count(df), "cleaned": label.endswith("(cleaned)"), "simulated": is_simulated(base)}
rep = rg.build_report({"title": title, "author": author, "institution": inst}, overview, include, notes, setting("decimals"))

tab_prev, tab_exp = st.tabs(["Preview", "Export"])
with tab_prev:
    html = rg.report_to_html(rep)
    if hasattr(st, "html"):
        st.html(html)
    else:  # older Streamlit: markdown needs the HTML on a single line
        st.markdown(html.replace("\n", ""), unsafe_allow_html=True)
with tab_exp:
    with st.container(border=True):
        card_title("Download")
        try:
            pdf = rg.report_to_pdf(rep)
        except Exception as exc:  # noqa: BLE001
            pdf = None
            st.error(f"PDF could not be generated: {exc}")
        files = rg.entries_to_csv_files(include)
        c1, c2 = st.columns(2)
        if pdf:
            if c1.download_button("Download PDF report", pdf, f"{title.replace(' ', '_')}.pdf", "application/pdf", type="primary", key="rep_pdf"):
                log_activity("Report exported", base, "PDF")
        if files:
            c2.download_button("Download all tables (ZIP of CSVs)", rg.zip_files(files), "tables.zip", "application/zip", key="rep_zip")
            for name, data in files.items():
                st.download_button(name, data, name, "text/csv", key=f"rep_csv_{name}")
        else:
            c2.caption("CSV export becomes available once you save descriptive, reliability, correlation or test results.")
