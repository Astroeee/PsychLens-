"""APA-style report builder: assembles saved results into a document structure,
then renders it as PDF (ReportLab), HTML preview, or CSV files.

Only content produced by the analyses is written to the report. Interpretations
are the neutral sentences generated alongside each test; any additional comments
come verbatim from the researcher's own notes.
"""
from __future__ import annotations

import base64
import io
import re
import zipfile
from datetime import datetime
from html import escape
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- helpers ---
_ITALIC = re.compile(r"(?<![A-Za-z])(t|F|p|r|d|g|U|W|z|N|n|M|SD|V|k|d_z|r_rb)(?=\s?[(=<])")


def apa_markup(text: str, html: bool = False) -> str:
    """Italicize statistical symbols as APA requires (t, p, F, M, SD, N ...)."""
    esc = escape(text) if html else xml_escape(text)
    return _ITALIC.sub(lambda m: f"<i>{m.group(1)}</i>", esc)


def fmt_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Convert numbers to strings; p-value-like columns follow APA (no leading zero)."""
    out = df.copy()
    int_cols = {"N", "Missing", "Modes (count)", "Frequency", "df", "k"}
    for c in out.columns:
        if str(c) in int_cols and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
        elif pd.api.types.is_float_dtype(out[c]):
            name = str(c).lower()
            def f(v, c=c, name=name):
                if pd.isna(v): return ""
                if name.startswith("p") and len(name) <= 14 and abs(v) <= 1:
                    return "< .001" if v < 0.001 else f"{v:.3f}".replace("0.", ".", 1)
                return f"{v:.{decimals}f}"
            out[c] = out[c].map(f)
        else:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else str(v))
    return out


def library_versions() -> str:
    import importlib.metadata as md
    parts = []
    for lib in ("pandas", "numpy", "scipy"):
        try: parts.append(f"{lib} {md.version(lib)}")
        except Exception: pass  # noqa: BLE001
    return ", ".join(parts)


# -------------------------------------------------------------- structure ---
def text(t): return {"type": "text", "text": t}
def table(df, caption, note=""): return {"type": "table", "df": df, "caption": caption, "note": note}
def image(png, caption): return {"type": "image", "png": png, "caption": caption}


def build_report(meta: dict, overview: dict, entries: list[dict], notes: str = "", decimals: int = 2) -> dict:
    """meta: title, author, institution. overview: dataset facts. entries: saved results."""
    sections = []
    tcount = [0]

    def tcap(title):
        tcount[0] += 1
        return f"Table {tcount[0]}", title

    ov = [text(f"Dataset: {overview['name']}. The dataset contains {overview['rows']} respondents (rows) and "
               f"{overview['cols']} variables ({overview['numeric']} numeric, {overview['categorical']} categorical). "
               f"{overview['missing_pct']:.1f}% of all cells are missing and {overview['duplicates']} duplicate row(s) were detected."
               + (" Analyses were run on a cleaned copy of the data; the original file was not modified." if overview.get("cleaned") else
                  " Analyses were run on the data as uploaded."))]
    if overview.get("simulated"):
        ov.append(text("Note: this dataset is simulated for demonstration purposes and contains no real respondents."))
    sections.append({"heading": "Dataset Overview", "blocks": ov})

    def of_kind(k): return [e for e in entries if e["kind"] == k]

    if of_kind("descriptives"):
        blocks = []
        for e in of_kind("descriptives"):
            n, ttl = tcap(f"Descriptive Statistics – {e['title']}")
            blocks.append(table(fmt_df(e["table"], decimals), f"{n}|{ttl}",
                                "Skewness and kurtosis are bias-corrected; kurtosis is excess kurtosis."))
        sections.append({"heading": "Descriptive Statistics", "blocks": blocks})

    if of_kind("reliability"):
        blocks = []
        for e in of_kind("reliability"):
            blocks.append(text(f"{e['title']}: {e['apa']}."))
            n, ttl = tcap(f"Item Statistics – {e['title']}")
            blocks.append(table(fmt_df(e["item_table"], decimals), f"{n}|{ttl}",
                                f"Listwise deletion; {e['n_excluded']} respondent(s) excluded."))
            blocks.append(text(e["interpretation"]))
            blocks += [text("Warning: " + w) for w in e["warnings"]]
        sections.append({"heading": "Reliability Analysis", "blocks": blocks})

    if of_kind("correlation"):
        blocks = []
        for e in of_kind("correlation"):
            n, ttl = tcap(f"{e['method_name']} Correlations – {e['title']}")
            blocks.append(table(e["matrix_apa"], f"{n}|{ttl}", e["note"]))
            n, ttl = tcap(f"Pairwise {e['method_name']} Correlations with Confidence Intervals – {e['title']}")
            blocks.append(table(fmt_df(e["pairs"], 3), f"{n}|{ttl}", e["note"]))
        sections.append({"heading": "Correlation Analysis", "blocks": blocks})

    if of_kind("test"):
        blocks = []
        for e in of_kind("test"):
            r = e["result"]
            blocks.append({"type": "subheading", "text": f"{r.name}: {r.variables}"})
            blocks.append(text(r.hypotheses))
            blocks.append(text(f"Result: {r.apa}."))
            if r.descriptives is not None:
                chi = r.name.startswith("Chi-square")
                n, ttl = tcap(f"{'Contingency Table (Observed Counts)' if chi else 'Descriptive Statistics'} – {r.variables}")
                blocks.append(table(fmt_df(r.descriptives, decimals), f"{n}|{ttl}"))
            a = pd.DataFrame(r.assumptions)[["Check", "Test", "Statistic", "p", "Result"]] if r.assumptions else None
            if a is not None:
                n, ttl = tcap(f"Assumption Checks – {r.variables}")
                blocks.append(table(fmt_df(a, 3), f"{n}|{ttl}"))
            for name, t in r.extra_tables.items():
                if name == "Observed counts":
                    continue
                n, ttl = tcap(f"{name} – {r.variables}")
                blocks.append(table(fmt_df(t, decimals), f"{n}|{ttl}"))
            blocks.append(text(r.interpretation))
            blocks += [text("Warning: " + w) for w in r.warnings]
        sections.append({"heading": "Hypothesis Testing", "blocks": blocks})

    if of_kind("chart"):
        blocks, i = [], 0
        for e in of_kind("chart"):
            i += 1
            blocks.append(image(e["png"], f"Figure {i}|{e['title']}"))
        sections.append({"heading": "Visualizations", "blocks": blocks})

    if notes.strip():
        sections.append({"heading": "Researcher Notes", "blocks": [text(p) for p in notes.strip().split("\n\n")]})

    sections.append({"heading": "Method Note", "blocks": [text(
        f"Analyses were generated with PsychLens using {library_versions()}. Statistical results are reported as computed; "
        "interpretive sentences are limited to what the statistics support and do not replace the researcher's judgment.")]})
    return {**meta, "date": datetime.now().strftime("%B %d, %Y"), "sections": sections}


# -------------------------------------------------------------------- HTML --
def report_to_html(rep: dict) -> str:
    h = [f'<div class="report-doc"><h1>{escape(rep["title"])}</h1>',
         f'<p class="report-meta">{escape(rep.get("author", ""))}'
         f'{" · " + escape(rep["institution"]) if rep.get("institution") else ""} · {rep["date"]}</p>']
    toc = "".join(f"<li>{escape(s['heading'])}</li>" for s in rep["sections"])
    h.append(f'<h2>Contents</h2><ol class="toc">{toc}</ol><hr/>')
    for s in rep["sections"]:
        h.append(f'<h2>{escape(s["heading"])}</h2>')
        for b in s["blocks"]:
            if b["type"] == "text":
                h.append(f'<p>{apa_markup(b["text"], html=True)}</p>')
            elif b["type"] == "subheading":
                h.append(f'<h3>{escape(b["text"])}</h3>')
            elif b["type"] == "table":
                num, ttl = b["caption"].split("|", 1)
                h.append(f'<p class="cap"><b>{escape(num)}</b><br/><i>{escape(ttl)}</i></p>')
                h.append(b["df"].to_html(index=False, border=0, classes="apa-table", escape=True))
                if b["note"]: h.append(f'<p class="note"><i>Note.</i> {escape(b["note"])}</p>')
            elif b["type"] == "image":
                num, ttl = b["caption"].split("|", 1)
                b64 = base64.b64encode(b["png"]).decode()
                h.append(f'<p class="cap"><b>{escape(num)}</b><br/><i>{escape(ttl)}</i></p>'
                         f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>')
    h.append("</div>")
    return "".join(h)


# --------------------------------------------------------------------- PDF --
def _register_fonts():
    """DejaVu Serif ships with Matplotlib and covers Greek letters (α, χ², η²)."""
    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    base = matplotlib.get_data_path() + "/fonts/ttf/"
    names = {"RSerif": "DejaVuSerif.ttf", "RSerif-B": "DejaVuSerif-Bold.ttf",
             "RSerif-I": "DejaVuSerif-Italic.ttf", "RSerif-BI": "DejaVuSerif-BoldItalic.ttf"}
    try:
        for n, f in names.items():
            if n not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(n, base + f))
        pdfmetrics.registerFontFamily("RSerif", normal="RSerif", bold="RSerif-B", italic="RSerif-I", boldItalic="RSerif-BI")
        return "RSerif"
    except Exception:  # noqa: BLE001
        return "Times-Roman"


def report_to_pdf(rep: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font = _register_fonts()
    bold = font + "-B" if font == "RSerif" else "Times-Bold"
    ital = font + "-I" if font == "RSerif" else "Times-Italic"
    S = {
        "title": ParagraphStyle("t", fontName=bold, fontSize=16, leading=20, alignment=1, spaceAfter=6),
        "meta": ParagraphStyle("m", fontName=font, fontSize=10, leading=13, alignment=1, spaceAfter=18),
        "h1": ParagraphStyle("h1", fontName=bold, fontSize=13, leading=16, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=11, leading=14, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", fontName=font, fontSize=10.5, leading=15, spaceAfter=6),
        "cap": ParagraphStyle("c", fontName=font, fontSize=10, leading=13, spaceBefore=8),
        "capi": ParagraphStyle("ci", fontName=ital, fontSize=10, leading=13, spaceAfter=3),
        "note": ParagraphStyle("n", fontName=font, fontSize=8.5, leading=11, spaceAfter=8),
    }
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=0.9 * inch,
                            title=rep["title"], author=rep.get("author", ""))
    width = letter[0] - 2 * inch

    def footer(canvas, d):
        canvas.setFont(font, 8)
        canvas.drawCentredString(letter[0] / 2, 0.5 * inch, f"{rep['title']}  ·  Page {d.page}")

    story = [Paragraph(xml_escape(rep["title"]), S["title"]),
             Paragraph(xml_escape(" · ".join(x for x in (rep.get("author"), rep.get("institution"), rep["date"]) if x)), S["meta"])]
    for s in rep["sections"]:
        story.append(Paragraph(xml_escape(s["heading"]), S["h1"]))
        for b in s["blocks"]:
            if b["type"] == "text":
                story.append(Paragraph(apa_markup(b["text"]), S["body"]))
            elif b["type"] == "subheading":
                story.append(Paragraph(xml_escape(b["text"]), S["h2"]))
            elif b["type"] == "table":
                num, ttl = b["caption"].split("|", 1)
                df = b["df"]
                ncol = max(len(df.columns), 1)
                cs = ParagraphStyle("cell", fontName=font, fontSize=8 if ncol > 6 else 9, leading=10 if ncol > 6 else 11)
                hs = ParagraphStyle("hcell", parent=cs, fontName=bold)
                data = [[Paragraph(xml_escape(str(c)), hs) for c in df.columns]]
                data += [[Paragraph(xml_escape(str(v)), cs) for v in row] for row in df.itertuples(index=False)]
                first = 1.6 * inch if ncol > 3 else width / ncol
                widths = [first] + [(width - first) / (ncol - 1)] * (ncol - 1) if ncol > 1 else [width]
                t = Table(data, colWidths=widths, repeatRows=1)
                t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
                                       ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                                       ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
                                       ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
                story += [Paragraph(f"<b>{xml_escape(num)}</b>", S["cap"]), Paragraph(xml_escape(ttl), S["capi"]), t]
                story.append(Paragraph(f"<i>Note.</i> {xml_escape(b['note'])}", S["note"]) if b["note"] else Spacer(1, 8))
            elif b["type"] == "image":
                num, ttl = b["caption"].split("|", 1)
                iw, ih = ImageReader(io.BytesIO(b["png"])).getSize()
                w = min(width, 6 * inch)
                story.append(KeepTogether([Paragraph(f"<b>{xml_escape(num)}</b>", S["cap"]), Paragraph(xml_escape(ttl), S["capi"]),
                                           Image(io.BytesIO(b["png"]), width=w, height=w * ih / iw), Spacer(1, 10)]))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# --------------------------------------------------------------------- CSV --
def entries_to_csv_files(entries: list[dict]) -> dict[str, bytes]:
    """CSV exports for every table-like result (name -> bytes)."""
    files: dict[str, bytes] = {}

    def put(name, df):
        files[name] = df.to_csv(index=False).encode("utf-8-sig")

    descs = [e["table"] for e in entries if e["kind"] == "descriptives"]
    if descs: put("descriptive_statistics.csv", pd.concat(descs, ignore_index=True))
    rel = [e["item_table"].assign(Scale=e["title"]) for e in entries if e["kind"] == "reliability"]
    if rel: put("reliability_item_statistics.csv", pd.concat(rel, ignore_index=True))
    cor = [e["pairs"].assign(Analysis=e["title"]) for e in entries if e["kind"] == "correlation"]
    if cor: put("correlations.csv", pd.concat(cor, ignore_index=True))
    tests = [e["result"].summary_row() for e in entries if e["kind"] == "test"]
    if tests: put("hypothesis_tests_summary.csv", pd.DataFrame(tests))
    return files


def zip_files(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    return buf.getvalue()
