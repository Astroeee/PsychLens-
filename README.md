# PsychLens – Psychological Research Data Analyzer

A Streamlit web app for psychology students and researchers: upload survey data, clean it,
run descriptive statistics, reliability analysis, correlations and hypothesis tests, build charts,
and export an APA-style PDF report.

## Quick start

```bash
cd psychological-research-analyzer
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501), then click **Load sample dataset**.

## Features

| Section | What it does |
|---|---|
| Dashboard | KPI cards, distribution preview, variable types, missing data, data-quality summary, recent analyses and activity |
| Data Management | CSV/XLSX upload (+ sheet picker), paginated preview with search/sort, column profile, missing/duplicate/invalid detection with custom valid ranges, guided cleaning, CSV/Excel download |
| Descriptive Statistics | Mean, median, mode, SD, variance, min, max, range, skewness, kurtosis, quartiles, IQR, Shapiro–Wilk; group splits; frequency tables; histogram, density, boxplot, bar charts |
| Reliability Analysis | Cronbach's α (+ Feldt CI, standardized α), corrected item-total r, α if item deleted, inter-item correlations, reverse scoring, neutral interpretation guide |
| Correlation Analysis | Pearson/Spearman, matrix with stars, p-values, Fisher-z CIs, Holm/Bonferroni/FDR adjustment, heatmap, scatterplot |
| Hypothesis Testing | Independent/paired/one-sample t, one-way ANOVA (Fisher/Welch + post-hoc), chi-square, Mann–Whitney U, Wilcoxon; assumption checks, effect sizes, CIs, APA strings |
| Data Visualization | 7 chart types, custom titles/labels, PNG and interactive HTML downloads |
| Report Generator | Select saved analyses, add your own notes, preview, export PDF and CSV/ZIP |
| Settings | Researcher name, α, confidence level, decimals, session data tools |

## Project structure

```
app.py                  entry point: theme, sidebar, routing
config.py               paths, colors, navigation, defaults
pages/                  one file per section (UI only)
components/             reusable UI: layout, cards, tables, dataset selector
modules/                statistics and logic (no Streamlit): data_processing, descriptive_statistics,
                        reliability, correlation, hypothesis_testing, visualization, report_generator,
                        sample_data, state (session helpers)
assets/style.css        custom SaaS-style theme
data/                   sample dataset (simulated) and saved settings
tests/                  pytest checks against SciPy / Pingouin / statsmodels
```

**How it fits together:** `app.py` loads the CSS, initializes session state and builds a hidden router;
each page reads the chosen dataset from `st.session_state`, calls pure functions in `modules/`, and
displays the results with `components/`. Pressing **Save to report** stores a result in session state;
the Report Generator turns saved results into PDF/HTML/CSV.

## Data safety and statistical principles

* The original dataset is never modified. Cleaning creates a separate copy after you confirm, with a change log.
* Analyses use pairwise or listwise deletion as documented on each page; no rows are silently dropped.
* No test is chosen for you. Each test asks you to confirm the research design, and assumption checks are informational.
* The report only contains computed statistics, generated neutral interpretations, and your own notes.
* The sample dataset is **simulated**; do not interpret it as real findings.
* Data lives in memory for the browser session; uploaded files are not stored on disk or sent anywhere.

## Tests

```bash
pytest -q
```
Checks t-tests, ANOVA, chi-square, non-parametric tests, correlations, Cronbach's α and p-value
adjustments against SciPy, Pingouin and statsmodels, plus cleaning and data-validation behavior.

## Common errors

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt`; run from the project folder |
| Sidebar menu looks wrong / `position="hidden"` error | `pip install -U streamlit` (needs ≥ 1.50) |
| PNG download from the interactive chart is not high-res | Install `kaleido`, or use the "Download PNG (print)" button (Matplotlib) |
| Excel upload fails | Use `.xlsx` (not `.xls`) and make sure `openpyxl` is installed |
| PDF shows boxes instead of Greek letters | Reinstall `matplotlib` (its DejaVu fonts are used for the PDF) |
| Data disappears on refresh | Expected: session data is in memory. Re-upload or reload the sample |

## Ideas for extending

Factor analysis, repeated-measures ANOVA, Kruskal–Wallis, regression, database storage, user accounts.
