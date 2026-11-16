# 🧠 PsychLens – Psychological Research Data Analyzer

![Python](https://img.shields.io/badge/Python-3.10%2B-1D66D6?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-2E9E6B)

**A web-based research analytics tool for psychology: upload survey data, clean it, test reliability,
run hypothesis tests, and export an APA-style report, all without writing code.**

![Dashboard](assets/screenshots/01-dashboard.png)

## Why I built this

I'm a psychology student, and statistics software was often either expensive, hard to learn, or
something I couldn't fully see inside. I built PsychLens to (1) have a free, transparent tool I can use in my
own future research, and (2) learn software development by solving a real problem from my field.

Every statistic is computed from the data you upload, checked against trusted libraries, and reported in
APA style. The app is deliberately cautious: it never changes your original data, it makes you confirm your
research design before running a test, and it never writes conclusions for you.

## Screenshots

| Data Management | Reliability Analysis |
|---|---|
| ![Data Management](assets/screenshots/02-data-management.png) | ![Reliability](assets/screenshots/03-reliability.png) |

| Hypothesis Testing | Correlation Heatmap |
|---|---|
| ![Hypothesis Testing](assets/screenshots/04-hypothesis-testing.png) | ![Correlation](assets/screenshots/05-correlation.png) |

| Data Visualization | APA-style Report |
|---|---|
| ![Visualization](assets/screenshots/06-visualization.png) | ![Report](assets/screenshots/07-report.png) |

## ⚠️ Data note

**The sample dataset included in this project is simulated (randomly generated). It does not come from real
participants.** It exists only so you can try every feature without uploading anything. Results from the sample data
(including the "effect" of the fictional Mindfulness condition) are artificial and must not be interpreted as
research findings. The sample also contains a few deliberate data-quality problems (missing values, one
duplicate row, one out-of-range Likert response) to demonstrate the cleaning tools.
It is generated with a fixed seed in `modules/sample_data.py`.

## Features

| Section | What it does |
|---|---|
| **Dashboard** | KPI cards, distribution preview, variable types, missing data, data-quality summary, recent activity |
| **Data Management** | CSV/XLSX upload, searchable preview, column profiling, missing/duplicate/invalid-value detection, guided cleaning with a change log, CSV/Excel download |
| **Descriptive Statistics** | Mean, median, mode, SD, variance, range, skewness, kurtosis, quartiles, IQR, normality test; group splits; frequency tables; histograms, boxplots, density plots |
| **Reliability Analysis** | Cronbach's α with confidence interval, corrected item-total correlations, α if item deleted, reverse scoring, neutral interpretation guide (never declares a scale "valid") |
| **Correlation Analysis** | Pearson and Spearman, p-values, confidence intervals, multiple-comparison correction (Holm, Bonferroni, FDR), heatmap, scatterplot |
| **Hypothesis Testing** | Independent, paired and one-sample t-tests; one-way ANOVA (Fisher/Welch + post-hoc); chi-square; Mann–Whitney U; Wilcoxon signed-rank. Assumption checks, effect sizes, CIs, APA strings |
| **Data Visualization** | 7 customizable chart types with PNG and interactive HTML download |
| **Report Generator** | Choose saved analyses, add your own notes, preview and export an APA-style PDF and CSV tables |
| **Settings** | Researcher profile, significance level, confidence level, decimal places |

## Tech stack

Python · Streamlit · pandas · NumPy · SciPy · statsmodels · Pingouin · Plotly · Matplotlib · Seaborn · OpenPyXL · ReportLab · pytest

## Run it locally

```bash
git clone https://github.com/YOUR-USERNAME/psychological-research-analyzer.git
cd psychological-research-analyzer
python -m venv venv
venv\Scripts\activate            # Windows (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 and click **Load sample dataset**.

## How it works

```
app.py                  entry point: theme, sign-in gate, sidebar, routing
pages/                  one file per section (UI only)
components/             reusable UI: layout, cards, tables, dataset selector, sign-in screen
modules/                all statistics and logic, independent of the UI
  data_processing.py        loading, validation, profiling, cleaning
  descriptive_statistics.py
  reliability.py
  correlation.py
  hypothesis_testing.py
  visualization.py          interactive (Plotly) and print-quality (Matplotlib) charts
  report_generator.py       APA-style PDF/HTML/CSV export
  sample_data.py            simulated dataset generator
assets/style.css        custom SaaS-style theme
tests/                  statistical accuracy tests
```

The statistics in `modules/` never import Streamlit, so they can be tested and reused independently of the interface.

## Statistical accuracy

The test suite (`pytest -q`) compares the app's results with SciPy, Pingouin and statsmodels for t-tests,
ANOVA (Fisher and Welch), chi-square and Cramér's V, Mann–Whitney U, Wilcoxon, Pearson/Spearman correlations,
p-value adjustments and Cronbach's α. It also checks that cleaning never changes the original dataset.

## Design principles

- **The original data is never modified.** Cleaning creates a separate copy after confirmation, with a change log.
- **No silent data loss.** Missing-data handling is stated on each page and rows are never dropped without notice.
- **The researcher chooses the test.** Assumption checks inform the decision but don't make it.
- **No fabricated findings.** Reports contain computed statistics, neutral generated interpretations, and your own notes.

## Limitations

PsychLens is a learning and research-support tool. It has not been validated for clinical or high-stakes use, and
results should be checked with your supervisor or against established software (e.g., jamovi, SPSS, R) before
publication. Data is held in memory for the browser session only. It currently covers the tests listed above (no regression,
factor analysis or repeated-measures designs yet).

## Roadmap

- [ ] Factor analysis (to support scale validation)
- [ ] Repeated-measures ANOVA and Kruskal–Wallis
- [ ] Linear and logistic regression
- [ ] Persistent projects (database storage)

## Author

**[Your Name]**, psychology student · [GitHub](https://github.com/YOUR-USERNAME) · [LinkedIn](https://www.linkedin.com/in/YOUR-PROFILE)

## License

MIT, see [LICENSE](LICENSE).
