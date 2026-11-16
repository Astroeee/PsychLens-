"""Hypothesis tests with assumption checks, effect sizes, CIs and APA output.

Supported: independent t (Student/Welch), paired t, one-sample t, one-way ANOVA
(Fisher/Welch + post-hoc), chi-square test of independence, Mann-Whitney U,
Wilcoxon signed-rank. No Streamlit imports.

Nothing here picks a test for you: the caller chooses the test from the research
design, and these functions report assumption checks as information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from modules.correlation import p_adjust


# ------------------------------------------------------------ formatting ----
def _nolead(x: float, d: int = 2) -> str:
    """APA style: no leading zero for statistics that cannot exceed 1."""
    s = f"{x:.{d}f}"
    return s.replace("0.", ".", 1) if abs(x) < 1 else s


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p): return "p = n/a"
    return "p < .001" if p < 0.001 else f"p = {_nolead(p, 3)}"


def _num(x: float, d: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _ci_text(ci, level, d=2) -> str:
    if ci is None or not all(np.isfinite(ci)): return ""
    return f"{level * 100:.0f}% CI [{_num(ci[0], d)}, {_num(ci[1], d)}]"


@dataclass
class TestResult:
    __test__ = False  # stop pytest from treating this as a test class
    name: str
    variables: str
    n: int
    statistic_name: str
    statistic: float
    df: object          # float, tuple or None
    p: float
    effect_name: str = ""
    effect: float = np.nan
    effect_label: str = ""
    ci_label: str = ""
    ci: tuple | None = None
    alpha: float = 0.05
    ci_level: float = 0.95
    descriptives: pd.DataFrame | None = None
    assumptions: list = field(default_factory=list)
    extra_tables: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    apa: str = ""
    interpretation: str = ""
    hypotheses: str = ""

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.p) and self.p < self.alpha)

    @property
    def df_text(self) -> str:
        if self.df is None: return ""
        if isinstance(self.df, tuple): return ", ".join(_num(v, 0 if float(v).is_integer() else 2) for v in self.df)
        return _num(self.df, 0 if float(self.df).is_integer() else 2)

    def summary_row(self) -> dict:
        return {"Test": self.name, "Variables": self.variables, "N": self.n,
                "Statistic": f"{self.statistic_name} = {_num(self.statistic, 3)}",
                "df": self.df_text, "p": fmt_p(self.p).replace("p = ", "").replace("p < ", "< "),
                "Effect size": f"{self.effect_name} = {_num(self.effect, 3)} ({self.effect_label})" if self.effect_name else "",
                self.ci_label or "CI": _ci_text(self.ci, self.ci_level, 2) if self.ci else "",
                "Significant": "Yes" if self.significant else "No"}


# --------------------------------------------------------- assumption checks --
def check_normality(x, label: str, alpha: float = 0.05) -> dict:
    x = pd.Series(x).dropna().astype(float)
    base = {"Check": f"Normality: {label}", "Test": "Shapiro–Wilk"}
    if len(x) < 3:
        return {**base, "Statistic": np.nan, "p": np.nan, "Result": "not assessed",
                "Note": "Fewer than 3 observations."}
    if x.nunique() == 1:
        return {**base, "Statistic": np.nan, "p": np.nan, "Result": "not assessed",
                "Note": "All values are identical."}
    w, p = stats.shapiro(x)
    ok = p > alpha
    note = ("No significant departure from normality detected; this does not prove normality."
            if ok else "Significant departure from normality; inspect the distribution and consider a non-parametric test.")
    if len(x) > 5000: note += " (n > 5000: Shapiro–Wilk is over-sensitive; rely on plots.)"
    return {**base, "Statistic": float(w), "p": float(p), "Result": "met" if ok else "flagged", "Note": note}


def check_equal_variances(groups, labels, alpha: float = 0.05) -> dict:
    base = {"Check": "Homogeneity of variance", "Test": "Levene (median-centred)"}
    groups = [pd.Series(g).dropna().astype(float) for g in groups]
    if any(len(g) < 2 for g in groups) or all(g.nunique() == 1 for g in groups):
        return {**base, "Statistic": np.nan, "p": np.nan, "Result": "not assessed", "Note": "Not enough variation."}
    w, p = stats.levene(*groups, center="median")
    ok = p > alpha
    return {**base, "Statistic": float(w), "p": float(p), "Result": "met" if ok else "flagged",
            "Note": "Variances look similar across groups." if ok else
                    "Variances differ significantly; Welch-type tests are more robust."}


def _design_note(text: str) -> dict:
    return {"Check": "Independence / design", "Test": "Researcher confirmation", "Statistic": np.nan,
            "p": np.nan, "Result": "confirmed by researcher", "Note": text}


def _label_d(d: float) -> str:
    a = abs(d)
    if not np.isfinite(a): return "n/a"
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def _label_r(r: float) -> str:
    a = abs(r)
    if not np.isfinite(a): return "n/a"
    return "negligible" if a < 0.1 else "small" if a < 0.3 else "medium" if a < 0.5 else "large"


def _label_eta(e: float) -> str:
    if not np.isfinite(e): return "n/a"
    return "negligible" if e < 0.01 else "small" if e < 0.06 else "medium" if e < 0.14 else "large"


def _clean(x) -> np.ndarray:
    return pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)


def _flagged(assumptions: list) -> list[str]:
    return [a["Check"] for a in assumptions if a["Result"] == "flagged"]


def _sig_text(p, alpha) -> str:
    return "statistically significant" if p < alpha else "not statistically significant"


def _caveat(assumptions) -> str:
    fl = _flagged(assumptions)
    txt = ("Assumption checks flagged: " + "; ".join(fl) + ". Interpret with caution or consider a robust/non-parametric alternative. "
           if fl else "")
    return txt + ("Statistical significance does not by itself indicate practical importance, and causal "
                  "conclusions depend on the research design, not on the test.")


# ------------------------------------------------------- independent t-test --
def independent_t_test(a, b, label_a="Group A", label_b="Group B", dv="Outcome",
                       equal_var=False, alpha=0.05, ci_level=0.95) -> TestResult:
    a, b = _clean(a), _clean(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group needs at least 2 valid observations.")
    m1, m2, v1, v2 = a.mean(), b.mean(), a.var(ddof=1), b.var(ddof=1)
    diff = m1 - m2
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if equal_var:
        se, dof = sp * np.sqrt(1 / n1 + 1 / n2), n1 + n2 - 2
    else:
        se = np.sqrt(v1 / n1 + v2 / n2)
        dof = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)) if se > 0 else np.nan
    warns = []
    if not se > 0:
        t, p, ci = np.nan, np.nan, (np.nan, np.nan)
        warns.append("Both groups have zero variance; the t statistic is undefined.")
    else:
        t = diff / se
        p = float(2 * stats.t.sf(abs(t), dof))
        crit = stats.t.ppf(1 - (1 - ci_level) / 2, dof)
        ci = (diff - crit * se, diff + crit * se)
    d = diff / sp if sp > 0 else np.nan
    g = d * (1 - 3 / (4 * (n1 + n2) - 9)) if np.isfinite(d) else np.nan
    desc = pd.DataFrame({"Group": [label_a, label_b], "N": [n1, n2], "Mean": [m1, m2],
                         "SD": [np.sqrt(v1), np.sqrt(v2)], "SE": [np.sqrt(v1 / n1), np.sqrt(v2 / n2)]})
    assum = [check_normality(a, label_a, alpha), check_normality(b, label_b, alpha),
             check_equal_variances([a, b], [label_a, label_b], alpha),
             _design_note("Each respondent must belong to only one group, and observations must not influence one another. "
                          "This cannot be tested statistically.")]
    name = "Independent-samples t-test (Student)" if equal_var else "Independent-samples t-test (Welch)"
    apa = (f"t({_num(dof, 0 if float(dof).is_integer() else 2)}) = {_num(t)}, {fmt_p(p)}, d = {_num(d)}, "
           f"{_ci_text(ci, ci_level)}") if np.isfinite(t) else "Test statistic undefined"
    interp = (f"On {dv}, {label_a} (M = {m1:.2f}, SD = {np.sqrt(v1):.2f}, n = {n1}) and {label_b} "
              f"(M = {m2:.2f}, SD = {np.sqrt(v2):.2f}, n = {n2}) differed by {diff:+.2f} "
              f"({_ci_text(ci, ci_level)}). This difference was {_sig_text(p, alpha)} at α = {_nolead(alpha, 2)}, "
              f"{apa.split(', d =')[0]}; the standardized effect was {_label_d(d)} (Cohen's d = {_num(d)}). "
              + _caveat(assum)) if np.isfinite(t) else "The test could not be computed."
    extra = {"Effect sizes": pd.DataFrame({"Measure": ["Cohen's d (pooled SD)", "Hedges' g"], "Value": [d, g]})}
    return TestResult(name, f"{dv} by group ({label_a} vs {label_b})", n1 + n2, "t", float(t), float(dof), p,
                      "Cohen's d", float(d), _label_d(d), "Mean difference 95% CI" if ci_level == .95 else "Mean difference CI",
                      tuple(map(float, ci)), alpha, ci_level, desc, assum, extra, warns, apa, interp,
                      f"H0: the population means of {label_a} and {label_b} are equal.")


# ---------------------------------------------------------- paired t-test ----
def _pairs(x, y):
    d = pd.concat([pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce"),
                   pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce")], axis=1).dropna()
    return d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)


def paired_t_test(x, y, label_x="Measure 1", label_y="Measure 2", alpha=0.05, ci_level=0.95) -> TestResult:
    x, y = _pairs(x, y)
    n = len(x)
    if n < 2:
        raise ValueError("At least 2 complete pairs are required.")
    diff = x - y
    md, sd = diff.mean(), diff.std(ddof=1)
    warns = []
    if sd == 0:
        t, p, ci, dz = np.nan, np.nan, (np.nan, np.nan), np.nan
        warns.append("All pairwise differences are identical; the test is undefined.")
    else:
        se = sd / np.sqrt(n)
        t = md / se
        p = float(2 * stats.t.sf(abs(t), n - 1))
        crit = stats.t.ppf(1 - (1 - ci_level) / 2, n - 1)
        ci, dz = (md - crit * se, md + crit * se), md / sd
    r_xy = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan
    desc = pd.DataFrame({"Measure": [label_x, label_y, f"Difference ({label_x} − {label_y})"],
                         "N": [n] * 3, "Mean": [x.mean(), y.mean(), md],
                         "SD": [x.std(ddof=1), y.std(ddof=1), sd],
                         "SE": [x.std(ddof=1) / np.sqrt(n), y.std(ddof=1) / np.sqrt(n), sd / np.sqrt(n)]})
    assum = [check_normality(diff, "pairwise differences", alpha),
             _design_note("Both measures must come from the same respondents (matched pairs), and pairs must be independent "
                          "of each other. This cannot be tested statistically.")]
    apa = (f"t({n - 1}) = {_num(t)}, {fmt_p(p)}, d_z = {_num(dz)}, {_ci_text(ci, ci_level)}"
           if np.isfinite(t) else "Test statistic undefined")
    interp = (f"Across {n} complete pairs, the mean difference between {label_x} and {label_y} was {md:+.2f} "
              f"({_ci_text(ci, ci_level)}), which was {_sig_text(p, alpha)} at α = {_nolead(alpha, 2)}, "
              f"{apa.split(', d_z')[0]}; the standardized effect was {_label_d(dz)} (d_z = {_num(dz)}). "
              + _caveat(assum)) if np.isfinite(t) else "The test could not be computed."
    extra = {"Correlation between measures": pd.DataFrame({"Measure": ["Pearson r"], "Value": [r_xy]})}
    return TestResult("Paired-samples t-test", f"{label_x} vs {label_y}", n, "t", float(t), float(n - 1), p,
                      "Cohen's d_z", float(dz), _label_d(dz), "Mean difference 95% CI", tuple(map(float, ci)),
                      alpha, ci_level, desc, assum, extra, warns, apa, interp,
                      f"H0: the mean of the paired differences ({label_x} − {label_y}) is zero.")


# ------------------------------------------------------- one-sample t-test ---
def one_sample_t_test(x, mu: float, label="Variable", alpha=0.05, ci_level=0.95) -> TestResult:
    x = _clean(x)
    n = len(x)
    if n < 2:
        raise ValueError("At least 2 valid observations are required.")
    m, sd = x.mean(), x.std(ddof=1)
    warns = []
    if sd == 0:
        t, p, ci, d = np.nan, np.nan, (np.nan, np.nan), np.nan
        warns.append("All values are identical; the test is undefined.")
    else:
        se = sd / np.sqrt(n)
        t = (m - mu) / se
        p = float(2 * stats.t.sf(abs(t), n - 1))
        crit = stats.t.ppf(1 - (1 - ci_level) / 2, n - 1)
        ci, d = (m - mu - crit * se, m - mu + crit * se), (m - mu) / sd
    desc = pd.DataFrame({"Variable": [label], "N": [n], "Mean": [m], "SD": [sd],
                         "SE": [sd / np.sqrt(n)], "Test value": [mu]})
    assum = [check_normality(x, label, alpha),
             _design_note("Observations must be independent and drawn from the population of interest. "
                          "This cannot be tested statistically.")]
    apa = (f"t({n - 1}) = {_num(t)}, {fmt_p(p)}, d = {_num(d)}, {_ci_text(ci, ci_level)}"
           if np.isfinite(t) else "Test statistic undefined")
    interp = (f"The sample mean of {label} (M = {m:.2f}, SD = {sd:.2f}, N = {n}) differed from the test value {mu:g} "
              f"by {m - mu:+.2f} ({_ci_text(ci, ci_level)}). This was {_sig_text(p, alpha)} at α = {_nolead(alpha, 2)}, "
              f"{apa.split(', d =')[0]}; effect size was {_label_d(d)} (d = {_num(d)}). " + _caveat(assum)
              ) if np.isfinite(t) else "The test could not be computed."
    return TestResult("One-sample t-test", f"{label} vs μ = {mu:g}", n, "t", float(t), float(n - 1), p,
                      "Cohen's d", float(d), _label_d(d), "Mean difference 95% CI", tuple(map(float, ci)),
                      alpha, ci_level, desc, assum, {}, warns, apa, interp,
                      f"H0: the population mean of {label} equals {mu:g}.")


# ----------------------------------------------------------- one-way ANOVA ---
def _welch_anova(groups):
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) for g in groups])
    if np.any(v == 0):
        return np.nan, np.nan, np.nan, np.nan
    w = n / v
    W = w.sum()
    mw = (w * m).sum() / W
    lam = ((1 - w / W) ** 2 / (n - 1)).sum()
    f = ((w * (m - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
    df1, df2 = k - 1, (k ** 2 - 1) / (3 * lam)
    return float(f), float(df1), float(df2), float(stats.f.sf(f, df1, df2))


def one_way_anova(df: pd.DataFrame, dv: str, factor: str, levels=None, welch=False,
                  alpha=0.05, ci_level=0.95) -> TestResult:
    data = df[[dv, factor]].copy()
    data[dv] = pd.to_numeric(data[dv], errors="coerce")
    data = data.dropna()
    if levels is not None:
        data = data[data[factor].isin(levels)]
    labels = [str(l) for l in (levels if levels is not None else sorted(data[factor].unique(), key=str))]
    groups = [data.loc[data[factor].astype(str) == l, dv].to_numpy(float) for l in labels]
    keep = [(l, g) for l, g in zip(labels, groups) if len(g) > 0]
    labels, groups = [l for l, _ in keep], [g for _, g in keep]
    k = len(groups)
    if k < 2:
        raise ValueError("At least 2 groups with data are required.")
    if any(len(g) < 2 for g in groups):
        raise ValueError("Every group needs at least 2 observations.")
    N = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    dfb, dfw = k - 1, N - k
    msw = ssw / dfw
    warns = []
    if msw == 0:
        raise ValueError("All groups have zero within-group variance; ANOVA is undefined.")
    f_fisher = (ssb / dfb) / msw
    p_fisher = float(stats.f.sf(f_fisher, dfb, dfw))
    eta2 = ssb / (ssb + ssw)
    omega2 = max((ssb - dfb * msw) / (ssb + ssw + msw), 0.0)
    if welch:
        f, df1, df2, p = _welch_anova(groups)
        name, dof = "One-way ANOVA (Welch)", (df1, df2)
    else:
        f, p, name, dof = f_fisher, p_fisher, "One-way ANOVA (Fisher)", (float(dfb), float(dfw))

    desc = pd.DataFrame({"Group": labels, "N": [len(g) for g in groups], "Mean": [g.mean() for g in groups],
                         "SD": [g.std(ddof=1) for g in groups],
                         "SE": [g.std(ddof=1) / np.sqrt(len(g)) for g in groups]})
    resid = np.concatenate([g - g.mean() for g in groups])
    norm = check_normality(resid, "model residuals", alpha)
    assum = [norm, check_equal_variances(groups, labels, alpha),
             _design_note("Groups must consist of different respondents and observations must be independent. "
                          "This cannot be tested statistically.")]
    table = pd.DataFrame({"Source": ["Between groups", "Within groups", "Total"],
                          "SS": [ssb, ssw, ssb + ssw], "df": [dfb, dfw, N - 1],
                          "MS": [ssb / dfb, msw, np.nan], "F": [f_fisher, np.nan, np.nan],
                          "p": [p_fisher, np.nan, np.nan]})
    extra = {"ANOVA table (Fisher)": table,
             "Effect sizes": pd.DataFrame({"Measure": ["η² (eta squared)", "ω² (omega squared)"], "Value": [eta2, omega2]})}

    # Post-hoc comparisons
    post = None
    if k > 2:
        rows = []
        if not welch:
            try:
                res = stats.tukey_hsd(*groups)
                try:
                    cis = res.confidence_interval(confidence_level=ci_level)
                    lo, hi = cis.low, cis.high
                except Exception:  # noqa: BLE001
                    lo = hi = None
                for i, j in combinations(range(k), 2):
                    rows.append({"Comparison": f"{labels[i]} − {labels[j]}",
                                 "Mean difference": groups[i].mean() - groups[j].mean(),
                                 "CI lower": np.nan if lo is None else lo[i, j],
                                 "CI upper": np.nan if hi is None else hi[i, j],
                                 "p (Tukey HSD)": float(res.pvalue[i, j])})
            except Exception as exc:  # noqa: BLE001
                warns.append(f"Tukey HSD could not be computed: {exc}")
        else:
            pairs = list(combinations(range(k), 2))
            rs = [independent_t_test(groups[i], groups[j], labels[i], labels[j], dv, False, alpha, ci_level) for i, j in pairs]
            adj = p_adjust([r.p for r in rs], "holm")
            for (i, j), r, pa in zip(pairs, rs, adj):
                rows.append({"Comparison": f"{labels[i]} − {labels[j]}",
                             "Mean difference": groups[i].mean() - groups[j].mean(),
                             "CI lower": r.ci[0], "CI upper": r.ci[1], "p (Welch, Holm-adjusted)": float(pa)})
        post = pd.DataFrame(rows)
        if p < alpha:
            extra["Post-hoc comparisons"] = post
        else:
            warns.append("The omnibus test was not significant, so post-hoc comparisons are not shown.")

    f_txt = f"F({_num(dof[0], 0)}, {_num(dof[1], 0 if float(dof[1]).is_integer() else 2)}) = {_num(f)}"
    apa = f"{f_txt}, {fmt_p(p)}, η² = {_nolead(eta2, 2)}" if np.isfinite(f) else "Test statistic undefined"
    interp = (f"Mean {dv} differed across the {k} groups ({', '.join(labels)}): {f_txt}, {fmt_p(p)} — "
              f"{_sig_text(p, alpha)} at α = {_nolead(alpha, 2)}. The effect size was {_label_eta(eta2)} "
              f"(η² = {_nolead(eta2, 2)}, ω² = {_nolead(omega2, 2)}). An omnibus test shows that at least one group mean "
              f"differs, not which ones; see the post-hoc table when available. " + _caveat(assum)
              ) if np.isfinite(f) else "The test could not be computed."
    return TestResult(name, f"{dv} by {factor}", N, "F", float(f), dof, p, "η²", float(eta2), _label_eta(eta2),
                      "", None, alpha, ci_level, desc, assum, extra, warns, apa, interp,
                      f"H0: all {k} group population means of {dv} are equal.")


# --------------------------------------------------------------- chi-square --
def chi_square_independence(df: pd.DataFrame, var_a: str, var_b: str, correction=True,
                            alpha=0.05) -> TestResult:
    d = df[[var_a, var_b]].dropna()
    excluded = len(df) - len(d)
    ct = pd.crosstab(d[var_a], d[var_b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("Both variables need at least 2 categories with data.")
    n = int(ct.to_numpy().sum())
    chi2, p, dof, exp = stats.chi2_contingency(ct, correction=correction)
    chi2_unc = stats.chi2_contingency(ct, correction=False)[0]
    k = min(ct.shape) - 1
    v = float(np.sqrt(chi2_unc / (n * k)))
    # effect-size benchmarks scale with df* = min(rows-1, cols-1) (Cohen, 1988)
    lab = "negligible" if v < 0.1 / np.sqrt(k) else "small" if v < 0.3 / np.sqrt(k) else \
          "medium" if v < 0.5 / np.sqrt(k) else "large"
    expd = pd.DataFrame(exp, index=ct.index, columns=ct.columns)
    rows, cols = ct.sum(axis=1).to_numpy()[:, None] / n, ct.sum(axis=0).to_numpy()[None, :] / n
    adj = pd.DataFrame((ct.to_numpy() - exp) / np.sqrt(exp * (1 - rows) * (1 - cols)),
                       index=ct.index, columns=ct.columns)
    pct_small = float((exp < 5).mean() * 100)
    warns = []
    if excluded:
        warns.append(f"{excluded} row(s) excluded because one of the variables was missing.")
    if exp.min() < 1 or pct_small > 20:
        warns.append(f"{pct_small:.0f}% of cells have expected counts below 5 (minimum {exp.min():.2f}); the chi-square "
                     "approximation may be unreliable. Consider merging categories or Fisher's exact test.")
    assum = [{"Check": "Expected cell counts", "Test": "Rule of thumb (≥ 80% of cells ≥ 5, none < 1)", "Statistic": float(exp.min()),
              "p": np.nan, "Result": "met" if not (exp.min() < 1 or pct_small > 20) else "flagged",
              "Note": f"Minimum expected count = {exp.min():.2f}; {pct_small:.0f}% of cells below 5."},
             _design_note("Each respondent must appear in exactly one cell, and observations must be independent. "
                          "This cannot be tested statistically.")]
    extra = {"Observed counts": ct.reset_index(), "Expected counts": expd.round(2).reset_index(),
             "Adjusted standardized residuals": adj.round(2).reset_index()}
    ci, ci_label = None, ""
    if ct.shape == (2, 2):
        a_, b_, c_, d_ = ct.to_numpy().ravel().astype(float)
        orr, fp = stats.fisher_exact(ct.to_numpy())
        rec = {"Statistic": ["Odds ratio (Fisher conditional estimate)", "Fisher exact p"], "Value": [orr, fp]}
        if min(a_, b_, c_, d_) > 0:
            se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
            ci = (float(np.exp(np.log(a_ * d_ / (b_ * c_)) - 1.96 * se)), float(np.exp(np.log(a_ * d_ / (b_ * c_)) + 1.96 * se)))
            ci_label = "Odds ratio 95% CI"
        extra["2×2 supplementary"] = pd.DataFrame(rec)
    corr_txt = " (Yates continuity correction applied)" if correction and ct.shape == (2, 2) else ""
    apa = f"χ²({dof}, N = {n}) = {_num(chi2)}, {fmt_p(p)}, V = {_nolead(v, 2)}"
    interp = (f"The association between {var_a} and {var_b} was {_sig_text(p, alpha)} at α = {_nolead(alpha, 2)}, "
              f"{apa}{corr_txt}. Cramér's V indicates a {lab} association. See the adjusted residuals (|value| > 1.96) to see "
              f"which cells differ from independence. " + _caveat(assum))
    df_obj = float(dof)
    return TestResult("Chi-square test of independence", f"{var_a} × {var_b}", n, "χ²", float(chi2), df_obj, float(p),
                      "Cramér's V", v, lab, ci_label, ci, alpha, 0.95, ct.reset_index(), assum, extra, warns, apa, interp,
                      f"H0: {var_a} and {var_b} are independent in the population.")


# ---------------------------------------------------------- Mann-Whitney U ---
def mann_whitney_u(a, b, label_a="Group A", label_b="Group B", dv="Outcome", alpha=0.05) -> TestResult:
    a, b = _clean(a), _clean(b)
    n1, n2 = len(a), len(b)
    if n1 < 1 or n2 < 1 or n1 + n2 < 3:
        raise ValueError("Each group needs at least 1 valid observation (3 in total).")
    res = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    u1 = float(res.statistic)
    allv = np.concatenate([a, b])
    N = n1 + n2
    ranks = stats.rankdata(allv)
    _, ties = np.unique(allv, return_counts=True)
    sigma = np.sqrt(n1 * n2 / 12 * ((N + 1) - (ties ** 3 - ties).sum() / (N * (N - 1)))) if N > 1 else np.nan
    z = (u1 - n1 * n2 / 2) / sigma if sigma and sigma > 0 else np.nan
    rrb = 2 * u1 / (n1 * n2) - 1
    cles = u1 / (n1 * n2)
    desc = pd.DataFrame({"Group": [label_a, label_b], "N": [n1, n2],
                         "Median": [np.median(a), np.median(b)],
                         "IQR": [np.subtract(*np.percentile(a, [75, 25])), np.subtract(*np.percentile(b, [75, 25]))],
                         "Mean rank": [ranks[:n1].mean(), ranks[n1:].mean()]})
    assum = [{"Check": "Similar spread (informational)", "Test": "Levene (median-centred)",
              **{k: v for k, v in check_equal_variances([a, b], [label_a, label_b], alpha).items() if k in ("Statistic", "p", "Result", "Note")}},
             _design_note("Groups must consist of different respondents; the outcome must be at least ordinal. If the two "
                          "distributions have a similar shape, the test compares medians; otherwise it compares overall "
                          "tendency to be larger (stochastic dominance).")]
    apa = f"U = {_num(u1, 1)}, z = {_num(z)}, {fmt_p(res.pvalue)}, r_rb = {_num(rrb)}"
    interp = (f"On {dv}, {label_a} (Mdn = {np.median(a):.2f}, n = {n1}) and {label_b} (Mdn = {np.median(b):.2f}, n = {n2}) "
              f"differed {_sig_text(res.pvalue, alpha)}ly at α = {_nolead(alpha, 2)}: {apa}. The rank-biserial correlation "
              f"indicates a {_label_r(rrb)} effect; the probability that a random {label_a} value exceeds a random {label_b} "
              f"value (ties counted half) is {cles:.2f}. " + _caveat(assum))
    extra = {"Effect sizes": pd.DataFrame({"Measure": ["Rank-biserial r", "Common-language effect size P(A > B)", "r = z/√N"],
                                           "Value": [rrb, cles, z / np.sqrt(N) if np.isfinite(z) else np.nan]})}
    return TestResult("Mann–Whitney U test", f"{dv} by group ({label_a} vs {label_b})", N, "U", u1, None,
                      float(res.pvalue), "Rank-biserial r", float(rrb), _label_r(rrb), "", None, alpha, 0.95, desc, assum,
                      extra, [], apa, interp,
                      f"H0: values in {label_a} and {label_b} come from the same distribution (no tendency for one group to be larger).")


# ------------------------------------------------------ Wilcoxon signed-rank -
def wilcoxon_signed_rank(x, y, label_x="Measure 1", label_y="Measure 2", alpha=0.05) -> TestResult:
    x, y = _pairs(x, y)
    diff = x - y
    nz = diff[diff != 0]
    n_zero = len(diff) - len(nz)
    if len(nz) < 1:
        raise ValueError("All paired differences are zero; the test is undefined.")
    res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="auto")
    ranks = stats.rankdata(np.abs(nz))
    wp, wm = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
    rrb = (wp - wm) / (wp + wm)
    n = len(nz)
    _, ties = np.unique(np.abs(nz), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (ties ** 3 - ties).sum() / 48
    z = (wp - n * (n + 1) / 4) / np.sqrt(var) if var > 0 else np.nan
    warns = []
    if n_zero:
        warns.append(f"{n_zero} pair(s) with zero difference were dropped, as is standard for this test.")
    skew = stats.skew(diff) if len(diff) > 2 and diff.std() > 0 else np.nan
    assum = [{"Check": "Symmetry of differences (informational)", "Test": "Sample skewness of differences",
              "Statistic": float(skew) if np.isfinite(skew) else np.nan, "p": np.nan,
              "Result": "met" if np.isfinite(skew) and abs(skew) < 1 else "flagged" if np.isfinite(skew) else "not assessed",
              "Note": "The test assumes the distribution of differences is roughly symmetric; |skewness| < 1 is a loose guide."},
             _design_note("Both measures must come from the same respondents (matched pairs), and pairs must be independent. "
                          "This cannot be tested statistically.")]
    desc = pd.DataFrame({"Measure": [label_x, label_y, "Difference"], "N": [len(x)] * 3,
                         "Median": [np.median(x), np.median(y), np.median(diff)],
                         "IQR": [np.subtract(*np.percentile(a, [75, 25])) for a in (x, y, diff)]})
    apa = f"W = {_num(wp, 1)}, z = {_num(z)}, {fmt_p(res.pvalue)}, r_rb = {_num(rrb)}"
    interp = (f"Across {len(x)} pairs ({n} with non-zero differences), {label_x} and {label_y} differed "
              f"{_sig_text(res.pvalue, alpha)}ly at α = {_nolead(alpha, 2)}: {apa} (W is the sum of positive ranks, i.e. "
              f"pairs where {label_x} > {label_y}). The matched-pairs rank-biserial correlation indicates a {_label_r(rrb)} "
              f"effect. " + _caveat(assum))
    extra = {"Rank sums": pd.DataFrame({"Measure": ["W+ (sum of positive ranks)", "W− (sum of negative ranks)"], "Value": [wp, wm]})}
    return TestResult("Wilcoxon signed-rank test", f"{label_x} vs {label_y}", len(x), "W", wp, None, float(res.pvalue),
                      "Rank-biserial r", float(rrb), _label_r(rrb), "", None, alpha, 0.95, desc, assum, extra, warns, apa,
                      interp, f"H0: the distribution of paired differences ({label_x} − {label_y}) is symmetric around zero.")
