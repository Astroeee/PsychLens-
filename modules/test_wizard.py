"""Guided choice of a statistical test from the research question and design.

The wizard explains *why* a test fits; it never runs anything and the researcher
stays responsible for the choice. No Streamlit imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

GOALS = [
    "Compare groups or conditions",
    "Compare one sample with a known value",
    "Examine the relationship between two variables",
    "Predict an outcome from one or more variables",
    "Check a questionnaire scale (reliability or structure)",
    "Plan sample size before collecting data",
]
OUTCOMES = {
    "numeric": "Numeric (interval/ratio: scores, minutes, totals/means of scales)",
    "ordinal": "Ordinal (ranked, single Likert item, non-normal ratings)",
    "categorical": "Categorical (categories such as yes/no or groups)",
}
DESIGNS = {"independent": "Independent groups (different people in each group)",
           "repeated": "Repeated measures (the same people measured more than once, or matched pairs)"}


@dataclass
class Recommendation:
    title: str
    why: str
    available: bool = True
    page: str | None = None              # app page that runs it
    test_key: str | None = None          # option to preselect on the Hypothesis Testing page
    assumptions: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)
    effect_size: str = ""
    report_as: str = ""
    notes: list = field(default_factory=list)


HT = "pages/6_Hypothesis_Testing.py"
N_AVAIL = "Not yet available in this app; use R, SPSS or jamovi for this test."


def recommend(goal: str, outcome: str = "numeric", groups: int = 2, design: str = "independent",
              normal: str = "yes", small_n: bool = False, both_numeric: bool = True, binary_outcome: bool = True,
              scale_task: str = "reliability") -> Recommendation:
    """normal: 'yes' | 'no' | 'unsure'. small_n: fewer than ~30 per group."""
    parametric = (outcome == "numeric") and (normal == "yes" or (not small_n and normal != "no"))

    if goal == GOALS[0]:
        if outcome == "categorical":
            if design == "independent":
                return Recommendation(
                    "Chi-square test of independence", "You are comparing how categories are distributed across independent groups.",
                    page=HT, test_key="Chi-square test of independence",
                    assumptions=["Each person appears in only one cell", "Expected counts ≥ 5 in at least 80% of cells (none < 1)"],
                    alternatives=["Fisher's exact test when expected counts are small (shown for 2×2 tables)"],
                    effect_size="Cramér's V (and odds ratio for 2×2 tables)", report_as="χ²(df, N = …) = …, p = …, V = …")
            return Recommendation(
                "McNemar's test (2 conditions) or Cochran's Q (3+)", "Categorical outcome measured on the same people.", available=False,
                notes=[N_AVAIL], effect_size="Odds ratio / proportion difference")
        if design == "independent":
            if groups == 2:
                if parametric:
                    return Recommendation(
                        "Independent-samples t-test (Welch)", "Two independent groups with a numeric outcome that is roughly normal or has enough cases per group.",
                        page=HT, test_key="Independent samples t-test",
                        assumptions=["Independent observations", "Roughly normal outcome in each group (or n ≥ ~30 per group)", "Welch's version does not need equal variances"],
                        alternatives=["Mann–Whitney U if the outcome is ordinal, very skewed, or n is small and non-normal"],
                        effect_size="Cohen's d / Hedges' g + 95% CI of the mean difference", report_as="t(df) = …, p = …, d = …, 95% CI […, …]")
                return Recommendation(
                    "Mann–Whitney U test", "Two independent groups, and the outcome is ordinal or not normal with a small sample.",
                    page=HT, test_key="Mann-Whitney U test",
                    assumptions=["Independent observations", "Outcome at least ordinal", "Similar distribution shapes if you want to interpret it as a difference in medians"],
                    alternatives=["Welch t-test if the data are roughly normal"], effect_size="Rank-biserial correlation",
                    report_as="U = …, z = …, p = …, r_rb = …")
            if parametric:
                return Recommendation(
                    "One-way ANOVA (Welch if variances differ)", "Three or more independent groups with a numeric, roughly normal outcome.",
                    page=HT, test_key="One-way ANOVA",
                    assumptions=["Independent observations", "Roughly normal residuals", "Similar variances (otherwise use Welch)"],
                    alternatives=["Kruskal–Wallis if the outcome is ordinal or strongly non-normal with small groups"],
                    effect_size="η² / ω² (plus post-hoc comparisons)", report_as="F(df1, df2) = …, p = …, η² = …",
                    notes=["Run post-hoc comparisons only if the overall test is significant."])
            return Recommendation(
                "Kruskal–Wallis test", "Three or more independent groups with an ordinal or non-normal outcome.", available=False,
                notes=[N_AVAIL, "Stop-gap in this app: One-way ANOVA (Welch) if your groups are reasonably sized, reported with a caution."],
                effect_size="Epsilon-squared", alternatives=["One-way ANOVA (Welch)"])
        # repeated measures
        if groups == 2:
            if parametric:
                return Recommendation(
                    "Paired-samples t-test", "The same people measured twice (or matched pairs) on a numeric outcome.",
                    page=HT, test_key="Paired samples t-test",
                    assumptions=["Pairs are independent of each other", "Differences are roughly normal (or n ≥ ~30 pairs)"],
                    alternatives=["Wilcoxon signed-rank if differences are non-normal or the outcome is ordinal"],
                    effect_size="Cohen's d_z + 95% CI of the mean difference", report_as="t(df) = …, p = …, d_z = …, 95% CI […, …]")
            return Recommendation(
                "Wilcoxon signed-rank test", "Two repeated measures where the differences are non-normal or the outcome is ordinal.",
                page=HT, test_key="Wilcoxon signed-rank test",
                assumptions=["Pairs are independent", "Differences roughly symmetric"], alternatives=["Paired t-test if differences are roughly normal"],
                effect_size="Matched-pairs rank-biserial correlation", report_as="W = …, z = …, p = …, r_rb = …")
        return Recommendation(
            "Repeated-measures ANOVA (or Friedman test if non-normal/ordinal)", "Three or more measurements on the same people.", available=False,
            notes=[N_AVAIL, "Stop-gap in this app: paired t-tests (or Wilcoxon) between pairs of time points with a Holm/Bonferroni correction, clearly labelled as such."],
            effect_size="Partial η² (generalized η²)")

    if goal == GOALS[1]:
        if outcome == "categorical":
            return Recommendation("Binomial test / chi-square goodness-of-fit", "Comparing observed category proportions with expected ones.",
                                  available=False, notes=[N_AVAIL])
        if parametric:
            return Recommendation(
                "One-sample t-test", "Comparing the mean of one numeric variable with a known or theoretical value.",
                page=HT, test_key="One-sample t-test", assumptions=["Independent observations", "Roughly normal (or n ≥ ~30)"],
                alternatives=["One-sample Wilcoxon / sign test if non-normal or ordinal (not in this app)"],
                effect_size="Cohen's d + 95% CI of the mean difference", report_as="t(df) = …, p = …, d = …, 95% CI […, …]")
        return Recommendation("One-sample Wilcoxon signed-rank (or sign) test", "Ordinal or non-normal data with a small sample, compared with a known value.",
                              available=False, notes=[N_AVAIL, "If n ≥ ~30, the one-sample t-test is usually robust."])

    if goal == GOALS[2]:
        if outcome == "categorical":
            return Recommendation(
                "Chi-square test of independence", "Both variables are categories.", page=HT, test_key="Chi-square test of independence",
                assumptions=["Independent observations", "Adequate expected counts"], effect_size="Cramér's V", report_as="χ²(df, N = …) = …, p = …, V = …")
        if outcome == "numeric" and both_numeric and (normal == "yes" or (not small_n and normal != "no")):
            return Recommendation(
                "Pearson correlation", "Two numeric variables with a roughly linear relationship and no extreme outliers.",
                page="pages/5_Correlation_Analysis.py", assumptions=["Linear relationship (check the scatterplot)", "No strong outliers", "Roughly bivariate normal for small samples"],
                alternatives=["Spearman if the relationship is monotonic but not linear, data are ordinal, or outliers are present"],
                effect_size="r (and r²) with 95% CI", report_as="r(df) = …, p = …, 95% CI […, …]")
        return Recommendation(
            "Spearman rank correlation", "Ordinal data, non-normal data, outliers, or a monotonic (not linear) relationship.",
            page="pages/5_Correlation_Analysis.py", assumptions=["Monotonic relationship"], alternatives=["Pearson if linear and roughly normal"],
            effect_size="ρ with 95% CI", report_as="ρ(df) = …, p = …")

    if goal == GOALS[3]:
        if outcome == "numeric":
            return Recommendation(
                "Multiple linear regression (hierarchical if you want to add predictors in steps)",
                "You want to predict a numeric outcome from one or more predictors and estimate each predictor's unique contribution.",
                page="pages/13_Regression.py", assumptions=["Linear relationships", "Independent, roughly normal residuals with constant variance", "No severe multicollinearity (VIF < 5–10)", "Roughly 10–20 cases per predictor"],
                alternatives=["Spearman/Pearson correlation if you only need a bivariate relationship"], effect_size="R², adjusted R², standardized β, ΔR² for blocks",
                report_as="R² = …, F(df1, df2) = …, p = …; B, SE, β, t, p for each predictor")
        if outcome == "categorical" and binary_outcome:
            return Recommendation(
                "Binary logistic regression", "You want to predict a yes/no (two-category) outcome from one or more predictors.",
                page="pages/13_Regression.py", assumptions=["Independent observations", "Linearity of the logit for numeric predictors", "No (quasi-)complete separation", "About 10+ events per predictor"],
                alternatives=["Chi-square test if there is only one categorical predictor"], effect_size="Odds ratios with 95% CI, Nagelkerke R², AUC",
                report_as="χ²(df, N = …) = …, p = …, Nagelkerke R² = …; OR [95% CI] per predictor")
        return Recommendation("Ordinal or multinomial logistic regression", "Outcome with ordered or more than two categories.", available=False, notes=[N_AVAIL])

    if goal == GOALS[4]:
        if scale_task == "reliability":
            return Recommendation(
                "Cronbach's alpha (internal consistency)", "You want to know whether the items of a scale hang together.",
                page="pages/4_Reliability_Analysis.py", assumptions=["Items measure one construct", "Items scored in the same direction (reverse-score first)", "Enough respondents (N ≥ 30–100)"],
                alternatives=["Factor analysis first if you are unsure the scale is one-dimensional", "McDonald's omega (not in this app)"],
                effect_size="α with 95% CI, corrected item-total correlations")
        return Recommendation(
            "Exploratory factor analysis (EFA)", "You want to see how many underlying dimensions the items measure and which items belong together.",
            page="pages/12_Factor_Analysis.py", assumptions=["Adequate sample (N ≥ 100, 5–10 per item)", "Items correlate (KMO > .60, Bartlett significant)", "Continuous or approximately continuous items"],
            alternatives=["Confirmatory factor analysis if you already have a theoretical structure (not in this app)"],
            effect_size="Loadings, communalities, variance explained")

    return Recommendation(
        "Power analysis / sample-size planning", "You want to know how many participants you need to detect an effect of a given size.",
        page="pages/14_Power_Analysis.py", assumptions=["A defensible expected effect size (prior studies or the smallest effect that matters)"],
        alternatives=["Sensitivity analysis: what effect could you detect with the sample you already have?"])


# ---------------------------------------------------------- data check -----
def data_snapshot(df: pd.DataFrame, outcome: str, group: str | None = None, alpha: float = 0.05) -> dict:
    """Facts from the user's own data that inform the normality / sample-size answers."""
    y = pd.to_numeric(df[outcome], errors="coerce")
    if group:
        parts = [(str(k), y[g.index].dropna()) for k, g in df.groupby(group, dropna=True)]
    else:
        parts = [(outcome, y.dropna())]
    rows = []
    for k, s in parts:
        p = stats.shapiro(s).pvalue if 3 <= len(s) <= 5000 and s.nunique() > 1 else np.nan
        rows.append({"Group": k, "n": len(s), "Mean": s.mean() if len(s) else np.nan, "SD": s.std(ddof=1) if len(s) > 1 else np.nan,
                     "Skewness": stats.skew(s, bias=False) if len(s) > 2 and s.nunique() > 1 else np.nan, "Shapiro–Wilk p": p})
    t = pd.DataFrame(rows)
    min_n = int(t["n"].min()) if len(t) else 0
    ps = t["Shapiro–Wilk p"].dropna()
    normal = "unsure" if ps.empty else ("yes" if (ps > alpha).all() else "no")
    lev = np.nan
    if group and len(parts) >= 2 and all(len(s) > 1 for _, s in parts):
        lev = stats.levene(*[s for _, s in parts], center="median").pvalue
    return {"table": t, "min_n": min_n, "small_n": min_n < 30, "normal": normal, "levene_p": lev, "n_groups": len(parts)}
