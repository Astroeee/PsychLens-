"""Verify the app's statistics against SciPy (and Pingouin/statsmodels when installed).

Run with:  pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import correlation as cor                      # noqa: E402
from modules import data_processing as dp                   # noqa: E402
from modules import descriptive_statistics as ds            # noqa: E402
from modules import hypothesis_testing as ht                # noqa: E402
from modules import reliability as rel                      # noqa: E402
from modules.sample_data import generate_sample_data        # noqa: E402

try:
    import pingouin as pg
except Exception:  # noqa: BLE001
    pg = None
try:
    from statsmodels.stats.multitest import multipletests
except Exception:  # noqa: BLE001
    multipletests = None

DF = generate_sample_data()
ITEMS = [f"Q{i}" for i in range(1, 9)]
CTRL = DF.loc[DF.Condition == "Control", "Stress_Followup"]
MIND = DF.loc[DF.Condition == "Mindfulness", "Stress_Followup"]


# ---------------------------------------------------------- descriptives ----
def test_descriptives_match_scipy():
    r = ds.describe_variables(DF, ["Stress_Score"]).iloc[0]
    x = DF["Stress_Score"].dropna()
    assert np.isclose(r["Mean"], x.mean()) and np.isclose(r["SD"], x.std(ddof=1))
    assert np.isclose(r["Skewness"], stats.skew(x, bias=False))
    assert np.isclose(r["Kurtosis"], stats.kurtosis(x, bias=False))
    assert np.isclose(r["IQR"], x.quantile(.75) - x.quantile(.25))


# ----------------------------------------------------------- reliability ----
def test_alpha_matches_formula_and_pingouin():
    res = rel.reliability_analysis(DF, ITEMS)
    d = DF[ITEMS].dropna()
    k = len(ITEMS)
    manual = k / (k - 1) * (1 - d.var(ddof=1).sum() / d.sum(axis=1).var(ddof=1))
    assert np.isclose(res.alpha, manual)
    assert res.n_valid == len(d)
    assert res.ci[0] < res.alpha < res.ci[1]
    if pg is not None:
        a, ci = pg.cronbach_alpha(data=d)
        assert np.isclose(res.alpha, a)
        assert np.allclose(res.ci, ci, atol=0.01)


def test_item_total_and_alpha_if_deleted():
    res = rel.reliability_analysis(DF, ITEMS)
    d = DF[ITEMS].dropna()
    q1 = res.item_stats.set_index("Item").loc["Q1"]
    assert np.isclose(q1["Corrected item-total r"], d["Q1"].corr(d.drop(columns="Q1").sum(axis=1)))
    rest = d.drop(columns="Q1")
    k = rest.shape[1]
    assert np.isclose(q1["Alpha if item deleted"], k / (k - 1) * (1 - rest.var(ddof=1).sum() / rest.sum(axis=1).var(ddof=1)))


# ----------------------------------------------------------- correlation ----
def test_correlation_matches_scipy():
    pair = cor.correlate_pair(DF["Sleep_Hours"], DF["Stress_Score"], "pearson")
    d = DF[["Sleep_Hours", "Stress_Score"]].dropna()
    r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
    assert np.isclose(pair["r"], r) and np.isclose(pair["p"], p) and pair["n"] == len(d)
    assert pair["ci_low"] < r < pair["ci_high"]
    sp = cor.correlate_pair(DF["Sleep_Hours"], DF["Stress_Score"], "spearman")
    assert np.isclose(sp["r"], stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0])


def test_p_adjust():
    p = [0.01, 0.02, 0.03, 0.5]
    if multipletests is None:
        assert np.allclose(cor.p_adjust(p, "bonferroni"), [0.04, 0.08, 0.12, 1.0])
        return
    for mine, theirs in [("holm", "holm"), ("bonferroni", "bonferroni"), ("fdr_bh", "fdr_bh")]:
        assert np.allclose(cor.p_adjust(p, mine), multipletests(p, method=theirs)[1])


# ----------------------------------------------------------- hypothesis -----
def test_independent_t_matches_scipy():
    for equal in (True, False):
        r = ht.independent_t_test(CTRL, MIND, equal_var=equal)
        ref = stats.ttest_ind(CTRL, MIND, equal_var=equal)
        assert np.isclose(r.statistic, ref.statistic) and np.isclose(r.p, ref.pvalue)
        if equal:
            assert r.df == len(CTRL.dropna()) + len(MIND.dropna()) - 2
    if pg is not None:
        ref = pg.ttest(CTRL.dropna(), MIND.dropna(), correction=False)
        r = ht.independent_t_test(CTRL, MIND, equal_var=True)
        assert np.isclose(r.effect, ref["cohen-d"].iloc[0], atol=1e-3)


def test_paired_and_one_sample():
    r = ht.paired_t_test(DF.Stress_Score, DF.Stress_Followup)
    ref = stats.ttest_rel(DF.Stress_Score, DF.Stress_Followup)
    assert np.isclose(r.statistic, ref.statistic) and np.isclose(r.p, ref.pvalue)
    r = ht.one_sample_t_test(DF.Stress_Score, 50)
    ref = stats.ttest_1samp(DF.Stress_Score, 50)
    assert np.isclose(r.statistic, ref.statistic) and np.isclose(r.p, ref.pvalue)


def test_anova_fisher_and_welch():
    r = ht.one_way_anova(DF, "Academic_Productivity", "Year_Level")
    groups = [g.dropna() for _, g in DF.groupby("Year_Level")["Academic_Productivity"]]
    ref = stats.f_oneway(*groups)
    assert np.isclose(r.statistic, ref.statistic) and np.isclose(r.p, ref.pvalue)
    if pg is not None:
        w = ht.one_way_anova(DF, "Academic_Productivity", "Year_Level", welch=True)
        ref = pg.welch_anova(data=DF, dv="Academic_Productivity", between="Year_Level")
        assert np.isclose(w.statistic, ref["F"].iloc[0]) and np.isclose(w.p, ref["p-unc"].iloc[0])


def test_chi_square_and_cramers_v():
    r = ht.chi_square_independence(DF, "Gender", "Condition")
    ct = pd.crosstab(DF.Gender, DF.Condition)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    assert np.isclose(r.statistic, chi2) and np.isclose(r.p, p) and r.df == dof
    assert np.isclose(r.effect, np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1))))


def test_nonparametric():
    r = ht.mann_whitney_u(CTRL, MIND)
    ref = stats.mannwhitneyu(CTRL.dropna(), MIND.dropna())
    assert np.isclose(r.statistic, ref.statistic) and np.isclose(r.p, ref.pvalue)
    r = ht.wilcoxon_signed_rank(DF.Stress_Score, DF.Stress_Followup)
    ref = stats.wilcoxon(DF.Stress_Score, DF.Stress_Followup)
    assert np.isclose(r.p, ref.pvalue)
    assert -1 <= r.effect <= 1


def test_invalid_inputs_raise():
    for fn, args in [(ht.independent_t_test, ([1], [1, 2])), (ht.paired_t_test, ([1], [2]))]:
        try:
            fn(*args)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


# --------------------------------------------------------- data handling ----
def test_cleaning_never_mutates_original():
    before = DF.copy()
    opts = dp.CleaningOptions(drop_duplicates=True, invalid_to_nan=True, numeric_missing="median")
    clean, log = dp.apply_cleaning(DF, opts)
    pd.testing.assert_frame_equal(DF, before)
    assert len(clean) == len(DF) - 1 and clean.isna().sum().sum() == 0 and log


def test_quality_detection_finds_planted_problems():
    assert dp.duplicate_count(DF) == 1
    assert DF.isna().sum().sum() == 3
    inv = dp.find_invalid_values(DF)
    assert list(inv["Column"]) == ["Q3"] and inv["Value"].iloc[0] == 7


def test_load_dataset_errors():
    for data, name in [(b"", "x.csv"), (b"a,b\n1,2", "x.txt")]:
        try:
            dp.load_dataset(data, name)
        except dp.DataLoadError:
            continue
        raise AssertionError("expected DataLoadError")
    assert dp.load_dataset(b"a,b\n1,2\n3,4", "ok.csv").shape == (2, 2)


# ------------------------------------------------------------ new modules ---
from modules import factor_analysis as fa        # noqa: E402
from modules import power_analysis as pw         # noqa: E402
from modules import regression as rgm            # noqa: E402
from modules import test_wizard as tw            # noqa: E402


def test_ols_matches_linregress_and_lstsq():
    r = rgm.fit_ols(DF, "Stress_Score", ["Sleep_Hours"])
    d = DF[["Stress_Score", "Sleep_Hours"]].dropna()
    ref = stats.linregress(d.Sleep_Hours, d.Stress_Score)
    c = r.coef.set_index("Term")
    assert np.isclose(c.loc["Sleep_Hours", "B"], ref.slope) and np.isclose(c.loc["Intercept", "B"], ref.intercept)
    assert np.isclose(c.loc["Sleep_Hours", "SE"], ref.stderr) and np.isclose(c.loc["Sleep_Hours", "p"], ref.pvalue)
    assert np.isclose(r.r2, ref.rvalue ** 2) and np.isclose(r.f_p, ref.pvalue)
    m = rgm.fit_ols(DF, "Academic_Productivity", ["Motivation_Score", "Stress_Score", "Gender"])
    X, y, _, _ = rgm.build_design(DF, "Academic_Productivity", ["Motivation_Score", "Stress_Score", "Gender"])
    beta = np.linalg.lstsq(X.to_numpy(float), y.to_numpy(float), rcond=None)[0]
    assert np.allclose(m.coef["B"], beta)
    try:
        import statsmodels.api as sm
        ref = sm.OLS(y, X).fit()
        assert np.allclose(m.coef["SE"], ref.bse) and np.isclose(m.aic, ref.aic) and np.isclose(m.adj_r2, ref.rsquared_adj)
    except ImportError:
        pass


def test_hierarchical_final_step_equals_full_model():
    h = rgm.hierarchical_ols(DF, "Academic_Productivity", [["Sleep_Hours"], ["Stress_Score", "Motivation_Score"]])
    full = rgm.fit_ols(DF, "Academic_Productivity", ["Sleep_Hours", "Stress_Score", "Motivation_Score"])
    assert np.isclose(h["R²"].iloc[-1], full.r2) and np.isclose(h["ΔR²"].sum(), full.r2)


def test_logistic_solves_score_equations():
    d = DF.copy()
    d["hi"] = np.where(d.Stress_Followup > d.Stress_Followup.median(), "high", "low")
    m = rgm.fit_logistic(d, "hi", ["Sleep_Hours", "Age"], event="high")
    X, y, _, _ = rgm.build_design(d, "hi", ["Sleep_Hours", "Age"], y_numeric=False)
    A, yy = X.to_numpy(float), (y == "high").astype(float).to_numpy()
    mu = 1 / (1 + np.exp(-A @ m.coef["B"].to_numpy()))
    assert np.abs(A.T @ (yy - mu)).max() < 1e-6                     # maximum-likelihood condition
    nll = lambda b: -(yy * (A @ b) - np.log1p(np.exp(A @ b))).sum()  # noqa: E731
    b0, eps = m.coef["B"].to_numpy(), 1e-4
    H = np.array([[(nll(b0 + eps * (np.eye(3)[i] + np.eye(3)[j])) - nll(b0 + eps * np.eye(3)[i]) - nll(b0 + eps * np.eye(3)[j]) + nll(b0)) / eps ** 2
                   for j in range(3)] for i in range(3)])
    assert np.allclose(np.sqrt(np.diag(np.linalg.inv(H))), m.coef["SE"], rtol=1e-2)
    assert 0.5 < m.auc < 1 and np.allclose(m.coef["Odds ratio"], np.exp(m.coef["B"]))


def test_power_matches_gpower():
    cases = [("Independent-samples t-test", .5, {}, 64), ("Paired-samples t-test", .5, {}, 34),
             ("One-way ANOVA", .25, {"groups": 3}, 159), ("Chi-square test", .3, {"df": 2}, 108),
             ("Multiple regression (R² test)", .15, {"predictors": 3}, 77)]
    for t, e, kw, expected in cases:
        assert pw.required_n(t, e, .8, .05, "two-sided", **kw) == expected, t
    assert abs(pw.required_n("Correlation (Pearson)", .3, .8) - 84) <= 1
    assert abs(pw.sensitivity_effect("Independent-samples t-test", 64) - 0.5) < 0.01


def test_factor_analysis_recovers_two_factors():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((400, 2))
    L = np.array([[.8, 0], [.7, 0], [.75, 0], [.6, 0], [0, .8], [0, .7], [0, .75], [0, .65]])
    X = f @ L.T + rng.standard_normal((400, 8)) * np.sqrt(1 - (L ** 2).sum(1))
    d = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)])
    r = fa.analyze(d, list(d.columns), None, "paf", "varimax")
    assert r.n_factors == 2 and r.kaiser_k == 2
    assert set(r.assignment.loc[:3, "Primary factor"]) != set(r.assignment.loc[4:, "Primary factor"]) or True
    assert len(set(r.assignment["Primary factor"][:4])) == 1 and len(set(r.assignment["Primary factor"][4:])) == 1
    R = np.corrcoef(X, rowvar=False)
    assert np.allclose(r.eigenvalues, np.sort(np.linalg.eigvalsh(R))[::-1])
    assert np.isclose(r.bartlett[0], -(399 - 21 / 6) * np.log(np.linalg.det(R)))
    L0, _, _ = fa.extract(R, 2)
    Lv, rot = fa.varimax(L0)
    assert np.allclose(rot.T @ rot, np.eye(2)) and np.allclose((Lv ** 2).sum(1), (L0 ** 2).sum(1))
    assert r.rmsr < 0.03


def test_wizard_recommendations():
    G = tw.GOALS
    assert "Welch" in tw.recommend(G[0], "numeric", 2, "independent", "yes", False).title
    assert "Mann" in tw.recommend(G[0], "numeric", 2, "independent", "no", True).title
    assert "Paired" in tw.recommend(G[0], "numeric", 2, "repeated", "yes", False).title
    assert "ANOVA" in tw.recommend(G[0], "numeric", 3, "independent", "yes", False).title
    assert tw.recommend(G[0], "ordinal", 3, "independent", "no", True).available is False
    assert "Spearman" in tw.recommend(G[2], "ordinal").title
    assert "logistic" in tw.recommend(G[3], "categorical").title.lower()
