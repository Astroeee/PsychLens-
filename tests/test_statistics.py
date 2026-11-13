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
