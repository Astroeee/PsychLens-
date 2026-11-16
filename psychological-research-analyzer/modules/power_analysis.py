"""Statistical power and sample-size planning (a priori and sensitivity analysis).

Power is computed from noncentral t, F and chi-square distributions (as in G*Power).
Correlation power uses the Fisher z approximation. No Streamlit imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

BENCHMARKS = {"d": (0.2, 0.5, 0.8), "r": (0.1, 0.3, 0.5), "f": (0.1, 0.25, 0.4), "w": (0.1, 0.3, 0.5), "f2": (0.02, 0.15, 0.35)}

TESTS = {
    "Independent-samples t-test": {"effect": "d", "label": "Cohen's d"},
    "Paired-samples t-test": {"effect": "d", "label": "Cohen's d_z"},
    "One-sample t-test": {"effect": "d", "label": "Cohen's d"},
    "One-way ANOVA": {"effect": "f", "label": "Cohen's f"},
    "Correlation (Pearson)": {"effect": "r", "label": "r"},
    "Chi-square test": {"effect": "w", "label": "Cohen's w"},
    "Multiple regression (R² test)": {"effect": "f2", "label": "Cohen's f²"},
}
EFFECT_MAX = {"d": 4.0, "r": 0.999, "f": 3.0, "w": 2.0, "f2": 8.0}
N_MIN = {"Independent-samples t-test": 2, "Paired-samples t-test": 3, "One-sample t-test": 3, "One-way ANOVA": 3,
         "Correlation (Pearson)": 4, "Chi-square test": 2, "Multiple regression (R² test)": 3}


def _tail(stat_sf, stat_cdf, alternative):
    return stat_sf if alternative != "two-sided" else stat_sf + stat_cdf


def power_t(d, n, alpha=0.05, alternative="two-sided", design="independent", ratio=1.0) -> float:
    """design: independent (n = size of group 1, group 2 = ratio*n), paired or one-sample (n = pairs/cases)."""
    d = abs(d)
    if design == "independent":
        n2 = n * ratio
        df, nc = n + n2 - 2, d * np.sqrt(n * n2 / (n + n2))
    else:
        df, nc = n - 1, d * np.sqrt(n)
    if df < 1:
        return np.nan
    if alternative == "two-sided":
        tc = stats.t.ppf(1 - alpha / 2, df)
        res = float(stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))
    else:
        res = float(stats.nct.sf(stats.t.ppf(1 - alpha, df), df, nc))
    return 1.0 if np.isnan(res) else res        # SciPy returns NaN for extremely large noncentrality (power ≈ 1)


def power_anova(f, k, n_total, alpha=0.05) -> float:
    df1, df2 = k - 1, n_total - k
    if df2 < 1:
        return np.nan
    res = float(stats.ncf.sf(stats.f.ppf(1 - alpha, df1, df2), df1, df2, f ** 2 * n_total))
    return 1.0 if np.isnan(res) else res


def power_correlation(r, n, alpha=0.05, alternative="two-sided") -> float:
    if n <= 3:
        return np.nan
    z = np.arctanh(abs(r)) * np.sqrt(n - 3)
    if alternative == "two-sided":
        zc = stats.norm.ppf(1 - alpha / 2)
        return float(stats.norm.sf(zc - z) + stats.norm.cdf(-zc - z))
    return float(stats.norm.sf(stats.norm.ppf(1 - alpha) - z))


def power_chisq(w, n, df, alpha=0.05) -> float:
    res = float(stats.ncx2.sf(stats.chi2.ppf(1 - alpha, df), df, n * w ** 2))
    return 1.0 if np.isnan(res) else res


def power_regression(f2, n, k, alpha=0.05) -> float:
    df1, df2 = k, n - k - 1
    if df2 < 1:
        return np.nan
    res = float(stats.ncf.sf(stats.f.ppf(1 - alpha, df1, df2), df1, df2, f2 * n))
    return 1.0 if np.isnan(res) else res


def _power_fn(test: str, alpha: float, alternative: str, **kw):
    """Return power as a function of (effect, n) where n is per group (independent t) or total/cases otherwise."""
    if test == "Independent-samples t-test":
        return lambda e, n: power_t(e, n, alpha, alternative, "independent", kw.get("ratio", 1.0))
    if test == "Paired-samples t-test":
        return lambda e, n: power_t(e, n, alpha, alternative, "paired")
    if test == "One-sample t-test":
        return lambda e, n: power_t(e, n, alpha, alternative, "one-sample")
    if test == "One-way ANOVA":
        return lambda e, n: power_anova(e, kw["groups"], n, alpha)
    if test == "Correlation (Pearson)":
        return lambda e, n: power_correlation(e, n, alpha, alternative)
    if test == "Chi-square test":
        return lambda e, n: power_chisq(e, n, kw["df"], alpha)
    if test == "Multiple regression (R² test)":
        return lambda e, n: power_regression(e, n, kw["predictors"], alpha)
    raise ValueError(f"Unknown test: {test}")


def required_n(test, effect, power=0.80, alpha=0.05, alternative="two-sided", **kw) -> int:
    fn = _power_fn(test, alpha, alternative, **kw)
    lo = N_MIN[test] if test != "One-way ANOVA" else kw["groups"] + 1
    if test == "Multiple regression (R² test)":
        lo = kw["predictors"] + 2
    if test == "Chi-square test":
        lo = 3
    hi = max(lo * 2, 8)
    while not (fn(effect, hi) >= power):
        hi *= 2
        if hi > 5_000_000:
            raise ValueError("Required sample size is impractically large for this effect size.")
    lo_i = lo
    while lo_i < hi:                                   # smallest n with power >= target
        mid = (lo_i + hi) // 2
        v = fn(effect, mid)
        if v >= power: hi = mid
        else: lo_i = mid + 1
    if test == "One-way ANOVA":                         # equal group sizes: round up to a multiple of k
        lo_i = int(np.ceil(lo_i / kw["groups"]) * kw["groups"])
    return int(lo_i)


def sensitivity_effect(test, n, power=0.80, alpha=0.05, alternative="two-sided", **kw) -> float:
    """Smallest effect size detectable with the given n and power."""
    fn = _power_fn(test, alpha, alternative, **kw)
    top = EFFECT_MAX[TESTS[test]["effect"]]
    if fn(top, n) < power:
        raise ValueError("Even a very large effect cannot reach this power with that sample size.")
    return float(optimize.brentq(lambda e: fn(e, n) - power, 1e-6, top, xtol=1e-6))


def achieved_power(test, effect, n, alpha=0.05, alternative="two-sided", **kw) -> float:
    return float(_power_fn(test, alpha, alternative, **kw)(effect, n))


def power_curve(test, effect, n_values, alpha=0.05, alternative="two-sided", **kw) -> pd.DataFrame:
    fn = _power_fn(test, alpha, alternative, **kw)
    return pd.DataFrame({"n": list(n_values), "Power": [fn(effect, int(n)) for n in n_values]})


@dataclass
class PowerResult:
    test: str
    solve_for: str
    effect_name: str
    effect: float
    alpha: float
    power: float
    n: int                      # per group (independent t) or total / cases
    n_total: int
    n_unit: str
    inputs: dict = field(default_factory=dict)
    sentence: str = ""
    warnings: list = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        rows = [("Test", self.test), ("Solving for", self.solve_for), (self.effect_name, f"{self.effect:.3f}"),
                ("α (significance level)", f"{self.alpha:g}"), ("Power (1 − β)", f"{self.power:.3f}"),
                (f"Sample size ({self.n_unit})", self.n), ("Total sample size", self.n_total)]
        rows += [(k, v) for k, v in self.inputs.items()]
        return pd.DataFrame(rows, columns=["Parameter", "Value"]).astype(str)


def _n_total(test, n, **kw):
    if test == "Independent-samples t-test":
        return int(n + int(np.ceil(n * kw.get("ratio", 1.0))))
    return int(n)


def analyze(test: str, solve_for: str, effect=None, power=0.80, n=None, alpha=0.05, alternative="two-sided",
            dropout=0.0, **kw) -> PowerResult:
    """solve_for: 'Sample size' | 'Power' | 'Detectable effect size'."""
    meta = TESTS[test]
    unit = "per group" if test == "Independent-samples t-test" else "participants" if test != "Paired-samples t-test" else "pairs"
    if test in ("Chi-square test", "Multiple regression (R² test)", "One-way ANOVA", "Correlation (Pearson)", "One-sample t-test"):
        unit = "participants"
    if solve_for == "Sample size":
        n = required_n(test, effect, power, alpha, alternative, **kw)
        power_out = achieved_power(test, effect, n, alpha, alternative, **kw)
    elif solve_for == "Power":
        power_out = achieved_power(test, effect, int(n), alpha, alternative, **kw)
    else:
        effect = sensitivity_effect(test, int(n), power, alpha, alternative, **kw)
        power_out = achieved_power(test, effect, int(n), alpha, alternative, **kw)
    n = int(n)
    total = _n_total(test, n, **kw)
    inputs = {}
    if test == "Independent-samples t-test" and kw.get("ratio", 1.0) != 1.0: inputs["Allocation ratio (group 2 / group 1)"] = kw["ratio"]
    if test == "One-way ANOVA": inputs["Number of groups"] = kw["groups"]
    if test == "Chi-square test": inputs["Degrees of freedom"] = kw["df"]
    if test == "Multiple regression (R² test)": inputs["Number of predictors"] = kw["predictors"]
    if test in ("Independent-samples t-test", "Paired-samples t-test", "One-sample t-test", "Correlation (Pearson)"):
        inputs["Tails"] = alternative
    warns = []
    if solve_for == "Sample size" and dropout > 0:
        adj = int(np.ceil(n / (1 - dropout)))
        inputs[f"Recruit to allow {dropout * 100:.0f}% attrition ({unit})"] = adj
    if effect is not None and effect <= BENCHMARKS[meta["effect"]][0] / 2:
        warns.append("A very small effect size requires a very large sample; check that this effect is realistic and meaningful.")
    if test == "Correlation (Pearson)":
        warns.append("Correlation power uses the Fisher z approximation (accurate to within about 1 participant of exact methods).")
    en, el = meta["effect"], meta["label"]
    label = f"{el} = {effect:.2f}"
    tail = f", {alternative}" if test in ("Independent-samples t-test", "Paired-samples t-test", "One-sample t-test", "Correlation (Pearson)") else ""
    tname = test[0].lower() + test[1:]
    base = f"A power analysis for the {tname} (α = {alpha:g}{tail}; {label})"
    if solve_for == "Sample size":
        sent = (f"{base} indicated that {n} {unit}" + (f" (N = {total})" if test == "Independent-samples t-test" else "")
                + f" are required to achieve power = {power:.2f}.")
    elif solve_for == "Power":
        sent = f"{base} with {n} {unit} has power = {power_out:.2f} to detect this effect."
    else:
        sent = (f"With {n} {unit}, α = {alpha:g} and power = {power:.2f}, the {tname} can detect an effect of at least "
                f"{el} = {effect:.2f}.")
    sent += " The assumed effect size should be justified from prior research or the smallest effect of practical interest."
    return PowerResult(test, solve_for, el, float(effect), alpha, float(power_out), n, total, unit, inputs, sent, warns)
