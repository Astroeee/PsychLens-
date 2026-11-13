"""Pearson and Spearman correlations with p-values, CIs and multiplicity correction."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def strength_label(r: float) -> str:
    a = abs(r)
    if not np.isfinite(a): return "n/a"
    if a < 0.1: return "negligible"
    if a < 0.3: return "small"
    if a < 0.5: return "moderate"
    return "large"


def direction_label(r: float) -> str:
    if not np.isfinite(r) or r == 0: return "none"
    return "positive" if r > 0 else "negative"


def correlate_pair(x: pd.Series, y: pd.Series, method: str = "pearson", ci_level: float = 0.95) -> dict:
    """Pairwise-complete correlation. Returns r, p, n, CI (Fisher z) and notes."""
    d = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    n = len(d)
    out = {"r": np.nan, "p": np.nan, "n": n, "ci_low": np.nan, "ci_high": np.nan}
    if n < 3 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return out
    a, b = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
    r, p = stats.pearsonr(a, b) if method == "pearson" else stats.spearmanr(a, b)
    out.update(r=float(r), p=float(p))
    if n > 3 and abs(r) < 1:
        se = 1 / np.sqrt(n - 3) if method == "pearson" else np.sqrt((1 + r ** 2 / 2) / (n - 3))
        z, crit = np.arctanh(r), stats.norm.ppf(1 - (1 - ci_level) / 2)
        out["ci_low"], out["ci_high"] = float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))
    return out


def p_adjust(pvals, method: str = "none") -> np.ndarray:
    """Bonferroni, Holm or Benjamini-Hochberg (fdr_bh) adjustment; NaNs are ignored."""
    p = np.asarray(pvals, dtype=float)
    out = p.copy()
    ok = np.isfinite(p)
    m = ok.sum()
    if method == "none" or m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    if method == "bonferroni":
        adj = np.minimum(pv * m, 1)
    elif method == "holm":
        s = pv[order] * (m - np.arange(m))
        adj = np.empty(m)
        adj[order] = np.minimum(np.maximum.accumulate(s), 1)
    elif method == "fdr_bh":
        s = pv[order] * m / (np.arange(m) + 1)
        s = np.minimum.accumulate(s[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.minimum(s, 1)
    else:
        raise ValueError(f"Unknown adjustment method: {method}")
    out[ok] = adj
    return out


class CorrelationResult:
    """Holds r, p, adjusted p and n matrices plus a tidy table of variable pairs."""

    def __init__(self, r, p, p_adj, n, pairs, method, ci_level, adjust):
        self.r, self.p, self.p_adj, self.n = r, p, p_adj, n
        self.pairs, self.method, self.ci_level, self.adjust = pairs, method, ci_level, adjust

    @property
    def symbol(self) -> str:
        return "r" if self.method == "pearson" else "ρ"


def correlation_matrix(df: pd.DataFrame, cols: list[str], method: str = "pearson",
                       ci_level: float = 0.95, adjust: str = "none", alpha: float = 0.05) -> CorrelationResult:
    if len(cols) < 2:
        raise ValueError("Select at least two variables.")
    k = len(cols)
    r = pd.DataFrame(np.eye(k), index=cols, columns=cols)
    p = pd.DataFrame(np.zeros((k, k)), index=cols, columns=cols)
    n = pd.DataFrame(0, index=cols, columns=cols, dtype=int)
    recs = []
    for a, b in combinations(cols, 2):
        res = correlate_pair(df[a], df[b], method, ci_level)
        r.loc[a, b] = r.loc[b, a] = res["r"]
        p.loc[a, b] = p.loc[b, a] = res["p"]
        n.loc[a, b] = n.loc[b, a] = res["n"]
        recs.append({"Variable 1": a, "Variable 2": b, "r": res["r"], "CI lower": res["ci_low"],
                     "CI upper": res["ci_high"], "p": res["p"], "N": res["n"]})
    for c in cols:
        n.loc[c, c] = int(df[c].notna().sum())
    pairs = pd.DataFrame(recs)
    pairs["p (adjusted)"] = p_adjust(pairs["p"].to_numpy(), adjust)
    pairs["r²"] = pairs["r"] ** 2
    pairs["Direction"] = pairs["r"].map(direction_label)
    pairs["Strength"] = pairs["r"].map(strength_label)
    pairs["Significant"] = np.where(pairs["p (adjusted)"] < alpha, "Yes", "No")
    p_adj = p.copy()
    for _, row in pairs.iterrows():
        p_adj.loc[row["Variable 1"], row["Variable 2"]] = row["p (adjusted)"]
        p_adj.loc[row["Variable 2"], row["Variable 1"]] = row["p (adjusted)"]
    if method == "spearman":
        pairs = pairs.rename(columns={"r": "ρ", "r²": "ρ²"})
    return CorrelationResult(r, p, p_adj, n, pairs, method, ci_level, adjust)


def _apa_r(x: float) -> str:
    s = f"{x:.2f}"
    return s.replace("0.", ".", 1) if abs(x) < 1 else s


def apa_matrix(res: CorrelationResult) -> pd.DataFrame:
    """Lower-triangle correlation table with significance stars (APA style)."""
    cols = list(res.r.columns)
    out = pd.DataFrame("", index=cols, columns=cols)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j:
                out.loc[a, b] = "—"
            elif j < i:
                v, pv = res.r.loc[a, b], res.p_adj.loc[a, b]
                if not np.isfinite(v):
                    out.loc[a, b] = "n/a"
                else:
                    stars = "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else ""
                    out.loc[a, b] = f"{_apa_r(v)}{stars}"
    return out.reset_index().rename(columns={"index": "Variable"})
