"""Exploratory factor analysis (EFA) with NumPy/SciPy.

KMO and Bartlett's test, eigenvalues with Kaiser and parallel-analysis guidance,
principal axis factoring (or PCA), varimax / promax / no rotation.
Pearson correlations are used; for short Likert items polychoric correlations are
often preferred, but are not implemented here. No Streamlit imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------- suitability ------
def bartlett_sphericity(R: np.ndarray, n: int) -> tuple[float, int, float]:
    p = R.shape[0]
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        return np.nan, p * (p - 1) // 2, np.nan
    chi2 = -(n - 1 - (2 * p + 5) / 6) * logdet
    df = p * (p - 1) // 2
    return float(chi2), df, float(stats.chi2.sf(chi2, df))


def kmo(R: np.ndarray) -> tuple[float, np.ndarray]:
    """Overall Kaiser–Meyer–Olkin measure and per-item MSA."""
    inv = np.linalg.pinv(R)
    d = np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    partial = -inv / d
    np.fill_diagonal(partial, 0)
    r = R.copy()
    np.fill_diagonal(r, 0)
    r2, p2 = (r ** 2).sum(), (partial ** 2).sum()
    msa = (r ** 2).sum(axis=0) / ((r ** 2).sum(axis=0) + (partial ** 2).sum(axis=0))
    return float(r2 / (r2 + p2)), msa


def kmo_label(v: float) -> str:
    return ("marvelous" if v >= .9 else "meritorious" if v >= .8 else "middling" if v >= .7 else
            "mediocre" if v >= .6 else "miserable" if v >= .5 else "unacceptable")


def parallel_analysis(n: int, p: int, n_iter: int = 200, percentile: float = 95, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eig = np.empty((n_iter, p))
    for i in range(n_iter):
        x = rng.standard_normal((n, p))
        eig[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(x, rowvar=False)))[::-1]
    return np.percentile(eig, percentile, axis=0)


# ----------------------------------------------------------- extraction -----
def extract(R: np.ndarray, m: int, method: str = "paf", max_iter: int = 500, tol: float = 1e-8):
    """Return (loadings p×m, communalities, converged)."""
    p = R.shape[0]
    if method == "pca":
        w, v = np.linalg.eigh(R)
        idx = np.argsort(w)[::-1][:m]
        L = v[:, idx] * np.sqrt(w[idx])
        return L, (L ** 2).sum(axis=1), True
    h = 1 - 1 / np.diag(np.linalg.pinv(R))             # squared multiple correlations
    converged = False
    for _ in range(max_iter):
        Rr = R.copy()
        np.fill_diagonal(Rr, h)
        w, v = np.linalg.eigh(Rr)
        idx = np.argsort(w)[::-1][:m]
        L = v[:, idx] * np.sqrt(np.clip(w[idx], 0, None))
        h_new = (L ** 2).sum(axis=1)
        if np.abs(h_new - h).max() < tol:
            h, converged = h_new, True
            break
        h = h_new
    return L, h, converged


# ------------------------------------------------------------- rotation -----
def varimax(L: np.ndarray, gamma: float = 1.0, max_iter: int = 1000, tol: float = 1e-8, normalize: bool = True):
    p, k = L.shape
    if k < 2:
        return L.copy(), np.eye(k)
    w = np.sqrt((L ** 2).sum(axis=1)) if normalize else np.ones(p)
    w[w == 0] = 1
    Ln = L / w[:, None]
    R, d = np.eye(k), 0.0
    for _ in range(max_iter):
        d_old = d
        Lam = Ln @ R
        u, s, vh = np.linalg.svd(Ln.T @ (Lam ** 3 - (gamma / p) * Lam @ np.diag((Lam ** 2).sum(axis=0))))
        R = u @ vh
        d = s.sum()
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return (Ln @ R) * w[:, None], R


def promax(L: np.ndarray, power: int = 4):
    """Oblique rotation. Returns (pattern matrix, factor correlation matrix)."""
    if L.shape[1] < 2:
        return L.copy(), np.eye(L.shape[1])
    Lv, _ = varimax(L)
    Q = Lv * np.abs(Lv) ** (power - 1)
    U = np.linalg.lstsq(Lv, Q, rcond=None)[0]
    d = np.diag(np.linalg.inv(U.T @ U))
    U = U @ np.diag(np.sqrt(d))
    P = Lv @ U
    Ui = np.linalg.inv(U)
    return P, Ui @ Ui.T


# --------------------------------------------------------------- driver -----
@dataclass
class EFAResult:
    items: list
    n: int
    n_total: int
    kmo_overall: float
    msa: pd.DataFrame
    bartlett: tuple
    eigenvalues: np.ndarray
    parallel: np.ndarray
    kaiser_k: int
    parallel_k: int
    n_factors: int
    method: str
    rotation: str
    loadings: pd.DataFrame            # pattern matrix
    structure: pd.DataFrame | None
    communalities: pd.DataFrame
    variance: pd.DataFrame
    factor_corr: pd.DataFrame | None
    assignment: pd.DataFrame
    rmsr: float
    pct_large_resid: float
    converged: bool
    cutoff: float
    warnings: list = field(default_factory=list)


def analyze(df: pd.DataFrame, items: list[str], n_factors: int | None = None, method: str = "paf",
            rotation: str = "varimax", cutoff: float = 0.30, n_iter: int = 200) -> EFAResult:
    if len(items) < 3:
        raise ValueError("Select at least 3 items.")
    data = df[items].apply(pd.to_numeric, errors="coerce")
    n_total = len(data)
    d = data.dropna()
    n, p = d.shape
    if n < p + 2:
        raise ValueError(f"Only {n} complete respondents for {p} items; more cases than items are required.")
    sd0 = d.std(ddof=1)
    if (sd0 == 0).any():
        raise ValueError("Items with no variance: " + ", ".join(sd0[sd0 == 0].index))
    R = np.corrcoef(d.to_numpy(float), rowvar=False)
    warns = []
    if n_total - n: warns.append(f"{n_total - n} respondent(s) excluded because of missing values (listwise deletion).")
    if n < 100 or n / p < 5:
        warns.append(f"Sample is small for factor analysis (N = {n}, {n / p:.1f} respondents per item); "
                     "common guidance is N ≥ 100 and 5–10 respondents per item. Treat results as exploratory.")
    if np.linalg.matrix_rank(R) < p or np.linalg.det(R) < 1e-10:
        warns.append("The correlation matrix is (nearly) singular: some items are almost perfectly correlated. "
                     "Consider removing redundant items.")
    bart = bartlett_sphericity(R, n)
    k_all, msa = kmo(R)
    if k_all < 0.6: warns.append(f"KMO = {k_all:.2f} is below .60: the data may not be suitable for factor analysis.")
    if np.isfinite(bart[2]) and bart[2] >= .05: warns.append("Bartlett's test is not significant: items may be uncorrelated.")
    eig = np.sort(np.linalg.eigvalsh(R))[::-1]
    pa = parallel_analysis(n, p, n_iter)
    kaiser_k = int((eig > 1).sum())
    par_k = 0
    for e, t in zip(eig, pa):
        if e > t: par_k += 1
        else: break
    m = int(n_factors or max(par_k, 1))
    if m >= p: raise ValueError("The number of factors must be smaller than the number of items.")

    L, h, conv = extract(R, m, method)
    if not conv: warns.append("The extraction did not fully converge.")
    if (h > 1).any(): warns.append("Heywood case: at least one communality exceeds 1. Try fewer factors.")
    # sign convention first, then rotate
    phi, structure = None, None
    if rotation == "varimax":
        P, _ = varimax(L)
    elif rotation == "promax":
        P, phi = promax(L)
    else:
        P = L.copy()
    ss_before = (P ** 2).sum(axis=0)
    order = np.argsort(ss_before)[::-1]
    P = P[:, order]
    if phi is not None:
        phi = phi[np.ix_(order, order)]
    for j in range(P.shape[1]):               # make the largest loading of each factor positive
        if P[np.abs(P[:, j]).argmax(), j] < 0:
            P[:, j] *= -1
            if phi is not None:
                phi[j, :] *= -1; phi[:, j] *= -1
    fnames = [f"F{i + 1}" for i in range(m)]
    comm = np.diag(P @ phi @ P.T) if phi is not None else (P ** 2).sum(axis=1)
    load_df = pd.DataFrame(P, index=items, columns=fnames)
    struct_df = pd.DataFrame(P @ phi, index=items, columns=fnames) if phi is not None else None
    comm_df = pd.DataFrame({"Item": items, "Extraction (communality)": comm, "Uniqueness": 1 - comm,
                            "KMO (MSA)": msa})
    ss = (P ** 2).sum(axis=0)
    var_df = pd.DataFrame({"Factor": fnames, "SS loadings": ss, "% of variance": ss / p * 100})
    if phi is None:
        var_df["Cumulative %"] = var_df["% of variance"].cumsum()
    else:
        warns.append("With an oblique (promax) rotation factors are correlated, so the variance shares overlap and are not additive.")
    # reproduced-correlation fit
    recon = (P @ phi @ P.T) if phi is not None else P @ P.T
    resid = R - recon
    off = ~np.eye(p, dtype=bool)
    rmsr = float(np.sqrt((resid[off] ** 2).mean()))
    large = float((np.abs(resid[off]) > 0.05).mean() * 100)
    primary = np.abs(P).argmax(axis=1)
    cross = [(np.abs(P[i]) >= cutoff).sum() > 1 for i in range(p)]
    assign = pd.DataFrame({"Item": items, "Primary factor": [fnames[j] for j in primary],
                           "Primary loading": P[np.arange(p), primary],
                           "Cross-loading (≥ cutoff on >1 factor)": ["Yes" if c else "No" for c in cross],
                           "No loading ≥ cutoff": ["Yes" if np.abs(P[i]).max() < cutoff else "No" for i in range(p)]})
    return EFAResult(items, n, n_total, k_all, pd.DataFrame({"Item": items, "MSA": msa}), bart, eig, pa, kaiser_k, par_k, m,
                     method, rotation, load_df, struct_df, comm_df, var_df,
                     pd.DataFrame(phi, index=fnames, columns=fnames) if phi is not None else None, assign, rmsr, large, conv,
                     cutoff, warns)


def apa_summary(r: EFAResult) -> str:
    b = r.bartlett
    meth = "principal axis factoring" if r.method == "paf" else "principal components analysis"
    rot = {"varimax": "varimax rotation", "promax": "promax rotation", "none": "no rotation"}[r.rotation]
    return (f"KMO = {r.kmo_overall:.2f}, Bartlett's χ²({b[1]}) = {b[0]:.2f}, p {'< .001' if b[2] < .001 else '= ' + f'{b[2]:.3f}'.lstrip('0')}; "
            f"{r.n_factors}-factor solution ({meth}, {rot}), N = {r.n}")


def interpret(r: EFAResult) -> str:
    cross = r.assignment[r.assignment["Cross-loading (≥ cutoff on >1 factor)"] == "Yes"]["Item"].tolist()
    none = r.assignment[r.assignment["No loading ≥ cutoff"] == "Yes"]["Item"].tolist()
    txt = (f"Sampling adequacy was {kmo_label(r.kmo_overall)} (KMO = {r.kmo_overall:.2f}). The Kaiser criterion (eigenvalue > 1) "
           f"suggests {r.kaiser_k} factor(s) and parallel analysis suggests {r.parallel_k}; a {r.n_factors}-factor solution was extracted. "
           f"Residual correlations were small (RMSR = {r.rmsr:.3f}; {r.pct_large_resid:.0f}% of residuals exceed |.05|). ")
    if cross: txt += f"Items loading ≥ {r.cutoff:.2f} on more than one factor: {', '.join(cross)}. "
    if none: txt += f"Items with no loading ≥ {r.cutoff:.2f}: {', '.join(none)}. "
    return txt + ("The number of factors and the naming of factors should rest on theory and item content, not on these "
                  "statistics alone. Consider confirming the structure in an independent sample (confirmatory factor analysis).")
