"""Linear (OLS), hierarchical and logistic regression implemented with NumPy/SciPy.

Categorical predictors are dummy-coded (first level = reference unless specified).
Rows with missing values on any model variable are excluded (listwise) and the
number excluded is always reported. No Streamlit imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

CAT_WARN = "Dummy-coded terms are shown as 'Variable[level]' versus the reference level."


# ----------------------------------------------------------------- design ---
def is_categorical(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def build_design(df: pd.DataFrame, dv: str, predictors: list[str], refs: dict | None = None,
                 y_numeric: bool = True):
    """Return (X DataFrame with 'Intercept', y Series, n_excluded, numeric_terms)."""
    refs = refs or {}
    cols = [dv] + list(predictors)
    d = df[cols].copy()
    n0 = len(d)
    d = d.dropna()
    y = pd.to_numeric(d[dv], errors="coerce") if y_numeric else d[dv]
    parts, numeric_terms = [pd.Series(1.0, index=d.index, name="Intercept")], []
    for p in predictors:
        s = d[p]
        if is_categorical(s):
            levels = sorted(s.astype(str).unique())
            ref = str(refs.get(p, levels[0]))
            if ref not in levels:
                ref = levels[0]
            for lv in levels:
                if lv != ref:
                    parts.append(((s.astype(str) == lv).astype(float)).rename(f"{p}[{lv}]"))
        else:
            parts.append(pd.to_numeric(s, errors="coerce").astype(float).rename(p))
            numeric_terms.append(p)
    X = pd.concat(parts, axis=1)
    ok = X.notna().all(axis=1) & (y.notna() if y_numeric else True)
    return X[ok], y[ok], n0 - int(ok.sum()), numeric_terms


def _vif(X: np.ndarray) -> np.ndarray:
    """VIF for each non-intercept column (X includes the intercept in column 0)."""
    k = X.shape[1] - 1
    out = np.full(k, np.nan)
    if k < 2:
        return out
    for j in range(1, k + 1):
        others = np.delete(X, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        res = X[:, j] - others @ beta
        tss = ((X[:, j] - X[:, j].mean()) ** 2).sum()
        r2 = 1 - (res ** 2).sum() / tss if tss > 0 else np.nan
        out[j - 1] = 1 / (1 - r2) if r2 is not np.nan and r2 < 1 else np.inf
    return out


# -------------------------------------------------------------------- OLS ---
@dataclass
class OLSResult:
    dv: str
    predictors: list
    n: int
    n_excluded: int
    k: int                       # number of non-intercept terms
    r2: float
    adj_r2: float
    f: float
    f_df: tuple
    f_p: float
    se_estimate: float
    aic: float
    bic: float
    coef: pd.DataFrame
    diagnostics: pd.DataFrame
    data: pd.DataFrame           # fitted, residual, standardized residual, leverage, Cook's D (indexed like input rows)
    warnings: list = field(default_factory=list)
    ci_level: float = 0.95
    alpha: float = 0.05


def fit_ols(df: pd.DataFrame, dv: str, predictors: list[str], refs: dict | None = None,
            ci_level: float = 0.95, alpha: float = 0.05) -> OLSResult:
    if not predictors:
        raise ValueError("Select at least one predictor.")
    if dv in predictors:
        raise ValueError("The outcome cannot also be a predictor.")
    X, y, n_excl, num_terms = build_design(df, dv, predictors, refs)
    n, p = X.shape
    k = p - 1
    if n <= p:
        raise ValueError(f"Not enough complete cases ({n}) for {p} parameters.")
    A, yv = X.to_numpy(float), y.to_numpy(float)
    if np.linalg.matrix_rank(A) < p:
        raise ValueError("Perfect multicollinearity or a constant predictor: remove redundant predictors.")
    if yv.std() == 0:
        raise ValueError("The outcome has no variation.")

    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    fitted = A @ beta
    resid = yv - fitted
    rss, tss = (resid ** 2).sum(), ((yv - yv.mean()) ** 2).sum()
    dfe = n - p
    s2 = rss / dfe
    xtx_inv = np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), dfe)
    crit = stats.t.ppf(1 - (1 - ci_level) / 2, dfe)
    r2 = 1 - rss / tss
    adj = 1 - (1 - r2) * (n - 1) / dfe
    f = ((tss - rss) / k) / s2
    fp = float(stats.f.sf(f, k, dfe))
    llf = -n / 2 * (np.log(2 * np.pi) + np.log(rss / n) + 1)
    sd_y = yv.std(ddof=1)
    std_beta = [np.nan] + [beta[i + 1] * A[:, i + 1].std(ddof=1) / sd_y if X.columns[i + 1] in num_terms else np.nan
                           for i in range(k)]
    vif = np.concatenate([[np.nan], _vif(A)])
    coef = pd.DataFrame({"Term": X.columns, "B": beta, "SE": se, "β (std.)": std_beta, "t": t, "p": pv,
                         "CI lower": beta - crit * se, "CI upper": beta + crit * se, "VIF": vif})

    q, _ = np.linalg.qr(A)
    h = (q ** 2).sum(axis=1)
    stud = resid / np.sqrt(s2 * (1 - h))
    cooks = stud ** 2 / p * h / (1 - h)
    data = pd.DataFrame({"Fitted": fitted, "Residual": resid, "Std. residual": stud,
                         "Leverage": h, "Cook's D": cooks}, index=X.index)

    e2 = resid ** 2
    aux_beta, *_ = np.linalg.lstsq(A, e2, rcond=None)
    aux_r2 = 1 - ((e2 - A @ aux_beta) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum() if e2.std() > 0 else 0
    bp_p = float(stats.chi2.sf(n * aux_r2, k)) if k > 0 else np.nan
    sh = stats.shapiro(resid) if 3 <= n <= 5000 else None
    dw = float(((np.diff(resid)) ** 2).sum() / (resid ** 2).sum())
    max_vif = np.nanmax(vif) if np.isfinite(vif).any() or np.isinf(vif).any() else np.nan
    diag = pd.DataFrame([
        ("Normality of residuals", "Shapiro–Wilk", sh.statistic if sh else np.nan, sh.pvalue if sh else np.nan,
         "met" if sh and sh.pvalue > alpha else "flagged" if sh else "not assessed"),
        ("Constant variance (homoscedasticity)", "Breusch–Pagan (studentized)", n * aux_r2, bp_p,
         "met" if bp_p > alpha else "flagged"),
        ("Independence of residuals (row order)", "Durbin–Watson (≈2 = no autocorrelation)", dw, np.nan,
         "met" if 1.5 <= dw <= 2.5 else "flagged"),
        ("Multicollinearity", "Largest VIF (> 5 = concern, > 10 = severe)", max_vif, np.nan,
         "not assessed" if np.isnan(max_vif) else "met" if max_vif <= 5 else "flagged"),
        ("Influential cases", "Cook's D > 4/n", float((cooks > 4 / n).sum()), np.nan,
         "met" if (cooks > 4 / n).sum() == 0 else "flagged"),
    ], columns=["Check", "Test / statistic", "Statistic", "p", "Result"])

    warns = []
    if n_excl: warns.append(f"{n_excl} row(s) excluded because of missing values (listwise deletion).")
    if n < 10 * k: warns.append(f"Only {n / max(k, 1):.1f} cases per predictor; 10–20 per predictor is a common minimum.")
    if (np.abs(stud) > 3).any(): warns.append(f"{int((np.abs(stud) > 3).sum())} case(s) with |standardized residual| > 3.")
    if (h > 2 * p / n).any(): warns.append(f"{int((h > 2 * p / n).sum())} high-leverage case(s) (leverage > 2p/n).")
    if any(c for c in X.columns if "[" in c): warns.append(CAT_WARN)
    return OLSResult(dv, predictors, n, n_excl, k, float(r2), float(adj), float(f), (k, dfe), fp, float(np.sqrt(s2)),
                     float(-2 * llf + 2 * p), float(-2 * llf + np.log(n) * p), coef, diag, data, warns, ci_level, alpha)


def hierarchical_ols(df: pd.DataFrame, dv: str, blocks: list[list[str]], refs: dict | None = None) -> pd.DataFrame:
    """Sequential (hierarchical) regression: R², ΔR² and F-change for each block,
    all fitted on the same complete cases."""
    blocks = [b for b in blocks if b]
    allp = [p for b in blocks for p in b]
    if len(set(allp)) != len(allp):
        raise ValueError("A predictor can only appear in one block.")
    rows, prev_r2, prev_p, cum = [], 0.0, 1, []
    for i, b in enumerate(blocks, 1):
        cum += b
        X, y, _, _ = build_design(df, dv, allp, refs)         # common sample
        cols = ["Intercept"] + [c for c in X.columns if any(c == p or c.startswith(p + "[") for p in cum)]
        A, yv = X[cols].to_numpy(float), y.to_numpy(float)
        n, p_ = A.shape
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        rss = ((yv - A @ beta) ** 2).sum()
        tss = ((yv - yv.mean()) ** 2).sum()
        r2 = 1 - rss / tss
        dfe = n - p_
        f_model = (r2 / (p_ - 1)) / ((1 - r2) / dfe)
        d_df = p_ - prev_p
        f_chg = ((r2 - prev_r2) / d_df) / ((1 - r2) / dfe)
        rows.append({"Model": f"Step {i}: + {', '.join(b)}", "N": n, "R²": r2, "Adj. R²": 1 - (1 - r2) * (n - 1) / dfe,
                     "ΔR²": r2 - prev_r2, "F change": f_chg, "df": f"{d_df}, {dfe}",
                     "p (change)": float(stats.f.sf(f_chg, d_df, dfe)), "F (model)": f_model,
                     "p (model)": float(stats.f.sf(f_model, p_ - 1, dfe))})
        prev_r2, prev_p = r2, p_
    return pd.DataFrame(rows)


# --------------------------------------------------------------- logistic ---
@dataclass
class LogitResult:
    dv: str
    event: str
    reference: str
    predictors: list
    n: int
    n_excluded: int
    n_events: int
    coef: pd.DataFrame
    fit: pd.DataFrame
    classification: pd.DataFrame
    auc: float
    hosmer_lemeshow: tuple
    data: pd.DataFrame
    threshold: float
    warnings: list = field(default_factory=list)
    converged: bool = True
    ci_level: float = 0.95


def _irls(A: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-10):
    beta = np.zeros(A.shape[1])
    ll_old = -np.inf
    for it in range(max_iter):
        eta = np.clip(A @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-10, None)
        z = eta + (y - mu) / w
        wa = A * w[:, None]
        try:
            new = np.linalg.solve(A.T @ wa, A.T @ (w * z))
        except np.linalg.LinAlgError:
            return beta, False, it
        beta = new
        eta = np.clip(A @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        ll = (y * np.log(np.clip(mu, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - mu, 1e-15, 1))).sum()
        if abs(ll - ll_old) < tol:
            return beta, True, it
        ll_old = ll
    return beta, False, max_iter


def _auc(y, p):
    r = stats.rankdata(p)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def fit_logistic(df: pd.DataFrame, dv: str, predictors: list[str], event=None, refs: dict | None = None,
                 threshold: float = 0.5, ci_level: float = 0.95, alpha: float = 0.05) -> LogitResult:
    if not predictors:
        raise ValueError("Select at least one predictor.")
    levels = sorted(df[dv].dropna().astype(str).unique())
    if len(levels) != 2:
        raise ValueError(f"The outcome must have exactly 2 categories (found {len(levels)}).")
    event = str(event) if event is not None else levels[1]
    ref_lv = [l for l in levels if l != event][0]
    X, ycat, n_excl, _ = build_design(df, dv, predictors, refs, y_numeric=False)
    y = (ycat.astype(str) == event).astype(float).to_numpy()
    A = X.to_numpy(float)
    n, p = A.shape
    if np.linalg.matrix_rank(A) < p:
        raise ValueError("Perfect multicollinearity or a constant predictor: remove redundant predictors.")
    n1 = int(y.sum())
    if n1 < 3 or n - n1 < 3:
        raise ValueError("Each outcome category needs at least 3 cases.")
    beta, ok, iters = _irls(A, y)
    eta = np.clip(A @ beta, -30, 30)
    mu = 1 / (1 + np.exp(-eta))
    w = mu * (1 - mu)
    cov = np.linalg.pinv(A.T @ (A * w[:, None]))
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2 * stats.norm.sf(np.abs(z))
    crit = stats.norm.ppf(1 - (1 - ci_level) / 2)
    ll = (y * np.log(np.clip(mu, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - mu, 1e-15, 1))).sum()
    p0 = n1 / n
    ll0 = n1 * np.log(p0) + (n - n1) * np.log(1 - p0)
    lr = 2 * (ll - ll0)
    k = p - 1
    cs = 1 - np.exp(-lr / n)
    nag = cs / (1 - np.exp(2 * ll0 / n))
    coef = pd.DataFrame({"Term": X.columns, "B": beta, "SE": se, "Wald z": z, "p": pv, "Odds ratio": np.exp(beta),
                         "OR CI lower": np.exp(beta - crit * se), "OR CI upper": np.exp(beta + crit * se)})
    fit = pd.DataFrame([("−2 log-likelihood (model)", -2 * ll), ("−2 log-likelihood (null)", -2 * ll0),
                        (f"Likelihood-ratio χ²({k})", lr), ("p (model)", float(stats.chi2.sf(lr, k))),
                        ("McFadden pseudo-R²", 1 - ll / ll0), ("Cox & Snell R²", cs), ("Nagelkerke R²", nag),
                        ("AIC", -2 * ll + 2 * p), ("BIC", -2 * ll + np.log(n) * p)], columns=["Statistic", "Value"])
    pred = (mu >= threshold).astype(float)
    tp, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    fp, fn = ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
    cls = pd.DataFrame([("Accuracy", (tp + tn) / n), ("Sensitivity (true positive rate)", tp / (tp + fn) if tp + fn else np.nan),
                        ("Specificity (true negative rate)", tn / (tn + fp) if tn + fp else np.nan),
                        ("Precision (positive predictive value)", tp / (tp + fp) if tp + fp else np.nan),
                        ("True positives / false negatives", f"{tp} / {fn}"), ("False positives / true negatives", f"{fp} / {tn}")],
                       columns=["Metric", f"Value (cut-off {threshold:g})"])
    hl = (np.nan, np.nan, np.nan)
    try:
        grp = pd.qcut(pd.Series(mu).rank(method="first"), 10, labels=False)
        chi = 0.0
        for g in range(10):
            m = (grp == g).to_numpy()
            ng, o, e = m.sum(), y[m].sum(), mu[m].sum()
            if ng and 0 < e < ng:
                chi += (o - e) ** 2 / (e * (1 - e / ng))
        hl = (chi, 8, float(stats.chi2.sf(chi, 8)))
    except Exception:  # noqa: BLE001
        pass
    warns = []
    if not ok or np.abs(beta).max() > 15 or se.max() > 1e3:
        warns.append("The model may not have converged properly or shows (quasi-)complete separation: some predictor "
                     "almost perfectly predicts the outcome. Estimates and odds ratios may be unreliable.")
    epv = min(n1, n - n1) / max(k, 1)
    if epv < 10: warns.append(f"Only {epv:.1f} events per predictor (the smaller outcome group has {min(n1, n - n1)} cases); 10+ is a common guideline.")
    if n_excl: warns.append(f"{n_excl} row(s) excluded because of missing values (listwise deletion).")
    if any("[" in c for c in X.columns): warns.append(CAT_WARN)
    data = pd.DataFrame({"Predicted probability": mu, "Observed": y}, index=X.index)
    return LogitResult(dv, event, ref_lv, predictors, n, n_excl, n1, coef, fit, cls, _auc(y, mu), hl, data, threshold,
                       warns, bool(ok), ci_level)


def roc_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """False/true positive rates over all thresholds (for plotting)."""
    order = np.argsort(-p)
    y_s = y[order]
    tps, fps = np.cumsum(y_s), np.cumsum(1 - y_s)
    return pd.DataFrame({"FPR": np.r_[0, fps / max(fps[-1], 1)], "TPR": np.r_[0, tps / max(tps[-1], 1)]})


# ----------------------------------------------------------------- output ---
def _nl(x, d=2):
    s = f"{x:.{d}f}"
    return s.replace("0.", ".", 1) if abs(x) < 1 else s


def _p(p):
    return "p < .001" if p < .001 else f"p = {_nl(p, 3)}"


def apa_ols(r: OLSResult) -> str:
    return f"R² = {_nl(r.r2)}, adjusted R² = {_nl(r.adj_r2)}, F({r.f_df[0]}, {r.f_df[1]}) = {r.f:.2f}, {_p(r.f_p)}, N = {r.n}"


def apa_logit(r: LogitResult) -> str:
    lr = r.fit.set_index("Statistic")["Value"]
    k = len(r.coef) - 1
    return (f"χ²({k}, N = {r.n}) = {lr[f'Likelihood-ratio χ²({k})']:.2f}, {_p(lr['p (model)'])}, Nagelkerke R² = {_nl(lr['Nagelkerke R²'])}, "
            f"AUC = {_nl(r.auc)}")


def interpret_ols(r: OLSResult) -> str:
    sig = r.coef[(r.coef["p"] < r.alpha) & (r.coef["Term"] != "Intercept")]
    txt = (f"The model with {r.k} predictor term(s) {'was' if r.f_p < r.alpha else 'was not'} statistically significant at α = {_nl(r.alpha)} "
           f"({apa_ols(r)}) and accounted for {r.r2 * 100:.1f}% of the variance in {r.dv} ({r.adj_r2 * 100:.1f}% adjusted). ")
    if len(sig):
        txt += "Terms with a statistically significant coefficient: " + "; ".join(
            f"{t} (B = {b:.2f}, {_p(p)})" for t, b, p in zip(sig["Term"], sig["B"], sig["p"])) + ". "
    else:
        txt += "No individual coefficient was statistically significant. "
    flagged = r.diagnostics.loc[r.diagnostics["Result"] == "flagged", "Check"].tolist()
    if flagged:
        txt += "Diagnostic checks flagged: " + "; ".join(flagged) + ", so interpret with caution. "
    return txt + ("Regression coefficients describe associations in this sample; causal conclusions depend on the study design, "
                  "not on the model.")


def interpret_logit(r: LogitResult) -> str:
    sig = r.coef[(r.coef["p"] < .05) & (r.coef["Term"] != "Intercept")]
    txt = (f"The model predicts the odds of {r.dv} = '{r.event}' (versus '{r.reference}'). {apa_logit(r)}. ")
    if len(sig):
        txt += "Statistically significant predictors: " + "; ".join(
            f"{t} (OR = {o:.2f}, {_p(p)})" for t, o, p in zip(sig["Term"], sig["Odds ratio"], sig["p"])) + ". "
    else:
        txt += "No individual predictor was statistically significant. "
    return txt + ("An odds ratio above 1 means higher odds of the event as the predictor increases (or versus the reference level); "
                  "below 1 means lower odds. Associations do not establish causation.")
