"""Scale reliability: Cronbach's alpha and item analysis (no Streamlit imports)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

GUIDELINES = pd.DataFrame(
    [("≥ .90", "Excellent (very high; may indicate redundant items)"),
     (".80 – .89", "Good"), (".70 – .79", "Acceptable"),
     (".60 – .69", "Questionable"), (".50 – .59", "Poor"), ("< .50", "Unacceptable by common conventions")],
    columns=["Alpha", "Common descriptive label"])


@dataclass
class ReliabilityResult:
    alpha: float
    alpha_standardized: float
    ci: tuple[float, float]
    ci_level: float
    n_items: int
    n_valid: int
    n_total: int
    mean_inter_item_r: float
    scale_mean: float
    scale_sd: float
    item_stats: pd.DataFrame
    inter_item_corr: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    @property
    def n_excluded(self) -> int:
        return self.n_total - self.n_valid


def reverse_score(df: pd.DataFrame, items: list[str], scale_min: float, scale_max: float) -> pd.DataFrame:
    """Return a COPY with the chosen items reverse-coded (min + max - x)."""
    out = df.copy()
    for c in items:
        out[c] = scale_min + scale_max - out[c]
    return out


def _alpha(data: np.ndarray) -> float:
    k = data.shape[1]
    total_var = data.sum(axis=1).var(ddof=1)
    if k < 2 or total_var == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - data.var(axis=0, ddof=1).sum() / total_var))


def alpha_confidence_interval(alpha: float, n: int, k: int, level: float = 0.95) -> tuple[float, float]:
    """Feldt (1965) F-based confidence interval for Cronbach's alpha."""
    if not np.isfinite(alpha) or n < 3 or k < 2:
        return (float("nan"), float("nan"))
    df1, df2 = n - 1, (n - 1) * (k - 1)
    g = 1 - level
    lower = 1 - (1 - alpha) * stats.f.ppf(1 - g / 2, df1, df2)
    upper = 1 - (1 - alpha) * stats.f.ppf(g / 2, df1, df2)
    return float(lower), float(upper)


def reliability_analysis(df: pd.DataFrame, items: list[str], ci_level: float = 0.95) -> ReliabilityResult:
    """Cronbach's alpha with listwise deletion (only complete respondents are used)."""
    if len(items) < 2:
        raise ValueError("Select at least two items.")
    data = df[items].apply(pd.to_numeric, errors="coerce")
    n_total = len(data)
    complete = data.dropna()
    n, k = len(complete), len(items)
    if n < 3:
        raise ValueError(f"Only {n} respondent(s) answered every selected item; at least 3 are required.")

    x = complete.to_numpy(dtype=float)
    alpha = _alpha(x)
    corr = complete.corr()
    off = corr.to_numpy()[np.triu_indices(k, 1)]
    rbar = float(np.nanmean(off)) if len(off) else float("nan")
    alpha_std = k * rbar / (1 + (k - 1) * rbar) if np.isfinite(rbar) else float("nan")
    total = x.sum(axis=1)

    rows = []
    for j, name in enumerate(items):
        rest = total - x[:, j]
        if x[:, j].std(ddof=1) == 0 or rest.std(ddof=1) == 0:
            r_it = float("nan")
        else:
            r_it = float(np.corrcoef(x[:, j], rest)[0, 1])
        a_del = _alpha(np.delete(x, j, axis=1)) if k > 2 else float("nan")
        rows.append({"Item": name, "Mean": x[:, j].mean(), "SD": x[:, j].std(ddof=1),
                     "Corrected item-total r": r_it, "Alpha if item deleted": a_del})
    item_stats = pd.DataFrame(rows)

    warns = []
    if n < 30:
        warns.append(f"Only {n} complete respondents: alpha estimates are unstable with small samples.")
    if k < 3:
        warns.append("Fewer than 3 items: alpha if item deleted cannot be computed and alpha is of limited use.")
    if n_total - n > 0:
        warns.append(f"{n_total - n} respondent(s) excluded because of missing values (listwise deletion).")
    zero = item_stats.loc[item_stats["SD"] == 0, "Item"].tolist()
    if zero:
        warns.append(f"Items with no variance: {', '.join(zero)}.")
    neg = item_stats.loc[item_stats["Corrected item-total r"] < 0, "Item"].tolist()
    if neg:
        warns.append(f"Negative corrected item-total correlation for {', '.join(neg)}: "
                     "check whether these items are reverse-worded and need reverse scoring.")
    if np.isfinite(alpha) and alpha < 0:
        warns.append("Negative alpha usually indicates items that are not all scored in the same direction.")

    return ReliabilityResult(
        alpha=alpha, alpha_standardized=alpha_std,
        ci=alpha_confidence_interval(alpha, n, k, ci_level), ci_level=ci_level,
        n_items=k, n_valid=n, n_total=n_total, mean_inter_item_r=rbar,
        scale_mean=float(total.mean()), scale_sd=float(total.std(ddof=1)),
        item_stats=item_stats, inter_item_corr=corr, warnings=warns)


def interpret_alpha(alpha: float) -> str:
    """Neutral description. It never declares a scale 'valid' or 'invalid'."""
    if not np.isfinite(alpha):
        return "Alpha could not be computed (check for items without variance)."
    if alpha >= 0.9: label = "excellent"
    elif alpha >= 0.8: label = "good"
    elif alpha >= 0.7: label = "acceptable"
    elif alpha >= 0.6: label = "questionable"
    elif alpha >= 0.5: label = "poor"
    else: label = "unacceptable"
    return (f"The alpha of {alpha:.2f} falls in the '{label}' range under commonly cited rules of thumb. "
            "These labels are conventions, not tests: alpha reflects internal consistency only, "
            "increases with the number of items, assumes items measure one construct, and says "
            "nothing on its own about validity.")


def apa_summary(res: ReliabilityResult) -> str:
    lo, hi = res.ci
    a = f"{res.alpha:.2f}".lstrip("0") if 0 <= res.alpha < 1 else f"{res.alpha:.2f}"
    ci = f"{res.ci_level * 100:.0f}% CI [{lo:.2f}, {hi:.2f}]".replace("0.", ".").replace("[-.", "[-.")
    return f"Cronbach's α = {a}, {ci}, k = {res.n_items} items, N = {res.n_valid}"
