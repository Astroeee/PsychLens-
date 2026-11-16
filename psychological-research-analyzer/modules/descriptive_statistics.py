"""Descriptive statistics (no Streamlit imports)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _stats_for(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    n = len(x)
    modes = x.mode() if n else pd.Series(dtype=float)
    q1, q3 = (x.quantile(0.25), x.quantile(0.75)) if n else (np.nan, np.nan)
    shapiro_p = np.nan
    if 3 <= n <= 5000 and x.nunique() > 1:
        shapiro_p = stats.shapiro(x).pvalue
    return {
        "N": n,
        "Missing": int(len(s) - n),
        "Mean": x.mean() if n else np.nan,
        "Median": x.median() if n else np.nan,
        "Mode": modes.iloc[0] if len(modes) else np.nan,
        "Modes (count)": len(modes),
        "SD": x.std(ddof=1) if n > 1 else np.nan,
        "Variance": x.var(ddof=1) if n > 1 else np.nan,
        "SE": x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
        "Min": x.min() if n else np.nan,
        "Max": x.max() if n else np.nan,
        "Range": (x.max() - x.min()) if n else np.nan,
        "Skewness": stats.skew(x, bias=False) if n > 2 and x.nunique() > 1 else np.nan,
        "Kurtosis": stats.kurtosis(x, fisher=True, bias=False) if n > 3 and x.nunique() > 1 else np.nan,
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "Shapiro-Wilk p": shapiro_p,
    }


def describe_variables(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One row per variable. Skewness/kurtosis use the bias-corrected (SPSS-style)
    estimators; kurtosis is *excess* kurtosis (normal distribution = 0)."""
    rows = {c: _stats_for(df[c]) for c in cols}
    return pd.DataFrame(rows).T.rename_axis("Variable").reset_index()


def describe_by_group(df: pd.DataFrame, var: str, group: str) -> pd.DataFrame:
    rows = []
    for level, sub in df.groupby(group, dropna=True, sort=True):
        st = _stats_for(sub[var])
        rows.append({"Group": level, "N": st["N"], "Mean": st["Mean"], "SD": st["SD"],
                     "SE": st["SE"], "Median": st["Median"], "Min": st["Min"], "Max": st["Max"]})
    return pd.DataFrame(rows)


def frequency_table(s: pd.Series) -> pd.DataFrame:
    """Counts, percent, valid percent and cumulative percent (missing shown separately)."""
    counts = s.value_counts(dropna=False)
    try:
        counts = counts.sort_index() if pd.api.types.is_numeric_dtype(s) else counts
    except TypeError:
        pass
    total = int(counts.sum())
    valid_total = int(s.notna().sum())
    out = pd.DataFrame({"Value": counts.index.astype(object).where(counts.index.notna(), "(missing)"),
                        "Frequency": counts.values})
    out["Percent"] = out["Frequency"] / total * 100 if total else np.nan
    is_valid = out["Value"] != "(missing)"
    out["Valid %"] = np.where(is_valid, out["Frequency"] / valid_total * 100 if valid_total else np.nan, np.nan)
    out["Cumulative %"] = out["Valid %"].fillna(0).cumsum().where(is_valid)
    return out


def format_table(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round floats for display; integers stay integers."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(decimals)
    return out
