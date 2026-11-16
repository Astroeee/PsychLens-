"""Charts. Every chart kind has an interactive Plotly version (build_figure)
and a Matplotlib version (render_png) that is used for image downloads and PDF reports.

Both read the same parameter dictionary:
    x, y, by, bins, title, xlabel, ylabel, agg, cols, method, trendline, kde
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy import stats

import config
from modules.correlation import correlation_matrix

try:  # Plotly is only needed for the interactive versions
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None

CHART_KINDS = ["Histogram", "Boxplot", "Bar chart", "Scatterplot", "Line graph",
               "Correlation heatmap", "Frequency distribution"]
AGGREGATES = ["mean", "median", "sum", "count"]
PRIMARY, ACCENT, PALETTE = config.COLORS["primary"], config.COLORS["accent"], config.PALETTE


# ------------------------------------------------------------ data helpers --
def _num(df, col) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def agg_table(df, x, y=None, agg="mean", by=None) -> pd.DataFrame:
    """Aggregate y by x (and optionally by a second grouping column).
    With no y, returns counts and percentages of x."""
    keys = [x] + ([by] if by else [])
    if y is None or agg == "count":
        t = df.groupby(keys, dropna=True, observed=True).size().rename("value").reset_index()
        t["se"] = np.nan
    else:
        d = df[keys].copy()
        d[y] = _num(df, y)
        g = d.dropna().groupby(keys, observed=True)[y]
        t = g.agg(agg).rename("value").reset_index()
        t["se"] = (g.std(ddof=1) / np.sqrt(g.count())).to_numpy() if agg == "mean" else np.nan
    tot = t.groupby(by)["value"].transform("sum") if by else t["value"].sum()
    t["percent"] = t["value"] / tot * 100
    try:
        t = t.sort_values(keys)
    except TypeError:
        pass
    return t.reset_index(drop=True)


def _trend(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 3 or d.iloc[:, 0].nunique() < 2:
        return None
    slope, icpt = np.polyfit(d.iloc[:, 0], d.iloc[:, 1], 1)
    r = np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1]
    xs = np.linspace(d.iloc[:, 0].min(), d.iloc[:, 0].max(), 50)
    return xs, icpt + slope * xs, slope, icpt, r ** 2


def _defaults(kind, p):
    p = dict(p)
    p.setdefault("title", kind)
    p.setdefault("bins", 20)
    p.setdefault("agg", "mean")
    return p


# ----------------------------------------------------------------- plotly ---
def _theme(fig, title, xlabel, ylabel, legend=True):
    fig.update_layout(
        template="plotly_white", title=dict(text=title, x=0.0, font=dict(size=16)),
        font=dict(family="Inter, Arial, sans-serif", size=12, color=config.COLORS["text"]),
        colorway=PALETTE, margin=dict(l=50, r=20, t=60, b=50), showlegend=legend,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title=xlabel, showgrid=False, linecolor="#D5D9E0")
    fig.update_yaxes(title=ylabel, gridcolor="#EEF0F3", zeroline=False)
    return fig


def build_figure(kind: str, df: pd.DataFrame, **params):
    """Return an interactive Plotly figure for the chosen chart kind."""
    if go is None:
        raise RuntimeError("Plotly is not installed. Run: pip install plotly")
    p = _defaults(kind, params)
    x, y, by = p.get("x"), p.get("y"), p.get("by")
    fig = go.Figure()

    if kind == "Histogram":
        groups = [(None, df)] if not by else list(df.groupby(by, observed=True))
        for i, (lvl, sub) in enumerate(groups):
            v = _num(sub, x).dropna()
            fig.add_trace(go.Histogram(x=v, nbinsx=p["bins"], name=str(lvl) if by else x,
                                       marker_color=PALETTE[i % len(PALETTE)],
                                       opacity=0.6 if by else 0.85))
            if p.get("kde") and v.nunique() > 1:
                xs = np.linspace(v.min(), v.max(), 200)
                width = (v.max() - v.min()) / p["bins"]
                fig.add_trace(go.Scatter(x=xs, y=stats.gaussian_kde(v)(xs) * len(v) * width, mode="lines",
                                         name=f"Density{'' if not by else ' – ' + str(lvl)}",
                                         line=dict(color=PALETTE[(i + 1) % len(PALETTE)], width=2)))
        fig.update_layout(barmode="overlay")
        return _theme(fig, p["title"], p.get("xlabel") or x, p.get("ylabel") or "Frequency", legend=bool(by or p.get("kde")))

    if kind == "Boxplot":
        if by:
            for i, (lvl, sub) in enumerate(df.groupby(by, observed=True)):
                fig.add_trace(go.Box(y=_num(sub, y).dropna(), name=str(lvl), boxpoints="outliers",
                                     marker_color=PALETTE[i % len(PALETTE)]))
        else:
            fig.add_trace(go.Box(y=_num(df, y).dropna(), name=y, boxpoints="outliers", marker_color=PRIMARY))
        return _theme(fig, p["title"], p.get("xlabel") or (by or ""), p.get("ylabel") or y, legend=False)

    if kind in ("Bar chart", "Frequency distribution"):
        agg = "count" if (kind == "Frequency distribution" or not y) else p["agg"]
        t = agg_table(df, x, None if agg == "count" else y, agg, by)
        t[x] = t[x].astype(str)
        if by:
            for i, (lvl, sub) in enumerate(t.groupby(by)):
                fig.add_trace(go.Bar(x=sub[x], y=sub["value"], name=str(lvl), marker_color=PALETTE[i % len(PALETTE)]))
            fig.update_layout(barmode="group")
        else:
            err = dict(type="data", array=t["se"].fillna(0)) if t["se"].notna().any() else None
            txt = [f"{v:.1f}%" for v in t["percent"]] if agg == "count" else None
            fig.add_trace(go.Bar(x=t[x], y=t["value"], marker_color=PRIMARY, error_y=err,
                                 text=txt, textposition="outside"))
        ylab = "Frequency" if agg == "count" else f"{agg.capitalize()} of {y}"
        return _theme(fig, p["title"], p.get("xlabel") or x, p.get("ylabel") or ylab, legend=bool(by))

    if kind == "Scatterplot":
        groups = [(None, df)] if not by else list(df.groupby(by, observed=True))
        for i, (lvl, sub) in enumerate(groups):
            fig.add_trace(go.Scatter(x=_num(sub, x), y=_num(sub, y), mode="markers", name=str(lvl) if by else "Observations",
                                     marker=dict(color=PALETTE[i % len(PALETTE)], size=8, opacity=0.75)))
        if p.get("trendline"):
            tr = _trend(_num(df, x), _num(df, y))
            if tr:
                fig.add_trace(go.Scatter(x=tr[0], y=tr[1], mode="lines",
                                         name=f"OLS fit (R² = {tr[4]:.2f})", line=dict(color=config.COLORS["danger"], width=2)))
        return _theme(fig, p["title"], p.get("xlabel") or x, p.get("ylabel") or y)

    if kind == "Line graph":
        d = df.copy()
        d[x] = pd.to_numeric(d[x], errors="coerce") if pd.api.types.is_numeric_dtype(d[x]) else d[x]
        t = agg_table(d, x, y, p["agg"], by)
        groups = [(None, t)] if not by else list(t.groupby(by))
        for i, (lvl, sub) in enumerate(groups):
            fig.add_trace(go.Scatter(x=sub[x], y=sub["value"], mode="lines+markers", name=str(lvl) if by else y,
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
        return _theme(fig, p["title"], p.get("xlabel") or x, p.get("ylabel") or f"{p['agg'].capitalize()} of {y}", legend=bool(by))

    if kind == "Correlation heatmap":
        res = correlation_matrix(df, p["cols"], p.get("method", "pearson"))
        z = res.r
        fig.add_trace(go.Heatmap(z=z.to_numpy(), x=list(z.columns), y=list(z.index), zmin=-1, zmax=1,
                                 colorscale=[[0, config.COLORS["danger"]], [0.5, "#FFFFFF"], [1, PRIMARY]],
                                 text=z.round(2).astype(str).to_numpy(), texttemplate="%{text}",
                                 colorbar=dict(title=res.symbol)))
        fig.update_yaxes(autorange="reversed")
        return _theme(fig, p["title"], p.get("xlabel") or "", p.get("ylabel") or "", legend=False)

    raise ValueError(f"Unknown chart kind: {kind}")


def figure_to_png(fig) -> bytes | None:
    """PNG bytes of a Plotly figure (needs the 'kaleido' package); None if unavailable."""
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------- matplotlib ---
def render_png(kind: str, df: pd.DataFrame, dpi: int = 160, size=(7.5, 4.6), **params) -> bytes:
    """Static PNG version of a chart (always available; used in the PDF report)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = _defaults(kind, params)
    x, y, by = p.get("x"), p.get("y"), p.get("by")
    fig, ax = plt.subplots(figsize=size)
    ax.spines[["top", "right"]].set_visible(False)

    if kind == "Histogram":
        groups = [(None, df)] if not by else list(df.groupby(by, observed=True))
        for i, (lvl, sub) in enumerate(groups):
            v = _num(sub, x).dropna()
            ax.hist(v, bins=p["bins"], color=PALETTE[i % len(PALETTE)], alpha=0.6 if by else 0.85,
                    label=str(lvl) if by else None, edgecolor="white")
            if p.get("kde") and v.nunique() > 1:
                xs = np.linspace(v.min(), v.max(), 200)
                ax.plot(xs, stats.gaussian_kde(v)(xs) * len(v) * (v.max() - v.min()) / p["bins"],
                        color=PALETTE[(i + 1) % len(PALETTE)])
        xl, yl = x, "Frequency"
    elif kind == "Boxplot":
        if by:
            data = [_num(s, y).dropna() for _, s in df.groupby(by, observed=True)]
            labels = [str(l) for l, _ in df.groupby(by, observed=True)]
        else:
            data, labels = [_num(df, y).dropna()], [y]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for i, box in enumerate(bp["boxes"]):
            box.set(facecolor=PALETTE[i % len(PALETTE)], alpha=0.6)
        xl, yl = by or "", y
    elif kind in ("Bar chart", "Frequency distribution"):
        agg = "count" if (kind == "Frequency distribution" or not y) else p["agg"]
        t = agg_table(df, x, None if agg == "count" else y, agg, by)
        t[x] = t[x].astype(str)
        if by:
            piv = t.pivot(index=x, columns=by, values="value")
            piv.plot(kind="bar", ax=ax, color=PALETTE[:piv.shape[1]], rot=0, width=0.8)
        else:
            bars = ax.bar(t[x], t["value"], color=PRIMARY, yerr=t["se"].fillna(0) if t["se"].notna().any() else None, capsize=3)
            if agg == "count":
                for b, pc in zip(bars, t["percent"]):
                    ax.annotate(f"{pc:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8)
        xl, yl = x, "Frequency" if agg == "count" else f"{agg.capitalize()} of {y}"
    elif kind == "Scatterplot":
        groups = [(None, df)] if not by else list(df.groupby(by, observed=True))
        for i, (lvl, sub) in enumerate(groups):
            ax.scatter(_num(sub, x), _num(sub, y), s=22, alpha=0.75, color=PALETTE[i % len(PALETTE)], label=str(lvl) if by else None)
        if p.get("trendline"):
            tr = _trend(_num(df, x), _num(df, y))
            if tr:
                ax.plot(tr[0], tr[1], color=config.COLORS["danger"], label=f"OLS fit (R² = {tr[4]:.2f})")
        xl, yl = x, y
    elif kind == "Line graph":
        t = agg_table(df, x, y, p["agg"], by)
        groups = [(None, t)] if not by else list(t.groupby(by))
        for i, (lvl, sub) in enumerate(groups):
            ax.plot(sub[x].astype(str) if not pd.api.types.is_numeric_dtype(sub[x]) else sub[x], sub["value"], marker="o",
                    color=PALETTE[i % len(PALETTE)], label=str(lvl) if by else None)
        xl, yl = x, f"{p['agg'].capitalize()} of {y}"
    elif kind == "Correlation heatmap":
        import seaborn as sns
        res = correlation_matrix(df, p["cols"], p.get("method", "pearson"))
        sns.heatmap(res.r, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax, cmap=sns.diverging_palette(15, 255, as_cmap=True),
                    cbar_kws={"label": res.symbol}, linewidths=0.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        xl = yl = ""
    else:
        plt.close(fig)
        raise ValueError(f"Unknown chart kind: {kind}")

    ax.set_title(p["title"], loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(p.get("xlabel") or xl)
    ax.set_ylabel(p.get("ylabel") or yl)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()
