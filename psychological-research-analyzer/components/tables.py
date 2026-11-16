"""Display helpers that work across Streamlit versions."""
import pandas as pd
import streamlit as st


def show_df(df: pd.DataFrame, height=None, **kwargs):
    kw = dict(hide_index=True, **kwargs)
    if height: kw["height"] = height
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def show_plot(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_image(data, caption=None):
    try:
        st.image(data, caption=caption, width="stretch")
    except TypeError:
        st.image(data, caption=caption, use_container_width=True)


def fmt(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round float columns for display."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(decimals)
    return out


def assumptions_table(assumptions: list[dict]) -> pd.DataFrame:
    icon = {"met": "✅ met", "flagged": "⚠️ flagged", "not assessed": "– not assessed",
            "confirmed by researcher": "☑ researcher"}
    df = pd.DataFrame(assumptions)
    df["Result"] = df["Result"].map(lambda r: icon.get(r, r))
    return df[["Check", "Test", "Statistic", "p", "Result", "Note"]]
