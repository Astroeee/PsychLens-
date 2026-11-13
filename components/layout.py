"""Shared layout: CSS loader, custom sidebar, page header."""
from datetime import datetime

import streamlit as st

import config
from modules.state import setting

LOGO = ('<svg viewBox="0 0 32 32" fill="none"><path d="M16 3 3 10l13 7 13-7z" fill="#1D66D6"/>'
        '<path d="m3 15 13 7 13-7" stroke="#1D66D6" stroke-width="2.4" stroke-linejoin="round"/>'
        '<path d="m3 20 13 7 13-7" stroke="#7AA7EA" stroke-width="2.4" stroke-linejoin="round"/></svg>')


def load_css():
    css = (config.ASSETS_DIR / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_sidebar():
    name = setting("researcher_name") or "Researcher"
    with st.sidebar:
        st.markdown(f'<div class="brand">{LOGO}<div>Psych<span>Lens</span></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="profile"><div class="av">{name[:1].upper()}</div><div>'
                    f'<div class="nm">{name}</div><div class="rl">Researcher workspace</div></div></div>',
                    unsafe_allow_html=True)
        for title, path, icon in config.NAV_MAIN:
            st.page_link(path, label=title, icon=icon)
        st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
        for title, path, icon in config.NAV_SECONDARY:
            st.page_link(path, label=title, icon=icon)
        active = st.session_state.get("active_dataset")
        if active:
            df = st.session_state.datasets.get(active)
            shape = f"{len(df)} rows × {df.shape[1]} columns" if df is not None else ""
            st.markdown(f'<div class="nav-divider"></div><div class="ds-chip">Active dataset<b>{active}</b>{shape}</div>',
                        unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    today = datetime.now().strftime("%B %d, %A")
    st.markdown(f'<div class="page-header"><h1>{title}</h1><p>{subtitle} · {today}</p></div>', unsafe_allow_html=True)
