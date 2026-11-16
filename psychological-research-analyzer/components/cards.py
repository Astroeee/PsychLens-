"""Reusable card-style UI pieces."""
from html import escape

import streamlit as st

ICONS = {
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2"/><circle cx="10" cy="7" r="4"/><path d="M21 21v-2a4 4 0 0 0-3-3.9"/><path d="M16 3.1a4 4 0 0 1 0 7.8"/>',
    "columns": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M15 3v18"/>',
    "alert": '<circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m8 12 3 3 5-6"/>',
    "copy": '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
}


def kpi_card(label: str, value, hint: str = "", icon: str = "database", tone: str = ""):
    st.markdown(f'<div class="kpi"><div class="ic {tone}"><svg viewBox="0 0 24 24">{ICONS[icon]}</svg></div>'
                f'<div><div class="label">{escape(label)}</div><div class="value">{value}</div>'
                f'<div class="hint">{escape(hint)}</div></div></div>', unsafe_allow_html=True)


def card_title(title: str, subtitle: str = ""):
    st.markdown(f'<div class="card-title">{escape(title)}</div>'
                + (f'<div class="card-sub">{escape(subtitle)}</div>' if subtitle else ""), unsafe_allow_html=True)


def callout(text: str, kind: str = "info"):
    """kind: info | warn | ok | plain. Text may contain simple HTML (<b>, <i>)."""
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def empty_state(title: str, text: str):
    st.markdown(f'<div class="empty"><h3>{escape(title)}</h3><div>{escape(text)}</div></div>', unsafe_allow_html=True)


def apa_box(text: str):
    from modules.report_generator import apa_markup
    st.markdown(f'<div class="apa-box">{apa_markup(text, html=True)}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "na") -> str:
    return f'<span class="pill {kind}">{escape(text)}</span>'
