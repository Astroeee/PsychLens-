"""PsychLens - Psychological Research Data Analyzer.

Run with:  streamlit run app.py
This file only sets up the page, theme, sidebar and routing; each section
lives in pages/ and the statistics live in modules/.
"""
import streamlit as st

import config
from components.auth_ui import render_auth_screen
from components.layout import load_css, render_sidebar
from modules.state import init_state, is_authenticated

st.set_page_config(page_title=config.APP_NAME, page_icon="🧠", layout="wide")

init_state()
load_css()

if not is_authenticated():          # nobody signed in: show sign-in / register / guest
    render_auth_screen()
    st.stop()

pages = [st.Page(path, title=title, icon=icon, default=(title == "Dashboard"))
         for title, path, icon in config.NAV_MAIN + config.NAV_SECONDARY]
current = st.navigation(pages, position="hidden")   # hide Streamlit's default menu
render_sidebar()                                    # draw our own
current.run()
