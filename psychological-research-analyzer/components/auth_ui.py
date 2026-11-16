"""Sign-in / create-account screen shown before the app when nobody is signed in."""
import streamlit as st

from components.layout import LOGO
from modules import auth
from modules.state import guest_login, login


def render_auth_screen():
    # No sidebar while signed out
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],"
                "[data-testid='collapsedControl']{display:none !important}</style>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        st.markdown(f'<div class="auth-brand">{LOGO}<div>Psych<span>Lens</span></div></div>'
                    '<div class="auth-sub">Psychological research analytics.<br>Sign in to keep your datasets and settings.</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            tab_in, tab_up = st.tabs(["Sign in", "Create account"])
            with tab_in:
                with st.form("login_form"):
                    ident = st.text_input("Username or email", key="auth_ident")
                    pw = st.text_input("Password", type="password", key="auth_pw")
                    go = st.form_submit_button("Sign in", type="primary")
                if go:
                    try:
                        login(auth.authenticate(ident, pw))
                        st.rerun()
                    except auth.AuthError as exc:
                        st.error(str(exc))
            with tab_up:
                with st.form("register_form"):
                    name = st.text_input("Display name", key="reg_name", help="Shown in the sidebar and on reports.")
                    user = st.text_input("Username", key="reg_user", help="3–30 characters: letters, numbers, . _ -")
                    email = st.text_input("Email (optional)", key="reg_email")
                    pw1 = st.text_input("Password", type="password", key="reg_pw1", help="At least 8 characters, with a letter and a number.")
                    pw2 = st.text_input("Confirm password", type="password", key="reg_pw2")
                    go = st.form_submit_button("Create account", type="primary")
                if go:
                    if pw1 != pw2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            login(auth.create_user(user, pw1, email, name))
                            st.rerun()
                        except auth.AuthError as exc:
                            st.error(str(exc))
        st.markdown('<div class="auth-or">or</div>', unsafe_allow_html=True)
        if st.button("Continue as guest", key="auth_guest"):
            guest_login()
            st.rerun()
        st.caption("Guest sessions work fully but nothing is saved after you close the tab.")
