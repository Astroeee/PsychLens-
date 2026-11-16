"""Account: profile, password, stored data, sign out and account deletion."""
import streamlit as st

from components.cards import callout, card_title, kpi_card
from components.layout import page_header
from modules import auth, storage
from modules.state import is_guest, logout, save_settings, setting

page_header("Account", "Manage your profile and saved data")
ss = st.session_state
user = ss.user

if is_guest():
    callout("<b>You are using guest mode.</b> Everything works, but datasets and settings are not saved. "
            "Create a free account to keep your work between sessions.", "warn")
    if st.button("Create an account / sign in", type="primary", key="acct_guest_exit"):
        logout()
        st.rerun()
    st.stop()

n_ds, nbytes = storage.usage(user["id"])
c1, c2, c3 = st.columns(3)
with c1: kpi_card("Username", user["username"], "sign-in name", "users")
with c2: kpi_card("Saved datasets", n_ds, f"{nbytes / 1024:.0f} KB stored", "database", "green")
with c3: kpi_card("Member since", str(user["created_at"])[:10], "account created", "check", "amber")
st.write("")

with st.container(border=True):
    card_title("Profile", "The name and institution appear on your reports")
    with st.form("profile_form"):
        a, b = st.columns(2)
        name = a.text_input("Display name", value=setting("researcher_name"))
        inst = b.text_input("Institution", value=setting("institution"))
        email = st.text_input("Email (optional)", value=user.get("email", ""))
        if st.form_submit_button("Save profile", type="primary"):
            try:
                auth.update_email(user["id"], email)
                auth.update_display_name(user["id"], name)
                ss.settings.update(researcher_name=name.strip() or user["username"], institution=inst.strip())
                ss.user = {**user, "email": email.strip(), "display_name": name.strip() or user["username"]}
                save_settings()
                st.success("Profile saved.")
                st.rerun()
            except auth.AuthError as exc:
                st.error(str(exc))

with st.container(border=True):
    card_title("Change password")
    with st.form("password_form"):
        old = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password", help="At least 8 characters, with a letter and a number.")
        new2 = st.text_input("Confirm new password", type="password")
        if st.form_submit_button("Update password", type="primary"):
            if new != new2:
                st.error("New passwords do not match.")
            else:
                try:
                    auth.change_password(user["id"], old, new)
                    st.success("Password updated.")
                except auth.AuthError as exc:
                    st.error(str(exc))

with st.container(border=True):
    card_title("Session")
    if st.button("Sign out", key="acct_signout"):
        logout()
        st.rerun()

with st.container(border=True):
    card_title("Delete account", "Permanently removes your account, saved datasets and settings. This cannot be undone.")
    with st.form("delete_form"):
        pw = st.text_input("Confirm with your password", type="password")
        sure = st.checkbox("I understand this is permanent")
        if st.form_submit_button("Delete my account"):
            if not sure:
                st.error("Tick the confirmation box first.")
            else:
                try:
                    auth.delete_user(user["id"], pw)
                    storage.delete_user_data(user["id"])
                    logout()
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))
