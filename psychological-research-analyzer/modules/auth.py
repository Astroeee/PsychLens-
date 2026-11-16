"""Accounts: registration, sign-in, profile and password management.

Uses only the standard library (sqlite3 + hashlib). Passwords are never stored:
we keep a random per-user salt and a PBKDF2-HMAC-SHA256 hash. Repeated failed
sign-ins temporarily lock the account. No Streamlit imports.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

import config

DB_PATH = config.DATA_DIR / "users.db"
ITERATIONS = 310_000          # PBKDF2 work factor (OWASP-recommended range)
MAX_FAILED = 5                # failed sign-ins before a temporary lock
LOCK_MINUTES = 5
GENERIC_FAIL = "Invalid username/email or password."

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    email TEXT UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    settings TEXT NOT NULL DEFAULT '{}',
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    created_at TEXT NOT NULL,
    last_login TEXT
)"""


class AuthError(Exception):
    """A problem the user can fix (bad password, name taken, ...)."""


@contextmanager
def _db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ------------------------------------------------------------- passwords ----
def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), ITERATIONS)
    return digest.hex(), salt


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    return hmac.compare_digest(hash_password(password, salt)[0], expected_hash)


def validate_username(u: str) -> str | None:
    return None if USERNAME_RE.match(u or "") else "Username must be 3–30 characters: letters, numbers, . _ -"


def validate_email(e: str) -> str | None:
    return None if not e or EMAIL_RE.match(e) else "Enter a valid email address (or leave it blank)."


def validate_password(p: str) -> str | None:
    if len(p or "") < 8: return "Password must be at least 8 characters."
    if len(p) > 128: return "Password is too long (max 128 characters)."
    if not (re.search(r"[A-Za-z]", p) and re.search(r"\d", p)): return "Password must contain at least one letter and one number."
    return None


def _user(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "username": row["username"], "email": row["email"] or "",
            "display_name": row["display_name"] or row["username"], "created_at": row["created_at"]}


# -------------------------------------------------------------- accounts ----
def create_user(username: str, password: str, email: str = "", display_name: str = "") -> dict:
    username, email, display_name = (username or "").strip(), (email or "").strip(), (display_name or "").strip()
    for err in (validate_username(username), validate_email(email), validate_password(password)):
        if err: raise AuthError(err)
    h, salt = hash_password(password)
    settings = {"researcher_name": display_name or username}
    try:
        with _db() as c:
            cur = c.execute("INSERT INTO users (username,email,password_hash,salt,display_name,settings,created_at) "
                            "VALUES (?,?,?,?,?,?,?)",
                            (username, email or None, h, salt, display_name or username, json.dumps(settings),
                             datetime.now().isoformat(timespec="seconds")))
            row = c.execute("SELECT * FROM users WHERE id=?", (cur.lastrowid,)).fetchone()
    except sqlite3.IntegrityError as exc:
        raise AuthError("That email is already registered." if "email" in str(exc).lower() else "That username is taken.") from exc
    return _user(row)


def authenticate(identifier: str, password: str) -> dict:
    ident = (identifier or "").strip()
    with _db() as c:
        row = c.execute("SELECT * FROM users WHERE username=? OR email=?", (ident, ident)).fetchone()
        if row is None:
            hash_password(password or "", "00" * 16)       # keep timing similar for unknown users
            raise AuthError(GENERIC_FAIL)
        if row["locked_until"] and datetime.fromisoformat(row["locked_until"]) > datetime.now():
            mins = int((datetime.fromisoformat(row["locked_until"]) - datetime.now()).total_seconds() // 60) + 1
            raise AuthError(f"Too many failed attempts. Try again in about {mins} minute(s).")
        if not verify_password(password or "", row["salt"], row["password_hash"]):
            n = row["failed_attempts"] + 1
            if n >= MAX_FAILED:
                c.execute("UPDATE users SET failed_attempts=0, locked_until=? WHERE id=?",
                          ((datetime.now() + timedelta(minutes=LOCK_MINUTES)).isoformat(), row["id"]))
            else:
                c.execute("UPDATE users SET failed_attempts=? WHERE id=?", (n, row["id"]))
            c.commit()
            raise AuthError(GENERIC_FAIL)
        c.execute("UPDATE users SET failed_attempts=0, locked_until=NULL, last_login=? WHERE id=?",
                  (datetime.now().isoformat(timespec="seconds"), row["id"]))
        return _user(row)


def get_user(user_id: int) -> dict | None:
    with _db() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return _user(row) if row else None


def update_email(user_id: int, email: str) -> None:
    email = (email or "").strip()
    if (err := validate_email(email)): raise AuthError(err)
    try:
        with _db() as c:
            c.execute("UPDATE users SET email=? WHERE id=?", (email or None, user_id))
    except sqlite3.IntegrityError as exc:
        raise AuthError("That email is already registered.") from exc


def update_display_name(user_id: int, name: str) -> None:
    with _db() as c:
        c.execute("UPDATE users SET display_name=? WHERE id=?", ((name or "").strip(), user_id))


def change_password(user_id: int, old: str, new: str) -> None:
    with _db() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if row is None or not verify_password(old or "", row["salt"], row["password_hash"]):
            raise AuthError("Current password is incorrect.")
        if (err := validate_password(new)): raise AuthError(err)
        h, salt = hash_password(new)
        c.execute("UPDATE users SET password_hash=?, salt=? WHERE id=?", (h, salt, user_id))


def delete_user(user_id: int, password: str) -> None:
    with _db() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if row is None or not verify_password(password or "", row["salt"], row["password_hash"]):
            raise AuthError("Password is incorrect.")
        c.execute("DELETE FROM users WHERE id=?", (user_id,))


# -------------------------------------------------------------- settings ----
def get_settings(user_id: int) -> dict:
    with _db() as c:
        row = c.execute("SELECT settings FROM users WHERE id=?", (user_id,)).fetchone()
    try:
        return json.loads(row["settings"]) if row else {}
    except (TypeError, ValueError):
        return {}


def save_settings(user_id: int, settings: dict) -> None:
    with _db() as c:
        c.execute("UPDATE users SET settings=? WHERE id=?", (json.dumps(settings), user_id))
