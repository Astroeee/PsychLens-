"""Account and storage tests (use a temporary database; no Streamlit needed)."""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import auth, storage  # noqa: E402

auth.ITERATIONS = 1_000  # fast hashing for tests only


def _fresh():
    tmp = Path(tempfile.mkdtemp())
    auth.DB_PATH = tmp / "users.db"
    storage.USERS_DIR = tmp / "users"
    return tmp


def _raises(fn, *a):
    try:
        fn(*a)
    except auth.AuthError as exc:
        return str(exc)
    raise AssertionError("expected AuthError")


def test_register_and_login():
    _fresh()
    u = auth.create_user("Alice_1", "secret123", "a@x.com", "Alice")
    assert u["username"] == "Alice_1" and u["display_name"] == "Alice"
    assert auth.authenticate("alice_1", "secret123")["id"] == u["id"]      # case-insensitive username
    assert auth.authenticate("A@X.com", "secret123")["id"] == u["id"]      # email works too
    assert auth.get_settings(u["id"])["researcher_name"] == "Alice"


def test_password_is_hashed_not_stored():
    _fresh()
    auth.create_user("bob", "secret123")
    with auth._db() as c:
        row = c.execute("SELECT * FROM users").fetchone()
    assert "secret123" not in str(dict(row)) and len(row["password_hash"]) == 64 and len(row["salt"]) == 32


def test_validation_and_duplicates():
    _fresh()
    for args in [("ab", "secret123"), ("bob", "short1"), ("bob", "allletters"), ("bob", "12345678"),
                 ("bob", "secret123", "not-an-email")]:
        _raises(auth.create_user, *args)
    auth.create_user("carol", "secret123", "c@x.com")
    assert "taken" in _raises(auth.create_user, "CAROL", "secret123")
    assert "email" in _raises(auth.create_user, "carol2", "secret123", "c@x.com")


def test_wrong_password_and_lockout():
    _fresh()
    auth.create_user("dave", "secret123")
    assert _raises(auth.authenticate, "nobody", "x") == auth.GENERIC_FAIL          # no user enumeration
    for _ in range(auth.MAX_FAILED):
        assert _raises(auth.authenticate, "dave", "wrongpass1") == auth.GENERIC_FAIL
    assert "Too many failed attempts" in _raises(auth.authenticate, "dave", "secret123")


def test_change_password_and_delete():
    _fresh()
    u = auth.create_user("erin", "secret123")
    _raises(auth.change_password, u["id"], "wrong", "newpass123")
    _raises(auth.change_password, u["id"], "secret123", "short")
    auth.change_password(u["id"], "secret123", "newpass123")
    _raises(auth.authenticate, "erin", "secret123")
    auth.authenticate("erin", "newpass123")
    _raises(auth.delete_user, u["id"], "wrong")
    auth.delete_user(u["id"], "newpass123")
    _raises(auth.authenticate, "erin", "newpass123")


def test_settings_roundtrip():
    _fresh()
    u = auth.create_user("fay", "secret123")
    auth.save_settings(u["id"], {"alpha": 0.01, "decimals": 3})
    assert auth.get_settings(u["id"]) == {"alpha": 0.01, "decimals": 3}


def test_storage_roundtrip_and_isolation():
    _fresh()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    storage.save_dataset(1, "My data", df)
    storage.save_cleaned(1, "My data", df.iloc[:2])
    storage.save_dataset(2, "Other", df)
    ds, cl = storage.load_all(1)
    assert list(ds) == ["My data"] and ds["My data"].shape == (3, 2) and cl["My data"].shape == (2, 2)
    assert list(storage.load_all(2)[0]) == ["Other"]                      # users never see each other's data
    storage.delete_cleaned(1, "My data")
    assert storage.load_all(1)[1] == {}
    storage.delete_dataset(1, "My data")
    assert storage.load_all(1)[0] == {}
    assert storage.usage(2)[0] == 1
    storage.delete_user_data(2)
    assert storage.load_all(2)[0] == {}
