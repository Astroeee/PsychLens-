"""Per-user dataset storage on disk (CSV files + a small manifest). No Streamlit imports."""
from __future__ import annotations

import hashlib
import json
import shutil

import pandas as pd

import config

USERS_DIR = config.DATA_DIR / "users"


def user_dir(uid: int):
    d = USERS_DIR / str(uid)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest(uid: int) -> dict:
    p = user_dir(uid) / "manifest.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_manifest(uid: int, m: dict) -> None:
    (user_dir(uid) / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")


def _file(name: str, kind: str) -> str:
    return f"{hashlib.sha1(name.encode()).hexdigest()[:12]}_{kind}.csv"


def save_dataset(uid: int, name: str, df: pd.DataFrame) -> None:
    m = _manifest(uid)
    entry = m.setdefault(name, {"original": _file(name, "orig"), "cleaned": None})
    df.to_csv(user_dir(uid) / entry["original"], index=False)
    _write_manifest(uid, m)


def save_cleaned(uid: int, name: str, df: pd.DataFrame) -> None:
    m = _manifest(uid)
    if name not in m:
        save_dataset(uid, name, df)
        m = _manifest(uid)
    m[name]["cleaned"] = _file(name, "clean")
    df.to_csv(user_dir(uid) / m[name]["cleaned"], index=False)
    _write_manifest(uid, m)


def delete_cleaned(uid: int, name: str) -> None:
    m = _manifest(uid)
    if name in m and m[name].get("cleaned"):
        (user_dir(uid) / m[name]["cleaned"]).unlink(missing_ok=True)
        m[name]["cleaned"] = None
        _write_manifest(uid, m)


def delete_dataset(uid: int, name: str) -> None:
    m = _manifest(uid)
    entry = m.pop(name, None)
    if entry:
        for f in (entry.get("original"), entry.get("cleaned")):
            if f: (user_dir(uid) / f).unlink(missing_ok=True)
        _write_manifest(uid, m)


def load_all(uid: int) -> tuple[dict, dict]:
    datasets, cleaned = {}, {}
    for name, e in _manifest(uid).items():
        try:
            datasets[name] = pd.read_csv(user_dir(uid) / e["original"])
            if e.get("cleaned"):
                cleaned[name] = pd.read_csv(user_dir(uid) / e["cleaned"])
        except (OSError, ValueError):
            continue
    return datasets, cleaned


def usage(uid: int) -> tuple[int, int]:
    """(number of datasets, bytes on disk)."""
    d = user_dir(uid)
    return len(_manifest(uid)), sum(f.stat().st_size for f in d.glob("*.csv"))


def delete_user_data(uid: int) -> None:
    shutil.rmtree(USERS_DIR / str(uid), ignore_errors=True)
