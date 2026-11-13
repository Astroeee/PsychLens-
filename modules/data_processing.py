"""Loading, validating, profiling and cleaning datasets.

Design rules
------------
* Functions never modify the DataFrame you pass in; they return new objects.
* Nothing is removed or changed silently: cleaning returns a change log.
* This module has no Streamlit imports, so it can be unit-tested on its own.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx")


class DataLoadError(Exception):
    """Raised with a user-friendly message when a file cannot be loaded."""


# ----------------------------------------------------------------- loading --
def excel_sheet_names(file_bytes: bytes) -> list[str]:
    try:
        return pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl").sheet_names
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Could not read the Excel workbook: {exc}") from exc


def load_dataset(file_bytes: bytes, filename: str, sheet: str | int | None = None) -> pd.DataFrame:
    """Read a CSV or XLSX file from raw bytes into a DataFrame."""
    name = filename.lower()
    if not name.endswith(SUPPORTED_EXTENSIONS):
        raise DataLoadError("Unsupported file type. Upload a .csv or .xlsx file.")
    if not file_bytes:
        raise DataLoadError("The file is empty.")
    try:
        if name.endswith(".csv"):
            df = None
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python",
                                     encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise DataLoadError("Could not decode the CSV file (try saving it as UTF-8).")
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet or 0, engine="openpyxl")
    except DataLoadError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The file contains no data.") from exc
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Could not read the file: {exc}") from exc
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise DataLoadError("The file has no rows or no columns.")
    return df


def validate_dataframe(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return a list of (level, message). level is 'error', 'warning' or 'info'."""
    issues: list[tuple[str, str]] = []
    if df.shape[0] < 5:
        issues.append(("warning", f"Only {df.shape[0]} rows: most statistics need more data."))
    if df.shape[1] < 2:
        issues.append(("warning", "Only one column detected. Check the file delimiter."))
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        issues.append(("warning", f"{len(unnamed)} unnamed column(s): {', '.join(map(str, unnamed))}."))
    spaced = [c for c in df.columns if str(c) != str(c).strip()]
    if spaced:
        issues.append(("info", f"Column names with leading/trailing spaces: {', '.join(map(repr, spaced))}."))
    all_missing = [c for c in df.columns if df[c].isna().all()]
    if all_missing:
        issues.append(("warning", f"Columns with no data: {', '.join(map(str, all_missing))}."))
    constant = [c for c in df.columns if df[c].nunique(dropna=True) == 1 and not df[c].isna().all()]
    if constant:
        issues.append(("info", f"Constant columns (one unique value): {', '.join(map(str, constant))}."))
    return issues


# --------------------------------------------------------------- profiling --
def _is_text(s: pd.Series) -> bool:
    """True for object and string dtypes (pandas 3 uses a dedicated string dtype)."""
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _is_identifier(name: str, s: pd.Series) -> bool:
    n = s.notna().sum()
    looks_like_id = bool(re.search(r"(^id$|_id$|^id_|\bid\b|respondent|participant)", str(name).lower()))
    return looks_like_id and n > 0 and s.nunique(dropna=True) / n >= 0.95


def column_kind(name: str, s: pd.Series) -> str:
    """Classify a column: numeric, categorical, identifier, datetime or text."""
    if _is_identifier(name, s):
        return "identifier"
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    nn = s.dropna()
    if len(nn) and nn.nunique() / len(nn) > 0.9 and len(nn) > 20:
        return "text"
    return "categorical"


def split_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    out = {"numeric": [], "categorical": [], "identifier": [], "datetime": [], "text": []}
    for c in df.columns:
        out[column_kind(c, df[c])].append(c)
    return out


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return split_columns(df)["numeric"]


def grouping_candidates(df: pd.DataFrame, max_levels: int = 12) -> list[str]:
    """Columns usable as a grouping factor: categorical, or coded numerics."""
    parts = split_columns(df)
    cols = list(parts["categorical"])
    for c in parts["numeric"]:
        if 2 <= df[c].nunique(dropna=True) <= max_levels:
            cols.append(c)
    return cols


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        rows.append({
            "Column": c,
            "Data type": str(s.dtype),
            "Variable type": column_kind(c, s),
            "Non-missing": int(s.notna().sum()),
            "Missing": int(s.isna().sum()),
            "Missing %": round(s.isna().mean() * 100, 1) if n else 0.0,
            "Unique": int(s.nunique(dropna=True)),
            "Example": "" if nn.empty else str(nn.iloc[0]),
        })
    return pd.DataFrame(rows)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna().sum()
    out = pd.DataFrame({"Column": m.index, "Missing": m.values})
    out["Missing %"] = (out["Missing"] / max(len(df), 1) * 100).round(1)
    return out.sort_values("Missing", ascending=False, ignore_index=True)


def missing_percentage(df: pd.DataFrame) -> float:
    return float(df.isna().sum().sum() / df.size * 100) if df.size else 0.0


def duplicate_rows(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Every row involved in a duplicate group (including the first occurrence)."""
    mask = df.duplicated(subset=subset, keep=False)
    return df[mask].sort_values(by=list(subset or df.columns[:1]))


def duplicate_count(df: pd.DataFrame, subset: list[str] | None = None) -> int:
    """Number of surplus duplicate rows (first occurrences are not counted)."""
    return int(df.duplicated(subset=subset, keep="first").sum())


def detect_likert_columns(df: pd.DataFrame) -> dict[str, int]:
    """Guess which numeric columns are Likert items; returns {column: scale_max}.

    A column qualifies when all values are whole numbers and at least 90% of
    them lie inside 1..5 (or 1..7), with at least 3 distinct values inside.
    """
    found: dict[str, int] = {}
    for c in numeric_columns(df):
        s = df[c].dropna()
        s = s[np.isfinite(s)]
        if len(s) < 10 or not np.all(np.mod(s, 1) == 0):
            continue
        for top in (5, 7):
            inside = s[(s >= 1) & (s <= top)]
            if len(inside) / len(s) >= 0.9 and inside.nunique() >= 3:
                found[c] = top
                break
    return found


def find_invalid_values(df: pd.DataFrame,
                        ranges: dict[str, tuple[float | None, float | None]] | None = None,
                        auto_likert: bool = True) -> pd.DataFrame:
    """List cells that look invalid. These are flags, not automatic deletions."""
    ranges = ranges or {}
    rows = []

    def add(idx, col, val, why):
        rows.append({"Row": int(idx) + 1, "Column": col, "Value": val, "Reason": why})

    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            for idx in s[np.isinf(s.astype(float))].index:
                add(df.index.get_loc(idx), c, s[idx], "Infinite value")
        elif _is_text(s):
            nn = s.dropna()
            conv = pd.to_numeric(nn, errors="coerce")
            if len(nn) and conv.notna().mean() >= 0.8 and conv.isna().any():
                for idx in nn[conv.isna()].index:
                    add(df.index.get_loc(idx), c, s[idx], "Text in a mostly numeric column")

    for c, (lo, hi) in ranges.items():
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = df[c]
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= s < lo
        if hi is not None:
            bad |= s > hi
        for idx in s[bad.fillna(False)].index:
            add(df.index.get_loc(idx), c, s[idx], f"Outside allowed range [{lo}, {hi}]")

    if auto_likert:
        for c, top in detect_likert_columns(df).items():
            if c in ranges:
                continue
            s = df[c]
            bad = (s < 1) | (s > top)
            for idx in s[bad.fillna(False)].index:
                add(df.index.get_loc(idx), c, s[idx], f"Outside detected 1-{top} response scale")
    cols = ["Row", "Column", "Value", "Reason"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- cleaning --
@dataclass
class CleaningOptions:
    drop_duplicates: bool = False
    invalid_to_nan: bool = False
    ranges: dict = field(default_factory=dict)
    numeric_missing: str = "keep"        # keep | drop_rows | mean | median | mode
    categorical_missing: str = "keep"    # keep | drop_rows | mode | label
    missing_label: str = "Missing"
    missing_columns: list | None = None  # None = every non-identifier column
    strip_text: bool = False
    text_case: str = "keep"              # keep | lower | upper | title
    standardize_names: bool = False
    to_numeric: list = field(default_factory=list)


def _snake(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_") or "column"


def apply_cleaning(df: pd.DataFrame, opts: CleaningOptions) -> tuple[pd.DataFrame, list[str]]:
    """Apply the chosen cleaning steps to a COPY and return (clean_df, log)."""
    out = df.copy()
    log: list[str] = []
    obj_cols = [c for c in out.columns if _is_text(out[c])]

    if opts.strip_text:
        changed = 0
        for c in obj_cols:
            new = out[c].map(lambda v: v.strip() if isinstance(v, str) else v)
            changed += int((new.astype(str) != out[c].astype(str)).sum())
            out[c] = new
        log.append(f"Trimmed whitespace in text columns ({changed} cells changed).")

    for c in opts.to_numeric:
        if c in out.columns:
            before = int(out[c].isna().sum())
            out[c] = pd.to_numeric(out[c], errors="coerce")
            log.append(f"Converted '{c}' to numeric ({int(out[c].isna().sum()) - before} "
                       f"non-numeric value(s) became missing).")

    if opts.invalid_to_nan:
        flagged = find_invalid_values(out, opts.ranges)
        for _, r in flagged.iterrows():
            out.iloc[int(r["Row"]) - 1, out.columns.get_loc(r["Column"])] = np.nan
        log.append(f"Set {len(flagged)} invalid value(s) to missing.")

    if opts.drop_duplicates:
        n = int(out.duplicated(keep="first").sum())
        out = out.drop_duplicates(keep="first")
        log.append(f"Removed {n} duplicate row(s) (first occurrence kept).")

    if opts.text_case != "keep":
        fn = {"lower": str.lower, "upper": str.upper, "title": str.title}[opts.text_case]
        for c in [c for c in out.columns if _is_text(out[c])]:
            out[c] = out[c].map(lambda v: fn(v) if isinstance(v, str) else v)
        log.append(f"Converted text values to {opts.text_case} case.")

    parts = split_columns(out)
    targets = opts.missing_columns if opts.missing_columns is not None else \
        [c for c in out.columns if c not in parts["identifier"]]
    num_t = [c for c in targets if c in parts["numeric"] and out[c].isna().any()]
    cat_t = [c for c in targets if c in parts["categorical"] + parts["text"] and out[c].isna().any()]

    drop_cols: list[str] = []
    for c in num_t:
        k = int(out[c].isna().sum())
        if opts.numeric_missing == "drop_rows":
            drop_cols.append(c)
        elif opts.numeric_missing in ("mean", "median", "mode"):
            if opts.numeric_missing == "mode":
                m = out[c].mode()
                fill = m.iloc[0] if len(m) else np.nan
            else:
                fill = getattr(out[c], opts.numeric_missing)()
            out[c] = out[c].fillna(fill)
            log.append(f"'{c}': filled {k} missing value(s) with the {opts.numeric_missing} ({fill:.3g}).")
    for c in cat_t:
        k = int(out[c].isna().sum())
        if opts.categorical_missing == "drop_rows":
            drop_cols.append(c)
        elif opts.categorical_missing == "mode":
            m = out[c].mode()
            if len(m):
                out[c] = out[c].fillna(m.iloc[0])
                log.append(f"'{c}': filled {k} missing value(s) with the most frequent value ('{m.iloc[0]}').")
        elif opts.categorical_missing == "label":
            out[c] = out[c].fillna(opts.missing_label)
            log.append(f"'{c}': labelled {k} missing value(s) as '{opts.missing_label}'.")
    if drop_cols:
        n0 = len(out)
        out = out.dropna(subset=drop_cols)
        log.append(f"Removed {n0 - len(out)} row(s) with missing values in: {', '.join(drop_cols)}.")

    if opts.standardize_names:
        new_names, seen = [], {}
        for c in out.columns:
            base = _snake(c)
            seen[base] = seen.get(base, 0) + 1
            new_names.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
        out.columns = new_names
        log.append("Standardized column names to snake_case.")

    if not log:
        log.append("No cleaning steps were selected; the copy is identical to the original.")
    log.append(f"Result: {len(out)} rows x {out.shape[1]} columns (original: {len(df)} x {df.shape[1]}).")
    return out.reset_index(drop=True), log


# ------------------------------------------------------------------ export --
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return buf.getvalue()
