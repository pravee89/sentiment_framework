"""
pipeline/ingest.py
──────────────────
Handles file loading and schema inference.

Public API
----------
load_file(path_or_buffer, file_type=None) -> pd.DataFrame
infer_schema(df)                          -> dict
validate_schema(df, schema)               -> (bool, list[str])
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union

import pandas as pd

from config.settings import (
    TEXT_COLUMN_HINTS,
    DATE_COLUMN_HINTS,
    SCORE_COLUMN_HINTS,
    CATEGORY_COLUMN_HINTS,
    MIN_TEXT_LENGTH,
)

# ─── Supported file extensions ─────────────────────────────────────────────────
LOADERS = {
    ".csv":     lambda f: pd.read_csv(f),
    ".tsv":     lambda f: pd.read_csv(f, sep="\t"),
    ".xlsx":    lambda f: pd.read_excel(f),
    ".xls":     lambda f: pd.read_excel(f),
    ".json":    lambda f: _load_json(f),
    ".parquet": lambda f: pd.read_parquet(f),
}


# ─── Public functions ──────────────────────────────────────────────────────────

def load_file(
    path_or_buffer: Union[str, Path, io.BytesIO],
    file_type: str | None = None,
) -> pd.DataFrame:
    """
    Load a file into a DataFrame.

    Parameters
    ----------
    path_or_buffer : file path, Path object, or BytesIO (Streamlit upload)
    file_type      : file extension hint (e.g. ".csv") — required for BytesIO

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(path_or_buffer, (str, Path)):
        ext = Path(path_or_buffer).suffix.lower()
    elif file_type:
        ext = file_type.lower() if file_type.startswith(".") else f".{file_type.lower()}"
    else:
        raise ValueError("Provide file_type when passing a buffer (e.g. '.csv')")

    if ext not in LOADERS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(LOADERS.keys())}"
        )

    df = LOADERS[ext](path_or_buffer)

    # Normalize column names: strip whitespace, lowercase
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_schema(df: pd.DataFrame) -> dict:
    """
    Auto-detect which columns map to text, date, score, and category.

    Returns
    -------
    dict with keys: text_col, date_col, score_col, category_col
    Each value is a column name string or None if not found.
    """
    text_col     = _find_text_column(df)
    date_col     = _find_date_column(df)
    score_col    = _find_score_column(df)
    category_col = _find_category_column(df, exclude=[text_col, date_col, score_col])

    return {
        "text_col":     text_col,
        "date_col":     date_col,
        "score_col":    score_col,
        "category_col": category_col,
    }


def validate_schema(df: pd.DataFrame, schema: dict) -> tuple[bool, list[str]]:
    """
    Validate the (possibly user-edited) schema against the DataFrame.

    Returns
    -------
    (is_valid: bool, errors: list[str])
    """
    errors = []
    text_col = schema.get("text_col")

    if not text_col:
        errors.append("A text column must be selected.")
    elif text_col not in df.columns:
        errors.append(f"Column '{text_col}' not found in the dataset.")
    else:
        # Check it actually has text content
        avg_len = df[text_col].dropna().astype(str).str.split().str.len().mean()
        if avg_len < MIN_TEXT_LENGTH:
            errors.append(
                f"Column '{text_col}' has very short average text "
                f"({avg_len:.1f} words). Is this the right column?"
            )

    # Validate optional columns if specified
    for key in ("date_col", "score_col", "category_col"):
        col = schema.get(key)
        if col and col not in df.columns:
            errors.append(f"Column '{col}' (selected as {key}) not found in the dataset.")

    return len(errors) == 0, errors


def get_column_previews(df: pd.DataFrame, schema: dict) -> dict:
    """
    Return a small sample of each schema column so the UI can display previews.
    """
    previews = {}
    for key, col in schema.items():
        if col and col in df.columns:
            previews[key] = df[col].dropna().head(3).tolist()
    return previews


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _load_json(f) -> pd.DataFrame:
    """Load JSON — handles both records and nested structures."""
    try:
        return pd.read_json(f, orient="records")
    except Exception:
        return pd.json_normalize(pd.read_json(f))


def _score_column_name(col: str, hints: list[str]) -> float:
    """Return a 0–1 confidence score for how well a column name matches hints."""
    col_lower = col.lower()
    for hint in hints:
        if hint == col_lower:
            return 1.0          # exact match
        if hint in col_lower:
            return 0.8          # partial match
        # fuzzy: check if hint words appear in col name
        if any(word in col_lower for word in hint.split("_")):
            return 0.5
    return 0.0


def _find_text_column(df: pd.DataFrame) -> str | None:
    """Find the column most likely to contain free-text feedback."""
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if not str_cols:
        return None

    candidates = []
    for col in str_cols:
        name_score = _score_column_name(col, TEXT_COLUMN_HINTS)
        # Average word count — higher = more likely to be free text
        avg_words = df[col].dropna().astype(str).str.split().str.len().mean()
        word_score = min(avg_words / 20.0, 1.0)   # normalise, cap at 1
        # Uniqueness ratio — free text should be mostly unique
        unique_ratio = df[col].nunique() / max(len(df), 1)
        candidates.append((col, name_score * 0.5 + word_score * 0.35 + unique_ratio * 0.15))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0] if candidates else None


def _find_date_column(df: pd.DataFrame) -> str | None:
    """Find a date/timestamp column."""
    # 1. Already datetime dtype
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # 2. Name-hint match on object columns
    str_cols = df.select_dtypes(include="object").columns
    best = (None, 0.0)
    for col in str_cols:
        score = _score_column_name(col, DATE_COLUMN_HINTS)
        if score > best[1]:
            best = (col, score)

    if best[0]:
        # Verify it actually parses as dates
        try:
            pd.to_datetime(df[best[0]].dropna().head(20), format="mixed", dayfirst=False)
            return best[0]
        except Exception:
            pass
    return None


def _find_score_column(df: pd.DataFrame) -> str | None:
    """Find a numeric rating/score column (typically 1–5 or 1–10)."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    candidates = []
    for col in numeric_cols:
        series = df[col].dropna()
        name_score = _score_column_name(col, SCORE_COLUMN_HINTS)
        # Check if values look like ratings
        in_rating_range = series.between(1, 10).mean()
        low_cardinality = 1.0 if series.nunique() <= 10 else 0.0
        score = name_score * 0.5 + in_rating_range * 0.3 + low_cardinality * 0.2
        candidates.append((col, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    if candidates and candidates[0][1] > 0.3:
        return candidates[0][0]
    return None


def _find_category_column(df: pd.DataFrame, exclude: list) -> str | None:
    """Find a categorical column (product type, department, etc.)."""
    exclude_set = {c for c in exclude if c}
    str_cols = [c for c in df.select_dtypes(include="object").columns if c not in exclude_set]

    candidates = []
    for col in str_cols:
        name_score = _score_column_name(col, CATEGORY_COLUMN_HINTS)
        n_unique = df[col].nunique()
        # Good category: few unique values, name hint match
        if 2 <= n_unique <= 100:
            cardinality_score = 1.0 - (n_unique / 100.0)
            candidates.append((col, name_score * 0.6 + cardinality_score * 0.4))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0] if candidates else None
