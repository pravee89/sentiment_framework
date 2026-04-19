"""
pipeline/preprocess.py
──────────────────────
Text preprocessing: language detection, cleaning, and lemmatization.

Two text variants are produced per row:
  clean_text      — lightly cleaned; fed to transformer models
  lemmatized_text — aggressively filtered (nouns + adjectives only);
                    used for word clouds and topic modelling
"""

from __future__ import annotations

import re
import warnings

import pandas as pd
import spacy
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from config.settings import SUPPORTED_LANGUAGES, SPACY_MODEL, MIN_TEXT_LENGTH

# Make langdetect deterministic
DetectorFactory.seed = 42

warnings.filterwarnings("ignore")

# Load spaCy model once at module import (lazy load in case spaCy isn't installed)
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    return _nlp


# ─── Public API ────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline on a DataFrame.

    Steps
    -----
    1. Detect language
    2. Filter to supported languages
    3. Light-clean text  → clean_text
    4. Lemmatize + POS-filter → lemmatized_text
    5. Drop rows with too-short text

    Returns a copy of the DataFrame with new columns added.
    """
    df = df.copy()

    print(f"[preprocess] Detecting language for {len(df)} rows…")
    df["lang"] = df[text_col].apply(_detect_language)

    before = len(df)
    df = df[df["lang"].isin(SUPPORTED_LANGUAGES)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[preprocess] Dropped {dropped} non-English rows.")

    print("[preprocess] Cleaning text…")
    df["clean_text"] = df[text_col].apply(_light_clean)

    # Drop rows where clean_text is too short
    word_counts = df["clean_text"].str.split().str.len()
    df = df[word_counts >= MIN_TEXT_LENGTH].copy()

    print("[preprocess] Lemmatizing (this may take a moment)…")
    nlp = _get_nlp()
    df["lemmatized_text"] = _batch_lemmatize(df["clean_text"].tolist(), nlp)

    df = df.reset_index(drop=True)
    print(f"[preprocess] Done. {len(df)} rows ready for analysis.")
    return df


def light_clean_single(text: str) -> str:
    """Clean a single text string (used for on-the-fly processing)."""
    return _light_clean(text)


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _detect_language(text) -> str:
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"


def _light_clean(text) -> str:
    """
    Minimal cleaning that preserves semantic content for transformer models.
    - Remove URLs, @mentions, HTML tags
    - Normalise whitespace and special chars
    - Keep punctuation (transformers handle it well)
    """
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                        # mentions
    text = re.sub(r"<[^>]+>", " ", text)                     # HTML tags
    text = re.sub(r"[^\w\s\'\"\.\!\?\,\-]", " ", text)       # special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _batch_lemmatize(texts: list[str], nlp) -> list[str]:
    """
    Process texts in batches using spaCy's pipe for efficiency.
    Keeps only NOUN and ADJ tokens; removes stopwords, punctuation, numbers.
    """
    results = []
    batch_size = 64

    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ("NOUN", "ADJ", "PROPN")
            and not token.is_stop
            and not token.is_punct
            and not token.like_num
            and not token.is_space
            and len(token.lemma_) > 2
            and token.lemma_.isalpha()
        ]
        results.append(" ".join(tokens) if tokens else "")

    return results
