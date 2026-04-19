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

from config.settings import SUPPORTED_LANGUAGES, SPACY_MODEL, MIN_TEXT_LENGTH, LEMMATIZE_ON_DEMAND

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

def preprocess(df: pd.DataFrame, text_col: str, lemmatize: bool = False) -> pd.DataFrame:
    """
    Run the preprocessing pipeline on a DataFrame.

    Steps
    -----
    1. Detect language
    2. Filter to supported languages
    3. Light-clean text  → clean_text
    4. (Optional) Lemmatize + POS-filter → lemmatized_text
    5. Drop rows with too-short text

    Parameters
    ----------
    lemmatize : if True, run spaCy lemmatization (slow — only needed for word clouds).
                Defaults to False so the main pipeline runs fast.
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

    if lemmatize or not LEMMATIZE_ON_DEMAND:
        print("[preprocess] Lemmatizing…")
        nlp = _get_nlp()
        df["lemmatized_text"] = _batch_lemmatize(df["clean_text"].tolist(), nlp)
    else:
        # Placeholder — will be filled on demand when word cloud tab is opened
        df["lemmatized_text"] = None

    df = df.reset_index(drop=True)
    print(f"[preprocess] Done. {len(df)} rows ready for analysis.")
    return df


def run_lemmatization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run lemmatization on demand (called lazily when word cloud tab is opened).
    Only processes rows that haven't been lemmatized yet.
    """
    if "lemmatized_text" in df.columns and df["lemmatized_text"].notna().all():
        return df   # already done

    print("[preprocess] Running on-demand lemmatization for word clouds…")
    nlp = _get_nlp()
    df = df.copy()
    df["lemmatized_text"] = _batch_lemmatize(df["clean_text"].tolist(), nlp)
    print("[preprocess] Lemmatization complete.")
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
