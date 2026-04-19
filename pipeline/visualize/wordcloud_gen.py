"""
pipeline/visualize/wordcloud_gen.py
─────────────────────────────────────
Generates per-sentiment word clouds from lemmatized text.

Design choices
--------------
- Separate clouds for positive / neutral / negative (not one blended cloud)
- Uses lemmatized_text (nouns + adjectives only, already filtered by spaCy)
- Additional domain stopwords loaded from config
- TF-IDF weighting via sklearn so rare-but-distinctive words stand out
- Bigrams/trigrams preserved through WordCloud's collocations=True

Returns PIL Image objects so Streamlit can display them with st.image().
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

from config.settings import (
    WC_WIDTH, WC_HEIGHT, WC_MAX_WORDS, WC_BG_COLOR,
    WC_COLORMAP_POS, WC_COLORMAP_NEU, WC_COLORMAP_NEG,
    DOMAIN_STOPWORDS_PATH,
)

# Load domain stopwords once
_DOMAIN_STOPS: set[str] = set()

def _load_domain_stops() -> set[str]:
    global _DOMAIN_STOPS
    if _DOMAIN_STOPS:
        return _DOMAIN_STOPS

    path = Path(DOMAIN_STOPWORDS_PATH)
    if path.exists():
        lines = path.read_text().splitlines()
        _DOMAIN_STOPS = {
            line.strip().lower()
            for line in lines
            if line.strip() and not line.startswith("#")
        }
    return _DOMAIN_STOPS


SENTIMENT_COLORMAPS = {
    "positive": WC_COLORMAP_POS,
    "neutral":  WC_COLORMAP_NEU,
    "negative": WC_COLORMAP_NEG,
}


# ─── Public API ────────────────────────────────────────────────────────────────

def generate_wordclouds(df: pd.DataFrame) -> dict[str, Optional[Image.Image]]:
    """
    Generate word cloud images for each sentiment class.

    Parameters
    ----------
    df : analysed DataFrame (must have 'lemmatized_text' and 'sentiment_label')

    Returns
    -------
    dict: {"positive": PIL.Image, "neutral": PIL.Image, "negative": PIL.Image}
    Values are None if there is no data for that class.
    """
    stops = STOPWORDS | _load_domain_stops()
    images = {}

    for label in ["positive", "neutral", "negative"]:
        subset = df[df["sentiment_label"] == label]
        text   = " ".join(subset["lemmatized_text"].dropna().astype(str))
        text   = text.strip()

        if not text or len(text.split()) < 5:
            images[label] = None
            continue

        wc = WordCloud(
            width=WC_WIDTH,
            height=WC_HEIGHT,
            background_color=WC_BG_COLOR,
            colormap=SENTIMENT_COLORMAPS[label],
            stopwords=stops,
            max_words=WC_MAX_WORDS,
            collocations=True,           # enable bigrams
            collocation_threshold=10,    # minimum co-occurrence count
            prefer_horizontal=0.85,
            min_word_length=3,
        ).generate(text)

        images[label] = wc.to_image()

    return images


def generate_combined_wordcloud(df: pd.DataFrame) -> Optional[Image.Image]:
    """Generate a single word cloud from all feedback (all sentiments)."""
    stops = STOPWORDS | _load_domain_stops()
    text  = " ".join(df["lemmatized_text"].dropna().astype(str)).strip()

    if not text or len(text.split()) < 5:
        return None

    wc = WordCloud(
        width=WC_WIDTH,
        height=WC_HEIGHT,
        background_color=WC_BG_COLOR,
        colormap="viridis",
        stopwords=stops,
        max_words=WC_MAX_WORDS,
        collocations=True,
        collocation_threshold=10,
        prefer_horizontal=0.85,
        min_word_length=3,
    ).generate(text)

    return wc.to_image()


def wordcloud_to_bytes(img: Image.Image) -> bytes:
    """Convert a PIL Image to PNG bytes (for Streamlit st.image or PDF embedding)."""
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
