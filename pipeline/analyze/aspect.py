"""
pipeline/analyze/aspect.py
───────────────────────────
Aspect / topic extraction using KeyBERT.

KeyBERT embeds the document and keyword candidates with the same sentence
transformer, then selects the keywords whose embeddings are most similar
to the document embedding. MMR (Maximal Marginal Relevance) is used to
ensure diversity — we don't want five variations of the same phrase.

This module also provides aspect aggregation: given a DataFrame with
per-row aspects and sentiment labels, it returns aspect-level sentiment
counts (the "heatmap" data).
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
from keybert import KeyBERT

from config.settings import (
    KEYBERT_MODEL,
    ASPECT_TOP_N,
    ASPECT_NGRAM_RANGE,
    ASPECT_MMR,
    ASPECT_DIVERSITY,
    ASPECT_MAX_ROWS,
)

_kw_model: KeyBERT | None = None


# ─── Public API ────────────────────────────────────────────────────────────────

def load_aspect_model() -> KeyBERT:
    """Load (or return cached) KeyBERT model."""
    global _kw_model
    if _kw_model is None:
        print(f"[aspect] Loading KeyBERT with '{KEYBERT_MODEL}'…")
        _kw_model = KeyBERT(model=KEYBERT_MODEL)
        print("[aspect] KeyBERT ready.")
    return _kw_model


def extract_aspects(texts: list[str], model: KeyBERT | None = None) -> list[list[str]]:
    """
    Extract keyphrases from each text.

    For large datasets, only runs KeyBERT on a representative sample
    (ASPECT_MAX_ROWS) and marks unsampled rows with an empty list.
    This keeps the pipeline fast without losing meaningful insight —
    aspect aggregation across 300 rows is just as informative as across 5000.

    Parameters
    ----------
    texts  : list of clean text strings
    model  : optional pre-loaded KeyBERT instance

    Returns
    -------
    list of lists — each inner list contains up to ASPECT_TOP_N keyphrases
    """
    if model is None:
        model = load_aspect_model()

    total = len(texts)
    results = [[] for _ in range(total)]

    # Determine which indices to sample
    if total <= ASPECT_MAX_ROWS:
        sample_indices = list(range(total))
    else:
        import random
        random.seed(42)
        sample_indices = sorted(random.sample(range(total), ASPECT_MAX_ROWS))
        print(f"[aspect] Dataset has {total} rows — sampling {ASPECT_MAX_ROWS} for aspect extraction.")

    for idx in sample_indices:
        text = texts[idx]
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            keywords = model.extract_keywords(
                text,
                keyphrase_ngram_range=ASPECT_NGRAM_RANGE,
                stop_words="english",
                top_n=ASPECT_TOP_N,
                use_mmr=ASPECT_MMR,
                diversity=ASPECT_DIVERSITY,
            )
            results[idx] = [kw for kw, _ in keywords]
        except Exception:
            results[idx] = []

    return results


def aggregate_aspect_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an aspect × sentiment count matrix.

    Expects columns: aspects (list[str]), sentiment_label (str).

    Returns
    -------
    pd.DataFrame with columns: aspect, positive, neutral, negative, total
    Sorted by total descending.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        aspects = row.get("aspects", [])
        label   = row.get("sentiment_label", "neutral")
        if not isinstance(aspects, list):
            continue
        for aspect in aspects:
            aspect = aspect.strip().lower()
            if aspect:
                counts[aspect][label] += 1

    records = []
    for aspect, sent_counts in counts.items():
        pos   = sent_counts.get("positive", 0)
        neu   = sent_counts.get("neutral",  0)
        neg   = sent_counts.get("negative", 0)
        total = pos + neu + neg
        records.append({
            "aspect":   aspect,
            "positive": pos,
            "neutral":  neu,
            "negative": neg,
            "total":    total,
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result = result.sort_values("total", ascending=False).reset_index(drop=True)
    return result


def get_top_aspects_by_sentiment(
    df: pd.DataFrame, label: str, top_n: int = 10
) -> list[str]:
    """
    Return the most-mentioned aspects for a specific sentiment label.

    Parameters
    ----------
    df    : DataFrame output from aggregate_aspect_sentiment
    label : "positive" | "neutral" | "negative"
    top_n : number of aspects to return
    """
    if df.empty or label not in df.columns:
        return []
    return df.nlargest(top_n, label)["aspect"].tolist()


def get_representative_quotes(
    df: pd.DataFrame,
    text_col: str,
    label: str,
    top_n: int = 5,
) -> list[str]:
    """
    Return the highest-confidence quotes for a given sentiment label.

    Parameters
    ----------
    df       : analysed DataFrame (must have sentiment_label and confidence cols)
    text_col : original text column name
    label    : "positive" | "neutral" | "negative"
    top_n    : number of quotes to return
    """
    subset = df[df["sentiment_label"] == label].copy()
    if subset.empty:
        return []

    subset = subset.sort_values("confidence", ascending=False)
    quotes = subset[text_col].dropna().head(top_n).tolist()
    return [str(q)[:400] for q in quotes]   # cap length for display
