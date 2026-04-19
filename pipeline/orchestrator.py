"""
pipeline/orchestrator.py
─────────────────────────
Coordinates the full analysis pipeline end-to-end.

Usage
-----
from pipeline.orchestrator import run_pipeline, compute_stats

df_results = run_pipeline(df, schema, progress_callback=None)
stats       = compute_stats(df_results, schema)
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from pipeline.preprocess import preprocess
from pipeline.analyze.sentiment import load_sentiment_model, run_sentiment
from pipeline.analyze.emotion import load_emotion_model, run_emotion
from pipeline.analyze.aspect import load_aspect_model, extract_aspects, aggregate_aspect_sentiment
from pipeline.analyze.summarizer import generate_summary
from config.settings import EMOTION_LABELS, SENTIMENT_LABELS, MAX_ROWS_WARNING


# ─── Main pipeline function ───────────────────────────────────────────────────

def run_pipeline(
    df: pd.DataFrame,
    schema: dict,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> pd.DataFrame:
    """
    Run the full sentiment analysis pipeline.

    Parameters
    ----------
    df                : raw input DataFrame
    schema            : dict with text_col, date_col, score_col, category_col
    progress_callback : optional fn(message: str, pct: float) for UI progress bars

    Returns
    -------
    DataFrame with all original columns plus:
      clean_text, lemmatized_text, lang,
      sentiment_label, sentiment_score, confidence,
      prob_positive, prob_neutral, prob_negative,
      dominant_emotion, emo_joy, emo_anger, … (one per EMOTION_LABELS),
      aspects (list[str])
    """
    def _update(msg: str, pct: float):
        print(f"[pipeline] {msg} ({int(pct * 100)}%)")
        if progress_callback:
            progress_callback(msg, pct)

    text_col = schema["text_col"]

    # ── Row cap warning ───────────────────────────────────────────────────────
    if len(df) > MAX_ROWS_WARNING:
        print(f"[pipeline] Large dataset: {len(df)} rows. Consider sampling for faster results.")

    # ── Stage 1: Preprocess (lemmatization deferred to word cloud tab) ────────
    _update("Preprocessing text…", 0.05)
    df = preprocess(df, text_col, lemmatize=False)

    if df.empty:
        raise ValueError(
            "No rows remained after preprocessing. "
            "Check that the text column contains English text and is long enough."
        )

    texts = df["clean_text"].tolist()

    # ── Stage 2: Load models ──────────────────────────────────────────────────
    _update("Loading sentiment model…", 0.15)
    sentiment_model = load_sentiment_model()

    _update("Loading emotion model…", 0.25)
    emotion_model = load_emotion_model()

    _update("Loading aspect model…", 0.35)
    aspect_model = load_aspect_model()

    # ── Stage 3: Sentiment ────────────────────────────────────────────────────
    _update("Running sentiment analysis…", 0.45)
    sentiment_results = run_sentiment(texts, sentiment_model)

    df["sentiment_label"]  = [r["label"]           for r in sentiment_results]
    df["sentiment_score"]  = [r["sentiment_score"]  for r in sentiment_results]
    df["confidence"]       = [r["confidence"]       for r in sentiment_results]
    df["prob_positive"]    = [r["prob_positive"]    for r in sentiment_results]
    df["prob_neutral"]     = [r["prob_neutral"]     for r in sentiment_results]
    df["prob_negative"]    = [r["prob_negative"]    for r in sentiment_results]

    # ── Stage 4: Emotion ──────────────────────────────────────────────────────
    _update("Running emotion detection…", 0.60)
    emotion_results = run_emotion(texts, emotion_model)

    df["dominant_emotion"] = [r["dominant_emotion"] for r in emotion_results]
    for emo in EMOTION_LABELS:
        df[f"emo_{emo}"] = [r.get(emo, 0.0) for r in emotion_results]

    # ── Stage 5: Aspect extraction ────────────────────────────────────────────
    _update("Extracting aspects and topics…", 0.75)
    df["aspects"] = extract_aspects(texts, aspect_model)

    # ── Stage 6: Parse dates ──────────────────────────────────────────────────
    date_col = schema.get("date_col")
    if date_col and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], format="mixed", dayfirst=False)
            print(f"[pipeline] Date column '{date_col}' parsed successfully.")
        except Exception as e:
            print(f"[pipeline] Date parsing failed for '{date_col}': {e}. Trend chart will be unavailable.")

    _update("Pipeline complete.", 1.0)
    return df


# ─── Stats aggregation ────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame, schema: dict) -> dict:
    """
    Compute aggregated statistics for the dashboard and LLM summary.

    Returns
    -------
    dict with all keys needed by visualize/ modules and summarizer.
    """
    n = len(df)
    if n == 0:
        return {}

    # Sentiment distribution
    label_counts = df["sentiment_label"].value_counts()
    pct = {lbl: round(label_counts.get(lbl, 0) / n * 100, 1) for lbl in SENTIMENT_LABELS}

    # Average score
    avg_score = round(df["sentiment_score"].mean(), 3)

    # Emotion distribution (average probability per emotion)
    emotion_dist = {}
    for emo in EMOTION_LABELS:
        col = f"emo_{emo}"
        if col in df.columns:
            emotion_dist[emo] = round(df[col].mean(), 4)

    dominant_emotion = max(emotion_dist, key=emotion_dist.get) if emotion_dist else "N/A"

    # Aspect-sentiment aggregation
    aspect_df = aggregate_aspect_sentiment(df)
    top_pos = aspect_df.nlargest(5, "positive")["aspect"].tolist() if not aspect_df.empty else []
    top_neg = aspect_df.nlargest(5, "negative")["aspect"].tolist() if not aspect_df.empty else []

    # Category (if available)
    category_col = schema.get("category_col")
    category = df[category_col].mode()[0] if category_col and category_col in df.columns else None

    # Time trend data (if date column available)
    date_col = schema.get("date_col")
    trend_data = None
    if date_col and date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        trend_data = _compute_trend(df, date_col)

    return {
        "total_records":    n,
        "pct_positive":     pct["positive"],
        "pct_neutral":      pct["neutral"],
        "pct_negative":     pct["negative"],
        "count_positive":   int(label_counts.get("positive", 0)),
        "count_neutral":    int(label_counts.get("neutral",  0)),
        "count_negative":   int(label_counts.get("negative", 0)),
        "avg_score":        avg_score,
        "emotion_dist":     emotion_dist,
        "dominant_emotion": dominant_emotion,
        "top_pos_aspects":  top_pos,
        "top_neg_aspects":  top_neg,
        "aspect_df":        aspect_df,
        "trend_data":       trend_data,
        "category":         category,
    }


def generate_executive_summary(stats: dict) -> str:
    """Call the LLM summarizer with computed stats."""
    return generate_summary(stats)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _compute_trend(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Compute weekly average sentiment score over time.
    Returns a DataFrame with columns: period, avg_score, count.
    """
    tmp = df[[date_col, "sentiment_score"]].copy()
    tmp = tmp.dropna(subset=[date_col])
    tmp = tmp.set_index(date_col)

    # Determine resample frequency based on date range
    date_range = (tmp.index.max() - tmp.index.min()).days
    freq = "W" if date_range > 30 else "D"

    trend = tmp.resample(freq)["sentiment_score"].agg(["mean", "count"]).reset_index()
    trend.columns = ["period", "avg_score", "count"]
    return trend
