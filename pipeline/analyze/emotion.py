"""
pipeline/analyze/emotion.py
────────────────────────────
Emotion detection using a DistilRoBERTa model.

Model : j-hartmann/emotion-english-distilroberta-base
Labels: joy | anger | sadness | fear | surprise | disgust | neutral

This is intentionally separate from sentiment:
  - Sentiment = polarity (positive / negative)
  - Emotion   = specific feeling (joy, anger, etc.)
A review can be positive in sentiment but express "surprise" rather than "joy".
"""

from __future__ import annotations

import torch
from transformers import pipeline as hf_pipeline

from config.settings import EMOTION_MODEL, EMOTION_LABELS, BATCH_SIZE, MAX_TOKEN_LENGTH

_model_pipeline = None


# ─── Public API ────────────────────────────────────────────────────────────────

def load_emotion_model():
    """Load (or return cached) emotion detection pipeline."""
    global _model_pipeline
    if _model_pipeline is None:
        device = 0 if torch.cuda.is_available() else -1
        print(f"[emotion] Loading model '{EMOTION_MODEL}' (device={device})…")
        _model_pipeline = hf_pipeline(
            "text-classification",
            model=EMOTION_MODEL,
            top_k=None,           # replaces deprecated return_all_scores=True
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            device=device,
        )
        print("[emotion] Model loaded.")
    return _model_pipeline


def run_emotion(texts: list[str], model=None) -> list[dict]:
    """
    Detect emotions in a list of texts.

    Parameters
    ----------
    texts : list of strings
    model : optional pre-loaded pipeline

    Returns
    -------
    list of dicts:
      {
        "dominant_emotion": "joy",
        "joy": 0.82,
        "anger": 0.04,
        ...  (one key per EMOTION_LABELS entry)
      }
    """
    if model is None:
        model = load_emotion_model()

    safe_texts = [t if isinstance(t, str) and t.strip() else "no content" for t in texts]

    results = []
    for i in range(0, len(safe_texts), BATCH_SIZE):
        batch = safe_texts[i: i + BATCH_SIZE]
        batch_output = model(batch)

        # Normalize output structure — same fix as sentiment.py
        if batch_output and isinstance(batch_output[0], dict):
            batch_output = [batch_output]

        for item_scores in batch_output:
            scores = {entry["label"].lower(): round(entry["score"], 4) for entry in item_scores}
            for label in EMOTION_LABELS:
                scores.setdefault(label, 0.0)
            dominant = max(EMOTION_LABELS, key=lambda l: scores[l])
            results.append({"dominant_emotion": dominant, **scores})

    return results


def get_emotion_distribution(emotion_results: list[dict]) -> dict:
    """
    Aggregate emotion probabilities across all rows.

    Returns
    -------
    dict: {emotion: average_probability}
    """
    if not emotion_results:
        return {label: 0.0 for label in EMOTION_LABELS}

    totals = {label: 0.0 for label in EMOTION_LABELS}
    for row in emotion_results:
        for label in EMOTION_LABELS:
            totals[label] += row.get(label, 0.0)

    n = len(emotion_results)
    return {label: round(totals[label] / n, 4) for label in EMOTION_LABELS}


