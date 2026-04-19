"""
pipeline/analyze/sentiment.py
──────────────────────────────
Polarity classification using a RoBERTa model fine-tuned on consumer text.

Model : cardiffnlp/twitter-roberta-base-sentiment-latest
Labels: positive | neutral | negative
Output: label, confidence score (0–1), continuous sentiment score (−1 to +1)

The continuous score = P(positive) − P(negative), giving a smooth signal
suitable for trend analysis and ranking.
"""

from __future__ import annotations

import torch
from transformers import pipeline as hf_pipeline

from config.settings import SENTIMENT_MODEL, BATCH_SIZE, MAX_TOKEN_LENGTH

# Module-level cache so the model loads only once per session
_model_pipeline = None


# ─── Public API ────────────────────────────────────────────────────────────────

def load_sentiment_model():
    """Load (or return cached) sentiment pipeline."""
    global _model_pipeline
    if _model_pipeline is None:
        device = 0 if torch.cuda.is_available() else -1
        print(f"[sentiment] Loading model '{SENTIMENT_MODEL}' (device={device})…")
        _model_pipeline = hf_pipeline(
            "text-classification",
            model=SENTIMENT_MODEL,
            top_k=None,           # replaces deprecated return_all_scores=True
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            device=device,
        )
        print("[sentiment] Model loaded.")
    return _model_pipeline


def run_sentiment(texts: list[str], model=None) -> list[dict]:
    """
    Classify a list of texts.

    Parameters
    ----------
    texts : list of strings (clean_text, not lemmatized)
    model : optional pre-loaded pipeline; loads default if None

    Returns
    -------
    list of dicts:
      {
        "label":           "positive" | "neutral" | "negative",
        "confidence":      float (0–1),
        "sentiment_score": float (−1 to +1),
        "prob_positive":   float,
        "prob_neutral":    float,
        "prob_negative":   float,
      }
    """
    if model is None:
        model = load_sentiment_model()

    # Replace empty / None texts with a placeholder
    safe_texts = [t if isinstance(t, str) and t.strip() else "no content" for t in texts]

    results = []
    for i in range(0, len(safe_texts), BATCH_SIZE):
        batch = safe_texts[i: i + BATCH_SIZE]
        batch_output = model(batch)

        # Normalize output structure — transformers versions differ:
        # Older: list of dicts (single text)  → [[dict, dict, dict]]
        # Newer: list of lists of dicts        → [[dict, dict, dict], ...]
        if batch_output and isinstance(batch_output[0], dict):
            batch_output = [batch_output]

        for item_scores in batch_output:
            scores = _normalise_scores(item_scores)
            label = max(scores, key=scores.get)
            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            results.append({
                "label":           label,
                "confidence":      round(scores[label], 4),
                "sentiment_score": round(pos - neg, 4),
                "prob_positive":   round(pos, 4),
                "prob_neutral":    round(scores.get("neutral", 0.0), 4),
                "prob_negative":   round(neg, 4),
            })
    return results


def sentiment_score_to_label(score: float) -> str:
    """Convert a continuous score (−1 to +1) to a discrete label."""
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    return "neutral"


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _normalise_scores(item_scores: list[dict]) -> dict:
    """
    Convert the model's output list:
      [{"label": "LABEL_2", "score": 0.9}, ...]
    into a clean dict:
      {"positive": 0.9, "neutral": 0.05, "negative": 0.05}

    The CardiffNLP model uses LABEL_0/1/2 → negative/neutral/positive.
    We also handle models that return "positive"/"neutral"/"negative" directly.
    """
    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
        "negative": "negative",
        "neutral":  "neutral",
        "positive": "positive",
    }
    out = {}
    for entry in item_scores:
        raw_label = entry["label"].lower()
        mapped    = label_map.get(raw_label, raw_label)
        out[mapped] = entry["score"]
    return out
