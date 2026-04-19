"""
pipeline/analyze/summarizer.py
───────────────────────────────
LLM-powered executive summary using the Anthropic API.

Takes aggregated stats (sentiment distribution, top aspects, dominant emotions)
and returns a 4–5 sentence natural language summary with one concrete
recommendation. Falls back gracefully if the API key is not configured.
"""

from __future__ import annotations

import os
from typing import Optional

from config.settings import LLM_MODEL, LLM_MAX_TOKENS, LLM_SUMMARY_ENABLED


# ─── Public API ────────────────────────────────────────────────────────────────

def generate_summary(stats: dict) -> str:
    """
    Generate a natural language executive summary from aggregated stats.

    Parameters
    ----------
    stats : dict with keys:
        total_records   : int
        pct_positive    : float (0–100)
        pct_neutral     : float
        pct_negative    : float
        avg_score       : float (−1 to +1)
        top_pos_aspects : list[str]
        top_neg_aspects : list[str]
        dominant_emotion: str
        emotion_dist    : dict[str, float]
        category        : str | None (e.g. "Electronics")

    Returns
    -------
    str — the summary text, or a fallback message if LLM is unavailable
    """
    if not LLM_SUMMARY_ENABLED:
        return _fallback_summary(stats)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _fallback_summary(stats)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        prompt = _build_prompt(stats)
        message = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    except Exception as e:
        print(f"[summarizer] LLM summary failed: {e}. Using fallback.")
        return _fallback_summary(stats)


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _build_prompt(stats: dict) -> str:
    category_line = f"for the '{stats['category']}' category " if stats.get("category") else ""

    pos_aspects = ", ".join(stats.get("top_pos_aspects", [])[:5]) or "N/A"
    neg_aspects = ", ".join(stats.get("top_neg_aspects", [])[:5]) or "N/A"

    emotion_dist = stats.get("emotion_dist", {})
    emotion_summary = ", ".join(
        f"{emo} ({round(pct * 100, 1)}%)"
        for emo, pct in sorted(emotion_dist.items(), key=lambda x: -x[1])
        if pct > 0.05
    )

    return f"""You are a retail analytics expert writing for a business stakeholder audience.

Analyse the following customer feedback summary {category_line}and write a concise 4–5 sentence executive summary followed by one concrete, actionable recommendation.

Data Summary:
- Total feedback records analysed: {stats.get('total_records', 0):,}
- Sentiment breakdown: {stats.get('pct_positive', 0):.1f}% positive, {stats.get('pct_neutral', 0):.1f}% neutral, {stats.get('pct_negative', 0):.1f}% negative
- Average sentiment score: {stats.get('avg_score', 0):.2f} (scale: −1 very negative to +1 very positive)
- What customers praise most: {pos_aspects}
- What customers complain about most: {neg_aspects}
- Dominant emotion: {stats.get('dominant_emotion', 'N/A')}
- Emotion distribution: {emotion_summary}

Write in a professional but accessible tone. Be specific — reference actual aspect names in your summary. End with a single bolded recommendation starting with "**Recommendation:**".
"""


def _fallback_summary(stats: dict) -> str:
    """Rule-based fallback summary when LLM is unavailable."""
    pct_pos = stats.get("pct_positive", 0)
    pct_neg = stats.get("pct_negative", 0)
    total   = stats.get("total_records", 0)
    pos_asp = stats.get("top_pos_aspects", [])
    neg_asp = stats.get("top_neg_aspects", [])

    sentiment_desc = (
        "largely positive" if pct_pos > 60 else
        "mixed" if pct_pos > 40 else
        "predominantly negative"
    )

    lines = [
        f"Analysis of {total:,} feedback records shows {sentiment_desc} sentiment "
        f"({pct_pos:.1f}% positive, {pct_neg:.1f}% negative).",
    ]
    if pos_asp:
        lines.append(f"Customers most frequently praised: {', '.join(pos_asp[:3])}.")
    if neg_asp:
        lines.append(
            f"The main areas of concern were: {', '.join(neg_asp[:3])}. "
            "These represent the clearest opportunities for improvement."
        )
    lines.append(
        "Note: set the ANTHROPIC_API_KEY environment variable to enable "
        "AI-generated summaries with tailored recommendations."
    )
    return " ".join(lines)
