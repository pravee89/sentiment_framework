"""
pipeline/visualize/charts.py
─────────────────────────────
All Plotly chart generators for the Streamlit dashboard.

Charts
------
sentiment_distribution_chart  — donut chart
emotion_breakdown_chart        — horizontal bar chart
sentiment_trend_chart          — line chart over time (requires date column)
aspect_heatmap_chart           — heatmap: aspect × sentiment count
score_histogram_chart          — distribution of continuous sentiment scores
rating_vs_sentiment_chart      — scatter/box: numeric rating vs sentiment score
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import (
    SENTIMENT_LABELS,
    SENTIMENT_COLORS,
    EMOTION_LABELS,
    EMOTION_COLORS,
)


# ─── Sentiment distribution ───────────────────────────────────────────────────

def sentiment_distribution_chart(stats: dict) -> go.Figure:
    """Donut chart showing positive / neutral / negative breakdown."""
    labels = [lbl.capitalize() for lbl in SENTIMENT_LABELS]
    values = [stats.get(f"count_{lbl}", 0) for lbl in SENTIMENT_LABELS]
    colors = [SENTIMENT_COLORS[lbl] for lbl in SENTIMENT_LABELS]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker_colors=colors,
        textinfo="percent+label",
        hovertemplate="%{label}: %{value:,} reviews (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Sentiment Distribution", font=dict(size=18)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        margin=dict(t=50, b=40, l=20, r=20),
        height=380,
    )
    return fig


# ─── Emotion breakdown ────────────────────────────────────────────────────────

def emotion_breakdown_chart(stats: dict) -> go.Figure:
    """Horizontal bar chart of average emotion probabilities."""
    emotion_dist = stats.get("emotion_dist", {})
    emotions = [e for e in EMOTION_LABELS if e in emotion_dist]
    values   = [round(emotion_dist[e] * 100, 1) for e in emotions]
    colors   = [EMOTION_COLORS[e] for e in emotions]

    # Sort by value
    paired = sorted(zip(values, emotions, colors), reverse=True)
    values, emotions, colors = zip(*paired) if paired else ([], [], [])

    fig = go.Figure(go.Bar(
        x=list(values),
        y=[e.capitalize() for e in emotions],
        orientation="h",
        marker_color=list(colors),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Emotion Breakdown", font=dict(size=18)),
        xaxis=dict(title="Average Probability (%)", range=[0, max(values) * 1.25 if values else 1]),
        yaxis=dict(title=""),
        margin=dict(t=50, b=40, l=20, r=60),
        height=380,
    )
    return fig


# ─── Sentiment trend over time ────────────────────────────────────────────────

def sentiment_trend_chart(trend_data: pd.DataFrame) -> go.Figure:
    """Line chart of average sentiment score over time."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend_data["period"],
        y=trend_data["avg_score"],
        mode="lines+markers",
        name="Avg Sentiment Score",
        line=dict(color="#3498db", width=2.5),
        marker=dict(size=6),
        hovertemplate="Period: %{x|%Y-%m-%d}<br>Score: %{y:.3f}<extra></extra>",
    ))

    # Zero line (neutral)
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)

    # Shaded positive region
    fig.add_hrect(y0=0, y1=1, fillcolor="#2ecc71", opacity=0.05, line_width=0)
    fig.add_hrect(y0=-1, y1=0, fillcolor="#e74c3c", opacity=0.05, line_width=0)

    fig.update_layout(
        title=dict(text="Sentiment Score Over Time", font=dict(size=18)),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Avg Sentiment Score (−1 to +1)", range=[-1.05, 1.05]),
        margin=dict(t=50, b=40, l=20, r=20),
        height=380,
    )
    return fig


# ─── Aspect heatmap ───────────────────────────────────────────────────────────

def aspect_heatmap_chart(aspect_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Heatmap: top aspects (y-axis) × sentiment (x-axis), coloured by count.
    """
    if aspect_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Aspect Sentiment Heatmap (no data)")
        return fig

    top = aspect_df.head(top_n).copy()
    top = top.set_index("aspect")[["positive", "neutral", "negative"]]

    fig = go.Figure(go.Heatmap(
        z=top.values,
        x=["Positive", "Neutral", "Negative"],
        y=top.index.tolist(),
        colorscale=[
            [0.0, "#f8f9fa"],
            [0.5, "#3498db"],
            [1.0, "#1a252f"],
        ],
        hovertemplate="Aspect: %{y}<br>Sentiment: %{x}<br>Count: %{z}<extra></extra>",
        text=top.values,
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title=dict(text=f"Top {top_n} Aspects by Sentiment", font=dict(size=18)),
        xaxis=dict(title="Sentiment"),
        yaxis=dict(title="Aspect", autorange="reversed"),
        margin=dict(t=50, b=60, l=160, r=20),
        height=max(350, top_n * 30),
    )
    return fig


# ─── Score histogram ──────────────────────────────────────────────────────────

def score_histogram_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of continuous sentiment scores (−1 to +1)."""
    fig = px.histogram(
        df,
        x="sentiment_score",
        nbins=40,
        color_discrete_sequence=["#3498db"],
        labels={"sentiment_score": "Sentiment Score (−1 to +1)"},
        title="Sentiment Score Distribution",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.6)
    fig.update_layout(
        bargap=0.05,
        margin=dict(t=50, b=40, l=20, r=20),
        height=350,
    )
    return fig


# ─── Rating vs sentiment ──────────────────────────────────────────────────────

def rating_vs_sentiment_chart(df: pd.DataFrame, score_col: str) -> go.Figure:
    """Box plot: numeric star rating vs sentiment score."""
    tmp = df[[score_col, "sentiment_score"]].dropna()
    tmp[score_col] = tmp[score_col].astype(str)

    fig = px.box(
        tmp,
        x=score_col,
        y="sentiment_score",
        color_discrete_sequence=["#3498db"],
        labels={
            score_col: f"Rating ({score_col})",
            "sentiment_score": "Sentiment Score",
        },
        title=f"Rating vs Sentiment Score",
    )
    fig.update_layout(
        margin=dict(t=50, b=40, l=20, r=20),
        height=380,
    )
    return fig
