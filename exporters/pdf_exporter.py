"""
exporters/pdf_exporter.py
──────────────────────────
Generate a stakeholder-ready PDF report using fpdf2.

Report sections
---------------
1. Cover page  — title, dataset info, generation date
2. Executive Summary — LLM or rule-based summary
3. Sentiment Overview — key metrics + donut chart (embedded as image)
4. Emotion Analysis — bar chart (embedded as image)
5. Top Aspects — positive and negative aspect tables
6. Representative Quotes — 3 quotes per sentiment class
7. Word Clouds — three side-by-side images
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from typing import Optional

from fpdf import FPDF

from config.settings import SENTIMENT_LABELS, EMOTION_LABELS


class SentimentReport(FPDF):
    """Custom FPDF subclass with header/footer."""

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "Sentiment Analysis Report", align="L")
            self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Sentiment Analysis Framework", align="C")


def export_pdf(
    stats: dict,
    schema: dict,
    summary_text: str,
    wc_images: dict,           # {"positive": PIL.Image, ...}
    chart_figs: dict,          # {"sentiment": go.Figure, "emotion": go.Figure, ...}
    df=None,
) -> bytes:
    """
    Build and return a PDF report as bytes.

    Parameters
    ----------
    stats        : aggregated stats dict from orchestrator.compute_stats()
    schema       : schema dict
    summary_text : executive summary string
    wc_images    : dict of PIL Images from wordcloud_gen
    chart_figs   : dict of Plotly figures from charts.py
    df           : optional DataFrame for quotes

    Returns
    -------
    bytes — PDF file contents
    """
    pdf = SentimentReport(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(20, 20, 20)

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Cover page ────────────────────────────────────────────────────────
        pdf.add_page()
        _cover_page(pdf, stats, schema)

        # ── Executive summary ─────────────────────────────────────────────────
        pdf.add_page()
        _section_title(pdf, "Executive Summary")
        _body_text(pdf, summary_text or "No summary available.")

        # ── Sentiment overview ────────────────────────────────────────────────
        pdf.add_page()
        _section_title(pdf, "Sentiment Overview")
        _metrics_table(pdf, stats)

        # Embed sentiment chart
        fig = chart_figs.get("sentiment")
        if fig:
            img_path = _save_figure(fig, tmpdir, "sentiment_chart.png")
            if img_path:
                pdf.image(img_path, x=20, w=170)

        # ── Emotion analysis ──────────────────────────────────────────────────
        pdf.add_page()
        _section_title(pdf, "Emotion Analysis")
        _emotion_table(pdf, stats)

        fig = chart_figs.get("emotion")
        if fig:
            img_path = _save_figure(fig, tmpdir, "emotion_chart.png")
            if img_path:
                pdf.image(img_path, x=20, w=170)

        # ── Top aspects ───────────────────────────────────────────────────────
        pdf.add_page()
        _section_title(pdf, "Top Aspects")
        _aspects_section(pdf, stats)

        # ── Quotes ────────────────────────────────────────────────────────────
        if df is not None:
            pdf.add_page()
            _section_title(pdf, "Representative Quotes")
            _quotes_section(pdf, df, schema)

        # ── Word clouds ───────────────────────────────────────────────────────
        if any(img is not None for img in wc_images.values()):
            pdf.add_page()
            _section_title(pdf, "Word Clouds by Sentiment")
            _wordcloud_section(pdf, wc_images, tmpdir)

    output = io.BytesIO()
    pdf_bytes = pdf.output()
    return bytes(pdf_bytes)


# ─── Section renderers ────────────────────────────────────────────────────────

def _cover_page(pdf: FPDF, stats: dict, schema: dict):
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 14, "Sentiment Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(100, 100, 100)

    category = stats.get("category")
    if category:
        pdf.cell(0, 8, f"Category: {category}", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.cell(0, 8, f"Records Analysed: {stats.get('total_records', 0):,}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y')}", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(44, 62, 80)
    pos  = stats.get("pct_positive", 0)
    neu  = stats.get("pct_neutral",  0)
    neg  = stats.get("pct_negative", 0)
    for label, pct, color in [
        ("Positive", pos, (46, 204, 113)),
        ("Neutral",  neu, (52, 152, 219)),
        ("Negative", neg, (231, 76, 60)),
    ]:
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(55, 16, f"{label}: {pct:.1f}%", align="C", fill=True, border=0)
        pdf.cell(5, 16, "")
    pdf.ln(30)


def _section_title(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(44, 62, 80)
    pdf.set_fill_color(245, 246, 250)
    pdf.cell(0, 10, title, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)


def _body_text(pdf: FPDF, text: str):
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 6, text)
    pdf.ln(4)


def _metrics_table(pdf: FPDF, stats: dict):
    rows = [
        ("Total Records",         f"{stats.get('total_records', 0):,}"),
        ("Positive Sentiment",    f"{stats.get('pct_positive', 0):.1f}%  ({stats.get('count_positive', 0):,} reviews)"),
        ("Neutral Sentiment",     f"{stats.get('pct_neutral', 0):.1f}%  ({stats.get('count_neutral', 0):,} reviews)"),
        ("Negative Sentiment",    f"{stats.get('pct_negative', 0):.1f}%  ({stats.get('count_negative', 0):,} reviews)"),
        ("Avg Sentiment Score",   f"{stats.get('avg_score', 0):.3f}  (−1 to +1)"),
        ("Dominant Emotion",      stats.get("dominant_emotion", "N/A").capitalize()),
    ]
    pdf.set_font("Helvetica", "", 11)
    for label, value in rows:
        pdf.set_text_color(80, 80, 80)
        pdf.cell(75, 7, label + ":", border="B")
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 7, value, border="B", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)


def _emotion_table(pdf: FPDF, stats: dict):
    emotion_dist = stats.get("emotion_dist", {})
    pdf.set_font("Helvetica", "", 11)
    for emo in sorted(emotion_dist, key=emotion_dist.get, reverse=True):
        pct = emotion_dist[emo] * 100
        pdf.set_text_color(80, 80, 80)
        pdf.cell(50, 7, emo.capitalize() + ":", border="B")
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 7, f"{pct:.1f}%", border="B", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)


def _aspects_section(pdf: FPDF, stats: dict):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(46, 204, 113)
    pdf.cell(0, 8, "Top Positive Aspects:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(50, 50, 50)
    for asp in stats.get("top_pos_aspects", []):
        pdf.cell(10); pdf.cell(0, 6, f"• {asp}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(231, 76, 60)
    pdf.cell(0, 8, "Top Negative Aspects:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(50, 50, 50)
    for asp in stats.get("top_neg_aspects", []):
        pdf.cell(10); pdf.cell(0, 6, f"• {asp}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)


def _quotes_section(pdf: FPDF, df, schema: dict):
    text_col = schema.get("text_col", "clean_text")
    if text_col not in df.columns:
        text_col = "clean_text"

    for label in SENTIMENT_LABELS:
        subset = df[df["sentiment_label"] == label].nlargest(3, "confidence")
        if subset.empty:
            continue

        color_map = {"positive": (46, 204, 113), "neutral": (52, 152, 219), "negative": (231, 76, 60)}
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*color_map[label])
        pdf.cell(0, 8, label.capitalize(), new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(60, 60, 60)
        for _, row in subset.iterrows():
            quote = str(row.get(text_col, ""))[:280]
            pdf.multi_cell(0, 5.5, f'"{quote}"')
            pdf.ln(2)
        pdf.ln(3)


def _wordcloud_section(pdf: FPDF, wc_images: dict, tmpdir: str):
    labels = [("positive", "Positive"), ("neutral", "Neutral"), ("negative", "Negative")]
    x_positions = [20, 75, 130]
    img_w = 55

    pdf.set_font("Helvetica", "B", 10)
    for (label, title), x in zip(labels, x_positions):
        pdf.set_xy(x, pdf.get_y())
        pdf.cell(img_w, 7, title, align="C")
    pdf.ln(8)

    y = pdf.get_y()
    for (label, _), x in zip(labels, x_positions):
        img = wc_images.get(label)
        if img:
            path = os.path.join(tmpdir, f"wc_{label}.png")
            img.save(path, format="PNG")
            pdf.image(path, x=x, y=y, w=img_w)

    pdf.ln(60)


# ─── Plotly figure → PNG ──────────────────────────────────────────────────────

def _save_figure(fig, tmpdir: str, filename: str) -> Optional[str]:
    """Save a Plotly figure to a temporary PNG file and return the path."""
    try:
        import plotly.io as pio
        path = os.path.join(tmpdir, filename)
        pio.write_image(fig, path, format="png", width=900, height=400, scale=1.5)
        return path
    except Exception as e:
        print(f"[pdf_exporter] Could not save chart {filename}: {e}")
        return None
