"""
exporters/xlsx_exporter.py
───────────────────────────
Export analysis results to a formatted Excel workbook (.xlsx).

Sheets
------
1. Summary       — key metrics and sentiment/emotion breakdown
2. Raw Results   — full DataFrame with all analysis columns
3. Aspects       — aspect-sentiment count matrix
4. Quotes        — top representative quotes per sentiment class
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import xlsxwriter

from config.settings import SENTIMENT_LABELS, EMOTION_LABELS, SENTIMENT_COLORS


# ─── Colour palette for Excel ─────────────────────────────────────────────────
EXCEL_COLORS = {
    "positive": "#2ecc71",
    "neutral":  "#3498db",
    "negative": "#e74c3c",
    "header":   "#2c3e50",
    "subheader":"#34495e",
    "light_bg": "#f8f9fa",
}


def export_xlsx(
    df: pd.DataFrame,
    stats: dict,
    schema: dict,
    summary_text: str = "",
) -> bytes:
    """
    Build a formatted Excel workbook and return it as bytes.

    Parameters
    ----------
    df           : fully analysed DataFrame
    stats        : dict from orchestrator.compute_stats()
    schema       : schema dict (for column name references)
    summary_text : LLM executive summary text (optional)

    Returns
    -------
    bytes — Excel file contents, ready for st.download_button
    """
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True})

    # ── Shared formats ────────────────────────────────────────────────────────
    fmt = _build_formats(workbook)

    # ── Sheet 1: Summary ──────────────────────────────────────────────────────
    _write_summary_sheet(workbook, fmt, stats, summary_text)

    # ── Sheet 2: Raw Results ──────────────────────────────────────────────────
    _write_raw_sheet(workbook, fmt, df, schema)

    # ── Sheet 3: Aspects ──────────────────────────────────────────────────────
    aspect_df = stats.get("aspect_df")
    if aspect_df is not None and not aspect_df.empty:
        _write_aspects_sheet(workbook, fmt, aspect_df)

    # ── Sheet 4: Quotes ───────────────────────────────────────────────────────
    _write_quotes_sheet(workbook, fmt, df, schema)

    workbook.close()
    return output.getvalue()


# ─── Sheet writers ────────────────────────────────────────────────────────────

def _write_summary_sheet(wb, fmt, stats: dict, summary_text: str):
    ws = wb.add_worksheet("Summary")
    ws.set_column("A:A", 32)
    ws.set_column("B:B", 20)

    row = 0
    ws.write(row, 0, "Sentiment Analysis Report", fmt["title"])
    ws.write(row, 1, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fmt["meta"])
    row += 2

    # Key metrics
    ws.write(row, 0, "KEY METRICS", fmt["section_header"])
    row += 1
    metrics = [
        ("Total Records Analysed", f"{stats.get('total_records', 0):,}"),
        ("Positive (%)",           f"{stats.get('pct_positive', 0):.1f}%"),
        ("Neutral (%)",            f"{stats.get('pct_neutral', 0):.1f}%"),
        ("Negative (%)",           f"{stats.get('pct_negative', 0):.1f}%"),
        ("Avg Sentiment Score",    f"{stats.get('avg_score', 0):.3f}"),
        ("Dominant Emotion",       stats.get("dominant_emotion", "N/A").capitalize()),
    ]
    for label, value in metrics:
        ws.write(row, 0, label, fmt["label"])
        ws.write(row, 1, value, fmt["value"])
        row += 1

    row += 1
    ws.write(row, 0, "SENTIMENT BREAKDOWN", fmt["section_header"])
    row += 1
    ws.write(row, 0, "Label",   fmt["col_header"])
    ws.write(row, 1, "Count",   fmt["col_header"])
    ws.write(row, 2, "Percent", fmt["col_header"])
    row += 1
    for lbl in SENTIMENT_LABELS:
        ws.write(row, 0, lbl.capitalize(), fmt["cell"])
        ws.write(row, 1, stats.get(f"count_{lbl}", 0), fmt["cell"])
        ws.write(row, 2, f"{stats.get(f'pct_{lbl}', 0):.1f}%", fmt["cell"])
        row += 1

    row += 1
    ws.write(row, 0, "EMOTION BREAKDOWN", fmt["section_header"])
    row += 1
    ws.write(row, 0, "Emotion", fmt["col_header"])
    ws.write(row, 1, "Avg Probability", fmt["col_header"])
    row += 1
    emotion_dist = stats.get("emotion_dist", {})
    for emo in sorted(emotion_dist, key=emotion_dist.get, reverse=True):
        ws.write(row, 0, emo.capitalize(), fmt["cell"])
        ws.write(row, 1, f"{emotion_dist[emo] * 100:.1f}%", fmt["cell"])
        row += 1

    if summary_text:
        row += 1
        ws.write(row, 0, "EXECUTIVE SUMMARY", fmt["section_header"])
        row += 1
        ws.merge_range(row, 0, row + 8, 2, summary_text, fmt["summary_text"])


def _write_raw_sheet(wb, fmt, df: pd.DataFrame, schema: dict):
    ws = wb.add_worksheet("Raw Results")

    # Choose columns to export (exclude internal processing columns)
    exclude = {"lemmatized_text", "lang"}
    cols = [c for c in df.columns if c not in exclude]

    # Header row
    for col_idx, col_name in enumerate(cols):
        ws.write(0, col_idx, col_name, fmt["col_header"])

    ws.set_column(0, len(cols) - 1, 18)

    # Data rows
    for row_idx, row in enumerate(df[cols].itertuples(index=False), start=1):
        for col_idx, value in enumerate(row):
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            ws.write(row_idx, col_idx, str(value) if value is not None else "", fmt["cell"])

    # Freeze header row
    ws.freeze_panes(1, 0)


def _write_aspects_sheet(wb, fmt, aspect_df: pd.DataFrame):
    ws = wb.add_worksheet("Aspects")
    headers = ["Aspect", "Positive", "Neutral", "Negative", "Total"]
    col_widths = [30, 12, 12, 12, 12]

    for col_idx, (h, w) in enumerate(zip(headers, col_widths)):
        ws.write(0, col_idx, h, fmt["col_header"])
        ws.set_column(col_idx, col_idx, w)

    for row_idx, row in enumerate(aspect_df.itertuples(index=False), start=1):
        ws.write(row_idx, 0, row.aspect, fmt["cell"])
        ws.write(row_idx, 1, row.positive, fmt["cell"])
        ws.write(row_idx, 2, row.neutral,  fmt["cell"])
        ws.write(row_idx, 3, row.negative, fmt["cell"])
        ws.write(row_idx, 4, row.total,    fmt["cell"])

    ws.freeze_panes(1, 0)


def _write_quotes_sheet(wb, fmt, df: pd.DataFrame, schema: dict):
    ws = wb.add_worksheet("Quotes")
    text_col = schema.get("text_col", "clean_text")
    if text_col not in df.columns:
        text_col = "clean_text"

    ws.set_column("A:A", 15)
    ws.set_column("B:B", 80)

    ws.write(0, 0, "Sentiment", fmt["col_header"])
    ws.write(0, 1, "Representative Quote",   fmt["col_header"])

    row = 1
    for label in SENTIMENT_LABELS:
        subset = df[df["sentiment_label"] == label].nlargest(5, "confidence")
        for _, r in subset.iterrows():
            ws.write(row, 0, label.capitalize(), fmt["cell"])
            ws.write(row, 1, str(r.get(text_col, ""))[:500], fmt["cell_wrap"])
            row += 1
        row += 1  # blank row between groups

    ws.freeze_panes(1, 0)


# ─── Format builder ───────────────────────────────────────────────────────────

def _build_formats(wb) -> dict:
    return {
        "title": wb.add_format({
            "bold": True, "font_size": 16, "font_color": EXCEL_COLORS["header"],
        }),
        "meta": wb.add_format({
            "italic": True, "font_size": 10, "font_color": "#7f8c8d",
        }),
        "section_header": wb.add_format({
            "bold": True, "font_size": 12, "font_color": "white",
            "bg_color": EXCEL_COLORS["header"], "bottom": 1,
        }),
        "col_header": wb.add_format({
            "bold": True, "bg_color": EXCEL_COLORS["subheader"],
            "font_color": "white", "border": 1, "align": "center",
        }),
        "label": wb.add_format({
            "bold": True, "font_color": EXCEL_COLORS["header"],
        }),
        "value": wb.add_format({
            "font_color": "#2c3e50",
        }),
        "cell": wb.add_format({
            "border": 1, "text_wrap": False, "valign": "top",
        }),
        "cell_wrap": wb.add_format({
            "border": 1, "text_wrap": True, "valign": "top",
        }),
        "summary_text": wb.add_format({
            "text_wrap": True, "valign": "top", "font_size": 11,
            "bg_color": EXCEL_COLORS["light_bg"],
        }),
    }
