"""
app.py
───────
Streamlit entry point for the Generic Sentiment Analysis Framework.

Run with:
    streamlit run app.py

Tabs
----
1. Upload & Configure — file upload, schema confirmation
2. Overview           — key metrics, sentiment + emotion charts
3. Deep Dive          — aspect heatmap, trend chart, score distribution
4. Word Clouds        — per-sentiment clouds
5. Export             — CSV, XLSX, PDF download buttons
"""

from __future__ import annotations

import io
import time

import pandas as pd
import streamlit as st

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Framework",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Pipeline imports (deferred to avoid slow startup) ────────────────────────
from pipeline.ingest import load_file, infer_schema, validate_schema, get_column_previews
from pipeline.orchestrator import run_pipeline, compute_stats, generate_executive_summary
from pipeline.visualize.charts import (
    sentiment_distribution_chart,
    emotion_breakdown_chart,
    sentiment_trend_chart,
    aspect_heatmap_chart,
    score_histogram_chart,
    rating_vs_sentiment_chart,
)
from pipeline.visualize.wordcloud_gen import generate_wordclouds, wordcloud_to_bytes
from exporters.xlsx_exporter import export_xlsx
from exporters.pdf_exporter import export_pdf


# ─── Session state helpers ────────────────────────────────────────────────────

def _reset_results():
    for key in ("df_results", "stats", "summary", "wc_images", "schema_confirmed"):
        st.session_state.pop(key, None)


def _init_state():
    defaults = {
        "schema": None,
        "df_raw": None,
        "df_results": None,
        "stats": None,
        "summary": None,
        "wc_images": None,
        "schema_confirmed": False,
        "uploaded_filename": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Custom CSS ───────────────────────────────────────────────────────────────

def _inject_css():
    st.markdown("""
    <style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-label { font-size: 0.82rem; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #2c3e50; }
    .metric-sub   { font-size: 0.78rem; color: #adb5bd; }
    .sentiment-pos { color: #2ecc71; }
    .sentiment-neg { color: #e74c3c; }
    .sentiment-neu { color: #3498db; }
    .quote-card {
        background: #fefefe;
        border-left: 4px solid #3498db;
        padding: 10px 16px;
        border-radius: 4px;
        margin-bottom: 10px;
        font-style: italic;
        font-size: 0.93rem;
        color: #444;
    }
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6c757d;
        margin-bottom: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    _init_state()
    _inject_css()

    st.title("💬 Sentiment Analysis Framework")
    st.caption("Upload any feedback dataset — the system handles the rest.")

    tabs = st.tabs([
        "📁 Upload & Configure",
        "📊 Overview",
        "🔍 Deep Dive",
        "☁️ Word Clouds",
        "💾 Export",
    ])

    with tabs[0]:
        _tab_upload()

    with tabs[1]:
        _tab_overview()

    with tabs[2]:
        _tab_deep_dive()

    with tabs[3]:
        _tab_wordclouds()

    with tabs[4]:
        _tab_export()


# ─── Tab 1: Upload & Configure ────────────────────────────────────────────────

def _tab_upload():
    st.subheader("Step 1 — Upload your dataset")

    uploaded = st.file_uploader(
        "Drag & drop or click to upload",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Supported: CSV, Excel, JSON, Parquet",
    )

    if uploaded is None:
        st.info("👆 Upload a file to get started. The system will auto-detect your schema.")
        return

    # Re-load only when a new file is uploaded
    if uploaded.name != st.session_state.get("uploaded_filename"):
        _reset_results()
        st.session_state["uploaded_filename"] = uploaded.name
        ext = "." + uploaded.name.rsplit(".", 1)[-1].lower()
        with st.spinner("Loading file…"):
            try:
                df = load_file(io.BytesIO(uploaded.read()), file_type=ext)
                st.session_state["df_raw"] = df
                st.session_state["schema"] = infer_schema(df)
                st.session_state["schema_confirmed"] = False
            except Exception as e:
                st.error(f"Could not load file: {e}")
                return

    df     = st.session_state["df_raw"]
    schema = st.session_state["schema"]

    st.success(f"✅ Loaded **{len(df):,} rows** × **{len(df.columns)} columns**")

    with st.expander("Preview (first 5 rows)", expanded=False):
        st.dataframe(df.head(), use_container_width=True)

    # ── Schema confirmation ────────────────────────────────────────────────────
    st.subheader("Step 2 — Confirm schema")
    st.caption("We've auto-detected the columns below. Override any that look wrong.")

    all_cols   = ["(none)"] + df.columns.tolist()
    str_cols   = ["(none)"] + df.select_dtypes(include="object").columns.tolist()
    num_cols   = ["(none)"] + df.select_dtypes(include="number").columns.tolist()

    def _col_index(col, options):
        return options.index(col) if col in options else 0

    col1, col2 = st.columns(2)
    with col1:
        text_col = st.selectbox(
            "📝 Text / feedback column *",
            str_cols,
            index=_col_index(schema.get("text_col"), str_cols),
            help="The column containing the free-text feedback",
        )
        date_col_raw = st.selectbox(
            "📅 Date column (optional)",
            all_cols,
            index=_col_index(schema.get("date_col"), all_cols),
        )
    with col2:
        score_col_raw = st.selectbox(
            "⭐ Rating / score column (optional)",
            all_cols,
            index=_col_index(schema.get("score_col"), all_cols),
        )
        cat_col_raw = st.selectbox(
            "🏷️ Category column (optional)",
            all_cols,
            index=_col_index(schema.get("category_col"), all_cols),
        )

    schema_edit = {
        "text_col":     text_col if text_col != "(none)" else None,
        "date_col":     date_col_raw if date_col_raw != "(none)" else None,
        "score_col":    score_col_raw if score_col_raw != "(none)" else None,
        "category_col": cat_col_raw if cat_col_raw != "(none)" else None,
    }

    # Show column previews
    previews = get_column_previews(df, schema_edit)
    if previews.get("text_col"):
        with st.expander("Preview: selected text column", expanded=True):
            for i, sample in enumerate(previews["text_col"][:3]):
                st.markdown(f'<div class="quote-card">{sample}</div>', unsafe_allow_html=True)

    st.session_state["schema"] = schema_edit

    valid, errors = validate_schema(df, schema_edit)
    if errors:
        for e in errors:
            st.warning(f"⚠️ {e}")

    # ── Run button ────────────────────────────────────────────────────────────
    st.subheader("Step 3 — Run analysis")

    run_col, info_col = st.columns([1, 3])
    with run_col:
        run_button = st.button(
            "▶ Run Analysis",
            type="primary",
            disabled=not valid,
            use_container_width=True,
        )
    with info_col:
        st.caption(
            "Analysis runs sentiment classification, emotion detection, and "
            "aspect extraction. First run downloads ~500 MB of models."
        )

    if run_button:
        _run_analysis(df, schema_edit)


def _run_analysis(df: pd.DataFrame, schema: dict):
    progress_bar = st.progress(0, text="Starting pipeline…")
    status_text  = st.empty()

    def on_progress(msg: str, pct: float):
        progress_bar.progress(pct, text=msg)
        status_text.caption(msg)

    try:
        t0 = time.time()
        df_results = run_pipeline(df, schema, progress_callback=on_progress)
        elapsed    = time.time() - t0

        progress_bar.progress(1.0, text="Computing statistics…")
        stats   = compute_stats(df_results, schema)
        summary = generate_executive_summary(stats)
        wc_images = generate_wordclouds(df_results)

        st.session_state["df_results"] = df_results
        st.session_state["stats"]      = stats
        st.session_state["summary"]    = summary
        st.session_state["wc_images"]  = wc_images
        st.session_state["schema_confirmed"] = True

        progress_bar.empty()
        status_text.empty()
        st.success(
            f"✅ Analysis complete — {len(df_results):,} records in {elapsed:.1f}s. "
            "Switch to the **Overview** tab to see results."
        )

    except Exception as e:
        progress_bar.empty()
        st.error(f"Pipeline failed: {e}")
        raise


# ─── Tab 2: Overview ──────────────────────────────────────────────────────────

def _tab_overview():
    if not st.session_state.get("schema_confirmed"):
        st.info("Run the analysis first (Upload & Configure tab).")
        return

    stats   = st.session_state["stats"]
    summary = st.session_state["summary"]

    # Key metric cards
    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, "Total Records",     f"{stats['total_records']:,}",      "")
    _metric_card(c2, "Positive",          f"{stats['pct_positive']:.1f}%",    f"{stats['count_positive']:,} reviews", "pos")
    _metric_card(c3, "Negative",          f"{stats['pct_negative']:.1f}%",    f"{stats['count_negative']:,} reviews", "neg")
    _metric_card(c4, "Avg Score",         f"{stats['avg_score']:+.3f}",       "−1 (neg) to +1 (pos)")

    st.divider()

    # Executive summary
    st.subheader("Executive Summary")
    st.markdown(summary)

    st.divider()

    # Charts
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            sentiment_distribution_chart(stats),
            use_container_width=True,
        )
    with col_right:
        st.plotly_chart(
            emotion_breakdown_chart(stats),
            use_container_width=True,
        )

    # Top aspects quick view
    st.divider()
    asp_col1, asp_col2 = st.columns(2)
    with asp_col1:
        st.markdown("**🟢 What customers love**")
        for asp in stats.get("top_pos_aspects", [])[:5]:
            st.markdown(f"&nbsp;&nbsp;• {asp}")
    with asp_col2:
        st.markdown("**🔴 What customers complain about**")
        for asp in stats.get("top_neg_aspects", [])[:5]:
            st.markdown(f"&nbsp;&nbsp;• {asp}")


def _metric_card(col, label, value, sub, sentiment=""):
    color_class = f"sentiment-{sentiment}" if sentiment else ""
    col.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ─── Tab 3: Deep Dive ─────────────────────────────────────────────────────────

def _tab_deep_dive():
    if not st.session_state.get("schema_confirmed"):
        st.info("Run the analysis first.")
        return

    df_results = st.session_state["df_results"]
    stats      = st.session_state["stats"]
    schema     = st.session_state["schema"]

    # Trend chart
    trend_data = stats.get("trend_data")
    if trend_data is not None and not trend_data.empty:
        st.plotly_chart(sentiment_trend_chart(trend_data), use_container_width=True)
    else:
        st.info("ℹ️ No date column detected — trend chart not available.")

    st.divider()

    # Score histogram
    st.plotly_chart(score_histogram_chart(df_results), use_container_width=True)

    st.divider()

    # Rating vs sentiment (if score column available)
    score_col = schema.get("score_col")
    if score_col and score_col in df_results.columns:
        st.plotly_chart(
            rating_vs_sentiment_chart(df_results, score_col),
            use_container_width=True,
        )

    st.divider()

    # Aspect heatmap
    aspect_df = stats.get("aspect_df")
    if aspect_df is not None and not aspect_df.empty:
        n_aspects = st.slider("Number of aspects to show", 5, 30, 15)
        st.plotly_chart(
            aspect_heatmap_chart(aspect_df, top_n=n_aspects),
            use_container_width=True,
        )
    else:
        st.info("No aspects extracted.")

    st.divider()

    # Representative quotes
    st.subheader("Representative Quotes")
    text_col = schema.get("text_col", "clean_text")

    for label, color in [("positive", "#2ecc71"), ("negative", "#e74c3c"), ("neutral", "#3498db")]:
        subset = df_results[df_results["sentiment_label"] == label].nlargest(3, "confidence")
        st.markdown(
            f'<span style="color:{color}; font-weight:700;">'
            f'{label.capitalize()} ({len(df_results[df_results["sentiment_label"] == label]):,} reviews)'
            f'</span>',
            unsafe_allow_html=True,
        )
        for _, row in subset.iterrows():
            quote = str(row.get(text_col, row.get("clean_text", "")))[:350]
            st.markdown(f'<div class="quote-card">{quote}</div>', unsafe_allow_html=True)
        st.markdown("")


# ─── Tab 4: Word Clouds ───────────────────────────────────────────────────────

def _tab_wordclouds():
    if not st.session_state.get("schema_confirmed"):
        st.info("Run the analysis first.")
        return

    wc_images = st.session_state["wc_images"]

    st.subheader("Word Clouds by Sentiment")
    st.caption(
        "Words are filtered to nouns and adjectives only, domain stopwords removed. "
        "Bigrams preserved. Each cloud represents that sentiment class."
    )

    cols = st.columns(3)
    labels = [("positive", "🟢 Positive"), ("neutral", "🔵 Neutral"), ("negative", "🔴 Negative")]

    for col, (label, title) in zip(cols, labels):
        with col:
            st.markdown(f"**{title}**")
            img = wc_images.get(label)
            if img:
                st.image(img, use_column_width=True)
            else:
                st.caption("Not enough data for this class.")


# ─── Tab 5: Export ────────────────────────────────────────────────────────────

def _tab_export():
    if not st.session_state.get("schema_confirmed"):
        st.info("Run the analysis first.")
        return

    df_results = st.session_state["df_results"]
    stats      = st.session_state["stats"]
    schema     = st.session_state["schema"]
    summary    = st.session_state["summary"]
    wc_images  = st.session_state["wc_images"]

    st.subheader("Download Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📄 CSV")
        st.caption("Full results with all analysis columns")
        csv_data = df_results.drop(columns=["lemmatized_text"], errors="ignore").to_csv(index=False).encode()
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="sentiment_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.markdown("#### 📊 Excel")
        st.caption("Multi-sheet workbook: Summary, Raw Results, Aspects, Quotes")
        if st.button("Generate Excel", use_container_width=True):
            with st.spinner("Building Excel workbook…"):
                xlsx_bytes = export_xlsx(df_results, stats, schema, summary)
            st.download_button(
                "Download Excel",
                data=xlsx_bytes,
                file_name="sentiment_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    with col3:
        st.markdown("#### 📋 PDF Report")
        st.caption("Stakeholder-ready report with charts and quotes")
        if st.button("Generate PDF", use_container_width=True):
            with st.spinner("Building PDF report…"):
                try:
                    charts = {
                        "sentiment": sentiment_distribution_chart(stats),
                        "emotion":   emotion_breakdown_chart(stats),
                    }
                    pdf_bytes = export_pdf(
                        stats=stats,
                        schema=schema,
                        summary_text=summary,
                        wc_images=wc_images,
                        chart_figs=charts,
                        df=df_results,
                    )
                    st.download_button(
                        "Download PDF",
                        data=pdf_bytes,
                        file_name="sentiment_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
                    st.caption("Tip: install `kaleido` for chart embedding: `pip install kaleido`")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
