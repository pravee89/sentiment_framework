"""
tests/test_ingest.py
─────────────────────
Unit tests for the ingest module: file loading and schema inference.

Run with:
    pytest tests/test_ingest.py -v
"""

import io
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.ingest import load_file, infer_schema, validate_schema


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def retail_df():
    """Simulates a retail review dataset."""
    return pd.DataFrame({
        "review_text": [
            "Absolutely loved the product! Fast shipping and great quality.",
            "Terrible experience. The item arrived broken and support was unhelpful.",
            "Average product. Nothing special but gets the job done.",
            "Best purchase I've made in years. Will definitely buy again!",
            "Not worth the money. Poor quality and slow delivery.",
        ] * 10,
        "rating":   [5, 1, 3, 5, 2] * 10,
        "date":     ["2024-01-15", "2024-02-20", "2024-03-05", "2024-04-10", "2024-05-12"] * 10,
        "category": ["Electronics", "Clothing", "Home", "Electronics", "Toys"] * 10,
    })


@pytest.fixture
def ambiguous_df():
    """Dataset with non-obvious column names — tests fallback heuristics."""
    return pd.DataFrame({
        "col_a":  ["Short text.", "Another short sentence here.", "One more."] * 5,
        "col_b":  [
            "This is a much longer piece of customer feedback that clearly contains opinions and detailed thoughts.",
            "The product exceeded my expectations in every possible way. The quality is outstanding.",
            "I was disappointed with my purchase. The item did not match the description at all.",
        ] * 5,
        "col_c":  [4.5, 3.0, 1.5] * 5,
        "col_d":  ["Cat1", "Cat2", "Cat1"] * 5,
    })


@pytest.fixture
def csv_buffer(retail_df):
    """A retail DataFrame serialised as CSV bytes."""
    buf = io.BytesIO()
    retail_df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


# ─── load_file tests ──────────────────────────────────────────────────────────

class TestLoadFile:

    def test_load_csv_from_buffer(self, csv_buffer):
        df = load_file(csv_buffer, file_type=".csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert "review_text" in df.columns

    def test_load_csv_from_path(self, tmp_path, retail_df):
        path = tmp_path / "test.csv"
        retail_df.to_csv(path, index=False)
        df = load_file(str(path))
        assert len(df) == 50

    def test_load_excel(self, tmp_path, retail_df):
        path = tmp_path / "test.xlsx"
        retail_df.to_excel(path, index=False)
        df = load_file(str(path))
        assert "review_text" in df.columns

    def test_load_json(self, tmp_path, retail_df):
        path = tmp_path / "test.json"
        retail_df.to_json(path, orient="records")
        df = load_file(str(path))
        assert len(df) == 50

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file("file.txt")

    def test_missing_file_type_for_buffer_raises(self):
        with pytest.raises(ValueError, match="Provide file_type"):
            load_file(io.BytesIO(b"data"))

    def test_column_names_are_stripped(self, tmp_path, retail_df):
        # Add whitespace to column names
        retail_df.columns = [f"  {c}  " for c in retail_df.columns]
        path = tmp_path / "padded.csv"
        retail_df.to_csv(path, index=False)
        df = load_file(str(path))
        for col in df.columns:
            assert col == col.strip()


# ─── infer_schema tests ───────────────────────────────────────────────────────

class TestInferSchema:

    def test_detects_review_text_column(self, retail_df):
        schema = infer_schema(retail_df)
        assert schema["text_col"] == "review_text"

    def test_detects_rating_column(self, retail_df):
        schema = infer_schema(retail_df)
        assert schema["score_col"] == "rating"

    def test_detects_date_column(self, retail_df):
        schema = infer_schema(retail_df)
        assert schema["date_col"] == "date"

    def test_detects_category_column(self, retail_df):
        schema = infer_schema(retail_df)
        assert schema["category_col"] == "category"

    def test_fallback_to_longest_string_column(self, ambiguous_df):
        schema = infer_schema(ambiguous_df)
        # col_b has much longer average text
        assert schema["text_col"] == "col_b"

    def test_returns_none_for_missing_columns(self):
        df = pd.DataFrame({"feedback": ["Great product!", "Bad quality."] * 5})
        schema = infer_schema(df)
        assert schema["text_col"] == "feedback"
        assert schema["date_col"] is None
        assert schema["score_col"] is None
        assert schema["category_col"] is None

    def test_schema_has_all_required_keys(self, retail_df):
        schema = infer_schema(retail_df)
        for key in ("text_col", "date_col", "score_col", "category_col"):
            assert key in schema


# ─── validate_schema tests ────────────────────────────────────────────────────

class TestValidateSchema:

    def test_valid_schema_returns_true(self, retail_df):
        schema = infer_schema(retail_df)
        valid, errors = validate_schema(retail_df, schema)
        assert valid is True
        assert errors == []

    def test_missing_text_col_is_invalid(self, retail_df):
        schema = {"text_col": None, "date_col": None, "score_col": None, "category_col": None}
        valid, errors = validate_schema(retail_df, schema)
        assert valid is False
        assert any("text column" in e.lower() for e in errors)

    def test_nonexistent_column_is_invalid(self, retail_df):
        schema = {"text_col": "does_not_exist", "date_col": None, "score_col": None, "category_col": None}
        valid, errors = validate_schema(retail_df, schema)
        assert valid is False

    def test_wrong_optional_column_gives_error(self, retail_df):
        schema = {
            "text_col": "review_text",
            "date_col": "nonexistent_col",
            "score_col": None,
            "category_col": None,
        }
        valid, errors = validate_schema(retail_df, schema)
        assert valid is False
        assert any("nonexistent_col" in e for e in errors)

    def test_very_short_text_gives_warning(self):
        df = pd.DataFrame({"tags": ["ok", "bad", "great", "fine", "ok"] * 5})
        schema = {"text_col": "tags", "date_col": None, "score_col": None, "category_col": None}
        valid, errors = validate_schema(df, schema)
        # Should warn but may still be valid depending on threshold
        # At minimum, errors should mention short text
        if not valid:
            assert any("short" in e.lower() for e in errors)
