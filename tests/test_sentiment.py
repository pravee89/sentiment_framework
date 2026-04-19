"""
tests/test_sentiment.py
────────────────────────
Unit tests for the sentiment and emotion analysis modules.

Note: These tests load real transformer models (~500 MB).
They are marked with pytest.mark.slow and skipped in CI unless
the RUN_SLOW_TESTS=1 environment variable is set.

For fast CI: only the pure-logic functions are tested by default.

Run all tests (including model tests):
    RUN_SLOW_TESTS=1 pytest tests/test_sentiment.py -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SLOW = pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to run model tests",
)


# ─── Pure logic tests (always run) ───────────────────────────────────────────

class TestSentimentLogic:
    """Test helper functions that don't require model loading."""

    def test_normalise_label2_mapping(self):
        from pipeline.analyze.sentiment import _normalise_scores
        raw = [
            {"label": "LABEL_2", "score": 0.85},
            {"label": "LABEL_1", "score": 0.10},
            {"label": "LABEL_0", "score": 0.05},
        ]
        result = _normalise_scores(raw)
        assert result["positive"] == pytest.approx(0.85)
        assert result["neutral"]  == pytest.approx(0.10)
        assert result["negative"] == pytest.approx(0.05)

    def test_normalise_direct_labels(self):
        from pipeline.analyze.sentiment import _normalise_scores
        raw = [
            {"label": "positive", "score": 0.9},
            {"label": "neutral",  "score": 0.06},
            {"label": "negative", "score": 0.04},
        ]
        result = _normalise_scores(raw)
        assert result["positive"] == pytest.approx(0.9)

    def test_score_to_label_positive(self):
        from pipeline.analyze.sentiment import sentiment_score_to_label
        assert sentiment_score_to_label(0.5)  == "positive"
        assert sentiment_score_to_label(0.11) == "positive"

    def test_score_to_label_negative(self):
        from pipeline.analyze.sentiment import sentiment_score_to_label
        assert sentiment_score_to_label(-0.5)  == "negative"
        assert sentiment_score_to_label(-0.11) == "negative"

    def test_score_to_label_neutral(self):
        from pipeline.analyze.sentiment import sentiment_score_to_label
        assert sentiment_score_to_label(0.0)  == "neutral"
        assert sentiment_score_to_label(0.05) == "neutral"
        assert sentiment_score_to_label(-0.05) == "neutral"


class TestEmotionLogic:

    def test_get_emotion_distribution_empty(self):
        from pipeline.analyze.emotion import get_emotion_distribution
        result = get_emotion_distribution([])
        from config.settings import EMOTION_LABELS
        for label in EMOTION_LABELS:
            assert label in result
            assert result[label] == 0.0

    def test_get_emotion_distribution_averages(self):
        from pipeline.analyze.emotion import get_emotion_distribution
        rows = [
            {"dominant_emotion": "joy",   "joy": 0.8, "anger": 0.2,
             "sadness": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "neutral": 0.0},
            {"dominant_emotion": "anger", "joy": 0.2, "anger": 0.8,
             "sadness": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "neutral": 0.0},
        ]
        dist = get_emotion_distribution(rows)
        assert dist["joy"]   == pytest.approx(0.5)
        assert dist["anger"] == pytest.approx(0.5)


class TestAspectLogic:

    def test_aggregate_returns_dataframe(self):
        import pandas as pd
        from pipeline.analyze.aspect import aggregate_aspect_sentiment

        df = pd.DataFrame({
            "aspects": [
                ["delivery", "packaging"],
                ["customer service"],
                ["delivery"],
            ],
            "sentiment_label": ["positive", "negative", "positive"],
        })
        result = aggregate_aspect_sentiment(df)
        assert not result.empty
        assert "aspect" in result.columns
        assert "positive" in result.columns
        assert "negative" in result.columns

    def test_aggregate_counts_correctly(self):
        import pandas as pd
        from pipeline.analyze.aspect import aggregate_aspect_sentiment

        df = pd.DataFrame({
            "aspects": [["speed"], ["speed"], ["price"]],
            "sentiment_label": ["positive", "positive", "negative"],
        })
        result = aggregate_aspect_sentiment(df)
        speed_row = result[result["aspect"] == "speed"].iloc[0]
        assert speed_row["positive"] == 2
        assert speed_row["negative"] == 0
        assert speed_row["total"] == 2

    def test_get_representative_quotes(self):
        import pandas as pd
        from pipeline.analyze.aspect import get_representative_quotes

        df = pd.DataFrame({
            "review_text": [
                "Great product!",
                "Terrible service.",
                "Amazing quality!",
                "Would not buy again.",
            ],
            "sentiment_label": ["positive", "negative", "positive", "negative"],
            "confidence":      [0.95, 0.90, 0.88, 0.85],
        })
        quotes = get_representative_quotes(df, "review_text", "positive", top_n=2)
        assert len(quotes) == 2
        assert quotes[0] == "Great product!"   # highest confidence positive


# ─── Model tests (slow — require downloading models) ─────────────────────────

@SLOW
class TestSentimentModel:

    @pytest.fixture(scope="class")
    def model(self):
        from pipeline.analyze.sentiment import load_sentiment_model
        return load_sentiment_model()

    def test_positive_text(self, model):
        from pipeline.analyze.sentiment import run_sentiment
        results = run_sentiment(["I absolutely love this product! It's amazing."], model)
        assert results[0]["label"] == "positive"
        assert results[0]["sentiment_score"] > 0

    def test_negative_text(self, model):
        from pipeline.analyze.sentiment import run_sentiment
        results = run_sentiment(["Terrible quality, broke after one day. Very disappointed."], model)
        assert results[0]["label"] == "negative"
        assert results[0]["sentiment_score"] < 0

    def test_neutral_text(self, model):
        from pipeline.analyze.sentiment import run_sentiment
        results = run_sentiment(["The product arrived on time."], model)
        # Neutral may be classified as either neutral or slight positive — just check score
        assert -0.5 < results[0]["sentiment_score"] < 0.5

    def test_output_keys_present(self, model):
        from pipeline.analyze.sentiment import run_sentiment
        results = run_sentiment(["Test text"], model)
        for key in ("label", "confidence", "sentiment_score", "prob_positive", "prob_neutral", "prob_negative"):
            assert key in results[0]

    def test_batch_processing(self, model):
        from pipeline.analyze.sentiment import run_sentiment
        texts = ["Great!", "Terrible!", "Okay."] * 20
        results = run_sentiment(texts, model)
        assert len(results) == 60

    def test_empty_text_handled(self, model):
        from pipeline.analyze.sentiment import run_sentiment
        results = run_sentiment(["", None, "  "], model)
        assert len(results) == 3
        for r in results:
            assert r["label"] in ("positive", "neutral", "negative")


@SLOW
class TestEmotionModel:

    @pytest.fixture(scope="class")
    def model(self):
        from pipeline.analyze.emotion import load_emotion_model
        return load_emotion_model()

    def test_joy_detected(self, model):
        from pipeline.analyze.emotion import run_emotion
        results = run_emotion(["I'm so happy and excited about this!", ], model)
        assert results[0]["dominant_emotion"] in ("joy", "surprise")

    def test_anger_detected(self, model):
        from pipeline.analyze.emotion import run_emotion
        results = run_emotion(["This is absolutely infuriating! I'm so angry."], model)
        assert results[0]["dominant_emotion"] == "anger"

    def test_all_emotion_keys_present(self, model):
        from pipeline.analyze.emotion import run_emotion
        from config.settings import EMOTION_LABELS
        results = run_emotion(["Some text here."], model)
        for label in EMOTION_LABELS:
            assert label in results[0]

    def test_probabilities_sum_to_one(self, model):
        from pipeline.analyze.emotion import run_emotion
        from config.settings import EMOTION_LABELS
        results = run_emotion(["This is a test."], model)
        total = sum(results[0][label] for label in EMOTION_LABELS)
        assert total == pytest.approx(1.0, abs=0.05)
