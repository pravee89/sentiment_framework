"""
Central configuration for the Sentiment Analysis Framework.
Edit these values to switch models, adjust thresholds, or tune behaviour.
"""

# ─── Model identifiers ─────────────────────────────────────────────────────────
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL   = "j-hartmann/emotion-english-distilroberta-base"
KEYBERT_MODEL   = "all-MiniLM-L6-v2"   # sentence-transformer backbone for KeyBERT
SPACY_MODEL     = "en_core_web_sm"

# ─── Inference settings ────────────────────────────────────────────────────────
BATCH_SIZE          = 32        # texts per inference batch
MAX_TOKEN_LENGTH    = 512       # truncation limit for transformers
MIN_TEXT_LENGTH     = 5         # skip texts shorter than this (words)
SUPPORTED_LANGUAGES = ["en"]    # keep only these language codes

# ─── Aspect extraction ─────────────────────────────────────────────────────────
ASPECT_TOP_N        = 5         # keywords per document
ASPECT_NGRAM_RANGE  = (1, 3)    # unigrams → trigrams
ASPECT_MMR          = True      # Maximal Marginal Relevance for diversity
ASPECT_DIVERSITY    = 0.5       # 0 = repetitive, 1 = maximally diverse

# ─── Word cloud ────────────────────────────────────────────────────────────────
WC_WIDTH            = 900
WC_HEIGHT           = 450
WC_MAX_WORDS        = 100
WC_BG_COLOR         = "white"
WC_COLORMAP_POS     = "Greens"
WC_COLORMAP_NEU     = "Blues"
WC_COLORMAP_NEG     = "Reds"
DOMAIN_STOPWORDS_PATH = "config/domain_stopwords.txt"

# ─── Schema inference hints ────────────────────────────────────────────────────
TEXT_COLUMN_HINTS = [
    "review", "comment", "feedback", "text", "description",
    "opinion", "note", "message", "body", "content", "remarks",
    "review_text", "customer_review", "user_comment"
]
DATE_COLUMN_HINTS   = ["date", "time", "created", "timestamp", "posted", "submitted"]
SCORE_COLUMN_HINTS  = ["rating", "score", "stars", "grade", "mark"]
CATEGORY_COLUMN_HINTS = ["category", "type", "product", "dept", "department",
                          "segment", "group", "channel"]

# ─── LLM summary (Anthropic) ───────────────────────────────────────────────────
LLM_MODEL           = "claude-opus-4-6"
LLM_MAX_TOKENS      = 512
LLM_SUMMARY_ENABLED = True      # set False to skip LLM summary

# ─── Emotion labels ────────────────────────────────────────────────────────────
EMOTION_LABELS = ["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"]

# ─── Sentiment labels ──────────────────────────────────────────────────────────
SENTIMENT_LABELS = ["positive", "neutral", "negative"]
SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral":  "#3498db",
    "negative": "#e74c3c",
}
EMOTION_COLORS = {
    "joy":      "#f1c40f",
    "anger":    "#e74c3c",
    "sadness":  "#3498db",
    "fear":     "#9b59b6",
    "surprise": "#e67e22",
    "disgust":  "#1abc9c",
    "neutral":  "#95a5a6",
}
