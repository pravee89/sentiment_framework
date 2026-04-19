#!/bin/bash
# setup.sh — one-shot environment setup for the Sentiment Analysis Framework

set -e

echo "=== Sentiment Analysis Framework — Setup ==="
echo ""

# 1. Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip --quiet

# 3. Install dependencies
echo "[3/5] Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet

# 4. Download spaCy language model
echo "[4/5] Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# 5. Generate sample data
echo "[5/5] Generating sample dataset..."
python tests/sample_data/generate_sample.py

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start the app:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "To run tests (no model downloads):"
echo "  pytest tests/ -v"
echo ""
echo "To run all tests including model tests:"
echo "  RUN_SLOW_TESTS=1 pytest tests/ -v"
echo ""
echo "Optional: set your Anthropic API key for LLM summaries:"
echo "  export ANTHROPIC_API_KEY=sk-ant-..."
