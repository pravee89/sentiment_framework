# Sentiment Analysis Framework — GitHub & Launch Guide

This guide covers two things:
1. **Pushing the code to your GitHub account** (step-by-step with authentication)
2. **Running the framework locally** from scratch

---

## PART 1 — Push Code to GitHub

### Step 1 — Find the project folder on your Mac

The code lives in your **Personal Projects** folder. Open Finder to locate the exact path, or in Terminal:

```bash
# If your Personal Projects folder is on the Desktop:
cd ~/Desktop/Personal\ Projects/sentiment-framework

# Or wherever your Personal Projects folder is located:
cd /path/to/Personal\ Projects/sentiment-framework

# Verify you're in the right place:
ls
# You should see: app.py  pipeline/  config/  exporters/  tests/  requirements.txt
```

---

### Step 2 — Set up GitHub Authentication (Personal Access Token)

Since you're not sure about your git auth setup, the safest method is a **Personal Access Token (PAT)** over HTTPS. This works on all machines without SSH key setup.

**Create your PAT:**

1. Go to [github.com](https://github.com) → click your profile picture (top right) → **Settings**
2. Scroll down the left sidebar → **Developer settings**
3. Click **Personal access tokens** → **Tokens (classic)**
4. Click **Generate new token (classic)**
5. Fill in:
   - **Note:** `sentiment-framework-push`
   - **Expiration:** 90 days (or No expiration)
   - **Scopes:** check ✅ `repo` (this gives full repo access)
6. Click **Generate token**
7. **Copy the token immediately** — GitHub only shows it once. It looks like: `ghp_xxxxxxxxxxxxxxxxxxxx`

**Save the token so you don't have to re-enter it:**

```bash
# On Mac — store credentials in the macOS Keychain (recommended)
git config --global credential.helper osxkeychain

# On Linux/Windows:
git config --global credential.helper store
```

---

### Step 3 — Initialize Git in the project folder

```bash
# Navigate to your project folder first (from Step 1)
cd ~/Desktop/Personal\ Projects/sentiment-framework    # adjust path if needed

# Initialize git
git init

# Set your identity (only needed once per machine)
git config user.name "Praveen"
git config user.email "praveelife@gmail.com"

# Rename branch to 'main' (GitHub default)
git branch -M main
```

---

### Step 4 — Add your GitHub repository as the remote

You said you already have a repo at `https://github.com/pravee89`. Use your repo URL:

```bash
# Replace YOUR-REPO-NAME with the actual repository name you created on GitHub
git remote add origin https://github.com/pravee89/YOUR-REPO-NAME.git

# Verify the remote was added:
git remote -v
# Should show:
# origin  https://github.com/pravee89/YOUR-REPO-NAME.git (fetch)
# origin  https://github.com/pravee89/YOUR-REPO-NAME.git (push)
```

> **Example:** If your repo is named `sentiment-analysis-framework`:
> ```bash
> git remote add origin https://github.com/pravee89/sentiment-analysis-framework.git
> ```

---

### Step 5 — Stage all files and create the first commit

```bash
# Stage everything (the .gitignore will automatically exclude venv/, .env, etc.)
git add .

# Verify what's staged — you should see all .py files, requirements.txt, etc.
git status

# Create the first commit
git commit -m "feat: initial commit — generic sentiment analysis framework

- Modular pipeline: ingest → preprocess → sentiment → emotion → aspect extraction
- Streamlit UI with 5-tab dashboard
- Export to CSV, Excel, and PDF
- Configurable via config/settings.py
- Unit tests for ingest and analysis modules"
```

---

### Step 6 — Push to GitHub

```bash
git push -u origin main
```

Git will ask for your credentials:
- **Username:** `pravee89` (your GitHub username)
- **Password:** paste your PAT token from Step 2 (not your GitHub password)

After this first push, credentials are saved in Keychain and you won't be asked again.

**Verify it worked:** Go to `https://github.com/pravee89/YOUR-REPO-NAME` in your browser — you should see all the files.

---

### Future pushes (after making changes)

```bash
git add .
git commit -m "your commit message"
git push
```

---

---

## PART 2 — How to Launch the Framework

### Prerequisites

- Python 3.10 or higher (check: `python3 --version`)
- pip (comes with Python)
- ~1.5 GB free disk space (for ML models that download on first run)
- Internet connection on first run (to download HuggingFace models)

---

### Step 1 — Navigate to the project folder

```bash
cd ~/Desktop/Personal\ Projects/sentiment-framework   # adjust path as needed
```

---

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
```

This creates an isolated Python environment so packages don't clash with other projects.

---

### Step 3 — Activate the virtual environment

```bash
# Mac / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

Your terminal prompt will change to show `(venv)` — this means the environment is active.

> ⚠️ You need to activate the venv **every time** you open a new terminal window.

---

### Step 4 — Install all dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs Streamlit, PyTorch, HuggingFace Transformers, spaCy, KeyBERT, and all other packages. Takes 3–5 minutes on first install.

---

### Step 5 — Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

This is the English language model used for lemmatization and POS tagging (word cloud filtering).

---

### Step 6 — (Optional) Set your Anthropic API key for LLM summaries

The framework generates AI-powered executive summaries using Claude. If you skip this, it falls back to a rule-based summary — everything else still works.

```bash
# Mac / Linux — add to your terminal session:
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# To make it permanent, add the line above to your ~/.zshrc or ~/.bash_profile
# then run: source ~/.zshrc
```

Get your API key from [console.anthropic.com](https://console.anthropic.com).

---

### Step 7 — Generate sample data (optional but recommended for first test)

```bash
python tests/sample_data/generate_sample.py
```

This creates `tests/sample_data/retail_reviews.csv` — a synthetic 500-row retail review dataset you can upload immediately to test the full pipeline.

---

### Step 8 — Launch the Streamlit app

```bash
streamlit run app.py
```

Streamlit will print something like:

```
  You can now view your Streamlit app in your browser.
  Local URL:  http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your browser should open automatically. If not, go to [http://localhost:8501](http://localhost:8501).

---

### Step 9 — Use the app

1. **Upload & Configure tab** — Upload `tests/sample_data/retail_reviews.csv`
2. The system auto-detects: `review_text` → text column, `rating` → score, `date` → date, `category` → category
3. Confirm the schema (or override any column)
4. Click **▶ Run Analysis**
   - First run: downloads ~500 MB of ML models (takes 3–5 min)
   - Subsequent runs: loads from cache (takes 30–60 sec)
5. Switch to **Overview** → see sentiment breakdown, emotion chart, and executive summary
6. **Deep Dive** → aspect heatmap, trend over time, representative quotes
7. **Word Clouds** → positive / neutral / negative side-by-side
8. **Export** → download CSV, Excel workbook, or PDF report

---

### Common issues and fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: spacy` | Run `pip install -r requirements.txt` with venv activated |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | Run `python -m spacy download en_core_web_sm` |
| App opens but analysis hangs | First run downloads models — wait 3–5 min, watch terminal for progress |
| `torch` install fails on Mac M1/M2 | Run `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| PDF export fails with "kaleido" error | Run `pip install kaleido` — needed for embedding Plotly charts in PDF |
| LLM summary shows fallback text | Set `ANTHROPIC_API_KEY` in your terminal session |

---

### Stopping the app

Press `Ctrl + C` in the terminal where Streamlit is running.

---

### Project structure quick reference

```
sentiment-framework/
├── app.py                    ← Streamlit UI (run this)
├── requirements.txt          ← All dependencies
├── config/settings.py        ← Change models, thresholds here
├── pipeline/
│   ├── ingest.py             ← File loading + schema inference
│   ├── preprocess.py         ← Cleaning + lemmatization
│   ├── orchestrator.py       ← Coordinates all stages
│   └── analyze/
│       ├── sentiment.py      ← RoBERTa polarity classifier
│       ├── emotion.py        ← DistilRoBERTa emotion detection
│       ├── aspect.py         ← KeyBERT aspect extraction
│       └── summarizer.py     ← LLM executive summary
├── pipeline/visualize/
│   ├── charts.py             ← All Plotly charts
│   └── wordcloud_gen.py      ← Per-sentiment word clouds
├── exporters/
│   ├── xlsx_exporter.py      ← Excel workbook export
│   └── pdf_exporter.py       ← PDF report export
└── tests/
    ├── test_ingest.py        ← Schema inference tests
    ├── test_sentiment.py     ← Model + logic tests
    └── sample_data/          ← Generated test CSV lives here
```
