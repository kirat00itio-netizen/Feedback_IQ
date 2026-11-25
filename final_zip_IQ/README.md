# FeedbackIQ — Amazon Review Intelligence and Chatbot

FeedbackIQ is a Python/Streamlit application for exploring Amazon review feedback through
sentiment summaries, aspect-aware recommendations, product comparisons, and a
product-isolated, FAISS-backed conversational review assistant.

This is a **GitHub-ready trimmed export** of the supplied project. It excludes local Git
history, secret environment files, temporary caches, large training datasets, and
transformer weight files. Source code, relevant scripts and tests, model comparison results,
notebooks (mostly cleared of execution output), and three pre-existing Parquet files are kept.

## Quick start (on a computer)

1. Install Python 3.10–3.12 (ideally Python 3.11).
2. Clone or download the repository.
3. In its root directory, run:

   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app/streamlit_app.py
   ```

4. On first startup, Sentence Transformers may download `all-MiniLM-L6-v2`.
   The optional LLM backends need separate setup (and, if applicable, private API keys).
   No private `.env` file is included; use `.env.example` as a reference.

## What is included and what changes

- `app/streamlit_app.py` — web interface; start with `streamlit run app/streamlit_app.py`.
- `src/` — preprocessing, analytics, retrieval, chatbot, recommender, and model code.
- `scripts/` — experiment and pipeline scripts.
- `tests/` — unit tests (some require additional models or access to original/raw data).
- `results/` — lightweight saved plots, metrics, and evaluation reports.
- `notebooks/` — experiment notebooks, with large notebook outputs cleared for GitHub.
- `data/processed/combined_processed.parquet` — 92,662 cleaned reviews (rating-derived labels).
- `data/processed/products.parquet` — product metadata table from the uploaded archive.
- `data/processed/product_insights.parquet` — saved product summary table.
- `data/aspect_annotations.csv` — saved aspect pseudo-label evaluation data.

**Important:** the uploaded archive does not contain `reviews_with_predictions.parquet`.
The current `ReviewRetriever` falls back to the stored `combined_processed.parquet`
and will use its rating-derived `sentiment` field when predicted sentiment is absent.
Thus a clean deployment is not identical to a fully computed RoBERTa inference run.
To reproduce model-based predictions, you must execute the relevant analytics pipeline
and download model weights using the documented dependencies.

The supplied archive also contains significantly more records than the 706-review
product-linked evaluation subset described in some research-paper versions. Do not
assume these bundled Parquet files reproduce that evaluation exactly.

## Publish on GitHub

1. Unzip `FeedbackIQ_GitHub_Ready_Under25MB.zip` on your computer.
2. Create a new **empty repository** at github.com.
3. Choose **Add file → Upload files** and upload **the extracted contents**
   (folders `app`, `src`, `scripts`, `data`, etc., and README/requirements).
4. Commit the uploaded files.

Uploading a ZIP by itself to a repo stores a ZIP file: GitHub does **not** unpack it
into a runnable project. The 25-MiB web-upload limit is **per individual file**, not
a requirement that the whole repository stay under 25 MiB.

GitHub stores the source; it is not itself a hosted Streamlit app. To make a public
web app, deploy the repository with a compatible Streamlit hosting service and set
its main file to `app/streamlit_app.py`. Hosting needs to install dependencies and
download model checkpoints at startup; disk, RAM and CPU limits may affect success.

## Data and model provenance

Amazon Reviews 2023 (McAuley Lab):
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

The application supports optional external AI APIs. Keep all API tokens in
host-specific secrets or environment variables — never commit `.env` or secrets.

## Limits of the packaged demo

The package has not been end-to-end run in this packaging environment (the full ML
dependency stack and checkpoint downloads are unavailable here). Python source
syntax and ZIP integrity were checked. A working deployment still requires the
dependencies and sufficient runtime resources.
