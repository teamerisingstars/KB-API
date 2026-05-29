FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/opt/nltk_data

# System deps needed only at install time — git for fetching demo corpora
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK corpora at build time so the container starts instantly
RUN python -m nltk.downloader -d "$NLTK_DATA" \
        punkt punkt_tab \
        averaged_perceptron_tagger averaged_perceptron_tagger_eng \
        wordnet stopwords omw-1.4

# Application code
COPY app ./app
COPY download_nltk_data.py ./
COPY action_validate.py ./

# Demo corpus — fetched at build time so the public demo has something real to query.
# Sparse-checkout pulls only the docs/ folder of each repo, keeping the image small.
RUN mkdir -p /app/knowledge \
 # --- FastAPI ---
 && git clone --depth 1 --filter=blob:none --sparse https://github.com/tiangolo/fastapi.git /tmp/fastapi \
 && (cd /tmp/fastapi && git sparse-checkout set docs/en/docs) \
 && mkdir -p /app/knowledge/fastapi \
 && cp -r /tmp/fastapi/docs/en/docs/. /app/knowledge/fastapi/ \
 && rm -rf /tmp/fastapi \
 # --- Pydantic ---
 && git clone --depth 1 --filter=blob:none --sparse https://github.com/pydantic/pydantic.git /tmp/pydantic \
 && (cd /tmp/pydantic && git sparse-checkout set docs) \
 && mkdir -p /app/knowledge/pydantic \
 && cp -r /tmp/pydantic/docs/. /app/knowledge/pydantic/ \
 && rm -rf /tmp/pydantic \
 # --- Starlette ---
 && git clone --depth 1 --filter=blob:none --sparse https://github.com/encode/starlette.git /tmp/starlette \
 && (cd /tmp/starlette && git sparse-checkout set docs) \
 && mkdir -p /app/knowledge/starlette \
 && cp -r /tmp/starlette/docs/. /app/knowledge/starlette/ \
 && rm -rf /tmp/starlette \
 # Drop any non-markdown junk that snuck in (images, CSS, .html, etc.)
 && find /app/knowledge -type f ! -name "*.md" -delete \
 && echo "Demo corpus: $(find /app/knowledge -name '*.md' | wc -l) markdown files"

EXPOSE 8000

# Tiny healthcheck — kb api is fast; if /health fails twice we're dead
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=2 \
  CMD python -c "import urllib.request, sys; \
                 sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
