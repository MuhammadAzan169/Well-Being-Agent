# ── WellBeing Agent — Backend (Render-ready, free-plan sized) ─────────
#
# Optional: Render's *native Python* runtime (see render.yaml) is the lighter
# path on the free plan. Use this Dockerfile if you prefer a container, or to
# run the backend locally in isolation:
#   docker build -t wellbeing-api . && docker run -p 8000:8000 --env-file .env wellbeing-api
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    # fastembed caches its ONNX model here — baked into the image at build time
    # so a cold start never has to download anything.
    FASTEMBED_CACHE_PATH=/app/var/fastembed-cache \
    # Prevent tokenizer / BLAS thread storms on single-core containers.
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# curl is used by the healthcheck. No build toolchain is needed: every wheel
# in requirements.txt ships prebuilt for linux/amd64.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, dataset, and the pre-built vector index.
COPY . .

# Writable runtime dirs (cache, conversation log, recordings, model cache).
RUN mkdir -p var/cache var/recorded_audio var/fastembed-cache

# Pre-download the embedding model into the image. The vector index itself is
# already committed under data/cancer_index_store, so nothing is rebuilt here.
RUN python -m scripts.warm_cache

EXPOSE 8000

# The RAG system loads in a background thread, so /health answers immediately;
# start-period just gives uvicorn time to bind.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f "http://localhost:${PORT}/health" || exit 1

# Honor the platform-provided $PORT (defaults to 8000).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
