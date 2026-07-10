# ─────────────────────────────────────────────────────────────
# Zintoo AI — container image
# Targets Hugging Face Spaces (Docker SDK), but works anywhere:
# Railway, Fly.io, Cloud Run, Render, a VPS.
#
# HF Spaces free tier: 2 vCPU / 16 GB RAM — comfortably fits
# onnxruntime + the model + the embedding index.
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

# HF Spaces routes traffic to 7860. Other hosts inject $PORT.
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# HF Spaces runs as a non-root user (uid 1000). Create it and own the app dir,
# otherwise the SQLite WAL files and the runtime dir are unwritable at boot.
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install deps first so Docker caches this layer across code changes.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

# Writable state. On HF Spaces the filesystem is ephemeral: the DB re-seeds and
# the catalogue re-imports on every cold start (see entrypoint below).
RUN mkdir -p /app/runtime && chown -R appuser:appuser /app/runtime
USER appuser

ENV ZINTOO_ENV=production \
    ZINTOO_JSON_LOGS=true \
    ZINTOO_DATA_DIR=/app/runtime \
    ZINTOO_MODELS_DIR=/app/models_artifacts \
    ZINTOO_VISION_ENABLED=true \
    ZINTOO_VISION_PRELOAD=true \
    ZINTOO_VISION_THREADS=2

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,os,sys; \
      sys.exit(0 if urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/readiness', timeout=4).status==200 else 1)"

# One worker: the SSE fan-out, the event bus and the ORT session are in-process
# and are not shared across workers. Scale vertically, not with --workers.
CMD ["sh", "-c", "python -m scripts.bootstrap && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 65"]
