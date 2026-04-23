#!/usr/bin/env bash
set -euo pipefail

# Render provides $PORT for web services.
STREAMLIT_PORT="${PORT:-8501}"

# Persist ChromaDB on a Render disk if configured; otherwise use the repo-local default.
CHROMA_DB_DIR="${CHROMA_DB_DIR:-prototype/.chroma_db}"
export CHROMA_DB_DIR

# Optional: skip expensive PDF ingestion on small instances.
SKIP_PDF_INGESTION="${SKIP_PDF_INGESTION:-0}"
export SKIP_PDF_INGESTION

# Ingest on first boot (or if the DB directory is empty). Can be disabled via AUTO_INGEST=0.
AUTO_INGEST="${AUTO_INGEST:-1}"
if [[ "${AUTO_INGEST}" == "1" ]]; then
  mkdir -p "${CHROMA_DB_DIR}"
  if [[ -z "$(ls -A "${CHROMA_DB_DIR}" 2>/dev/null || true)" ]]; then
    echo "[render] ChromaDB directory is empty; running ingestion..."
    python -m prototype.ingest || python prototype/ingest.py
  else
    echo "[render] ChromaDB directory already has data; skipping ingestion."
  fi
else
  echo "[render] AUTO_INGEST=0; skipping ingestion."
fi

exec python -m streamlit run prototype/app.py \
  --server.address 0.0.0.0 \
  --server.port "${STREAMLIT_PORT}" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false

