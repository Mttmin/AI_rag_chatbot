#!/usr/bin/env bash
# One-shot launcher: ensures venv + deps + Ollama models + seed, then starts the app.
# Usage:  ./run.sh              # runs on http://localhost:8000
#         PORT=9000 ./run.sh    # custom port
#         MISTRAL_MODEL=ministral-3:14b ./run.sh
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-5173}"
HOST="${HOST:-0.0.0.0}"
MISTRAL_MODEL="${MISTRAL_MODEL:-ministral-3:8b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
export MISTRAL_MODEL EMBED_MODEL

green() { printf "\033[1;32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[1;33m%s\033[0m\n" "$*"; }
red() { printf "\033[1;31m%s\033[0m\n" "$*"; }

# --- 1. Ollama running? ---
if ! command -v ollama >/dev/null 2>&1; then
  red "Ollama not installed. See https://ollama.com/download"
  exit 1
fi
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  yellow "Ollama daemon not responding on :11434 — starting it…"
  (ollama serve >/tmp/ollama.log 2>&1 &)
  for i in {1..20}; do
    sleep 0.5
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then break; fi
  done
  if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    red "Could not start Ollama. Check /tmp/ollama.log"
    exit 1
  fi
fi
green "✓ Ollama up"

# 2. Required models pulled?
ensure_model() {
  local m="$1"
  # ollama list shows "name:tag" — implicit "latest" tag is rendered as ":latest"
  local needle="$m"
  [[ "$m" != *:* ]] && needle="${m}:latest"
  if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$needle"; then
    green "✓ Model $m present"
  else
    yellow "Pulling $m (this may take a while)…"
    ollama pull "$m"
  fi
}
ensure_model "$MISTRAL_MODEL"
ensure_model "$EMBED_MODEL"

# 3. Python env + deps
# Prefer an active conda env; otherwise fall back to a project-local .venv.
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  PY="$CONDA_PREFIX/bin/python"
  PIP="$CONDA_PREFIX/bin/pip"
  green "✓ Using conda env: ${CONDA_DEFAULT_ENV:-base} ($CONDA_PREFIX)"
else
  if [ ! -d .venv ]; then
    yellow "Creating .venv…"
    python3 -m venv .venv
  fi
  PY=.venv/bin/python
  PIP=.venv/bin/pip
  green "✓ Using .venv"
fi
if ! "$PY" -c "import fastapi, ollama, chromadb, sse_starlette" 2>/dev/null; then
  yellow "Installing Python dependencies…"
  "$PIP" install -q --upgrade pip
  "$PIP" install -q -r requirements.txt
fi
green "✓ Python dependencies ready"

# 4. Seed DB + Chroma if missing
if [ ! -f data/bank.sqlite ] || [ ! -d data/chroma ] || [ -z "$(ls -A data/chroma 2>/dev/null)" ]; then
  yellow "Seeding mock bank + RAG index…"
  "$PY" -m app.seed
fi
green "✓ Data seeded"

# 5. Launch
green "Starting BNP × Mistral assistant on http://${HOST}:${PORT}"
echo "    Model:      $MISTRAL_MODEL"
echo "    Embeddings: $EMBED_MODEL"
echo "    Stop with Ctrl-C"
echo
exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
