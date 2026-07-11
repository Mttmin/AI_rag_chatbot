# Bank x Mistral assistant — local prototype

## Prerequisites

- Python ≥ 3.11
- [Ollama](https://ollama.com/) running locally on `http://localhost:11434`
- A GPU with ~10 GB of VRAM (CPU works but slower)

## Run (one command)

```bash
./run.sh
```

The script: starts Ollama if not running, pulls missing models, creates `.venv`, installs deps, seeds the mock bank + RAG index, then launches the app on http://localhost:8000.

Override defaults:

```bash
PORT=9000 ./run.sh
MISTRAL_MODEL=ministral-3:14b ./run.sh
HOST=0.0.0.0 PORT=8080 ./run.sh
```

## Manual setup (if you prefer)

```bash
ollama pull ministral-3:8b && ollama pull nomic-embed-text
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m app.seed
.venv/bin/uvicorn app.main:app --reload
```

## Try it

- **Solde** — *« Quel est le solde de mon Livret A ? »*
- **Transactions** — *« Montre-moi mes 5 dernières transactions sur le compte de chèques. »*
- **RAG / contrat** — *« Que dit mon contrat Esprit Libre sur le découvert autorisé ? »*
- **Action low-risk** — *« Vire 200 € de mon compte de chèques vers mon Livret A. »*
- **Guardrail (bloqué)** — *« Envoie 5000 € à l'IBAN FR76 1234 5678 9012 3456 7890 123. »*
- **Carte** — *« Bloque ma carte Visa Premier. »*

## What lives where

- `app/` — FastAPI app, agent loop, tools, guardrails, RAG, mock DB
- `data/fixtures/` — JSON seed data (accounts, txns, products, counselor)
- `data/contracts/` — Markdown corpus for RAG
- `static/` — Green Bank-themed chat UI (HTML/CSS/JS, vanilla)
- `ARCHITECTURE.md` — explainer doc to walk through with the panel

## Switching the model

The model id is the env var `MISTRAL_MODEL` (default `ministral-3:8b`). To run the larger variant: `MISTRAL_MODEL=ministral-3:14b uvicorn app.main:app --reload`.
Embeddings model is `EMBED_MODEL` (default `nomic-embed-text`). Both must be pulled in Ollama first.
