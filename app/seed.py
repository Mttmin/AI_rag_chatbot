"""Seed SQLite mock-bank data and Chroma RAG index. Idempotent."""
from __future__ import annotations

from app import db, rag


def main() -> None:
    print("[seed] resetting SQLite mock bank…")
    db.reset_and_seed()
    print(f"[seed] SQLite ready at {db.DB_PATH}")

    print("[seed] building Chroma RAG index (Ollama embeddings)…")
    n = rag.build_index()
    print(f"[seed] indexed {n} chunks at {rag.CHROMA_DIR}")


if __name__ == "__main__":
    main()
