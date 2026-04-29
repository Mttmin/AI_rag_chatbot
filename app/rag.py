"""RAG over contract markdown via local Chroma + Ollama embeddings."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
import ollama

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = ROOT / "data" / "contracts"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "contracts"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


def _client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _embed(texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        r = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        out.append(r["embedding"])
    return out


def _chunk(text: str, target_chars: int = 500, overlap: int = 80) -> list[str]:
    """Paragraph-aware chunking. Markdown sections kept together when small enough."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + len(p) + 2 <= target_chars:
            buf = f"{buf}\n\n{p}"
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = f"{tail}\n\n{p}" if tail else p
    if buf:
        chunks.append(buf)
    return chunks


def build_index() -> int:
    client = _client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    docs: list[str] = []
    metas: list[dict[str, Any]] = []
    ids: list[str] = []
    for path in sorted(CONTRACTS_DIR.glob("*.md")):
        text = path.read_text()
        for i, chunk in enumerate(_chunk(text)):
            # Prefix source name so the embedding picks up document identity.
            embedded = f"[{path.stem}] {chunk}"
            docs.append(embedded)
            metas.append({"source": path.name, "chunk": i})
            ids.append(f"{path.stem}__{i}")

    if not docs:
        return 0
    embeddings = _embed(docs)
    coll.add(documents=docs, metadatas=metas, ids=ids, embeddings=embeddings)
    return len(docs)


_STOP = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "mon", "ma",
    "mes", "sur", "dans", "pour", "que", "qui", "ce", "se", "au", "aux", "est",
    "the", "a", "an", "of", "to", "in", "is", "my",
}


def _tokens(s: str) -> set[str]:
    import re

    return {t for t in re.findall(r"[a-zàâäéèêëïîôöùûüç]+", s.lower()) if t not in _STOP and len(t) > 2}


def retrieve(query: str, k: int = 4) -> list[dict[str, Any]]:
    if not query or not query.strip():
        return []
    """Hybrid retrieval: vector top-(3k) + lexical re-rank.

    nomic-embed-text on French is uneven; a small lexical boost on shared
    content tokens (and on filename match) pulls the right document up
    when the query mentions the product by name (« Esprit Libre »,
    « Livret A », etc.).
    """
    client = _client()
    try:
        coll = client.get_collection(COLLECTION)
    except Exception:
        return []
    qemb = _embed([query])[0]
    pool = max(k * 3, 12)
    qtoks = _tokens(query)

    # Vector pool
    res = coll.query(query_embeddings=[qemb], n_results=pool)
    seen: dict[str, dict[str, Any]] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        vec = 1 - dist
        seen[f"{meta['source']}#{meta['chunk']}"] = {
            "text": doc, "source": meta["source"], "vec": vec
        }

    # Lexical pass: pull ALL chunks whose source filename matches a query token.
    # Catches "Esprit Libre", "Livret A" etc. when vector recall buries them.
    all_chunks = coll.get(include=["documents", "metadatas"])
    for doc, meta in zip(all_chunks["documents"], all_chunks["metadatas"]):
        key = f"{meta['source']}#{meta['chunk']}"
        if any(t in meta["source"].lower() for t in qtoks):
            seen.setdefault(key, {"text": doc, "source": meta["source"], "vec": 0.0})

    # Score
    candidates: list[dict[str, Any]] = []
    for c in seen.values():
        dtoks = _tokens(c["text"])
        overlap = len(qtoks & dtoks)
        src_match = 1.0 if any(t in c["source"].lower() for t in qtoks) else 0.0
        c["score"] = c["vec"] + 0.05 * overlap + 0.6 * src_match
        candidates.append(c)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:k]
