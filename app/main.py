"""FastAPI app: serves BNP-themed chat UI and an SSE endpoint backed by the agent loop."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app import agent, db

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"

app = FastAPI(title="BNP × Mistral assistant (prototype)")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat")
def chat(req: ChatRequest) -> EventSourceResponse:
    history: list[dict[str, Any]] = [m.model_dump() for m in req.messages]

    def gen():
        for event in agent.run_turn(history):
            yield {"event": event["type"], "data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(gen())


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model": agent.MODEL}


@app.get("/state")
def state() -> dict[str, Any]:
    """Live mock-bank state for the side panel. Refreshed by the UI after every agent turn."""
    client = db.get_client()
    return {
        "client": {"first_name": client["first_name"], "last_name": client["last_name"]},
        "accounts": db.list_accounts(),
        "cards": db.list_cards(),
        "products": db.list_products(),
    }
