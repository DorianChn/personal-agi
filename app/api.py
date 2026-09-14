"""FastAPI 服务：提供 OpenAI 兼容的 /v1/chat/completions 接口。"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import yaml  # noqa: E402
from src.inference.client import get_client  # noqa: E402

app = FastAPI(title="Personal AGI API")

with (PROJECT_ROOT / "config" / "config.yaml").open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
gen = cfg.get("generation", {})
client = get_client(cfg)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    messages = [m.model_dump() for m in req.messages]
    resp = client.chat(
        messages,
        temperature=(
            req.temperature if req.temperature is not None else gen.get("temperature", 0.7)
        ),
        max_tokens=(
            req.max_tokens if req.max_tokens is not None else gen.get("max_tokens", 1024)
        ),
    )
    return {
        "id": "chatcmpl-personal-agi",
        "object": "chat.completion",
        "model": client.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": resp}}],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": client.model}
