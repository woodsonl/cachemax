"""OpenAI-compatible HTTP surface backed by an `Engine`.

Only the fields the cachemax proxy reads are produced. The response carries no
`cached_tokens`: mlx-lm exposes no cache truth, so the proxy records
`no_cache_truth` rather than a fabricated figure.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from .engine import Engine


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = None


def build_app(engine: Engine) -> FastAPI:
    """Mount the OpenAI-compatible routes over a given engine."""
    app = FastAPI(title="cachemax-sidecar", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest) -> dict[str, Any]:
        completion = engine.generate(req.messages, req.max_tokens or 256)
        total = completion.prompt_tokens + completion.completion_tokens
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion.text},
                    "finish_reason": "stop",
                }
            ],
            # No prompt_tokens_details: mlx-lm has no cache truth to report.
            "usage": {
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": total,
            },
        }

    return app
