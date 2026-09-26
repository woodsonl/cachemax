"""Engine abstraction for the cachemax sidecar.

An `Engine` turns an OpenAI-shaped chat request into a chat completion and, for
the real path, exposes the per-token TTFT that the warm/cold discrimination
depends on. Two implementations:

- `MlxEngine` — real mlx-lm inference (macOS / Apple Silicon only).
- `FakeEngine` — deterministic, in-process, for contract tests on any platform.

The server never imports mlx-lm at module scope; `MlxEngine` loads it lazily so
the fake path and the whole test suite run where mlx-lm cannot be installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Completion:
    """One chat completion plus the timing the proxy measures."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float


class Engine(Protocol):
    """The minimal surface the server needs from a backend engine."""

    def generate(self, messages: list[dict], max_tokens: int) -> Completion:
        """Produce a completion for the OpenAI-shaped `messages` list."""
        ...


@dataclass
class FakeEngine:
    """Deterministic engine for contract tests.

    Echoes a canned reply and reports a fixed token split so tests can assert
    the wire shape without a model. `ttft_ms` is constant; the real warm/cold
    behaviour is a property of mlx-lm, not of this stand-in.
    """

    reply: str = "ok"
    ttft_ms: float = 1.0

    def generate(self, messages: list[dict], max_tokens: int) -> Completion:
        prompt = sum(len(str(m.get("content", ""))) for m in messages)
        return Completion(
            text=self.reply,
            prompt_tokens=max(1, prompt // 4),
            completion_tokens=max(1, len(self.reply) // 4),
            ttft_ms=self.ttft_ms,
        )


@dataclass
class MlxEngine:
    """Real mlx-lm inference. Loads the model once and reuses it.

    The model load is deferred to `load()` so importing this module never
    requires mlx-lm. macOS / Apple Silicon only; construction does not touch mlx.
    """

    model_name: str
    _model: object = field(default=None, init=False)
    _tokenizer: object = field(default=None, init=False)

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_lm import load  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - platform dependent
            raise RuntimeError(
                "mlx-lm is not installed. The mlx-lm backend is macOS / Apple "
                "Silicon only; install it with `uv pip install mlx-lm`."
            ) from e
        self._model, self._tokenizer = load(self.model_name)

    def generate(self, messages: list[dict], max_tokens: int) -> Completion:
        self.load()
        from mlx_lm import stream_generate  # type: ignore[import-not-found]

        prompt = _apply_chat_template(self._tokenizer, messages)

        # stream_generate yields a GenerationResponse per token; the wall-clock
        # of the first yield is the TTFT the warm/cold discrimination reads.
        start = time.perf_counter()
        ttft_ms: float | None = None
        pieces: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        for response in stream_generate(
            self._model, self._tokenizer, prompt=prompt, max_tokens=max_tokens
        ):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            pieces.append(response.text)
            prompt_tokens = response.prompt_tokens
            completion_tokens = response.generation_tokens
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return Completion(
            text="".join(pieces),
            prompt_tokens=prompt_tokens or len(self._tokenizer.encode(prompt)),
            completion_tokens=max(1, completion_tokens),
            ttft_ms=ttft_ms if ttft_ms is not None else elapsed_ms,
        )


def _apply_chat_template(tokenizer: object, messages: list[dict]) -> str:
    """Render `messages` with the model's chat template when it has one."""
    if apply := getattr(tokenizer, "apply_chat_template", None):
        return apply(messages, add_generation_prompt=True, tokenize=False)
    return "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)
