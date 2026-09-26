"""Warm/cold TTFT discrimination — the mlx-lm sidecar's reported signal.

mlx-lm exposes no cache truth, so there is no hit-rate number. It does expose
per-token timing, so the sidecar reports per-turn TTFT and the warm/cold ratio.
It does NOT assert a speedup threshold; reporting the number is the contract.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .engine import Engine


@dataclass
class Discrimination:
    warm_ms: list[float]
    cold_ms: list[float]

    @property
    def ratios(self) -> list[float]:
        """Per-turn cold/warm ratio: >1 means warm was faster."""
        return [c / w if w else 0.0 for c, w in zip(self.cold_ms, self.warm_ms)]

    @property
    def median_ratio(self) -> float:
        return statistics.median(self.ratios) if self.ratios else 0.0


def _prompt(size: int) -> list[dict]:
    filler = "The quick brown fox jumps over the lazy dog. " * (size // 45)
    return [
        {"role": "system", "content": filler},
        {"role": "user", "content": "Continue."},
    ]


def warm_cold_discrimination(engine: Engine, turns: int = 5, size: int = 600) -> Discrimination:
    """Measure warm vs cold TTFT over `turns` comparable-length prompts.

    Cold: a fresh, unique prompt each turn. Warm: the same prompt re-sent, so
    its prefix would be resident under any real cache. Both are the same length,
    so the recorded difference is the engine's, not the prompt size's.
    """
    warm_ms: list[float] = []
    cold_ms: list[float] = []
    for i in range(turns):
        # Cold: a distinct prompt whose prefix has not been seen.
        cold = _prompt(size)
        cold[0]["content"] = f"variant {i} " + cold[0]["content"]
        cold_ms.append(engine.generate(cold, max_tokens=16).ttft_ms)

        # Warm: re-send the cold prompt unchanged; its prefix is now resident.
        warm_ms.append(engine.generate(cold, max_tokens=16).ttft_ms)
    return Discrimination(warm_ms=warm_ms, cold_ms=cold_ms)

