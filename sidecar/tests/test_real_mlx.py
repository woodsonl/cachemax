"""Opt-in real mlx-lm check: warm/cold TTFT discrimination is measured.

Skipped unless `CACHEMAX_MLX_REAL=1` is set, so the default suite stays fast
and platform-independent. Run explicitly on an Apple Silicon host:

    CACHEMAX_MLX_REAL=1 uv run pytest tests/test_real_mlx.py -s

It downloads a small model on first run. The assertion is that the measurement
is produced and well-formed, not that a warm speedup exists: real mlx-lm
clusters around 1.0x at these sizes (see docs/designs/cachemax-measurement-core.md).
"""

import os

import pytest

from cache_max.engine import MlxEngine
from cache_max.measure import warm_cold_discrimination

pytestmark = pytest.mark.skipif(
    os.environ.get("CACHEMAX_MLX_REAL") != "1",
    reason="set CACHEMAX_MLX_REAL=1 to run the real mlx-lm check",
)


def test_warm_cold_ttft_is_measured_and_well_formed():
    model = os.environ.get("CACHEMAX_MLX_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    engine = MlxEngine(model_name=model)
    result = warm_cold_discrimination(engine, turns=5)

    assert len(result.warm_ms) == 5
    assert len(result.cold_ms) == 5
    assert all(t > 0 for t in result.warm_ms), "every warm TTFT is a real, positive duration"
    assert all(t > 0 for t in result.cold_ms), "every cold TTFT is a real, positive duration"
    assert len(result.ratios) == 5
    print(f"warm={result.warm_ms}\ncold={result.cold_ms}\nmedian ratio={result.median_ratio:.2f}x")
