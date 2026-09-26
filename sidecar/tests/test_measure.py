"""Unit tests for the warm/cold measurement math (no engine, any platform)."""

import threading
import time

from cache_max.engine import MlxEngine
from cache_max.measure import Discrimination


def test_ratios_are_cold_over_warm():
    d = Discrimination(warm_ms=[100.0, 50.0], cold_ms=[200.0, 50.0])
    assert d.ratios == [2.0, 1.0]


def test_median_ratio_odd_and_even():
    assert Discrimination(warm_ms=[1, 1, 1], cold_ms=[1, 4, 9]).median_ratio == 4.0
    assert Discrimination(warm_ms=[1, 1, 1, 1], cold_ms=[1, 2, 3, 4]).median_ratio == 2.5


def test_zero_warm_is_guarded():
    d = Discrimination(warm_ms=[0.0], cold_ms=[10.0])
    assert d.ratios == [0.0]


def test_lazy_load_runs_exactly_once_under_concurrency():
    # FastAPI serves the sync handler from a threadpool, so concurrent first
    # requests must not each trigger a model load. `load()` is idempotent and
    # lock-guarded.
    engine = MlxEngine(model_name="unused")
    loads = []

    def fake_load_locked() -> None:
        if engine._model is not None:  # mirror the real idempotence guard
            return
        loads.append(1)
        time.sleep(0.05)  # widen the race window
        engine._model = object()  # mark loaded

    engine._load_locked = fake_load_locked  # type: ignore[method-assign]
    threads = [threading.Thread(target=engine.load) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(loads) == 1, f"model loaded {len(loads)} times, expected 1"

