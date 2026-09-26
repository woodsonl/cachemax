# cachemax sidecar

The mlx-lm precision path for [cachemax](../README.md). A Python subprocess the
Rust core invokes off the hot path, macOS/Apple Silicon only. Not required to run
cachemax or show a curve.

```bash
uv sync
uv run pytest
```
