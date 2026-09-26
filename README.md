# cachemax

An OpenAI-compatible proxy that sits in front of an LLM endpoint, measures how
much of each request's re-sent history was served from the provider's prompt
cache, and reports the cost and speed consequence on a live dashboard.

It measures first. It does not repair yet.

- **Cloud (typical):** point an app or agent at the proxy instead of directly at
  OpenAI, Anthropic, or OpenRouter. See how much prompt-cache reuse you get and
  what it costs.
- **Local (minority):** run llama.cpp, mlx-lm, or vLLM and watch TTFT collapse
  as the cache warms. (mlx-lm is macOS/Apple Silicon only.)

Both are first-class targets.

See [docs/designs/cachemax-measurement-core.md](docs/designs/cachemax-measurement-core.md)
for the design.

## Status

Spec complete, zero implementation. The core is Rust (axum/tokio), built at the
repo root. The Python package under `sidecar/` is the mlx-lm precision path
(subprocess, off the hot path) plus dev tooling; it is not required to run
cachemax.

## Layout

- repo root — the Rust core (`Cargo.toml`, `src/*.rs`), the dashboard, the spec link above
- `sidecar/` — the Python mlx-lm sidecar (`cache_max/`), its tests, and its uv project

## Development

```bash
# Rust core (not yet scaffolded)
cargo test

# Python sidecar
uv sync --project sidecar
uv run --project sidecar pytest
uv run --project sidecar cachemax-sidecar  # placeholder entry point
```
