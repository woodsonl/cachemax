# cache-maxing

An OpenAI-compatible proxy that sits in front of an LLM endpoint, measures how
much of each request's re-sent history was served from the provider's prompt
cache, and reports the cost and speed consequence on a live dashboard.

It measures first. It does not repair yet.

- **Cloud (typical):** point an app or agent at the proxy instead of directly at
  OpenAI, Anthropic, or OpenRouter. See how much prompt-cache reuse you get and
  what it costs.
- **Local (minority):** run llama.cpp, mlx-lm, or vLLM and watch TTFT collapse
  as the cache warms.

Both are first-class targets.

See [docs/designs/cache-maxing-measurement-core.md](docs/designs/cache-maxing-measurement-core.md)
for the design.

## Status

Spec complete, zero implementation. The core will be Rust (axum/tokio); the repo
scaffolding below is Python, retained for the mlx-lm precision sidecar and dev
tooling.

## Development

```bash
uv sync
uv run pytest
uv run cache-maxing  # placeholder entry point
```
