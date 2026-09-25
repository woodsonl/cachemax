# cache-maxing

Universal cache-maxingimizing inference proxy. An OpenAI-compatible daemon that
normalizes prompt formatting so prefix caches actually hit, with prefix-affinity
routing and hit-rate/cost observability.

See [docs/designs/cache-maxing-proxy.md](docs/designs/cache-maxing-proxy.md) for the
design.

## Status

v0 skeleton. Not functional yet.

## Development

```bash
uv sync
uv run pytest
uv run cache-maxing  # placeholder entry point
```
