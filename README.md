# cache-max

Universal cache-maximizing inference proxy. An OpenAI-compatible daemon that
normalizes prompt formatting so prefix caches actually hit, with prefix-affinity
routing and hit-rate/cost observability.

See [docs/designs/cache-max-proxy.md](docs/designs/cache-max-proxy.md) for the
design.

## Status

v0 skeleton. Not functional yet.

## Development

```bash
uv sync
uv run pytest
uv run cache-max  # placeholder entry point
```
