"""The mlx-lm precision path for the cachemax proxy.

The proxy talks to this sidecar as an ordinary OpenAI-compatible upstream:
`cachemax serve --backend mlxlm --upstream-url http://127.0.0.1:8080/v1`.
Because mlx-lm exposes no cache truth, the sidecar emits no `cached_tokens`
field; the proxy's mlxlm adapter records `no_cache_truth` and the dashboard
renders `—`. The value this path adds is TTFT warm/cold discrimination on a
local Apple Silicon engine.
"""

__version__ = "0.2.0"
