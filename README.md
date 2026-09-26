# cachemax

An OpenAI-compatible proxy that sits in front of an LLM endpoint, measures how
much of each request's re-sent history was served from the provider's prompt
cache, and reports the cost and speed consequence on a live dashboard.

It measures first. It does not repair yet.

- **Cloud (typical):** point an app or agent at the proxy instead of directly at
  OpenAI, Anthropic, or OpenRouter. See how much prompt-cache reuse you get and
  what it costs.
- **Local (minority):** run llama.cpp, vLLM, or mlx-lm and watch TTFT collapse
  as the cache warms. (mlx-lm is macOS/Apple Silicon only.)

Both are first-class targets.

## Install

```bash
cargo install --git https://github.com/woodsonl/cachemax
```

Or from a checkout: `cargo install --path .`. This builds a single binary named
`cachemax` — no runtime dependencies, no separate dashboard server.

## First run

```bash
# Point cachemax at your provider (OpenAI shown; see OpenRouter below).
cachemax serve --upstream-url https://api.openai.com/v1
```

It binds `127.0.0.1:8787` by default. You now have two things on that port:

- the **proxy** — an OpenAI-compatible endpoint at
  `http://127.0.0.1:8787/v1/chat/completions`
- the **dashboard** — open **http://127.0.0.1:8787/** in a browser

Point your client or agent at the proxy instead of the provider, send a turn or
two, and watch the dashboard. The first turn is cold (it establishes the cache);
the payoff lands on the next turn. There is no demo command — the moment comes
from your real traffic.

### OpenRouter

Same adapter, different endpoint:

```bash
cachemax serve --upstream-url https://openrouter.ai/api/v1
```

### Local engines

```bash
cachemax serve --backend llamacpp --upstream-url http://127.0.0.1:8080/v1
cachemax serve --backend vllm     --upstream-url http://127.0.0.1:8000/v1
cachemax serve --backend mlxlm    --upstream-url http://127.0.0.1:8080/v1  # macOS only
```

## Point a client at it

Any OpenAI-compatible client works: change its `base_url` to the proxy. Your
provider API key is passed through untouched and never stored or logged.

**OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8787/v1",   # <- the only change
    api_key="sk-...",                       # your real key, forwarded as-is
)

# A multi-turn conversation is what warms the cache. Re-send the history each
# turn, as any chat client does — cachemax measures what the provider reuses.
messages = [{"role": "user", "content": "Summarize the CAC benchmark in one line."}]
for _ in range(3):
    resp = client.chat.completions.create(model="gpt-4o", messages=messages)
    messages.append(resp.choices[0].message)
    messages.append({"role": "user", "content": "Say more."})
```

**Anthropic** via an OpenAI-compatible shim, or point an Anthropic SDK's
`base_url` at the proxy with `--backend anthropic`. The rule is the same: swap
the host, keep your key.

### One named agent, worked example

Any agent that takes a `base_url` works. Here is OpenClaude-style config:

```jsonc
{
  "model": "gpt-4o",
  "base_url": "http://127.0.0.1:8787/v1",
  "api_key": "dummy-key-accepted-and-forwarded",
  "max_tokens": 1024
}
```

Start the proxy, launch the agent, and ask it something with a follow-up. The
status strip flips to `● live` on the first request; the hero fills in the cost
saved on the first warm turn. If the agent requires a non-empty key, a dummy is
fine — cachemax forwards it and the provider decides.

## Export

One artifact, two triggers, byte-identical output:

```bash
cachemax export                       # writes ./cachemax-<session>.jsonl
cachemax export --out ./my-session.jsonl
```

or click **export** on the dashboard to download the same JSONL. It is
metrics-only — never message content.

```json
{"session_id":1,"turn":1,"status":"complete","source":"provider_reported","ttft_ms":120.0,"cached_tokens":1020,"cache_written_tokens":0,"resent_history_tokens":1550,"billed_input_tokens":1750,"broke_prefix":false,"cost_usd":0.02,"cost_saved_usd":0.01}
```

## Commands

| Command | What it does |
|---|---|
| `cachemax serve` | Run the proxy + dashboard (default). |
| `cachemax check` | Check the upstream is reachable; exit non-zero if not. |
| `cachemax export` | Write the running proxy's session as JSONL. |

| Flag | Meaning |
|---|---|
| `--upstream-url <url>` | Provider endpoint (required for `serve`/`check`). |
| `--backend <name>` | `openai` \| `anthropic` \| `llamacpp` \| `vllm` \| `mlxlm`. |
| `--bind <addr>` | Loopback address (default `127.0.0.1:8787`). |
| `--tokenizer <name>` | Prefix-hash tokenizer (default `cl100k_base`). |
| `--rates <file>` | Override the built-in rate table. |
| `--verbose` | Debug logging. Metadata only — never message content. |

## Security

Loopback-only by default, no auth (single user). Cloud keys are pass-through:
never stored, never logged. Metrics live in memory only and nothing is written
to disk unless you run `export`. Binding non-loopback is explicit (`--bind`) and
should be done only on a trusted host.

## Docs

- [How measurement works](docs/how-measurement-works.md) — the record schema, the
  hit-rate formula in plain words, what `provider_reported` means, the Anthropic
  write/read split.
- [Troubleshooting](docs/troubleshooting.md) — every failure, its cause, and the
  fix.
- [Design](docs/designs/cachemax-measurement-core.md) — the full spec.

## Status & layout

The measurement core is implemented: proxy, five adapters, sessions, latency
budget, dashboard, export, CLI. The Python package under `sidecar/` is the
mlx-lm precision path (subprocess, off the hot path) plus dev tooling; it is not
required to run cachemax.

- repo root — the Rust core (`Cargo.toml`, `src/*.rs`) and the embedded dashboard
- `sidecar/` — the Python mlx-lm sidecar and its uv project
- `menubar/` — the macOS menu bar status item (AppKit, no dependencies)

### macOS menu bar (optional)

A glanceable status item: `● <hit rate>` when the proxy is live, `○ —` when it
is not running, `⚠ <n>` when turns are incomplete. Clicking it opens the web
dashboard, which remains the single full UI. It polls `/api/state`; it shows no
panels of its own.

```bash
menubar/build.sh    # builds menubar/cachemax-menubar.app
open menubar/cachemax-menubar.app
# point at a non-default proxy (default bind is 127.0.0.1:8787).
# -n forces a fresh instance: `open` reuses a running one and would ignore this.
CACHEMAX_URL=http://127.0.0.1:9000 open -n menubar/cachemax-menubar.app
```

Requires the Swift toolchain (Xcode Command Line Tools). macOS only.

### mlx-lm sidecar (macOS / Apple Silicon)

mlx-lm exposes no cache truth: it produces no hit-rate number, only per-turn
TTFT. The sidecar serves mlx-lm behind the OpenAI wire shape so the proxy can
point at it like any other upstream:

```bash
uv sync --project sidecar --extra mlx        # installs mlx-lm (Apple Silicon)
uv run --project sidecar cachemax-sidecar serve --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
# in another shell:
cachemax serve --backend mlxlm --upstream-url http://127.0.0.1:8080/v1
```

The dashboard shows no hit rate for mlx-lm (`—`, source `no_cache_truth`); the
signal is the TTFT cold→warm readout. To print the warm/cold comparison
directly, run `cachemax-sidecar measure`. It reports the ratio; it asserts no
speed threshold, because real mlx-lm clusters near 1.0x (prefill is a small
share of first-token latency at small sizes).

## Development

```bash
cargo test                 # Rust core
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check

uv sync --project sidecar  # Python sidecar (fake engine; no mlx-lm needed)
uv run --project sidecar pytest
CACHEMAX_MLX_REAL=1 uv run --project sidecar pytest tests/test_real_mlx.py -s  # real mlx-lm, opt-in

menubar/test.sh            # menu bar render logic (macOS)
```
