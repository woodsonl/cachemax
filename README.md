# cachemax

An OpenAI-compatible proxy that sits in front of an LLM endpoint, measures how
much of each request's re-sent history was served from the provider's prompt
cache, repairs drifted request history so the provider re-sees what it already
cached, and reports the cost and speed consequence on a live dashboard.

Measure → repair → prove, on one loopback port:

- **Measure (always on):** point an app or agent at the proxy instead of
  directly at OpenAI, Anthropic, or OpenRouter, and see how much prompt-cache
  reuse you get and what it costs. Local engines (llama.cpp, vLLM, mlx-lm) are
  a first-class target too: watch TTFT collapse as the cache warms.
- **Repair (dry-run by default):** agent frameworks re-serialize tool-call
  arguments and re-wrap text on every turn, and each re-serialization breaks
  the provider's cache key. cachemax classifies that drift per turn — in
  `--repair on` it rewrites the drifted history back to the exact
  serialization the provider already cached, so the cache hits again. It never
  invents content: a rewrite replaces bytes with bytes the provider already
  accepted, and anything semantically different passes through untouched and
  flagged.
- **Prove:** every turn's record shows what happened — drift classified,
  tokens repaired, cache-served tokens recovered — on the dashboard and in the
  metrics-only JSONL export.

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
{"session_id":1,"turn":1,"status":"complete","source":"provider_reported","ttft_ms":120.0,"cached_tokens":1020,"cache_written_tokens":0,"resent_history_tokens":1550,"billed_input_tokens":1750,"broke_prefix":false,"cost_usd":0.02,"cost_saved_usd":0.01,"repair_mode":"dry_run","repaired":false,"matches_canonical":false,"drift_kind":"tool_arg_reserialization","canonicalized_tokens":312,"breakpoint_count":null}
```

The last five fields are the drift annotation: `repair_mode` (`off` |
`dry_run` | `on`), `repaired` (true only when a rewrite happened), and — when
the turn was examined — whether the re-sent history matched the canonical
chain under semantic JSON equality, the classified drift flavor, and the
tokens at risk. `breakpoint_count` is the cache hints on the request as
forwarded (`null` when breakpoint management is off). Still metrics only:
never message content.

### Prove it yourself: `cachemax replay`

The A/B that produced the numbers above is reproducible on your own traffic:

1. Run `cachemax serve` with `--repair dry-run` (the default) in front of
   your provider for a while — the ledger records the canonical chains.
2. Run `cachemax replay` (add `--ledger-dir <path>` for a non-default
   location; ignores `--no-ledger` — it reads the recorded directory).
   For every recorded chain it prints one JSONL line holding the
   same request twice: `a_drifted`, re-serialized the way agent frameworks
   do (tool-argument keys reordered, interior whitespace collapsed), and
   `b_canonical`, the exact content the provider already cached.
3. Drive your endpoint with both bodies — `a_drifted` is the cache-miss
   baseline, `b_canonical` is what repair forwards — and compare the
   provider's `cached_tokens` per variant. The dashboard's *recovered by
   repair* line shows the same delta live, on repaired turns.

Or let `replay` drive it for you:

```
cachemax replay --execute \
  --upstream-url https://api.openai.com/v1 \
  --backend openai --n 5
```

`--execute` sends each form to the endpoint `--n` times (bypassing the
proxy, so the number is the provider's own cache, not ours) and prints a
table of cached tokens — median and max — per form, plus how many distinct
upstream instances answered. Auth is read from an environment variable
(`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, or `--api-key-env <VAR>`); the key
is never printed or written. A single send is not a measurement on a routed
endpoint — different instances have different cache namespaces — so read
`--n`'s max, not one line; the table says so when more than one instance
answered.

### Declaring conversation affinity

By default a request joins the session whose token-prefix it extends, which
is what you want for an ordinary client. An agent that re-sends a
truncated or re-based history every turn can defeat prefix matching; send an
`x-cachemax-session: <id>` header to name the conversation explicitly:

```
POST /v1/chat/completions
x-cachemax-session: my-agent-run-42
```

Every request under the same key is one session, whatever the bytes — the
client's word is the authority, no prefix-fork inference. Distinct keys
never cross, even with identical history. Omit the header for the default
prefix-based behavior.

## Commands

| Command | What it does |
|---|---|
| `cachemax serve` | Run the proxy + dashboard (default). |
| `cachemax check` | Check the upstream is reachable; exit non-zero if not. |
| `cachemax export` | Write the running proxy's session as JSONL. |
| `cachemax purge` | Delete the on-disk repair ledger (see Security). |
| `cachemax replay` | Print A/B request bodies (drifted vs canonical) from the recorded ledger, as JSONL. With `--execute`, drive them against a real endpoint and report cached tokens per form. |

| Flag | Meaning |
|---|---|
| `--upstream-url <url>` | Provider endpoint (required for `serve`/`check`/`replay --execute`). |
| `--backend <name>` | `openai` \| `anthropic` \| `llamacpp` \| `vllm` \| `mlxlm`. |
| `--bind <addr>` | Loopback address (default `127.0.0.1:8787`). |
| `--tokenizer <name>` | Prefix-hash tokenizer (default `cl100k_base`). |
| `--rates <file>` | Override the built-in rate table. |
| `--repair <mode>` | `dry-run` (default) \| `on` \| `off`. Rewrite drifted history to the canonical serialization the provider already saw (`on`); every rewrite is logged. Per-request header `x-cachemax-repair: on\|off` overrides. |
| `--ledger-dir <path>` | Where the local repair ledger lives (default `~/.cache/cachemax/ledger`). |
| `--no-ledger` | Keep the repair ledger in memory only; write nothing to disk. |
| `--manage-breakpoints` | Anthropic only: place `cache_control` breakpoints per the incremental-breakpoint guidance (last system block + last user/tool-result blocks, ≤ 4). Requests carrying client-placed breakpoints pass through untouched. |
| `--force-breakpoints` | With `--manage-breakpoints`: re-derive breakpoints even over client-placed ones. |
| `--verbose` | Debug logging. Metadata only — never message content. |

`replay` takes its own flags: `--execute`, `--upstream-url`, `--backend`,
`--api-key-env <VAR>`, and `--n <samples>`.

## Security

Loopback-only by default, no auth (single user). Cloud keys are pass-through:
never stored, never logged. Metrics live in memory only, and metrics export
(`cachemax export`) writes nothing but per-turn counts — never message content.

One thing does touch disk: the **repair ledger**. To repair a broken cache
prefix, cachemax must remember the exact message content it forwarded and
received, so `serve` persists that record locally under
`~/.cache/cachemax/ledger/` (one `<session>.jsonl` file per session). It
stays on your machine and is never included in exports or logs. `cachemax
purge` (or `cachemax purge --ledger-dir <path>` for a non-default location)
deletes every `<session>.jsonl` file directly inside that directory — other
files, subdirectories, and the directory itself stay; symlinks are never
followed — and reports what it removed. `purge` ignores `--no-ledger` and
always targets the on-disk directory: it is how you purge what a previous
run wrote. Run `serve` with `--no-ledger` to keep the ledger in memory only,
so nothing is ever written at all.

Binding non-loopback is explicit (`--bind`) and should be done only on a
trusted host.

## Docs

- [How measurement works](docs/how-measurement-works.md) — the record schema, the
  hit-rate formula in plain words, what `provider_reported` means, the Anthropic
  write/read split.
- [Dogfood report](docs/dogfood-v0.2.0.md) — v0.2.0 measured against real routed
  traffic: verified rewrites, the drift token surcharge, and routing-noise
  methodology.
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
