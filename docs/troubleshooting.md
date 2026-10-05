# Troubleshooting

Every failure below names the **problem**, the **cause**, the **fix**, and links
here. cachemax does not print a raw panic for expected failures.

> **Before you paste a log anywhere:** cachemax finalize logs carry metadata and
> counts only — never message content — but a provider's *own* error body might.
> Redact before sharing.

## upstream-unreachable

**Problem:** `cachemax --check` or `serve` reports the upstream unreachable.

**Cause:** the host in `--upstream-url` did not answer — wrong URL, no network,
or a typo in the scheme/host.

**Fix:** confirm the URL. `cachemax check --upstream-url <url>` should print
`reachable (HTTP …)`. A reachable host that returns `401 Unauthorized` is
**fine** — cachemax checks reachability, not authorization. The API key is
yours and is passed through untouched.

```bash
cachemax check --upstream-url https://api.openai.com/v1
# cachemax --check: upstream https://api.openai.com/v1 reachable (HTTP 401 Unauthorized)
```

## no-cache-signal

**Problem:** the dashboard shows `—` for the hit rate on a cloud run, or a local
engine shows `no_cache_truth`.

**Cause:** the response carried no cache figure cachemax recognizes, or you are
on an engine that exposes none.

**Fix:** check `--backend` matches the upstream dialect. `openai` covers OpenAI
and OpenRouter; `anthropic` covers the Anthropic Messages API; `llamacpp`,
`vllm`, and `mlxlm` cover the local engines. mlx-lm exposes no cache truth by
design — it shows TTFT warm/cold discrimination only, and `—` for the hit rate,
which is correct, not a bug.

## mlx-sidecar-unavailable

**Problem:** `--backend mlxlm` reaches nothing, or the sidecar exits at start.

**Cause:** the mlx-lm sidecar is macOS / Apple Silicon only and mlx-lm is an
opt-in extra.

**Fix:** on Apple Silicon,
`uv sync --project sidecar --extra mlx`, then
`uv run --project sidecar cachemax-sidecar serve`, then point the proxy at it
with `--upstream-url http://127.0.0.1:8080/v1`. On Linux/Windows the mlx-lm
backend cannot run; use `llamacpp` or `vllm` instead.

## key-rejected

**Problem:** the upstream returns `401` / `403` to your client's requests.

**Cause:** the API key your client sends is missing, malformed, or lacks quota.

**Fix:** cachemax never stores or logs keys; it forwards whatever your client
sends. If your agent needs a non-empty key to start, give it a dummy — cachemax
passes it through and the provider (or your local engine) decides. Point the
error at the key your *client* holds, not at cachemax.

## tokenizer-unavailable

**Problem:** startup fails with `tokenizer unavailable`.

**Cause:** `--tokenizer` named an encoding cachemax doesn't know.

**Fix:** use the default `cl100k_base`, another encoding (`o200k_base`,
`p50k_base`, `r50k_base`), or a known model name routed through the built-in
table. The default needs no configuration.

```bash
cachemax serve --tokenizer cl100k_base --upstream-url https://api.openai.com/v1
```

## oversized-prompt

**Problem:** a turn is recorded `incomplete`.

**Cause:** the prompt exceeded the upstream's context window; the provider
rejected or truncated it.

**Fix:** shorten the conversation or raise the provider's limit. An incomplete
record is excluded from the cumulative hit rate — it never silently becomes a
`0`. The dashboard shows the turn with `?`.

## engine-timeout

**Problem:** a turn is recorded `incomplete` and the client saw the stream end
early.

**Cause:** the local engine (or the connection) stalled mid-stream.

**Fix:** the record stays `incomplete`, excluded from the cumulative, so a
stalled turn can't pollute your numbers. Check the engine's health; a local
engine restart shows a **session break** (not a splice) in the tape.

## tokenizer-version-mismatch

**Problem:** prefix hashes look discontinuous across a provider or engine change.

**Cause:** the tokenizer used for prefix hashing differs from the one that
produced an earlier session's hashes. Continuity is defined at the token level.

**Fix:** keep `--tokenizer` stable across a session. If you change it, expect a
new session rather than continued continuity — start with a fresh conversation.

## no-running-proxy

**Problem:** `cachemax export` reports no running proxy.

**Cause:** state is in-memory and ephemeral by design; nothing is written to disk.
`export` reads the *running* proxy at `/api/export`.

**Fix:** start the proxy first, then export from it. Reload the dashboard if the
proxy restarted — a **metrics reset** banner tells you the state is fresh.

```bash
cachemax serve --upstream-url https://api.openai.com/v1 &
cachemax export --out ./session.jsonl
```

## rates-file

**Problem:** `--rates` fails to load.

**Cause:** the JSON is malformed or missing the `models` object.

**Fix:** provide the documented shape; entries replace same-prefix built-ins.

```json
{
  "models": {
    "my-model": { "input_per_mtok": 3.0, "output_per_mtok": 15.0,
                  "cached_input_mult": 0.5, "cache_write_mult": 0.0 }
  }
}
```

`cached_input_mult` and `cache_write_mult` are fractions of the full input rate
and default to `0.5` and `0.0` when omitted.

## ledger

**Problem:** `serve` reports the ledger directory unusable.

**Cause:** the `--ledger-dir` path (default `~/.cache/cachemax/ledger`) cannot
be created or written — permissions, a read-only volume, or the path is a file.

**Fix:** pass a writable `--ledger-dir`, or `--no-ledger` to keep the ledger in
memory only (repair then works within a single run and forgets the chain on
restart). The ledger stores the message content cachemax forwards and receives,
locally, so repair can extend the provider-seen prefix; it is never exported or
logged. Delete the directory to purge it.

## repair-mode

**Problem:** `--repair` rejects the given mode.

**Cause:** the value is not one of the three modes.

**Fix:** `--repair dry-run` (the default: detect and annotate drift, never
touch a byte), `--repair on` (rewrite drifted history to the canonical
serialization the provider already saw — every rewrite is logged), or
`--repair off` (no classification). A per-request `x-cachemax-repair: on|off`
header overrides the configured mode for that one request. `on` never
rewrites what it cannot prove equivalent: changed tool arguments, changed
system prompts, model switches, and first turns all pass through untouched
and are flagged instead.
