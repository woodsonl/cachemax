# cache-maxing — measurement core spec

Repo: woodsonl/cache-maxing (github.com/woodsonl/cache-maxing)
Branch: main
Status: SPEC — zero implementation. Every number here is a design decision, not a measurement.
Supersedes: docs/designs/cache-maxing-proxy.md

## What this is

An OpenAI-compatible proxy that sits in front of an LLM endpoint — **cloud (typical) or local (minority)** — measures how much of each request's re-sent history was served from the provider's prompt cache, and reports the cost and speed consequence on a live dashboard.

It measures first. It does not repair yet (repair is a later phase).

## Who uses it

- **Typical user:** points an app or agent at the proxy instead of directly at a cloud LLM (OpenAI or Anthropic). Wants to know how much prompt-cache reuse they are getting and what it costs.
- **Minority user:** runs a local engine (llama.cpp, mlx-lm, vLLM) and wants to watch TTFT collapse as the cache warms.

Both are first-class targets. Local is also where token-space verification is provable (the engine reports real cached-token counts); cloud is where most users are.

## Problem

Every LLM API is stateless, so clients re-send the whole conversation each turn. Providers cache the prefix, but clients silently break it — a regenerated reply differs by a space, tool definitions get reordered, nothing warms the cache before the first real call. The ds4 article reported a 12x warm-turn speedup sitting on the table (reported claim, not reproduced here). Most users cannot see whether they are getting any of it, on cloud or local.

## Payoff by backend

| Backend | Hero metric | Why |
|---|---|---|
| Cloud (OpenAI, Anthropic) | **Cost** — billed input tokens saved | Warm turns cut billed input; TTFT barely moves when RTT dominates |
| Local (llama.cpp, mlx-lm, vLLM) | **Speed** — TTFT collapse cold→warm | The proxy's overhead is a visible share of TTFT |

Dashboard shows both; the hero surface weights by backend.

## Binding constraints

- **Proxy overhead: ≤1-2 ms p50 (≈5 ms p95) added TTFT vs direct, both backends.** Measured client-observed first-token timing with equivalent cache conditions on both arms, p95 gate over N runs. Every microsecond counts; one budget, no cloud relaxation. Engineless CI runs structural checks only (byte-match, record integrity) and cannot certify the budget. A >5 ms p95 regression fails the build.
- **Streaming (SSE) passes through unbuffered** while the record builds; the stored record must byte-match what the client received. Any feature that would buffer, batch, or delay the client stream is rejected at design time.
- **Secrets are pass-through only, never stored, never logged.** The client sends its provider key; the proxy forwards it and forgets it.
- **Request bodies are recorded only with explicit opt-in.** No-opt-in model: session-scoped token-prefix hashes and per-request numeric metrics (TTFT, token counts, cached-token counts, cost) in memory only — no bodies, no disk. The tape renders hash-level match/mismatch; forensic byte diagnosis requires opt-in.
- **Hit-rate formula (binding):** per turn t ≥ 1 (turn 0 is cold, excluded), `hit_rate = Σ cached_tokens(t) / Σ resent_history_tokens(t)`. `resent_history_tokens(t)` is the token count of the re-sent message list — system prompt plus all prior user, assistant, and tool messages, excluding turn t's new content. `cached_tokens` is the provider/engine-reported count of prefix tokens served from cache. On local engines it is clamped to the history span (ratio ≤ 100%); on cloud it is the provider-reported figure as-is, labeled `provider_reported`. Reported per-turn and session-cumulative; acceptance compares per-turn counts individually, not just the session sum.
- **Session definition:** a session is a tracked conversation — a request belongs to the session whose remembered token-prefix it extends; no match starts a new session; forks resolve by longest-matching-prefix (ties: most recent activity). A request whose prefix breaks against a tracked session stays in that session and is measured as a miss — it does not become turn 0 of a silent new session.
- **Incomplete records** (dropped stream, engine/provider error mid-request) are marked `incomplete`, excluded from session-cumulative aggregation, and shown with an explicit marker.
- **Rewrite failure fallback:** on any failure or ambiguity, send the client's original request unmodified (applies when repair lands).
- Every unexposed field renders `—`, never `0`.

## Backends and adapters

One normalized record type, five adapters. `--backend` selects; default `openai`.

| Adapter | Cache signal | Verification |
|---|---|---|
| **openai** (cloud, primary) | `usage.prompt_tokens_details.cached_tokens` | Provider-reported. No independent ground truth; labeled. |
| **anthropic** (cloud, primary) | `usage.cache_read_input_tokens` + `cache_creation_input_tokens` (write/read split) | Provider-reported. Write/read tracked separately. |
| **llama.cpp** (local) | `/slots` cached-token counts, prompt echo / `tokens_evaluated`, slot `chat_format`/`generation_prompt` | Token-space ground truth (±5%, this is the only engine where token-space matching itself is verified). |
| **vLLM** (local) | `/metrics` token counters, delta-sampled per request | Record fidelity (±5% at session-aggregate; fallback to log-line parsing if noisy). |
| **mlx-lm** (local) | none exposed | TTFT warm/cold discrimination only. No hit-rate number. |

Anthropic is a different cache model, not a hit count: `cache_creation_input_tokens` are tokens written to cache (billed at a premium) and `cache_read_input_tokens` are served from cache (billed at a discount). The dashboard shows the write/read split and the resulting cost, not a single percentage.

Cloud cache behavior is provider-reported and cannot be independently verified (no closed provider guarantees true cached-token counts). The plan says so and labels cloud figures accordingly.

## Hit rate and cost

- **Local:** hit-rate per the binding formula; llama.cpp measured, vLLM measured-with-caveat, mlx-lm shows no number (discrimination only, labeled "no cache truth on this engine").
- **Cloud:** hit-rate from provider-reported cached tokens; cost = billed input tokens (with cache discounts/premiums applied per provider's published rates) vs a no-cache counterfactual.

## Success criteria

- **llama.cpp (ground truth):** one scripted multi-turn conversation produces a record whose session-cumulative hit-rate matches `/slots` within ±5%, per-turn counts compared individually. Includes a cleared-cache control run to separate measured hits from reuse potential.
- **vLLM (record fidelity):** cached-token figures agree with `/metrics` within ±5% at session-aggregate, single-client serialized run. If delta-sampling is too noisy, restate to log-line parsing; ±5% bar unchanged.
- **mlx-lm:** TTFT warm turns ≥2x faster than comparable-length cold turns on ≥90% of warm turns; no hit-rate number.
- **OpenAI:** the proxy reports `cached_tokens` from the response usage and computes hit-rate + cost consistently; record matches provider usage exactly (it is the provider's own number).
- **Anthropic:** the proxy reports cache read/creation tokens and the write/read split; cost reflects the premium/discount; figures match the provider usage exactly.
- **Dashboard:** live curve visible during a real session, both a cloud run and a local run.
- **Cloud cost gate:** on a replayed real trace, ≥30% reduction in billed-at-full-rate input tokens (deferred until repair; the measurement core proves the measurement, not the reduction).
- **Repair (later):** zero tool-call correctness regressions; fallback rule exercised on every ambiguity.

## Architecture

```
client (app / agent) ──► cache-maxing proxy ──► LLM endpoint
   OpenAI dialect          │   (cloud: api.openai.com, api.anthropic.com)
                           │   (local: llama.cpp, vLLM, mlx-lm)
                           │
                           ├─ forward first, unbuffered SSE
                           ├─ tokenize + hash prefixes concurrently (never blocks first byte)
                           ├─ session store (in-memory prefix hashes)
                           ├─ adapter reads cache signal (usage / slots / metrics)
                           └─ normalized record ──► dashboard + JSONL export
```

One normalized record type is the load-bearing abstraction: repair, the tape, and a future cloud gate all consume it.

## Data flow

```
INPUT (client request)
  -> VALIDATE (shape; count what parses)
  -> TOKENIZE (concurrently; never blocks forward)
  -> FORWARD (unbuffered to endpoint)
  -> OBSERVE (stream to client WHILE record accumulates:
              TTFT at first token, token counts, cache signal)
  -> FINALIZE (status: complete | incomplete | provider_reported | no_cache_truth)
  -> OUTPUT (dashboard + session-cumulative)
```

Edge cases: empty messages → `—` (zero denominator, not 0); oversized prompt → incomplete + error surfaced; engine timeout → incomplete; concurrent requests extending one prefix → atomic aggregate append (no `.await` inside the lock guard); tokenizer version mismatch → flagged, marked accordingly; engine restart mid-session → curve shows a session break, not a splice; proxy restart → "metrics reset" banner (state is ephemeral by design).

## Stack

- **Core: Rust** (axum/tokio, hyper unbuffered SSE, reqwest, HF `tokenizers` crate when a local tokenizer is needed). Single binary, cross-platform (macOS Intel/AS, Linux x86_64/aarch64, Windows MSVC).
- **Local mlx-lm precision path: Python sidecar** (subprocess, off the hot path). Never imported by the core.
- **No tokenizer and no Python required to run and show a curve.** Tokenizers and the sidecar are opt-in precision layers.
- Repo scaffolding (uv + FastAPI skeleton) remains only for the Python sidecar and dev tooling.
- Distribution: `cargo install cache-maxing`, release binaries, `cargo binstall`. Dashboard is a single HTML file embedded in the binary.
- Tests: `cargo test` for the core (unit + integration + latency system test), engineless via trait mocks; pytest for the sidecar's contract tests. CI: macOS + Linux + Windows.

## Latency budget method

The CI test measures proxy-vs-direct first-token timing, warm-pinned, median-of-N, p95 gate. A micro-profile breaks the hot path into stages (serde, tokenize, hash, session lookup, forward) so any budget verdict names the culprit. Failures at >5 ms p95 on an injected delay.

## Security posture

Local binary binding loopback by default; no auth (single user, no multi-user surface, no object IDs). If bound non-loopback, that must be explicit config with a stated warning. Cloud keys are pass-through, never stored, never logged. Opt-in body recording writes local disk only, no sync, and the dashboard escapes rendered content (no HTML/JS injection via recorded bodies). Finalize logs are metadata + counts only, never message content; malformed-chunk logs record parser position + byte count + error category, never raw bytes.

## Tasks

### Core

- [ ] **C1 — proxy core.** axum/tokio SSE passthrough, concurrent tokenization, record builder, normalized record type. Files: `src/proxy.rs`, `src/record.rs`. Verify: byte-match fixture (stream == record, reassembled by concatenation); 50K-token prompt TTFT regression; incomplete-record unit test.
- [ ] **C2 — cloud adapters.** openai (`cached_tokens`) and anthropic (read/creation split). Files: `src/adapters/openai.rs`, `src/adapters/anthropic.rs`. Verify: record matches provider usage exactly on a live run; cost math reflects published rates.
- [ ] **C3 — local adapters.** llama.cpp (ground truth), vLLM (metrics), mlx-lm (discrimination only). Files: `src/adapters/llamacpp.rs`, `src/adapters/vllm.rs`, `src/adapters/mlxlm.rs`. Verify: ±5% llama.cpp criterion; mocked-adapter CI suite passes engineless.
- [ ] **C4 — sessions + aggregation.** prefix-continuity store, fork resolution, atomic aggregate. Files: `src/sessions.rs`. Verify: interleaved-requests atomicity test (real interleave, tokio); fork tie-break fixture; collision log test.
- [ ] **C5 — latency budget CI.** warm-pinned median-of-N proxy-vs-direct TTFT + micro-profile. Files: `tests/` + CI workflow. Verify: >5 ms p95 fails on injected delay; passes on clean tree.
- [ ] **C6 — dashboard.** single-file, hero surface per backend (cost on cloud, TTFT on local), hit rate, prefix tape, session view. Files: `src/dashboard.rs`. Verify: state-map rows each render; tape legible without color (glyph, not just green/red); no fake detail in hash-level mode.
- [ ] **C7 — observability.** JSONL export (metrics-only by default) + structured finalize logs. Files: `src/export.rs`. Verify: export matches dashboard numbers; logs grep-able without bodies.
- [ ] **C8 — CLI.** `serve`, `--backend`, `--upstream-url`, `--check`, `export`, `--verbose` (metadata only), engine/precision flags. Files: `src/main.rs`. Verify: `--check` fails loudly on unreachable upstream; `--help` lists everything; defaults work with zero flags.

### Getting started (T0 contract)

- [ ] **D1 — README + docs.** Install command, first command, dashboard URL, cost-capable client snippet. Files: `README.md`. Verify: the sequence runs end to end; no stale proxy-vision text; typo fixed.
- [ ] **D2 — one named agent config.** Point a real agent at the proxy (OpenAI `base_url` swap, model passed through, dummy `api_key` accepted-and-ignored if required) with one worked example. Files: `README.md`. Verify: the named agent completes a multi-turn conversation through the proxy.
- [ ] **D3 — error contract.** Problem + cause + fix + docs link for: upstream unreachable, no cache signal, key rejected, tokenizer unavailable. Files: `src/main.rs`, `docs/troubleshooting.md`. Verify: each path emits the four parts; no raw panic by default.
- [ ] **D4 — docs structure.** `README.md` (getting started), `docs/troubleshooting.md`, `docs/how-measurement-works.md` (record schema, formula in plain words, what provider-reported means, write/read split). Verify: `provider_reported` and the Anthropic split explained; troubleshooting does not invite sharing content-bearing logs.

## Not in scope

- Repair engine, murderer demo mode, cloud cost gate — later phases; measurement core first.
- Prefix-affinity routing, cross-dialect translation, cache warming — superseded doc's scope.
- Hosted/deployable service — the proxy is a local binary fronting cloud upstreams.
- LICENSE selection, issue templates, changelog ownership — at ship time.
- Demo command — cut; the magical moment is real traffic (cold turn, then warm turn).

## What already exists

- Repo scaffolding: uv project, FastAPI skeleton, placeholder CLI, 1 passing test. Repurposed for the sidecar/dev tooling; the core is new Rust.
- llamacpp-stats-dashboard (MIT): patterns borrowed — read1() streaming, caps-field display, pre-warm TTFT testing.
- LMCache: per-request hit-rate attribute pattern, token-level counters, blake3 hashing option.
- Rust ecosystem: axum/hyper SSE, reqwest, HF `tokenizers`.

## Open questions

- vLLM: poll `/metrics` per request or scrape continuously — attribution granularity differs.
- Tape from the proxy's own token stream (portable) or engine ground truth where available (accurate). Default: engine truth when available, labeled otherwise.
- Binary name: `cache-maxing` vs shorter `cache-max` CLI. Decide before first release.
