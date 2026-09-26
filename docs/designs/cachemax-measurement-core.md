# cachemax: measurement core spec

Repo: woodsonl/cachemax (github.com/woodsonl/cachemax)
Branch: main
Status: SPEC, zero implementation. Every number here is a design decision, not a measurement.
Supersedes: docs/designs/cache-maxing-proxy.md

## What this is

An OpenAI-compatible proxy in front of an LLM endpoint. It fronts either a cloud provider (typical) or a local engine (minority). It measures how much of each request's re-sent history the provider served from its prompt cache, then reports the cost and speed consequence on a live dashboard.

It measures. Repair is a later phase.

## Who uses it

- **Typical user:** points an app or agent at the proxy instead of directly at a cloud LLM (OpenAI, Anthropic, or OpenRouter). Wants to know how much prompt-cache reuse they get and what it costs.
- **Minority user:** runs a local engine (llama.cpp, mlx-lm, vLLM) and watches TTFT collapse as the cache warms.

Both are first-class targets. Local is where token-space verification is provable: the engine reports real cached-token counts. Cloud is where most users are.

## Problem

Every LLM API is stateless, so clients re-send the whole conversation each turn. Providers cache the prefix, but clients break it: a regenerated reply differs by a space, tool definitions get reordered, nothing warms the cache before the first real call. The ds4 article reported a 12x warm-turn speedup (reported claim, not reproduced here). Most users cannot see whether they get any of it, on cloud or local.

## Payoff by backend

| Backend | Hero metric | Why |
|---|---|---|
| Cloud (OpenAI, Anthropic, OpenRouter) | **Cost**: billed input tokens saved | Warm turns cut billed input; TTFT barely moves when RTT dominates |
| Local (llama.cpp, mlx-lm, vLLM) | **Speed**: TTFT collapse cold to warm | The proxy's overhead is a visible share of TTFT |

The dashboard shows both; the hero surface weights by backend.

## Binding constraints

- **Proxy overhead: ≤1-2 ms p50 (≈5 ms p95) added TTFT vs direct, both backends.** Measured as the proxy-vs-direct first-token delta (not absolute client-observed TTFT, which RTT dominates) with equivalent cache conditions on both arms, warm-pinned, median-of-N, p95 gate. Every microsecond counts; one budget, no cloud relaxation. The target is ≤5 ms p95; the CI fail gate sits strictly above at >6 ms p95, so measurement noise near the target does not flake the build. Engineless CI cannot certify the real budget (no engine); it runs structural checks plus a latency harness on an injected delay that must fail the build above 6 ms p95, proving the gate detects regressions.
- **Streaming (SSE) passes through unbuffered** while the record builds; the stored record must byte-match what the client received. Any feature that would buffer, batch, or delay the client stream is rejected at design time.
- **Secrets are pass-through only, never stored, never logged.** The client sends its provider key; the proxy forwards it and forgets it.
- **Request bodies are recorded only with explicit opt-in.** No-opt-in model: session-scoped token-prefix hashes and per-request numeric metrics (TTFT, token counts, cached-token counts, cost) live in memory only and nothing is written to disk. The tape renders hash-level match/mismatch; forensic byte diagnosis requires opt-in. One export artifact, two triggers: `cachemax export` writes metrics-only JSONL to a path (default `./cachemax-<session>.jsonl`, `--out` to override) and the dashboard `export` control downloads the same JSONL via the browser. Body export requires the same body opt-in.
- **Hit-rate formula (binding):** per turn t ≥ 1 (turn 0 is cold, excluded), `hit_rate = Σ cached_tokens(t) / Σ resent_history_tokens(t)`. `resent_history_tokens(t)` is the token count of the re-sent message list: system prompt plus all prior user, assistant, and tool messages, excluding turn t's new content. `cached_tokens` is the provider- or engine-reported count of prefix tokens served from cache. On local engines it is clamped to the history span (ratio ≤ 100%); on cloud it is the provider-reported figure as-is, labeled `provider_reported`. Reported per-turn and session-cumulative; acceptance compares per-turn counts individually, not just the session sum.
- **Session definition:** a session is a tracked conversation. A request belongs to the session whose remembered token-prefix it extends; no match starts a new session; forks resolve by longest-matching-prefix (ties: most recent activity). A request whose prefix breaks against a tracked session stays in that session and is measured as a miss. It does not become turn 0 of a silent new session.
- **Incomplete records** (dropped stream, engine/provider error mid-request) are marked `incomplete`, excluded from session-cumulative aggregation, and shown with an explicit marker.
- **Rewrite failure fallback:** on any failure or ambiguity, send the client's original request unmodified (applies when repair lands).
- Every unexposed field renders `—`, never `0`.

## Backends and adapters

One normalized record type, five adapters. `--backend` selects; default `openai`.

| Adapter | Cache signal | Verification |
|---|---|---|
| **openai** (cloud, primary) | `usage.prompt_tokens_details.cached_tokens` | Provider-reported. Also covers OpenRouter by pointing the adapter's provider endpoint at `https://openrouter.ai/api/v1` (OpenAI-compatible; set via `--upstream-url`). |
| **anthropic** (cloud, primary) | `usage.cache_read_input_tokens` + `cache_creation_input_tokens` (write/read split) | Provider-reported. Write/read tracked separately. |
| **llama.cpp** (local) | `/slots` cached-token counts, prompt echo / `tokens_evaluated`, slot `chat_format`/`generation_prompt` | Token-space ground truth (±5%, this is the only engine where token-space matching itself is verified). |
| **vLLM** (local) | `/metrics` token counters, delta-sampled per request | Record fidelity (±5% at session-aggregate; fallback to log-line parsing if noisy). |
| **mlx-lm** (local, macOS/Apple Silicon only) | none exposed | TTFT warm/cold discrimination only. No hit-rate number. Requires the Python sidecar; unavailable on Linux/Windows. |

Anthropic uses a different cache model, not a single hit count: `cache_creation_input_tokens` are tokens written to cache (billed at a premium) and `cache_read_input_tokens` are served from cache (billed at a discount). The dashboard shows the write/read split (creation vs read) alongside cost, and derives a hit-rate from `cache_read_input_tokens` over `cache_read + cache_creation`, labeled `provider_reported` like the rest of the cloud path.

Cloud figures carry the compact `provider_reported` label.

## Hit rate and cost

- **Local:** hit-rate per the binding formula. llama.cpp measured, vLLM measured-with-caveat, mlx-lm shows no number (discrimination only, labeled "no cache truth on this engine").
- **Cloud:** hit-rate from provider-reported cached tokens; cost = billed input tokens (with cache discounts/premiums per the provider's published rates) against a no-cache counterfactual.

## Success criteria

- **llama.cpp (ground truth):** one scripted multi-turn conversation produces a record whose session-cumulative hit-rate matches `/slots` within ±5%, per-turn counts compared individually. Includes a cleared-cache control run to separate measured hits from reuse potential.
- **vLLM (record fidelity):** cached-token figures agree with `/metrics` within ±5% at session-aggregate, single-client serialized run. If delta-sampling is too noisy, restate to log-line parsing; ±5% bar unchanged.
- **mlx-lm (macOS/Apple Silicon only):** TTFT warm turns ≥2x faster than comparable-length cold turns on ≥90% of warm turns; no hit-rate number. Skipped on Linux/Windows, where the engine cannot run.
- **OpenAI:** the proxy reports `cached_tokens` from the response usage and computes hit-rate + cost consistently; record matches provider usage exactly (it is the provider's own number). OpenRouter uses the same adapter with the provider endpoint set to its URL.
- **Anthropic:** the proxy reports cache read/creation tokens, the write/read split, and a derived hit-rate (`cache_read` over `cache_read + cache_creation`); cost reflects the premium/discount; figures match the provider usage exactly.
- **Dashboard:** live curve visible during a real session, both a cloud run and a local run.
- **Cloud cost gate:** on a replayed real trace, ≥30% reduction in billed-at-full-rate input tokens (deferred until repair; the measurement core proves the measurement, not the reduction).
- **Repair (later):** zero tool-call correctness regressions; fallback rule exercised on every ambiguity.

## Architecture

```
client (app / agent) ──► cachemax proxy ──► LLM endpoint
   OpenAI dialect          │   (cloud: api.openai.com, api.anthropic.com)
                           │   (local: llama.cpp, vLLM, mlx-lm)
                           │
                           ├─ forward first, unbuffered SSE
                           ├─ tokenize + hash prefixes concurrently (never blocks first byte)
                           ├─ session store (in-memory prefix hashes)
                           ├─ adapter reads cache signal (usage / slots / metrics)
                           └─ normalized record ──► dashboard + JSONL export
```

One normalized record type is the shared abstraction: repair, the tape, and a future cloud gate all read it.

## Data flow

```
INPUT (client request)
  -> VALIDATE (shape; count what parses)
  -> TOKENIZE (concurrently; never blocks forward)
  -> FORWARD (unbuffered to endpoint)
  -> OBSERVE (stream to client WHILE record accumulates:
              TTFT at first token, token counts, cache signal)
  -> FINALIZE (status: complete | incomplete;
              source label: provider_reported | engine_measured | no_cache_truth)
  -> OUTPUT (dashboard + session-cumulative)
```

Status (`complete` | `incomplete`) is orthogonal to the source label. A cloud request can be `complete` with source `provider_reported`; an mlx-lm request is `complete` with source `no_cache_truth`. Only `incomplete` records are excluded from session-cumulative aggregation.

Edge cases: empty messages → `—` (zero denominator, not 0); oversized prompt → incomplete + error surfaced; engine timeout → incomplete; concurrent requests extending one prefix → atomic aggregate append (no `.await` inside the lock guard); tokenizer version mismatch → flagged and marked; engine restart mid-session → curve shows a session break, not a splice; proxy restart → "metrics reset" banner (state is ephemeral by design).

## Stack

- **Core: Rust** (axum/tokio, hyper unbuffered SSE, reqwest, `tiktoken-rs` for token-prefix hashing). Live at the repo root (`Cargo.toml`, `src/*.rs`). Single binary, cross-platform (macOS Intel/AS, Linux x86_64/aarch64, Windows MSVC).
- **Prefix hashing is token-level from the start.** `tiktoken-rs` embeds the vocabulary, so no external asset ships. Default encoding is `cl100k_base` (the OpenAI-family lingua franca); `--tokenizer <name|path>` overrides (`o200k_base`, `p50k_base`, or a model name via `bpe_for_model`).
- **Local mlx-lm precision path: Python sidecar** (subprocess, off the hot path). Lives under `sidecar/`. Never imported by the core.
- **No user-supplied tokenizer and no Python are required to run and show a curve.** The bundled tokenizer always drives the tape's prefix hashing; token counting on the cloud path stays provider-reported, never locally re-derived. The sidecar is an opt-in precision layer.
- Repo scaffolding (uv project under `sidecar/`) remains only for the Python sidecar and dev tooling.
- Distribution: `cargo install cachemax`, release binaries, `cargo binstall`. The dashboard is a single HTML file embedded in the binary.
- Tests: `cargo test` for the core (unit + integration + latency system test), engineless via trait mocks; pytest for the sidecar's contract tests. CI: macOS + Linux + Windows.

## Dashboard interaction states

What the user sees per surface, per state. Empty states name the one next action, never "No data."

| Surface | Loading | Empty | Error | Success | Partial |
|---|---|---|---|---|---|
| Hero band | Skeleton bar + "connecting to upstream…" | "No session yet. Point your app or agent at this proxy and send a request." + the proxy URL to copy | Upstream unreachable: red status line with cause + fix (mirrors the D3 error contract); hero shows `—` | Live numbers updating per turn | `incomplete` request: hero keeps last good value, badge `⚠ 1 incomplete` |
| Session view | Table shell, 3 placeholder rows | One ghost row using the `—` convention | Table frozen at last good row + error banner | Per-turn rows append as turns complete | Incomplete turn row shows `?` (its own glyph), excluded from cumulative |
| Prefix tape | Empty tape with legend | Tape shows a single `·` cold cell; caption "waiting for turn 1" | Tape dims, error banner over it | Per-turn cells fill left-to-right | Break cell `┊`, incomplete cell `?` |
| Status strip | "connecting…" | "idle" | "upstream error" + docs link | "● live" | "N incomplete" |
| Whole dashboard | | | | | **metrics reset banner** after proxy restart (state is ephemeral by design); **session break** (not a splice) when the engine restarts mid-session |

`—` (not `0`) is the unexposed-value token. Source labels (`provider_reported`, `engine_measured`, `no_cache_truth`) are labeled inline wherever the figure appears.

## User journey (cloud-primary)

The dashboard must prove it is alive before it proves it saves money. A user who cannot tell the proxy is working will not wait for the cost number.

| Step | User does | User feels | Plan specifies |
|---|---|---|---|
| 1 | Runs install + first command, opens the dashboard URL | Curious, impatient | D1 (install, first command, URL); empty state names the next action |
| 2 | Points an agent at the proxy (`base_url` swap) | Doubt: "did I point it right?" | D2 (named agent config); status strip shows `● live` on first request |
| 3 | Sends turn 0 (cold) | "Is it even working?" | Hero + session row appear within one turn; `cold` cell visible |
| 4 | Sends turn 1 (first warm turn) | Relief: "there it is" | Hit rate + cached/resent + the mode's source label (`provider_reported` cloud, `engine_measured` local) land; tape shows HIT |
| 5 | Reads the session-cumulative row | "So that is what cache reuse is worth" | Cumulative hit rate + cost saved in the hero (cloud weighting) |
| 6 | Watches a longer agentic session (long system prompt, tools) | Trust builds | Per-turn rows compare individually; incomplete/miss markers stay honest |

The local path reuses steps 1, 4, and 5 with the hero weighted to TTFT collapse instead of cost.

## Latency budget method

The CI test measures proxy-vs-direct first-token timing, warm-pinned, median-of-N, p95 gate. The target is ≤5 ms p95; the fail gate is >6 ms p95 (strictly above the target, so noise does not flake the build). A micro-profile breaks the hot path into stages (serde, tokenize, hash, session lookup, forward), so a budget verdict names the culprit. A >6 ms p95 failure on an injected delay fails the build.

## Security posture

Local binary binds loopback by default, with no auth (single user, no multi-user surface, no object IDs). Binding non-loopback must be explicit config with a stated warning. Cloud keys are pass-through, never stored, never logged. Opt-in body recording writes local disk only, no sync, and the dashboard escapes rendered content (no HTML/JS injection via recorded bodies). Finalize logs carry metadata and counts only, never message content; malformed-chunk logs record parser position, byte count, and error category, never raw bytes.

## Tasks

### Core

- [x] **C1 — proxy core.** axum/tokio SSE passthrough, concurrent tokenization, record builder, normalized record type, and the binding hit-rate formula (per-turn and cumulative, denominator = `resent_history_tokens`). Files: `src/proxy.rs`, `src/record.rs`. Verify: byte-match fixture (stream == record, reassembled by concatenation); 50K-token prompt TTFT regression; incomplete-record unit test; **hit-rate formula unit fixture**: a known multi-turn conversation with hand-computed per-turn `cached / resent_history` ratios (denominator = system + all prior messages, excluding the turn's new content), asserting every per-turn value and the cumulative sum individually.
- [x] **C2 — cloud adapters.** openai (`cached_tokens`; covers OpenAI and OpenRouter by setting the provider endpoint to its URL) and anthropic (read/creation split). Files: `src/adapters/openai.rs`, `src/adapters/anthropic.rs`, `src/rates.rs`. The no-cache cost counterfactual is priced from a static per-model rate table (`src/rates.rs`) with `--rates <file>` overrides; rates are USD per 1M tokens plus cache multipliers (`cached_input_mult`, `cache_write_mult`). `cost_saved` = the re-sent history priced at full input rate minus its actual cached cost (and minus the cache-write premium where the provider charges one). Anthropic's derived `read/(read+creation)` is a **secondary** split shown beside cost; the headline hit-rate stays the binding `cached / resent_history`. Verify: record matches provider usage exactly on a live run; cost math reflects published rates (unit fixtures per provider).
- [x] **C3 — local adapters.** llama.cpp (ground truth), vLLM (metrics), mlx-lm (discrimination only). Files: `src/adapters/llamacpp.rs`, `src/adapters/vllm.rs`, `src/adapters/mlxlm.rs`. Verify: ±5% llama.cpp criterion; mocked-adapter CI suite passes engineless.
- [x] **C4 — sessions + aggregation.** prefix-continuity store, fork resolution, atomic aggregate. Files: `src/sessions.rs`. Verify: interleaved-requests atomicity test (real interleave, tokio); fork tie-break fixture; collision log test.
- [x] **C5 — latency budget CI.** warm-pinned median-of-N proxy-vs-direct TTFT + micro-profile. Files: `tests/c5_latency.rs`, `tests/c5_microprofile.rs`, `.github/workflows/ci.yml`. Verify: >6 ms p95 fails on injected delay; passes on clean tree; target is ≤5 ms p95. Measured clean-tree release p95 ≈0.2 ms; the micro-profile asserts each hot-path stage (prefix-hash, resolve, observe+build) stays bounded and prefix hashing scales linearly (no O(n²) re-tokenize).
- [x] **C6 — dashboard.** single-file, single-screen, no navigation. Primary workspace is one composition, hero-first:
  1. **Status strip (top):** live indicator, tape mode (hash-level / byte-level), export, incomplete count.
  2. **Hero band (weights by backend):** cloud shows cost saved (`billed input`, `cache-served`, hit rate labeled `provider_reported`); local shows TTFT cold→warm with the speedup factor. The cold-to-warm readout sits beside it always.
  3. **Session view (left, 42%):** per-turn table (turn, hit, `cached / resent history`, cost) plus cumulative row. The denominator is `resent_history_tokens`, which grows each turn (system + all prior messages); turn 0 shows `— / —` since it is cold and excluded.
  4. **Prefix tape (right, 58%):** per-turn prefix map (hit / resent / cold / break / miss / incomplete) with a legend. Turn rows align with the session view.
  Files: `src/dashboard.rs`. Verify: state-map rows each render; tape legible without color (glyph, not just green/red); no fake detail in hash-level mode; every color/type/space value comes from DESIGN.md tokens.
  > Layout and tokens are defined in `DESIGN.md` (direction A: warm bone/graphite instrument; the one amber accent marks verified cache-served data and the instrument's live/focus states only). No ad-hoc palette, type scale, or layout beyond DESIGN.md.
  > Responsive + a11y contract: minimum supported viewport 1024px; below 1024px the session view and tape stack vertically with the hero staying full-width (no nav to collapse). Keyboard: session rows and tape cells are focusable, `export` is keyboard-reachable, focus ring always visible. Contrast ≥4.5:1 on body text. All figures use tabular numerals. The tape is legible with color removed (glyph rule, above). Any interactive control has a ≥44px target.
  > Theme: light **and** dark, following the OS `prefers-color-scheme`. Both palettes come from DESIGN.md CSS variables; no hard-coded colors.
- [x] **C7 — observability.** JSONL export (metrics-only by default; CLI writes a file, dashboard downloads the same schema) + structured finalize logs. Files: `src/export.rs`. Verify: export matches dashboard numbers; CLI file and browser download are byte-identical for one session; logs grep-able without bodies. Both triggers call one `to_jsonl`, so the file and the download are byte-identical by construction (tested); the CLI reads the running proxy's `/api/export` (state is in-memory) and honors `--out`. Finalize logs are `target: "cachemax_finalize"`, counts and labels only.
- [x] **C8 — CLI.** `serve`, `--backend`, `--upstream-url` (provider endpoint; OpenRouter sets it to `https://openrouter.ai/api/v1`), `--check`, `export`, `--verbose` (metadata only), `--rates <file>`, `--tokenizer`, `--bind`, engine/precision flags. Files: `src/main.rs`. Verify: `--check` fails loudly on unreachable upstream; `--help` lists everything; defaults work with zero flags. `--check` exits 1 on a transport error and 0 on any HTTP answer (even 401). Tests in `tests/c8_cli.rs`.

### Getting started (T0 contract)

- [x] **D1 — README + docs.** Install command, first command, dashboard URL, cost-capable client snippet. Files: `README.md`. Verify: the sequence runs end to end; no stale proxy-vision text; typo fixed.
- [x] **D2 — one named agent config.** Point a real agent at the proxy (OpenAI `base_url` swap, which works for OpenAI, Anthropic, or OpenRouter; model passed through; dummy `api_key` accepted and ignored if required) with one worked example. Files: `README.md`. Verify: the named agent completes a multi-turn conversation through the proxy.
- [x] **D3 — error contract.** Problem + cause + fix + docs link for every failure the spec can produce: upstream unreachable, no cache signal, key rejected, tokenizer unavailable, oversized prompt, engine timeout, tokenizer version mismatch. Files: `src/main.rs`, `docs/troubleshooting.md`. Verify: each path emits the four parts; no raw panic by default.
- [x] **D4 — docs structure.** `README.md` (getting started), `docs/troubleshooting.md`, `docs/how-measurement-works.md` (record schema, formula in plain words, what provider-reported means, write/read split). Verify: `provider_reported` and the Anthropic split explained; troubleshooting does not invite sharing content-bearing logs.

### Later (post-T0)

- [ ] **M1 — macOS menu bar item.** Glanceable status only (proxy running / live / error, current hit rate), click-through opens the existing web dashboard. The web dashboard remains the single full UI; the menu bar item duplicates no panels. macOS only. Not T0.

## Not in scope

- Repair engine, murderer demo mode, cloud cost gate: later phases. Measurement core first.
- Prefix-affinity routing, cross-dialect translation, cache warming: the superseded doc's scope.
- Hosted or deployable service. The proxy is a local binary fronting cloud upstreams.
- LICENSE selection, issue templates, changelog ownership: at ship time.
- Demo command: cut. The moment comes from real traffic (cold turn, then warm turn).

## What already exists

- Repo scaffolding: uv project under `sidecar/`, FastAPI skeleton, placeholder CLI, 1 passing test. Repurposed for the sidecar and dev tooling; the core is new Rust at the repo root.
- llamacpp-stats-dashboard (MIT): patterns borrowed (read1() streaming, caps-field display, pre-warm TTFT testing).
- LMCache: per-request hit-rate attribute pattern, token-level counters, blake3 hashing option.
- Rust ecosystem: axum/hyper SSE, reqwest, `tiktoken-rs` (token-prefix hashing).

## Open questions

- vLLM: poll `/metrics` per request or scrape continuously? Attribution granularity differs.
- Tape from the proxy's own token stream (portable) or engine ground truth where available (accurate)? Default: engine truth when available, labeled otherwise.
- Binary name: `cachemax`. Decided (was `cache-maxing` or `cache-max`).

## Implementation Tasks (design review)

Synthesized from the design review's findings. Each task derives from a specific
finding above. Run with Claude Code or Codex; checkbox as you ship.

- [x] **T1 (P1, human: ~1h / CC: ~10min)** — dashboard — build the hero-first single-screen layout (status strip, hero band, session view, prefix tape)
  - Surfaced by: Pass 1 (Info Arch 3/10): no hierarchy or layout specified
  - Files: `src/dashboard.rs`
  - Verify: all four regions render; no navigation; hero weights by backend
- [x] **T2 (P1, human: ~1h / CC: ~10min)** — dashboard — implement the interaction state table (loading/empty/error/success/partial) for every surface
  - Surfaced by: Pass 2 (States 2/10): no loading/empty/error states; reset banner and session break undrawn
  - Files: `src/dashboard.rs`
  - Verify: each state in the table renders; `—` never `0`; reset banner + session break appear
- [x] **T3 (P2, human: ~30min / CC: ~5min)** — dashboard — make the cloud journey's "is it alive" signal land on turn 0 and the cost payoff on turn 1
  - Surfaced by: Pass 3 (Journey 2/10): no emotional arc; first-turn liveness unaddressed
  - Files: `src/dashboard.rs`
  - Verify: cold turn (t0) shows activity within one turn; first warm turn (t1) shows hit rate + the mode's source label (`provider_reported` cloud, `engine_measured` local)
- [x] **T4 (P1, human: ~1h / CC: ~10min)** — dashboard — responsive + a11y contract (1024px floor, stack below, keyboard, contrast, tabular numerals, 44px targets, light/dark via `prefers-color-scheme`)
  - Surfaced by: Pass 6 (Responsive 2/10): no viewport/a11y spec
  - Files: `src/dashboard.rs`
  - Verify: keyboard-only walkthrough; contrast check; both themes render
- [x] **T5 (P1, human: ~2h / CC: ~20min)** — design system — create `DESIGN.md` via `/design-consultation`
  - Surfaced by: Pass 4+5 (AI Slop 4/10, Design Sys 2/10): no DESIGN.md; visual language deferred by owner
  - Files: `DESIGN.md`, `src/dashboard.rs`
  - Verify: every dashboard color/type/space value references a DESIGN.md token
  - Done: DESIGN.md created (direction A: warm bone/graphite instrument); C6 token check now unblocked.


## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | issues_open | HOLD SCOPE, 0 critical gaps |
| Outside Review | codex (`/plan-ceo-review`) | Independent 2nd opinion | 1 | completed | 7 findings adopted |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 2 | issues_open | 14 issues, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 1 | issues_open | score: 2/10 → 7/10, 6 decisions |
| DX Review | `/plan-devex-review` | Developer experience gaps | 1 | issues_open | score: 3/10 → 6.5/10, TTHW: >10min → 2-5min |

- **OUTSIDE COVERAGE:** codex, plan-review (CEO phase), completed, 7 findings adopted. No outside voice ran for the design or DX phases.
- **CROSS-MODEL:** native CEO/eng/design/DX reviews plus one completed codex outside voice; overlap on the codex plan-review findings (all 7 adopted). No distinct-model inference beyond recorded provider.
- **VERDICT:** CEO + ENG CLEARED, ready to implement. The design review found UI specification gaps (the dashboard had no layout, states, or a11y spec); fixes approved and applied. Design review is required before dashboard styling is design-complete.
- **UNRESOLVED DECISIONS:** None. Visual language was deferred to `/design-consultation`, which produced `DESIGN.md` (direction A, task T5, done).

NO UNRESOLVED DECISIONS
