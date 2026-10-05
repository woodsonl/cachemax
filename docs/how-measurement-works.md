# How measurement works

cachemax measures one thing precisely: of the history your client re-sends on
every turn, how much did the provider serve from its prompt cache? Everything
else — cost, speed, the tape — is derived from that.

## The problem it measures

LLM APIs are stateless. Every turn, your client re-sends the whole
conversation: the system prompt, the tools, and every prior user, assistant, and
tool message. Providers cache the *prefix* of that re-sent history so they don't
recompute it — but the cache only helps if the prefix is byte-identical to last
time. A regenerated reply, a reordered tool list, or a prompt that changes
slightly each turn breaks the prefix, and you pay full price to recompute
everything.

cachemax sits in front of the provider and shows you which is happening, turn by
turn.

## The record

One record per request. Nothing about message content is in it — metrics only:

| Field | Meaning |
|---|---|
| `session_id` | Which conversation this request belongs to. |
| `turn` | Turn index within the session. Turn 0 is cold. |
| `status` | `complete` or `incomplete` (the stream ended early, or the upstream answered non-2xx). |
| `source` | Where the cache figure came from: `provider_reported`, `engine_measured`, or `no_cache_truth`. |
| `ttft_ms` | Time to first token. |
| `cached_tokens` | Prefix tokens the provider/engine reports it served from cache. |
| `cache_written_tokens` | Prefix tokens written to cache this turn (Anthropic only; 0 elsewhere). |
| `resent_history_tokens` | The binding denominator: system + all prior messages, excluding this turn's new content. |
| `billed_input_tokens` | Billed input tokens for the turn. |
| `broke_prefix` | `true` when this turn's prompt diverged from the tracked session's prefix (a prefix break). |
| `cost_usd` | Cost at the provider's published rate, if the model is known. `null` for an incomplete turn. |
| `cost_saved_usd` | The no-cache counterfactual minus the actual cost, if rates are known. `null` for an incomplete turn. |
| `repair_mode` | Whether drift classification ran: `off`, `dry_run` (the default), or `on`. |
| `repaired` | `true` only when the outgoing history was rewritten (`on` mode). |
| `matches_canonical` | `null` when not examined; else whether the re-sent history byte-matched the canonical chain. |
| `drift_kind` | The classified drift flavor, when there was one. |
| `canonicalized_tokens` | Tokens of drifted history that would be (`dry_run`) or were (`on`) canonicalized. An estimate for annotation. |

## The hit-rate formula

For every turn `t ≥ 1`:

```
hit_rate(t) = cached_tokens(t) / resent_history_tokens(t)
```

Session-cumulative hit rate is the same ratio over the sums, complete turns only:

```
cumulative = Σ cached_tokens / Σ resent_history_tokens
```

Two rules that matter:

- **Turn 0 is cold.** It establishes the cache, so it has no history to reuse.
  Its hit rate is `—`, not `0`, and it is excluded from the cumulative.
- **The denominator is the re-sent history**, not the whole prompt. It is the
  system prompt plus every prior user/assistant/tool message — the part that
  *could* have been served from cache. This turn's new content is not in it.
  That is why the denominator grows every turn, and why the hit rate is the
  honest answer to "how much of what I re-sent came back from cache?"

`—` always means *unexposed*, and is styled to look unlike `0`. A zero
denominator renders `—`, never `0`.

## What `provider_reported` means

On the cloud path, the cache figure is the provider's own number. cachemax never
re-derives or second-guesses it. OpenAI and OpenRouter report
`usage.prompt_tokens_details.cached_tokens`; cachemax reads it and labels it
`provider_reported`. If the provider says 1,455 tokens were cached, the record
says 1,455.

The label is not a disclaimer about what the number *isn't* — it is a statement
of what it *is*: the provider's reported count. Local engines are labeled
`engine_measured` (llama.cpp, vLLM) or `no_cache_truth` (mlx-lm).

A local engine reports cached tokens in its own token space, which can drift
slightly from cachemax's tokenizer. To keep the ratio honest, an
`engine_measured` count is clamped to the history span, so a local hit rate
never exceeds 100%. A `provider_reported` count is the provider's own figure and
is shown as-is.

## The Anthropic write/read split

Anthropic does not expose a single hit count. It splits cache activity in two:

- `cache_creation_input_tokens` — tokens **written** to the cache this turn.
  Billed at a **premium** (1.25× for the 5-minute TTL).
- `cache_read_input_tokens` — tokens **read** from the cache. Billed at a
  **discount** (0.1×).

The dashboard shows this split beside cost, and a **derived** share:

```
derived = cache_read / (cache_read + cache_creation)
```

That derived figure is a secondary read on Anthropic's cache economics. The
**headline** hit rate stays `cached / resent_history`, the same binding formula
every other backend uses, so sessions are comparable across providers.

## Cost

Cost is priced from a static per-model rate table (`src/rates.rs`), overridable
with `--rates <file>`. Two derived figures:

- `cost_usd` — the turn's billed input tokens at the model's input rate.
- `cost_saved_usd` — the **no-cache counterfactual**: the re-sent history priced
  at the full input rate, minus what it actually cost with the provider's cache
  discount (and minus the write premium, where the provider charges one). This
  is "what cache reuse was worth" on that turn.

Unknown models carry no cost rather than a guessed one.

## What is not measured

cachemax measures cache reuse and its cost/speed consequence. It does not repair
a broken prefix, warm a cache, or route by prefix affinity — those are later
phases. It measures first.
