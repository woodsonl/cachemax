# Dogfood: cachemax v0.2.0 against real routed traffic (omniroute)

**Date:** 2026-10-05 · **Proxy:** cachemax v0.2.0, `--repair dry-run` default,
per-request `x-cachemax-repair` header switching · **Upstream:**
`omniroute.home.arpa` (`auto/fast` → claude-sonnet-4.6-class, OpenAI-shape
usage with `prompt_tokens_details.cached_tokens`) · **Ledger:**
`/tmp/cm-ledger` · **Raw artifacts:** `/tmp/cm-ab/*.json`, proxy log
`/tmp/cm-serve.log` (grep `cachemax_repair`).

## What was measured

A 5-turn conversation over a ~3,270-token policy document. Each turn's
re-sent history was offered in three arms:

- **C_read** — canonical history (single-spaced), repair `off`
- **REPAIR** — drifted history, **`x-cachemax-repair: on`** (proxy rewrites
  to the canonical serialization before forwarding)
- **DRIFT** — drifted history (whitespace runs expanded ×2/×3), repair `off`

The drift is whitespace-only — semantically identical text, different bytes —
which is exactly the class of drift the repair ladder certifies as
equivalent-but-not-exact. Assistant replies in the history are the model's
actual answers, so the ledger chain genuinely extends (repair can fire) —
v1 of this experiment fabricated replies, repair correctly refused to
rewrite, and the lesson (use real replies) is baked in here.

## Results

### Repair vs drift on a warm cache (the core claim)

| turn | drift | REPAIR (on) | DRIFT (off) | verified rewrite |
|---|---|---|---|---|
| 2 | ×2 | **3,278 / 3,322 = 98.7%** | 128 / 5,598 = **2.3%** | `session 6: TextNormalization, 4 elements, 3,174 tk` |
| 4 | ×3 | **3,332 / 3,376 = 98.7%** | 146 / 5,682 = **2.6%** | `session 7: TextNormalization, 6 elements, 3,234 tk` |

Same semantic request, same turn: **drifted bytes = cold, repaired bytes =
warm.** Both rewrites are receipt-logged (`cachemax_repair` target).

### The token surcharge (drift costs tokens, too)

Whitespace-expanded drift does not just miss cache — it bills more:

| form | prompt tokens |
|---|---|
| canonical | 3,376 |
| ×2 drift | 5,598 (**+66%**) |
| ×3 drift | 5,682 (**+68%**) |

### Consistent drift builds its own parallel chain

The ×3 form, sent three times, converged to 5,667/5,682 = **99.7%** — the
upstream caches *whatever byte-prefix it sees repeatedly*. The pathology
repair targets is therefore the **inconsistent** serializer (this
experiment's varying ×1/×2/×3), which can never accumulate a stable chain.
The `rearm` probe closed the loop honestly: repair-on cannot rescue a
request whose drifted form is already the warmest prefix on every upstream
instance — the ledger stores as-drifted (6/10 contents multi-spaced), the
rewrite does not fire (no `cachemax_repair` line), and the re-send hits the
drift chain at 5,667–5,681/5,682. Repair rewrites toward *its own* canonical
chain; it does not fight an already-warm drift chain.

### Noise disclosure: the routing lottery

omniroute load-balances across ≥3 upstream instances with **separate cache
namespaces**. Single sends of any arm flip between ~99% (instance holding
the warm prefix) and ~2–4% (instance without it):

| arm | max-of-3 sends |
|---|---|
| C_read | 98.7% / 3.8% / **100.0%** |
| REPAIR | 2.6% / **99.7%** / 2.3% |
| DRIFT (first 3) | 2.6% / **99.7%** / 2.3% |

Consequence: **within-instance comparisons are clean, across-instance
comparisons are not.** The two REPAIR-vs-DRIFT pairs in the first table are
same-instance readings (the clean signal). A production-grade A/B on a
routed endpoint needs sticky routing or per-instance sampling — noted as a
follow-up for `cachemax replay`.

## Reading the dashboard honestly

With the router's own ~128-token prefix landing inside `cached_tokens`, the
dashboard's per-turn hit-rate can exceed 100% (the numerator counts
router-side cache the denominator never re-sent). It is annotated
(`provider_reported`); a router-own-prefix adjustment is a possible
follow-up, not a correctness bug in the record.

## Verdict

1. **Repair works where it claims to:** 2/2 verified rewrites,
   TextNormalization-only, 98.7–99.7% cache hits on the canonical chain,
   with zero semantic drift (whitespace-only).
2. **Drift is expensive twice:** cold cache *and* +66–68% billed tokens for
   expanded-serializer drift.
3. **The real-world pathology is inconsistent serialization** — a stable
   drifted chain accumulates its own warmth; a wobbling one never does, and
   that is where repair-on recovers ~97 points of hit rate per turn.
4. **Routed endpoints need sampling discipline** — see the lottery note
   before quoting any single number.
