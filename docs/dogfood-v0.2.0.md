# Dogfood: cachemax v0.2.0 against real routed traffic (omniroute)

**Date:** 2026-10-05 · **Proxy:** cachemax v0.2.0, `--repair dry-run` default,
per-request `x-cachemax-repair` header switching · **Upstream:**
omniroute routed endpoint (`auto/fast`; OpenAI-shape usage with
`prompt_tokens_details.cached_tokens`) · **Ledger:** `/tmp/cm-ledger` ·
**Raw artifacts:** `/tmp/cm-ab/*.json`, proxy log `/tmp/cm-serve.log`
(grep `cachemax_repair`). All numbers in this report trace to those
artifacts; no sends were made after the fact.

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

### Repair vs drift on a cold drifted form (the core claim)

| turn | drift | REPAIR (on) | DRIFT (off) | verified rewrite |
|---|---|---|---|---|
| 2 | ×2 | **3,278 / 3,322 = 98.7%** | 128 / 5,598 = **2.3%** | `session 6: TextNormalization, 4 elements, 3,174 tk` |
| 4 | ×3 | **3,332 / 3,376 = 98.7%** | 128 / 5,682 = **2.3%** | `session 7: TextNormalization, 6 elements, 3,234 tk` |

Same semantic request, same turn: **the drifted byte-form had never been
sent to any upstream instance, so it read cold regardless of routing, while
the repair arm forwarded canonical bytes the cache already held.** Both
rewrites are receipt-logged (`cachemax_repair` target). The turn-4 rewrite
occurred on the first drifted×3 exposure of that conversation (a fresh
proxy session); the turn-4 DRIFT cell is that conversation's own drifted
send, taken before any ×3 bytes existed anywhere.

Why these two turns and not the others: turns 1 and 3 used ×1 "drift"
(collapse-to-single = identity on the canonical form), so all three arms
forwarded effectively identical bytes — REPAIR 99.5%, DRIFT 99.5% (t1) and
REPAIR 3.8%, DRIFT 99.5% (t3, pure routing luck). Those rows carry no drift
signal and are excluded; the run-wide single-sample averages were C_read
27.7%, REPAIR 75.2%, DRIFT 50.9% — dominated by the lottery, which is why
per-arm max-of-3 sampling (below) is the methodology going forward.

### The token surcharge (drift costs tokens, too)

Whitespace-expanded drift does not just miss cache — it bills more:

| form | prompt tokens | vs same-turn canonical |
|---|---|---|
| canonical (turn 4) | 3,376 | — |
| ×2 drift (turn 2; canonical there: 3,322) | 5,598 (**+68.5%**) | cross-turn, same content |
| ×3 drift (turn 4) | 5,682 (**+68%**) | same-turn |

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
chain; it does not fight an already-warm drift chain. (Rearm's header is
not recorded in its log, but turn4.py's `REPAIR_x3` — verifiably
repair-on, same ledger outcome, no receipt — carries the same conclusion.)

### Noise disclosure: the routing lottery

omniroute load-balances across ≥3 upstream instances with **separate cache
namespaces** — proven in-session by the *identical* request body reading
2.6% / 99.7% / 2.3% on three consecutive sends (session 7, turns 2–4).
No artifact carries instance identity, so per-send "same instance" cannot
be claimed; what the headline pairs exploit instead is that a **first-ever
drift byte-form is cold on every instance** — routing cannot warm it —
while the canonical chain is warm on at least the instance repair lands on.
Max-of-3 samples per arm:

| arm | 3 sends |
|---|---|
| C_read (canonical, off) | 98.7% / 3.8% / **100.0%** |
| REPAIR_x3 (drifted, on; rewrite *declined* — see rearm) | 2.6% / **99.7%** / 2.3% |
| DRIFT_x3 (drifted, off) | **99.7% / 99.7% / 99.7%** |

DRIFT_x3's uniform 99.7% is not a refutation of repair — it is the
parallel-chain effect: by the time DRIFT_x3 ran, the ×3 byte-form had
already been written (its own first send + REPAIR_x3's declines forwarded
drifted bytes too). Consequence for methodology: **within-turn first-exposure
comparisons are clean; anything after a byte-form's first send needs
per-instance sampling or sticky routing** — noted as a `cachemax replay`
follow-up.

## Reading the dashboard honestly

With the router's own ~128-token prefix landing inside `cached_tokens`, the
dashboard's per-turn hit-rate can exceed 100% (the numerator counts
router-side cache the denominator never re-sent). It is annotated
(`provider_reported`); a router-own-prefix adjustment is a possible
follow-up, not a correctness bug in the record.

## Verdict

1. **Repair works where it claims to:** 2/2 verified rewrites,
   TextNormalization-only, 98.7% canonical-chain cache hits, with zero
   semantic drift (whitespace-only).
2. **Drift is expensive twice:** cold cache on first exposure *and* +68%
   billed tokens for expanded-serializer drift.
3. **The real-world pathology is inconsistent serialization** — a stable
   drifted chain accumulates its own warmth (99.7% converged); a wobbling
   one never does, and that is where repair-on recovers ~96 points of hit
   rate per turn.
4. **Routed endpoints need sampling discipline** — see the lottery note
   before quoting any single number.
