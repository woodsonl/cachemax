# Drift-cost matrix

What semantically-identical-but-byte-different request bodies actually cost
against a live endpoint, per drift class. Run:

```
cachemax drift-matrix --upstream-url <endpoint> --backend openai --n 5 --yes
```

Each class is a (canonical, drifted) request pair — the same conversation,
different serialization. Both forms are sampled `--n` times interleaved on
fresh connections, and the delta between cached-token medians is the cache
that repair would recover for that class on that endpoint. A class reading
`absorbed` costs nothing there: the endpoint normalizes it away, and repair
has nothing to recover. The verdict is the per-endpoint priority list —
spend ladder work where the table says `costs`.

Sends are billable on cloud endpoints. The planned total prints before the
first send, and runs above 40 sends need `--yes`.

**Routed endpoints: a measurement belongs to the provider instance that
served it.** OpenRouter (and omniroute) forward to inference providers,
each with its own caching behavior; one run measures the provider(s) that
answered those requests, not the router as a whole. Results below are
attributed to the provider slug that ran them. Two caveats the tool
itself cannot close on OpenRouter: the router exposes no routing headers,
so the instance fingerprint is blind and cross-provider mixing within one
run is invisible; and the same model slug can be served by different
providers over time. A run is a sample of that provider's behavior that
day, not a permanent property of the router.

## omniroute.local (auto/fast) — local models (2026-10-05)

Local router to Apple-silicon models. Five classes, 5 samples per form,
50 sends total, one upstream instance answering. The normalization
credited below cannot be pinned to router or model — the fingerprint
cannot split them — so it is attributed to the serving stack as a whole.

```
drift-cost matrix · https://omniroute.home.arpa · n=5
  class                    drifted m/m canonical m/m   delta send  verdict
  tool-arg-reorder             390/390     390/390       0  5/5  absorbed
  whitespace                   393/393     390/390      +3  5/5  inverted: drifted cached more
  key-order                    390/390     390/390       0  5/5  absorbed
  number-text                  392/392     390/390      +2  5/5  inverted: drifted cached more
  content-string-vs-array      390/390     390/390       0  5/5  absorbed
```

No class costs cache here: the router normalizes or re-tokenizes prompts
before its cache keys them, so serialization drift recovers nothing.
Three classes read absorbed; two read inverted — the drifted form cached
slightly MORE (393 vs 390, 392 vs 390), the router's tokenizer turning the
altered text into a few more cache-served tokens. An inversion is a
router artifact, not a signal in either direction, and the table says so
rather than folding it into absorbed.

The consequence for repair: on this endpoint, repair-on recovers nothing
for these five classes, and the ladder work that matters elsewhere is not
prioritized here. The same matrix against a byte-identity provider
(OpenAI direct, Anthropic without normalization) is where `costs` verdicts
are expected; those runs are user-gated.

## NVIDIA provider + auto-router, via OpenRouter free tier (2026-10-05)

One fixed provider and one auto-routed slug, 3 samples per form, 36 sends
each: nvidia/nemotron-3-ultra-550b-a55b:free (NVIDIA's serving infra) and
openrouter/free (OpenRouter's auto-router — not a provider: it mixes
providers per request, so that run's readings may span more than one
backend, invisible to the fingerprint). Auth via the OpenRouter key, `--model` selects the
fixture model.

```
nvidia/nemotron-3-ultra-550b-a55b:free
  class                    drifted m/m canonical m/m   delta send  verdict
  tool-arg-reorder                 0/0         0/0       0  3/3  no caching on this endpoint
  whitespace                       0/0         0/0       0  3/3  no caching on this endpoint
  key-order                        0/0         0/0       0  2/3  no caching on this endpoint
  number-text                      0/0         0/0       0  2/2  no caching on this endpoint
  tools-reserialization            0/0         0/0       0  3/3  no caching on this endpoint
  content-string-vs-array            —         0/0       —  0/2  unmeasured

openrouter/free
  class                    drifted m/m canonical m/m   delta send  verdict
  tool-arg-reorder                 0/0         0/0       0  3/3  no caching on this endpoint
  whitespace                       0/0         0/0       0  3/3  no caching on this endpoint
  key-order                        0/0         0/0       0  3/3  no caching on this endpoint
  number-text                      0/0         0/0       0  3/3  no caching on this endpoint
  tools-reserialization          0/384         0/0       0  3/3  absorbed
  content-string-vs-array          0/0         0/0       0  3/3  no caching on this endpoint
```

The free tier reports a cache figure (the field exists) but on the
fixed-provider model every reading on both forms is a reported zero: the
providers do not serve from prompt cache for free-tier traffic, or do not
report it. Repair recovers nothing
here because nothing is cached — a different mechanism than omniroute's
absorption, and the verdict says so. The openrouter/free run shows the
routing-lottery hazard of auto-routing in one number: a single drifted
send landed on a caching instance (max 384) while every other reading
read zero — the median held, and the max column is why readings are
published at all.

The byte-identity question (do key order, number text, argument
reordering cost cache on a provider that caches raw tokens?) needs a paid
model; the free tier cannot answer it. A cents-scale run on a paid
OpenRouter model or a direct provider key is the remaining measurement.

## DeepSeek provider (deepseek-chat-v3.1), via OpenRouter paid (2026-10-06)

The byte-identity leg — attributed to the DeepSeek provider that served
it, not to OpenRouter. DeepSeek V3.1 does automatic prefix caching with no
breakpoints and reports hit counts through the OpenAI-compatible surface —
the mechanism that punishes byte-level drift. The protocol here is the
corrected one: warm the canonical form (2 pairs), then send the drifted
form — the FIRST drifted reading against the cache the canonical form
demonstrably established (the warm's maximum; individual warm sends can
read zero seconds after establishing it) is the drift cost. 60 sends,
about 2 cents. The fingerprint is blind on this path, so provider mixing
within the run cannot be ruled out; repeat on a different day or a
different provider serving the same class of cache before generalizing.

```
drift-cost matrix · https://openrouter.ai/api/v1 · n=3
  class                    drifted 1st/m canonical m/m   delta send  verdict
  tool-arg-reorder             148/199     127/204      56  3/2  costs 56 tk (first send)
  whitespace                    12/204       0/127     115  3/2  costs 115 tk (first send)
  key-order                    201/201     201/201       0  3/2  absorbed
  number-text                  163/203     127/201      38  3/2  costs 38 tk (first send)
  tools-reserialization          0/334       4/334     334  3/2  costs 334 tk (first send)
  content-string-vs-array        0/205     205/205     205  3/2  costs 205 tk (first send)
```

**Serialization drift costs real cache on this provider, and the classes
repair fixes are the classes it punishes.** Tool-argument reordering costs
the drifted tail (~56 tokens: the prefix up to the tool call survives,
the rest misses). Prose whitespace in the system costs most of the prefix
(115-197 across runs). Number text costs a partial-to-full miss (38-201
across runs). Tool-definition reserialization costs the whole
tools-bearing prefix (334 — and repair's tools-prefix batch exists exactly
for this). Content shape (string vs parts array) cost the full prefix this
run; an earlier run read a full hit for it — that class flips between
runs and is recorded as variable. Key order alone is absorbed, stable
across every run: DeepSeek's serving layer re-serializes object keys
before caching, so key-order drift is free there and repair recovers
nothing for it.

Two provider behaviors recorded during measurement: the cache
intermittently reads zero on a repeat seconds after establishment
(eviction or shard routing — indistinguishable here because OpenRouter
exposes no routing headers, so the instance fingerprint is blind), which
is why the baseline is the warm's maximum and the readings are published;
and first-send costs vary run to run (56-204 for tool-arg reorder), which
is why the verdict is the direction, not a single number.

The omniroute and free-tier tables above predate the corrected protocol
(they measured interleaved alternation, which lets each form warm its own
entry); their verdicts stand as recorded but the DeepSeek-provider table
is the one to cite for what drift costs a caching provider. The Anthropic-dialect
classes could not be measured through OpenRouter: its `/v1/messages` path
reports zero cache writes for `cache_control` payloads at any length
(3216-token probe, three sends), and the OpenAI-compat path with a Claude
model also read zero — OpenRouter does not wire Anthropic prompt caching
on either path. That leg needs a direct Anthropic key.

## api.anthropic.com · subscription OAuth (2026-10-06)

The Anthropic-dialect leg, run directly against Anthropic's API with a
Claude subscription token
(`sk-ant-oat…`) on the native `/v1/messages` endpoint, 48 sends, n=2: every
reading on every class is a reported zero — the subscription auth path
reports the cache fields but never a nonzero figure, so drift cost cannot
be measured on it. This is the third auth path surveyed and the third answer: the omniroute
local models normalize, the free-tier providers surveyed do not cache, and
Anthropic subscription OAuth does not report caching to raw API callers —
each a statement about the providers that served the runs, not a permanent
property of any router.
The Anthropic-dialect classes (hint placement, system shape, the
tools-bearing prefix with `cache_control`) remain unmeasured pending a
console API key (`sk-ant-api…`), which is the one auth path documented to
bill and report prompt caching.

## Reading the table

- `costs N tk` — the drifted bytes lose N cached tokens to the drift; repair
  recovers them on this endpoint.
- `absorbed` — the endpoint normalizes the class away; drift costs nothing
  there, and repair has nothing to recover for it.
- `no caching on this endpoint` — nothing ever read above zero: the
  endpoint did not serve from cache at all during the run. Drift costs
  nothing because caching costs nothing — a different mechanism than
  absorption, stated differently.
- The drifted column shows first-send/max; the canonical column shows
  median/max of the warm. (The omniroute and free-tier tables predate the
  corrected protocol and show median/max on both sides.) The verdict compares the drifted FIRST send
  against the warm MAXIMUM (the established baseline) — one flaky zero in
  the warm cannot declare the cache absent.
- `unmeasured` — every send failed or carried no cache figure; the cause is
  on stderr. A gap, never a zero.

A class can be `costs` on one endpoint and `absorbed` on another: providers
that cache raw token identity punish serialization drift; routers that
re-tokenize prompts absorb it. That difference is why the ladder is
per-endpoint prioritized from measured data instead of guessed.
