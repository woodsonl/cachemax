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

## omniroute.home.arpa · auto/fast (2026-10-05)

Local routed endpoint, Apple-silicon inference. Five classes, 5 samples per
form, 50 sends total, one upstream instance answering.

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

## openrouter.ai · free tier (2026-10-05)

Two free models, 3 samples per form, 36 sends each: nvidia/nemotron-3-ultra-
550b-a55b:free and openrouter/free (auto-router). Auth via the OpenRouter
key, `--model` selects the fixture model.

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

## Reading the table

- `costs N tk` — the drifted bytes lose N cached tokens to the drift; repair
  recovers them on this endpoint.
- `absorbed` — the endpoint normalizes the class away; drift costs nothing
  there, and repair has nothing to recover for it.
- `no caching on this endpoint` — every reading on both forms was a
  reported zero (median AND max): the endpoint did not serve from cache at
  all during the run. Drift costs nothing because caching costs nothing —
  a different mechanism than absorption, stated differently.
- `unmeasured` — every send failed or carried no cache figure; the cause is
  on stderr. A gap, never a zero.

A class can be `costs` on one endpoint and `absorbed` on another: providers
that cache raw token identity punish serialization drift; routers that
re-tokenize prompts absorb it. That difference is why the ladder is
per-endpoint prioritized from measured data instead of guessed.
